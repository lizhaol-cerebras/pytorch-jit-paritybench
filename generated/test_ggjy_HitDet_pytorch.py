import sys
_module = sys.modules[__name__]
del sys
faster_fpn_x101_32x4d = _module
faster_fpn_x50_32x4d = _module
fbhit = _module
faster_rcnn_1333x800_r50_fpn_1x = _module
faster_rcnn_640x640_r50_fpn_1x = _module
run_faster_detnas_1G_nasfpn_4conv1fc = _module
run_faster_nasfpn = _module
faster_rcnn_r50_panet_syncbn = _module
faster_rcnn_r50_fpn_voc = _module
train_hit_paper_voc = _module
retinanet_1333x800_r50_fpn_1x = _module
mmcv = _module
arraymisc = _module
quantization = _module
cnn = _module
alexnet = _module
resnet = _module
vgg = _module
weight_init = _module
fileio = _module
handlers = _module
base = _module
json_handler = _module
pickle_handler = _module
yaml_handler = _module
io = _module
parse = _module
image = _module
transforms = _module
colorspace = _module
geometry = _module
normalize = _module
resize = _module
opencv_info = _module
parallel = _module
_functions = _module
collate = _module
data_container = _module
data_parallel = _module
distributed = _module
scatter_gather = _module
runner = _module
checkpoint = _module
dist_utils = _module
hooks = _module
closure = _module
hook = _module
iter_timer = _module
logger = _module
pavi = _module
tensorboard = _module
text = _module
lr_updater = _module
memory = _module
optimizer = _module
sampler_seed = _module
log_buffer = _module
parallel_test = _module
priority = _module
runner = _module
utils = _module
config = _module
misc = _module
path = _module
progressbar = _module
timer = _module
version = _module
video = _module
optflow = _module
optflow_warp = _module
processing = _module
visualization = _module
color = _module
mmdet = _module
apis = _module
env = _module
inference = _module
train = _module
core = _module
anchor = _module
anchor_generator = _module
anchor_target = _module
guided_anchor_target = _module
bbox = _module
assign_sampling = _module
assigners = _module
approx_max_iou_assigner = _module
assign_result = _module
base_assigner = _module
max_iou_assigner = _module
bbox_target = _module
geometry = _module
samplers = _module
base_sampler = _module
combined_sampler = _module
instance_balanced_pos_sampler = _module
iou_balanced_neg_sampler = _module
ohem_sampler = _module
pseudo_sampler = _module
random_sampler = _module
sampling_result = _module
transforms = _module
evaluation = _module
bbox_overlaps = _module
class_names = _module
coco_utils = _module
eval_hooks = _module
mean_ap = _module
recall = _module
fp16 = _module
decorators = _module
hooks = _module
utils = _module
mask = _module
mask_target = _module
post_processing = _module
bbox_nms = _module
merge_augs = _module
dist_utils = _module
datasets = _module
builder = _module
cityscapes = _module
coco = _module
custom = _module
dataset_wrappers = _module
loader = _module
build_loader = _module
sampler = _module
pipelines = _module
compose = _module
formating = _module
loading = _module
test_aug = _module
registry = _module
transforms = _module
utils = _module
voc = _module
wider_face = _module
xml_style = _module
models = _module
anchor_heads = _module
anchor_head = _module
fcos_head = _module
ga_retina_head = _module
ga_rpn_head = _module
guided_anchor_head = _module
retina_head = _module
rpn_head = _module
ssd_head = _module
backbones = _module
detnas = _module
dropblock = _module
fbnet = _module
fbnet_arch = _module
fbnet_blocks = _module
hrnet = _module
mnasnet = _module
mobilenetv2 = _module
resnet = _module
resnext = _module
ssd_vgg = _module
utils = _module
bbox_heads = _module
auto_head = _module
build_head = _module
mbblock_head_search = _module
mbblock_ops = _module
bbox_head = _module
convfc_bbox_head = _module
double_bbox_head = _module
builder = _module
detectors = _module
base = _module
cascade_rcnn = _module
double_head_rcnn = _module
fast_rcnn = _module
faster_rcnn = _module
fcos = _module
grid_rcnn = _module
htc = _module
mask_rcnn = _module
mask_scoring_rcnn = _module
retinanet = _module
rpn = _module
single_stage = _module
test_mixins = _module
two_stage = _module
losses = _module
accuracy = _module
balanced_l1_loss = _module
cross_entropy_loss = _module
focal_loss = _module
ghm_loss = _module
iou_loss = _module
mse_loss = _module
smooth_l1_loss = _module
utils = _module
mask_heads = _module
fcn_mask_head = _module
fused_semantic_head = _module
grid_head = _module
htc_mask_head = _module
maskiou_head = _module
necks = _module
auto_neck = _module
build_neck = _module
hit_neck_search = _module
hit_ops = _module
bfp = _module
fpn = _module
fpn_panet = _module
hrfpn = _module
nas_fpn = _module
search_pafpn = _module
plugins = _module
generalized_attention = _module
non_local = _module
roi_extractors = _module
single_level = _module
shared_heads = _module
res_layer = _module
conv_module = _module
conv_ws = _module
norm = _module
quant_conv = _module
scale = _module
weight_init = _module
ops = _module
dcn = _module
functions = _module
deform_conv = _module
deform_pool = _module
modules = _module
deform_conv = _module
deform_pool = _module
setup = _module
gcb = _module
context_block = _module
masked_conv = _module
masked_conv = _module
masked_conv = _module
setup = _module
nms = _module
nms_wrapper = _module
setup = _module
roi_align = _module
roi_align = _module
gradcheck = _module
roi_align = _module
roi_align = _module
setup = _module
roi_pool = _module
roi_pool = _module
gradcheck = _module
roi_pool = _module
setup = _module
sigmoid_focal_loss = _module
sigmoid_focal_loss = _module
sigmoid_focal_loss = _module
setup = _module
collect_env = _module
contextmanagers = _module
flops_counter = _module
profiling = _module
registry = _module
util_mixins = _module
test = _module
analyze_logs = _module
coco_eval = _module
pascal_voc = _module
detectron2pytorch = _module
get_flops = _module
publish_model = _module
test = _module
upgrade_model_version = _module
voc_eval = _module
train = _module

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


import logging


import torch.nn as nn


import torch.utils.checkpoint as cp


import torch


from torch.nn.parallel._functions import _get_stream


import collections


import torch.nn.functional as F


from torch.utils.data.dataloader import default_collate


import functools


from torch.nn.parallel import DataParallel


import torch.distributed as dist


from torch._utils import _flatten_dense_tensors


from torch._utils import _take_tensors


from torch._utils import _unflatten_dense_tensors


from torch.nn.parallel._functions import Scatter as OrigScatter


import time


import warnings


from collections import OrderedDict


import torchvision


from torch.utils import model_zoo


import torch.multiprocessing as mp


import numpy as np


from torch.nn.utils import clip_grad


import math


import random


import matplotlib.pyplot as plt


import re


from abc import ABCMeta


from abc import abstractmethod


from torch.utils.data import Dataset


from inspect import getfullargspec


import copy


from collections import abc


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


from functools import partial


import torch.utils.data.sampler as _sampler


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler as _DistributedSampler


from torch.utils.data import Sampler


from collections.abc import Sequence


from collections import Sequence


from torch.nn.modules.batchnorm import _BatchNorm


from torch import nn


from torch.autograd import Variable


from torch.nn import functional as F


from torch.nn.modules.utils import _pair


from torch.utils.checkpoint import checkpoint


from torch.autograd import Function


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd import gradcheck


from torch.nn.modules.module import Module


from torch.autograd.function import once_differentiable


from collections import defaultdict


from typing import List


from torch.nn.modules.conv import _ConvNd


from torch.nn.modules.conv import _ConvTransposeMixin


from torch.nn.modules.pooling import _AdaptiveAvgPoolNd


from torch.nn.modules.pooling import _AdaptiveMaxPoolNd


from torch.nn.modules.pooling import _AvgPoolNd


from torch.nn.modules.pooling import _MaxPoolNd


import inspect


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.
    Args:
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    own_state = module.state_dict()
    state_dict_modify = state_dict.copy()
    for name, param in state_dict.items():
        """ for mobilenet v2
        if 'features' in name:
            name = name.replace('features.','features')
        """
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if 'conv2' in name and 'layer4.0.conv2_d2.weight' in own_state.keys():
            d1 = name.replace('conv2', 'conv2_d1')
            d1_c = own_state[d1].size(0)
            own_state[d1].copy_(param[:d1_c, :, :, :])
            state_dict_modify[d1] = param[:d1_c, :, :, :]
            d2 = name.replace('conv2', 'conv2_d2')
            d2_c = own_state[d2].size(0)
            own_state[d2].copy_(param[d1_c:d1_c + d2_c, :, :, :])
            state_dict_modify[d2] = param[d1_c:d1_c + d2_c, :, :, :]
            d3 = name.replace('conv2', 'conv2_d3')
            own_state[d3].copy_(param[d1_c + d2_c:, :, :, :])
            state_dict_modify[d3] = param[d1_c + d2_c:, :, :, :]
        else:
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
    missing_keys = set(own_state.keys()) - set(state_dict_modify.keys())
    """
    if 'layer4.0.conv2_d2.weight' in own_state.keys():
        missing_keys = set(own_state.keys()) - set(state_dict_modify.keys())
    else:
        # for mobilenetv2
        own_state_set = []
        for name in set(own_state.keys()):
            own_state_set.append(name.replace('features','features.'))
        missing_keys = set(own_state_set) - set(state_dict.keys())
    """
    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
    err_msg = '\n'.join(err_msg)
    if err_msg:
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warn(err_msg)
        else:
            None


def load_checkpoint(model, filename, strict=False, logger=None):
    checkpoint = torch.load(filename)
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('No state_dict found in checkpoint file {}'.format(filename))
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class AlexNet(nn.Module):
    """AlexNet backbone.

    Args:
        num_classes (int): number of classes for classification.
    """

    def __init__(self, num_classes=-1):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))
        if self.num_classes > 0:
            self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Linear(4096, num_classes))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.features(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifier(x)
        return x


def conv_ws_2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, eps=1e-05):
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class ConvWS2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, eps=1e-05):
        super(ConvWS2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x):
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.eps)


class BirealActivation(Function):
    """
    take a real value x
    output sign(x)
    """

    @staticmethod
    def forward(ctx, input, nbit_a=1):
        ctx.save_for_backward(input)
        return input.clamp(-1, 1).sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = (2 + 2 * input) * input.lt(0).float() + (2 - 2 * input) * input.ge(0).float()
        grad_input = torch.clamp(grad_input, 0)
        grad_input *= grad_output
        return grad_input, None


def bireal_a(input, nbit_a=1, *args, **kwargs):
    return BirealActivation.apply(input)


class Signer(Function):
    """
    take a real value x
    output sign(x)
    """

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def sign(input):
    return Signer.apply(input)


def bireal_w(w, nbit_w=1, *args, **kwargs):
    if nbit_w != 1:
        raise ValueError('nbit_w must be 1 in Bi-Real-Net.')
    return sign(w) * torch.mean(torch.abs(w.clone().detach()))


class Quantizer(Function):
    """
    take a real value x in alpha*[0,1] or alpha*[-1,1]
    output a discrete-valued x in alpha*{0, 1/(2^k-1), ..., (2^k-1)/(2^k-1)} or likeness
    where k is nbit
    """

    @staticmethod
    def forward(ctx, input, nbit, alpha=None, offset=None):
        ctx.alpha = alpha
        ctx.offset = offset
        scale = 2 ** nbit - 1 if alpha is None else (2 ** nbit - 1) / alpha
        ctx.scale = scale
        return torch.round(input * scale) / scale if offset is None else (torch.round(input * scale) + torch.round(offset)) / scale

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.offset is None:
            return grad_output, None, None, None
        else:
            return grad_output, None, None, torch.sum(grad_output) / ctx.scale


def quantize(input, nbit, alpha=None, offset=None):
    return Quantizer.apply(input, nbit, alpha, offset)


def dorefa_a(input, nbit_a, *args, **kwargs):
    return quantize(torch.clamp(input, 0, 1.0), nbit_a, *args, **kwargs)


class ScaleSigner(Function):
    """
    take a real value x
    output sign(x) * E(|x|)
    """

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def scale_sign(input):
    return ScaleSigner.apply(input)


def dorefa_w(w, nbit_w, *args, **kwargs):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = torch.tanh(w)
        w = w / (2 * torch.max(torch.abs(w))) + 0.5
        w = 2 * quantize(w, nbit_w) - 1
    return w


def pact_a(input, nbit_a, alpha, *args, **kwargs):
    x = 0.5 * (torch.abs(input) - torch.abs(input - alpha) + alpha)
    return quantize(x, nbit_a, alpha, *args, **kwargs)


def wrpn_w(w, nbit_w, *args, **kwargs):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = quantize(torch.clamp(w, -1, 1), nbit_w - 1)
    return w


class Xnor(Function):
    """
    take a real value x
    output sign(x_c) * E(|x_c|)
    """

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input), dim=[1, 2, 3], keepdim=True)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def xnor(input):
    return Xnor.apply(input)


def xnor_w(w, nbit_w=1, *args, **kwargs):
    if nbit_w != 1:
        raise ValueError('nbit_w must be 1 in XNOR-Net.')
    return xnor(w)


class QuantConv(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QuantConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_custome_parameters()
        self.quant_config()

    def quant_config(self, quant_name_w='dorefa', quant_name_a='dorefa', nbit_w=1, nbit_a=1, has_offset=False):
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': dorefa_w, 'pact': dorefa_w, 'wrpn': wrpn_w, 'xnor': xnor_w, 'bireal': bireal_w}
        name_a_dict = {'dorefa': dorefa_a, 'pact': pact_a, 'wrpn': dorefa_a, 'xnor': dorefa_a, 'bireal': bireal_a}
        self.quant_w = name_w_dict[quant_name_w]
        self.quant_a = name_a_dict[quant_name_a]
        if quant_name_a == 'pact':
            self.alpha_a = nn.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_parameter('alpha_a', None)
        if quant_name_w == 'pact':
            self.alpha_w = nn.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_parameter('alpha_w', None)
        if has_offset:
            self.offset = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('offset', None)
        if self.alpha_a is not None:
            nn.init.constant_(self.alpha_a, 10)
        if self.alpha_w is not None:
            nn.init.constant_(self.alpha_w, 10)
        if self.offset is not None:
            nn.init.constant_(self.offset, 0)

    def reset_custome_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        if self.nbit_w == 0 or self.nbit_a == 0:
            diff_channels = self.out_channels - self.in_channels
            if self.stride == 2 or self.stride == (2, 2):
                x = F.pad(input[:, :, ::2, ::2], (0, 0, 0, 0, diff_channels // 2, diff_channels - diff_channels // 2), 'constant', 0)
                return x
            else:
                x = F.pad(input, (0, 0, 0, 0, diff_channels // 2, diff_channels - diff_channels // 2), 'constant', 0)
                return x
        if self.nbit_w < 32:
            w = self.quant_w(self.weight, self.nbit_w, self.alpha_w, self.offset)
        else:
            w = self.weight
        if self.nbit_a < 32:
            x = self.quant_a(input, self.nbit_a, self.alpha_a)
        else:
            x = F.relu(input)
        x = F.conv2d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
        return x


conv_cfg = {'Conv': nn.Conv2d, 'ConvWS': ConvWS2d, 'QuantConv': QuantConv}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]
    layer = conv_layer(*args, **kwargs, **cfg_)
    return layer


norm_cfg = {'BN': ('bn', nn.BatchNorm2d), 'SyncBN': ('bn', nn.SyncBatchNorm), 'GN': ('gn', nn.GroupNorm)}


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-05)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return name, layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=3, dilation=1, downsample=None, style='pytorch', with_cp=False, conv2_split=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, gcb=None, gen_attention=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert gen_attention is None, 'Not implemented yet.'
        assert gcb is None, 'Not implemented yet.'
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(input, offset, weight)
        output = input.new_empty(DeformConvFunction._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride))
        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            deform_conv_cuda.deform_conv_forward_cuda(input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_cuda.deform_conv_backward_input_cuda(input, offset, grad_output, grad_input, grad_offset, weight, ctx.bufs_[0], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_cuda.deform_conv_backward_parameters_cuda(input, offset, grad_output, grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1, cur_im2col_step)
        return grad_input, grad_offset, grad_weight, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
        return output_size


deform_conv = DeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()
        assert not bias
        assert in_channels % groups == 0, 'in_channels {} cannot be divisible by groups {}'.format(in_channels, groups)
        assert out_channels % groups == 0, 'out_channels {} cannot be divisible by groups {}'.format(out_channels, groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        deform_conv_cuda.modulated_deform_conv_cuda_forward(input, weight, bias, ctx._bufs[0], offset, mask, output, ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        deform_conv_cuda.modulated_deform_conv_cuda_backward(input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None, None

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = ModulatedDeformConvFunction.apply


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


BACKBONES = Registry('backbone')


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def make_res_layer(block, inplanes, planes, blocks, stride=1, dilation=1, groups=1, base_width=4, style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, gcb=None):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(build_conv_layer(conv_cfg, inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), build_norm_layer(norm_cfg, planes * block.expansion)[1])
    layers = []
    layers.append(block(inplanes=inplanes, planes=planes, stride=stride, dilation=dilation, downsample=downsample, groups=groups, base_width=base_width, style=style, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, dcn=dcn, gcb=gcb))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes=inplanes, planes=planes, stride=1, dilation=dilation, groups=groups, base_width=base_width, style=style, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, dcn=dcn, gcb=gcb))
    return nn.Sequential(*layers)


def conv3x3(in_planes, out_planes, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation)


def make_vgg_layer(inplanes, planes, num_blocks, dilation=1, with_bn=False, ceil_mode=False):
    layers = []
    for _ in range(num_blocks):
        layers.append(conv3x3(inplanes, planes, dilation))
        if with_bn:
            layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        inplanes = planes
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))
    return layers


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


class VGG(nn.Module):
    """VGG backbone.

    Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        with_bn (bool): Use BatchNorm or not.
        num_classes (int): number of classes for classification.
        num_stages (int): VGG stages, normally 5.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers as eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
    """
    arch_settings = {(11): (1, 1, 2, 2, 2), (13): (2, 2, 2, 2, 2), (16): (2, 2, 3, 3, 3), (19): (2, 2, 4, 4, 4)}

    def __init__(self, depth, with_bn=False, num_classes=-1, num_stages=5, dilations=(1, 1, 1, 1, 1), out_indices=(0, 1, 2, 3, 4), frozen_stages=-1, bn_eval=True, bn_frozen=False, ceil_mode=False, with_last_pool=True):
        super(VGG, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for vgg'.format(depth))
        assert num_stages >= 1 and num_stages <= 5
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        assert len(dilations) == num_stages
        assert max(out_indices) <= num_stages
        self.num_classes = num_classes
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.inplanes = 3
        start_idx = 0
        vgg_layers = []
        self.range_sub_modules = []
        for i, num_blocks in enumerate(self.stage_blocks):
            num_modules = num_blocks * (2 + with_bn) + 1
            end_idx = start_idx + num_modules
            dilation = dilations[i]
            planes = 64 * 2 ** i if i < 4 else 512
            vgg_layer = make_vgg_layer(self.inplanes, planes, num_blocks, dilation=dilation, with_bn=with_bn, ceil_mode=ceil_mode)
            vgg_layers.extend(vgg_layer)
            self.inplanes = planes
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx
        if not with_last_pool:
            vgg_layers.pop(-1)
            self.range_sub_modules[-1][1] -= 1
        self.module_name = 'features'
        self.add_module(self.module_name, nn.Sequential(*vgg_layers))
        if self.num_classes > 0:
            self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i, num_blocks in enumerate(self.stage_blocks):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(VGG, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        vgg_layers = getattr(self, self.module_name)
        if mode and self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                for j in range(*self.range_sub_modules[i]):
                    mod = vgg_layers[j]
                    mod.eval()
                    for param in mod.parameters():
                        param.requires_grad = False


def assert_tensor_type(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError('{} has no attribute {} for type {}'.format(args[0].__class__.__name__, func.__name__, args[0].datatype))
        return func(*args, **kwargs)
    return wrapper


class DataContainer(object):
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self, data, stack=False, padding_value=0, cpu_only=False, pad_dims=2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self.data))

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def pad_dims(self):
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()


def get_input_device(input):
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, torch.Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception('Unknown type {}.'.format(type(input)))


def synchronize_stream(output, devices, streams):
    if isinstance(output, list):
        chunk_size = len(output) // len(devices)
        for i in range(len(devices)):
            for j in range(chunk_size):
                synchronize_stream(output[i * chunk_size + j], [devices[i]], [streams[i]])
    elif isinstance(output, torch.Tensor):
        if output.numel() != 0:
            with torch.device(devices[0]):
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)
    else:
        raise Exception('Unknown type {}.'.format(type(output)))


class Scatter(object):

    @staticmethod
    def forward(target_gpus, input):
        input_device = get_input_device(input)
        streams = None
        if input_device == -1:
            streams = [_get_stream(device) for device in target_gpus]
        outputs = scatter(input, target_gpus, streams)
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)
        return tuple(outputs)


def scatter(inputs, target_gpus, dim=0):
    """Scatter inputs to target gpus.

    The only difference from original :func:`scatter` is to add support for
    :type:`~mmcv.parallel.DataContainer`.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return OrigScatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, DataContainer):
            if obj.cpu_only:
                return obj.data
            else:
                return Scatter.forward(target_gpus, obj.data)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj for targets in target_gpus]
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class MMDataParallel(DataParallel):

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)


class MMDistributedDataParallel(nn.Module):

    def __init__(self, module, dim=0, broadcast_buffers=True, bucket_cap_mb=25):
        super(MMDistributedDataParallel, self).__init__()
        self.module = module
        self.dim = dim
        self.broadcast_buffers = broadcast_buffers
        self.broadcast_bucket_size = bucket_cap_mb * 1024 * 1024
        self._sync_params()

    def _dist_broadcast_coalesced(self, tensors, buffer_size):
        for tensors in _take_tensors(tensors, buffer_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.broadcast(flat_tensors, 0)
            for tensor, synced in zip(tensors, _unflatten_dense_tensors(flat_tensors, tensors)):
                tensor.copy_(synced)

    def _sync_params(self):
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            self._dist_broadcast_coalesced(module_states, self.broadcast_bucket_size)
        if self.broadcast_buffers:
            if torch.__version__ < '1.0':
                buffers = [b.data for b in self.module._all_buffers()]
            else:
                buffers = [b.data for b in self.module.buffers()]
            if len(buffers) > 0:
                self._dist_broadcast_coalesced(buffers, self.broadcast_bucket_size)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, [torch.cuda.current_device()])
        return self.module(*inputs[0], **kwargs[0])


class AnchorGenerator(object):

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr
        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)
        base_anchors = torch.stack([x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)], dim=-1).round()
        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid


HEADS = Registry('head')


class SamplingResult(object):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]
        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])


class BaseSampler(metaclass=ABCMeta):

    def __init__(self, num, pos_fraction, neg_pos_ub=-1, add_gt_as_proposals=True, **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        pass

    def sample(self, assign_result, bboxes, gt_bboxes, gt_labels=None, **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        bboxes = bboxes[:, :4]
        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])
        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()
        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)


class PseudoSampler(BaseSampler):

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
        return sampling_result


def anchor_inside_flags(flat_anchors, valid_flags, img_shape, allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & (flat_anchors[:, 0] >= -allowed_border) & (flat_anchors[:, 1] >= -allowed_border) & (flat_anchors[:, 2] < img_w + allowed_border) & (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def build_assigner(cfg, **kwargs):
    if isinstance(cfg, assigners.BaseAssigner):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, assigners, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(type(cfg)))


def build_sampler(cfg, **kwargs):
    if isinstance(cfg, samplers.BaseSampler):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, samplers, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(type(cfg)))


def assign_and_sample(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels, cfg):
    bbox_assigner = build_assigner(cfg.assigner)
    bbox_sampler = build_sampler(cfg.sampler)
    assign_result = bbox_assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels)
    sampling_result = bbox_sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels)
    return assign_result, sampling_result


def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    assert proposals.size() == gt.size()
    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0
    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret


def anchor_target_single(flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, target_means, target_stds, cfg, label_channels=1, sampling=True, unmap_outputs=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], cfg.allowed_border)
    if not inside_flags.any():
        return (None,) * 6
    anchors = flat_anchors[inside_flags, :]
    if sampling:
        assign_result, sampling_result = assign_and_sample(anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes, gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors, gt_bboxes)
    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes, target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
    return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def anchor_target(anchor_list, valid_flag_list, gt_bboxes_list, img_metas, target_means, target_stds, cfg, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, sampling=True, unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(anchor_target_single, anchor_list, valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, target_means=target_means, target_stds=target_stds, cfg=cfg, label_channels=label_channels, sampling=sampling, unmap_outputs=unmap_outputs)
    if any([(labels is None) for labels in all_labels]):
        return None
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg


LOSSES = Registry('loss')


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if mmcv.is_str(obj_type):
        obj_type = registry.get(obj_type)
        if obj_type is None:
            raise KeyError('{} is not in the {} registry'.format(obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_loss(cfg):
    return build(cfg, LOSSES)


def delta2bbox(rois, deltas, means=[0, 0, 0, 0], stds=[1, 1, 1, 1], max_shape=None, wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)
    gy = torch.addcmul(py, 1, ph, dy)
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


def cast_tensor_type(inputs, src_type, dst_type):
    if isinstance(inputs, torch.Tensor):
        return inputs
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({k: cast_tensor_type(v, src_type, dst_type) for k, v in inputs.items()})
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


def force_fp32(apply_to=None, out_fp16=False):
    """Decorator to convert input arguments to fp32 in force.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If there are some inputs that must be processed
    in fp32 mode, then this decorator can handle it. If inputs arguments are
    fp16 tensors, they will be converted to fp32 automatically. Arguments other
    than fp16 tensors are ignored.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp16 (bool): Whether to convert the output back to fp16.

    :Example:

        class MyModule1(nn.Module)

            # Convert x and y to fp32
            @force_fp32()
            def loss(self, x, y):
                pass

        class MyModule2(nn.Module):

            # convert pred to fp32
            @force_fp32(apply_to=('pred', ))
            def post_process(self, pred, others):
                pass
    """

    def force_fp32_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@force_fp32 can only be used to decorate the method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            args_info = getfullargspec(old_func)
            args_to_cast = args_info.args if apply_to is None else apply_to
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(arg_value, torch.half, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value
            output = old_func(*new_args, **new_kwargs)
            if out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output
        return new_func
    return force_fp32_wrapper


def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0],), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
    return bboxes, labels


class AnchorHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """

    def __init__(self, num_classes, in_channels, feat_channels=256, anchor_scales=[8, 16, 32], anchor_ratios=[0.5, 1.0, 2.0], anchor_strides=[4, 8, 16, 32, 64], anchor_base_sizes=None, target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0), loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.fp16_enabled = False
        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))
        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self._init_layers()

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, num_total_samples, cfg):
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, cfg, gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(anchor_list, valid_flag_list, gt_bboxes, img_metas, self.target_means, self.target_stds, cfg, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels, sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos
        losses_cls, losses_bbox = multi_apply(self.loss_single, cls_scores, bbox_preds, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_samples=num_total_samples, cfg=cfg)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg, rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        mlvl_anchors = [self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:], self.anchor_strides[i]) for i in range(num_levels)]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, mlvl_anchors, img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means, self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels


class ConvModule(nn.Module):
    """Conv-Norm-Activation block.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str or None): Activation type, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        bottle_first (bool): Whether to apply the activation layer in the
            last. (Do not use this flag since the behavior and api may be
            changed in the future.)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias='auto', conv_cfg=None, norm_cfg=None, activation='relu', inplace=True, bottle_first='conv'):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.bottle_first = bottle_first
        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias
        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')
        self.conv = build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups
        if self.with_norm:
            norm_channels = in_channels if self.bottle_first == 'bn' else out_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        if self.with_activatation:
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        nonlinearity = 'relu' if self.activation is None else self.activation
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        if self.bottle_first == 'conv':
            x = self.conv(x)
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activatation:
                x = self.activate(x)
        elif self.bottle_first == 'relu':
            if activate and self.with_activatation:
                x = self.activate(x)
            x = self.conv(x)
            if norm and self.with_norm:
                x = self.norm(x)
        elif self.bottle_first == 'bn':
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activatation:
                x = self.activate(x)
            x = self.conv(x)
        else:
            raise KeyError('bottle_first is invalid.')
        return x


INF = 100000000.0


class Scale(nn.Module):

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


def bias_init_with_prob(prior_prob):
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    return torch.stack([x1, y1, x2, y2], -1)


class FCOSHead(nn.Module):

    def __init__(self, num_classes, in_channels, feat_channels=256, stacked_convs=4, strides=(4, 8, 16, 32, 64), regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)), loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), loss_bbox=dict(type='IoULoss', loss_weight=1.0), loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), conv_cfg=None, norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(FCOSHead, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None))
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)
        centerness = self.fcos_centerness(cls_feat)
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = scale(self.fcos_reg(reg_feat)).float().exp()
        return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self, cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels, img_metas, cfg, gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        labels, bbox_targets = self.fcos_target(all_level_points, gt_bboxes, gt_labels)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_centerness = [centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])
        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos + num_imgs)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        if num_pos > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(pos_decoded_bbox_preds, pos_decoded_target_preds, weight=pos_centerness_targets, avg_factor=pos_centerness_targets.sum())
            loss_centerness = self.loss_centerness(pos_centerness, pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_centerness=loss_centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self, cls_scores, bbox_preds, centernesses, img_metas, cfg, rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            centerness_pred_list = [centernesses[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list, centerness_pred_list, mlvl_points, img_shape, scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self, cls_scores, bbox_preds, centernesses, mlvl_points, img_shape, scale_factor, cfg, rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img, score_factors=mlvl_centerness)
        return det_bboxes, det_labels

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(self.get_points_single(featmap_sizes[i], self.strides[i], dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def fcos_target(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i]) for i in range(num_levels)]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        labels_list, bbox_targets_list = multi_apply(self.fcos_target_single, gt_bboxes_list, gt_labels_list, points=concat_points, regress_ranges=concat_regress_ranges)
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list]
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets

    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (max_regress_distance >= regress_ranges[..., 0]) & (max_regress_distance <= regress_ranges[..., 1])
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0] * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)


class FeatureAdaption(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(2, deformable_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x


class MaskedConv2dFunction(Function):

    @staticmethod
    def forward(ctx, features, mask, weight, bias, padding=0, stride=1):
        assert mask.dim() == 3 and mask.size(0) == 1
        assert features.dim() == 4 and features.size(0) == 1
        assert features.size()[2:] == mask.size()[1:]
        pad_h, pad_w = _pair(padding)
        stride_h, stride_w = _pair(stride)
        if stride_h != 1 or stride_w != 1:
            raise ValueError('Stride could not only be 1 in masked_conv2d currently.')
        if not features.is_cuda:
            raise NotImplementedError
        out_channel, in_channel, kernel_h, kernel_w = weight.size()
        batch_size = features.size(0)
        out_h = int(math.floor((features.size(2) + 2 * pad_h - (kernel_h - 1) - 1) / stride_h + 1))
        out_w = int(math.floor((features.size(3) + 2 * pad_w - (kernel_h - 1) - 1) / stride_w + 1))
        mask_inds = torch.nonzero(mask[0] > 0)
        output = features.new_zeros(batch_size, out_channel, out_h, out_w)
        if mask_inds.numel() > 0:
            mask_h_idx = mask_inds[:, 0].contiguous()
            mask_w_idx = mask_inds[:, 1].contiguous()
            data_col = features.new_zeros(in_channel * kernel_h * kernel_w, mask_inds.size(0))
            masked_conv2d_cuda.masked_im2col_forward(features, mask_h_idx, mask_w_idx, kernel_h, kernel_w, pad_h, pad_w, data_col)
            masked_output = torch.addmm(1, bias[:, None], 1, weight.view(out_channel, -1), data_col)
            masked_conv2d_cuda.masked_col2im_forward(masked_output, mask_h_idx, mask_w_idx, out_h, out_w, out_channel, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return (None,) * 5


masked_conv2d = MaskedConv2dFunction.apply


class MaskedConv2d(nn.Conv2d):
    """A MaskedConv2d which inherits the official Conv2d.

    The masked forward doesn't implement the backward function and only
    supports the stride parameter to be 1 currently.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input, mask=None):
        if mask is None:
            return super(MaskedConv2d, self).forward(input)
        else:
            return masked_conv2d(input, mask, self.weight, self.bias, self.padding)


def calc_region(bbox, ratio, featmap_size=None):
    """Calculate a proportional bbox region.

    The bbox center are fixed and the new h' and w' is h * ratio and w * ratio.

    Args:
        bbox (Tensor): Bboxes to calculate regions, shape (n, 4)
        ratio (float): Ratio of the output region.
        featmap_size (tuple): Feature map size used for clipping the boundary.

    Returns:
        tuple: x1, y1, x2, y2
    """
    x1 = torch.round((1 - ratio) * bbox[0] + ratio * bbox[2]).long()
    y1 = torch.round((1 - ratio) * bbox[1] + ratio * bbox[3]).long()
    x2 = torch.round(ratio * bbox[0] + (1 - ratio) * bbox[2]).long()
    y2 = torch.round(ratio * bbox[1] + (1 - ratio) * bbox[3]).long()
    if featmap_size is not None:
        x1 = x1.clamp(min=0, max=featmap_size[1] - 1)
        y1 = y1.clamp(min=0, max=featmap_size[0] - 1)
        x2 = x2.clamp(min=0, max=featmap_size[1] - 1)
        y2 = y2.clamp(min=0, max=featmap_size[0] - 1)
    return x1, y1, x2, y2


def ga_loc_target(gt_bboxes_list, featmap_sizes, anchor_scale, anchor_strides, center_ratio=0.2, ignore_ratio=0.5):
    """Compute location targets for guided anchoring.

    Each feature map is divided into positive, negative and ignore regions.
    - positive regions: target 1, weight 1
    - ignore regions: target 0, weight 0
    - negative regions: target 0, weight 0.1

    Args:
        gt_bboxes_list (list[Tensor]): Gt bboxes of each image.
        featmap_sizes (list[tuple]): Multi level sizes of each feature maps.
        anchor_scale (int): Anchor scale.
        anchor_strides ([list[int]]): Multi level anchor strides.
        center_ratio (float): Ratio of center region.
        ignore_ratio (float): Ratio of ignore region.

    Returns:
        tuple
    """
    img_per_gpu = len(gt_bboxes_list)
    num_lvls = len(featmap_sizes)
    r1 = (1 - center_ratio) / 2
    r2 = (1 - ignore_ratio) / 2
    all_loc_targets = []
    all_loc_weights = []
    all_ignore_map = []
    for lvl_id in range(num_lvls):
        h, w = featmap_sizes[lvl_id]
        loc_targets = torch.zeros(img_per_gpu, 1, h, w, device=gt_bboxes_list[0].device, dtype=torch.float32)
        loc_weights = torch.full_like(loc_targets, -1)
        ignore_map = torch.zeros_like(loc_targets)
        all_loc_targets.append(loc_targets)
        all_loc_weights.append(loc_weights)
        all_ignore_map.append(ignore_map)
    for img_id in range(img_per_gpu):
        gt_bboxes = gt_bboxes_list[img_id]
        scale = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1))
        min_anchor_size = scale.new_full((1,), float(anchor_scale * anchor_strides[0]))
        target_lvls = torch.floor(torch.log2(scale) - torch.log2(min_anchor_size) + 0.5)
        target_lvls = target_lvls.clamp(min=0, max=num_lvls - 1).long()
        for gt_id in range(gt_bboxes.size(0)):
            lvl = target_lvls[gt_id].item()
            gt_ = gt_bboxes[gt_id, :4] / anchor_strides[lvl]
            ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(gt_, r2, featmap_sizes[lvl])
            ctr_x1, ctr_y1, ctr_x2, ctr_y2 = calc_region(gt_, r1, featmap_sizes[lvl])
            all_loc_targets[lvl][img_id, 0, ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1
            all_loc_weights[lvl][img_id, 0, ignore_y1:ignore_y2 + 1, ignore_x1:ignore_x2 + 1] = 0
            all_loc_weights[lvl][img_id, 0, ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1
            if lvl > 0:
                d_lvl = lvl - 1
                gt_ = gt_bboxes[gt_id, :4] / anchor_strides[d_lvl]
                ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(gt_, r2, featmap_sizes[d_lvl])
                all_ignore_map[d_lvl][img_id, 0, ignore_y1:ignore_y2 + 1, ignore_x1:ignore_x2 + 1] = 1
            if lvl < num_lvls - 1:
                u_lvl = lvl + 1
                gt_ = gt_bboxes[gt_id, :4] / anchor_strides[u_lvl]
                ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(gt_, r2, featmap_sizes[u_lvl])
                all_ignore_map[u_lvl][img_id, 0, ignore_y1:ignore_y2 + 1, ignore_x1:ignore_x2 + 1] = 1
    for lvl_id in range(num_lvls):
        all_loc_weights[lvl_id][(all_loc_weights[lvl_id] < 0) & (all_ignore_map[lvl_id] > 0)] = 0
        all_loc_weights[lvl_id][all_loc_weights[lvl_id] < 0] = 0.1
    loc_avg_factor = sum([(t.size(0) * t.size(-1) * t.size(-2)) for t in all_loc_targets]) / 200
    return all_loc_targets, all_loc_weights, loc_avg_factor


def ga_shape_target_single(flat_approxs, inside_flags, flat_squares, gt_bboxes, gt_bboxes_ignore, img_meta, approxs_per_octave, cfg, sampling=True, unmap_outputs=True):
    """Compute guided anchoring targets.

    This function returns sampled anchors and gt bboxes directly
    rather than calculates regression targets.

    Args:
        flat_approxs (Tensor): flat approxs of a single image,
            shape (n, 4)
        inside_flags (Tensor): inside flags of a single image,
            shape (n, ).
        flat_squares (Tensor): flat squares of a single image,
            shape (approxs_per_octave * n, 4)
        gt_bboxes (Tensor): Ground truth bboxes of a single image.
        img_meta (dict): Meta info of a single image.
        approxs_per_octave (int): number of approxs per octave
        cfg (dict): RPN train configs.
        sampling (bool): sampling or not.
        unmap_outputs (bool): unmap outputs or not.

    Returns:
        tuple
    """
    if not inside_flags.any():
        return (None,) * 6
    expand_inside_flags = inside_flags[:, None].expand(-1, approxs_per_octave).reshape(-1)
    approxs = flat_approxs[expand_inside_flags, :]
    squares = flat_squares[inside_flags, :]
    bbox_assigner = build_assigner(cfg.ga_assigner)
    assign_result = bbox_assigner.assign(approxs, squares, approxs_per_octave, gt_bboxes, gt_bboxes_ignore)
    if sampling:
        bbox_sampler = build_sampler(cfg.ga_sampler)
    else:
        bbox_sampler = PseudoSampler()
    sampling_result = bbox_sampler.sample(assign_result, squares, gt_bboxes)
    bbox_anchors = torch.zeros_like(squares)
    bbox_gts = torch.zeros_like(squares)
    bbox_weights = torch.zeros_like(squares)
    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        bbox_anchors[pos_inds, :] = sampling_result.pos_bboxes
        bbox_gts[pos_inds, :] = sampling_result.pos_gt_bboxes
        bbox_weights[pos_inds, :] = 1.0
    if unmap_outputs:
        num_total_anchors = flat_squares.size(0)
        bbox_anchors = unmap(bbox_anchors, num_total_anchors, inside_flags)
        bbox_gts = unmap(bbox_gts, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
    return bbox_anchors, bbox_gts, bbox_weights, pos_inds, neg_inds


def ga_shape_target(approx_list, inside_flag_list, square_list, gt_bboxes_list, img_metas, approxs_per_octave, cfg, gt_bboxes_ignore_list=None, sampling=True, unmap_outputs=True):
    """Compute guided anchoring targets.

    Args:
        approx_list (list[list]): Multi level approxs of each image.
        inside_flag_list (list[list]): Multi level inside flags of each image.
        square_list (list[list]): Multi level squares of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        approxs_per_octave (int): number of approxs per octave
        cfg (dict): RPN train configs.
        gt_bboxes_ignore_list (list[Tensor]): ignore list of gt bboxes.
        sampling (bool): sampling or not.
        unmap_outputs (bool): unmap outputs or not.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(approx_list) == len(inside_flag_list) == len(square_list) == num_imgs
    num_level_squares = [squares.size(0) for squares in square_list[0]]
    inside_flag_flat_list = []
    approx_flat_list = []
    square_flat_list = []
    for i in range(num_imgs):
        assert len(square_list[i]) == len(inside_flag_list[i])
        inside_flag_flat_list.append(torch.cat(inside_flag_list[i]))
        approx_flat_list.append(torch.cat(approx_list[i]))
        square_flat_list.append(torch.cat(square_list[i]))
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    all_bbox_anchors, all_bbox_gts, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(ga_shape_target_single, approx_flat_list, inside_flag_flat_list, square_flat_list, gt_bboxes_list, gt_bboxes_ignore_list, img_metas, approxs_per_octave=approxs_per_octave, cfg=cfg, sampling=sampling, unmap_outputs=unmap_outputs)
    if any([(bbox_anchors is None) for bbox_anchors in all_bbox_anchors]):
        return None
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    bbox_anchors_list = images_to_levels(all_bbox_anchors, num_level_squares)
    bbox_gts_list = images_to_levels(all_bbox_gts, num_level_squares)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_squares)
    return bbox_anchors_list, bbox_gts_list, bbox_weights_list, num_total_pos, num_total_neg


class GuidedAnchorHead(AnchorHead):
    """Guided-Anchor-based head (GA-RPN, GA-RetinaNet, etc.).

    This GuidedAnchorHead will predict high-quality feature guided
    anchors and locations where anchors will be kept in inference.
    There are mainly 3 categories of bounding-boxes.
    - Sampled (9) pairs for target assignment. (approxes)
    - The square boxes where the predicted anchors are based on.
        (squares)
    - Guided anchors.
    Please refer to https://arxiv.org/abs/1901.03278 for more details.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        octave_base_scale (int): Base octave scale of each level of
            feature map.
        scales_per_octave (int): Number of octave scales in each level of
            feature map
        octave_ratios (Iterable): octave aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        anchoring_means (Iterable): Mean values of anchoring targets.
        anchoring_stds (Iterable): Std values of anchoring targets.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        deformable_groups: (int): Group number of DCN in
            FeatureAdaption module.
        loc_filter_thr (float): Threshold to filter out unconcerned regions.
        loss_loc (dict): Config of location loss.
        loss_shape (dict): Config of anchor shape loss.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of bbox regression loss.
    """

    def __init__(self, num_classes, in_channels, feat_channels=256, octave_base_scale=8, scales_per_octave=3, octave_ratios=[0.5, 1.0, 2.0], anchor_strides=[4, 8, 16, 32, 64], anchor_base_sizes=None, anchoring_means=(0.0, 0.0, 0.0, 0.0), anchoring_stds=(1.0, 1.0, 1.0, 1.0), target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0), deformable_groups=4, loc_filter_thr=0.01, loss_loc=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0), loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.octave_scales = octave_base_scale * np.array([(2 ** (i / scales_per_octave)) for i in range(scales_per_octave)])
        self.approxs_per_octave = len(self.octave_scales) * len(octave_ratios)
        self.octave_ratios = octave_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.anchoring_means = anchoring_means
        self.anchoring_stds = anchoring_stds
        self.target_means = target_means
        self.target_stds = target_stds
        self.deformable_groups = deformable_groups
        self.loc_filter_thr = loc_filter_thr
        self.approx_generators = []
        self.square_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.approx_generators.append(AnchorGenerator(anchor_base, self.octave_scales, self.octave_ratios))
            self.square_generators.append(AnchorGenerator(anchor_base, [self.octave_base_scale], [1.0]))
        self.num_anchors = 1
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.cls_focal_loss = loss_cls['type'] in ['FocalLoss']
        self.loc_focal_loss = loss_loc['type'] in ['FocalLoss']
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes
        self.loss_loc = build_loss(loss_loc)
        self.loss_shape = build_loss(loss_shape)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.conv_loc = nn.Conv2d(self.feat_channels, 1, 1)
        self.conv_shape = nn.Conv2d(self.feat_channels, self.num_anchors * 2, 1)
        self.feature_adaption = FeatureAdaption(self.feat_channels, self.feat_channels, kernel_size=3, deformable_groups=self.deformable_groups)
        self.conv_cls = MaskedConv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = MaskedConv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_loc, std=0.01, bias=bias_cls)
        normal_init(self.conv_shape, std=0.01)
        self.feature_adaption.init_weights()

    def forward_single(self, x):
        loc_pred = self.conv_loc(x)
        shape_pred = self.conv_shape(x)
        x = self.feature_adaption(x, shape_pred)
        if not self.training:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None
        cls_score = self.conv_cls(x, mask)
        bbox_pred = self.conv_reg(x, mask)
        return cls_score, bbox_pred, shape_pred, loc_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_sampled_approxs(self, featmap_sizes, img_metas, cfg):
        """Get sampled approxs and inside flags according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: approxes of each image, inside flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        multi_level_approxs = []
        for i in range(num_levels):
            approxs = self.approx_generators[i].grid_anchors(featmap_sizes[i], self.anchor_strides[i])
            multi_level_approxs.append(approxs)
        approxs_list = [multi_level_approxs for _ in range(num_imgs)]
        inside_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            multi_level_approxs = approxs_list[img_id]
            for i in range(num_levels):
                approxs = multi_level_approxs[i]
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.approx_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
                inside_flags_list = []
                for i in range(self.approxs_per_octave):
                    split_valid_flags = flags[i::self.approxs_per_octave]
                    split_approxs = approxs[i::self.approxs_per_octave, :]
                    inside_flags = anchor_inside_flags(split_approxs, split_valid_flags, img_meta['img_shape'][:2], cfg.allowed_border)
                    inside_flags_list.append(inside_flags)
                inside_flags = torch.stack(inside_flags_list, 0).sum(dim=0) > 0
                multi_level_flags.append(inside_flags)
            inside_flag_list.append(multi_level_flags)
        return approxs_list, inside_flag_list

    def get_anchors(self, featmap_sizes, shape_preds, loc_preds, img_metas, use_loc_filter=False):
        """Get squares according to feature map sizes and guided
        anchors.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            shape_preds (list[tensor]): Multi-level shape predictions.
            loc_preds (list[tensor]): Multi-level location predictions.
            img_metas (list[dict]): Image meta info.
            use_loc_filter (bool): Use loc filter or not.

        Returns:
            tuple: square approxs of each image, guided anchors of each image,
                loc masks of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        multi_level_squares = []
        for i in range(num_levels):
            squares = self.square_generators[i].grid_anchors(featmap_sizes[i], self.anchor_strides[i])
            multi_level_squares.append(squares)
        squares_list = [multi_level_squares for _ in range(num_imgs)]
        guided_anchors_list = []
        loc_mask_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_guided_anchors = []
            multi_level_loc_mask = []
            for i in range(num_levels):
                squares = squares_list[img_id][i]
                shape_pred = shape_preds[i][img_id]
                loc_pred = loc_preds[i][img_id]
                guided_anchors, loc_mask = self.get_guided_anchors_single(squares, shape_pred, loc_pred, use_loc_filter=use_loc_filter)
                multi_level_guided_anchors.append(guided_anchors)
                multi_level_loc_mask.append(loc_mask)
            guided_anchors_list.append(multi_level_guided_anchors)
            loc_mask_list.append(multi_level_loc_mask)
        return squares_list, guided_anchors_list, loc_mask_list

    def get_guided_anchors_single(self, squares, shape_pred, loc_pred, use_loc_filter=False):
        """Get guided anchors and loc masks for a single level.

        Args:
            square (tensor): Squares of a single level.
            shape_pred (tensor): Shape predections of a single level.
            loc_pred (tensor): Loc predections of a single level.
            use_loc_filter (list[tensor]): Use loc filter or not.

        Returns:
            tuple: guided anchors, location masks
        """
        loc_pred = loc_pred.sigmoid().detach()
        if use_loc_filter:
            loc_mask = loc_pred >= self.loc_filter_thr
        else:
            loc_mask = loc_pred >= 0.0
        mask = loc_mask.permute(1, 2, 0).expand(-1, -1, self.num_anchors)
        mask = mask.contiguous().view(-1)
        squares = squares[mask]
        anchor_deltas = shape_pred.permute(1, 2, 0).contiguous().view(-1, 2).detach()[mask]
        bbox_deltas = anchor_deltas.new_full(squares.size(), 0)
        bbox_deltas[:, 2:] = anchor_deltas
        guided_anchors = delta2bbox(squares, bbox_deltas, self.anchoring_means, self.anchoring_stds, wh_ratio_clip=1e-06)
        return guided_anchors, mask

    def loss_shape_single(self, shape_pred, bbox_anchors, bbox_gts, anchor_weights, anchor_total_num):
        shape_pred = shape_pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        bbox_anchors = bbox_anchors.contiguous().view(-1, 4)
        bbox_gts = bbox_gts.contiguous().view(-1, 4)
        anchor_weights = anchor_weights.contiguous().view(-1, 4)
        bbox_deltas = bbox_anchors.new_full(bbox_anchors.size(), 0)
        bbox_deltas[:, 2:] += shape_pred
        inds = torch.nonzero(anchor_weights[:, 0] > 0).squeeze(1)
        bbox_deltas_ = bbox_deltas[inds]
        bbox_anchors_ = bbox_anchors[inds]
        bbox_gts_ = bbox_gts[inds]
        anchor_weights_ = anchor_weights[inds]
        pred_anchors_ = delta2bbox(bbox_anchors_, bbox_deltas_, self.anchoring_means, self.anchoring_stds, wh_ratio_clip=1e-06)
        loss_shape = self.loss_shape(pred_anchors_, bbox_gts_, anchor_weights_, avg_factor=anchor_total_num)
        return loss_shape

    def loss_loc_single(self, loc_pred, loc_target, loc_weight, loc_avg_factor, cfg):
        loss_loc = self.loss_loc(loc_pred.reshape(-1, 1), loc_target.reshape(-1, 1).long(), loc_weight.reshape(-1, 1), avg_factor=loc_avg_factor)
        return loss_loc

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'shape_preds', 'loc_preds'))
    def loss(self, cls_scores, bbox_preds, shape_preds, loc_preds, gt_bboxes, gt_labels, img_metas, cfg, gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.approx_generators)
        loc_targets, loc_weights, loc_avg_factor = ga_loc_target(gt_bboxes, featmap_sizes, self.octave_base_scale, self.anchor_strides, center_ratio=cfg.center_ratio, ignore_ratio=cfg.ignore_ratio)
        approxs_list, inside_flag_list = self.get_sampled_approxs(featmap_sizes, img_metas, cfg)
        squares_list, guided_anchors_list, _ = self.get_anchors(featmap_sizes, shape_preds, loc_preds, img_metas)
        sampling = False if not hasattr(cfg, 'ga_sampler') else True
        shape_targets = ga_shape_target(approxs_list, inside_flag_list, squares_list, gt_bboxes, img_metas, self.approxs_per_octave, cfg, sampling=sampling)
        if shape_targets is None:
            return None
        bbox_anchors_list, bbox_gts_list, anchor_weights_list, anchor_fg_num, anchor_bg_num = shape_targets
        anchor_total_num = anchor_fg_num if not sampling else anchor_fg_num + anchor_bg_num
        sampling = False if self.cls_focal_loss else True
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(guided_anchors_list, inside_flag_list, gt_bboxes, img_metas, self.target_means, self.target_stds, cfg, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels, sampling=sampling)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_total_samples = num_total_pos if self.cls_focal_loss else num_total_pos + num_total_neg
        losses_cls, losses_bbox = multi_apply(self.loss_single, cls_scores, bbox_preds, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_samples=num_total_samples, cfg=cfg)
        losses_loc = []
        for i in range(len(loc_preds)):
            loss_loc = self.loss_loc_single(loc_preds[i], loc_targets[i], loc_weights[i], loc_avg_factor=loc_avg_factor, cfg=cfg)
            losses_loc.append(loss_loc)
        losses_shape = []
        for i in range(len(shape_preds)):
            loss_shape = self.loss_shape_single(shape_preds[i], bbox_anchors_list[i], bbox_gts_list[i], anchor_weights_list[i], anchor_total_num=anchor_total_num)
            losses_shape.append(loss_shape)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_shape=losses_shape, loss_loc=losses_loc)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'shape_preds', 'loc_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, shape_preds, loc_preds, img_metas, cfg, rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(shape_preds) == len(loc_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        _, guided_anchors, loc_masks = self.get_anchors(featmap_sizes, shape_preds, loc_preds, img_metas, use_loc_filter=not self.training)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            guided_anchor_list = [guided_anchors[img_id][i].detach() for i in range(num_levels)]
            loc_mask_list = [loc_masks[img_id][i].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, guided_anchor_list, loc_mask_list, img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, mlvl_masks, img_shape, scale_factor, cfg, rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors, mask in zip(cls_scores, bbox_preds, mlvl_anchors, mlvl_masks):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            if mask.sum() == 0:
                continue
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            scores = scores[mask, :]
            bbox_pred = bbox_pred[mask, :]
            if scores.dim() == 0:
                anchors = anchors.unsqueeze(0)
                scores = scores.unsqueeze(0)
                bbox_pred = bbox_pred.unsqueeze(0)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means, self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels


norm_cfg_ = {'BN': nn.BatchNorm2d, 'SyncBN': nn.SyncBatchNorm, 'GN': nn.GroupNorm}


class ConvBNReLU(nn.Module):

    def __init__(self, input_size, in_channels, out_channels, kernel_size, stride, bias, relu_type, bn_type, groups=1, dilation=1):
        super(ConvBNReLU, self).__init__()
        assert relu_type in ['relu', 'none']
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)
        if bn_type == 'none':
            self.bn = nn.Sequential()
        elif bn_type == 'GN':
            norm_layer = norm_cfg_[bn_type]
            self.bn = norm_layer(num_channels=out_channels, num_groups=32)
        else:
            norm_layer = norm_cfg_[bn_type]
            self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu_type == 'relu' else nn.Sequential()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(self.bn(out))
        return out


class ConvBlock(nn.Module):

    def __init__(self, input_size, in_channels, out_channels, expansion, stride, kernel_size, dilation, groups=1, bn='none'):
        super(ConvBlock, self).__init__()
        self.conv = ConvBNReLU(input_size, in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=True, relu_type='none', bn_type=bn, groups=groups, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out


class Identity(nn.Module):

    def __init__(self, input_size, in_channels, out_channels, stride):
        super(Identity, self).__init__()
        if in_channels != out_channels or stride != 1:
            self.conv = ConvBNReLU(input_size, in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False, relu_type='relu', bn_type='bn')
        else:
            self.conv = nn.Sequential()

    def forward(self, x):
        return self.conv(x)


class MBBlock(nn.Module):

    def __init__(self, in_channels, out_channels, expansion, stride, kernel_size, dilation=1, groups=1):
        super(MBBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        mid_channels = in_channels * expansion
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, stride=1, padding=0, dilation=1, bias=False, groups=groups), nn.SyncBatchNorm(mid_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, groups=mid_channels), nn.SyncBatchNorm(mid_channels), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(mid_channels, out_channels, 1, stride=1, padding=0, dilation=1, bias=False, groups=groups), nn.SyncBatchNorm(out_channels))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.in_channels == self.out_channels and self.stride == 1:
            out = out + x
        return out


class SepBlock(nn.Module):

    def __init__(self, input_size, in_channels, out_channels, expansion, stride, kernel_size, dilation, groups=1, bn='BN'):
        super(SepBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = ConvBNReLU(input_size, in_channels, in_channels, kernel_size=kernel_size, stride=stride, bias=False, relu_type='relu', bn_type=bn, groups=in_channels, dilation=dilation)
        self.conv2 = ConvBNReLU(input_size // stride, in_channels, out_channels, kernel_size=1, stride=1, bias=False, relu_type='none', bn_type=bn, groups=groups)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.in_channels == self.out_channels and self.stride == 1:
            out = out + x
        return out


OPS = {'skip': lambda input_size, in_channels, out_channels, stride, bn='BN': Identity(input_size, in_channels, out_channels, stride), 'ir_k3_e1': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 1, stride, 3, bn=bn), 'ir_k3_e1_d2': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 1, stride, 3, dilation=2, bn=bn), 'ir_k3_e3': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 3, stride, 3, bn=bn), 'ir_k3_e6': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 6, stride, 3, bn=bn), 'ir_k3_e6_d3': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 6, stride, 3, dilation=3, bn=bn), 'ir_k3_s2': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 1, stride, 3, 2, bn=bn), 'ir_k5_e1': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 1, stride, 5, bn=bn), 'ir_k5_e1_d2': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 1, stride, 5, dilation=2, bn=bn), 'ir_k5_e3': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 3, stride, 5, bn=bn), 'ir_k5_e6': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 6, stride, 5, bn=bn), 'ir_k5_e6_d2': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 6, stride, 5, dilation=2, bn=bn), 'ir_k5_e6_d3': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 6, stride, 5, dilation=3, bn=bn), 'ir_k5_s2': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 1, stride, 5, 2, bn=bn), 'ir_k7_e3': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 3, stride, 7, bn=bn), 'ir_k7_e6': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 6, stride, 7, bn=bn), 'sd_k3_d1': lambda input_size, in_channels, out_channels, stride, bn='BN': SepBlock(input_size, in_channels, out_channels, 1, stride, 3, 1, bn=bn), 'sd_k3_d3': lambda input_size, in_channels, out_channels, stride, bn='BN': SepBlock(input_size, in_channels, out_channels, 1, stride, 3, 3, bn=bn), 'sd_k5_d2': lambda input_size, in_channels, out_channels, stride, bn='BN': SepBlock(input_size, in_channels, out_channels, 1, stride, 5, 2, bn=bn), 'sd_k5_d3': lambda input_size, in_channels, out_channels, stride, bn='BN': SepBlock(input_size, in_channels, out_channels, 1, stride, 5, 3, bn=bn), 'conv_1x1': lambda input_size, in_channels, out_channels, stride, bn='BN': ConvBlock(input_size, in_channels, out_channels, 1, stride, 1, 1, bn=bn)}


class MbblockHead(nn.Module):

    def __init__(self, latency=None, gamma=0.02, genotype=None, **kwargs):
        super(MbblockHead, self).__init__()
        self.latency = latency
        self.gamma = gamma
        self.genotype = genotype
        self.last_dim = kwargs.get('out_channels', [256])[-1]
        self.strides = kwargs.get('strides')
        self.out_channels = kwargs.get('out_channels')
        bn_type = kwargs.get('bn_type', 'BN')
        self.cells = nn.ModuleList()
        input_size = 7
        _in_channel = self.last_dim
        for _genotype, _stride, _out_channel in zip(genotype, self.strides, self.out_channels):
            self.cells.append(OPS[_genotype](input_size, _in_channel, _out_channel, _stride, bn=bn_type))
            input_size = input_size // _stride
            _in_channel = _out_channel
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)

    def forward(self, x):
        for cell in self.cells:
            x = cell(x)
        return x, None


def build_search_head(cfg):
    """Build head model from config dict.
    """
    if cfg is not None:
        cfg_ = cfg.copy()
        head_type = cfg_.pop('type')
        if head_type == 'DARTS':
            raise NotImplementedError
        elif head_type == 'MBBlock':
            return MbblockHead(**cfg_)
        else:
            raise KeyError('Invalid head type {}'.fromat(head_type))
    else:
        return None


class RetinaHead(AnchorHead):

    def __init__(self, num_classes, in_channels, stacked_convs=4, octave_base_scale=4, scales_per_octave=3, search_head=None, conv_cfg=None, norm_cfg=None, **kwargs):
        self.stacked_convs = stacked_convs
        self.search_head = search_head
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array([(2 ** (i / scales_per_octave)) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(RetinaHead, self).__init__(num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        if self.search_head is not None:
            if 'cls' in self.search_head.branch:
                self.cls_convs = build_search_head(self.search_head)
            else:
                for i in range(self.stacked_convs):
                    chn = self.in_channels if i == 0 else self.feat_channels
                    self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            if 'reg' in self.search_head.branch:
                self.reg_convs = build_search_head(self.search_head)
            else:
                for i in range(self.stacked_convs):
                    chn = self.in_channels if i == 0 else self.feat_channels
                    self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        else:
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
                self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        if self.search_head is not None:
            if 'cls' in self.search_head.branch:
                cls_feat = self.cls_convs(cls_feat)[0]
            else:
                for cls_conv in self.cls_convs:
                    cls_feat = cls_conv(cls_feat)
            if 'reg' in self.search_head.branch:
                reg_feat = self.reg_convs(reg_feat)[0]
            else:
                for reg_conv in self.reg_convs:
                    reg_feat = reg_conv(reg_feat)
        else:
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets)
    else:
        raise TypeError('dets must be either a Tensor or numpy array, but got {}'.format(type(dets)))
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    elif dets_th.is_cuda:
        inds = nms_cuda.nms(dets_th, iou_thr)
    else:
        inds = nms_cpu.nms(dets_th, iou_thr)
    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


class RPNHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self, cls_scores, bbox_preds, gt_bboxes, img_metas, cfg, gt_bboxes_ignore=None):
        losses = super(RPNHead, self).loss(cls_scores, bbox_preds, gt_bboxes, None, img_metas, cfg, gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means, self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        loss = loss.sum() / avg_factor
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
    return wrapper


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


class SSDHead(AnchorHead):

    def __init__(self, input_size=300, num_classes=81, in_channels=(512, 1024, 512, 256, 256, 256), anchor_strides=(8, 16, 32, 64, 100, 300), basesize_ratio_range=(0.1, 0.9), anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]), target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0)):
        super(AnchorHead, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes
        num_anchors = [(len(ratios) * 2 + 2) for ratios in anchor_ratios]
        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(nn.Conv2d(in_channels[i], num_anchors[i] * 4, kernel_size=3, padding=1))
            cls_convs.append(nn.Conv2d(in_channels[i], num_anchors[i] * num_classes, kernel_size=3, padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)
        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        step = int(np.floor(max_ratio - min_ratio) / (len(in_channels) - 2))
        min_sizes = []
        max_sizes = []
        for r in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(int(input_size * r / 100))
            max_sizes.append(int(input_size * (r + step) / 100))
        if input_size == 300:
            if basesize_ratio_range[0] == 0.15:
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
            elif basesize_ratio_range[0] == 0.2:
                min_sizes.insert(0, int(input_size * 10 / 100))
                max_sizes.insert(0, int(input_size * 20 / 100))
        elif input_size == 512:
            if basesize_ratio_range[0] == 0.1:
                min_sizes.insert(0, int(input_size * 4 / 100))
                max_sizes.insert(0, int(input_size * 10 / 100))
            elif basesize_ratio_range[0] == 0.15:
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
        self.anchor_generators = []
        self.anchor_strides = anchor_strides
        for k in range(len(anchor_strides)):
            base_size = min_sizes[k]
            stride = anchor_strides[k]
            ctr = (stride - 1) / 2.0, (stride - 1) / 2.0
            scales = [1.0, np.sqrt(max_sizes[k] / min_sizes[k])]
            ratios = [1.0]
            for r in anchor_ratios[k]:
                ratios += [1 / r, r]
            anchor_generator = AnchorGenerator(base_size, scales, ratios, scale_major=False, ctr=ctr)
            indices = list(range(len(ratios)))
            indices.insert(1, len(indices))
            anchor_generator.base_anchors = torch.index_select(anchor_generator.base_anchors, 0, torch.LongTensor(indices))
            self.anchor_generators.append(anchor_generator)
        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.fp16_enabled = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs, self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    def loss_single(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, num_total_samples, cfg):
        loss_cls_all = F.cross_entropy(cls_score, labels, reduction='none') * label_weights
        pos_inds = (labels > 0).nonzero().view(-1)
        neg_inds = (labels == 0).nonzero().view(-1)
        num_pos_samples = pos_inds.size(0)
        num_neg_samples = cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
        loss_bbox = smooth_l1_loss(bbox_pred, bbox_targets, bbox_weights, beta=cfg.smoothl1_beta, avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, cfg, gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas)
        cls_reg_targets = anchor_target(anchor_list, valid_flag_list, gt_bboxes, img_metas, self.target_means, self.target_stds, cfg, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=1, sampling=False, unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_images = len(img_metas)
        all_cls_scores = torch.cat([s.permute(0, 2, 3, 1).reshape(num_images, -1, self.cls_out_channels) for s in cls_scores], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list, -1).view(num_images, -1)
        all_bbox_preds = torch.cat([b.permute(0, 2, 3, 1).reshape(num_images, -1, 4) for b in bbox_preds], -2)
        all_bbox_targets = torch.cat(bbox_targets_list, -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list, -2).view(num_images, -1, 4)
        losses_cls, losses_bbox = multi_apply(self.loss_single, all_cls_scores, all_bbox_preds, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, num_total_samples=num_total_pos, cfg=cfg)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)


norm_layer = norm_cfg_['BN']


def create_spatial_conv2d_group_bn_relu(prefix, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=False, has_bn=True, has_relu=True, channel_shuffle=False, has_spatial_conv=True, has_spatial_conv_bn=True, conv_name_fun=None, bn_name_fun=None, bn_training=True, fix_weights=False):
    conv_name = prefix
    if conv_name_fun:
        conv_name = conv_name_fun(prefix)
    layer = nn.Sequential()
    if has_spatial_conv:
        spatial_conv_name = conv_name + '_s'
        layer.add_module(spatial_conv_name, nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias))
        if fix_weights:
            pass
        if has_spatial_conv_bn:
            layer.add_module(spatial_conv_name + '_bn', norm_layer(in_channels))
    if channel_shuffle:
        pass
    assert in_channels % groups == 0
    assert out_channels % groups == 0
    layer.add_module(conv_name, nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=bias))
    if fix_weights:
        pass
    if has_bn:
        bn_name = 'bn_' + prefix
        if bn_name_fun:
            bn_name = bn_name_fun(prefix)
        layer.add_module(bn_name, norm_layer(out_channels))
        if bn_training:
            pass
    if has_relu:
        layer.add_module('relu' + prefix, nn.ReLU(inplace=True))
    return layer


def conv1x1_dwconv_conv1x1(prefix, in_channels, out_channels, mid_channels, kernel_size, stride, bn_training=True):
    mid_channels = int(mid_channels)
    layer = list()
    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2a', in_channels=in_channels, out_channels=mid_channels, kernel_size=-1, stride=1, padding=0, groups=1, has_bn=True, has_relu=True, channel_shuffle=False, has_spatial_conv=False, has_spatial_conv_bn=False, conv_name_fun=lambda p: 'interstellar' + p, bn_name_fun=lambda p: 'bn' + p, bn_training=bn_training))
    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2b', in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=1, has_bn=True, has_relu=False, channel_shuffle=False, has_spatial_conv=True, has_spatial_conv_bn=True, conv_name_fun=lambda p: 'interstellar' + p, bn_name_fun=lambda p: 'bn' + p, bn_training=bn_training))
    return nn.Sequential(*layer)


def xception(prefix, in_channels, out_channels, mid_channels, stride, bn_training=True):
    mid_channels = int(mid_channels)
    layer = list()
    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2a', in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=1, groups=1, has_bn=True, has_relu=True, channel_shuffle=False, has_spatial_conv=True, has_spatial_conv_bn=True, conv_name_fun=lambda p: 'interstellar' + p, bn_name_fun=lambda p: 'bn' + p, bn_training=bn_training))
    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2b', in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, groups=1, has_bn=True, has_relu=True, channel_shuffle=False, has_spatial_conv=True, has_spatial_conv_bn=True, conv_name_fun=lambda p: 'interstellar' + p, bn_name_fun=lambda p: 'bn' + p, bn_training=bn_training))
    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2c', in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=1, has_bn=True, has_relu=False, channel_shuffle=False, has_spatial_conv=True, has_spatial_conv_bn=True, conv_name_fun=lambda p: 'interstellar' + p, bn_name_fun=lambda p: 'bn' + p, bn_training=bn_training))
    return nn.Sequential(*layer)


Blocks = {'shufflenet_3x3': lambda prefix, in_channels, output_channels, base_mid_channels, stride, bn_training: conv1x1_dwconv_conv1x1(prefix, in_channels, output_channels, base_mid_channels, 3, stride, bn_training), 'shufflenet_5x5': lambda prefix, in_channels, output_channels, base_mid_channels, stride, bn_training: conv1x1_dwconv_conv1x1(prefix, in_channels, output_channels, base_mid_channels, 5, stride, bn_training), 'shufflenet_7x7': lambda prefix, in_channels, output_channels, base_mid_channels, stride, bn_training: conv1x1_dwconv_conv1x1(prefix, in_channels, output_channels, base_mid_channels, 7, stride, bn_training), 'xception_3x3': lambda prefix, in_channels, output_channels, base_mid_channels, stride, bn_training: xception(prefix, in_channels, output_channels, base_mid_channels, stride, bn_training)}


blocks_key = ['shufflenet_3x3', 'shufflenet_5x5', 'shufflenet_7x7', 'xception_3x3']


def channel_shuffle2(x):
    channels = x.shape[1]
    assert channels % 4 == 0
    height = x.shape[2]
    width = x.shape[3]
    x = x.reshape(x.shape[0] * channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, channels // 2, height, width)
    return x[0], x[1]


class ShuffleNetV2BlockSearched(nn.Module):

    def __init__(self, prefix, in_channels, out_channels, stride, base_mid_channels, i_th, architecture):
        super(ShuffleNetV2BlockSearched, self).__init__()
        op = blocks_key[architecture[i_th]]
        self.ksize = int(op.split('_')[1][0])
        self.stride = stride
        if self.stride == 2:
            self.conv = Blocks[op](prefix + '_' + op, in_channels, out_channels - in_channels, base_mid_channels, stride, True)
        else:
            self.conv = Blocks[op](prefix + '_' + op, in_channels // 2, out_channels // 2, base_mid_channels, stride, True)
        if stride > 1:
            self.proj_conv = create_spatial_conv2d_group_bn_relu(prefix + '_proj', in_channels, in_channels, self.ksize, stride, self.ksize // 2, has_bn=True, has_relu=True, channel_shuffle=False, has_spatial_conv=True, has_spatial_conv_bn=True, conv_name_fun=lambda p: 'interstellar' + p, bn_name_fun=lambda p: 'bn' + p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_in):
        if self.stride == 1:
            x_proj, x = channel_shuffle2(x_in)
        else:
            x_proj = x_in
            x = x_in
            x_proj = self.proj_conv(x_proj)
        x = self.relu(self.conv(x))
        return torch.cat((x_proj, x), dim=1)


class DetNas(nn.Module):

    def __init__(self, model_size='VOC_FPN_300M', out_indices=(3, 7, 15, 19), frozen_stages=-1):
        super(DetNas, self).__init__()
        None
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        if model_size == 'COCO_FPN_3.8G':
            architecture = [0, 0, 3, 1, 2, 1, 0, 2, 0, 3, 1, 2, 3, 3, 2, 0, 2, 1, 1, 3, 2, 0, 2, 2, 2, 1, 3, 1, 0, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3]
            stage_repeats = [8, 8, 16, 8]
            stage_out_channels = [-1, 72, 172, 432, 864, 1728, 1728]
        elif model_size == 'COCO_FPN_1.3G':
            architecture = [0, 0, 3, 1, 2, 1, 0, 2, 0, 3, 1, 2, 3, 3, 2, 0, 2, 1, 1, 3, 2, 0, 2, 2, 2, 1, 3, 1, 0, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3]
            stage_repeats = [8, 8, 16, 8]
            stage_out_channels = [-1, 48, 96, 240, 480, 960, 1024]
        elif model_size == 'COCO_FPN_300M':
            architecture = [2, 1, 2, 0, 2, 1, 1, 2, 3, 3, 1, 3, 0, 0, 3, 1, 3, 1, 3, 2]
            stage_repeats = [4, 4, 8, 4]
            stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        elif model_size == 'COCO_RetinaNet_300M':
            architecture = [2, 3, 1, 1, 3, 2, 1, 3, 3, 1, 1, 1, 3, 3, 2, 0, 3, 3, 3, 3]
            stage_repeats = [4, 4, 8, 4]
            stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        elif model_size == 'VOC_FPN_300M':
            architecture = [2, 1, 0, 3, 1, 3, 0, 3, 2, 0, 1, 1, 3, 3, 3, 3, 3, 3, 3, 1]
            stage_repeats = [4, 4, 8, 4]
            stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        elif model_size == 'VOC_RetinaNet_300M':
            architecture = [1, 3, 0, 0, 2, 3, 3, 3, 2, 3, 3, 3, 3, 2, 2, 0, 2, 3, 1, 1]
            stage_repeats = [4, 4, 8, 4]
            stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        else:
            raise NotImplementedError
        self.first_conv = ConvBNReLU(in_channel=3, out_channel=stage_out_channels[1], k_size=3, stride=2, padding=1, gaussian_init=True)
        self.features = list()
        in_channels = stage_out_channels[1]
        i_th = 0
        for id_stage in range(1, len(stage_repeats) + 1):
            out_channels = stage_out_channels[id_stage + 1]
            repeats = stage_repeats[id_stage - 1]
            for id_repeat in range(repeats):
                prefix = str(id_stage) + chr(ord('a') + id_repeat)
                stride = 1 if id_repeat > 0 else 2
                self.features.append(ShuffleNetV2BlockSearched(prefix, in_channels=in_channels, out_channels=out_channels, stride=stride, base_mid_channels=out_channels // 2, i_th=i_th, architecture=architecture))
                in_channels = out_channels
                i_th += 1
        self.features = nn.Sequential(*self.features)
        if self.out_indices[-1] == len(self.features):
            self.last_conv = ConvBNReLU(in_channel=in_channels, out_channel=stage_out_channels[-1], k_size=1, stride=1, padding=0)
        self._initialize_weights()
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.first_conv.bn.eval()
            for m in [self.first_conv]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(self.frozen_stages):
            self.features[i].eval()
            for param in self.features[i].parameters():
                param.requires_grad = False

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        x = self.first_conv(x)
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i in self.out_indices:
                outs.append(x)
        if self.out_indices[-1] == len(self.features):
            x = self.last_conv(x)
            outs.append(x)
        return tuple(outs)


class DropBlock2D(nn.Module):
    """Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        assert x.dim() == 4, 'Expected input with 4 dimensions (bsize, channels, height, width)'
        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = torch.rand(x.shape[0], *x.shape[2:]) < gamma
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[:, None, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :], kernel_size=(self.block_size, self.block_size), stride=(1, 1), padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / self.block_size ** 2


class DropBlock3D(DropBlock2D):
    """Randomly zeroes 3D spatial blocks of the input tensor.
    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__(drop_prob, block_size)

    def forward(self, x):
        assert x.dim() == 5, 'Expected input with 5 dimensions (bsize, channels, depth, height, width)'
        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = torch.rand(x.shape[0], *x.shape[2:]) < gamma
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[:, None, :, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask[:, None, :, :, :], kernel_size=(self.block_size, self.block_size, self.block_size), stride=(1, 1, 1), padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / self.block_size ** 3


class DropBlockScheduled(nn.Module):

    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(DropBlockScheduled, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        if self.training:
            self.step()
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]
        self.i += 1


PRIMITIVES = ['conv_1x1', 'ir_k3_e6_d3', 'ir_k5_e6', 'ir_k5_e6_d3', 'sd_k3_d1', 'sd_k3_d3', 'sd_k5_d2', 'sd_k5_d3']


predefine_archs = {'fbnet_b': {'genotypes': ['conv3', 'ir_k3_e1', 'ir_k3_e6', 'ir_k5_e6', 'ir_k3_e1', 'ir_k3_e1', 'ir_k5_e6', 'ir_k5_e3', 'ir_k3_e6', 'ir_k5_e6', 'ir_k5_e6', 'ir_k5_e1', 'skip', 'ir_k5_e3', 'ir_k5_e6', 'ir_k3_e1', 'ir_k5_e1', 'ir_k5_e3', 'ir_k5_e6', 'ir_k5_e1', 'ir_k5_e6', 'ir_k5_e6', 'ir_k3_e6', 'conv1', 'avgpool'], 'strides': [2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 7], 'out_channels': [16, 16, 24, 24, 24, 24, 32, 32, 32, 32, 64, 64, 64, 64, 112, 112, 112, 112, 184, 184, 184, 184, 352, 1984, 1984], 'dropout_ratio': 0.2, 'search_space': 'fbsb'}, 'fbnet_hit': {'genotypes': ['conv3', 'ir_k3_e3', 'ir_k3_e3', 'ir_k3_e3_r2', 'ir_k3_e3', 'ir_k5_e6', 'ir_k5_e6', 'ir_k3_e3', 'ir_k3_e3', 'ir_k7_e6', 'ir_k5_e6', 'ir_k5_e6_r2', 'ir_k5_e3', 'ir_k5_e6', 'ir_k5_e6_r2', 'ir_k5_e6', 'ir_k5_e6_r2', 'ir_k7_e6', 'ir_k5_e6', 'ir_k5_e6_r2', 'ir_k5_e6', 'ir_k3_e3', 'conv1'], 'strides': [2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1], 'out_channels': [16, 48, 48, 48, 48, 96, 96, 96, 96, 184, 184, 184, 184, 256, 256, 256, 256, 352, 352, 352, 352, 1024, 2048], 'dropout_ratio': 0.2, 'search_space': 'fbsb'}}


class FBNet(nn.Module):

    def __init__(self, arch='fbnet_c', out_indices=(5, 9, 17, 22), frozen_stages=-1):
        super(FBNet, self).__init__()
        None
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.arch = arch
        self.input_size = 800
        self.build_backbone(self.arch, self.input_size)

    def build_backbone(self, arch, input_size):
        genotypes = predefine_archs[arch]['genotypes']
        strides = predefine_archs[arch]['strides']
        out_channels = predefine_archs[arch]['out_channels']
        self.layers = nn.ModuleList()
        self.layers.append(ConvBNReLU(input_size, in_channels=3, out_channels=out_channels[0], kernel_size=3, stride=strides[0], padding=1, bias=True, relu_type='relu', bn_type='bn'))
        input_size = input_size // strides[0]
        _in_channels = out_channels[0]
        for genotype, stride, _out_channels in zip(genotypes[1:], strides[1:], out_channels[1:]):
            if genotype.endswith('sb'):
                self.layers.append(SUPER_PRIMITIVES[genotype](input_size, _in_channels, _out_channels, stride))
            else:
                self.layers.append(PRIMITIVES[genotype](input_size, _in_channels, _out_channels, stride))
            input_size = input_size // stride
            _in_channels = _out_channels
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, alphas=None):
        outs = []
        cnt = 0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class AvgPool(nn.Module):

    def __init__(self, stride):
        super(AvgPool, self).__init__()
        self.stride = stride

    def forward(self, x):
        return F.avg_pool2d(x, self.stride)


class ChannelShuffle(nn.Module):

    def __init__(self, groups=1):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        if self.groups == 1:
            return x
        N, C, H, W = x.size()
        cpg = C // self.groups
        out = x.view(N, self.groups, cpg, H, W)
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out.view(N, C, H, W)
        return out


class HRModule(nn.Module):
    """ High-Resolution Module for HRNet. In this module, every branch
    has 4 BasicBlocks/Bottlenecks. Fusion/Exchange is in this module.
    """

    def __init__(self, num_branches, blocks, num_blocks, in_channels, num_channels, multiscale_output=True, with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN')):
        super(HRModule, self).__init__()
        self._check_branches(num_branches, num_blocks, in_channels, num_channels)
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, num_blocks, in_channels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            raise ValueError(error_msg)
        if num_branches != len(in_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(in_channels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.in_channels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(build_conv_layer(self.conv_cfg, self.in_channels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), build_norm_layer(self.norm_cfg, num_channels[branch_index] * block.expansion)[1])
        layers = []
        layers.append(block(self.in_channels[branch_index], num_channels[branch_index], stride, downsample=downsample, with_cp=self.with_cp, norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg))
        self.in_channels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.in_channels[branch_index], num_channels[branch_index], with_cp=self.with_cp, norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[i], kernel_size=1, stride=1, padding=0, bias=False), build_norm_layer(self.norm_cfg, in_channels[i])[1], nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[i], kernel_size=3, stride=2, padding=1, bias=False), build_norm_layer(self.norm_cfg, in_channels[i])[1]))
                        else:
                            conv_downsamples.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[j], kernel_size=3, stride=2, padding=1, bias=False), build_norm_layer(self.norm_cfg, in_channels[j])[1], nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, kernel_size=3, rf_series=1, rf_sd=1, rf_bn=True, rf_relu=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 1, groups=hidden_dim, bias=False), norm_layer(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), norm_layer(oup))
        else:
            self.conv = []
            self.conv.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            self.conv.append(norm_layer(hidden_dim))
            self.conv.append(nn.ReLU6(inplace=True))
            for idx in range(rf_series):
                self.conv.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=int((kernel_size - 1) * (idx + 1) / 2), dilation=idx + 1, groups=hidden_dim, bias=False))
                if rf_bn:
                    self.conv.append(norm_layer(hidden_dim))
                if rf_relu:
                    self.conv.append(nn.ReLU6(inplace=True))
            if not rf_bn:
                self.conv.append(norm_layer(hidden_dim))
            if not rf_relu:
                self.conv.append(nn.ReLU6(inplace=True))
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(norm_layer(oup))
            self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def Conv_1x1(inp, oup, activation=nn.ReLU6, act_params={'inplace': True}):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), activation(**act_params))


def Conv_3x3(inp, oup, stride, activation=nn.ReLU6, act_params={'inplace': True}):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), activation(**act_params))


def SepConv_3x3(inp, oup, activation=nn.ReLU6, act_params={'inplace': True}):
    return nn.Sequential(nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), activation(**act_params), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))


class MnasNet(nn.Module):

    def __init__(self, out_indices=(1, 2, 3, 4), width_mult=1.0, drop_prob=0.0, num_steps=300000.0, activation=nn.ReLU6, act_params={'inplace': True}):
        super(MnasNet, self).__init__()
        self.out_indices = out_indices
        self.activation = activation
        self.act_params = act_params
        self.interverted_residual_setting = [[3, 24, 3, 2, 3, 0], [3, 40, 3, 2, 5, 0], [6, 80, 3, 2, 5, 0], [6, 96, 2, 1, 3, drop_prob], [6, 192, 4, 2, 5, drop_prob], [6, 320, 1, 1, 3, drop_prob]]
        self.num_steps = num_steps
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [Conv_3x3(3, input_channel, 2, self.activation, self.act_params), SepConv_3x3(input_channel, 16, self.activation, self.act_params)]
        input_channel = 16
        for t, c, n, s, k, dp in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t, k, dp, self.num_steps, self.activation, self.act_params))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t, k, dp, self.num_steps, self.activation, self.act_params))
                input_channel = output_channel
        self.features.append(Conv_1x1(input_channel, self.last_channel, self.activation, self.act_params))
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        pdb.set_trace()
        outs = []
        cnt = 0
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

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


class MobileNetV2(nn.Module):

    def __init__(self, width_mult=1.0, input_channel=32, last_channel=1280, kernel_size=3, out_indices=(2, 5, 12, 17), style='pytorch', frozen_stages=-1, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        self.kernel_size = kernel_size
        self.out_indices = out_indices
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.mv2_layer = []
        features = []
        features.append(nn.Sequential(nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False), norm_layer(input_channel), nn.ReLU6(inplace=True)))
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    features.append(block(input_channel, output_channel, s, expand_ratio=t, kernel_size=3))
                else:
                    features.append(block(input_channel, output_channel, 1, expand_ratio=t, kernel_size=kernel_size))
                input_channel = output_channel
        features.append(nn.Sequential(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False), norm_layer(last_channel), nn.ReLU6(inplace=True)))
        for i, module in enumerate(features):
            layer_name = 'features{}'.format(i)
            self.add_module(layer_name, module)
            self.mv2_layer.append(layer_name)
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)
        self._freeze_stages()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.mv2_layer):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
    """
    def train(self, mode=True):
        super(MobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
    """


class StructualDeformBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(StructualDeformBlock, self).__init__()
        assert out_channels == 2 * kernel_size ** 2
        self.out_channels = out_channels + 6
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        coord = (torch.arange(kernel_size, dtype=torch.float) - kernel_size // 2) * dilation
        coord = list(torch.meshgrid([coord, coord]))
        coord.append(torch.ones(kernel_size, kernel_size))
        self.coord_map = torch.autograd.Variable(torch.stack(coord, dim=0).view(3, -1), requires_grad=False)

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}'
        return s.format(**self.__dict__)

    def forward(self, x_):
        offset_affine = F.conv2d(x_, self.weight, self.bias, self.stride, self.padding, self.dilation)
        n, c, h, w = offset_affine.shape
        deform_params = offset_affine[:, -6:].view(n, 2, 3, h, w)
        structural_offset = torch.einsum('nijhw,jk->nikhw', (deform_params, self.coord_map))
        offset = structural_offset.reshape(n, -1, h, w) + offset_affine[:, :-6]
        return offset


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20.0, eps=1e-10):
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) * x_float / norm).type_as(x)


class SSDVGG(VGG):
    extra_setting = {(300): (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256), (512): (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128)}

    def __init__(self, input_size, depth, with_last_pool=False, ceil_mode=True, out_indices=(3, 4), out_feature_indices=(22, 34), l2_norm_scale=20.0):
        super(SSDVGG, self).__init__(depth, with_last_pool=with_last_pool, ceil_mode=ceil_mode, out_indices=out_indices)
        assert input_size in (300, 512)
        self.input_size = input_size
        self.features.add_module(str(len(self.features)), nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.features.add_module(str(len(self.features)), nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
        self.features.add_module(str(len(self.features)), nn.ReLU(inplace=True))
        self.features.add_module(str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1))
        self.features.add_module(str(len(self.features)), nn.ReLU(inplace=True))
        self.out_feature_indices = out_feature_indices
        self.inplanes = 1024
        self.extra = self._make_extra_layers(self.extra_setting[input_size])
        self.l2_norm = L2Norm(self.features[out_feature_indices[0] - 1].out_channels, l2_norm_scale)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')
        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        constant_init(self.l2_norm, self.l2_norm.scale)

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_feature_indices:
                outs.append(x)
        for i, layer in enumerate(self.extra):
            x = F.relu(layer(x), inplace=True)
            if i % 2 == 1:
                outs.append(x)
        outs[0] = self.l2_norm(outs[0])
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _make_extra_layers(self, outplanes):
        layers = []
        kernel_sizes = 1, 3
        num_layers = 0
        outplane = None
        for i in range(len(outplanes)):
            if self.inplanes == 'S':
                self.inplanes = outplane
                continue
            k = kernel_sizes[num_layers % 2]
            if outplanes[i] == 'S':
                outplane = outplanes[i + 1]
                conv = nn.Conv2d(self.inplanes, outplane, k, stride=2, padding=1)
            else:
                outplane = outplanes[i]
                conv = nn.Conv2d(self.inplanes, outplane, k, stride=1, padding=0)
            layers.append(conv)
            self.inplanes = outplanes[i]
            num_layers += 1
        if self.input_size == 512:
            layers.append(nn.Conv2d(self.inplanes, 256, 4, padding=1))
        return nn.Sequential(*layers)


class SE(nn.Module):

    def __init__(self, input_size, in_channels, se_ratio):
        super(SE, self).__init__()
        self.in_channels, self.se_ratio = in_channels, se_ratio
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channels, max(1, int(in_channels * se_ratio)), 1, bias=False)
        self.fc2 = nn.Conv2d(max(1, int(in_channels * se_ratio)), in_channels, 1, bias=False)

    def forward(self, x):
        raise NotImplementedError
        out = self.pooling(x)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out


class SepConv(nn.Module):

    def __init__(self, input_size, in_channels, out_channels, expansion, stride, kernel_size, groups=1, bn_type='BN'):
        super(SepConv, self).__init__()
        self.conv1 = ConvBNReLU(input_size, in_channels, in_channels, kernel_size=kernel_size, stride=stride, bias=False, relu_type='relu', bn_type=bn_type, groups=in_channels)
        self.conv2 = ConvBNReLU(input_size // stride, in_channels, out_channels, kernel_size=1, stride=1, bias=False, relu_type='none', bn_type=bn_type, groups=groups)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = topk,
        return_single = True
    else:
        return_single = False
    maxk = max(topk)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def auto_fp16(apply_to=None, out_fp32=False):
    """Decorator to enable fp16 training automatically.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If inputs arguments are fp32 tensors, they will
    be converted to fp16 automatically. Arguments other than fp32 tensors are
    ignored.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp32 (bool): Whether to convert the output back to fp32.

    :Example:

        class MyModule1(nn.Module)

            # Convert x and y to fp16
            @auto_fp16()
            def forward(self, x, y):
                pass

        class MyModule2(nn.Module):

            # convert pred to fp16
            @auto_fp16(apply_to=('pred', ))
            def do_something(self, pred, others):
                pass
    """

    def auto_fp16_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@auto_fp16 can only be used to decorate the method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            args_info = getfullargspec(old_func)
            args_to_cast = args_info.args if apply_to is None else apply_to
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(cast_tensor_type(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(arg_value, torch.float, torch.half)
                    else:
                        new_kwargs[arg_name] = arg_value
            output = old_func(*new_args, **new_kwargs)
            if out_fp32:
                output = cast_tensor_type(output, torch.half, torch.float)
            return output
        return new_func
    return auto_fp16_wrapper


def bbox_target_single(pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg, reg_classes=1, target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means, target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0
    return labels, label_weights, bbox_targets, bbox_weights


def bbox_target(pos_bboxes_list, neg_bboxes_list, pos_gt_bboxes_list, pos_gt_labels_list, cfg, reg_classes=1, target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0], concat=True):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(bbox_target_single, pos_bboxes_list, neg_bboxes_list, pos_gt_bboxes_list, pos_gt_labels_list, cfg=cfg, reg_classes=reg_classes, target_means=target_means, target_stds=target_stds)
    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self, with_avg_pool=False, with_cls=True, with_reg=True, roi_feat_size=7, in_channels=256, num_classes=81, target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2], reg_class_agnostic=False, loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(pos_proposals, neg_proposals, pos_gt_bboxes, pos_gt_labels, rcnn_train_cfg, reg_classes, target_means=self.target_means, target_stds=self.target_stds)
        return cls_reg_targets

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)
            losses['loss_cls'] = self.loss_cls(cls_score, labels, label_weights, avg_factor=avg_factor, reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(pos_bbox_pred, bbox_targets[pos_inds], bbox_weights[pos_inds], avg_factor=bbox_targets.size(0), reduction_override=reduction_override)
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_det_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means, self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)
        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = torch.from_numpy(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(bboxes.size()[0], -1)
        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds',))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)
        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()
            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]
            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_, img_meta_)
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep
            bboxes_list.append(bboxes[keep_inds])
        return bboxes_list

    @force_fp32(apply_to=('bbox_pred',))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5
        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4
        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means, self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means, self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)
        return new_rois


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.relu = nn.ReLU()
        if in_channel != out_channel:
            self.downsample = nn.Conv2d(in_channel, out_channel, 1, bias=False)
            self.conv = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False), nn.SyncBatchNorm(in_channel), nn.Conv2d(in_channel, out_channel, 1, bias=False), nn.SyncBatchNorm(out_channel))
        else:
            self.downsample = nn.Sequential()
            self.conv = nn.Sequential(nn.Conv2d(in_channel, in_channel // 4, 1, bias=False), nn.SyncBatchNorm(in_channel // 4), nn.Conv2d(in_channel // 4, in_channel // 4, 3, padding=1, bias=False), nn.SyncBatchNorm(in_channel // 4), nn.Conv2d(in_channel // 4, out_channel, 1, bias=False), nn.SyncBatchNorm(out_channel))
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)

    def forward(self, x):
        out = self.conv(x)
        short_cut = self.downsample(x)
        out = self.relu(out + short_cut)
        return out


class ConvFCBBoxHead(BBoxHead):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \\-> reg convs -> reg fcs -> reg
    """

    def __init__(self, num_shared_convs=0, num_shared_fcs=0, num_cls_convs=0, num_cls_fcs=0, num_reg_convs=0, num_reg_fcs=0, convs_kernel=3, conv_out_channels=256, fc_out_channels=1024, conv_cfg=None, norm_cfg=None, search_head=None, toy_replace=None, bottle_first='conv', *args, **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert num_shared_convs + num_shared_fcs + num_cls_convs + num_cls_fcs + num_reg_convs + num_reg_fcs >= 0
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.convs_kernel = convs_kernel
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.bottle_first = bottle_first
        self.SearchHead = build_search_head(search_head)
        self.toy_replace = toy_replace
        self.shared_convs, self.shared_fcs, last_layer_dim = self._add_conv_fc_branch(self.num_shared_convs, self.num_shared_fcs, self.in_channels, True, toy_replace)
        self.shared_out_channels = last_layer_dim
        self.cls_convs, self.cls_fcs, self.cls_last_dim = self._add_conv_fc_branch_2head(self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)
        self.reg_convs, self.reg_fcs, self.reg_last_dim = self._add_conv_fc_branch_2head(self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)
        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_size * self.roi_feat_size
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_size * self.roi_feat_size
        if self.SearchHead is not None and self.num_shared_fcs == 0:
            self.cls_last_dim = self.SearchHead.last_dim
            self.reg_last_dim = self.SearchHead.last_dim
        self.relu = nn.ReLU(inplace=True)
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self, num_branch_convs, num_branch_fcs, in_channels, is_shared=False, toy_replace=None):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = last_layer_dim if i == 0 else self.conv_out_channels
                if toy_replace is not None and i == toy_replace.get('stage', 30):
                    if toy_replace.get('block', 'res') == 'ir':
                        branch_convs.append(MBBlock(conv_in_channels, self.conv_out_channels, 1, 1, toy_replace.get('conv_kernel'), dilation=toy_replace.get('dilation'), groups=1))
                    else:
                        branch_convs.append(ConvModule(conv_in_channels, self.conv_out_channels, toy_replace.get('conv_kernel'), padding=(toy_replace.get('conv_kernel') - 1) * toy_replace.get('dilation') // 2, dilation=toy_replace.get('dilation'), conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bottle_first=self.bottle_first))
                else:
                    branch_convs.append(ConvModule(conv_in_channels, self.conv_out_channels, self.convs_kernel, padding=(self.convs_kernel - 1) // 2, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bottle_first=self.bottle_first))
            last_layer_dim = self.conv_out_channels
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if (is_shared or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = last_layer_dim if i == 0 else self.fc_out_channels
                branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def _add_conv_fc_branch_2head(self, num_branch_convs, num_branch_fcs, in_channels):
        """convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = last_layer_dim if i == 0 else self.conv_out_channels
                branch_convs.append(ResidualBlock(conv_in_channels, self.conv_out_channels))
            last_layer_dim = self.conv_out_channels
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            for i in range(num_branch_fcs):
                fc_in_channels = last_layer_dim * self.roi_feat_size * self.roi_feat_size if i == 0 else self.fc_out_channels
                branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        loss_latency = None
        if self.SearchHead is not None:
            x, loss_latency = self.SearchHead(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        x_cls = x
        x_reg = x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, loss_latency


class SharedFCBBoxHead(ConvFCBBoxHead):

    def __init__(self, num_convs=0, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        super(SharedFCBBoxHead, self).__init__(*args, num_shared_convs=num_convs, num_shared_fcs=num_fcs, fc_out_channels=fc_out_channels, **kwargs)


class BasicResBlock(nn.Module):
    """Basic residual block.
    This block is a little different from the block in the ResNet backbone.
    The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.
    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
    """

    def __init__(self, in_channels, out_channels, conv_cfg=None, norm_cfg=dict(type='BN')):
        super(BasicResBlock, self).__init__()
        self.conv1 = ConvModule(in_channels, in_channels, kernel_size=3, padding=1, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv2 = ConvModule(in_channels, out_channels, kernel_size=1, bias=False, activation=None, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv_identity = ConvModule(in_channels, out_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=None)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        identity = self.conv_identity(identity)
        out = x + identity
        out = self.relu(out)
        return out


class DoubleConvFCBBoxHead(BBoxHead):
    """Bbox head used in Double-Head R-CNN
                                      /-> cls
                  /-> shared convs ->
                                      \\-> reg
    roi features
                                      /-> cls
                  \\-> shared fc    ->
                                      \\-> reg
    """

    def __init__(self, num_convs=0, num_fcs=0, conv_out_channels=1024, fc_out_channels=1024, conv_cfg=None, norm_cfg=dict(type='BN'), **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(DoubleConvFCBBoxHead, self).__init__(**kwargs)
        assert self.with_avg_pool
        assert num_convs > 0
        assert num_fcs > 0
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.res_block = BasicResBlock(self.in_channels, self.conv_out_channels)
        self.conv_branch = self._add_conv_branch()
        self.fc_branch = self._add_fc_branch()
        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg)
        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes)
        self.relu = nn.ReLU(inplace=True)

    def _add_conv_branch(self):
        """Add the fc branch which consists of a sequential of conv layers"""
        branch_convs = nn.ModuleList()
        for i in range(self.num_convs):
            branch_convs.append(Bottleneck(inplanes=self.conv_out_channels, planes=self.conv_out_channels // 4, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        return branch_convs

    def _add_fc_branch(self):
        """Add the fc branch which consists of a sequential of fc layers"""
        branch_fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = self.in_channels * self.roi_feat_area if i == 0 else self.fc_out_channels
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        return branch_fcs

    def init_weights(self):
        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_reg, std=0.001)
        for m in self.fc_branch.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def forward(self, x_cls, x_reg):
        x_conv = self.res_block(x_reg)
        for conv in self.conv_branch:
            x_conv = conv(x_conv)
        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)
        x_conv = x_conv.view(x_conv.size(0), -1)
        bbox_pred = self.fc_reg(x_conv)
        x_fc = x_cls.view(x_cls.size(0), -1)
        for fc in self.fc_branch:
            x_fc = self.relu(fc(x_fc))
        cls_score = self.fc_cls(x_fc)
        return cls_score, bbox_pred


dataset_aliases = {'voc': ['voc', 'pascal_voc', 'voc07', 'voc12'], 'imagenet_det': ['det', 'imagenet_det', 'ilsvrc_det'], 'imagenet_vid': ['vid', 'imagenet_vid', 'ilsvrc_vid'], 'coco': ['coco', 'mscoco', 'ms_coco'], 'wider_face': ['WIDERFaceDataset', 'wider_face', 'WDIERFace']}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name
    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


class BaseDetector(nn.Module):
    """Base class for detectors"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): list of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

             **kwargs: specific to concrete implementation
        """
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_meta (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError('num of augmentations ({}) != num of image meta ({})'.format(len(imgs), len(img_metas)))
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1
        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_meta, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=False`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=True`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def show_result(self, data, result, img_norm_cfg, dataset=None, score_thr=0.5, out_file=None):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)
        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError('dataset must be a valid dataset name or a sequence of class names, not {}'.format(type(dataset)))
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            bboxes = np.vstack(bbox_result)
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
            labels = np.concatenate(labels)
            mmcv.imshow_det_bboxes(img_show, bboxes, labels, class_names=class_names, score_thr=score_thr, show=out_file is None, out_file=out_file)


DETECTORS = Registry('detector')


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


def bbox_mapping_back(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = bbox_flip(bboxes, img_shape) if flip else bboxes
    new_bboxes = new_bboxes / scale_factor
    return new_bboxes


def merge_aug_proposals(aug_proposals, img_metas, rpn_test_cfg):
    """Merge augmented proposals (multiscale, flip, etc.)

    Args:
        aug_proposals (list[Tensor]): proposals from different testing
            schemes, shape (n, 5). Note that they are not rescaled to the
            original image size.
        img_metas (list[dict]): image info including "shape_scale" and "flip".
        rpn_test_cfg (dict): rpn test config.

    Returns:
        Tensor: shape (n, 4), proposals corresponding to original image scale.
    """
    recovered_proposals = []
    for proposals, img_info in zip(aug_proposals, img_metas):
        img_shape = img_info['img_shape']
        scale_factor = img_info['scale_factor']
        flip = img_info['flip']
        _proposals = proposals.clone()
        _proposals[:, :4] = bbox_mapping_back(_proposals[:, :4], img_shape, scale_factor, flip)
        recovered_proposals.append(_proposals)
    aug_proposals = torch.cat(recovered_proposals, dim=0)
    merged_proposals, _ = nms(aug_proposals, rpn_test_cfg.nms_thr)
    scores = merged_proposals[:, 4]
    _, order = scores.sort(0, descending=True)
    num = min(rpn_test_cfg.max_num, merged_proposals.shape[0])
    order = order[:num]
    merged_proposals = merged_proposals[order, :]
    return merged_proposals


class RPNTestMixin(object):

    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        aug_img_metas = []
        for i in range(imgs_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
        merged_proposals = [merge_aug_proposals(proposals, aug_img_meta, rpn_test_cfg) for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)]
        return merged_proposals


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def merge_aug_masks(aug_masks, img_metas, rcnn_test_cfg, weights=None):
    """Merge augmented mask prediction.

    Args:
        aug_masks (list[ndarray]): shape (n, #class, h, w)
        img_shapes (list[ndarray]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_masks = [(mask if not img_info[0]['flip'] else mask[..., ::-1]) for mask, img_info in zip(aug_masks, img_metas)]
    if weights is None:
        merged_masks = np.mean(recovered_masks, axis=0)
    else:
        merged_masks = np.average(np.array(recovered_masks), axis=0, weights=np.array(weights))
    return merged_masks


class CascadeRCNN(BaseDetector, RPNTestMixin):

    def __init__(self, num_stages, backbone, neck=None, shared_head=None, rpn_head=None, bbox_roi_extractor=None, bbox_head=None, mask_roi_extractor=None, mask_head=None, train_cfg=None, test_cfg=None, pretrained=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        super(CascadeRCNN, self).__init__()
        self.num_stages = num_stages
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)
        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)
        if bbox_head is not None:
            self.bbox_roi_extractor = nn.ModuleList()
            self.bbox_head = nn.ModuleList()
            if not isinstance(bbox_roi_extractor, list):
                bbox_roi_extractor = [bbox_roi_extractor for _ in range(num_stages)]
            if not isinstance(bbox_head, list):
                bbox_head = [bbox_head for _ in range(num_stages)]
            assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
            for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
                self.bbox_roi_extractor.append(builder.build_roi_extractor(roi_extractor))
                self.bbox_head.append(builder.build_head(head))
        if mask_head is not None:
            self.mask_head = nn.ModuleList()
            if not isinstance(mask_head, list):
                mask_head = [mask_head for _ in range(num_stages)]
            assert len(mask_head) == self.num_stages
            for head in mask_head:
                self.mask_head.append(builder.build_head(head))
            if mask_roi_extractor is not None:
                self.share_roi_extractor = False
                self.mask_roi_extractor = nn.ModuleList()
                if not isinstance(mask_roi_extractor, list):
                    mask_roi_extractor = [mask_roi_extractor for _ in range(num_stages)]
                assert len(mask_roi_extractor) == self.num_stages
                for roi_extractor in mask_roi_extractor:
                    self.mask_roi_extractor.append(builder.build_roi_extractor(roi_extractor))
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(CascadeRCNN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mask:
                if not self.share_roi_extractor:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None):
        x = self.extract_feat(img)
        losses = dict()
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta, self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
                bbox_sampler = build_sampler(rcnn_train_cfg.sampler, context=self)
                num_imgs = img.size(0)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j], gt_labels[j])
                    sampling_result = bbox_sampler.sample(assign_result, proposal_list[j], gt_bboxes[j], gt_labels[j], feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = bbox_head(bbox_feats)
            bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
            loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = value * lw if 'loss' in name else value
            if self.with_mask:
                if not self.share_roi_extractor:
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
                    mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs], pos_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats)
                else:
                    pos_inds = []
                    device = bbox_feats.device
                    for res in sampling_results:
                        pos_inds.append(torch.ones(res.pos_bboxes.shape[0], device=device, dtype=torch.uint8))
                        pos_inds.append(torch.zeros(res.neg_bboxes.shape[0], device=device, dtype=torch.uint8))
                    pos_inds = torch.cat(pos_inds)
                    mask_feats = bbox_feats[pos_inds]
                mask_head = self.mask_head[i]
                mask_pred = mask_head(mask_feats)
                mask_targets = mask_head.get_target(sampling_results, gt_masks, rcnn_train_cfg)
                pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
                loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = value * lw if 'loss' in name else value
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = bbox_targets[0]
                with torch.no_grad():
                    proposal_list = bbox_head.refine_bboxes(rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn
        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]
            bbox_feats = bbox_roi_extractor(x[:len(bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = bbox_head(bbox_feats)
            ms_scores.append(cls_score)
            if self.test_cfg.keep_all_stages:
                det_bboxes, det_labels = bbox_head.get_det_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=rescale, cfg=rcnn_test_cfg)
                bbox_result = bbox2result(det_bboxes, det_labels, bbox_head.num_classes)
                ms_bbox_result['stage{}'.format(i)] = bbox_result
                if self.with_mask:
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_head = self.mask_head[i]
                    if det_bboxes.shape[0] == 0:
                        segm_result = [[] for _ in range(mask_head.num_classes - 1)]
                    else:
                        _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
                        mask_rois = bbox2roi([_bboxes])
                        mask_feats = mask_roi_extractor(x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                        if self.with_shared_head:
                            mask_feats = self.shared_head(mask_feats, i)
                        mask_pred = mask_head(mask_feats)
                        segm_result = mask_head.get_seg_masks(mask_pred, _bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale)
                    ms_segm_result['stage{}'.format(i)] = segm_result
            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred, img_meta[0])
        cls_score = sum(ms_scores) / self.num_stages
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=rescale, cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels, self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result
        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head[-1].num_classes - 1)]
            else:
                _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                for i in range(self.num_stages):
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_feats = mask_roi_extractor(x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats)
                    mask_pred = self.mask_head[i](mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks, [img_meta] * self.num_stages, self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(merged_masks, _bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result
        if not self.test_cfg.keep_all_stages:
            if self.with_mask:
                results = ms_bbox_result['ensemble'], ms_segm_result['ensemble']
            else:
                results = ms_bbox_result['ensemble']
        elif self.with_mask:
            results = {stage: (ms_bbox_result[stage], ms_segm_result[stage]) for stage in ms_bbox_result}
        else:
            results = ms_bbox_result
        return results

    def aug_test(self, img, img_meta, proposals=None, rescale=False):
        raise NotImplementedError

    def show_result(self, data, result, img_norm_cfg, **kwargs):
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = ms_bbox_result['ensemble'], ms_segm_result['ensemble']
        elif isinstance(result, dict):
            result = result['ensemble']
        super(CascadeRCNN, self).show_result(data, result, img_norm_cfg, **kwargs)


class HybridTaskCascade(CascadeRCNN):

    def __init__(self, num_stages, backbone, semantic_roi_extractor=None, semantic_head=None, semantic_fusion=('bbox', 'mask'), interleaved=True, mask_info_flow=True, **kwargs):
        super(HybridTaskCascade, self).__init__(num_stages, backbone, **kwargs)
        assert self.with_bbox and self.with_mask
        assert not self.with_shared_head
        if semantic_head is not None:
            self.semantic_roi_extractor = builder.build_roi_extractor(semantic_roi_extractor)
            self.semantic_head = builder.build_head(semantic_head)
        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow

    @property
    def with_semantic(self):
        if hasattr(self, 'semantic_head') and self.semantic_head is not None:
            return True
        else:
            return False

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg, semantic_feat=None):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat], rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat
        cls_score, bbox_pred = bbox_head(bbox_feats)
        bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
        return loss_bbox, rois, bbox_targets, bbox_pred

    def _mask_forward_train(self, stage, x, sampling_results, gt_masks, rcnn_train_cfg, semantic_feat=None):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs], pos_rois)
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat], pos_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat
        if self.mask_info_flow:
            last_feat = None
            for i in range(stage):
                last_feat = self.mask_head[i](mask_feats, last_feat, return_logits=False)
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
        else:
            mask_pred = mask_head(mask_feats)
        mask_targets = mask_head.get_target(sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
        return loss_mask

    def _bbox_forward_test(self, stage, x, rois, semantic_feat=None):
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:len(bbox_roi_extractor.featmap_strides)], rois)
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat], rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat
        cls_score, bbox_pred = bbox_head(bbox_feats)
        return cls_score, bbox_pred

    def _mask_forward_test(self, stage, x, bboxes, semantic_feat=None):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_rois = bbox2roi([bboxes])
        mask_feats = mask_roi_extractor(x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat], mask_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat
        if self.mask_info_flow:
            last_feat = None
            last_pred = None
            for i in range(stage):
                mask_pred, last_feat = self.mask_head[i](mask_feats, last_feat)
                if last_pred is not None:
                    mask_pred = mask_pred + last_pred
                last_pred = mask_pred
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
            if last_pred is not None:
                mask_pred = mask_pred + last_pred
        else:
            mask_pred = mask_head(mask_feats)
        return mask_pred

    def forward_train(self, img, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, gt_semantic_seg=None, proposals=None):
        x = self.extract_feat(img)
        losses = dict()
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta, self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]
            sampling_results = []
            bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
            bbox_sampler = build_sampler(rcnn_train_cfg.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j], gt_labels[j])
                sampling_result = bbox_sampler.sample(assign_result, proposal_list[j], gt_bboxes[j], gt_labels[j], feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
            loss_bbox, rois, bbox_targets, bbox_pred = self._bbox_forward_train(i, x, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_targets[0]
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = value * lw if 'loss' in name else value
            if self.with_mask:
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
                        sampling_results = []
                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j], gt_labels[j])
                            sampling_result = bbox_sampler.sample(assign_result, proposal_list[j], gt_bboxes[j], gt_labels[j], feats=[lvl_feat[j][None] for lvl_feat in x])
                            sampling_results.append(sampling_result)
                loss_mask = self._mask_forward_train(i, x, sampling_results, gt_masks, rcnn_train_cfg, semantic_feat)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = value * lw if 'loss' in name else value
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None
        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn
        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            cls_score, bbox_pred = self._bbox_forward_test(i, x, rois, semantic_feat=semantic_feat)
            ms_scores.append(cls_score)
            if self.test_cfg.keep_all_stages:
                det_bboxes, det_labels = bbox_head.get_det_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=rescale, nms_cfg=rcnn_test_cfg)
                bbox_result = bbox2result(det_bboxes, det_labels, bbox_head.num_classes)
                ms_bbox_result['stage{}'.format(i)] = bbox_result
                if self.with_mask:
                    mask_head = self.mask_head[i]
                    if det_bboxes.shape[0] == 0:
                        segm_result = [[] for _ in range(mask_head.num_classes - 1)]
                    else:
                        _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
                        mask_pred = self._mask_forward_test(i, x, _bboxes, semantic_feat=semantic_feat)
                        segm_result = mask_head.get_seg_masks(mask_pred, _bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale)
                    ms_segm_result['stage{}'.format(i)] = segm_result
            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred, img_meta[0])
        cls_score = sum(ms_scores) / float(len(ms_scores))
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=rescale, cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels, self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result
        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head[-1].num_classes - 1)]
            else:
                _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor([semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None
                for i in range(self.num_stages):
                    mask_head = self.mask_head[i]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks, [img_meta] * self.num_stages, self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(merged_masks, _bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result
        if not self.test_cfg.keep_all_stages:
            if self.with_mask:
                results = ms_bbox_result['ensemble'], ms_segm_result['ensemble']
            else:
                results = ms_bbox_result['ensemble']
        elif self.with_mask:
            results = {stage: (ms_bbox_result[stage], ms_segm_result[stage]) for stage in ms_bbox_result}
        else:
            results = ms_bbox_result
        return results

    def aug_test(self, img, img_meta, proposals=None, rescale=False):
        raise NotImplementedError


class SingleStageDetector(BaseDetector):

    def __init__(self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None, pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.
        
        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        out = self.extract_feat(img)
        if len(out) >= 4:
            x = out
            loss_latency = None
        else:
            x = out[0]
            loss_latency = out[1]
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses, loss_latency

    def simple_test(self, img, img_meta, rescale=False):
        out = self.extract_feat(img)
        if len(out) >= 4:
            x = out
        else:
            x = out[0]
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes) for det_bboxes, det_labels in bbox_list]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError


def bbox_mapping(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from the original image scale to testing scale"""
    new_bboxes = bboxes * scale_factor
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape)
    return new_bboxes


def merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg):
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 4*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
        recovered_bboxes.append(bboxes)
    bboxes = torch.stack(recovered_bboxes).mean(dim=0)
    if aug_scores is None:
        return bboxes
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores


class BBoxTestMixin(object):

    def simple_test_bboxes(self, x, img_meta, proposals, rcnn_test_cfg, rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        out_list = self.bbox_head(roi_feats)
        cls_score = out_list[0]
        bbox_pred = out_list[1]
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=rescale, cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape, scale_factor, flip)
            rois = bbox2roi([proposals])
            roi_feats = self.bbox_roi_extractor(x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            bboxes, scores = self.bbox_head.get_det_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        merged_bboxes, merged_scores = merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin(object):

    def simple_test_mask(self, x, img_meta, det_bboxes, det_labels, rescale=False):
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape, scale_factor, rescale)
        return segm_result

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape, scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                mask_pred = self.mask_head(mask_feats)
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas, self.test_cfg.rcnn)
            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(merged_masks, det_bboxes, det_labels, self.test_cfg.rcnn, ori_shape, scale_factor=1.0, rescale=False)
        return segm_result


class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin, MaskTestMixin):

    def __init__(self, backbone, neck=None, shared_head=None, rpn_head=None, bbox_roi_extractor=None, bbox_head=None, mask_roi_extractor=None, mask_head=None, train_cfg=None, test_cfg=None, cls_roi_scale_factor=None, reg_roi_scale_factor=None, pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.cls_roi_scale_factor = cls_roi_scale_factor
        self.reg_roi_scale_factor = reg_roi_scale_factor
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)
        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)
        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
            if len(x) >= 2:
                if x[1] is not None:
                    x = x
                else:
                    x = x[0]
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        outs = ()
        x = self.extract_feat(img)
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4)
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            outs = outs + (mask_pred,)
        return outs

    def forward_train(self, img, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None):
        out = self.extract_feat(img)
        if len(out) >= 4:
            x = out
            loss_latency = None
        else:
            x = out[0]
            loss_latency = out[1]
        losses = dict()
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta, self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
                sampling_result = bbox_sampler.sample(assign_result, proposal_list[i], gt_bboxes[i], gt_labels[i], feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
            """
            bbox_feats_cls = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs],
                rois,
                roi_scale_factor=self.cls_roi_scale_factor)
            bbox_feats_reg = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs],
                rois,
                roi_scale_factor=self.reg_roi_scale_factor)
            """
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred, loss_latency_head = self.bbox_head(bbox_feats)
            if loss_latency_head is not None:
                if loss_latency is not None:
                    loss_latency = loss_latency + loss_latency_head
                else:
                    loss_latency = loss_latency_head
            bbox_targets = self.bbox_head.get_target(sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            losses.update(loss_bbox)
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(torch.ones(res.pos_bboxes.shape[0], device=device, dtype=torch.uint8))
                    pos_inds.append(torch.zeros(res.neg_bboxes.shape[0], device=device, dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)
            mask_targets = self.mask_head.get_target(sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets, pos_labels)
            losses.update(loss_mask)
        return losses, loss_latency
    """
    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        rois = bbox2roi(proposals)
        bbox_cls_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], 
            rois,
            roi_scale_factor=self.cls_roi_scale_factor)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            roi_scale_factor=self.reg_roi_scale_factor)
        if self.with_shared_head:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)
        cls_score, bbox_pred = self.bbox_head((bbox_cls_feats, bbox_reg_feats))
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels
    """

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        out = self.extract_feat(img)
        if len(out) >= 4:
            x = out
        else:
            x = out[0]
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        det_bboxes, det_labels = self.simple_test_bboxes(x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        proposal_list = self.aug_test_rpn(self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(self.extract_feats(imgs), img_metas, proposal_list, self.test_cfg.rcnn)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels, self.bbox_head.num_classes)
        if self.with_mask:
            segm_results = self.aug_test_mask(self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results


class Accuracy(nn.Module):

    def __init__(self, topk=(1,)):
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        return accuracy(pred, target, self.topk)


@weighted_loss
def balanced_l1_loss(pred, target, beta=1.0, alpha=0.5, gamma=1.5, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(diff < beta, alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff, gamma * diff + gamma / b - alpha * beta)
    return loss


class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """

    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0, reduction='mean', loss_weight=1.0):
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * balanced_l1_loss(pred, target, weight, alpha=self.alpha, gamma=self.gamma, beta=self.beta, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_bbox


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)
    return loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    loss = F.cross_entropy(pred, label, reduction='none')
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None):
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice, target, reduction='mean')[None]


class CrossEntropyLoss(nn.Module):

    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean', loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert use_sigmoid is False or use_mask is False
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * self.cls_criterion(cls_score, label, weight, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_cls


class SigmoidFocalLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target, gamma=2.0, alpha=0.25):
        ctx.save_for_backward(input, target)
        num_classes = input.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha
        loss = sigmoid_focal_loss_cuda.forward(input, target, num_classes, gamma, alpha)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        input, target = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_input = sigmoid_focal_loss_cuda.backward(input, target, d_loss, num_classes, gamma, alpha)
        return d_input, None, None, None, None


sigmoid_focal_loss = SigmoidFocalLossFunction.apply


class FocalLoss(nn.Module):

    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(pred, target, weight, gamma=self.gamma, alpha=self.alpha, reduction=reduction, avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float() / bins
        self.edges[-1] += 1e-06
        if momentum > 0:
            self.acc_sum = torch.zeros(bins)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        if pred.dim() != target.dim():
            target, label_weight = _expand_binary_labels(target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)
        g = torch.abs(pred.sigmoid().detach() - target)
        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n
        loss = F.binary_cross_entropy_with_logits(pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight


class GHMR(nn.Module):
    """GHM Regression Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector"
    https://arxiv.org/abs/1811.05181

    Args:
        mu (float): The parameter for the Authentic Smooth L1 loss.
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        loss_weight (float): The weight of the total GHM-R loss.
    """

    def __init__(self, mu=0.02, bins=10, momentum=0, loss_weight=1.0):
        super(GHMR, self).__init__()
        self.mu = mu
        self.bins = bins
        self.edges = torch.arange(bins + 1).float() / bins
        self.edges[-1] = 1000.0
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = torch.zeros(bins)
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, avg_factor=None):
        """Calculate the GHM-R loss.

        Args:
            pred (float tensor of size [batch_num, 4 (* class_num)]):
                The prediction of box regression layer. Channel number can be 4
                or 4 * class_num depending on whether it is class-agnostic.
            target (float tensor of size [batch_num, 4 (* class_num)]):
                The target regression values with the same size of pred.
            label_weight (float tensor of size [batch_num, 4 (* class_num)]):
                The weight of each sample, 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum
        diff = pred - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        weights = torch.zeros_like(g)
        valid = label_weight > 0
        tot = max(label_weight.float().sum().item(), 1.0)
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n
        loss = loss * weights
        loss = loss.sum() / tot
        return loss * self.loss_weight


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """
    assert mode in ['iou', 'iof']
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)
    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / area1[:, None]
    return ious


@weighted_loss
def iou_loss(pred, target, eps=1e-06):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    loss = -ious.log()
    return loss


class IoULoss(nn.Module):

    def __init__(self, eps=1e-06, reduction='mean', loss_weight=1.0):
        super(IoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * iou_loss(pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


@weighted_loss
def bounded_iou_loss(pred, target, beta=0.2, eps=0.001):
    """Improving Object Localization with Fitness NMS and Bounded IoU Loss,
    https://arxiv.org/abs/1711.00164.

    Args:
        pred (tensor): Predicted bboxes.
        target (tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0] + 1
    pred_h = pred[:, 3] - pred[:, 1] + 1
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0] + 1
        target_h = target[:, 3] - target[:, 1] + 1
    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry
    loss_dx = 1 - torch.max((target_w - 2 * dx.abs()) / (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max((target_h - 2 * dy.abs()) / (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w / (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h / (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh], dim=-1).view(loss_dx.size(0), -1)
    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta, loss_comb - 0.5 * beta)
    return loss


class BoundedIoULoss(nn.Module):

    def __init__(self, beta=0.2, eps=0.001, reduction='mean', loss_weight=1.0):
        super(BoundedIoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * bounded_iou_loss(pred, target, weight, beta=self.beta, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


mse_loss = weighted_loss(F.mse_loss)


class MSELoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        loss = self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction, avg_factor=avg_factor)
        return loss


class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * smooth_l1_loss(pred, target, weight, beta=self.beta, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_bbox


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    mask_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w], (mask_size, mask_size))
            mask_targets.append(target)
        mask_targets = torch.from_numpy(np.stack(mask_targets)).float()
    else:
        mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
    return mask_targets


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


class FCNMaskHead(nn.Module):

    def __init__(self, num_convs=4, roi_feat_size=14, in_channels=256, conv_kernel_size=3, conv_out_channels=256, upsample_method='deconv', upsample_ratio=2, num_classes=81, class_agnostic=False, conv_cfg=None, norm_cfg=None, loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(FCNMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError('Invalid upsample method {}, accepted methods are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else self.conv_out_channels
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(ConvModule(in_channels, self.conv_out_channels, self.conv_kernel_size, padding=padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
        upsample_in_channels = self.conv_out_channels if self.num_convs > 0 else in_channels
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(upsample_in_channels, self.conv_out_channels, self.upsample_ratio, stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(scale_factor=self.upsample_ratio, mode=self.upsample_method)
        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = self.conv_out_channels if self.upsample_method == 'deconv' else upsample_in_channels
        self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds, gt_masks, rcnn_train_cfg)
        return mask_targets

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets, torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)
        mask_pred = mask_pred.astype(np.float32)
        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1
        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0
        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)
            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)
        return cls_segms


class FusedSemanticHead(nn.Module):
    """Multi-level fused semantic segmentation head.

    in_1 -> 1x1 conv ---
                        |
    in_2 -> 1x1 conv -- |
                       ||
    in_3 -> 1x1 conv - ||
                      |||                  /-> 1x1 conv (mask prediction)
    in_4 -> 1x1 conv -----> 3x3 convs (*4)
                        |                  \\-> 1x1 conv (feature)
    in_5 -> 1x1 conv ---
    """

    def __init__(self, num_ins, fusion_level, num_convs=4, in_channels=256, conv_out_channels=256, num_classes=183, ignore_label=255, loss_weight=0.2, conv_cfg=None, norm_cfg=None):
        super(FusedSemanticHead, self).__init__()
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(ConvModule(self.in_channels, self.in_channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, inplace=False))
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else conv_out_channels
            self.convs.append(ConvModule(in_channels, conv_out_channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        self.conv_embedding = ConvModule(conv_out_channels, conv_out_channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        self.conv_logits = nn.Conv2d(conv_out_channels, self.num_classes, 1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)

    def init_weights(self):
        kaiming_init(self.conv_logits)

    @auto_fp16()
    def forward(self, feats):
        x = self.lateral_convs[self.fusion_level](feats[self.fusion_level])
        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(feats):
            if i != self.fusion_level:
                feat = F.interpolate(feat, size=fused_size, mode='bilinear', align_corners=True)
                x += self.lateral_convs[i](feat)
        for i in range(self.num_convs):
            x = self.convs[i](x)
        mask_pred = self.conv_logits(x)
        x = self.conv_embedding(x)
        return mask_pred, x

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_pred, labels):
        labels = labels.squeeze(1).long()
        loss_semantic_seg = self.criterion(mask_pred, labels)
        loss_semantic_seg *= self.loss_weight
        return loss_semantic_seg


class GridHead(nn.Module):

    def __init__(self, grid_points=9, num_convs=8, roi_feat_size=14, in_channels=256, conv_kernel_size=3, point_feat_channels=64, deconv_kernel_size=4, class_agnostic=False, loss_grid=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=15), conv_cfg=None, norm_cfg=dict(type='GN', num_groups=36)):
        super(GridHead, self).__init__()
        self.grid_points = grid_points
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.point_feat_channels = point_feat_channels
        self.conv_out_channels = self.point_feat_channels * self.grid_points
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        if isinstance(norm_cfg, dict) and norm_cfg['type'] == 'GN':
            assert self.conv_out_channels % norm_cfg['num_groups'] == 0
        assert self.grid_points >= 4
        self.grid_size = int(np.sqrt(self.grid_points))
        if self.grid_size * self.grid_size != self.grid_points:
            raise ValueError('grid_points must be a square number')
        self.whole_map_size = self.roi_feat_size * 4
        self.sub_regions = self.calc_sub_regions()
        self.convs = []
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else self.conv_out_channels
            stride = 2 if i == 0 else 1
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(ConvModule(in_channels, self.conv_out_channels, self.conv_kernel_size, stride=stride, padding=padding, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=True))
        self.convs = nn.Sequential(*self.convs)
        self.deconv1 = nn.ConvTranspose2d(self.conv_out_channels, self.conv_out_channels, kernel_size=deconv_kernel_size, stride=2, padding=(deconv_kernel_size - 2) // 2, groups=grid_points)
        self.norm1 = nn.GroupNorm(grid_points, self.conv_out_channels)
        self.deconv2 = nn.ConvTranspose2d(self.conv_out_channels, grid_points, kernel_size=deconv_kernel_size, stride=2, padding=(deconv_kernel_size - 2) // 2, groups=grid_points)
        self.neighbor_points = []
        grid_size = self.grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                neighbors = []
                if i > 0:
                    neighbors.append((i - 1) * grid_size + j)
                if j > 0:
                    neighbors.append(i * grid_size + j - 1)
                if j < grid_size - 1:
                    neighbors.append(i * grid_size + j + 1)
                if i < grid_size - 1:
                    neighbors.append((i + 1) * grid_size + j)
                self.neighbor_points.append(tuple(neighbors))
        self.num_edges = sum([len(p) for p in self.neighbor_points])
        self.forder_trans = nn.ModuleList()
        self.sorder_trans = nn.ModuleList()
        for neighbors in self.neighbor_points:
            fo_trans = nn.ModuleList()
            so_trans = nn.ModuleList()
            for _ in range(len(neighbors)):
                fo_trans.append(nn.Sequential(nn.Conv2d(self.point_feat_channels, self.point_feat_channels, 5, stride=1, padding=2, groups=self.point_feat_channels), nn.Conv2d(self.point_feat_channels, self.point_feat_channels, 1)))
                so_trans.append(nn.Sequential(nn.Conv2d(self.point_feat_channels, self.point_feat_channels, 5, 1, 2, groups=self.point_feat_channels), nn.Conv2d(self.point_feat_channels, self.point_feat_channels, 1)))
            self.forder_trans.append(fo_trans)
            self.sorder_trans.append(so_trans)
        self.loss_grid = build_loss(loss_grid)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_init(m)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
        nn.init.constant_(self.deconv2.bias, -np.log(0.99 / 0.01))

    def forward(self, x):
        assert x.shape[-1] == x.shape[-2] == self.roi_feat_size
        x = self.convs(x)
        c = self.point_feat_channels
        x_fo = [None for _ in range(self.grid_points)]
        for i, points in enumerate(self.neighbor_points):
            x_fo[i] = x[:, i * c:(i + 1) * c]
            for j, point_idx in enumerate(points):
                x_fo[i] = x_fo[i] + self.forder_trans[i][j](x[:, point_idx * c:(point_idx + 1) * c])
        x_so = [None for _ in range(self.grid_points)]
        for i, points in enumerate(self.neighbor_points):
            x_so[i] = x[:, i * c:(i + 1) * c]
            for j, point_idx in enumerate(points):
                x_so[i] = x_so[i] + self.sorder_trans[i][j](x_fo[point_idx])
        x2 = torch.cat(x_so, dim=1)
        x2 = self.deconv1(x2)
        x2 = F.relu(self.norm1(x2), inplace=True)
        heatmap = self.deconv2(x2)
        if self.training:
            x1 = x
            x1 = self.deconv1(x1)
            x1 = F.relu(self.norm1(x1), inplace=True)
            heatmap_unfused = self.deconv2(x1)
        else:
            heatmap_unfused = heatmap
        return dict(fused=heatmap, unfused=heatmap_unfused)

    def calc_sub_regions(self):
        """Compute point specific representation regions.

        See Grid R-CNN Plus (https://arxiv.org/abs/1906.05688) for details.
        """
        half_size = self.whole_map_size // 4 * 2
        sub_regions = []
        for i in range(self.grid_points):
            x_idx = i // self.grid_size
            y_idx = i % self.grid_size
            if x_idx == 0:
                sub_x1 = 0
            elif x_idx == self.grid_size - 1:
                sub_x1 = half_size
            else:
                ratio = x_idx / (self.grid_size - 1) - 0.25
                sub_x1 = max(int(ratio * self.whole_map_size), 0)
            if y_idx == 0:
                sub_y1 = 0
            elif y_idx == self.grid_size - 1:
                sub_y1 = half_size
            else:
                ratio = y_idx / (self.grid_size - 1) - 0.25
                sub_y1 = max(int(ratio * self.whole_map_size), 0)
            sub_regions.append((sub_x1, sub_y1, sub_x1 + half_size, sub_y1 + half_size))
        return sub_regions

    def get_target(self, sampling_results, rcnn_train_cfg):
        pos_bboxes = torch.cat([res.pos_bboxes for res in sampling_results], dim=0).cpu()
        pos_gt_bboxes = torch.cat([res.pos_gt_bboxes for res in sampling_results], dim=0).cpu()
        assert pos_bboxes.shape == pos_gt_bboxes.shape
        x1 = pos_bboxes[:, 0] - (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / 2
        y1 = pos_bboxes[:, 1] - (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / 2
        x2 = pos_bboxes[:, 2] + (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / 2
        y2 = pos_bboxes[:, 3] + (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / 2
        pos_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        pos_bbox_ws = (pos_bboxes[:, 2] - pos_bboxes[:, 0]).unsqueeze(-1)
        pos_bbox_hs = (pos_bboxes[:, 3] - pos_bboxes[:, 1]).unsqueeze(-1)
        num_rois = pos_bboxes.shape[0]
        map_size = self.whole_map_size
        targets = torch.zeros((num_rois, self.grid_points, map_size, map_size), dtype=torch.float)
        factors = []
        for j in range(self.grid_points):
            x_idx = j // self.grid_size
            y_idx = j % self.grid_size
            factors.append((1 - x_idx / (self.grid_size - 1), 1 - y_idx / (self.grid_size - 1)))
        radius = rcnn_train_cfg.pos_radius
        radius2 = radius ** 2
        for i in range(num_rois):
            if pos_bbox_ws[i] <= self.grid_size or pos_bbox_hs[i] <= self.grid_size:
                continue
            for j in range(self.grid_points):
                factor_x, factor_y = factors[j]
                gridpoint_x = factor_x * pos_gt_bboxes[i, 0] + (1 - factor_x) * pos_gt_bboxes[i, 2]
                gridpoint_y = factor_y * pos_gt_bboxes[i, 1] + (1 - factor_y) * pos_gt_bboxes[i, 3]
                cx = int((gridpoint_x - pos_bboxes[i, 0]) / pos_bbox_ws[i] * map_size)
                cy = int((gridpoint_y - pos_bboxes[i, 1]) / pos_bbox_hs[i] * map_size)
                for x in range(cx - radius, cx + radius + 1):
                    for y in range(cy - radius, cy + radius + 1):
                        if x >= 0 and x < map_size and y >= 0 and y < map_size:
                            if (x - cx) ** 2 + (y - cy) ** 2 <= radius2:
                                targets[i, j, y, x] = 1
        sub_targets = []
        for i in range(self.grid_points):
            sub_x1, sub_y1, sub_x2, sub_y2 = self.sub_regions[i]
            sub_targets.append(targets[:, [i], sub_y1:sub_y2, sub_x1:sub_x2])
        sub_targets = torch.cat(sub_targets, dim=1)
        sub_targets = sub_targets
        return sub_targets

    def loss(self, grid_pred, grid_targets):
        loss_fused = self.loss_grid(grid_pred['fused'], grid_targets)
        loss_unfused = self.loss_grid(grid_pred['unfused'], grid_targets)
        loss_grid = loss_fused + loss_unfused
        return dict(loss_grid=loss_grid)

    def get_bboxes(self, det_bboxes, grid_pred, img_meta):
        assert det_bboxes.shape[0] == grid_pred.shape[0]
        det_bboxes = det_bboxes.cpu()
        cls_scores = det_bboxes[:, [4]]
        det_bboxes = det_bboxes[:, :4]
        grid_pred = grid_pred.sigmoid().cpu()
        R, c, h, w = grid_pred.shape
        half_size = self.whole_map_size // 4 * 2
        assert h == w == half_size
        assert c == self.grid_points
        grid_pred = grid_pred.view(R * c, h * w)
        pred_scores, pred_position = grid_pred.max(dim=1)
        xs = pred_position % w
        ys = pred_position // w
        for i in range(self.grid_points):
            xs[i::self.grid_points] += self.sub_regions[i][0]
            ys[i::self.grid_points] += self.sub_regions[i][1]
        pred_scores, xs, ys = tuple(map(lambda x: x.view(R, c), [pred_scores, xs, ys]))
        widths = (det_bboxes[:, 2] - det_bboxes[:, 0]).unsqueeze(-1)
        heights = (det_bboxes[:, 3] - det_bboxes[:, 1]).unsqueeze(-1)
        x1 = det_bboxes[:, 0, None] - widths / 2
        y1 = det_bboxes[:, 1, None] - heights / 2
        abs_xs = (xs.float() + 0.5) / w * widths + x1
        abs_ys = (ys.float() + 0.5) / h * heights + y1
        x1_inds = [i for i in range(self.grid_size)]
        y1_inds = [(i * self.grid_size) for i in range(self.grid_size)]
        x2_inds = [(self.grid_points - self.grid_size + i) for i in range(self.grid_size)]
        y2_inds = [((i + 1) * self.grid_size - 1) for i in range(self.grid_size)]
        bboxes_x1 = (abs_xs[:, x1_inds] * pred_scores[:, x1_inds]).sum(dim=1, keepdim=True) / pred_scores[:, x1_inds].sum(dim=1, keepdim=True)
        bboxes_y1 = (abs_ys[:, y1_inds] * pred_scores[:, y1_inds]).sum(dim=1, keepdim=True) / pred_scores[:, y1_inds].sum(dim=1, keepdim=True)
        bboxes_x2 = (abs_xs[:, x2_inds] * pred_scores[:, x2_inds]).sum(dim=1, keepdim=True) / pred_scores[:, x2_inds].sum(dim=1, keepdim=True)
        bboxes_y2 = (abs_ys[:, y2_inds] * pred_scores[:, y2_inds]).sum(dim=1, keepdim=True) / pred_scores[:, y2_inds].sum(dim=1, keepdim=True)
        bbox_res = torch.cat([bboxes_x1, bboxes_y1, bboxes_x2, bboxes_y2, cls_scores], dim=1)
        bbox_res[:, [0, 2]].clamp_(min=0, max=img_meta[0]['img_shape'][1] - 1)
        bbox_res[:, [1, 3]].clamp_(min=0, max=img_meta[0]['img_shape'][0] - 1)
        return bbox_res


class MaskIoUHead(nn.Module):
    """Mask IoU Head.

    This head predicts the IoU of predicted masks and corresponding gt masks.
    """

    def __init__(self, num_convs=4, num_fcs=2, roi_feat_size=14, in_channels=256, conv_out_channels=256, fc_out_channels=1024, num_classes=81, loss_iou=dict(type='MSELoss', loss_weight=0.5)):
        super(MaskIoUHead, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            if i == 0:
                in_channels = self.in_channels + 1
            else:
                in_channels = self.conv_out_channels
            stride = 2 if i == num_convs - 1 else 1
            self.convs.append(nn.Conv2d(in_channels, self.conv_out_channels, 3, stride=stride, padding=1))
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = self.conv_out_channels * (roi_feat_size // 2) ** 2 if i == 0 else self.fc_out_channels
            self.fcs.append(nn.Linear(in_channels, self.fc_out_channels))
        self.fc_mask_iou = nn.Linear(self.fc_out_channels, self.num_classes)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.loss_iou = build_loss(loss_iou)

    def init_weights(self):
        for conv in self.convs:
            kaiming_init(conv)
        for fc in self.fcs:
            kaiming_init(fc, a=1, mode='fan_in', nonlinearity='leaky_relu', distribution='uniform')
        normal_init(self.fc_mask_iou, std=0.01)

    def forward(self, mask_feat, mask_pred):
        mask_pred = mask_pred.sigmoid()
        mask_pred_pooled = self.max_pool(mask_pred.unsqueeze(1))
        x = torch.cat((mask_feat, mask_pred_pooled), 1)
        for conv in self.convs:
            x = self.relu(conv(x))
        x = x.view(x.size(0), -1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_iou = self.fc_mask_iou(x)
        return mask_iou

    def loss(self, mask_iou_pred, mask_iou_targets):
        pos_inds = mask_iou_targets > 0
        if pos_inds.sum() > 0:
            loss_mask_iou = self.loss_iou(mask_iou_pred[pos_inds], mask_iou_targets[pos_inds])
        else:
            loss_mask_iou = mask_iou_pred * 0
        return dict(loss_mask_iou=loss_mask_iou)

    def get_target(self, sampling_results, gt_masks, mask_pred, mask_targets, rcnn_train_cfg):
        """Compute target of mask IoU.

        Mask IoU target is the IoU of the predicted mask (inside a bbox) and
        the gt mask of corresponding gt mask (the whole instance).
        The intersection area is computed inside the bbox, and the gt mask area
        is computed with two steps, firstly we compute the gt area inside the
        bbox, then divide it by the area ratio of gt area inside the bbox and
        the gt area of the whole instance.

        Args:
            sampling_results (list[:obj:`SamplingResult`]): sampling results.
            gt_masks (list[ndarray]): Gt masks (the whole instance) of each
                image, binary maps with the same shape of the input image.
            mask_pred (Tensor): Predicted masks of each positive proposal,
                shape (num_pos, h, w).
            mask_targets (Tensor): Gt mask of each positive proposal,
                binary map of the shape (num_pos, h, w).
            rcnn_train_cfg (dict): Training config for R-CNN part.

        Returns:
            Tensor: mask iou target (length == num positive).
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        area_ratios = map(self._get_area_ratio, pos_proposals, pos_assigned_gt_inds, gt_masks)
        area_ratios = torch.cat(list(area_ratios))
        assert mask_targets.size(0) == area_ratios.size(0)
        mask_pred = (mask_pred > rcnn_train_cfg.mask_thr_binary).float()
        mask_pred_areas = mask_pred.sum((-1, -2))
        overlap_areas = (mask_pred * mask_targets).sum((-1, -2))
        gt_full_areas = mask_targets.sum((-1, -2)) / (area_ratios + 1e-07)
        mask_iou_targets = overlap_areas / (mask_pred_areas + gt_full_areas - overlap_areas)
        return mask_iou_targets

    def _get_area_ratio(self, pos_proposals, pos_assigned_gt_inds, gt_masks):
        """Compute area ratio of the gt mask inside the proposal and the gt
        mask of the corresponding instance"""
        num_pos = pos_proposals.size(0)
        if num_pos > 0:
            area_ratios = []
            proposals_np = pos_proposals.cpu().numpy()
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            gt_instance_mask_area = gt_masks.sum((-1, -2))
            for i in range(num_pos):
                gt_mask = gt_masks[pos_assigned_gt_inds[i]]
                x1, y1, x2, y2 = proposals_np[i, :].astype(np.int32)
                gt_mask_in_proposal = gt_mask[y1:y2 + 1, x1:x2 + 1]
                ratio = gt_mask_in_proposal.sum() / (gt_instance_mask_area[pos_assigned_gt_inds[i]] + 1e-07)
                area_ratios.append(ratio)
            area_ratios = torch.from_numpy(np.stack(area_ratios)).float()
        else:
            area_ratios = pos_proposals.new_zeros((0,))
        return area_ratios

    def get_mask_scores(self, mask_iou_pred, det_bboxes, det_labels):
        """Get the mask scores.

        mask_score = bbox_score * mask_iou
        """
        inds = range(det_labels.size(0))
        mask_scores = mask_iou_pred[inds, det_labels + 1] * det_bboxes[inds, -1]
        mask_scores = mask_scores.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        return [mask_scores[det_labels == i] for i in range(self.num_classes - 1)]


class HitNeck(nn.Module):

    def __init__(self, num_fm=4, in_channel=[256], out_channel=256, latency=None, gamma=0.02, genotype=None, **kwargs):
        super(HitNeck, self).__init__()
        self.num_fm = num_fm
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.genotype = genotype
        bn_type = kwargs.get('bn_type', 'BN')
        self.cells = nn.ModuleList()
        input_size = [160, 80, 40, 20]
        for i, ops in enumerate(genotype):
            if i < self.num_fm:
                cell = OPS[PRIMITIVES[ops]](input_size[i % self.num_fm], in_channel[i % self.num_fm], out_channel, 1, bn=bn_type)
            else:
                cell = OPS[PRIMITIVES[ops]](input_size[i % self.num_fm], out_channel, out_channel, 1, bn=bn_type)
            self.cells.append(cell)
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)

    def forward(self, x, step):
        assert step in [1, 2]
        _step = step - 1
        out = []
        for i in range(_step * self.num_fm, step * self.num_fm):
            out.append(self.cells[i](x[i % self.num_fm]))
        return out


NECKS = Registry('neck')


class NonLocal2D(nn.Module):
    """Non-local module.

    See https://arxiv.org/abs/1711.07971 for details.

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
        conv_cfg (dict): The config dict for convolution layers.
            (only applicable to conv_out)
        norm_cfg (dict): The config dict for normalization layers.
            (only applicable to conv_out)
        mode (str): Options are `embedded_gaussian` and `dot_product`.
    """

    def __init__(self, in_channels, reduction=2, use_scale=True, conv_cfg=None, norm_cfg=None, mode='embedded_gaussian'):
        super(NonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']
        self.g = ConvModule(self.in_channels, self.inter_channels, kernel_size=1, activation=None)
        self.theta = ConvModule(self.in_channels, self.inter_channels, kernel_size=1, activation=None)
        self.phi = ConvModule(self.in_channels, self.inter_channels, kernel_size=1, activation=None)
        self.conv_out = ConvModule(self.inter_channels, self.in_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=None)
        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.g, self.theta, self.phi]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1] ** -0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        n, _, h, w = x.shape
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(n, self.inter_channels, -1)
        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)
        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)
        output = x + self.conv_out(y)
        return output


class BFP(nn.Module):
    """BFP (Balanced Feature Pyrmamids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    https://arxiv.org/pdf/1904.02701.pdf for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
    """

    def __init__(self, in_channels, num_levels, refine_level=2, refine_type=None, conv_cfg=None, norm_cfg=None):
        super(BFP, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']
        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels
        if self.refine_type == 'conv':
            self.refine = ConvModule(self.in_channels, self.in_channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2D(self.in_channels, reduction=1, use_scale=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.num_levels
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)
        bsf = sum(feats) / len(feats)
        if self.refine_type is not None:
            bsf = self.refine(bsf)
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])
        return tuple(outs)


class FPN(nn.Module):

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, add_extra_convs=False, extra_convs_on_inputs=True, relu_before_extra_convs=False, conv_cfg=None, norm_cfg=None, activation=None, fpn_kernel=3, lateral_kernel=1, depthwise=None, toy_replace=None, dense_add=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False
        self.fpn_kernel = fpn_kernel
        self.lateral_kernel = lateral_kernel
        self.dense_add = dense_add
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, lateral_kernel, padding=(lateral_kernel - 1) // 2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=self.activation, inplace=False)
            if depthwise is not None:
                if depthwise == 'sep':
                    fpn_conv = nn.Conv2d(out_channels, out_channels, self.fpn_kernel, padding=int((self.fpn_kernel - 1) / 2), groups=out_channels)
                elif depthwise == 'sep-depth':
                    fpn_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, self.fpn_kernel, padding=int((self.fpn_kernel - 1) / 2), groups=out_channels), nn.Conv2d(out_channels, out_channels, 1, padding=0))
            elif toy_replace is not None and i == toy_replace.get('stage', 30):
                if toy_replace.get('block', 'res') == 'ir':
                    fpn_conv = MBBlock(out_channels, out_channels, 1, 1, toy_replace.get('conv_kernel'), dilation=toy_replace.get('dilation'), groups=1)
                else:
                    fpn_conv = ConvModule(out_channels, out_channels, toy_replace.get('conv_kernel'), padding=(toy_replace.get('conv_kernel') - 1) * toy_replace.get('dilation') // 2, dilation=toy_replace.get('dilation'), conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=self.activation, inplace=False)
            else:
                fpn_conv = ConvModule(out_channels, out_channels, self.fpn_kernel, padding=int((self.fpn_kernel - 1) / 2), conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=self.activation, inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(in_channels, out_channels, 3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=self.activation, inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        if self.dense_add is not None:
            if self.dense_add == 'no':
                laterals = laterals
            elif self.dense_add == 'all':
                laterals_ = [(0) for i in range(len(laterals))]
                for i in range(used_backbone_levels - 1, -1, -1):
                    h, w = laterals[i].size(2), laterals[i].size(3)
                    for j in range(len(laterals)):
                        for k in range(i - j):
                            if k == 0:
                                tmp_lateral = F.max_pool2d(laterals[j], 2, stride=2)
                            else:
                                tmp_lateral = F.max_pool2d(tmp_lateral, 2, stride=2)
                        if i > j:
                            laterals_[i] += F.interpolate(tmp_lateral, size=(h, w), mode='bilinear', align_corners=True)
                        else:
                            laterals_[i] += F.interpolate(laterals[j], size=(h, w), mode='bilinear', align_corners=True)
                laterals = laterals_
            elif self.dense_add == 'top-down':
                laterals_ = [(0) for i in range(len(laterals))]
                for i in range(used_backbone_levels - 1, -1, -1):
                    h, w = laterals[i].size(2), laterals[i].size(3)
                    for j in range(used_backbone_levels - 1, i - 1, -1):
                        laterals_[i] += F.interpolate(laterals[j], size=(h, w), mode='nearest')
                laterals = laterals_
            elif self.dense_add == 'bottom-up-nearest':
                for i in range(0, used_backbone_levels - 1, 1):
                    laterals[i + 1] += F.max_pool2d(laterals[i], 1, stride=2)
            elif self.dense_add == 'bottom-up':
                laterals_ = [(0) for i in range(len(laterals))]
                for i in range(used_backbone_levels - 1, -1, -1):
                    h, w = laterals[i].size(2), laterals[i].size(3)
                    for j in range(i + 1):
                        for k in range(i - j):
                            if k == 0:
                                tmp_lateral = F.max_pool2d(laterals[j], 2, stride=2)
                            else:
                                tmp_lateral = F.max_pool2d(tmp_lateral, 2, stride=2)
                        if i > j:
                            laterals_[i] += F.interpolate(tmp_lateral, size=(h, w), mode='bilinear', align_corners=True)
                        else:
                            laterals_[i] += F.interpolate(laterals[j], size=(h, w), mode='bilinear', align_corners=True)
                laterals = laterals_
        else:
            for i in range(used_backbone_levels - 1, 0, -1):
                laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
        if self.fpn_kernel == 1 or self.fpn_kernel == 3:
            outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        else:
            outs = [laterals[i] for i in range(used_backbone_levels)]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class PAFPN(nn.Module):
    """ PAFPN Arch
        lateral      TD    3x3    BU
    C5 --------> C5     P5     N5    N5
        lateral    
    C4 --------> C4     P4     N4    N4
        lateral    
    C3 --------> C3     P3     N3    N3
        lateral    
    C2 --------> C2     P2     N2    N2
    """

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, add_extra_convs=False, extra_convs_on_inputs=True, relu_before_extra_convs=False, conv_cfg=None, norm_cfg=None, activation=None, lateral_kernel=1, fpn_kernel=3, bottom_up_kernel=3, pa_kernel=3):
        super(PAFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False
        self.fpn_kernel = fpn_kernel
        self.lateral_kernel = lateral_kernel
        self.bottom_up_kernel = bottom_up_kernel
        self.pa_kernel = pa_kernel
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.bottom_up_convs = nn.ModuleList()
        self.pa_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, lateral_kernel, padding=(lateral_kernel - 1) // 2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=None, inplace=True)
            fpn_conv = ConvModule(out_channels, out_channels, fpn_kernel, padding=(fpn_kernel - 1) // 2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=None, inplace=True)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        for i in range(self.start_level, self.backbone_end_level - 1):
            if bottom_up_kernel > 0:
                bottom_up_conv = ConvModule(out_channels, out_channels, bottom_up_kernel, stride=2, padding=(bottom_up_kernel - 1) // 2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=activation, inplace=True)
                self.bottom_up_convs.append(bottom_up_conv)
            if pa_kernel > 0:
                pa_conv = ConvModule(out_channels, out_channels, pa_kernel, padding=(pa_kernel - 1) // 2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=activation, inplace=True)
                self.pa_convs.append(pa_conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(in_channels, out_channels, 3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=self.activation, inplace=True)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
        fpn_middle = [fpn_conv(laterals[i]) for i, fpn_conv in enumerate(self.fpn_convs)]
        if self.pa_kernel > 0:
            outs = [fpn_middle[0]]
            for i in range(0, self.backbone_end_level - self.start_level - 1):
                if self.bottom_up_kernel > 0:
                    tmp = self.bottom_up_convs[i](outs[i]) + fpn_middle[i + 1]
                else:
                    tmp = F.max_pool2d(outs[i], 2, stride=2) + fpn_middle[i + 1]
                outs.append(self.pa_convs[i](tmp))
        else:
            outs = fpn_middle
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


def caffe2_xavier_init(module, bias=0):
    kaiming_init(module, a=1, mode='fan_in', nonlinearity='leaky_relu', distribution='uniform')


class HRFPN(nn.Module):
    """HRFPN (High Resolution Feature Pyrmamids)

    arXiv: https://arxiv.org/abs/1904.04514

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    def __init__(self, in_channels, out_channels, num_outs=5, pooling_type='AVG', conv_cfg=None, norm_cfg=None, with_cp=False):
        super(HRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reduction_conv = ConvModule(sum(in_channels), out_channels, kernel_size=1, conv_cfg=self.conv_cfg, activation=None)
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_convs.append(ConvModule(out_channels, out_channels, kernel_size=3, padding=1, conv_cfg=self.conv_cfg, activation=None))
        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        assert len(inputs) == self.num_ins
        outs = [inputs[0]]
        for i in range(1, self.num_ins):
            outs.append(F.interpolate(inputs[i], scale_factor=2 ** i, mode='bilinear'))
        out = torch.cat(outs, dim=1)
        if out.requires_grad and self.with_cp:
            out = checkpoint(self.reduction_conv, out)
        else:
            out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, self.num_outs):
            outs.append(self.pooling(out, kernel_size=2 ** i, stride=2 ** i))
        outputs = []
        for i in range(self.num_outs):
            if outs[i].requires_grad and self.with_cp:
                tmp_out = checkpoint(self.fpn_convs[i], outs[i])
            else:
                tmp_out = self.fpn_convs[i](outs[i])
            outputs.append(tmp_out)
        return tuple(outputs)


class MergingCell(nn.Module):

    def __init__(self, channels=256, with_conv=True, norm_type='BN'):
        super(MergingCell, self).__init__()
        self.with_conv = with_conv
        norm_layer = norm_cfg_[norm_type]
        if self.with_conv:
            self.conv_out = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, 1, 1), norm_layer(channels))

    def _binary_op(self, x1, x2):
        raise NotImplementedError

    def _resize(self, x, size):
        if x.shape[-2:] == size:
            return x
        elif x.shape[-2:] < size:
            return F.interpolate(x, size=size, mode='nearest')
        else:
            assert x.shape[-2] % size[-2] == 0 and x.shape[-1] % size[-1] == 0
            kernel_size = x.shape[-1] // size[-1]
            x = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
            return x

    def forward(self, x1, x2, out_size):
        assert x1.shape[:2] == x2.shape[:2]
        assert len(out_size) == 2
        x1 = self._resize(x1, out_size)
        x2 = self._resize(x2, out_size)
        x = self._binary_op(x1, x2)
        if self.with_conv:
            x = self.conv_out(x)
        return x


class SumCell(MergingCell):

    def _binary_op(self, x1, x2):
        return x1 + x2


class GPCell(MergingCell):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _binary_op(self, x1, x2):
        x2_att = self.global_pool(x2).sigmoid()
        return x2 + x2_att * x1


class NASFPN(nn.Module):

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, add_extra_convs=False, stack_times=7, lateral_kernel=1, norm_type='SyncBN'):
        super(NASFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.stack_times = stack_times
        self.norm_type = norm_type
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.start_level + num_outs):
            in_channel = in_channels[i] if i < self.backbone_end_level else in_channels[-1]
            padding = (lateral_kernel - 1) // 2
            l_conv = nn.Conv2d(in_channel, out_channels, kernel_size=lateral_kernel, padding=padding)
            self.lateral_convs.append(l_conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_downsamples = nn.ModuleList()
        for i in range(extra_levels):
            if self.add_extra_convs:
                extra_conv = nn.Conv2d(in_channels[-1], in_channels[-1], 3, stride=2, padding=1)
                self.extra_downsamples.append(extra_conv)
            else:
                self.extra_downsamples.append(nn.MaxPool2d(2, stride=2))
        self.fpn_stages = nn.ModuleList()
        for _ in range(self.stack_times):
            stage = nn.ModuleDict()
            stage['gp_64_4'] = GPCell(out_channels, norm_type=norm_type)
            stage['sum_44_4'] = SumCell(out_channels, norm_type=norm_type)
            stage['sum_43_3'] = SumCell(out_channels, norm_type=norm_type)
            stage['sum_34_4'] = SumCell(out_channels, norm_type=norm_type)
            stage['gp_43_5'] = GPCell(with_conv=False)
            stage['sum_55_5'] = SumCell(out_channels, norm_type=norm_type)
            stage['gp_54_7'] = GPCell(with_conv=False)
            stage['sum_77_7'] = SumCell(out_channels, norm_type=norm_type)
            stage['gp_75_6'] = GPCell(out_channels, norm_type=norm_type)
            self.fpn_stages.append(stage)
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = list(inputs)
        for downsample in self.extra_downsamples:
            inputs.append(downsample(inputs[-1]))
        feats = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        p3, p4, p5, p6, p7 = feats
        for stage in self.fpn_stages:
            p4_1 = stage['gp_64_4'](p6, p4, out_size=p4.shape[-2:])
            p4_2 = stage['sum_44_4'](p4_1, p4, out_size=p4.shape[-2:])
            p3 = stage['sum_43_3'](p4_2, p3, out_size=p3.shape[-2:])
            p4 = stage['sum_34_4'](p3, p4_2, out_size=p4.shape[-2:])
            p5_tmp = stage['gp_43_5'](p4, p3, out_size=p5.shape[-2:])
            p5 = stage['sum_55_5'](p5, p5_tmp, out_size=p5.shape[-2:])
            p7_tmp = stage['gp_54_7'](p5, p4_2, out_size=p7.shape[-2:])
            p7 = stage['sum_77_7'](p7, p7_tmp, out_size=p7.shape[-2:])
            p6 = stage['gp_75_6'](p7, p5, out_size=p6.shape[-2:])
        return tuple([p3, p4, p5, p6, p7])


def build_search_neck(cfg):
    """Build neck model from config dict.
    """
    if cfg is not None:
        cfg_ = cfg.copy()
        neck_type = cfg_.pop('type')
        if neck_type == 'DARTS':
            raise NotImplementedError
        elif neck_type == 'HitDet':
            return HitNeck(**cfg_)
        else:
            raise KeyError('Invalid neck type {}'.fromat(neck_type))
    else:
        return None


class SearchPAFPN(nn.Module):
    """ PAFPN Arch
        TBS      TD      TBS      BU
    C5 -----> C5     P5 -----> N5    N5
            
    C4 -----> C4     P4 -----> N4    N4
            
    C3 -----> C3     P3 -----> N3    N3
           
    C2 -----> C2     P2 -----> N2    N2
    """

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, add_extra_convs=False, extra_convs_on_inputs=True, relu_before_extra_convs=False, conv_cfg=None, norm_cfg=None, activation=None, pa_kernel=3, search_neck=None):
        super(SearchPAFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False
        self.pa_kernel = pa_kernel
        self.SearchNeck = build_search_neck(search_neck)
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.pa_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level - 1):
            if pa_kernel > 0:
                pa_conv = ConvModule(out_channels, out_channels, pa_kernel, padding=(pa_kernel - 1) // 2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=activation, inplace=True)
                self.pa_convs.append(pa_conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            self.fpn_convs = nn.ModuleList()
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channel = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channel = out_channels
                extra_fpn_conv = ConvModule(in_channel, out_channels, 3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=self.activation, inplace=True)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = self.SearchNeck(inputs[self.start_level:], 1)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
        laterals_mid = self.SearchNeck(laterals, 2)
        if self.pa_kernel > 0:
            outs = [laterals_mid[0]]
            for i in range(0, self.backbone_end_level - self.start_level - 1):
                tmp = F.max_pool2d(outs[i], 2, stride=2) + laterals_mid[i + 1]
                outs.append(self.pa_convs[i](tmp))
        else:
            outs = laterals_mid
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[0](orig))
                else:
                    outs.append(self.fpn_convs[0](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i - used_backbone_levels](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i - used_backbone_levels](outs[-1]))
        return tuple(outs), None


class GeneralizedAttention(nn.Module):
    """GeneralizedAttention module.

    See 'An Empirical Study of Spatial Attention Mechanisms in Deep Networks'
    (https://arxiv.org/abs/1711.07971) for details.

    Args:
        in_dim (int): Channels of the input feature map.
        spatial_range (int): The spatial range.
            -1 indicates no spatial range constraint.
        num_heads (int): The head number of empirical_attention module.
        position_embedding_dim (int): The position embedding dimension.
        position_magnitude (int): A multiplier acting on coord difference.
        kv_stride (int): The feature stride acting on key/value feature map.
        q_stride (int): The feature stride acting on query feature map.
        attention_type (str): A binary indicator string for indicating which
            items in generalized empirical_attention module are used.
            '1000' indicates 'query and key content' (appr - appr) item,
            '0100' indicates 'query content and relative position'
              (appr - position) item,
            '0010' indicates 'key content only' (bias - appr) item,
            '0001' indicates 'relative position only' (bias - position) item.
    """

    def __init__(self, in_dim, spatial_range=-1, num_heads=9, position_embedding_dim=-1, position_magnitude=1, kv_stride=2, q_stride=1, attention_type='1111'):
        super(GeneralizedAttention, self).__init__()
        self.position_embedding_dim = position_embedding_dim if position_embedding_dim > 0 else in_dim
        self.position_magnitude = position_magnitude
        self.num_heads = num_heads
        self.channel_in = in_dim
        self.spatial_range = spatial_range
        self.kv_stride = kv_stride
        self.q_stride = q_stride
        self.attention_type = [bool(int(_)) for _ in attention_type]
        self.qk_embed_dim = in_dim // num_heads
        out_c = self.qk_embed_dim * num_heads
        if self.attention_type[0] or self.attention_type[1]:
            self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_c, kernel_size=1, bias=False)
            self.query_conv.kaiming_init = True
        if self.attention_type[0] or self.attention_type[2]:
            self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_c, kernel_size=1, bias=False)
            self.key_conv.kaiming_init = True
        self.v_dim = in_dim // num_heads
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.v_dim * num_heads, kernel_size=1, bias=False)
        self.value_conv.kaiming_init = True
        if self.attention_type[1] or self.attention_type[3]:
            self.appr_geom_fc_x = nn.Linear(self.position_embedding_dim // 2, out_c, bias=False)
            self.appr_geom_fc_x.kaiming_init = True
            self.appr_geom_fc_y = nn.Linear(self.position_embedding_dim // 2, out_c, bias=False)
            self.appr_geom_fc_y.kaiming_init = True
        if self.attention_type[2]:
            stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
            appr_bias_value = -2 * stdv * torch.rand(out_c) + stdv
            self.appr_bias = nn.Parameter(appr_bias_value)
        if self.attention_type[3]:
            stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
            geom_bias_value = -2 * stdv * torch.rand(out_c) + stdv
            self.geom_bias = nn.Parameter(geom_bias_value)
        self.proj_conv = nn.Conv2d(in_channels=self.v_dim * num_heads, out_channels=in_dim, kernel_size=1, bias=True)
        self.proj_conv.kaiming_init = True
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.spatial_range >= 0:
            if in_dim == 256:
                max_len = 84
            elif in_dim == 512:
                max_len = 42
            max_len_kv = int((max_len - 1.0) / self.kv_stride + 1)
            local_constraint_map = np.ones((max_len, max_len, max_len_kv, max_len_kv), dtype=np.int)
            for iy in range(max_len):
                for ix in range(max_len):
                    local_constraint_map[iy, ix, max((iy - self.spatial_range) // self.kv_stride, 0):min((iy + self.spatial_range + 1) // self.kv_stride + 1, max_len), max((ix - self.spatial_range) // self.kv_stride, 0):min((ix + self.spatial_range + 1) // self.kv_stride + 1, max_len)] = 0
            self.local_constraint_map = nn.Parameter(torch.from_numpy(local_constraint_map).byte(), requires_grad=False)
        if self.q_stride > 1:
            self.q_downsample = nn.AvgPool2d(kernel_size=1, stride=self.q_stride)
        else:
            self.q_downsample = None
        if self.kv_stride > 1:
            self.kv_downsample = nn.AvgPool2d(kernel_size=1, stride=self.kv_stride)
        else:
            self.kv_downsample = None
        self.init_weights()

    def get_position_embedding(self, h, w, h_kv, w_kv, q_stride, kv_stride, device, feat_dim, wave_length=1000):
        h_idxs = torch.linspace(0, h - 1, h)
        h_idxs = h_idxs.view((h, 1)) * q_stride
        w_idxs = torch.linspace(0, w - 1, w)
        w_idxs = w_idxs.view((w, 1)) * q_stride
        h_kv_idxs = torch.linspace(0, h_kv - 1, h_kv)
        h_kv_idxs = h_kv_idxs.view((h_kv, 1)) * kv_stride
        w_kv_idxs = torch.linspace(0, w_kv - 1, w_kv)
        w_kv_idxs = w_kv_idxs.view((w_kv, 1)) * kv_stride
        h_diff = h_idxs.unsqueeze(1) - h_kv_idxs.unsqueeze(0)
        h_diff *= self.position_magnitude
        w_diff = w_idxs.unsqueeze(1) - w_kv_idxs.unsqueeze(0)
        w_diff *= self.position_magnitude
        feat_range = torch.arange(0, feat_dim / 4)
        dim_mat = torch.Tensor([wave_length])
        dim_mat = dim_mat ** (4.0 / feat_dim * feat_range)
        dim_mat = dim_mat.view((1, 1, -1))
        embedding_x = torch.cat(((w_diff / dim_mat).sin(), (w_diff / dim_mat).cos()), dim=2)
        embedding_y = torch.cat(((h_diff / dim_mat).sin(), (h_diff / dim_mat).cos()), dim=2)
        return embedding_x, embedding_y

    def forward(self, x_input):
        num_heads = self.num_heads
        if self.q_downsample is not None:
            x_q = self.q_downsample(x_input)
        else:
            x_q = x_input
        n, _, h, w = x_q.shape
        if self.kv_downsample is not None:
            x_kv = self.kv_downsample(x_input)
        else:
            x_kv = x_input
        _, _, h_kv, w_kv = x_kv.shape
        if self.attention_type[0] or self.attention_type[1]:
            proj_query = self.query_conv(x_q).view((n, num_heads, self.qk_embed_dim, h * w))
            proj_query = proj_query.permute(0, 1, 3, 2)
        if self.attention_type[0] or self.attention_type[2]:
            proj_key = self.key_conv(x_kv).view((n, num_heads, self.qk_embed_dim, h_kv * w_kv))
        if self.attention_type[1] or self.attention_type[3]:
            position_embed_x, position_embed_y = self.get_position_embedding(h, w, h_kv, w_kv, self.q_stride, self.kv_stride, x_input.device, self.position_embedding_dim)
            position_feat_x = self.appr_geom_fc_x(position_embed_x).view(1, w, w_kv, num_heads, self.qk_embed_dim).permute(0, 3, 1, 2, 4).repeat(n, 1, 1, 1, 1)
            position_feat_y = self.appr_geom_fc_y(position_embed_y).view(1, h, h_kv, num_heads, self.qk_embed_dim).permute(0, 3, 1, 2, 4).repeat(n, 1, 1, 1, 1)
            position_feat_x /= math.sqrt(2)
            position_feat_y /= math.sqrt(2)
        if np.sum(self.attention_type) == 1 and self.attention_type[2]:
            appr_bias = self.appr_bias.view(1, num_heads, 1, self.qk_embed_dim).repeat(n, 1, 1, 1)
            energy = torch.matmul(appr_bias, proj_key).view(n, num_heads, 1, h_kv * w_kv)
            h = 1
            w = 1
        else:
            if not self.attention_type[0]:
                energy = torch.zeros(n, num_heads, h, w, h_kv, w_kv, dtype=x_input.dtype, device=x_input.device)
            if self.attention_type[0] or self.attention_type[2]:
                if self.attention_type[0] and self.attention_type[2]:
                    appr_bias = self.appr_bias.view(1, num_heads, 1, self.qk_embed_dim)
                    energy = torch.matmul(proj_query + appr_bias, proj_key).view(n, num_heads, h, w, h_kv, w_kv)
                elif self.attention_type[0]:
                    energy = torch.matmul(proj_query, proj_key).view(n, num_heads, h, w, h_kv, w_kv)
                elif self.attention_type[2]:
                    appr_bias = self.appr_bias.view(1, num_heads, 1, self.qk_embed_dim).repeat(n, 1, 1, 1)
                    energy += torch.matmul(appr_bias, proj_key).view(n, num_heads, 1, 1, h_kv, w_kv)
            if self.attention_type[1] or self.attention_type[3]:
                if self.attention_type[1] and self.attention_type[3]:
                    geom_bias = self.geom_bias.view(1, num_heads, 1, self.qk_embed_dim)
                    proj_query_reshape = (proj_query + geom_bias).view(n, num_heads, h, w, self.qk_embed_dim)
                    energy_x = torch.matmul(proj_query_reshape.permute(0, 1, 3, 2, 4), position_feat_x.permute(0, 1, 2, 4, 3))
                    energy_x = energy_x.permute(0, 1, 3, 2, 4).unsqueeze(4)
                    energy_y = torch.matmul(proj_query_reshape, position_feat_y.permute(0, 1, 2, 4, 3))
                    energy_y = energy_y.unsqueeze(5)
                    energy += energy_x + energy_y
                elif self.attention_type[1]:
                    proj_query_reshape = proj_query.view(n, num_heads, h, w, self.qk_embed_dim)
                    proj_query_reshape = proj_query_reshape.permute(0, 1, 3, 2, 4)
                    position_feat_x_reshape = position_feat_x.permute(0, 1, 2, 4, 3)
                    position_feat_y_reshape = position_feat_y.permute(0, 1, 2, 4, 3)
                    energy_x = torch.matmul(proj_query_reshape, position_feat_x_reshape)
                    energy_x = energy_x.permute(0, 1, 3, 2, 4).unsqueeze(4)
                    energy_y = torch.matmul(proj_query_reshape, position_feat_y_reshape)
                    energy_y = energy_y.unsqueeze(5)
                    energy += energy_x + energy_y
                elif self.attention_type[3]:
                    geom_bias = self.geom_bias.view(1, num_heads, self.qk_embed_dim, 1).repeat(n, 1, 1, 1)
                    position_feat_x_reshape = position_feat_x.view(n, num_heads, w * w_kv, self.qk_embed_dim)
                    position_feat_y_reshape = position_feat_y.view(n, num_heads, h * h_kv, self.qk_embed_dim)
                    energy_x = torch.matmul(position_feat_x_reshape, geom_bias)
                    energy_x = energy_x.view(n, num_heads, 1, w, 1, w_kv)
                    energy_y = torch.matmul(position_feat_y_reshape, geom_bias)
                    energy_y = energy_y.view(n, num_heads, h, 1, h_kv, 1)
                    energy += energy_x + energy_y
            energy = energy.view(n, num_heads, h * w, h_kv * w_kv)
        if self.spatial_range >= 0:
            cur_local_constraint_map = self.local_constraint_map[:h, :w, :h_kv, :w_kv].contiguous().view(1, 1, h * w, h_kv * w_kv)
            energy = energy.masked_fill_(cur_local_constraint_map, float('-inf'))
        attention = F.softmax(energy, 3)
        proj_value = self.value_conv(x_kv)
        proj_value_reshape = proj_value.view((n, num_heads, self.v_dim, h_kv * w_kv)).permute(0, 1, 3, 2)
        out = torch.matmul(attention, proj_value_reshape).permute(0, 1, 3, 2).contiguous().view(n, self.v_dim * self.num_heads, h, w)
        out = self.proj_conv(out)
        out = self.gamma * out + x_input
        return out

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'kaiming_init') and m.kaiming_init:
                kaiming_init(m, mode='fan_in', nonlinearity='leaky_relu', bias=0, distribution='uniform', a=1)


ROI_EXTRACTORS = Registry('roi_extractor')


class SingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self, roi_layer, out_channels, featmap_strides, finest_scale=56):
        super(SingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.fp16_enabled = False

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList([layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt((rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-06))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def roi_rescale(self, rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5 + 0.5
        x2 = cx + new_w * 0.5 - 0.5
        y1 = cy - new_h * 0.5 + 0.5
        y2 = cy + new_h * 0.5 - 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    @force_fp32(apply_to=('feats',), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)
        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = feats[0].new_zeros(rois.size(0), self.out_channels, *out_size)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
        return roi_feats


SHARED_HEADS = Registry('shared_head')


class ResLayer(nn.Module):

    def __init__(self, depth, stage=3, stride=2, dilation=1, style='pytorch', norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True, with_cp=False, dcn=None):
        super(ResLayer, self).__init__()
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.stage = stage
        self.fp16_enabled = False
        block, stage_blocks = ResNet.arch_settings[depth]
        stage_block = stage_blocks[stage]
        planes = 64 * 2 ** stage
        inplanes = 64 * 2 ** (stage - 1) * block.expansion
        res_layer = make_res_layer(block, inplanes, planes, stage_block, stride=stride, dilation=dilation, style=style, with_cp=with_cp, norm_cfg=self.norm_cfg, dcn=dcn)
        self.add_module('layer{}'.format(stage + 1), res_layer)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    @auto_fp16()
    def forward(self, x):
        res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
        out = res_layer(x)
        return out

    def train(self, mode=True):
        super(ResLayer, self).train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


class DeformConvPack(DeformConv):

    def __init__(self, *args, **kwargs):
        super(DeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(self.in_channels, self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset_mask = nn.Conv2d(self.in_channels, self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx, data, rois, offset, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        ctx.spatial_scale = spatial_scale
        ctx.out_size = out_size
        ctx.out_channels = out_channels
        ctx.no_trans = no_trans
        ctx.group_size = group_size
        ctx.part_size = out_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std
        assert 0.0 <= ctx.trans_std <= 1.0
        if not data.is_cuda:
            raise NotImplementedError
        n = rois.shape[0]
        output = data.new_empty(n, out_channels, out_size, out_size)
        output_count = data.new_empty(n, out_channels, out_size, out_size)
        deform_pool_cuda.deform_psroi_pooling_cuda_forward(data, rois, offset, output, output_count, ctx.no_trans, ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, rois, offset)
        ctx.output_count = output_count
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        data, rois, offset = ctx.saved_tensors
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_rois = None
        grad_offset = torch.zeros_like(offset)
        deform_pool_cuda.deform_psroi_pooling_cuda_backward(grad_output, data, rois, offset, output_count, grad_input, grad_offset, ctx.no_trans, ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        return grad_input, grad_rois, grad_offset, None, None, None, None, None, None, None, None


deform_roi_pooling = DeformRoIPoolingFunction.apply


class DeformRoIPooling(nn.Module):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.out_size = out_size
        self.out_channels = out_channels
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = out_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, data, rois, offset):
        if self.no_trans:
            offset = data.new_empty(0)
        return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)


class DeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, num_offset_fcs=3, deform_fc_channels=1024):
        super(DeformRoIPoolingPack, self).__init__(spatial_scale, out_size, out_channels, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.num_offset_fcs = num_offset_fcs
        self.deform_fc_channels = deform_fc_channels
        if not no_trans:
            seq = []
            ic = self.out_size * self.out_size * self.out_channels
            for i in range(self.num_offset_fcs):
                if i < self.num_offset_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size * self.out_size * 2
                seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_offset_fcs - 1:
                    seq.append(nn.ReLU(inplace=True))
            self.offset_fc = nn.Sequential(*seq)
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
        else:
            n = rois.shape[0]
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, True, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size, self.out_size)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)


class ModulatedDeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, num_offset_fcs=3, num_mask_fcs=2, deform_fc_channels=1024):
        super(ModulatedDeformRoIPoolingPack, self).__init__(spatial_scale, out_size, out_channels, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.num_offset_fcs = num_offset_fcs
        self.num_mask_fcs = num_mask_fcs
        self.deform_fc_channels = deform_fc_channels
        if not no_trans:
            offset_fc_seq = []
            ic = self.out_size * self.out_size * self.out_channels
            for i in range(self.num_offset_fcs):
                if i < self.num_offset_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size * self.out_size * 2
                offset_fc_seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_offset_fcs - 1:
                    offset_fc_seq.append(nn.ReLU(inplace=True))
            self.offset_fc = nn.Sequential(*offset_fc_seq)
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()
            mask_fc_seq = []
            ic = self.out_size * self.out_size * self.out_channels
            for i in range(self.num_mask_fcs):
                if i < self.num_mask_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size * self.out_size
                mask_fc_seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_mask_fcs - 1:
                    mask_fc_seq.append(nn.ReLU(inplace=True))
                else:
                    mask_fc_seq.append(nn.Sigmoid())
            self.mask_fc = nn.Sequential(*mask_fc_seq)
            self.mask_fc[-2].weight.data.zero_()
            self.mask_fc[-2].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
        else:
            n = rois.shape[0]
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, True, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size, self.out_size)
            mask = self.mask_fc(x.view(n, -1))
            mask = mask.view(n, 1, self.out_size, self.out_size)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std) * mask


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock(nn.Module):

    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([(f in valid_fusion_types) for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True
        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class RoIAlignFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale, sample_num=0):
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new_zeros(num_rois, num_channels, out_h, out_w)
        if features.is_cuda:
            roi_align_cuda.forward(features, rois, out_h, out_w, spatial_scale, sample_num, output)
        else:
            raise NotImplementedError
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output.size(3)
        out_h = grad_output.size(2)
        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, data_height, data_width)
            roi_align_cuda.backward(grad_output.contiguous(), rois, out_h, out_w, spatial_scale, sample_num, grad_input)
        return grad_input, grad_rois, None, None, None


roi_align = RoIAlignFunction.apply


class RoIAlign(nn.Module):

    def __init__(self, out_size, spatial_scale, sample_num=0, use_torchvision=False):
        super(RoIAlign, self).__init__()
        self.out_size = _pair(out_size)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.use_torchvision = use_torchvision

    def forward(self, features, rois):
        if self.use_torchvision:
            from torchvision.ops import roi_align as tv_roi_align
            return tv_roi_align(features, rois, self.out_size, self.spatial_scale, self.sample_num)
        else:
            return roi_align(features, rois, self.out_size, self.spatial_scale, self.sample_num)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}, sample_num={}'.format(self.out_size, self.spatial_scale, self.sample_num)
        format_str += ', use_torchvision={})'.format(self.use_torchvision)
        return format_str


class RoIPoolFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError('"out_size" must be an integer or tuple of integers')
        ctx.save_for_backward(rois)
        num_channels = features.size(1)
        num_rois = rois.size(0)
        out_size = num_rois, num_channels, out_h, out_w
        output = features.new_zeros(out_size)
        argmax = features.new_zeros(out_size, dtype=torch.int)
        roi_pool_cuda.forward(features, rois, out_h, out_w, spatial_scale, output, argmax)
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = features.size()
        ctx.argmax = argmax
        return output

    @staticmethod
    def backward(ctx, grad_output):
        spatial_scale = ctx.spatial_scale
        feature_size = ctx.feature_size
        argmax = ctx.argmax
        rois = ctx.saved_tensors[0]
        assert feature_size is not None
        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.new_zeros(feature_size)
            roi_pool_cuda.backward(grad_output.contiguous(), rois, argmax, spatial_scale, grad_input)
        return grad_input, grad_rois, None, None


roi_pool = RoIPoolFunction.apply


class RoIPool(Module):

    def __init__(self, out_size, spatial_scale):
        super(RoIPool, self).__init__()
        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return roi_pool(features, rois, self.out_size, self.spatial_scale)


class SigmoidFocalLoss(nn.Module):

    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        loss = sigmoid_focal_loss(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'gamma=' + str(self.gamma)
        tmpstr += ', alpha=' + str(self.alpha)
        tmpstr += ')'
        return tmpstr


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AlexNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (AvgPool,
     lambda: ([], {'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BalancedL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BaseDetector,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BoundedIoULoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelShuffle,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContextBlock,
     lambda: ([], {'inplanes': 4, 'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvBlock,
     lambda: ([], {'input_size': 4, 'in_channels': 4, 'out_channels': 4, 'expansion': 4, 'stride': 1, 'kernel_size': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvWS2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DropBlock2D,
     lambda: ([], {'drop_prob': 4, 'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DropBlock3D,
     lambda: ([], {'drop_prob': 4, 'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (DropBlockScheduled,
     lambda: ([], {'dropblock': _mock_layer(), 'start_value': 4, 'stop_value': 4, 'nr_steps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GHMC,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GHMR,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Identity,
     lambda: ([], {'input_size': 4, 'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IoULoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (L2Norm,
     lambda: ([], {'n_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MaskedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (NonLocal2D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Scale,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SepBlock,
     lambda: ([], {'input_size': 4, 'in_channels': 4, 'out_channels': 4, 'expansion': 4, 'stride': 1, 'kernel_size': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 2, 2])], {}),
     True),
    (SepConv,
     lambda: ([], {'input_size': 4, 'in_channels': 4, 'out_channels': 4, 'expansion': 4, 'stride': 1, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SmoothL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_ggjy_HitDet_pytorch(_paritybench_base):
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

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

