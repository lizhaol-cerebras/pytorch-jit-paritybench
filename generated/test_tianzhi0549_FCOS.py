import sys
_module = sys.modules[__name__]
del sys
fcos_demo = _module
predictor = _module
webcam = _module
jupyter_notebook_config = _module
fcos = _module
fcos = _module
fcos_core = _module
config = _module
defaults = _module
paths_catalog = _module
data = _module
build = _module
collate_batch = _module
datasets = _module
coco = _module
concat_dataset = _module
evaluation = _module
coco_eval = _module
voc = _module
voc_eval = _module
list_dataset = _module
voc = _module
samplers = _module
distributed = _module
grouped_batch_sampler = _module
iteration_based_batch_sampler = _module
transforms = _module
transforms = _module
engine = _module
bbox_aug = _module
inference = _module
trainer = _module
layers = _module
_utils = _module
batch_norm = _module
dcn = _module
deform_conv_func = _module
deform_conv_module = _module
deform_pool_func = _module
deform_pool_module = _module
iou_loss = _module
misc = _module
nms = _module
roi_align = _module
roi_pool = _module
scale = _module
sigmoid_focal_loss = _module
smooth_l1_loss = _module
modeling = _module
backbone = _module
backbone = _module
fbnet = _module
fbnet_builder = _module
fbnet_modeldef = _module
fpn = _module
mobilenet = _module
resnet = _module
balanced_positive_negative_sampler = _module
box_coder = _module
detector = _module
detectors = _module
generalized_rcnn = _module
make_layers = _module
matcher = _module
poolers = _module
registry = _module
roi_heads = _module
box_head = _module
box_head = _module
inference = _module
loss = _module
roi_box_feature_extractors = _module
roi_box_predictors = _module
keypoint_head = _module
inference = _module
keypoint_head = _module
loss = _module
roi_keypoint_feature_extractors = _module
roi_keypoint_predictors = _module
mask_head = _module
inference = _module
loss = _module
mask_head = _module
roi_mask_feature_extractors = _module
roi_mask_predictors = _module
roi_heads = _module
rpn = _module
anchor_generator = _module
fcos = _module
inference = _module
loss = _module
inference = _module
loss = _module
retinanet = _module
inference = _module
loss = _module
retinanet = _module
rpn = _module
utils = _module
utils = _module
solver = _module
build = _module
lr_scheduler = _module
structures = _module
bounding_box = _module
boxlist_ops = _module
image_list = _module
keypoint = _module
segmentation_mask = _module
c2_model_loading = _module
checkpoint = _module
collect_env = _module
comm = _module
cv2_util = _module
env = _module
imports = _module
logger = _module
metric_logger = _module
miscellaneous = _module
model_serialization = _module
model_zoo = _module
timer = _module
export_model_to_onnx = _module
test_fcos_onnx_model = _module
setup = _module
checkpoint = _module
test_backbones = _module
test_box_coder = _module
test_configs = _module
test_data_samplers = _module
test_detectors = _module
test_fbnet = _module
test_feature_extractors = _module
test_metric_logger = _module
test_nms = _module
test_predictors = _module
test_rpn_heads = _module
test_segmentation_mask = _module
convert_cityscapes_to_coco = _module
instances2dict_with_polygons = _module
remove_solver_states = _module
test_net = _module
train_net = _module

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


import torch


from torchvision import transforms as T


import numpy as np


import matplotlib.pyplot as plt


import copy


import logging


import torch.utils.data


import torchvision


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


from collections import OrderedDict


import math


import torch.distributed as dist


from torch.utils.data.sampler import Sampler


import itertools


from torch.utils.data.sampler import BatchSampler


import random


from torchvision.transforms import functional as F


import torchvision.transforms as TT


import time


from torch import nn


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


import torch.nn as nn


from torch.nn.modules.utils import _ntuple


import torch.nn.functional as F


from torch.nn import BatchNorm2d


from collections import namedtuple


from torch.nn import functional as F


from torch.utils.collect_env import get_pretty_env_info


from collections import defaultdict


from collections import deque


from torchvision.ops.boxes import batched_nms


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.data.sampler import SequentialSampler


from torch.utils.data.sampler import RandomSampler


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


def _load_C_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir = os.path.dirname(this_dir)
    this_dir = os.path.join(this_dir, 'csrc')
    main_file = glob.glob(os.path.join(this_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(this_dir, 'cpu', '*.cpp'))
    source_cuda = glob.glob(os.path.join(this_dir, 'cuda', '*.cu'))
    source = main_file + source_cpu
    extra_cflags = []
    if torch.cuda.is_available() and CUDA_HOME is not None:
        source.extend(source_cuda)
        extra_cflags = ['-DWITH_CUDA']
    source = [os.path.join(this_dir, s) for s in source]
    extra_include_paths = [this_dir]
    return load_ext('torchvision', source, extra_cflags=extra_cflags, extra_include_paths=extra_include_paths)


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
            _C.deform_conv_forward(input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    @once_differentiable
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
                _C.deform_conv_backward_input(input, offset, grad_output, grad_input, grad_offset, weight, ctx.bufs_[0], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                _C.deform_conv_backward_parameters(input, offset, grad_output, grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1, cur_im2col_step)
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
        self.with_bias = bias
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
        if self.with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.with_bias:
            torch.nn.init.constant_(self.bias, 0.0)

    def forward(self, input, offset):
        y = deform_conv(input, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)
        if self.with_bias:
            assert len(y.size()) == 4
            y = y + self.bias.reshape(1, -1, 1, 1)
        return y

    def __repr__(self):
        return ''.join(['{}('.format(self.__class__.__name__), 'in_channels={}, '.format(self.in_channels), 'out_channels={}, '.format(self.out_channels), 'kernel_size={}, '.format(self.kernel_size), 'stride={}, '.format(self.stride), 'dilation={}, '.format(self.dilation), 'padding={}, '.format(self.padding), 'groups={}, '.format(self.groups), 'deformable_groups={}, '.format(self.deformable_groups), 'bias={})'.format(self.with_bias)])


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
        _C.modulated_deform_conv_forward(input, weight, bias, ctx._bufs[0], offset, mask, output, ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        _C.modulated_deform_conv_backward(input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
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

    def forward(self, input, offset, mask):
        return modulated_deform_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)

    def __repr__(self):
        return ''.join(['{}('.format(self.__class__.__name__), 'in_channels={}, '.format(self.in_channels), 'out_channels={}, '.format(self.out_channels), 'kernel_size={}, '.format(self.kernel_size), 'stride={}, '.format(self.stride), 'dilation={}, '.format(self.dilation), 'padding={}, '.format(self.padding), 'groups={}, '.format(self.groups), 'deformable_groups={}, '.format(self.deformable_groups), 'bias={})'.format(self.with_bias)])


class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConvPack, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, deformable_groups, bias)
        self.conv_offset_mask = nn.Conv2d(self.in_channels // self.groups, self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


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
        _C.deform_psroi_pooling_forward(data, rois, offset, output, output_count, ctx.no_trans, ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, rois, offset)
        ctx.output_count = output_count
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        data, rois, offset = ctx.saved_tensors
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_rois = None
        grad_offset = torch.zeros_like(offset)
        _C.deform_psroi_pooling_backward(grad_output, data, rois, offset, output_count, grad_input, grad_offset, ctx.no_trans, ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
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

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, deform_fc_channels=1024):
        super(DeformRoIPoolingPack, self).__init__(spatial_scale, out_size, out_channels, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.deform_fc_channels = deform_fc_channels
        if not no_trans:
            self.offset_fc = nn.Sequential(nn.Linear(self.out_size * self.out_size * self.out_channels, self.deform_fc_channels), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_channels, self.deform_fc_channels), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_channels, self.out_size * self.out_size * 2))
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

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, deform_fc_channels=1024):
        super(ModulatedDeformRoIPoolingPack, self).__init__(spatial_scale, out_size, out_channels, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.deform_fc_channels = deform_fc_channels
        if not no_trans:
            self.offset_fc = nn.Sequential(nn.Linear(self.out_size * self.out_size * self.out_channels, self.deform_fc_channels), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_channels, self.deform_fc_channels), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_channels, self.out_size * self.out_size * 2))
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()
            self.mask_fc = nn.Sequential(nn.Linear(self.out_size * self.out_size * self.out_channels, self.deform_fc_channels), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_channels, self.out_size * self.out_size * 1), nn.Sigmoid())
            self.mask_fc[2].weight.data.zero_()
            self.mask_fc[2].bias.data.zero_()

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


class IOULoss(nn.Module):

    def __init__(self, loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]
        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]
        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-07
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()


class _NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // d + 1) for i, p, di, k, d in zip(x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride)]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class ConvTranspose2d(torch.nn.ConvTranspose2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d, self).forward(x)
        output_shape = [((i - 1) * d - 2 * p + (di * (k - 1) + 1) + op) for i, p, di, k, d, op in zip(x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride, self.output_padding)]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class BatchNorm2d(torch.nn.BatchNorm2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(BatchNorm2d, self).forward(x)
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class DFConv2d(torch.nn.Module):
    """Deformable convolutional layer"""

    def __init__(self, in_channels, out_channels, with_modulated_dcn=True, kernel_size=3, stride=1, groups=1, padding=1, dilation=1, deformable_groups=1, bias=False):
        super(DFConv2d, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert len(kernel_size) == 2
            offset_base_channels = kernel_size[0] * kernel_size[1]
        else:
            offset_base_channels = kernel_size * kernel_size
        if with_modulated_dcn:
            offset_channels = offset_base_channels * 3
            conv_block = ModulatedDeformConv
        else:
            offset_channels = offset_base_channels * 2
            conv_block = DeformConv
        self.offset = Conv2d(in_channels, deformable_groups * offset_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=1, dilation=dilation)
        for l in [self.offset]:
            torch.nn.init.kaiming_uniform_(l.weight, a=1)
            torch.nn.init.constant_(l.bias, 0.0)
        self.conv = conv_block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.offset_base_channels = offset_base_channels

    def forward(self, x):
        assert x.numel() > 0, 'only non-empty tensors are supported'
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                offset = self.offset(x)
                x = self.conv(x, offset)
            else:
                offset_mask = self.offset(x)
                split_point = self.offset_base_channels * 2
                offset = offset_mask[:, :split_point, :, :]
                mask = offset_mask[:, split_point:, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            return x


class _ROIAlign(Function):

    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.roi_align_forward(input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_backward(grad_output, rois, spatial_scale, output_size[0], output_size[1], bs, ch, h, w, sampling_ratio)
        return grad_input, None, None, None, None


roi_align = _ROIAlign.apply


class ROIAlign(nn.Module):

    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', sampling_ratio=' + str(self.sampling_ratio)
        tmpstr += ')'
        return tmpstr


class _ROIPool(Function):

    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = input.size()
        output, argmax = _C.roi_pool_forward(input, roi, spatial_scale, output_size[0], output_size[1])
        ctx.save_for_backward(input, roi, argmax)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, argmax = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_pool_backward(grad_output, input, rois, argmax, spatial_scale, output_size[0], output_size[1], bs, ch, h, w)
        return grad_input, None, None, None


roi_pool = _ROIPool.apply


class ROIPool(nn.Module):

    def __init__(self, output_size, spatial_scale):
        super(ROIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ')'
        return tmpstr


class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    gamma = gamma[0]
    alpha = alpha[0]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)
    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)
    return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)


class _SigmoidFocalLoss(Function):

    @staticmethod
    def forward(ctx, logits, targets, gamma, alpha):
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha
        losses = _C.sigmoid_focalloss_forward(logits, targets, num_classes, gamma, alpha)
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_logits = _C.sigmoid_focalloss_backward(logits, targets, d_loss, num_classes, gamma, alpha)
        return d_logits, None, None, None, None


sigmoid_focal_loss_cuda = _SigmoidFocalLoss.apply


class SigmoidFocalLoss(nn.Module):

    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        device = logits.device
        if logits.is_cuda:
            loss_func = sigmoid_focal_loss_cuda
        else:
            loss_func = sigmoid_focal_loss_cpu
        loss = loss_func(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'gamma=' + str(self.gamma)
        tmpstr += ', alpha=' + str(self.alpha)
        tmpstr += ')'
        return tmpstr


def _get_trunk_cfg(arch_def):
    """ Get all stages except the last one """
    num_stages = mbuilder.get_num_stages(arch_def)
    trunk_stages = arch_def.get('backbone', range(num_stages - 1))
    ret = mbuilder.get_blocks(arch_def, stage_indices=trunk_stages)
    return ret


class FBNetTrunk(nn.Module):

    def __init__(self, builder, arch_def, dim_in):
        super(FBNetTrunk, self).__init__()
        self.first = builder.add_first(arch_def['first'], dim_in=dim_in)
        trunk_cfg = _get_trunk_cfg(arch_def)
        self.stages = builder.add_blocks(trunk_cfg['stages'])

    def forward(self, x):
        y = self.first(x)
        y = self.stages(y)
        ret = [y]
        return ret


logger = logging.getLogger(__name__)


def _get_rpn_stage(arch_def, num_blocks):
    rpn_stage = arch_def.get('rpn')
    ret = mbuilder.get_blocks(arch_def, stage_indices=rpn_stage)
    if num_blocks > 0:
        logger.warn('Use last {} blocks in {} as rpn'.format(num_blocks, ret))
        block_count = len(ret['stages'])
        assert num_blocks <= block_count, 'use block {}, block count {}'.format(num_blocks, block_count)
        blocks = range(block_count - num_blocks, block_count)
        ret = mbuilder.get_blocks(ret, block_indices=blocks)
    return ret['stages']


class FBNetRPNHead(nn.Module):

    def __init__(self, cfg, in_channels, builder, arch_def):
        super(FBNetRPNHead, self).__init__()
        assert in_channels == builder.last_depth
        rpn_bn_type = cfg.MODEL.FBNET.RPN_BN_TYPE
        if len(rpn_bn_type) > 0:
            builder.bn_type = rpn_bn_type
        use_blocks = cfg.MODEL.FBNET.RPN_HEAD_BLOCKS
        stages = _get_rpn_stage(arch_def, use_blocks)
        self.head = builder.add_blocks(stages)
        self.out_channels = builder.last_depth

    def forward(self, x):
        x = [self.head(y) for y in x]
        return x


ARCH_CFG_NAME_MAPPING = {'bbox': 'ROI_BOX_HEAD', 'kpts': 'ROI_KEYPOINT_HEAD', 'mask': 'ROI_MASK_HEAD'}


def _get_head_stage(arch, head_name, blocks):
    if head_name not in arch:
        head_name = 'head'
    head_stage = arch.get(head_name)
    ret = mbuilder.get_blocks(arch, stage_indices=head_stage, block_indices=blocks)
    return ret['stages']


class FBNetROIHead(nn.Module):

    def __init__(self, cfg, in_channels, builder, arch_def, head_name, use_blocks, stride_init, last_layer_scale):
        super(FBNetROIHead, self).__init__()
        assert in_channels == builder.last_depth
        assert isinstance(use_blocks, list)
        head_cfg_name = ARCH_CFG_NAME_MAPPING[head_name]
        self.pooler = poolers.make_pooler(cfg, head_cfg_name)
        stage = _get_head_stage(arch_def, head_name, use_blocks)
        assert stride_init in [0, 1, 2]
        if stride_init != 0:
            stage[0]['block'][3] = stride_init
        blocks = builder.add_blocks(stage)
        last_info = copy.deepcopy(arch_def['last'])
        last_info[1] = last_layer_scale
        last = builder.add_last(last_info)
        self.head = nn.Sequential(OrderedDict([('blocks', blocks), ('last', last)]))
        self.out_channels = builder.last_depth

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


class ConvBNRelu(nn.Sequential):

    def __init__(self, input_depth, output_depth, kernel, stride, pad, no_bias, use_relu, bn_type, group=1, *args, **kwargs):
        super(ConvBNRelu, self).__init__()
        assert use_relu in ['relu', None]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == 'gn'
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ['bn', 'af', 'gn', None]
        assert stride in [1, 2, 4]
        op = Conv2d(input_depth, output_depth, *args, kernel_size=kernel, stride=stride, padding=pad, bias=not no_bias, groups=group, **kwargs)
        nn.init.kaiming_normal_(op.weight, mode='fan_out', nonlinearity='relu')
        if op.bias is not None:
            nn.init.constant_(op.bias, 0.0)
        self.add_module('conv', op)
        if bn_type == 'bn':
            bn_op = BatchNorm2d(output_depth)
        elif bn_type == 'gn':
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth)
        elif bn_type == 'af':
            bn_op = FrozenBatchNorm2d(output_depth)
        if bn_type is not None:
            self.add_module('bn', bn_op)
        if use_relu == 'relu':
            self.add_module('relu', nn.ReLU(inplace=True))


class Identity(nn.Module):

    def __init__(self, C_in, C_out, stride):
        super(Identity, self).__init__()
        self.conv = ConvBNRelu(C_in, C_out, kernel=1, stride=stride, pad=0, no_bias=1, use_relu='relu', bn_type='bn') if C_in != C_out or stride != 1 else None

    def forward(self, x):
        if self.conv:
            out = self.conv(x)
        else:
            out = x
        return out


class CascadeConv3x3(nn.Sequential):

    def __init__(self, C_in, C_out, stride):
        assert stride in [1, 2]
        ops = [Conv2d(C_in, C_in, 3, stride, 1, bias=False), BatchNorm2d(C_in), nn.ReLU(inplace=True), Conv2d(C_in, C_out, 3, 1, 1, bias=False), BatchNorm2d(C_out)]
        super(CascadeConv3x3, self).__init__(*ops)
        self.res_connect = stride == 1 and C_in == C_out

    def forward(self, x):
        y = super(CascadeConv3x3, self).forward(x)
        if self.res_connect:
            y += x
        return y


class Shift(nn.Module):

    def __init__(self, C, kernel_size, stride, padding):
        super(Shift, self).__init__()
        self.C = C
        kernel = torch.zeros((C, 1, kernel_size, kernel_size), dtype=torch.float32)
        ch_idx = 0
        assert stride in [1, 2]
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = 1
        hks = kernel_size // 2
        ksq = kernel_size ** 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == hks and j == hks:
                    num_ch = C // ksq + C % ksq
                else:
                    num_ch = C // ksq
                kernel[ch_idx:ch_idx + num_ch, 0, i, j] = 1
                ch_idx += num_ch
        self.register_parameter('bias', None)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        if x.numel() > 0:
            return nn.functional.conv2d(x, self.kernel, self.bias, (self.stride, self.stride), (self.padding, self.padding), self.dilation, self.C)
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // d + 1) for i, p, di, k, d in zip(x.shape[-2:], (self.padding, self.dilation), (self.dilation, self.dilation), (self.kernel_size, self.kernel_size), (self.stride, self.stride))]
        output_shape = [x.shape[0], self.C] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


def _py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def _get_divisible_by(num, divisible_by, min_val):
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret


class ShiftBlock5x5(nn.Sequential):

    def __init__(self, C_in, C_out, expansion, stride):
        assert stride in [1, 2]
        self.res_connect = stride == 1 and C_in == C_out
        C_mid = _get_divisible_by(C_in * expansion, 8, 8)
        ops = [Conv2d(C_in, C_mid, 1, 1, 0, bias=False), BatchNorm2d(C_mid), nn.ReLU(inplace=True), Shift(C_mid, 5, stride, 2), Conv2d(C_mid, C_out, 1, 1, 0, bias=False), BatchNorm2d(C_out)]
        super(ShiftBlock5x5, self).__init__(*ops)

    def forward(self, x):
        y = super(ShiftBlock5x5, self).forward(x)
        if self.res_connect:
            y += x
        return y


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, 'Incompatible group size {} for input channel {}'.format(g, C)
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class SEModule(nn.Module):
    reduction = 4

    def __init__(self, C):
        super(SEModule, self).__init__()
        mid = max(C // self.reduction, 8)
        conv1 = Conv2d(C, mid, 1, 1, 0)
        conv2 = Conv2d(mid, C, 1, 1, 0)
        self.op = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv1, nn.ReLU(inplace=True), conv2, nn.Sigmoid())

    def forward(self, x):
        return x * self.op(x)


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if size is not None and scale_factor is not None:
            raise ValueError('only one of size or scale_factor should be defined')
        if scale_factor is not None and isinstance(scale_factor, tuple) and len(scale_factor) != dim:
            raise ValueError('scale_factor shape must match input shape. Input is {}D, scale_factor size is {}'.format(dim, len(scale_factor)))

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        return [int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)]
    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)


class Upsample(nn.Module):

    def __init__(self, scale_factor, mode, align_corners=None):
        super(Upsample, self).__init__()
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=self.align_corners)


def _get_upsample_op(stride):
    assert stride in [1, 2, 4] or stride in [-1, -2, -4] or isinstance(stride, tuple) and all(x in [-1, -2, -4] for x in stride)
    scales = stride
    ret = None
    if isinstance(stride, tuple) or stride < 0:
        scales = [(-x) for x in stride] if isinstance(stride, tuple) else -stride
        stride = 1
        ret = Upsample(scale_factor=scales, mode='nearest', align_corners=None)
    return ret, stride


class IRFBlock(nn.Module):

    def __init__(self, input_depth, output_depth, expansion, stride, bn_type='bn', kernel=3, width_divisor=1, shuffle_type=None, pw_group=1, se=False, cdw=False, dw_skip_bn=False, dw_skip_relu=False):
        super(IRFBlock, self).__init__()
        assert kernel in [1, 3, 5, 7], kernel
        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth
        mid_depth = int(input_depth * expansion)
        mid_depth = _get_divisible_by(mid_depth, width_divisor, width_divisor)
        self.pw = ConvBNRelu(input_depth, mid_depth, kernel=1, stride=1, pad=0, no_bias=1, use_relu='relu', bn_type=bn_type, group=pw_group)
        self.upscale, stride = _get_upsample_op(stride)
        if kernel == 1:
            self.dw = nn.Sequential()
        elif cdw:
            dw1 = ConvBNRelu(mid_depth, mid_depth, kernel=kernel, stride=stride, pad=kernel // 2, group=mid_depth, no_bias=1, use_relu='relu', bn_type=bn_type)
            dw2 = ConvBNRelu(mid_depth, mid_depth, kernel=kernel, stride=1, pad=kernel // 2, group=mid_depth, no_bias=1, use_relu='relu' if not dw_skip_relu else None, bn_type=bn_type if not dw_skip_bn else None)
            self.dw = nn.Sequential(OrderedDict([('dw1', dw1), ('dw2', dw2)]))
        else:
            self.dw = ConvBNRelu(mid_depth, mid_depth, kernel=kernel, stride=stride, pad=kernel // 2, group=mid_depth, no_bias=1, use_relu='relu' if not dw_skip_relu else None, bn_type=bn_type if not dw_skip_bn else None)
        self.pwl = ConvBNRelu(mid_depth, output_depth, kernel=1, stride=1, pad=0, no_bias=1, use_relu=None, bn_type=bn_type, group=pw_group)
        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)
        self.se4 = SEModule(output_depth) if se else nn.Sequential()
        self.output_depth = output_depth

    def forward(self, x):
        y = self.pw(x)
        if self.shuffle_type == 'mid':
            y = self.shuffle(y)
        if self.upscale is not None:
            y = self.upscale(y)
        y = self.dw(y)
        y = self.pwl(y)
        if self.use_res_connect:
            y += x
        y = self.se4(y)
        return y


class LastLevelMaxPool(nn.Module):

    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(self, in_channels_list, out_channels, conv_block, top_blocks=None):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = 'fpn_inner{}'.format(idx)
            layer_block = 'fpn_layer{}'.format(idx)
            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]):
            if not inner_block:
                continue
            inner_lateral = getattr(self, inner_block)(feature)
            inner_top_down = F.interpolate(last_inner, size=(int(inner_lateral.shape[-2]), int(inner_lateral.shape[-1])), mode='nearest')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))
        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)
        return tuple(results)


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_bn(inp, oup, stride):
    return nn.Sequential(Conv2d(inp, oup, 3, stride, 1, bias=False), BatchNorm2d(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):
    """
    Should freeze bn
    """

    def __init__(self, cfg, n_class=1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.return_features_indices = [3, 6, 13, 17]
        self.return_features_num_channels = []
        self.features = nn.ModuleList([conv_bn(3, input_channel, 2)])
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
                if len(self.features) - 1 in self.return_features_indices:
                    self.return_features_num_channels.append(output_channel)
        self._initialize_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False

    def forward(self, x):
        res = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.return_features_indices:
                res.append(x)
        return res

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    """
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    """

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        if module is not None:
            _register_generic(self, module_name, module)
            return

        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn
        return register_fn


StageSpec = namedtuple('StageSpec', ['index', 'block_count', 'return_features'])


ResNet101FPNStagesTo5 = tuple(StageSpec(index=i, block_count=c, return_features=r) for i, c, r in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True)))


ResNet101StagesTo4 = tuple(StageSpec(index=i, block_count=c, return_features=r) for i, c, r in ((1, 3, False), (2, 4, False), (3, 23, True)))


ResNet101StagesTo5 = tuple(StageSpec(index=i, block_count=c, return_features=r) for i, c, r in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True)))


ResNet152FPNStagesTo5 = tuple(StageSpec(index=i, block_count=c, return_features=r) for i, c, r in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True)))


ResNet50FPNStagesTo5 = tuple(StageSpec(index=i, block_count=c, return_features=r) for i, c, r in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True)))


ResNet50StagesTo4 = tuple(StageSpec(index=i, block_count=c, return_features=r) for i, c, r in ((1, 3, False), (2, 4, False), (3, 6, True)))


ResNet50StagesTo5 = tuple(StageSpec(index=i, block_count=c, return_features=r) for i, c, r in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True)))


_STAGE_SPECS = Registry({'R-50-C4': ResNet50StagesTo4, 'R-50-C5': ResNet50StagesTo5, 'R-101-C4': ResNet101StagesTo4, 'R-101-C5': ResNet101StagesTo5, 'R-50-FPN': ResNet50FPNStagesTo5, 'R-50-FPN-RETINANET': ResNet50FPNStagesTo5, 'R-101-FPN': ResNet101FPNStagesTo5, 'R-101-FPN-RETINANET': ResNet101FPNStagesTo5, 'R-152-FPN': ResNet152FPNStagesTo5})


class BaseStem(nn.Module):

    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()
        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        self.conv1 = Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_func(out_channels)
        for l in [self.conv1]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class StemWithFixedBatchNorm(BaseStem):

    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(cfg, norm_func=FrozenBatchNorm2d)


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, 'GroupNorm: can only specify G or C/G.'
    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, 'dim: {}, dim_per_gp: {}'.format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, 'dim: {}, num_groups: {}'.format(dim, num_groups)
        group_gn = num_groups
    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    eps = cfg.MODEL.GROUP_NORM.EPSILON
    return torch.nn.GroupNorm(get_group_gn(out_channels, dim_per_gp, num_groups), out_channels, eps, affine)


class StemWithGN(BaseStem):

    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=group_norm)


_STEM_MODULES = Registry({'StemWithFixedBatchNorm': StemWithFixedBatchNorm, 'StemWithGN': StemWithGN})


class Bottleneck(nn.Module):

    def __init__(self, in_channels, bottleneck_channels, out_channels, num_groups, stride_in_1x1, stride, dilation, norm_func, dcn_config):
        super(Bottleneck, self).__init__()
        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size=1, stride=down_stride, bias=False), norm_func(out_channels))
            for modules in [self.downsample]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)
        if dilation > 1:
            stride = 1
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride_1x1, bias=False)
        self.bn1 = norm_func(bottleneck_channels)
        with_dcn = dcn_config.get('stage_with_dcn', False)
        if with_dcn:
            deformable_groups = dcn_config.get('deformable_groups', 1)
            with_modulated_dcn = dcn_config.get('with_modulated_dcn', False)
            self.conv2 = DFConv2d(bottleneck_channels, bottleneck_channels, with_modulated_dcn=with_modulated_dcn, kernel_size=3, stride=stride_3x3, groups=num_groups, dilation=dilation, deformable_groups=deformable_groups, bias=False)
        else:
            self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride_3x3, padding=dilation, bias=False, groups=num_groups, dilation=dilation)
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)
        self.bn2 = norm_func(bottleneck_channels)
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm_func(out_channels)
        for l in [self.conv1, self.conv3]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)
        out0 = self.conv3(out)
        out = self.bn3(out0)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu_(out)
        return out


class BottleneckWithFixedBatchNorm(Bottleneck):

    def __init__(self, in_channels, bottleneck_channels, out_channels, num_groups=1, stride_in_1x1=True, stride=1, dilation=1, dcn_config=None):
        super(BottleneckWithFixedBatchNorm, self).__init__(in_channels=in_channels, bottleneck_channels=bottleneck_channels, out_channels=out_channels, num_groups=num_groups, stride_in_1x1=stride_in_1x1, stride=stride, dilation=dilation, norm_func=FrozenBatchNorm2d, dcn_config=dcn_config)


class BottleneckWithGN(Bottleneck):

    def __init__(self, in_channels, bottleneck_channels, out_channels, num_groups=1, stride_in_1x1=True, stride=1, dilation=1, dcn_config=None):
        super(BottleneckWithGN, self).__init__(in_channels=in_channels, bottleneck_channels=bottleneck_channels, out_channels=out_channels, num_groups=num_groups, stride_in_1x1=stride_in_1x1, stride=stride, dilation=dilation, norm_func=group_norm, dcn_config=dcn_config)


_TRANSFORMATION_MODULES = Registry({'BottleneckWithFixedBatchNorm': BottleneckWithFixedBatchNorm, 'BottleneckWithGN': BottleneckWithGN})


def _make_stage(transformation_module, in_channels, bottleneck_channels, out_channels, block_count, num_groups, stride_in_1x1, first_stride, dilation=1, dcn_config=None):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(transformation_module(in_channels, bottleneck_channels, out_channels, num_groups, stride_in_1x1, stride, dilation=dilation, dcn_config=dcn_config))
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


class ResNet(nn.Module):

    def __init__(self, cfg):
        super(ResNet, self).__init__()
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]
        self.stem = stem_module(cfg)
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = 'layer' + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
            module = _make_stage(transformation_module, in_channels, bottleneck_channels, out_channels, stage_spec.block_count, num_groups, cfg.MODEL.RESNETS.STRIDE_IN_1X1, first_stride=int(stage_spec.index > 1) + 1, dcn_config={'stage_with_dcn': stage_with_dcn, 'with_modulated_dcn': cfg.MODEL.RESNETS.WITH_MODULATED_DCN, 'deformable_groups': cfg.MODEL.RESNETS.DEFORMABLE_GROUPS})
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem
            else:
                m = getattr(self, 'layer' + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


class ResNetHead(nn.Module):

    def __init__(self, block_module, stages, num_groups=1, width_per_group=64, stride_in_1x1=True, stride_init=None, res2_out_channels=256, dilation=1, dcn_config=None):
        super(ResNetHead, self).__init__()
        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
        block_module = _TRANSFORMATION_MODULES[block_module]
        self.stages = []
        stride = stride_init
        for stage in stages:
            name = 'layer' + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(block_module, in_channels, bottleneck_channels, out_channels, stage.block_count, num_groups, stride_in_1x1, first_stride=stride, dilation=dilation, dcn_config=dcn_config)
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, 'cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry'.format(cfg.MODEL.BACKBONE.CONV_BODY)
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                mask_features = x
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)
        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            if self.training and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                keypoint_features = x
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)
        return x, detections, losses


def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR]
    return func(cfg, in_channels)


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        return pos_idx, neg_idx


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000.0 / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        TO_REMOVE = 1
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights
        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights
        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        boxes = boxes
        TO_REMOVE = 1
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_boxes = torch.zeros_like(rel_codes)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1
        return pred_boxes


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError('No ground-truth boxes available for one of the images during training')
            else:
                raise ValueError('No proposal boxes available for one of the images during training')
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS
        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)
        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.nonzero(match_quality_matrix == highest_quality_foreach_gt[:, None])
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError('boxlists should have same image size, got {}, {}'.format(boxlist1, boxlist2))
    N = len(boxlist1)
    M = len(boxlist2)
    area1 = boxlist1.area()
    area2 = boxlist2.area()
    box1, box2 = boxlist1.bbox, boxlist2.bbox
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def smooth_l1_loss(input, target, beta=1.0 / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, cls_agnostic_bbox_reg=False):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields('labels')
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = matched_targets.get_field('labels')
            labels_per_image = labels_per_image
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1
            regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, proposals_per_image.bbox)
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
        return labels, regression_targets

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        labels, regression_targets = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        proposals = list(proposals)
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(labels, regression_targets, proposals):
            proposals_per_image.add_field('labels', labels_per_image)
            proposals_per_image.add_field('regression_targets', regression_targets_per_image)
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image
        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device
        if not hasattr(self, '_proposals'):
            raise RuntimeError('subsample needs to be called before')
        proposals = self._proposals
        labels = cat([proposal.get_field('labels') for proposal in proposals], dim=0)
        regression_targets = cat([proposal.get_field('regression_targets') for proposal in proposals], dim=0)
        classification_loss = F.cross_entropy(class_logits, labels)
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)
        box_loss = smooth_l1_loss(box_regression[sampled_pos_inds_subset[:, None], map_inds], regression_targets[sampled_pos_inds_subset], size_average=False, beta=1)
        box_loss = box_loss / labels.numel()
        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    fg_bg_sampler = BalancedPositiveNegativeSampler(cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION)
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    loss_evaluator = FastRCNNLossComputation(matcher, fg_bg_sampler, box_coder, cls_agnostic_bbox_reg)
    return loss_evaluator


FLIP_LEFT_RIGHT = 0


FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode='xyxy'):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device('cpu')
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError('bbox should have 2 dimensions, got {}'.format(bbox.ndimension()))
        if bbox.size(-1) != 4:
            raise ValueError('last dimension of bbox should have a size of 4, got {}'.format(bbox.size(-1)))
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        self.bbox = bbox
        self.size = image_size
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == 'xyxy':
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat((xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == 'xyxy':
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == 'xywh':
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmin + (w - TO_REMOVE).clamp(min=0), ymin + (h - TO_REMOVE).clamp(min=0)
        else:
            raise RuntimeError('Should not be here')

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox
        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat((scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1)
        bbox = BoxList(scaled_box, size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError('Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented')
        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin
        transposed_boxes = torch.cat((transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1)
        bbox = BoxList(transposed_boxes, self.size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)
        cropped_box = torch.cat((cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1)
        bbox = BoxList(cropped_box, (w, h), mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def to(self, device):
        bbox = BoxList(self.bbox, self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, 'to'):
                v = v
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == 'xyxy':
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == 'xywh':
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError('Should not be here')
        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_boxes={}, '.format(len(self))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'mode={})'.format(self.mode)
        return s


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field='scores'):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert('xyxy')
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[:max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)
    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)
    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)
    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)
    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)
    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)
    return cat_boxes


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(self, score_thresh=0.05, nms=0.5, detections_per_img=100, box_coder=None, cls_agnostic_bbox_reg=False, bbox_aug_enabled=False):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(box_regression.view(sum(boxes_per_image), -1), concat_boxes)
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])
        num_classes = class_prob.shape[1]
        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        results = []
        for prob, boxes_per_img, image_shape in zip(class_prob, proposals, image_shapes):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            if not self.bbox_aug_enabled:
                boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode='xyxy')
        boxlist.add_field('scores', scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field('scores').reshape(-1, num_classes)
        device = scores.device
        result = []
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4:(j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode='xyxy')
            boxlist_for_class.add_field('scores', scores_j)
            boxlist_for_class = boxlist_nms(boxlist_for_class, self.nms)
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field('labels', torch.full((num_labels,), j, dtype=torch.int64, device=device))
            result.append(boxlist_for_class)
        result = cat_boxlist(result)
        number_of_detections = len(result)
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field('scores')
            image_thresh, _ = torch.kthvalue(cls_scores.cpu(), number_of_detections - self.detections_per_img + 1)
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED
    postprocessor = PostProcessor(score_thresh, nms_thresh, detections_per_img, box_coder, cls_agnostic_bbox_reg, bbox_aug_enabled)
    return postprocessor


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)
        x = self.feature_extractor(features, proposals)
        class_logits, box_regression = self.predictor(x)
        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}
        loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression])
        return x, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)


def make_roi_keypoint_feature_extractor(cfg, in_channels):
    func = registry.ROI_KEYPOINT_FEATURE_EXTRACTORS[cfg.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR]
    return func(cfg, in_channels)


def _within_box(points, boxes):
    """Validate which keypoints are contained inside a given box.
    points: NxKx2
    boxes: Nx4
    output: NxK
    """
    x_within = (points[..., 0] >= boxes[:, 0, None]) & (points[..., 0] <= boxes[:, 2, None])
    y_within = (points[..., 1] >= boxes[:, 1, None]) & (points[..., 1] <= boxes[:, 3, None])
    return x_within & y_within


def keypoints_to_heat_map(keypoints, rois, heatmap_size):
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])
    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]
    x = keypoints[..., 0]
    y = keypoints[..., 1]
    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]
    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()
    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1
    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()
    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid
    return heatmaps, valid


def project_keypoints_to_heatmap(keypoints, proposals, discretization_size):
    proposals = proposals.convert('xyxy')
    return keypoints_to_heat_map(keypoints.keypoints, proposals.bbox, discretization_size)


class KeypointRCNNLossComputation(object):

    def __init__(self, proposal_matcher, fg_bg_sampler, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.discretization_size = discretization_size

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(['labels', 'keypoints'])
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        keypoints = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = matched_targets.get_field('labels')
            labels_per_image = labels_per_image
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0
            keypoints_per_image = matched_targets.get_field('keypoints')
            within_box = _within_box(keypoints_per_image.keypoints, matched_targets.bbox)
            vis_kp = keypoints_per_image.keypoints[..., 2] > 0
            is_visible = (within_box & vis_kp).sum(1) > 0
            labels_per_image[~is_visible] = -1
            labels.append(labels_per_image)
            keypoints.append(keypoints_per_image)
        return labels, keypoints

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        labels, keypoints = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        proposals = list(proposals)
        for labels_per_image, keypoints_per_image, proposals_per_image in zip(labels, keypoints, proposals):
            proposals_per_image.add_field('labels', labels_per_image)
            proposals_per_image.add_field('keypoints', keypoints_per_image)
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image
        self._proposals = proposals
        return proposals

    def __call__(self, proposals, keypoint_logits):
        heatmaps = []
        valid = []
        for proposals_per_image in proposals:
            kp = proposals_per_image.get_field('keypoints')
            heatmaps_per_image, valid_per_image = project_keypoints_to_heatmap(kp, proposals_per_image, self.discretization_size)
            heatmaps.append(heatmaps_per_image.view(-1))
            valid.append(valid_per_image.view(-1))
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0)
        valid = torch.nonzero(valid).squeeze(1)
        if keypoint_targets.numel() == 0 or len(valid) == 0:
            return keypoint_logits.sum() * 0
        N, K, H, W = keypoint_logits.shape
        keypoint_logits = keypoint_logits.view(N * K, H * W)
        keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
        return keypoint_loss


def make_roi_keypoint_loss_evaluator(cfg):
    matcher = Matcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)
    fg_bg_sampler = BalancedPositiveNegativeSampler(cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION)
    resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION
    loss_evaluator = KeypointRCNNLossComputation(matcher, fg_bg_sampler, resolution)
    return loss_evaluator


class Keypoints(object):

    def __init__(self, keypoints, size, mode=None):
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device('cpu')
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        num_keypoints = keypoints.shape[0]
        if num_keypoints:
            keypoints = keypoints.view(num_keypoints, -1, 3)
        self.keypoints = keypoints
        self.size = size
        self.mode = mode
        self.extra_fields = {}

    def crop(self, box):
        raise NotImplementedError()

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data = self.keypoints.clone()
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
        keypoints = type(self)(resized_data, size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError('Only FLIP_LEFT_RIGHT implemented')
        flip_inds = type(self).FLIP_INDS
        flipped_data = self.keypoints[:, flip_inds]
        width = self.size[0]
        TO_REMOVE = 1
        flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE
        inds = flipped_data[..., 2] == 0
        flipped_data[inds] = 0
        keypoints = type(self)(flipped_data, self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def to(self, *args, **kwargs):
        keypoints = type(self)(self.keypoints, self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, 'to'):
                v = v
            keypoints.add_field(k, v)
        return keypoints

    def __getitem__(self, item):
        keypoints = type(self)(self.keypoints[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v[item])
        return keypoints

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.keypoints))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s


class PersonKeypoints(Keypoints):
    NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    FLIP_MAP = {'left_eye': 'right_eye', 'left_ear': 'right_ear', 'left_shoulder': 'right_shoulder', 'left_elbow': 'right_elbow', 'left_wrist': 'right_wrist', 'left_hip': 'right_hip', 'left_knee': 'right_knee', 'left_ankle': 'right_ankle'}


class KeypointPostProcessor(nn.Module):

    def __init__(self, keypointer=None):
        super(KeypointPostProcessor, self).__init__()
        self.keypointer = keypointer

    def forward(self, x, boxes):
        mask_prob = x
        scores = None
        if self.keypointer:
            mask_prob, scores = self.keypointer(x, boxes)
        assert len(boxes) == 1, 'Only non-batched inference supported for now'
        boxes_per_image = [box.bbox.size(0) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)
        scores = scores.split(boxes_per_image, dim=0)
        results = []
        for prob, box, score in zip(mask_prob, boxes, scores):
            bbox = BoxList(box.bbox, box.size, mode='xyxy')
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            prob = PersonKeypoints(prob, box.size)
            prob.add_field('logits', score)
            bbox.add_field('keypoints', prob)
            results.append(bbox)
        return results


def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = np.maximum(widths, 1)
    heights = np.maximum(heights, 1)
    widths_ceil = np.ceil(widths)
    heights_ceil = np.ceil(heights)
    maps = np.transpose(maps, [0, 2, 3, 1])
    min_size = 0
    num_keypoints = maps.shape[3]
    xy_preds = np.zeros((len(rois), 3, num_keypoints), dtype=np.float32)
    end_scores = np.zeros((len(rois), num_keypoints), dtype=np.float32)
    for i in range(len(rois)):
        if min_size > 0:
            roi_map_width = int(np.maximum(widths_ceil[i], min_size))
            roi_map_height = int(np.maximum(heights_ceil[i], min_size))
        else:
            roi_map_width = widths_ceil[i]
            roi_map_height = heights_ceil[i]
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = cv2.resize(maps[i], (roi_map_width, roi_map_height), interpolation=cv2.INTER_CUBIC)
        roi_map = np.transpose(roi_map, [2, 0, 1])
        w = roi_map.shape[2]
        pos = roi_map.reshape(num_keypoints, -1).argmax(axis=1)
        x_int = pos % w
        y_int = (pos - x_int) // w
        x = (x_int + 0.5) * width_correction
        y = (y_int + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
        end_scores[i, :] = roi_map[np.arange(num_keypoints), y_int, x_int]
    return np.transpose(xy_preds, [0, 2, 1]), end_scores


class Keypointer(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, padding=0):
        self.padding = padding

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]
        assert len(boxes) == 1
        result, scores = heatmaps_to_keypoints(masks.detach().cpu().numpy(), boxes[0].bbox.cpu().numpy())
        return torch.from_numpy(result), torch.as_tensor(scores, device=masks.device)


def make_roi_keypoint_post_processor(cfg):
    keypointer = Keypointer()
    keypoint_post_processor = KeypointPostProcessor(keypointer)
    return keypoint_post_processor


def make_roi_keypoint_predictor(cfg, in_channels):
    func = registry.ROI_KEYPOINT_PREDICTOR[cfg.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR]
    return func(cfg, in_channels)


class ROIKeypointHead(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(ROIKeypointHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_keypoint_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_keypoint_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_keypoint_post_processor(cfg)
        self.loss_evaluator = make_roi_keypoint_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)
        x = self.feature_extractor(features, proposals)
        kp_logits = self.predictor(x)
        if not self.training:
            result = self.post_processor(kp_logits, proposals)
            return x, result, {}
        loss_kp = self.loss_evaluator(proposals, kp_logits)
        return x, proposals, dict(loss_kp=loss_kp)


def build_roi_keypoint_head(cfg, in_channels):
    return ROIKeypointHead(cfg, in_channels)


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field('labels')
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field('labels')
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


def make_roi_mask_feature_extractor(cfg, in_channels):
    func = registry.ROI_MASK_FEATURE_EXTRACTORS[cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR]
    return func(cfg, in_channels)


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert('xyxy')
    assert segmentation_masks.size == proposals.size, '{}, {}'.format(segmentation_masks, proposals)
    proposals = proposals.bbox
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.get_mask_tensor()
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0)


class MaskRCNNLossComputation(object):

    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(['labels', 'masks'])
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = matched_targets.get_field('labels')
            labels_per_image = labels_per_image
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)
            segmentation_masks = matched_targets.get_field('masks')
            segmentation_masks = segmentation_masks[positive_inds]
            positive_proposals = proposals_per_image[positive_inds]
            masks_per_image = project_masks_on_boxes(segmentation_masks, positive_proposals, self.discretization_size)
            labels.append(labels_per_image)
            masks.append(masks_per_image)
        return labels, masks

    def __call__(self, proposals, mask_logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, mask_targets = self.prepare_targets(proposals, targets)
        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)
        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0
        mask_loss = F.binary_cross_entropy_with_logits(mask_logits[positive_inds, labels_pos], mask_targets)
        return mask_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)
    loss_evaluator = MaskRCNNLossComputation(matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION)
    return loss_evaluator


class MaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, masker=None):
        super(MaskPostProcessor, self).__init__()
        self.masker = masker

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        mask_prob = x.sigmoid()
        num_masks = x.shape[0]
        labels = [bbox.get_field('labels') for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, None]
        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)
        if self.masker:
            mask_prob = self.masker(mask_prob, boxes)
        results = []
        for prob, box in zip(mask_prob, boxes):
            bbox = BoxList(box.bbox, box.size, mode='xyxy')
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field('mask', prob)
            results.append(bbox)
        return results


def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5
    w_half *= scale
    h_half *= scale
    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)
    mask = mask.expand((1, 1, -1, -1))
    mask = mask
    mask = interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]
    if thresh >= 0:
        mask = mask > thresh
    else:
        mask = mask * 255
    im_mask = torch.zeros((im_h, im_w), dtype=torch.uint8)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)
    im_mask[y_0:y_1, x_0:x_1] = mask[y_0 - box[1]:y_1 - box[1], x_0 - box[0]:x_1 - box[0]]
    return im_mask


class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert('xyxy')
        im_w, im_h = boxes.size
        res = [paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding) for mask, box in zip(masks, boxes.bbox)]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, None]
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]
        assert len(boxes) == len(masks), 'Masks and boxes should have the same length.'
        results = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), 'Number of objects should be the same.'
            result = self.forward_single_image(mask, box)
            results.append(result)
        return results


def make_roi_mask_post_processor(cfg):
    if cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS:
        mask_threshold = cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
        masker = Masker(threshold=mask_threshold, padding=1)
    else:
        masker = None
    mask_post_processor = MaskPostProcessor(masker)
    return mask_post_processor


def make_roi_mask_predictor(cfg, in_channels):
    func = registry.ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg, in_channels)


class ROIMaskHead(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_mask_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)
        mask_logits = self.predictor(x)
        if not self.training:
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}
        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)
        return x, all_proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg, in_channels):
    return ROIMaskHead(cfg, in_channels)


def build_roi_heads(cfg, in_channels):
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(('box', build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(('mask', build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(('keypoint', build_roi_keypoint_head(cfg, in_channels)))
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)
    return roi_heads


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack([_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])])
    return torch.from_numpy(anchors)


def generate_anchors(stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(stride, np.array(sizes, dtype=np.float) / stride, np.array(aspect_ratios, dtype=np.float))


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0), anchor_strides=(8, 16, 32), straddle_thresh=0):
        super(AnchorGenerator, self).__init__()
        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchors = [generate_anchors(anchor_stride, sizes, aspect_ratios).float()]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError('FPN should have #anchor_strides == #sizes')
            cell_anchors = [generate_anchors(anchor_stride, size if isinstance(size, (tuple, list)) else (size,), aspect_ratios).float() for anchor_stride, size in zip(anchor_strides, sizes)]
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, grid_height * stride, step=stride, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
        return anchors

    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (anchors[..., 0] >= -self.straddle_thresh) & (anchors[..., 1] >= -self.straddle_thresh) & (anchors[..., 2] < image_width + self.straddle_thresh) & (anchors[..., 3] < image_height + self.straddle_thresh)
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
        boxlist.add_field('visibility', inds_inside)

    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(anchors_per_feature_map, (image_width, image_height), mode='xyxy')
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors


def make_anchor_generator(config):
    anchor_sizes = config.MODEL.RPN.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH
    if config.MODEL.RPN.USE_FPN:
        assert len(anchor_stride) == len(anchor_sizes), 'FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)'
    else:
        assert len(anchor_stride) == 1, 'Non-FPN should have a single ANCHOR_STRIDE'
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh)
    return anchor_generator


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(copied_fields)
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(anchors_per_image, targets_per_image, self.copied_fields)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0
            if 'not_visibility' in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field('visibility')] = -1
            if 'between_thresholds' in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1
            regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, anchors_per_image.bbox)
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness, box_regression = concat_box_prediction_layers(objectness, box_regression)
        objectness = objectness.squeeze()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        box_loss = smooth_l1_loss(box_regression[sampled_pos_inds], regression_targets[sampled_pos_inds], beta=1.0 / 9, size_average=False) / sampled_inds.numel()
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])
        return objectness_loss, box_loss


def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field('matched_idxs')
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(cfg.MODEL.RPN.FG_IOU_THRESHOLD, cfg.MODEL.RPN.BG_IOU_THRESHOLD, allow_low_quality_matches=True)
    fg_bg_sampler = BalancedPositiveNegativeSampler(cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION)
    loss_evaluator = RPNLossComputation(matcher, fg_bg_sampler, box_coder, generate_rpn_labels)
    return loss_evaluator


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    xywh_boxes = boxlist.convert('xywh').bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(self, pre_nms_top_n, post_nms_top_n, nms_thresh, min_size, box_coder=None, fpn_post_nms_top_n=None):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder
        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        device = proposals[0].bbox.device
        gt_boxes = [target.copy_with_fields([]) for target in targets]
        for gt_box in gt_boxes:
            gt_box.add_field('objectness', torch.ones(len(gt_box), device=device))
        proposals = [cat_boxlist((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]
        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        objectness = objectness.sigmoid()
        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        num_anchors = A * H * W
        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)
        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx]
        image_shapes = [box.size for box in anchors]
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]
        proposals = self.box_coder.decode(box_regression.view(-1, 4), concat_anchors.view(-1, 4))
        proposals = proposals.view(N, -1, 4)
        result = []
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            boxlist = BoxList(proposal, im_shape, mode='xyxy')
            boxlist.add_field('objectness', score)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            boxlist = boxlist_nms(boxlist, self.nms_thresh, max_proposals=self.post_nms_top_n, score_field='objectness')
            result.append(boxlist)
        return result

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))
        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)
        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        if self.training:
            objectness = torch.cat([boxlist.get_field('objectness') for boxlist in boxlists], dim=0)
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field('objectness')
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST
    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = RPNPostProcessor(pre_nms_top_n=pre_nms_top_n, post_nms_top_n=post_nms_top_n, nms_thresh=nms_thresh, min_size=min_size, box_coder=rpn_box_coder, fpn_post_nms_top_n=fpn_post_nms_top_n)
    return box_selector


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):
        super(RPNModule, self).__init__()
        self.cfg = cfg.clone()
        anchor_generator = make_anchor_generator(cfg)
        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(cfg, in_channels, anchor_generator.num_anchors_per_location()[0])
        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)
        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)
        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            boxes = anchors
        else:
            with torch.no_grad():
                boxes = self.box_selector_train(anchors, objectness, rpn_box_regression, targets)
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(anchors, objectness, rpn_box_regression, targets)
        losses = {'loss_objectness': loss_objectness, 'loss_rpn_box_reg': loss_rpn_box_reg}
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            inds = [box.get_field('objectness').sort(descending=True)[1] for box in boxes]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


class FCOSHead(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER
        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            if self.use_dcn_in_tower and i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d
            cls_tower.append(conv_func(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(conv_func(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))
            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, centerness


INF = 100000000


def get_num_gpus():
    return int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(cfg.MODEL.FCOS.LOSS_GAMMA, cfg.MODEL.FCOS.LOSS_ALPHA)
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction='sum')

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        """
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        """
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1))
        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(points_all_level, targets, expanded_object_sizes_of_interest)
        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0))
            reg_targets_per_level = torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)
        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == 'xyxy'
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field('labels')
            area = targets_per_im.area()
            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(bboxes, self.fpn_strides, self.num_points_per_level, xs, ys, radius=self.center_sampling_radius)
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            is_cared_in_the_level = (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])
            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0] * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets = self.prepare_targets(locations, targets)
        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)
        cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten.int()) / num_pos_avg_per_gpu
        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            sum_centerness_targets_avg_per_gpu = reduce_sum(centerness_targets.sum()).item() / float(num_gpus)
            reg_loss = self.box_reg_loss_func(box_regression_flatten, reg_targets_flatten, centerness_targets) / sum_centerness_targets_avg_per_gpu
            centerness_loss = self.centerness_loss_func(centerness_flatten, centerness_targets) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()
        return cls_loss, reg_loss, centerness_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """

    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh, fpn_post_nms_top_n, min_size, num_classes):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes

    def forward_for_single_feature_map(self, locations, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        box_cls = box_cls * centerness[:, :, None]
        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1
            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]
            per_pre_nms_top_n = pre_nms_top_n[i]
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
            detections = torch.stack([per_locations[:, 0] - per_box_regression[:, 0], per_locations[:, 1] - per_box_regression[:, 1], per_locations[:, 0] + per_box_regression[:, 2], per_locations[:, 1] + per_box_regression[:, 3]], dim=1)
            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode='xyxy')
            boxlist.add_field('labels', per_class)
            boxlist.add_field('scores', torch.sqrt(per_box_cls))
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)
        return results

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(self.forward_for_single_feature_map(l, o, b, c, image_sizes))
        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)
        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            keep = batched_nms(boxlists[i].bbox, boxlists[i].get_field('scores'), boxlists[i].get_field('labels'), self.nms_thresh)
            result = boxlists[i][keep]
            number_of_detections = len(result)
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field('scores')
                image_thresh, _ = torch.kthvalue(cls_scores.cpu(), number_of_detections - self.fpn_post_nms_top_n + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_fcos_postprocessor(config):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    bbox_aug_enabled = config.TEST.BBOX_AUG.ENABLED
    box_selector = FCOSPostProcessor(pre_nms_thresh=pre_nms_thresh, pre_nms_top_n=pre_nms_top_n, nms_thresh=nms_thresh, fpn_post_nms_top_n=fpn_post_nms_top_n, min_size=0, num_classes=config.MODEL.FCOS.NUM_CLASSES, bbox_aug_enabled=bbox_aug_enabled)
    return box_selector


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()
        head = FCOSHead(cfg, in_channels)
        box_selector_test = make_fcos_postprocessor(cfg)
        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)
        if self.training:
            return self._forward_train(locations, box_cls, box_regression, centerness, targets)
        else:
            return self._forward_test(locations, box_cls, box_regression, centerness, images.image_sizes)

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(locations, box_cls, box_regression, centerness, targets)
        losses = {'loss_cls': loss_box_cls, 'loss_reg': loss_box_reg, 'loss_centerness': loss_centerness}
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(locations, box_cls, box_regression, centerness, image_sizes)
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(h, w, self.fpn_strides[level], feature.device)
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)


class RetinaNetHead(torch.nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RetinaNetHead, self).__init__()
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE
        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.RETINANET.NUM_CONVS):
            cls_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            bbox_tower.append(nn.ReLU())
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits, self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            logits.append(self.cls_logits(self.cls_tower(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
        return logits, bbox_reg


def make_anchor_generator_retinanet(config):
    anchor_sizes = config.MODEL.RETINANET.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RETINANET.ASPECT_RATIOS
    anchor_strides = config.MODEL.RETINANET.ANCHOR_STRIDES
    straddle_thresh = config.MODEL.RETINANET.STRADDLE_THRESH
    octave = config.MODEL.RETINANET.OCTAVE
    scales_per_octave = config.MODEL.RETINANET.SCALES_PER_OCTAVE
    assert len(anchor_strides) == len(anchor_sizes), 'Only support FPN now'
    new_anchor_sizes = []
    for size in anchor_sizes:
        per_layer_anchor_sizes = []
        for scale_per_octave in range(scales_per_octave):
            octave_scale = octave ** (scale_per_octave / float(scales_per_octave))
            per_layer_anchor_sizes.append(octave_scale * size)
        new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
    anchor_generator = AnchorGenerator(tuple(new_anchor_sizes), aspect_ratios, anchor_strides, straddle_thresh)
    return anchor_generator


class RetinaNetLossComputation(RPNLossComputation):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, proposal_matcher, box_coder, generate_labels_func, sigmoid_focal_loss, bbox_reg_beta=0.11, regress_norm=1.0):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.box_cls_loss_func = sigmoid_focal_loss
        self.bbox_reg_beta = bbox_reg_beta
        self.copied_fields = ['labels']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['between_thresholds']
        self.regress_norm = regress_norm

    def __call__(self, anchors, box_cls, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            retinanet_cls_loss (Tensor)
            retinanet_regression_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        N = len(labels)
        box_cls, box_regression = concat_box_prediction_layers(box_cls, box_regression)
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)
        retinanet_regression_loss = smooth_l1_loss(box_regression[pos_inds], regression_targets[pos_inds], beta=self.bbox_reg_beta, size_average=False) / max(1, pos_inds.numel() * self.regress_norm)
        labels = labels.int()
        retinanet_cls_loss = self.box_cls_loss_func(box_cls, labels) / (pos_inds.numel() + N)
        return retinanet_cls_loss, retinanet_regression_loss


def generate_retinanet_labels(matched_targets):
    labels_per_image = matched_targets.get_field('labels')
    return labels_per_image


def make_retinanet_loss_evaluator(cfg, box_coder):
    matcher = Matcher(cfg.MODEL.RETINANET.FG_IOU_THRESHOLD, cfg.MODEL.RETINANET.BG_IOU_THRESHOLD, allow_low_quality_matches=True)
    sigmoid_focal_loss = SigmoidFocalLoss(cfg.MODEL.RETINANET.LOSS_GAMMA, cfg.MODEL.RETINANET.LOSS_ALPHA)
    loss_evaluator = RetinaNetLossComputation(matcher, box_coder, generate_retinanet_labels, sigmoid_focal_loss, bbox_reg_beta=cfg.MODEL.RETINANET.BBOX_REG_BETA, regress_norm=cfg.MODEL.RETINANET.BBOX_REG_WEIGHT)
    return loss_evaluator


class RetinaNetPostProcessor(RPNPostProcessor):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """

    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh, fpn_post_nms_top_n, min_size, num_classes, box_coder=None):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(RetinaNetPostProcessor, self).__init__(pre_nms_thresh, 0, nms_thresh, min_size)
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        if box_coder is None:
            box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.box_coder = box_coder

    def add_gt_proposals(self, proposals, targets):
        """
        This function is not used in RetinaNet
        """
        pass

    def forward_for_single_feature_map(self, anchors, box_cls, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = box_cls.device
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()
        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)
        num_anchors = A * H * W
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        results = []
        for per_box_cls, per_box_regression, per_pre_nms_top_n, per_candidate_inds, per_anchors in zip(box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors):
            per_box_cls = per_box_cls[per_candidate_inds]
            per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
            per_candidate_nonzeros = per_candidate_inds.nonzero()[top_k_indices, :]
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]
            per_class += 1
            detections = self.box_coder.decode(per_box_regression[per_box_loc, :].view(-1, 4), per_anchors.bbox[per_box_loc, :].view(-1, 4))
            boxlist = BoxList(detections, per_anchors.size, mode='xyxy')
            boxlist.add_field('labels', per_class)
            boxlist.add_field('scores', per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)
        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field('scores')
            labels = boxlists[i].get_field('labels')
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)
                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode='xyxy')
                boxlist_for_class.add_field('scores', scores_j)
                boxlist_for_class = boxlist_nms(boxlist_for_class, self.nms_thresh, score_field='scores')
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field('labels', torch.full((num_labels,), j, dtype=torch.int64, device=scores.device))
                result.append(boxlist_for_class)
            result = cat_boxlist(result)
            number_of_detections = len(result)
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field('scores')
                image_thresh, _ = torch.kthvalue(cls_scores.cpu(), number_of_detections - self.fpn_post_nms_top_n + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_retinanet_postprocessor(config, rpn_box_coder, is_train):
    pre_nms_thresh = config.MODEL.RETINANET.INFERENCE_TH
    pre_nms_top_n = config.MODEL.RETINANET.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.RETINANET.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    min_size = 0
    box_selector = RetinaNetPostProcessor(pre_nms_thresh=pre_nms_thresh, pre_nms_top_n=pre_nms_top_n, nms_thresh=nms_thresh, fpn_post_nms_top_n=fpn_post_nms_top_n, min_size=min_size, num_classes=config.MODEL.RETINANET.NUM_CLASSES, box_coder=rpn_box_coder)
    return box_selector


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and
    RetinaNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(RetinaNetModule, self).__init__()
        self.cfg = cfg.clone()
        anchor_generator = make_anchor_generator_retinanet(cfg)
        head = RetinaNetHead(cfg, in_channels)
        box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        box_selector_test = make_retinanet_postprocessor(cfg, box_coder, is_train=False)
        loss_evaluator = make_retinanet_loss_evaluator(cfg, box_coder)
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)
        if self.training:
            return self._forward_train(anchors, box_cls, box_regression, targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression)

    def _forward_train(self, anchors, box_cls, box_regression, targets):
        loss_box_cls, loss_box_reg = self.loss_evaluator(anchors, box_cls, box_regression, targets)
        losses = {'loss_retina_cls': loss_box_cls, 'loss_retina_reg': loss_box_reg}
        return anchors, losses

    def _forward_test(self, anchors, box_cls, box_regression):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        return boxes, {}


def build_retinanet(cfg, in_channels):
    return RetinaNetModule(cfg, in_channels)


def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.FCOS_ON:
        return build_fcos(cfg, in_channels)
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)
    return RPNModule(cfg, in_channels)


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors
        return ImageList(cast_tensor, self.image_sizes)


def to_image_list(tensors, size_divisible=0):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]
    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
        if size_divisible > 0:
            import math
            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)
        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        image_sizes = [im.shape[-2:] for im in tensors]
        return ImageList(batched_imgs, image_sizes)
    else:
        raise TypeError('Unsupported type for to_image_list: {}'.format(type(tensors)))


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError('In training mode, targets should be passed')
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            x = features
            result = proposals
            detector_losses = {}
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        return result


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-06):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls - self.k_min


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(ROIAlign(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio))
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat([torch.full((len(b), 1), i, dtype=dtype, device=device) for i, b in enumerate(boxes)], dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)
        levels = self.map_levels(boxes)
        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]
        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros((num_rois, num_channels, output_size, output_size), dtype=dtype, device=device)
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level)
        return result


class ResNet50Conv5ROIFeatureExtractor(nn.Module):

    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()
        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(block_module=config.MODEL.RESNETS.TRANS_FUNC, stages=(stage,), num_groups=config.MODEL.RESNETS.NUM_GROUPS, width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP, stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1, stride_init=None, res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS, dilation=config.MODEL.RESNETS.RES5_DILATION)
        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


def make_fc(dim_in, hidden_dim, use_gn=False):
    """
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    """
    if use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, group_norm(hidden_dim))
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor, self).__init__()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        self.pooler = pooler
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION
        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(nn.Conv2d(in_channels, conv_head_dim, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False if use_gn else True))
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))
        self.add_module('xconvs', nn.Sequential(*xconvs))
        for modules in [self.xconvs]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)
        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


class FastRCNNPredictor(nn.Module):

    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)
        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


class FPNPredictor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels
        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class KeypointRCNNFeatureExtractor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(KeypointRCNNFeatureExtractor, self).__init__()
        resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        self.pooler = pooler
        input_features = in_channels
        layers = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS
        next_feature = input_features
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = 'conv_fcn{}'.format(layer_idx)
            module = Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))
        return x


class KeypointRCNNPredictor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(KeypointRCNNPredictor, self).__init__()
        input_features = in_channels
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES
        deconv_kernel = 4
        self.kps_score_lowres = layers.ConvTranspose2d(input_features, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1)
        nn.init.kaiming_normal_(self.kps_score_lowres.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        x = layers.interpolate(x, scale_factor=self.up_scale, mode='bilinear', align_corners=False)
        return x


class MaskPostProcessorCOCOFormat(MaskPostProcessor):
    """
    From the results of the CNN, post process the results
    so that the masks are pasted in the image, and
    additionally convert the results to COCO format.
    """

    def forward(self, x, boxes):
        import numpy as np
        results = super(MaskPostProcessorCOCOFormat, self).forward(x, boxes)
        for result in results:
            masks = result.get_field('mask').cpu()
            rles = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], order='F'))[0] for mask in masks]
            for rle in rles:
                rle['counts'] = rle['counts'].decode('utf-8')
            result.add_field('mask', rles)
        return results


def make_conv3x3(in_channels, out_channels, dilation=1, stride=1, use_gn=False, use_relu=False, kaiming_init=True):
    conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False if use_gn else True)
    if kaiming_init:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    module = [conv]
    if use_gn:
        module.append(group_norm(out_channels))
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv


class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()
        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels
        self.pooler = pooler
        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION
        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = 'mask_fcn{}'.format(layer_idx)
            module = make_conv3x3(next_feature, layer_features, dilation=dilation, stride=1, use_gn=use_gn)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))
        return x


class MaskRCNNC4Predictor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        num_inputs = in_channels
        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


class MaskRCNNConv1x1Predictor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(MaskRCNNConv1x1Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_inputs = in_channels
        self.mask_fcn_logits = Conv2d(num_inputs, num_classes, 1, 1, 0)
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.mask_fcn_logits(x)


class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeadConvRegressor, self).__init__()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]
        return logits, bbox_reg


class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(RPNHeadFeatureSingleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
        self.out_channels = in_channels

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        x = [F.relu(self.conv(z)) for z in x]
        return x


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class ONNX_FCOS(nn.Module):

    def __init__(self, onnx_model_path, cfg):
        super(ONNX_FCOS, self).__init__()
        self.onnx_model = backend.prepare(onnx.load(onnx_model_path), device=cfg.MODEL.DEVICE.upper())
        self.postprocessing = FCOSPostProcessor(pre_nms_thresh=cfg.MODEL.FCOS.INFERENCE_TH, pre_nms_top_n=cfg.MODEL.FCOS.PRE_NMS_TOP_N, nms_thresh=cfg.MODEL.FCOS.NMS_TH, fpn_post_nms_top_n=cfg.TEST.DETECTIONS_PER_IMG, min_size=0, num_classes=cfg.MODEL.FCOS.NUM_CLASSES)
        self.cfg = cfg
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images):
        outputs = self.onnx_model.run(images.tensors.cpu().numpy())
        outputs = [torch.from_numpy(o) for o in outputs]
        num_outputs = len(outputs) // 3
        logits = outputs[:num_outputs]
        bbox_reg = outputs[num_outputs:2 * num_outputs]
        centerness = outputs[2 * num_outputs:]
        locations = self.compute_locations(logits)
        boxes = self.postprocessing(locations, logits, bbox_reg, centerness, images.image_sizes)
        return boxes

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(h, w, self.fpn_strides[level], feature.device)
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


import torch
from torch.nn import MSELoss, ReLU
from paritybench._paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CascadeConv3x3,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelShuffle,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvBNRelu,
     lambda: ([], {'input_depth': 1, 'output_depth': 1, 'kernel': 4, 'stride': 1, 'pad': 4, 'no_bias': 4, 'use_relu': 'relu', 'bn_type': 'bn'}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (ConvTranspose2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FrozenBatchNorm2d,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IOULoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (IRFBlock,
     lambda: ([], {'input_depth': 1, 'output_depth': 1, 'expansion': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (Identity,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LastLevelMaxPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LastLevelP6P7,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RPNHead,
     lambda: ([], {'cfg': _mock_config(), 'in_channels': 4, 'num_anchors': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RPNHeadConvRegressor,
     lambda: ([], {'cfg': _mock_config(), 'in_channels': 4, 'num_anchors': 4}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     True),
    (RPNHeadFeatureSingleConv,
     lambda: ([], {'cfg': _mock_config(), 'in_channels': 4}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     True),
    (SEModule,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Scale,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Shift,
     lambda: ([], {'C': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ShiftBlock5x5,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'expansion': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_tianzhi0549_FCOS(_paritybench_base):
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

