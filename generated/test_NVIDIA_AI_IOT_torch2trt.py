
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


from string import Template


import torch


import torchvision


import torchvision.transforms as transforms


import torch.nn as nn


import numpy as np


import torch.optim as optim


import time


import collections


import torchvision.models as models


import re


import math


from typing import Literal


from enum import Enum


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import CppExtension


import torch.nn.functional as F


import copy


import inspect


from torch import nn


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


from torch.nn.modules.conv import _ConvTransposeNd


from uuid import uuid1


from collections import defaultdict


class qconv2d(torch.nn.Module):
    """
    common layer for qat and non qat mode
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int'=3, stride: 'int'=1, padding: 'int'=0, groups: 'int'=1, dilation: 'int'=1, bias=None, padding_mode: 'str'='zeros', eps: 'float'=1e-05, momentum: 'float'=0.1, freeze_bn=False, act: 'bool'=True, norm: 'bool'=True, qat: 'bool'=False, infer: 'bool'=False):
        super().__init__()
        if qat:
            if infer:
                if norm:
                    layer_list = [IQuantConvBN2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias, padding_mode=padding_mode)]
                else:
                    layer_list = [IQuantConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias, padding_mode=padding_mode)]
            elif norm:
                layer_list = [QuantConvBN2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias, padding_mode=padding_mode, quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)]
            else:
                layer_list = [QuantConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias, padding_mode=padding_mode, quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)]
            if act:
                if infer:
                    layer_list.append(IQuantReLU())
                else:
                    layer_list.append(QuantReLU())
            self.qconv = nn.Sequential(*layer_list)
        else:
            layer_list = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias, groups=groups)]
            if norm:
                layer_list.append(nn.BatchNorm2d(out_channels))
            if act:
                layer_list.append(nn.ReLU())
            self.qconv = nn.Sequential(*layer_list)

    def forward(self, inputs):
        return self.qconv(inputs)


class vanilla_cnn(nn.Module):

    def __init__(self, qat_mode=False, infer=False):
        super().__init__()
        self.qat = qat_mode
        self.layer1 = qconv2d(3, 32, padding=1, qat=qat_mode, infer=infer)
        self.layer2 = qconv2d(32, 64, padding=1, qat=qat_mode, infer=infer)
        self.layer3 = qconv2d(64, 128, padding=1, qat=qat_mode, infer=infer)
        self.layer4 = qconv2d(128, 256, padding=1, qat=qat_mode, infer=infer)
        self.layer5 = nn.MaxPool2d(kernel_size=2, stride=8)
        self.fcs = nn.Sequential(nn.Linear(4096, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, norm=False, act=False, qat_mode=False, infer=False):
    """3x3 convolution with padding"""
    return qconv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, groups=groups, padding=dilation, dilation=dilation, bias=False, act=act, norm=norm, qat=qat_mode, infer=infer)


class qrelu(torch.nn.Module):

    def __init__(self, inplace=False, qat=False, infer=False):
        super().__init__()
        if qat:
            if infer:
                self.relu = IQuantReLU(inplace)
            else:
                self.relu = QuantReLU(inplace)
        else:
            self.relu = nn.ReLU(inplace)

    def forward(self, input):
        return self.relu(input)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, qat_mode=False, infer=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride, norm=True, qat_mode=qat_mode, infer=infer)
        self.relu1 = qrelu(inplace=True, qat=qat_mode, infer=infer)
        self.conv2 = conv3x3(planes, planes, norm=True, qat_mode=qat_mode, infer=infer)
        self.relu2 = qrelu(inplace=True, qat=qat_mode, infer=infer)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)
        return out


def conv1x1(in_planes, out_planes, stride=1, norm=False, act=False, qat_mode=False, infer=False):
    """1x1 convolution"""
    return qconv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, bias=False, act=act, norm=norm, qat=qat_mode, infer=infer)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, qat_mode=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width, qat_mode=qat_mode)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, qat_mode=qat_mode)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, qat_mode=qat_mode)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, qat_mode=False, infer=False):
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
        self.conv1 = qconv2d(in_channels=3, out_channels=self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, norm=True, act=False, qat=qat_mode, infer=infer)
        self.relu = qrelu(inplace=True, qat=qat_mode, infer=infer)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], qat_mode=qat_mode, infer=infer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], qat_mode=qat_mode, infer=infer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], qat_mode=qat_mode, infer=infer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], qat_mode=qat_mode, infer=infer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, qat_mode=False, infer=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride, norm=True, qat_mode=qat_mode, infer=infer))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, qat_mode, infer=infer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, qat_mode=qat_mode, infer=infer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class PoolFix(torch.nn.Module):

    def forward(self, x):
        return torch.mean(x, dim=-1, keepdim=True)


class UnaryModule(torch.nn.Module):

    def __init__(self, fn):
        super(UnaryModule, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class BinaryModule(torch.nn.Module):

    def __init__(self, fn):
        super(BinaryModule, self).__init__()
        self.fn = fn

    def forward(self, a, b):
        return self.fn(a, b)


class YOLOXFocusTestModule(nn.Module):

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return x


class FlattenModule(torch.nn.Module):

    def __init__(self, start_dim, end_dim):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return torch.flatten(x, self.start_dim, self.end_dim)


class ModelWrapper(torch.nn.Module):

    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['out']


class TensorQuantizer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('learned_amax', torch.tensor(1.0))


class Unflatten(nn.Module):

    def __init__(self, module, input_flattener=None, output_flattener=None):
        super().__init__()
        self.module = module
        self.input_flattener = input_flattener
        self.output_flattener = output_flattener

    def forward(self, *args):
        if self.input_flattener is not None:
            args = self.input_flattener.flatten(args)
        output = self.module(*args)
        if self.output_flattener is not None:
            output = self.output_flattener.unflatten(output)
        return output


class Flatten(nn.Module):

    def __init__(self, module, input_flattener=None, output_flattener=None):
        super().__init__()
        self.module = module
        self.input_flattener = input_flattener
        self.output_flattener = output_flattener

    def forward(self, *args):
        if self.input_flattener is not None:
            args = self.input_flattener.unflatten(*args)
        output = self.module(*args)
        if self.output_flattener is not None:
            output = self.output_flattener.flatten(output)
        return output


def _default_condition(x):
    return isinstance(x, torch.Tensor) and (x.dtype is torch.half or x.dtype is torch.float or x.dtype == torch.bool)


def _make_schema_from_value(value, condition=_default_condition, size=0):
    if condition(value):
        return size, size + 1
    elif isinstance(value, list) or isinstance(value, tuple):
        schema = []
        for child_value in value:
            child_schema, size = _make_schema_from_value(child_value, condition, size)
            schema.append(child_schema)
        if isinstance(value, tuple):
            schema = tuple(schema)
        return schema, size
    elif isinstance(value, dict):
        schema = {}
        for child_key in sorted(value.keys()):
            child_value = value[child_key]
            child_schema, size = _make_schema_from_value(child_value, condition, size)
            schema[child_key] = child_schema
        return schema, size
    else:
        return None, size


class Flattener(object):

    def __init__(self, schema, size):
        self._schema = schema
        self._size = size

    @staticmethod
    def from_value(value, condition=_default_condition):
        return Flattener(*_make_schema_from_value(value, condition))

    @staticmethod
    def from_dict(x):
        return Flattener(x['schema'], x['size'])

    def dict(self):
        return {'schema': self.schema, 'size': self.size}

    @property
    def schema(self):
        return self._schema

    @property
    def size(self):
        return self._size

    def __len__(self):
        return self._size

    def _flatten(self, value, result):
        if isinstance(self._schema, int):
            result[self._schema] = value
        elif isinstance(self._schema, list) or isinstance(self._schema, tuple):
            for child_value, child_schema in zip(value, self._schema):
                Flattener(child_schema, self.size)._flatten(child_value, result)
        elif isinstance(self._schema, dict):
            for key in sorted(self._schema.keys()):
                child_value = value[key]
                child_schema = self._schema[key]
                Flattener(child_schema, self.size)._flatten(child_value, result)

    def flatten(self, value):
        result = [None for i in range(self.size)]
        self._flatten(value, result)
        return result

    def unflatten(self, flattened):
        if isinstance(self._schema, int):
            return flattened[self._schema]
        elif isinstance(self._schema, list) or isinstance(self._schema, tuple):
            result = []
            for child_schema in self._schema:
                result.append(Flattener(child_schema, self.size).unflatten(flattened))
            if isinstance(self._schema, tuple):
                result = tuple(result)
            return result
        elif isinstance(self._schema, dict):
            result = {}
            for child_key in sorted(self._schema.keys()):
                child_schema = self._schema[child_key]
                result[child_key] = Flattener(child_schema, self.size).unflatten(flattened)
            return result
        else:
            return None


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


def trt_version():
    return Version(trt.__version__)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


class TRTModule(torch.nn.Module):

    def __init__(self, engine=None, input_names=None, output_names=None, input_flattener=None, output_flattener=None):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)
        if isinstance(engine, str):
            with open(engine, 'rb') as f:
                engine = f.read()
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(engine)
        elif isinstance(engine, trt.IHostMemory):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(engine)
        self.engine = engine
        if self.engine is not None:
            self.context = self.engine.create_execution_context()
            self._update_name_binindgs_maps()
        self.input_names = input_names
        self.output_names = output_names
        self.input_flattener = input_flattener
        self.output_flattener = output_flattener

    def _update_name_binindgs_maps(self):
        if trt_version() >= '10.0':
            self._update_name_binding_maps_trt_10()
        else:
            self._update_name_binding_maps_pre_trt_10()

    def _update_name_binding_maps_trt_10(self):
        self._name_to_binding = {}
        self._binding_to_name = {}
        for i in range(self.engine.num_io_tensors):
            name_i = self.engine.get_tensor_name(i)
            self._name_to_binding[name_i] = i
            self._binding_to_name[i] = name_i

    def _update_name_binding_maps_pre_trt_10(self):
        self._name_to_binding = {}
        self._binding_to_name = {}
        for i in range(self.engine.num_bindings):
            name_i = self.engine.get_binding_name(i)
            self._name_to_binding[name_i] = i
            self._binding_to_name[i] = name_i

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + 'engine'] = bytearray(self.engine.serialize())
        state_dict[prefix + 'input_names'] = self.input_names
        state_dict[prefix + 'output_names'] = self.output_names
        state_dict[prefix + 'input_flattener'] = self.input_flattener.dict()
        state_dict[prefix + 'output_flattener'] = self.output_flattener.dict()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        engine_bytes = state_dict[prefix + 'engine']
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()
        self.input_names = state_dict[prefix + 'input_names']
        self.output_names = state_dict[prefix + 'output_names']
        if 'input_flattener' in state_dict:
            self.input_flattener = Flattener.from_dict(state_dict['input_flattener'])
        else:
            self.input_flattener = None
        if 'output_flattener' in state_dict:
            self.output_flattener = Flattener.from_dict(state_dict['output_flattener'])
        else:
            self.output_flattener = None
        self._update_name_binindgs_maps()

    def _forward_pre_10(self, *inputs):
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        if self.input_flattener is not None:
            inputs = self.input_flattener.flatten(inputs)
        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            shape = tuple(inputs[i].shape)
            bindings[idx] = inputs[i].contiguous().data_ptr()
            self.context.set_binding_shape(idx, shape)
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
        if self.output_flattener is not None:
            outputs = self.output_flattener.unflatten(outputs)
        else:
            outputs = tuple(outputs)
            if len(outputs) == 1:
                outputs = outputs[0]
        return outputs

    def _forward_post_10(self, *inputs):
        if self.input_flattener is not None:
            inputs = self.input_flattener.flatten(inputs)
        for i, input_name in enumerate(self.input_names):
            shape = tuple(inputs[i].shape)
            data_ptr = inputs[i].contiguous().data_ptr()
            self.context.set_tensor_address(input_name, data_ptr)
            self.context.set_input_shape(input_name, shape)
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            dtype = torch_dtype_from_trt(self.engine.get_tensor_dtype(output_name))
            shape = tuple(self.context.get_tensor_shape(output_name))
            device = torch_device_from_trt(self.engine.get_tensor_location(output_name))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            self.context.set_tensor_address(output_name, output.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if self.output_flattener is not None:
            outputs = self.output_flattener.unflatten(outputs)
        else:
            outputs = tuple(outputs)
            if len(outputs) == 1:
                outputs = outputs[0]
        return outputs

    def forward(self, *inputs):
        if trt_version() < '10.0':
            return self._forward_pre_10(*inputs)
        else:
            return self._forward_post_10(*inputs)

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FlattenModule,
     lambda: ([], {'start_dim': 4, 'end_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (PoolFix,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UnaryModule,
     lambda: ([], {'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (YOLOXFocusTestModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (qconv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (qrelu,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

