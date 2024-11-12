
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


import matplotlib.pyplot as plt


import numpy as np


import torch


from copy import deepcopy


from typing import Any


from typing import List


from typing import Tuple


from torch.utils.data._utils.collate import default_collate


import logging


from typing import Optional


from torchvision.transforms.functional import to_tensor


import math


from functools import partial


from typing import Dict


from torch import nn


from torchvision.models import mobilenetv3


from torchvision.models.mobilenetv3 import MobileNetV3


from typing import Union


from typing import Callable


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import ResNet as TVResNet


from torchvision.models.resnet import resnet18 as tv_resnet18


from torchvision.models.resnet import resnet34 as tv_resnet34


from torchvision.models.resnet import resnet50 as tv_resnet50


from torchvision.models import vgg as tv_vgg


from torch import Tensor


from torch.nn.functional import max_pool2d


from torch.nn import functional as F


from torchvision.models import resnet34


from torchvision.models import resnet50


from torchvision.models._utils import IntermediateLayerGetter


from torchvision.ops.deform_conv import DeformConv2d


import torch.nn as nn


from torchvision.transforms import functional as F


from torchvision.transforms import transforms as T


import tensorflow as tf


from itertools import groupby


from itertools import permutations


from typing import Sequence


import random


from torch.nn.functional import pad


import time


from torch.nn.functional import cross_entropy


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import MultiplicativeLR


from torch.optim.lr_scheduler import OneCycleLR


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torchvision.transforms.v2 import Compose


from torchvision.transforms.v2 import GaussianBlur


from torchvision.transforms.v2 import InterpolationMode


from torchvision.transforms.v2 import Normalize


from torchvision.transforms.v2 import RandomGrayscale


from torchvision.transforms.v2 import RandomPerspective


from torchvision.transforms.v2 import RandomPhotometricDistort


from torchvision.transforms.v2 import RandomRotation


from torchvision.transforms import Normalize


from torch.optim.lr_scheduler import PolynomialLR


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data.distributed import DistributedSampler


import re


from collections import namedtuple


class MAGC(nn.Module):
    """Implements the Multi-Aspect Global Context Attention, as described in
    <https://arxiv.org/pdf/1910.02562.pdf>`_.

    Args:
    ----
        inplanes: input channels
        headers: number of headers to split channels
        attn_scale: if True, re-scale attention to counteract the variance distibutions
        ratio: bottleneck ratio
        **kwargs
    """

    def __init__(self, inplanes: 'int', headers: 'int'=8, attn_scale: 'bool'=False, ratio: 'float'=0.0625, cfg: 'Optional[Dict[str, Any]]'=None) ->None:
        super().__init__()
        self.headers = headers
        self.inplanes = inplanes
        self.attn_scale = attn_scale
        self.planes = int(inplanes * ratio)
        self.single_header_inplanes = int(inplanes / headers)
        self.conv_mask = nn.Conv2d(self.single_header_inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.transform = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

    def forward(self, inputs: 'torch.Tensor') ->torch.Tensor:
        batch, _, height, width = inputs.size()
        x = inputs.view(batch * self.headers, self.single_header_inplanes, height, width)
        shortcut = x
        shortcut = shortcut.view(batch * self.headers, self.single_header_inplanes, height * width)
        context_mask = self.conv_mask(x)
        context_mask = context_mask.view(batch * self.headers, -1)
        if self.attn_scale and self.headers > 1:
            context_mask = context_mask / math.sqrt(self.single_header_inplanes)
        context_mask = self.softmax(context_mask)
        context = (shortcut * context_mask.unsqueeze(1)).sum(-1)
        context = context.view(batch, self.headers * self.single_header_inplanes, 1, 1)
        transformed = self.transform(context)
        return inputs + transformed


def _addindent(s_, num_spaces):
    s = s_.split('\n')
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * ' ' + line) for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class NestedObject:
    """Base class for all nested objects in doctr"""
    _children_names: 'List[str]'

    def extra_repr(self) ->str:
        return ''

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        if hasattr(self, '_children_names'):
            for key in self._children_names:
                child = getattr(self, key)
                if isinstance(child, list) and len(child) > 0:
                    child_str = ',\n'.join([repr(subchild) for subchild in child])
                    if len(child) > 1:
                        child_str = _addindent(f'\n{child_str},', 2) + '\n'
                    child_str = f'[{child_str}]'
                else:
                    child_str = repr(child)
                child_str = _addindent(child_str, 2)
                child_lines.append('(' + key + '): ' + child_str)
        lines = extra_lines + child_lines
        main_str = self.__class__.__name__ + '('
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str


class Resize(T.Resize):
    """Resize the input image to the given size"""

    def __init__(self, size: 'Union[int, Tuple[int, int]]', interpolation=F.InterpolationMode.BILINEAR, preserve_aspect_ratio: 'bool'=False, symmetric_pad: 'bool'=False) ->None:
        super().__init__(size, interpolation, antialias=True)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        if not isinstance(self.size, (int, tuple, list)):
            raise AssertionError('size should be either a tuple, a list or an int')

    def forward(self, img: 'torch.Tensor', target: 'Optional[np.ndarray]'=None) ->Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
        if isinstance(self.size, int):
            target_ratio = img.shape[-2] / img.shape[-1]
        else:
            target_ratio = self.size[0] / self.size[1]
        actual_ratio = img.shape[-2] / img.shape[-1]
        if not self.preserve_aspect_ratio or target_ratio == actual_ratio and isinstance(self.size, (tuple, list)):
            if target is not None:
                return super().forward(img), target
            return super().forward(img)
        else:
            if isinstance(self.size, (tuple, list)):
                if actual_ratio > target_ratio:
                    tmp_size = self.size[0], max(int(self.size[0] / actual_ratio), 1)
                else:
                    tmp_size = max(int(self.size[1] * actual_ratio), 1), self.size[1]
            elif isinstance(self.size, int):
                if img.shape[-2] <= img.shape[-1]:
                    tmp_size = max(int(self.size * actual_ratio), 1), self.size
                else:
                    tmp_size = self.size, max(int(self.size / actual_ratio), 1)
            img = F.resize(img, tmp_size, self.interpolation, antialias=True)
            raw_shape = img.shape[-2:]
            if isinstance(self.size, (tuple, list)):
                _pad = 0, self.size[1] - img.shape[-1], 0, self.size[0] - img.shape[-2]
                if self.symmetric_pad:
                    half_pad = math.ceil(_pad[1] / 2), math.ceil(_pad[3] / 2)
                    _pad = half_pad[0], _pad[1] - half_pad[0], half_pad[1], _pad[3] - half_pad[1]
                img = pad(img, _pad)
            if target is not None:
                if self.symmetric_pad:
                    offset = half_pad[0] / img.shape[-1], half_pad[1] / img.shape[-2]
                if self.preserve_aspect_ratio:
                    if target.shape[1:] == (4,):
                        if isinstance(self.size, (tuple, list)) and self.symmetric_pad:
                            target[:, [0, 2]] = offset[0] + target[:, [0, 2]] * raw_shape[-1] / img.shape[-1]
                            target[:, [1, 3]] = offset[1] + target[:, [1, 3]] * raw_shape[-2] / img.shape[-2]
                        else:
                            target[:, [0, 2]] *= raw_shape[-1] / img.shape[-1]
                            target[:, [1, 3]] *= raw_shape[-2] / img.shape[-2]
                    elif target.shape[1:] == (4, 2):
                        if isinstance(self.size, (tuple, list)) and self.symmetric_pad:
                            target[..., 0] = offset[0] + target[..., 0] * raw_shape[-1] / img.shape[-1]
                            target[..., 1] = offset[1] + target[..., 1] * raw_shape[-2] / img.shape[-2]
                        else:
                            target[..., 0] *= raw_shape[-1] / img.shape[-1]
                            target[..., 1] *= raw_shape[-2] / img.shape[-2]
                    else:
                        raise AssertionError('Boxes should be in the format (n_boxes, 4, 2) or (n_boxes, 4)')
                return img, np.clip(target, 0, 1)
            return img

    def __repr__(self) ->str:
        interpolate_str = self.interpolation.value
        _repr = f"output_size={self.size}, interpolation='{interpolate_str}'"
        if self.preserve_aspect_ratio:
            _repr += f', preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}'
        return f'{self.__class__.__name__}({_repr})'


ENV_VARS_TRUE_VALUES = {'1', 'ON', 'YES', 'TRUE'}


class PreProcessor(NestedObject):
    """Implements an abstract preprocessor object which performs casting, resizing, batching and normalization.

    Args:
    ----
        output_size: expected size of each page in format (H, W)
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
        **kwargs: additional arguments for the resizing operation
    """
    _children_names: 'List[str]' = ['resize', 'normalize']

    def __init__(self, output_size: 'Tuple[int, int]', batch_size: 'int', mean: 'Tuple[float, float, float]'=(0.5, 0.5, 0.5), std: 'Tuple[float, float, float]'=(1.0, 1.0, 1.0), **kwargs: Any) ->None:
        self.batch_size = batch_size
        self.resize = Resize(output_size, **kwargs)
        self.normalize = Normalize(mean, std)
        self._runs_on_cuda = tf.config.list_physical_devices('GPU') != []

    def batch_inputs(self, samples: 'List[tf.Tensor]') ->List[tf.Tensor]:
        """Gather samples into batches for inference purposes

        Args:
        ----
            samples: list of samples (tf.Tensor)

        Returns:
        -------
            list of batched samples
        """
        num_batches = int(math.ceil(len(samples) / self.batch_size))
        batches = [tf.stack(samples[idx * self.batch_size:min((idx + 1) * self.batch_size, len(samples))], axis=0) for idx in range(int(num_batches))]
        return batches

    def sample_transforms(self, x: 'Union[np.ndarray, tf.Tensor]') ->tf.Tensor:
        if x.ndim != 3:
            raise AssertionError('expected list of 3D Tensors')
        if isinstance(x, np.ndarray):
            if x.dtype not in (np.uint8, np.float32):
                raise TypeError('unsupported data type for numpy.ndarray')
            x = tf.convert_to_tensor(x)
        elif x.dtype not in (tf.uint8, tf.float16, tf.float32):
            raise TypeError('unsupported data type for torch.Tensor')
        if x.dtype == tf.uint8:
            x = tf.image.convert_image_dtype(x, dtype=tf.float32)
        x = self.resize(x)
        return x

    def __call__(self, x: 'Union[tf.Tensor, np.ndarray, List[Union[tf.Tensor, np.ndarray]]]') ->List[tf.Tensor]:
        """Prepare document data for model forwarding

        Args:
        ----
            x: list of images (np.array) or tensors (already resized and batched)

        Returns:
        -------
            list of page batches
        """
        if isinstance(x, (np.ndarray, tf.Tensor)):
            if x.ndim != 4:
                raise AssertionError('expected 4D Tensor')
            if isinstance(x, np.ndarray):
                if x.dtype not in (np.uint8, np.float32):
                    raise TypeError('unsupported data type for numpy.ndarray')
                x = tf.convert_to_tensor(x)
            elif x.dtype not in (tf.uint8, tf.float16, tf.float32):
                raise TypeError('unsupported data type for torch.Tensor')
            if x.dtype == tf.uint8:
                x = tf.image.convert_image_dtype(x, dtype=tf.float32)
            if (x.shape[1], x.shape[2]) != self.resize.output_size:
                x = tf.image.resize(x, self.resize.output_size, method=self.resize.method, antialias=self.resize.antialias)
            batches = [x]
        elif isinstance(x, list) and all(isinstance(sample, (np.ndarray, tf.Tensor)) for sample in x):
            samples = list(multithread_exec(self.sample_transforms, x, threads=1 if self._runs_on_cuda else None))
            batches = self.batch_inputs(samples)
        else:
            raise TypeError(f'invalid input type: {type(x)}')
        batches = list(multithread_exec(self.normalize, batches, threads=1 if self._runs_on_cuda else None))
        return batches


def set_device_and_dtype(model: 'Any', batches: 'List[torch.Tensor]', device: 'Union[str, torch.device]', dtype: 'torch.dtype') ->Tuple[Any, List[torch.Tensor]]:
    """Set the device and dtype of a model and its batches

    >>> import torch
    >>> from torch import nn
    >>> from doctr.models.utils import set_device_and_dtype
    >>> model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    >>> batches = [torch.rand(8) for _ in range(2)]
    >>> model, batches = set_device_and_dtype(model, batches, device="cuda", dtype=torch.float16)

    Args:
    ----
        model: the model to be set
        batches: the batches to be set
        device: the device to be used
        dtype: the dtype to be used

    Returns:
    -------
        the model and batches set
    """
    return model, [batch for batch in batches]


class OrientationPredictor(nn.Module):
    """Implements an object able to detect the reading direction of a text box or a page.
    4 possible orientations: 0, 90, 180, 270 (-90) degrees counter clockwise.

    Args:
    ----
        pre_processor: transform inputs for easier batched model inference
        model: core classification architecture (backbone + classification head)
    """

    def __init__(self, pre_processor: 'Optional[PreProcessor]', model: 'Optional[nn.Module]') ->None:
        super().__init__()
        self.pre_processor = pre_processor if isinstance(pre_processor, PreProcessor) else None
        self.model = model.eval() if isinstance(model, nn.Module) else None

    @torch.inference_mode()
    def forward(self, inputs: 'List[Union[np.ndarray, torch.Tensor]]') ->List[Union[List[int], List[float]]]:
        if any(input.ndim != 3 for input in inputs):
            raise ValueError('incorrect input shape: all inputs are expected to be multi-channel 2D images.')
        if self.model is None or self.pre_processor is None:
            return [[0] * len(inputs), [0] * len(inputs), [1.0] * len(inputs)]
        processed_batches = self.pre_processor(inputs)
        _params = next(self.model.parameters())
        self.model, processed_batches = set_device_and_dtype(self.model, processed_batches, _params.device, _params.dtype)
        predicted_batches = [self.model(batch) for batch in processed_batches]
        probs = [torch.max(torch.softmax(batch, dim=1), dim=1).values.cpu().detach().numpy() for batch in predicted_batches]
        predicted_batches = [out_batch.argmax(dim=1).cpu().detach().numpy() for out_batch in predicted_batches]
        class_idxs = [int(pred) for batch in predicted_batches for pred in batch]
        classes = [int(self.model.cfg['classes'][idx]) for idx in class_idxs]
        confs = [round(float(p), 2) for prob in probs for p in prob]
        return [class_idxs, classes, confs]


def conv_sequence_pt(in_channels: 'int', out_channels: 'int', relu: 'bool'=False, bn: 'bool'=False, **kwargs: Any) ->List[nn.Module]:
    """Builds a convolutional-based layer sequence

    >>> from torch.nn import Sequential
    >>> from doctr.models import conv_sequence
    >>> module = Sequential(conv_sequence(3, 32, True, True, kernel_size=3))

    Args:
    ----
        in_channels: number of input channels
        out_channels: number of output channels
        relu: whether ReLU should be used
        bn: should a batch normalization layer be added
        **kwargs: additional arguments to be passed to the convolutional layer

    Returns:
    -------
        list of layers
    """
    kwargs['bias'] = kwargs.get('bias', not bn)
    conv_seq: 'List[nn.Module]' = [nn.Conv2d(in_channels, out_channels, **kwargs)]
    if bn:
        conv_seq.append(nn.BatchNorm2d(out_channels))
    if relu:
        conv_seq.append(nn.ReLU(inplace=True))
    return conv_seq


def resnet_stage(in_channels: 'int', out_channels: 'int', num_blocks: 'int', stride: 'int') ->List[nn.Module]:
    """Build a ResNet stage"""
    _layers: 'List[nn.Module]' = []
    in_chan = in_channels
    s = stride
    for _ in range(num_blocks):
        downsample = None
        if in_chan != out_channels:
            downsample = nn.Sequential(*conv_sequence_pt(in_chan, out_channels, False, True, kernel_size=1, stride=s))
        _layers.append(BasicBlock(in_chan, out_channels, stride=s, downsample=downsample))
        in_chan = out_channels
        s = 1
    return _layers


class ResNet(nn.Sequential):
    """Implements a ResNet-31 architecture from `"Show, Attend and Read:A Simple and Strong Baseline for Irregular
    Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Args:
    ----
        num_blocks: number of resnet block in each stage
        output_channels: number of channels in each stage
        stage_conv: whether to add a conv_sequence after each stage
        stage_pooling: pooling to add after each stage (if None, no pooling)
        origin_stem: whether to use the orginal ResNet stem or ResNet-31's
        stem_channels: number of output channels of the stem convolutions
        attn_module: attention module to use in each stage
        include_top: whether the classifier head should be instantiated
        num_classes: number of output classes
    """

    def __init__(self, num_blocks: 'List[int]', output_channels: 'List[int]', stage_stride: 'List[int]', stage_conv: 'List[bool]', stage_pooling: 'List[Optional[Tuple[int, int]]]', origin_stem: 'bool'=True, stem_channels: 'int'=64, attn_module: 'Optional[Callable[[int], nn.Module]]'=None, include_top: 'bool'=True, num_classes: 'int'=1000, cfg: 'Optional[Dict[str, Any]]'=None) ->None:
        _layers: 'List[nn.Module]'
        if origin_stem:
            _layers = [*conv_sequence_pt(3, stem_channels, True, True, kernel_size=7, padding=3, stride=2), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        else:
            _layers = [*conv_sequence_pt(3, stem_channels // 2, True, True, kernel_size=3, padding=1), *conv_sequence_pt(stem_channels // 2, stem_channels, True, True, kernel_size=3, padding=1), nn.MaxPool2d(2)]
        in_chans = [stem_channels] + output_channels[:-1]
        for n_blocks, in_chan, out_chan, stride, conv, pool in zip(num_blocks, in_chans, output_channels, stage_stride, stage_conv, stage_pooling):
            _stage = resnet_stage(in_chan, out_chan, n_blocks, stride)
            if attn_module is not None:
                _stage.append(attn_module(out_chan))
            if conv:
                _stage.extend(conv_sequence_pt(out_chan, out_chan, True, True, kernel_size=3, padding=1))
            if pool is not None:
                _stage.append(nn.MaxPool2d(pool))
            _layers.append(nn.Sequential(*_stage))
        if include_top:
            _layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(output_channels[-1], num_classes, bias=True)])
        super().__init__(*_layers)
        self.cfg = cfg
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class FASTConvLayer(nn.Module):
    """Convolutional layer used in the TextNet and FAST architectures"""

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'Union[int, Tuple[int, int]]', stride: 'int'=1, dilation: 'int'=1, groups: 'int'=1, bias: 'bool'=False) ->None:
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.converted_ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.hor_conv, self.hor_bn = None, None
        self.ver_conv, self.ver_bn = None, None
        padding = int((self.converted_ks[0] - 1) * dilation / 2), int((self.converted_ks[1] - 1) * dilation / 2)
        self.activation = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.converted_ks, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.converted_ks[1] != 1:
            self.ver_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(self.converted_ks[0], 1), padding=(int((self.converted_ks[0] - 1) * dilation / 2), 0), stride=stride, dilation=dilation, groups=groups, bias=bias)
            self.ver_bn = nn.BatchNorm2d(out_channels)
        if self.converted_ks[0] != 1:
            self.hor_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, self.converted_ks[1]), padding=(0, int((self.converted_ks[1] - 1) * dilation / 2)), stride=stride, dilation=dilation, groups=groups, bias=bias)
            self.hor_bn = nn.BatchNorm2d(out_channels)
        self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if hasattr(self, 'fused_conv'):
            return self.activation(self.fused_conv(x))
        main_outputs = self.bn(self.conv(x))
        vertical_outputs = self.ver_bn(self.ver_conv(x)) if self.ver_conv is not None and self.ver_bn is not None else 0
        horizontal_outputs = self.hor_bn(self.hor_conv(x)) if self.hor_bn is not None and self.hor_conv is not None else 0
        id_out = self.rbr_identity(x) if self.rbr_identity is not None else 0
        return self.activation(main_outputs + vertical_outputs + horizontal_outputs + id_out)

    def _identity_to_conv(self, identity: 'Union[nn.BatchNorm2d, None]') ->Union[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
        if identity is None or identity.running_var is None:
            return 0, 0
        if not hasattr(self, 'id_tensor'):
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 0, 0] = 1
            id_tensor = torch.from_numpy(kernel_value)
            self.id_tensor = self._pad_to_mxn_tensor(id_tensor)
        kernel = self.id_tensor
        std = (identity.running_var + identity.eps).sqrt()
        t = (identity.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, identity.bias - identity.running_mean * identity.weight / std

    def _fuse_bn_tensor(self, conv: 'nn.Conv2d', bn: 'nn.BatchNorm2d') ->Tuple[torch.Tensor, torch.Tensor]:
        kernel = conv.weight
        kernel = self._pad_to_mxn_tensor(kernel)
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def _get_equivalent_kernel_bias(self) ->Tuple[torch.Tensor, torch.Tensor]:
        kernel_mxn, bias_mxn = self._fuse_bn_tensor(self.conv, self.bn)
        if self.ver_conv is not None:
            kernel_mx1, bias_mx1 = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        else:
            kernel_mx1, bias_mx1 = 0, 0
        if self.hor_conv is not None:
            kernel_1xn, bias_1xn = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        else:
            kernel_1xn, bias_1xn = 0, 0
        kernel_id, bias_id = self._identity_to_conv(self.rbr_identity)
        kernel_mxn = kernel_mxn + kernel_mx1 + kernel_1xn + kernel_id
        bias_mxn = bias_mxn + bias_mx1 + bias_1xn + bias_id
        return kernel_mxn, bias_mxn

    def _pad_to_mxn_tensor(self, kernel: 'torch.Tensor') ->torch.Tensor:
        kernel_height, kernel_width = self.converted_ks
        height, width = kernel.shape[2:]
        pad_left_right = (kernel_width - width) // 2
        pad_top_down = (kernel_height - height) // 2
        return torch.nn.functional.pad(kernel, [pad_left_right, pad_left_right, pad_top_down, pad_top_down], value=0)

    def reparameterize_layer(self):
        if hasattr(self, 'fused_conv'):
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        for attr in ['conv', 'bn', 'ver_conv', 'ver_bn', 'hor_conv', 'hor_bn']:
            if hasattr(self, attr):
                self.__delattr__(attr)
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')


class TextNet(nn.Sequential):
    """Implements TextNet architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    Args:
    ----
        stages (List[Dict[str, List[int]]]): List of dictionaries containing the parameters of each stage.
        include_top (bool, optional): Whether to include the classifier head. Defaults to True.
        num_classes (int, optional): Number of output classes. Defaults to 1000.
        cfg (Optional[Dict[str, Any]], optional): Additional configuration. Defaults to None.
    """

    def __init__(self, stages: 'List[Dict[str, List[int]]]', input_shape: 'Tuple[int, int, int]'=(3, 32, 32), num_classes: 'int'=1000, include_top: 'bool'=True, cfg: 'Optional[Dict[str, Any]]'=None) ->None:
        _layers: 'List[nn.Module]' = [*conv_sequence_pt(in_channels=3, out_channels=64, relu=True, bn=True, kernel_size=3, stride=2, padding=(1, 1)), *[nn.Sequential(*[FASTConvLayer(**params) for params in [{key: stage[key][i] for key in stage} for i in range(len(stage['in_channels']))]]) for stage in stages]]
        if include_top:
            _layers.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(stages[-1]['out_channels'][-1], num_classes)))
        super().__init__(*_layers)
        self.cfg = cfg
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ClassifierHead(nn.Module):
    """Classifier head for Vision Transformer

    Args:
    ----
        in_channels: number of input channels
        num_classes: number of output classes
    """

    def __init__(self, in_channels: 'int', num_classes: 'int') ->None:
        super().__init__()
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.head(x[:, 0])


def scaled_dot_product_attention(query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
    """Scaled Dot-Product Attention"""
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = torch.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(self, num_heads: 'int', d_model: 'int', dropout: 'float'=0.1) ->None:
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask=None) ->torch.Tensor:
        batch_size = query.size(0)
        query, key, value = [linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) for linear, x in zip(self.linear_layers, (query, key, value))]
        x, attn = scaled_dot_product_attention(query, key, value, mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.output_linear(x)


class PositionwiseFeedForward(nn.Sequential):
    """Position-wise Feed-Forward Network"""

    def __init__(self, d_model: 'int', ffd: 'int', dropout: 'float'=0.1, activation_fct: 'Callable[[Any], Any]'=nn.ReLU()) ->None:
        super().__init__(nn.Linear(d_model, ffd), activation_fct, nn.Dropout(p=dropout), nn.Linear(ffd, d_model), nn.Dropout(p=dropout))


class EncoderBlock(nn.Module):
    """Transformer Encoder Block"""

    def __init__(self, num_layers: 'int', num_heads: 'int', d_model: 'int', dff: 'int', dropout: 'float', activation_fct: 'Callable[[Any], Any]'=nn.ReLU()) ->None:
        super().__init__()
        self.num_layers = num_layers
        self.layer_norm_input = nn.LayerNorm(d_model, eps=1e-05)
        self.layer_norm_attention = nn.LayerNorm(d_model, eps=1e-05)
        self.layer_norm_output = nn.LayerNorm(d_model, eps=1e-05)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.ModuleList([MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)])
        self.position_feed_forward = nn.ModuleList([PositionwiseFeedForward(d_model, dff, dropout, activation_fct) for _ in range(self.num_layers)])

    def forward(self, x: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        output = x
        for i in range(self.num_layers):
            normed_output = self.layer_norm_input(output)
            output = output + self.dropout(self.attention[i](normed_output, normed_output, normed_output, mask))
            normed_output = self.layer_norm_attention(output)
            output = output + self.dropout(self.position_feed_forward[i](normed_output))
        return self.layer_norm_output(output)


class PatchEmbedding(nn.Module):
    """Compute 2D patch embeddings with cls token and positional encoding"""

    def __init__(self, input_shape: 'Tuple[int, int, int]', embed_dim: 'int', patch_size: 'Tuple[int, int]') ->None:
        super().__init__()
        channels, height, width = input_shape
        self.patch_size = patch_size
        self.interpolate = True if patch_size[0] == patch_size[1] else False
        self.grid_size = tuple(s // p for s, p in zip((height, width), self.patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.projection = nn.Conv2d(channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def interpolate_pos_encoding(self, embeddings: 'torch.Tensor', height: 'int', width: 'int') ->torch.Tensor:
        """100 % borrowed from:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py

        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py
        """
        num_patches = embeddings.shape[1] - 1
        num_positions = self.positions.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.positions
        class_pos_embed = self.positions[:, 0]
        patch_pos_embed = self.positions[:, 1:]
        dim = embeddings.shape[-1]
        h0 = float(height // self.patch_size[0])
        w0 = float(width // self.patch_size[1])
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)), mode='bilinear', align_corners=False, recompute_scale_factor=True)
        assert int(h0) == patch_pos_embed.shape[-2], "height of interpolated patch embedding doesn't match"
        assert int(w0) == patch_pos_embed.shape[-1], "width of interpolated patch embedding doesn't match"
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, 'Image height must be divisible by patch height'
        assert W % self.patch_size[1] == 0, 'Image width must be divisible by patch width'
        patches = self.projection(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat([cls_tokens, patches], dim=1)
        if self.interpolate:
            embeddings += self.interpolate_pos_encoding(embeddings, H, W)
        else:
            embeddings += self.positions
        return embeddings


class VisionTransformer(nn.Sequential):
    """VisionTransformer architecture as described in
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_.

    Args:
    ----
        d_model: dimension of the transformer layers
        num_layers: number of transformer layers
        num_heads: number of attention heads
        ffd_ratio: multiplier for the hidden dimension of the feedforward layer
        patch_size: size of the patches
        input_shape: size of the input image
        dropout: dropout rate
        num_classes: number of output classes
        include_top: whether the classifier head should be instantiated
    """

    def __init__(self, d_model: 'int', num_layers: 'int', num_heads: 'int', ffd_ratio: 'int', patch_size: 'Tuple[int, int]'=(4, 4), input_shape: 'Tuple[int, int, int]'=(3, 32, 32), dropout: 'float'=0.0, num_classes: 'int'=1000, include_top: 'bool'=True, cfg: 'Optional[Dict[str, Any]]'=None) ->None:
        _layers: 'List[nn.Module]' = [PatchEmbedding(input_shape, d_model, patch_size), EncoderBlock(num_layers, num_heads, d_model, d_model * ffd_ratio, dropout, nn.GELU())]
        if include_top:
            _layers.append(ClassifierHead(d_model, num_classes))
        super().__init__(*_layers)
        self.cfg = cfg


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, in_channels: 'List[int]', out_channels: 'int', deform_conv: 'bool'=False) ->None:
        super().__init__()
        out_chans = out_channels // len(in_channels)
        conv_layer = DeformConv2d if deform_conv else nn.Conv2d
        self.in_branches = nn.ModuleList([nn.Sequential(conv_layer(chans, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)) for idx, chans in enumerate(in_channels)])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_branches = nn.ModuleList([nn.Sequential(conv_layer(out_channels, out_chans, 3, padding=1, bias=False), nn.BatchNorm2d(out_chans), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2 ** idx, mode='bilinear', align_corners=True)) for idx, chans in enumerate(in_channels)])

    def forward(self, x: 'List[torch.Tensor]') ->torch.Tensor:
        if len(x) != len(self.out_branches):
            raise AssertionError
        _x: 'List[torch.Tensor]' = [branch(t) for branch, t in zip(self.in_branches, x)]
        out: 'List[torch.Tensor]' = [_x[-1]]
        for t in _x[:-1][::-1]:
            out.append(self.upsample(out[-1]) + t)
        out = [branch(t) for branch, t in zip(self.out_branches, out[::-1])]
        return torch.cat(out, dim=1)


class DetectionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
    ----
        box_thresh (float): minimal objectness score to consider a box
        bin_thresh (float): threshold to apply to segmentation raw heatmap
        assume straight_pages (bool): if True, fit straight boxes only
    """

    def __init__(self, box_thresh: 'float'=0.5, bin_thresh: 'float'=0.5, assume_straight_pages: 'bool'=True) ->None:
        self.box_thresh = box_thresh
        self.bin_thresh = bin_thresh
        self.assume_straight_pages = assume_straight_pages
        self._opening_kernel: 'np.ndarray' = np.ones((3, 3), dtype=np.uint8)

    def extra_repr(self) ->str:
        return f'bin_thresh={self.bin_thresh}, box_thresh={self.box_thresh}'

    @staticmethod
    def box_score(pred: 'np.ndarray', points: 'np.ndarray', assume_straight_pages: 'bool'=True) ->float:
        """Compute the confidence score for a polygon : mean of the p values on the polygon

        Args:
        ----
            pred (np.ndarray): p map returned by the model
            points: coordinates of the polygon
            assume_straight_pages: if True, fit straight boxes only

        Returns:
        -------
            polygon objectness
        """
        h, w = pred.shape[:2]
        if assume_straight_pages:
            xmin = np.clip(np.floor(points[:, 0].min()).astype(np.int32), 0, w - 1)
            xmax = np.clip(np.ceil(points[:, 0].max()).astype(np.int32), 0, w - 1)
            ymin = np.clip(np.floor(points[:, 1].min()).astype(np.int32), 0, h - 1)
            ymax = np.clip(np.ceil(points[:, 1].max()).astype(np.int32), 0, h - 1)
            return pred[ymin:ymax + 1, xmin:xmax + 1].mean()
        else:
            mask: 'np.ndarray' = np.zeros((h, w), np.int32)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1.0)
            product = pred * mask
            return np.sum(product) / np.count_nonzero(product)

    def bitmap_to_boxes(self, pred: 'np.ndarray', bitmap: 'np.ndarray') ->np.ndarray:
        raise NotImplementedError

    def __call__(self, proba_map) ->List[List[np.ndarray]]:
        """Performs postprocessing for a list of model outputs

        Args:
        ----
            proba_map: probability map of shape (N, H, W, C)

        Returns:
        -------
            list of N class predictions (for each input sample), where each class predictions is a list of C tensors
        of shape (*, 5) or (*, 6)
        """
        if proba_map.ndim != 4:
            raise AssertionError(f'arg `proba_map` is expected to be 4-dimensional, got {proba_map.ndim}.')
        bin_map = [[cv2.morphologyEx(bmap[..., idx], cv2.MORPH_OPEN, self._opening_kernel) for idx in range(proba_map.shape[-1])] for bmap in (proba_map >= self.bin_thresh).astype(np.uint8)]
        return [[self.bitmap_to_boxes(pmaps[..., idx], bmaps[idx]) for idx in range(proba_map.shape[-1])] for pmaps, bmaps in zip(proba_map, bin_map)]


Point2D = Tuple[float, float]


Polygon = List[Point2D]


class DBPostProcessor(DetectionPostProcessor):
    """Implements a post processor for DBNet adapted from the implementation of `xuannianz
    <https://github.com/xuannianz/DifferentiableBinarization>`_.

    Args:
    ----
        unclip ratio: ratio used to unshrink polygons
        min_size_box: minimal length (pix) to keep a box
        max_candidates: maximum boxes to consider in a single page
        box_thresh: minimal objectness score to consider a box
        bin_thresh: threshold used to binzarized p_map at inference time

    """

    def __init__(self, box_thresh: 'float'=0.1, bin_thresh: 'float'=0.3, assume_straight_pages: 'bool'=True) ->None:
        super().__init__(box_thresh, bin_thresh, assume_straight_pages)
        self.unclip_ratio = 1.5

    def polygon_to_box(self, points: 'np.ndarray') ->np.ndarray:
        """Expand a polygon (points) by a factor unclip_ratio, and returns a polygon

        Args:
        ----
            points: The first parameter.

        Returns:
        -------
            a box in absolute coordinates (xmin, ymin, xmax, ymax) or (4, 2) array (quadrangle)
        """
        if not self.assume_straight_pages:
            rect = cv2.minAreaRect(points)
            points = cv2.boxPoints(rect)
            area = (rect[1][0] + 1) * (1 + rect[1][1])
            length = 2 * (rect[1][0] + rect[1][1]) + 2
        else:
            poly = Polygon(points)
            area = poly.area
            length = poly.length
        distance = area * self.unclip_ratio / length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        _points = offset.Execute(distance)
        idx = 0
        if len(_points) > 1:
            max_size = 0
            for _idx, p in enumerate(_points):
                if len(p) > max_size:
                    idx = _idx
                    max_size = len(p)
            _points = [_points[idx]]
        expanded_points: 'np.ndarray' = np.asarray(_points)
        if len(expanded_points) < 1:
            return None
        return cv2.boundingRect(expanded_points) if self.assume_straight_pages else np.roll(cv2.boxPoints(cv2.minAreaRect(expanded_points)), -1, axis=0)

    def bitmap_to_boxes(self, pred: 'np.ndarray', bitmap: 'np.ndarray') ->np.ndarray:
        """Compute boxes from a bitmap/pred_map: find connected components then filter boxes

        Args:
        ----
            pred: Pred map from differentiable binarization output
            bitmap: Bitmap map computed from pred (binarized)
            angle_tol: Comparison tolerance of the angle with the median angle across the page
            ratio_tol: Under this limit aspect ratio, we cannot resolve the direction of the crop

        Returns:
        -------
            np tensor boxes for the bitmap, each box is a 5-element list
                containing x, y, w, h, score for the box
        """
        height, width = bitmap.shape[:2]
        min_size_box = 2
        boxes: 'List[Union[np.ndarray, List[float]]]' = []
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if np.any(contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0) < min_size_box):
                continue
            if self.assume_straight_pages:
                x, y, w, h = cv2.boundingRect(contour)
                points: 'np.ndarray' = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
                score = self.box_score(pred, points, assume_straight_pages=True)
            else:
                score = self.box_score(pred, contour, assume_straight_pages=False)
            if score < self.box_thresh:
                continue
            if self.assume_straight_pages:
                _box = self.polygon_to_box(points)
            else:
                _box = self.polygon_to_box(np.squeeze(contour))
            if self.assume_straight_pages:
                if _box is None or _box[2] < min_size_box or _box[3] < min_size_box:
                    continue
            elif np.linalg.norm(_box[2, :] - _box[0, :], axis=-1) < min_size_box:
                continue
            if self.assume_straight_pages:
                x, y, w, h = _box
                xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
                boxes.append([xmin, ymin, xmax, ymax, score])
            else:
                if not isinstance(_box, np.ndarray) and _box.shape == (4, 2):
                    raise AssertionError('When assume straight pages is false a box is a (4, 2) array (polygon)')
                _box[:, 0] /= width
                _box[:, 1] /= height
                boxes.append(np.vstack([_box, np.array([0.0, score])]))
        if not self.assume_straight_pages:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5, 2), dtype=pred.dtype)
        else:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5), dtype=pred.dtype)


class _DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_.

    Args:
    ----
        feature extractor: the backbone serving as feature extractor
        fpn_channels: number of channels each extracted feature maps is mapped to
    """
    shrink_ratio = 0.4
    thresh_min = 0.3
    thresh_max = 0.7
    min_size_box = 3
    assume_straight_pages: 'bool' = True

    @staticmethod
    def compute_distance(xs: 'np.ndarray', ys: 'np.ndarray', a: 'np.ndarray', b: 'np.ndarray', eps: 'float'=1e-06) ->float:
        """Compute the distance for each point of the map (xs, ys) to the (a, b) segment

        Args:
        ----
            xs : map of x coordinates (height, width)
            ys : map of y coordinates (height, width)
            a: first point defining the [ab] segment
            b: second point defining the [ab] segment
            eps: epsilon to avoid division by zero

        Returns:
        -------
            The computed distance

        """
        square_dist_1 = np.square(xs - a[0]) + np.square(ys - a[1])
        square_dist_2 = np.square(xs - b[0]) + np.square(ys - b[1])
        square_dist = np.square(a[0] - b[0]) + np.square(a[1] - b[1])
        cosin = (square_dist - square_dist_1 - square_dist_2) / (2 * np.sqrt(square_dist_1 * square_dist_2) + eps)
        cosin = np.clip(cosin, -1.0, 1.0)
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_dist_1 * square_dist_2 * square_sin / square_dist + eps)
        result[cosin < 0] = np.sqrt(np.fmin(square_dist_1, square_dist_2))[cosin < 0]
        return result

    def draw_thresh_map(self, polygon: 'np.ndarray', canvas: 'np.ndarray', mask: 'np.ndarray') ->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw a polygon treshold map on a canvas, as described in the DB paper

        Args:
        ----
            polygon : array of coord., to draw the boundary of the polygon
            canvas : threshold map to fill with polygons
            mask : mask for training on threshold polygons
        """
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise AttributeError('polygon should be a 2 dimensional array of coords')
        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(coor) for coor in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon: 'np.ndarray' = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)
        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin
        xs: 'np.ndarray' = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys: 'np.ndarray' = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))
        distance_map = np.zeros((polygon.shape[0], height, width), dtype=polygon.dtype)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.compute_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = np.min(distance_map, axis=0)
        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(1 - distance_map[ymin_valid - ymin:ymax_valid - ymax + height, xmin_valid - xmin:xmax_valid - xmax + width], canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])
        return polygon, canvas, mask

    def build_target(self, target: 'List[Dict[str, np.ndarray]]', output_shape: 'Tuple[int, int, int]', channels_last: 'bool'=True) ->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if any(t.dtype != np.float32 for tgt in target for t in tgt.values()):
            raise AssertionError("the expected dtype of target 'boxes' entry is 'np.float32'.")
        if any(np.any((t[:, :4] > 1) | (t[:, :4] < 0)) for tgt in target for t in tgt.values()):
            raise ValueError("the 'boxes' entry of the target is expected to take values between 0 & 1.")
        input_dtype = next(iter(target[0].values())).dtype if len(target) > 0 else np.float32
        h: 'int'
        w: 'int'
        if channels_last:
            h, w, num_classes = output_shape
        else:
            num_classes, h, w = output_shape
        target_shape = len(target), num_classes, h, w
        seg_target: 'np.ndarray' = np.zeros(target_shape, dtype=np.uint8)
        seg_mask: 'np.ndarray' = np.ones(target_shape, dtype=bool)
        thresh_target: 'np.ndarray' = np.zeros(target_shape, dtype=np.float32)
        thresh_mask: 'np.ndarray' = np.zeros(target_shape, dtype=np.uint8)
        for idx, tgt in enumerate(target):
            for class_idx, _tgt in enumerate(tgt.values()):
                if _tgt.shape[0] == 0:
                    seg_mask[idx, class_idx] = False
                abs_boxes = _tgt.copy()
                if abs_boxes.ndim == 3:
                    abs_boxes[:, :, 0] *= w
                    abs_boxes[:, :, 1] *= h
                    polys = abs_boxes
                    boxes_size = np.linalg.norm(abs_boxes[:, 2, :] - abs_boxes[:, 0, :], axis=-1)
                    abs_boxes = np.concatenate((abs_boxes.min(1), abs_boxes.max(1)), -1).round().astype(np.int32)
                else:
                    abs_boxes[:, [0, 2]] *= w
                    abs_boxes[:, [1, 3]] *= h
                    abs_boxes = abs_boxes.round().astype(np.int32)
                    polys = np.stack([abs_boxes[:, [0, 1]], abs_boxes[:, [0, 3]], abs_boxes[:, [2, 3]], abs_boxes[:, [2, 1]]], axis=1)
                    boxes_size = np.minimum(abs_boxes[:, 2] - abs_boxes[:, 0], abs_boxes[:, 3] - abs_boxes[:, 1])
                for poly, box, box_size in zip(polys, abs_boxes, boxes_size):
                    if box_size < self.min_size_box:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    polygon = Polygon(poly)
                    distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                    subject = [tuple(coor) for coor in poly]
                    padding = pyclipper.PyclipperOffset()
                    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                    shrunken = padding.Execute(-distance)
                    if len(shrunken) == 0:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    shrunken = np.array(shrunken[0]).reshape(-1, 2)
                    if shrunken.shape[0] <= 2 or not Polygon(shrunken).is_valid:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    cv2.fillPoly(seg_target[idx, class_idx], [shrunken.astype(np.int32)], 1.0)
                    poly, thresh_target[idx, class_idx], thresh_mask[idx, class_idx] = self.draw_thresh_map(poly, thresh_target[idx, class_idx], thresh_mask[idx, class_idx])
        if channels_last:
            seg_target = seg_target.transpose((0, 2, 3, 1))
            seg_mask = seg_mask.transpose((0, 2, 3, 1))
            thresh_target = thresh_target.transpose((0, 2, 3, 1))
            thresh_mask = thresh_mask.transpose((0, 2, 3, 1))
        thresh_target = thresh_target.astype(input_dtype) * (self.thresh_max - self.thresh_min) + self.thresh_min
        seg_target = seg_target.astype(input_dtype)
        seg_mask = seg_mask.astype(bool)
        thresh_target = thresh_target.astype(input_dtype)
        thresh_mask = thresh_mask.astype(bool)
        return seg_target, seg_mask, thresh_target, thresh_mask


def _bf16_to_float32(x: 'torch.Tensor') ->torch.Tensor:
    return x.float() if x.dtype == torch.bfloat16 else x


class FastNeck(nn.Module):
    """Neck of the FAST architecture, composed of a series of 3x3 convolutions and upsampling layers.

    Args:
    ----
        in_channels: number of input channels
        out_channels: number of output channels
    """

    def __init__(self, in_channels: 'int', out_channels: 'int'=128) ->None:
        super().__init__()
        self.reduction = nn.ModuleList([FASTConvLayer(in_channels * scale, out_channels, kernel_size=3) for scale in [1, 2, 4, 8]])

    def _upsample(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        return F.interpolate(x, size=y.shape[-2:], mode='bilinear')

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        f1, f2, f3, f4 = x
        f1, f2, f3, f4 = [reduction(f) for reduction, f in zip(self.reduction, (f1, f2, f3, f4))]
        f2, f3, f4 = [self._upsample(f, f1) for f in (f2, f3, f4)]
        f = torch.cat((f1, f2, f3, f4), 1)
        return f


class FastHead(nn.Sequential):
    """Head of the FAST architecture

    Args:
    ----
        in_channels: number of input channels
        num_classes: number of output classes
        out_channels: number of output channels
        dropout: dropout probability
    """

    def __init__(self, in_channels: 'int', num_classes: 'int', out_channels: 'int'=128, dropout: 'float'=0.1) ->None:
        _layers: 'List[nn.Module]' = [FASTConvLayer(in_channels, out_channels, kernel_size=3), nn.Dropout(dropout), nn.Conv2d(out_channels, num_classes, kernel_size=1, bias=False)]
        super().__init__(*_layers)


class FASTPostProcessor(DetectionPostProcessor):
    """Implements a post processor for FAST model.

    Args:
    ----
        bin_thresh: threshold used to binzarized p_map at inference time
        box_thresh: minimal objectness score to consider a box
        assume_straight_pages: whether the inputs were expected to have horizontal text elements
    """

    def __init__(self, bin_thresh: 'float'=0.1, box_thresh: 'float'=0.1, assume_straight_pages: 'bool'=True) ->None:
        super().__init__(box_thresh, bin_thresh, assume_straight_pages)
        self.unclip_ratio = 1.0

    def polygon_to_box(self, points: 'np.ndarray') ->np.ndarray:
        """Expand a polygon (points) by a factor unclip_ratio, and returns a polygon

        Args:
        ----
            points: The first parameter.

        Returns:
        -------
            a box in absolute coordinates (xmin, ymin, xmax, ymax) or (4, 2) array (quadrangle)
        """
        if not self.assume_straight_pages:
            rect = cv2.minAreaRect(points)
            points = cv2.boxPoints(rect)
            area = (rect[1][0] + 1) * (1 + rect[1][1])
            length = 2 * (rect[1][0] + rect[1][1]) + 2
        else:
            poly = Polygon(points)
            area = poly.area
            length = poly.length
        distance = area * self.unclip_ratio / length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        _points = offset.Execute(distance)
        idx = 0
        if len(_points) > 1:
            max_size = 0
            for _idx, p in enumerate(_points):
                if len(p) > max_size:
                    idx = _idx
                    max_size = len(p)
            _points = [_points[idx]]
        expanded_points: 'np.ndarray' = np.asarray(_points)
        if len(expanded_points) < 1:
            return None
        return cv2.boundingRect(expanded_points) if self.assume_straight_pages else np.roll(cv2.boxPoints(cv2.minAreaRect(expanded_points)), -1, axis=0)

    def bitmap_to_boxes(self, pred: 'np.ndarray', bitmap: 'np.ndarray') ->np.ndarray:
        """Compute boxes from a bitmap/pred_map: find connected components then filter boxes

        Args:
        ----
            pred: Pred map from differentiable linknet output
            bitmap: Bitmap map computed from pred (binarized)
            angle_tol: Comparison tolerance of the angle with the median angle across the page
            ratio_tol: Under this limit aspect ratio, we cannot resolve the direction of the crop

        Returns:
        -------
            np tensor boxes for the bitmap, each box is a 6-element list
                containing x, y, w, h, alpha, score for the box
        """
        height, width = bitmap.shape[:2]
        boxes: 'List[Union[np.ndarray, List[float]]]' = []
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if np.any(contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0) < 2):
                continue
            if self.assume_straight_pages:
                x, y, w, h = cv2.boundingRect(contour)
                points: 'np.ndarray' = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
                score = self.box_score(pred, points, assume_straight_pages=True)
            else:
                score = self.box_score(pred, contour, assume_straight_pages=False)
            if score < self.box_thresh:
                continue
            if self.assume_straight_pages:
                _box = self.polygon_to_box(points)
            else:
                _box = self.polygon_to_box(np.squeeze(contour))
            if self.assume_straight_pages:
                x, y, w, h = _box
                xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
                boxes.append([xmin, ymin, xmax, ymax, score])
            else:
                _box[:, 0] /= width
                _box[:, 1] /= height
                boxes.append(np.vstack([_box, np.array([0.0, score])]))
        if not self.assume_straight_pages:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5, 2), dtype=pred.dtype)
        else:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5), dtype=pred.dtype)


class BaseModel(NestedObject):
    """Implements abstract DetectionModel class"""

    def __init__(self, cfg: 'Optional[Dict[str, Any]]'=None) ->None:
        super().__init__()
        self.cfg = cfg


class _FAST(BaseModel):
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_.
    """
    min_size_box: 'int' = 3
    assume_straight_pages: 'bool' = True
    shrink_ratio = 0.4

    def build_target(self, target: 'List[Dict[str, np.ndarray]]', output_shape: 'Tuple[int, int, int]', channels_last: 'bool'=True) ->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the target, and it's mask to be used from loss computation.

        Args:
        ----
            target: target coming from dataset
            output_shape: shape of the output of the model without batch_size
            channels_last: whether channels are last or not

        Returns:
        -------
            the new formatted target, mask and shrunken text kernel
        """
        if any(t.dtype != np.float32 for tgt in target for t in tgt.values()):
            raise AssertionError("the expected dtype of target 'boxes' entry is 'np.float32'.")
        if any(np.any((t[:, :4] > 1) | (t[:, :4] < 0)) for tgt in target for t in tgt.values()):
            raise ValueError("the 'boxes' entry of the target is expected to take values between 0 & 1.")
        h: 'int'
        w: 'int'
        if channels_last:
            h, w, num_classes = output_shape
        else:
            num_classes, h, w = output_shape
        target_shape = len(target), num_classes, h, w
        seg_target: 'np.ndarray' = np.zeros(target_shape, dtype=np.uint8)
        seg_mask: 'np.ndarray' = np.ones(target_shape, dtype=bool)
        shrunken_kernel: 'np.ndarray' = np.zeros(target_shape, dtype=np.uint8)
        for idx, tgt in enumerate(target):
            for class_idx, _tgt in enumerate(tgt.values()):
                if _tgt.shape[0] == 0:
                    seg_mask[idx, class_idx] = False
                abs_boxes = _tgt.copy()
                if abs_boxes.ndim == 3:
                    abs_boxes[:, :, 0] *= w
                    abs_boxes[:, :, 1] *= h
                    polys = abs_boxes
                    boxes_size = np.linalg.norm(abs_boxes[:, 2, :] - abs_boxes[:, 0, :], axis=-1)
                    abs_boxes = np.concatenate((abs_boxes.min(1), abs_boxes.max(1)), -1).round().astype(np.int32)
                else:
                    abs_boxes[:, [0, 2]] *= w
                    abs_boxes[:, [1, 3]] *= h
                    abs_boxes = abs_boxes.round().astype(np.int32)
                    polys = np.stack([abs_boxes[:, [0, 1]], abs_boxes[:, [0, 3]], abs_boxes[:, [2, 3]], abs_boxes[:, [2, 1]]], axis=1)
                    boxes_size = np.minimum(abs_boxes[:, 2] - abs_boxes[:, 0], abs_boxes[:, 3] - abs_boxes[:, 1])
                for poly, box, box_size in zip(polys, abs_boxes, boxes_size):
                    if box_size < self.min_size_box:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    polygon = Polygon(poly)
                    distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                    subject = [tuple(coor) for coor in poly]
                    padding = pyclipper.PyclipperOffset()
                    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                    shrunken = padding.Execute(-distance)
                    if len(shrunken) == 0:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    shrunken = np.array(shrunken[0]).reshape(-1, 2)
                    if shrunken.shape[0] <= 2 or not Polygon(shrunken).is_valid:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    cv2.fillPoly(shrunken_kernel[idx, class_idx], [shrunken.astype(np.int32)], 1.0)
                    cv2.fillPoly(seg_target[idx, class_idx], [poly.astype(np.int32)], 1.0)
        if channels_last:
            seg_target = seg_target.transpose((0, 2, 3, 1))
            seg_mask = seg_mask.transpose((0, 2, 3, 1))
            shrunken_kernel = shrunken_kernel.transpose((0, 2, 3, 1))
        return seg_target, seg_mask, shrunken_kernel


class LinkNetFPN(nn.Module):

    def __init__(self, layer_shapes: 'List[Tuple[int, int, int]]') ->None:
        super().__init__()
        strides = [(1 if in_shape[-1] == out_shape[-1] else 2) for in_shape, out_shape in zip(layer_shapes[:-1], layer_shapes[1:])]
        chans = [shape[0] for shape in layer_shapes]
        _decoder_layers = [self.decoder_block(ochan, ichan, stride) for ichan, ochan, stride in zip(chans[:-1], chans[1:], strides)]
        self.decoders = nn.ModuleList(_decoder_layers)

    @staticmethod
    def decoder_block(in_chan: 'int', out_chan: 'int', stride: 'int') ->nn.Sequential:
        """Creates a LinkNet decoder block"""
        mid_chan = in_chan // 4
        return nn.Sequential(nn.Conv2d(in_chan, mid_chan, kernel_size=1, bias=False), nn.BatchNorm2d(mid_chan), nn.ReLU(inplace=True), nn.ConvTranspose2d(mid_chan, mid_chan, 3, padding=1, output_padding=stride - 1, stride=stride, bias=False), nn.BatchNorm2d(mid_chan), nn.ReLU(inplace=True), nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=False), nn.BatchNorm2d(out_chan), nn.ReLU(inplace=True))

    def forward(self, feats: 'List[torch.Tensor]') ->torch.Tensor:
        out = feats[-1]
        for decoder, fmap in zip(self.decoders[::-1], feats[:-1][::-1]):
            out = decoder(out) + fmap
        out = self.decoders[0](out)
        return out


class LinkNetPostProcessor(DetectionPostProcessor):
    """Implements a post processor for LinkNet model.

    Args:
    ----
        bin_thresh: threshold used to binzarized p_map at inference time
        box_thresh: minimal objectness score to consider a box
        assume_straight_pages: whether the inputs were expected to have horizontal text elements
    """

    def __init__(self, bin_thresh: 'float'=0.1, box_thresh: 'float'=0.1, assume_straight_pages: 'bool'=True) ->None:
        super().__init__(box_thresh, bin_thresh, assume_straight_pages)
        self.unclip_ratio = 1.5

    def polygon_to_box(self, points: 'np.ndarray') ->np.ndarray:
        """Expand a polygon (points) by a factor unclip_ratio, and returns a polygon

        Args:
        ----
            points: The first parameter.

        Returns:
        -------
            a box in absolute coordinates (xmin, ymin, xmax, ymax) or (4, 2) array (quadrangle)
        """
        if not self.assume_straight_pages:
            rect = cv2.minAreaRect(points)
            points = cv2.boxPoints(rect)
            area = (rect[1][0] + 1) * (1 + rect[1][1])
            length = 2 * (rect[1][0] + rect[1][1]) + 2
        else:
            poly = Polygon(points)
            area = poly.area
            length = poly.length
        distance = area * self.unclip_ratio / length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        _points = offset.Execute(distance)
        idx = 0
        if len(_points) > 1:
            max_size = 0
            for _idx, p in enumerate(_points):
                if len(p) > max_size:
                    idx = _idx
                    max_size = len(p)
            _points = [_points[idx]]
        expanded_points: 'np.ndarray' = np.asarray(_points)
        if len(expanded_points) < 1:
            return None
        return cv2.boundingRect(expanded_points) if self.assume_straight_pages else np.roll(cv2.boxPoints(cv2.minAreaRect(expanded_points)), -1, axis=0)

    def bitmap_to_boxes(self, pred: 'np.ndarray', bitmap: 'np.ndarray') ->np.ndarray:
        """Compute boxes from a bitmap/pred_map: find connected components then filter boxes

        Args:
        ----
            pred: Pred map from differentiable linknet output
            bitmap: Bitmap map computed from pred (binarized)
            angle_tol: Comparison tolerance of the angle with the median angle across the page
            ratio_tol: Under this limit aspect ratio, we cannot resolve the direction of the crop

        Returns:
        -------
            np tensor boxes for the bitmap, each box is a 6-element list
                containing x, y, w, h, alpha, score for the box
        """
        height, width = bitmap.shape[:2]
        boxes: 'List[Union[np.ndarray, List[float]]]' = []
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if np.any(contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0) < 2):
                continue
            if self.assume_straight_pages:
                x, y, w, h = cv2.boundingRect(contour)
                points: 'np.ndarray' = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
                score = self.box_score(pred, points, assume_straight_pages=True)
            else:
                score = self.box_score(pred, contour, assume_straight_pages=False)
            if score < self.box_thresh:
                continue
            if self.assume_straight_pages:
                _box = self.polygon_to_box(points)
            else:
                _box = self.polygon_to_box(np.squeeze(contour))
            if self.assume_straight_pages:
                x, y, w, h = _box
                xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
                boxes.append([xmin, ymin, xmax, ymax, score])
            else:
                _box[:, 0] /= width
                _box[:, 1] /= height
                boxes.append(np.vstack([_box, np.array([0.0, score])]))
        if not self.assume_straight_pages:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5, 2), dtype=pred.dtype)
        else:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5), dtype=pred.dtype)


class _LinkNet(BaseModel):
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    Args:
    ----
        out_chan: number of channels for the output
    """
    min_size_box: 'int' = 3
    assume_straight_pages: 'bool' = True
    shrink_ratio = 0.5

    def build_target(self, target: 'List[Dict[str, np.ndarray]]', output_shape: 'Tuple[int, int, int]', channels_last: 'bool'=True) ->Tuple[np.ndarray, np.ndarray]:
        """Build the target, and it's mask to be used from loss computation.

        Args:
        ----
            target: target coming from dataset
            output_shape: shape of the output of the model without batch_size
            channels_last: whether channels are last or not

        Returns:
        -------
            the new formatted target and the mask
        """
        if any(t.dtype != np.float32 for tgt in target for t in tgt.values()):
            raise AssertionError("the expected dtype of target 'boxes' entry is 'np.float32'.")
        if any(np.any((t[:, :4] > 1) | (t[:, :4] < 0)) for tgt in target for t in tgt.values()):
            raise ValueError("the 'boxes' entry of the target is expected to take values between 0 & 1.")
        h: 'int'
        w: 'int'
        if channels_last:
            h, w, num_classes = output_shape
        else:
            num_classes, h, w = output_shape
        target_shape = len(target), num_classes, h, w
        seg_target: 'np.ndarray' = np.zeros(target_shape, dtype=np.uint8)
        seg_mask: 'np.ndarray' = np.ones(target_shape, dtype=bool)
        for idx, tgt in enumerate(target):
            for class_idx, _tgt in enumerate(tgt.values()):
                if _tgt.shape[0] == 0:
                    seg_mask[idx, class_idx] = False
                abs_boxes = _tgt.copy()
                if abs_boxes.ndim == 3:
                    abs_boxes[:, :, 0] *= w
                    abs_boxes[:, :, 1] *= h
                    polys = abs_boxes
                    boxes_size = np.linalg.norm(abs_boxes[:, 2, :] - abs_boxes[:, 0, :], axis=-1)
                    abs_boxes = np.concatenate((abs_boxes.min(1), abs_boxes.max(1)), -1).round().astype(np.int32)
                else:
                    abs_boxes[:, [0, 2]] *= w
                    abs_boxes[:, [1, 3]] *= h
                    abs_boxes = abs_boxes.round().astype(np.int32)
                    polys = np.stack([abs_boxes[:, [0, 1]], abs_boxes[:, [0, 3]], abs_boxes[:, [2, 3]], abs_boxes[:, [2, 1]]], axis=1)
                    boxes_size = np.minimum(abs_boxes[:, 2] - abs_boxes[:, 0], abs_boxes[:, 3] - abs_boxes[:, 1])
                for poly, box, box_size in zip(polys, abs_boxes, boxes_size):
                    if box_size < self.min_size_box:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    polygon = Polygon(poly)
                    distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                    subject = [tuple(coor) for coor in poly]
                    padding = pyclipper.PyclipperOffset()
                    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                    shrunken = padding.Execute(-distance)
                    if len(shrunken) == 0:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    shrunken = np.array(shrunken[0]).reshape(-1, 2)
                    if shrunken.shape[0] <= 2 or not Polygon(shrunken).is_valid:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    cv2.fillPoly(seg_target[idx, class_idx], [shrunken.astype(np.int32)], 1.0)
        if channels_last:
            seg_target = seg_target.transpose((0, 2, 3, 1))
            seg_mask = seg_mask.transpose((0, 2, 3, 1))
        return seg_target, seg_mask


def _remove_padding(pages: 'List[np.ndarray]', loc_preds: 'List[Dict[str, np.ndarray]]', preserve_aspect_ratio: 'bool', symmetric_pad: 'bool', assume_straight_pages: 'bool') ->List[Dict[str, np.ndarray]]:
    """Remove padding from the localization predictions

    Args:
    ----
        pages: list of pages
        loc_preds: list of localization predictions
        preserve_aspect_ratio: whether the aspect ratio was preserved during padding
        symmetric_pad: whether the padding was symmetric
        assume_straight_pages: whether the pages are assumed to be straight

    Returns:
    -------
        list of unpaded localization predictions
    """
    if preserve_aspect_ratio:
        rectified_preds = []
        for page, dict_loc_preds in zip(pages, loc_preds):
            for k, loc_pred in dict_loc_preds.items():
                h, w = page.shape[0], page.shape[1]
                if h > w:
                    if symmetric_pad:
                        if assume_straight_pages:
                            loc_pred[:, [0, 2]] = (loc_pred[:, [0, 2]] - 0.5) * h / w + 0.5
                        else:
                            loc_pred[:, :, 0] = (loc_pred[:, :, 0] - 0.5) * h / w + 0.5
                    elif assume_straight_pages:
                        loc_pred[:, [0, 2]] *= h / w
                    else:
                        loc_pred[:, :, 0] *= h / w
                elif w > h:
                    if symmetric_pad:
                        if assume_straight_pages:
                            loc_pred[:, [1, 3]] = (loc_pred[:, [1, 3]] - 0.5) * w / h + 0.5
                        else:
                            loc_pred[:, :, 1] = (loc_pred[:, :, 1] - 0.5) * w / h + 0.5
                    elif assume_straight_pages:
                        loc_pred[:, [1, 3]] *= w / h
                    else:
                        loc_pred[:, :, 1] *= w / h
                rectified_preds.append({k: np.clip(loc_pred, 0, 1)})
        return rectified_preds
    return loc_preds


class DetectionPredictor(nn.Module):
    """Implements an object able to localize text elements in a document

    Args:
    ----
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
    """

    def __init__(self, pre_processor: 'PreProcessor', model: 'nn.Module') ->None:
        super().__init__()
        self.pre_processor = pre_processor
        self.model = model.eval()

    @torch.inference_mode()
    def forward(self, pages: 'List[Union[np.ndarray, torch.Tensor]]', return_maps: 'bool'=False, **kwargs: Any) ->Union[List[Dict[str, np.ndarray]], Tuple[List[Dict[str, np.ndarray]], List[np.ndarray]]]:
        preserve_aspect_ratio = self.pre_processor.resize.preserve_aspect_ratio
        symmetric_pad = self.pre_processor.resize.symmetric_pad
        assume_straight_pages = self.model.assume_straight_pages
        if any(page.ndim != 3 for page in pages):
            raise ValueError('incorrect input shape: all pages are expected to be multi-channel 2D images.')
        processed_batches = self.pre_processor(pages)
        _params = next(self.model.parameters())
        self.model, processed_batches = set_device_and_dtype(self.model, processed_batches, _params.device, _params.dtype)
        predicted_batches = [self.model(batch, return_preds=True, return_model_output=True, **kwargs) for batch in processed_batches]
        preds = _remove_padding(pages, [pred for batch in predicted_batches for pred in batch['preds']], preserve_aspect_ratio=preserve_aspect_ratio, symmetric_pad=symmetric_pad, assume_straight_pages=assume_straight_pages)
        if return_maps:
            seg_maps = [pred.permute(1, 2, 0).detach().cpu().numpy() for batch in predicted_batches for pred in batch['out_map']]
            return preds, seg_maps
        return preds


class Element(NestedObject):
    """Implements an abstract document element with exporting and text rendering capabilities"""
    _children_names: 'List[str]' = []
    _exported_keys: 'List[str]' = []

    def __init__(self, **kwargs: Any) ->None:
        for k, v in kwargs.items():
            if k in self._children_names:
                setattr(self, k, v)
            else:
                raise KeyError(f"{self.__class__.__name__} object does not have any attribute named '{k}'")

    def export(self) ->Dict[str, Any]:
        """Exports the object into a nested dict format"""
        export_dict = {k: getattr(self, k) for k in self._exported_keys}
        for children_name in self._children_names:
            if children_name in ['predictions']:
                export_dict[children_name] = {k: [item.export() for item in c] for k, c in getattr(self, children_name).items()}
            else:
                export_dict[children_name] = [c.export() for c in getattr(self, children_name)]
        return export_dict

    @classmethod
    def from_dict(cls, save_dict: 'Dict[str, Any]', **kwargs):
        raise NotImplementedError

    def render(self) ->str:
        raise NotImplementedError


class Artefact(Element):
    """Implements a non-textual element

    Args:
    ----
        artefact_type: the type of artefact
        confidence: the confidence of the type prediction
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size.
    """
    _exported_keys: 'List[str]' = ['geometry', 'type', 'confidence']
    _children_names: 'List[str]' = []

    def __init__(self, artefact_type: 'str', confidence: 'float', geometry: 'BoundingBox') ->None:
        super().__init__()
        self.geometry = geometry
        self.type = artefact_type
        self.confidence = confidence

    def render(self) ->str:
        """Renders the full text of the element"""
        return f'[{self.type.upper()}]'

    def extra_repr(self) ->str:
        return f"type='{self.type}', confidence={self.confidence:.2}"

    @classmethod
    def from_dict(cls, save_dict: 'Dict[str, Any]', **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        return cls(**kwargs)


class Word(Element):
    """Implements a word element

    Args:
    ----
        value: the text string of the word
        confidence: the confidence associated with the text prediction
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
        the page's size
        objectness_score: the objectness score of the detection
        crop_orientation: the general orientation of the crop in degrees and its confidence
    """
    _exported_keys: 'List[str]' = ['value', 'confidence', 'geometry', 'objectness_score', 'crop_orientation']
    _children_names: 'List[str]' = []

    def __init__(self, value: 'str', confidence: 'float', geometry: 'Union[BoundingBox, np.ndarray]', objectness_score: 'float', crop_orientation: 'Dict[str, Any]') ->None:
        super().__init__()
        self.value = value
        self.confidence = confidence
        self.geometry = geometry
        self.objectness_score = objectness_score
        self.crop_orientation = crop_orientation

    def render(self) ->str:
        """Renders the full text of the element"""
        return self.value

    def extra_repr(self) ->str:
        return f"value='{self.value}', confidence={self.confidence:.2}"

    @classmethod
    def from_dict(cls, save_dict: 'Dict[str, Any]', **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        return cls(**kwargs)


BoundingBox = Tuple[Point2D, Point2D]


def resolve_enclosing_bbox(bboxes: 'Union[List[BoundingBox], np.ndarray]') ->Union[BoundingBox, np.ndarray]:
    """Compute enclosing bbox either from:

    Args:
    ----
        bboxes: boxes in one of the following formats:

            - an array of boxes: (*, 4), where boxes have this shape:
            (xmin, ymin, xmax, ymax)

            - a list of BoundingBox

    Returns:
    -------
        a (1, 4) array (enclosing boxarray), or a BoundingBox
    """
    if isinstance(bboxes, np.ndarray):
        xmin, ymin, xmax, ymax = np.split(bboxes, 4, axis=1)
        return np.array([xmin.min(), ymin.min(), xmax.max(), ymax.max()])
    else:
        x, y = zip(*[point for box in bboxes for point in box])
        return (min(x), min(y)), (max(x), max(y))


def resolve_enclosing_rbbox(rbboxes: 'List[np.ndarray]', intermed_size: 'int'=1024) ->np.ndarray:
    """Compute enclosing rotated bbox either from:

    Args:
    ----
        rbboxes: boxes in one of the following formats:

            - an array of boxes: (*, 4, 2), where boxes have this shape:
            (x1, y1), (x2, y2), (x3, y3), (x4, y4)

            - a list of BoundingBox
        intermed_size: size of the intermediate image

    Returns:
    -------
        a (4, 2) array (enclosing rotated box)
    """
    cloud: 'np.ndarray' = np.concatenate(rbboxes, axis=0)
    cloud *= intermed_size
    rect = cv2.minAreaRect(cloud.astype(np.int32))
    return cv2.boxPoints(rect) / intermed_size


class Line(Element):
    """Implements a line element as a collection of words

    Args:
    ----
        words: list of word elements
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size. If not specified, it will be resolved by default to the smallest bounding box enclosing
            all words in it.
    """
    _exported_keys: 'List[str]' = ['geometry', 'objectness_score']
    _children_names: 'List[str]' = ['words']
    words: 'List[Word]' = []

    def __init__(self, words: 'List[Word]', geometry: 'Optional[Union[BoundingBox, np.ndarray]]'=None, objectness_score: 'Optional[float]'=None) ->None:
        if objectness_score is None:
            objectness_score = float(np.mean([w.objectness_score for w in words]))
        if geometry is None:
            box_resolution_fn = resolve_enclosing_rbbox if len(words[0].geometry) == 4 else resolve_enclosing_bbox
            geometry = box_resolution_fn([w.geometry for w in words])
        super().__init__(words=words)
        self.geometry = geometry
        self.objectness_score = objectness_score

    def render(self) ->str:
        """Renders the full text of the element"""
        return ' '.join(w.render() for w in self.words)

    @classmethod
    def from_dict(cls, save_dict: 'Dict[str, Any]', **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({'words': [Word.from_dict(_dict) for _dict in save_dict['words']]})
        return cls(**kwargs)


class Block(Element):
    """Implements a block element as a collection of lines and artefacts

    Args:
    ----
        lines: list of line elements
        artefacts: list of artefacts
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size. If not specified, it will be resolved by default to the smallest bounding box enclosing
            all lines and artefacts in it.
    """
    _exported_keys: 'List[str]' = ['geometry', 'objectness_score']
    _children_names: 'List[str]' = ['lines', 'artefacts']
    lines: 'List[Line]' = []
    artefacts: 'List[Artefact]' = []

    def __init__(self, lines: 'List[Line]'=[], artefacts: 'List[Artefact]'=[], geometry: 'Optional[Union[BoundingBox, np.ndarray]]'=None, objectness_score: 'Optional[float]'=None) ->None:
        if objectness_score is None:
            objectness_score = float(np.mean([w.objectness_score for line in lines for w in line.words]))
        if geometry is None:
            line_boxes = [word.geometry for line in lines for word in line.words]
            artefact_boxes = [artefact.geometry for artefact in artefacts]
            box_resolution_fn = resolve_enclosing_rbbox if isinstance(lines[0].geometry, np.ndarray) else resolve_enclosing_bbox
            geometry = box_resolution_fn(line_boxes + artefact_boxes)
        super().__init__(lines=lines, artefacts=artefacts)
        self.geometry = geometry
        self.objectness_score = objectness_score

    def render(self, line_break: 'str'='\n') ->str:
        """Renders the full text of the element"""
        return line_break.join(line.render() for line in self.lines)

    @classmethod
    def from_dict(cls, save_dict: 'Dict[str, Any]', **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({'lines': [Line.from_dict(_dict) for _dict in save_dict['lines']], 'artefacts': [Artefact.from_dict(_dict) for _dict in save_dict['artefacts']]})
        return cls(**kwargs)


def requires_package(name: 'str', extra_message: 'Optional[str]'=None) ->None:
    """
    package requirement helper

    Args:
    ----
        name: name of the package
        extra_message: additional message to display if the package is not found
    """
    try:
        _pkg_version = importlib.metadata.version(name)
        logging.info(f'{name} version {_pkg_version} available.')
    except importlib.metadata.PackageNotFoundError:
        raise ImportError(f"\n\n{extra_message if extra_message is not None else ''} \nPlease install it with the following command: pip install {name}\n")


def _warn_rotation(entry: 'Dict[str, Any]') ->None:
    global ROTATION_WARNING
    if not ROTATION_WARNING and len(entry['geometry']) == 4:
        logging.warning('Polygons with larger rotations will lead to inaccurate rendering')
        ROTATION_WARNING = True


def synthesize_page(page: 'Dict[str, Any]', draw_proba: 'bool'=False, font_family: 'Optional[str]'=None, smoothing_factor: 'float'=0.95, min_font_size: 'int'=8, max_font_size: 'int'=50) ->np.ndarray:
    """Draw a the content of the element page (OCR response) on a blank page.

    Args:
    ----
        page: exported Page object to represent
        draw_proba: if True, draw words in colors to represent confidence. Blue: p=1, red: p=0
        font_family: family of the font
        smoothing_factor: factor to smooth the font size
        min_font_size: minimum font size
        max_font_size: maximum font size

    Returns:
    -------
        the synthesized page
    """
    h, w = page['dimensions']
    response = Image.new('RGB', (w, h), color=(255, 255, 255))
    for block in page['blocks']:
        if len(block['lines']) > 1:
            for line in block['lines']:
                _warn_rotation(block)
                response = _synthesize(response=response, entry=line, w=w, h=h, draw_proba=draw_proba, font_family=font_family, smoothing_factor=smoothing_factor, min_font_size=min_font_size, max_font_size=max_font_size)
        else:
            for line in block['lines']:
                _warn_rotation(block)
                for word in line['words']:
                    response = _synthesize(response=response, entry=word, w=w, h=h, draw_proba=draw_proba, font_family=font_family, smoothing_factor=smoothing_factor, min_font_size=min_font_size, max_font_size=max_font_size)
    return np.array(response, dtype=np.uint8)


def estimate_page_angle(polys: 'np.ndarray') ->float:
    """Takes a batch of rotated previously ORIENTED polys (N, 4, 2) (rectified by the classifier) and return the
    estimated angle ccw in degrees
    """
    xleft = polys[:, 0, 0] + polys[:, 3, 0]
    yleft = polys[:, 0, 1] + polys[:, 3, 1]
    xright = polys[:, 1, 0] + polys[:, 2, 0]
    yright = polys[:, 1, 1] + polys[:, 2, 1]
    with np.errstate(divide='raise', invalid='raise'):
        try:
            return float(np.median(np.arctan((yleft - yright) / (xright - xleft)) * 180 / np.pi))
        except FloatingPointError:
            return 0.0


def remap_boxes(loc_preds: 'np.ndarray', orig_shape: 'Tuple[int, int]', dest_shape: 'Tuple[int, int]') ->np.ndarray:
    """Remaps a batch of rotated locpred (N, 4, 2) expressed for an origin_shape to a destination_shape.
    This does not impact the absolute shape of the boxes, but allow to calculate the new relative RotatedBbox
    coordinates after a resizing of the image.

    Args:
    ----
        loc_preds: (N, 4, 2) array of RELATIVE loc_preds
        orig_shape: shape of the origin image
        dest_shape: shape of the destination image

    Returns:
    -------
        A batch of rotated loc_preds (N, 4, 2) expressed in the destination referencial
    """
    if len(dest_shape) != 2:
        raise ValueError(f'Mask length should be 2, was found at: {len(dest_shape)}')
    if len(orig_shape) != 2:
        raise ValueError(f'Image_shape length should be 2, was found at: {len(orig_shape)}')
    orig_height, orig_width = orig_shape
    dest_height, dest_width = dest_shape
    mboxes = loc_preds.copy()
    mboxes[:, :, 0] = (loc_preds[:, :, 0] * orig_width + (dest_width - orig_width) / 2) / dest_width
    mboxes[:, :, 1] = (loc_preds[:, :, 1] * orig_height + (dest_height - orig_height) / 2) / dest_height
    return mboxes


def rotate_boxes(loc_preds: 'np.ndarray', angle: 'float', orig_shape: 'Tuple[int, int]', min_angle: 'float'=1.0, target_shape: 'Optional[Tuple[int, int]]'=None) ->np.ndarray:
    """Rotate a batch of straight bounding boxes (xmin, ymin, xmax, ymax, c) or rotated bounding boxes
    (4, 2) of an angle, if angle > min_angle, around the center of the page.
    If target_shape is specified, the boxes are remapped to the target shape after the rotation. This
    is done to remove the padding that is created by rotate_page(expand=True)

    Args:
    ----
        loc_preds: (N, 4) or (N, 4, 2) array of RELATIVE boxes
        angle: angle between -90 and +90 degrees
        orig_shape: shape of the origin image
        min_angle: minimum angle to rotate boxes
        target_shape: shape of the destination image

    Returns:
    -------
        A batch of rotated boxes (N, 4, 2): or a batch of straight bounding boxes
    """
    _boxes = loc_preds.copy()
    if _boxes.ndim == 2:
        _boxes = np.stack([_boxes[:, [0, 1]], _boxes[:, [2, 1]], _boxes[:, [2, 3]], _boxes[:, [0, 3]]], axis=1)
    if abs(angle) < min_angle or abs(angle) > 90 - min_angle:
        return _boxes
    angle_rad = angle * np.pi / 180.0
    rotation_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]], dtype=_boxes.dtype)
    points: 'np.ndarray' = np.stack((_boxes[:, :, 0] * orig_shape[1], _boxes[:, :, 1] * orig_shape[0]), axis=-1)
    image_center = orig_shape[1] / 2, orig_shape[0] / 2
    rotated_points = image_center + np.matmul(points - image_center, rotation_mat)
    rotated_boxes: 'np.ndarray' = np.stack((rotated_points[:, :, 0] / orig_shape[1], rotated_points[:, :, 1] / orig_shape[0]), axis=-1)
    if target_shape is not None:
        rotated_boxes = remap_boxes(rotated_boxes, orig_shape=orig_shape, dest_shape=target_shape)
    return rotated_boxes


class Prediction(Word):
    """Implements a prediction element"""

    def render(self) ->str:
        """Renders the full text of the element"""
        return self.value

    def extra_repr(self) ->str:
        return f"value='{self.value}', confidence={self.confidence:.2}, bounding_box={self.geometry}"


def synthesize_kie_page(page: 'Dict[str, Any]', draw_proba: 'bool'=False, font_family: 'Optional[str]'=None) ->np.ndarray:
    """Draw a the content of the element page (OCR response) on a blank page.

    Args:
    ----
        page: exported Page object to represent
        draw_proba: if True, draw words in colors to represent confidence. Blue: p=1, red: p=0
        font_family: family of the font
        smoothing_factor: factor to smooth the font size
        min_font_size: minimum font size
        max_font_size: maximum font size

    Returns:
    -------
        the synthesized page
    """
    h, w = page['dimensions']
    response = Image.new('RGB', (w, h), color=(255, 255, 255))
    for predictions in page['predictions'].values():
        for prediction in predictions:
            _warn_rotation(prediction)
            response = _synthesize(response=response, entry=prediction, w=w, h=h, draw_proba=draw_proba, font_family=font_family)
    return np.array(response, dtype=np.uint8)


def get_colors(num_colors: 'int') ->List[Tuple[float, float, float]]:
    """Generate num_colors color for matplotlib

    Args:
    ----
        num_colors: number of colors to generate

    Returns:
    -------
        colors: list of generated colors
    """
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def get_max_width_length_ratio(contour: 'np.ndarray') ->float:
    """Get the maximum shape ratio of a contour.

    Args:
    ----
        contour: the contour from cv2.findContour

    Returns:
    -------
        the maximum shape ratio
    """
    _, (w, h), _ = cv2.minAreaRect(contour)
    return max(w / h, h / w)


def rotate_abs_points(points: 'np.ndarray', angle: 'float'=0.0) ->np.ndarray:
    """Rotate points counter-clockwise.

    Args:
    ----
        points: array of size (N, 2)
        angle: angle between -90 and +90 degrees

    Returns:
    -------
        Rotated points
    """
    angle_rad = angle * np.pi / 180.0
    rotation_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]], dtype=points.dtype)
    return np.matmul(points, rotation_mat.T)


def compute_expanded_shape(img_shape: 'Tuple[int, int]', angle: 'float') ->Tuple[int, int]:
    """Compute the shape of an expanded rotated image

    Args:
    ----
        img_shape: the height and width of the image
        angle: angle between -90 and +90 degrees

    Returns:
    -------
        the height and width of the rotated image
    """
    points: 'np.ndarray' = np.array([[img_shape[1] / 2, img_shape[0] / 2], [-img_shape[1] / 2, img_shape[0] / 2]])
    rotated_points = rotate_abs_points(points, angle)
    wh_shape = 2 * np.abs(rotated_points).max(axis=0)
    return wh_shape[1], wh_shape[0]


def rotate_image(image: 'np.ndarray', angle: 'float', expand: 'bool'=False, preserve_origin_shape: 'bool'=False) ->np.ndarray:
    """Rotate an image counterclockwise by an given angle.

    Args:
    ----
        image: numpy tensor to rotate
        angle: rotation angle in degrees, between -90 and +90
        expand: whether the image should be padded before the rotation
        preserve_origin_shape: if expand is set to True, resizes the final output to the original image size

    Returns:
    -------
        Rotated array, padded by 0 by default.
    """
    exp_img: 'np.ndarray'
    if expand:
        exp_shape = compute_expanded_shape(image.shape[:2], angle)
        h_pad, w_pad = int(max(0, ceil(exp_shape[0] - image.shape[0]))), int(max(0, ceil(exp_shape[1] - image.shape[1])))
        exp_img = np.pad(image, ((h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2), (0, 0)))
    else:
        exp_img = image
    height, width = exp_img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rot_img = cv2.warpAffine(exp_img, rot_mat, (width, height))
    if expand:
        if image.shape[0] / image.shape[1] != rot_img.shape[0] / rot_img.shape[1]:
            if rot_img.shape[0] / rot_img.shape[1] > image.shape[0] / image.shape[1]:
                h_pad, w_pad = 0, int(rot_img.shape[0] * image.shape[1] / image.shape[0] - rot_img.shape[1])
            else:
                h_pad, w_pad = int(rot_img.shape[1] * image.shape[0] / image.shape[1] - rot_img.shape[0]), 0
            rot_img = np.pad(rot_img, ((h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2), (0, 0)))
        if preserve_origin_shape:
            rot_img = cv2.resize(rot_img, image.shape[:-1][::-1], interpolation=cv2.INTER_LINEAR)
    return rot_img


def estimate_orientation(img: 'np.ndarray', general_page_orientation: 'Optional[Tuple[int, float]]'=None, n_ct: 'int'=70, ratio_threshold_for_lines: 'float'=3, min_confidence: 'float'=0.2, lower_area: 'int'=100) ->int:
    """Estimate the angle of the general document orientation based on the
     lines of the document and the assumption that they should be horizontal.

    Args:
    ----
        img: the img or bitmap to analyze (H, W, C)
        general_page_orientation: the general orientation of the page (angle [0, 90, 180, 270 (-90)], confidence)
            estimated by a model
        n_ct: the number of contours used for the orientation estimation
        ratio_threshold_for_lines: this is the ratio w/h used to discriminates lines
        min_confidence: the minimum confidence to consider the general_page_orientation
        lower_area: the minimum area of a contour to be considered

    Returns:
    -------
        the estimated angle of the page (clockwise, negative for left side rotation, positive for right side rotation)
    """
    assert len(img.shape) == 3 and img.shape[-1] in [1, 3], f'Image shape {img.shape} not supported'
    thresh = None
    if img.shape[-1] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.medianBlur(gray_img, 5)
        thresh = cv2.threshold(gray_img, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    else:
        thresh = img.astype(np.uint8)
    page_orientation, orientation_confidence = general_page_orientation or (None, 0.0)
    if page_orientation and orientation_confidence >= min_confidence:
        thresh = rotate_image(thresh, -page_orientation)
    else:
        h, w = img.shape[:2]
        k_x = max(1, floor(w / 100))
        k_y = max(1, floor(h / 100))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_x, k_y))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted([contour for contour in contours if cv2.contourArea(contour) > lower_area], key=get_max_width_length_ratio, reverse=True)
    angles = []
    for contour in contours[:n_ct]:
        _, (w, h), angle = cv2.minAreaRect(contour)
        if w / h > ratio_threshold_for_lines:
            angles.append(angle)
        elif w / h < 1 / ratio_threshold_for_lines:
            angles.append(angle - 90)
    if len(angles) == 0:
        estimated_angle = 0
    else:
        median = -median_low(angles)
        estimated_angle = -round(median) if abs(median) != 0 else 0
    if page_orientation and orientation_confidence >= min_confidence:
        if abs(estimated_angle) == abs(page_orientation):
            return page_orientation
        estimated_angle = estimated_angle if page_orientation == 0 else page_orientation + estimated_angle
        if estimated_angle > 180:
            estimated_angle -= 360
    return estimated_angle


def extract_crops(img: 'np.ndarray', boxes: 'np.ndarray', channels_last: 'bool'=True) ->List[np.ndarray]:
    """Created cropped images from list of bounding boxes

    Args:
    ----
        img: input image
        boxes: bounding boxes of shape (N, 4) where N is the number of boxes, and the relative
            coordinates (xmin, ymin, xmax, ymax)
        channels_last: whether the channel dimensions is the last one instead of the last one

    Returns:
    -------
        list of cropped images
    """
    if boxes.shape[0] == 0:
        return []
    if boxes.shape[1] != 4:
        raise AssertionError('boxes are expected to be relative and in order (xmin, ymin, xmax, ymax)')
    _boxes = boxes.copy()
    h, w = img.shape[:2] if channels_last else img.shape[-2:]
    if not np.issubdtype(_boxes.dtype, np.integer):
        _boxes[:, [0, 2]] *= w
        _boxes[:, [1, 3]] *= h
        _boxes = _boxes.round().astype(int)
        _boxes[2:] += 1
    if channels_last:
        return deepcopy([img[box[1]:box[3], box[0]:box[2]] for box in _boxes])
    return deepcopy([img[:, box[1]:box[3], box[0]:box[2]] for box in _boxes])


def extract_rcrops(img: 'np.ndarray', polys: 'np.ndarray', dtype=np.float32, channels_last: 'bool'=True, assume_horizontal: 'bool'=False) ->List[np.ndarray]:
    """Created cropped images from list of rotated bounding boxes

    Args:
    ----
        img: input image
        polys: bounding boxes of shape (N, 4, 2)
        dtype: target data type of bounding boxes
        channels_last: whether the channel dimensions is the last one instead of the last one
        assume_horizontal: whether the boxes are assumed to be only horizontally oriented

    Returns:
    -------
        list of cropped images
    """
    if polys.shape[0] == 0:
        return []
    if polys.shape[1:] != (4, 2):
        raise AssertionError('polys are expected to be quadrilateral, of shape (N, 4, 2)')
    _boxes = polys.copy()
    height, width = img.shape[:2] if channels_last else img.shape[-2:]
    if not np.issubdtype(_boxes.dtype, np.integer):
        _boxes[:, :, 0] *= width
        _boxes[:, :, 1] *= height
    src_img = img if channels_last else img.transpose(1, 2, 0)
    if assume_horizontal:
        crops = []
        for box in _boxes:
            centroid = np.mean(box, axis=0)
            left_points = box[box[:, 0] < centroid[0]]
            right_points = box[box[:, 0] >= centroid[0]]
            left_points = left_points[np.argsort(left_points[:, 1])]
            top_left_pt = left_points[0]
            bottom_left_pt = left_points[-1]
            right_points = right_points[np.argsort(right_points[:, 1])]
            top_right_pt = right_points[0]
            bottom_right_pt = right_points[-1]
            box_points = np.array([top_left_pt, bottom_left_pt, top_right_pt, bottom_right_pt], dtype=dtype)
            width_upper = np.linalg.norm(top_right_pt - top_left_pt)
            width_lower = np.linalg.norm(bottom_right_pt - bottom_left_pt)
            height_left = np.linalg.norm(bottom_left_pt - top_left_pt)
            height_right = np.linalg.norm(bottom_right_pt - top_right_pt)
            rect_width = max(int(width_upper), int(width_lower))
            rect_height = max(int(height_left), int(height_right))
            dst_pts = np.array([[0, 0], [0, rect_height - 1], [rect_width - 1, 0], [rect_width - 1, rect_height - 1]], dtype=dtype)
            affine_mat = cv2.getPerspectiveTransform(box_points, dst_pts)
            crop = cv2.warpPerspective(src_img, affine_mat, (rect_width, rect_height))
            crops.append(crop)
    else:
        src_pts = _boxes[:, :3].astype(np.float32)
        d1 = np.linalg.norm(src_pts[:, 0] - src_pts[:, 1], axis=-1)
        d2 = np.linalg.norm(src_pts[:, 1] - src_pts[:, 2], axis=-1)
        dst_pts = np.zeros((_boxes.shape[0], 3, 2), dtype=dtype)
        dst_pts[:, 1, 0] = dst_pts[:, 2, 0] = d1 - 1
        dst_pts[:, 2, 1] = d2 - 1
        crops = [cv2.warpAffine(src_img, cv2.getAffineTransform(src_pts[idx], dst_pts[idx]), (int(d1[idx]), int(d2[idx]))) for idx in range(_boxes.shape[0])]
    return crops


def rectify_crops(crops: 'List[np.ndarray]', orientations: 'List[int]') ->List[np.ndarray]:
    """Rotate each crop of the list according to the predicted orientation:
    0: already straight, no rotation
    1: 90 ccw, rotate 3 times ccw
    2: 180, rotate 2 times ccw
    3: 270 ccw, rotate 1 time ccw
    """
    orientations = [(4 - pred if pred != 0 else 0) for pred in orientations]
    return [(crop if orientation == 0 else np.rot90(crop, orientation)) for orientation, crop in zip(orientations, crops)] if len(orientations) > 0 else []


def rectify_loc_preds(page_loc_preds: 'np.ndarray', orientations: 'List[int]') ->Optional[np.ndarray]:
    """Orient the quadrangle (Polygon4P) according to the predicted orientation,
    so that the points are in this order: top L, top R, bot R, bot L if the crop is readable
    """
    return np.stack([np.roll(page_loc_pred, orientation, axis=0) for orientation, page_loc_pred in zip(orientations, page_loc_preds)], axis=0) if len(orientations) > 0 else None


def remove_image_padding(image: 'np.ndarray') ->np.ndarray:
    """Remove black border padding from an image

    Args:
    ----
        image: numpy tensor to remove padding from

    Returns:
    -------
        Image with padding removed
    """
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return image[rmin:rmax + 1, cmin:cmax + 1]


class _OCRPredictor:
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
    ----
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        straighten_pages: if True, estimates the page general orientation based on the median line orientation.
            Then, rotates page before passing it to the deep learning modules. The final predictions will be remapped
            accordingly. Doing so will improve performances for documents with page-uniform rotations.
        preserve_aspect_ratio: if True, resize preserving the aspect ratio (with padding)
        symmetric_pad: if True and preserve_aspect_ratio is True, pas the image symmetrically.
        detect_orientation: if True, the estimated general page orientation will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        **kwargs: keyword args of `DocumentBuilder`
    """
    crop_orientation_predictor: 'Optional[OrientationPredictor]'
    page_orientation_predictor: 'Optional[OrientationPredictor]'

    def __init__(self, assume_straight_pages: 'bool'=True, straighten_pages: 'bool'=False, preserve_aspect_ratio: 'bool'=True, symmetric_pad: 'bool'=True, detect_orientation: 'bool'=False, **kwargs: Any) ->None:
        self.assume_straight_pages = assume_straight_pages
        self.straighten_pages = straighten_pages
        self._page_orientation_disabled = kwargs.pop('disable_page_orientation', False)
        self._crop_orientation_disabled = kwargs.pop('disable_crop_orientation', False)
        self.crop_orientation_predictor = None if assume_straight_pages else crop_orientation_predictor(pretrained=True, disabled=self._crop_orientation_disabled)
        self.page_orientation_predictor = page_orientation_predictor(pretrained=True, disabled=self._page_orientation_disabled) if detect_orientation or straighten_pages or not assume_straight_pages else None
        self.doc_builder = DocumentBuilder(**kwargs)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.hooks: 'List[Callable]' = []

    def _general_page_orientations(self, pages: 'List[np.ndarray]') ->List[Tuple[int, float]]:
        _, classes, probs = zip(self.page_orientation_predictor(pages))
        page_orientations = [(orientation, prob) for page_classes, page_probs in zip(classes, probs) for orientation, prob in zip(page_classes, page_probs)]
        return page_orientations

    def _get_orientations(self, pages: 'List[np.ndarray]', seg_maps: 'List[np.ndarray]') ->Tuple[List[Tuple[int, float]], List[int]]:
        general_pages_orientations = self._general_page_orientations(pages)
        origin_page_orientations = [estimate_orientation(seq_map, general_orientation) for seq_map, general_orientation in zip(seg_maps, general_pages_orientations)]
        return general_pages_orientations, origin_page_orientations

    def _straighten_pages(self, pages: 'List[np.ndarray]', seg_maps: 'List[np.ndarray]', general_pages_orientations: 'Optional[List[Tuple[int, float]]]'=None, origin_pages_orientations: 'Optional[List[int]]'=None) ->List[np.ndarray]:
        general_pages_orientations = general_pages_orientations if general_pages_orientations else self._general_page_orientations(pages)
        origin_pages_orientations = origin_pages_orientations if origin_pages_orientations else [estimate_orientation(seq_map, general_orientation) for seq_map, general_orientation in zip(seg_maps, general_pages_orientations)]
        return [remove_image_padding(rotate_image(page, angle, expand=page.shape[0] != page.shape[1])) for page, angle in zip(pages, origin_pages_orientations)]

    @staticmethod
    def _generate_crops(pages: 'List[np.ndarray]', loc_preds: 'List[np.ndarray]', channels_last: 'bool', assume_straight_pages: 'bool'=False, assume_horizontal: 'bool'=False) ->List[List[np.ndarray]]:
        if assume_straight_pages:
            crops = [extract_crops(page, _boxes[:, :4], channels_last=channels_last) for page, _boxes in zip(pages, loc_preds)]
        else:
            crops = [extract_rcrops(page, _boxes[:, :4], channels_last=channels_last, assume_horizontal=assume_horizontal) for page, _boxes in zip(pages, loc_preds)]
        return crops

    @staticmethod
    def _prepare_crops(pages: 'List[np.ndarray]', loc_preds: 'List[np.ndarray]', channels_last: 'bool', assume_straight_pages: 'bool'=False, assume_horizontal: 'bool'=False) ->Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        crops = _OCRPredictor._generate_crops(pages, loc_preds, channels_last, assume_straight_pages, assume_horizontal)
        is_kept = [[all(s > 0 for s in crop.shape) for crop in page_crops] for page_crops in crops]
        crops = [[crop for crop, _kept in zip(page_crops, page_kept) if _kept] for page_crops, page_kept in zip(crops, is_kept)]
        loc_preds = [_boxes[_kept] for _boxes, _kept in zip(loc_preds, is_kept)]
        return crops, loc_preds

    def _rectify_crops(self, crops: 'List[List[np.ndarray]]', loc_preds: 'List[np.ndarray]') ->Tuple[List[List[np.ndarray]], List[np.ndarray], List[Tuple[int, float]]]:
        orientations, classes, probs = zip(*[self.crop_orientation_predictor(page_crops) for page_crops in crops])
        rect_crops = [rectify_crops(page_crops, orientation) for page_crops, orientation in zip(crops, orientations)]
        rect_loc_preds = [(rectify_loc_preds(page_loc_preds, orientation) if len(page_loc_preds) > 0 else page_loc_preds) for page_loc_preds, orientation in zip(loc_preds, orientations)]
        crop_orientations = [(orientation, prob) for page_classes, page_probs in zip(classes, probs) for orientation, prob in zip(page_classes, page_probs)]
        return rect_crops, rect_loc_preds, crop_orientations

    @staticmethod
    def _process_predictions(loc_preds: 'List[np.ndarray]', word_preds: 'List[Tuple[str, float]]', crop_orientations: 'List[Dict[str, Any]]') ->Tuple[List[np.ndarray], List[List[Tuple[str, float]]], List[List[Dict[str, Any]]]]:
        text_preds = []
        crop_orientation_preds = []
        if len(loc_preds) > 0:
            _idx = 0
            for page_boxes in loc_preds:
                text_preds.append(word_preds[_idx:_idx + page_boxes.shape[0]])
                crop_orientation_preds.append(crop_orientations[_idx:_idx + page_boxes.shape[0]])
                _idx += page_boxes.shape[0]
        return loc_preds, text_preds, crop_orientation_preds

    def add_hook(self, hook: 'Callable') ->None:
        """Add a hook to the predictor

        Args:
        ----
            hook: a callable that takes as input the `loc_preds` and returns the modified `loc_preds`
        """
        self.hooks.append(hook)


class _KIEPredictor(_OCRPredictor):
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
    ----
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        straighten_pages: if True, estimates the page general orientation based on the median line orientation.
            Then, rotates page before passing it to the deep learning modules. The final predictions will be remapped
            accordingly. Doing so will improve performances for documents with page-uniform rotations.
        preserve_aspect_ratio: if True, resize preserving the aspect ratio (with padding)
        symmetric_pad: if True and preserve_aspect_ratio is True, pas the image symmetrically.
        detect_orientation: if True, the estimated general page orientation will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        kwargs: keyword args of `DocumentBuilder`
    """
    crop_orientation_predictor: 'Optional[OrientationPredictor]'
    page_orientation_predictor: 'Optional[OrientationPredictor]'

    def __init__(self, assume_straight_pages: 'bool'=True, straighten_pages: 'bool'=False, preserve_aspect_ratio: 'bool'=True, symmetric_pad: 'bool'=True, detect_orientation: 'bool'=False, **kwargs: Any) ->None:
        super().__init__(assume_straight_pages, straighten_pages, preserve_aspect_ratio, symmetric_pad, detect_orientation, **kwargs)
        kwargs.pop('disable_page_orientation', None)
        kwargs.pop('disable_crop_orientation', None)
        self.doc_builder: 'KIEDocumentBuilder' = KIEDocumentBuilder(**kwargs)


def detach_scores(boxes: 'List[np.ndarray]') ->Tuple[List[np.ndarray], List[np.ndarray]]:
    """Detach the objectness scores from box predictions

    Args:
    ----
        boxes: list of arrays with boxes of shape (N, 5) or (N, 5, 2)

    Returns:
    -------
        a tuple of two lists: the first one contains the boxes without the objectness scores,
        the second one contains the objectness scores
    """

    def _detach(boxes: 'np.ndarray') ->Tuple[np.ndarray, np.ndarray]:
        if boxes.ndim == 2:
            return boxes[:, :-1], boxes[:, -1]
        return boxes[:, :-1], boxes[:, -1, -1]
    loc_preds, obj_scores = zip(*(_detach(box) for box in boxes))
    return list(loc_preds), list(obj_scores)


def get_language(text: 'str') ->Tuple[str, float]:
    """Get languages of a text using langdetect model.
    Get the language with the highest probability or no language if only a few words or a low probability

    Args:
    ----
        text (str): text

    Returns:
    -------
        The detected language in ISO 639 code and confidence score
    """
    try:
        lang = detect_langs(text.lower())[0]
    except LangDetectException:
        return 'unknown', 0.0
    if len(text) <= 1 or len(text) <= 5 and lang.prob <= 0.2:
        return 'unknown', 0.0
    return lang.lang, lang.prob


def invert_data_structure(x: 'Union[List[Dict[str, Any]], Dict[str, List[Any]]]') ->Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
    """Invert a List of Dict of elements to a Dict of list of elements and the other way around

    Args:
    ----
        x: a list of dictionaries with the same keys or a dictionary of lists of the same length

    Returns:
    -------
        dictionary of list when x is a list of dictionaries or a list of dictionaries when x is dictionary of lists
    """
    if isinstance(x, dict):
        assert len({len(v) for v in x.values()}) == 1, 'All the lists in the dictionnary should have the same length.'
        return [dict(zip(x, t)) for t in zip(*x.values())]
    elif isinstance(x, list):
        return {k: [dic[k] for dic in x] for k in x[0]}
    else:
        raise TypeError(f'Expected input to be either a dict or a list, got {type(input)} instead.')


class PositionalEncoding(nn.Module):
    """Compute positional encoding"""

    def __init__(self, d_model: 'int', dropout: 'float'=0.1, max_len: 'int'=5000) ->None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward pass

        Args:
        ----
            x: embeddings (batch, max_len, d_model)

        Returns
        -------
            positional embeddings (batch, max_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Decoder(nn.Module):
    """Transformer Decoder"""

    def __init__(self, num_layers: 'int', num_heads: 'int', d_model: 'int', vocab_size: 'int', dropout: 'float'=0.2, dff: 'int'=2048, maximum_position_encoding: 'int'=50) ->None:
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.layer_norm_input = nn.LayerNorm(d_model, eps=1e-05)
        self.layer_norm_masked_attention = nn.LayerNorm(d_model, eps=1e-05)
        self.layer_norm_attention = nn.LayerNorm(d_model, eps=1e-05)
        self.layer_norm_output = nn.LayerNorm(d_model, eps=1e-05)
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, maximum_position_encoding)
        self.attention = nn.ModuleList([MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)])
        self.source_attention = nn.ModuleList([MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)])
        self.position_feed_forward = nn.ModuleList([PositionwiseFeedForward(d_model, dff, dropout) for _ in range(self.num_layers)])

    def forward(self, tgt: 'torch.Tensor', memory: 'torch.Tensor', source_mask: 'Optional[torch.Tensor]'=None, target_mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        tgt = self.embed(tgt) * math.sqrt(self.d_model)
        pos_enc_tgt = self.positional_encoding(tgt)
        output = pos_enc_tgt
        for i in range(self.num_layers):
            normed_output = self.layer_norm_input(output)
            output = output + self.dropout(self.attention[i](normed_output, normed_output, normed_output, target_mask))
            normed_output = self.layer_norm_masked_attention(output)
            output = output + self.dropout(self.source_attention[i](normed_output, memory, memory, source_mask))
            normed_output = self.layer_norm_attention(output)
            output = output + self.dropout(self.position_feed_forward[i](normed_output))
        return self.layer_norm_output(output)


class RecognitionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
    ----
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(self, vocab: 'str') ->None:
        self.vocab = vocab
        self._embedding = list(self.vocab) + ['<eos>']

    def extra_repr(self) ->str:
        return f'vocab_size={len(self.vocab)}'


def decode_sequence(input_seq: 'Union[np.ndarray, SequenceType[int]]', mapping: 'str') ->str:
    """Given a predefined mapping, decode the sequence of numbers to a string

    Args:
    ----
        input_seq: array to decode
        mapping: vocabulary (string), the encoding is given by the indexing of the character sequence

    Returns:
    -------
        A string, decoded from input_seq
    """
    if not isinstance(input_seq, (Sequence, np.ndarray)):
        raise TypeError('Invalid sequence type')
    if isinstance(input_seq, np.ndarray) and (input_seq.dtype != np.int_ or input_seq.max() >= len(mapping)):
        raise AssertionError('Input must be an array of int, with max less than mapping size')
    return ''.join(map(mapping.__getitem__, input_seq))


def encode_string(input_string: 'str', vocab: 'str') ->List[int]:
    """Given a predefined mapping, encode the string to a sequence of numbers

    Args:
    ----
        input_string: string to encode
        vocab: vocabulary (string), the encoding is given by the indexing of the character sequence

    Returns:
    -------
        A list encoding the input_string
    """
    try:
        return list(map(vocab.index, input_string))
    except ValueError:
        raise ValueError(f"some characters cannot be found in 'vocab'.                          Please check the input string {input_string} and the vocabulary {vocab}")


def encode_sequences(sequences: 'List[str]', vocab: 'str', target_size: 'Optional[int]'=None, eos: 'int'=-1, sos: 'Optional[int]'=None, pad: 'Optional[int]'=None, dynamic_seq_length: 'bool'=False) ->np.ndarray:
    """Encode character sequences using a given vocab as mapping

    Args:
    ----
        sequences: the list of character sequences of size N
        vocab: the ordered vocab to use for encoding
        target_size: maximum length of the encoded data
        eos: encoding of End Of String
        sos: optional encoding of Start Of String
        pad: optional encoding for padding. In case of padding, all sequences are followed by 1 EOS then PAD
        dynamic_seq_length: if `target_size` is specified, uses it as upper bound and enables dynamic sequence size

    Returns:
    -------
        the padded encoded data as a tensor
    """
    if 0 <= eos < len(vocab):
        raise ValueError("argument 'eos' needs to be outside of vocab possible indices")
    if not isinstance(target_size, int) or dynamic_seq_length:
        max_length = max(len(w) for w in sequences) + 1
        if isinstance(sos, int):
            max_length += 1
        if isinstance(pad, int):
            max_length += 1
        target_size = max_length if not isinstance(target_size, int) else min(max_length, target_size)
    if isinstance(pad, int):
        if 0 <= pad < len(vocab):
            raise ValueError("argument 'pad' needs to be outside of vocab possible indices")
        default_symbol = pad
    else:
        default_symbol = eos
    encoded_data: 'np.ndarray' = np.full([len(sequences), target_size], default_symbol, dtype=np.int32)
    for idx, seq in enumerate(map(partial(encode_string, vocab=vocab), sequences)):
        if isinstance(pad, int):
            seq.append(eos)
        encoded_data[idx, :min(len(seq), target_size)] = seq[:min(len(seq), target_size)]
    if isinstance(sos, int):
        if 0 <= sos < len(vocab):
            raise ValueError("argument 'sos' needs to be outside of vocab possible indices")
        encoded_data = np.roll(encoded_data, 1)
        encoded_data[:, 0] = sos
    return encoded_data


class RecognitionModel(NestedObject):
    """Implements abstract RecognitionModel class"""
    vocab: 'str'
    max_length: 'int'

    def build_target(self, gts: 'List[str]') ->Tuple[np.ndarray, List[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
        ----
            gts: list of ground-truth labels

        Returns:
        -------
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(sequences=gts, vocab=self.vocab, target_size=self.max_length, eos=len(self.vocab))
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


class CRNN(RecognitionModel, nn.Module):
    """Implements a CRNN architecture as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    Args:
    ----
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of units in the LSTM layers
        exportable: onnx exportable returns only logits
        cfg: configuration dictionary
    """
    _children_names: 'List[str]' = ['feat_extractor', 'decoder', 'linear', 'postprocessor']

    def __init__(self, feature_extractor: 'nn.Module', vocab: 'str', rnn_units: 'int'=128, input_shape: 'Tuple[int, int, int]'=(3, 32, 128), exportable: 'bool'=False, cfg: 'Optional[Dict[str, Any]]'=None) ->None:
        super().__init__()
        self.vocab = vocab
        self.cfg = cfg
        self.max_length = 32
        self.exportable = exportable
        self.feat_extractor = feature_extractor
        with torch.inference_mode():
            out_shape = self.feat_extractor(torch.zeros((1, *input_shape))).shape
        lstm_in = out_shape[1] * out_shape[2]
        self.decoder = nn.LSTM(input_size=lstm_in, hidden_size=rnn_units, batch_first=True, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(in_features=2 * rnn_units, out_features=len(vocab) + 1)
        self.postprocessor = CTCPostProcessor(vocab=vocab)
        for n, m in self.named_modules():
            if n.startswith('feat_extractor.'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def compute_loss(self, model_output: 'torch.Tensor', target: 'List[str]') ->torch.Tensor:
        """Compute CTC loss for the model.

        Args:
        ----
            model_output: predicted logits of the model
            target: list of target strings

        Returns:
        -------
            The loss of the model on the batch
        """
        gt, seq_len = self.build_target(target)
        batch_len = model_output.shape[0]
        input_length = model_output.shape[1] * torch.ones(size=(batch_len,), dtype=torch.int32)
        logits = model_output.permute(1, 0, 2)
        probs = F.log_softmax(logits, dim=-1)
        ctc_loss = F.ctc_loss(probs, torch.from_numpy(gt), input_length, torch.tensor(seq_len, dtype=torch.int), len(self.vocab), zero_infinity=True)
        return ctc_loss

    def forward(self, x: 'torch.Tensor', target: 'Optional[List[str]]'=None, return_model_output: 'bool'=False, return_preds: 'bool'=False) ->Dict[str, Any]:
        if self.training and target is None:
            raise ValueError('Need to provide labels during training')
        features = self.feat_extractor(x)
        c, h, w = features.shape[1], features.shape[2], features.shape[3]
        features_seq = torch.reshape(features, shape=(-1, h * c, w))
        features_seq = torch.transpose(features_seq, 1, 2)
        logits, _ = self.decoder(features_seq)
        logits = self.linear(logits)
        out: 'Dict[str, Any]' = {}
        if self.exportable:
            out['logits'] = logits
            return out
        if return_model_output:
            out['out_map'] = logits
        if target is None or return_preds:
            out['preds'] = self.postprocessor(logits)
        if target is not None:
            out['loss'] = self.compute_loss(logits, target)
        return out


class _MASTERPostProcessor(RecognitionPostProcessor):
    """Abstract class to postprocess the raw output of the model

    Args:
    ----
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(self, vocab: 'str') ->None:
        super().__init__(vocab)
        self._embedding = list(vocab) + ['<eos>'] + ['<sos>'] + ['<pad>']


class MASTERPostProcessor(_MASTERPostProcessor):
    """Post processor for MASTER architectures"""

    def __call__(self, logits: 'torch.Tensor') ->List[Tuple[str, float]]:
        out_idxs = logits.argmax(-1)
        probs = torch.gather(torch.softmax(logits, -1), -1, out_idxs.unsqueeze(-1)).squeeze(-1)
        probs = probs.min(dim=1).values.detach().cpu()
        word_values = [''.join(self._embedding[idx] for idx in encoded_seq).split('<eos>')[0] for encoded_seq in out_idxs.cpu().numpy()]
        return list(zip(word_values, probs.numpy().clip(0, 1).tolist()))


class _MASTER:
    vocab: 'str'
    max_length: 'int'

    def build_target(self, gts: 'List[str]') ->Tuple[np.ndarray, List[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
        ----
            gts: list of ground-truth labels

        Returns:
        -------
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(sequences=gts, vocab=self.vocab, target_size=self.max_length, eos=len(self.vocab), sos=len(self.vocab) + 1, pad=len(self.vocab) + 2)
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


class MASTER(_MASTER, nn.Module):
    """Implements MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/wenwenyu/MASTER-pytorch>`_.

    Args:
    ----
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary, (without EOS, SOS, PAD)
        d_model: d parameter for the transformer decoder
        dff: depth of the pointwise feed-forward layer
        num_heads: number of heads for the mutli-head attention module
        num_layers: number of decoder layers to stack
        max_length: maximum length of character sequence handled by the model
        dropout: dropout probability of the decoder
        input_shape: size of the image inputs
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    def __init__(self, feature_extractor: 'nn.Module', vocab: 'str', d_model: 'int'=512, dff: 'int'=2048, num_heads: 'int'=8, num_layers: 'int'=3, max_length: 'int'=50, dropout: 'float'=0.2, input_shape: 'Tuple[int, int, int]'=(3, 32, 128), exportable: 'bool'=False, cfg: 'Optional[Dict[str, Any]]'=None) ->None:
        super().__init__()
        self.exportable = exportable
        self.max_length = max_length
        self.d_model = d_model
        self.vocab = vocab
        self.cfg = cfg
        self.vocab_size = len(vocab)
        self.feat_extractor = feature_extractor
        self.positional_encoding = PositionalEncoding(self.d_model, dropout, max_len=input_shape[1] * input_shape[2])
        self.decoder = Decoder(num_layers=num_layers, d_model=self.d_model, num_heads=num_heads, vocab_size=self.vocab_size + 3, dff=dff, dropout=dropout, maximum_position_encoding=self.max_length)
        self.linear = nn.Linear(self.d_model, self.vocab_size + 3)
        self.postprocessor = MASTERPostProcessor(vocab=self.vocab)
        for n, m in self.named_modules():
            if n.startswith('feat_extractor.'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_source_and_target_mask(self, source: 'torch.Tensor', target: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        target_pad_mask = (target != self.vocab_size + 2).unsqueeze(1).unsqueeze(1)
        target_length = target.size(1)
        target_sub_mask = torch.tril(torch.ones((target_length, target_length), device=source.device), diagonal=0)
        source_mask = torch.ones((target_length, source.size(1)), dtype=torch.uint8, device=source.device)
        target_mask = target_pad_mask & target_sub_mask
        return source_mask, target_mask.int()

    @staticmethod
    def compute_loss(model_output: 'torch.Tensor', gt: 'torch.Tensor', seq_len: 'torch.Tensor') ->torch.Tensor:
        """Compute categorical cross-entropy loss for the model.
        Sequences are masked after the EOS character.

        Args:
        ----
            gt: the encoded tensor with gt labels
            model_output: predicted logits of the model
            seq_len: lengths of each gt word inside the batch

        Returns:
        -------
            The loss of the model on the batch
        """
        input_len = model_output.shape[1]
        seq_len = seq_len + 1
        cce = F.cross_entropy(model_output[:, :-1, :].permute(0, 2, 1), gt[:, 1:], reduction='none')
        mask_2d = torch.arange(input_len - 1, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0
        ce_loss = cce.sum(1) / seq_len
        return ce_loss.mean()

    def forward(self, x: 'torch.Tensor', target: 'Optional[List[str]]'=None, return_model_output: 'bool'=False, return_preds: 'bool'=False) ->Dict[str, Any]:
        """Call function for training

        Args:
        ----
            x: images
            target: list of str labels
            return_model_output: if True, return logits
            return_preds: if True, decode logits

        Returns:
        -------
            A dictionnary containing eventually loss, logits and predictions.
        """
        features = self.feat_extractor(x)['features']
        b, c, h, w = features.shape
        features = features.view(b, c, h * w).permute((0, 2, 1))
        encoded = self.positional_encoding(features)
        out: 'Dict[str, Any]' = {}
        if self.training and target is None:
            raise ValueError('Need to provide labels during training')
        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt), torch.tensor(_seq_len)
            gt, seq_len = gt, seq_len
            source_mask, target_mask = self.make_source_and_target_mask(encoded, gt)
            output = self.decoder(gt, encoded, source_mask, target_mask)
            logits = self.linear(output)
        else:
            logits = self.decode(encoded)
        logits = _bf16_to_float32(logits)
        if self.exportable:
            out['logits'] = logits
            return out
        if target is not None:
            out['loss'] = self.compute_loss(logits, gt, seq_len)
        if return_model_output:
            out['out_map'] = logits
        if return_preds:
            out['preds'] = self.postprocessor(logits)
        return out

    def decode(self, encoded: 'torch.Tensor') ->torch.Tensor:
        """Decode function for prediction

        Args:
        ----
            encoded: input tensor

        Returns:
        -------
            A Tuple of torch.Tensor: predictions, logits
        """
        b = encoded.size(0)
        ys = torch.full((b, self.max_length), self.vocab_size + 2, dtype=torch.long, device=encoded.device)
        ys[:, 0] = self.vocab_size + 1
        for i in range(self.max_length - 1):
            source_mask, target_mask = self.make_source_and_target_mask(encoded, ys)
            output = self.decoder(ys, encoded, source_mask, target_mask)
            logits = self.linear(output)
            prob = torch.softmax(logits, dim=-1)
            next_token = torch.max(prob, dim=-1).indices
            ys[:, i + 1] = next_token[:, i]
        return logits


class CharEmbedding(nn.Module):
    """Implements the character embedding module

    Args:
    ----
        vocab_size: size of the vocabulary
        d_model: dimension of the model
    """

    def __init__(self, vocab_size: 'int', d_model: 'int'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return math.sqrt(self.d_model) * self.embedding(x)


class PARSeqDecoder(nn.Module):
    """Implements decoder module of the PARSeq model

    Args:
    ----
        d_model: dimension of the model
        num_heads: number of attention heads
        ffd: dimension of the feed forward layer
        ffd_ratio: depth multiplier for the feed forward layer
        dropout: dropout rate
    """

    def __init__(self, d_model: 'int', num_heads: 'int'=12, ffd: 'int'=2048, ffd_ratio: 'int'=4, dropout: 'float'=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.cross_attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.position_feed_forward = PositionwiseFeedForward(d_model, ffd * ffd_ratio, dropout, nn.GELU())
        self.attention_norm = nn.LayerNorm(d_model, eps=1e-05)
        self.cross_attention_norm = nn.LayerNorm(d_model, eps=1e-05)
        self.query_norm = nn.LayerNorm(d_model, eps=1e-05)
        self.content_norm = nn.LayerNorm(d_model, eps=1e-05)
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=1e-05)
        self.output_norm = nn.LayerNorm(d_model, eps=1e-05)
        self.attention_dropout = nn.Dropout(dropout)
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, target, content, memory, target_mask: 'Optional[torch.Tensor]'=None):
        query_norm = self.query_norm(target)
        content_norm = self.content_norm(content)
        target = target.clone() + self.attention_dropout(self.attention(query_norm, content_norm, content_norm, mask=target_mask))
        target = target.clone() + self.cross_attention_dropout(self.cross_attention(self.query_norm(target), memory, memory))
        target = target.clone() + self.feed_forward_dropout(self.position_feed_forward(self.feed_forward_norm(target)))
        return self.output_norm(target)


class _PARSeqPostProcessor(RecognitionPostProcessor):
    """Abstract class to postprocess the raw output of the model

    Args:
    ----
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(self, vocab: 'str') ->None:
        super().__init__(vocab)
        self._embedding = list(vocab) + ['<eos>', '<sos>', '<pad>']


class PARSeqPostProcessor(_PARSeqPostProcessor):
    """Post processor for PARSeq architecture

    Args:
    ----
        vocab: string containing the ordered sequence of supported characters
    """

    def __call__(self, logits: 'torch.Tensor') ->List[Tuple[str, float]]:
        out_idxs = logits.argmax(-1)
        preds_prob = torch.softmax(logits, -1).max(dim=-1)[0]
        word_values = [''.join(self._embedding[idx] for idx in encoded_seq).split('<eos>')[0] for encoded_seq in out_idxs.cpu().numpy()]
        probs = [(preds_prob[i, :len(word)].clip(0, 1).mean().item() if word else 0.0) for i, word in enumerate(word_values)]
        return list(zip(word_values, probs))


class _PARSeq:
    vocab: 'str'
    max_length: 'int'

    def build_target(self, gts: 'List[str]') ->Tuple[np.ndarray, List[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
        ----
            gts: list of ground-truth labels

        Returns:
        -------
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(sequences=gts, vocab=self.vocab, target_size=self.max_length, eos=len(self.vocab), sos=len(self.vocab) + 1, pad=len(self.vocab) + 2)
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


class PARSeq(_PARSeq, nn.Module):
    """Implements a PARSeq architecture as described in `"Scene Text Recognition
    with Permuted Autoregressive Sequence Models" <https://arxiv.org/pdf/2207.06966>`_.
    Slightly modified implementation based on the official Pytorch implementation: <https://github.com/baudm/parseq/tree/main`_.

    Args:
    ----
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        embedding_units: number of embedding units
        max_length: maximum word length handled by the model
        dropout_prob: dropout probability for the decoder
        dec_num_heads: number of attention heads in the decoder
        dec_ff_dim: dimension of the feed forward layer in the decoder
        dec_ffd_ratio: depth multiplier for the feed forward layer in the decoder
        input_shape: input shape of the image
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    def __init__(self, feature_extractor, vocab: 'str', embedding_units: 'int', max_length: 'int'=32, dropout_prob: 'float'=0.1, dec_num_heads: 'int'=12, dec_ff_dim: 'int'=384, dec_ffd_ratio: 'int'=4, input_shape: 'Tuple[int, int, int]'=(3, 32, 128), exportable: 'bool'=False, cfg: 'Optional[Dict[str, Any]]'=None) ->None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length
        self.vocab_size = len(vocab)
        self.rng = np.random.default_rng()
        self.feat_extractor = feature_extractor
        self.decoder = PARSeqDecoder(embedding_units, dec_num_heads, dec_ff_dim, dec_ffd_ratio, dropout_prob)
        self.head = nn.Linear(embedding_units, self.vocab_size + 1)
        self.embed = CharEmbedding(self.vocab_size + 3, embedding_units)
        self.pos_queries = nn.Parameter(torch.Tensor(1, self.max_length + 1, embedding_units))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.postprocessor = PARSeqPostProcessor(vocab=self.vocab)
        nn.init.trunc_normal_(self.pos_queries, std=0.02)
        for n, m in self.named_modules():
            if n.startswith('feat_extractor.'):
                continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def generate_permutations(self, seqlen: 'torch.Tensor') ->torch.Tensor:
        max_num_chars = int(seqlen.max().item())
        perms = [torch.arange(max_num_chars, device=seqlen.device)]
        max_perms = math.factorial(max_num_chars) // 2
        num_gen_perms = min(3, max_perms)
        if max_num_chars < 5:
            if max_num_chars == 4:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)), device=seqlen.device)[selector]
            perm_pool = perm_pool[1:]
            final_perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(final_perms), replace=False)
                final_perms = torch.cat([final_perms, perm_pool[i]])
        else:
            perms.extend([torch.randperm(max_num_chars, device=seqlen.device) for _ in range(num_gen_perms - len(perms))])
            final_perms = torch.stack(perms)
        comp = final_perms.flip(-1)
        final_perms = torch.stack([final_perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        sos_idx = torch.zeros(len(final_perms), 1, device=seqlen.device)
        eos_idx = torch.full((len(final_perms), 1), max_num_chars + 1, device=seqlen.device)
        combined = torch.cat([sos_idx, final_perms + 1, eos_idx], dim=1).int()
        if len(combined) > 1:
            combined[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=seqlen.device)
        return combined

    def generate_permutations_attention_masks(self, permutation: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        sz = permutation.shape[0]
        mask = torch.ones((sz, sz), device=permutation.device)
        for i in range(sz):
            query_idx = permutation[i]
            masked_keys = permutation[i + 1:]
            mask[query_idx, masked_keys] = 0.0
        source_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=permutation.device)] = 0.0
        target_mask = mask[1:, :-1]
        return source_mask.int(), target_mask.int()

    def decode(self, target: 'torch.Tensor', memory: 'torch.Tensor', target_mask: 'Optional[torch.Tensor]'=None, target_query: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Add positional information to the target sequence and pass it through the decoder."""
        batch_size, sequence_length = target.shape
        null_ctx = self.embed(target[:, :1])
        content = self.pos_queries[:, :sequence_length - 1] + self.embed(target[:, 1:])
        content = self.dropout(torch.cat([null_ctx, content], dim=1))
        if target_query is None:
            target_query = self.pos_queries[:, :sequence_length].expand(batch_size, -1, -1)
        target_query = self.dropout(target_query)
        return self.decoder(target_query, content, memory, target_mask)

    def decode_autoregressive(self, features: 'torch.Tensor', max_len: 'Optional[int]'=None) ->torch.Tensor:
        """Generate predictions for the given features."""
        max_length = max_len if max_len is not None else self.max_length
        max_length = min(max_length, self.max_length) + 1
        ys = torch.full((features.size(0), max_length), self.vocab_size + 2, dtype=torch.long, device=features.device)
        ys[:, 0] = self.vocab_size + 1
        pos_queries = self.pos_queries[:, :max_length].expand(features.size(0), -1, -1)
        query_mask = torch.tril(torch.ones((max_length, max_length), device=features.device), diagonal=0).int()
        pos_logits = []
        for i in range(max_length):
            tgt_out = self.decode(ys[:, :i + 1], features, query_mask[i:i + 1, :i + 1], target_query=pos_queries[:, i:i + 1])
            pos_prob = self.head(tgt_out)
            pos_logits.append(pos_prob)
            if i + 1 < max_length:
                ys[:, i + 1] = pos_prob.squeeze().argmax(-1)
                if not self.exportable and max_len is None and (ys == self.vocab_size).any(dim=-1).all():
                    break
        logits = torch.cat(pos_logits, dim=1)
        query_mask[torch.triu(torch.ones(max_length, max_length, dtype=torch.bool, device=features.device), 2)] = 1
        sos = torch.full((features.size(0), 1), self.vocab_size + 1, dtype=torch.long, device=features.device)
        ys = torch.cat([sos, logits[:, :-1].argmax(-1)], dim=1)
        target_pad_mask = ~((ys == self.vocab_size).int().cumsum(-1) > 0).unsqueeze(1).unsqueeze(1)
        mask = (target_pad_mask.bool() & query_mask[:, :ys.shape[1]].bool()).int()
        logits = self.head(self.decode(ys, features, mask, target_query=pos_queries))
        return logits

    def forward(self, x: 'torch.Tensor', target: 'Optional[List[str]]'=None, return_model_output: 'bool'=False, return_preds: 'bool'=False) ->Dict[str, Any]:
        features = self.feat_extractor(x)['features']
        features = features[:, 1:, :]
        if self.training and target is None:
            raise ValueError('Need to provide labels during training')
        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long), torch.tensor(_seq_len)
            gt = gt[:, :int(seq_len.max().item()) + 2]
            if self.training:
                tgt_perms = self.generate_permutations(seq_len)
                gt_in = gt[:, :-1]
                gt_out = gt[:, 1:]
                padding_mask = ~(((gt_in == self.vocab_size + 2) | (gt_in == self.vocab_size)).int().cumsum(-1) > 0).unsqueeze(1).unsqueeze(1)
                loss = torch.tensor(0.0, device=features.device)
                loss_numel: 'Union[int, float]' = 0
                n = (gt_out != self.vocab_size + 2).sum().item()
                for i, perm in enumerate(tgt_perms):
                    _, target_mask = self.generate_permutations_attention_masks(perm)
                    mask = (target_mask.bool() & padding_mask.bool()).int()
                    logits = self.head(self.decode(gt_in, features, mask)).flatten(end_dim=1)
                    loss += n * F.cross_entropy(logits, gt_out.flatten(), ignore_index=self.vocab_size + 2)
                    loss_numel += n
                    if i == 1:
                        gt_out = torch.where(gt_out == self.vocab_size, self.vocab_size + 2, gt_out)
                        n = (gt_out != self.vocab_size + 2).sum().item()
                loss /= loss_numel
            else:
                gt = gt[:, 1:]
                max_len = gt.shape[1] - 1
                logits = self.decode_autoregressive(features, max_len)
                loss = F.cross_entropy(logits.flatten(end_dim=1), gt.flatten(), ignore_index=self.vocab_size + 2)
        else:
            logits = self.decode_autoregressive(features)
        logits = _bf16_to_float32(logits)
        out: 'Dict[str, Any]' = {}
        if self.exportable:
            out['logits'] = logits
            return out
        if return_model_output:
            out['out_map'] = logits
        if target is None or return_preds:
            out['preds'] = self.postprocessor(logits)
        if target is not None:
            out['loss'] = loss
        return out


def merge_strings(a: 'str', b: 'str', dil_factor: 'float') ->str:
    """Merges 2 character sequences in the best way to maximize the alignment of their overlapping characters.

    Args:
    ----
        a: first char seq, suffix should be similar to b's prefix.
        b: second char seq, prefix should be similar to a's suffix.
        dil_factor: dilation factor of the boxes to overlap, should be > 1. This parameter is
            only used when the mother sequence is splitted on a character repetition

    Returns:
    -------
        A merged character sequence.

    Example::
        >>> from doctr.model.recognition.utils import merge_sequences
        >>> merge_sequences('abcd', 'cdefgh', 1.4)
        'abcdefgh'
        >>> merge_sequences('abcdi', 'cdefgh', 1.4)
        'abcdefgh'
    """
    seq_len = min(len(a), len(b))
    if seq_len == 0:
        return b if len(a) == 0 else a
    min_score, index = 1.0, 0
    scores = [(Levenshtein.distance(a[-i:], b[:i], processor=None) / i) for i in range(1, seq_len + 1)]
    if len(scores) > 1 and (scores[0], scores[1]) == (0, 0):
        n_overlap = round(len(b) * (dil_factor - 1) / dil_factor)
        n_zeros = sum(val == 0 for val in scores)
        min_score, index = 0, min(n_zeros, n_overlap)
    else:
        for i, score in enumerate(scores):
            if score < min_score:
                min_score, index = score, i + 1
    if index == 0:
        return a + b
    return a[:-1] + b[index - 1:]


def merge_multi_strings(seq_list: 'List[str]', dil_factor: 'float') ->str:
    """Recursively merges consecutive string sequences with overlapping characters.

    Args:
    ----
        seq_list: list of sequences to merge. Sequences need to be ordered from left to right.
        dil_factor: dilation factor of the boxes to overlap, should be > 1. This parameter is
            only used when the mother sequence is splitted on a character repetition

    Returns:
    -------
        A merged character sequence

    Example::
        >>> from doctr.model.recognition.utils import merge_multi_sequences
        >>> merge_multi_sequences(['abc', 'bcdef', 'difghi', 'aijkl'], 1.4)
        'abcdefghijkl'
    """

    def _recursive_merge(a: 'str', seq_list: 'List[str]', dil_factor: 'float') ->str:
        if len(seq_list) == 1:
            return merge_strings(a, seq_list[0], dil_factor)
        return _recursive_merge(merge_strings(a, seq_list[0], dil_factor), seq_list[1:], dil_factor)
    return _recursive_merge('', seq_list, dil_factor)


def remap_preds(preds: 'List[Tuple[str, float]]', crop_map: 'List[Union[int, Tuple[int, int]]]', dilation: 'float') ->List[Tuple[str, float]]:
    remapped_out = []
    for _idx in crop_map:
        if isinstance(_idx, int):
            remapped_out.append(preds[_idx])
        else:
            vals, probs = zip(*preds[_idx[0]:_idx[1]])
            remapped_out.append((merge_multi_strings(vals, dilation), min(probs)))
    return remapped_out


def split_crops(crops: 'List[np.ndarray]', max_ratio: 'float', target_ratio: 'int', dilation: 'float', channels_last: 'bool'=True) ->Tuple[List[np.ndarray], List[Union[int, Tuple[int, int]]], bool]:
    """Chunk crops horizontally to match a given aspect ratio

    Args:
    ----
        crops: list of numpy array of shape (H, W, 3) if channels_last or (3, H, W) otherwise
        max_ratio: the maximum aspect ratio that won't trigger the chunk
        target_ratio: when crops are chunked, they will be chunked to match this aspect ratio
        dilation: the width dilation of final chunks (to provide some overlaps)
        channels_last: whether the numpy array has dimensions in channels last order

    Returns:
    -------
        a tuple with the new crops, their mapping, and a boolean specifying whether any remap is required
    """
    _remap_required = False
    crop_map: 'List[Union[int, Tuple[int, int]]]' = []
    new_crops: 'List[np.ndarray]' = []
    for crop in crops:
        h, w = crop.shape[:2] if channels_last else crop.shape[-2:]
        aspect_ratio = w / h
        if aspect_ratio > max_ratio:
            num_subcrops = int(aspect_ratio // target_ratio)
            width = dilation * w / num_subcrops
            centers = [(w / num_subcrops * (1 / 2 + idx)) for idx in range(num_subcrops)]
            if channels_last:
                _crops = [crop[:, max(0, int(round(center - width / 2))):min(w - 1, int(round(center + width / 2))), :] for center in centers]
            else:
                _crops = [crop[:, :, max(0, int(round(center - width / 2))):min(w - 1, int(round(center + width / 2)))] for center in centers]
            _crops = [crop for crop in _crops if all(s > 0 for s in crop.shape)]
            crop_map.append((len(new_crops), len(new_crops) + len(_crops)))
            new_crops.extend(_crops)
            _remap_required = True
        else:
            crop_map.append(len(new_crops))
            new_crops.append(crop)
    return new_crops, crop_map, _remap_required


class RecognitionPredictor(nn.Module):
    """Implements an object able to identify character sequences in images

    Args:
    ----
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
        split_wide_crops: wether to use crop splitting for high aspect ratio crops
    """

    def __init__(self, pre_processor: 'PreProcessor', model: 'nn.Module', split_wide_crops: 'bool'=True) ->None:
        super().__init__()
        self.pre_processor = pre_processor
        self.model = model.eval()
        self.split_wide_crops = split_wide_crops
        self.critical_ar = 8
        self.dil_factor = 1.4
        self.target_ar = 6

    @torch.inference_mode()
    def forward(self, crops: 'Sequence[Union[np.ndarray, torch.Tensor]]', **kwargs: Any) ->List[Tuple[str, float]]:
        if len(crops) == 0:
            return []
        if any(crop.ndim != 3 for crop in crops):
            raise ValueError('incorrect input shape: all crops are expected to be multi-channel 2D images.')
        remapped = False
        if self.split_wide_crops:
            new_crops, crop_map, remapped = split_crops(crops, self.critical_ar, self.target_ar, self.dil_factor, isinstance(crops[0], np.ndarray))
            if remapped:
                crops = new_crops
        processed_batches = self.pre_processor(crops)
        _params = next(self.model.parameters())
        self.model, processed_batches = set_device_and_dtype(self.model, processed_batches, _params.device, _params.dtype)
        raw = [self.model(batch, return_preds=True, **kwargs)['preds'] for batch in processed_batches]
        out = [charseq for batch in raw for charseq in batch]
        if self.split_wide_crops and remapped:
            out = remap_preds(out, crop_map, self.dil_factor)
        return out


class SAREncoder(nn.Module):

    def __init__(self, in_feats: 'int', rnn_units: 'int', dropout_prob: 'float'=0.0) ->None:
        super().__init__()
        self.rnn = nn.LSTM(in_feats, rnn_units, 2, batch_first=True, dropout=dropout_prob)
        self.linear = nn.Linear(rnn_units, rnn_units)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        encoded = self.rnn(x)[0]
        return self.linear(encoded[:, -1, :])


class AttentionModule(nn.Module):

    def __init__(self, feat_chans: 'int', state_chans: 'int', attention_units: 'int') ->None:
        super().__init__()
        self.feat_conv = nn.Conv2d(feat_chans, attention_units, kernel_size=3, padding=1)
        self.state_conv = nn.Conv2d(state_chans, attention_units, kernel_size=1, bias=False)
        self.attention_projector = nn.Conv2d(attention_units, 1, kernel_size=1, bias=False)

    def forward(self, features: 'torch.Tensor', hidden_state: 'torch.Tensor') ->torch.Tensor:
        H_f, W_f = features.shape[2:]
        feat_projection = self.feat_conv(features)
        hidden_state = hidden_state.view(hidden_state.size(0), hidden_state.size(1), 1, 1)
        state_projection = self.state_conv(hidden_state)
        state_projection = state_projection.expand(-1, -1, H_f, W_f)
        attention_weights = torch.tanh(feat_projection + state_projection)
        attention_weights = self.attention_projector(attention_weights)
        B, C, H, W = attention_weights.size()
        attention_weights = torch.softmax(attention_weights.view(B, -1), dim=-1).view(B, C, H, W)
        return (features * attention_weights).sum(dim=(2, 3))


class SARDecoder(nn.Module):
    """Implements decoder module of the SAR model

    Args:
    ----
        rnn_units: number of hidden units in recurrent cells
        max_length: maximum length of a sequence
        vocab_size: number of classes in the model alphabet
        embedding_units: number of hidden embedding units
        attention_units: number of hidden attention units

    """

    def __init__(self, rnn_units: 'int', max_length: 'int', vocab_size: 'int', embedding_units: 'int', attention_units: 'int', feat_chans: 'int'=512, dropout_prob: 'float'=0.0) ->None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed = nn.Linear(self.vocab_size + 1, embedding_units)
        self.embed_tgt = nn.Embedding(embedding_units, self.vocab_size + 1)
        self.attention_module = AttentionModule(feat_chans, rnn_units, attention_units)
        self.lstm_cell = nn.LSTMCell(rnn_units, rnn_units)
        self.output_dense = nn.Linear(2 * rnn_units, self.vocab_size + 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, features: 'torch.Tensor', holistic: 'torch.Tensor', gt: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        if gt is not None:
            gt_embedding = self.embed_tgt(gt)
        logits_list: 'List[torch.Tensor]' = []
        for t in range(self.max_length + 1):
            if t == 0:
                hidden_state_init = cell_state_init = torch.zeros(features.size(0), features.size(1), device=features.device, dtype=features.dtype)
                hidden_state, cell_state = hidden_state_init, cell_state_init
                prev_symbol = holistic
            elif t == 1:
                prev_symbol = torch.zeros(features.size(0), self.vocab_size + 1, device=features.device, dtype=features.dtype)
                prev_symbol = self.embed(prev_symbol)
            elif gt is not None and self.training:
                prev_symbol = self.embed(gt_embedding[:, t - 2])
            else:
                index = logits_list[t - 1].argmax(-1)
                prev_symbol = self.embed(self.embed_tgt(index))
            hidden_state_init, cell_state_init = self.lstm_cell(prev_symbol, (hidden_state_init, cell_state_init))
            hidden_state, cell_state = self.lstm_cell(hidden_state_init, (hidden_state, cell_state))
            glimpse = self.attention_module(features, hidden_state)
            logits = torch.cat([hidden_state, glimpse], dim=1)
            logits = self.dropout(logits)
            logits_list.append(self.output_dense(logits))
        return torch.stack(logits_list[1:]).permute(1, 0, 2)


class SARPostProcessor(RecognitionPostProcessor):
    """Post processor for SAR architectures

    Args:
    ----
        vocab: string containing the ordered sequence of supported characters
    """

    def __call__(self, logits: 'torch.Tensor') ->List[Tuple[str, float]]:
        out_idxs = logits.argmax(-1)
        probs = torch.gather(torch.softmax(logits, -1), -1, out_idxs.unsqueeze(-1)).squeeze(-1)
        probs = probs.min(dim=1).values.detach().cpu()
        word_values = [''.join(self._embedding[idx] for idx in encoded_seq).split('<eos>')[0] for encoded_seq in out_idxs.detach().cpu().numpy()]
        return list(zip(word_values, probs.numpy().clip(0, 1).tolist()))


class SAR(nn.Module, RecognitionModel):
    """Implements a SAR architecture as described in `"Show, Attend and Read:A Simple and Strong Baseline for
    Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Args:
    ----
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of hidden units in both encoder and decoder LSTM
        embedding_units: number of embedding units
        attention_units: number of hidden units in attention module
        max_length: maximum word length handled by the model
        dropout_prob: dropout probability of the encoder LSTM
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    def __init__(self, feature_extractor, vocab: 'str', rnn_units: 'int'=512, embedding_units: 'int'=512, attention_units: 'int'=512, max_length: 'int'=30, dropout_prob: 'float'=0.0, input_shape: 'Tuple[int, int, int]'=(3, 32, 128), exportable: 'bool'=False, cfg: 'Optional[Dict[str, Any]]'=None) ->None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length + 1
        self.feat_extractor = feature_extractor
        self.feat_extractor.eval()
        with torch.no_grad():
            out_shape = self.feat_extractor(torch.zeros((1, *input_shape)))['features'].shape
        self.feat_extractor.train()
        self.encoder = SAREncoder(out_shape[1], rnn_units, dropout_prob)
        self.decoder = SARDecoder(rnn_units, self.max_length, len(self.vocab), embedding_units, attention_units, dropout_prob=dropout_prob)
        self.postprocessor = SARPostProcessor(vocab=vocab)
        for n, m in self.named_modules():
            if n.startswith('feat_extractor.'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: 'torch.Tensor', target: 'Optional[List[str]]'=None, return_model_output: 'bool'=False, return_preds: 'bool'=False) ->Dict[str, Any]:
        features = self.feat_extractor(x)['features']
        pooled_features = features.max(dim=-2).values
        pooled_features = pooled_features.permute(0, 2, 1).contiguous()
        encoded = self.encoder(pooled_features)
        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt), torch.tensor(_seq_len)
            gt, seq_len = gt, seq_len
        if self.training and target is None:
            raise ValueError('Need to provide labels during training for teacher forcing')
        decoded_features = _bf16_to_float32(self.decoder(features, encoded, gt=None if target is None else gt))
        out: 'Dict[str, Any]' = {}
        if self.exportable:
            out['logits'] = decoded_features
            return out
        if return_model_output:
            out['out_map'] = decoded_features
        if target is None or return_preds:
            out['preds'] = self.postprocessor(decoded_features)
        if target is not None:
            out['loss'] = self.compute_loss(decoded_features, gt, seq_len)
        return out

    @staticmethod
    def compute_loss(model_output: 'torch.Tensor', gt: 'torch.Tensor', seq_len: 'torch.Tensor') ->torch.Tensor:
        """Compute categorical cross-entropy loss for the model.
        Sequences are masked after the EOS character.

        Args:
        ----
            model_output: predicted logits of the model
            gt: the encoded tensor with gt labels
            seq_len: lengths of each gt word inside the batch

        Returns:
        -------
            The loss of the model on the batch
        """
        input_len = model_output.shape[1]
        seq_len = seq_len + 1
        cce = F.cross_entropy(model_output.permute(0, 2, 1), gt, reduction='none')
        mask_2d = torch.arange(input_len, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0
        ce_loss = cce.sum(1) / seq_len
        return ce_loss.mean()


class _ViTSTRPostProcessor(RecognitionPostProcessor):
    """Abstract class to postprocess the raw output of the model

    Args:
    ----
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(self, vocab: 'str') ->None:
        super().__init__(vocab)
        self._embedding = list(vocab) + ['<eos>', '<sos>']


class ViTSTRPostProcessor(_ViTSTRPostProcessor):
    """Post processor for ViTSTR architecture

    Args:
    ----
        vocab: string containing the ordered sequence of supported characters
    """

    def __call__(self, logits: 'torch.Tensor') ->List[Tuple[str, float]]:
        out_idxs = logits.argmax(-1)
        preds_prob = torch.softmax(logits, -1).max(dim=-1)[0]
        word_values = [''.join(self._embedding[idx] for idx in encoded_seq).split('<eos>')[0] for encoded_seq in out_idxs.cpu().numpy()]
        probs = [(preds_prob[i, :len(word)].clip(0, 1).mean().item() if word else 0.0) for i, word in enumerate(word_values)]
        return list(zip(word_values, probs))


class _ViTSTR:
    vocab: 'str'
    max_length: 'int'

    def build_target(self, gts: 'List[str]') ->Tuple[np.ndarray, List[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
        ----
            gts: list of ground-truth labels

        Returns:
        -------
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(sequences=gts, vocab=self.vocab, target_size=self.max_length, eos=len(self.vocab), sos=len(self.vocab) + 1)
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


class ViTSTR(_ViTSTR, nn.Module):
    """Implements a ViTSTR architecture as described in `"Vision Transformer for Fast and
    Efficient Scene Text Recognition" <https://arxiv.org/pdf/2105.08582.pdf>`_.

    Args:
    ----
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        embedding_units: number of embedding units
        max_length: maximum word length handled by the model
        dropout_prob: dropout probability of the encoder LSTM
        input_shape: input shape of the image
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    def __init__(self, feature_extractor, vocab: 'str', embedding_units: 'int', max_length: 'int'=32, input_shape: 'Tuple[int, int, int]'=(3, 32, 128), exportable: 'bool'=False, cfg: 'Optional[Dict[str, Any]]'=None) ->None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length + 2
        self.feat_extractor = feature_extractor
        self.head = nn.Linear(embedding_units, len(self.vocab) + 1)
        self.postprocessor = ViTSTRPostProcessor(vocab=self.vocab)

    def forward(self, x: 'torch.Tensor', target: 'Optional[List[str]]'=None, return_model_output: 'bool'=False, return_preds: 'bool'=False) ->Dict[str, Any]:
        features = self.feat_extractor(x)['features']
        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt), torch.tensor(_seq_len)
            gt, seq_len = gt, seq_len
        if self.training and target is None:
            raise ValueError('Need to provide labels during training')
        features = features[:, :self.max_length]
        B, N, E = features.size()
        features = features.reshape(B * N, E)
        logits = self.head(features).view(B, N, len(self.vocab) + 1)
        decoded_features = _bf16_to_float32(logits[:, 1:])
        out: 'Dict[str, Any]' = {}
        if self.exportable:
            out['logits'] = decoded_features
            return out
        if return_model_output:
            out['out_map'] = decoded_features
        if target is None or return_preds:
            out['preds'] = self.postprocessor(decoded_features)
        if target is not None:
            out['loss'] = self.compute_loss(decoded_features, gt, seq_len)
        return out

    @staticmethod
    def compute_loss(model_output: 'torch.Tensor', gt: 'torch.Tensor', seq_len: 'torch.Tensor') ->torch.Tensor:
        """Compute categorical cross-entropy loss for the model.
        Sequences are masked after the EOS character.

        Args:
        ----
            model_output: predicted logits of the model
            gt: the encoded tensor with gt labels
            seq_len: lengths of each gt word inside the batch

        Returns:
        -------
            The loss of the model on the batch
        """
        input_len = model_output.shape[1]
        seq_len = seq_len + 1
        cce = F.cross_entropy(model_output.permute(0, 2, 1), gt[:, 1:], reduction='none')
        mask_2d = torch.arange(input_len, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0
        ce_loss = cce.sum(1) / seq_len
        return ce_loss.mean()


class GaussianNoise(torch.nn.Module):
    """Adds Gaussian Noise to the input tensor

    >>> import torch
    >>> from doctr.transforms import GaussianNoise
    >>> transfo = GaussianNoise(0., 1.)
    >>> out = transfo(torch.rand((3, 224, 224)))

    Args:
    ----
        mean : mean of the gaussian distribution
        std : std of the gaussian distribution
    """

    def __init__(self, mean: 'float'=0.0, std: 'float'=1.0) ->None:
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        noise = self.mean + 2 * self.std * torch.rand(x.shape, device=x.device) - self.std
        if x.dtype == torch.uint8:
            return (x + 255 * noise).round().clamp(0, 255)
        else:
            return (x + noise).clamp(0, 1)

    def extra_repr(self) ->str:
        return f'mean={self.mean}, std={self.std}'


class ChannelShuffle(torch.nn.Module):
    """Randomly shuffle channel order of a given image"""

    def __init__(self):
        super().__init__()

    def forward(self, img: 'torch.Tensor') ->torch.Tensor:
        chan_order = torch.rand(img.shape[0]).argsort()
        return img[chan_order]


def expand_line(line: 'np.ndarray', target_shape: 'Tuple[int, int]') ->Tuple[float, float]:
    """Expands a 2-point line, so that the first is on the edge. In other terms, we extend the line in
    the same direction until we meet one of the edges.

    Args:
    ----
        line: array of shape (2, 2) of the point supposed to be on one edge, and the shadow tip.
        target_shape: the desired mask shape

    Returns:
    -------
        2D coordinates of the first point once we extended the line (on one of the edges)
    """
    if any(coord == 0 or coord == size for coord, size in zip(line[0], target_shape[::-1])):
        return line[0]
    _tmp = line[1] - line[0]
    _direction = _tmp > 0
    _flat = _tmp == 0
    if _tmp[0] == 0:
        solutions = [(line[0, 0], 0), (line[0, 0], target_shape[0])]
    elif _tmp[1] == 0:
        solutions = [(0, line[0, 1]), (target_shape[1], line[0, 1])]
    else:
        alpha = _tmp[1] / _tmp[0]
        beta = line[1, 1] - alpha * line[1, 0]
        solutions = [(0, beta), (-beta / alpha, 0), (target_shape[1], alpha * target_shape[1] + beta), ((target_shape[0] - beta) / alpha, target_shape[0])]
    for point in solutions:
        if any(val < 0 or val > size for val, size in zip(point, target_shape[::-1])):
            continue
        if all(val == ref if _same else val < ref if _dir else val > ref for val, ref, _dir, _same in zip(point, line[1], _direction, _flat)):
            return point
    raise ValueError


def rotate_abs_geoms(geoms: 'np.ndarray', angle: 'float', img_shape: 'Tuple[int, int]', expand: 'bool'=True) ->np.ndarray:
    """Rotate a batch of bounding boxes or polygons by an angle around the
    image center.

    Args:
    ----
        geoms: (N, 4) or (N, 4, 2) array of ABSOLUTE coordinate boxes
        angle: anti-clockwise rotation angle in degrees
        img_shape: the height and width of the image
        expand: whether the image should be padded to avoid information loss

    Returns:
    -------
        A batch of rotated polygons (N, 4, 2)
    """
    polys = np.stack([geoms[:, [0, 1]], geoms[:, [2, 1]], geoms[:, [2, 3]], geoms[:, [0, 3]]], axis=1) if geoms.ndim == 2 else geoms
    polys = polys.astype(np.float32)
    polys[..., 0] -= img_shape[1] / 2
    polys[..., 1] = img_shape[0] / 2 - polys[..., 1]
    rotated_polys = rotate_abs_points(polys.reshape(-1, 2), angle).reshape(-1, 4, 2)
    target_shape = compute_expanded_shape(img_shape, angle) if expand else img_shape
    rotated_polys[..., 0] = (rotated_polys[..., 0] + target_shape[1] / 2).clip(0, target_shape[1])
    rotated_polys[..., 1] = (target_shape[0] / 2 - rotated_polys[..., 1]).clip(0, target_shape[0])
    return rotated_polys


def create_shadow_mask(target_shape: 'Tuple[int, int]', min_base_width=0.3, max_tip_width=0.5, max_tip_height=0.3) ->np.ndarray:
    """Creates a random shadow mask

    Args:
    ----
        target_shape: the target shape (H, W)
        min_base_width: the relative minimum shadow base width
        max_tip_width: the relative maximum shadow tip width
        max_tip_height: the relative maximum shadow tip height

    Returns:
    -------
        a numpy ndarray of shape (H, W, 1) with values in the range [0, 1]
    """
    _params = np.random.rand(6)
    base_width = min_base_width + (1 - min_base_width) * _params[0]
    base_center = base_width / 2 + (1 - base_width) * _params[1]
    tip_width = min(_params[2] * base_width * target_shape[0] / target_shape[1], max_tip_width)
    tip_center = tip_width / 2 + (1 - tip_width) * _params[3]
    tip_height = _params[4] * max_tip_height
    tip_mid = tip_height / 2 + (1 - tip_height) * _params[5]
    _order = tip_center < base_center
    contour: 'np.ndarray' = np.array([[base_center - base_width / 2, 0], [base_center + base_width / 2, 0], [tip_center + tip_width / 2, tip_mid + tip_height / 2 if _order else tip_mid - tip_height / 2], [tip_center - tip_width / 2, tip_mid - tip_height / 2 if _order else tip_mid + tip_height / 2]], dtype=np.float32)
    abs_contour: 'np.ndarray' = np.stack((contour[:, 0] * target_shape[1], contour[:, 1] * target_shape[0]), axis=-1).round().astype(np.int32)
    _params = np.random.rand(1)
    rotated_contour = rotate_abs_geoms(abs_contour[None, ...], 360 * _params[0], target_shape, expand=False)[0].round().astype(np.int32)
    quad_idx = int(_params[0] / 0.25)
    if quad_idx % 2 == 0:
        intensity_mask = np.repeat(np.arange(target_shape[0])[:, None], target_shape[1], axis=1) / (target_shape[0] - 1)
        if quad_idx == 0:
            intensity_mask = 1 - intensity_mask
    else:
        intensity_mask = np.repeat(np.arange(target_shape[1])[None, :], target_shape[0], axis=0) / (target_shape[1] - 1)
        if quad_idx == 1:
            intensity_mask = 1 - intensity_mask
    final_contour = rotated_contour.copy()
    final_contour[0] = expand_line(final_contour[[0, 3]], target_shape)
    final_contour[1] = expand_line(final_contour[[1, 2]], target_shape)
    if not np.any(final_contour[0] == final_contour[1]):
        corner_x = 0 if max(final_contour[0, 0], final_contour[1, 0]) < target_shape[1] else target_shape[1]
        corner_y = 0 if max(final_contour[0, 1], final_contour[1, 1]) < target_shape[0] else target_shape[0]
        corner: 'np.ndarray' = np.array([corner_x, corner_y])
        final_contour = np.concatenate((final_contour[:1], corner[None, ...], final_contour[1:]), axis=0)
    mask: 'np.ndarray' = np.zeros((*target_shape, 1), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [final_contour], (255,), lineType=cv2.LINE_AA)[..., 0]
    return (mask / 255).astype(np.float32).clip(0, 1) * intensity_mask.astype(np.float32)


def random_shadow(img: 'torch.Tensor', opacity_range: 'Tuple[float, float]', **kwargs) ->torch.Tensor:
    """Crop and image and associated bboxes

    Args:
    ----
        img: image to modify
        opacity_range: the minimum and maximum desired opacity of the shadow
        **kwargs: additional arguments to pass to `create_shadow_mask`

    Returns:
    -------
        shaded image
    """
    shadow_mask = create_shadow_mask(img.shape[1:], **kwargs)
    opacity = np.random.uniform(*opacity_range)
    shadow_tensor = 1 - torch.from_numpy(shadow_mask[None, ...])
    k = 7 + 2 * int(4 * np.random.rand(1))
    sigma = np.random.uniform(0.5, 5.0)
    shadow_tensor = F.gaussian_blur(shadow_tensor, k, sigma=[sigma, sigma])
    return opacity * shadow_tensor * img + (1 - opacity) * img


class RandomShadow(torch.nn.Module):
    """Adds random shade to the input image

    >>> import torch
    >>> from doctr.transforms import RandomShadow
    >>> transfo = RandomShadow((0., 1.))
    >>> out = transfo(torch.rand((3, 64, 64)))

    Args:
    ----
        opacity_range : minimum and maximum opacity of the shade
    """

    def __init__(self, opacity_range: 'Optional[Tuple[float, float]]'=None) ->None:
        super().__init__()
        self.opacity_range = opacity_range if isinstance(opacity_range, tuple) else (0.2, 0.8)

    def __call__(self, x: 'torch.Tensor') ->torch.Tensor:
        try:
            if x.dtype == torch.uint8:
                return (255 * random_shadow(x.to(dtype=torch.float32) / 255, self.opacity_range)).round().clip(0, 255)
            else:
                return random_shadow(x, self.opacity_range).clip(0, 1)
        except ValueError:
            return x

    def extra_repr(self) ->str:
        return f'opacity_range={self.opacity_range}'


class RandomResize(torch.nn.Module):
    """Randomly resize the input image and align corresponding targets

    >>> import torch
    >>> from doctr.transforms import RandomResize
    >>> transfo = RandomResize((0.3, 0.9), preserve_aspect_ratio=True, symmetric_pad=True, p=0.5)
    >>> out = transfo(torch.rand((3, 64, 64)))

    Args:
    ----
        scale_range: range of the resizing factor for width and height (independently)
        preserve_aspect_ratio: whether to preserve the aspect ratio of the image,
            given a float value, the aspect ratio will be preserved with this probability
        symmetric_pad: whether to symmetrically pad the image,
            given a float value, the symmetric padding will be applied with this probability
        p: probability to apply the transformation
    """

    def __init__(self, scale_range: 'Tuple[float, float]'=(0.3, 0.9), preserve_aspect_ratio: 'Union[bool, float]'=False, symmetric_pad: 'Union[bool, float]'=False, p: 'float'=0.5) ->None:
        super().__init__()
        self.scale_range = scale_range
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.p = p
        self._resize = Resize

    def forward(self, img: 'torch.Tensor', target: 'np.ndarray') ->Tuple[torch.Tensor, np.ndarray]:
        if torch.rand(1) < self.p:
            scale_h = np.random.uniform(*self.scale_range)
            scale_w = np.random.uniform(*self.scale_range)
            new_size = int(img.shape[-2] * scale_h), int(img.shape[-1] * scale_w)
            _img, _target = self._resize(new_size, preserve_aspect_ratio=self.preserve_aspect_ratio if isinstance(self.preserve_aspect_ratio, bool) else bool(torch.rand(1) <= self.symmetric_pad), symmetric_pad=self.symmetric_pad if isinstance(self.symmetric_pad, bool) else bool(torch.rand(1) <= self.symmetric_pad))(img, target)
            return _img, _target
        return img, target

    def extra_repr(self) ->str:
        return f'scale_range={self.scale_range}, preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}, p={self.p}'


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentionModule,
     lambda: ([], {'feat_chans': 4, 'state_chans': 4, 'attention_units': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 1, 1])], {})),
    (ChannelShuffle,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ClassifierHead,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EncoderBlock,
     lambda: ([], {'num_layers': 1, 'num_heads': 4, 'd_model': 4, 'dff': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4])], {})),
    (FASTConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 2, 2])], {})),
    (FastHead,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GaussianNoise,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiHeadAttention,
     lambda: ([], {'num_heads': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (OrientationPredictor,
     lambda: ([], {'pre_processor': 4, 'model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PatchEmbedding,
     lambda: ([], {'input_shape': [4, 4, 4], 'embed_dim': 4, 'patch_size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionwiseFeedForward,
     lambda: ([], {'d_model': 4, 'ffd': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RandomResize,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Resize,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SAREncoder,
     lambda: ([], {'in_feats': 4, 'rnn_units': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (VisionTransformer,
     lambda: ([], {'d_model': 4, 'num_layers': 1, 'num_heads': 4, 'ffd_ratio': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

