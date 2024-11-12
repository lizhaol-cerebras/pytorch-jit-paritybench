
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


import torch.utils.cpp_extension


import torch


import logging


from typing import List


from typing import Optional


import torch.utils.data


import numpy as np


import time


from collections import defaultdict


import torchvision


from typing import ClassVar


from typing import Tuple


import typing as t


import torchvision.models


from typing import Callable


from typing import Dict


from typing import Set


from typing import Type


import warnings


import functools


import math


import copy


import random


import torchvision.transforms.functional


import itertools


class DecoderModule(torch.nn.Module):

    def __init__(self, cif_meta, caf_meta):
        super().__init__()
        self.cpp_decoder = torch.classes.openpifpaf_decoder.CifCaf(len(cif_meta.keypoints), torch.LongTensor(caf_meta.skeleton) - 1)
        self.cif_stride = cif_meta.stride
        self.caf_stride = caf_meta.stride

    def forward(self, cif_field, caf_field):
        return self.cpp_decoder.call(cif_field, self.cif_stride, caf_field, self.caf_stride)


class EncoderDecoder(torch.nn.Module):

    def __init__(self, traced_encoder, decoder):
        super().__init__()
        self.traced_encoder = traced_encoder
        self.decoder = decoder

    def forward(self, x):
        cif_head_batch, caf_head_batch = self.traced_encoder(x)
        o = [self.decoder(cif_head, caf_head) for cif_head, caf_head in zip(cif_head_batch, caf_head_batch)]
        return o


LOG = logging.getLogger(__name__)


class BaseNetwork(torch.nn.Module):
    """Common base network.

    :param name: a short name for the base network, e.g. resnet50
    :param stride: total stride from input to output
    :param out_features: number of output features
    """

    def __init__(self, name: 'str', *, stride: int, out_features: int):
        super().__init__()
        self.name = name
        self.stride = stride
        self.out_features = out_features
        LOG.info('%s: stride = %d, output features = %d', name, stride, out_features)

    @classmethod
    def cli(cls, parser: 'argparse.ArgumentParser'):
        """Command line interface (CLI) to extend argument parser."""

    @classmethod
    def configure(cls, args: 'argparse.Namespace'):
        """Take the parsed argument parser output and configure class variables."""

    def forward(self, x):
        raise NotImplementedError


class ShuffleNetV2(BaseNetwork):
    pretrained = True

    def __init__(self, name, torchvision_shufflenetv2, out_features=2048):
        super().__init__(name, stride=16, out_features=out_features)
        base_vision = torchvision_shufflenetv2(self.pretrained)
        self.conv1 = base_vision.conv1
        self.stage2 = base_vision.stage2
        self.stage3 = base_vision.stage3
        self.stage4 = base_vision.stage4
        self.conv5 = base_vision.conv5

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x

    @classmethod
    def cli(cls, parser: 'argparse.ArgumentParser'):
        group = parser.add_argument_group('ShuffleNetv2')
        assert cls.pretrained
        group.add_argument('--shufflenetv2-no-pretrain', dest='shufflenetv2_pretrained', default=True, action='store_false', help='use randomly initialized models')

    @classmethod
    def configure(cls, args: 'argparse.Namespace'):
        cls.pretrained = args.shufflenetv2_pretrained


class Resnet(BaseNetwork):
    pretrained = True
    pool0_stride = 0
    input_conv_stride = 2
    input_conv2_stride = 0
    remove_last_block = False
    block5_dilation = 1

    def __init__(self, name, torchvision_resnet, out_features=2048):
        modules = list(torchvision_resnet(self.pretrained).children())
        stride = 32
        input_modules = modules[:4]
        if self.pool0_stride:
            if self.pool0_stride != 2:
                input_modules[3].stride = torch.nn.modules.utils._pair(self.pool0_stride)
                stride = int(stride * 2 / self.pool0_stride)
        else:
            input_modules.pop(3)
            stride //= 2
        if self.input_conv_stride != 2:
            input_modules[0].stride = torch.nn.modules.utils._pair(self.input_conv_stride)
            stride = int(stride * 2 / self.input_conv_stride)
        if self.input_conv2_stride:
            assert not self.pool0_stride
            channels = input_modules[0].out_channels
            conv2 = torch.nn.Sequential(torch.nn.Conv2d(channels, channels, 3, 2, 1, bias=False), torch.nn.BatchNorm2d(channels), torch.nn.ReLU(inplace=True))
            input_modules.append(conv2)
            stride *= 2
            LOG.debug('replaced max pool with [3x3 conv, bn, relu] with %d channels', channels)
        block5 = modules[7]
        if self.remove_last_block:
            block5 = None
            stride //= 2
            out_features //= 2
        if self.block5_dilation != 1:
            stride //= 2
            for m in block5.modules():
                if not isinstance(m, torch.nn.Conv2d):
                    continue
                m.stride = torch.nn.modules.utils._pair(1)
                if m.kernel_size[0] == 1:
                    continue
                m.dilation = torch.nn.modules.utils._pair(self.block5_dilation)
                padding = (m.kernel_size[0] - 1) // 2 * self.block5_dilation
                m.padding = torch.nn.modules.utils._pair(padding)
        super().__init__(name, stride=stride, out_features=out_features)
        self.input_block = torch.nn.Sequential(*input_modules)
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]
        self.block5 = block5

    def forward(self, x):
        x = self.input_block(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

    @classmethod
    def cli(cls, parser: 'argparse.ArgumentParser'):
        group = parser.add_argument_group('ResNet')
        assert cls.pretrained
        group.add_argument('--resnet-no-pretrain', dest='resnet_pretrained', default=True, action='store_false', help='use randomly initialized models')
        group.add_argument('--resnet-pool0-stride', default=cls.pool0_stride, type=int, help='stride of zero removes the pooling op')
        group.add_argument('--resnet-input-conv-stride', default=cls.input_conv_stride, type=int, help='stride of the input convolution')
        group.add_argument('--resnet-input-conv2-stride', default=cls.input_conv2_stride, type=int, help='stride of the optional 2nd input convolution')
        group.add_argument('--resnet-block5-dilation', default=cls.block5_dilation, type=int, help='use dilated convs in block5')
        assert not cls.remove_last_block
        group.add_argument('--resnet-remove-last-block', default=False, action='store_true', help='create a network without the last block')

    @classmethod
    def configure(cls, args: 'argparse.Namespace'):
        cls.pretrained = args.resnet_pretrained
        cls.pool0_stride = args.resnet_pool0_stride
        cls.input_conv_stride = args.resnet_input_conv_stride
        cls.input_conv2_stride = args.resnet_input_conv2_stride
        cls.block5_dilation = args.resnet_block5_dilation
        cls.remove_last_block = args.resnet_remove_last_block


class InvertedResidualK(torch.nn.Module):
    """Based on torchvision.models.shufflenet.InvertedResidual."""

    def __init__(self, inp, oup, first_in_stage, *, stride=1, layer_norm, non_linearity, dilation=1, kernel_size=3):
        super().__init__()
        assert (stride != 1 or dilation != 1 or inp != oup) or not first_in_stage
        LOG.debug('InvResK: %d %d %s, stride=%d, dilation=%d', inp, oup, first_in_stage, stride, dilation)
        self.first_in_stage = first_in_stage
        branch_features = oup // 2
        padding = (kernel_size - 1) // 2 * dilation
        self.branch1 = None
        if self.first_in_stage:
            self.branch1 = torch.nn.Sequential(self.depthwise_conv(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation), layer_norm(inp), torch.nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False), layer_norm(branch_features), non_linearity())
        self.branch2 = torch.nn.Sequential(torch.nn.Conv2d(inp if first_in_stage else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), layer_norm(branch_features), non_linearity(), self.depthwise_conv(branch_features, branch_features, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation), layer_norm(branch_features), torch.nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), layer_norm(branch_features), non_linearity())

    @staticmethod
    def depthwise_conv(in_f, out_f, kernel_size, stride=1, padding=0, bias=False, dilation=1):
        return torch.nn.Conv2d(in_f, out_f, kernel_size, stride, padding, bias=bias, groups=in_f, dilation=dilation)

    def forward(self, x):
        if self.branch1 is None:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = torchvision.models.shufflenetv2.channel_shuffle(out, 2)
        return out


class ShuffleNetV2K(BaseNetwork):
    """Based on torchvision.models.ShuffleNetV2 where
    the kernel size in stages 2,3,4 is 5 instead of 3."""
    input_conv2_stride = 0
    input_conv2_outchannels = None
    layer_norm = None
    stage4_dilation = 1
    kernel_width = 5
    conv5_as_stage = False
    non_linearity = None

    def __init__(self, name, stages_repeats, stages_out_channels):
        layer_norm = ShuffleNetV2K.layer_norm
        if layer_norm is None:
            layer_norm = torch.nn.BatchNorm2d
        non_linearity = ShuffleNetV2K.non_linearity
        if non_linearity is None:
            non_linearity = lambda : torch.nn.ReLU(inplace=True)
        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        _stage_out_channels = stages_out_channels
        stride = 16
        input_modules = []
        input_channels = 3
        output_channels = _stage_out_channels[0]
        conv1 = torch.nn.Sequential(torch.nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False), layer_norm(output_channels), non_linearity())
        input_modules.append(conv1)
        input_channels = output_channels
        if self.input_conv2_stride:
            output_channels = self.input_conv2_outchannels or input_channels
            conv2 = torch.nn.Sequential(torch.nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False), layer_norm(output_channels), non_linearity())
            input_modules.append(conv2)
            stride *= 2
            input_channels = output_channels
            LOG.debug('replaced max pool with [3x3 conv, bn, relu]')
        stages = []
        for repeats, output_channels, dilation in zip(stages_repeats, _stage_out_channels[1:], [1, 1, self.stage4_dilation]):
            stage_stride = 2 if dilation == 1 else 1
            stride = int(stride * stage_stride / 2)
            seq = [InvertedResidualK(input_channels, output_channels, True, kernel_size=self.kernel_width, layer_norm=layer_norm, non_linearity=non_linearity, dilation=dilation, stride=stage_stride)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidualK(output_channels, output_channels, False, kernel_size=self.kernel_width, layer_norm=layer_norm, non_linearity=non_linearity, dilation=dilation))
            stages.append(torch.nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = _stage_out_channels[-1]
        if self.conv5_as_stage:
            use_first_in_stage = input_channels != output_channels
            conv5 = torch.nn.Sequential(InvertedResidualK(input_channels, output_channels, use_first_in_stage, kernel_size=self.kernel_width, layer_norm=layer_norm, non_linearity=non_linearity, dilation=self.stage4_dilation), InvertedResidualK(output_channels, output_channels, False, kernel_size=self.kernel_width, layer_norm=layer_norm, non_linearity=non_linearity, dilation=self.stage4_dilation))
        else:
            conv5 = torch.nn.Sequential(torch.nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False), layer_norm(output_channels), non_linearity())
        super().__init__(name, stride=stride, out_features=output_channels)
        self.input_block = torch.nn.Sequential(*input_modules)
        self.stage2 = stages[0]
        self.stage3 = stages[1]
        self.stage4 = stages[2]
        self.conv5 = conv5

    def forward(self, x):
        x = self.input_block(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x

    @classmethod
    def cli(cls, parser: 'argparse.ArgumentParser'):
        group = parser.add_argument_group('shufflenetv2k')
        group.add_argument('--shufflenetv2k-input-conv2-stride', default=cls.input_conv2_stride, type=int, help='stride of the optional 2nd input convolution')
        group.add_argument('--shufflenetv2k-input-conv2-outchannels', default=cls.input_conv2_outchannels, type=int, help='out channels of the optional 2nd input convolution')
        group.add_argument('--shufflenetv2k-stage4-dilation', default=cls.stage4_dilation, type=int, help='dilation factor of stage 4')
        group.add_argument('--shufflenetv2k-kernel', default=cls.kernel_width, type=int, help='kernel width')
        assert not cls.conv5_as_stage
        group.add_argument('--shufflenetv2k-conv5-as-stage', default=False, action='store_true')
        layer_norm_group = group.add_mutually_exclusive_group()
        layer_norm_group.add_argument('--shufflenetv2k-instance-norm', default=False, action='store_true')
        layer_norm_group.add_argument('--shufflenetv2k-group-norm', default=False, action='store_true')
        non_linearity_group = group.add_mutually_exclusive_group()
        non_linearity_group.add_argument('--shufflenetv2k-leaky-relu', default=False, action='store_true')

    @classmethod
    def configure(cls, args: 'argparse.Namespace'):
        cls.input_conv2_stride = args.shufflenetv2k_input_conv2_stride
        cls.input_conv2_outchannels = args.shufflenetv2k_input_conv2_outchannels
        cls.stage4_dilation = args.shufflenetv2k_stage4_dilation
        cls.kernel_width = args.shufflenetv2k_kernel
        cls.conv5_as_stage = args.shufflenetv2k_conv5_as_stage
        if args.shufflenetv2k_instance_norm:
            cls.layer_norm = lambda x: torch.nn.InstanceNorm2d(x, affine=True, track_running_stats=True)
        if args.shufflenetv2k_group_norm:
            cls.layer_norm = lambda x: torch.nn.GroupNorm((32 if x % 32 == 0 else 29) if x > 100 else 4, x)
        if args.shufflenetv2k_leaky_relu:
            cls.non_linearity = lambda : torch.nn.LeakyReLU(inplace=True)


class MobileNetV2(BaseNetwork):
    pretrained = True

    def __init__(self, name, torchvision_mobilenetv2, out_features=1280):
        super().__init__(name, stride=32, out_features=out_features)
        base_vision = torchvision_mobilenetv2(self.pretrained)
        self.backbone = list(base_vision.children())[0]

    def forward(self, x):
        x = self.backbone(x)
        return x

    @classmethod
    def cli(cls, parser: 'argparse.ArgumentParser'):
        group = parser.add_argument_group('MobileNetV2')
        assert cls.pretrained
        group.add_argument('--mobilenetv2-no-pretrain', dest='mobilenetv2_pretrained', default=True, action='store_false', help='use randomly initialized models')

    @classmethod
    def configure(cls, args: 'argparse.Namespace'):
        cls.pretrained = args.mobilenetv2_pretrained


class MobileNetV3(BaseNetwork):
    pretrained = True

    def __init__(self, name, torchvision_mobilenetv3, out_features=960):
        super().__init__(name, stride=16, out_features=out_features)
        base_vision = torchvision_mobilenetv3(self.pretrained)
        self.backbone = list(base_vision.children())[0]
        input_conv = list(self.backbone)[0][0]
        input_conv.stride = torch.nn.modules.utils._pair(1)

    def forward(self, x):
        x = self.backbone(x)
        return x

    @classmethod
    def cli(cls, parser: 'argparse.ArgumentParser'):
        group = parser.add_argument_group('MobileNetV3')
        assert cls.pretrained
        group.add_argument('--mobilenetv3-no-pretrain', dest='mobilenetv3_pretrained', default=True, action='store_false', help='use randomly initialized models')

    @classmethod
    def configure(cls, args: 'argparse.Namespace'):
        cls.pretrained = args.mobilenetv3_pretrained


class SqueezeNet(BaseNetwork):
    pretrained = True

    def __init__(self, name, torchvision_squeezenet, out_features=512):
        super().__init__(name, stride=16, out_features=out_features)
        base_vision = torchvision_squeezenet(self.pretrained)
        for m in base_vision.modules():
            if isinstance(m, (torch.nn.MaxPool2d,)) and m.padding != 1:
                LOG.debug('adjusting maxpool2d padding to 1 from padding=%d, kernel=%d, stride=%d', m.padding, m.kernel_size, m.stride)
                m.padding = 1
            if isinstance(m, (torch.nn.Conv2d,)):
                target_padding = (m.kernel_size[0] - 1) // 2
                if m.padding[0] != target_padding:
                    LOG.debug('adjusting conv2d padding to %d (kernel=%d, padding=%d)', target_padding, m.kernel_size, m.padding)
                    m.padding = torch.nn.modules.utils._pair(target_padding)
        self.backbone = list(base_vision.children())[0]

    def forward(self, x):
        x = self.backbone(x)
        return x

    @classmethod
    def cli(cls, parser: 'argparse.ArgumentParser'):
        group = parser.add_argument_group('SqueezeNet')
        assert cls.pretrained
        group.add_argument('--squeezenet-no-pretrain', dest='squeezenet_pretrained', default=True, action='store_false', help='use randomly initialized models')

    @classmethod
    def configure(cls, args: 'argparse.Namespace'):
        cls.pretrained = args.squeezenet_pretrained


class PifHFlip(torch.nn.Module):

    def __init__(self, keypoints, hflip):
        super().__init__()
        flip_indices = torch.LongTensor([(keypoints.index(hflip[kp_name]) if kp_name in hflip else kp_i) for kp_i, kp_name in enumerate(keypoints)])
        LOG.debug('hflip indices: %s', flip_indices)
        self.register_buffer('flip_indices', flip_indices)

    def forward(self, *args):
        out = []
        for field in args:
            field = torch.index_select(field, 1, self.flip_indices)
            field = torch.flip(field, dims=[len(field.shape) - 1])
            out.append(field)
        out[1][:, :, 0, :, :] *= -1.0
        return out


class PafHFlip(torch.nn.Module):

    def __init__(self, keypoints, skeleton, hflip):
        super().__init__()
        skeleton_names = [(keypoints[j1 - 1], keypoints[j2 - 1]) for j1, j2 in skeleton]
        flipped_skeleton_names = [(hflip[j1] if j1 in hflip else j1, hflip[j2] if j2 in hflip else j2) for j1, j2 in skeleton_names]
        LOG.debug('skeleton = %s, flipped_skeleton = %s', skeleton_names, flipped_skeleton_names)
        flip_indices = list(range(len(skeleton)))
        reverse_direction = []
        for paf_i, (n1, n2) in enumerate(skeleton_names):
            if (n1, n2) in flipped_skeleton_names:
                flip_indices[paf_i] = flipped_skeleton_names.index((n1, n2))
            if (n2, n1) in flipped_skeleton_names:
                flip_indices[paf_i] = flipped_skeleton_names.index((n2, n1))
                reverse_direction.append(paf_i)
        LOG.debug('hflip indices: %s, reverse: %s', flip_indices, reverse_direction)
        self.register_buffer('flip_indices', torch.LongTensor(flip_indices))
        self.register_buffer('reverse_direction', torch.LongTensor(reverse_direction))

    def forward(self, *args):
        out = []
        for field in args:
            field = torch.index_select(field, 1, self.flip_indices)
            field = torch.flip(field, dims=[len(field.shape) - 1])
            out.append(field)
        out[1][:, :, 0, :, :] *= -1.0
        out[2][:, :, 0, :, :] *= -1.0
        for paf_i in self.reverse_direction:
            cc = torch.clone(out[1][:, paf_i])
            out[1][:, paf_i] = out[2][:, paf_i]
            out[2][:, paf_i] = cc
        return out


class HeadNetwork(torch.nn.Module):
    """Base class for head networks.

    :param meta: head meta instance to configure this head network
    :param in_features: number of input features which should be equal to the
        base network's output features
    """

    def __init__(self, meta: 'headmeta.Base', in_features: 'int'):
        super().__init__()
        self.meta = meta
        self.in_features = in_features

    @classmethod
    def cli(cls, parser: 'argparse.ArgumentParser'):
        """Command line interface (CLI) to extend argument parser."""

    @classmethod
    def configure(cls, args: 'argparse.Namespace'):
        """Take the parsed argument parser output and configure class variables."""

    def forward(self, x):
        raise NotImplementedError


@functools.lru_cache(16)
def index_field_torch(shape: 't.Tuple[int, int]', device: 'torch.device', unsqueeze: 't.Tuple[int, int]'=(0, 0)) ->torch.Tensor:
    assert len(shape) == 2
    xy = torch.empty((2, shape[0], shape[1]), device=device)
    xy[0] = torch.arange(shape[1], device=device)
    xy[1] = torch.arange(shape[0], device=device).unsqueeze(1)
    for dim in unsqueeze:
        xy = torch.unsqueeze(xy, dim)
    return xy


class CompositeField3(HeadNetwork):
    dropout_p = 0.0
    inplace_ops: 'bool' = True

    def __init__(self, meta: 'headmeta.Base', in_features, *, kernel_size=1, padding=0, dilation=1):
        super().__init__(meta, in_features)
        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d kernel = %d, padding = %d, dilation = %d', meta.name, meta.n_fields, meta.n_confidences, meta.n_vectors, meta.n_scales, kernel_size, padding, dilation)
        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)
        out_features = meta.n_fields * (meta.n_confidences + meta.n_vectors * 3 + meta.n_scales)
        self.conv = torch.nn.Conv2d(in_features, out_features * meta.upsample_stride ** 2, kernel_size, padding=padding, dilation=dilation)
        assert meta.upsample_stride >= 1
        self.upsample_op = None
        if meta.upsample_stride > 1:
            self.upsample_op = torch.nn.PixelShuffle(meta.upsample_stride)

    @classmethod
    def cli(cls, parser: 'argparse.ArgumentParser'):
        group = parser.add_argument_group('CompositeField3')
        group.add_argument('--cf3-dropout', default=cls.dropout_p, type=float, help='[experimental] zeroing probability of feature in head input')
        assert cls.inplace_ops
        group.add_argument('--cf3-no-inplace-ops', dest='cf3_inplace_ops', default=True, action='store_false', help='alternative graph without inplace ops')

    @classmethod
    def configure(cls, args: 'argparse.Namespace'):
        cls.dropout_p = args.cf3_dropout
        cls.inplace_ops = args.cf3_inplace_ops

    @property
    def sparse_task_parameters(self):
        return [self.conv.weight]

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        if self.upsample_op is not None:
            x = self.upsample_op(x)
            low_cut = (self.meta.upsample_stride - 1) // 2
            high_cut = math.ceil((self.meta.upsample_stride - 1) / 2.0)
            if self.training:
                x = x[:, :, low_cut:-high_cut, low_cut:-high_cut]
            else:
                x = x[:, :, low_cut:int(x.shape[2]) - high_cut, low_cut:int(x.shape[3]) - high_cut]
        x_size = x.size()
        batch_size = x_size[0]
        feature_height = int(x_size[2])
        feature_width = int(x_size[3])
        x = x.view(batch_size, self.meta.n_fields, self.meta.n_confidences + self.meta.n_vectors * 3 + self.meta.n_scales, feature_height, feature_width)
        if not self.training and self.inplace_ops:
            classes_x = x[:, :, 0:self.meta.n_confidences]
            torch.sigmoid_(classes_x)
            if self.meta.n_vectors > 0:
                index_field = index_field_torch((feature_height, feature_width), device=x.device)
                first_reg_feature = self.meta.n_confidences
                for i, do_offset in enumerate(self.meta.vector_offsets):
                    if not do_offset:
                        continue
                    reg_x = x[:, :, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                    reg_x.add_(index_field)
            first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
            scales_x = x[:, :, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x[:] = torch.nn.functional.softplus(scales_x)
            first_width_feature = self.meta.n_confidences + self.meta.n_vectors * 2
            x = torch.cat([x[:, :, first_width_feature:first_width_feature + 1], x[:, :, :first_width_feature], x[:, :, self.meta.n_confidences + self.meta.n_vectors * 3:]], dim=2)
        elif not self.training and not self.inplace_ops:
            x = torch.transpose(x, 1, 2)
            classes_x = x[:, 0:self.meta.n_confidences]
            classes_x = torch.sigmoid(classes_x)
            first_reg_feature = self.meta.n_confidences
            regs_x = [x[:, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2] for i in range(self.meta.n_vectors)]
            index_field = index_field_torch((feature_height, feature_width), device=x.device, unsqueeze=(1, 0))
            index_field = torch.from_numpy(index_field.numpy())
            regs_x = [(reg_x + index_field if do_offset else reg_x) for reg_x, do_offset in zip(regs_x, self.meta.vector_offsets)]
            first_reglogb_feature = self.meta.n_confidences + self.meta.n_vectors * 2
            single_reg_logb = x[:, first_reglogb_feature:first_reglogb_feature + 1]
            first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
            scales_x = x[:, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x = torch.nn.functional.softplus(scales_x)
            x = torch.cat([single_reg_logb, classes_x, *regs_x, scales_x], dim=1)
            x = torch.transpose(x, 1, 2)
        return x


class CompositeField4(HeadNetwork):
    dropout_p = 0.0
    inplace_ops: 'bool' = True

    def __init__(self, meta: 'headmeta.Base', in_features, *, kernel_size=1, padding=0, dilation=1):
        super().__init__(meta, in_features)
        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d kernel = %d, padding = %d, dilation = %d', meta.name, meta.n_fields, meta.n_confidences, meta.n_vectors, meta.n_scales, kernel_size, padding, dilation)
        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)
        self.n_components = 1 + meta.n_confidences + meta.n_vectors * 2 + meta.n_scales
        self.conv = torch.nn.Conv2d(in_features, meta.n_fields * self.n_components * meta.upsample_stride ** 2, kernel_size, padding=padding, dilation=dilation)
        self.n_fields = meta.n_fields
        self.n_confidences = meta.n_confidences
        self.n_vectors = meta.n_vectors
        self.n_scales = meta.n_scales
        self.vector_offsets = tuple(meta.vector_offsets)
        self.upsample_stride = meta.upsample_stride
        assert meta.upsample_stride >= 1
        self.upsample_op = None
        if meta.upsample_stride > 1:
            self.upsample_op = torch.nn.PixelShuffle(meta.upsample_stride)

    @classmethod
    def cli(cls, parser: 'argparse.ArgumentParser'):
        group = parser.add_argument_group('CompositeField4')
        group.add_argument('--cf4-dropout', default=cls.dropout_p, type=float, help='[experimental] zeroing probability of feature in head input')
        assert cls.inplace_ops
        group.add_argument('--cf4-no-inplace-ops', dest='cf4_inplace_ops', default=True, action='store_false', help='alternative graph without inplace ops')

    @classmethod
    def configure(cls, args: 'argparse.Namespace'):
        cls.dropout_p = args.cf4_dropout
        cls.inplace_ops = args.cf4_inplace_ops

    @property
    def sparse_task_parameters(self):
        return [self.conv.weight]

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.dropout(x)
        x = self.conv(x)
        if self.upsample_op is not None:
            x = self.upsample_op(x)
            low_cut = (self.upsample_stride - 1) // 2
            high_cut = math.ceil((self.upsample_stride - 1) / 2.0)
            if self.training:
                x = x[:, :, low_cut:-high_cut, low_cut:-high_cut]
            else:
                x = x[:, :, low_cut:int(x.shape[2]) - high_cut, low_cut:int(x.shape[3]) - high_cut]
        x_size = x.size()
        batch_size = x_size[0]
        feature_height = x_size[2]
        feature_width = x_size[3]
        x = x.view(batch_size, self.n_fields, self.n_components, feature_height, feature_width)
        if not self.training and self.inplace_ops:
            classes_x = x[:, :, 1:1 + self.n_confidences]
            torch.sigmoid_(classes_x)
            if self.n_vectors > 0:
                index_field = index_field_torch((feature_height, feature_width), device=x.device)
                first_reg_feature = 1 + self.n_confidences
                for i, do_offset in enumerate(self.vector_offsets):
                    if not do_offset:
                        continue
                    reg_x = x[:, :, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                    reg_x.add_(index_field)
            first_scale_feature = 1 + self.n_confidences + self.n_vectors * 2
            scales_x = x[:, :, first_scale_feature:first_scale_feature + self.n_scales]
            scales_x[:] = torch.nn.functional.softplus(scales_x)
        elif not self.training and not self.inplace_ops:
            x = torch.transpose(x, 1, 2)
            width_x = x[:, 0:1]
            classes_x = x[:, 1:1 + self.n_confidences]
            classes_x = torch.sigmoid(classes_x)
            first_reg_feature = 1 + self.n_confidences
            regs_x = [x[:, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2] for i in range(self.n_vectors)]
            index_field = index_field_torch((feature_height, feature_width), device=x.device, unsqueeze=(1, 0))
            index_field = torch.from_numpy(index_field.numpy())
            regs_x = [(reg_x + index_field if do_offset else reg_x) for reg_x, do_offset in zip(regs_x, self.vector_offsets)]
            first_scale_feature = 1 + self.n_confidences + self.n_vectors * 2
            scales_x = x[:, first_scale_feature:first_scale_feature + self.n_scales]
            scales_x = torch.nn.functional.softplus(scales_x)
            x = torch.cat([width_x, classes_x] + regs_x + [scales_x], dim=1)
            x = torch.transpose(x, 1, 2)
        return x


class SoftClamp(torch.nn.Module):

    def __init__(self, max_value):
        super().__init__()
        self.max_value = max_value

    def forward(self, x):
        above_max = x > self.max_value
        x[above_max] = self.max_value + torch.log(1 - self.max_value + x[above_max])
        return x


class Base(torch.nn.Module):

    def __init__(self, xi: 'List[int]', ti: 'List[int]'):
        super().__init__()
        self.xi = xi
        self.ti = ti

    def forward(self, x_all, t_all):
        return x_all[:, :, :, :, self.xi], t_all[:, :, :, :, self.ti]


class Bce(Base):
    focal_alpha = 0.5
    focal_gamma = 1.0
    soft_clamp_value = 5.0
    background_clamp = -15.0

    def __init__(self, xi: 'List[int]', ti: 'List[int]', weights=None, **kwargs):
        super().__init__(xi, ti)
        self.weights = weights
        for n, v in kwargs.items():
            assert hasattr(self, n)
            setattr(self, n, v)
        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = SoftClamp(self.soft_clamp_value)

    @classmethod
    def cli(cls, parser: 'argparse.ArgumentParser'):
        group = parser.add_argument_group('Bce Loss')
        group.add_argument('--focal-alpha', default=cls.focal_alpha, type=float, help='scale parameter of focal loss')
        group.add_argument('--focal-gamma', default=cls.focal_gamma, type=float, help='use focal loss with the given gamma')
        group.add_argument('--bce-soft-clamp', default=cls.soft_clamp_value, type=float, help='soft clamp for BCE')
        group.add_argument('--bce-background-clamp', default=cls.background_clamp, type=float, help='background clamp for BCE')

    @classmethod
    def configure(cls, args: 'argparse.Namespace'):
        cls.focal_alpha = args.focal_alpha
        cls.focal_gamma = args.focal_gamma
        cls.soft_clamp_value = args.bce_soft_clamp
        cls.background_clamp = args.bce_background_clamp

    def forward(self, x_all, t_all):
        x, t = super().forward(x_all, t_all)
        mask = t >= 0.0
        t = t[mask]
        x = x[mask]
        t_sign = t.clone()
        t_sign[t > 0.0] = 1.0
        t_sign[t <= 0.0] = -1.0
        x_detached = x.detach()
        focal_loss_modification = 1.0
        p_bar = 1.0 / (1.0 + torch.exp(t_sign * x_detached))
        if self.focal_alpha:
            focal_loss_modification *= self.focal_alpha
        if self.focal_gamma == 1.0:
            p = 1.0 - p_bar
            neg_ln_p = torch.nn.functional.softplus(-t_sign * x_detached)
            focal_loss_modification = focal_loss_modification * (p_bar + p * neg_ln_p)
        elif self.focal_gamma > 0.0:
            p = 1.0 - p_bar
            neg_ln_p = torch.nn.functional.softplus(-t_sign * x_detached)
            focal_loss_modification = focal_loss_modification * (p_bar ** self.focal_gamma + self.focal_gamma * p_bar ** (self.focal_gamma - 1.0) * p * neg_ln_p)
        elif self.focal_gamma == 0.0:
            pass
        else:
            raise NotImplementedError
        target = x_detached + t_sign * p_bar * focal_loss_modification
        l = torch.nn.functional.smooth_l1_loss(x, target, reduction='none')
        if self.background_clamp:
            l[(x_detached < self.background_clamp) & (t_sign == -1.0)] = 0.0
        if self.soft_clamp is not None:
            l = self.soft_clamp(l)
        mask_foreground = t > 0.0
        x_logs2 = x_all[:, :, :, :, 0:1][mask][mask_foreground]
        x_logs2 = 3.0 * torch.tanh(x_logs2 / 3.0)
        l[mask_foreground] = 0.5 * l[mask_foreground] * torch.exp(-x_logs2) + 0.5 * x_logs2
        if self.weights is not None:
            full_weights = torch.empty_like(t_all[:, :, :, :, 0:1])
            full_weights[:] = self.weights
            l = full_weights[mask] * l
        return l


class Scale(Base):
    b = 1.0
    log_space = False
    relative = True
    relative_eps = 0.1
    clip = None
    soft_clamp_value = 5.0

    def __init__(self, xi: 'List[int]', ti: 'List[int]', weights=None, **kwargs):
        super().__init__(xi, ti)
        self.weights = weights
        for n, v in kwargs.items():
            assert hasattr(self, n)
            setattr(self, n, v)
        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = SoftClamp(self.soft_clamp_value)

    @classmethod
    def cli(cls, parser: 'argparse.ArgumentParser'):
        group = parser.add_argument_group('Scale Loss')
        group.add_argument('--b-scale', default=cls.b, type=float, help='Laplace width b for scale loss')
        assert not cls.log_space
        group.add_argument('--scale-log', default=False, action='store_true')
        group.add_argument('--scale-soft-clamp', default=cls.soft_clamp_value, type=float, help='soft clamp for scale')

    @classmethod
    def configure(cls, args: 'argparse.Namespace'):
        cls.b = args.b_scale
        cls.log_space = args.scale_log
        if args.scale_log:
            cls.relative = False
        cls.soft_clamp_value = args.scale_soft_clamp

    def forward(self, x_all, t_all):
        x, t = super().forward(x_all, t_all)
        scale_mask = torch.isfinite(t)
        x = x[scale_mask]
        t = t[scale_mask]
        assert not (self.log_space and self.relative)
        x = torch.nn.functional.softplus(x)
        d = torch.nn.functional.l1_loss(x if not self.log_space else torch.log(x), t if not self.log_space else torch.log(t), reduction='none')
        if self.clip is not None:
            d = torch.clamp(d, self.clip[0], self.clip[1])
        denominator = self.b
        if self.relative:
            denominator = self.b * (self.relative_eps + t)
        d = d / denominator
        if self.soft_clamp is not None:
            d = self.soft_clamp(d)
        loss = torch.nn.functional.smooth_l1_loss(d, torch.zeros_like(d), reduction='none')
        if self.weights is not None:
            full_weights = torch.empty_like(t_all[:, :, :, :, 0:1])
            full_weights[:] = self.weights
            loss = full_weights[scale_mask] * loss
        return loss


class Regression(Base):
    soft_clamp_value = 5.0

    def __init__(self, xi: 'List[int]', ti: 'List[int]', weights=None, *, sigma_from_scale: float=0.5, scale_from_wh: bool=False):
        super().__init__(xi, ti)
        self.weights = weights
        self.sigma_from_scale = sigma_from_scale
        self.scale_from_wh = scale_from_wh
        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = SoftClamp(self.soft_clamp_value)

    @classmethod
    def cli(cls, parser: 'argparse.ArgumentParser'):
        group = parser.add_argument_group('Regression loss')
        group.add_argument('--regression-soft-clamp', default=cls.soft_clamp_value, type=float, help='soft clamp for scale')

    @classmethod
    def configure(cls, args: 'argparse.Namespace'):
        cls.soft_clamp_value = args.scale_soft_clamp

    def forward(self, x_all, t_all):
        """Only t_regs is guaranteed to be valid.
        Imputes t_sigma_min and and t_scales with guesses.
        """
        x, t = super().forward(x_all, t_all)
        x_regs = x[:, :, :, :, 0:2]
        x_scales = x[:, :, :, :, 2:3]
        t_regs = t[:, :, :, :, 0:2]
        t_sigma_min = t[:, :, :, :, 2:3]
        t_scales = t[:, :, :, :, 3:4]
        if self.scale_from_wh:
            x_scales = torch.linalg.norm(x[:, :, :, :, 2:4], ord=2, dim=4, keepdim=True)
            t_scales = torch.linalg.norm(t[:, :, :, :, 3:5], ord=2, dim=4, keepdim=True)
        finite = torch.isfinite(t_regs)
        reg_mask = torch.all(finite, dim=4)
        x_regs = x_regs[reg_mask]
        x_scales = x_scales[reg_mask]
        t_regs = t_regs[reg_mask]
        t_sigma_min = t_sigma_min[reg_mask]
        t_scales = t_scales[reg_mask]
        t_scales = t_scales.clone()
        invalid_t_scales = torch.isnan(t_scales)
        t_scales[invalid_t_scales] = torch.nn.functional.softplus(x_scales.detach()[invalid_t_scales])
        d = x_regs - t_regs
        t_sigma_min_imputed = t_sigma_min.clone()
        t_sigma_min_imputed[torch.isnan(t_sigma_min)] = 0.1
        d = torch.cat([d, t_sigma_min_imputed], dim=1)
        d = torch.linalg.norm(d, ord=2, dim=1, keepdim=True)
        t_sigma = self.sigma_from_scale * t_scales
        l = 1.0 / t_sigma * d
        if self.soft_clamp is not None:
            l = self.soft_clamp(l)
        x_logs2 = x_all[:, :, :, :, 0:1][reg_mask]
        x_logs2 = 3.0 * torch.tanh(x_logs2 / 3.0)
        x_logb = 0.5 * x_logs2 + 0.69314
        l = l * torch.exp(-x_logb) + x_logb
        if self.weights is not None:
            full_weights = torch.empty_like(t_all[:, :, :, :, 0:1])
            full_weights[:] = self.weights
            l = full_weights[reg_mask] * l
        return l


class CompositeLoss(torch.nn.Module):
    """Default loss since v0.13"""

    @classmethod
    def factory_from_headmeta(cls, head_meta):
        LOG.debug('%s: n_vectors = %d, n_scales = %d', head_meta.name, head_meta.n_vectors, head_meta.n_scales)
        weights = None
        if head_meta.training_weights is not None:
            assert len(head_meta.training_weights) == head_meta.n_fields
            weights = torch.Tensor(head_meta.training_weights).reshape(1, -1, 1, 1, 1)
            LOG.debug('The weights for the keypoints are %s', weights)
        loss_components: 'Dict[str, List[components.Base]]' = {f'{head_meta.dataset}.{head_meta.name}.c': [components.Bce([1], [0], weights=weights)]}
        regression_components: 'List[components.Base]' = []
        if head_meta.n_vectors <= head_meta.n_scales:
            regression_components = [components.Regression([2 + vi * 2, 2 + vi * 2 + 1, 2 + head_meta.n_vectors * 2 + vi], [1 + vi * 2, 1 + vi * 2 + 1, 1 + head_meta.n_vectors * 2 + vi, 1 + head_meta.n_vectors * 3 + vi], weights=weights) for vi in range(head_meta.n_vectors)]
        elif head_meta.n_vectors == 2 and head_meta.n_scales == 0:
            regression_components = [components.Regression([2 + vi * 2, 2 + vi * 2 + 1, 2 + 1 * 2, 2 + 1 * 2 + 1], [1 + vi * 2, 1 + vi * 2 + 1, 1 + 2 * 2 + vi, 1 + 1 * 2, 1 + 1 * 2 + 1], weights=weights, sigma_from_scale=0.1, scale_from_wh=True) for vi in range(head_meta.n_vectors)]
        if regression_components:
            loss_components[f'{head_meta.dataset}.{head_meta.name}.vec'] = regression_components
        if head_meta.n_scales:
            loss_components[f'{head_meta.dataset}.{head_meta.name}.scales'] = [components.Scale([2 + head_meta.n_vectors * 2 + si], [1 + head_meta.n_vectors * 3 + si], weights=weights) for si in range(head_meta.n_scales)]
        return cls(loss_components)

    def __init__(self, loss_components: 'Dict[str, List[components.Base]]'):
        super().__init__()
        self.loss_components = loss_components
        self.previous_losses = None

    @property
    def field_names(self):
        return self.loss_components.keys()

    @classmethod
    def cli(cls, parser: 'argparse.ArgumentParser'):
        pass

    @classmethod
    def configure(cls, args: 'argparse.Namespace'):
        pass

    def forward(self, x, t):
        LOG.debug('loss for %s', self.field_names)
        if t is None:
            return [None for _ in self.loss_components]
        x = torch.transpose(x, 2, 4)
        t = torch.transpose(t, 2, 4)
        losses = {name: [l(x, t) for l in loss_components] for name, loss_components in self.loss_components.items()}
        batch_size = t.shape[0]
        losses = [(None if not ls else torch.sum(ls[0] if len(ls) == 1 else torch.cat(ls)) / batch_size) for ls in losses.values()]
        if not all(torch.isfinite(l).item() if l is not None else True for l in losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'.format(losses, self.previous_losses))
        self.previous_losses = [(float(l.item()) if l is not None else None) for l in losses]
        return losses


class MultiHeadLoss(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas):
        super().__init__()
        if not lambdas:
            lambdas = [(1.0) for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)
        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.field_names = [n for l in self.losses for n in l.field_names]
        assert len(self.field_names) == len(self.lambdas)
        LOG.info('multihead loss: %s, %s', self.field_names, self.lambdas)

    def forward(self, head_fields, head_targets):
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        assert self.task_sparsity_weight == 0.0
        flat_head_losses = [ll for l, f, t in zip(self.losses, head_fields, head_targets) for ll in l(f, t)]
        assert len(self.lambdas) == len(flat_head_losses)
        loss_values = [(lam * l) for lam, l in zip(self.lambdas, flat_head_losses) if l is not None]
        total_loss = sum(loss_values) if loss_values else None
        return total_loss, flat_head_losses


class MultiHeadLossAutoTuneKendall(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas, *, sparse_task_parameters=None, tune=None):
        """Auto-tuning multi-head loss.

        Uses idea from "Multi-Task Learning Using Uncertainty to Weigh Losses
        for Scene Geometry and Semantics" by Kendall, Gal and Cipolla.

        Individual losses must not be negative for Kendall's prescription.

        In the common setting, use lambdas of zero and one to deactivate and
        activate the tasks you want to train. Less common, if you have
        secondary tasks, you can reduce their importance by choosing a
        lambda value between zero and one.
        """
        super().__init__()
        if not lambdas:
            lambdas = [(1.0) for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)
        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.sparse_task_parameters = sparse_task_parameters
        self.tune = tune
        self.log_sigmas = torch.nn.Parameter(torch.zeros((len(lambdas),), dtype=torch.float64), requires_grad=True)
        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss with autotune: %s', self.field_names)
        assert len(self.field_names) == len(self.lambdas)
        assert len(self.field_names) == len(self.log_sigmas)
        if self.tune is None:

            def tune_from_name(name):
                if '.vec' in name:
                    return 'none'
                if '.scale' in name:
                    return 'laplace'
                return 'gauss'
            self.tune = [tune_from_name(n) for l in self.losses for n in l.field_names]
        LOG.info('tune config: %s', self.tune)

    def batch_meta(self):
        return {'mtl_sigmas': [round(float(s), 3) for s in self.log_sigmas.exp()]}

    def forward(self, head_fields, head_targets):
        LOG.debug('losses = %d, fields = %d, targets = %d', len(self.losses), len(head_fields), len(head_targets))
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll for l, f, t in zip(self.losses, head_fields, head_targets) for ll in l(f, t)]
        assert len(self.lambdas) == len(flat_head_losses)
        assert len(self.log_sigmas) == len(flat_head_losses)
        constrained_log_sigmas = 3.0 * torch.tanh(self.log_sigmas / 3.0)

        def tuned_loss(tune, log_sigma, loss):
            if tune == 'none':
                return loss
            if tune == 'laplace':
                return 0.694 + log_sigma + loss * torch.exp(-log_sigma)
            if tune == 'gauss':
                return 0.919 + log_sigma + loss * 0.5 * torch.exp(-2.0 * log_sigma)
            raise Exception('unknown tune: {}'.format(tune))
        loss_values = [(lam * tuned_loss(t, log_sigma, l)) for lam, t, log_sigma, l in zip(self.lambdas, self.tune, constrained_log_sigmas, flat_head_losses) if l is not None]
        total_loss = sum(loss_values) if loss_values else None
        if self.task_sparsity_weight and self.sparse_task_parameters is not None:
            head_sparsity_loss = sum(param.abs().max(dim=1)[0].clamp(min=1e-06).sum() for param in self.sparse_task_parameters)
            LOG.debug('l1 head sparsity loss = %f (total = %f)', head_sparsity_loss, total_loss)
            total_loss = total_loss + self.task_sparsity_weight * head_sparsity_loss
        return total_loss, flat_head_losses


class MultiHeadLossAutoTuneVariance(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas, *, sparse_task_parameters=None):
        """Auto-tuning multi-head loss based on loss-variance.

        In the common setting, use lambdas of zero and one to deactivate and
        activate the tasks you want to train. Less common, if you have
        secondary tasks, you can reduce their importance by choosing a
        lambda value between zero and one.
        """
        super().__init__()
        if not lambdas:
            lambdas = [(1.0) for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)
        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.sparse_task_parameters = sparse_task_parameters
        self.epsilons = torch.ones((len(lambdas),), dtype=torch.float64)
        self.buffer = torch.full((len(lambdas), 53), float('nan'), dtype=torch.float64)
        self.buffer_index = -1
        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss with autotune: %s', self.field_names)
        assert len(self.field_names) == len(self.lambdas)
        assert len(self.field_names) == len(self.epsilons)

    def batch_meta(self):
        return {'mtl_sigmas': [round(float(s), 3) for s in self.epsilons]}

    def forward(self, head_fields, head_targets):
        LOG.debug('losses = %d, fields = %d, targets = %d', len(self.losses), len(head_fields), len(head_targets))
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll for l, f, t in zip(self.losses, head_fields, head_targets) for ll in l(f, t)]
        self.buffer_index = (self.buffer_index + 1) % self.buffer.shape[1]
        for i, ll in enumerate(flat_head_losses):
            if not hasattr(ll, 'data'):
                continue
            self.buffer[i, self.buffer_index] = ll.data
        self.epsilons = torch.sqrt(torch.mean(self.buffer ** 2, dim=1) - torch.sum(self.buffer, dim=1) ** 2 / self.buffer.shape[1] ** 2)
        self.epsilons[torch.isnan(self.epsilons)] = 10.0
        self.epsilons = self.epsilons.clamp(0.01, 100.0)
        LOG.debug('eps before norm: %s', self.epsilons)
        self.epsilons = self.epsilons * torch.sum(1.0 / self.epsilons) / self.epsilons.shape[0]
        LOG.debug('eps after norm: %s', self.epsilons)
        assert len(self.lambdas) == len(flat_head_losses)
        assert len(self.epsilons) == len(flat_head_losses)
        loss_values = [(lam * l / eps) for lam, eps, l in zip(self.lambdas, self.epsilons, flat_head_losses) if l is not None]
        total_loss = sum(loss_values) if loss_values else None
        if self.task_sparsity_weight and self.sparse_task_parameters is not None:
            head_sparsity_loss = sum(param.abs().max(dim=1)[0].clamp(min=1e-06).sum() for param in self.sparse_task_parameters)
            LOG.debug('l1 head sparsity loss = %f (total = %f)', head_sparsity_loss, total_loss)
            total_loss = total_loss + self.task_sparsity_weight * head_sparsity_loss
        return total_loss, flat_head_losses


class Shell(torch.nn.Module):

    def __init__(self, base_net, head_nets, *, process_input=None, process_heads=None):
        super().__init__()
        self.base_net = base_net
        self.head_nets = None
        self.process_input = process_input
        self.process_heads = process_heads
        self.set_head_nets(head_nets)

    @property
    def head_metas(self):
        if self.head_nets is None:
            return None
        return [hn.meta for hn in self.head_nets]

    def set_head_nets(self, head_nets):
        if not isinstance(head_nets, torch.nn.ModuleList):
            head_nets = torch.nn.ModuleList(head_nets)
        for hn_i, hn in enumerate(head_nets):
            hn.meta.head_index = hn_i
            hn.meta.base_stride = self.base_net.stride
        self.head_nets = head_nets

    def forward(self, image_batch, head_mask=None):
        if self.process_input is not None:
            image_batch = self.process_input(image_batch)
        x = self.base_net(image_batch)
        if head_mask is not None:
            head_outputs = tuple(hn(x) if m else None for hn, m in zip(self.head_nets, head_mask))
        else:
            head_outputs = tuple(hn(x) for hn in self.head_nets)
        if self.process_heads is not None:
            head_outputs = self.process_heads(head_outputs)
        return head_outputs


class CrossTalk(torch.nn.Module):

    def __init__(self, strength=0.2):
        super().__init__()
        self.strength = strength

    def forward(self, image_batch):
        if self.training and self.strength:
            rolled_images = torch.cat((image_batch[-1:], image_batch[:-1]))
            image_batch += rolled_images * self.cross_talk
        return image_batch


class RunningCache(torch.nn.Module):

    def __init__(self, cached_items):
        super().__init__()
        self.cached_items = cached_items
        self.duration = abs(min(cached_items)) + 1
        self.cache = [None for _ in range(self.duration)]
        self.index = 0
        LOG.debug('running cache of length %d', len(self.cache))

    def incr(self):
        self.index = (self.index + 1) % self.duration

    def get_index(self, index):
        while index < 0:
            index += self.duration
        while index >= self.duration:
            index -= self.duration
        LOG.debug('retrieving cache at index %d', index)
        v = self.cache[index]
        if v is not None:
            v = v.detach()
        return v

    def get(self):
        return [self.get_index(i + self.index) for i in self.cached_items]

    def set_next(self, data):
        self.incr()
        self.cache[self.index] = data
        LOG.debug('set new data at index %d', self.index)
        return self

    def forward(self, *args):
        LOG.debug('----------- running cache --------------')
        x = args[0]
        o = []
        for x_i in x:
            o += self.set_next(x_i).get()
        if any(oo is None for oo in o):
            o = [(oo if oo is not None else o[0]) for oo in o]
        if len(o) >= 2:
            image_sizes = [tuple(oo.shape[-2:]) for oo in o]
            if not all(ims == image_sizes[0] for ims in image_sizes[1:]):
                freq = defaultdict(int)
                for ims in image_sizes:
                    freq[ims] += 1
                max_freq = max(freq.values())
                ref_image_size = next(iter(ims for ims, f in freq.items() if f == max_freq))
                for i, ims in enumerate(image_sizes):
                    if ims == ref_image_size:
                        continue
                    for s in range(1, len(image_sizes)):
                        target_i = (i + s) % len(image_sizes)
                        if image_sizes[target_i] == ref_image_size:
                            break
                    LOG.warning('replacing %d (%s) with %d (%s) for ref %s', i, ims, target_i, image_sizes[target_i], ref_image_size)
                    o[i] = o[target_i]
        return torch.stack(o)


class TBaseSingleImage(HeadNetwork):
    """Filter the feature map so that they can be used by single image loss.

    Training: only apply loss to image 0 of an image pair of image 0 and 1.
    Evaluation with forward tracking pose: only keep image 0.
    Evaluation with full tracking pose: keep all but stack group along feature dim.
    """
    forward_tracking_pose = True
    tracking_pose_length = 2

    def __init__(self, meta, in_features):
        super().__init__(meta, in_features)
        self.head = CompositeField4(meta, in_features)

    def forward(self, *args):
        x = args[0]
        if self.training:
            x = x[::2]
        elif self.forward_tracking_pose:
            x = x[::self.tracking_pose_length]
        x = self.head(x)
        if not self.training and not self.forward_tracking_pose:
            raise NotImplementedError
        return x


class Tcaf(HeadNetwork):
    """Filter the feature map so that they can be used by single image loss.

    Training: only apply loss to image 0 of an image pair of image 0 and 1.
    Evaluation with forward tracking pose: only keep image 0.
    Evaluation with full tracking pose: keep all.
    """
    tracking_pose_length = 2
    reduced_features = 512
    _global_feature_reduction = None
    _global_feature_compute = None

    def __init__(self, meta, in_features):
        super().__init__(meta, in_features)
        if self._global_feature_reduction is None:
            self.__class__._global_feature_reduction = torch.nn.Sequential(torch.nn.Conv2d(in_features, self.reduced_features, kernel_size=1, bias=True), torch.nn.ReLU(inplace=True))
        self.feature_reduction = self._global_feature_reduction
        if self._global_feature_compute is None:
            self.__class__._global_feature_compute = torch.nn.Sequential(torch.nn.Conv2d(self.reduced_features * 2, self.reduced_features * 2, kernel_size=1, bias=True), torch.nn.ReLU(inplace=True))
        self.feature_compute = self._global_feature_compute
        self.head = CompositeField4(meta, self.reduced_features * 2)

    def forward(self, *args):
        x = args[0]
        if len(x) % 2 == 1:
            return None
        x = self.feature_reduction(x)
        group_length = 2 if self.training else self.tracking_pose_length
        primary = x[::group_length]
        others = [x[i::group_length] for i in range(1, group_length)]
        x = torch.stack([torch.cat([primary, o], dim=1) for o in others], dim=1)
        x_shape = x.size()
        x = torch.reshape(x, [x_shape[0] * x_shape[1]] + list(x_shape[2:]))
        x = self.feature_compute(x)
        x = self.head(x)
        if self.tracking_pose_length != 2:
            raise NotImplementedError
        return x


class ModuleUsingCifHr(torch.nn.Module):

    def forward(self, x):
        cifhr = torch.classes.openpifpaf_decoder_utils.CifHr()
        with torch.no_grad():
            cifhr.reset(x.shape[1:], 8)
            cifhr.accumulate(x[1:], 8, 0.0, 1.0)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CrossTalk,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ShuffleNetV2K,
     lambda: ([], {'name': 4, 'stages_repeats': [4, 4, 4], 'stages_out_channels': [4, 4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SoftClamp,
     lambda: ([], {'max_value': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

