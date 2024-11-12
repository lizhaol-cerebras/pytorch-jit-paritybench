
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


from torch.nn import GroupNorm


from torch.optim import Adam


import warnings


from typing import List


from typing import Optional


from typing import Union


import numpy as np


import torch


import torch.nn as nn


import inspect


import logging


from collections import defaultdict


from typing import Callable


from typing import Dict


from typing import Generator


from typing import Iterable


from typing import Sequence


from typing import Tuple


from functools import partial


from itertools import product


from typing import Any


from torch import Tensor


import torch.nn.functional as F


from torch.nn import SiLU


from torch.nn import SyncBatchNorm


from torch.optim import AdamW


import copy


import random


import itertools


import math


from typing import Iterator


from typing import Sized


from torch.utils.data import Sampler


from collections import OrderedDict


from torch import nn


from torch.nn import LayerNorm


from torch.optim import Optimizer


from abc import ABCMeta


from abc import abstractmethod


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.functional import pad


import torch.utils.checkpoint as cp


import copy as cp


from copy import deepcopy


import types


from itertools import zip_longest


from torch.nn.modules.utils import _pair


from torch.nn import functional as F


from torch import distributions


from abc import abstractproperty


from collections import abc


from typing import Type


import torch.distributed as dist


import torch.multiprocessing as mp


from torchvision.transforms import ToPILImage


from torch.utils.data.dataset import ConcatDataset


from torch.utils.data import Dataset


import torchvision.transforms.functional as F


from torchvision import transforms


import torch.utils.checkpoint as checkpoint


from scipy import interpolate


from torch.nn import BatchNorm2d


from typing import TypeVar


import torch.optim as optim


from torch.nn.modules import GroupNorm


from torch.nn.modules import AvgPool2d


class Focus(nn.Module):
    """Focus width and height information into channel space.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): The kernel size of the convolution. Default: 1
        stride (int): The stride of the convolution. Default: 1
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, conv_cfg=None, norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='Swish')):
        super().__init__()
        self.conv = ConvModule(in_channels * 4, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)


class RSoftmax(nn.Module):
    """Radix Softmax module in ``SplitAttentionConv2d``.

    Args:
        radix (int): Radix of input.
        groups (int): Groups of input.
    """

    def __init__(self, radix, groups):
        super().__init__()
        self.radix = radix
        self.groups = groups

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       4 for ``ViPNAS_Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, ViPNAS_Bottleneck):
            expansion = 1
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')
    return expansion


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self, block, num_blocks, in_channels, out_channels, expansion=None, stride=1, avg_down=False, conv_cfg=None, norm_cfg=dict(type='BN'), downsample_first=True, **kwargs):
        norm_cfg = copy.deepcopy(norm_cfg)
        self.block = block
        self.expansion = get_expansion(block, expansion)
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
            downsample.extend([build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=1, stride=conv_stride, bias=False), build_norm_layer(norm_cfg, out_channels)[1]])
            downsample = nn.Sequential(*downsample)
        layers = []
        if downsample_first:
            layers.append(block(in_channels=in_channels, out_channels=out_channels, expansion=self.expansion, stride=stride, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
            in_channels = out_channels
            for _ in range(1, num_blocks):
                layers.append(block(in_channels=in_channels, out_channels=out_channels, expansion=self.expansion, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        else:
            for i in range(0, num_blocks - 1):
                layers.append(block(in_channels=in_channels, out_channels=in_channels, expansion=self.expansion, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
            layers.append(block(in_channels=in_channels, out_channels=out_channels, expansion=self.expansion, stride=stride, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        super().__init__(*layers)


class SELayer(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
    """

    def __init__(self, channels, ratio=16, conv_cfg=None, act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = act_cfg, act_cfg
        assert len(act_cfg) == 2
        assert mmengine.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(in_channels=channels, out_channels=int(channels / ratio), kernel_size=1, stride=1, conv_cfg=conv_cfg, act_cfg=act_cfg[0])
        self.conv2 = ConvModule(in_channels=int(channels / ratio), out_channels=channels, kernel_size=1, stride=1, conv_cfg=conv_cfg, act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class InvertedResidual(nn.Module):
    """Inverted Residual Block.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        mid_channels (int): The input channels of the depthwise convolution.
        kernel_size (int): The kernel size of the depthwise convolution.
            Default: 3.
        groups (None or int): The group number of the depthwise convolution.
            Default: None, which means group number = mid_channels.
        stride (int): The stride of the depthwise convolution. Default: 1.
        se_cfg (dict): Config dict for se layer. Default: None, which means no
            se layer.
        with_expand_conv (bool): Use expand conv or not. If set False,
            mid_channels must be the same with in_channels.
            Default: True.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, in_channels, out_channels, mid_channels, kernel_size=3, groups=None, stride=1, se_cfg=None, with_expand_conv=True, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), with_cp=False):
        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        super().__init__()
        self.with_res_shortcut = stride == 1 and in_channels == out_channels
        assert stride in [1, 2]
        self.with_cp = with_cp
        self.with_se = se_cfg is not None
        self.with_expand_conv = with_expand_conv
        if groups is None:
            groups = mid_channels
        if self.with_se:
            assert isinstance(se_cfg, dict)
        if not self.with_expand_conv:
            assert mid_channels == in_channels
        if self.with_expand_conv:
            self.expand_conv = ConvModule(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.depthwise_conv = ConvModule(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if self.with_se:
            self.se = SELayer(**se_cfg)
        self.linear_conv = ConvModule(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):

        def _inner_forward(x):
            out = x
            if self.with_expand_conv:
                out = self.expand_conv(out)
            out = self.depthwise_conv(out)
            if self.with_se:
                out = self.se(out)
            out = self.linear_conv(out)
            if self.with_res_shortcut:
                return x + out
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class TruncSigmoid(nn.Sigmoid):
    """A sigmoid activation function that truncates the output to the given
    range.

    Args:
        min (float, optional): The minimum value to clamp the output to.
            Defaults to 0.0
        max (float, optional): The maximum value to clamp the output to.
            Defaults to 1.0
    """

    def __init__(self, min: 'float'=0.0, max: 'float'=1.0):
        super(TruncSigmoid, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input: 'Tensor') ->Tensor:
        """Computes the truncated sigmoid activation of the input tensor."""
        output = torch.sigmoid(input)
        output = output.clamp(min=self.min, max=self.max)
        return output


class SpatialAttention(nn.Module):
    """Spatial-wise attention module introduced in `CID`.

    Args:
        in_channels (int): The number of channels of the input instance
            vectors.
        out_channels (int): The number of channels of the transformed instance
            vectors.
    """

    def __init__(self, in_channels, out_channels):
        super(SpatialAttention, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels)
        self.feat_stride = 4
        self.conv = nn.Conv2d(3, 1, 5, 1, 2)

    def _get_pixel_coords(self, heatmap_size: 'Tuple', device: 'str'='cpu'):
        """Get pixel coordinates for each element in the heatmap.

        Args:
            heatmap_size (tuple): Size of the heatmap in (W, H) format.
            device (str): Device to put the resulting tensor on.

        Returns:
            Tensor of shape (batch_size, num_pixels, 2) containing the pixel
            coordinates for each element in the heatmap.
        """
        w, h = heatmap_size
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        pixel_coords = torch.stack((x, y), dim=-1).reshape(-1, 2)
        pixel_coords = pixel_coords.float() + 0.5
        return pixel_coords

    def forward(self, global_feats: 'Tensor', instance_feats: 'Tensor', instance_coords: 'Tensor') ->Tensor:
        """Perform spatial attention.

        Args:
            global_feats (Tensor): Tensor containing the global features.
            instance_feats (Tensor): Tensor containing the instance feature
                vectors.
            instance_coords (Tensor): Tensor containing the root coordinates
                of the instances.

        Returns:
            Tensor containing the modulated global features.
        """
        B, C, H, W = global_feats.size()
        instance_feats = self.atn(instance_feats).reshape(B, C, 1, 1)
        feats = global_feats * instance_feats.expand_as(global_feats)
        fsum = torch.sum(feats, dim=1, keepdim=True)
        pixel_coords = self._get_pixel_coords((W, H), feats.device)
        relative_coords = instance_coords.reshape(-1, 1, 2) - pixel_coords.reshape(1, -1, 2)
        relative_coords = relative_coords.permute(0, 2, 1) / 32.0
        relative_coords = relative_coords.reshape(B, 2, H, W)
        input_feats = torch.cat((fsum, relative_coords), dim=1)
        mask = self.conv(input_feats).sigmoid()
        return global_feats * mask


def make_linear_layers(feat_dims, relu_final=False):
    """Make linear layers."""
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
        if i < len(feat_dims) - 2 or i == len(feat_dims) - 2 and relu_final:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Heatmap1DHead(nn.Module):
    """Heatmap1DHead is a sub-module of Interhand3DHead, and outputs 1D
    heatmaps.

    Args:
        in_channels (int): Number of input channels. Defaults to 2048.
        heatmap_size (int): Heatmap size. Defaults to 64.
        hidden_dims (Sequence[int]): Number of feature dimension of FC layers.
            Defaults to ``(512, )``.
    """

    def __init__(self, in_channels: 'int'=2048, heatmap_size: 'int'=64, hidden_dims: 'Sequence[int]'=(512,)):
        super().__init__()
        self.in_channels = in_channels
        self.heatmap_size = heatmap_size
        feature_dims = [in_channels, *hidden_dims, heatmap_size]
        self.fc = make_linear_layers(feature_dims, relu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)
        accu = heatmap1d * torch.arange(self.heatmap_size, dtype=heatmap1d.dtype, device=heatmap1d.device)[None, :]
        coord = accu.sum(dim=1)
        return coord

    def forward(self, feats: 'Tuple[Tensor]') ->Tensor:
        """Forward the network.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        x = self.fc(feats)
        x = self.soft_argmax_1d(x).view(-1, 1)
        return x

    def init_weights(self):
        """Initialize model weights."""
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)


class MultilabelClassificationHead(nn.Module):
    """MultilabelClassificationHead is a sub-module of Interhand3DHead, and
    outputs hand type classification.

    Args:
        in_channels (int): Number of input channels. Defaults to 2048.
        num_labels (int): Number of labels. Defaults to 2.
        hidden_dims (Sequence[int]): Number of hidden dimension of FC layers.
            Defaults to ``(512, )``.
    """

    def __init__(self, in_channels: 'int'=2048, num_labels: 'int'=2, hidden_dims: 'Sequence[int]'=(512,)):
        super().__init__()
        self.in_channels = in_channels
        feature_dims = [in_channels, *hidden_dims, num_labels]
        self.fc = make_linear_layers(feature_dims, relu_final=False)

    def init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)

    def forward(self, x):
        """Forward function."""
        labels = self.fc(x)
        return labels


class PRM(nn.Module):
    """Pose Refine Machine.

    Please refer to "Learning Delicate Local Representations
    for Multi-Person Pose Estimation" (ECCV 2020).

    Args:
        out_channels (int): Number of the output channels, equals to
            the number of keypoints.
        norm_cfg (Config): Config to construct the norm layer.
            Defaults to ``dict(type='BN')``
    """

    def __init__(self, out_channels: 'int', norm_cfg: 'ConfigType'=dict(type='BN')):
        super().__init__()
        norm_cfg = copy.deepcopy(norm_cfg)
        self.out_channels = out_channels
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_path = nn.Sequential(Linear(self.out_channels, self.out_channels), build_norm_layer(dict(type='BN1d'), out_channels)[1], build_activation_layer(dict(type='ReLU')), Linear(self.out_channels, self.out_channels), build_norm_layer(dict(type='BN1d'), out_channels)[1], build_activation_layer(dict(type='ReLU')), build_activation_layer(dict(type='Sigmoid')))
        self.bottom_path = nn.Sequential(ConvModule(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0, norm_cfg=norm_cfg, inplace=False), DepthwiseSeparableConvModule(self.out_channels, 1, kernel_size=9, stride=1, padding=4, norm_cfg=norm_cfg, inplace=False), build_activation_layer(dict(type='Sigmoid')))
        self.conv_bn_relu_prm_1 = ConvModule(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg, inplace=False)

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward the network. The input heatmaps will be refined.

        Args:
            x (Tensor): The input heatmaps.

        Returns:
            Tensor: output heatmaps.
        """
        out = self.conv_bn_relu_prm_1(x)
        out_1 = out
        out_2 = self.global_pooling(out_1)
        out_2 = out_2.view(out_2.size(0), -1)
        out_2 = self.middle_path(out_2)
        out_2 = out_2.unsqueeze(2)
        out_2 = out_2.unsqueeze(3)
        out_3 = self.bottom_path(out_1)
        out = out_1 * (1 + out_2 * out_3)
        return out


class PredictHeatmap(nn.Module):
    """Predict the heatmap for an input feature.

    Args:
        unit_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        out_shape (tuple): Shape of the output heatmaps.
        use_prm (bool): Whether to use pose refine machine. Default: False.
        norm_cfg (Config): Config to construct the norm layer.
            Defaults to ``dict(type='BN')``
    """

    def __init__(self, unit_channels: 'int', out_channels: 'int', out_shape: 'tuple', use_prm: 'bool'=False, norm_cfg: 'ConfigType'=dict(type='BN')):
        super().__init__()
        norm_cfg = copy.deepcopy(norm_cfg)
        self.unit_channels = unit_channels
        self.out_channels = out_channels
        self.out_shape = out_shape
        self.use_prm = use_prm
        if use_prm:
            self.prm = PRM(out_channels, norm_cfg=norm_cfg)
        self.conv_layers = nn.Sequential(ConvModule(unit_channels, unit_channels, kernel_size=1, stride=1, padding=0, norm_cfg=norm_cfg, inplace=False), ConvModule(unit_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=None, inplace=False))

    def forward(self, feature: 'Tensor') ->Tensor:
        """Forward the network.

        Args:
            feature (Tensor): The input feature maps.

        Returns:
            Tensor: output heatmaps.
        """
        feature = self.conv_layers(feature)
        output = nn.functional.interpolate(feature, size=self.out_shape, mode='bilinear', align_corners=True)
        if self.use_prm:
            output = self.prm(output)
        return output


class AssociativeEmbeddingLoss(nn.Module):
    """Associative Embedding loss.

    Details can be found in
    `Associative Embedding <https://arxiv.org/abs/1611.05424>`_

    Note:

        - batch size: B
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - embedding tag dimension: L
        - heatmap size: [W, H]

    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0
        push_loss_factor (float): A factor that controls the weight between
            the push loss and the pull loss. Defaults to 0.5
    """

    def __init__(self, loss_weight: 'float'=1.0, push_loss_factor: 'float'=0.5) ->None:
        super().__init__()
        self.loss_weight = loss_weight
        self.push_loss_factor = push_loss_factor

    def _ae_loss_per_image(self, tags: 'Tensor', keypoint_indices: 'Tensor'):
        """Compute associative embedding loss for one image.

        Args:
            tags (Tensor): Tagging heatmaps in shape (K*L, H, W)
            keypoint_indices (Tensor): Ground-truth keypint position indices
                in shape (N, K, 2)
        """
        K = keypoint_indices.shape[1]
        C, H, W = tags.shape
        L = C // K
        tags = tags.view(L, K, H * W)
        instance_tags = []
        instance_kpt_tags = []
        for keypoint_indices_n in keypoint_indices:
            _kpt_tags = []
            for k in range(K):
                if keypoint_indices_n[k, 1]:
                    _kpt_tags.append(tags[:, k, keypoint_indices_n[k, 0]])
            if _kpt_tags:
                kpt_tags = torch.stack(_kpt_tags)
                instance_kpt_tags.append(kpt_tags)
                instance_tags.append(kpt_tags.mean(dim=0))
        N = len(instance_kpt_tags)
        if N == 0:
            pull_loss = tags.new_zeros(size=(), requires_grad=True)
            push_loss = tags.new_zeros(size=(), requires_grad=True)
        else:
            pull_loss = sum(F.mse_loss(_kpt_tags, _tag.expand_as(_kpt_tags)) for _kpt_tags, _tag in zip(instance_kpt_tags, instance_tags))
            if N == 1:
                push_loss = tags.new_zeros(size=(), requires_grad=True)
            else:
                tag_mat = torch.stack(instance_tags)
                diff = tag_mat[None] - tag_mat[:, None]
                push_loss = torch.sum(torch.exp(-diff.pow(2)))
            eps = 1e-06
            pull_loss = pull_loss / (N + eps)
            push_loss = push_loss / ((N - 1) * N + eps)
        return pull_loss, push_loss

    def forward(self, tags: 'Tensor', keypoint_indices: 'Union[List[Tensor], Tensor]'):
        """Compute associative embedding loss on a batch of data.

        Args:
            tags (Tensor): Tagging heatmaps in shape (B, L*K, H, W)
            keypoint_indices (Tensor|List[Tensor]): Ground-truth keypint
                position indices represented by a Tensor in shape
                (B, N, K, 2), or a list of B Tensors in shape (N_i, K, 2)
                Each keypoint's index is represented as [i, v], where i is the
                position index in the heatmap (:math:`i=y*w+x`) and v is the
                visibility

        Returns:
            tuple:
            - pull_loss (Tensor)
            - push_loss (Tensor)
        """
        assert tags.shape[0] == len(keypoint_indices)
        pull_loss = 0.0
        push_loss = 0.0
        for i in range(tags.shape[0]):
            _pull, _push = self._ae_loss_per_image(tags[i], keypoint_indices[i])
            pull_loss += _pull * self.loss_weight
            push_loss += _push * self.loss_weight * self.push_loss_factor
        return pull_loss, push_loss


def fp16_clamp(x, min_val=None, max_val=None):
    if not x.is_cuda and x.dtype == torch.float16:
        return x.float().clamp(min_val, max_val).half()
    return x.clamp(min_val, max_val)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-06) ->torch.Tensor:
    """Calculate overlap between two sets of bounding boxes.

    Args:
        bboxes1 (torch.Tensor): Bounding boxes of shape (..., m, 4) or empty.
        bboxes2 (torch.Tensor): Bounding boxes of shape (..., n, 4) or empty.
        mode (str): "iou" (intersection over union),
                    "iof" (intersection over foreground),
                    or "giou" (generalized intersection over union).
                    Defaults to "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A small constant added to the denominator for
            numerical stability. Default 1e-6.

    Returns:
        torch.Tensor: Overlap values of shape (..., m, n) if is_aligned is
            False, else shape (..., m).

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )
    """
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0
    if bboxes1.ndim == 1:
        bboxes1 = bboxes1.unsqueeze(0)
    if bboxes2.ndim == 1:
        bboxes2 = bboxes2.unsqueeze(0)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = fp16_clamp(rb - lt, min_val=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = fp16_clamp(rb - lt, min_val=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    eps_tensor = union.new_tensor([eps])
    union = torch.max(union, eps_tensor)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    elif mode == 'giou':
        enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min_val=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps_tensor)
        gious = ious - (enclose_area - union) / enclose_area
        return gious


class IoULoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        reduction (str): Options are "none", "mean" and "sum".
        eps (float): Epsilon to avoid log(0).
        loss_weight (float): Weight of the loss. Default: 1.0.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self, reduction='mean', mode='log', eps: 'float'=1e-16, loss_weight=1.0):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), f"the argument `reduction` should be either 'mean', 'sum' or 'none', but got {reduction}"
        assert mode in ('linear', 'square', 'log'), f"the argument `reduction` should be either 'linear', 'square' or 'log', but got {mode}"
        self.reduction = reduction
        self.criterion = partial(F.cross_entropy, reduction='none')
        self.loss_weight = loss_weight
        self.mode = mode
        self.eps = eps

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
        """
        ious = bbox_overlaps(output, target, is_aligned=True).clamp(min=self.eps)
        if self.mode == 'linear':
            loss = 1 - ious
        elif self.mode == 'square':
            loss = 1 - ious.pow(2)
        elif self.mode == 'log':
            loss = -ious.log()
        else:
            raise NotImplementedError
        if target_weight is not None:
            for i in range(loss.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = loss * target_weight
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss * self.loss_weight


class BCELoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of the loss. Default: 1.0.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            before output. Defaults to False.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.0, reduction='mean', use_sigmoid=False):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), f"the argument `reduction` should be either 'mean', 'sum' or 'none', but got {reduction}"
        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        criterion = F.binary_cross_entropy if use_sigmoid else F.binary_cross_entropy_with_logits
        self.criterion = partial(criterion, reduction='none')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output, target)
            if target_weight.dim() == 1:
                target_weight = target_weight[:, None]
            loss = loss * target_weight
        else:
            loss = self.criterion(output, target)
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss * self.loss_weight


class JSDiscretLoss(nn.Module):
    """Discrete JS Divergence loss for DSNT with Gaussian Heatmap.

    Modified from `the official implementation
    <https://github.com/anibali/dsntnn/blob/master/dsntnn/__init__.py>`_.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
    """

    def __init__(self, use_target_weight=True, size_average: 'bool'=True):
        super(JSDiscretLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.size_average = size_average
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def kl(self, p, q):
        """Kullback-Leibler Divergence."""
        eps = 1e-24
        kl_values = self.kl_loss((q + eps).log(), p)
        return kl_values

    def js(self, pred_hm, gt_hm):
        """Jensen-Shannon Divergence."""
        m = 0.5 * (pred_hm + gt_hm)
        js_values = 0.5 * (self.kl(pred_hm, m) + self.kl(gt_hm, m))
        return js_values

    def forward(self, pred_hm, gt_hm, target_weight=None):
        """Forward function.

        Args:
            pred_hm (torch.Tensor[N, K, H, W]): Predicted heatmaps.
            gt_hm (torch.Tensor[N, K, H, W]): Target heatmaps.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.

        Returns:
            torch.Tensor: Loss value.
        """
        if self.use_target_weight:
            assert target_weight is not None
            assert pred_hm.ndim >= target_weight.ndim
            for i in range(pred_hm.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = self.js(pred_hm * target_weight, gt_hm * target_weight)
        else:
            loss = self.js(pred_hm, gt_hm)
        if self.size_average:
            loss /= len(gt_hm)
        return loss.sum()


class KLDiscretLoss(nn.Module):
    """Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.
    Modified from `the official implementation.

    <https://github.com/leeyegy/SimCC>`_.
    Args:
        beta (float): Temperature factor of Softmax. Default: 1.0.
        label_softmax (bool): Whether to use Softmax on labels.
            Default: False.
        label_beta (float): Temperature factor of Softmax on labels.
            Default: 1.0.
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        mask (list[int]): Index of masked keypoints.
        mask_weight (float): Weight of masked keypoints. Default: 1.0.
    """

    def __init__(self, beta=1.0, label_softmax=False, label_beta=10.0, use_target_weight=True, mask=None, mask_weight=1.0):
        super(KLDiscretLoss, self).__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.label_beta = label_beta
        self.use_target_weight = use_target_weight
        self.mask = mask
        self.mask_weight = mask_weight
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        """Criterion function."""
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = F.softmax(labels * self.label_beta, dim=1)
        loss = torch.mean(self.kl_loss(log_pt, labels), dim=1)
        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis.
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        N, K, _ = pred_simcc[0].shape
        loss = 0
        if self.use_target_weight:
            weight = target_weight.reshape(-1)
        else:
            weight = 1.0
        for pred, target in zip(pred_simcc, gt_simcc):
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1, target.size(-1))
            t_loss = self.criterion(pred, target).mul(weight)
            if self.mask is not None:
                t_loss = t_loss.reshape(N, K)
                t_loss[:, self.mask] = t_loss[:, self.mask] * self.mask_weight
            loss = loss + t_loss.sum()
        return loss / K


class InfoNCELoss(nn.Module):
    """InfoNCE loss for training a discriminative representation space with a
    contrastive manner.

    `Representation Learning with Contrastive Predictive Coding
    arXiv: <https://arxiv.org/abs/1611.05424>`_.

    Args:
        temperature (float, optional): The temperature to use in the softmax
            function. Higher temperatures lead to softer probability
            distributions. Defaults to 1.0.
        loss_weight (float, optional): The weight to apply to the loss.
            Defaults to 1.0.
    """

    def __init__(self, temperature: 'float'=1.0, loss_weight=1.0) ->None:
        super(InfoNCELoss, self).__init__()
        assert temperature > 0, f'the argument `temperature` must be positive, but got {temperature}'
        self.temp = temperature
        self.loss_weight = loss_weight

    def forward(self, features: 'torch.Tensor') ->torch.Tensor:
        """Computes the InfoNCE loss.

        Args:
            features (Tensor): A tensor containing the feature
                representations of different samples.

        Returns:
            Tensor: A tensor of shape (1,) containing the InfoNCE loss.
        """
        n = features.size(0)
        features_norm = F.normalize(features, dim=1)
        logits = features_norm.mm(features_norm.t()) / self.temp
        targets = torch.arange(n, dtype=torch.long, device=features.device)
        loss = F.cross_entropy(logits, targets, reduction='sum')
        return loss * self.loss_weight


class VariFocalLoss(nn.Module):
    """Varifocal loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of the loss. Default: 1.0.
        alpha (float): A balancing factor for the negative part of
            Varifocal Loss. Defaults to 0.75.
        gamma (float): Gamma parameter for the modulating factor.
            Defaults to 2.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.0, reduction='mean', alpha=0.75, gamma=2.0):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), f"the argument `reduction` should be either 'mean', 'sum' or 'none', but got {reduction}"
        self.reduction = reduction
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.gamma = gamma

    def criterion(self, output, target):
        label = target > 0.0001
        weight = self.alpha * output.sigmoid().pow(self.gamma) * (1 - label) + target
        output = output.clip(min=-10, max=10)
        vfl = F.binary_cross_entropy_with_logits(output, target, reduction='none') * weight
        return vfl

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output, target)
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss = loss * target_weight
        else:
            loss = self.criterion(output, target)
        loss[torch.isinf(loss)] = 0.0
        loss[torch.isnan(loss)] = 0.0
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss * self.loss_weight


class FeaLoss(nn.Module):
    """PyTorch version of feature-based distillation from DWPose Modified from
    the official implementation.

    <https://github.com/IDEA-Research/DWPose>
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        alpha_fea (float, optional): Weight of dis_loss. Defaults to 0.00007
    """

    def __init__(self, name, use_this, student_channels, teacher_channels, alpha_fea=7e-05):
        super(FeaLoss, self).__init__()
        self.alpha_fea = alpha_fea
        if teacher_channels != student_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

    def forward(self, preds_S, preds_T):
        """Forward function.

        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        if self.align is not None:
            outs = self.align(preds_S)
        else:
            outs = preds_S
        loss = self.get_dis_loss(outs, preds_T)
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape
        dis_loss = loss_mse(preds_S, preds_T) / N * self.alpha_fea
        return dis_loss


class KeypointMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self, use_target_weight: 'bool'=False, skip_empty_channel: 'bool'=False, loss_weight: 'float'=1.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight

    def forward(self, output: 'Tensor', target: 'Tensor', target_weights: 'Optional[Tensor]'=None, mask: 'Optional[Tensor]'=None) ->Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """
        _mask = self._get_mask(target, target_weights, mask)
        if _mask is None:
            loss = F.mse_loss(output, target)
        else:
            _loss = F.mse_loss(output, target, reduction='none')
            loss = (_loss * _mask).mean()
        return loss * self.loss_weight

    def _get_mask(self, target: 'Tensor', target_weights: 'Optional[Tensor]', mask: 'Optional[Tensor]') ->Optional[Tensor]:
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        if mask is not None:
            assert mask.ndim == target.ndim and all(d_m == d_t or d_m == 1 for d_m, d_t in zip(mask.shape, target.shape)), f'mask and target have mismatched shapes {mask.shape} v.s.{target.shape}'
        if target_weights is not None:
            assert target_weights.ndim in (2, 4) and target_weights.shape == target.shape[:target_weights.ndim], f'target_weights and target have mismatched shapes {target_weights.shape} v.s. {target.shape}'
            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape + (1,) * ndim_pad)
            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1,) * ndim_pad)
            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask
        return mask


class CombinedTargetMSELoss(nn.Module):
    """MSE loss for combined target.

    CombinedTarget: The combination of classification target
    (response map) and regression target (offset map).
    Paper ref: Huang et al. The Devil is in the Details: Delving into
    Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self, use_target_weight: 'bool'=False, loss_weight: 'float'=1.0):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output: 'Tensor', target: 'Tensor', target_weights: 'Tensor') ->Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_channels: C
            - heatmaps height: H
            - heatmaps weight: W
            - num_keypoints: K
            Here, C = 3 * K

        Args:
            output (Tensor): The output feature maps with shape [B, C, H, W].
            target (Tensor): The target feature maps with shape [B, C, H, W].
            target_weights (Tensor): The target weights of differet keypoints,
                with shape [B, K].

        Returns:
            Tensor: The calculated loss.
        """
        batch_size = output.size(0)
        num_channels = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_channels, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_channels, -1)).split(1, 1)
        loss = 0.0
        num_joints = num_channels // 3
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx * 3].squeeze()
            heatmap_gt = heatmaps_gt[idx * 3].squeeze()
            offset_x_pred = heatmaps_pred[idx * 3 + 1].squeeze()
            offset_x_gt = heatmaps_gt[idx * 3 + 1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 3 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 3 + 2].squeeze()
            if self.use_target_weight:
                target_weight = target_weights[:, idx, None]
                heatmap_pred = heatmap_pred * target_weight
                heatmap_gt = heatmap_gt * target_weight
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_x_pred, heatmap_gt * offset_x_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_y_pred, heatmap_gt * offset_y_gt)
        return loss / num_joints * self.loss_weight


class KeypointOHKMMSELoss(nn.Module):
    """MSE loss with online hard keypoint mining.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        topk (int): Only top k joint losses are kept. Defaults to 8
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self, use_target_weight: 'bool'=False, topk: 'int'=8, loss_weight: 'float'=1.0):
        super().__init__()
        assert topk > 0
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.loss_weight = loss_weight

    def _ohkm(self, losses: 'Tensor') ->Tensor:
        """Online hard keypoint mining.

        Note:
            - batch_size: B
            - num_keypoints: K

        Args:
            loss (Tensor): The losses with shape [B, K]

        Returns:
            Tensor: The calculated loss.
        """
        ohkm_loss = 0.0
        B = losses.shape[0]
        for i in range(B):
            sub_loss = losses[i]
            _, topk_idx = torch.topk(sub_loss, k=self.topk, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= B
        return ohkm_loss

    def forward(self, output: 'Tensor', target: 'Tensor', target_weights: 'Tensor') ->Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W].
            target (Tensor): The target heatmaps with shape [B, K, H, W].
            target_weights (Tensor): The target weights of differet keypoints,
                with shape [B, K].

        Returns:
            Tensor: The calculated loss.
        """
        num_keypoints = output.size(1)
        if num_keypoints < self.topk:
            raise ValueError(f'topk ({self.topk}) should not be larger than num_keypoints ({num_keypoints}).')
        losses = []
        for idx in range(num_keypoints):
            if self.use_target_weight:
                target_weight = target_weights[:, idx, None, None]
                losses.append(self.criterion(output[:, idx] * target_weight, target[:, idx] * target_weight))
            else:
                losses.append(self.criterion(output[:, idx], target[:, idx]))
        losses = [loss.mean(dim=(1, 2)).unsqueeze(dim=1) for loss in losses]
        losses = torch.cat(losses, dim=1)
        return self._ohkm(losses) * self.loss_weight


class AdaptiveWingLoss(nn.Module):
    """Adaptive wing loss. paper ref: 'Adaptive Wing Loss for Robust Face
    Alignment via Heatmap Regression' Wang et al. ICCV'2019.

    Args:
        alpha (float), omega (float), epsilon (float), theta (float)
            are hyper-parameters.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5, use_target_weight=False, loss_weight=1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def criterion(self, pred, target):
        """Criterion of wingloss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            pred (torch.Tensor[NxKxHxW]): Predicted heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
        """
        H, W = pred.shape[2:4]
        delta = (target - pred).abs()
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))) * (self.alpha - target) * torch.pow(self.theta / self.epsilon, self.alpha - target - 1) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        losses = torch.where(delta < self.theta, self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - target)), A * delta - C)
        return torch.mean(losses)

    def forward(self, output: 'Tensor', target: 'Tensor', target_weights: 'Optional[Tensor]'=None):
        """Forward function.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, H, W]): Output heatmaps.
            target (torch.Tensor[N, K, H, W]): Target heatmaps.
            target_weight (torch.Tensor[N, K]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weights.ndim in (2, 4) and target_weights.shape == target.shape[:target_weights.ndim], f'target_weights and target have mismatched shapes {target_weights.shape} v.s. {target.shape}'
            ndim_pad = target.ndim - target_weights.ndim
            target_weights = target_weights.view(target_weights.shape + (1,) * ndim_pad)
            loss = self.criterion(output * target_weights, target * target_weights)
        else:
            loss = self.criterion(output, target)
        return loss * self.loss_weight


class FocalHeatmapLoss(KeypointMSELoss):
    """A class for calculating the modified focal loss for heatmap prediction.

    This loss function is exactly the same as the one used in CornerNet. It
    runs faster and costs a little bit more memory.

    `CornerNet: Detecting Objects as Paired Keypoints
    arXiv: <https://arxiv.org/abs/1808.01244>`_.

    Arguments:
        alpha (int): The alpha parameter in the focal loss equation.
        beta (int): The beta parameter in the focal loss equation.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self, alpha: 'int'=2, beta: 'int'=4, use_target_weight: 'bool'=False, skip_empty_channel: 'bool'=False, loss_weight: 'float'=1.0):
        super(FocalHeatmapLoss, self).__init__(use_target_weight, skip_empty_channel, loss_weight)
        self.alpha = alpha
        self.beta = beta

    def forward(self, output: 'Tensor', target: 'Tensor', target_weights: 'Optional[Tensor]'=None, mask: 'Optional[Tensor]'=None) ->Tensor:
        """Calculate the modified focal loss for heatmap prediction.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """
        _mask = self._get_mask(target, target_weights, mask)
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        if _mask is not None:
            pos_inds = pos_inds * _mask
            neg_inds = neg_inds * _mask
        neg_weights = torch.pow(1 - target, self.beta)
        pos_loss = torch.log(output) * torch.pow(1 - output, self.alpha) * pos_inds
        neg_loss = torch.log(1 - output) * torch.pow(output, self.alpha) * neg_weights * neg_inds
        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss.sum()
        else:
            loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss * self.loss_weight


class MLECCLoss(nn.Module):
    """Maximum Likelihood Estimation loss for Coordinate Classification.

    This loss function is designed to work with coordinate classification
    problems where the likelihood of each target coordinate is maximized.

    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
        mode (str): Specifies the mode of calculating loss:
            'linear' | 'square' | 'log'. Default: 'log'.
        use_target_weight (bool): If True, uses weighted loss. Different
            joint types may have different target weights. Defaults to False.
        loss_weight (float): Weight of the loss. Defaults to 1.0.

    Raises:
        AssertionError: If the `reduction` or `mode` arguments are not in the
                        expected choices.
        NotImplementedError: If the selected mode is not implemented.
    """

    def __init__(self, reduction: 'str'='mean', mode: 'str'='log', use_target_weight: 'bool'=False, loss_weight: 'float'=1.0):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), f"`reduction` should be either 'mean', 'sum', or 'none', but got {reduction}"
        assert mode in ('linear', 'square', 'log'), f"`mode` should be either 'linear', 'square', or 'log', but got {mode}"
        self.reduction = reduction
        self.mode = mode
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, outputs, targets, target_weight=None):
        """Forward pass for the MLECCLoss.

        Args:
            outputs (torch.Tensor): The predicted outputs.
            targets (torch.Tensor): The ground truth targets.
            target_weight (torch.Tensor, optional): Optional tensor of weights
                for each target.

        Returns:
            torch.Tensor: Calculated loss based on the specified mode and
                reduction.
        """
        assert len(outputs) == len(targets), 'Outputs and targets must have the same length'
        prob = 1.0
        for o, t in zip(outputs, targets):
            prob *= (o * t).sum(dim=-1)
        if self.mode == 'linear':
            loss = 1.0 - prob
        elif self.mode == 'square':
            loss = 1.0 - prob.pow(2)
        elif self.mode == 'log':
            loss = -torch.log(prob + 0.0001)
        loss[torch.isnan(loss)] = 0.0
        if self.use_target_weight:
            assert target_weight is not None
            for i in range(loss.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = loss * target_weight
        if self.reduction == 'sum':
            loss = loss.flatten(1).sum(dim=1)
        elif self.reduction == 'mean':
            loss = loss.flatten(1).mean(dim=1)
        return loss * self.loss_weight


class KDLoss(nn.Module):
    """PyTorch version of logit-based distillation from DWPose Modified from
    the official implementation.

    <https://github.com/IDEA-Research/DWPose>
    Args:
        weight (float, optional): Weight of dis_loss. Defaults to 1.0
    """

    def __init__(self, name, use_this, weight=1.0):
        super(KDLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.weight = weight

    def forward(self, pred, pred_t, beta, target_weight):
        ls_x, ls_y = pred
        lt_x, lt_y = pred_t
        lt_x = lt_x.detach()
        lt_y = lt_y.detach()
        num_joints = ls_x.size(1)
        loss = 0
        loss += self.loss(ls_x, lt_x, beta, target_weight)
        loss += self.loss(ls_y, lt_y, beta, target_weight)
        return loss / num_joints

    def loss(self, logit_s, logit_t, beta, weight):
        N = logit_s.shape[0]
        if len(logit_s.shape) == 3:
            K = logit_s.shape[1]
            logit_s = logit_s.reshape(N * K, -1)
            logit_t = logit_t.reshape(N * K, -1)
        s_i = self.log_softmax(logit_s * beta)
        t_i = F.softmax(logit_t * beta, dim=1)
        loss_all = torch.sum(self.kl_loss(s_i, t_i), dim=1)
        loss_all = loss_all.reshape(N, K).sum(dim=1).mean()
        loss_all = self.weight * loss_all
        return loss_all


class MultipleLossWrapper(nn.Module):
    """A wrapper to collect multiple loss functions together and return a list
    of losses in the same order.

    Args:
        losses (list): List of Loss Config
    """

    def __init__(self, losses: 'list'):
        super().__init__()
        self.num_losses = len(losses)
        loss_modules = []
        for loss_cfg in losses:
            t_loss = MODELS.build(loss_cfg)
            loss_modules.append(t_loss)
        self.loss_modules = nn.ModuleList(loss_modules)

    def forward(self, input_list, target_list, keypoint_weights=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            input_list (List[Tensor]): List of inputs.
            target_list (List[Tensor]): List of targets.
            keypoint_weights (Tensor[N, K, D]):
                Weights across different joint types.
        """
        assert isinstance(input_list, list), ''
        assert isinstance(target_list, list), ''
        assert len(input_list) == len(target_list), ''
        losses = []
        for i in range(self.num_losses):
            input_i = input_list[i]
            target_i = target_list[i]
            loss_i = self.loss_modules[i](input_i, target_i, keypoint_weights)
            losses.append(loss_i)
        return losses


class CombinedLoss(nn.ModuleDict):
    """A wrapper to combine multiple loss functions. These loss functions can
    have different input type (e.g. heatmaps or regression values), and can
    only be involed individually and explixitly.

    Args:
        losses (Dict[str, ConfigType]): The names and configs of loss
            functions to be wrapped

    Example::
        >>> heatmap_loss_cfg = dict(type='KeypointMSELoss')
        >>> ae_loss_cfg = dict(type='AssociativeEmbeddingLoss')
        >>> loss_module = CombinedLoss(
        ...     losses=dict(
        ...         heatmap_loss=heatmap_loss_cfg,
        ...         ae_loss=ae_loss_cfg))
        >>> loss_hm = loss_module.heatmap_loss(pred_heatmap, gt_heatmap)
        >>> loss_ae = loss_module.ae_loss(pred_tags, keypoint_indices)
    """

    def __init__(self, losses: 'Dict[str, ConfigType]'):
        super().__init__()
        for loss_name, loss_cfg in losses.items():
            self.add_module(loss_name, MODELS.build(loss_cfg))


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model

    `Density estimation using Real NVP
    arXiv: <https://arxiv.org/abs/1605.08803>`_.

    Code is modified from `the official implementation of RLE
    <https://github.com/Jeff-sjtu/res-loglikelihood-regression>`_.

    See also `real-nvp-pytorch
    <https://github.com/senya-ashukha/real-nvp-pytorch>`_.
    """

    @staticmethod
    def get_scale_net():
        """Get the scale model in a single invertable mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def get_trans_net():
        """Get the translation model in a single invertable mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super(RealNVP, self).__init__()
        self.register_buffer('loc', torch.zeros(2))
        self.register_buffer('cov', torch.eye(2))
        self.register_buffer('mask', torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))
        self.s = torch.nn.ModuleList([self.get_scale_net() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.get_trans_net() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialization model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping form the data space to the latent space and calculate
        the log determinant of the Jacobian matrix."""
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det


class RLELoss(nn.Module):
    """RLE Loss.

    `Human Pose Regression With Residual Log-Likelihood Estimation
    arXiv: <https://arxiv.org/abs/2107.11291>`_.

    Code is modified from `the official implementation
    <https://github.com/Jeff-sjtu/res-loglikelihood-regression>`_.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
        residual (bool): Option to add L1 loss and let the flow
            learn the residual error distribution.
        q_dis (string): Option for the identity Q(error) distribution,
            Options: "laplace" or "gaussian"
    """

    def __init__(self, use_target_weight=False, size_average=True, residual=True, q_distribution='laplace'):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual
        self.q_distribution = q_distribution
        self.flow_model = RealNVP()

    def forward(self, pred, sigma, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            pred (Tensor[N, K, D]): Output regression.
            sigma (Tensor[N, K, D]): Output sigma.
            target (Tensor[N, K, D]): Target regression.
            target_weight (Tensor[N, K, D]):
                Weights across different joint types.
        """
        sigma = sigma.sigmoid()
        error = (pred - target) / (sigma + 1e-09)
        log_phi = self.flow_model.log_prob(error.reshape(-1, 2))
        log_phi = log_phi.reshape(target.shape[0], target.shape[1], 1)
        log_sigma = torch.log(sigma).reshape(target.shape[0], target.shape[1], 2)
        nf_loss = log_sigma - log_phi
        if self.residual:
            assert self.q_distribution in ['laplace', 'gaussian']
            if self.q_distribution == 'laplace':
                loss_q = torch.log(sigma * 2) + torch.abs(error)
            else:
                loss_q = torch.log(sigma * math.sqrt(2 * math.pi)) + 0.5 * error ** 2
            loss = nf_loss + loss_q
        else:
            loss = nf_loss
        if self.use_target_weight:
            assert target_weight is not None
            loss *= target_weight
        if self.size_average:
            loss /= len(loss)
        return loss.sum()


class SmoothL1Loss(nn.Module):
    """SmoothL1Loss loss.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.0):
        super().__init__()
        self.criterion = F.smooth_l1_loss
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            assert output.ndim >= target_weight.ndim
            for i in range(output.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)
        return loss * self.loss_weight


class SoftWeightSmoothL1Loss(nn.Module):
    """Smooth L1 loss with soft weight for regression.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        supervise_empty (bool): Whether to supervise the output with zero
            weight.
        beta (float):  Specifies the threshold at which to change between
            L1 and L2 loss.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, supervise_empty=True, beta=1.0, loss_weight=1.0):
        super().__init__()
        reduction = 'none' if use_target_weight else 'mean'
        self.criterion = partial(self.smooth_l1_loss, reduction=reduction, beta=beta)
        self.supervise_empty = supervise_empty
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    @staticmethod
    def smooth_l1_loss(input, target, reduction='none', beta=1.0):
        """Re-implement torch.nn.functional.smooth_l1_loss with beta to support
        pytorch <= 1.6."""
        delta = input - target
        mask = delta.abs() < beta
        delta[mask] = delta[mask].pow(2) / (2 * beta)
        delta[~mask] = delta[~mask].abs() - beta / 2
        if reduction == 'mean':
            return delta.mean()
        elif reduction == 'sum':
            return delta.sum()
        elif reduction == 'none':
            return delta
        else:
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', but got '{reduction}'")

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            assert output.ndim >= target_weight.ndim
            for i in range(output.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = self.criterion(output, target) * target_weight
            if self.supervise_empty:
                loss = loss.mean()
            else:
                num_elements = torch.nonzero(target_weight > 0).size()[0]
                loss = loss.sum() / max(num_elements, 1.0)
        else:
            loss = self.criterion(output, target)
        return loss * self.loss_weight


class WingLoss(nn.Module):
    """Wing Loss. paper ref: 'Wing Loss for Robust Facial Landmark Localisation
    with Convolutional Neural Networks' Feng et al. CVPR'2018.

    Args:
        omega (float): Also referred to as width.
        epsilon (float): Also referred to as curvature.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, omega=10.0, epsilon=2.0, use_target_weight=False, loss_weight=1.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.C = self.omega * (1.0 - math.log(1.0 + self.omega / self.epsilon))

    def criterion(self, pred, target):
        """Criterion of wingloss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            pred (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
        """
        delta = (target - pred).abs()
        losses = torch.where(delta < self.omega, self.omega * torch.log(1.0 + delta / self.epsilon), delta - self.C)
        return torch.mean(torch.sum(losses, dim=[1, 2]), dim=0)

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N,K,D]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)
        return loss * self.loss_weight


class SoftWingLoss(nn.Module):
    """Soft Wing Loss 'Structure-Coherent Deep Feature Learning for Robust Face
    Alignment' Lin et al. TIP'2021.

    loss =
        1. |x|                           , if |x| < omega1
        2. omega2*ln(1+|x|/epsilon) + B, if |x| >= omega1

    Args:
        omega1 (float): The first threshold.
        omega2 (float): The second threshold.
        epsilon (float): Also referred to as curvature.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, omega1=2.0, omega2=20.0, epsilon=0.5, use_target_weight=False, loss_weight=1.0):
        super().__init__()
        self.omega1 = omega1
        self.omega2 = omega2
        self.epsilon = epsilon
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.B = self.omega1 - self.omega2 * math.log(1.0 + self.omega1 / self.epsilon)

    def criterion(self, pred, target):
        """Criterion of wingloss.

        Note:
            batch_size: N
            num_keypoints: K
            dimension of keypoints: D (D=2 or D=3)

        Args:
            pred (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
        """
        delta = (target - pred).abs()
        losses = torch.where(delta < self.omega1, delta, self.omega2 * torch.log(1.0 + delta / self.epsilon) + self.B)
        return torch.mean(torch.sum(losses, dim=[1, 2]), dim=0)

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            batch_size: N
            num_keypoints: K
            dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)
        return loss * self.loss_weight


class MPJPEVelocityJointLoss(nn.Module):
    """MPJPE (Mean Per Joint Position Error) loss.

    Args:
        loss_weight (float): Weight of the loss. Default: 1.0.
        lambda_scale (float): Factor of the N-MPJPE loss. Default: 0.5.
        lambda_3d_velocity (float): Factor of the velocity loss. Default: 20.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.0, lambda_scale=0.5, lambda_3d_velocity=20.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.lambda_scale = lambda_scale
        self.lambda_3d_velocity = lambda_3d_velocity

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N,K,D]):
                Weights across different joint types.
        """
        norm_output = torch.mean(torch.sum(torch.square(output), dim=-1, keepdim=True), dim=-2, keepdim=True)
        norm_target = torch.mean(torch.sum(target * output, dim=-1, keepdim=True), dim=-2, keepdim=True)
        velocity_output = output[..., 1:, :, :] - output[..., :-1, :, :]
        velocity_target = target[..., 1:, :, :] - target[..., :-1, :, :]
        if self.use_target_weight:
            assert target_weight is not None
            mpjpe = torch.mean(torch.norm((output - target) * target_weight, dim=-1))
            nmpjpe = torch.mean(torch.norm((norm_target / norm_output * output - target) * target_weight, dim=-1))
            loss_3d_velocity = torch.mean(torch.norm((velocity_output - velocity_target) * target_weight, dim=-1))
        else:
            mpjpe = torch.mean(torch.norm(output - target, dim=-1))
            nmpjpe = torch.mean(torch.norm(norm_target / norm_output * output - target, dim=-1))
            loss_3d_velocity = torch.mean(torch.norm(velocity_output - velocity_target, dim=-1))
        loss = mpjpe + nmpjpe * self.lambda_scale + loss_3d_velocity * self.lambda_3d_velocity
        return loss * self.loss_weight


class MPJPELoss(nn.Module):
    """MPJPE (Mean Per Joint Position Error) loss.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N,K,D]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            loss = torch.mean(torch.norm((output - target) * target_weight, dim=-1))
        else:
            loss = torch.mean(torch.norm(output - target, dim=-1))
        return loss * self.loss_weight


class L1Loss(nn.Module):
    """L1Loss loss."""

    def __init__(self, reduction='mean', use_target_weight=False, loss_weight=1.0):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), f"the argument `reduction` should be either 'mean', 'sum' or 'none', but got {reduction}"
        self.criterion = partial(F.l1_loss, reduction=reduction)
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output regression.
            target (torch.Tensor[N, K, 2]): Target regression.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            for _ in range(target.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)
        return loss * self.loss_weight


class MSELoss(nn.Module):
    """MSE loss for coordinate regression."""

    def __init__(self, use_target_weight=False, loss_weight=1.0):
        super().__init__()
        self.criterion = F.mse_loss
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output regression.
            target (torch.Tensor[N, K, 2]): Target regression.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)
        return loss * self.loss_weight


class BoneLoss(nn.Module):
    """Bone length loss.

    Args:
        joint_parents (list): Indices of each joint's parent joint.
        use_target_weight (bool): Option to use weighted bone loss.
            Different bone types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, joint_parents, use_target_weight: 'bool'=False, loss_weight: 'float'=1.0, loss_name: 'str'='loss_bone'):
        super().__init__()
        self.joint_parents = joint_parents
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.non_root_indices = []
        for i in range(len(self.joint_parents)):
            if i != self.joint_parents[i]:
                self.non_root_indices.append(i)
        self._loss_name = loss_name

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K-1]):
                Weights across different bone types.
        """
        output_bone = torch.norm(output - output[:, self.joint_parents, :], dim=-1)[:, self.non_root_indices]
        target_bone = torch.norm(target - target[:, self.joint_parents, :], dim=-1)[:, self.non_root_indices]
        if self.use_target_weight:
            assert target_weight is not None
            target_weight = target_weight[:, self.non_root_indices]
            loss = torch.mean(torch.abs((output_bone * target_weight).mean(dim=0) - (target_bone * target_weight).mean(dim=0)))
        else:
            loss = torch.mean(torch.abs(output_bone.mean(dim=0) - target_bone.mean(dim=0)))
        return loss * self.loss_weight

    @property
    def loss_name(self):
        """Loss Name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


class SemiSupervisionLoss(nn.Module):
    """Semi-supervision loss for unlabeled data. It is composed of projection
    loss and bone loss.

    Paper ref: `3D human pose estimation in video with temporal convolutions
    and semi-supervised training` Dario Pavllo et al. CVPR'2019.

    Args:
        joint_parents (list): Indices of each joint's parent joint.
        projection_loss_weight (float): Weight for projection loss.
        bone_loss_weight (float): Weight for bone loss.
        warmup_iterations (int): Number of warmup iterations. In the first
            `warmup_iterations` iterations, the model is trained only on
            labeled data, and semi-supervision loss will be 0.
            This is a workaround since currently we cannot access
            epoch number in loss functions. Note that the iteration number in
            an epoch can be changed due to different GPU numbers in multi-GPU
            settings. So please set this parameter carefully.
            warmup_iterations = dataset_size // samples_per_gpu // gpu_num
            * warmup_epochs
    """

    def __init__(self, joint_parents, projection_loss_weight=1.0, bone_loss_weight=1.0, warmup_iterations=0):
        super().__init__()
        self.criterion_projection = MPJPELoss(loss_weight=projection_loss_weight)
        self.criterion_bone = BoneLoss(joint_parents, loss_weight=bone_loss_weight)
        self.warmup_iterations = warmup_iterations
        self.num_iterations = 0

    @staticmethod
    def project_joints(x, intrinsics):
        """Project 3D joint coordinates to 2D image plane using camera
        intrinsic parameters.

        Args:
            x (torch.Tensor[N, K, 3]): 3D joint coordinates.
            intrinsics (torch.Tensor[N, 4] | torch.Tensor[N, 9]): Camera
                intrinsics: f (2), c (2), k (3), p (2).
        """
        while intrinsics.dim() < x.dim():
            intrinsics.unsqueeze_(1)
        f = intrinsics[..., :2]
        c = intrinsics[..., 2:4]
        _x = torch.clamp(x[:, :, :2] / x[:, :, 2:], -1, 1)
        if intrinsics.shape[-1] == 9:
            k = intrinsics[..., 4:7]
            p = intrinsics[..., 7:9]
            r2 = torch.sum(_x[:, :, :2] ** 2, dim=-1, keepdim=True)
            radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=-1), dim=-1, keepdim=True)
            tan = torch.sum(p * _x, dim=-1, keepdim=True)
            _x = _x * (radial + tan) + p * r2
        _x = f * _x + c
        return _x

    def forward(self, output, target):
        losses = dict()
        self.num_iterations += 1
        if self.num_iterations <= self.warmup_iterations:
            return losses
        labeled_pose = output['labeled_pose']
        unlabeled_pose = output['unlabeled_pose']
        unlabeled_traj = output['unlabeled_traj']
        unlabeled_target_2d = target['unlabeled_target_2d']
        intrinsics = target['intrinsics']
        unlabeled_output = unlabeled_pose + unlabeled_traj
        unlabeled_output_2d = self.project_joints(unlabeled_output, intrinsics)
        loss_proj = self.criterion_projection(unlabeled_output_2d, unlabeled_target_2d, None)
        losses['proj_loss'] = loss_proj
        loss_bone = self.criterion_bone(unlabeled_pose, labeled_pose, None)
        losses['bone_loss'] = loss_bone
        return losses


def parse_pose_metainfo(metainfo: 'dict'):
    """Load meta information of pose dataset and check its integrity.

    Args:
        metainfo (dict): Raw data of pose meta information, which should
            contain following contents:

            - "dataset_name" (str): The name of the dataset
            - "keypoint_info" (dict): The keypoint-related meta information,
                e.g., name, upper/lower body, and symmetry
            - "skeleton_info" (dict): The skeleton-related meta information,
                e.g., start/end keypoint of limbs
            - "joint_weights" (list[float]): The loss weights of keypoints
            - "sigmas" (list[float]): The keypoint distribution parameters
                to calculate OKS score. See `COCO keypoint evaluation
                <https://cocodataset.org/#keypoints-eval>`__.

            An example of metainfo is shown as follows.

            .. code-block:: none
                {
                    "dataset_name": "coco",
                    "keypoint_info":
                    {
                        0:
                        {
                            "name": "nose",
                            "type": "upper",
                            "swap": "",
                            "color": [51, 153, 255],
                        },
                        1:
                        {
                            "name": "right_eye",
                            "type": "upper",
                            "swap": "left_eye",
                            "color": [51, 153, 255],
                        },
                        ...
                    },
                    "skeleton_info":
                    {
                        0:
                        {
                            "link": ("left_ankle", "left_knee"),
                            "color": [0, 255, 0],
                        },
                        ...
                    },
                    "joint_weights": [1., 1., ...],
                    "sigmas": [0.026, 0.025, ...],
                }


            A special case is that `metainfo` can have the key "from_file",
            which should be the path of a config file. In this case, the
            actual metainfo will be loaded by:

            .. code-block:: python
                metainfo = mmengine.Config.fromfile(metainfo['from_file'])

    Returns:
        Dict: pose meta information that contains following contents:

        - "dataset_name" (str): Same as ``"dataset_name"`` in the input
        - "num_keypoints" (int): Number of keypoints
        - "keypoint_id2name" (dict): Mapping from keypoint id to name
        - "keypoint_name2id" (dict): Mapping from keypoint name to id
        - "upper_body_ids" (list): Ids of upper-body keypoint
        - "lower_body_ids" (list): Ids of lower-body keypoint
        - "flip_indices" (list): The Id of each keypoint's symmetric keypoint
        - "flip_pairs" (list): The Ids of symmetric keypoint pairs
        - "keypoint_colors" (numpy.ndarray): The keypoint color matrix of
            shape [K, 3], where each row is the color of one keypint in bgr
        - "num_skeleton_links" (int): The number of links
        - "skeleton_links" (list): The links represented by Id pairs of start
             and end points
        - "skeleton_link_colors" (numpy.ndarray): The link color matrix
        - "dataset_keypoint_weights" (numpy.ndarray): Same as the
            ``"joint_weights"`` in the input
        - "sigmas" (numpy.ndarray): Same as the ``"sigmas"`` in the input
    """
    if 'from_file' in metainfo:
        cfg_file = metainfo['from_file']
        if not osp.isfile(cfg_file):
            mmpose_path = osp.dirname(mmpose.__file__)
            _cfg_file = osp.join(mmpose_path, '.mim', 'configs', '_base_', 'datasets', osp.basename(cfg_file))
            if osp.isfile(_cfg_file):
                warnings.warn(f'The metainfo config file "{cfg_file}" does not exist. A matched config file "{_cfg_file}" will be used instead.')
                cfg_file = _cfg_file
            else:
                raise FileNotFoundError(f'The metainfo config file "{cfg_file}" does not exist.')
        metainfo = Config.fromfile(cfg_file).dataset_info
    assert 'dataset_name' in metainfo
    assert 'keypoint_info' in metainfo
    assert 'skeleton_info' in metainfo
    assert 'joint_weights' in metainfo
    assert 'sigmas' in metainfo
    parsed = dict(dataset_name=None, num_keypoints=None, keypoint_id2name={}, keypoint_name2id={}, upper_body_ids=[], lower_body_ids=[], flip_indices=[], flip_pairs=[], keypoint_colors=[], num_skeleton_links=None, skeleton_links=[], skeleton_link_colors=[], dataset_keypoint_weights=None, sigmas=None)
    parsed['dataset_name'] = metainfo['dataset_name']
    parsed['num_keypoints'] = len(metainfo['keypoint_info'])
    for kpt_id, kpt in metainfo['keypoint_info'].items():
        kpt_name = kpt['name']
        parsed['keypoint_id2name'][kpt_id] = kpt_name
        parsed['keypoint_name2id'][kpt_name] = kpt_id
        parsed['keypoint_colors'].append(kpt.get('color', [255, 128, 0]))
        kpt_type = kpt.get('type', '')
        if kpt_type == 'upper':
            parsed['upper_body_ids'].append(kpt_id)
        elif kpt_type == 'lower':
            parsed['lower_body_ids'].append(kpt_id)
        swap_kpt = kpt.get('swap', '')
        if swap_kpt == kpt_name or swap_kpt == '':
            parsed['flip_indices'].append(kpt_name)
        else:
            parsed['flip_indices'].append(swap_kpt)
            pair = swap_kpt, kpt_name
            if pair not in parsed['flip_pairs']:
                parsed['flip_pairs'].append(pair)
    parsed['num_skeleton_links'] = len(metainfo['skeleton_info'])
    for _, sk in metainfo['skeleton_info'].items():
        parsed['skeleton_links'].append(sk['link'])
        parsed['skeleton_link_colors'].append(sk.get('color', [96, 96, 255]))
    parsed['dataset_keypoint_weights'] = np.array(metainfo['joint_weights'], dtype=np.float32)
    parsed['sigmas'] = np.array(metainfo['sigmas'], dtype=np.float32)
    if 'stats_info' in metainfo:
        parsed['stats_info'] = {}
        for name, val in metainfo['stats_info'].items():
            parsed['stats_info'][name] = np.array(val, dtype=np.float32)

    def _map(src, mapping: 'dict'):
        if isinstance(src, (list, tuple)):
            cls = type(src)
            return cls(_map(s, mapping) for s in src)
        else:
            return mapping[src]
    parsed['flip_pairs'] = _map(parsed['flip_pairs'], mapping=parsed['keypoint_name2id'])
    parsed['flip_indices'] = _map(parsed['flip_indices'], mapping=parsed['keypoint_name2id'])
    parsed['skeleton_links'] = _map(parsed['skeleton_links'], mapping=parsed['keypoint_name2id'])
    parsed['keypoint_colors'] = np.array(parsed['keypoint_colors'], dtype=np.uint8)
    parsed['skeleton_link_colors'] = np.array(parsed['skeleton_link_colors'], dtype=np.uint8)
    return parsed


class OKSLoss(nn.Module):
    """A PyTorch implementation of the Object Keypoint Similarity (OKS) loss as
    described in the paper "YOLO-Pose: Enhancing YOLO for Multi Person Pose
    Estimation Using Object Keypoint Similarity Loss" by Debapriya et al.
    (2022).

    The OKS loss is used for keypoint-based object recognition and consists
    of a measure of the similarity between predicted and ground truth
    keypoint locations, adjusted by the size of the object in the image.

    The loss function takes as input the predicted keypoint locations, the
    ground truth keypoint locations, a mask indicating which keypoints are
    valid, and bounding boxes for the objects.

    Args:
        metainfo (Optional[str]): Path to a JSON file containing information
            about the dataset's annotations.
        reduction (str): Options are "none", "mean" and "sum".
        eps (float): Epsilon to avoid log(0).
        loss_weight (float): Weight of the loss. Default: 1.0.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'linear'
        norm_target_weight (bool): whether to normalize the target weight
            with number of visible keypoints. Defaults to False.
    """

    def __init__(self, metainfo: 'Optional[str]'=None, reduction='mean', mode='linear', eps=1e-08, norm_target_weight=False, loss_weight=1.0):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), f"the argument `reduction` should be either 'mean', 'sum' or 'none', but got {reduction}"
        assert mode in ('linear', 'square', 'log'), f"the argument `reduction` should be either 'linear', 'square' or 'log', but got {mode}"
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.mode = mode
        self.norm_target_weight = norm_target_weight
        self.eps = eps
        if metainfo is not None:
            metainfo = parse_pose_metainfo(dict(from_file=metainfo))
            sigmas = metainfo.get('sigmas', None)
            if sigmas is not None:
                self.register_buffer('sigmas', torch.as_tensor(sigmas))

    def forward(self, output, target, target_weight=None, areas=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints coordinates.
            target (torch.Tensor[N, K, 2]): Target keypoints coordinates..
            target_weight (torch.Tensor[N, K]): Loss weight for each keypoint.
            areas (torch.Tensor[N]): Instance size which is adopted as
                normalization factor.
        """
        dist = torch.norm(output - target, dim=-1)
        if areas is not None:
            dist = dist / areas.pow(0.5).clip(min=self.eps).unsqueeze(-1)
        if hasattr(self, 'sigmas'):
            sigmas = self.sigmas.reshape(*((1,) * (dist.ndim - 1)), -1)
            dist = dist / (sigmas * 2)
        oks = torch.exp(-dist.pow(2) / 2)
        if target_weight is not None:
            if self.norm_target_weight:
                target_weight = target_weight / target_weight.sum(dim=-1, keepdims=True).clip(min=self.eps)
            else:
                target_weight = target_weight / target_weight.size(-1)
            oks = oks * target_weight
        oks = oks.sum(dim=-1)
        if self.mode == 'linear':
            loss = 1 - oks
        elif self.mode == 'square':
            loss = 1 - oks.pow(2)
        elif self.mode == 'log':
            loss = -oks.log()
        else:
            raise NotImplementedError()
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss * self.loss_weight


def resize(input: 'torch.Tensor', size: 'Optional[Union[Tuple[int, int], torch.Size]]'=None, scale_factor: 'Optional[float]'=None, mode: 'str'='nearest', align_corners: 'Optional[bool]'=None, warning: 'bool'=True) ->torch.Tensor:
    """Resize a given input tensor using specified size or scale_factor.

    Args:
        input (torch.Tensor): The input tensor to be resized.
        size (Optional[Union[Tuple[int, int], torch.Size]]): The desired
            output size. Defaults to None.
        scale_factor (Optional[float]): The scaling factor for resizing.
            Defaults to None.
        mode (str): The interpolation mode. Defaults to 'nearest'.
        align_corners (Optional[bool]): Determines whether to align the
            corners when using certain interpolation modes. Defaults to None.
        warning (bool): Whether to display a warning when the input and
            output sizes are not ideal for alignment. Defaults to True.

    Returns:
        torch.Tensor: The resized tensor.
    """
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1) and (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1):
                    warnings.warn(f'When align_corners={align_corners}, the output would be more aligned if input size {input_h, input_w} is `x+1` and out size {output_h, output_w} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class FeatureMapProcessor(nn.Module):
    """A PyTorch module for selecting, concatenating, and rescaling feature
    maps.

    Args:
        select_index (Optional[Union[int, Tuple[int]]], optional): Index or
            indices of feature maps to select. Defaults to None, which means
            all feature maps are used.
        concat (bool, optional): Whether to concatenate the selected feature
            maps. Defaults to False.
        scale_factor (float, optional): The scaling factor to apply to the
            feature maps. Defaults to 1.0.
        apply_relu (bool, optional): Whether to apply ReLU on input feature
            maps. Defaults to False.
        align_corners (bool, optional): Whether to align corners when resizing
            the feature maps. Defaults to False.
    """

    def __init__(self, select_index: 'Optional[Union[int, Tuple[int]]]'=None, concat: 'bool'=False, scale_factor: 'float'=1.0, apply_relu: 'bool'=False, align_corners: 'bool'=False):
        super().__init__()
        if isinstance(select_index, int):
            select_index = select_index,
        self.select_index = select_index
        self.concat = concat
        assert scale_factor > 0, f'the argument `scale_factor` must be positive, but got {scale_factor}'
        self.scale_factor = scale_factor
        self.apply_relu = apply_relu
        self.align_corners = align_corners

    def forward(self, inputs: 'Union[Tensor, Sequence[Tensor]]') ->Union[Tensor, List[Tensor]]:
        if not isinstance(inputs, (tuple, list)):
            sequential_input = False
            inputs = [inputs]
        else:
            sequential_input = True
            if self.select_index is not None:
                inputs = [inputs[i] for i in self.select_index]
            if self.concat:
                inputs = self._concat(inputs)
        if self.apply_relu:
            inputs = [F.relu(x) for x in inputs]
        if self.scale_factor != 1.0:
            inputs = self._rescale(inputs)
        if not sequential_input:
            inputs = inputs[0]
        return inputs

    def _concat(self, inputs: 'Sequence[Tensor]') ->List[Tensor]:
        size = inputs[0].shape[-2:]
        resized_inputs = [resize(x, size=size, mode='bilinear', align_corners=self.align_corners) for x in inputs]
        return [torch.cat(resized_inputs, dim=1)]

    def _rescale(self, inputs: 'Sequence[Tensor]') ->List[Tensor]:
        rescaled_inputs = [resize(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=self.align_corners) for x in inputs]
        return rescaled_inputs


class FPN(nn.Module):
    """Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, add_extra_convs=False, relu_before_extra_convs=False, no_norm_on_lateral=False, conv_cfg=None, norm_cfg=None, act_cfg=None, upsample_cfg=dict(mode='nearest')):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            self.add_extra_convs = 'on_input'
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg if not self.no_norm_on_lateral else None, act_cfg=act_cfg, inplace=False)
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(in_channels, out_channels, 3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return outs


class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.
    """

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        """Forward function."""
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple([out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, list):
            outs = [self.gap(x) for x in inputs]
            outs = [out.view(x.size(0), -1) for out, x in zip(outs, inputs)]
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs


class FrozenBatchNorm2d(torch.nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are
    fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without
    which any other models than torchvision.models.resnet[18,34,50,101] produce
    nans.
    """

    def __init__(self, n, eps: 'int'=1e-05):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Scale(nn.Module):
    """Scale vector by element multiplications.

    Args:
        dim (int): The dimension of the scale vector.
        init_value (float, optional): The initial value of the scale vector.
            Defaults to 1.0.
        trainable (bool, optional): Whether the scale vector is trainable.
            Defaults to True.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        """Forward function."""
        return x * self.scale


class ScaleNorm(nn.Module):
    """Scale Norm.

    Args:
        dim (int): The dimension of the scale vector.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.

    Reference:
        `Transformers without Tears: Improving the Normalization
        of Self-Attention <https://arxiv.org/abs/1910.05895>`_
    """

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The tensor after applying scale norm.
        """
        if torch.onnx.is_in_onnx_export() and digit_version(TORCH_VERSION) >= digit_version('1.12'):
            norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        else:
            norm = torch.norm(x, dim=-1, keepdim=True)
        norm = norm * self.scale
        return x / norm.clamp(min=self.eps) * self.g


def rope(x, dim):
    """Applies Rotary Position Embedding to input tensor.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int | list[int]): The spatial dimension(s) to apply
            rotary position embedding.

    Returns:
        torch.Tensor: The tensor after applying rotary position
            embedding.

    Reference:
        `RoFormer: Enhanced Transformer with Rotary
        Position Embedding <https://arxiv.org/abs/2104.09864>`_
    """
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]
    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i
    position = torch.reshape(torch.arange(total_len, dtype=torch.int, device=x.device), spatial_shape)
    for i in range(dim[-1] + 1, len(shape) - 1, 1):
        position = torch.unsqueeze(position, dim=-1)
    half_size = shape[-1] // 2
    freq_seq = -torch.arange(half_size, dtype=torch.int, device=x.device) / float(half_size)
    inv_freq = 10000 ** -freq_seq
    sinusoid = position[..., None] * inv_freq[None, None, :]
    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RTMCCBlock(nn.Module):
    """Gated Attention Unit (GAU) in RTMBlock.

    Args:
        num_token (int): The number of tokens.
        in_token_dims (int): The input token dimension.
        out_token_dims (int): The output token dimension.
        expansion_factor (int, optional): The expansion factor of the
            intermediate token dimension. Defaults to 2.
        s (int, optional): The self-attention feature dimension.
            Defaults to 128.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.
        dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
        drop_path (float, optional): The drop path rate. Defaults to 0.0.
        attn_type (str, optional): Type of attention which should be one of
            the following options:

            - 'self-attn': Self-attention.
            - 'cross-attn': Cross-attention.

            Defaults to 'self-attn'.
        act_fn (str, optional): The activation function which should be one
            of the following options:

            - 'ReLU': ReLU activation.
            - 'SiLU': SiLU activation.

            Defaults to 'SiLU'.
        bias (bool, optional): Whether to use bias in linear layers.
            Defaults to False.
        use_rel_bias (bool, optional): Whether to use relative bias.
            Defaults to True.
        pos_enc (bool, optional): Whether to use rotary position
            embedding. Defaults to False.

    Reference:
        `Transformer Quality in Linear Time
        <https://arxiv.org/abs/2202.10447>`_
    """

    def __init__(self, num_token, in_token_dims, out_token_dims, expansion_factor=2, s=128, eps=1e-05, dropout_rate=0.0, drop_path=0.0, attn_type='self-attn', act_fn='SiLU', bias=False, use_rel_bias=True, pos_enc=False):
        super(RTMCCBlock, self).__init__()
        self.s = s
        self.num_token = num_token
        self.use_rel_bias = use_rel_bias
        self.attn_type = attn_type
        self.pos_enc = pos_enc
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.e = int(in_token_dims * expansion_factor)
        if use_rel_bias:
            if attn_type == 'self-attn':
                self.w = nn.Parameter(torch.rand([2 * num_token - 1], dtype=torch.float))
            else:
                self.a = nn.Parameter(torch.rand([1, s], dtype=torch.float))
                self.b = nn.Parameter(torch.rand([1, s], dtype=torch.float))
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)
        if attn_type == 'self-attn':
            self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
            self.gamma = nn.Parameter(torch.rand((2, self.s)))
            self.beta = nn.Parameter(torch.rand((2, self.s)))
        else:
            self.uv = nn.Linear(in_token_dims, self.e + self.s, bias=bias)
            self.k_fc = nn.Linear(in_token_dims, self.s, bias=bias)
            self.v_fc = nn.Linear(in_token_dims, self.e, bias=bias)
            nn.init.xavier_uniform_(self.k_fc.weight)
            nn.init.xavier_uniform_(self.v_fc.weight)
        self.ln = ScaleNorm(in_token_dims, eps=eps)
        nn.init.xavier_uniform_(self.uv.weight)
        if act_fn == 'SiLU' or act_fn == nn.SiLU:
            assert digit_version(TORCH_VERSION) >= digit_version('1.7.0'), 'SiLU activation requires PyTorch version >= 1.7'
            self.act_fn = nn.SiLU(True)
        elif act_fn == 'ReLU' or act_fn == nn.ReLU:
            self.act_fn = nn.ReLU(True)
        else:
            raise NotImplementedError
        if in_token_dims == out_token_dims:
            self.shortcut = True
            self.res_scale = Scale(in_token_dims)
        else:
            self.shortcut = False
        self.sqrt_s = math.sqrt(s)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def rel_pos_bias(self, seq_len, k_len=None):
        """Add relative position bias."""
        if self.attn_type == 'self-attn':
            t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            a = rope(self.a.repeat(seq_len, 1), dim=0)
            b = rope(self.b.repeat(k_len, 1), dim=0)
            t = torch.bmm(a, b.permute(0, 2, 1))
        return t

    def _forward(self, inputs):
        """GAU Forward function."""
        if self.attn_type == 'self-attn':
            x = inputs
        else:
            x, k, v = inputs
        x = self.ln(x)
        uv = self.uv(x)
        uv = self.act_fn(uv)
        if self.attn_type == 'self-attn':
            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
            base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
            if self.pos_enc:
                base = rope(base, dim=1)
            q, k = torch.unbind(base, dim=2)
        else:
            u, q = torch.split(uv, [self.e, self.s], dim=2)
            k = self.k_fc(k)
            v = self.v_fc(v)
            if self.pos_enc:
                q = rope(q, 1)
                k = rope(k, 1)
        qk = torch.bmm(q, k.permute(0, 2, 1))
        if self.use_rel_bias:
            if self.attn_type == 'self-attn':
                bias = self.rel_pos_bias(q.size(1))
            else:
                bias = self.rel_pos_bias(q.size(1), k.size(1))
            qk += bias[:, :q.size(1), :k.size(1)]
        kernel = torch.square(F.relu(qk / self.sqrt_s))
        if self.dropout_rate > 0.0:
            kernel = self.dropout(kernel)
        x = u * torch.bmm(kernel, v)
        x = self.o(x)
        return x

    def forward(self, x):
        """Forward function."""
        if self.shortcut:
            if self.attn_type == 'cross-attn':
                res_shortcut = x[0]
            else:
                res_shortcut = x
            main_branch = self.drop_path(self._forward(x))
            return self.res_scale(res_shortcut) + main_branch
        else:
            return self.drop_path(self._forward(x))


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super(AdaptivePadding, self).__init__()
        assert padding in ('same', 'corner')
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        """Get horizontal and vertical padding shapes."""
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        """Forward function."""
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x


class ChannelWiseScale(nn.Module):
    """Scale vector by element multiplications.

    Args:
        dim (int): The dimension of the scale vector.
        init_value (float, optional): The initial value of the scale vector.
            Defaults to 1.0.
        trainable (bool, optional): Whether the scale vector is trainable.
            Defaults to True.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        """Forward function."""
        return x * self.scale


def norm_targets(targets, patch_size):
    assert patch_size % 2 == 1
    targets_ = targets
    targets_count = torch.ones_like(targets)
    targets_square = targets ** 2.0
    targets_mean = F.avg_pool2d(targets, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_square_mean = F.avg_pool2d(targets_square, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_count = F.avg_pool2d(targets_count, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=True) * patch_size ** 2
    targets_var = (targets_square_mean - targets_mean ** 2.0) * (targets_count / (targets_count - 1))
    targets_var = torch.clamp(targets_var, min=0.0)
    targets_ = (targets_ - targets_mean) / (targets_var + 1e-06) ** 0.5
    return targets_


class SimMIM(nn.Module):

    def __init__(self, config, encoder, encoder_stride, in_chans, patch_size):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.decoder = nn.Sequential(nn.Conv2d(in_channels=self.encoder.num_features, out_channels=self.encoder_stride ** 2 * 3, kernel_size=1), nn.PixelShuffle(self.encoder_stride))
        self.in_chans = in_chans
        self.patch_size = patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        if self.config.NORM_TARGET.ENABLE:
            x = norm_targets(x, self.config.NORM_TARGET.PATCH_SIZE)
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-05) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {('encoder.' + i) for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {('encoder.' + i) for i in self.encoder.no_weight_decay_keywords()}
        return {}


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinMLPBlock(nn.Module):
    """ Swin MLP Block.

    Args: dim (int): Number of input channels. input_resolution (tuple[int]):
    Input resolution. num_heads (int): Number of attention heads. window_size
    (int): Window size. shift_size (int): Shift size for SW-MSA. mlp_ratio (
    float): Ratio of mlp hidden dim to embedding dim. drop (float, optional):
    Dropout rate. Default: 0.0 drop_path (float, optional): Stochastic depth
    rate. Default: 0.0 act_layer (nn.Module, optional): Activation layer.
    Default: nn.GELU norm_layer (nn.Module, optional): Normalization layer.
    Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.padding = [self.window_size - self.shift_size, self.shift_size, self.window_size - self.shift_size, self.shift_size]
        self.norm1 = norm_layer(dim)
        self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2, self.num_heads * self.window_size ** 2, kernel_size=1, groups=self.num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], 'constant', 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
        x_windows_heads = x_windows_heads.transpose(1, 2)
        x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size, C // self.num_heads)
        spatial_mlp_windows = self.spatial_mlp(x_windows_heads)
        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size, C // self.num_heads).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        if self.shift_size > 0:
            nW = (H / self.window_size + 1) * (W / self.window_size + 1)
        else:
            nW = H * W / self.window_size / self.window_size
        flops += nW * self.dim * (self.window_size * self.window_size) * (self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    """ Patch Merging Layer.

    Args: input_resolution (tuple[int]): Resolution of input feature. dim (
    int): Number of input channels. norm_layer (nn.Module, optional):
    Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.reduction(x)
        x = self.norm(x)
        return x

    def extra_repr(self) ->str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        H, W = self.input_resolution
        flops = H // 2 * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative
    position bias. It supports both of shifted and non-shifted window.

    Args: dim (int): Number of input channels. window_size (tuple[int]): The
    height and width of the window. num_heads (int): Number of attention
    heads. qkv_bias (bool, optional):  If True, add a learnable bias to
    query, key, value. Default: True attn_drop (float, optional): Dropout
    ratio of attention weight. Default: 0.0 proj_drop (float, optional):
    Dropout ratio of output. Default: 0.0 pretrained_window_size (tuple[
    int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False))
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer('relative_coords_table', relative_coords_table)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args: x: input features with shape of (num_windows*B, N, C) mask: (
        0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01, device=x.device))).exp()
        attn = attn * logit_scale
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, window_size={self.window_size}, '(f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}')

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args: dim (int): Number of input channels. input_resolution (tuple[int]):
    Input resolution. num_heads (int): Number of attention heads. window_size
    (int): Window size. shift_size (int): Shift size for SW-MSA. mlp_ratio (
    float): Ratio of mlp hidden dim to embedding dim. qkv_bias (bool,
    optional): If True, add a learnable bias to query, key, value. Default:
    True drop (float, optional): Dropout rate. Default: 0.0 attn_drop (float,
    optional): Attention dropout rate. Default: 0.0 drop_path (float,
    optional): Stochastic depth rate. Default: 0.0 act_layer (nn.Module,
    optional): Activation layer. Default: nn.GELU norm_layer (nn.Module,
    optional): Normalization layer.  Default: nn.LayerNorm
    pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, pretrained_window_size=to_2tuple(pretrained_window_size))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args: dim (int): Number of input channels. input_resolution (tuple[int]):
    Input resolution. depth (int): Number of blocks. num_heads (int): Number
    of attention heads. window_size (int): Local window size. mlp_ratio (
    float): Ratio of mlp hidden dim to embedding dim. qkv_bias (bool,
    optional): If True, add a learnable bias to query, key, value. Default:
    True drop (float, optional): Dropout rate. Default: 0.0 attn_drop (float,
    optional): Attention dropout rate. Default: 0.0 drop_path (float | tuple[
    float], optional): Stochastic depth rate. Default: 0.0 norm_layer (
    nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    downsample (nn.Module | None, optional): Downsample layer at the end of
    the layer. Default: None use_checkpoint (bool): Whether to use
    checkpointing to save memory. Default: False. pretrained_window_size (
    int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, pretrained_window_size=pretrained_window_size) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class SwinMLP(nn.Module):
    """ Swin MLP

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head.
        Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin MLP layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch
        embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding.
        Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory.
        Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, drop=drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // 2 ** self.num_layers
        flops += self.num_features * self.num_classes
        return flops


class SwinTransformer(nn.Module):
    """ Swin Transformer A PyTorch impl of : `Swin Transformer: Hierarchical
    Vision Transformer using Shifted Windows`  -
    https://arxiv.org/pdf/2103.14030

    Args: img_size (int | tuple(int)): Input image size. Default 224
    patch_size (int | tuple(int)): Patch size. Default: 4 in_chans (int):
    Number of input image channels. Default: 3 num_classes (int): Number of
    classes for classification head. Default: 1000 embed_dim (int): Patch
    embedding dimension. Default: 96 depths (tuple(int)): Depth of each Swin
    Transformer layer. num_heads (tuple(int)): Number of attention heads in
    different layers. window_size (int): Window size. Default: 7 mlp_ratio (
    float): Ratio of mlp hidden dim to embedding dim. Default: 4 qkv_bias (
    bool): If True, add a learnable bias to query, key, value. Default: True
    qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
    Default: None drop_rate (float): Dropout rate. Default: 0 attn_drop_rate
    (float): Attention dropout rate. Default: 0 drop_path_rate (float):
    Stochastic depth rate. Default: 0.1 norm_layer (nn.Module): Normalization
    layer. Default: nn.LayerNorm. ape (bool): If True, add absolute position
    embedding to the patch embedding. Default: False patch_norm (bool): If
    True, add normalization after patch embedding. Default: True
    use_checkpoint (bool): Whether to use checkpointing to save memory.
    Default: False fused_window_process (bool, optional): If True, use one
    kernel to fused window shift & window partition for acceleration, similar
    for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint, fused_window_process=fused_window_process)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // 2 ** self.num_layers
        flops += self.num_features * self.num_classes
        return flops


class MoEMlp(nn.Module):

    def __init__(self, in_features, hidden_features, num_local_experts, top_value, capacity_factor=1.25, cosine_router=False, normalize_gate=False, use_bpr=True, is_gshard_loss=True, gate_noise=1.0, cosine_router_dim=256, cosine_router_init_t=0.5, moe_drop=0.0, init_std=0.02, mlp_fc2_bias=True):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_local_experts = num_local_experts
        self.top_value = top_value
        self.capacity_factor = capacity_factor
        self.cosine_router = cosine_router
        self.normalize_gate = normalize_gate
        self.use_bpr = use_bpr
        self.init_std = init_std
        self.mlp_fc2_bias = mlp_fc2_bias
        self.dist_rank = dist.get_rank()
        self._dropout = nn.Dropout(p=moe_drop)
        _gate_type = {'type': 'cosine_top' if cosine_router else 'top', 'k': top_value, 'capacity_factor': capacity_factor, 'gate_noise': gate_noise, 'fp32_gate': True}
        if cosine_router:
            _gate_type['proj_dim'] = cosine_router_dim
            _gate_type['init_t'] = cosine_router_init_t
        self._moe_layer = tutel_moe.moe_layer(gate_type=_gate_type, model_dim=in_features, experts={'type': 'ffn', 'count_per_node': num_local_experts, 'hidden_size_per_expert': hidden_features, 'activation_fn': lambda x: self._dropout(F.gelu(x))}, scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True), seeds=(1, self.dist_rank + 1, self.dist_rank + 1), batch_prioritized_routing=use_bpr, normalize_gate=normalize_gate, is_gshard_loss=is_gshard_loss)
        if not self.mlp_fc2_bias:
            self._moe_layer.experts.batched_fc2_bias.requires_grad = False

    def forward(self, x):
        x = self._moe_layer(x)
        return x, x.l_aux

    def extra_repr(self) ->str:
        return f'[Statistics-{self.dist_rank}] param count for MoE, in_features = {self.in_features}, hidden_features = {self.hidden_features}, num_local_experts = {self.num_local_experts}, top_value = {self.top_value}, cosine_router={self.cosine_router} normalize_gate={self.normalize_gate}, use_bpr = {self.use_bpr}'

    def _init_weights(self):
        if hasattr(self._moe_layer, 'experts'):
            trunc_normal_(self._moe_layer.experts.batched_fc1_w, std=self.init_std)
            trunc_normal_(self._moe_layer.experts.batched_fc2_w, std=self.init_std)
            nn.init.constant_(self._moe_layer.experts.batched_fc1_bias, 0)
            nn.init.constant_(self._moe_layer.experts.batched_fc2_bias, 0)


class SwinTransformerMoE(nn.Module):
    """ Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision
        Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head.
        Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if
        set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch
        embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding.
        Default: True
        mlp_fc2_bias (bool): Whether to add bias in fc2 of Mlp. Default: True
        init_std: Initialization std. Default: 0.02
        use_checkpoint (bool): Whether to use checkpointing to save memory.
        Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each
        layer.
        moe_blocks (tuple(tuple(int))): The index of each MoE block in each
        layer.
        num_local_experts (int): number of local experts in each device (
        GPU). Default: 1
        top_value (int): the value of k in top-k gating. Default: 1
        capacity_factor (float): the capacity factor in MoE. Default: 1.25
        cosine_router (bool): Whether to use cosine router Default: False
        normalize_gate (bool): Whether to normalize the gating score in top-k
        gating. Default: False
        use_bpr (bool): Whether to use batch-prioritized-routing. Default: True
        is_gshard_loss (bool): If True, use Gshard balance loss.
                               If False, use the load loss and importance
                               loss in "arXiv:1701.06538". Default: False
        gate_noise (float): the noise ratio in top-k gating. Default: 1.0
        cosine_router_dim (int): Projection dimension in cosine router.
        cosine_router_init_t (float): Initialization temperature in cosine
        router.
        moe_drop (float): Dropout rate in MoE. Default: 0.0
        aux_loss_weight (float): auxiliary loss weight. Default: 0.1
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, mlp_fc2_bias=True, init_std=0.02, use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], moe_blocks=[[-1], [-1], [-1], [-1]], num_local_experts=1, top_value=1, capacity_factor=1.25, cosine_router=False, normalize_gate=False, use_bpr=True, is_gshard_loss=True, gate_noise=1.0, cosine_router_dim=256, cosine_router_init_t=0.5, moe_drop=0.0, aux_loss_weight=0.01, **kwargs):
        super().__init__()
        self._ddp_params_and_buffers_to_ignore = list()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.init_std = init_std
        self.aux_loss_weight = aux_loss_weight
        self.num_local_experts = num_local_experts
        self.global_experts = num_local_experts * dist.get_world_size() if num_local_experts > 0 else dist.get_world_size() // -num_local_experts
        self.sharded_count = 1.0 / num_local_experts if num_local_experts > 0 else -num_local_experts
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=self.init_std)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, mlp_fc2_bias=mlp_fc2_bias, init_std=init_std, use_checkpoint=use_checkpoint, pretrained_window_size=pretrained_window_sizes[i_layer], moe_block=moe_blocks[i_layer], num_local_experts=num_local_experts, top_value=top_value, capacity_factor=capacity_factor, cosine_router=cosine_router, normalize_gate=normalize_gate, use_bpr=use_bpr, is_gshard_loss=is_gshard_loss, gate_noise=gate_noise, cosine_router_dim=cosine_router_dim, cosine_router_init_t=cosine_router_init_t, moe_drop=moe_drop)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, MoEMlp):
            m._init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'cpb_mlp', 'relative_position_bias_table', 'fc1_bias', 'fc2_bias', 'temperature', 'cosine_projector', 'sim_matrix'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        l_aux = 0.0
        for layer in self.layers:
            x, cur_l_aux = layer(x)
            l_aux = cur_l_aux + l_aux
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x, l_aux

    def forward(self, x):
        x, l_aux = self.forward_features(x)
        x = self.head(x)
        return x, l_aux * self.aux_loss_weight

    def add_param_to_skip_allreduce(self, param_name):
        self._ddp_params_and_buffers_to_ignore.append(param_name)

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // 2 ** self.num_layers
        flops += self.num_features * self.num_classes
        return flops


class SwinTransformerV2(nn.Module):
    """ Swin Transformer A PyTorch impl of : `Swin Transformer: Hierarchical
    Vision Transformer using Shifted Windows`  -
    https://arxiv.org/pdf/2103.14030

    Args: img_size (int | tuple(int)): Input image size. Default 224
    patch_size (int | tuple(int)): Patch size. Default: 4 in_chans (int):
    Number of input image channels. Default: 3 num_classes (int): Number of
    classes for classification head. Default: 1000 embed_dim (int): Patch
    embedding dimension. Default: 96 depths (tuple(int)): Depth of each Swin
    Transformer layer. num_heads (tuple(int)): Number of attention heads in
    different layers. window_size (int): Window size. Default: 7 mlp_ratio (
    float): Ratio of mlp hidden dim to embedding dim. Default: 4 qkv_bias (
    bool): If True, add a learnable bias to query, key, value. Default: True
    drop_rate (float): Dropout rate. Default: 0 attn_drop_rate (float):
    Attention dropout rate. Default: 0 drop_path_rate (float): Stochastic
    depth rate. Default: 0.1 norm_layer (nn.Module): Normalization layer.
    Default: nn.LayerNorm. ape (bool): If True, add absolute position
    embedding to the patch embedding. Default: False patch_norm (bool): If
    True, add normalization after patch embedding. Default: True
    use_checkpoint (bool): Whether to use checkpointing to save memory.
    Default: False pretrained_window_sizes (tuple(int)): Pretrained window
    sizes of each layer.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], multi_scale=False, upsample='deconv', **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint, pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.multi_scale = multi_scale
        if self.multi_scale:
            self.scales = [1, 2, 4, 4]
            self.upsample = nn.ModuleList()
            features = [int(embed_dim * 2 ** i) for i in range(1, self.num_layers)] + [self.num_features]
            self.multi_scale_fuse = nn.Conv2d(sum(features), self.num_features, 1)
            for i in range(self.num_layers):
                self.upsample.append(nn.Upsample(scale_factor=self.scales[i]))
        elif upsample == 'deconv':
            self.upsample = nn.ConvTranspose2d(self.num_features, self.num_features, 2, stride=2)
        elif upsample == 'new_deconv':
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(self.num_features, self.num_features, 3, stride=1, padding=1), nn.BatchNorm2d(self.num_features), nn.ReLU(inplace=True))
        elif upsample == 'new_deconv2':
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(self.num_features, self.num_features, 3, stride=1, padding=1), nn.BatchNorm2d(self.num_features), nn.ReLU(inplace=True))
        elif upsample == 'bilinear':
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'cpb_mlp', 'logit_scale', 'relative_position_bias_table'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        if self.multi_scale:
            features = []
            for i, layer in enumerate(self.layers):
                x = layer(x)
                x_2d = x.view(B, H // (8 * self.scales[i]), W // (8 * self.scales[i]), -1).permute(0, 3, 1, 2)
                features.append(self.upsample[i](x_2d))
            x = torch.cat(features, dim=1)
            x = self.multi_scale_fuse(x)
            x = x.view(B, self.num_features, -1).permute(0, 2, 1)
            x = self.norm(x)
            x = x.view(B, H // 8, W // 8, self.num_features).permute(0, 3, 1, 2)
        else:
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            x = x.view(B, H // 32, W // 32, self.num_features).permute(0, 3, 1, 2)
            x = self.upsample(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // 2 ** self.num_layers
        flops += self.num_features * self.num_classes
        return flops


class TokenDecodeMLP(nn.Module):
    """The MLP used to predict coordinates from the support keypoints
    tokens."""

    def __init__(self, in_channels, hidden_channels, out_channels=2, num_layers=3):
        super(TokenDecodeMLP, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_channels, hidden_channels))
                layers.append(nn.GELU())
            else:
                layers.append(nn.Linear(hidden_channels, hidden_channels))
                layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def build_positional_encoding(cfg, default_args=None):
    """Build backbone."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


def inverse_sigmoid(x, eps=0.001):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _calc_distances(preds: 'np.ndarray', gts: 'np.ndarray', mask: 'np.ndarray', norm_factor: 'np.ndarray') ->np.ndarray:
    """Calculate the normalized distances between preds and target.

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        gts (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        norm_factor (np.ndarray[N, D]): Normalization factor.
            Typical value is heatmap_size.

    Returns:
        np.ndarray[K, N]: The normalized distances.             If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    _mask = mask.copy()
    _mask[np.where((norm_factor == 0).sum(1))[0], :] = False
    distances = np.full((N, K), -1, dtype=np.float32)
    norm_factor[np.where(norm_factor <= 0)] = 1000000.0
    distances[_mask] = np.linalg.norm(((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1)
    return distances.T


def _distance_acc(distances: 'np.ndarray', thr: 'float'=0.5) ->float:
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        - instance number: N

    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold.             If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def keypoint_pck_accuracy(pred: 'np.ndarray', gt: 'np.ndarray', mask: 'np.ndarray', thr: 'np.ndarray', norm_factor: 'np.ndarray') ->tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        norm_factor (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, norm_factor)
    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0.0
    return acc, avg_acc, cnt


def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    scale = scale * 200.0
    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]
    target_coords = coords.copy()
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    return target_coords


class PoseHead(nn.Module):
    """In two stage regression A3, the proposal generator are moved into
    transformer.

    All valid proposals will be added with an positional embedding to better
    regress the location
    """

    def __init__(self, in_channels, transformer=None, positional_encoding=dict(type='SinePositionalEncoding', num_feats=128, normalize=True), encoder_positional_encoding=dict(type='SinePositionalEncoding', num_feats=512, normalize=True), share_kpt_branch=False, num_decoder_layer=3, with_heatmap_loss=False, with_bb_loss=False, bb_temperature=0.2, heatmap_loss_weight=2.0, support_order_dropout=-1, extra=None, train_cfg=None, test_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.encoder_positional_encoding = build_positional_encoding(encoder_positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.d_model
        self.with_heatmap_loss = with_heatmap_loss
        self.with_bb_loss = with_bb_loss
        self.bb_temperature = bb_temperature
        self.heatmap_loss_weight = heatmap_loss_weight
        self.support_order_dropout = support_order_dropout
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should'(f' be exactly 2 times of num_feats. Found {self.embed_dims} and {num_feats}.')
        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')
        """Initialize layers of the transformer head."""
        self.input_proj = Conv2d(self.in_channels, self.embed_dims, kernel_size=1)
        self.query_proj = Linear(self.in_channels, self.embed_dims)
        kpt_branch = TokenDecodeMLP(in_channels=self.embed_dims, hidden_channels=self.embed_dims)
        if share_kpt_branch:
            self.kpt_branch = nn.ModuleList([kpt_branch for i in range(num_decoder_layer)])
        else:
            self.kpt_branch = nn.ModuleList([deepcopy(kpt_branch) for i in range(num_decoder_layer)])
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatMap')

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        """Initialize weights of the transformer head."""
        self.transformer.init_weights()
        for mlp in self.kpt_branch:
            nn.init.constant_(mlp.mlp[-1].weight.data, 0)
            nn.init.constant_(mlp.mlp[-1].bias.data, 0)
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
        nn.init.constant_(self.input_proj.bias, 0)
        nn.init.xavier_uniform_(self.query_proj.weight, gain=1)
        nn.init.constant_(self.query_proj.bias, 0)

    def forward(self, x, feature_s, target_s, mask_s, skeleton):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        x = self.input_proj(x)
        bs, dim, h, w = x.shape
        support_order_embedding = x.new_zeros((bs, self.embed_dims, 1, target_s[0].shape[1]))
        masks = x.new_zeros((x.shape[0], x.shape[2], x.shape[3]))
        pos_embed = self.positional_encoding(masks)
        query_embed_list = []
        for i, (feature, target) in enumerate(zip(feature_s, target_s)):
            resized_feature = resize(input=feature, size=target.shape[-2:], mode='bilinear', align_corners=False)
            target = target / (target.sum(dim=-1).sum(dim=-1)[:, :, None, None] + 1e-08)
            support_keypoints = target.flatten(2) @ resized_feature.flatten(2).permute(0, 2, 1)
            query_embed_list.append(support_keypoints)
        support_keypoints = torch.mean(torch.stack(query_embed_list, dim=0), 0)
        support_keypoints = support_keypoints * mask_s
        support_keypoints = self.query_proj(support_keypoints)
        masks_query = (~mask_s).squeeze(-1)
        outs_dec, initial_proposals, out_points, similarity_map = self.transformer(x, masks, support_keypoints, pos_embed, support_order_embedding, masks_query, self.positional_encoding, self.kpt_branch, skeleton)
        output_kpts = []
        for idx in range(outs_dec.shape[0]):
            layer_delta_unsig = self.kpt_branch[idx](outs_dec[idx])
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(out_points[idx])
            output_kpts.append(layer_outputs_unsig.sigmoid())
        return torch.stack(output_kpts, dim=0), initial_proposals, similarity_map

    def get_loss(self, output, initial_proposals, similarity_map, target, target_heatmap, target_weight, target_sizes):
        losses = dict()
        num_dec_layer, bs, nq = output.shape[:3]
        target_sizes = target_sizes
        target = target / target_sizes
        target = target[None, :, :, :].repeat(num_dec_layer, 1, 1, 1)
        normalizer = target_weight.squeeze(dim=-1).sum(dim=-1)
        normalizer[normalizer == 0] = 1
        if self.with_heatmap_loss:
            losses['heatmap_loss'] = self.heatmap_loss(similarity_map, target_heatmap, target_weight, normalizer) * self.heatmap_loss_weight
        proposal_l1_loss = F.l1_loss(initial_proposals, target[0], reduction='none')
        proposal_l1_loss = proposal_l1_loss.sum(dim=-1, keepdim=False) * target_weight.squeeze(dim=-1)
        proposal_l1_loss = proposal_l1_loss.sum(dim=-1, keepdim=False) / normalizer
        losses['proposal_loss'] = proposal_l1_loss.sum() / bs
        for idx in range(num_dec_layer):
            layer_output, layer_target = output[idx], target[idx]
            l1_loss = F.l1_loss(layer_output, layer_target, reduction='none')
            l1_loss = l1_loss.sum(dim=-1, keepdim=False) * target_weight.squeeze(dim=-1)
            l1_loss = l1_loss.sum(dim=-1, keepdim=False) / normalizer
            losses['l1_loss' + '_layer' + str(idx)] = l1_loss.sum() / bs
        return losses

    def get_max_coords(self, heatmap, heatmap_size=64):
        B, C, H, W = heatmap.shape
        heatmap = heatmap.view(B, C, -1)
        max_cor = heatmap.argmax(dim=2)
        row, col = torch.floor(max_cor / heatmap_size), max_cor % heatmap_size
        support_joints = torch.cat((row.unsqueeze(-1), col.unsqueeze(-1)), dim=-1)
        return support_joints

    def heatmap_loss(self, similarity_map, target_heatmap, target_weight, normalizer):
        h, w = similarity_map.shape[-2:]
        similarity_map = similarity_map.sigmoid()
        target_heatmap = F.interpolate(target_heatmap, size=(h, w), mode='bilinear')
        target_heatmap = target_heatmap / (target_heatmap.max(dim=-1)[0].max(dim=-1)[0] + 1e-10)[:, :, None, None]
        l2_loss = F.mse_loss(similarity_map, target_heatmap, reduction='none')
        l2_loss = l2_loss * target_weight[:, :, :, None]
        l2_loss = l2_loss.flatten(2, 3).sum(-1) / (h * w)
        l2_loss = l2_loss.sum(-1) / normalizer
        return l2_loss.mean()

    def get_accuracy(self, output, target, target_weight, target_sizes, height=256):
        """Calculate accuracy for top-down keypoint loss.

        Args:
            output (torch.Tensor[NxKx2]): estimated keypoints in ABSOLUTE
            coordinates.
            target (torch.Tensor[NxKx2]): gt keypoints in ABSOLUTE coordinates.
            target_weight (torch.Tensor[NxKx1]): Weights across different
            joint types.
            target_sizes (torch.Tensor[Nx2): shapes of the image.
        """
        accuracy = dict()
        output = output * float(height)
        output, target, target_weight, target_sizes = output.detach().cpu().numpy(), target.detach().cpu().numpy(), target_weight.squeeze(-1).long().detach().cpu().numpy(), target_sizes.squeeze(1).detach().cpu().numpy()
        _, avg_acc, _ = keypoint_pck_accuracy(output, target, target_weight.astype(np.bool8), thr=0.2, normalize=target_sizes)
        accuracy['acc_pose'] = float(avg_acc)
        return accuracy

    def decode(self, img_metas, output, img_size, **kwargs):
        """Decode the predicted keypoints from prediction.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        batch_size = len(img_metas)
        W, H = img_size
        output = output * np.array([W, H])[None, None, :]
        if 'bbox_id' or 'query_bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None
        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['query_center']
            s[i, :] = img_metas[i]['query_scale']
            image_paths.append(img_metas[i]['query_image_file'])
            if 'query_bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['query_bbox_score']).reshape(-1)
            if 'bbox_id' in img_metas[i]:
                bbox_ids.append(img_metas[i]['bbox_id'])
            elif 'query_bbox_id' in img_metas[i]:
                bbox_ids.append(img_metas[i]['query_bbox_id'])
        preds = np.zeros(output.shape)
        for idx in range(output.shape[0]):
            preds[i] = transform_preds(output[i], c[i], s[i], [W, H], use_udp=self.test_cfg.get('use_udp', False))
        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = 1.0
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score
        result = {}
        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids
        return result


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ProposalGenerator(nn.Module):

    def __init__(self, hidden_dim, proj_dim, dynamic_proj_dim):
        super().__init__()
        self.support_proj = nn.Linear(hidden_dim, proj_dim)
        self.query_proj = nn.Linear(hidden_dim, proj_dim)
        self.dynamic_proj = nn.Sequential(nn.Linear(hidden_dim, dynamic_proj_dim), nn.ReLU(), nn.Linear(dynamic_proj_dim, hidden_dim))
        self.dynamic_act = nn.Tanh()

    def forward(self, query_feat, support_feat, spatial_shape):
        """
        Args:
            support_feat: [query, bs, c]
            query_feat: [hw, bs, c]
            spatial_shape: h, w
        """
        device = query_feat.device
        _, bs, c = query_feat.shape
        h, w = spatial_shape
        side_normalizer = torch.tensor([w, h])[None, None, :]
        query_feat = query_feat.transpose(0, 1)
        support_feat = support_feat.transpose(0, 1)
        nq = support_feat.shape[1]
        fs_proj = self.support_proj(support_feat)
        fq_proj = self.query_proj(query_feat)
        pattern_attention = self.dynamic_act(self.dynamic_proj(fs_proj))
        fs_feat = (pattern_attention + 1) * fs_proj
        similarity = torch.bmm(fq_proj, fs_feat.transpose(1, 2))
        similarity = similarity.transpose(1, 2).reshape(bs, nq, h, w)
        grid_y, grid_x = torch.meshgrid(torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device), torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device))
        coord_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).unsqueeze(0).repeat(bs, nq, 1, 1, 1)
        coord_grid = coord_grid.permute(0, 1, 3, 4, 2)
        similarity_softmax = similarity.flatten(2, 3).softmax(dim=-1)
        similarity_coord_grid = similarity_softmax[:, :, :, None] * coord_grid.flatten(2, 3)
        proposal_for_loss = similarity_coord_grid.sum(dim=2, keepdim=False)
        proposal_for_loss = proposal_for_loss / side_normalizer
        max_pos = torch.argmax(similarity.reshape(bs, nq, -1), dim=-1, keepdim=True)
        max_mask = F.one_hot(max_pos, num_classes=w * h)
        max_mask = max_mask.reshape(bs, nq, w, h).type(torch.float)
        local_max_mask = F.max_pool2d(input=max_mask, kernel_size=3, stride=1, padding=1).reshape(bs, nq, w * h, 1)
        """
        proposal = (similarity_coord_grid * local_max_mask).sum(
            dim=2, keepdim=False) / torch.count_nonzero(
                local_max_mask, dim=2)
        """
        local_similarity_softmax = similarity_softmax[:, :, :, None] * local_max_mask
        local_similarity_softmax = local_similarity_softmax / (local_similarity_softmax.sum(dim=-2, keepdim=True) + 1e-10)
        proposals = local_similarity_softmax * coord_grid.flatten(2, 3)
        proposals = proposals.sum(dim=2) / side_normalizer
        return proposal_for_loss, similarity, proposals


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class GraphTransformerDecoder(nn.Module):

    def __init__(self, d_model, decoder_layer, num_layers, norm=None, return_intermediate=False, look_twice=False, detach_support_feat=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.ref_point_head = MLP(d_model, d_model, d_model, num_layers=2)
        self.look_twice = look_twice
        self.detach_support_feat = detach_support_feat

    def forward(self, support_feat, query_feat, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None, position_embedding=None, initial_proposals=None, kpt_branch=None, skeleton=None, return_attn_map=False):
        """
        position_embedding: Class used to compute positional embedding
        initial_proposals: [bs, nq, 2], normalized coordinates of initial
        proposals kpt_branch: MLP used to predict the offsets for each query.
        """
        refined_support_feat = support_feat
        intermediate = []
        attn_maps = []
        bi = initial_proposals.detach()
        bi_tag = initial_proposals.detach()
        query_points = [initial_proposals.detach()]
        tgt_key_padding_mask_remove_all_true = tgt_key_padding_mask.clone()
        tgt_key_padding_mask_remove_all_true[tgt_key_padding_mask.logical_not().sum(dim=-1) == 0, 0] = False
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 0:
                query_pos_embed = query_pos.transpose(0, 1)
            else:
                query_pos_embed = position_embedding.forward_coordinates(bi)
                query_pos_embed = query_pos_embed.transpose(0, 1)
            query_pos_embed = self.ref_point_head(query_pos_embed)
            if self.detach_support_feat:
                refined_support_feat = refined_support_feat.detach()
            refined_support_feat, attn_map = layer(refined_support_feat, query_feat, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask_remove_all_true, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos_embed, skeleton=skeleton)
            if self.return_intermediate:
                intermediate.append(self.norm(refined_support_feat))
            if return_attn_map:
                attn_maps.append(attn_map)
            delta_bi = kpt_branch[layer_idx](refined_support_feat.transpose(0, 1))
            if self.look_twice:
                bi_pred = self.update(bi_tag, delta_bi)
                bi_tag = self.update(bi, delta_bi)
            else:
                bi_tag = self.update(bi, delta_bi)
                bi_pred = bi_tag
            bi = bi_tag.detach()
            query_points.append(bi_pred)
        if self.norm is not None:
            refined_support_feat = self.norm(refined_support_feat)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(refined_support_feat)
        if self.return_intermediate:
            return torch.stack(intermediate), query_points, attn_maps
        return refined_support_feat.unsqueeze(0), query_points, attn_maps

    def update(self, query_coordinates, delta_unsig):
        query_coordinates_unsigmoid = inverse_sigmoid(query_coordinates)
        new_query_coordinates = query_coordinates_unsigmoid + delta_unsig
        new_query_coordinates = new_query_coordinates.sigmoid()
        return new_query_coordinates


class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=2, use_bias=True, activation=nn.ReLU(inplace=True), batch_first=True):
        super(GCNLayer, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features * kernel_size, kernel_size=1, padding=0, stride=1, dilation=1, bias=use_bias)
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_first = batch_first

    def forward(self, x, adj):
        assert adj.size(1) == self.kernel_size
        if not self.batch_first:
            x = x.permute(1, 2, 0)
        else:
            x = x.transpose(1, 2)
        x = self.conv(x)
        b, kc, v = x.size()
        x = x.view(b, self.kernel_size, kc // self.kernel_size, v)
        x = torch.einsum('bkcv,bkvw->bcw', (x, adj))
        if self.activation is not None:
            x = self.activation(x)
        if not self.batch_first:
            x = x.permute(2, 0, 1)
        else:
            x = x.transpose(1, 2)
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


def adj_from_skeleton(num_pts, skeleton, mask, device='cuda'):
    adj_mx = torch.empty(0, device=device)
    batch_size = len(skeleton)
    for b in range(batch_size):
        edges = torch.tensor(skeleton[b])
        adj = torch.zeros(num_pts, num_pts, device=device)
        adj[edges[:, 0], edges[:, 1]] = 1
        adj_mx = torch.cat((adj_mx, adj.unsqueeze(0)), dim=0)
    trans_adj_mx = torch.transpose(adj_mx, 1, 2)
    cond = (trans_adj_mx > adj_mx).float()
    adj = adj_mx + trans_adj_mx * cond - adj_mx * cond
    adj = adj * ~mask[..., None] * ~mask[:, None]
    adj = torch.nan_to_num(adj / adj.sum(dim=-1, keepdim=True))
    adj = torch.stack((torch.diag_embed(~mask), adj), dim=1)
    return adj


class GraphTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, graph_decoder=None):
        super().__init__()
        self.graph_decoder = graph_decoder
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
        self.choker = nn.Linear(in_features=2 * d_model, out_features=d_model)
        if self.graph_decoder is None:
            self.ffn1 = nn.Linear(d_model, dim_feedforward)
            self.ffn2 = nn.Linear(dim_feedforward, d_model)
        elif self.graph_decoder == 'pre':
            self.ffn1 = GCNLayer(d_model, dim_feedforward, batch_first=False)
            self.ffn2 = nn.Linear(dim_feedforward, d_model)
        elif self.graph_decoder == 'post':
            self.ffn1 = nn.Linear(d_model, dim_feedforward)
            self.ffn2 = GCNLayer(dim_feedforward, d_model, batch_first=False)
        else:
            self.ffn1 = GCNLayer(d_model, dim_feedforward, batch_first=False)
            self.ffn2 = GCNLayer(dim_feedforward, d_model, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward(self, refined_support_feat, refined_query_feat, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None, skeleton: 'Optional[list]'=None):
        q = k = self.with_pos_embed(refined_support_feat, query_pos + pos[refined_query_feat.shape[0]:])
        tgt2 = self.self_attn(q, k, value=refined_support_feat, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        refined_support_feat = refined_support_feat + self.dropout1(tgt2)
        refined_support_feat = self.norm1(refined_support_feat)
        cross_attn_q = torch.cat((refined_support_feat, query_pos + pos[refined_query_feat.shape[0]:]), dim=-1)
        cross_attn_k = torch.cat((refined_query_feat, pos[:refined_query_feat.shape[0]]), dim=-1)
        tgt2, attn_map = self.multihead_attn(query=cross_attn_q, key=cross_attn_k, value=refined_query_feat, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        refined_support_feat = refined_support_feat + self.dropout2(self.choker(tgt2))
        refined_support_feat = self.norm2(refined_support_feat)
        if self.graph_decoder is not None:
            num_pts, b, c = refined_support_feat.shape
            adj = adj_from_skeleton(num_pts=num_pts, skeleton=skeleton, mask=tgt_key_padding_mask, device=refined_support_feat.device)
            if self.graph_decoder == 'pre':
                tgt2 = self.ffn2(self.dropout(self.activation(self.ffn1(refined_support_feat, adj))))
            elif self.graph_decoder == 'post':
                tgt2 = self.ffn2(self.dropout(self.activation(self.ffn1(refined_support_feat))), adj)
            else:
                tgt2 = self.ffn2(self.dropout(self.activation(self.ffn1(refined_support_feat, adj))), adj)
        else:
            tgt2 = self.ffn2(self.dropout(self.activation(self.ffn1(refined_support_feat))))
        refined_support_feat = refined_support_feat + self.dropout3(tgt2)
        refined_support_feat = self.norm3(refined_support_feat)
        return refined_support_feat, attn_map


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, query, mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, query_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        n, bs, c = src.shape
        src_cat = torch.cat((src, query), dim=0)
        mask_cat = torch.cat((src_key_padding_mask, query_key_padding_mask), dim=1)
        output = src_cat
        for layer in self.layers:
            output = layer(output, query_length=n, src_mask=mask, src_key_padding_mask=mask_cat, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        refined_query = output[n:, :, :]
        output = output[:n, :, :]
        return output, refined_query


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward(self, src, query_length, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        src = self.with_pos_embed(src, pos)
        q = k = src
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class EncoderDecoder(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3, graph_decoder=None, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, similarity_proj_dim=256, dynamic_proj_dim=128, return_intermediate_dec=True, look_twice=False, detach_support_feat=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = GraphTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before, graph_decoder)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = GraphTransformerDecoder(d_model, decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec, look_twice=look_twice, detach_support_feat=detach_support_feat)
        self.proposal_generator = ProposalGenerator(hidden_dim=d_model, proj_dim=similarity_proj_dim, dynamic_proj_dim=dynamic_proj_dim)

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')

    def forward(self, src, mask, support_embed, pos_embed, support_order_embed, query_padding_mask, position_embedding, kpt_branch, skeleton, return_attn_map=False):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        support_order_embed = support_order_embed.flatten(2).permute(2, 0, 1)
        pos_embed = torch.cat((pos_embed, support_order_embed))
        query_embed = support_embed.transpose(0, 1)
        mask = mask.flatten(1)
        query_embed, refined_support_embed = self.encoder(src, query_embed, src_key_padding_mask=mask, query_key_padding_mask=query_padding_mask, pos=pos_embed)
        initial_proposals_for_loss, similarity_map, initial_proposals = self.proposal_generator(query_embed, refined_support_embed, spatial_shape=[h, w])
        initial_position_embedding = position_embedding.forward_coordinates(initial_proposals)
        outs_dec, out_points, attn_maps = self.decoder(refined_support_embed, query_embed, memory_key_padding_mask=mask, pos=pos_embed, query_pos=initial_position_embedding, tgt_key_padding_mask=query_padding_mask, position_embedding=position_embedding, initial_proposals=initial_proposals, kpt_branch=kpt_branch, skeleton=skeleton, return_attn_map=return_attn_map)
        return outs_dec.transpose(1, 2), initial_proposals_for_loss, out_points, similarity_map


class RTMDet(nn.Module):
    """Load RTMDet model and add postprocess.

    Args:
        model (nn.Module): The RTMDet model.
    """

    def __init__(self, model: 'nn.Module') ->None:
        super().__init__()
        self.model = model
        self.stage = [80, 40, 20]
        self.input_shape = 640

    def forward(self, inputs):
        """model forward function."""
        boxes = []
        neck_outputs = self.model(inputs)
        for i, (cls, box) in enumerate(zip(*neck_outputs)):
            cls = cls.permute(0, 2, 3, 1)
            box = box.permute(0, 2, 3, 1)
            box = self.decode(box, cls, i)
            boxes.append(box)
        result_box = torch.cat(boxes, dim=1)
        return result_box

    def decode(self, box: 'torch.Tensor', cls: 'torch.Tensor', stage: 'int'):
        """RTMDet postprocess function.

        Args:
            box (torch.Tensor): output boxes.
            cls (torch.Tensor): output cls.
            stage (int): RTMDet output stage.

        Returns:
            torch.Tensor: The decode boxes.
                Format is [x1, y1, x2, y2, class, confidence]
        """
        cls = F.sigmoid(cls)
        conf = torch.max(cls, dim=3, keepdim=True)[0]
        cls = torch.argmax(cls, dim=3, keepdim=True)
        box = torch.cat([box, cls, conf], dim=-1)
        step = self.input_shape // self.stage[stage]
        block_step = torch.linspace(0, self.stage[stage] - 1, steps=self.stage[stage], device='cuda') * step
        block_x = torch.broadcast_to(block_step, [self.stage[stage], self.stage[stage]])
        block_y = torch.transpose(block_x, 1, 0)
        block_x = torch.unsqueeze(block_x, 0)
        block_y = torch.unsqueeze(block_y, 0)
        block = torch.stack([block_x, block_y], -1)
        box[..., :2] = block - box[..., :2]
        box[..., 2:4] = block + box[..., 2:4]
        box = box.reshape(1, -1, 6)
        return box


class OksLoss(nn.Module):
    """A PyTorch implementation of the Object Keypoint Similarity (OKS) loss as
    described in the paper "YOLO-Pose: Enhancing YOLO for Multi Person Pose
    Estimation Using Object Keypoint Similarity Loss" by Debapriya et al.
    (2022).

    The OKS loss is used for keypoint-based object recognition and consists
    of a measure of the similarity between predicted and ground truth
    keypoint locations, adjusted by the size of the object in the image.

    The loss function takes as input the predicted keypoint locations, the
    ground truth keypoint locations, a mask indicating which keypoints are
    valid, and bounding boxes for the objects.

    Args:
        metainfo (Optional[str]): Path to a JSON file containing information
            about the dataset's annotations.
        loss_weight (float): Weight for the loss.
    """

    def __init__(self, metainfo: 'Optional[str]'=None, loss_weight: 'float'=1.0):
        super().__init__()
        if metainfo is not None:
            metainfo = parse_pose_metainfo(dict(from_file=metainfo))
            sigmas = metainfo.get('sigmas', None)
            if sigmas is not None:
                self.register_buffer('sigmas', torch.as_tensor(sigmas))
        self.loss_weight = loss_weight

    def forward(self, output: 'Tensor', target: 'Tensor', target_weights: 'Tensor', bboxes: 'Optional[Tensor]'=None) ->Tensor:
        oks = self.compute_oks(output, target, target_weights, bboxes)
        loss = 1 - oks
        return loss * self.loss_weight

    def compute_oks(self, output: 'Tensor', target: 'Tensor', target_weights: 'Tensor', bboxes: 'Optional[Tensor]'=None) ->Tensor:
        """Calculates the OKS loss.

        Args:
            output (Tensor): Predicted keypoints in shape N x k x 2, where N
                is batch size, k is the number of keypoints, and 2 are the
                xy coordinates.
            target (Tensor): Ground truth keypoints in the same shape as
                output.
            target_weights (Tensor): Mask of valid keypoints in shape N x k,
                with 1 for valid and 0 for invalid.
            bboxes (Optional[Tensor]): Bounding boxes in shape N x 4,
                where 4 are the xyxy coordinates.

        Returns:
            Tensor: The calculated OKS loss.
        """
        dist = torch.norm(output - target, dim=-1)
        if hasattr(self, 'sigmas'):
            sigmas = self.sigmas.reshape(*((1,) * (dist.ndim - 1)), -1)
            dist = dist / sigmas
        if bboxes is not None:
            area = torch.norm(bboxes[..., 2:] - bboxes[..., :2], dim=-1)
            dist = dist / area.clip(min=1e-08).unsqueeze(-1)
        return (torch.exp(-dist.pow(2) / 2) * target_weights).sum(dim=-1) / target_weights.sum(dim=-1).clip(min=1e-08)


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class AELoss(nn.Module):
    """Associative Embedding loss in MMPose v0.x."""

    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    @staticmethod
    def _make_input(t, requires_grad=False, device=torch.device('cpu')):
        """Make zero inputs for AE loss.

        Args:
            t (torch.Tensor): input
            requires_grad (bool): Option to use requires_grad.
            device: torch device

        Returns:
            torch.Tensor: zero input.
        """
        inp = torch.autograd.Variable(t, requires_grad=requires_grad)
        inp = inp.sum()
        inp = inp
        return inp

    def singleTagLoss(self, pred_tag, joints):
        """Associative embedding loss for one image.

        Note:
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            pred_tag (torch.Tensor[KxHxW,1]): tag of output for one image.
            joints (torch.Tensor[M,K,2]): joints information for one image.
        """
        tags = []
        pull = 0
        pred_tag = pred_tag.view(17, -1, 1)
        for joints_per_person in joints:
            tmp = []
            for k, joint in enumerate(joints_per_person):
                if joint[1] > 0:
                    tmp.append(pred_tag[k, joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp)) ** 2)
        num_tags = len(tags)
        if num_tags == 0:
            return self._make_input(torch.zeros(1).float(), device=pred_tag.device), self._make_input(torch.zeros(1).float(), device=pred_tag.device)
        elif num_tags == 1:
            return self._make_input(torch.zeros(1).float(), device=pred_tag.device), pull
        tags = torch.stack(tags)
        size = num_tags, num_tags
        A = tags.expand(*size)
        B = A.permute(1, 0)
        diff = A - B
        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push)
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unknown ae loss type')
        push_loss = push / ((num_tags - 1) * num_tags) * 0.5
        pull_loss = pull / num_tags
        return push_loss, pull_loss

    def forward(self, tags, keypoint_indices):
        assert tags.shape[0] == len(keypoint_indices)
        pull_loss = 0.0
        push_loss = 0.0
        for i in range(tags.shape[0]):
            _push, _pull = self.singleTagLoss(tags[i].view(-1, 1), keypoint_indices[i])
            pull_loss += _pull
            push_loss += _push
        return pull_loss, push_loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AdaptiveWingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ChannelWiseScale,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CombinedTargetMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FeaLoss,
     lambda: ([], {'name': 4, 'use_this': 4, 'student_channels': 4, 'teacher_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FeatureMapProcessor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FocalHeatmapLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FrozenBatchNorm2d,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GlobalAveragePooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (KLDiscretLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (MLECCLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MPJPELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MPJPEVelocityJointLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (OKSLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (OksLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Scale,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScaleNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SoftWeightSmoothL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SoftWingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TokenDecodeMLP,
     lambda: ([], {'in_channels': 4, 'hidden_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TruncSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (WingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

