
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


from typing import Dict


from typing import Iterable


from typing import List


from typing import Optional


from typing import Sequence


from typing import Tuple


from typing import Union


import numpy as np


from torch import Tensor


from typing import Callable


import math


from typing import Iterator


from typing import Sized


import torch


from torch.utils.data import Sampler


from scipy.sparse import csr_matrix


from scipy.sparse.csgraph import maximum_bipartite_matching


import torch.nn as nn


import torch.utils.checkpoint as cp


import torch.nn.functional as F


import warnings


from torch import nn


from torch.nn import functional as F


from numbers import Number


from abc import ABCMeta


from abc import abstractmethod


from numpy import ndarray


from torch.nn import init


from numpy.linalg import norm


from numpy.fft import fft


from functools import partial


import functools


from numpy.fft import ifft


import copy


from queue import PriorityQueue


import itertools


from typing import Any


from functools import reduce


from matplotlib.font_manager import FontProperties


from matplotlib.collections import PatchCollection


from matplotlib.patches import FancyArrow


import random


import re


from copy import deepcopy


import time


from itertools import chain


from itertools import permutations


import logging


import matplotlib.pyplot as plt


class UpConvBlock(nn.Module):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self, conv_block, in_channels, skip_channels, out_channels, num_convs=2, stride=1, dilation=1, with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), upsample_cfg=dict(type='InterpConv'), dcn=None, plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        self.conv_block = conv_block(in_channels=2 * skip_channels, out_channels=out_channels, num_convs=num_convs, stride=stride, dilation=dilation, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, dcn=None, plugins=None)
        if upsample_cfg is not None:
            upsample_cfg.update(dict(in_channels=in_channels, out_channels=skip_channels, with_cp=with_cp, norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.upsample = MODELS.build(upsample_cfg)
        else:
            self.upsample = ConvModule(in_channels, skip_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, skip, x):
        """Forward function."""
        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)
        return out


class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self, in_channels, out_channels, num_convs=2, stride=1, dilation=1, with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), dcn=None, plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(ConvModule(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, kernel_size=3, stride=stride if i == 0 else 1, dilation=1 if i == 0 else dilation, padding=1 if i == 0 else dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out


class DeconvModule(nn.Module):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
    """

    def __init__(self, in_channels, out_channels, with_cp=False, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), *, kernel_size=4, scale_factor=2):
        super().__init__()
        assert kernel_size - scale_factor >= 0 and (kernel_size - scale_factor) % 2 == 0, f'kernel_size should be greater than or equal to scale_factor and (kernel_size - scale_factor) should be even numbers, while the kernel size is {kernel_size} and scale_factor is {scale_factor}.'
        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        self.with_cp = with_cp
        deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        _, norm = build_norm_layer(norm_cfg, out_channels)
        activate = MODELS.build(act_cfg)
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        """Forward function."""
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.deconv_upsamping, x)
        else:
            out = self.deconv_upsamping(x)
        return out


class InterpConv(nn.Module):
    """Interpolation upsample module in decoder for UNet.

    This module uses interpolation to upsample feature map in the decoder
    of UNet. It consists of one interpolation upsample layer and one
    convolutional layer. It can be one interpolation upsample layer followed
    by one convolutional layer (conv_first=False) or one convolutional layer
    followed by one interpolation upsample layer (conv_first=True).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        conv_first (bool): Whether convolutional layer or interpolation
            upsample layer first. Default: False. It means interpolation
            upsample layer followed by one convolutional layer.
        kernel_size (int): Kernel size of the convolutional layer. Default: 1.
        stride (int): Stride of the convolutional layer. Default: 1.
        padding (int): Padding of the convolutional layer. Default: 1.
        upsample_cfg (dict): Interpolation config of the upsample layer.
            Default: dict(
                scale_factor=2, mode='bilinear', align_corners=False).
    """

    def __init__(self, in_channels, out_channels, with_cp=False, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), *, conv_cfg=None, conv_first=False, kernel_size=1, stride=1, padding=0, upsample_cfg=dict(scale_factor=2, mode='bilinear', align_corners=False)):
        super().__init__()
        self.with_cp = with_cp
        conv = ConvModule(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        upsample = nn.Upsample(**upsample_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)

    def forward(self, x):
        """Forward function."""
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.interp_upsample, x)
        else:
            out = self.interp_upsample(x)
        return out


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention Module. This code is adopted from
    https://github.com/jadore801120/attention-is-all-you-need-pytorch.

    Args:
        temperature (float): The scale factor for softmax input.
        attn_dropout (float): Dropout layer on attn_output_weights.
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module.

    Args:
        n_head (int): The number of heads in the
            multiheadattention models (default=8).
        d_model (int): The number of expected features
            in the decoder inputs (default=512).
        d_k (int): Total number of features in key.
        d_v (int): Total number of features in value.
        dropout (float): Dropout layer on attn_output_weights.
        qkv_bias (bool): Add bias in projection layer. Default: False.
    """

    def __init__(self, n_head=8, d_model=512, d_k=64, d_v=64, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v
        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias=qkv_bias)
        self.attention = ScaledDotProductAttention(d_k ** 0.5, dropout)
        self.fc = nn.Linear(self.dim_v, d_model, bias=qkv_bias)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, _ = q.size()
        _, len_k, _ = k.size()
        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.linear_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.linear_v(v).view(batch_size, len_k, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
        attn_out, _ = self.attention(q, k, v, mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, len_q, self.dim_v)
        attn_out = self.fc(attn_out)
        attn_out = self.proj_drop(attn_out)
        return attn_out


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward module.

    Args:
        d_in (int): The dimension of the input for feedforward
            network model.
        d_hid (int): The dimension of the feedforward
            network model.
        dropout (float): Dropout layer on feedforward output.
        act_cfg (dict): Activation cfg for feedforward module.
    """

    def __init__(self, d_in, d_hid, dropout=0.1, act_cfg=dict(type='Relu')):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.act = MODELS.build(act_cfg)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class TFDecoderLayer(nn.Module):
    """Transformer Decoder Layer.

    Args:
        d_model (int): The number of expected features
            in the decoder inputs (default=512).
        d_inner (int): The dimension of the feedforward
            network model (default=256).
        n_head (int): The number of heads in the
            multiheadattention models (default=8).
        d_k (int): Total number of features in key.
        d_v (int): Total number of features in value.
        dropout (float): Dropout layer on attn_output_weights.
        qkv_bias (bool): Add bias in projection layer. Default: False.
        act_cfg (dict): Activation cfg for feedforward module.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'enc_dec_attn',
            'norm', 'ffn', 'norm') or ('norm', 'self_attn', 'norm',
            'enc_dec_attn', 'norm', 'ffn').
            Default：None.
    """

    def __init__(self, d_model=512, d_inner=256, n_head=8, d_k=64, d_v=64, dropout=0.1, qkv_bias=False, act_cfg=dict(type='mmengine.GELU'), operation_order=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)
        self.mlp = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, act_cfg=act_cfg)
        self.operation_order = operation_order
        if self.operation_order is None:
            self.operation_order = 'norm', 'self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn'
        assert self.operation_order in [('norm', 'self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn'), ('self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn', 'norm')]

    def forward(self, dec_input, enc_output, self_attn_mask=None, dec_enc_attn_mask=None):
        if self.operation_order == ('self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn', 'norm'):
            dec_attn_out = self.self_attn(dec_input, dec_input, dec_input, self_attn_mask)
            dec_attn_out += dec_input
            dec_attn_out = self.norm1(dec_attn_out)
            enc_dec_attn_out = self.enc_attn(dec_attn_out, enc_output, enc_output, dec_enc_attn_mask)
            enc_dec_attn_out += dec_attn_out
            enc_dec_attn_out = self.norm2(enc_dec_attn_out)
            mlp_out = self.mlp(enc_dec_attn_out)
            mlp_out += enc_dec_attn_out
            mlp_out = self.norm3(mlp_out)
        elif self.operation_order == ('norm', 'self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn'):
            dec_input_norm = self.norm1(dec_input)
            dec_attn_out = self.self_attn(dec_input_norm, dec_input_norm, dec_input_norm, self_attn_mask)
            dec_attn_out += dec_input
            enc_dec_attn_in = self.norm2(dec_attn_out)
            enc_dec_attn_out = self.enc_attn(enc_dec_attn_in, enc_output, enc_output, dec_enc_attn_mask)
            enc_dec_attn_out += dec_attn_out
            mlp_out = self.mlp(self.norm3(enc_dec_attn_out))
            mlp_out += enc_dec_attn_out
        return mlp_out


class MaskedBalancedBCEWithLogitsLoss(nn.Module):
    """This loss combines a Sigmoid layers and a masked balanced BCE loss in
    one single class. It's AMP-eligible.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'none'.
        negative_ratio (float or int, optional): Maximum ratio of negative
            samples to positive ones. Defaults to 3.
        fallback_negative_num (int, optional): When the mask contains no
            positive samples, the number of negative samples to be sampled.
            Defaults to 0.
        eps (float, optional): Eps to avoid zero-division error.  Defaults to
            1e-6.
    """

    def __init__(self, reduction: 'str'='none', negative_ratio: 'Union[float, int]'=3, fallback_negative_num: 'int'=0, eps: 'float'=1e-06) ->None:
        super().__init__()
        assert reduction in ['none', 'mean', 'sum']
        assert isinstance(negative_ratio, (float, int))
        assert isinstance(fallback_negative_num, int)
        assert isinstance(eps, float)
        self.eps = eps
        self.negative_ratio = negative_ratio
        self.reduction = reduction
        self.fallback_negative_num = fallback_negative_num
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred: 'torch.Tensor', gt: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction in any shape.
            gt (torch.Tensor): The learning target of the prediction in the
                same shape as pred.
            mask (torch.Tensor, optional): Binary mask in the same shape of
                pred, indicating positive regions to calculate the loss. Whole
                region will be taken into account if not provided. Defaults to
                None.

        Returns:
            torch.Tensor: The loss value.
        """
        assert pred.size() == gt.size() and gt.numel() > 0
        if mask is None:
            mask = torch.ones_like(gt)
        assert mask.size() == gt.size()
        positive = (gt * mask).float()
        negative = ((1 - gt) * mask).float()
        positive_count = int(positive.sum())
        if positive_count == 0:
            negative_count = min(int(negative.sum()), self.fallback_negative_num)
        else:
            negative_count = min(int(negative.sum()), int(positive_count * self.negative_ratio))
        assert gt.max() <= 1 and gt.min() >= 0
        loss = self.loss(pred, gt)
        positive_loss = loss * positive
        negative_loss = loss * negative
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)
        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        return balance_loss


class MaskedBalancedBCELoss(MaskedBalancedBCEWithLogitsLoss):
    """Masked Balanced BCE loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'none'.
        negative_ratio (float or int): Maximum ratio of negative
            samples to positive ones. Defaults to 3.
        fallback_negative_num (int): When the mask contains no
            positive samples, the number of negative samples to be sampled.
            Defaults to 0.
        eps (float): Eps to avoid zero-division error.  Defaults to
            1e-6.
    """

    def __init__(self, reduction: 'str'='none', negative_ratio: 'Union[float, int]'=3, fallback_negative_num: 'int'=0, eps: 'float'=1e-06) ->None:
        super().__init__()
        assert reduction in ['none', 'mean', 'sum']
        assert isinstance(negative_ratio, (float, int))
        assert isinstance(fallback_negative_num, int)
        assert isinstance(eps, float)
        self.eps = eps
        self.negative_ratio = negative_ratio
        self.reduction = reduction
        self.fallback_negative_num = fallback_negative_num
        self.loss = nn.BCELoss(reduction=reduction)

    def forward(self, pred: 'torch.Tensor', gt: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction in any shape.
            gt (torch.Tensor): The learning target of the prediction in the
                same shape as pred.
            mask (torch.Tensor, optional): Binary mask in the same shape of
                pred, indicating positive regions to calculate the loss. Whole
                region will be taken into account if not provided. Defaults to
                None.

        Returns:
            torch.Tensor: The loss value.
        """
        assert pred.max() <= 1 and pred.min() >= 0
        return super().forward(pred, gt, mask)


class MaskedBCEWithLogitsLoss(nn.Module):
    """This loss combines a Sigmoid layers and a masked BCE loss in one single
    class. It's AMP-eligible.

    Args:
        eps (float): Eps to avoid zero-division error.  Defaults to
            1e-6.
    """

    def __init__(self, eps: 'float'=1e-06) ->None:
        super().__init__()
        assert isinstance(eps, float)
        self.eps = eps
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred: 'torch.Tensor', gt: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction in any shape.
            gt (torch.Tensor): The learning target of the prediction in the
                same shape as pred.
            mask (torch.Tensor, optional): Binary mask in the same shape of
                pred, indicating positive regions to calculate the loss. Whole
                region will be taken into account if not provided. Defaults to
                None.

        Returns:
            torch.Tensor: The loss value.
        """
        assert pred.size() == gt.size() and gt.numel() > 0
        if mask is None:
            mask = torch.ones_like(gt)
        assert mask.size() == gt.size()
        assert gt.max() <= 1 and gt.min() >= 0
        loss = self.loss(pred, gt)
        return (loss * mask).sum() / (mask.sum() + self.eps)


class MaskedBCELoss(MaskedBCEWithLogitsLoss):
    """Masked BCE loss.

    Args:
        eps (float): Eps to avoid zero-division error.  Defaults to
            1e-6.
    """

    def __init__(self, eps: 'float'=1e-06) ->None:
        super().__init__()
        assert isinstance(eps, float)
        self.eps = eps
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, pred: 'torch.Tensor', gt: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction in any shape.
            gt (torch.Tensor): The learning target of the prediction in the
                same shape as pred.
            mask (torch.Tensor, optional): Binary mask in the same shape of
                pred, indicating positive regions to calculate the loss. Whole
                region will be taken into account if not provided. Defaults to
                None.

        Returns:
            torch.Tensor: The loss value.
        """
        assert pred.max() <= 1 and pred.min() >= 0
        return super().forward(pred, gt, mask)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross entropy loss."""


class MaskedDiceLoss(nn.Module):
    """Masked dice loss.

    Args:
        eps (float, optional): Eps to avoid zero-divison error.  Defaults to
            1e-6.
    """

    def __init__(self, eps: 'float'=1e-06) ->None:
        super().__init__()
        assert isinstance(eps, float)
        self.eps = eps

    def forward(self, pred: 'torch.Tensor', gt: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction in any shape.
            gt (torch.Tensor): The learning target of the prediction in the
                same shape as pred.
            mask (torch.Tensor, optional): Binary mask in the same shape of
                pred, indicating positive regions to calculate the loss. Whole
                region will be taken into account if not provided. Defaults to
                None.

        Returns:
            torch.Tensor: The loss value.
        """
        assert pred.size() == gt.size() and gt.numel() > 0
        if mask is None:
            mask = torch.ones_like(gt)
        assert mask.size() == gt.size()
        pred = pred.contiguous().view(pred.size(0), -1)
        gt = gt.contiguous().view(gt.size(0), -1)
        mask = mask.contiguous().view(mask.size(0), -1)
        pred = pred * mask
        gt = gt * mask
        dice_coeff = 2 * (pred * gt).sum() / (pred.sum() + gt.sum() + self.eps)
        return 1 - dice_coeff


class MaskedSquareDiceLoss(nn.Module):
    """Masked square dice loss.

    Args:
        eps (float, optional): Eps to avoid zero-divison error.  Defaults to
            1e-3.
    """

    def __init__(self, eps: 'float'=0.001) ->None:
        super().__init__()
        assert isinstance(eps, float)
        self.eps = eps

    def forward(self, pred: 'torch.Tensor', gt: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction in any shape.
            gt (torch.Tensor): The learning target of the prediction in the
                same shape as pred.
            mask (torch.Tensor, optional): Binary mask in the same shape of
                pred, indicating positive regions to calculate the loss. Whole
                region will be taken into account if not provided. Defaults to
                None.

        Returns:
            torch.Tensor: The loss value.
        """
        assert pred.size() == gt.size() and gt.numel() > 0
        if mask is None:
            mask = torch.ones_like(gt)
        assert mask.size() == gt.size()
        batch_size = pred.size(0)
        pred = pred.contiguous().view(batch_size, -1)
        gt = gt.contiguous().view(batch_size, -1).float()
        mask = mask.contiguous().view(batch_size, -1).float()
        pred = pred * mask
        gt = gt * mask
        a = torch.sum(pred * gt, dim=1)
        b = torch.sum(pred * pred, dim=1) + self.eps
        c = torch.sum(gt * gt, dim=1) + self.eps
        d = 2 * a / (b + c)
        loss = 1 - d
        loss = torch.mean(loss)
        return loss


class SmoothL1Loss(nn.SmoothL1Loss):
    """Smooth L1 loss."""


class MaskedSmoothL1Loss(nn.Module):
    """Masked Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.
        eps (float, optional): Eps to avoid zero-division error.  Defaults to
            1e-6.
    """

    def __init__(self, beta: 'Union[float, int]'=1, eps: 'float'=1e-06) ->None:
        super().__init__()
        if digit_version(torch.__version__) > digit_version('1.6.0'):
            if digit_version(torch.__version__) >= digit_version('1.13.0') and beta == 0:
                beta = beta + eps
            self.smooth_l1_loss = nn.SmoothL1Loss(beta=beta, reduction='none')
        self.eps = eps
        self.beta = beta

    def forward(self, pred: 'torch.Tensor', gt: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction in any shape.
            gt (torch.Tensor): The learning target of the prediction in the
                same shape as pred.
            mask (torch.Tensor, optional): Binary mask in the same shape of
                pred, indicating positive regions to calculate the loss. Whole
                region will be taken into account if not provided. Defaults to
                None.

        Returns:
            torch.Tensor: The loss value.
        """
        assert pred.size() == gt.size() and gt.numel() > 0
        if mask is None:
            mask = torch.ones_like(gt).bool()
        assert mask.size() == gt.size()
        x = pred * mask
        y = gt * mask
        if digit_version(torch.__version__) > digit_version('1.6.0'):
            loss = self.smooth_l1_loss(x, y)
        else:
            loss = torch.zeros_like(gt)
            diff = torch.abs(x - y)
            mask_beta = diff < self.beta
            loss[mask_beta] = 0.5 * torch.square(diff)[mask_beta] / self.beta
            loss[~mask_beta] = diff[~mask_beta] - 0.5 * self.beta
        return loss.sum() / (mask.sum() + self.eps)


class PositionalEncoding(nn.Module):
    """Fixed positional encoding with sine and cosine functions."""

    def __init__(self, d_hid=512, n_position=200, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('position_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        denominator = torch.Tensor([(1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)])
        denominator = denominator.view(1, -1)
        pos_tensor = torch.arange(n_position).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
        return sinusoid_table.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor of shape (batch_size, pos_len, d_hid, ...)
        """
        self.device = x.device
        x = x + self.position_table[:, :x.size(1)].clone().detach()
        return self.dropout(x)


class AvgPool2d(nn.Module):
    """Applies a 2D average pooling over an input signal composed of several
    input planes.

    It can also be used as a network plugin.

    Args:
        kernel_size (int or tuple(int)): the size of the window.
        stride (int or tuple(int), optional): the stride of the window.
            Defaults to None.
        padding (int or tuple(int)): implicit zero padding. Defaults to 0.
    """

    def __init__(self, kernel_size: 'Union[int, Tuple[int]]', stride: 'Optional[Union[int, Tuple[int]]]'=None, padding: 'Union[int, Tuple[int]]'=0, **kwargs) ->None:
        super().__init__()
        self.model = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward function.
        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output tensor after Avgpooling layer.
        """
        return self.model(x)


class GNNLayer(nn.Module):
    """GNN layer for SDMGR.

    Args:
        node_dim (int): Dimension of node embedding. Defaults to 256.
        edge_dim (int): Dimension of edge embedding. Defaults to 256.
    """

    def __init__(self, node_dim: 'int'=256, edge_dim: 'int'=256) ->None:
        super().__init__()
        self.in_fc = nn.Linear(node_dim * 2 + edge_dim, node_dim)
        self.coef_fc = nn.Linear(node_dim, 1)
        self.out_fc = nn.Linear(node_dim, node_dim)
        self.relu = nn.ReLU()

    def forward(self, nodes: 'Tensor', edges: 'Tensor', nums: 'List[int]') ->Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            nodes (Tensor): Concatenated node embeddings.
            edges (Tensor): Concatenated edge embeddings.
            nums (List[int]): List of number of nodes in each batch.

        Returns:
            tuple(Tensor, Tensor):

            - nodes (Tensor): New node embeddings.
            - edges (Tensor): New edge embeddings.
        """
        start, cat_nodes = 0, []
        for num in nums:
            sample_nodes = nodes[start:start + num]
            cat_nodes.append(torch.cat([sample_nodes.unsqueeze(1).expand(-1, num, -1), sample_nodes.unsqueeze(0).expand(num, -1, -1)], -1).view(num ** 2, -1))
            start += num
        cat_nodes = torch.cat([torch.cat(cat_nodes), edges], -1)
        cat_nodes = self.relu(self.in_fc(cat_nodes))
        coefs = self.coef_fc(cat_nodes)
        start, residuals = 0, []
        for num in nums:
            residual = F.softmax(-torch.eye(num).unsqueeze(-1) * 1000000000.0 + coefs[start:start + num ** 2].view(num, num, -1), 1)
            residuals.append((residual * cat_nodes[start:start + num ** 2].view(num, num, -1)).sum(1))
            start += num ** 2
        nodes += self.relu(self.out_fc(torch.cat(residuals)))
        return nodes, cat_nodes


class FusionBlock(nn.Module):
    """Fusion block of SDMGR.

    Args:
        input_dims (tuple(int, int)): Visual dimension and node embedding
            dimension.
        output_dim (int): Output dimension.
        mm_dim (int): Model dimension. Defaults to 1600.
        chunks (int): Number of chunks. Defaults to 20.
        rank (int): Rank number. Defaults to 15.
        shared (bool): Whether to share the project layer between visual and
            node embedding features. Defaults to False.
        dropout_input (float): Dropout rate after the first projection layer.
            Defaults to 0.
        dropout_pre_lin (float): Dropout rate before the final project layer.
            Defaults to 0.
        dropout_pre_lin (float): Dropout rate after the final project layer.
            Defaults to 0.
        pos_norm (str): The normalization position. Options are 'before_cat'
            and 'after_cat'. Defaults to 'before_cat'.
    """

    def __init__(self, input_dims: 'Tuple[int, int]', output_dim: 'int', mm_dim: 'int'=1600, chunks: 'int'=20, rank: 'int'=15, shared: 'bool'=False, dropout_input: 'float'=0.0, dropout_pre_lin: 'float'=0.0, dropout_output: 'float'=0.0, pos_norm: 'str'='before_cat') ->None:
        super().__init__()
        self.rank = rank
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert pos_norm in ['before_cat', 'after_cat']
        self.pos_norm = pos_norm
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = self.linear0 if shared else nn.Linear(input_dims[1], mm_dim)
        self.merge_linears0 = nn.ModuleList()
        self.merge_linears1 = nn.ModuleList()
        self.chunks = self.chunk_sizes(mm_dim, chunks)
        for size in self.chunks:
            ml0 = nn.Linear(size, size * rank)
            self.merge_linears0.append(ml0)
            ml1 = ml0 if shared else nn.Linear(size, size * rank)
            self.merge_linears1.append(ml1)
        self.linear_out = nn.Linear(mm_dim, output_dim)

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward function."""
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bs = x1.size(0)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = torch.split(x0, self.chunks, -1)
        x1_chunks = torch.split(x1, self.chunks, -1)
        zs = []
        for x0_c, x1_c, m0, m1 in zip(x0_chunks, x1_chunks, self.merge_linears0, self.merge_linears1):
            m = m0(x0_c) * m1(x1_c)
            m = m.view(bs, self.rank, -1)
            z = torch.sum(m, 1)
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z)
            zs.append(z)
        z = torch.cat(zs, 1)
        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z

    @staticmethod
    def chunk_sizes(dim: 'int', chunks: 'int') ->List[int]:
        """Compute chunk sizes."""
        split_size = (dim + chunks - 1) // chunks
        sizes_list = [split_size] * chunks
        sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)
        return sizes_list


class SDMGRModuleLoss(nn.Module):
    """The implementation the loss of key information extraction proposed in
    the paper: `Spatial Dual-Modality Graph Reasoning for Key Information
    Extraction <https://arxiv.org/abs/2103.14470>`_.

    Args:
        weight_node (float): Weight of node loss. Defaults to 1.0.
        weight_edge (float): Weight of edge loss. Defaults to 1.0.
        ignore_idx (int): Node label to ignore. Defaults to -100.
    """

    def __init__(self, weight_node: 'float'=1.0, weight_edge: 'float'=1.0, ignore_idx: 'int'=-100) ->None:
        super().__init__()
        self.loss_node = nn.CrossEntropyLoss(ignore_index=ignore_idx)
        self.loss_edge = nn.CrossEntropyLoss(ignore_index=-1)
        self.weight_node = weight_node
        self.weight_edge = weight_edge
        self.ignore_idx = ignore_idx

    def forward(self, preds: 'Tuple[Tensor, Tensor]', data_samples: 'List[KIEDataSample]') ->Dict:
        """Forward function.

        Args:
            preds (tuple(Tensor, Tensor)):
            data_samples (list[KIEDataSample]): A list of datasamples
                containing ``gt_instances.labels`` and
                ``gt_instances.edge_labels``.

        Returns:
            dict(str, Tensor): Loss dict, containing ``loss_node``,
            ``loss_edge``, ``acc_node`` and ``acc_edge``.
        """
        node_preds, edge_preds = preds
        node_gts, edge_gts = [], []
        for data_sample in data_samples:
            node_gts.append(data_sample.gt_instances.labels)
            edge_gts.append(data_sample.gt_instances.edge_labels.reshape(-1))
        node_gts = torch.cat(node_gts).long()
        edge_gts = torch.cat(edge_gts).long()
        node_valids = torch.nonzero(node_gts != self.ignore_idx, as_tuple=False).reshape(-1)
        edge_valids = torch.nonzero(edge_gts != -1, as_tuple=False).reshape(-1)
        return dict(loss_node=self.weight_node * self.loss_node(node_preds, node_gts), loss_edge=self.weight_edge * self.loss_edge(edge_preds, edge_gts), acc_node=accuracy(node_preds[node_valids], node_gts[node_valids]), acc_edge=accuracy(edge_preds[edge_valids], edge_gts[edge_valids]))


class BaseTextDetModuleLoss(nn.Module, metaclass=ABCMeta):
    """Base class for text detection module loss.
    """

    def __init__(self) ->None:
        super().__init__()

    @abstractmethod
    def forward(self, inputs: 'INPUT_TYPES', data_samples: 'DetSampleList'=None) ->Dict:
        """Calculates losses from a batch of inputs and data samples. Returns a
        dict of losses.

        Args:
            inputs (Tensor or list[Tensor] or dict): The raw tensor outputs
                from the model.
            data_samples (list(TextDetDataSample)): Datasamples containing
                ground truth data.

        Returns:
            dict: A dict of losses.
        """
        pass

    @abstractmethod
    def get_targets(self, data_samples: 'DetSampleList') ->Tuple:
        """Generates loss targets from data samples. Returns a tuple of target
        tensors.

        Args:
            data_samples (list(TextDetDataSample)): Ground truth data samples.

        Returns:
            tuple: A tuple of target tensors.
        """
        pass


class PANEmbLossV1(nn.Module):
    """The class for implementing EmbLossV1. This was partially adapted from
    https://github.com/whai362/pan_pp.pytorch.

    Args:
        feature_dim (int): The dimension of the feature. Defaults to 4.
        delta_aggregation (float): The delta for aggregation. Defaults to 0.5.
        delta_discrimination (float): The delta for discrimination.
            Defaults to 1.5.
    """

    def __init__(self, feature_dim: 'int'=4, delta_aggregation: 'float'=0.5, delta_discrimination: 'float'=1.5) ->None:
        super().__init__()
        self.feature_dim = feature_dim
        self.delta_aggregation = delta_aggregation
        self.delta_discrimination = delta_discrimination
        self.weights = 1.0, 1.0

    def _forward_single(self, emb: 'torch.Tensor', instance: 'torch.Tensor', kernel: 'torch.Tensor', training_mask: 'torch.Tensor') ->torch.Tensor:
        """Compute the loss for a single image.

        Args:
            emb (torch.Tensor): The embedding feature.
            instance (torch.Tensor): The instance feature.
            kernel (torch.Tensor): The kernel feature.
            training_mask (torch.Tensor): The effective mask.
        """
        training_mask = (training_mask > 0.5).float()
        kernel = (kernel > 0.5).float()
        instance = instance * training_mask
        instance_kernel = (instance * kernel).view(-1)
        instance = instance.view(-1)
        emb = emb.view(self.feature_dim, -1)
        unique_labels, unique_ids = torch.unique(instance_kernel, sorted=True, return_inverse=True)
        num_instance = unique_labels.size(0)
        if num_instance <= 1:
            return 0
        emb_mean = emb.new_zeros((self.feature_dim, num_instance), dtype=torch.float32)
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind_k = instance_kernel == lb
            emb_mean[:, i] = torch.mean(emb[:, ind_k], dim=1)
        l_agg = emb.new_zeros(num_instance, dtype=torch.float32)
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind = instance == lb
            emb_ = emb[:, ind]
            dist = (emb_ - emb_mean[:, i:i + 1]).norm(p=2, dim=0)
            dist = F.relu(dist - self.delta_aggregation) ** 2
            l_agg[i] = torch.mean(torch.log(dist + 1.0))
        l_agg = torch.mean(l_agg[1:])
        if num_instance > 2:
            emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
            emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(-1, self.feature_dim)
            mask = (1 - torch.eye(num_instance, dtype=torch.int8)).view(-1, 1).repeat(1, self.feature_dim)
            mask = mask.view(num_instance, num_instance, -1)
            mask[0, :, :] = 0
            mask[:, 0, :] = 0
            mask = mask.view(num_instance * num_instance, -1)
            dist = emb_interleave - emb_band
            dist = dist[mask > 0].view(-1, self.feature_dim).norm(p=2, dim=1)
            dist = F.relu(2 * self.delta_discrimination - dist) ** 2
            l_dis = torch.mean(torch.log(dist + 1.0))
        else:
            l_dis = 0
        l_agg = self.weights[0] * l_agg
        l_dis = self.weights[1] * l_dis
        l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        loss = l_agg + l_dis + l_reg
        return loss

    def forward(self, emb: 'torch.Tensor', instance: 'torch.Tensor', kernel: 'torch.Tensor', training_mask: 'torch.Tensor') ->torch.Tensor:
        """Compute the loss for a batch image.

        Args:
            emb (torch.Tensor): The embedding feature.
            instance (torch.Tensor): The instance feature.
            kernel (torch.Tensor): The kernel feature.
            training_mask (torch.Tensor): The effective mask.
        """
        loss_batch = emb.new_zeros(emb.size(0), dtype=torch.float32)
        for i in range(loss_batch.size(0)):
            loss_batch[i] = self._forward_single(emb[i], instance[i], kernel[i], training_mask[i])
        return loss_batch


ArrayLike = 'ArrayLike'


def offset_polygon(poly: 'ArrayLike', distance: 'float') ->ArrayLike:
    """Offset (expand/shrink) the polygon by the target distance. It's a
    wrapper around pyclipper based on Vatti clipping algorithm.

    Warning:
        Polygon coordinates will be casted to int type in PyClipper. Mind the
        potential precision loss caused by the casting.

    Args:
        poly (ArrayLike): A polygon. In any form can be converted
            to an 1-D numpy array. E.g. list[float], np.ndarray,
            or torch.Tensor. Polygon is written in
            [x1, y1, x2, y2, ...].
        distance (float): The offset distance. Positive value means expanding,
            negative value means shrinking.

    Returns:
        np.array: 1-D Offsetted polygon ndarray in float32 type. If the
        result polygon is invalid or has been split into several parts,
        return an empty array.
    """
    poly = np.array(poly).reshape(-1, 2)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    result = np.array(pco.Execute(distance), dtype=object)
    if len(result) > 0 and isinstance(result[0], list):
        result = np.array([])
    result = result.astype(np.float32)
    return result if len(result) == 0 else result[0].flatten()


class Embeddings(nn.Module):
    """Construct the word embeddings given vocab size and embed dim.

    Args:
        d_model (int): The embedding dimension.
        vocab (int): Vocablury size.
    """

    def __init__(self, d_model: 'int', vocab: 'int'):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input: torch.Tensor) ->torch.Tensor:
        """Forward the embeddings.

        Args:
            input (torch.Tensor): The input tensors.

        Returns:
            torch.Tensor: The embeddings.
        """
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)


def conv1x1(in_planes, out_planes):
    """1x1 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_conv1x1=False, plugins=None):
        super().__init__()
        if use_conv1x1:
            self.conv1 = conv1x1(inplanes, planes)
            self.conv2 = conv3x3(planes, planes * self.expansion, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes * self.expansion)
        self.with_plugins = False
        if plugins:
            if isinstance(plugins, dict):
                plugins = [plugins]
            self.with_plugins = True
            self.before_conv1_plugin = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'before_conv1']
            self.after_conv1_plugin = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_conv1']
            self.after_conv2_plugin = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_conv2']
            self.after_shortcut_plugin = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_shortcut']
        self.planes = planes
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride
        if self.with_plugins:
            self.before_conv1_plugin_names = self.make_block_plugins(inplanes, self.before_conv1_plugin)
            self.after_conv1_plugin_names = self.make_block_plugins(planes, self.after_conv1_plugin)
            self.after_conv2_plugin_names = self.make_block_plugins(planes, self.after_conv2_plugin)
            self.after_shortcut_plugin_names = self.make_block_plugins(planes, self.after_shortcut_plugin)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(plugin, in_channels=in_channels, out_channels=in_channels, postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    def forward(self, x):
        if self.with_plugins:
            x = self.forward_plugin(x, self.before_conv1_plugin_names)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.with_plugins:
            out = self.forward_plugin(out, self.after_conv1_plugin_names)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.with_plugins:
            out = self.forward_plugin(out, self.after_conv2_plugin_names)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if self.with_plugins:
            out = self.forward_plugin(out, self.after_shortcut_plugin_names)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if downsample:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes * self.expansion, 1, stride, bias=False), nn.BatchNorm2d(planes * self.expansion))
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class DotProductAttentionLayer(nn.Module):

    def __init__(self, dim_model=None):
        super().__init__()
        self.scale = dim_model ** -0.5 if dim_model is not None else 1.0

    def forward(self, query, key, value, mask=None):
        n, seq_len = mask.size()
        logits = torch.matmul(query.permute(0, 2, 1), key) * self.scale
        if mask is not None:
            mask = mask.view(n, 1, seq_len)
            logits = logits.masked_fill(mask, float('-inf'))
        weights = F.softmax(logits, dim=2)
        glimpse = torch.matmul(weights, value.transpose(1, 2))
        glimpse = glimpse.permute(0, 2, 1).contiguous()
        return glimpse


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super().__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class PositionAwareLayer(nn.Module):

    def __init__(self, dim_model, rnn_layers=2):
        super().__init__()
        self.dim_model = dim_model
        self.rnn = nn.LSTM(input_size=dim_model, hidden_size=dim_model, num_layers=rnn_layers, batch_first=True)
        self.mixer = nn.Sequential(nn.Conv2d(dim_model, dim_model, kernel_size=3, stride=1, padding=1), nn.ReLU(True), nn.Conv2d(dim_model, dim_model, kernel_size=3, stride=1, padding=1))

    def forward(self, img_feature):
        n, c, h, w = img_feature.size()
        rnn_input = img_feature.permute(0, 2, 3, 1).contiguous()
        rnn_input = rnn_input.view(n * h, w, c)
        rnn_output, _ = self.rnn(rnn_input)
        rnn_output = rnn_output.view(n, h, w, c)
        rnn_output = rnn_output.permute(0, 3, 1, 2).contiguous()
        out = self.mixer(rnn_output)
        return out


def list_from_file(filename, encoding='utf-8'):
    """Load a text file and parse the content as a list of strings. The
    trailing "\\r" and "\\n" of each line will be removed.

    Note:
        This will be replaced by mmcv's version after it supports encoding.

    Args:
        filename (str): Filename.
        encoding (str): Encoding used to open the file. Default utf-8.

    Returns:
        list[str]: A list of strings.
    """
    item_list = []
    with open(filename, encoding=encoding) as f:
        for line in f:
            item_list.append(line.rstrip('\n\r'))
    return item_list


class Dictionary:
    """The class generates a dictionary for recognition. It pre-defines four
    special tokens: ``start_token``, ``end_token``, ``pad_token``, and
    ``unknown_token``, which will be sequentially placed at the end of the
    dictionary when their corresponding flags are True.

    Args:
        dict_file (str): The path of Character dict file which a single
            character must occupies a line.
        with_start (bool): The flag to control whether to include the start
            token. Defaults to False.
        with_end (bool): The flag to control whether to include the end token.
            Defaults to False.
        same_start_end (bool): The flag to control whether the start token and
            end token are the same. It only works when both ``with_start`` and
            ``with_end`` are True. Defaults to False.
        with_padding (bool):The padding token may represent more than a
            padding. It can also represent tokens like the blank token in CTC
            or the background token in SegOCR. Defaults to False.
        with_unknown (bool): The flag to control whether to include the
            unknown token. Defaults to False.
        start_token (str): The start token as a string. Defaults to '<BOS>'.
        end_token (str): The end token as a string. Defaults to '<EOS>'.
        start_end_token (str): The start/end token as a string. if start and
            end is the same. Defaults to '<BOS/EOS>'.
        padding_token (str): The padding token as a string.
            Defaults to '<PAD>'.
        unknown_token (str, optional): The unknown token as a string. If it's
            set to None and ``with_unknown`` is True, the unknown token will be
            skipped when converting string to index. Defaults to '<UKN>'.
    """

    def __init__(self, dict_file: 'str', with_start: 'bool'=False, with_end: 'bool'=False, same_start_end: 'bool'=False, with_padding: 'bool'=False, with_unknown: 'bool'=False, start_token: 'str'='<BOS>', end_token: 'str'='<EOS>', start_end_token: 'str'='<BOS/EOS>', padding_token: 'str'='<PAD>', unknown_token: 'str'='<UKN>') ->None:
        self.with_start = with_start
        self.with_end = with_end
        self.same_start_end = same_start_end
        self.with_padding = with_padding
        self.with_unknown = with_unknown
        self.start_end_token = start_end_token
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token
        self.unknown_token = unknown_token
        assert isinstance(dict_file, str)
        self._dict = []
        for line_num, line in enumerate(list_from_file(dict_file)):
            line = line.strip('\r\n')
            if len(line) > 1:
                raise ValueError(f'Expect each line has 0 or 1 character, got {len(line)} characters at line {line_num + 1}')
            if line != '':
                self._dict.append(line)
        self._char2idx = {char: idx for idx, char in enumerate(self._dict)}
        self._update_dict()
        assert len(set(self._dict)) == len(self._dict), 'Invalid dictionary: Has duplicated characters.'

    @property
    def num_classes(self) ->int:
        """int: Number of output classes. Special tokens are counted.
        """
        return len(self._dict)

    @property
    def dict(self) ->list:
        """list: Returns a list of characters to recognize, where special
        tokens are counted."""
        return self._dict

    def char2idx(self, char: 'str', strict: 'bool'=True) ->int:
        """Convert a character to an index via ``Dictionary.dict``.

        Args:
            char (str): The character to convert to index.
            strict (bool): The flag to control whether to raise an exception
                when the character is not in the dictionary. Defaults to True.

        Return:
            int: The index of the character.
        """
        char_idx = self._char2idx.get(char, None)
        if char_idx is None:
            if self.with_unknown:
                return self.unknown_idx
            elif not strict:
                return None
            else:
                raise Exception(f'Chararcter: {char} not in dict, please check gt_label and use custom dict file, or set "with_unknown=True"')
        return char_idx

    def str2idx(self, string: 'str') ->List:
        """Convert a string to a list of indexes via ``Dictionary.dict``.

        Args:
            string (str): The string to convert to indexes.

        Return:
            list: The list of indexes of the string.
        """
        idx = list()
        for s in string:
            char_idx = self.char2idx(s)
            if char_idx is None:
                if self.with_unknown:
                    continue
                raise Exception(f'Chararcter: {s} not in dict, please check gt_label and use custom dict file, or set "with_unknown=True"')
            idx.append(char_idx)
        return idx

    def idx2str(self, index: 'Sequence[int]') ->str:
        """Convert a list of index to string.

        Args:
            index (list[int]): The list of indexes to convert to string.

        Return:
            str: The converted string.
        """
        assert isinstance(index, (list, tuple))
        string = ''
        for i in index:
            assert i < len(self._dict), f'Index: {i} out of range! Index must be less than {len(self._dict)}'
            string += self._dict[i]
        return string

    def _update_dict(self):
        """Update the dict with tokens according to parameters."""
        self.start_idx = None
        self.end_idx = None
        if self.with_start and self.with_end and self.same_start_end:
            self._dict.append(self.start_end_token)
            self.start_idx = len(self._dict) - 1
            self.end_idx = self.start_idx
        else:
            if self.with_start:
                self._dict.append(self.start_token)
                self.start_idx = len(self._dict) - 1
            if self.with_end:
                self._dict.append(self.end_token)
                self.end_idx = len(self._dict) - 1
        self.padding_idx = None
        if self.with_padding:
            self._dict.append(self.padding_token)
            self.padding_idx = len(self._dict) - 1
        self.unknown_idx = None
        if self.with_unknown and self.unknown_token is not None:
            self._dict.append(self.unknown_token)
            self.unknown_idx = len(self._dict) - 1
        self._char2idx = {}
        for idx, char in enumerate(self._dict):
            self._char2idx[char] = idx


class Maxpool2d(nn.Module):
    """A wrapper around nn.Maxpool2d().

    Args:
        kernel_size (int or tuple(int)): Kernel size for max pooling layer
        stride (int or tuple(int)): Stride for max pooling layer
        padding (int or tuple(int)): Padding for pooling layer
    """

    def __init__(self, kernel_size: 'Union[int, Tuple[int]]', stride: 'Union[int, Tuple[int]]', padding: 'Union[int, Tuple[int]]'=0, **kwargs) ->None:
        super().__init__()
        self.model = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x) ->torch.Tensor:
        """Forward function.
        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output tensor after Maxpooling layer.
        """
        return self.model(x)


class GCAModule(nn.Module):
    """GCAModule in MASTER.

    Args:
        in_channels (int): Channels of input tensor.
        ratio (float): Scale ratio of in_channels.
        n_head (int): Numbers of attention head.
        pooling_type (str): Spatial pooling type. Options are [``avg``,
            ``att``].
        scale_attn (bool): Whether to scale the attention map. Defaults to
            False.
        fusion_type (str): Fusion type of input and context. Options are
            [``channel_add``, ``channel_mul``, ``channel_concat``].
    """

    def __init__(self, in_channels: 'int', ratio: 'float', n_head: 'int', pooling_type: 'str'='att', scale_attn: 'bool'=False, fusion_type: 'str'='channel_add', **kwargs) ->None:
        super().__init__()
        assert pooling_type in ['avg', 'att']
        assert fusion_type in ['channel_add', 'channel_mul', 'channel_concat']
        assert in_channels % n_head == 0 and in_channels >= 8
        self.n_head = n_head
        self.in_channels = in_channels
        self.ratio = ratio
        self.planes = int(in_channels * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.scale_attn = scale_attn
        self.single_header_inplanes = int(in_channels / n_head)
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(self.single_header_inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if fusion_type == 'channel_add':
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.in_channels, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
        elif fusion_type == 'channel_concat':
            self.channel_concat_conv = nn.Sequential(nn.Conv2d(self.in_channels, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
            self.cat_conv = nn.Conv2d(2 * self.in_channels, self.in_channels, kernel_size=1)
        elif fusion_type == 'channel_mul':
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.in_channels, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.in_channels, kernel_size=1))

    def spatial_pool(self, x: 'torch.Tensor') ->torch.Tensor:
        """Spatial pooling function.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output tensor after spatial pooling.
        """
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            x = x.view(batch * self.n_head, self.single_header_inplanes, height, width)
            input_x = x
            input_x = input_x.view(batch * self.n_head, self.single_header_inplanes, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch * self.n_head, 1, height * width)
            if self.scale_attn and self.n_head > 1:
                context_mask = context_mask / torch.sqrt(self.single_header_inplanes)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, self.n_head * self.single_header_inplanes, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward function.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output tensor after GCAModule.
        """
        context = self.spatial_pool(x)
        out = x
        if self.fusion_type == 'channel_mul':
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == 'channel_add':
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            channel_concat_term = self.channel_concat_conv(context)
            _, C1, _, _ = channel_concat_term.shape
            N, C2, H, W = out.shape
            out = torch.cat([out, channel_concat_term.expand(-1, -1, H, W)], dim=1)
            out = self.cat_conv(out)
            out = nn.functional.layer_norm(out, [self.in_channels, H, W])
            out = nn.functional.relu(out)
        return out


class TPStransform(nn.Module):
    """Implement TPS transform.

    This was partially adapted from https://github.com/ayumiymk/aster.pytorch

    Args:
        output_image_size (tuple[int, int]): The size of the output image.
            Defaults to (32, 128).
        num_control_points (int): The number of control points. Defaults to 20.
        margins (tuple[float, float]): The margins for control points to the
            top and down side of the image. Defaults to [0.05, 0.05].
    """

    def __init__(self, output_image_size: 'Tuple[int, int]'=(32, 100), num_control_points: 'int'=20, margins: 'Tuple[float, float]'=[0.05, 0.05]) ->None:
        super().__init__()
        self.output_image_size = output_image_size
        self.num_control_points = num_control_points
        self.margins = margins
        self.target_height, self.target_width = output_image_size
        target_control_points = self._build_output_control_points(num_control_points, margins)
        N = num_control_points
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = self._compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        inverse_kernel = torch.inverse(forward_kernel).contiguous()
        HW = self.target_height * self.target_width
        tgt_coord = list(itertools.product(range(self.target_height), range(self.target_width)))
        tgt_coord = torch.Tensor(tgt_coord)
        Y, X = tgt_coord.split(1, dim=1)
        Y = Y / (self.target_height - 1)
        X = X / (self.target_width - 1)
        tgt_coord = torch.cat([X, Y], dim=1)
        tgt_coord_partial_repr = self._compute_partial_repr(tgt_coord, target_control_points)
        tgt_coord_repr = torch.cat([tgt_coord_partial_repr, torch.ones(HW, 1), tgt_coord], dim=1)
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', tgt_coord_repr)
        self.register_buffer('target_control_points', target_control_points)

    def forward(self, input: 'torch.Tensor', source_control_points: 'torch.Tensor') ->torch.Tensor:
        """Forward function of the TPS block.

        Args:
            input (Tensor): The input image.
            source_control_points (Tensor): The control points of the source
                image of shape (N, self.num_control_points, 2).
        Returns:
            Tensor: The output image after TPS transform.
        """
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_control_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)
        Y = torch.cat([source_control_points, self.padding_matrix.expand(batch_size, 3, 2)], 1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)
        grid = source_coordinate.view(-1, self.target_height, self.target_width, 2)
        grid = torch.clamp(grid, 0, 1)
        grid = 2.0 * grid - 1.0
        output_maps = self._grid_sample(input, grid, canvas=None)
        return output_maps

    def _grid_sample(self, input: 'torch.Tensor', grid: 'torch.Tensor', canvas: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Sample the input image at the given grid.

        Args:
            input (Tensor): The input image.
            grid (Tensor): The grid to sample the input image.
            canvas (Optional[Tensor]): The canvas to store the output image.
        Returns:
            Tensor: The sampled image.
        """
        output = F.grid_sample(input, grid, align_corners=True)
        if canvas is None:
            return output
        else:
            input_mask = input.data.new(input.size()).fill_(1)
            output_mask = F.grid_sample(input_mask, grid, align_corners=True)
            padded_output = output * output_mask + canvas * (1 - output_mask)
            return padded_output

    def _compute_partial_repr(self, input_points: 'torch.Tensor', control_points: 'torch.Tensor') ->torch.Tensor:
        """Compute the partial representation matrix.

        Args:
            input_points (Tensor): The input points.
            control_points (Tensor): The control points.
        Returns:
            Tensor: The partial representation matrix.
        """
        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
        mask = repr_matrix != repr_matrix
        repr_matrix.masked_fill_(mask, 0)
        return repr_matrix

    def _build_output_control_points(self, num_control_points: 'torch.Tensor', margins: 'Tuple[float, float]') ->torch.Tensor:
        """Build the output control points.

        The output points will be fix at
        top and down side of the image.
        Args:
            num_control_points (Tensor): The number of control points.
            margins (Tuple[float, float]): The margins for control points to
                the top and down side of the image.
        Returns:
            Tensor: The output control points.
        """
        margin_x, margin_y = margins
        num_ctrl_pts_per_side = num_control_points // 2
        ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
        ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
        ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        output_ctrl_pts_arr = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        output_ctrl_pts = torch.Tensor(output_ctrl_pts_arr)
        return output_ctrl_pts


INF = 100000000.0


def bezier_coefficient(n, t, k):
    return t ** k * (1 - t) ** (n - k) * n_over_k(n, k)


def bezier_coefficients(time, point_num, ratios):
    return [[bezier_coefficient(time, ratio, num) for num in range(point_num)] for ratio in ratios]


def linear_interpolation(point1: 'np.ndarray', point2: 'np.ndarray', number: 'int'=2) ->np.ndarray:
    t = np.linspace(0, 1, number + 2).reshape(-1, 1)
    return point1 + (point2 - point1) * t


def curve2bezier(curve: 'ArrayLike'):
    curve = np.array(curve).reshape(-1, 2)
    if len(curve) == 2:
        return linear_interpolation(curve[0], curve[1])
    diff = curve[1:] - curve[:-1]
    distance = np.linalg.norm(diff, axis=-1)
    norm_distance = distance / distance.sum()
    norm_distance = np.hstack(([0], norm_distance))
    cum_norm_dis = norm_distance.cumsum()
    pseudo_inv = np.linalg.pinv(bezier_coefficients(3, 4, cum_norm_dis))
    control_points = pseudo_inv.dot(curve)
    return control_points


def poly2bezier(poly):
    poly = np.array(poly).reshape(-1, 2)
    points_num = len(poly)
    up_curve = poly[:points_num // 2]
    down_curve = poly[points_num // 2:]
    up_bezier = curve2bezier(up_curve)
    down_bezier = curve2bezier(down_curve)
    up_bezier[0] = up_curve[0]
    up_bezier[-1] = up_curve[-1]
    down_bezier[0] = down_curve[0]
    down_bezier[-1] = down_curve[-1]
    return np.vstack((up_bezier, down_bezier)).flatten().tolist()


class ABCNetDetModuleLoss(BaseTextDetModuleLoss):

    def __init__(self, num_classes: 'int'=1, bbox_coder: 'ConfigType'=dict(type='mmdet.DistancePointBBoxCoder'), regress_ranges: 'RangeType'=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)), strides: 'List[int]'=(8, 16, 32, 64, 128), center_sampling: 'bool'=True, center_sample_radius: 'float'=1.5, norm_on_bbox: 'bool'=True, loss_cls: 'ConfigType'=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), loss_bbox: 'ConfigType'=dict(type='mmdet.GIoULoss', loss_weight=1.0), loss_centerness: 'ConfigType'=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bezier: 'ConfigType'=dict(type='mmdet.SmoothL1Loss', reduction='mean', loss_weight=1.0)) ->None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.prior_generator = MlvlPointGenerator(strides)
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.loss_centerness = MODELS.build(loss_centerness)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_bezier = MODELS.build(loss_bezier)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

    def forward(self, inputs: 'Tuple[Tensor]', data_samples: 'DetSampleList') ->Dict:
        """Compute ABCNet loss.

        Args:
            inputs (tuple(tensor)): Raw predictions from model, containing
                ``cls_scores``, ``bbox_preds``, ``beizer_preds`` and
                ``centernesses``.
                Each is a tensor of shape :math:`(N, H, W)`.
            data_samples (list[TextDetDataSample]): The data samples.

        Returns:
            dict: The dict for abcnet-det losses with loss_cls, loss_bbox,
            loss_centerness and loss_bezier.
        """
        cls_scores, bbox_preds, centernesses, beizer_preds = inputs
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(beizer_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
        labels, bbox_targets, bezier_targets = self.get_targets(all_level_points, data_samples)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_centerness = [centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses]
        flatten_bezier_preds = [bezier_pred.permute(0, 2, 3, 1).reshape(-1, 16) for bezier_pred in beizer_preds]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_bezier_preds = torch.cat(flatten_bezier_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_bezier_targets = torch.cat(bezier_targets)
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bezier_preds = flatten_bezier_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        pos_bezier_targets = flatten_bezier_targets[pos_inds]
        centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-06)
        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(pos_decoded_bbox_preds, pos_decoded_target_preds, weight=pos_centerness_targets, avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(pos_centerness, pos_centerness_targets, avg_factor=num_pos)
            loss_bezier = self.loss_bezier(pos_bezier_preds, pos_bezier_targets, weight=pos_centerness_targets[:, None], avg_factor=centerness_denorm)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            loss_bezier = pos_bezier_preds.sum()
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_centerness=loss_centerness, loss_bezier=loss_bezier)

    def get_targets(self, points: 'List[Tensor]', data_samples: 'DetSampleList') ->Tuple[List[Tensor], List[Tensor]]:
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            data_samples: Batch of data samples. Each data sample contains
                a gt_instance, which usually includes bboxes and labels
                attributes.

        Returns:
            tuple: Targets of each level.

            - concat_lvl_labels (list[Tensor]): Labels of each level.
            - concat_lvl_bbox_targets (list[Tensor]): BBox targets of each             level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i]) for i in range(num_levels)]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]
        labels_list, bbox_targets_list, bezier_targets_list = multi_apply(self._get_targets_single, data_samples, points=concat_points, regress_ranges=concat_regress_ranges, num_points_per_lvl=num_points)
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list]
        bezier_targets_list = [bezier_targets.split(num_points, 0) for bezier_targets in bezier_targets_list]
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_bezier_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list])
            bezier_targets = torch.cat([bezier_targets[i] for bezier_targets in bezier_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
                bezier_targets = bezier_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_bezier_targets.append(bezier_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_bezier_targets

    def _get_targets_single(self, data_sample: 'TextDetDataSample', points: 'Tensor', regress_ranges: 'Tensor', num_points_per_lvl: 'List[int]') ->Tuple[Tensor, Tensor, Tensor]:
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        gt_instances = data_sample.gt_instances
        gt_instances = gt_instances[~gt_instances.ignored]
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        data_sample.gt_instances = gt_instances
        polygons = gt_instances.polygons
        beziers = gt_bboxes.new([poly2bezier(poly) for poly in polygons])
        gt_instances.beziers = beziers
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), gt_bboxes.new_zeros((num_points, 4)), gt_bboxes.new_zeros((num_points, 16))
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
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
        beziers = beziers.reshape(-1, 8, 2)[None].expand(num_points, num_gts, 8, 2)
        beziers_left = beziers[..., 0] - xs[..., None]
        beziers_right = beziers[..., 1] - ys[..., None]
        bezier_targets = torch.stack((beziers_left, beziers_right), dim=-1)
        bezier_targets = bezier_targets.view(num_points, num_gts, 16)
        if self.center_sampling:
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2], gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3], gt_bboxes[..., 3], y_maxs)
            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack((cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (max_regress_distance >= regress_ranges[..., 0]) & (max_regress_distance <= regress_ranges[..., 1])
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        bezier_targets = bezier_targets[range(num_points), min_area_inds]
        return labels, bbox_targets, bezier_targets

    def centerness_target(self, pos_bbox_targets: 'Tensor') ->Tensor:
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0] * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)


class PositionEmbeddingSine(nn.Module):
    """This is a more standard version of the position embedding, very similar
    to the one used by the Attention is all you need paper, generalized to work
    on images.

    Adapted from https://github.com/shannanyinxiang/SPTS.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask: 'Tensor'):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class DecoderEmbeddings(nn.Module):

    def __init__(self, num_classes: 'int', padding_idx: 'int', hidden_dim, max_position_embeddings, dropout):
        super(DecoderEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(num_classes, hidden_dim, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        self.LayerNorm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
        if self.norm is not None:
            output = self.norm(output)
        return output.unsqueeze(0)


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
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

    def forward_post(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
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

    def forward_post(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TDAugment(torch.nn.Module):

    def forward(self, inputs, data_samples):
        return inputs, data_samples


class Augment(torch.nn.Module):

    def forward(self, inputs, data_samples):
        return inputs, data_samples


class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def test_step(self, x):
        return self.forward(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Augment,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (AvgPool2d,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BidirectionalLSTM,
     lambda: ([], {'nIn': 4, 'nHidden': 4, 'nOut': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (CrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DummyModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FusionBlock,
     lambda: ([], {'input_dims': [4, 4], 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MaskedBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MaskedBCEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MaskedBalancedBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MaskedBalancedBCEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MaskedDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MaskedSquareDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Maxpool2d,
     lambda: ([], {'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionAwareLayer,
     lambda: ([], {'dim_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 512])], {})),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SmoothL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TDAugment,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TransformerDecoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

