
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


import random


import numpy as np


import torch


from scipy import ndimage


from scipy.ndimage.interpolation import zoom


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from scipy.ndimage.morphology import binary_dilation


import torch.nn as nn


from torch.nn import functional as F


from typing import Tuple


from torch import nn


import logging


import torch.backends.cudnn as cudnn


import warnings


import time


import torch.optim as optim


from torch.nn.modules.loss import CrossEntropyLoss


from torchvision import transforms


import matplotlib.pyplot as plt


import pandas as pd


from scipy.ndimage import zoom


class Cross_Attention(nn.Module):

    def __init__(self, key_channels, value_channels, height, width, head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.height = height
        self.width = width
        self.reprojection = nn.Conv2d(value_channels, 2 * value_channels, 1)
        self.norm = nn.LayerNorm(2 * value_channels)

    def forward(self, x1, x2):
        B, N, D = x1.size()
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels:(i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels:(i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels:(i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = context.transpose(1, 2) @ query
            attended_values.append(attended_value)
        aggregated_values = torch.cat(attended_values, dim=1).reshape(B, D, self.height, self.width)
        reprojected_value = self.reprojection(aggregated_values).reshape(B, 2 * D, N).permute(0, 2, 1)
        reprojected_value = self.norm(reprojected_value)
        return reprojected_value


class MLP_FFN(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DWConv(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: 'torch.Tensor', H, W) ->torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: 'torch.Tensor', H, W) ->torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out


class MixFFN_skip(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: 'torch.Tensor', H, W) ->torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """

    def __init__(self, in_dim, key_dim, value_dim, height, width, head_count=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.H = height
        self.W = width
        self.attn = Cross_Attention(key_dim, value_dim, height, width, head_count=head_count)
        self.norm2 = nn.LayerNorm(in_dim * 2)
        if token_mlp == 'mix':
            self.mlp = MixFFN(in_dim * 2, int(in_dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(in_dim * 2, int(in_dim * 4))
        else:
            self.mlp = MLP_FFN(in_dim * 2, int(in_dim * 4))

    def forward(self, x1: 'torch.Tensor', x2: 'torch.Tensor') ->torch.Tensor:
        norm_1 = self.norm1(x1)
        norm_2 = self.norm1(x2)
        attn = self.attn(norm_1, norm_2)
        residual = torch.cat([x1, x2], dim=2)
        tx = residual + attn
        mx = tx + self.mlp(self.norm2(tx), self.H, self.W)
        return mx


class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]

    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually

    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels:(i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels:(i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels:(i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)
        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)
        return attention


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DualTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = ChannelAttention(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)
        if token_mlp == 'mix':
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x: 'torch.Tensor', H, W) ->torch.Tensor:
        norm1 = self.norm1(x)
        norm1 = Rearrange('b (h w) d -> b d h w', h=H, w=W)(norm1)
        attn = self.attn(norm1)
        attn = Rearrange('b d h w -> b (h w) d')(attn)
        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W)
        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        channel_attn = self.channel_attn(norm3)
        add3 = add2 + channel_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, H, W)
        mx = add3 + mlp2
        return mx


class OverlapPatchEmbeddings(nn.Module):

    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W


class EfficientSelfAtten(nn.Module):

    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: 'torch.Tensor', H, W) ->torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn_score = attn.softmax(dim=-1)
        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)
        return out


class TransformerBlock(nn.Module):

    def __init__(self, dim, head, reduction_ratio=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp == 'mix':
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x: 'torch.Tensor', H, W) ->torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx


class MiT(nn.Module):

    def __init__(self, image_size, dims, layers, token_mlp='mix_skip'):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], dims[0], dims[1])
        self.patch_embed3 = OverlapPatchEmbeddings(image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], dims[1], dims[2])
        self.patch_embed4 = OverlapPatchEmbeddings(image_size // 16, patch_sizes[3], strides[3], padding_sizes[3], dims[2], dims[3])
        self.block1 = nn.ModuleList([TransformerBlock(dims[0], heads[0], reduction_ratios[0], token_mlp) for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(dims[0])
        self.block2 = nn.ModuleList([TransformerBlock(dims[1], heads[1], reduction_ratios[1], token_mlp) for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(dims[1])
        self.block3 = nn.ModuleList([TransformerBlock(dims[2], heads[2], reduction_ratios[2], token_mlp) for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(dims[2])
        self.block4 = nn.ModuleList([TransformerBlock(dims[3], heads[3], reduction_ratios[3], token_mlp) for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(dims[3])

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        B = x.shape[0]
        outs = []
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs


class PatchExpand(nn.Module):

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())
        return x


class FinalPatchExpand_X4(nn.Module):

    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C // self.dim_scale ** 2)
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())
        return x


class MyDecoderLayer(nn.Module):

    def __init__(self, input_size, in_out_chan, head_count, token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode)
            self.concat_linear = nn.Linear(2 * dims, out_dim)
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode)
            self.concat_linear = nn.Linear(4 * dims, out_dim)
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)
        self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            x1_expand = self.x1_linear(x1)
            cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2))
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)
            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            out = self.layer_up(x1)
        return out


class DAEFormer(nn.Module):

    def __init__(self, num_classes=9, head_count=1, token_mlp_mode='mix_skip'):
        super().__init__()
        dims, key_dim, value_dim, layers = [[128, 320, 512], [128, 320, 512], [128, 320, 512], [2, 2, 2]]
        self.backbone = MiT(image_size=224, in_dim=dims, key_dim=key_dim, value_dim=value_dim, layers=layers, head_count=head_count, token_mlp=token_mlp_mode)
        d_base_feat_size = 7
        in_out_chan = [[64, 128, 128, 128, 160], [320, 320, 320, 320, 256], [512, 512, 512, 512, 512]]
        self.decoder_2 = MyDecoderLayer((d_base_feat_size * 2, d_base_feat_size * 2), in_out_chan[2], head_count, token_mlp_mode, n_class=num_classes)
        self.decoder_1 = MyDecoderLayer((d_base_feat_size * 4, d_base_feat_size * 4), in_out_chan[1], head_count, token_mlp_mode, n_class=num_classes)
        self.decoder_0 = MyDecoderLayer((d_base_feat_size * 8, d_base_feat_size * 8), in_out_chan[0], head_count, token_mlp_mode, n_class=num_classes, is_last=True)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        output_enc = self.backbone(x)
        b, c, _, _ = output_enc[2].shape
        tmp_2 = self.decoder_2(output_enc[2].permute(0, 2, 3, 1).view(b, -1, c))
        tmp_1 = self.decoder_1(tmp_2, output_enc[1].permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc[0].permute(0, 2, 3, 1))
        return tmp_0


class SelfAtten(nn.Module):

    def __init__(self, dim, head):
        super().__init__()
        self.head = head
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn_score = attn.softmax(dim=-1)
        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)
        return out


class Scale_reduce(nn.Module):

    def __init__(self, dim, reduction_ratio):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        if len(self.reduction_ratio) == 4:
            self.sr0 = nn.Conv2d(dim, dim, reduction_ratio[3], reduction_ratio[3])
            self.sr1 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
            self.sr2 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[1], reduction_ratio[1])
        elif len(self.reduction_ratio) == 3:
            self.sr0 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
            self.sr1 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[1], reduction_ratio[1])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        B, N, C = x.shape
        if len(self.reduction_ratio) == 4:
            tem0 = x[:, :3136, :].reshape(B, 56, 56, C).permute(0, 3, 1, 2)
            tem1 = x[:, 3136:4704, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
            tem2 = x[:, 4704:5684, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
            tem3 = x[:, 5684:6076, :]
            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)
            reduce_out = self.norm(torch.cat([sr_0, sr_1, sr_2, tem3], -2))
        if len(self.reduction_ratio) == 3:
            tem0 = x[:, :1568, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
            tem1 = x[:, 1568:2548, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
            tem2 = x[:, 2548:2940, :]
            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            reduce_out = self.norm(torch.cat([sr_0, sr_1, tem2], -2))
        return reduce_out


class M_EfficientSelfAtten(nn.Module):

    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        if reduction_ratio is not None:
            self.scale_reduce = Scale_reduce(dim, reduction_ratio)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn_score = attn.softmax(dim=-1)
        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)
        return out


class LocalEnhance_EfficientSelfAtten(nn.Module):

    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.local_pos = DWConv(dim)
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: 'torch.Tensor', H, W) ->torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn_score = attn.softmax(dim=-1)
        local_v = v.permute(0, 2, 1, 3).reshape(B, N, C)
        local_pos = self.local_pos(local_v).reshape(B, -1, self.head, C // self.head).permute(0, 2, 1, 3)
        x_atten = (attn_score @ v + local_pos).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)
        return out


class MixD_FFN(nn.Module):

    def __init__(self, c1, c2, fuse_mode='add'):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1) if fuse_mode == 'add' else nn.Linear(c2 * 2, c1)
        self.fuse_mode = fuse_mode

    def forward(self, x):
        ax = self.dwconv(self.fc1(x), H, W)
        fuse = self.act(ax + self.fc1(x)) if self.fuse_mode == 'add' else self.act(torch.cat([ax, self.fc1(x)], 2))
        out = self.fc2(ax)
        return out


class MLP(nn.Module):

    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class ConvModule(nn.Module):

    def __init__(self, c1, c2, k):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class FuseTransformerBlock(nn.Module):

    def __init__(self, dim, head, reduction_ratio=1, fuse_mode='add'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixD_FFN(dim, int(dim * 4), fuse_mode)

    def forward(self, x: 'torch.Tensor', H, W) ->torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx


class FuseMiT(nn.Module):

    def __init__(self, image_size, dims, layers, fuse_mode='add'):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], dims[0], dims[1])
        self.patch_embed3 = OverlapPatchEmbeddings(image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], dims[1], dims[2])
        self.patch_embed4 = OverlapPatchEmbeddings(image_size // 16, patch_sizes[3], strides[3], padding_sizes[3], dims[2], dims[3])
        self.block1 = nn.ModuleList([FuseTransformerBlock(dims[0], heads[0], reduction_ratios[0], fuse_mode) for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(dims[0])
        self.block2 = nn.ModuleList([FuseTransformerBlock(dims[1], heads[1], reduction_ratios[1], fuse_mode) for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(dims[1])
        self.block3 = nn.ModuleList([FuseTransformerBlock(dims[2], heads[2], reduction_ratios[2], fuse_mode) for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(dims[2])
        self.block4 = nn.ModuleList([FuseTransformerBlock(dims[3], heads[3], reduction_ratios[3], fuse_mode) for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(dims[3])

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        B = x.shape[0]
        outs = []
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs


class Decoder(nn.Module):

    def __init__(self, dims, embed_dim, num_classes):
        super().__init__()
        self.linear_c1 = MLP(dims[0], embed_dim)
        self.linear_c2 = MLP(dims[1], embed_dim)
        self.linear_c3 = MLP(dims[2], embed_dim)
        self.linear_c4 = MLP(dims[3], embed_dim)
        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim, 1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.conv_seg = nn.Conv2d(128, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs: 'Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]') ->torch.Tensor:
        c1, c2, c3, c4 = inputs
        n = c1.shape[0]
        c1f = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        c2f = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        c2f = F.interpolate(c2f, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c3f = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        c3f = F.interpolate(c3f, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c4f = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        c4f = F.interpolate(c4f, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c = self.linear_fuse(torch.cat([c4f, c3f, c2f, c1f], dim=1))
        c = self.dropout(c)
        return self.linear_pred(c)


segformer_settings = {'B0': [[32, 64, 160, 256], [2, 2, 2, 2], 256], 'B1': [[64, 128, 320, 512], [2, 2, 2, 2], 256], 'B2': [[64, 128, 320, 512], [3, 4, 6, 3], 768], 'B3': [[64, 128, 320, 512], [3, 4, 18, 3], 768], 'B4': [[64, 128, 320, 512], [3, 8, 27, 3], 768], 'B5': [[64, 128, 320, 512], [3, 6, 40, 3], 768]}


class SegFormer(nn.Module):

    def __init__(self, model_name: 'str'='B0', num_classes: 'int'=19, image_size: 'int'=224) ->None:
        super().__init__()
        assert model_name in segformer_settings.keys(), f'SegFormer model name should be in {list(segformer_settings.keys())}'
        dims, layers, embed_dim = segformer_settings[model_name]
        self.backbone = MiT(image_size, dims, layers)
        self.decode_head = Decoder(dims, embed_dim, num_classes)

    def init_weights(self, pretrained: 'str'=None) ->None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoder_outs = self.backbone(x)
        return self.decode_head(encoder_outs)


class DiceLoss(nn.Module):

    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-05
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ConvModule,
     lambda: ([], {'c1': 4, 'c2': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EfficientAttention,
     lambda: ([], {'in_channels': 4, 'key_channels': 4, 'value_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'dim': 4, 'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP_FFN,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OverlapPatchEmbeddings,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SegFormer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SelfAtten,
     lambda: ([], {'dim': 4, 'head': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

