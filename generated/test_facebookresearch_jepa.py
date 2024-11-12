
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


import copy


import time


import numpy as np


import torch


import torch.multiprocessing as mp


import torch.nn.functional as F


from torch.nn.parallel import DistributedDataParallel


import torchvision.transforms as transforms


import logging


import warnings


import torch.nn as nn


from logging import getLogger


import torchvision


import numbers


import math


import random


import torchvision.transforms.functional as F


from torchvision import transforms


from typing import Iterator


from typing import Optional


from torch.utils.data import Dataset


from torch.utils.data import Sampler


from torch.utils.data import DistributedSampler


from torch.utils.data import WeightedRandomSampler


import pandas as pd


from functools import partial


import torch.distributed as dist


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches [0,N) to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class FrameAggregation(nn.Module):
    """
    Process each frame independently and concatenate all tokens
    """

    def __init__(self, model, max_frames=10000, use_pos_embed=False, attend_across_segments=False):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.attend_across_segments = attend_across_segments
        self.pos_embed = None
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_frames)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):
        num_views_per_clip = len(x[0])
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=2)
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        outputs = self.model(x)
        _, N, D = outputs.size()
        outputs = outputs.reshape(B, T, N, D).flatten(1, 2)
        B = B // num_views_per_clip
        all_outputs = []
        for i in range(num_views_per_clip):
            o = outputs[i * B:(i + 1) * B]
            if self.pos_embed is not None and clip_indices is not None:
                pos_embed = self.pos_embed.repeat(B, 1, 1)
                pos_embed = apply_masks(pos_embed, clip_indices, concat=False)
                pos_embed = torch.cat(pos_embed, dim=1)
                pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, N, 1)
                pos_embed = pos_embed.flatten(1, 2)
                o += pos_embed
            all_outputs += [o]
        return all_outputs


class ClipAggregation(nn.Module):
    """
    Process each clip independently and concatenate all tokens
    """

    def __init__(self, model, tubelet_size=2, max_frames=10000, use_pos_embed=False, attend_across_segments=False):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.attend_across_segments = attend_across_segments
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(torch.zeros(1, max_T, embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):
        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, T, H, W = x[0][0].size()
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)
        outputs = self.model(x)
        _, N, D = outputs.size()
        T = T // self.tubelet_size
        N = N // T
        eff_B = B * num_views_per_clip
        all_outputs = [[] for _ in range(num_views_per_clip)]
        for i in range(num_clips):
            o = outputs[i * eff_B:(i + 1) * eff_B]
            for j in range(num_views_per_clip):
                all_outputs[j].append(o[j * B:(j + 1) * B])
        if not self.attend_across_segments:
            return all_outputs
        for i, outputs in enumerate(all_outputs):
            outputs = [o.reshape(B, T, N, D) for o in outputs]
            outputs = torch.cat(outputs, dim=1).flatten(1, 2)
            if self.pos_embed is not None and clip_indices is not None:
                clip_indices = [c[:, ::self.tubelet_size] for c in clip_indices]
                pos_embed = self.pos_embed.repeat(B, 1, 1)
                pos_embed = apply_masks(pos_embed, clip_indices, concat=False)
                pos_embed = torch.cat(pos_embed, dim=1)
                pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, N, 1)
                pos_embed = pos_embed.flatten(1, 2)
                outputs += pos_embed
            all_outputs[i] = outputs
        return all_outputs


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob)
                attn = None
        else:
            attn = q @ k.transpose(-2, -1) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class MLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, grid_size=None, grid_depth=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, mask=None):
        y, attn = self.attn(self.norm1(x), mask=mask)
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=12, qkv_bias=False, use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = q @ k.transpose(-2, -1) * self.scale
            xattn = xattn.softmax(dim=-1)
            q = xattn @ v
        q = q.transpose(1, 2).reshape(B, n, C)
        q = self.proj(q)
        return q


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class AttentivePooler(nn.Module):
    """ Attentive Pooler """

    def __init__(self, num_queries=1, embed_dim=768, num_heads=12, mlp_ratio=4.0, depth=1, norm_layer=nn.LayerNorm, init_std=0.02, qkv_bias=True, complete_block=True):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
        else:
            self.cross_attention_block = CrossAttention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=False, norm_layer=norm_layer) for i in range(depth - 1)])
        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)
        return q


class AttentiveClassifier(nn.Module):
    """ Attentive Classifier """

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, depth=1, norm_layer=nn.LayerNorm, init_std=0.02, qkv_bias=True, num_classes=1000, complete_block=True):
        super().__init__()
        self.pooler = AttentivePooler(num_queries=1, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, depth=depth, norm_layer=norm_layer, init_std=init_std, qkv_bias=qkv_bias, complete_block=complete_block)
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        x = self.linear(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    returns:
        pos_embed: [grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False):
    """
    grid_size: int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)
    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)
    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([torch.cat([x[i * B:(i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)], dim=0)
    return x


class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """

    def __init__(self, img_size=224, patch_size=16, num_frames=1, tubelet_size=2, embed_dim=768, predictor_embed_dim=384, depth=6, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, norm_layer=nn.LayerNorm, init_std=0.02, uniform_power=False, use_mask_tokens=False, num_mask_tokens=2, zero_init_mask_tokens=True, **kwargs):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) for i in range(num_mask_tokens)])
        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1
        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size
        if self.is_video:
            self.num_patches = num_patches = num_frames // tubelet_size * (img_size // patch_size) * (img_size // patch_size)
        else:
            self.num_patches = num_patches = img_size // patch_size * (img_size // patch_size)
        self.uniform_power = uniform_power
        self.predictor_pos_embed = None
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False)
        self.predictor_blocks = nn.ModuleList([Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, act_layer=nn.GELU, attn_drop=attn_drop_rate, grid_size=grid_size, grid_depth=grid_depth, norm_layer=norm_layer) for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        if self.predictor_pos_embed is not None:
            self._init_pos_embed(self.predictor_pos_embed.data)
        self.init_std = init_std
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=self.uniform_power)
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def diffusion(self, x, noise_beta=(0.5, 1.0), steps=1000):
        b1, b2 = noise_beta
        beta_scheduler = (b1 + i * (b2 - b1) / steps for i in range(steps))
        alpha_scheduler = []
        _alpha = 1.0
        for _beta in beta_scheduler:
            _alpha *= 1.0 - _beta
            alpha_scheduler += [_alpha]
        T = torch.randint(0, steps, (len(x),))
        alpha = torch.tensor(alpha_scheduler, device=x.device)[T].unsqueeze(-1).unsqueeze(-1)
        x = torch.nn.functional.layer_norm(x, (x.size(-1),))
        x = alpha ** 0.5 * x + (1.0 - alpha) ** 0.5 * torch.randn(x.shape, device=x.device)
        return x

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt, mask_index=1):
        """
        :param ctxt: context tokens
        :param tgt: target tokens
        :param masks_ctxt: indices of context tokens in input
        :params masks_tgt: indices of target tokens in input
        """
        assert masks_ctxt is not None and masks_tgt is not None, 'Cannot run predictor without mask indices'
        if not isinstance(masks_ctxt, list):
            masks_ctxt = [masks_ctxt]
        if not isinstance(masks_tgt, list):
            masks_tgt = [masks_tgt]
        B = len(ctxt) // len(masks_ctxt)
        x = self.predictor_embed(ctxt)
        _, N_ctxt, D = x.shape
        if self.predictor_pos_embed is not None:
            ctxt_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(ctxt_pos_embed, masks_ctxt)
        if self.mask_tokens is None:
            pred_tokens = self.predictor_embed(tgt)
            pred_tokens = self.diffusion(pred_tokens)
        else:
            mask_index = mask_index % self.num_mask_tokens
            pred_tokens = self.mask_tokens[mask_index]
            pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
            pred_tokens = apply_masks(pred_tokens, masks_tgt)
        if self.predictor_pos_embed is not None:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_tgt)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_ctxt))
            pred_tokens += pos_embs
        x = x.repeat(len(masks_tgt), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)
        masks_ctxt = torch.cat(masks_ctxt, dim=0)
        masks_tgt = torch.cat(masks_tgt, dim=0)
        masks = torch.cat([masks_ctxt, masks_tgt], dim=1)
        for blk in self.predictor_blocks:
            x = blk(x, mask=masks)
        x = self.predictor_norm(x)
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)
        return x


class MultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks=None):
        if masks is None:
            return self.backbone(x)
        if masks is not None and not isinstance(masks, list):
            masks = [masks]
        outs = []
        for m in masks:
            outs += [self.backbone(x, masks=m)]
        return outs


class PredictorMultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt):
        if type(ctxt) is not list:
            ctxt = [ctxt]
        if type(tgt) is not list:
            tgt = [tgt]
        if type(masks_ctxt) is not list:
            masks_ctxt = [masks_ctxt]
        if type(masks_tgt) is not list:
            masks_tgt = [masks_tgt]
        outs = []
        for i, (zi, hi, mc, mt) in enumerate(zip(ctxt, tgt, masks_ctxt, masks_tgt)):
            outs += [self.backbone(zi, hi, mc, mt, mask_index=i)]
        return outs


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, kernel_size=(tubelet_size, patch_size, patch_size), stride=(tubelet_size, patch_size, patch_size))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """

    def __init__(self, img_size=224, patch_size=16, num_frames=1, tubelet_size=2, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, norm_layer=nn.LayerNorm, init_std=0.02, out_layers=None, uniform_power=False, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1
        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size
        if self.is_video:
            self.patch_embed = PatchEmbed3D(patch_size=patch_size, tubelet_size=tubelet_size, in_chans=in_chans, embed_dim=embed_dim)
            self.num_patches = num_frames // tubelet_size * (img_size // patch_size) * (img_size // patch_size)
        else:
            self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.num_patches = img_size // patch_size * (img_size // patch_size)
        self.uniform_power = uniform_power
        self.pos_embed = None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, act_layer=nn.GELU, grid_size=grid_size, grid_depth=grid_depth, attn_drop=attn_drop_rate, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=self.uniform_power)
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def forward(self, x, masks=None):
        """
        :param x: input image/video
        :param masks: indices of patch tokens to mask (remove)
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]
        pos_embed = self.pos_embed
        if pos_embed is not None:
            pos_embed = self.interpolate_pos_encoding(x, pos_embed)
        x = self.patch_embed(x)
        if pos_embed is not None:
            x += pos_embed
        B, N, D = x.shape
        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask=masks)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))
        if self.out_layers is not None:
            return outs
        if self.norm is not None:
            x = self.norm(x)
        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        _, N, dim = pos_embed.shape
        if self.is_video:
            _, _, T, H, W = x.shape
            if H == self.input_size and W == self.input_size and T == self.num_frames:
                return pos_embed
            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size
            N_t = self.num_frames // self.tubelet_size
            N_h = N_w = self.input_size // self.patch_size
            assert N_h * N_w * N_t == N, 'Positional embedding initialized incorrectly'
            scale_factor = T / N_t, H / N_h, W / N_w
            pos_embed = nn.functional.interpolate(pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3), scale_factor=scale_factor, mode='trilinear')
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed
        else:
            _, _, H, W = x.shape
            if H == self.input_size and W == self.input_size:
                return pos_embed
            npatch = H // self.patch_size * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)
            pos_embed = nn.functional.interpolate(pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), scale_factor=scale_factor, mode='bicubic')
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (MLP,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiMaskWrapper,
     lambda: ([], {'backbone': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (PatchEmbed3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64, 64])], {})),
]

