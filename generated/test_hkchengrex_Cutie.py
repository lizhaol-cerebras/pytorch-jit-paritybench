
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


import logging


import numpy as np


import torch


from torch.utils.data import DataLoader


from torch.utils.data.dataset import Dataset


from torchvision import transforms


from torchvision.transforms import InterpolationMode


from typing import Dict


from typing import List


from typing import Tuple


from typing import Optional


import copy


import warnings


from typing import Iterable


import torch.nn.functional as F


from typing import Literal


from collections import defaultdict


from typing import Union


from queue import Queue


import torch.nn as nn


import math


import torch.optim as optim


from torch import nn


from typing import Callable


from torch import Tensor


from collections import OrderedDict


from torch.utils import model_zoo


import time


import torch.distributed as distributed


from torch.utils.tensorboard import SummaryWriter


from torch.nn import functional as F


import functools


from torch import autocast


from torchvision.transforms.functional import to_tensor


from torch.utils.data import Dataset


from torchvision.transforms import ToTensor


from time import time


from scipy.optimize import fmin_l_bfgs_b


import torch._utils


from torch import nn as nn


from torch import distributed as dist


from torch.utils import data


from functools import partial


from functools import wraps


from copy import deepcopy


import inspect


from time import perf_counter


class GConv2d(nn.Conv2d):

    def forward(self, g: 'torch.Tensor') ->torch.Tensor:
        batch_size, num_objects = g.shape[:2]
        g = super().forward(g.flatten(start_dim=0, end_dim=1))
        return g.view(batch_size, num_objects, *g.shape[1:])


class LinearPredictor(nn.Module):

    def __init__(self, x_dim: 'int', pix_dim: 'int'):
        super().__init__()
        self.projection = GConv2d(x_dim, pix_dim + 1, kernel_size=1)

    def forward(self, pix_feat: 'torch.Tensor', x: 'torch.Tensor') ->torch.Tensor:
        num_objects = x.shape[1]
        x = self.projection(x)
        pix_feat = pix_feat.unsqueeze(1).expand(-1, num_objects, -1, -1, -1)
        logits = (pix_feat * x[:, :, :-1]).sum(dim=2) + x[:, :, -1]
        return logits


class DirectPredictor(nn.Module):

    def __init__(self, x_dim: 'int'):
        super().__init__()
        self.projection = GConv2d(x_dim, 1, kernel_size=1)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        logits = self.projection(x).squeeze(2)
        return logits


def aggregate(prob: 'torch.Tensor', dim: 'int') ->torch.Tensor:
    with torch.amp.autocast(enabled=False):
        prob = prob.float()
        new_prob = torch.cat([torch.prod(1 - prob, dim=dim, keepdim=True), prob], dim).clamp(1e-07, 1 - 1e-07)
        logits = torch.log(new_prob / (1 - new_prob))
        return logits


class AuxComputer(nn.Module):

    def __init__(self, cfg: 'DictConfig'):
        super().__init__()
        use_sensory_aux = cfg.model.aux_loss.sensory.enabled
        self.use_query_aux = cfg.model.aux_loss.query.enabled
        sensory_dim = cfg.model.sensory_dim
        embed_dim = cfg.model.embed_dim
        if use_sensory_aux:
            self.sensory_aux = LinearPredictor(sensory_dim, embed_dim)
        else:
            self.sensory_aux = None

    def _aggregate_with_selector(self, logits: 'torch.Tensor', selector: 'torch.Tensor') ->torch.Tensor:
        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector
        logits = aggregate(prob, dim=1)
        return logits

    def forward(self, pix_feat: 'torch.Tensor', aux_input: 'Dict[str, torch.Tensor]', selector: 'torch.Tensor') ->Dict[str, torch.Tensor]:
        sensory = aux_input['sensory']
        q_logits = aux_input['q_logits']
        aux_output = {}
        aux_output['attn_mask'] = aux_input['attn_mask']
        if self.sensory_aux is not None:
            logits = self.sensory_aux(pix_feat, sensory)
            aux_output['sensory_logits'] = self._aggregate_with_selector(logits, selector)
        if self.use_query_aux and q_logits is not None:
            aux_output['q_logits'] = self._aggregate_with_selector(torch.stack(q_logits, dim=2), selector.unsqueeze(2) if selector is not None else None)
        return aux_output


class PixelEncoder(nn.Module):

    def __init__(self, model_cfg: 'DictConfig'):
        super().__init__()
        self.is_resnet = 'resnet' in model_cfg.pixel_encoder.type
        resnet_model_path = model_cfg.get('resnet_model_path')
        if self.is_resnet:
            if model_cfg.pixel_encoder.type == 'resnet18':
                network = resnet.resnet18(pretrained=True, model_dir=resnet_model_path)
            elif model_cfg.pixel_encoder.type == 'resnet50':
                network = resnet.resnet50(pretrained=True, model_dir=resnet_model_path)
            else:
                raise NotImplementedError
            self.conv1 = network.conv1
            self.bn1 = network.bn1
            self.relu = network.relu
            self.maxpool = network.maxpool
            self.res2 = network.layer1
            self.layer2 = network.layer2
            self.layer3 = network.layer3
        else:
            raise NotImplementedError

    def forward(self, x: 'torch.Tensor') ->(torch.Tensor, torch.Tensor, torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f4 = self.res2(x)
        f8 = self.layer2(f4)
        f16 = self.layer3(f8)
        return f16, f8, f4

    def train(self, mode=True):
        self.training = False
        for module in self.children():
            module.train(False)
        return self


class KeyProjection(nn.Module):

    def __init__(self, model_cfg: 'DictConfig'):
        super().__init__()
        in_dim = model_cfg.pixel_encoder.ms_dims[0]
        mid_dim = model_cfg.pixel_dim
        key_dim = model_cfg.key_dim
        self.pix_feat_proj = nn.Conv2d(in_dim, mid_dim, kernel_size=1)
        self.key_proj = nn.Conv2d(mid_dim, key_dim, kernel_size=3, padding=1)
        self.d_proj = nn.Conv2d(mid_dim, 1, kernel_size=3, padding=1)
        self.e_proj = nn.Conv2d(mid_dim, key_dim, kernel_size=3, padding=1)
        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x: 'torch.Tensor', *, need_s: bool, need_e: bool) ->(torch.Tensor, torch.Tensor, torch.Tensor):
        x = self.pix_feat_proj(x)
        shrinkage = self.d_proj(x) ** 2 + 1 if need_s else None
        selection = torch.sigmoid(self.e_proj(x)) if need_e else None
        return self.key_proj(x), shrinkage, selection


class CAResBlock(nn.Module):

    def __init__(self, in_dim: 'int', out_dim: 'int', residual: 'bool'=True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        t = int((abs(math.log2(out_dim)) + 1) // 2)
        k = t if t % 2 else t + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        if self.residual:
            if in_dim == out_dim:
                self.downsample = nn.Identity()
            else:
                self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        r = x
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        b, c = x.shape[:2]
        w = self.pool(x).view(b, 1, c)
        w = self.conv(w).transpose(-1, -2).unsqueeze(-1).sigmoid()
        if self.residual:
            x = x * w + self.downsample(r)
        else:
            x = x * w
        return x


class MainToGroupDistributor(nn.Module):

    def __init__(self, x_transform: 'Optional[nn.Module]'=None, g_transform: 'Optional[nn.Module]'=None, method: 'str'='cat', reverse_order: 'bool'=False):
        super().__init__()
        self.x_transform = x_transform
        self.g_transform = g_transform
        self.method = method
        self.reverse_order = reverse_order

    def forward(self, x: 'torch.Tensor', g: 'torch.Tensor', skip_expand: 'bool'=False) ->torch.Tensor:
        num_objects = g.shape[1]
        if self.x_transform is not None:
            x = self.x_transform(x)
        if self.g_transform is not None:
            g = self.g_transform(g)
        if not skip_expand:
            x = x.unsqueeze(1).expand(-1, num_objects, -1, -1, -1)
        if self.method == 'cat':
            if self.reverse_order:
                g = torch.cat([g, x], 2)
            else:
                g = torch.cat([x, g], 2)
        elif self.method == 'add':
            g = x + g
        elif self.method == 'mulcat':
            g = torch.cat([x * g, g], dim=2)
        elif self.method == 'muladd':
            g = x * g + g
        else:
            raise NotImplementedError
        return g


class GroupFeatureFusionBlock(nn.Module):

    def __init__(self, x_in_dim: 'int', g_in_dim: 'int', out_dim: 'int'):
        super().__init__()
        x_transform = nn.Conv2d(x_in_dim, out_dim, kernel_size=1)
        g_transform = GConv2d(g_in_dim, out_dim, kernel_size=1)
        self.distributor = MainToGroupDistributor(x_transform=x_transform, g_transform=g_transform, method='add')
        self.block1 = CAResBlock(out_dim, out_dim)
        self.block2 = CAResBlock(out_dim, out_dim)

    def forward(self, x: 'torch.Tensor', g: 'torch.Tensor') ->torch.Tensor:
        batch_size, num_objects = g.shape[:2]
        g = self.distributor(x, g)
        g = g.flatten(start_dim=0, end_dim=1)
        g = self.block1(g)
        g = self.block2(g)
        g = g.view(batch_size, num_objects, *g.shape[1:])
        return g


def _recurrent_update(h: 'torch.Tensor', values: 'torch.Tensor') ->torch.Tensor:
    dim = values.shape[2] // 3
    forget_gate = torch.sigmoid(values[:, :, :dim])
    update_gate = torch.sigmoid(values[:, :, dim:dim * 2])
    new_value = torch.tanh(values[:, :, dim * 2:])
    new_h = forget_gate * h * (1 - update_gate) + update_gate * new_value
    return new_h


class SensoryDeepUpdater(nn.Module):

    def __init__(self, f_dim: 'int', sensory_dim: 'int'):
        super().__init__()
        self.transform = GConv2d(f_dim + sensory_dim, sensory_dim * 3, kernel_size=3, padding=1)
        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g: 'torch.Tensor', h: 'torch.Tensor') ->torch.Tensor:
        with torch.amp.autocast(enabled=False):
            g = g.float()
            h = h.float()
            values = self.transform(torch.cat([g, h], dim=2))
            new_h = _recurrent_update(h, values)
        return new_h


class MaskEncoder(nn.Module):

    def __init__(self, model_cfg: 'DictConfig', single_object=False):
        super().__init__()
        pixel_dim = model_cfg.pixel_dim
        value_dim = model_cfg.value_dim
        sensory_dim = model_cfg.sensory_dim
        final_dim = model_cfg.mask_encoder.final_dim
        self.single_object = single_object
        extra_dim = 1 if single_object else 2
        resnet_model_path = model_cfg.get('resnet_model_path')
        if model_cfg.mask_encoder.type == 'resnet18':
            network = resnet.resnet18(pretrained=True, extra_dim=extra_dim, model_dir=resnet_model_path)
        elif model_cfg.mask_encoder.type == 'resnet50':
            network = resnet.resnet50(pretrained=True, extra_dim=extra_dim, model_dir=resnet_model_path)
        else:
            raise NotImplementedError
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.distributor = MainToGroupDistributor()
        self.fuser = GroupFeatureFusionBlock(pixel_dim, final_dim, value_dim)
        self.sensory_update = SensoryDeepUpdater(value_dim, sensory_dim)

    def forward(self, image: 'torch.Tensor', pix_feat: 'torch.Tensor', sensory: 'torch.Tensor', masks: 'torch.Tensor', others: 'torch.Tensor', *, deep_update: bool=True, chunk_size: int=-1) ->(torch.Tensor, torch.Tensor):
        if self.single_object:
            g = masks.unsqueeze(2)
        else:
            g = torch.stack([masks, others], dim=2)
        g = self.distributor(image, g)
        batch_size, num_objects = g.shape[:2]
        if chunk_size < 1 or chunk_size >= num_objects:
            chunk_size = num_objects
            fast_path = True
            new_sensory = sensory
        else:
            if deep_update:
                new_sensory = torch.empty_like(sensory)
            else:
                new_sensory = sensory
            fast_path = False
        all_g = []
        for i in range(0, num_objects, chunk_size):
            if fast_path:
                g_chunk = g
            else:
                g_chunk = g[:, i:i + chunk_size]
            actual_chunk_size = g_chunk.shape[1]
            g_chunk = g_chunk.flatten(start_dim=0, end_dim=1)
            g_chunk = self.conv1(g_chunk)
            g_chunk = self.bn1(g_chunk)
            g_chunk = self.maxpool(g_chunk)
            g_chunk = self.relu(g_chunk)
            g_chunk = self.layer1(g_chunk)
            g_chunk = self.layer2(g_chunk)
            g_chunk = self.layer3(g_chunk)
            g_chunk = g_chunk.view(batch_size, actual_chunk_size, *g_chunk.shape[1:])
            g_chunk = self.fuser(pix_feat, g_chunk)
            all_g.append(g_chunk)
            if deep_update:
                if fast_path:
                    new_sensory = self.sensory_update(g_chunk, sensory)
                else:
                    new_sensory[:, i:i + chunk_size] = self.sensory_update(g_chunk, sensory[:, i:i + chunk_size])
        g = torch.cat(all_g, dim=1)
        return g, new_sensory

    def train(self, mode=True):
        self.training = False
        for module in self.children():
            module.train(False)
        return self


class PixelFeatureFuser(nn.Module):

    def __init__(self, model_cfg: 'DictConfig', single_object=False):
        super().__init__()
        value_dim = model_cfg.value_dim
        sensory_dim = model_cfg.sensory_dim
        pixel_dim = model_cfg.pixel_dim
        embed_dim = model_cfg.embed_dim
        self.single_object = single_object
        self.fuser = GroupFeatureFusionBlock(pixel_dim, value_dim, embed_dim)
        if self.single_object:
            self.sensory_compress = GConv2d(sensory_dim + 1, value_dim, kernel_size=1)
        else:
            self.sensory_compress = GConv2d(sensory_dim + 2, value_dim, kernel_size=1)

    def forward(self, pix_feat: 'torch.Tensor', pixel_memory: 'torch.Tensor', sensory_memory: 'torch.Tensor', last_mask: 'torch.Tensor', last_others: 'torch.Tensor', *, chunk_size: int=-1) ->torch.Tensor:
        batch_size, num_objects = pixel_memory.shape[:2]
        if self.single_object:
            last_mask = last_mask.unsqueeze(2)
        else:
            last_mask = torch.stack([last_mask, last_others], dim=2)
        if chunk_size < 1:
            chunk_size = num_objects
        all_p16 = []
        for i in range(0, num_objects, chunk_size):
            sensory_readout = self.sensory_compress(torch.cat([sensory_memory[:, i:i + chunk_size], last_mask[:, i:i + chunk_size]], 2))
            p16 = pixel_memory[:, i:i + chunk_size] + sensory_readout
            p16 = self.fuser(pix_feat, p16)
            all_p16.append(p16)
        p16 = torch.cat(all_p16, dim=1)
        return p16


class DecoderFeatureProcessor(nn.Module):

    def __init__(self, decoder_dims: 'List[int]', out_dims: 'List[int]'):
        super().__init__()
        self.transforms = nn.ModuleList([nn.Conv2d(d_dim, p_dim, kernel_size=1) for d_dim, p_dim in zip(decoder_dims, out_dims)])

    def forward(self, multi_scale_features: 'Iterable[torch.Tensor]') ->List[torch.Tensor]:
        outputs = [func(x) for x, func in zip(multi_scale_features, self.transforms)]
        return outputs


class GroupResBlock(nn.Module):

    def __init__(self, in_dim: 'int', out_dim: 'int'):
        super().__init__()
        if in_dim == out_dim:
            self.downsample = nn.Identity()
        else:
            self.downsample = GConv2d(in_dim, out_dim, kernel_size=1)
        self.conv1 = GConv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = GConv2d(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, g: 'torch.Tensor') ->torch.Tensor:
        out_g = self.conv1(F.relu(g))
        out_g = self.conv2(F.relu(out_g))
        g = self.downsample(g)
        return out_g + g


def interpolate_groups(g: 'torch.Tensor', ratio: 'float', mode: 'str', align_corners: 'bool') ->torch.Tensor:
    batch_size, num_objects = g.shape[:2]
    g = F.interpolate(g.flatten(start_dim=0, end_dim=1), scale_factor=ratio, mode=mode, align_corners=align_corners)
    g = g.view(batch_size, num_objects, *g.shape[1:])
    return g


def upsample_groups(g: 'torch.Tensor', ratio: 'float'=2, mode: 'str'='bilinear', align_corners: 'bool'=False) ->torch.Tensor:
    return interpolate_groups(g, ratio, mode, align_corners)


class MaskUpsampleBlock(nn.Module):

    def __init__(self, in_dim: 'int', out_dim: 'int', scale_factor: 'int'=2):
        super().__init__()
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(in_dim, out_dim)
        self.scale_factor = scale_factor

    def forward(self, in_g: 'torch.Tensor', skip_f: 'torch.Tensor') ->torch.Tensor:
        g = upsample_groups(in_g, ratio=self.scale_factor)
        g = self.distributor(skip_f, g)
        g = self.out_conv(g)
        return g


def downsample_groups(g: 'torch.Tensor', ratio: 'float'=1 / 2, mode: 'str'='area', align_corners: 'bool'=None) ->torch.Tensor:
    return interpolate_groups(g, ratio, mode, align_corners)


class SensoryUpdater(nn.Module):

    def __init__(self, g_dims: 'List[int]', mid_dim: 'int', sensory_dim: 'int'):
        super().__init__()
        self.g16_conv = GConv2d(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = GConv2d(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = GConv2d(g_dims[2], mid_dim, kernel_size=1)
        self.transform = GConv2d(mid_dim + sensory_dim, sensory_dim * 3, kernel_size=3, padding=1)
        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g: 'torch.Tensor', h: 'torch.Tensor') ->torch.Tensor:
        g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1 / 2)) + self.g4_conv(downsample_groups(g[2], ratio=1 / 4))
        with torch.amp.autocast(enabled=False):
            g = g.float()
            h = h.float()
            values = self.transform(torch.cat([g, h], dim=2))
            new_h = _recurrent_update(h, values)
        return new_h


class MaskDecoder(nn.Module):

    def __init__(self, model_cfg: 'DictConfig'):
        super().__init__()
        embed_dim = model_cfg.embed_dim
        sensory_dim = model_cfg.sensory_dim
        ms_image_dims = model_cfg.pixel_encoder.ms_dims
        up_dims = model_cfg.mask_decoder.up_dims
        assert embed_dim == up_dims[0]
        self.sensory_update = SensoryUpdater([up_dims[0], up_dims[1], up_dims[2] + 1], sensory_dim, sensory_dim)
        self.decoder_feat_proc = DecoderFeatureProcessor(ms_image_dims[1:], up_dims[:-1])
        self.up_16_8 = MaskUpsampleBlock(up_dims[0], up_dims[1])
        self.up_8_4 = MaskUpsampleBlock(up_dims[1], up_dims[2])
        self.pred = nn.Conv2d(up_dims[-1], 1, kernel_size=3, padding=1)

    def forward(self, ms_image_feat: 'Iterable[torch.Tensor]', memory_readout: 'torch.Tensor', sensory: 'torch.Tensor', *, chunk_size: int=-1, update_sensory: bool=True) ->(torch.Tensor, torch.Tensor):
        batch_size, num_objects = memory_readout.shape[:2]
        f8, f4 = self.decoder_feat_proc(ms_image_feat[1:])
        if chunk_size < 1 or chunk_size >= num_objects:
            chunk_size = num_objects
            fast_path = True
            new_sensory = sensory
        else:
            if update_sensory:
                new_sensory = torch.empty_like(sensory)
            else:
                new_sensory = sensory
            fast_path = False
        all_logits = []
        for i in range(0, num_objects, chunk_size):
            if fast_path:
                p16 = memory_readout
            else:
                p16 = memory_readout[:, i:i + chunk_size]
            actual_chunk_size = p16.shape[1]
            p8 = self.up_16_8(p16, f8)
            p4 = self.up_8_4(p8, f4)
            with torch.amp.autocast(enabled=False):
                logits = self.pred(F.relu(p4.flatten(start_dim=0, end_dim=1).float()))
            if update_sensory:
                p4 = torch.cat([p4, logits.view(batch_size, actual_chunk_size, 1, *logits.shape[-2:])], 2)
                if fast_path:
                    new_sensory = self.sensory_update([p16, p8, p4], sensory)
                else:
                    new_sensory[:, i:i + chunk_size] = self.sensory_update([p16, p8, p4], sensory[:, i:i + chunk_size])
            all_logits.append(logits)
        logits = torch.cat(all_logits, dim=0)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])
        return new_sensory, logits


def get_emb(sin_inp: 'torch.Tensor') ->torch.Tensor:
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding(nn.Module):

    def __init__(self, dim: 'int', scale: 'float'=math.pi * 2, temperature: 'float'=10000, normalize: 'bool'=True, channel_last: 'bool'=True, transpose_output: 'bool'=False):
        super().__init__()
        dim = int(np.ceil(dim / 4) * 2)
        self.dim = dim
        inv_freq = 1.0 / temperature ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)
        self.normalize = normalize
        self.scale = scale
        self.eps = 1e-06
        self.channel_last = channel_last
        self.transpose_output = transpose_output
        self.cached_penc = None

    def forward(self, tensor: 'torch.Tensor') ->torch.Tensor:
        """
        :param tensor: A 4/5d tensor of size 
            channel_last=True: (batch_size, h, w, c) or (batch_size, k, h, w, c)
            channel_last=False: (batch_size, c, h, w) or (batch_size, k, c, h, w)
        :return: positional encoding tensor that has the same shape as the input if the input is 4d
                 if the input is 5d, the output is broadcastable along the k-dimension
        """
        if len(tensor.shape) != 4 and len(tensor.shape) != 5:
            raise RuntimeError(f'The input tensor has to be 4/5d, got {tensor.shape}!')
        if len(tensor.shape) == 5:
            num_objects = tensor.shape[1]
            tensor = tensor[:, 0]
        else:
            num_objects = None
        if self.channel_last:
            batch_size, h, w, c = tensor.shape
        else:
            batch_size, c, h, w = tensor.shape
        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            if num_objects is None:
                return self.cached_penc
            else:
                return self.cached_penc.unsqueeze(1)
        self.cached_penc = None
        pos_y = torch.arange(h, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_x = torch.arange(w, device=tensor.device, dtype=self.inv_freq.dtype)
        if self.normalize:
            pos_y = pos_y / (pos_y[-1] + self.eps) * self.scale
            pos_x = pos_x / (pos_x[-1] + self.eps) * self.scale
        sin_inp_y = torch.einsum('i,j->ij', pos_y, self.inv_freq)
        sin_inp_x = torch.einsum('i,j->ij', pos_x, self.inv_freq)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((h, w, self.dim * 2), device=tensor.device, dtype=tensor.dtype)
        emb[:, :, :self.dim] = emb_x
        emb[:, :, self.dim:] = emb_y
        if not self.channel_last and self.transpose_output:
            pass
        elif not self.channel_last or self.transpose_output:
            emb = emb.permute(2, 0, 1)
        self.cached_penc = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        if num_objects is None:
            return self.cached_penc
        else:
            return self.cached_penc.unsqueeze(1)


def _weighted_pooling(masks: 'torch.Tensor', value: 'torch.Tensor', logits: 'torch.Tensor') ->(torch.Tensor, torch.Tensor):
    weights = logits.sigmoid() * masks
    sums = torch.einsum('bkhwq,bkhwc->bkqc', weights, value)
    area = weights.flatten(start_dim=2, end_dim=3).sum(2).unsqueeze(-1)
    return sums, area


class ObjectSummarizer(nn.Module):

    def __init__(self, model_cfg: 'DictConfig'):
        super().__init__()
        this_cfg = model_cfg.object_summarizer
        self.value_dim = model_cfg.value_dim
        self.embed_dim = this_cfg.embed_dim
        self.num_summaries = this_cfg.num_summaries
        self.add_pe = this_cfg.add_pe
        self.pixel_pe_scale = model_cfg.pixel_pe_scale
        self.pixel_pe_temperature = model_cfg.pixel_pe_temperature
        if self.add_pe:
            self.pos_enc = PositionalEncoding(self.embed_dim, scale=self.pixel_pe_scale, temperature=self.pixel_pe_temperature)
        self.input_proj = nn.Linear(self.value_dim, self.embed_dim)
        self.feature_pred = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU(inplace=True), nn.Linear(self.embed_dim, self.embed_dim))
        self.weights_pred = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU(inplace=True), nn.Linear(self.embed_dim, self.num_summaries))

    def forward(self, masks: 'torch.Tensor', value: 'torch.Tensor', need_weights: 'bool'=False) ->(torch.Tensor, Optional[torch.Tensor]):
        h, w = value.shape[-2:]
        masks = F.interpolate(masks, size=(h, w), mode='area')
        masks = masks.unsqueeze(-1)
        inv_masks = 1 - masks
        repeated_masks = torch.cat([masks.expand(-1, -1, -1, -1, self.num_summaries // 2), inv_masks.expand(-1, -1, -1, -1, self.num_summaries // 2)], dim=-1)
        value = value.permute(0, 1, 3, 4, 2)
        value = self.input_proj(value)
        if self.add_pe:
            pe = self.pos_enc(value)
            value = value + pe
        with torch.amp.autocast(enabled=False):
            value = value.float()
            feature = self.feature_pred(value)
            logits = self.weights_pred(value)
            sums, area = _weighted_pooling(repeated_masks, feature, logits)
        summaries = torch.cat([sums, area], dim=-1)
        if need_weights:
            return summaries, logits
        else:
            return summaries, None


class CrossAttention(nn.Module):

    def __init__(self, dim: 'int', nhead: 'int', dropout: 'float'=0.0, batch_first: 'bool'=True, add_pe_to_qkv: 'List[bool]'=[True, True, False], residual: 'bool'=True, norm: 'bool'=True):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=batch_first)
        if norm:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.add_pe_to_qkv = add_pe_to_qkv
        self.residual = residual

    def forward(self, x: 'torch.Tensor', mem: 'torch.Tensor', x_pe: 'torch.Tensor', mem_pe: 'torch.Tensor', attn_mask: 'bool'=None, *, need_weights: bool=False) ->(torch.Tensor, torch.Tensor):
        x = self.norm(x)
        if self.add_pe_to_qkv[0]:
            q = x + x_pe
        else:
            q = x
        if any(self.add_pe_to_qkv[1:]):
            mem_with_pe = mem + mem_pe
            k = mem_with_pe if self.add_pe_to_qkv[1] else mem
            v = mem_with_pe if self.add_pe_to_qkv[2] else mem
        else:
            k = v = mem
        r = x
        x, weights = self.cross_attn(q, k, v, attn_mask=attn_mask, need_weights=need_weights, average_attn_weights=False)
        if self.residual:
            return r + self.dropout(x), weights
        else:
            return self.dropout(x), weights


def _get_activation_fn(activation: 'str') ->Callable[[Tensor], Tensor]:
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    raise RuntimeError('activation should be relu/gelu, not {}'.format(activation))


class FFN(nn.Module):

    def __init__(self, dim_in: 'int', dim_ff: 'int', activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_ff)
        self.linear2 = nn.Linear(dim_ff, dim_in)
        self.norm = nn.LayerNorm(dim_in)
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        r = x
        x = self.norm(x)
        x = self.linear2(self.activation(self.linear1(x)))
        x = r + x
        return x


class PixelFFN(nn.Module):

    def __init__(self, dim: 'int'):
        super().__init__()
        self.dim = dim
        self.conv = CAResBlock(dim, dim)

    def forward(self, pixel: 'torch.Tensor', pixel_flat: 'torch.Tensor') ->torch.Tensor:
        bs, num_objects, _, h, w = pixel.shape
        pixel_flat = pixel_flat.view(bs * num_objects, h, w, self.dim)
        pixel_flat = pixel_flat.permute(0, 3, 1, 2).contiguous()
        x = self.conv(pixel_flat)
        x = x.view(bs, num_objects, self.dim, h, w)
        return x


class SelfAttention(nn.Module):

    def __init__(self, dim: 'int', nhead: 'int', dropout: 'float'=0.0, batch_first: 'bool'=True, add_pe_to_qkv: 'List[bool]'=[True, True, False]):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=batch_first)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.add_pe_to_qkv = add_pe_to_qkv

    def forward(self, x: 'torch.Tensor', pe: 'torch.Tensor', attn_mask: 'bool'=None, key_padding_mask: 'bool'=None) ->torch.Tensor:
        x = self.norm(x)
        if any(self.add_pe_to_qkv):
            x_with_pe = x + pe
            q = x_with_pe if self.add_pe_to_qkv[0] else x
            k = x_with_pe if self.add_pe_to_qkv[1] else x
            v = x_with_pe if self.add_pe_to_qkv[2] else x
        else:
            q = k = v = x
        r = x
        x = self.self_attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        return r + self.dropout(x)


class QueryTransformerBlock(nn.Module):

    def __init__(self, model_cfg: 'DictConfig'):
        super().__init__()
        this_cfg = model_cfg.object_transformer
        self.embed_dim = this_cfg.embed_dim
        self.num_heads = this_cfg.num_heads
        self.num_queries = this_cfg.num_queries
        self.ff_dim = this_cfg.ff_dim
        self.read_from_pixel = CrossAttention(self.embed_dim, self.num_heads, add_pe_to_qkv=this_cfg.read_from_pixel.add_pe_to_qkv)
        self.self_attn = SelfAttention(self.embed_dim, self.num_heads, add_pe_to_qkv=this_cfg.query_self_attention.add_pe_to_qkv)
        self.ffn = FFN(self.embed_dim, self.ff_dim)
        self.read_from_query = CrossAttention(self.embed_dim, self.num_heads, add_pe_to_qkv=this_cfg.read_from_query.add_pe_to_qkv, norm=this_cfg.read_from_query.output_norm)
        self.pixel_ffn = PixelFFN(self.embed_dim)

    def forward(self, x: 'torch.Tensor', pixel: 'torch.Tensor', query_pe: 'torch.Tensor', pixel_pe: 'torch.Tensor', attn_mask: 'torch.Tensor', need_weights: 'bool'=False) ->(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        pixel_flat = pixel.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()
        x, q_weights = self.read_from_pixel(x, pixel_flat, query_pe, pixel_pe, attn_mask=attn_mask, need_weights=need_weights)
        x = self.self_attn(x, query_pe)
        x = self.ffn(x)
        pixel_flat, p_weights = self.read_from_query(pixel_flat, x, pixel_pe, query_pe, need_weights=need_weights)
        pixel = self.pixel_ffn(pixel, pixel_flat)
        if need_weights:
            bs, num_objects, _, h, w = pixel.shape
            q_weights = q_weights.view(bs, num_objects, self.num_heads, self.num_queries, h, w)
            p_weights = p_weights.transpose(2, 3).view(bs, num_objects, self.num_heads, self.num_queries, h, w)
        return x, pixel, q_weights, p_weights


class QueryTransformer(nn.Module):

    def __init__(self, model_cfg: 'DictConfig'):
        super().__init__()
        this_cfg = model_cfg.object_transformer
        self.value_dim = model_cfg.value_dim
        self.embed_dim = this_cfg.embed_dim
        self.num_heads = this_cfg.num_heads
        self.num_queries = this_cfg.num_queries
        self.query_init = nn.Embedding(self.num_queries, self.embed_dim)
        self.query_emb = nn.Embedding(self.num_queries, self.embed_dim)
        self.summary_to_query_init = nn.Linear(self.embed_dim, self.embed_dim)
        self.summary_to_query_emb = nn.Linear(self.embed_dim, self.embed_dim)
        self.pixel_pe_scale = model_cfg.pixel_pe_scale
        self.pixel_pe_temperature = model_cfg.pixel_pe_temperature
        self.pixel_init_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.pixel_emb_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.spatial_pe = PositionalEncoding(self.embed_dim, scale=self.pixel_pe_scale, temperature=self.pixel_pe_temperature, channel_last=False, transpose_output=True)
        self.num_blocks = this_cfg.num_blocks
        self.blocks = nn.ModuleList(QueryTransformerBlock(model_cfg) for _ in range(self.num_blocks))
        self.mask_pred = nn.ModuleList(nn.Sequential(nn.ReLU(), GConv2d(self.embed_dim, 1, kernel_size=1)) for _ in range(self.num_blocks + 1))
        self.act = nn.ReLU(inplace=True)

    def forward(self, pixel: 'torch.Tensor', obj_summaries: 'torch.Tensor', selector: 'Optional[torch.Tensor]'=None, need_weights: 'bool'=False) ->(torch.Tensor, Dict[str, torch.Tensor]):
        T = obj_summaries.shape[2]
        bs, num_objects, _, H, W = pixel.shape
        obj_summaries = obj_summaries.view(bs * num_objects, T, self.num_queries, self.embed_dim + 1)
        obj_sums = obj_summaries[:, :, :, :-1].sum(dim=1)
        obj_area = obj_summaries[:, :, :, -1:].sum(dim=1)
        obj_values = obj_sums / (obj_area + 0.0001)
        obj_init = self.summary_to_query_init(obj_values)
        obj_emb = self.summary_to_query_emb(obj_values)
        query = self.query_init.weight.unsqueeze(0).expand(bs * num_objects, -1, -1) + obj_init
        query_emb = self.query_emb.weight.unsqueeze(0).expand(bs * num_objects, -1, -1) + obj_emb
        pixel_init = self.pixel_init_proj(pixel)
        pixel_emb = self.pixel_emb_proj(pixel)
        pixel_pe = self.spatial_pe(pixel.flatten(0, 1))
        pixel_emb = pixel_emb.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()
        pixel_pe = pixel_pe.flatten(1, 2) + pixel_emb
        pixel = pixel_init
        aux_features = {'logits': []}
        aux_logits = self.mask_pred[0](pixel).squeeze(2)
        attn_mask = self._get_aux_mask(aux_logits, selector)
        aux_features['logits'].append(aux_logits)
        for i in range(self.num_blocks):
            query, pixel, q_weights, p_weights = self.blocks[i](query, pixel, query_emb, pixel_pe, attn_mask, need_weights=need_weights)
            if self.training or i <= self.num_blocks - 1 or need_weights:
                aux_logits = self.mask_pred[i + 1](pixel).squeeze(2)
                attn_mask = self._get_aux_mask(aux_logits, selector)
                aux_features['logits'].append(aux_logits)
        aux_features['q_weights'] = q_weights
        aux_features['p_weights'] = p_weights
        if self.training:
            aux_features['attn_mask'] = attn_mask.view(bs, num_objects, self.num_heads, self.num_queries, H, W)[:, :, 0]
        return pixel, aux_features

    def _get_aux_mask(self, logits: 'torch.Tensor', selector: 'torch.Tensor') ->torch.Tensor:
        if selector is None:
            prob = logits.sigmoid()
        else:
            prob = logits.sigmoid() * selector
        logits = aggregate(prob, dim=1)
        is_foreground = logits[:, 1:] >= logits.max(dim=1, keepdim=True)[0]
        foreground_mask = is_foreground.bool().flatten(start_dim=2)
        inv_foreground_mask = ~foreground_mask
        inv_background_mask = foreground_mask
        aux_foreground_mask = inv_foreground_mask.unsqueeze(2).unsqueeze(2).repeat(1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)
        aux_background_mask = inv_background_mask.unsqueeze(2).unsqueeze(2).repeat(1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)
        aux_mask = torch.cat([aux_foreground_mask, aux_background_mask], dim=1)
        aux_mask[torch.where(aux_mask.sum(-1) == aux_mask.shape[-1])] = False
        return aux_mask


def do_softmax(similarity: 'torch.Tensor', top_k: 'Optional[int]'=None, inplace: 'bool'=False, return_usage: 'bool'=False) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1)
        x_exp = values.exp_()
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp)
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp)
    else:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum
        indices = None
    if return_usage:
        return affinity, affinity.sum(dim=2)
    return affinity


def get_similarity(mk: 'torch.Tensor', ms: 'torch.Tensor', qk: 'torch.Tensor', qe: 'torch.Tensor', add_batch_dim: 'bool'=False) ->torch.Tensor:
    if add_batch_dim:
        mk, ms = mk.unsqueeze(0), ms.unsqueeze(0)
        qk, qe = qk.unsqueeze(0), qe.unsqueeze(0)
    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None
    if qe is not None:
        mk = mk.transpose(1, 2)
        a_sq = mk.pow(2) @ qe
        two_ab = 2 * (mk @ (qk * qe))
        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
        similarity = -a_sq + two_ab - b_sq
    else:
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        two_ab = 2 * (mk.transpose(1, 2) @ qk)
        similarity = -a_sq + two_ab
    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)
    else:
        similarity = similarity / math.sqrt(CK)
    return similarity


def get_affinity(mk: 'torch.Tensor', ms: 'torch.Tensor', qk: 'torch.Tensor', qe: 'torch.Tensor') ->torch.Tensor:
    similarity = get_similarity(mk, ms, qk, qe)
    affinity = do_softmax(similarity)
    return affinity


log = logging.getLogger()


def readout(affinity: 'torch.Tensor', mv: 'torch.Tensor') ->torch.Tensor:
    B, CV, T, H, W = mv.shape
    mo = mv.view(B, CV, T * H * W)
    mem = torch.bmm(mo, affinity)
    mem = mem.view(B, CV, H, W)
    return mem


class CUTIE(nn.Module):

    def __init__(self, cfg: 'DictConfig', *, single_object=False):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model
        self.ms_dims = model_cfg.pixel_encoder.ms_dims
        self.key_dim = model_cfg.key_dim
        self.value_dim = model_cfg.value_dim
        self.sensory_dim = model_cfg.sensory_dim
        self.pixel_dim = model_cfg.pixel_dim
        self.embed_dim = model_cfg.embed_dim
        self.single_object = single_object
        self.object_transformer_enabled = model_cfg.object_transformer.num_blocks > 0
        log.info(f'Single object: {self.single_object}')
        log.info(f'Object transformer enabled: {self.object_transformer_enabled}')
        self.pixel_encoder = PixelEncoder(model_cfg)
        self.pix_feat_proj = nn.Conv2d(self.ms_dims[0], self.pixel_dim, kernel_size=1)
        self.key_proj = KeyProjection(model_cfg)
        self.mask_encoder = MaskEncoder(model_cfg, single_object=single_object)
        self.mask_decoder = MaskDecoder(model_cfg)
        self.pixel_fuser = PixelFeatureFuser(model_cfg, single_object=single_object)
        if self.object_transformer_enabled:
            self.object_transformer = QueryTransformer(model_cfg)
            self.object_summarizer = ObjectSummarizer(model_cfg)
        self.aux_computer = AuxComputer(cfg)
        self.register_buffer('pixel_mean', torch.Tensor(model_cfg.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(model_cfg.pixel_std).view(-1, 1, 1), False)

    def _get_others(self, masks: 'torch.Tensor') ->torch.Tensor:
        if self.single_object:
            return None
        num_objects = masks.shape[1]
        if num_objects >= 1:
            others = (masks.sum(dim=1, keepdim=True) - masks).clamp(0, 1)
        else:
            others = torch.zeros_like(masks)
        return others

    def encode_image(self, image: 'torch.Tensor') ->(Iterable[torch.Tensor], torch.Tensor):
        image = (image - self.pixel_mean) / self.pixel_std
        ms_image_feat = self.pixel_encoder(image)
        return ms_image_feat, self.pix_feat_proj(ms_image_feat[0])

    def encode_mask(self, image: 'torch.Tensor', ms_features: 'List[torch.Tensor]', sensory: 'torch.Tensor', masks: 'torch.Tensor', *, deep_update: bool=True, chunk_size: int=-1, need_weights: bool=False) ->(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        image = (image - self.pixel_mean) / self.pixel_std
        others = self._get_others(masks)
        mask_value, new_sensory = self.mask_encoder(image, ms_features, sensory, masks, others, deep_update=deep_update, chunk_size=chunk_size)
        if self.object_transformer_enabled:
            object_summaries, object_logits = self.object_summarizer(masks, mask_value, need_weights)
        else:
            object_summaries, object_logits = None, None
        return mask_value, new_sensory, object_summaries, object_logits

    def transform_key(self, final_pix_feat: 'torch.Tensor', *, need_sk: bool=True, need_ek: bool=True) ->(torch.Tensor, torch.Tensor, torch.Tensor):
        key, shrinkage, selection = self.key_proj(final_pix_feat, need_s=need_sk, need_e=need_ek)
        return key, shrinkage, selection

    def read_memory(self, query_key: 'torch.Tensor', query_selection: 'torch.Tensor', memory_key: 'torch.Tensor', memory_shrinkage: 'torch.Tensor', msk_value: 'torch.Tensor', obj_memory: 'torch.Tensor', pix_feat: 'torch.Tensor', sensory: 'torch.Tensor', last_mask: 'torch.Tensor', selector: 'torch.Tensor') ->(torch.Tensor, Dict[str, torch.Tensor]):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        msk_value       : B * num_objects * CV * T * H * W
        obj_memory      : B * num_objects * T * num_summaries * C
        pixel_feature   : B * C * H * W
        """
        batch_size, num_objects = msk_value.shape[:2]
        with torch.amp.autocast(enabled=False):
            affinity = get_affinity(memory_key.float(), memory_shrinkage.float(), query_key.float(), query_selection.float())
            msk_value = msk_value.flatten(start_dim=1, end_dim=2).float()
            pixel_readout = readout(affinity, msk_value)
            pixel_readout = pixel_readout.view(batch_size, num_objects, self.value_dim, *pixel_readout.shape[-2:])
        pixel_readout = self.pixel_fusion(pix_feat, pixel_readout, sensory, last_mask)
        mem_readout, aux_features = self.readout_query(pixel_readout, obj_memory, selector=selector)
        aux_output = {'sensory': sensory, 'q_logits': aux_features['logits'] if aux_features else None, 'attn_mask': aux_features['attn_mask'] if aux_features else None}
        return mem_readout, aux_output

    def pixel_fusion(self, pix_feat: 'torch.Tensor', pixel: 'torch.Tensor', sensory: 'torch.Tensor', last_mask: 'torch.Tensor', *, chunk_size: int=-1) ->torch.Tensor:
        last_mask = F.interpolate(last_mask, size=sensory.shape[-2:], mode='area')
        last_others = self._get_others(last_mask)
        fused = self.pixel_fuser(pix_feat, pixel, sensory, last_mask, last_others, chunk_size=chunk_size)
        return fused

    def readout_query(self, pixel_readout, obj_memory, *, selector=None, need_weights=False) ->(torch.Tensor, Dict[str, torch.Tensor]):
        if not self.object_transformer_enabled:
            return pixel_readout, None
        return self.object_transformer(pixel_readout, obj_memory, selector=selector, need_weights=need_weights)

    def segment(self, ms_image_feat: 'List[torch.Tensor]', memory_readout: 'torch.Tensor', sensory: 'torch.Tensor', *, selector: bool=None, chunk_size: int=-1, update_sensory: bool=True) ->(torch.Tensor, torch.Tensor, torch.Tensor):
        """
        multi_scale_features is from the key encoder for skip-connection
        memory_readout is from working/long-term memory
        sensory is the sensory memory
        last_mask is the mask from the last frame, supplementing sensory memory
        selector is 1 if an object exists, and 0 otherwise. We use it to filter padded objects
            during training.
        """
        sensory, logits = self.mask_decoder(ms_image_feat, memory_readout, sensory, chunk_size=chunk_size, update_sensory=update_sensory)
        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector
        logits = aggregate(prob, dim=1)
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        prob = F.softmax(logits, dim=1)
        return sensory, logits, prob

    def compute_aux(self, pix_feat: 'torch.Tensor', aux_inputs: 'Dict[str, torch.Tensor]', selector: 'torch.Tensor') ->Dict[str, torch.Tensor]:
        return self.aux_computer(pix_feat, aux_inputs, selector)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def load_weights(self, src_dict, init_as_zero_if_needed=False) ->None:
        if not self.single_object:
            for k in list(src_dict.keys()):
                if k == 'mask_encoder.conv1.weight':
                    if src_dict[k].shape[1] == 4:
                        log.info(f'Converting {k} from single object to multiple objects.')
                        pads = torch.zeros((64, 1, 7, 7), device=src_dict[k].device)
                        if not init_as_zero_if_needed:
                            nn.init.orthogonal_(pads)
                            log.info(f'Randomly initialized padding for {k}.')
                        else:
                            log.info(f'Zero-initialized padding for {k}.')
                        src_dict[k] = torch.cat([src_dict[k], pads], 1)
                elif k == 'pixel_fuser.sensory_compress.weight':
                    if src_dict[k].shape[1] == self.sensory_dim + 1:
                        log.info(f'Converting {k} from single object to multiple objects.')
                        pads = torch.zeros((self.value_dim, 1, 1, 1), device=src_dict[k].device)
                        if not init_as_zero_if_needed:
                            nn.init.orthogonal_(pads)
                            log.info(f'Randomly initialized padding for {k}.')
                        else:
                            log.info(f'Zero-initialized padding for {k}.')
                        src_dict[k] = torch.cat([src_dict[k], pads], 1)
        elif self.single_object:
            """
            If the model is multiple-object and we are training in single-object, 
            we strip the last channel of conv1.
            This is not supposed to happen in standard training except when users are trying to
            finetune a trained model with single object datasets.
            """
            k = 'mask_encoder.conv1.weight'
            if src_dict[k].shape[1] == 5:
                log.warning(f'Converting {k} from multiple objects to single object.This is not supposed to happen in standard training.')
                src_dict[k] = src_dict[k][:, :-1]
        for k in src_dict:
            if k not in self.state_dict():
                log.info(f'Key {k} found in src_dict but not in self.state_dict()!!!')
        for k in self.state_dict():
            if k not in src_dict:
                log.info(f'Key {k} found in self.state_dict() but not in src_dict!!!')
        self.load_state_dict(src_dict, strict=False)

    @property
    def device(self) ->torch.device:
        return self.pixel_mean.device


class CutieTrainWrapper(CUTIE):

    def __init__(self, cfg: 'DictConfig', stage_cfg: 'DictConfig'):
        super().__init__(cfg, single_object=stage_cfg.num_objects == 1)
        self.sensory_dim = cfg.model.sensory_dim
        self.seq_length = stage_cfg.seq_length
        self.num_ref_frames = stage_cfg.num_ref_frames
        self.deep_update_prob = stage_cfg.deep_update_prob
        self.use_amp = stage_cfg.amp
        self.move_t_out_of_batch = Rearrange('(b t) c h w -> b t c h w', t=self.seq_length)
        self.move_t_from_batch_to_volume = Rearrange('(b t) c h w -> b c t h w', t=self.seq_length)

    def forward(self, data: 'Dict'):
        out = {}
        frames = data['rgb']
        first_frame_gt = data['first_frame_gt'].float()
        b, seq_length = frames.shape[:2]
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        max_num_objects = max(num_filled_objects)
        first_frame_gt = first_frame_gt[:, :, :max_num_objects]
        selector = data['selector'][:, :max_num_objects].unsqueeze(2).unsqueeze(2)
        num_objects = first_frame_gt.shape[2]
        out['num_filled_objects'] = num_filled_objects

        def get_ms_feat_ti(ti):
            return [f[:, ti] for f in ms_feat]
        with torch.amp.autocast(enabled=self.use_amp):
            frames_flat = frames.view(b * seq_length, *frames.shape[2:])
            ms_feat, pix_feat = self.encode_image(frames_flat)
            with torch.amp.autocast(enabled=False):
                keys, shrinkages, selections = self.transform_key(ms_feat[0].float())
            h, w = keys.shape[-2:]
            keys = self.move_t_from_batch_to_volume(keys)
            shrinkages = self.move_t_from_batch_to_volume(shrinkages)
            selections = self.move_t_from_batch_to_volume(selections)
            ms_feat = [self.move_t_out_of_batch(f) for f in ms_feat]
            pix_feat = self.move_t_out_of_batch(pix_feat)
            sensory = torch.zeros((b, num_objects, self.sensory_dim, h, w), device=frames.device)
            msk_val, sensory, obj_val, _ = self.encode_mask(frames[:, 0], pix_feat[:, 0], sensory, first_frame_gt[:, 0])
            masks = first_frame_gt[:, 0]
            msk_values = msk_val.unsqueeze(3)
            obj_values = obj_val.unsqueeze(2) if obj_val is not None else None
            for ti in range(1, seq_length):
                if ti <= self.num_ref_frames:
                    ref_msk_values = msk_values
                    ref_keys = keys[:, :, :ti]
                    ref_shrinkages = shrinkages[:, :, :ti] if shrinkages is not None else None
                else:
                    ridx = [torch.randperm(ti)[:self.num_ref_frames] for _ in range(b)]
                    ref_msk_values = torch.stack([msk_values[bi, :, :, ridx[bi]] for bi in range(b)], 0)
                    ref_keys = torch.stack([keys[bi, :, ridx[bi]] for bi in range(b)], 0)
                    ref_shrinkages = torch.stack([shrinkages[bi, :, ridx[bi]] for bi in range(b)], 0)
                readout, aux_input = self.read_memory(keys[:, :, ti], selections[:, :, ti], ref_keys, ref_shrinkages, ref_msk_values, obj_values, pix_feat[:, ti], sensory, masks, selector)
                aux_output = self.compute_aux(pix_feat[:, ti], aux_input, selector)
                sensory, logits, masks = self.segment(get_ms_feat_ti(ti), readout, sensory, selector=selector)
                masks = masks[:, 1:]
                if ti < self.seq_length - 1:
                    deep_update = np.random.rand() < self.deep_update_prob
                    msk_val, sensory, obj_val, _ = self.encode_mask(frames[:, ti], pix_feat[:, ti], sensory, masks, deep_update=deep_update)
                    msk_values = torch.cat([msk_values, msk_val.unsqueeze(3)], 3)
                    obj_values = torch.cat([obj_values, obj_val.unsqueeze(2)], 2) if obj_val is not None else None
                out[f'masks_{ti}'] = masks
                out[f'logits_{ti}'] = logits
                out[f'aux_{ti}'] = aux_output
        return out


class OutputFFN(nn.Module):

    def __init__(self, dim_in: 'int', dim_out: 'int', activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_out, dim_out)
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return x


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers=(3, 4, 23, 3), extra_dim=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3 + extra_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)


class BRSMaskLoss(torch.nn.Module):

    def __init__(self, eps=1e-05):
        super().__init__()
        self._eps = eps

    def forward(self, result, pos_mask, neg_mask):
        pos_diff = (1 - result) * pos_mask
        pos_target = torch.sum(pos_diff ** 2)
        pos_target = pos_target / (torch.sum(pos_mask) + self._eps)
        neg_diff = result * neg_mask
        neg_target = torch.sum(neg_diff ** 2)
        neg_target = neg_target / (torch.sum(neg_mask) + self._eps)
        loss = pos_target + neg_target
        with torch.no_grad():
            f_max_pos = torch.max(torch.abs(pos_diff)).item()
            f_max_neg = torch.max(torch.abs(neg_diff)).item()
        return loss, f_max_pos, f_max_neg


class SigmoidBinaryCrossEntropyLoss(nn.Module):

    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))
        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label + torch.log(1.0 - pred + eps) * (1.0 - label))
        loss = self._weight * (loss * sample_weight)
        return torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))


class OracleMaskLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.gt_mask = None
        self.loss = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
        self.predictor = None
        self.history = []

    def set_gt_mask(self, gt_mask):
        self.gt_mask = gt_mask
        self.history = []

    def forward(self, result, pos_mask, neg_mask):
        gt_mask = self.gt_mask
        if self.predictor.object_roi is not None:
            r1, r2, c1, c2 = self.predictor.object_roi[:4]
            gt_mask = gt_mask[:, :, r1:r2 + 1, c1:c2 + 1]
            gt_mask = torch.nn.functional.interpolate(gt_mask, result.size()[2:], mode='bilinear', align_corners=True)
        if result.shape[0] == 2:
            gt_mask_flipped = torch.flip(gt_mask, dims=[3])
            gt_mask = torch.cat([gt_mask, gt_mask_flipped], dim=0)
        loss = self.loss(result, gt_mask)
        self.history.append(loss.detach().cpu().numpy()[0])
        if len(self.history) > 5 and abs(self.history[-5] - self.history[-1]) < 1e-05:
            return 0, 0, 0
        return loss, 1.0, 1.0


class BatchImageNormalize:

    def __init__(self, mean, std, dtype=torch.float):
        self.mean = torch.as_tensor(mean, dtype=dtype)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype)[None, :, None, None]

    def __call__(self, tensor):
        tensor = tensor.clone()
        tensor.sub_(self.mean).div_(self.std)
        return tensor


class DistMaps(nn.Module):

    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks
        if self.cpu_mode:
            self._get_dist_maps = get_dist_maps

    def get_coord_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(self._get_dist_maps(points[i].cpu().float().numpy(), rows, cols, norm_delimeter))
            coords = torch.from_numpy(np.stack(coords, axis=0)).float()
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order = torch.split(points, [2, 1], dim=1)
            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)
            coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)
            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy)
            if not self.use_disks:
                coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)
            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]
            coords[invalid_points, :, :, :] = 1000000.0
            coords = coords.view(-1, num_points, 1, rows, cols)
            coords = coords.min(dim=1)[0]
            coords = coords.view(-1, 2, rows, cols)
        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()
        return coords

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])


class LRMult(object):

    def __init__(self, lr_mult=1.0):
        self.lr_mult = lr_mult

    def __call__(self, m):
        if getattr(m, 'weight', None) is not None:
            m.weight.lr_mult = self.lr_mult
        if getattr(m, 'bias', None) is not None:
            m.bias.lr_mult = self.lr_mult


class ScaleLayer(nn.Module):

    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(torch.full((1,), init_value / lr_mult, dtype=torch.float32))

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


def split_points_by_order(tpoints: 'torch.Tensor', groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2
    groups = [(x if x > 0 else num_points) for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32) for x in groups]
    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int32)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size
    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue
            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or group_id == 0 and is_negative:
                group_id = num_groups - 1
            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1
            group_points[group_id][bindx, new_point_indx, :] = point
    group_points = [torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device) for x in group_points]
    return group_points


class ISModel(nn.Module):

    def __init__(self, use_rgb_conv=True, with_aux_output=False, norm_radius=260, use_disks=False, cpu_dist_maps=False, clicks_groups=None, with_prev_mask=False, use_leaky_relu=False, binary_prev_mask=False, conv_extend=False, norm_layer=nn.BatchNorm2d, norm_mean_std=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
        super().__init__()
        self.with_aux_output = with_aux_output
        self.clicks_groups = clicks_groups
        self.with_prev_mask = with_prev_mask
        self.binary_prev_mask = binary_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])
        self.coord_feature_ch = 2
        if clicks_groups is not None:
            self.coord_feature_ch *= len(clicks_groups)
        if self.with_prev_mask:
            self.coord_feature_ch += 1
        if use_rgb_conv:
            rgb_conv_layers = [nn.Conv2d(in_channels=3 + self.coord_feature_ch, out_channels=6 + self.coord_feature_ch, kernel_size=1), norm_layer(6 + self.coord_feature_ch), nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True), nn.Conv2d(in_channels=6 + self.coord_feature_ch, out_channels=3, kernel_size=1)]
            self.rgb_conv = nn.Sequential(*rgb_conv_layers)
        elif conv_extend:
            self.rgb_conv = None
            self.maps_transform = nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=64, kernel_size=3, stride=2, padding=1)
            self.maps_transform.apply(LRMult(0.1))
        else:
            self.rgb_conv = None
            mt_layers = [nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1), nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True), nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1), ScaleLayer(init_value=0.05, lr_mult=1)]
            self.maps_transform = nn.Sequential(*mt_layers)
        if self.clicks_groups is not None:
            self.dist_maps = nn.ModuleList()
            for click_radius in self.clicks_groups:
                self.dist_maps.append(DistMaps(norm_radius=click_radius, spatial_scale=1.0, cpu_mode=cpu_dist_maps, use_disks=use_disks))
        else:
            self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0, cpu_mode=cpu_dist_maps, use_disks=use_disks)

    def forward(self, image, points):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        if self.rgb_conv is not None:
            x = self.rgb_conv(torch.cat((image, coord_features), dim=1))
            outputs = self.backbone_forward(x)
        else:
            coord_features = self.maps_transform(coord_features)
            outputs = self.backbone_forward(image, coord_features)
        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:], mode='bilinear', align_corners=True)
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:], mode='bilinear', align_corners=True)
        return outputs

    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]
            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).float()
        image = self.normalization(image)
        return image, prev_mask

    def backbone_forward(self, image, coord_features=None):
        raise NotImplementedError

    def get_coord_features(self, image, prev_mask, points):
        if self.clicks_groups is not None:
            points_groups = split_points_by_order(points, groups=(2,) + (1,) * (len(self.clicks_groups) - 2) + (-1,))
            coord_features = [dist_map(image, pg) for dist_map, pg in zip(self.dist_maps, points_groups)]
            coord_features = torch.cat(coord_features, dim=1)
        else:
            coord_features = self.dist_maps(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features


class NormalizedFocalLossSigmoid(nn.Module):

    def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12, from_sigmoid=False, detach_delimeter=True, batch_axis=0, weight=None, size_average=True, ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis
        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label
        if not self._from_logits:
            pred = torch.sigmoid(pred)
        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))
        beta = (1 - pt) ** self._gamma
        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)
        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
            sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()
                beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax
        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float)))
        loss = self._weight * (loss * sample_weight)
        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))
        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
        sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)


class FocalLoss(nn.Module):

    def __init__(self, axis=-1, alpha=0.25, gamma=2, from_logits=False, batch_axis=0, weight=None, num_class=None, eps=1e-09, size_average=True, scale=1.0, ignore_label=-1):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis
        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label
        if not self._from_logits:
            pred = torch.sigmoid(pred)
        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))
        beta = (1 - pt) ** self._gamma
        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float)))
        loss = self._weight * (loss * sample_weight)
        if self._size_average:
            tsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(label.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (tsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))
        return self._scale * loss


class SoftIoU(nn.Module):

    def __init__(self, from_sigmoid=False, ignore_label=-1):
        super().__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        if not self._from_sigmoid:
            pred = torch.sigmoid(pred)
        loss = 1.0 - torch.sum(pred * label * sample_weight, dim=(1, 2, 3)) / (torch.sum(torch.max(pred, label) * sample_weight, dim=(1, 2, 3)) + 1e-08)
        return loss


class ConvHead(nn.Module):

    def __init__(self, out_channels, in_channels=32, num_layers=1, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d):
        super(ConvHead, self).__init__()
        convhead = []
        for i in range(num_layers):
            convhead.extend([nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding), nn.ReLU(), norm_layer(in_channels) if norm_layer is not None else nn.Identity()])
        convhead.append(nn.Conv2d(in_channels, out_channels, 1, padding=0))
        self.convhead = nn.Sequential(*convhead)

    def forward(self, *inputs):
        return self.convhead(inputs[0])


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_padding, dw_stride=1, activation=None, use_bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        _activation = ops.select_activation_function(activation)
        self.body = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=dw_kernel, stride=dw_stride, padding=dw_padding, bias=use_bias, groups=in_channels), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=use_bias), norm_layer(out_channels) if norm_layer is not None else nn.Identity(), _activation())

    def forward(self, x):
        return self.body(x)


class SepConvHead(nn.Module):

    def __init__(self, num_outputs, in_channels, mid_channels, num_layers=1, kernel_size=3, padding=1, dropout_ratio=0.0, dropout_indx=0, norm_layer=nn.BatchNorm2d):
        super(SepConvHead, self).__init__()
        sepconvhead = []
        for i in range(num_layers):
            sepconvhead.append(SeparableConv2d(in_channels=in_channels if i == 0 else mid_channels, out_channels=mid_channels, dw_kernel=kernel_size, dw_padding=padding, norm_layer=norm_layer, activation='relu'))
            if dropout_ratio > 0 and dropout_indx == i:
                sepconvhead.append(nn.Dropout(dropout_ratio))
        sepconvhead.append(nn.Conv2d(in_channels=mid_channels, out_channels=num_outputs, kernel_size=1, padding=0))
        self.layers = nn.Sequential(*sepconvhead)

    def forward(self, *inputs):
        x = inputs[0]
        return self.layers(x)


class BottleneckV1b(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


GLUON_RESNET_TORCH_HUB = 'rwightman/pytorch-pretrained-gluonresnet'


class ResNetV1b(nn.Module):
    """ Pre-trained ResNetV1b Model, which produces the strides of 8 featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm2d`)
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.

    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, block, layers, classes=1000, dilated=True, deep_stem=False, stem_width=32, avg_down=False, final_drop=0.0, norm_layer=nn.BatchNorm2d):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNetV1b, self).__init__()
        if not deep_stem:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False), norm_layer(stem_width), nn.ReLU(True), nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(stem_width), nn.ReLU(True), nn.Conv2d(stem_width, 2 * stem_width, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], avg_down=avg_down, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, avg_down=avg_down, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, avg_down=avg_down, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, avg_down=avg_down, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, avg_down=avg_down, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, avg_down=avg_down, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = None
        if final_drop > 0.0:
            self.drop = nn.Dropout(final_drop)
        self.fc = nn.Linear(512 * block.expansion, classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, avg_down=False, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = []
            if avg_down:
                if dilation == 1:
                    downsample.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
                else:
                    downsample.append(nn.AvgPool2d(kernel_size=1, stride=1, ceil_mode=True, count_include_pad=False))
                downsample.extend([nn.Conv2d(self.inplanes, out_channels=planes * block.expansion, kernel_size=1, stride=1, bias=False), norm_layer(planes * block.expansion)])
                downsample = nn.Sequential(*downsample)
            else:
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, out_channels=planes * block.expansion, kernel_size=1, stride=stride, bias=False), norm_layer(planes * block.expansion))
        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)
        return x


def _safe_state_dict_filtering(orig_dict, model_dict_keys):
    filtered_orig_dict = {}
    for k, v in orig_dict.items():
        if k in model_dict_keys:
            filtered_orig_dict[k] = v
        else:
            None
    return filtered_orig_dict


def resnet101_v1s(pretrained=False, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(torch.hub.load(GLUON_RESNET_TORCH_HUB, 'gluon_resnet101_v1s', pretrained=True).state_dict(), model_dict.keys())
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


def resnet152_v1s(pretrained=False, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(torch.hub.load(GLUON_RESNET_TORCH_HUB, 'gluon_resnet152_v1s', pretrained=True).state_dict(), model_dict.keys())
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


class BasicBlockV1b(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


def resnet34_v1b(pretrained=False, **kwargs):
    model = ResNetV1b(BasicBlockV1b, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(torch.hub.load(GLUON_RESNET_TORCH_HUB, 'gluon_resnet34_v1b', pretrained=True).state_dict(), model_dict.keys())
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


def resnet50_v1s(pretrained=False, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(torch.hub.load(GLUON_RESNET_TORCH_HUB, 'gluon_resnet50_v1s', pretrained=True).state_dict(), model_dict.keys())
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


class ResNetBackbone(torch.nn.Module):

    def __init__(self, backbone='resnet50', pretrained_base=True, dilated=True, **kwargs):
        super(ResNetBackbone, self).__init__()
        if backbone == 'resnet34':
            pretrained = resnet34_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet50':
            pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet101':
            pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet152':
            pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        else:
            raise RuntimeError(f'unknown backbone: {backbone}')
        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4

    def forward(self, x, additional_features=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if additional_features is not None:
            x = x + torch.nn.functional.pad(additional_features, [0, 0, 0, 0, 0, x.size(1) - additional_features.size(1)], mode='constant', value=0)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4


def _ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=atrous_rate, dilation=atrous_rate, bias=False), norm_layer(out_channels), nn.ReLU())
    return block


class _AsppPooling(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False), norm_layer(out_channels), nn.ReLU())

    def forward(self, x):
        pool = self.gap(x)
        return F.interpolate(pool, x.size()[2:], mode='bilinear', align_corners=True)


class _ASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates, out_channels=256, project_dropout=0.5, norm_layer=nn.BatchNorm2d):
        super(_ASPP, self).__init__()
        b0 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False), norm_layer(out_channels), nn.ReLU())
        rate1, rate2, rate3 = tuple(atrous_rates)
        b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer)
        b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer)
        b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer)
        b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer)
        self.concurent = nn.ModuleList([b0, b1, b2, b3, b4])
        project = [nn.Conv2d(in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, bias=False), norm_layer(out_channels), nn.ReLU()]
        if project_dropout > 0:
            project.append(nn.Dropout(project_dropout))
        self.project = nn.Sequential(*project)

    def forward(self, x):
        x = torch.cat([block(x) for block in self.concurent], dim=1)
        return self.project(x)


class _DeepLabHead(nn.Module):

    def __init__(self, out_channels, in_channels, mid_channels=256, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        self.block = nn.Sequential(SeparableConv2d(in_channels=in_channels, out_channels=mid_channels, dw_kernel=3, dw_padding=1, activation='relu', norm_layer=norm_layer), SeparableConv2d(in_channels=mid_channels, out_channels=mid_channels, dw_kernel=3, dw_padding=1, activation='relu', norm_layer=norm_layer), nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1))

    def forward(self, x):
        return self.block(x)


class _SkipProject(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(_SkipProject, self).__init__()
        _activation = ops.select_activation_function('relu')
        self.skip_project = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), norm_layer(out_channels), _activation())

    def forward(self, x):
        return self.skip_project(x)


class DeepLabV3Plus(nn.Module):

    def __init__(self, backbone='resnet50', norm_layer=nn.BatchNorm2d, backbone_norm_layer=None, ch=256, project_dropout=0.5, inference_mode=False, **kwargs):
        super(DeepLabV3Plus, self).__init__()
        if backbone_norm_layer is None:
            backbone_norm_layer = norm_layer
        self.backbone_name = backbone
        self.norm_layer = norm_layer
        self.backbone_norm_layer = backbone_norm_layer
        self.inference_mode = False
        self.ch = ch
        self.aspp_in_channels = 2048
        self.skip_project_in_channels = 256
        self._kwargs = kwargs
        if backbone == 'resnet34':
            self.aspp_in_channels = 512
            self.skip_project_in_channels = 64
        self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False, norm_layer=self.backbone_norm_layer, **kwargs)
        self.head = _DeepLabHead(in_channels=ch + 32, mid_channels=ch, out_channels=ch, norm_layer=self.norm_layer)
        self.skip_project = _SkipProject(self.skip_project_in_channels, 32, norm_layer=self.norm_layer)
        self.aspp = _ASPP(in_channels=self.aspp_in_channels, atrous_rates=[12, 24, 36], out_channels=ch, project_dropout=project_dropout, norm_layer=self.norm_layer)
        if inference_mode:
            self.set_prediction_mode()

    def load_pretrained_weights(self):
        pretrained = ResNetBackbone(backbone=self.backbone_name, pretrained_base=True, norm_layer=self.backbone_norm_layer, **self._kwargs)
        backbone_state_dict = self.backbone.state_dict()
        pretrained_state_dict = pretrained.state_dict()
        backbone_state_dict.update(pretrained_state_dict)
        self.backbone.load_state_dict(backbone_state_dict)
        if self.inference_mode:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def set_prediction_mode(self):
        self.inference_mode = True
        self.eval()

    def forward(self, x, additional_features=None):
        with ExitStack() as stack:
            if self.inference_mode:
                stack.enter_context(torch.no_grad())
            c1, _, c3, c4 = self.backbone(x, additional_features)
            c1 = self.skip_project(c1)
            x = self.aspp(c4)
            x = F.interpolate(x, c1.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, c1), dim=1)
            x = self.head(x)
        return x,


relu_inplace = True


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True, norm_layer=nn.BatchNorm2d, align_corners=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.norm_layer = norm_layer
        self.align_corners = align_corners
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), self.norm_layer(num_channels[branch_index] * block.expansion))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample=downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], norm_layer=self.norm_layer))
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
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(in_channels=num_inchannels[j], out_channels=num_inchannels[i], kernel_size=1, bias=False), self.norm_layer(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, kernel_size=3, stride=2, padding=1, bias=False), self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, kernel_size=3, stride=2, padding=1, bias=False), self.norm_layer(num_outchannels_conv3x3), nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=[height_output, width_output], mode='bilinear', align_corners=self.align_corners)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock2D(nn.Module):
    """
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    """

    def __init__(self, in_channels, key_channels, scale=1, norm_layer=nn.BatchNorm2d, align_corners=True):
        super(ObjectAttentionBlock2D, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.align_corners = align_corners
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)), nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)))
        self.f_object = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)), nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)))
        self.f_down = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)))
        self.f_up = nn.Sequential(nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.in_channels), nn.ReLU(inplace=True)))

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)
        sim_map = self.key_channels ** -0.5 * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=self.align_corners)
        return context


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1, norm_layer=nn.BatchNorm2d, align_corners=True):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale, norm_layer, align_corners)
        _in_channels = 2 * in_channels
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False), nn.Sequential(norm_layer(out_channels), nn.ReLU(inplace=True)), nn.Dropout2d(dropout))

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class HighResolutionNet(nn.Module):

    def __init__(self, width, num_classes, ocr_width=256, small=False, norm_layer=nn.BatchNorm2d, align_corners=True):
        super(HighResolutionNet, self).__init__()
        self.norm_layer = norm_layer
        self.width = width
        self.ocr_width = ocr_width
        self.align_corners = align_corners
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = norm_layer(64)
        self.relu = nn.ReLU(inplace=relu_inplace)
        num_blocks = 2 if small else 4
        stage1_num_channels = 64
        self.layer1 = self._make_layer(BottleneckV1b, 64, stage1_num_channels, blocks=num_blocks)
        stage1_out_channel = BottleneckV1b.expansion * stage1_num_channels
        self.stage2_num_branches = 2
        num_channels = [width, 2 * width]
        num_inchannels = [(num_channels[i] * BasicBlockV1b.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_inchannels)
        self.stage2, pre_stage_channels = self._make_stage(BasicBlockV1b, num_inchannels=num_inchannels, num_modules=1, num_branches=self.stage2_num_branches, num_blocks=2 * [num_blocks], num_channels=num_channels)
        self.stage3_num_branches = 3
        num_channels = [width, 2 * width, 4 * width]
        num_inchannels = [(num_channels[i] * BasicBlockV1b.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_inchannels)
        self.stage3, pre_stage_channels = self._make_stage(BasicBlockV1b, num_inchannels=num_inchannels, num_modules=3 if small else 4, num_branches=self.stage3_num_branches, num_blocks=3 * [num_blocks], num_channels=num_channels)
        self.stage4_num_branches = 4
        num_channels = [width, 2 * width, 4 * width, 8 * width]
        num_inchannels = [(num_channels[i] * BasicBlockV1b.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_inchannels)
        self.stage4, pre_stage_channels = self._make_stage(BasicBlockV1b, num_inchannels=num_inchannels, num_modules=2 if small else 3, num_branches=self.stage4_num_branches, num_blocks=4 * [num_blocks], num_channels=num_channels)
        last_inp_channels = np.int32(np.sum(pre_stage_channels))
        if self.ocr_width > 0:
            ocr_mid_channels = 2 * self.ocr_width
            ocr_key_channels = self.ocr_width
            self.conv3x3_ocr = nn.Sequential(nn.Conv2d(last_inp_channels, ocr_mid_channels, kernel_size=3, stride=1, padding=1), norm_layer(ocr_mid_channels), nn.ReLU(inplace=relu_inplace))
            self.ocr_gather_head = SpatialGather_Module(num_classes)
            self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels, key_channels=ocr_key_channels, out_channels=ocr_mid_channels, scale=1, dropout=0.05, norm_layer=norm_layer, align_corners=align_corners)
            self.cls_head = nn.Conv2d(ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            self.aux_head = nn.Sequential(nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1, stride=1, padding=0), norm_layer(last_inp_channels), nn.ReLU(inplace=relu_inplace), nn.Conv2d(last_inp_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        else:
            self.cls_head = nn.Sequential(nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=3, stride=1, padding=1), norm_layer(last_inp_channels), nn.ReLU(inplace=relu_inplace), nn.Conv2d(last_inp_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=3, stride=1, padding=1, bias=False), self.norm_layer(num_channels_cur_layer[i]), nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False), self.norm_layer(outchannels), nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), self.norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))
        return nn.Sequential(*layers)

    def _make_stage(self, block, num_inchannels, num_modules, num_branches, num_blocks, num_channels, fuse_method='SUM', multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output, norm_layer=self.norm_layer, align_corners=self.align_corners))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, additional_features=None):
        feats = self.compute_hrnet_feats(x, additional_features)
        if self.ocr_width > 0:
            out_aux = self.aux_head(feats)
            feats = self.conv3x3_ocr(feats)
            context = self.ocr_gather_head(feats, out_aux)
            feats = self.ocr_distri_head(feats, context)
            out = self.cls_head(feats)
            return [out, out_aux]
        else:
            return [self.cls_head(feats), None]

    def compute_hrnet_feats(self, x, additional_features):
        x = self.compute_pre_stage_features(x, additional_features)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_num_branches):
            if self.transition2[i] is not None:
                if i < self.stage2_num_branches:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                if i < self.stage3_num_branches:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        return self.aggregate_hrnet_features(x)

    def compute_pre_stage_features(self, x, additional_features):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if additional_features is not None:
            x = x + additional_features
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x)

    def aggregate_hrnet_features(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)
        return torch.cat([x[0], x1, x2, x3], 1)

    def load_pretrained_weights(self, pretrained_path=''):
        model_dict = self.state_dict()
        if not os.path.exists(pretrained_path):
            None
            None
            exit(1)
        pretrained_dict = torch.load(pretrained_path, map_location={'cuda:0': 'cpu'})
        pretrained_dict = {k.replace('last_layer', 'aux_head').replace('model.', ''): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class BilinearConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, in_channels, out_channels, scale, groups=1):
        kernel_size = 2 * scale - scale % 2
        self.scale = scale
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=scale, padding=1, groups=groups, bias=False)
        self.apply(initializer.Bilinear(scale=scale, in_channels=in_channels, groups=groups))


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BRSMaskLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlockV1b,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CAResBlock,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CrossAttention,
     lambda: ([], {'dim': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (DecoderFeatureProcessor,
     lambda: ([], {'decoder_dims': [4, 4], 'out_dims': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FFN,
     lambda: ([], {'dim_in': 4, 'dim_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 1, 4, 4])], {})),
    (GroupResBlock,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 1, 4, 4])], {})),
    (HighResolutionNet,
     lambda: ([], {'width': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ObjectAttentionBlock2D,
     lambda: ([], {'in_channels': 4, 'key_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (OutputFFN,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PixelFFN,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4]), torch.rand([16, 4, 4, 4])], {})),
    (PixelFeatureFuser,
     lambda: ([], {'model_cfg': SimpleNamespace(value_dim=4, sensory_dim=4, pixel_dim=4, embed_dim=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNetBackbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ScaleLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SelfAttention,
     lambda: ([], {'dim': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (SoftIoU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SpatialGather_Module,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SpatialOCR_Module,
     lambda: ([], {'in_channels': 4, 'key_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (_ASPP,
     lambda: ([], {'in_channels': 4, 'atrous_rates': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_AsppPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'norm_layer': torch.nn.ReLU}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

