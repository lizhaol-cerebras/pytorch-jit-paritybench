
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


import torch


import torch.nn.functional as F


import numpy as np


import torchvision.transforms as transforms


import logging


from torch.utils.tensorboard import SummaryWriter


import torch.nn as nn


from enum import Enum


from torchvision.transforms import ColorJitter


from torchvision.transforms import GaussianBlur


from typing import Mapping


from typing import Tuple


from typing import Union


from typing import Any


from typing import Optional


from collections import defaultdict


from matplotlib import cm


import matplotlib.pyplot as plt


from torchvision.transforms import Compose


import random


from torchvision import transforms


import math


import types


from torch.utils.checkpoint import checkpoint


from torchvision.transforms import Normalize


import itertools


from scipy import ndimage


import re


import matplotlib


import matplotlib.cm


import torch.distributed as dist


import torch.nn


import torch.utils.data.distributed


from torchvision.transforms import ToTensor


from torch.utils.data import DataLoader


from numpy import random


from torch._C import dtype


from torch._C import set_flush_denormal


from functools import partial


import torchvision.transforms.functional as TF


from sklearn.decomposition import PCA


from torch.cuda.amp import autocast


import collections


from itertools import repeat


import torchvision.models as tvm


from torch.nn import Module


from torch.nn import Dropout


import copy


import typing


from sklearn.cluster import SpectralClustering


from torch.autograd import Variable


from collections import OrderedDict


from torch.nn import init


from typing import Type


import time


import torchvision.ops as ops


import torchvision


from torch.utils.data import Dataset


from typing import Callable


from typing import List


from typing import TypeVar


from torch.utils.data import Sampler


import warnings


from torch.utils.data.sampler import Sampler


from typing import Sequence


from typing import Dict


from torch.nn.functional import one_hot


from torch.nn.functional import softmax


from torch.nn.parallel import DistributedDataParallel


import torch.backends.cudnn as cudnn


import torch.distributed


from torch import nn


from torch.utils.data import TensorDataset


from torch import Tensor


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.distributed.fsdp import ShardingStrategy


from torch.distributed.fsdp import MixedPrecision


from torch.distributed.fsdp import StateDictType


from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


from torch.distributed.fsdp.wrap import ModuleWrapPolicy


from torch.distributed.fsdp._runtime_utils import _reshard


from torch.nn.init import trunc_normal_


from torch.nn.utils import weight_norm


from collections import deque


import torch.utils.checkpoint


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(self, width, height, resize_target=True, keep_aspect_ratio=False, ensure_multiple_of=1, resize_method='lower_bound'):
        """Init.
        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        None
        None
        None
        None
        None
        None
        None
        self.__width = width
        self.__height = height
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)
        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)
        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)
        return y

    def get_size(self, width, height):
        scale_height = self.__height / height
        scale_width = self.__width / width
        if self.__keep_aspect_ratio:
            if self.__resize_method == 'lower_bound':
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == 'upper_bound':
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == 'minimal':
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise ValueError(f'resize_method {self.__resize_method} not implemented')
        if self.__resize_method == 'lower_bound':
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == 'upper_bound':
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == 'minimal':
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f'resize_method {self.__resize_method} not implemented')
        return new_width, new_height

    def __call__(self, x):
        width, height = self.get_size(*x.shape[-2:][::-1])
        return nn.functional.interpolate(x, (int(height), int(width)), mode='bilinear', align_corners=True)


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not self.bn, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not self.bn, groups=self.groups)
        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)
        if self.groups > 1:
            out = self.conv_merge(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(features, nn.ReLU(False), deconv=False, bn=use_bn, expand=False, align_corners=True)


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch


class DPTHeadEnc(nn.Module):

    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False, out_c=128):
        super(DPTHeadEnc, self).__init__()
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=1, stride=1, padding=0) for out_channel in out_channels])
        self.resize_layers = nn.ModuleList([nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0), nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0), nn.Identity(), nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1)])
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()))
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        head_features_1 = features
        self.scratch.output_conv = nn.Sequential(nn.Conv2d(head_features_1, out_c, kernel_size=3, stride=1, padding=1))

    def forward(self, out_features, patch_h, patch_w, enc_only=True):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        layer_1, layer_2, layer_3, layer_4 = out
        if enc_only == True:
            layer_1_rs = F.interpolate(layer_1, (int(patch_h * 14), int(patch_w * 14)), mode='bilinear', align_corners=True)
            layer_2_rs = F.interpolate(layer_2, (int(patch_h * 14), int(patch_w * 14)), mode='bilinear', align_corners=True)
            layer_3_rs = F.interpolate(layer_3, (int(patch_h * 14), int(patch_w * 14)), mode='bilinear', align_corners=True)
            layer_4_rs = F.interpolate(layer_4, (int(patch_h * 14), int(patch_w * 14)), mode='bilinear', align_corners=True)
            return layer_4_rs + layer_3_rs + layer_2_rs + layer_1_rs
        else:
            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)
            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
            out = self.scratch.output_conv(path_1)
            out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode='bilinear', align_corners=True)
            return out


class DPT_DINOv2(nn.Module):

    def __init__(self, encoder='vits', features=64, out_channels=[48, 96, 192, 384], use_bn=True, use_clstoken=False, localhub=True, stride=2, enc_only=True):
        super(DPT_DINOv2, self).__init__()
        self.stride = stride
        self.enc_only = enc_only
        assert encoder in ['vits', 'vitb', 'vitl']
        if localhub:
            self.pretrained = torch.hub.load('models/torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
        state_dict = torch.load('models/monoD/zoeDepth/ckpts/dinov2_vits14_pretrain.pth')
        self.pretrained.load_state_dict(state_dict, strict=True)
        self.pretrained.requires_grad_(False)
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        if enc_only == True:
            out_channels = [128, 128, 128, 128]
        self.DPThead = DPTHeadEnc(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x):
        mean_ = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std_ = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x + 1) / 2
        x = (x - mean_) / std_
        h, w = x.shape[-2:]
        h_re, w_re = 560, 560
        x_resize = F.interpolate(x, size=(h_re, w_re), mode='bilinear', align_corners=False)
        with torch.no_grad():
            features = self.pretrained.get_intermediate_layers(x_resize, 4, return_class_token=True)
        patch_h, patch_w = h_re // 14, w_re // 14
        feat = self.DPThead(features, patch_h, patch_w, self.enc_only)
        feat = F.interpolate(feat, size=(h // self.stride, w // self.stride), mode='bilinear', align_corners=True)
        return feat


def build(config):
    """
        Build the model from the config
        NOTE: the config should contain the following
        - encoder: the encoder type of the model
        - load_from: the path to the pretrained model
    """
    args = config
    assert args.encoder in ['vits', 'vitb', 'vitl']
    if args.encoder == 'vits':
        depth_anything = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], localhub=args.localhub)
    elif args.encoder == 'vitb':
        depth_anything = DPT_DINOv2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768], localhub=args.localhub)
    else:
        depth_anything = DPT_DINOv2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], localhub=args.localhub)
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'), strict=True)
    total_params = sum(param.numel() for param in depth_anything.parameters())
    None
    depth_anything.eval()
    return depth_anything


class DepthAnything(nn.Module):

    def __init__(self, args):
        super(DepthAnything, self).__init__()
        self.dpAny = build(args)

    def infer(self, rgbs):
        """
            Infer the depth map from the input RGB image
        
        Args:
            rgbs: the input RGB image B x 3 x H x W (Cuda Tensor)
        
        Asserts:
            the input should be a cuda tensor
        """
        T, C, H, W = rgbs.shape
        Resizer = Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC)
        width, height = Resizer.get_size(rgbs.shape[2], rgbs.shape[3])
        rgbs = F.interpolate(rgbs, (int(height), int(width)), mode='bicubic', align_corners=False)
        mean_ = torch.tensor([0.485, 0.456, 0.406], device=rgbs.device).view(1, 3, 1, 1)
        std_ = torch.tensor([0.229, 0.224, 0.225], device=rgbs.device).view(1, 3, 1, 1)
        rgbs = (rgbs - mean_) / std_
        disp = self.dpAny(rgbs)
        disp = F.interpolate(disp[:, None], (H, W), mode='bilinear', align_corners=False)
        depth_map = disp
        return depth_map


COMMON_TRAINING_CONFIG = {'dataset': 'nyu', 'distributed': True, 'workers': 16, 'clip_grad': 0.1, 'use_shared_dict': False, 'shared_dict': None, 'use_amp': False, 'aug': True, 'random_crop': False, 'random_translate': False, 'translate_prob': 0.2, 'max_translation': 100, 'validate_every': 0.25, 'log_images_every': 0.1, 'prefetch': False}


KEYS_TYPE_BOOL = ['use_amp', 'distributed', 'use_shared_dict', 'same_lr', 'aug', 'three_phase', 'prefetch', 'cycle_momentum']


def check_choices(name, value, choices):
    if value not in choices:
        raise ValueError(f'{name} {value} not in supported choices {choices}')


def flatten(config, except_keys='bin_conf'):

    def recurse(inp):
        if isinstance(inp, dict):
            for key, value in inp.items():
                if key in except_keys:
                    yield key, value
                if isinstance(value, dict):
                    yield from recurse(value)
                else:
                    yield key, value
    return dict(list(recurse(config)))


def infer_type(x):
    if not isinstance(x, str):
        return x
    try:
        x = int(x)
        return x
    except ValueError:
        pass
    try:
        x = float(x)
        return x
    except ValueError:
        pass
    return x


def parse_list(config, key, dtype=int):
    """Parse a list of values for the key if the value is a string. The values are separated by a comma. 
    Modifies the config in place.
    """
    if key in config:
        if isinstance(config[key], str):
            config[key] = list(map(dtype, config[key].split(',')))
        assert isinstance(config[key], list) and all([isinstance(e, dtype) for e in config[key]]), f'{key} should be a list of values dtype {dtype}. Given {config[key]} of type {type(config[key])} with values of type {[type(e) for e in config[key]]}.'


def split_combined_args(kwargs):
    """Splits the arguments that are combined with '__' into multiple arguments.
       Combined arguments should have equal number of keys and values.
       Keys are separated by '__' and Values are separated with ';'.
       For example, '__n_bins__lr=256;0.001'

    Args:
        kwargs (dict): key-value pairs of arguments where key-value is optionally combined according to the above format. 

    Returns:
        dict: Parsed dict with the combined arguments split into individual key-value pairs.
    """
    new_kwargs = dict(kwargs)
    for key, value in kwargs.items():
        if key.startswith('__'):
            keys = key.split('__')[1:]
            values = value.split(';')
            assert len(keys) == len(values), f"Combined arguments should have equal number of keys and values. Keys are separated by '__' and Values are separated with ';'. For example, '__n_bins__lr=256;0.001. Given (keys,values) is ({keys}, {values})"
            for k, v in zip(keys, values):
                new_kwargs[k] = v
    return new_kwargs


def get_model_config(model_name, model_version=None):
    """Find and parse the .json config file for the model.

    Args:
        model_name (str): name of the model. The config file should be named config_{model_name}[_{model_version}].json under the models/{model_name} directory.
        model_version (str, optional): Specific config version. If specified config_{model_name}_{model_version}.json is searched for and used. Otherwise config_{model_name}.json is used. Defaults to None.

    Returns:
        easydict: the config dictionary for the model.
    """
    config_fname = f'config_{model_name}_{model_version}.json' if model_version is not None else f'config_{model_name}.json'
    config_file = os.path.join(ROOT, 'models', model_name, config_fname)
    if not os.path.exists(config_file):
        return None
    with open(config_file, 'r') as f:
        config = edict(json.load(f))
    if 'inherit' in config.train and config.train.inherit is not None:
        inherit_config = get_model_config(config.train['inherit']).train
        for key, value in inherit_config.items():
            if key not in config.train:
                config.train[key] = value
    return edict(config)


def update_model_config(config, mode, model_name, model_version=None, strict=False):
    model_config = get_model_config(model_name, model_version)
    if model_config is not None:
        config = {**config, **flatten({**model_config.model, **model_config[mode]})}
    elif strict:
        raise ValueError(f'Config file for model {model_name} not found.')
    return config


def get_config(model_name, mode='train', dataset=None, **overwrite_kwargs):
    """Main entry point to get the config for the model.

    Args:
        model_name (str): name of the desired model.
        mode (str, optional): "train" or "infer". Defaults to 'train'.
        dataset (str, optional): If specified, the corresponding dataset configuration is loaded as well. Defaults to None.
    
    Keyword Args: key-value pairs of arguments to overwrite the default config.

    The order of precedence for overwriting the config is (Higher precedence first):
        # 1. overwrite_kwargs
        # 2. "config_version": Config file version if specified in overwrite_kwargs. The corresponding config loaded is config_{model_name}_{config_version}.json
        # 3. "version_name": Default Model version specific config specified in overwrite_kwargs. The corresponding config loaded is config_{model_name}_{version_name}.json
        # 4. common_config: Default config for all models specified in COMMON_CONFIG

    Returns:
        easydict: The config dictionary for the model.
    """
    check_choices('Model', model_name, ['zoedepth', 'zoedepth_nk'])
    check_choices('Mode', mode, ['train', 'infer', 'eval'])
    if mode == 'train':
        check_choices('Dataset', dataset, ['nyu', 'kitti', 'mix', None])
    config = flatten({**COMMON_CONFIG, **COMMON_TRAINING_CONFIG})
    config = update_model_config(config, mode, model_name)
    version_name = overwrite_kwargs.get('version_name', config['version_name'])
    config = update_model_config(config, mode, model_name, version_name)
    config_version = overwrite_kwargs.get('config_version', None)
    if config_version is not None:
        None
        config = update_model_config(config, mode, model_name, config_version)
    overwrite_kwargs = split_combined_args(overwrite_kwargs)
    config = {**config, **overwrite_kwargs}
    for key in KEYS_TYPE_BOOL:
        if key in config:
            config[key] = bool(config[key])
    parse_list(config, 'n_attractors')
    if 'bin_conf' in config and 'n_bins' in overwrite_kwargs:
        bin_conf = config['bin_conf']
        n_bins = overwrite_kwargs['n_bins']
        new_bin_conf = []
        for conf in bin_conf:
            conf['n_bins'] = n_bins
            new_bin_conf.append(conf)
        config['bin_conf'] = new_bin_conf
    if mode == 'train':
        orig_dataset = dataset
        if dataset == 'mix':
            dataset = 'nyu'
        if dataset is not None:
            config['project'] = f'MonoDepth3-{orig_dataset}'
    if dataset is not None:
        config['dataset'] = dataset
        config = {**DATASETS_CONFIG[dataset], **config}
    config['model'] = model_name
    typed_config = {k: infer_type(v) for k, v in config.items()}
    config['hostname'] = platform.node()
    return edict(typed_config)


class MonoDEst(nn.Module):

    def __init__(self, args):
        super(MonoDEst, self).__init__()
        if args.mde_name == 'zoedepth_nk':
            conf = get_config('zoedepth_nk', 'infer')
            model_zoe_nk = build_model(conf)
            model_zoe_nk.eval()
            model_zoe_nk = model_zoe_nk
            self.model = model_zoe_nk
        elif args.mde_name == 'zoedepth_k':
            conf = get_config('zoedepth', 'infer', config_version='kitti')
            model_zoe_k = build_model(conf)
            model_zoe_k.eval()
            model_zoe_k = model_zoe_k
            self.model = model_zoe_k
        elif args.mde_name == 'depthAny':
            cfg = edict({'encoder': 'vits', 'load_from': 'models/monoD/depth_anything/ckpts/depth_anything_vits14.pth', 'localhub': True})
            self.model = DepthAnything(cfg)
            conf = get_config('zoedepth_nk', 'infer')
            model_zoe_nk = build_model(conf)
            model_zoe_nk.eval()
            model_zoe_nk = model_zoe_nk
            self.metric3d = model_zoe_nk
        self.mde_name = args.mde_name

    def infer(self, rgbs, scale=None, shift=None):
        """
            Infer the depth map from the input RGB image
        """
        if self.mde_name == 'depthAny':
            depth_map = self.model.infer(rgbs)
            metric_dp = self.metric3d.infer(rgbs[:20])
            metric_dp_inv = 1 / metric_dp
            dp_0_rel = depth_map[:20]
            scale, shift = np.polyfit(dp_0_rel.view(-1).cpu().numpy(), metric_dp_inv.view(-1).cpu().numpy(), 1)
            depth_map = depth_map * scale + shift
            depth_map = (1 / depth_map).clamp(0.01, 65)
        else:
            depth_map = self.model.infer(rgbs)
        return depth_map


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, padding_mode='zeros')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, padding_mode='zeros')
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()
        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):

    def __init__(self, input_dim=3, output_dim=128, stride=8, norm_fn='batch', dropout=0.0, Embed3D=False):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn
        self.in_planes = 64
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim * 2)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(output_dim * 2)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim * 2)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode='zeros')
        self.relu1 = nn.ReLU(inplace=True)
        self.shallow = False
        if self.shallow:
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.conv2 = nn.Conv2d(128 + 96 + 64, output_dim, kernel_size=1)
        else:
            if Embed3D:
                self.conv_fuse = nn.Conv2d(64 + 63, self.in_planes, kernel_size=3, padding=1)
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.layer4 = self._make_layer(128, stride=2)
            self.conv2 = nn.Conv2d(128 + 128 + 96 + 64, output_dim * 2, kernel_size=3, padding=1, padding_mode='zeros')
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = layer1, layer2
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, feat_PE=None):
        _, _, H, W = x.shape
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        if self.shallow:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            a = F.interpolate(a, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            b = F.interpolate(b, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            c = F.interpolate(c, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            x = self.conv2(torch.cat([a, b, c], dim=1))
        else:
            if feat_PE is not None:
                x = self.conv_fuse(torch.cat([x, feat_PE], dim=1))
                a = self.layer1(x)
            else:
                a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            d = self.layer4(c)
            a = F.interpolate(a, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            b = F.interpolate(b, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            c = F.interpolate(c, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            d = F.interpolate(d, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            x = self.conv2(torch.cat([a, b, c, d], dim=1))
            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv3(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim: 'int', num_heads: 'int'=8, qkv_bias: 'bool'=False, proj_bias: 'bool'=True, attn_drop: 'float'=0.0, proj_drop: 'float'=0.0) ->None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: 'Tensor') ->Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features: 'int', hidden_features: 'Optional[int]'=None, out_features: 'Optional[int]'=None, act_layer: 'Callable[..., nn.Module]'=nn.GELU, drop: 'float'=0.0, bias: 'bool'=True) ->None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttnBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, flash=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.flash = flash
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, flash=flash, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda : nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class UpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(self, space_depth=12, time_depth=12, input_dim=320, hidden_size=384, num_heads=8, output_dim=130, mlp_ratio=4.0, add_space_attn=True):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        self.time_blocks = nn.ModuleList([AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(time_depth)])
        if add_space_attn:
            self.space_blocks = nn.ModuleList([AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(space_depth)])
            assert len(self.time_blocks) >= len(self.space_blocks)
        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, input_tensor):
        x = self.input_transform(input_tensor)
        j = 0
        for i in range(len(self.time_blocks)):
            B, N, T, _ = x.shape
            x_time = rearrange(x, 'b n t c -> (b n) t c', b=B, t=T, n=N)
            x_time = self.time_blocks[i](x_time)
            x = rearrange(x_time, '(b n) t c -> b n t c ', b=B, t=T, n=N)
            if self.add_space_attn and i % (len(self.time_blocks) // len(self.space_blocks)) == 0:
                x_space = rearrange(x, 'b n t c -> (b t) n c ', b=B, t=T, n=N)
                x_space = self.space_blocks[j](x_space)
                x = rearrange(x_space, '(b t) n c -> b n t c  ', b=B, t=T, n=N)
                j += 1
        flow = self.flow_head(x)
        return flow


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


class CorrBlock:

    def __init__(self, fmaps, num_levels=4, radius=4, depths_dnG=None):
        B, S, C, H_prev, W_prev = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H_prev, W_prev
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.depth_pyramid = []
        self.fmaps_pyramid.append(fmaps)
        if depths_dnG is not None:
            self.depth_pyramid.append(depths_dnG)
        for i in range(self.num_levels - 1):
            if depths_dnG is not None:
                depths_dnG_ = depths_dnG.reshape(B * S, 1, H_prev, W_prev)
                depths_dnG_ = F.avg_pool2d(depths_dnG_, 2, stride=2)
                _, _, H, W = depths_dnG_.shape
                depths_dnG = depths_dnG_.reshape(B, S, 1, H, W)
                self.depth_pyramid.append(depths_dnG)
            fmaps_ = fmaps.reshape(B * S, C, H_prev, W_prev)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            H_prev = H
            W_prev = W
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2
        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]
            _, _, _, H, W = corrs.shape
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)
            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            corrs = bilinear_sampler(corrs.reshape(B * S * N, 1, H, W), coords_lvl)
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)
        out = torch.cat(out_pyramid, dim=-1)
        return out.contiguous().float()

    def corr(self, targets):
        B, S, N, C = targets.shape
        assert C == self.C
        assert S == self.S
        fmap1 = targets
        self.corrs_pyramid = []
        for fmaps in self.fmaps_pyramid:
            _, _, _, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)

    def corr_sample(self, targets, coords, coords_dp=None):
        B, S, N, C = targets.shape
        r = self.radius
        Dim_c = (2 * r + 1) ** 2
        assert C == self.C
        assert S == self.S
        out_pyramid = []
        out_pyramid_dp = []
        for i in range(self.num_levels):
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)
            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            fmaps = self.fmaps_pyramid[i]
            _, _, _, H, W = fmaps.shape
            fmap2s = fmaps.view(B * S, C, H, W)
            if len(self.depth_pyramid) > 0:
                depths_dnG_i = self.depth_pyramid[i]
                depths_dnG_i = depths_dnG_i.view(B * S, 1, H, W)
                dnG_sample = bilinear_sampler(depths_dnG_i, coords_lvl.view(B * S, 1, N * Dim_c, 2))
                dp_corrs = (dnG_sample.view(B * S, N, -1) - coords_dp[0]).abs() / coords_dp[0]
                out_pyramid_dp.append(dp_corrs)
            fmap2s_sample = bilinear_sampler(fmap2s, coords_lvl.view(B * S, 1, N * Dim_c, 2))
            fmap2s_sample = fmap2s_sample.permute(0, 3, 1, 2)
            corrs = torch.matmul(targets.reshape(B * S * N, 1, -1), fmap2s_sample.reshape(B * S * N, Dim_c, -1).permute(0, 2, 1))
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)
        out = torch.cat(out_pyramid, dim=-1)
        if len(self.depth_pyramid) > 0:
            out_dp = torch.cat(out_pyramid_dp, dim=-1)
            self.fcorrD = out_dp.contiguous().float()
        else:
            self.fcorrD = torch.zeros_like(out).contiguous().float()
        return out.contiguous().float()


def bilinear_sample2d(im, x, y, return_inbounds=False):
    B, C, H, W = list(im.shape)
    N = list(x.shape)[1]
    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)
    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()
    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H
    base = torch.arange(0, B, dtype=torch.int64, device=x.device) * dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])
    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2
    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip
    im_flat = im.permute(0, 2, 3, 1).reshape(B * H * W, C)
    i_y0_x0 = im_flat[idx_y0_x0.long()]
    i_y0_x1 = im_flat[idx_y0_x1.long()]
    i_y1_x0 = im_flat[idx_y1_x0.long()]
    i_y1_x1 = im_flat[idx_y1_x1.long()]
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()
    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)
    output = w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    if return_inbounds:
        x_valid = (x > -0.5).byte() & (x < float(W_f - 0.5)).byte()
        y_valid = (y > -0.5).byte() & (y < float(H_f - 0.5)).byte()
        inbounds = (x_valid & y_valid).float()
        inbounds = inbounds.reshape(B, N)
        return output, inbounds
    return output


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_embedding(xy, C, cat_coords=True):
    B, N, D = xy.shape
    assert D == 2
    x = xy[:, :, 0:1]
    y = xy[:, :, 1:2]
    div_term = (torch.arange(0, C, 2, device=xy.device, dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C / 2))
    pe_x = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)
    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)
    pe = torch.cat([pe_x, pe_y], dim=2)
    if cat_coords:
        pe = torch.cat([xy, pe], dim=2)
    return pe


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, tuple):
        grid_size_h, grid_size_w = grid_size
    else:
        grid_size_h = grid_size_w = grid_size
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0
    B, S, N, _ = grid.shape
    gridx = grid[..., 0].view(B * S * N).detach().cpu().numpy()
    gridy = grid[..., 1].view(B * S * N).detach().cpu().numpy()
    gridz = grid[..., 2].view(B * S * N).detach().cpu().numpy()
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, gridx)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, gridy)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, gridz)
    emb = np.concatenate([emb_h, emb_w, emb_z], axis=1)
    emb = torch.from_numpy(emb)
    return emb.view(B, S, N, embed_dim)


def sample_pos_embed(grid_size, embed_dim, coords):
    if coords.shape[-1] == 2:
        pos_embed = get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size)
        pos_embed = torch.from_numpy(pos_embed).reshape(grid_size[0], grid_size[1], embed_dim).float().unsqueeze(0)
        sampled_pos_embed = bilinear_sample2d(pos_embed.permute(0, 3, 1, 2), coords[:, 0, :, 0], coords[:, 0, :, 1])
    elif coords.shape[-1] == 3:
        sampled_pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, coords[:, :1, ...]).float()[:, 0, ...].permute(0, 2, 1)
    return sampled_pos_embed


def smart_cat(tensor1, tensor2, dim):
    if tensor1 is None:
        return tensor2
    return torch.cat([tensor1, tensor2], dim=dim)


def vis_PCA(fmaps, save_dir):
    """
        visualize the PCA of the feature maps
    args:
        fmaps: feature maps  1 C H W
        save_dir: the directory to save the PCA visualization
    """
    pca = PCA(n_components=3)
    fmap_vis = fmaps[0, ...]
    fmap_vnorm = (fmap_vis - fmap_vis.min()) / (fmap_vis.max() - fmap_vis.min())
    H_vis, W_vis = fmap_vis.shape[1:]
    fmap_vnorm = fmap_vnorm.reshape(fmap_vnorm.shape[0], -1).permute(1, 0)
    fmap_pca = pca.fit_transform(fmap_vnorm.detach().cpu().numpy())
    pca = fmap_pca.reshape(H_vis, W_vis, 3)
    plt.imsave(save_dir, (pca - pca.min()) / (pca.max() - pca.min()))


class CoTracker(nn.Module):

    def __init__(self, S=8, stride=8, add_space_attn=True, num_heads=8, hidden_size=384, space_depth=12, time_depth=12):
        super(CoTracker, self).__init__()
        self.S = S
        self.stride = stride
        self.hidden_dim = 256
        self.latent_dim = latent_dim = 128
        self.corr_levels = 4
        self.corr_radius = 3
        self.add_space_attn = add_space_attn
        self.fnet = BasicEncoder(output_dim=self.latent_dim, norm_fn='instance', dropout=0, stride=stride)
        self.updateformer = UpdateFormer(space_depth=space_depth, time_depth=time_depth, input_dim=456, hidden_size=hidden_size, num_heads=num_heads, output_dim=latent_dim + 2, mlp_ratio=4.0, add_space_attn=add_space_attn)
        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.ffeat_updater = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.GELU())
        self.vis_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

    def forward_iteration(self, fmaps, coords_init, feat_init=None, vis_init=None, track_mask=None, iters=4):
        B, S_init, N, D = coords_init.shape
        assert D == 2
        assert B == 1
        B, S, __, H8, W8 = fmaps.shape
        device = fmaps.device
        if S_init < S:
            coords = torch.cat([coords_init, coords_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1)
            vis_init = torch.cat([vis_init, vis_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1)
        else:
            coords = coords_init.clone()
        fcorr_fn = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)
        ffeats = feat_init.clone()
        times_ = torch.linspace(0, S - 1, S).reshape(1, S, 1)
        pos_embed = sample_pos_embed(grid_size=(H8, W8), embed_dim=456, coords=coords)
        pos_embed = rearrange(pos_embed, 'b e n -> (b n) e').unsqueeze(1)
        times_embed = torch.from_numpy(get_1d_sincos_pos_embed_from_grid(456, times_[0]))[None].repeat(B, 1, 1).float()
        coord_predictions = []
        debug = False
        if debug == True:
            import matplotlib.pyplot as plt
            pcd_idx = 43
            vis_PCA(fmaps[0, :1], './xy.png')
            vis_PCA(fmaps[0, -1:], './xy_.png')
            fxy_q = feat_init[0, 0, pcd_idx:pcd_idx + 1, :, None, None]
            corr_map = (fxy_q * fmaps[0, -1:]).sum(dim=1)
            argmax_idx = torch.argmax(corr_map.view(-1))
            x_max, y_max = argmax_idx % corr_map.shape[-1], argmax_idx // corr_map.shape[-1]
            coord_last = coords[0, -1, pcd_idx:pcd_idx + 1]
            corr_map_vis = corr_map[0].detach().cpu().numpy()
            img_feat_new = cv2.imread('./xy_.png')
            cv2.circle(img_feat_new, (int(x_max), int(y_max)), 3, (0, 255, 0), -1)
            plt.imsave('./corr_map.png', corr_map_vis)
            img_feat = cv2.imread('./xy.png')
            cv2.circle(img_feat, (int(coord_last[0, 0]), int(coord_last[0, 1])), 2, (0, 0, 255), -1)
            cv2.imwrite('./xy_coord.png', img_feat)
            ipdb.set_trace()
        for __ in range(iters):
            coords = coords.detach()
            fcorr_fn.corr(ffeats)
            fcorrs = fcorr_fn.sample(coords)
            LRR = fcorrs.shape[3]
            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)
            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 2)
            flows_cat = get_2d_embedding(flows_, 64, cat_coords=True)
            ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)
            if track_mask.shape[1] < vis_init.shape[1]:
                track_mask = torch.cat([track_mask, torch.zeros_like(track_mask[:, 0]).repeat(1, vis_init.shape[1] - track_mask.shape[1], 1, 1)], dim=1)
            concat = torch.cat([track_mask, vis_init], dim=2).permute(0, 2, 1, 3).reshape(B * N, S, 2)
            transformer_input = torch.cat([flows_cat, fcorrs_, ffeats_, concat], dim=2)
            x = transformer_input + pos_embed + times_embed
            x = rearrange(x, '(b n) t d -> b n t d', b=B)
            delta = self.updateformer(x)
            delta = rearrange(delta, ' b n t d -> (b n) t d')
            delta_coords_ = delta[:, :, :2]
            delta_feats_ = delta[:, :, 2:]
            delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)
            ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N * S, self.latent_dim)
            ffeats_ = self.ffeat_updater(self.norm(delta_feats_)) + ffeats_
            ffeats = ffeats_.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)
            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0, 2, 1, 3)
            coord_predictions.append(coords * self.stride)
        vis_e = self.vis_predictor(ffeats.reshape(B * S * N, self.latent_dim)).reshape(B, S, N)
        return coord_predictions, vis_e, feat_init

    def forward(self, rgbs, queries, iters=4, feat_init=None, is_train=False):
        B, T, C, H, W = rgbs.shape
        B, N, __ = queries.shape
        device = rgbs.device
        assert B == 1
        first_positive_inds = queries[:, :, 0].long()
        __, sort_inds = torch.sort(first_positive_inds[0], dim=0, descending=False)
        inv_sort_inds = torch.argsort(sort_inds, dim=0)
        first_positive_sorted_inds = first_positive_inds[0][sort_inds]
        assert torch.allclose(first_positive_inds[0], first_positive_inds[0][sort_inds][inv_sort_inds])
        coords_init = queries[:, :, 1:].reshape(B, 1, N, 2).repeat(1, self.S, 1, 1) / float(self.stride)
        rgbs = 2 * (rgbs / 255.0) - 1.0
        traj_e = torch.zeros((B, T, N, 2), device=device)
        vis_e = torch.zeros((B, T, N), device=device)
        ind_array = torch.arange(T, device=device)
        ind_array = ind_array[None, :, None].repeat(B, 1, N)
        track_mask = (ind_array >= first_positive_inds[:, None, :]).unsqueeze(-1)
        vis_init = torch.ones((B, self.S, N, 1), device=device).float() * 10
        ind = 0
        track_mask_ = track_mask[:, :, sort_inds].clone()
        coords_init_ = coords_init[:, :, sort_inds].clone()
        vis_init_ = vis_init[:, :, sort_inds].clone()
        prev_wind_idx = 0
        fmaps_ = None
        vis_predictions = []
        coord_predictions = []
        wind_inds = []
        while ind < T - self.S // 2:
            rgbs_seq = rgbs[:, ind:ind + self.S]
            S = S_local = rgbs_seq.shape[1]
            if S < self.S:
                rgbs_seq = torch.cat([rgbs_seq, rgbs_seq[:, -1, None].repeat(1, self.S - S, 1, 1, 1)], dim=1)
                S = rgbs_seq.shape[1]
            rgbs_ = rgbs_seq.reshape(B * S, C, H, W)
            if fmaps_ is None:
                fmaps_ = self.fnet(rgbs_)
            else:
                fmaps_ = torch.cat([fmaps_[self.S // 2:], self.fnet(rgbs_[self.S // 2:])], dim=0)
            fmaps = fmaps_.reshape(B, S, self.latent_dim, H // self.stride, W // self.stride)
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            None
            curr_wind_points = torch.nonzero(first_positive_sorted_inds < ind + self.S)
            if curr_wind_points.shape[0] == 0:
                ind = ind + self.S // 2
                continue
            wind_idx = curr_wind_points[-1] + 1
            if prev_wind_idx > 0:
                new_coords = coords[-1][:, self.S // 2:] / float(self.stride)
                coords_init_[:, :self.S // 2, :prev_wind_idx] = new_coords
                coords_init_[:, self.S // 2:, :prev_wind_idx] = new_coords[:, -1].repeat(1, self.S // 2, 1, 1)
                new_vis = vis[:, self.S // 2:].unsqueeze(-1)
                vis_init_[:, :self.S // 2, :prev_wind_idx] = new_vis
                vis_init_[:, self.S // 2:, :prev_wind_idx] = new_vis[:, -1].repeat(1, self.S // 2, 1, 1)
            if wind_idx - prev_wind_idx > 0:
                fmaps_sample = fmaps[:, first_positive_sorted_inds[prev_wind_idx:wind_idx] - ind]
                feat_init_ = bilinear_sample2d(fmaps_sample, coords_init_[:, 0, prev_wind_idx:wind_idx, 0], coords_init_[:, 0, prev_wind_idx:wind_idx, 1]).permute(0, 2, 1)
                feat_init_ = feat_init_.unsqueeze(1).repeat(1, self.S, 1, 1)
                feat_init = smart_cat(feat_init, feat_init_, dim=2)
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            None
            coords, vis, __ = self.forward_iteration(fmaps=fmaps, coords_init=coords_init_[:, :, :wind_idx], feat_init=feat_init[:, :, :wind_idx], vis_init=vis_init_[:, :, :wind_idx], track_mask=track_mask_[:, ind:ind + self.S, :wind_idx], iters=iters)
            if is_train:
                vis_predictions.append(torch.sigmoid(vis[:, :S_local]))
                coord_predictions.append([coord[:, :S_local] for coord in coords])
                wind_inds.append(wind_idx)
            traj_e[:, ind:ind + self.S, :wind_idx] = coords[-1][:, :S_local]
            vis_e[:, ind:ind + self.S, :wind_idx] = vis[:, :S_local]
            track_mask_[:, :ind + self.S, :wind_idx] = 0.0
            ind = ind + self.S // 2
            prev_wind_idx = wind_idx
        traj_e = traj_e[:, :, inv_sort_inds]
        vis_e = vis_e[:, :, inv_sort_inds]
        vis_e = torch.sigmoid(vis_e)
        train_data = (vis_predictions, coord_predictions, wind_inds, sort_inds) if is_train else None
        return traj_e, feat_init, vis_e, train_data


def normalize_grid2d(grid_y, grid_x, Y, X, clamp_extreme=True):
    grid_y = 2.0 * (grid_y / float(Y - 1)) - 1.0
    grid_x = 2.0 * (grid_x / float(X - 1)) - 1.0
    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
    return grid_y, grid_x


def meshgrid2d(B, Y, X, stack=False, norm=False, device='cuda', on_chans=False):
    grid_y = torch.linspace(0.0, Y - 1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)
    grid_x = torch.linspace(0.0, X - 1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)
    if norm:
        grid_y, grid_x = normalize_grid2d(grid_y, grid_x, Y, X)
    if stack:
        if on_chans:
            grid = torch.stack([grid_x, grid_y], dim=1)
        else:
            grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x


def get_points_on_a_grid(grid_size, interp_shape, grid_center=(0, 0), device='cuda'):
    if grid_size == 1:
        return torch.tensor([interp_shape[1] / 2, interp_shape[0] / 2], device=device)[None, None]
    grid_y, grid_x = meshgrid2d(1, grid_size, grid_size, stack=False, norm=False, device=device)
    step = interp_shape[1] // 64
    if grid_center[0] != 0 or grid_center[1] != 0:
        grid_y = grid_y - grid_size / 2.0
        grid_x = grid_x - grid_size / 2.0
    grid_y = step + grid_y.reshape(1, -1) / float(grid_size - 1) * (interp_shape[0] - step * 2)
    grid_x = step + grid_x.reshape(1, -1) / float(grid_size - 1) * (interp_shape[1] - step * 2)
    grid_y = grid_y + grid_center[0]
    grid_x = grid_x + grid_center[1]
    xy = torch.stack([grid_x, grid_y], dim=-1)
    return xy


class EvaluationPredictor(torch.nn.Module):

    def __init__(self, cotracker_model: 'CoTracker', interp_shape: 'Tuple[int, int]'=(384, 512), grid_size: 'int'=6, local_grid_size: 'int'=6, single_point: 'bool'=True, n_iters: 'int'=6) ->None:
        super(EvaluationPredictor, self).__init__()
        self.grid_size = grid_size
        self.local_grid_size = local_grid_size
        self.single_point = single_point
        self.interp_shape = interp_shape
        self.n_iters = n_iters
        self.model = cotracker_model
        self.model.eval()

    def forward(self, video, queries):
        queries = queries.clone()
        B, T, C, H, W = video.shape
        B, N, D = queries.shape
        assert D == 3
        assert B == 1
        rgbs = video.reshape(B * T, C, H, W)
        rgbs = F.interpolate(rgbs, tuple(self.interp_shape), mode='bilinear')
        rgbs = rgbs.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])
        device = rgbs.device
        queries[:, :, 1] *= self.interp_shape[1] / W
        queries[:, :, 2] *= self.interp_shape[0] / H
        if self.single_point:
            traj_e = torch.zeros((B, T, N, 2), device=device)
            vis_e = torch.zeros((B, T, N), device=device)
            for pind in range(N):
                query = queries[:, pind:pind + 1]
                t = query[0, 0, 0].long()
                traj_e_pind, vis_e_pind = self._process_one_point(rgbs, query)
                traj_e[:, t:, pind:pind + 1] = traj_e_pind[:, :, :1]
                vis_e[:, t:, pind:pind + 1] = vis_e_pind[:, :, :1]
        else:
            if self.grid_size > 0:
                xy = get_points_on_a_grid(self.grid_size, rgbs.shape[3:], device=device)
                xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2)
                queries = torch.cat([queries, xy], dim=1)
            traj_e, __, vis_e, __ = self.model(rgbs=rgbs, queries=queries, iters=self.n_iters)
        traj_e[:, :, :, 0] *= W / float(self.interp_shape[1])
        traj_e[:, :, :, 1] *= H / float(self.interp_shape[0])
        return traj_e, vis_e

    def _process_one_point(self, rgbs, query):
        t = query[0, 0, 0].long()
        device = rgbs.device
        if self.local_grid_size > 0:
            xy_target = get_points_on_a_grid(self.local_grid_size, (50, 50), [query[0, 0, 2], query[0, 0, 1]])
            xy_target = torch.cat([torch.zeros_like(xy_target[:, :, :1]), xy_target], dim=2)
            query = torch.cat([query, xy_target], dim=1)
        if self.grid_size > 0:
            xy = get_points_on_a_grid(self.grid_size, rgbs.shape[3:], device=device)
            xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2)
            query = torch.cat([query, xy], dim=1)
        query[0, 0, 0] = 0
        traj_e_pind, __, vis_e_pind, __ = self.model(rgbs=rgbs[:, t:], queries=query, iters=self.n_iters)
        return traj_e_pind, vis_e_pind


def _build_cotracker(stride, sequence_len, checkpoint=None):
    cotracker = CoTracker(stride=stride, S=sequence_len, add_space_attn=True, space_depth=6, time_depth=6)
    if checkpoint is not None:
        with open(checkpoint, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
        cotracker.load_state_dict(state_dict)
    return cotracker


def build_cotracker_stride_4_wind_12(checkpoint=None):
    return _build_cotracker(stride=4, sequence_len=12, checkpoint=checkpoint)


def build_cotracker_stride_4_wind_8(checkpoint=None):
    return _build_cotracker(stride=4, sequence_len=8, checkpoint=checkpoint)


def build_cotracker_stride_8_wind_16(checkpoint=None):
    return _build_cotracker(stride=8, sequence_len=16, checkpoint=checkpoint)


def build_cotracker(checkpoint: 'str'):
    if checkpoint is None:
        return build_cotracker_stride_4_wind_8()
    model_name = checkpoint.split('/')[-1].split('.')[0]
    if model_name == 'cotracker_stride_4_wind_8':
        return build_cotracker_stride_4_wind_8(checkpoint=checkpoint)
    elif model_name == 'cotracker_stride_4_wind_12':
        return build_cotracker_stride_4_wind_12(checkpoint=checkpoint)
    elif model_name == 'cotracker_stride_8_wind_16':
        return build_cotracker_stride_8_wind_16(checkpoint=checkpoint)
    elif 'model' in model_name:
        return build_cotracker_stride_4_wind_8(checkpoint=checkpoint)
    else:
        raise ValueError(f'Unknown model name {model_name}')


class CoTrackerPredictor(torch.nn.Module):

    def __init__(self, checkpoint='cotracker/checkpoints/cotracker_stride_4_wind_8.pth'):
        super().__init__()
        self.interp_shape = 384, 512
        self.support_grid_size = 6
        model = build_cotracker(checkpoint)
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(self, video, queries: 'torch.Tensor'=None, segm_mask: 'torch.Tensor'=None, grid_size: 'int'=0, grid_query_frame: 'int'=0, backward_tracking: 'bool'=False):
        if queries is None and grid_size == 0:
            tracks, visibilities = self._compute_dense_tracks(video, grid_query_frame=grid_query_frame, backward_tracking=backward_tracking)
        else:
            tracks, visibilities = self._compute_sparse_tracks(video, queries, segm_mask, grid_size, add_support_grid=grid_size == 0 or segm_mask is not None, grid_query_frame=grid_query_frame, backward_tracking=backward_tracking)
        return tracks, visibilities

    def _compute_dense_tracks(self, video, grid_query_frame, grid_size=30, backward_tracking=False):
        *_, H, W = video.shape
        grid_step = W // grid_size
        grid_width = W // grid_step
        grid_height = H // grid_step
        tracks = visibilities = None
        grid_pts = torch.zeros((1, grid_width * grid_height, 3))
        grid_pts[0, :, 0] = grid_query_frame
        for offset in tqdm(range(grid_step * grid_step)):
            ox = offset % grid_step
            oy = offset // grid_step
            grid_pts[0, :, 1] = torch.arange(grid_width).repeat(grid_height) * grid_step + ox
            grid_pts[0, :, 2] = torch.arange(grid_height).repeat_interleave(grid_width) * grid_step + oy
            tracks_step, visibilities_step = self._compute_sparse_tracks(video=video, queries=grid_pts, backward_tracking=backward_tracking)
            tracks = smart_cat(tracks, tracks_step, dim=2)
            visibilities = smart_cat(visibilities, visibilities_step, dim=2)
        return tracks, visibilities

    def _compute_sparse_tracks(self, video, queries, segm_mask=None, grid_size=0, add_support_grid=False, grid_query_frame=0, backward_tracking=False):
        B, T, C, H, W = video.shape
        assert B == 1
        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode='bilinear')
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])
        if queries is not None:
            queries = queries.clone()
            B, N, D = queries.shape
            assert D == 3
            queries[:, :, 1] *= self.interp_shape[1] / W
            queries[:, :, 2] *= self.interp_shape[0] / H
        elif grid_size > 0:
            grid_pts = get_points_on_a_grid(grid_size, self.interp_shape, device=video.device)
            if segm_mask is not None:
                segm_mask = F.interpolate(segm_mask, tuple(self.interp_shape), mode='nearest')
                point_mask = segm_mask[0, 0][grid_pts[0, :, 1].round().long().cpu(), grid_pts[0, :, 0].round().long().cpu()].bool()
                grid_pts = grid_pts[:, point_mask]
            queries = torch.cat([torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts], dim=2)
        if add_support_grid:
            grid_pts = get_points_on_a_grid(self.support_grid_size, self.interp_shape, device=video.device)
            grid_pts = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)
            queries = torch.cat([queries, grid_pts], dim=1)
        tracks, __, visibilities, __ = self.model(rgbs=video, queries=queries, iters=6)
        if backward_tracking:
            tracks, visibilities = self._compute_backward_tracks(video, queries, tracks, visibilities)
            if add_support_grid:
                queries[:, -self.support_grid_size ** 2:, 0] = T - 1
        if add_support_grid:
            tracks = tracks[:, :, :-self.support_grid_size ** 2]
            visibilities = visibilities[:, :, :-self.support_grid_size ** 2]
        thr = 0.9
        visibilities = visibilities > thr
        for i in range(len(queries)):
            queries_t = queries[i, :tracks.size(2), 0]
            arange = torch.arange(0, len(queries_t))
            tracks[i, queries_t, arange] = queries[i, :tracks.size(2), 1:]
            visibilities[i, queries_t, arange] = True
        tracks[:, :, :, 0] *= W / float(self.interp_shape[1])
        tracks[:, :, :, 1] *= H / float(self.interp_shape[0])
        return tracks, visibilities

    def _compute_backward_tracks(self, video, queries, tracks, visibilities):
        inv_video = video.flip(1).clone()
        inv_queries = queries.clone()
        inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1
        inv_tracks, __, inv_visibilities, __ = self.model(rgbs=inv_video, queries=inv_queries, iters=6)
        inv_tracks = inv_tracks.flip(1)
        inv_visibilities = inv_visibilities.flip(1)
        mask = tracks == 0
        tracks[mask] = inv_tracks[mask]
        visibilities[mask[:, :, :, 0]] = inv_visibilities[mask[:, :, :, 0]]
        return tracks, visibilities


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
        return output


class DPTHead(nn.Module):

    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=1, stride=1, padding=0) for out_channel in out_channels])
        self.resize_layers = nn.ModuleList([nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0), nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0), nn.Identity(), nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1)])
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()))
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        head_features_1 = features
        head_features_2 = 32
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1), nn.ReLU(True), nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0))
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            self.scratch.output_conv2 = nn.Sequential(nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1), nn.ReLU(True), nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0), nn.ReLU(True), nn.Identity())

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode='bilinear', align_corners=True)
        out = self.scratch.output_conv2(out)
        return out


class ConvTransposeNorm(nn.Sequential):
    """
    Modification of
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/levit.py: ConvNorm
    such that ConvTranspose2d is used instead of Conv2d.
    """

    def __init__(self, in_chs, out_chs, kernel_size=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.ConvTranspose2d(in_chs, out_chs, kernel_size, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_chs))
        nn.init.constant_(self.bn.weight, bn_weight_init)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.ConvTranspose2d(w.size(1), w.size(0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Slice(nn.Module):

    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class AddReadout(nn.Module):

    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):

    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)
        return self.project(features)


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


class BaseModel(torch.nn.Module):

    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))
        if 'optimizer' in parameters:
            parameters = parameters['model']
        self.load_state_dict(parameters)


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4
    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
    return _make_resnet_backbone(resnet)


def _make_pretrained_vit_tiny(pretrained, use_readout='ignore', hooks=None, use_vit_only=False, enable_attention_hooks=False):
    model = timm.create_model('vit_tiny_r_s16_p8_384', pretrained=pretrained)
    ipdb.set_trace()
    hooks = [0, 1, 8, 11] if hooks == None else hooks
    return _make_vit_tiny_backbone(model, features=[256, 512, 768, 768], size=[384, 384], hooks=hooks, use_vit_only=use_vit_only, use_readout=use_readout, enable_attention_hooks=enable_attention_hooks)


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = posemb[:, :self.start_index], posemb[0, self.start_index:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


activations = {}


attention = {}


def forward_flex(self, x):
    b, c, h, w = x.shape
    pos_embed = self._resize_pos_embed(self.pos_embed, h // self.patch_size[1], w // self.patch_size[0])
    B = x.shape[0]
    if hasattr(self.patch_embed, 'backbone'):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
    if getattr(self, 'dist_token', None) is not None:
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
    x = x + pos_embed
    x = self.pos_drop(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    return x


def get_activation(name):

    def hook(model, input, output):
        activations[name] = output
    return hook


def get_attention(name):

    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * module.scale
        attn = attn.softmax(dim=-1)
        attention[name] = attn
    return hook


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == 'ignore':
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == 'add':
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == 'project':
        readout_oper = [ProjectReadout(vit_features, start_index) for out_feat in features]
    else:
        assert False, "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"
    return readout_oper


def _make_vit_b16_backbone(model, features=[96, 192, 384, 768], size=[384, 384], hooks=[2, 5, 8, 11], vit_features=768, use_readout='ignore', start_index=1, enable_attention_hooks=False):
    pretrained = nn.Module()
    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation('1'))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation('2'))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation('3'))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation('4'))
    pretrained.activations = activations
    if enable_attention_hooks:
        pretrained.model.blocks[hooks[0]].attn.register_forward_hook(get_attention('attn_1'))
        pretrained.model.blocks[hooks[1]].attn.register_forward_hook(get_attention('attn_2'))
        pretrained.model.blocks[hooks[2]].attn.register_forward_hook(get_attention('attn_3'))
        pretrained.model.blocks[hooks[3]].attn.register_forward_hook(get_attention('attn_4'))
        pretrained.attention = attention
    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)
    pretrained.act_postprocess1 = nn.Sequential(readout_oper[0], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=vit_features, out_channels=features[0], kernel_size=1, stride=1, padding=0), nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0], kernel_size=4, stride=4, padding=0, bias=True, dilation=1, groups=1))
    pretrained.act_postprocess2 = nn.Sequential(readout_oper[1], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=vit_features, out_channels=features[1], kernel_size=1, stride=1, padding=0), nn.ConvTranspose2d(in_channels=features[1], out_channels=features[1], kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1))
    pretrained.act_postprocess3 = nn.Sequential(readout_oper[2], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=vit_features, out_channels=features[2], kernel_size=1, stride=1, padding=0))
    pretrained.act_postprocess4 = nn.Sequential(readout_oper[3], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=vit_features, out_channels=features[3], kernel_size=1, stride=1, padding=0), nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=2, padding=1))
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed, pretrained.model)
    return pretrained


def _make_pretrained_vitb16_384(pretrained, use_readout='ignore', hooks=None, enable_attention_hooks=False):
    model = timm.create_model('vit_base_patch16_384', pretrained=pretrained)
    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(model, features=[96, 192, 384, 768], hooks=hooks, use_readout=use_readout, enable_attention_hooks=enable_attention_hooks)


def _make_vit_b_rn50_backbone(model, features=[256, 512, 768, 768], size=[384, 384], hooks=[0, 1, 8, 11], vit_features=384, use_vit_only=False, use_readout='ignore', start_index=1, enable_attention_hooks=False):
    pretrained = nn.Module()
    pretrained.model = model
    pretrained.model.patch_size = [32, 32]
    ps = pretrained.model.patch_size[0]
    if use_vit_only == True:
        pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation('1'))
        pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation('2'))
    else:
        pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(get_activation('1'))
        pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(get_activation('2'))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation('3'))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation('4'))
    if enable_attention_hooks:
        pretrained.model.blocks[2].attn.register_forward_hook(get_attention('attn_1'))
        pretrained.model.blocks[5].attn.register_forward_hook(get_attention('attn_2'))
        pretrained.model.blocks[8].attn.register_forward_hook(get_attention('attn_3'))
        pretrained.model.blocks[11].attn.register_forward_hook(get_attention('attn_4'))
        pretrained.attention = attention
    pretrained.activations = activations
    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)
    if use_vit_only == True:
        pretrained.act_postprocess1 = nn.Sequential(readout_oper[0], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // ps, size[1] // ps])), nn.Conv2d(in_channels=vit_features, out_channels=features[0], kernel_size=1, stride=1, padding=0), nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0], kernel_size=4, stride=4, padding=0, bias=True, dilation=1, groups=1))
        pretrained.act_postprocess2 = nn.Sequential(readout_oper[1], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // ps, size[1] // ps])), nn.Conv2d(in_channels=vit_features, out_channels=features[1], kernel_size=1, stride=1, padding=0), nn.ConvTranspose2d(in_channels=features[1], out_channels=features[1], kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1))
    else:
        pretrained.act_postprocess1 = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity())
        pretrained.act_postprocess2 = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity())
    pretrained.act_postprocess3 = nn.Sequential(readout_oper[2], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // ps, size[1] // ps])), nn.Conv2d(in_channels=vit_features, out_channels=features[2], kernel_size=1, stride=1, padding=0))
    pretrained.act_postprocess4 = nn.Sequential(readout_oper[3], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // ps, size[1] // ps])), nn.Conv2d(in_channels=vit_features, out_channels=features[3], kernel_size=1, stride=1, padding=0), nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=2, padding=1))
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [32, 32]
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed, pretrained.model)
    return pretrained


def _make_pretrained_vitb_rn50_384(pretrained, use_readout='ignore', hooks=None, use_vit_only=False, enable_attention_hooks=False):
    model = timm.create_model('vit_small_r26_s32_384', pretrained=pretrained)
    hooks = [0, 1, 8, 11] if hooks == None else hooks
    return _make_vit_b_rn50_backbone(model, features=[128, 256, 384, 384], size=[384, 384], hooks=hooks, use_vit_only=use_vit_only, use_readout=use_readout, enable_attention_hooks=enable_attention_hooks)


def _make_pretrained_vitl16_384(pretrained, use_readout='ignore', hooks=None, enable_attention_hooks=False):
    model = timm.create_model('vit_large_patch16_384', pretrained=pretrained)
    hooks = [5, 11, 17, 23] if hooks == None else hooks
    return _make_vit_b16_backbone(model, features=[256, 512, 1024, 1024], hooks=hooks, vit_features=1024, use_readout=use_readout, enable_attention_hooks=enable_attention_hooks)


def _make_encoder(backbone, features, use_pretrained, groups=1, expand=False, exportable=True, hooks=None, use_vit_only=False, use_readout='ignore', enable_attention_hooks=False):
    if backbone == 'vitl16_384':
        pretrained = _make_pretrained_vitl16_384(use_pretrained, hooks=hooks, use_readout=use_readout, enable_attention_hooks=enable_attention_hooks)
        scratch = _make_scratch([256, 512, 1024, 1024], features, groups=groups, expand=expand)
    elif backbone == 'vitb_rn50_384':
        pretrained = _make_pretrained_vitb_rn50_384(use_pretrained, hooks=hooks, use_vit_only=use_vit_only, use_readout=use_readout, enable_attention_hooks=enable_attention_hooks)
        scratch = _make_scratch([256, 512, 768, 768], features, groups=groups, expand=expand)
    elif backbone == 'vitb16_384':
        pretrained = _make_pretrained_vitb16_384(use_pretrained, hooks=hooks, use_readout=use_readout, enable_attention_hooks=enable_attention_hooks)
        scratch = _make_scratch([96, 192, 384, 768], features, groups=groups, expand=expand)
    elif backbone == 'resnext101_wsl':
        pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
        scratch = _make_scratch([256, 512, 1024, 2048], features, groups=groups, expand=expand)
    elif backbone == 'vit_tiny_r_s16_p8_384':
        pretrained = _make_pretrained_vit_tiny(use_pretrained, hooks=hooks, use_readout=use_readout, enable_attention_hooks=enable_attention_hooks)
        scratch = _make_scratch([96, 192, 384, 768], features, groups=groups, expand=expand)
    else:
        None
        assert False
    return pretrained, scratch


def forward_vit(pretrained, x):
    b, c, h, w = x.shape
    glob = pretrained.model.forward_flex(x)
    layer_1 = pretrained.activations['1']
    layer_2 = pretrained.activations['2']
    layer_3 = pretrained.activations['3']
    layer_4 = pretrained.activations['4']
    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)
    unflatten = nn.Sequential(nn.Unflatten(2, torch.Size([h // pretrained.model.patch_size[1], w // pretrained.model.patch_size[0]])))
    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)
    layer_1 = pretrained.act_postprocess1[3:len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3:len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3:len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3:len(pretrained.act_postprocess4)](layer_4)
    return layer_1, layer_2, layer_3, layer_4


class DPT(BaseModel):

    def __init__(self, head, features=256, backbone='vitb_rn50_384', readout='project', channels_last=False, use_bn=True, enable_attention_hooks=False):
        super(DPT, self).__init__()
        self.channels_last = channels_last
        hooks = {'vitb_rn50_384': [0, 1, 8, 11], 'vitb16_384': [2, 5, 8, 11], 'vitl16_384': [5, 11, 17, 23], 'vit_tiny_r_s16_p8_384': [0, 1, 2, 3]}
        self.pretrained, self.scratch = _make_encoder(backbone, features, False, groups=1, expand=False, exportable=False, hooks=hooks[backbone], use_readout=readout, enable_attention_hooks=enable_attention_hooks)
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        self.scratch.output_conv = head
        self.proj_out = nn.Sequential(nn.Conv2d(256 + 512 + 384 + 384, 256, kernel_size=3, padding=1, padding_mode='zeros'), nn.BatchNorm2d(128 * 2), nn.ReLU(True), nn.Conv2d(128 * 2, 128, kernel_size=3, padding=1, padding_mode='zeros'))

    def forward(self, x, only_enc=False):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        if only_enc:
            layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)
            a = layer_1
            b = F.interpolate(layer_2, scale_factor=2, mode='bilinear', align_corners=True)
            c = F.interpolate(layer_3, scale_factor=8, mode='bilinear', align_corners=True)
            d = F.interpolate(layer_4, scale_factor=16, mode='bilinear', align_corners=True)
            x = self.proj_out(torch.cat([a, b, c, d], dim=1))
            return x
        else:
            layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        _, _, H_out, W_out = path_1.size()
        path_2_up = F.interpolate(path_2, size=(H_out, W_out), mode='bilinear', align_corners=True)
        path_3_up = F.interpolate(path_3, size=(H_out, W_out), mode='bilinear', align_corners=True)
        path_4_up = F.interpolate(path_4, size=(H_out, W_out), mode='bilinear', align_corners=True)
        out = self.scratch.output_conv(path_1 + path_2_up + path_3_up + path_4_up)
        return out


class DPTDepthModel(DPT):

    def __init__(self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs):
        features = kwargs['features'] if 'features' in kwargs else 256
        self.scale = scale
        self.shift = shift
        self.invert = invert
        head = nn.Sequential(nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1), Interpolate(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(True), nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), nn.ReLU(True) if non_negative else nn.Identity(), nn.Identity())
        super().__init__(head, **kwargs)
        if path is not None:
            self.load(path)

    def forward(self, x):
        inv_depth = super().forward(x).squeeze(dim=1)
        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-08] = 1e-08
            depth = 1.0 / depth
            return depth
        else:
            return inv_depth


class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        None
        super(MidasNet, self).__init__()
        use_pretrained = False if path is None else True
        self.pretrained, self.scratch = _make_encoder(backbone='resnext101_wsl', features=features, use_pretrained=use_pretrained)
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        self.scratch.output_conv = nn.Sequential(nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1), Interpolate(scale_factor=2, mode='bilinear'), nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(True), nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), nn.ReLU(True) if non_negative else nn.Identity())
        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv(path_1)
        return torch.squeeze(out, dim=1)


class MidasNet_small(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=64, backbone='efficientnet_lite3', non_negative=True, exportable=True, channels_last=False, align_corners=True, blocks={'expand': True}):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        None
        super(MidasNet_small, self).__init__()
        use_pretrained = False if path else True
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone
        self.groups = 1
        features1 = features
        features2 = features
        features3 = features
        features4 = features
        self.expand = False
        if 'expand' in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1 = features
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8
        self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)
        self.scratch.activation = nn.ReLU(False)
        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)
        self.scratch.output_conv = nn.Sequential(nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, groups=self.groups), Interpolate(scale_factor=2, mode='bilinear'), nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1), self.scratch.activation, nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), nn.ReLU(True) if non_negative else nn.Identity(), nn.Identity())
        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        if self.channels_last == True:
            None
            x.contiguous(memory_format=torch.channels_last)
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv(path_1)
        return torch.squeeze(out, dim=1)


nchannels2models = {tuple([256] * 5): ['DPT_BEiT_L_384', 'DPT_BEiT_L_512', 'DPT_BEiT_B_384', 'DPT_SwinV2_L_384', 'DPT_SwinV2_B_384', 'DPT_SwinV2_T_256', 'DPT_Large', 'DPT_Hybrid'], (512, 256, 128, 64, 64): ['MiDaS_small']}


MIDAS_SETTINGS = {m: k for k, v in nchannels2models.items() for m in v}


class PrepForMidas(object):

    def __init__(self, resize_mode='minimal', keep_aspect_ratio=True, img_size=384, do_resize=True):
        if isinstance(img_size, int):
            img_size = img_size, img_size
        net_h, net_w = img_size
        self.normalization = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resizer = Resize(net_w, net_h, keep_aspect_ratio=keep_aspect_ratio, ensure_multiple_of=32, resize_method=resize_mode) if do_resize else nn.Identity()

    def __call__(self, x):
        return self.normalization(self.resizer(x))


def denormalize(x):
    """Reverses the imagenet normalization applied to the input.

    Args:
        x (torch.Tensor - shape(N,3,H,W)): input tensor

    Returns:
        torch.Tensor - shape(N,3,H,W): Denormalized input
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return x * std + mean


class MidasCore(nn.Module):

    def __init__(self, midas, trainable=False, fetch_features=True, layer_names=('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1'), freeze_bn=False, keep_aspect_ratio=True, img_size=384, **kwargs):
        """Midas Base model used for multi-scale feature extraction.

        Args:
            midas (torch.nn.Module): Midas model.
            trainable (bool, optional): Train midas model. Defaults to False.
            fetch_features (bool, optional): Extract multi-scale features. Defaults to True.
            layer_names (tuple, optional): Layers used for feature extraction. Order = (head output features, last layer features, ...decoder features). Defaults to ('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1').
            freeze_bn (bool, optional): Freeze BatchNorm. Generally results in better finetuning performance. Defaults to False.
            keep_aspect_ratio (bool, optional): Keep the aspect ratio of input images while resizing. Defaults to True.
            img_size (int, tuple, optional): Input resolution. Defaults to 384.
        """
        super().__init__()
        self.core = midas
        self.output_channels = None
        self.core_out = {}
        self.trainable = trainable
        self.fetch_features = fetch_features
        self.handles = []
        self.layer_names = layer_names
        self.set_trainable(trainable)
        self.set_fetch_features(fetch_features)
        self.prep = PrepForMidas(keep_aspect_ratio=keep_aspect_ratio, img_size=img_size, do_resize=kwargs.get('do_resize', True))
        if freeze_bn:
            self.freeze_bn()

    def set_trainable(self, trainable):
        self.trainable = trainable
        if trainable:
            self.unfreeze()
        else:
            self.freeze()
        return self

    def set_fetch_features(self, fetch_features):
        self.fetch_features = fetch_features
        if fetch_features:
            if len(self.handles) == 0:
                self.attach_hooks(self.core)
        else:
            self.remove_hooks()
        return self

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.trainable = False
        return self

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.trainable = True
        return self

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x, denorm=False, return_rel_depth=False):
        with torch.no_grad():
            if denorm:
                x = denormalize(x)
            x = self.prep(x)
        with torch.set_grad_enabled(self.trainable):
            rel_depth = self.core(x)
            if not self.fetch_features:
                return rel_depth
        out = [self.core_out[k] for k in self.layer_names]
        if return_rel_depth:
            return rel_depth, out
        return out

    def get_rel_pos_params(self):
        for name, p in self.core.pretrained.named_parameters():
            if 'relative_position' in name:
                yield p

    def get_enc_params_except_rel_pos(self):
        for name, p in self.core.pretrained.named_parameters():
            if 'relative_position' not in name:
                yield p

    def freeze_encoder(self, freeze_rel_pos=False):
        if freeze_rel_pos:
            for p in self.core.pretrained.parameters():
                p.requires_grad = False
        else:
            for p in self.get_enc_params_except_rel_pos():
                p.requires_grad = False
        return self

    def attach_hooks(self, midas):
        if len(self.handles) > 0:
            self.remove_hooks()
        if 'out_conv' in self.layer_names:
            self.handles.append(list(midas.scratch.output_conv.children())[3].register_forward_hook(get_activation('out_conv', self.core_out)))
        if 'r4' in self.layer_names:
            self.handles.append(midas.scratch.refinenet4.register_forward_hook(get_activation('r4', self.core_out)))
        if 'r3' in self.layer_names:
            self.handles.append(midas.scratch.refinenet3.register_forward_hook(get_activation('r3', self.core_out)))
        if 'r2' in self.layer_names:
            self.handles.append(midas.scratch.refinenet2.register_forward_hook(get_activation('r2', self.core_out)))
        if 'r1' in self.layer_names:
            self.handles.append(midas.scratch.refinenet1.register_forward_hook(get_activation('r1', self.core_out)))
        if 'l4_rn' in self.layer_names:
            self.handles.append(midas.scratch.layer4_rn.register_forward_hook(get_activation('l4_rn', self.core_out)))
        return self

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        return self

    def __del__(self):
        self.remove_hooks()

    def set_output_channels(self, model_type):
        self.output_channels = MIDAS_SETTINGS[model_type]

    @staticmethod
    def build(midas_model_type='DPT_BEiT_L_384', train_midas=False, use_pretrained_midas=True, fetch_features=False, freeze_bn=True, force_keep_ar=False, force_reload=False, **kwargs):
        if midas_model_type not in MIDAS_SETTINGS:
            raise ValueError(f'Invalid model type: {midas_model_type}. Must be one of {list(MIDAS_SETTINGS.keys())}')
        if 'img_size' in kwargs:
            kwargs = MidasCore.parse_img_size(kwargs)
        img_size = kwargs.pop('img_size', [384, 384])
        None
        hubconf = importlib.import_module(f'models.monoD.zoeDepth.midas_c.hubconf')
        midas = getattr(hubconf, midas_model_type)(pretrained=False)
        ckpt_path = 'models/monoD/zoeDepth/ckpts/dpt_beit_large_384.pt'
        midas_ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        midas.load_state_dict(midas_ckpt)
        kwargs.update({'keep_aspect_ratio': force_keep_ar})
        midas_core = MidasCore(midas, trainable=train_midas, fetch_features=fetch_features, freeze_bn=freeze_bn, img_size=img_size, **kwargs)
        midas_core.set_output_channels(midas_model_type)
        return midas_core

    @staticmethod
    def build_from_config(config):
        return MidasCore.build(**config)

    @staticmethod
    def parse_img_size(config):
        assert 'img_size' in config
        if isinstance(config['img_size'], str):
            assert ',' in config['img_size'], 'img_size should be a string with comma separated img_size=H,W'
            config['img_size'] = list(map(int, config['img_size'].split(',')))
            assert len(config['img_size']) == 2, 'img_size should be a string with comma separated img_size=H,W'
        elif isinstance(config['img_size'], int):
            config['img_size'] = [config['img_size'], config['img_size']]
        else:
            assert isinstance(config['img_size'], list) and len(config['img_size']) == 2, 'img_size should be a list of H,W'
        return config


@torch.jit.script
def exp_attractor(dx, alpha: 'float'=300, gamma: 'int'=2):
    """Exponential attractor: dc = exp(-alpha*|dx|^gamma) * dx , where dx = a - c, a = attractor point, c = bin center, dc = shift in bin centermmary for exp_attractor

    Args:
        dx (torch.Tensor): The difference tensor dx = Ai - Cj, where Ai is the attractor point and Cj is the bin center.
        alpha (float, optional): Proportional Attractor strength. Determines the absolute strength. Lower alpha = greater attraction. Defaults to 300.
        gamma (int, optional): Exponential Attractor strength. Determines the "region of influence" and indirectly number of bin centers affected. Lower gamma = farther reach. Defaults to 2.

    Returns:
        torch.Tensor : Delta shifts - dc; New bin centers = Old bin centers + dc
    """
    return torch.exp(-alpha * torch.abs(dx) ** gamma) * dx


@torch.jit.script
def inv_attractor(dx, alpha: 'float'=300, gamma: 'int'=2):
    """Inverse attractor: dc = dx / (1 + alpha*dx^gamma), where dx = a - c, a = attractor point, c = bin center, dc = shift in bin center
    This is the default one according to the accompanying paper. 

    Args:
        dx (torch.Tensor): The difference tensor dx = Ai - Cj, where Ai is the attractor point and Cj is the bin center.
        alpha (float, optional): Proportional Attractor strength. Determines the absolute strength. Lower alpha = greater attraction. Defaults to 300.
        gamma (int, optional): Exponential Attractor strength. Determines the "region of influence" and indirectly number of bin centers affected. Lower gamma = farther reach. Defaults to 2.

    Returns:
        torch.Tensor: Delta shifts - dc; New bin centers = Old bin centers + dc
    """
    return dx.div(1 + alpha * dx.pow(gamma))


class AttractorLayer(nn.Module):

    def __init__(self, in_features, n_bins, n_attractors=16, mlp_dim=128, min_depth=0.001, max_depth=10, alpha=300, gamma=2, kind='sum', attractor_type='exp', memory_efficient=False):
        """
        Attractor layer for bin centers. Bin centers are bounded on the interval (min_depth, max_depth)
        """
        super().__init__()
        self.n_attractors = n_attractors
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alpha = alpha
        self.gamma = gamma
        self.kind = kind
        self.attractor_type = attractor_type
        self.memory_efficient = memory_efficient
        self._net = nn.Sequential(nn.Conv2d(in_features, mlp_dim, 1, 1, 0), nn.ReLU(inplace=True), nn.Conv2d(mlp_dim, n_attractors * 2, 1, 1, 0), nn.ReLU(inplace=True))

    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        Args:
            x (torch.Tensor) : feature block; shape - n, c, h, w
            b_prev (torch.Tensor) : previous bin centers normed; shape - n, prev_nbins, h, w
        
        Returns:
            tuple(torch.Tensor,torch.Tensor) : new bin centers normed and scaled; shape - n, nbins, h, w
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + prev_b_embedding
        A = self._net(x)
        eps = 0.001
        A = A + eps
        n, c, h, w = A.shape
        A = A.view(n, self.n_attractors, 2, h, w)
        A_normed = A / A.sum(dim=2, keepdim=True)
        A_normed = A[:, :, 0, ...]
        b_prev = nn.functional.interpolate(b_prev, (h, w), mode='bilinear', align_corners=True)
        b_centers = b_prev
        if self.attractor_type == 'exp':
            dist = exp_attractor
        else:
            dist = inv_attractor
        if not self.memory_efficient:
            func = {'mean': torch.mean, 'sum': torch.sum}[self.kind]
            delta_c = func(dist(A_normed.unsqueeze(2) - b_centers.unsqueeze(1)), dim=1)
        else:
            delta_c = torch.zeros_like(b_centers, device=b_centers.device)
            for i in range(self.n_attractors):
                delta_c += dist(A_normed[:, i, ...].unsqueeze(1) - b_centers)
            if self.kind == 'mean':
                delta_c = delta_c / self.n_attractors
        b_new_centers = b_centers + delta_c
        B_centers = (self.max_depth - self.min_depth) * b_new_centers + self.min_depth
        B_centers, _ = torch.sort(B_centers, dim=1)
        B_centers = torch.clip(B_centers, self.min_depth, self.max_depth)
        return b_new_centers, B_centers


class AttractorLayerUnnormed(nn.Module):

    def __init__(self, in_features, n_bins, n_attractors=16, mlp_dim=128, min_depth=0.001, max_depth=10, alpha=300, gamma=2, kind='sum', attractor_type='exp', memory_efficient=False):
        """
        Attractor layer for bin centers. Bin centers are unbounded
        """
        super().__init__()
        self.n_attractors = n_attractors
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alpha = alpha
        self.gamma = gamma
        self.kind = kind
        self.attractor_type = attractor_type
        self.memory_efficient = memory_efficient
        self._net = nn.Sequential(nn.Conv2d(in_features, mlp_dim, 1, 1, 0), nn.ReLU(inplace=True), nn.Conv2d(mlp_dim, n_attractors, 1, 1, 0), nn.Softplus())

    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        Args:
            x (torch.Tensor) : feature block; shape - n, c, h, w
            b_prev (torch.Tensor) : previous bin centers normed; shape - n, prev_nbins, h, w
        
        Returns:
            tuple(torch.Tensor,torch.Tensor) : new bin centers unbounded; shape - n, nbins, h, w. Two outputs just to keep the API consistent with the normed version
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + prev_b_embedding
        A = self._net(x)
        n, c, h, w = A.shape
        b_prev = nn.functional.interpolate(b_prev, (h, w), mode='bilinear', align_corners=True)
        b_centers = b_prev
        if self.attractor_type == 'exp':
            dist = exp_attractor
        else:
            dist = inv_attractor
        if not self.memory_efficient:
            func = {'mean': torch.mean, 'sum': torch.sum}[self.kind]
            delta_c = func(dist(A.unsqueeze(2) - b_centers.unsqueeze(1)), dim=1)
        else:
            delta_c = torch.zeros_like(b_centers, device=b_centers.device)
            for i in range(self.n_attractors):
                delta_c += dist(A[:, i, ...].unsqueeze(1) - b_centers)
            if self.kind == 'mean':
                delta_c = delta_c / self.n_attractors
        b_new_centers = b_centers + delta_c
        B_centers = b_new_centers
        return b_new_centers, B_centers


def log_binom(n, k, eps=1e-07):
    """ log(nCk) using stirling approximation """
    n = n + eps
    k = k + eps
    return n * torch.log(n) - k * torch.log(k) - (n - k) * torch.log(n - k + eps)


class LogBinomial(nn.Module):

    def __init__(self, n_classes=256, act=torch.softmax):
        """Compute log binomial distribution for n_classes

        Args:
            n_classes (int, optional): number of output classes. Defaults to 256.
        """
        super().__init__()
        self.K = n_classes
        self.act = act
        self.register_buffer('k_idx', torch.arange(0, n_classes).view(1, -1, 1, 1))
        self.register_buffer('K_minus_1', torch.Tensor([self.K - 1]).view(1, -1, 1, 1))

    def forward(self, x, t=1.0, eps=0.0001):
        """Compute log binomial distribution for x

        Args:
            x (torch.Tensor - NCHW): probabilities
            t (float, torch.Tensor - NCHW, optional): Temperature of distribution. Defaults to 1..
            eps (float, optional): Small number for numerical stability. Defaults to 1e-4.

        Returns:
            torch.Tensor -NCHW: log binomial distribution logbinomial(p;t)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        one_minus_x = torch.clamp(1 - x, eps, 1)
        x = torch.clamp(x, eps, 1)
        y = log_binom(self.K_minus_1, self.k_idx) + self.k_idx * torch.log(x) + (self.K - 1 - self.k_idx) * torch.log(one_minus_x)
        return self.act(y / t, dim=1)


class ConditionalLogBinomial(nn.Module):

    def __init__(self, in_features, condition_dim, n_classes=256, bottleneck_factor=2, p_eps=0.0001, max_temp=50, min_temp=1e-07, act=torch.softmax):
        """Conditional Log Binomial distribution

        Args:
            in_features (int): number of input channels in main feature
            condition_dim (int): number of input channels in condition feature
            n_classes (int, optional): Number of classes. Defaults to 256.
            bottleneck_factor (int, optional): Hidden dim factor. Defaults to 2.
            p_eps (float, optional): small eps value. Defaults to 1e-4.
            max_temp (float, optional): Maximum temperature of output distribution. Defaults to 50.
            min_temp (float, optional): Minimum temperature of output distribution. Defaults to 1e-7.
        """
        super().__init__()
        self.p_eps = p_eps
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.log_binomial_transform = LogBinomial(n_classes, act=act)
        bottleneck = (in_features + condition_dim) // bottleneck_factor
        self.mlp = nn.Sequential(nn.Conv2d(in_features + condition_dim, bottleneck, kernel_size=1, stride=1, padding=0), nn.GELU(), nn.Conv2d(bottleneck, 2 + 2, kernel_size=1, stride=1, padding=0), nn.Softplus())

    def forward(self, x, cond):
        """Forward pass

        Args:
            x (torch.Tensor - NCHW): Main feature
            cond (torch.Tensor - NCHW): condition feature

        Returns:
            torch.Tensor: Output log binomial distribution
        """
        pt = self.mlp(torch.concat((x, cond), dim=1))
        p, t = pt[:, :2, ...], pt[:, 2:, ...]
        p = p + self.p_eps
        p = p[:, 0, ...] / (p[:, 0, ...] + p[:, 1, ...])
        t = t + self.p_eps
        t = t[:, 0, ...] / (t[:, 0, ...] + t[:, 1, ...])
        t = t.unsqueeze(1)
        t = (self.max_temp - self.min_temp) * t + self.min_temp
        return self.log_binomial_transform(p, t)


class SeedBinRegressor(nn.Module):

    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=0.001, max_depth=10):
        """Bin center regressor network. Bin centers are bounded on (min_depth, max_depth) interval.

        Args:
            in_features (int): input channels
            n_bins (int, optional): Number of bin centers. Defaults to 16.
            mlp_dim (int, optional): Hidden dimension. Defaults to 256.
            min_depth (float, optional): Min depth value. Defaults to 1e-3.
            max_depth (float, optional): Max depth value. Defaults to 10.
        """
        super().__init__()
        self.version = '1_1'
        self.min_depth = min_depth
        self.max_depth = max_depth
        self._net = nn.Sequential(nn.Conv2d(in_features, mlp_dim, 1, 1, 0), nn.ReLU(inplace=True), nn.Conv2d(mlp_dim, n_bins, 1, 1, 0), nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Returns tensor of bin_width vectors (centers). One vector b for every pixel
        """
        B = self._net(x)
        eps = 0.001
        B = B + eps
        B_widths_normed = B / B.sum(dim=1, keepdim=True)
        B_widths = (self.max_depth - self.min_depth) * B_widths_normed
        B_widths = nn.functional.pad(B_widths, (0, 0, 0, 0, 1, 0), mode='constant', value=self.min_depth)
        B_edges = torch.cumsum(B_widths, dim=1)
        B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:, 1:, ...])
        return B_widths_normed, B_centers


class SeedBinRegressorUnnormed(nn.Module):

    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=0.001, max_depth=10):
        """Bin center regressor network. Bin centers are unbounded

        Args:
            in_features (int): input channels
            n_bins (int, optional): Number of bin centers. Defaults to 16.
            mlp_dim (int, optional): Hidden dimension. Defaults to 256.
            min_depth (float, optional): Not used. (for compatibility with SeedBinRegressor)
            max_depth (float, optional): Not used. (for compatibility with SeedBinRegressor)
        """
        super().__init__()
        self.version = '1_1'
        self._net = nn.Sequential(nn.Conv2d(in_features, mlp_dim, 1, 1, 0), nn.ReLU(inplace=True), nn.Conv2d(mlp_dim, n_bins, 1, 1, 0), nn.Softplus())

    def forward(self, x):
        """
        Returns tensor of bin_width vectors (centers). One vector b for every pixel
        """
        B_centers = self._net(x)
        return B_centers, B_centers


class Projector(nn.Module):

    def __init__(self, in_features, out_features, mlp_dim=128):
        """Projector MLP

        Args:
            in_features (int): input channels
            out_features (int): output channels
            mlp_dim (int, optional): hidden dimension. Defaults to 128.
        """
        super().__init__()
        self._net = nn.Sequential(nn.Conv2d(in_features, mlp_dim, 1, 1, 0), nn.ReLU(inplace=True), nn.Conv2d(mlp_dim, out_features, 1, 1, 0))

    def forward(self, x):
        return self._net(x)


class LinearSplitter(nn.Module):

    def __init__(self, in_features, prev_nbins, split_factor=2, mlp_dim=128, min_depth=0.001, max_depth=10):
        super().__init__()
        self.prev_nbins = prev_nbins
        self.split_factor = split_factor
        self.min_depth = min_depth
        self.max_depth = max_depth
        self._net = nn.Sequential(nn.Conv2d(in_features, mlp_dim, 1, 1, 0), nn.GELU(), nn.Conv2d(mlp_dim, prev_nbins * split_factor, 1, 1, 0), nn.ReLU())

    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        x : feature block; shape - n, c, h, w
        b_prev : previous bin widths normed; shape - n, prev_nbins, h, w
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + prev_b_embedding
        S = self._net(x)
        eps = 0.001
        S = S + eps
        n, c, h, w = S.shape
        S = S.view(n, self.prev_nbins, self.split_factor, h, w)
        S_normed = S / S.sum(dim=2, keepdim=True)
        b_prev = nn.functional.interpolate(b_prev, (h, w), mode='bilinear', align_corners=True)
        b_prev = b_prev / b_prev.sum(dim=1, keepdim=True)
        b = b_prev.unsqueeze(2) * S_normed
        b = b.flatten(1, 2)
        B_widths = (self.max_depth - self.min_depth) * b
        B_widths = nn.functional.pad(B_widths, (0, 0, 0, 0, 1, 0), mode='constant', value=self.min_depth)
        B_edges = torch.cumsum(B_widths, dim=1)
        B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:, 1:, ...])
        return b, B_centers


class PatchTransformerEncoder(nn.Module):

    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4, use_class_token=False):
        """ViT-like transformer block

        Args:
            in_channels (int): Input channels
            patch_size (int, optional): patch size. Defaults to 10.
            embedding_dim (int, optional): Embedding dimension in transformer model. Defaults to 128.
            num_heads (int, optional): number of attention heads. Defaults to 4.
            use_class_token (bool, optional): Whether to use extra token at the start for global accumulation (called as "class token"). Defaults to False.
        """
        super(PatchTransformerEncoder, self).__init__()
        self.use_class_token = use_class_token
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)
        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)

    def positional_encoding_1d(self, sequence_length, batch_size, embedding_dim, device='cpu'):
        """Generate positional encodings

        Args:
            sequence_length (int): Sequence length
            embedding_dim (int): Embedding dimension

        Returns:
            torch.Tensor SBE: Positional encodings
        """
        position = torch.arange(0, sequence_length, dtype=torch.float32, device=device).unsqueeze(1)
        index = torch.arange(0, embedding_dim, 2, dtype=torch.float32, device=device).unsqueeze(0)
        div_term = torch.exp(index * (-torch.log(torch.tensor(10000.0, device=device)) / embedding_dim))
        pos_encoding = position * div_term
        pos_encoding = torch.cat([torch.sin(pos_encoding), torch.cos(pos_encoding)], dim=1)
        pos_encoding = pos_encoding.unsqueeze(1).repeat(1, batch_size, 1)
        return pos_encoding

    def forward(self, x):
        """Forward pass

        Args:
            x (torch.Tensor - NCHW): Input feature tensor

        Returns:
            torch.Tensor - SNE: Transformer output embeddings. S - sequence length (=HW/patch_size^2), N - batch size, E - embedding dim
        """
        embeddings = self.embedding_convPxP(x).flatten(2)
        if self.use_class_token:
            embeddings = nn.functional.pad(embeddings, (1, 0))
        embeddings = embeddings.permute(2, 0, 1)
        S, N, E = embeddings.shape
        embeddings = embeddings + self.positional_encoding_1d(S, N, E, device=embeddings.device)
        x = self.transformer_encoder(embeddings)
        return x


def load_state_dict(model, state_dict):
    """Load state_dict into model, handling DataParallel and DistributedDataParallel. Also checks for "model" key in state_dict.

    DataParallel prefixes state_dict keys with 'module.' when saving.
    If the model is not a DataParallel model but the state_dict is, then prefixes are removed.
    If the model is a DataParallel model but the state_dict is not, then prefixes are added.
    """
    state_dict = state_dict.get('model', state_dict)
    do_prefix = isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    state = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not do_prefix:
            k = k[7:]
        if not k.startswith('module.') and do_prefix:
            k = 'module.' + k
        state[k] = v
    model.load_state_dict(state)
    None
    return model


def load_state_dict_from_url(model, url, **kwargs):
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', **kwargs)
    return load_state_dict(model, state_dict)


def load_wts(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return load_state_dict(model, ckpt)


def load_state_from_resource(model, resource: 'str'):
    """Loads weights to the model from a given resource. A resource can be of following types:
        1. URL. Prefixed with "url::"
                e.g. url::http(s)://url.resource.com/ckpt.pt

        2. Local path. Prefixed with "local::"
                e.g. local::/path/to/ckpt.pt


    Args:
        model (torch.nn.Module): Model
        resource (str): resource string

    Returns:
        torch.nn.Module: Model with loaded weights
    """
    None
    if resource.startswith('url::'):
        url = resource.split('url::')[1]
        return load_state_dict_from_url(model, url, progress=True)
    elif resource.startswith('local::'):
        path = resource.split('local::')[1]
        return load_wts(model, path)
    else:
        raise ValueError('Invalid resource type, only url:: and local:: are supported')


class Embedder_Fourier(nn.Module):

    def __init__(self, input_dim, max_freq_log2, N_freqs, log_sampling=True, include_input=True, periodic_fns=(torch.sin, torch.cos)):
        """
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        """
        super(Embedder_Fourier, self).__init__()
        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns
        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim
        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)
        if log_sampling:
            self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq_log2, N_freqs)
        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: 'torch.Tensor', rescale: 'float'=1.0):
        """
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        """
        assert input.shape[-1] == self.input_dim
        out = []
        if self.include_input:
            out.append(input / rescale)
        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)
        assert out.shape[-1] == self.out_dim
        return out


class VitEncoder(nn.Module):

    def __init__(self, input_dim=4, output_dim=128, stride=4):
        super(VitEncoder, self).__init__()
        self.vit = vitEnc(img_size=512, depth=6, num_heads=8, in_chans=input_dim, out_chans=output_dim, embed_dim=384)
        self.stride = stride

    def forward(self, x):
        T, C, H, W = x.shape
        x_resize = F.interpolate(x.view(-1, C, H, W), size=(512, 512), mode='bilinear', align_corners=False)
        x_resize = self.vit(x_resize)
        x = F.interpolate(x_resize, size=(H // self.stride, W // self.stride), mode='bilinear', align_corners=False)
        return x


class DPTEncoder(DPT):

    def __init__(self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs):
        features = kwargs['features'] if 'features' in kwargs else 256
        self.scale = scale
        self.shift = shift
        head = nn.Sequential(nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1))
        super().__init__(head, **kwargs)
        if path is not None:
            self.load(path)

    def forward(self, x):
        features = super().forward(x, only_enc=True).squeeze(dim=1)
        return features


class DPTEnc(nn.Module):

    def __init__(self, input_dim=3, output_dim=128, stride=2):
        super(DPTEnc, self).__init__()
        self.dpt = DPTEncoder()
        self.stride = stride

    def forward(self, x):
        T, C, H, W = x.shape
        x = (x - 0.5) / 0.5
        x_resize = F.interpolate(x.view(-1, C, H, W), size=(384, 384), mode='bilinear', align_corners=False)
        x_resize = self.dpt(x_resize)
        x = F.interpolate(x_resize, size=(H // self.stride, W // self.stride), mode='bilinear', align_corners=False)
        return x


class VGG19(nn.Module):

    def __init__(self, pretrained=False, amp=False, amp_dtype=torch.float16) ->None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast('cuda', enabled=self.amp, dtype=self.amp_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale * 2
                x = layer(x)
            return feats


class CNNandDinov2(nn.Module):

    def __init__(self, cnn_kwargs=None, amp=True, amp_dtype=torch.float16):
        super().__init__()
        self.dinov2_vitl14 = torch.hub.load('models/torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format('vitl'), source='local', pretrained=False)
        state_dict = torch.load('models/monoD/zoeDepth/ckpts/dinov2_vitl14_pretrain.pth')
        self.dinov2_vitl14.load_state_dict(state_dict, strict=True)
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            dinov2_vitl14 = dinov2_vitl14
        self.dinov2_vitl14 = [dinov2_vitl14]

    def train(self, mode: 'bool'=True):
        return self.cnn.train(mode)

    def forward(self, x, upsample=False):
        B, C, H, W = x.shape
        feature_pyramid = self.cnn(x)
        if not upsample:
            with torch.no_grad():
                if self.dinov2_vitl14[0].device != x.device:
                    self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device)
                dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x)
                features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0, 2, 1).reshape(B, 1024, H // 14, W // 14)
                del dinov2_features_16
                feature_pyramid[16] = features_16
        return feature_pyramid


class Dinov2(nn.Module):

    def __init__(self, amp=True, amp_dtype=torch.float16):
        super().__init__()
        self.dinov2_vitl14 = torch.hub.load('models/torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format('vitl'), source='local', pretrained=False)
        state_dict = torch.load('models/monoD/zoeDepth/ckpts/dinov2_vitl14_pretrain.pth')
        self.dinov2_vitl14.load_state_dict(state_dict, strict=True)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            self.dinov2_vitl14 = self.dinov2_vitl14

    def forward(self, x, upsample=False):
        B, C, H, W = x.shape
        mean_ = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std_ = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x + 1) / 2
        x = (x - mean_) / std_
        h_re, w_re = 560, 560
        x_resize = F.interpolate(x, size=(h_re, w_re), mode='bilinear', align_corners=True)
        if not upsample:
            with torch.no_grad():
                dinov2_features_16 = self.dinov2_vitl14.forward_features(x_resize)
                features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0, 2, 1).reshape(B, 1024, h_re // 14, w_re // 14)
                del dinov2_features_16
        features_16 = F.interpolate(features_16, size=(H // 8, W // 8), mode='bilinear', align_corners=True)
        return features_16


class CrossAttnBlock(nn.Module):

    def __init__(self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, flash=True, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = Attention(hidden_size, context_dim=context_dim, num_heads=num_heads, qkv_bias=True, **block_kwargs, flash=flash)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda : nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, context):
        with autocast():
            x = x + self.cross_attn(self.norm1(x), self.norm_context(context))
        x = x + self.mlp(self.norm2(x))
        return x


class FullAttention(Module):

    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        input_args = [x.half().contiguous() for x in [queries.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3), values.permute(0, 2, 1, 3)]]
        queried_values = F.scaled_dot_product_attention(*input_args).permute(0, 2, 1, 3).float()
        return queried_values.contiguous()


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead):
        super(TransformerEncoderLayer, self).__init__()
        self.dim = d_model // nhead
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d_model * 2, d_model * 2, bias=False), nn.ReLU(True), nn.Linear(d_model * 2, d_model, bias=False))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))
        message = self.norm1(message)
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)
        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()
        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = TransformerEncoderLayer(config['d_model'], config['nhead'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        assert self.d_model == feat0.size(2), 'the feature number of src and transformer must be equal'
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
        return feat0, feat1


class EUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(self, space_depth=12, time_depth=12, input_dim=320, hidden_size=384, num_heads=8, output_dim=130, mlp_ratio=4.0, vq_depth=3, add_space_attn=True, add_time_attn=True, flash=True):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.flash = flash
        self.flow_head = nn.Sequential(nn.Linear(hidden_size, output_dim, bias=True), nn.ReLU(inplace=True), nn.Linear(output_dim, output_dim, bias=True), nn.ReLU(inplace=True), nn.Linear(output_dim, output_dim, bias=True))
        cross_attn_kwargs = {'d_model': 384, 'nhead': 4, 'layer_names': ['self', 'cross'] * 3}
        self.gnn = LocalFeatureTransformer(cross_attn_kwargs)
        self.time_blocks = nn.ModuleList([(AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, flash=flash) if add_time_attn else nn.Identity()) for _ in range(time_depth)])
        if add_space_attn:
            self.space_blocks = nn.ModuleList([AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, flash=flash) for _ in range(space_depth)])
            assert len(self.time_blocks) >= len(self.space_blocks)
        self.RigidProj = nn.Linear(self.hidden_size, 128, bias=True)
        self.Proj = nn.Linear(self.hidden_size, 128, bias=True)
        self.se3_dec = nn.Linear(384, 3, bias=True)
        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, input_tensor, se3_feature):
        """ Updating with Transformer

        Args:
            input_tensor: B, N, T, C
            arap_embed: B, N, T, C
        """
        B, N, T, C = input_tensor.shape
        x = self.input_transform(input_tensor)
        tokens = x
        K = 0
        j = 0
        for i in range(len(self.time_blocks)):
            tokens_time = rearrange(tokens, 'b n t c -> (b n) t c', b=B, t=T, n=N + K)
            tokens_time = self.time_blocks[i](tokens_time)
            tokens = rearrange(tokens_time, '(b n) t c -> b n t c ', b=B, t=T, n=N + K)
            if self.add_space_attn and i % (len(self.time_blocks) // len(self.space_blocks)) == 0:
                tokens_space = rearrange(tokens, 'b n t c -> (b t) n c ', b=B, t=T, n=N)
                tokens_space = self.space_blocks[j](tokens_space)
                tokens = rearrange(tokens_space, '(b t) n c -> b n t c  ', b=B, t=T, n=N)
                j += 1
        B, N, S, _ = tokens.shape
        feat0, feat1 = self.gnn(tokens.view(B * N * S, -1)[None, ...], se3_feature[None, ...])
        so3 = F.tanh(self.se3_dec(feat0.view(B * N * S, -1)[None, ...].view(B, N, S, -1)) / 100)
        flow = self.flow_head(feat0.view(B, N, S, -1))
        return flow, _, _, feat1, so3


class FusionFormer(nn.Module):
    """ 
    Fuse the feature tracks info with the low rank motion tokens
    """

    def __init__(self, d_model=64, nhead=8, attn_iters=4, mlp_ratio=4.0, flash=False, input_dim=35, output_dim=384 + 3):
        super().__init__()
        self.flash = flash
        self.in_proj = nn.ModuleList([nn.Linear(input_dim, d_model) for _ in range(2)])
        self.out_proj = nn.Linear(d_model, output_dim, bias=True)
        self.time_blocks = nn.ModuleList([CrossAttnBlock(d_model, d_model, nhead, mlp_ratio=mlp_ratio) for _ in range(attn_iters)])
        self.space_blocks = nn.ModuleList([AttnBlock(d_model, nhead, mlp_ratio=mlp_ratio, flash=self.flash) for _ in range(attn_iters)])
        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        self.out_proj.weight.data.fill_(0)
        self.out_proj.bias.data.fill_(0)

    def forward(self, x, token_cls):
        """ Fuse the feature tracks info with the low rank motion tokens

        Args:
            x: B, S, N, C
            Traj_whole: B T N C
        
        """
        B, S, N, C = x.shape
        _, T, _, _ = token_cls.shape
        x = self.in_proj[0](x)
        token_cls = self.in_proj[1](token_cls)
        token_cls = rearrange(token_cls, 'b t n c -> (b n) t c')
        for i in range(len(self.space_blocks)):
            x = rearrange(x, 'b s n c -> (b n) s c')
            x = self.time_blocks[i](x, token_cls)
            x = self.space_blocks[i](x.permute(1, 0, 2))
            x = rearrange(x, '(b s) n c -> b s n c', b=B, s=S, n=N)
        x = self.out_proj(x)
        delta_xyz = x[..., :3]
        feat_traj = x[..., 3:]
        return delta_xyz, feat_traj


class MidasNet_large(BaseModel):
    """Network for monocular depth estimation."""

    def __init__(self, path=None, features=256, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        None
        super(MidasNet_large, self).__init__()
        use_pretrained = False if path is None else True
        self.pretrained, self.scratch = _make_encoder(backbone='resnext101_wsl', features=features, use_pretrained=use_pretrained)
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        self.scratch.output_conv = nn.Sequential(nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1), Interpolate(scale_factor=2, mode='bilinear'), nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(True), nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), nn.ReLU(True) if non_negative else nn.Identity())
        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv(path_1)
        return torch.squeeze(out, dim=1)


class DPTSegmentationModel(DPT):

    def __init__(self, num_classes, path=None, **kwargs):
        features = kwargs['features'] if 'features' in kwargs else 256
        kwargs['use_bn'] = True
        head = nn.Sequential(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(features), nn.ReLU(True), nn.Dropout(0.1, False), nn.Conv2d(features, num_classes, kernel_size=1), Interpolate(scale_factor=2, mode='bilinear', align_corners=True))
        super().__init__(head, **kwargs)
        self.auxlayer = nn.Sequential(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(features), nn.ReLU(True), nn.Dropout(0.1, False), nn.Conv2d(features, num_classes, kernel_size=1))
        if path is not None:
            self.load(path)


class ResnetBlockFC(nn.Module):
    """ Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in
        if size_h is None:
            size_h = min(size_in, size_out)
        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()
        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), conv1x1(in_channels, out_channels))


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5, start_filts=64, up_mode='transpose', merge_mode='concat', **kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError('"{}" is not a valid mode for upsampling. Only "transpose" and "upsample" are allowed.'.format(up_mode))
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError('"{}" is not a valid mode formerging up and down paths. Only "concat" and "add" are allowed.'.format(up_mode))
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError('up_mode "upsample" is incompatible with merge_mode "add" at the moment because it doesn\'t make sense to use nearest neighbour to reduce depth channels (by half).')
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.down_convs = []
        self.up_convs = []
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * 2 ** i
            pooling = True if i < depth - 1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode)
            self.up_convs.append(up_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.conv_final = conv1x1(outs, self.num_classes)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)
        x = self.conv_final(x)
        return x


class LocalSoftSplat(nn.Module):

    def __init__(self, ch=128, dim=3, hidden_dim=128, scatter_type='max', unet=True, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, hw=None, grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=4, splat_func=None):
        super().__init__()
        c_dim = ch
        self.c_dim = c_dim
        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)])
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim
        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None
        self.splat_func = splat_func

    def forward(self, img_feat, Fxy2xz, Fxy2yz, Dz, gridxy=None):
        """
        Args:
            img_feat (tensor): image features
            Fxy2xz (tensor): transformation matrix from xy to xz
            Fxy2yz (tensor): transformation matrix from xy to yz
        """
        B, T, _, H, W = img_feat.shape
        fea_reshp = rearrange(img_feat, 'b t c h w -> (b h w) t c', c=img_feat.shape[2], h=H, w=W)
        gridyz = gridxy + Fxy2yz
        gridxz = gridxy + Fxy2xz
        gridyz[:, 0, ...] = (gridyz[:, 0, ...] / (H - 1) - 0.5) * 2
        gridyz[:, 1, ...] = (gridyz[:, 1, ...] / (Dz - 1) - 0.5) * 2
        gridxz[:, 0, ...] = (gridxz[:, 0, ...] / (W - 1) - 0.5) * 2
        gridxz[:, 1, ...] = (gridxz[:, 1, ...] / (Dz - 1) - 0.5) * 2
        if len(self.blocks) > 0:
            net = self.fc_pos(fea_reshp)
            net = self.blocks[0](net)
            for block in self.blocks[1:]:
                net_plane = rearrange(net, '(b h w) t c -> (b t) c h w', b=B, h=H, w=W)
                net_planeYZ = self.splat_func(net_plane, Fxy2yz, None, strMode='avg', tenoutH=Dz, tenoutW=H)
                net_planeXZ = self.splat_func(net_plane, Fxy2xz, None, strMode='avg', tenoutH=Dz, tenoutW=W)
                net_plane = net_plane + (F.grid_sample(net_planeYZ, gridyz.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border') + F.grid_sample(net_planeXZ, gridxz.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border'))
                pooled = rearrange(net_plane, 't c h w -> (h w) t c', c=net_plane.shape[1], h=H, w=W)
                net = torch.cat([net, pooled], dim=2)
                net = block(net)
            c = self.fc_c(net)
            net_plane = rearrange(c, '(b h w) t c -> (b t) c h w', b=B, h=H, w=W)
        else:
            net_plane = rearrange(img_feat, 'b t c h w -> (b t) c h w', c=img_feat.shape[2], h=H, w=W)
        net_planeYZ = self.splat_func(net_plane, Fxy2yz, None, strMode='avg', tenoutH=Dz, tenoutW=H)
        net_planeXZ = self.splat_func(net_plane, Fxy2xz, None, strMode='avg', tenoutH=Dz, tenoutW=W)
        return net_plane[None], net_planeYZ[None], net_planeXZ[None]


def coordinate2index(x, reso, coord_type='2d'):
    """ Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    """
    x = (x * reso).long()
    if coord_type == '2d':
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d':
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index


def normalize_3d_coordinate(p, padding=0.1):
    """ Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """
    p_nor = p / (1 + padding + 0.001)
    p_nor = p_nor + 0.5
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 0.001
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


def normalize_coordinate(p, padding=0.1, plane='xz'):
    """ Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    """
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane == 'xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]
    xy = torch.clamp(xy, min=1e-06, max=1.0 - 1e-06)
    return xy


class LocalPoolPointnet(nn.Module):
    """ PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    """

    def __init__(self, ch=128, dim=3, hidden_dim=128, scatter_type='max', unet=True, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, hw=None, grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5):
        super().__init__()
        c_dim = ch
        unet3d = False
        plane_type = ['xy', 'xz', 'yz']
        plane_resolution = hw
        self.c_dim = c_dim
        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)])
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim
        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None
        if unet3d:
            raise NotImplementedError()
        else:
            self.unet3d = None
        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding
        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

    def generate_plane_features(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        index = coordinate2index(xy, self.reso_plane)
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)
        fea_plane = scatter_mean(c, index, out=fea_plane)
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane)
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)
        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)
        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)
        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()
        c_out = 0
        for key in keys:
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid ** 3)
            else:
                c_permute = c.permute(0, 2, 1)
                fea = self.scatter(c_permute, index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out = c_out + fea
        return c_out.permute(0, 2, 1)

    def forward(self, p_input, img_feats=None):
        """
        Args:
            p_input (tensor): input points    T 3 H W
            img_feats (tensor): image features  T C H W
        """
        T, _, H, W = img_feats.size()
        p = rearrange(p_input, 't c h w -> (h w) t c', c=3, h=H, w=W)
        fea_reshp = rearrange(img_feats, 't c h w -> (h w) t c', c=img_feats.shape[1], h=H, w=W)
        coord = {}
        index = {}
        if 'xz' in self.plane_type:
            coord['xz'] = normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
            index['xz'] = coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
            index['xy'] = coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
            index['yz'] = coordinate2index(coord['yz'], self.reso_plane)
        if 'grid' in self.plane_type:
            coord['grid'] = normalize_3d_coordinate(p.clone(), padding=self.padding)
            index['grid'] = coordinate2index(coord['grid'], self.reso_grid, coord_type='3d')
        net = self.fc_pos(p) + fea_reshp
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)
        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')
        ret = torch.stack([fea['xy'], fea['xz'], fea['yz']]).permute((1, 0, 2, 3, 4))
        return ret


class positional_encoding(object):
    """ Positional Encoding (presented in NeRF)

    Args:
        basis_function (str): basis function
    """

    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function
        L = 10
        freq_bands = 2.0 ** np.linspace(0, L - 1, L)
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = 2.0 * p - 1.0
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=2)
        return p


class map2local(object):
    """ Add new keys to the given input

    Args:
        s (float): the defined voxel size
        pos_encoding (str): method for the positional encoding, linear|sin_cos
    """

    def __init__(self, s, pos_encoding='linear'):
        super().__init__()
        self.s = s
        self.pe = positional_encoding(basis_function=pos_encoding)

    def __call__(self, p):
        p = torch.remainder(p, self.s) / self.s
        p = self.pe(p)
        return p


class PatchLocalPoolPointnet(nn.Module):
    """ PointNet-based encoder network with ResNet blocks.
        First transform input points to local system based on the given voxel size.
        Support non-fixed number of point cloud, but need to precompute the index
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
        local_coord (bool): whether to use local coordinate
        pos_encoding (str): method for the positional encoding, linear|sin_cos
        unit_size (float): defined voxel unit size for local system
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5, local_coord=False, pos_encoding='linear', unit_size=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.blocks = nn.ModuleList([ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)])
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding
        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None
        if unet3d:
            raise NotImplementedError()
        else:
            self.unet3d = None
        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')
        if local_coord:
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None
        if pos_encoding == 'sin_cos':
            self.fc_pos = nn.Linear(60, 2 * hidden_dim)
        else:
            self.fc_pos = nn.Linear(dim, 2 * hidden_dim)

    def generate_plane_features(self, index, c):
        c = c.permute(0, 2, 1)
        if index.max() < self.reso_plane ** 2:
            fea_plane = c.new_zeros(c.size(0), self.c_dim, self.reso_plane ** 2)
            fea_plane = scatter_mean(c, index, out=fea_plane)
        else:
            fea_plane = scatter_mean(c, index)
            if fea_plane.shape[-1] > self.reso_plane ** 2:
                fea_plane = fea_plane[:, :, :-1]
        fea_plane = fea_plane.reshape(c.size(0), self.c_dim, self.reso_plane, self.reso_plane)
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)
        return fea_plane

    def generate_grid_features(self, index, c):
        c = c.permute(0, 2, 1)
        if index.max() < self.reso_grid ** 3:
            fea_grid = c.new_zeros(c.size(0), self.c_dim, self.reso_grid ** 3)
            fea_grid = scatter_mean(c, index, out=fea_grid)
        else:
            fea_grid = scatter_mean(c, index)
            if fea_grid.shape[-1] > self.reso_grid ** 3:
                fea_grid = fea_grid[:, :, :-1]
        fea_grid = fea_grid.reshape(c.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)
        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)
        return fea_grid

    def pool_local(self, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = index.keys()
        c_out = 0
        for key in keys:
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key])
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key])
            if self.scatter == scatter_max:
                fea = fea[0]
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, inputs):
        p = inputs['points']
        index = inputs['index']
        batch_size, T, D = p.size()
        if self.map2local:
            pp = self.map2local(p)
            net = self.fc_pos(pp)
        else:
            net = self.fc_pos(p)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)
        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(index['grid'], c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(index['xz'], c)
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(index['xy'], c)
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(index['yz'], c)
        return fea


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):

    def __init__(self, eps=1e-06):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]
        v_length = values.size(1)
        values = values / v_length
        KV = torch.einsum('nshd,nshv->nhdv', K, values)
        Z = 1 / (torch.einsum('nlhd,nhd->nlh', Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum('nlhd,nhdv,nlh->nlhv', Q, KV, Z) * v_length
        return queried_values.contiguous()


class Lie:
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self, w):
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I + A * wx + B * wx @ wx
        return R

    def SO3_to_so3(self, R, eps=1e-07):
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[..., None, None] % np.pi
        lnR = 1 / (2 * self.taylor_A(theta) + 1e-08) * (R - R.transpose(-2, -1))
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    def se3_to_SE3(self, wu):
        w, u = wu.split([3, 3], dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I + A * wx + B * wx @ wx
        V = I + B * wx + C * wx @ wx
        Rt = torch.cat([R, V @ u[..., None]], dim=-1)
        return Rt

    def SE3_to_se3(self, Rt, eps=1e-08):
        R, t = Rt.split([3, 1], dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I - 0.5 * wx + (1 - A / (2 * B)) / (theta ** 2 + eps) * wx @ wx
        u = (invV @ t)[..., 0]
        wu = torch.cat([w, u], dim=-1)
        return wu

    def skew_symmetric(self, w):
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O, -w2, w1], dim=-1), torch.stack([w2, O, -w0], dim=-1), torch.stack([-w1, w0, O], dim=-1)], dim=-2)
        return wx

    def taylor_A(self, x, nth=10):
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            if i > 0:
                denom *= 2 * i * (2 * i + 1)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_B(self, x, nth=10):
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            denom *= (2 * i + 1) * (2 * i + 2)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_C(self, x, nth=10):
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            denom *= (2 * i + 2) * (2 * i + 3)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans


def cam2pix(coords, intr):
    """
    Args:
        coords: [B, T, N, 3]
        intr: [B, T, 3, 3]
    """
    coords = coords.detach()
    B, S, N, _ = coords.shape
    xy_src = coords.reshape(B * S * N, 3).clone()
    intr = intr[:, :, None, ...].repeat(1, 1, N, 1, 1).reshape(B * S * N, 3, 3)
    xy_src = xy_src / (xy_src[..., 2:] + 1e-05)
    xyz_src = (intr @ xy_src[..., None])[..., 0]
    dp_pred = coords[..., 2]
    xyz_src[..., 2] *= dp_pred.reshape(S * N)
    xyz_src = xyz_src.reshape(B, S, N, 3)
    return xyz_src


def get_3d_embedding(xyz, C, cat_coords=True):
    B, N, D = xyz.shape
    assert D == 3
    x = xyz[:, :, 0:1]
    y = xyz[:, :, 1:2]
    z = xyz[:, :, 2:3]
    div_term = (torch.arange(0, C, 2, device=xyz.device, dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C / 2))
    pe_x = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_z = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)
    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)
    pe_z[:, :, 0::2] = torch.sin(z * div_term)
    pe_z[:, :, 1::2] = torch.cos(z * div_term)
    pe = torch.cat([pe_x, pe_y, pe_z], dim=2)
    if cat_coords:
        pe = torch.cat([pe, xyz], dim=2)
    return pe


def pix2cam(coords, intr):
    """
    Args:
        coords: [B, T, N, 3]
        intr: [B, T, 3, 3]
    """
    coords = coords.detach()
    B, S, N, _ = coords.shape
    xy_src = coords.reshape(B * S * N, 3)
    intr = intr[:, :, None, ...].repeat(1, 1, N, 1, 1).reshape(B * S * N, 3, 3)
    xy_src = torch.cat([xy_src[..., :2], torch.ones_like(xy_src[..., :1])], dim=-1)
    xyz_src = (torch.inverse(intr) @ xy_src[..., None])[..., 0]
    dp_pred = coords[..., 2]
    xyz_src_ = xyz_src * dp_pred.reshape(S * N, 1)
    xyz_src_ = xyz_src_.reshape(B, S, N, 3)
    return xyz_src_


def sample_features5d(input, coords):
    """Sample spatio-temporal features

    `sample_features5d(input, coords)` works in the same way as
    :func:`sample_features4d` but for spatio-temporal features and points:
    :attr:`input` is a 5D tensor :math:`(B, T, C, H, W)`, :attr:`coords` is
    a :math:`(B, R1, R2, 3)` tensor of spatio-temporal point :math:`(t_i,
    x_i, y_i)`. The output tensor has shape :math:`(B, R1, R2, C)`.

    Args:
        input (Tensor): spatio-temporal features.
        coords (Tensor): spatio-temporal points.

    Returns:
        Tensor: sampled features.
    """
    B, T, _, _, _ = input.shape
    input = input.permute(0, 2, 1, 3, 4)
    coords = coords.unsqueeze(3)
    feats = bilinear_sampler(input, coords)
    return feats.permute(0, 2, 3, 1, 4).view(B, feats.shape[2], feats.shape[3], feats.shape[1])


def cuda_int32(intIn: 'int'):
    return cupy.int32(intIn)


objCudacache = {}


def cuda_kernel(strFunction: 'str', strKernel: 'str', objVariables: 'typing.Dict'):
    if 'device' not in objCudacache:
        objCudacache['device'] = torch.cuda.get_device_name()
    strKey = strFunction
    for strVariable in objVariables:
        objValue = objVariables[strVariable]
        strKey += strVariable
        if objValue is None:
            continue
        elif type(objValue) == int:
            strKey += str(objValue)
        elif type(objValue) == float:
            strKey += str(objValue)
        elif type(objValue) == bool:
            strKey += str(objValue)
        elif type(objValue) == str:
            strKey += objValue
        elif type(objValue) == torch.Tensor:
            strKey += str(objValue.dtype)
            strKey += str(objValue.shape)
            strKey += str(objValue.stride())
        elif True:
            None
            assert False
    strKey += objCudacache['device']
    if strKey not in objCudacache:
        for strVariable in objVariables:
            objValue = objVariables[strVariable]
            if objValue is None:
                continue
            elif type(objValue) == int:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))
            elif type(objValue) == float:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))
            elif type(objValue) == bool:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))
            elif type(objValue) == str:
                strKernel = strKernel.replace('{{' + strVariable + '}}', objValue)
            elif type(objValue) == torch.Tensor and objValue.dtype == torch.uint8:
                strKernel = strKernel.replace('{{type}}', 'unsigned char')
            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float16:
                strKernel = strKernel.replace('{{type}}', 'half')
            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float32:
                strKernel = strKernel.replace('{{type}}', 'float')
            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float64:
                strKernel = strKernel.replace('{{type}}', 'double')
            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int32:
                strKernel = strKernel.replace('{{type}}', 'int')
            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int64:
                strKernel = strKernel.replace('{{type}}', 'long')
            elif type(objValue) == torch.Tensor:
                None
                assert False
            elif True:
                None
                assert False
        while True:
            objMatch = re.search('(SIZE_)([0-4])(\\()([^\\)]*)(\\))', strKernel)
            if objMatch is None:
                break
            intArg = int(objMatch.group(2))
            strTensor = objMatch.group(4)
            intSizes = objVariables[strTensor].size()
            strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg] if torch.is_tensor(intSizes[intArg]) == False else intSizes[intArg].item()))
        while True:
            objMatch = re.search('(OFFSET_)([0-4])(\\()', strKernel)
            if objMatch is None:
                break
            intStart = objMatch.span()[1]
            intStop = objMatch.span()[1]
            intParentheses = 1
            while True:
                intParentheses += 1 if strKernel[intStop] == '(' else 0
                intParentheses -= 1 if strKernel[intStop] == ')' else 0
                if intParentheses == 0:
                    break
                intStop += 1
            intArgs = int(objMatch.group(2))
            strArgs = strKernel[intStart:intStop].split(',')
            assert intArgs == len(strArgs) - 1
            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()
            strIndex = []
            for intArg in range(intArgs):
                strIndex.append('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')')
            strKernel = strKernel.replace('OFFSET_' + str(intArgs) + '(' + strKernel[intStart:intStop] + ')', '(' + str.join('+', strIndex) + ')')
        while True:
            objMatch = re.search('(VALUE_)([0-4])(\\()', strKernel)
            if objMatch is None:
                break
            intStart = objMatch.span()[1]
            intStop = objMatch.span()[1]
            intParentheses = 1
            while True:
                intParentheses += 1 if strKernel[intStop] == '(' else 0
                intParentheses -= 1 if strKernel[intStop] == ')' else 0
                if intParentheses == 0:
                    break
                intStop += 1
            intArgs = int(objMatch.group(2))
            strArgs = strKernel[intStart:intStop].split(',')
            assert intArgs == len(strArgs) - 1
            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()
            strIndex = []
            for intArg in range(intArgs):
                strIndex.append('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')')
            strKernel = strKernel.replace('VALUE_' + str(intArgs) + '(' + strKernel[intStart:intStop] + ')', strTensor + '[' + str.join('+', strIndex) + ']')
        objCudacache[strKey] = {'strFunction': strFunction, 'strKernel': strKernel}
    return strKey


def softsplat(tenIn: 'torch.Tensor', tenFlow: 'torch.Tensor', tenMetric: 'torch.Tensor', strMode: 'str', tenoutH=None, tenoutW=None):
    assert strMode.split('-')[0] in ['sum', 'avg', 'linear', 'soft']
    if strMode == 'sum':
        assert tenMetric is None
    if strMode == 'avg':
        assert tenMetric is None
    if strMode.split('-')[0] == 'linear':
        assert tenMetric is not None
    if strMode.split('-')[0] == 'soft':
        assert tenMetric is not None
    if strMode == 'avg':
        tenIn = torch.cat([tenIn, tenIn.new_ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]])], 1)
    elif strMode.split('-')[0] == 'linear':
        tenIn = torch.cat([tenIn * tenMetric, tenMetric], 1)
    elif strMode.split('-')[0] == 'soft':
        tenIn = torch.cat([tenIn * tenMetric.exp(), tenMetric.exp()], 1)
    tenOut = softsplat_func.apply(tenIn, tenFlow, tenoutH, tenoutW)
    if strMode.split('-')[0] in ['avg', 'linear', 'soft']:
        tenNormalize = tenOut[:, -1:, :, :]
        if len(strMode.split('-')) == 1:
            tenNormalize = tenNormalize + 1e-07
        elif strMode.split('-')[1] == 'addeps':
            tenNormalize = tenNormalize + 1e-07
        elif strMode.split('-')[1] == 'zeroeps':
            tenNormalize[tenNormalize == 0.0] = 1.0
        elif strMode.split('-')[1] == 'clipeps':
            tenNormalize = tenNormalize.clip(1e-07, None)
        tenOut = tenOut[:, :-1, :, :] / tenNormalize
    return tenOut


class MLPBlock(nn.Module):

    def __init__(self, embedding_dim: 'int', mlp_dim: 'int', act: 'Type[nn.Module]'=nn.GELU) ->None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels: 'int', eps: 'float'=1e-06) ->None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):

    def __init__(self, dim: 'int', init_values: 'Union[float, Tensor]'=1e-05, inplace: 'bool'=False) ->None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: 'Tensor') ->Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_add_residual_stochastic_depth(x: 'Tensor', residual_func: 'Callable[[Tensor], Tensor]', sample_drop_ratio: 'float'=0.0) ->Tensor:
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = torch.randperm(b, device=x.device)[:sample_subset_size]
    x_subset = x[brange]
    residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    residual_scale_factor = b / sample_subset_size
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual, alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


class Block(nn.Module):

    def __init__(self, dim: 'int', num_heads: 'int', mlp_ratio: 'float'=4.0, qkv_bias: 'bool'=False, proj_bias: 'bool'=True, ffn_bias: 'bool'=True, drop: 'float'=0.0, attn_drop: 'float'=0.0, init_values=None, drop_path: 'float'=0.0, act_layer: 'Callable[..., nn.Module]'=nn.GELU, norm_layer: 'Callable[..., nn.Module]'=nn.LayerNorm, attn_class: 'Callable[..., nn.Module]'=Attention, ffn_layer: 'Callable[..., nn.Module]'=Mlp) ->None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: 'Tensor') ->Tensor:

        def attn_residual_func(x: 'Tensor') ->Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: 'Tensor') ->Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(x, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio)
            x = drop_add_residual_stochastic_depth(x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return x, x


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(self, img_size: 'Union[int, Tuple[int, int]]'=224, patch_size: 'Union[int, Tuple[int, int]]'=16, in_chans: 'int'=3, embed_dim: 'int'=768, norm_layer: 'Optional[Callable]'=None, flatten_embedding: 'bool'=True) ->None:
        super().__init__()
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1]
        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: 'Tensor') ->Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size
        assert H % patch_H == 0, f'Input image height {H} is not a multiple of patch height {patch_H}'
        assert W % patch_W == 0, f'Input image width {W} is not a multiple of patch width: {patch_W}'
        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x

    def flops(self) ->float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class ImageEncoderViT(nn.Module):

    def __init__(self, img_size: 'int'=1024, patch_size: 'int'=16, in_chans: 'int'=3, embed_dim: 'int'=768, depth: 'int'=12, num_heads: 'int'=12, mlp_ratio: 'float'=4.0, out_chans: 'int'=256, qkv_bias: 'bool'=True, norm_layer: 'Type[nn.Module]'=nn.LayerNorm, act_layer: 'Type[nn.Module]'=nn.GELU, use_abs_pos: 'bool'=True, use_rel_pos: 'bool'=False, rel_pos_zero_init: 'bool'=True, window_size: 'int'=0, global_attn_indexes: 'Tuple[int, ...]'=()) ->None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed: 'Optional[nn.Parameter]' = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, act_layer=act_layer, use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init, window_size=window_size if i not in global_attn_indexes else 0, input_size=(img_size // patch_size, img_size // patch_size))
            self.blocks.append(block)
        self.neck = nn.Sequential(nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False), LayerNorm2d(out_chans), nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False), LayerNorm2d(out_chans))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x


def _build_spatracker(stride, sequence_len, checkpoint=None):
    spatracker = SpaTracker(stride=stride, S=sequence_len, add_space_attn=True, space_depth=6, time_depth=6)
    if checkpoint is not None:
        with open(checkpoint, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
            if 'model' in state_dict:
                model_paras = spatracker.state_dict()
                paras_dict = {k: v for k, v in state_dict['model'].items() if k in spatracker.state_dict()}
                model_paras.update(paras_dict)
                state_dict = model_paras
        spatracker.load_state_dict(state_dict)
    return spatracker


def build_spatracker_from_cfg(checkpoint=None, seq_length=8):
    return _build_spatracker(stride=4, sequence_len=seq_length, checkpoint=checkpoint)


def build_spatracker(checkpoint: 'str', seq_length: 'int'=8):
    model_name = checkpoint.split('/')[-1].split('.')[0]
    return build_spatracker_from_cfg(checkpoint=checkpoint, seq_length=seq_length)


class SpaTrackerPredictor(torch.nn.Module):

    def __init__(self, checkpoint='cotracker/checkpoints/cotracker_stride_4_wind_8.pth', interp_shape=(384, 512), seq_length=16):
        super().__init__()
        self.interp_shape = interp_shape
        self.support_grid_size = 6
        model = build_spatracker(checkpoint, seq_length=seq_length)
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(self, video, video_depth=None, queries: 'torch.Tensor'=None, segm_mask: 'torch.Tensor'=None, grid_size: 'int'=0, grid_query_frame: 'int'=0, backward_tracking: 'bool'=False, depth_predictor=None, wind_length: 'int'=8):
        if queries is None and grid_size == 0:
            tracks, visibilities, T_Firsts = self._compute_dense_tracks(video, grid_query_frame=grid_query_frame, backward_tracking=backward_tracking, video_depth=video_depth, depth_predictor=depth_predictor, wind_length=wind_length)
        else:
            tracks, visibilities, T_Firsts = self._compute_sparse_tracks(video, queries, segm_mask, grid_size, add_support_grid=grid_size == 0 or segm_mask is not None, grid_query_frame=grid_query_frame, backward_tracking=backward_tracking, video_depth=video_depth, depth_predictor=depth_predictor, wind_length=wind_length)
        return tracks, visibilities, T_Firsts

    def _compute_dense_tracks(self, video, grid_query_frame, grid_size=30, backward_tracking=False, depth_predictor=None, video_depth=None, wind_length=8):
        *_, H, W = video.shape
        grid_step = W // grid_size
        grid_width = W // grid_step
        grid_height = H // grid_step
        tracks = visibilities = T_Firsts = None
        grid_pts = torch.zeros((1, grid_width * grid_height, 3))
        grid_pts[0, :, 0] = grid_query_frame
        for offset in tqdm(range(grid_step * grid_step)):
            ox = offset % grid_step
            oy = offset // grid_step
            grid_pts[0, :, 1] = torch.arange(grid_width).repeat(grid_height) * grid_step + ox
            grid_pts[0, :, 2] = torch.arange(grid_height).repeat_interleave(grid_width) * grid_step + oy
            tracks_step, visibilities_step, T_First_step = self._compute_sparse_tracks(video=video, queries=grid_pts, backward_tracking=backward_tracking, wind_length=wind_length, video_depth=video_depth, depth_predictor=depth_predictor)
            tracks = smart_cat(tracks, tracks_step, dim=2)
            visibilities = smart_cat(visibilities, visibilities_step, dim=2)
            T_Firsts = smart_cat(T_Firsts, T_First_step, dim=1)
        return tracks, visibilities, T_Firsts

    def _compute_sparse_tracks(self, video, queries, segm_mask=None, grid_size=0, add_support_grid=False, grid_query_frame=0, backward_tracking=False, depth_predictor=None, video_depth=None, wind_length=8):
        B, T, C, H, W = video.shape
        assert B == 1
        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode='bilinear')
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])
        if queries is not None:
            queries = queries.clone()
            B, N, D = queries.shape
            assert D == 3
            queries[:, :, 1] *= self.interp_shape[1] / W
            queries[:, :, 2] *= self.interp_shape[0] / H
        elif grid_size > 0:
            grid_pts = get_points_on_a_grid(grid_size, self.interp_shape, device=video.device)
            if segm_mask is not None:
                segm_mask = F.interpolate(segm_mask, tuple(self.interp_shape), mode='nearest')
                point_mask = segm_mask[0, 0][grid_pts[0, :, 1].round().long().cpu(), grid_pts[0, :, 0].round().long().cpu()].bool()
                grid_pts_extra = grid_pts[:, point_mask]
            else:
                grid_pts_extra = None
            if grid_pts_extra is not None:
                total_num = int(grid_pts_extra.shape[1])
                total_num = min(800, total_num)
                pick_idx = torch.randperm(grid_pts_extra.shape[1])[:total_num]
                grid_pts_extra = grid_pts_extra[:, pick_idx]
                queries_extra = torch.cat([torch.ones_like(grid_pts_extra[:, :, :1]) * grid_query_frame, grid_pts_extra], dim=2)
            queries = torch.cat([torch.randint_like(grid_pts[:, :, :1], T), grid_pts], dim=2)
            queries = torch.cat([queries, queries_extra], dim=1)
        if add_support_grid:
            grid_pts = get_points_on_a_grid(self.support_grid_size, self.interp_shape, device=video.device)
            grid_pts = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)
            queries = torch.cat([queries, grid_pts], dim=1)
        if video_depth is None:
            with torch.no_grad():
                if video[0].shape[0] > 30:
                    vidDepths = []
                    for i in range(video[0].shape[0] // 30 + 1):
                        if (i + 1) * 30 > video[0].shape[0]:
                            end_idx = video[0].shape[0]
                        else:
                            end_idx = (i + 1) * 30
                        if end_idx == i * 30:
                            break
                        video_ = video[0][i * 30:end_idx]
                        vidDepths.append(depth_predictor.infer(video_ / 255))
                    video_depth = torch.cat(vidDepths, dim=0)
                else:
                    video_depth = depth_predictor.infer(video[0] / 255)
        video_depth = F.interpolate(video_depth, tuple(self.interp_shape), mode='nearest')
        depths = video_depth
        rgbds = torch.cat([video, depths[None, ...]], dim=2)
        depth_interp = []
        for i in range(queries.shape[1]):
            depth_interp_i = bilinear_sample2d(video_depth[queries[:, i:i + 1, 0].long()], queries[:, i:i + 1, 1], queries[:, i:i + 1, 2])
            depth_interp.append(depth_interp_i)
        depth_interp = torch.cat(depth_interp, dim=1)
        queries = smart_cat(queries, depth_interp, dim=-1)
        del depth_predictor
        torch.cuda.empty_cache()
        t0 = time.time()
        tracks, __, visibilities = self.model(rgbds=rgbds, queries=queries, iters=6, wind_S=wind_length)
        None
        if backward_tracking:
            tracks, visibilities = self._compute_backward_tracks(rgbds, queries, tracks, visibilities)
            if add_support_grid:
                queries[:, -self.support_grid_size ** 2:, 0] = T - 1
        if add_support_grid:
            tracks = tracks[:, :, :-self.support_grid_size ** 2]
            visibilities = visibilities[:, :, :-self.support_grid_size ** 2]
        thr = 0.9
        visibilities = visibilities > thr
        for i in range(len(queries)):
            queries_t = queries[i, :tracks.size(2), 0]
            arange = torch.arange(0, len(queries_t))
            tracks[i, queries_t, arange] = queries[i, :tracks.size(2), 1:]
            visibilities[i, queries_t, arange] = True
        T_First = queries[..., :tracks.size(2), 0]
        tracks[:, :, :, 0] *= W / float(self.interp_shape[1])
        tracks[:, :, :, 1] *= H / float(self.interp_shape[0])
        return tracks, visibilities, T_First

    def _compute_backward_tracks(self, video, queries, tracks, visibilities):
        inv_video = video.flip(1).clone()
        inv_queries = queries.clone()
        inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1
        inv_tracks, __, inv_visibilities = self.model(rgbds=inv_video, queries=queries, iters=6)
        inv_tracks = inv_tracks.flip(1)
        inv_visibilities = inv_visibilities.flip(1)
        mask = tracks == 0
        tracks[mask] = inv_tracks[mask]
        visibilities[mask[:, :, :, 0]] = inv_visibilities[mask[:, :, :, 0]]
        return tracks, visibilities


class KnnModule(torch.nn.Module):
    """
    Gets knn of test features from all processes on a chunk of the train features

    Each rank gets a chunk of the train features as well as a chunk of the test features.
    In `compute_neighbors`, for each rank one after the other, its chunk of test features
    is sent to all devices, partial knns are computed with each chunk of train features
    then collated back on the original device.
    """

    def __init__(self, train_features, train_labels, nb_knn, T, device, num_classes=1000):
        super().__init__()
        self.global_rank = distributed.get_global_rank()
        self.global_size = distributed.get_global_size()
        self.device = device
        self.train_features_rank_T = train_features.chunk(self.global_size)[self.global_rank].T
        self.candidates = train_labels.chunk(self.global_size)[self.global_rank].view(1, -1)
        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = T
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True)
        neighbors_labels = torch.gather(train_labels, 1, indices)
        return topk_sims, neighbors_labels

    def _similarity_for_rank(self, features_rank, source_rank):
        broadcast_shape = torch.tensor(features_rank.shape)
        torch.distributed.broadcast(broadcast_shape, source_rank)
        broadcasted = features_rank
        if self.global_rank != source_rank:
            broadcasted = torch.zeros(*broadcast_shape, dtype=features_rank.dtype, device=self.device)
        torch.distributed.broadcast(broadcasted, source_rank)
        similarity_rank = torch.mm(broadcasted, self.train_features_rank_T)
        candidate_labels = self.candidates.expand(len(similarity_rank), -1)
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def _gather_all_knn_for_rank(self, topk_sims, neighbors_labels, target_rank):
        topk_sims_rank = retrieved_rank = None
        if self.global_rank == target_rank:
            topk_sims_rank = [torch.zeros_like(topk_sims) for _ in range(self.global_size)]
            retrieved_rank = [torch.zeros_like(neighbors_labels) for _ in range(self.global_size)]
        torch.distributed.gather(topk_sims, topk_sims_rank, dst=target_rank)
        torch.distributed.gather(neighbors_labels, retrieved_rank, dst=target_rank)
        if self.global_rank == target_rank:
            topk_sims_rank = torch.cat(topk_sims_rank, dim=1)
            retrieved_rank = torch.cat(retrieved_rank, dim=1)
            results = self._get_knn_sims_and_labels(topk_sims_rank, retrieved_rank)
            return results
        return None

    def compute_neighbors(self, features_rank):
        for rank in range(self.global_size):
            topk_sims, neighbors_labels = self._similarity_for_rank(features_rank, rank)
            results = self._gather_all_knn_for_rank(topk_sims, neighbors_labels, rank)
            if results is not None:
                topk_sims_rank, neighbors_labels_rank = results
        return topk_sims_rank, neighbors_labels_rank

    def forward(self, features_rank):
        """
        Compute the results on all values of `self.nb_knn` neighbors from the full `self.max_k`
        """
        assert all(k <= self.max_k for k in self.nb_knn)
        topk_sims, neighbors_labels = self.compute_neighbors(features_rank)
        batch_size = neighbors_labels.shape[0]
        topk_sims_transform = softmax(topk_sims / self.T, 1)
        matmul = torch.mul(one_hot(neighbors_labels, num_classes=self.num_classes), topk_sims_transform.view(batch_size, -1, 1))
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.nb_knn}
        return probas_for_k


class DictKeysModule(torch.nn.Module):

    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def forward(self, features_dict, targets):
        for k in self.keys:
            features_dict = features_dict[k]
        return {'preds': features_dict, 'target': targets}


class ModuleDictWithForward(torch.nn.ModuleDict):

    def forward(self, *args, **kwargs):
        return {k: module(*args, **kwargs) for k, module in self._modules.items()}


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat((output, torch.mean(intermediate_output[-1][0], dim=1)), dim=-1)
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


class AllClassifiers(nn.Module):

    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class LinearPostprocessor(nn.Module):

    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer('class_mapping', None if class_mapping is None else torch.LongTensor(class_mapping))

    def forward(self, samples, targets):
        preds = self.linear_classifier(samples)
        return {'preds': preds[:, self.class_mapping] if self.class_mapping is not None else preds, 'target': targets}


DEFAULT_MAX_ITER = 1000


_CPU_DEVICE = torch.device('cpu')


class LogRegModule(nn.Module):

    def __init__(self, C, max_iter=DEFAULT_MAX_ITER, dtype=torch.float64, device=_CPU_DEVICE):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.estimator = LogisticRegression(penalty='l2', C=C, max_iter=max_iter, output_type='numpy', tol=1e-12, linesearch_max_iter=50)

    def forward(self, samples, targets):
        samples_device = samples.device
        samples = samples
        if self.device == _CPU_DEVICE:
            samples = samples.numpy()
        probas = self.estimator.predict_proba(samples)
        return {'preds': torch.from_numpy(probas), 'target': targets}

    def fit(self, train_features, train_labels):
        train_features = train_features
        train_labels = train_labels
        if self.device == _CPU_DEVICE:
            train_features = train_features.numpy()
            train_labels = train_labels.numpy()
        self.estimator.fit(train_features, train_labels)


class ModelWithNormalize(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=1, p=2)


class ModelWithIntermediateLayers(nn.Module):

    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(images, self.n_last_blocks, return_class_token=True)
        return features


class MemEffAttention(Attention):

    def forward(self, x: 'Tensor', attn_bias=None) ->Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, 'xFormers is required for nested tensors usage'
            return super().forward(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual, alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(x, brange, residual, scaling=scaling_vector, alpha=residual_scale_factor)
    return x_plus_residual


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias
    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)
    return attn_bias_cache[all_shapes], cat_tensors


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = torch.randperm(b, device=x.device)[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def drop_add_residual_stochastic_depth_list(x_list: 'List[Tensor]', residual_func: 'Callable[[Tensor, Any], Tensor]', sample_drop_ratio: 'float'=0.0, scaling_vector=None) ->Tensor:
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))
    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):

    def forward_nested(self, x_list: 'List[Tensor]') ->List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)
        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: 'Tensor', attn_bias=None) ->Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: 'Tensor', attn_bias=None) ->Tensor:
                return self.mlp(self.norm2(x))
            x_list = drop_add_residual_stochastic_depth_list(x_list, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio, scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None)
            x_list = drop_add_residual_stochastic_depth_list(x_list, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio, scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None)
            return x_list
        else:

            def attn_residual_func(x: 'Tensor', attn_bias=None) ->Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: 'Tensor', attn_bias=None) ->Tensor:
                return self.ls2(self.mlp(self.norm2(x)))
            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            assert XFORMERS_AVAILABLE, 'Please install xFormers for nested tensors usage'
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)


class DINOHead(nn.Module):

    def __init__(self, in_dim, out_dim, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256, mlp_bias=True):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-06 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


class SwiGLUFFN(nn.Module):

    def __init__(self, in_features: 'int', hidden_features: 'Optional[int]'=None, out_features: 'Optional[int]'=None, act_layer: 'Callable[..., nn.Module]'=None, drop: 'float'=0.0, bias: 'bool'=True) ->None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: 'Tensor') ->Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class DINOLoss(nn.Module):

    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()
        B = Q.shape[1] * world_size
        K = Q.shape[0]
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q
        for it in range(n_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)
            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)
            self.updated = True


class iBOTPatchLoss(nn.Module):

    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, 1, patch_out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        self.apply_center_update()
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3):
        teacher_output = teacher_output.float()
        Q = torch.exp(teacher_output / teacher_temp).t()
        B = n_masked_patches_tensor
        dist.all_reduce(B)
        K = Q.shape[0]
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q
        for it in range(n_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_patch_tokens: (B, N, D) tensor
        teacher_patch_tokens: (B, N, D) tensor
        student_masks_flat: (B, N) tensor
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = torch.sum(loss * student_masks_flat.float(), dim=-1) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return -loss.mean()

    def forward_masked(self, student_patch_tokens_masked, teacher_patch_tokens_masked, student_masks_flat, n_masked_patches=None, masks_weight=None):
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        loss = lossfunc(t, s, self.student_temp)
        if masks_weight is None:
            masks_weight = (1 / student_masks_flat.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(student_masks_flat)[student_masks_flat]
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)
            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)
            self.updated = True


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-08)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[::n + 1].fill_(-1)
        _, I = torch.max(dots, dim=1)
        return I

    def forward(self, student_output, eps=1e-08):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.amp.autocast(enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)
            distances = self.pdist(student_output, student_output[I])
            loss = -torch.log(distances + eps).mean()
        return loss


class BlockChunk(nn.ModuleList):

    def forward(self, x):
        for b in self:
            x = b(x)
        return x


def init_weights_vit_timm(module: 'nn.Module', name: 'str'=''):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


logger = logging.getLogger('dinov2')


def named_apply(fn: 'Callable', module: 'nn.Module', name='', depth_first=True, include_root=False) ->nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class DinoVisionTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, ffn_bias=True, proj_bias=True, drop_path_rate=0.0, drop_path_uniform=False, init_values=None, embed_layer=PatchEmbed, act_layer=nn.GELU, block_fn=Block, ffn_layer='mlp', block_chunks=1, num_register_tokens=0, interpolate_antialias=False, interpolate_offset=0.1):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-06)
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if ffn_layer == 'mlp':
            logger.info('using MLP layer as FFN')
            ffn_layer = Mlp
        elif ffn_layer == 'swiglufused' or ffn_layer == 'swiglu':
            logger.info('using SwiGLU layer as FFN')
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == 'identity':
            logger.info('using Identity layer as FFN')

            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError
        blocks_list = [block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, ffn_layer=ffn_layer, init_values=init_values) for i in range(depth)]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i:i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-06)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-06)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2), scale_factor=(sx, sy), mode='bicubic', antialias=self.interpolate_antialias)
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)
        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append({'x_norm_clstoken': x_norm[:, 0], 'x_norm_regtokens': x_norm[:, 1:self.num_register_tokens + 1], 'x_norm_patchtokens': x_norm[:, self.num_register_tokens + 1:], 'x_prenorm': x, 'masks': masks})
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {'x_norm_clstoken': x_norm[:, 0], 'x_norm_regtokens': x_norm[:, 1:self.num_register_tokens + 1], 'x_norm_patchtokens': x_norm[:, self.num_register_tokens + 1:], 'x_prenorm': x, 'masks': masks}

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f'only {len(output)} / {len(blocks_to_take)} blocks found'
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f'only {len(output)} / {len(blocks_to_take)} blocks found'
        return output

    def get_intermediate_layers(self, x: 'torch.Tensor', n: 'Union[int, Sequence]'=1, reshape: 'bool'=False, return_class_token: 'bool'=False, norm=True) ->Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous() for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret['x_norm_clstoken'])


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)


def fuse_params_groups(all_params_groups, keys=('lr_multiplier', 'wd_multiplier', 'is_last_layer')):
    fused_params_groups = defaultdict(lambda : {'params': []})
    for d in all_params_groups:
        identifier = ''
        for k in keys:
            identifier += k + str(d[k]) + '_'
        for k in keys:
            fused_params_groups[identifier][k] = d[k]
        fused_params_groups[identifier]['params'].append(d['params'])
    return fused_params_groups.values()


def get_fsdp_modules(x):
    return FSDP.fsdp_modules(x)


def get_fsdp_wrapper(model_cfg, modules_to_wrap=set()):
    sharding_strategy_dict = {'NO_SHARD': ShardingStrategy.NO_SHARD, 'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP, 'FULL_SHARD': ShardingStrategy.FULL_SHARD}
    dtype_dict = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    mixed_precision_config = MixedPrecision(param_dtype=dtype_dict[model_cfg.mixed_precision.param_dtype], reduce_dtype=dtype_dict[model_cfg.mixed_precision.reduce_dtype], buffer_dtype=dtype_dict[model_cfg.mixed_precision.buffer_dtype])
    sharding_strategy_config = sharding_strategy_dict[model_cfg.sharding_strategy]
    local_rank = distributed.get_local_rank()
    fsdp_wrapper = partial(FSDP, sharding_strategy=sharding_strategy_config, mixed_precision=mixed_precision_config, device_id=local_rank, sync_module_states=True, use_orig_params=True, auto_wrap_policy=ModuleWrapPolicy(modules_to_wrap))
    return fsdp_wrapper


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12, force_is_backbone=False, chunked_blocks=False):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith('backbone') or force_is_backbone:
        if '.pos_embed' in name or '.patch_embed' in name or '.mask_token' in name or '.cls_token' in name:
            layer_id = 0
        elif force_is_backbone and ('pos_embed' in name or 'patch_embed' in name or 'mask_token' in name or 'cls_token' in name):
            layer_id = 0
        elif '.blocks.' in name and '.residual.' not in name:
            layer_id = int(name[name.find('.blocks.'):].split('.')[2]) + 1
        elif chunked_blocks and 'blocks.' in name and 'residual.' not in name:
            layer_id = int(name[name.find('blocks.'):].split('.')[2]) + 1
        elif 'blocks.' in name and 'residual.' not in name:
            layer_id = int(name[name.find('blocks.'):].split('.')[1]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_params_groups_with_decay(model, lr_decay_rate=1.0, patch_embed_lr_mult=1.0):
    chunked_blocks = False
    if hasattr(model, 'n_blocks'):
        logger.info('chunked fsdp')
        n_blocks = model.n_blocks
        chunked_blocks = model.chunked_blocks
    elif hasattr(model, 'blocks'):
        logger.info('first code branch')
        n_blocks = len(model.blocks)
    elif hasattr(model, 'backbone'):
        logger.info('second code branch')
        n_blocks = len(model.backbone.blocks)
    else:
        logger.info('else code branch')
        n_blocks = 0
    all_param_groups = []
    for name, param in model.named_parameters():
        name = name.replace('_fsdp_wrapped_module.', '')
        if not param.requires_grad:
            continue
        decay_rate = get_vit_lr_decay_rate(name, lr_decay_rate, num_layers=n_blocks, force_is_backbone=n_blocks > 0, chunked_blocks=chunked_blocks)
        d = {'params': param, 'is_last_layer': False, 'lr_multiplier': decay_rate, 'wd_multiplier': 1.0, 'name': name}
        if 'last_layer' in name:
            d.update({'is_last_layer': True})
        if name.endswith('.bias') or 'norm' in name or 'gamma' in name:
            d.update({'wd_multiplier': 0.0})
        if 'patch_embed' in name:
            d.update({'lr_multiplier': d['lr_multiplier'] * patch_embed_lr_mult})
        all_param_groups.append(d)
        logger.info(f"{name}: lr_multiplier: {d['lr_multiplier']}, wd_multiplier: {d['wd_multiplier']}")
    return all_param_groups


def has_batchnorms(model):
    bn_types = nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def is_fsdp(x):
    return isinstance(x, FSDP)


def is_sharded_fsdp(x):
    return is_fsdp(x) and x.sharding_strategy is not ShardingStrategy.NO_SHARD


def free_if_fsdp(x):
    if is_sharded_fsdp(x):
        handles = x._handles
        true_list = [(True) for h in handles]
        _reshard(x, handles, true_list)


def reshard_fsdp_model(x):
    for m in get_fsdp_modules(x):
        free_if_fsdp(m)


class SSLMetaArch(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None
        student_model_dict = dict()
        teacher_model_dict = dict()
        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict['backbone'] = student_backbone
        teacher_model_dict['backbone'] = teacher_backbone
        logger.info(f'OPTIONS -- architecture : embed_dim: {embed_dim}')
        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f'OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}')
            student_backbone.load_state_dict(chkpt['model'], strict=False)
        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes
        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head
        logger.info('OPTIONS -- DINO')
        if self.do_dino:
            logger.info(f'OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}')
            logger.info(f'OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}')
            logger.info(f'OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}')
            logger.info(f'OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}')
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(DINOHead, in_dim=embed_dim, out_dim=cfg.dino.head_n_prototypes, hidden_dim=cfg.dino.head_hidden_dim, bottleneck_dim=cfg.dino.head_bottleneck_dim, nlayers=cfg.dino.head_nlayers)
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info('OPTIONS -- DINO -- applying KOLEO regularization')
                self.koleo_loss = KoLeoLoss()
        else:
            logger.info('OPTIONS -- DINO -- not using DINO')
        if self.do_dino or self.do_ibot:
            student_model_dict['dino_head'] = dino_head()
            teacher_model_dict['dino_head'] = dino_head()
        logger.info('OPTIONS -- IBOT')
        logger.info(f'OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}')
        logger.info(f'OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}')
        logger.info(f'OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}')
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, 'please provide a positive mask ratio tuple for ibot'
            assert cfg.ibot.mask_sample_probability > 0, 'please provide a positive mask probability for ibot'
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f'OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}')
                logger.info(f'OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}')
                logger.info(f'OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}')
                logger.info(f'OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}')
                ibot_head = partial(DINOHead, in_dim=embed_dim, out_dim=cfg.ibot.head_n_prototypes, hidden_dim=cfg.ibot.head_hidden_dim, bottleneck_dim=cfg.ibot.head_bottleneck_dim, nlayers=cfg.ibot.head_nlayers)
                student_model_dict['ibot_head'] = ibot_head()
                teacher_model_dict['ibot_head'] = ibot_head()
            else:
                logger.info('OPTIONS -- IBOT -- head shared with DINO')
        self.need_to_synchronize_fsdp_streams = True
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f'Student and Teacher are built: they are both {cfg.student.arch} network.')

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number
        global_crops = images['collated_global_crops']
        local_crops = images['collated_local_crops']
        masks = images['collated_masks']
        mask_indices_list = images['mask_indices_list']
        n_masked_patches_tensor = images['n_masked_patches']
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images['upperbound']
        masks_weight = images['masks_weight']
        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
        do_dino = self.do_dino
        do_ibot = self.do_ibot
        ibot_loss_scale = 1.0 / n_global_crops

        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict['x_norm_clstoken']
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            ibot_teacher_patch_tokens = teacher_backbone_output_dict['x_norm_patchtokens']
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]
            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(ibot_teacher_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list, out=buffer_tensor_teacher[n_cls_tokens:n_cls_tokens + n_masked_patches])
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[n_cls_tokens:n_cls_tokens + n_masked_patches]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(ibot_teacher_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list, out=buffer_tensor_teacher[:n_masked_patches])
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[:n_masked_patches]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None
            if self.cfg.train.centering == 'centering':
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(teacher_cls_tokens_after_head, teacher_temp=teacher_temp).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp)
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])
            elif self.cfg.train.centering == 'sinkhorn_knopp':
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(teacher_cls_tokens_after_head, teacher_temp=teacher_temp).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(masked_teacher_patch_tokens_after_head, teacher_temp=teacher_temp, n_masked_patches_tensor=n_masked_patches_tensor)
            else:
                raise NotImplementedError
            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered
        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        reshard_fsdp_model(self.teacher)
        loss_dict = {}
        loss_accumulator = 0
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone([global_crops, local_crops], masks=[masks, None], is_training=True)
        inputs_for_student_head_list = []
        student_local_cls_tokens = student_local_backbone_output_dict['x_norm_clstoken']
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))
        student_global_cls_tokens = student_global_backbone_output_dict['x_norm_clstoken']
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))
        if do_ibot:
            _dim = student_global_backbone_output_dict['x_norm_clstoken'].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict['x_norm_patchtokens']
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list))
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[:n_masked_patches]
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]
        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops), teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            loss_dict['dino_local_crops_loss'] = dino_local_crops_loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss
        loss_scales = 2
        if do_dino:
            dino_global_crops_loss = self.dino_loss(student_output_list=[student_global_cls_tokens_after_head], teacher_out_softmaxed_centered_list=[teacher_dino_softmaxed_centered_list.flatten(0, 1)]) * loss_scales / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            loss_dict['dino_global_crops_loss'] = dino_global_crops_loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss
            student_cls_tokens = student_global_cls_tokens
            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(self.koleo_loss(p) for p in student_cls_tokens.chunk(2))
                loss_accumulator += koleo_loss
                loss_dict['koleo_loss'] = koleo_loss / loss_scales
        if do_ibot:
            ibot_patch_loss = self.ibot_patch_loss.forward_masked(student_global_masked_patch_tokens_after_head, masked_teacher_ibot_softmaxed_centered, student_masks_flat=masks, n_masked_patches=n_masked_patches, masks_weight=masks_weight) * loss_scales * ibot_loss_scale
            loss_dict['ibot_loss'] = ibot_patch_loss / 2
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss
        self.backprop_loss(loss_accumulator)
        self.fsdp_synchronize_streams()
        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.dino_head._streams = self.teacher.dino_head._streams = self.student.backbone._streams = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(model=m, lr_decay_rate=self.cfg.optim.layerwise_decay, patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult)
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info('fusing param groups')
        for g in fused_params_groups:
            g['foreach'] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info('DISTRIBUTED FSDP -- preparing model for distributed training')
        if has_batchnorms(self.student):
            raise NotImplementedError
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])


class CenterPadding(nn.Module):

    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


class Graph(nn.Module):

    def __init__(self, rel_dp):
        """
            rel_dp: the relative depth map T x 1 x H x W, torch.Tensor
        """
        super(Graph, self).__init__()
        _, T, C, H, W = rel_dp.shape
        self.paras_scale_shift = nn.Parameter(torch.ones(T, 2, 1, 1, requires_grad=True))
        self.focal = nn.Parameter(512 * torch.ones(1, requires_grad=True))
        self.rel_dp = rel_dp

    def forward(self, rgbs, tracker, query2d):
        """
        Args:
            rgbs: the input images B x T x 3 x H x W, torch.Tensor
        """
        tracker = tracker.eval()
        rgbds = torch.cat([rgbs, self.rel_dp], dim=2)
        zeros_one = torch.zeros(query2d.shape[0], query2d.shape[1], 1)
        query2d_input = torch.cat([zeros_one, query2d], dim=-1)
        metric_dp = self.rel_dp * self.paras_scale_shift[None, :, :1, :, :] + self.paras_scale_shift[None, :, 1:, :, :]
        depth_sample = bilinear_sample2d(metric_dp[0, :1], query2d[:, :, 0], query2d[:, :, 1])
        query3d_input = torch.cat([query2d_input, depth_sample.permute(0, 2, 1)], dim=-1)
        tracker.args.depth_near = 0.01
        tracker.args.depth_far = 65
        tracker.args.if_ARAP = True
        tracker.args.Embed3D = True
        with torch.no_grad():
            traj_e, _, vis_e, _ = tracker(rgbds, query3d_input, iters=4, wind_S=12)
        vis = torch.sigmoid(vis_e)
        depth_est = bilinear_sample2d(metric_dp[0, ...], traj_e[0, :, :, 0], traj_e[0, :, :, 1])
        ln = ((depth_est[:, 0, :] - traj_e[0, :, :, 2]) * vis[0]).sum()
        return ln


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AddReadout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AllClassifiers,
     lambda: ([], {'classifiers_dict': {'relu': ReLU()}}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AttractorLayer,
     lambda: ([], {'in_features': 4, 'n_bins': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (AttractorLayerUnnormed,
     lambda: ([], {'in_features': 4, 'n_bins': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BasicEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Block,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BlockChunk,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CenterPadding,
     lambda: ([], {'multiple': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConditionalLogBinomial,
     lambda: ([], {'in_features': 4, 'condition_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DINOHead,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DINOLoss,
     lambda: ([], {'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DownConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Embedder_Fourier,
     lambda: ([], {'input_dim': 4, 'max_freq_log2': 4, 'N_freqs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FullAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm2d,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerScale,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LinearAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (LinearPostprocessor,
     lambda: ([], {'linear_classifier': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (LinearSplitter,
     lambda: ([], {'in_features': 4, 'prev_nbins': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (LogBinomial,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 4, 4])], {})),
    (MLPBlock,
     lambda: ([], {'embedding_dim': 4, 'mlp_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MidasNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (MidasNet_large,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ModelWithNormalize,
     lambda: ([], {'model': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ModuleDictWithForward,
     lambda: ([], {}),
     lambda: ([], {})),
    (NestedTensorBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PatchTransformerEncoder,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (ProjectReadout,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Projector,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualConvUnit,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualConvUnit_custom,
     lambda: ([], {'features': 4, 'activation': torch.nn.ReLU(), 'bn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResnetBlockFC,
     lambda: ([], {'size_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SeedBinRegressor,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SeedBinRegressorUnnormed,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Slice,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwiGLUFFN,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Transpose,
     lambda: ([], {'dim0': 4, 'dim1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (UNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (VGG19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (iBOTPatchLoss,
     lambda: ([], {'patch_out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

