
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


import logging


import torch.nn as nn


from torch import nn as nn


import warnings


import inspect


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.modules.instancenorm import _InstanceNorm


import torch


import torch.nn.functional as F


import torch.utils.checkpoint as cp


import numpy as np


from torch.nn.parallel._functions import _get_stream


import collections


from torch.utils.data.dataloader import default_collate


import functools


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


import torch.distributed as dist


from torch._utils import _flatten_dense_tensors


from torch._utils import _take_tensors


from torch._utils import _unflatten_dense_tensors


from torch.nn.parallel._functions import Scatter as OrigScatter


import time


from collections import OrderedDict


import torchvision


from torch.utils import model_zoo


import torch.multiprocessing as mp


import numbers


from torch.nn.utils import clip_grad


import random


from abc import ABCMeta


from abc import abstractmethod


from torch.utils.data import DataLoader


from inspect import getfullargspec


import copy


from collections import abc


from torch.nn.modules.utils import _pair


from torch.optim import SGD


from torch.nn import GroupNorm


from torch.nn import LayerNorm


from functools import partial


from torch.utils.data import Dataset


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


from collections.abc import Sequence


from torch.utils.data import DistributedSampler as _DistributedSampler


import math


from torch.utils.data import Sampler


from torch import nn


import numpy.matlib


from torch.utils.checkpoint import checkpoint


from torch.autograd import Function


from torch.nn.modules.module import Module


from torch.autograd import gradcheck


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _single


from collections import defaultdict


from typing import List


from torch.nn.modules.conv import _ConvNd


from torch.nn.modules.conv import _ConvTransposeMixin


from torch.nn.modules.pooling import _AdaptiveAvgPoolNd


from torch.nn.modules.pooling import _AdaptiveMaxPoolNd


from torch.nn.modules.pooling import _AvgPoolNd


from torch.nn.modules.pooling import _MaxPoolNd


from torch.utils.cpp_extension import CppExtension


from torch.nn.modules import AvgPool2d


from torch.nn.modules import GroupNorm


from itertools import product


from torch.onnx import OperatorExportTypes


import re


def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    elif dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def load_url_dist(url):
    """ In distributed setting, this function only download checkpoint at
    local rank 0 """
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url)
    return checkpoint


open_mmlab_model_urls = {'vgg16_caffe': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/vgg16_caffe-292e1171.pth', 'resnet50_caffe': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_caffe-788b5fa3.pth', 'resnet50_caffe_bgr': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/pretrain/third_party/resnet50_msra-5891d200.pth', 'resnet101_caffe': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_caffe-3ad79236.pth', 'resnet101_caffe_bgr': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/pretrain/third_party/resnet101_msra-6cc46731.pth', 'resnext50_32x4d': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50-32x4d-0ab1a123.pth', 'resnext101_32x4d': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d-a5af3160.pth', 'resnext101_64x4d': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_64x4d-ee2c6f71.pth', 'contrib/resnet50_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn_thangvubk-ad1730dd.pth', 'detectron/resnet50_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn-9186a21c.pth', 'detectron/resnet101_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_gn-cac0ab98.pth', 'jhu/resnet50_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn_ws-15beedd8.pth', 'jhu/resnet101_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_gn_ws-3e3c308c.pth', 'jhu/resnext50_32x4d_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50_32x4d_gn_ws-0d87ac85.pth', 'jhu/resnext101_32x4d_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d_gn_ws-34ac1a9e.pth', 'jhu/resnext50_32x4d_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50_32x4d_gn-c7e8b754.pth', 'jhu/resnext101_32x4d_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d_gn-ac3bb84e.pth', 'msra/hrnetv2_w18': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/hrnetv2_w18-00eb2006.pth', 'msra/hrnetv2_w32': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/hrnetv2_w32-dc9eeb4f.pth', 'msra/hrnetv2_w40': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/hrnetv2_w40-ed0b031c.pth', 'bninception_caffe': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/pretrain/third_party/bn_inception_caffe-ed2e8665.pth', 'kin400/i3d_r50_f32s2_k400': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/pretrain/third_party/i3d_r50_f32s2_k400-2c57e077.pth', 'kin400/nl3d_r50_f32s2_k400': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/pretrain/third_party/nl3d_r50_f32s2_k400-fa7e7caa.pth'}


def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Either a filepath or URI.
        map_location (str | None): Same as :func:`torch.load`. Default: None.

    Returns:
        dict | OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.
    """
    if filename.startswith('modelzoo://'):
        warnings.warn('The URL scheme of "modelzoo://" is deprecated, please use "torchvision://" instead')
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('torchvision://'):
        model_urls = get_torchvision_models()
        model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('open-mmlab://'):
        model_name = filename[13:]
        checkpoint = load_url_dist(open_mmlab_model_urls[model_name])
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_url_dist(filename)
    else:
        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True, all_missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    load(module)
    load = None
    missing_keys = [key for key in all_missing_keys if 'num_batches_tracked' not in key]
    if unexpected_keys:
        err_msg.append(f"unexpected key in source state_dict: {', '.join(unexpected_keys)}\n")
    if missing_keys:
        err_msg.append(f"missing keys in source state_dict: {', '.join(missing_keys)}\n")
    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            None


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class AlexNet(nn.Module):
    """AlexNet backbone.

    Args:
        num_classes (int): number of classes for classification.
    """

    def __init__(self, num_classes=-1):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))
        if self.num_classes > 0:
            self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Linear(4096, num_classes))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.features(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifier(x)
        return x


class Registry(object):
    """A registry to map strings to classes.

    Args:
        name (str): Registry name.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + f'(name={self._name}, items={self._module_dict})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError(f'module must be a class, but got {type(module_class)}')
        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered in {self.name}')
        self._module_dict[module_name] = module_class

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn('The old API of register_module(module, force=False) is deprecated and will be removed, please use the new API register_module(name=None, force=False, module=None) instead.', DeprecationWarning)
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)
        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module
        if not (name is None or isinstance(name, str)):
            raise TypeError(f'name must be a str, but got {type(name)}')

        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls
        return _register


ACTIVATION_LAYERS = Registry('activation layer')


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(f'the cfg dict must contain the key "type", but got {cfg}')
    if not isinstance(registry, Registry):
        raise TypeError(f'registry must be an mmcv.Registry object, but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(f'default_args must be a dict or None, but got {type(default_args)}')
    args = cfg.copy()
    obj_type = args.pop('type')
    if is_str(obj_type):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)


def build_activation_layer(cfg):
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    return build_from_cfg(cfg, ACTIVATION_LAYERS)


CONV_LAYERS = Registry('conv layer')


def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = CONV_LAYERS.get(layer_type)
    layer = conv_layer(*args, **kwargs, **cfg_)
    return layer


NORM_LAYERS = Registry('norm layer')


def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "abbr", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, 'abbr'):
        return class_type.abbr
    if issubclass(class_type, _InstanceNorm):
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm'


def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]:
            name (str): The layer name consisting of abbreviation and postfix,
                e.g., bn1, gn.
            layer (nn.Module): Created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    norm_layer = NORM_LAYERS.get(layer_type)
    abbr = infer_abbr(norm_layer)
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-05)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return name, layer


PADDING_LAYERS = Registry('padding layer')


def build_padding_layer(cfg, *args, **kwargs):
    """Build padding layer.

    Args:
        cfg (None or dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in PADDING_LAYERS:
        raise KeyError(f'Unrecognized padding type {padding_type}.')
    else:
        padding_layer = PADDING_LAYERS.get(padding_type)
    layer = padding_layer(*args, **kwargs, **cfg_)
    return layer


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias='auto', conv_cfg=None, norm_cfg=None, act_cfg=dict(type='ReLU'), inplace=True, with_spectral_norm=False, padding_mode='zeros', order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias
        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')
        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)
        conv_padding = 0 if self.with_explicit_padding else padding
        self.conv = build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size, stride=stride, padding=conv_padding, dilation=dilation, groups=groups, bias=bias)
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups
        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
        if self.with_norm:
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
            nonlinearity = 'leaky_relu'
            a = self.act_cfg.get('negative_slope', 0.01)
        else:
            nonlinearity = 'relu'
            a = 0
        kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer.

    This module packs `F.pixel_shuffle()` and a nn.Conv2d module together to
    achieve a simple upsampling with pixel shuffle.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of the conv layer to expand the
            channels.
    """

    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super(PixelShufflePack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(self.in_channels, self.out_channels * scale_factor * scale_factor, self.upsample_kernel, padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        xavier_init(self.upsample_conv, distribution='uniform')

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, plugins=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self, block, inplanes, planes, num_blocks, stride=1, avg_down=False, conv_cfg=None, norm_cfg=dict(type='BN'), **kwargs):
        self.block = block
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
            downsample.extend([build_conv_layer(conv_cfg, inplanes, planes * block.expansion, kernel_size=1, stride=conv_stride, bias=False), build_norm_layer(norm_cfg, planes * block.expansion)[1]])
            downsample = nn.Sequential(*downsample)
        layers = []
        layers.append(block(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(inplanes=inplanes, planes=planes, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        super(ResLayer, self).__init__(*layers)


logger_initialized = {}


def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)
    logger_initialized[name] = True
    return logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)
    return logger


def conv3x3(in_planes, out_planes, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation)


def make_vgg_layer(inplanes, planes, num_blocks, dilation=1, with_bn=False, ceil_mode=False):
    layers = []
    for _ in range(num_blocks):
        layers.append(conv3x3(inplanes, planes, dilation))
        if with_bn:
            layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        inplanes = planes
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))
    return layers


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class VGG(nn.Module):
    """VGG backbone.

    Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        with_bn (bool): Use BatchNorm or not.
        num_classes (int): number of classes for classification.
        num_stages (int): VGG stages, normally 5.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers as eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
    """
    arch_settings = {(11): (1, 1, 2, 2, 2), (13): (2, 2, 2, 2, 2), (16): (2, 2, 3, 3, 3), (19): (2, 2, 4, 4, 4)}

    def __init__(self, depth, with_bn=False, num_classes=-1, num_stages=5, dilations=(1, 1, 1, 1, 1), out_indices=(0, 1, 2, 3, 4), frozen_stages=-1, bn_eval=True, bn_frozen=False, ceil_mode=False, with_last_pool=True):
        super(VGG, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for vgg')
        assert num_stages >= 1 and num_stages <= 5
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        assert len(dilations) == num_stages
        assert max(out_indices) <= num_stages
        self.num_classes = num_classes
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.inplanes = 3
        start_idx = 0
        vgg_layers = []
        self.range_sub_modules = []
        for i, num_blocks in enumerate(self.stage_blocks):
            num_modules = num_blocks * (2 + with_bn) + 1
            end_idx = start_idx + num_modules
            dilation = dilations[i]
            planes = 64 * 2 ** i if i < 4 else 512
            vgg_layer = make_vgg_layer(self.inplanes, planes, num_blocks, dilation=dilation, with_bn=with_bn, ceil_mode=ceil_mode)
            vgg_layers.extend(vgg_layer)
            self.inplanes = planes
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx
        if not with_last_pool:
            vgg_layers.pop(-1)
            self.range_sub_modules[-1][1] -= 1
        self.module_name = 'features'
        self.add_module(self.module_name, nn.Sequential(*vgg_layers))
        if self.num_classes > 0:
            self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i, num_blocks in enumerate(self.stage_blocks):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(VGG, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        vgg_layers = getattr(self, self.module_name)
        if mode and self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                for j in range(*self.range_sub_modules[i]):
                    mod = vgg_layers[j]
                    mod.eval()
                    for param in mod.parameters():
                        param.requires_grad = False


def assert_tensor_type(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(f'{args[0].__class__.__name__} has no attribute {func.__name__} for type {args[0].datatype}')
        return func(*args, **kwargs)
    return wrapper


class DataContainer(object):
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self, data, stack=False, padding_value=0, cpu_only=False, pad_dims=2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.data)})'

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def pad_dims(self):
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()


def get_input_device(input):
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, torch.Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception(f'Unknown type {type(input)}.')

