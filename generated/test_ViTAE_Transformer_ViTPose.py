
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


from torch.optim import Optimizer


import time


import warnings


from collections import OrderedDict


import torchvision


from torch.utils import model_zoo


from torch.nn import functional as F


from scipy import interpolate


import numpy as np


import math


import re


import copy


import torch.distributed as dist


import torch.nn as nn


from torch.cuda._utils import _get_device_index


import functools


from inspect import getfullargspec


from collections import abc


from torch._utils import _flatten_dense_tensors


from torch._utils import _take_tensors


from torch._utils import _unflatten_dense_tensors


from abc import ABCMeta


from abc import abstractmethod


from abc import abstractproperty


import random


from functools import partial


from torch.utils.data.dataset import ConcatDataset


from torch.utils.data import Dataset


import copy as cp


from torch.utils.data import ConcatDataset


from torch.utils.data import WeightedRandomSampler


from torch.utils.data import DistributedSampler as _DistributedSampler


import logging


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.functional import pad


import torch.nn.functional as F


import torch.utils.checkpoint as cp


import torch.utils.checkpoint as checkpoint


import torch.multiprocessing as mp


from torch.nn.modules import GroupNorm


from torch.nn.modules import AvgPool2d


from torch.utils.data import DataLoader


from numpy.testing import assert_almost_equal


from numpy.testing import assert_array_almost_equal


from typing import Optional


from torch.hub import HASH_REGEX


from torch.hub import download_url_to_file


class DistributedDataParallelWrapper(nn.Module):
    """A DistributedDataParallel wrapper for models in 3D mesh estimation task.

    In  3D mesh estimation task, there is a need to wrap different modules in
    the models with separate DistributedDataParallel. Otherwise, it will cause
    errors for GAN training.
    More specific, the GAN model, usually has two sub-modules:
    generator and discriminator. If we wrap both of them in one
    standard DistributedDataParallel, it will cause errors during training,
    because when we update the parameters of the generator (or discriminator),
    the parameters of the discriminator (or generator) is not updated, which is
    not allowed for DistributedDataParallel.
    So we design this wrapper to separately wrap DistributedDataParallel
    for generator and discriminator.

    In this wrapper, we perform two operations:
    1. Wrap the modules in the models with separate MMDistributedDataParallel.
        Note that only modules with parameters will be wrapped.
    2. Do scatter operation for 'forward', 'train_step' and 'val_step'.

    Note that the arguments of this wrapper is the same as those in
    `torch.nn.parallel.distributed.DistributedDataParallel`.

    Args:
        module (nn.Module): Module that needs to be wrapped.
        device_ids (list[int | `torch.device`]): Same as that in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
        dim (int, optional): Same as that in the official scatter function in
            pytorch. Defaults to 0.
        broadcast_buffers (bool): Same as that in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
            Defaults to False.
        find_unused_parameters (bool, optional): Same as that in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
            Traverse the autograd graph of all tensors contained in returned
            value of the wrapped module’s forward function. Defaults to False.
        kwargs (dict): Other arguments used in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
    """

    def __init__(self, module, device_ids, dim=0, broadcast_buffers=False, find_unused_parameters=False, **kwargs):
        super().__init__()
        assert len(device_ids) == 1, f'Currently, DistributedDataParallelWrapper only supports onesingle CUDA device for each process.The length of device_ids must be 1, but got {len(device_ids)}.'
        self.module = module
        self.dim = dim
        self.to_ddp(device_ids=device_ids, dim=dim, broadcast_buffers=broadcast_buffers, find_unused_parameters=find_unused_parameters, **kwargs)
        self.output_device = _get_device_index(device_ids[0], True)

    def to_ddp(self, device_ids, dim, broadcast_buffers, find_unused_parameters, **kwargs):
        """Wrap models with separate MMDistributedDataParallel.

        It only wraps the modules with parameters.
        """
        for name, module in self.module._modules.items():
            if next(module.parameters(), None) is None:
                module = module
            elif all(not p.requires_grad for p in module.parameters()):
                module = module
            else:
                module = MMDistributedDataParallel(module, device_ids=device_ids, dim=dim, broadcast_buffers=broadcast_buffers, find_unused_parameters=find_unused_parameters, **kwargs)
            self.module._modules[name] = module

    def scatter(self, inputs, kwargs, device_ids):
        """Scatter function.

        Args:
            inputs (Tensor): Input Tensor.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
            device_ids (int): Device id.
        """
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        """Forward function.

        Args:
            inputs (tuple): Input data.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs, [torch.cuda.current_device()])
        return self.module(*inputs[0], **kwargs[0])

    def train_step(self, *inputs, **kwargs):
        """Train step function.

        Args:
            inputs (Tensor): Input Tensor.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs, [torch.cuda.current_device()])
        output = self.module.train_step(*inputs[0], **kwargs[0])
        return output

    def val_step(self, *inputs, **kwargs):
        """Validation step function.

        Args:
            inputs (tuple): Input data.
            kwargs (dict): Args for ``scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs, [torch.cuda.current_device()])
        output = self.module.val_step(*inputs[0], **kwargs[0])
        return output


DEFAULT_CACHE_DIR = '~/.cache'


ENV_MMCV_HOME = 'MMCV_HOME'


ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'


def mkdir_or_exist(dir_name, mode=511):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def _get_mmcv_home():
    mmcv_home = os.path.expanduser(os.getenv(ENV_MMCV_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'mmcv')))
    mkdir_or_exist(mmcv_home)
    return mmcv_home


def _process_mmcls_checkpoint(checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k[9:]] = v
    new_checkpoint = dict(state_dict=new_state_dict)
    return new_checkpoint


def get_deprecated_model_names():
    deprecate_json_path = osp.join(mmcv.__path__[0], 'model_zoo/deprecated.json')
    deprecate_urls = load_file(deprecate_json_path)
    assert isinstance(deprecate_urls, dict)
    return deprecate_urls


def get_external_models():
    mmcv_home = _get_mmcv_home()
    default_json_path = osp.join(mmcv.__path__[0], 'model_zoo/open_mmlab.json')
    default_urls = load_file(default_json_path)
    assert isinstance(default_urls, dict)
    external_json_path = osp.join(mmcv_home, 'open_mmlab.json')
    if osp.exists(external_json_path):
        external_urls = load_file(external_json_path)
        assert isinstance(external_urls, dict)
        default_urls.update(external_urls)
    return default_urls


def get_mmcls_models():
    mmcls_json_path = osp.join(mmcv.__path__[0], 'model_zoo/mmcls.json')
    mmcls_urls = load_file(mmcls_json_path)
    return mmcls_urls


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


def load_fileclient_dist(filename, backend, map_location):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    allowed_backends = ['ceph']
    if backend not in allowed_backends:
        raise ValueError(f'Load from Backend {backend} is not supported.')
    if rank == 0:
        fileclient = FileClient(backend=backend)
        buffer = io.BytesIO(fileclient.get(filename))
        checkpoint = torch.load(buffer, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            fileclient = FileClient(backend=backend)
            buffer = io.BytesIO(fileclient.get(filename))
            checkpoint = torch.load(buffer, map_location=map_location)
    return checkpoint


def load_url_dist(url, model_dir=None, map_location='cpu'):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url, model_dir=model_dir, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url, model_dir=model_dir, map_location=map_location)
    return checkpoint


def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
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
        model_urls = get_external_models()
        model_name = filename[13:]
        deprecated_urls = get_deprecated_model_names()
        if model_name in deprecated_urls:
            warnings.warn(f'open-mmlab://{model_name} is deprecated in favor of open-mmlab://{deprecated_urls[model_name]}')
            model_name = deprecated_urls[model_name]
        model_url = model_urls[model_name]
        if model_url.startswith(('http://', 'https://')):
            checkpoint = load_url_dist(model_url)
        else:
            filename = osp.join(_get_mmcv_home(), model_url)
            if not osp.isfile(filename):
                raise IOError(f'{filename} is not a checkpoint file')
            checkpoint = torch.load(filename, map_location=map_location)
    elif filename.startswith('mmcls://'):
        model_urls = get_mmcls_models()
        model_name = filename[8:]
        checkpoint = load_url_dist(model_urls[model_name])
        checkpoint = _process_mmcls_checkpoint(checkpoint)
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_url_dist(filename)
    elif filename.startswith('pavi://'):
        model_path = filename[7:]
        checkpoint = load_pavimodel_dist(model_path, map_location=map_location)
    elif filename.startswith('s3://'):
        checkpoint = load_fileclient_dist(filename, backend='ceph', map_location=map_location)
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
        if is_module_wrapper(module):
            module = module.module
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


def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict_tmp = checkpoint['state_dict']
    else:
        state_dict_tmp = checkpoint
    state_dict = OrderedDict()
    for k, v in state_dict_tmp.items():
        if k.startswith('module.backbone.'):
            state_dict[k[16:]] = v
        elif k.startswith('module.'):
            state_dict[k[7:]] = v
        elif k.startswith('backbone.'):
            state_dict[k[9:]] = v
        else:
            state_dict[k] = v
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    """Base backbone.

    This class defines the basic functions of a backbone. Any backbone that
    inherits this class should at least define its own `forward` function.
    """

    def init_weights(self, pretrained=None, patch_padding='pad', part_features=None):
        """Init backbone weights.

        Args:
            pretrained (str | None): If pretrained is a string, then it
                initializes backbone weights by loading the pretrained
                checkpoint. If pretrained is None, then it follows default
                initializer or customized initializer in subclasses.
        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger, patch_padding=patch_padding, part_features=part_features)
        elif pretrained is None:
            pass
        else:
            raise TypeError(f'pretrained must be a str or None. But received {type(pretrained)}.')

    @abstractmethod
    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """


class CpmBlock(nn.Module):
    """CpmBlock for Convolutional Pose Machine.

    Args:
        in_channels (int): Input channels of this block.
        channels (list): Output channels of each conv module.
        kernels (list): Kernel sizes of each conv module.
    """

    def __init__(self, in_channels, channels=(128, 128, 128), kernels=(11, 11, 11), norm_cfg=None):
        super().__init__()
        assert len(channels) == len(kernels)
        layers = []
        for i in range(len(channels)):
            if i == 0:
                input_channels = in_channels
            else:
                input_channels = channels[i - 1]
            layers.append(ConvModule(input_channels, channels[i], kernels[i], padding=(kernels[i] - 1) // 2, norm_cfg=norm_cfg))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Model forward function."""
        out = self.model(x)
        return out


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use `get_logger` method in mmcv to get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmpose".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)


class CPM(BaseBackbone):
    """CPM backbone.

    Convolutional Pose Machines.
    More details can be found in the `paper
    <https://arxiv.org/abs/1602.00134>`__ .

    Args:
        in_channels (int): The input channels of the CPM.
        out_channels (int): The output channels of the CPM.
        feat_channels (int): Feature channel of each CPM stage.
        middle_channels (int): Feature channel of conv after the middle stage.
        num_stages (int): Number of stages.
        norm_cfg (dict): Dictionary to construct and config norm layer.

    Example:
        >>> from mmpose.models import CPM
        >>> import torch
        >>> self = CPM(3, 17)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 368, 368)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 17, 46, 46)
        (1, 17, 46, 46)
        (1, 17, 46, 46)
        (1, 17, 46, 46)
        (1, 17, 46, 46)
        (1, 17, 46, 46)
    """

    def __init__(self, in_channels, out_channels, feat_channels=128, middle_channels=32, num_stages=6, norm_cfg=dict(type='BN', requires_grad=True)):
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        assert in_channels == 3
        self.num_stages = num_stages
        assert self.num_stages >= 1
        self.stem = nn.Sequential(ConvModule(in_channels, 128, 9, padding=4, norm_cfg=norm_cfg), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), ConvModule(128, 128, 9, padding=4, norm_cfg=norm_cfg), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), ConvModule(128, 128, 9, padding=4, norm_cfg=norm_cfg), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), ConvModule(128, 32, 5, padding=2, norm_cfg=norm_cfg), ConvModule(32, 512, 9, padding=4, norm_cfg=norm_cfg), ConvModule(512, 512, 1, padding=0, norm_cfg=norm_cfg), ConvModule(512, out_channels, 1, padding=0, act_cfg=None))
        self.middle = nn.Sequential(ConvModule(in_channels, 128, 9, padding=4, norm_cfg=norm_cfg), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), ConvModule(128, 128, 9, padding=4, norm_cfg=norm_cfg), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), ConvModule(128, 128, 9, padding=4, norm_cfg=norm_cfg), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.cpm_stages = nn.ModuleList([CpmBlock(middle_channels + out_channels, channels=[feat_channels, feat_channels, feat_channels], kernels=[11, 11, 11], norm_cfg=norm_cfg) for _ in range(num_stages - 1)])
        self.middle_conv = nn.ModuleList([nn.Sequential(ConvModule(128, middle_channels, 5, padding=2, norm_cfg=norm_cfg)) for _ in range(num_stages - 1)])
        self.out_convs = nn.ModuleList([nn.Sequential(ConvModule(feat_channels, feat_channels, 1, padding=0, norm_cfg=norm_cfg), ConvModule(feat_channels, out_channels, 1, act_cfg=None)) for _ in range(num_stages - 1)])

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Model forward function."""
        stage1_out = self.stem(x)
        middle_out = self.middle(x)
        out_feats = []
        out_feats.append(stage1_out)
        for ind in range(self.num_stages - 1):
            single_stage = self.cpm_stages[ind]
            out_conv = self.out_convs[ind]
            inp_feat = torch.cat([out_feats[-1], self.middle_conv[ind](middle_out)], 1)
            cpm_feat = single_stage(inp_feat)
            out_feat = out_conv(cpm_feat)
            out_feats.append(out_feat)
        return out_feats


class BasicBlock(nn.Module):
    """BasicBlock for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1. Default: 1.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): `pytorch` or `caffe`. It is unused and reserved for
            unified API with Bottleneck.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self, in_channels, out_channels, expansion=1, stride=1, dilation=1, downsample=None, style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN')):
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, out_channels, postfix=2)
        self.conv1 = build_conv_layer(conv_cfg, in_channels, self.mid_channels, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg, self.mid_channels, out_channels, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)
        return out


class ViPNAS_Bottleneck(nn.Module):
    """Bottleneck block for ViPNAS_ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        kernel_size (int): kernel size of conv2 searched in ViPANS.
        groups (int): group number of conv2 searched in ViPNAS.
        attention (bool): whether to use attention module in the end of
            the block.
    """

    def __init__(self, in_channels, out_channels, expansion=4, stride=1, dilation=1, downsample=None, style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), kernel_size=3, groups=1, attention=False):
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        assert style in ['pytorch', 'caffe']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, out_channels, postfix=3)
        self.conv1 = build_conv_layer(conv_cfg, in_channels, self.mid_channels, kernel_size=1, stride=self.conv1_stride, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg, self.mid_channels, self.mid_channels, kernel_size=kernel_size, stride=self.conv2_stride, padding=kernel_size // 2, groups=groups, dilation=dilation, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(conv_cfg, self.mid_channels, out_channels, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        if attention:
            self.attention = ContextBlock(out_channels, max(1.0 / 16, 16.0 / out_channels))
        else:
            self.attention = None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.norm3(out)
            if self.attention is not None:
                out = self.attention(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)
        return out


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


class HourglassModule(nn.Module):
    """Hourglass Module for HourglassNet backbone.

    Generate module recursively and use BasicBlock as the base unit.

    Args:
        depth (int): Depth of current HourglassModule.
        stage_channels (list[int]): Feature channels of sub-modules in current
            and follow-up HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in current and
            follow-up HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self, depth, stage_channels, stage_blocks, norm_cfg=dict(type='BN', requires_grad=True)):
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.depth = depth
        cur_block = stage_blocks[0]
        next_block = stage_blocks[1]
        cur_channel = stage_channels[0]
        next_channel = stage_channels[1]
        self.up1 = ResLayer(BasicBlock, cur_block, cur_channel, cur_channel, norm_cfg=norm_cfg)
        self.low1 = ResLayer(BasicBlock, cur_block, cur_channel, next_channel, stride=2, norm_cfg=norm_cfg)
        if self.depth > 1:
            self.low2 = HourglassModule(depth - 1, stage_channels[1:], stage_blocks[1:])
        else:
            self.low2 = ResLayer(BasicBlock, next_block, next_channel, next_channel, norm_cfg=norm_cfg)
        self.low3 = ResLayer(BasicBlock, cur_block, next_channel, cur_channel, norm_cfg=norm_cfg, downsample_first=False)
        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        """Model forward function."""
        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class HourglassNet(BaseBackbone):
    """HourglassNet backbone.

    Stacked Hourglass Networks for Human Pose Estimation.
    More details can be found in the `paper
    <https://arxiv.org/abs/1603.06937>`__ .

    Args:
        downsample_times (int): Downsample times in a HourglassModule.
        num_stacks (int): Number of HourglassModule modules stacked,
            1 for Hourglass-52, 2 for Hourglass-104.
        stage_channels (list[int]): Feature channel of each sub-module in a
            HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in a
            HourglassModule.
        feat_channel (int): Feature channel of conv after a HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.

    Example:
        >>> from mmpose.models import HourglassNet
        >>> import torch
        >>> self = HourglassNet()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 256, 128, 128)
        (1, 256, 128, 128)
    """

    def __init__(self, downsample_times=5, num_stacks=2, stage_channels=(256, 256, 384, 384, 384, 512), stage_blocks=(2, 2, 2, 2, 2, 4), feat_channel=256, norm_cfg=dict(type='BN', requires_grad=True)):
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.num_stacks = num_stacks
        assert self.num_stacks >= 1
        assert len(stage_channels) == len(stage_blocks)
        assert len(stage_channels) > downsample_times
        cur_channel = stage_channels[0]
        self.stem = nn.Sequential(ConvModule(3, 128, 7, padding=3, stride=2, norm_cfg=norm_cfg), ResLayer(BasicBlock, 1, 128, 256, stride=2, norm_cfg=norm_cfg))
        self.hourglass_modules = nn.ModuleList([HourglassModule(downsample_times, stage_channels, stage_blocks) for _ in range(num_stacks)])
        self.inters = ResLayer(BasicBlock, num_stacks - 1, cur_channel, cur_channel, norm_cfg=norm_cfg)
        self.conv1x1s = nn.ModuleList([ConvModule(cur_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None) for _ in range(num_stacks - 1)])
        self.out_convs = nn.ModuleList([ConvModule(cur_channel, feat_channel, 3, padding=1, norm_cfg=norm_cfg) for _ in range(num_stacks)])
        self.remap_convs = nn.ModuleList([ConvModule(feat_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None) for _ in range(num_stacks - 1)])
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Model forward function."""
        inter_feat = self.stem(x)
        out_feats = []
        for ind in range(self.num_stacks):
            single_hourglass = self.hourglass_modules[ind]
            out_conv = self.out_convs[ind]
            hourglass_feat = single_hourglass(inter_feat)
            out_feat = out_conv(hourglass_feat)
            out_feats.append(out_feat)
            if ind < self.num_stacks - 1:
                inter_feat = self.conv1x1s[ind](inter_feat) + self.remap_convs[ind](out_feat)
                inter_feat = self.inters[ind](self.relu(inter_feat))
        return out_feats


class HourglassAEModule(nn.Module):
    """Modified Hourglass Module for HourglassNet_AE backbone.

    Generate module recursively and use BasicBlock as the base unit.

    Args:
        depth (int): Depth of current HourglassModule.
        stage_channels (list[int]): Feature channels of sub-modules in current
            and follow-up HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self, depth, stage_channels, norm_cfg=dict(type='BN', requires_grad=True)):
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.depth = depth
        cur_channel = stage_channels[0]
        next_channel = stage_channels[1]
        self.up1 = ConvModule(cur_channel, cur_channel, 3, padding=1, norm_cfg=norm_cfg)
        self.pool1 = MaxPool2d(2, 2)
        self.low1 = ConvModule(cur_channel, next_channel, 3, padding=1, norm_cfg=norm_cfg)
        if self.depth > 1:
            self.low2 = HourglassAEModule(depth - 1, stage_channels[1:])
        else:
            self.low2 = ConvModule(next_channel, next_channel, 3, padding=1, norm_cfg=norm_cfg)
        self.low3 = ConvModule(next_channel, cur_channel, 3, padding=1, norm_cfg=norm_cfg)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        """Model forward function."""
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class HourglassAENet(BaseBackbone):
    """Hourglass-AE Network proposed by Newell et al.

    Associative Embedding: End-to-End Learning for Joint
    Detection and Grouping.

    More details can be found in the `paper
    <https://arxiv.org/abs/1611.05424>`__ .

    Args:
        downsample_times (int): Downsample times in a HourglassModule.
        num_stacks (int): Number of HourglassModule modules stacked,
            1 for Hourglass-52, 2 for Hourglass-104.
        stage_channels (list[int]): Feature channel of each sub-module in a
            HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in a
            HourglassModule.
        feat_channels (int): Feature channel of conv after a HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.

    Example:
        >>> from mmpose.models import HourglassAENet
        >>> import torch
        >>> self = HourglassAENet()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 512, 512)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 34, 128, 128)
    """

    def __init__(self, downsample_times=4, num_stacks=1, out_channels=34, stage_channels=(256, 384, 512, 640, 768), feat_channels=256, norm_cfg=dict(type='BN', requires_grad=True)):
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.num_stacks = num_stacks
        assert self.num_stacks >= 1
        assert len(stage_channels) > downsample_times
        cur_channels = stage_channels[0]
        self.stem = nn.Sequential(ConvModule(3, 64, 7, padding=3, stride=2, norm_cfg=norm_cfg), ConvModule(64, 128, 3, padding=1, norm_cfg=norm_cfg), MaxPool2d(2, 2), ConvModule(128, 128, 3, padding=1, norm_cfg=norm_cfg), ConvModule(128, feat_channels, 3, padding=1, norm_cfg=norm_cfg))
        self.hourglass_modules = nn.ModuleList([nn.Sequential(HourglassAEModule(downsample_times, stage_channels, norm_cfg=norm_cfg), ConvModule(feat_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg), ConvModule(feat_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)) for _ in range(num_stacks)])
        self.out_convs = nn.ModuleList([ConvModule(cur_channels, out_channels, 1, padding=0, norm_cfg=None, act_cfg=None) for _ in range(num_stacks)])
        self.remap_out_convs = nn.ModuleList([ConvModule(out_channels, feat_channels, 1, norm_cfg=norm_cfg, act_cfg=None) for _ in range(num_stacks - 1)])
        self.remap_feature_convs = nn.ModuleList([ConvModule(feat_channels, feat_channels, 1, norm_cfg=norm_cfg, act_cfg=None) for _ in range(num_stacks - 1)])
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Model forward function."""
        inter_feat = self.stem(x)
        out_feats = []
        for ind in range(self.num_stacks):
            single_hourglass = self.hourglass_modules[ind]
            out_conv = self.out_convs[ind]
            hourglass_feat = single_hourglass(inter_feat)
            out_feat = out_conv(hourglass_feat)
            out_feats.append(out_feat)
            if ind < self.num_stacks - 1:
                inter_feat = inter_feat + self.remap_out_convs[ind](out_feat) + self.remap_feature_convs[ind](hourglass_feat)
        return out_feats


class HRModule(nn.Module):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(self, num_branches, blocks, num_blocks, in_channels, num_channels, multiscale_output=False, with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), upsample_cfg=dict(mode='nearest', align_corners=None)):
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self._check_branches(num_branches, num_blocks, in_channels, num_channels)
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.upsample_cfg = upsample_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _check_branches(num_branches, num_blocks, in_channels, num_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) != NUM_BLOCKS({len(num_blocks)})'
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) != NUM_CHANNELS({len(num_channels)})'
            raise ValueError(error_msg)
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) != NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        """Make one branch."""
        downsample = None
        if stride != 1 or self.in_channels[branch_index] != num_channels[branch_index] * get_expansion(block):
            downsample = nn.Sequential(build_conv_layer(self.conv_cfg, self.in_channels[branch_index], num_channels[branch_index] * get_expansion(block), kernel_size=1, stride=stride, bias=False), build_norm_layer(self.norm_cfg, num_channels[branch_index] * get_expansion(block))[1])
        layers = []
        layers.append(block(self.in_channels[branch_index], num_channels[branch_index] * get_expansion(block), stride=stride, downsample=downsample, with_cp=self.with_cp, norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg))
        self.in_channels[branch_index] = num_channels[branch_index] * get_expansion(block)
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.in_channels[branch_index], num_channels[branch_index] * get_expansion(block), with_cp=self.with_cp, norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """Make branches."""
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[i], kernel_size=1, stride=1, padding=0, bias=False), build_norm_layer(self.norm_cfg, in_channels[i])[1], nn.Upsample(scale_factor=2 ** (j - i), mode=self.upsample_cfg['mode'], align_corners=self.upsample_cfg['align_corners'])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[i], kernel_size=3, stride=2, padding=1, bias=False), build_norm_layer(self.norm_cfg, in_channels[i])[1]))
                        else:
                            conv_downsamples.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[j], kernel_size=3, stride=2, padding=1, bias=False), build_norm_layer(self.norm_cfg, in_channels[j])[1], nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class SpatialWeighting(nn.Module):
    """Spatial weighting module.

    Args:
        channels (int): The channels of the module.
        ratio (int): channel reduction ratio.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: (dict(type='ReLU'), dict(type='Sigmoid')).
            The last ConvModule uses Sigmoid by default.
    """

    def __init__(self, channels, ratio=16, conv_cfg=None, norm_cfg=None, act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = act_cfg, act_cfg
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(in_channels=channels, out_channels=int(channels / ratio), kernel_size=1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg[0])
        self.conv2 = ConvModule(in_channels=int(channels / ratio), out_channels=channels, kernel_size=1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class CrossResolutionWeighting(nn.Module):
    """Cross-resolution channel weighting module.

    Args:
        channels (int): The channels of the module.
        ratio (int): channel reduction ratio.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: (dict(type='ReLU'), dict(type='Sigmoid')).
            The last ConvModule uses Sigmoid by default.
    """

    def __init__(self, channels, ratio=16, conv_cfg=None, norm_cfg=None, act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = act_cfg, act_cfg
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.channels = channels
        total_channel = sum(channels)
        self.conv1 = ConvModule(in_channels=total_channel, out_channels=int(total_channel / ratio), kernel_size=1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg[0])
        self.conv2 = ConvModule(in_channels=int(total_channel / ratio), out_channels=total_channel, kernel_size=1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg[1])

    def forward(self, x):
        mini_size = x[-1].size()[-2:]
        out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [(s * F.interpolate(a, size=s.size()[-2:], mode='nearest')) for s, a in zip(x, out)]
        return out


def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """
    batch_size, num_channels, height, width = x.size()
    assert num_channels % groups == 0, 'num_channels should be divisible by groups'
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


class ConditionalChannelWeighting(nn.Module):
    """Conditional channel weighting block.

    Args:
        in_channels (int): The input channels of the block.
        stride (int): Stride of the 3x3 convolution layer.
        reduce_ratio (int): channel reduction ratio.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self, in_channels, stride, reduce_ratio, conv_cfg=None, norm_cfg=dict(type='BN'), with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.stride = stride
        assert stride in [1, 2]
        branch_channels = [(channel // 2) for channel in in_channels]
        self.cross_resolution_weighting = CrossResolutionWeighting(branch_channels, ratio=reduce_ratio, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.depthwise_convs = nn.ModuleList([ConvModule(channel, channel, kernel_size=3, stride=self.stride, padding=1, groups=channel, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None) for channel in branch_channels])
        self.spatial_weighting = nn.ModuleList([SpatialWeighting(channels=channel, ratio=4) for channel in branch_channels])

    def forward(self, x):

        def _inner_forward(x):
            x = [s.chunk(2, dim=1) for s in x]
            x1 = [s[0] for s in x]
            x2 = [s[1] for s in x]
            x2 = self.cross_resolution_weighting(x2)
            x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
            x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]
            out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
            out = [channel_shuffle(s, 2) for s in out]
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class Stem(nn.Module):
    """Stem network block.

    Args:
        in_channels (int): The input channels of the block.
        stem_channels (int): Output channels of the stem layer.
        out_channels (int): The output channels of the block.
        expand_ratio (int): adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self, in_channels, stem_channels, out_channels, expand_ratio, conv_cfg=None, norm_cfg=dict(type='BN'), with_cp=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.conv1 = ConvModule(in_channels=in_channels, out_channels=stem_channels, kernel_size=3, stride=2, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=dict(type='ReLU'))
        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels
        self.branch1 = nn.Sequential(ConvModule(branch_channels, branch_channels, kernel_size=3, stride=2, padding=1, groups=branch_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None), ConvModule(branch_channels, inc_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU')))
        self.expand_conv = ConvModule(branch_channels, mid_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
        self.depthwise_conv = ConvModule(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.linear_conv = ConvModule(mid_channels, branch_channels if stem_channels == self.out_channels else stem_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))

    def forward(self, x):

        def _inner_forward(x):
            x = self.conv1(x)
            x1, x2 = x.chunk(2, dim=1)
            x2 = self.expand_conv(x2)
            x2 = self.depthwise_conv(x2)
            x2 = self.linear_conv(x2)
            out = torch.cat((self.branch1(x1), x2), dim=1)
            out = channel_shuffle(out, 2)
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class IterativeHead(nn.Module):
    """Extra iterative head for feature learning.

    Args:
        in_channels (int): The input channels of the block.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
    """

    def __init__(self, in_channels, norm_cfg=dict(type='BN')):
        super().__init__()
        projects = []
        num_branchs = len(in_channels)
        self.in_channels = in_channels[::-1]
        for i in range(num_branchs):
            if i != num_branchs - 1:
                projects.append(DepthwiseSeparableConvModule(in_channels=self.in_channels[i], out_channels=self.in_channels[i + 1], kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'), dw_act_cfg=None, pw_act_cfg=dict(type='ReLU')))
            else:
                projects.append(DepthwiseSeparableConvModule(in_channels=self.in_channels[i], out_channels=self.in_channels[i], kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'), dw_act_cfg=None, pw_act_cfg=dict(type='ReLU')))
        self.projects = nn.ModuleList(projects)

    def forward(self, x):
        x = x[::-1]
        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(last_x, size=s.size()[-2:], mode='bilinear', align_corners=True)
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s
        return y[::-1]


class ShuffleUnit(nn.Module):
    """ShuffleUnit block.

    ShuffleNet unit with pointwise group convolution (GConv) and channel
    shuffle.

    Args:
        in_channels (int): The input channels of the ShuffleUnit.
        out_channels (int): The output channels of the ShuffleUnit.
        groups (int, optional): The number of groups to be used in grouped 1x1
            convolutions in each ShuffleUnit. Default: 3
        first_block (bool, optional): Whether it is the first ShuffleUnit of a
            sequential ShuffleUnits. Default: True, which means not using the
            grouped 1x1 convolution.
        combine (str, optional): The ways to combine the input and output
            branches. Default: 'add'.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, in_channels, out_channels, groups=3, first_block=True, combine='add', conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), with_cp=False):
        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_block = first_block
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4
        self.with_cp = with_cp
        if self.combine == 'add':
            self.depthwise_stride = 1
            self._combine_func = self._add
            assert in_channels == out_channels, 'in_channels must be equal to out_channels when combine is add'
        elif self.combine == 'concat':
            self.depthwise_stride = 2
            self._combine_func = self._concat
            self.out_channels -= self.in_channels
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f'Cannot combine tensors with {self.combine}. Only "add" and "concat" are supported')
        self.first_1x1_groups = 1 if first_block else self.groups
        self.g_conv_1x1_compress = ConvModule(in_channels=self.in_channels, out_channels=self.bottleneck_channels, kernel_size=1, groups=self.first_1x1_groups, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.depthwise_conv3x3_bn = ConvModule(in_channels=self.bottleneck_channels, out_channels=self.bottleneck_channels, kernel_size=3, stride=self.depthwise_stride, padding=1, groups=self.bottleneck_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.g_conv_1x1_expand = ConvModule(in_channels=self.bottleneck_channels, out_channels=self.out_channels, kernel_size=1, groups=self.groups, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.act = build_activation_layer(act_cfg)

    @staticmethod
    def _add(x, out):
        return x + out

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, x):

        def _inner_forward(x):
            residual = x
            out = self.g_conv_1x1_compress(x)
            out = self.depthwise_conv3x3_bn(out)
            if self.groups > 1:
                out = channel_shuffle(out, self.groups)
            out = self.g_conv_1x1_expand(out)
            if self.combine == 'concat':
                residual = self.avgpool(residual)
                out = self.act(out)
                out = self._combine_func(residual, out)
            else:
                out = self._combine_func(residual, out)
                out = self.act(out)
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class LiteHRModule(nn.Module):
    """High-Resolution Module for LiteHRNet.

    It contains conditional channel weighting blocks and
    shuffle blocks.


    Args:
        num_branches (int): Number of branches in the module.
        num_blocks (int): Number of blocks in the module.
        in_channels (list(int)): Number of input image channels.
        reduce_ratio (int): Channel reduction ratio.
        module_type (str): 'LITE' or 'NAIVE'
        multiscale_output (bool): Whether to output multi-scale features.
        with_fuse (bool): Whether to use fuse layers.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    def __init__(self, num_branches, num_blocks, in_channels, reduce_ratio, module_type, multiscale_output=False, with_fuse=True, conv_cfg=None, norm_cfg=dict(type='BN'), with_cp=False):
        super().__init__()
        self._check_branches(num_branches, in_channels)
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.module_type = module_type
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        if self.module_type.upper() == 'LITE':
            self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio)
        elif self.module_type.upper() == 'NAIVE':
            self.layers = self._make_naive_branches(num_branches, num_blocks)
        else:
            raise ValueError("module_type should be either 'LITE' or 'NAIVE'.")
        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _check_branches(self, num_branches, in_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) != NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, reduce_ratio, stride=1):
        """Make channel weighting blocks."""
        layers = []
        for i in range(num_blocks):
            layers.append(ConditionalChannelWeighting(self.in_channels, stride=stride, reduce_ratio=reduce_ratio, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, with_cp=self.with_cp))
        return nn.Sequential(*layers)

    def _make_one_branch(self, branch_index, num_blocks, stride=1):
        """Make one branch."""
        layers = []
        layers.append(ShuffleUnit(self.in_channels[branch_index], self.in_channels[branch_index], stride=stride, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=dict(type='ReLU'), with_cp=self.with_cp))
        for i in range(1, num_blocks):
            layers.append(ShuffleUnit(self.in_channels[branch_index], self.in_channels[branch_index], stride=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=dict(type='ReLU'), with_cp=self.with_cp))
        return nn.Sequential(*layers)

    def _make_naive_branches(self, num_branches, num_blocks):
        """Make branches."""
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, num_blocks))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[i], kernel_size=1, stride=1, padding=0, bias=False), build_norm_layer(self.norm_cfg, in_channels[i])[1], nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[j], kernel_size=3, stride=2, padding=1, groups=in_channels[j], bias=False), build_norm_layer(self.norm_cfg, in_channels[j])[1], build_conv_layer(self.conv_cfg, in_channels[j], in_channels[i], kernel_size=1, stride=1, padding=0, bias=False), build_norm_layer(self.norm_cfg, in_channels[i])[1]))
                        else:
                            conv_downsamples.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[j], kernel_size=3, stride=2, padding=1, groups=in_channels[j], bias=False), build_norm_layer(self.norm_cfg, in_channels[j])[1], build_conv_layer(self.conv_cfg, in_channels[j], in_channels[j], kernel_size=1, stride=1, padding=0, bias=False), build_norm_layer(self.norm_cfg, in_channels[j])[1], nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.layers[0](x[0])]
        if self.module_type.upper() == 'LITE':
            out = self.layers(x)
        elif self.module_type.upper() == 'NAIVE':
            for i in range(self.num_branches):
                x[i] = self.layers[i](x[i])
            out = x
        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.num_branches):
                    if i == j:
                        y += out[j]
                    else:
                        y += self.fuse_layers[i][j](out[j])
                out_fuse.append(self.relu(y))
            out = out_fuse
        if not self.multiscale_output:
            out = [out[0]]
        return out


class LiteHRNet(nn.Module):
    """Lite-HRNet backbone.

    `Lite-HRNet: A Lightweight High-Resolution Network
    <https://arxiv.org/abs/2104.06403>`_.

    Code adapted from 'https://github.com/HRNet/Lite-HRNet'.

    Args:
        extra (dict): detailed configuration for each stage of HRNet.
        in_channels (int): Number of input image channels. Default: 3.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.

    Example:
        >>> from mmpose.models import LiteHRNet
        >>> import torch
        >>> extra=dict(
        >>>    stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
        >>>    num_stages=3,
        >>>    stages_spec=dict(
        >>>        num_modules=(2, 4, 2),
        >>>        num_branches=(2, 3, 4),
        >>>        num_blocks=(2, 2, 2),
        >>>        module_type=('LITE', 'LITE', 'LITE'),
        >>>        with_fuse=(True, True, True),
        >>>        reduce_ratios=(8, 8, 8),
        >>>        num_channels=(
        >>>            (40, 80),
        >>>            (40, 80, 160),
        >>>            (40, 80, 160, 320),
        >>>        )),
        >>>    with_head=False)
        >>> self = LiteHRNet(extra, in_channels=1)
        >>> self.eval()
        >>> inputs = torch.rand(1, 1, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 40, 8, 8)
    """

    def __init__(self, extra, in_channels=3, conv_cfg=None, norm_cfg=dict(type='BN'), norm_eval=False, with_cp=False):
        super().__init__()
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.stem = Stem(in_channels, stem_channels=self.extra['stem']['stem_channels'], out_channels=self.extra['stem']['out_channels'], expand_ratio=self.extra['stem']['expand_ratio'], conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        self.num_stages = self.extra['num_stages']
        self.stages_spec = self.extra['stages_spec']
        num_channels_last = [self.stem.out_channels]
        for i in range(self.num_stages):
            num_channels = self.stages_spec['num_channels'][i]
            num_channels = [num_channels[i] for i in range(len(num_channels))]
            setattr(self, f'transition{i}', self._make_transition_layer(num_channels_last, num_channels))
            stage, num_channels_last = self._make_stage(self.stages_spec, i, num_channels, multiscale_output=True)
            setattr(self, f'stage{i}', stage)
        self.with_head = self.extra['with_head']
        if self.with_head:
            self.head_layer = IterativeHead(in_channels=num_channels_last, norm_cfg=self.norm_cfg)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(build_conv_layer(self.conv_cfg, num_channels_pre_layer[i], num_channels_pre_layer[i], kernel_size=3, stride=1, padding=1, groups=num_channels_pre_layer[i], bias=False), build_norm_layer(self.norm_cfg, num_channels_pre_layer[i])[1], build_conv_layer(self.conv_cfg, num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=1, stride=1, padding=0, bias=False), build_norm_layer(self.norm_cfg, num_channels_cur_layer[i])[1], nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False), build_norm_layer(self.norm_cfg, in_channels)[1], build_conv_layer(self.conv_cfg, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), build_norm_layer(self.norm_cfg, out_channels)[1], nn.ReLU()))
                transition_layers.append(nn.Sequential(*conv_downsamples))
        return nn.ModuleList(transition_layers)

    def _make_stage(self, stages_spec, stage_index, in_channels, multiscale_output=True):
        num_modules = stages_spec['num_modules'][stage_index]
        num_branches = stages_spec['num_branches'][stage_index]
        num_blocks = stages_spec['num_blocks'][stage_index]
        reduce_ratio = stages_spec['reduce_ratios'][stage_index]
        with_fuse = stages_spec['with_fuse'][stage_index]
        module_type = stages_spec['module_type'][stage_index]
        modules = []
        for i in range(num_modules):
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True
            modules.append(LiteHRModule(num_branches, num_blocks, in_channels, reduce_ratio, module_type, multiscale_output=reset_multiscale_output, with_fuse=with_fuse, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, with_cp=self.with_cp))
            in_channels = modules[-1].in_channels
        return nn.Sequential(*modules), in_channels

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.stem(x)
        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, f'transition{i}')
            for j in range(self.stages_spec['num_branches'][i]):
                if transition[j]:
                    if j >= len(y_list):
                        x_list.append(transition[j](y_list[-1]))
                    else:
                        x_list.append(transition[j](y_list[j]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, f'stage{i}')(x_list)
        x = y_list
        if self.with_head:
            x = self.head_layer(x)
        return [x[0]]

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


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
        assert mmcv.is_tuple_of(act_cfg, dict)
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


def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number down to the nearest value that can
    be divisible by the divisor.

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int, optional): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float, optional): The minimum ratio of the rounded channel
            number to the original channel number. Default: 0.9.
    Returns:
        int: The modified output channel number
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class MobileNetV2(BaseBackbone):
    """MobileNetV2 backbone.

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (None or Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    arch_settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]

    def __init__(self, widen_factor=1.0, out_indices=(7,), frozen_stages=-1, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU6'), norm_eval=False, with_cp=False):
        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        super().__init__()
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 8):
                raise ValueError(f'the item in out_indices must in range(0, 8). But received {index}')
        if frozen_stages not in range(-1, 8):
            raise ValueError(f'frozen_stages must be in range(-1, 8). But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.in_channels = make_divisible(32 * widen_factor, 8)
        self.conv1 = ConvModule(in_channels=3, out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.layers = []
        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, stride = layer_cfg
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(out_channels=out_channels, num_blocks=num_blocks, stride=stride, expand_ratio=expand_ratio)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)
        if widen_factor > 1.0:
            self.out_channel = int(1280 * widen_factor)
        else:
            self.out_channel = 1280
        layer = ConvModule(in_channels=self.in_channels, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.add_module('conv2', layer)
        self.layers.append('conv2')

    def make_layer(self, out_channels, num_blocks, stride, expand_ratio):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            layers.append(InvertedResidual(self.in_channels, out_channels, stride, expand_ratio=expand_ratio, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, with_cp=self.with_cp))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class MobileNetV3(BaseBackbone):
    """MobileNetV3 backbone.

    Args:
        arch (str): Architecture of mobilnetv3, from {small, big}.
            Default: small.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        out_indices (None or Sequence[int]): Output from which stages.
            Default: (-1, ), which means output tensors from final stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed.
            Default: False.
    """
    arch_settings = {'small': [[3, 16, 16, True, 'ReLU', 2], [3, 72, 24, False, 'ReLU', 2], [3, 88, 24, False, 'ReLU', 1], [5, 96, 40, True, 'HSwish', 2], [5, 240, 40, True, 'HSwish', 1], [5, 240, 40, True, 'HSwish', 1], [5, 120, 48, True, 'HSwish', 1], [5, 144, 48, True, 'HSwish', 1], [5, 288, 96, True, 'HSwish', 2], [5, 576, 96, True, 'HSwish', 1], [5, 576, 96, True, 'HSwish', 1]], 'big': [[3, 16, 16, False, 'ReLU', 1], [3, 64, 24, False, 'ReLU', 2], [3, 72, 24, False, 'ReLU', 1], [5, 72, 40, True, 'ReLU', 2], [5, 120, 40, True, 'ReLU', 1], [5, 120, 40, True, 'ReLU', 1], [3, 240, 80, False, 'HSwish', 2], [3, 200, 80, False, 'HSwish', 1], [3, 184, 80, False, 'HSwish', 1], [3, 184, 80, False, 'HSwish', 1], [3, 480, 112, True, 'HSwish', 1], [3, 672, 112, True, 'HSwish', 1], [5, 672, 160, True, 'HSwish', 1], [5, 672, 160, True, 'HSwish', 2], [5, 960, 160, True, 'HSwish', 1]]}

    def __init__(self, arch='small', conv_cfg=None, norm_cfg=dict(type='BN'), out_indices=(-1,), frozen_stages=-1, norm_eval=False, with_cp=False):
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        assert arch in self.arch_settings
        for index in out_indices:
            if index not in range(-len(self.arch_settings[arch]), len(self.arch_settings[arch])):
                raise ValueError(f'the item in out_indices must in range(0, {len(self.arch_settings[arch])}). But received {index}')
        if frozen_stages not in range(-1, len(self.arch_settings[arch])):
            raise ValueError(f'frozen_stages must be in range(-1, {len(self.arch_settings[arch])}). But received {frozen_stages}')
        self.arch = arch
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.in_channels = 16
        self.conv1 = ConvModule(in_channels=3, out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=dict(type='HSwish'))
        self.layers = self._make_layer()
        self.feat_dim = self.arch_settings[arch][-1][2]

    def _make_layer(self):
        layers = []
        layer_setting = self.arch_settings[self.arch]
        for i, params in enumerate(layer_setting):
            kernel_size, mid_channels, out_channels, with_se, act, stride = params
            if with_se:
                se_cfg = dict(channels=mid_channels, ratio=4, act_cfg=(dict(type='ReLU'), dict(type='HSigmoid')))
            else:
                se_cfg = None
            layer = InvertedResidual(in_channels=self.in_channels, out_channels=out_channels, mid_channels=mid_channels, kernel_size=kernel_size, stride=stride, se_cfg=se_cfg, with_expand_conv=True, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=dict(type=act), with_cp=self.with_cp)
            self.in_channels = out_channels
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, layer)
            layers.append(layer_name)
        return layers

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
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices or i - len(self.layers) in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class DownsampleModule(nn.Module):
    """Downsample module for MSPN.

    Args:
        block (nn.Module): Downsample block.
        num_blocks (list): Number of blocks in each downsample unit.
        num_units (int): Numbers of downsample units. Default: 4
        has_skip (bool): Have skip connections from prior upsample
            module or not. Default:False
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        in_channels (int): Number of channels of the input feature to
            downsample module. Default: 64
    """

    def __init__(self, block, num_blocks, num_units=4, has_skip=False, norm_cfg=dict(type='BN'), in_channels=64):
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.has_skip = has_skip
        self.in_channels = in_channels
        assert len(num_blocks) == num_units
        self.num_blocks = num_blocks
        self.num_units = num_units
        self.norm_cfg = norm_cfg
        self.layer1 = self._make_layer(block, in_channels, num_blocks[0])
        for i in range(1, num_units):
            module_name = f'layer{i + 1}'
            self.add_module(module_name, self._make_layer(block, in_channels * pow(2, i), num_blocks[i], stride=2))

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = ConvModule(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, padding=0, norm_cfg=self.norm_cfg, act_cfg=None, inplace=True)
        units = list()
        units.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample, norm_cfg=self.norm_cfg))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            units.append(block(self.in_channels, out_channels))
        return nn.Sequential(*units)

    def forward(self, x, skip1, skip2):
        out = list()
        for i in range(self.num_units):
            module_name = f'layer{i + 1}'
            module_i = getattr(self, module_name)
            x = module_i(x)
            if self.has_skip:
                x = x + skip1[i] + skip2[i]
            out.append(x)
        out.reverse()
        return tuple(out)


class UpsampleUnit(nn.Module):
    """Upsample unit for upsample module.

    Args:
        ind (int): Indicates whether to interpolate (>0) and whether to
           generate feature map for the next hourglass-like module.
        num_units (int): Number of units that form a upsample module. Along
            with ind and gen_cross_conv, nm_units is used to decide whether
            to generate feature map for the next hourglass-like module.
        in_channels (int): Channel number of the skip-in feature maps from
            the corresponding downsample unit.
        unit_channels (int): Channel number in this unit. Default:256.
        gen_skip: (bool): Whether or not to generate skips for the posterior
            downsample module. Default:False
        gen_cross_conv (bool): Whether to generate feature map for the next
            hourglass-like module. Default:False
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        out_channels (int): Number of channels of feature output by upsample
            module. Must equal to in_channels of downsample module. Default:64
    """

    def __init__(self, ind, num_units, in_channels, unit_channels=256, gen_skip=False, gen_cross_conv=False, norm_cfg=dict(type='BN'), out_channels=64):
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.num_units = num_units
        self.norm_cfg = norm_cfg
        self.in_skip = ConvModule(in_channels, unit_channels, kernel_size=1, stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=None, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.ind = ind
        if self.ind > 0:
            self.up_conv = ConvModule(unit_channels, unit_channels, kernel_size=1, stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=None, inplace=True)
        self.gen_skip = gen_skip
        if self.gen_skip:
            self.out_skip1 = ConvModule(in_channels, in_channels, kernel_size=1, stride=1, padding=0, norm_cfg=self.norm_cfg, inplace=True)
            self.out_skip2 = ConvModule(unit_channels, in_channels, kernel_size=1, stride=1, padding=0, norm_cfg=self.norm_cfg, inplace=True)
        self.gen_cross_conv = gen_cross_conv
        if self.ind == num_units - 1 and self.gen_cross_conv:
            self.cross_conv = ConvModule(unit_channels, out_channels, kernel_size=1, stride=1, padding=0, norm_cfg=self.norm_cfg, inplace=True)

    def forward(self, x, up_x):
        out = self.in_skip(x)
        if self.ind > 0:
            up_x = F.interpolate(up_x, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            up_x = self.up_conv(up_x)
            out = out + up_x
        out = self.relu(out)
        skip1 = None
        skip2 = None
        if self.gen_skip:
            skip1 = self.out_skip1(x)
            skip2 = self.out_skip2(out)
        cross_conv = None
        if self.ind == self.num_units - 1 and self.gen_cross_conv:
            cross_conv = self.cross_conv(out)
        return out, skip1, skip2, cross_conv


class UpsampleModule(nn.Module):
    """Upsample module for MSPN.

    Args:
        unit_channels (int): Channel number in the upsample units.
            Default:256.
        num_units (int): Numbers of upsample units. Default: 4
        gen_skip (bool): Whether to generate skip for posterior downsample
            module or not. Default:False
        gen_cross_conv (bool): Whether to generate feature map for the next
            hourglass-like module. Default:False
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        out_channels (int): Number of channels of feature output by upsample
            module. Must equal to in_channels of downsample module. Default:64
    """

    def __init__(self, unit_channels=256, num_units=4, gen_skip=False, gen_cross_conv=False, norm_cfg=dict(type='BN'), out_channels=64):
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = list()
        for i in range(num_units):
            self.in_channels.append(Bottleneck.expansion * out_channels * pow(2, i))
        self.in_channels.reverse()
        self.num_units = num_units
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.norm_cfg = norm_cfg
        for i in range(num_units):
            module_name = f'up{i + 1}'
            self.add_module(module_name, UpsampleUnit(i, self.num_units, self.in_channels[i], unit_channels, self.gen_skip, self.gen_cross_conv, norm_cfg=self.norm_cfg, out_channels=64))

    def forward(self, x):
        out = list()
        skip1 = list()
        skip2 = list()
        cross_conv = None
        for i in range(self.num_units):
            module_i = getattr(self, f'up{i + 1}')
            if i == 0:
                outi, skip1_i, skip2_i, _ = module_i(x[i], None)
            elif i == self.num_units - 1:
                outi, skip1_i, skip2_i, cross_conv = module_i(x[i], out[i - 1])
            else:
                outi, skip1_i, skip2_i, _ = module_i(x[i], out[i - 1])
            out.append(outi)
            skip1.append(skip1_i)
            skip2.append(skip2_i)
        skip1.reverse()
        skip2.reverse()
        return out, skip1, skip2, cross_conv


class SingleStageNetwork(nn.Module):
    """Single_stage Network.

    Args:
        unit_channels (int): Channel number in the upsample units. Default:256.
        num_units (int): Numbers of downsample/upsample units. Default: 4
        gen_skip (bool): Whether to generate skip for posterior downsample
            module or not. Default:False
        gen_cross_conv (bool): Whether to generate feature map for the next
            hourglass-like module. Default:False
        has_skip (bool): Have skip connections from prior upsample
            module or not. Default:False
        num_blocks (list): Number of blocks in each downsample unit.
            Default: [2, 2, 2, 2] Note: Make sure num_units==len(num_blocks)
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        in_channels (int): Number of channels of the feature from ResNetTop.
            Default: 64.
    """

    def __init__(self, has_skip=False, gen_skip=False, gen_cross_conv=False, unit_channels=256, num_units=4, num_blocks=[2, 2, 2, 2], norm_cfg=dict(type='BN'), in_channels=64):
        norm_cfg = cp.deepcopy(norm_cfg)
        num_blocks = cp.deepcopy(num_blocks)
        super().__init__()
        assert len(num_blocks) == num_units
        self.has_skip = has_skip
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.num_units = num_units
        self.unit_channels = unit_channels
        self.num_blocks = num_blocks
        self.norm_cfg = norm_cfg
        self.downsample = DownsampleModule(Bottleneck, num_blocks, num_units, has_skip, norm_cfg, in_channels)
        self.upsample = UpsampleModule(unit_channels, num_units, gen_skip, gen_cross_conv, norm_cfg, in_channels)

    def forward(self, x, skip1, skip2):
        mid = self.downsample(x, skip1, skip2)
        out, skip1, skip2, cross_conv = self.upsample(mid)
        return out, skip1, skip2, cross_conv


class ResNetTop(nn.Module):
    """ResNet top for MSPN.

    Args:
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        channels (int): Number of channels of the feature output by ResNetTop.
    """

    def __init__(self, norm_cfg=dict(type='BN'), channels=64):
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.top = nn.Sequential(ConvModule(3, channels, kernel_size=7, stride=2, padding=3, norm_cfg=norm_cfg, inplace=True), MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, img):
        return self.top(img)


def get_state_dict(filename, map_location='cpu'):
    """Get state_dict from a file or URI.

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        map_location (str): Same as :func:`torch.load`.

    Returns:
        OrderedDict: The state_dict.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict_tmp = checkpoint['state_dict']
    else:
        state_dict_tmp = checkpoint
    state_dict = OrderedDict()
    for k, v in state_dict_tmp.items():
        if k.startswith('module.backbone.'):
            state_dict[k[16:]] = v
        elif k.startswith('module.'):
            state_dict[k[7:]] = v
        elif k.startswith('backbone.'):
            state_dict[k[9:]] = v
        else:
            state_dict[k] = v
    return state_dict


class MSPN(BaseBackbone):
    """MSPN backbone. Paper ref: Li et al. "Rethinking on Multi-Stage Networks
    for Human Pose Estimation" (CVPR 2020).

    Args:
        unit_channels (int): Number of Channels in an upsample unit.
            Default: 256
        num_stages (int): Number of stages in a multi-stage MSPN. Default: 4
        num_units (int): Number of downsample/upsample units in a single-stage
            network. Default: 4
            Note: Make sure num_units == len(self.num_blocks)
        num_blocks (list): Number of bottlenecks in each
            downsample unit. Default: [2, 2, 2, 2]
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        res_top_channels (int): Number of channels of feature from ResNetTop.
            Default: 64.

    Example:
        >>> from mmpose.models import MSPN
        >>> import torch
        >>> self = MSPN(num_stages=2,num_units=2,num_blocks=[2,2])
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     for feature in level_output:
        ...         print(tuple(feature.shape))
        ...
        (1, 256, 64, 64)
        (1, 256, 128, 128)
        (1, 256, 64, 64)
        (1, 256, 128, 128)
    """

    def __init__(self, unit_channels=256, num_stages=4, num_units=4, num_blocks=[2, 2, 2, 2], norm_cfg=dict(type='BN'), res_top_channels=64):
        norm_cfg = cp.deepcopy(norm_cfg)
        num_blocks = cp.deepcopy(num_blocks)
        super().__init__()
        self.unit_channels = unit_channels
        self.num_stages = num_stages
        self.num_units = num_units
        self.num_blocks = num_blocks
        self.norm_cfg = norm_cfg
        assert self.num_stages > 0
        assert self.num_units > 1
        assert self.num_units == len(self.num_blocks)
        self.top = ResNetTop(norm_cfg=norm_cfg)
        self.multi_stage_mspn = nn.ModuleList([])
        for i in range(self.num_stages):
            if i == 0:
                has_skip = False
            else:
                has_skip = True
            if i != self.num_stages - 1:
                gen_skip = True
                gen_cross_conv = True
            else:
                gen_skip = False
                gen_cross_conv = False
            self.multi_stage_mspn.append(SingleStageNetwork(has_skip, gen_skip, gen_cross_conv, unit_channels, num_units, num_blocks, norm_cfg, res_top_channels))

    def forward(self, x):
        """Model forward function."""
        out_feats = []
        skip1 = None
        skip2 = None
        x = self.top(x)
        for i in range(self.num_stages):
            out, skip1, skip2, x = self.multi_stage_mspn[i](x, skip1, skip2)
            out_feats.append(out)
        return out_feats

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        if isinstance(pretrained, str):
            logger = get_root_logger()
            state_dict_tmp = get_state_dict(pretrained)
            state_dict = OrderedDict()
            state_dict['top'] = OrderedDict()
            state_dict['bottlenecks'] = OrderedDict()
            for k, v in state_dict_tmp.items():
                if k.startswith('layer'):
                    if 'downsample.0' in k:
                        state_dict['bottlenecks'][k.replace('downsample.0', 'downsample.conv')] = v
                    elif 'downsample.1' in k:
                        state_dict['bottlenecks'][k.replace('downsample.1', 'downsample.bn')] = v
                    else:
                        state_dict['bottlenecks'][k] = v
                elif k.startswith('conv1'):
                    state_dict['top'][k.replace('conv1', 'top.0.conv')] = v
                elif k.startswith('bn1'):
                    state_dict['top'][k.replace('bn1', 'top.0.bn')] = v
            load_state_dict(self.top, state_dict['top'], strict=False, logger=logger)
            for i in range(self.num_stages):
                load_state_dict(self.multi_stage_mspn[i].downsample, state_dict['bottlenecks'], strict=False, logger=logger)
        else:
            for m in self.multi_stage_mspn.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
            for m in self.top.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)


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


class SplitAttentionConv2d(nn.Module):
    """Split-Attention Conv2d.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        radix (int): Radix of SpltAtConv2d. Default: 2
        reduction_factor (int): Reduction factor of SplitAttentionConv2d.
            Default: 4.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
    """

    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, radix=2, reduction_factor=4, conv_cfg=None, norm_cfg=dict(type='BN')):
        super().__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.groups = groups
        self.channels = channels
        self.conv = build_conv_layer(conv_cfg, in_channels, channels * radix, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups * radix, bias=False)
        self.norm0_name, norm0 = build_norm_layer(norm_cfg, channels * radix, postfix=0)
        self.add_module(self.norm0_name, norm0)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = build_conv_layer(None, channels, inter_channels, 1, groups=self.groups)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, inter_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.fc2 = build_conv_layer(None, inter_channels, channels * radix, 1, groups=self.groups)
        self.rsoftmax = RSoftmax(radix, groups)

    @property
    def norm0(self):
        return getattr(self, self.norm0_name)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm0(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splits = x.view(batch, self.radix, -1, *x.shape[2:])
            gap = splits.sum(dim=1)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.norm1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            attens = atten.view(batch, self.radix, -1, *atten.shape[2:])
            out = torch.sum(attens * splits, dim=1)
        else:
            out = atten * x
        return out.contiguous()


class RSB(nn.Module):
    """Residual Steps block for RSN. Paper ref: Cai et al. "Learning Delicate
    Local Representations for Multi-Person Pose Estimation" (ECCV 2020).

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        num_steps (int): Numbers of steps in RSB
        stride (int): stride of the block. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        expand_times (int): Times by which the in_channels are expanded.
            Default:26.
        res_top_channels (int): Number of channels of feature output by
            ResNet_top. Default:64.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, num_steps=4, stride=1, downsample=None, with_cp=False, norm_cfg=dict(type='BN'), expand_times=26, res_top_channels=64):
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        assert num_steps > 1
        self.in_channels = in_channels
        self.branch_channels = self.in_channels * expand_times
        self.branch_channels //= res_top_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample
        self.with_cp = with_cp
        self.norm_cfg = norm_cfg
        self.num_steps = num_steps
        self.conv_bn_relu1 = ConvModule(self.in_channels, self.num_steps * self.branch_channels, kernel_size=1, stride=self.stride, padding=0, norm_cfg=self.norm_cfg, inplace=False)
        for i in range(self.num_steps):
            for j in range(i + 1):
                module_name = f'conv_bn_relu2_{i + 1}_{j + 1}'
                self.add_module(module_name, ConvModule(self.branch_channels, self.branch_channels, kernel_size=3, stride=1, padding=1, norm_cfg=self.norm_cfg, inplace=False))
        self.conv_bn3 = ConvModule(self.num_steps * self.branch_channels, self.out_channels * self.expansion, kernel_size=1, stride=1, padding=0, act_cfg=None, norm_cfg=self.norm_cfg, inplace=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        """Forward function."""
        identity = x
        x = self.conv_bn_relu1(x)
        spx = torch.split(x, self.branch_channels, 1)
        outputs = list()
        outs = list()
        for i in range(self.num_steps):
            outputs_i = list()
            outputs.append(outputs_i)
            for j in range(i + 1):
                if j == 0:
                    inputs = spx[i]
                else:
                    inputs = outputs[i][j - 1]
                if i > j:
                    inputs = inputs + outputs[i - 1][j]
                module_name = f'conv_bn_relu2_{i + 1}_{j + 1}'
                module_i_j = getattr(self, module_name)
                outputs[i].append(module_i_j(inputs))
            outs.append(outputs[i][i])
        out = torch.cat(tuple(outs), 1)
        out = self.conv_bn3(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        out = self.relu(out)
        return out


class Downsample_module(nn.Module):
    """Downsample module for RSN.

    Args:
        block (nn.Module): Downsample block.
        num_blocks (list): Number of blocks in each downsample unit.
        num_units (int): Numbers of downsample units. Default: 4
        has_skip (bool): Have skip connections from prior upsample
            module or not. Default:False
        num_steps (int): Number of steps in a block. Default:4
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        in_channels (int): Number of channels of the input feature to
            downsample module. Default: 64
        expand_times (int): Times by which the in_channels are expanded.
            Default:26.
    """

    def __init__(self, block, num_blocks, num_steps=4, num_units=4, has_skip=False, norm_cfg=dict(type='BN'), in_channels=64, expand_times=26):
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.has_skip = has_skip
        self.in_channels = in_channels
        assert len(num_blocks) == num_units
        self.num_blocks = num_blocks
        self.num_units = num_units
        self.num_steps = num_steps
        self.norm_cfg = norm_cfg
        self.layer1 = self._make_layer(block, in_channels, num_blocks[0], expand_times=expand_times, res_top_channels=in_channels)
        for i in range(1, num_units):
            module_name = f'layer{i + 1}'
            self.add_module(module_name, self._make_layer(block, in_channels * pow(2, i), num_blocks[i], stride=2, expand_times=expand_times, res_top_channels=in_channels))

    def _make_layer(self, block, out_channels, blocks, stride=1, expand_times=26, res_top_channels=64):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = ConvModule(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, padding=0, norm_cfg=self.norm_cfg, act_cfg=None, inplace=True)
        units = list()
        units.append(block(self.in_channels, out_channels, num_steps=self.num_steps, stride=stride, downsample=downsample, norm_cfg=self.norm_cfg, expand_times=expand_times, res_top_channels=res_top_channels))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            units.append(block(self.in_channels, out_channels, num_steps=self.num_steps, expand_times=expand_times, res_top_channels=res_top_channels))
        return nn.Sequential(*units)

    def forward(self, x, skip1, skip2):
        out = list()
        for i in range(self.num_units):
            module_name = f'layer{i + 1}'
            module_i = getattr(self, module_name)
            x = module_i(x)
            if self.has_skip:
                x = x + skip1[i] + skip2[i]
            out.append(x)
        out.reverse()
        return tuple(out)


class Upsample_unit(nn.Module):
    """Upsample unit for upsample module.

    Args:
        ind (int): Indicates whether to interpolate (>0) and whether to
           generate feature map for the next hourglass-like module.
        num_units (int): Number of units that form a upsample module. Along
            with ind and gen_cross_conv, nm_units is used to decide whether
            to generate feature map for the next hourglass-like module.
        in_channels (int): Channel number of the skip-in feature maps from
            the corresponding downsample unit.
        unit_channels (int): Channel number in this unit. Default:256.
        gen_skip: (bool): Whether or not to generate skips for the posterior
            downsample module. Default:False
        gen_cross_conv (bool): Whether to generate feature map for the next
            hourglass-like module. Default:False
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        out_channels (in): Number of channels of feature output by upsample
            module. Must equal to in_channels of downsample module. Default:64
    """

    def __init__(self, ind, num_units, in_channels, unit_channels=256, gen_skip=False, gen_cross_conv=False, norm_cfg=dict(type='BN'), out_channels=64):
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.num_units = num_units
        self.norm_cfg = norm_cfg
        self.in_skip = ConvModule(in_channels, unit_channels, kernel_size=1, stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=None, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.ind = ind
        if self.ind > 0:
            self.up_conv = ConvModule(unit_channels, unit_channels, kernel_size=1, stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=None, inplace=True)
        self.gen_skip = gen_skip
        if self.gen_skip:
            self.out_skip1 = ConvModule(in_channels, in_channels, kernel_size=1, stride=1, padding=0, norm_cfg=self.norm_cfg, inplace=True)
            self.out_skip2 = ConvModule(unit_channels, in_channels, kernel_size=1, stride=1, padding=0, norm_cfg=self.norm_cfg, inplace=True)
        self.gen_cross_conv = gen_cross_conv
        if self.ind == num_units - 1 and self.gen_cross_conv:
            self.cross_conv = ConvModule(unit_channels, out_channels, kernel_size=1, stride=1, padding=0, norm_cfg=self.norm_cfg, inplace=True)

    def forward(self, x, up_x):
        out = self.in_skip(x)
        if self.ind > 0:
            up_x = F.interpolate(up_x, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            up_x = self.up_conv(up_x)
            out = out + up_x
        out = self.relu(out)
        skip1 = None
        skip2 = None
        if self.gen_skip:
            skip1 = self.out_skip1(x)
            skip2 = self.out_skip2(out)
        cross_conv = None
        if self.ind == self.num_units - 1 and self.gen_cross_conv:
            cross_conv = self.cross_conv(out)
        return out, skip1, skip2, cross_conv


class Upsample_module(nn.Module):
    """Upsample module for RSN.

    Args:
        unit_channels (int): Channel number in the upsample units.
            Default:256.
        num_units (int): Numbers of upsample units. Default: 4
        gen_skip (bool): Whether to generate skip for posterior downsample
            module or not. Default:False
        gen_cross_conv (bool): Whether to generate feature map for the next
            hourglass-like module. Default:False
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        out_channels (int): Number of channels of feature output by upsample
            module. Must equal to in_channels of downsample module. Default:64
    """

    def __init__(self, unit_channels=256, num_units=4, gen_skip=False, gen_cross_conv=False, norm_cfg=dict(type='BN'), out_channels=64):
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = list()
        for i in range(num_units):
            self.in_channels.append(RSB.expansion * out_channels * pow(2, i))
        self.in_channels.reverse()
        self.num_units = num_units
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.norm_cfg = norm_cfg
        for i in range(num_units):
            module_name = f'up{i + 1}'
            self.add_module(module_name, Upsample_unit(i, self.num_units, self.in_channels[i], unit_channels, self.gen_skip, self.gen_cross_conv, norm_cfg=self.norm_cfg, out_channels=64))

    def forward(self, x):
        out = list()
        skip1 = list()
        skip2 = list()
        cross_conv = None
        for i in range(self.num_units):
            module_i = getattr(self, f'up{i + 1}')
            if i == 0:
                outi, skip1_i, skip2_i, _ = module_i(x[i], None)
            elif i == self.num_units - 1:
                outi, skip1_i, skip2_i, cross_conv = module_i(x[i], out[i - 1])
            else:
                outi, skip1_i, skip2_i, _ = module_i(x[i], out[i - 1])
            out.append(outi)
            skip1.append(skip1_i)
            skip2.append(skip2_i)
        skip1.reverse()
        skip2.reverse()
        return out, skip1, skip2, cross_conv


class Single_stage_RSN(nn.Module):
    """Single_stage Residual Steps Network.

    Args:
        unit_channels (int): Channel number in the upsample units. Default:256.
        num_units (int): Numbers of downsample/upsample units. Default: 4
        gen_skip (bool): Whether to generate skip for posterior downsample
            module or not. Default:False
        gen_cross_conv (bool): Whether to generate feature map for the next
            hourglass-like module. Default:False
        has_skip (bool): Have skip connections from prior upsample
            module or not. Default:False
        num_steps (int): Number of steps in RSB. Default: 4
        num_blocks (list): Number of blocks in each downsample unit.
            Default: [2, 2, 2, 2] Note: Make sure num_units==len(num_blocks)
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        in_channels (int): Number of channels of the feature from ResNet_Top.
            Default: 64.
        expand_times (int): Times by which the in_channels are expanded in RSB.
            Default:26.
    """

    def __init__(self, has_skip=False, gen_skip=False, gen_cross_conv=False, unit_channels=256, num_units=4, num_steps=4, num_blocks=[2, 2, 2, 2], norm_cfg=dict(type='BN'), in_channels=64, expand_times=26):
        norm_cfg = cp.deepcopy(norm_cfg)
        num_blocks = cp.deepcopy(num_blocks)
        super().__init__()
        assert len(num_blocks) == num_units
        self.has_skip = has_skip
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.num_units = num_units
        self.num_steps = num_steps
        self.unit_channels = unit_channels
        self.num_blocks = num_blocks
        self.norm_cfg = norm_cfg
        self.downsample = Downsample_module(RSB, num_blocks, num_steps, num_units, has_skip, norm_cfg, in_channels, expand_times)
        self.upsample = Upsample_module(unit_channels, num_units, gen_skip, gen_cross_conv, norm_cfg, in_channels)

    def forward(self, x, skip1, skip2):
        mid = self.downsample(x, skip1, skip2)
        out, skip1, skip2, cross_conv = self.upsample(mid)
        return out, skip1, skip2, cross_conv


class ResNet_top(nn.Module):
    """ResNet top for RSN.

    Args:
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        channels (int): Number of channels of the feature output by ResNet_top.
    """

    def __init__(self, norm_cfg=dict(type='BN'), channels=64):
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.top = nn.Sequential(ConvModule(3, channels, kernel_size=7, stride=2, padding=3, norm_cfg=norm_cfg, inplace=True), MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, img):
        return self.top(img)


class RSN(BaseBackbone):
    """Residual Steps Network backbone. Paper ref: Cai et al. "Learning
    Delicate Local Representations for Multi-Person Pose Estimation" (ECCV
    2020).

    Args:
        unit_channels (int): Number of Channels in an upsample unit.
            Default: 256
        num_stages (int): Number of stages in a multi-stage RSN. Default: 4
        num_units (int): NUmber of downsample/upsample units in a single-stage
            RSN. Default: 4 Note: Make sure num_units == len(self.num_blocks)
        num_blocks (list): Number of RSBs (Residual Steps Block) in each
            downsample unit. Default: [2, 2, 2, 2]
        num_steps (int): Number of steps in a RSB. Default:4
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        res_top_channels (int): Number of channels of feature from ResNet_top.
            Default: 64.
        expand_times (int): Times by which the in_channels are expanded in RSB.
            Default:26.
    Example:
        >>> from mmpose.models import RSN
        >>> import torch
        >>> self = RSN(num_stages=2,num_units=2,num_blocks=[2,2])
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     for feature in level_output:
        ...         print(tuple(feature.shape))
        ...
        (1, 256, 64, 64)
        (1, 256, 128, 128)
        (1, 256, 64, 64)
        (1, 256, 128, 128)
    """

    def __init__(self, unit_channels=256, num_stages=4, num_units=4, num_blocks=[2, 2, 2, 2], num_steps=4, norm_cfg=dict(type='BN'), res_top_channels=64, expand_times=26):
        norm_cfg = cp.deepcopy(norm_cfg)
        num_blocks = cp.deepcopy(num_blocks)
        super().__init__()
        self.unit_channels = unit_channels
        self.num_stages = num_stages
        self.num_units = num_units
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.norm_cfg = norm_cfg
        assert self.num_stages > 0
        assert self.num_steps > 1
        assert self.num_units > 1
        assert self.num_units == len(self.num_blocks)
        self.top = ResNet_top(norm_cfg=norm_cfg)
        self.multi_stage_rsn = nn.ModuleList([])
        for i in range(self.num_stages):
            if i == 0:
                has_skip = False
            else:
                has_skip = True
            if i != self.num_stages - 1:
                gen_skip = True
                gen_cross_conv = True
            else:
                gen_skip = False
                gen_cross_conv = False
            self.multi_stage_rsn.append(Single_stage_RSN(has_skip, gen_skip, gen_cross_conv, unit_channels, num_units, num_steps, num_blocks, norm_cfg, res_top_channels, expand_times))

    def forward(self, x):
        """Model forward function."""
        out_feats = []
        skip1 = None
        skip2 = None
        x = self.top(x)
        for i in range(self.num_stages):
            out, skip1, skip2, x = self.multi_stage_rsn[i](x, skip1, skip2)
            out_feats.append(out)
        return out_feats

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        for m in self.multi_stage_rsn.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
        for m in self.top.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)


class SCConv(nn.Module):
    """SCConv (Self-calibrated Convolution)

    Args:
        in_channels (int): The input channels of the SCConv.
        out_channels (int): The output channel of the SCConv.
        stride (int): stride of SCConv.
        pooling_r (int): size of pooling for scconv.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self, in_channels, out_channels, stride, pooling_r, conv_cfg=None, norm_cfg=dict(type='BN', momentum=0.1)):
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        assert in_channels == out_channels
        self.k2 = nn.Sequential(nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), build_conv_layer(conv_cfg, in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False), build_norm_layer(norm_cfg, in_channels)[1])
        self.k3 = nn.Sequential(build_conv_layer(conv_cfg, in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False), build_norm_layer(norm_cfg, in_channels)[1])
        self.k4 = nn.Sequential(build_conv_layer(conv_cfg, in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False), build_norm_layer(norm_cfg, out_channels)[1], nn.ReLU(inplace=True))

    def forward(self, x):
        """Forward function."""
        identity = x
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))
        out = torch.mul(self.k3(x), out)
        out = self.k4(out)
        return out


class ShuffleNetV1(BaseBackbone):
    """ShuffleNetV1 backbone.

    Args:
        groups (int, optional): The number of groups to be used in grouped 1x1
            convolutions in each ShuffleUnit. Default: 3.
        widen_factor (float, optional): Width multiplier - adjusts the number
            of channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (2, )
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self, groups=3, widen_factor=1.0, out_indices=(2,), frozen_stages=-1, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), norm_eval=False, with_cp=False):
        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        super().__init__()
        self.stage_blocks = [4, 8, 4]
        self.groups = groups
        for index in out_indices:
            if index not in range(0, 3):
                raise ValueError(f'the item in out_indices must in range(0, 3). But received {index}')
        if frozen_stages not in range(-1, 3):
            raise ValueError(f'frozen_stages must be in range(-1, 3). But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        if groups == 1:
            channels = 144, 288, 576
        elif groups == 2:
            channels = 200, 400, 800
        elif groups == 3:
            channels = 240, 480, 960
        elif groups == 4:
            channels = 272, 544, 1088
        elif groups == 8:
            channels = 384, 768, 1536
        else:
            raise ValueError(f'{groups} groups is not supported for 1x1 Grouped Convolutions')
        channels = [make_divisible(ch * widen_factor, 8) for ch in channels]
        self.in_channels = int(24 * widen_factor)
        self.conv1 = ConvModule(in_channels=3, out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            first_block = i == 0
            layer = self.make_layer(channels[i], num_blocks, first_block)
            self.layers.append(layer)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(self.frozen_stages):
            layer = self.layers[i]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    if 'conv1' in name:
                        normal_init(m, mean=0, std=0.01)
                    else:
                        normal_init(m, mean=0, std=1.0 / m.weight.shape[1])
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, val=1, bias=0.0001)
                    if isinstance(m, _BatchNorm):
                        if m.running_mean is not None:
                            nn.init.constant_(m.running_mean, 0)
        else:
            raise TypeError(f'pretrained must be a str or None. But received {type(pretrained)}')

    def make_layer(self, out_channels, num_blocks, first_block=False):
        """Stack ShuffleUnit blocks to make a layer.

        Args:
            out_channels (int): out_channels of the block.
            num_blocks (int): Number of blocks.
            first_block (bool, optional): Whether is the first ShuffleUnit of a
                sequential ShuffleUnits. Default: False, which means using
                the grouped 1x1 convolution.
        """
        layers = []
        for i in range(num_blocks):
            first_block = first_block if i == 0 else False
            combine_mode = 'concat' if i == 0 else 'add'
            layers.append(ShuffleUnit(self.in_channels, out_channels, groups=self.groups, first_block=first_block, combine=combine_mode, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, with_cp=self.with_cp))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class ShuffleNetV2(BaseBackbone):
    """ShuffleNetV2 backbone.

    Args:
        widen_factor (float): Width multiplier - adjusts the number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self, widen_factor=1.0, out_indices=(3,), frozen_stages=-1, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), norm_eval=False, with_cp=False):
        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        super().__init__()
        self.stage_blocks = [4, 8, 4]
        for index in out_indices:
            if index not in range(0, 4):
                raise ValueError(f'the item in out_indices must in range(0, 4). But received {index}')
        if frozen_stages not in range(-1, 4):
            raise ValueError(f'frozen_stages must be in range(-1, 4). But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        if widen_factor == 0.5:
            channels = [48, 96, 192, 1024]
        elif widen_factor == 1.0:
            channels = [116, 232, 464, 1024]
        elif widen_factor == 1.5:
            channels = [176, 352, 704, 1024]
        elif widen_factor == 2.0:
            channels = [244, 488, 976, 2048]
        else:
            raise ValueError(f'widen_factor must be in [0.5, 1.0, 1.5, 2.0]. But received {widen_factor}')
        self.in_channels = 24
        self.conv1 = ConvModule(in_channels=3, out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            layer = self._make_layer(channels[i], num_blocks)
            self.layers.append(layer)
        output_channels = channels[-1]
        self.layers.append(ConvModule(in_channels=self.in_channels, out_channels=output_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))

    def _make_layer(self, out_channels, num_blocks):
        """Stack blocks to make a layer.

        Args:
            out_channels (int): out_channels of the block.
            num_blocks (int): number of blocks.
        """
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(InvertedResidual(in_channels=self.in_channels, out_channels=out_channels, stride=stride, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, with_cp=self.with_cp))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    if 'conv1' in name:
                        normal_init(m, mean=0, std=0.01)
                    else:
                        normal_init(m, mean=0, std=1.0 / m.weight.shape[1])
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m.weight, val=1, bias=0.0001)
                    if isinstance(m, _BatchNorm):
                        if m.running_mean is not None:
                            nn.init.constant_(m.running_mean, 0)
        else:
            raise TypeError(f'pretrained must be a str or None. But received {type(pretrained)}')

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


class BasicTemporalBlock(nn.Module):
    """Basic block for VideoPose3D.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        mid_channels (int): The output channels of conv1. Default: 1024.
        kernel_size (int): Size of the convolving kernel. Default: 3.
        dilation (int): Spacing between kernel elements. Default: 3.
        dropout (float): Dropout rate. Default: 0.25.
        causal (bool): Use causal convolutions instead of symmetric
            convolutions (for real-time applications). Default: False.
        residual (bool): Use residual connection. Default: True.
        use_stride_conv (bool): Use optimized TCN that designed
            specifically for single-frame batching, i.e. where batches have
            input length = receptive field, and output length = 1. This
            implementation replaces dilated convolutions with strided
            convolutions to avoid generating unused intermediate results.
            Default: False.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: dict(type='Conv1d').
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN1d').
    """

    def __init__(self, in_channels, out_channels, mid_channels=1024, kernel_size=3, dilation=3, dropout=0.25, causal=False, residual=True, use_stride_conv=False, conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d')):
        conv_cfg = copy.deepcopy(conv_cfg)
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        self.causal = causal
        self.residual = residual
        self.use_stride_conv = use_stride_conv
        self.pad = (kernel_size - 1) * dilation // 2
        if use_stride_conv:
            self.stride = kernel_size
            self.causal_shift = kernel_size // 2 if causal else 0
            self.dilation = 1
        else:
            self.stride = 1
            self.causal_shift = kernel_size // 2 * dilation if causal else 0
        self.conv1 = nn.Sequential(ConvModule(in_channels, mid_channels, kernel_size=kernel_size, stride=self.stride, dilation=self.dilation, bias='auto', conv_cfg=conv_cfg, norm_cfg=norm_cfg))
        self.conv2 = nn.Sequential(ConvModule(mid_channels, out_channels, kernel_size=1, bias='auto', conv_cfg=conv_cfg, norm_cfg=norm_cfg))
        if residual and in_channels != out_channels:
            self.short_cut = build_conv_layer(conv_cfg, in_channels, out_channels, 1)
        else:
            self.short_cut = None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        """Forward function."""
        if self.use_stride_conv:
            assert self.causal_shift + self.kernel_size // 2 < x.shape[2]
        else:
            assert 0 <= self.pad + self.causal_shift < x.shape[2] - self.pad + self.causal_shift <= x.shape[2]
        out = self.conv1(x)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        if self.dropout is not None:
            out = self.dropout(out)
        if self.residual:
            if self.use_stride_conv:
                res = x[:, :, self.causal_shift + self.kernel_size // 2::self.kernel_size]
            else:
                res = x[:, :, self.pad + self.causal_shift:x.shape[2] - self.pad + self.causal_shift]
            if self.short_cut is not None:
                res = self.short_cut(res)
            out = out + res
        return out


class PytorchModuleHook(metaclass=ABCMeta):
    """Base class for PyTorch module hook registers.

    An instance of a subclass of PytorchModuleHook can be used to
    register hook to a pytorch module using the `register` method like:
        hook_register.register(module)

    Subclasses should add/overwrite the following methods:
        - __init__
        - hook
        - hook_type
    """

    @abstractmethod
    def hook(self, *args, **kwargs):
        """Hook function."""

    @abstractproperty
    def hook_type(self) ->str:
        """Hook type Subclasses should overwrite this function to return a
        string value in.

        {`forward`, `forward_pre`, `backward`}
        """

    def register(self, module):
        """Register the hook function to the module.

        Args:
            module (pytorch module): the module to register the hook.

        Returns:
            handle (torch.utils.hooks.RemovableHandle): a handle to remove
                the hook by calling handle.remove()
        """
        assert isinstance(module, torch.nn.Module)
        if self.hook_type == 'forward':
            h = module.register_forward_hook(self.hook)
        elif self.hook_type == 'forward_pre':
            h = module.register_forward_pre_hook(self.hook)
        elif self.hook_type == 'backward':
            h = module.register_backward_hook(self.hook)
        else:
            raise ValueError(f'Invalid hook type {self.hook}')
        return h


class WeightNormClipHook(PytorchModuleHook):
    """Apply weight norm clip regularization.

    The module's parameter will be clip to a given maximum norm before each
    forward pass.

    Args:
        max_norm (float): The maximum norm of the parameter.
        module_param_names (str|list): The parameter name (or name list) to
            apply weight norm clip.
    """

    def __init__(self, max_norm=1.0, module_param_names='weight'):
        self.module_param_names = module_param_names if isinstance(module_param_names, list) else [module_param_names]
        self.max_norm = max_norm

    @property
    def hook_type(self):
        return 'forward_pre'

    def hook(self, module, _input):
        for name in self.module_param_names:
            assert name in module._parameters, f'{name} is not a parameter of the module {type(module)}'
            param = module._parameters[name]
            with torch.no_grad():
                m = param.norm().item()
                if m > self.max_norm:
                    param.mul_(self.max_norm / (m + 1e-06))


class TCN(BaseBackbone):
    """TCN backbone.

    Temporal Convolutional Networks.
    More details can be found in the
    `paper <https://arxiv.org/abs/1811.11742>`__ .

    Args:
        in_channels (int): Number of input channels, which equals to
            num_keypoints * num_features.
        stem_channels (int): Number of feature channels. Default: 1024.
        num_blocks (int): NUmber of basic temporal convolutional blocks.
            Default: 2.
        kernel_sizes (Sequence[int]): Sizes of the convolving kernel of
            each basic block. Default: ``(3, 3, 3)``.
        dropout (float): Dropout rate. Default: 0.25.
        causal (bool): Use causal convolutions instead of symmetric
            convolutions (for real-time applications).
            Default: False.
        residual (bool): Use residual connection. Default: True.
        use_stride_conv (bool): Use TCN backbone optimized for
            single-frame batching, i.e. where batches have input length =
            receptive field, and output length = 1. This implementation
            replaces dilated convolutions with strided convolutions to avoid
            generating unused intermediate results. The weights are
            interchangeable with the reference implementation. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: dict(type='Conv1d').
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN1d').
        max_norm (float|None): if not None, the weight of convolution layers
            will be clipped to have a maximum norm of max_norm.

    Example:
        >>> from mmpose.models import TCN
        >>> import torch
        >>> self = TCN(in_channels=34)
        >>> self.eval()
        >>> inputs = torch.rand(1, 34, 243)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 1024, 235)
        (1, 1024, 217)
    """

    def __init__(self, in_channels, stem_channels=1024, num_blocks=2, kernel_sizes=(3, 3, 3), dropout=0.25, causal=False, residual=True, use_stride_conv=False, conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d'), max_norm=None):
        conv_cfg = copy.deepcopy(conv_cfg)
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.num_blocks = num_blocks
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.causal = causal
        self.residual = residual
        self.use_stride_conv = use_stride_conv
        self.max_norm = max_norm
        assert num_blocks == len(kernel_sizes) - 1
        for ks in kernel_sizes:
            assert ks % 2 == 1, 'Only odd filter widths are supported.'
        self.expand_conv = ConvModule(in_channels, stem_channels, kernel_size=kernel_sizes[0], stride=kernel_sizes[0] if use_stride_conv else 1, bias='auto', conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        dilation = kernel_sizes[0]
        self.tcn_blocks = nn.ModuleList()
        for i in range(1, num_blocks + 1):
            self.tcn_blocks.append(BasicTemporalBlock(in_channels=stem_channels, out_channels=stem_channels, mid_channels=stem_channels, kernel_size=kernel_sizes[i], dilation=dilation, dropout=dropout, causal=causal, residual=residual, use_stride_conv=use_stride_conv, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
            dilation *= kernel_sizes[i]
        if self.max_norm is not None:
            weight_clip = WeightNormClipHook(self.max_norm)
            for module in self.modules():
                if isinstance(module, nn.modules.conv._ConvNd):
                    weight_clip.register(module)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        """Forward function."""
        x = self.expand_conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        outs = []
        for i in range(self.num_blocks):
            x = self.tcn_blocks[i](x)
            outs.append(x)
        return tuple(outs)

    def init_weights(self, pretrained=None):
        """Initialize the weights."""
        super().init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.modules.conv._ConvNd):
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)


class Basic3DBlock(nn.Module):
    """A basic 3D convolutional block.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        kernel_size (int): Kernel size of the convolution operation
        conv_cfg (dict): Dictionary to construct and config conv layer.
            Default: dict(type='Conv3d')
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN3d')
    """

    def __init__(self, in_channels, out_channels, kernel_size, conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d')):
        super(Basic3DBlock, self).__init__()
        self.block = ConvModule(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=True)

    def forward(self, x):
        """Forward function."""
        return self.block(x)


class Res3DBlock(nn.Module):
    """A residual 3D convolutional block.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        kernel_size (int): Kernel size of the convolution operation
            Default: 3
        conv_cfg (dict): Dictionary to construct and config conv layer.
            Default: dict(type='Conv3d')
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN3d')
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d')):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(ConvModule(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=True), ConvModule(out_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None, bias=True))
        if in_channels == out_channels:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = ConvModule(in_channels, out_channels, 1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None, bias=True)

    def forward(self, x):
        """Forward function."""
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):
    """A 3D max-pool block.

    Args:
        pool_size (int): Pool size of the 3D max-pool layer
    """

    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        """Forward function."""
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample3DBlock(nn.Module):
    """A 3D upsample block.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        kernel_size (int): Kernel size of the transposed convolution operation.
            Default: 2
        stride (int):  Kernel size of the transposed convolution operation.
            Default: 2
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(Upsample3DBlock, self).__init__()
        assert kernel_size == 2
        assert stride == 2
        self.block = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0), nn.BatchNorm3d(out_channels), nn.ReLU(True))

    def forward(self, x):
        """Forward function."""
        return self.block(x)


class EncoderDecorder(nn.Module):
    """An encoder-decoder block.

    Args:
        in_channels (int): Input channels of this block
    """

    def __init__(self, in_channels=32):
        super(EncoderDecorder, self).__init__()
        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(in_channels, in_channels * 2)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(in_channels * 2, in_channels * 4)
        self.mid_res = Res3DBlock(in_channels * 4, in_channels * 4)
        self.decoder_res2 = Res3DBlock(in_channels * 4, in_channels * 4)
        self.decoder_upsample2 = Upsample3DBlock(in_channels * 4, in_channels * 2, 2, 2)
        self.decoder_res1 = Res3DBlock(in_channels * 2, in_channels * 2)
        self.decoder_upsample1 = Upsample3DBlock(in_channels * 2, in_channels, 2, 2)
        self.skip_res1 = Res3DBlock(in_channels, in_channels)
        self.skip_res2 = Res3DBlock(in_channels * 2, in_channels * 2)

    def forward(self, x):
        """Forward function."""
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        x = self.mid_res(x)
        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1
        return x


class V2VNet(BaseBackbone):
    """V2VNet.

    Please refer to the `paper <https://arxiv.org/abs/1711.07399>`
        for details.

    Args:
        input_channels (int):
            Number of channels of the input feature volume.
        output_channels (int):
            Number of channels of the output volume.
        mid_channels (int):
            Input and output channels of the encoder-decoder block.
    """

    def __init__(self, input_channels, output_channels, mid_channels=32):
        super(V2VNet, self).__init__()
        self.front_layers = nn.Sequential(Basic3DBlock(input_channels, mid_channels // 2, 7), Res3DBlock(mid_channels // 2, mid_channels))
        self.encoder_decoder = EncoderDecorder(in_channels=mid_channels)
        self.output_layer = nn.Conv3d(mid_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self._initialize_weights()

    def forward(self, x):
        """Forward function."""
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


def make_vgg_layer(in_channels, out_channels, num_blocks, conv_cfg=None, norm_cfg=None, act_cfg=dict(type='ReLU'), dilation=1, with_norm=False, ceil_mode=False):
    layers = []
    for _ in range(num_blocks):
        layer = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=dilation, padding=dilation, bias=True, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        layers.append(layer)
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))
    return layers


class VGG(BaseBackbone):
    """VGG backbone.

    Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        with_norm (bool): Use BatchNorm or not.
        num_classes (int): number of classes for classification.
        num_stages (int): VGG stages, normally 5.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. When it is None, the default behavior depends on
            whether num_classes is specified. If num_classes <= 0, the default
            value is (4, ), outputting the last feature map before classifier.
            If num_classes > 0, the default value is (5, ), outputting the
            classification score. Default: None.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        ceil_mode (bool): Whether to use ceil_mode of MaxPool. Default: False.
        with_last_pool (bool): Whether to keep the last pooling before
            classifier. Default: True.
    """
    arch_settings = {(11): (1, 1, 2, 2, 2), (13): (2, 2, 2, 2, 2), (16): (2, 2, 3, 3, 3), (19): (2, 2, 4, 4, 4)}

    def __init__(self, depth, num_classes=-1, num_stages=5, dilations=(1, 1, 1, 1, 1), out_indices=None, frozen_stages=-1, conv_cfg=None, norm_cfg=None, act_cfg=dict(type='ReLU'), norm_eval=False, ceil_mode=False, with_last_pool=True):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for vgg')
        assert num_stages >= 1 and num_stages <= 5
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        assert len(dilations) == num_stages
        self.num_classes = num_classes
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        with_norm = norm_cfg is not None
        if out_indices is None:
            out_indices = (5,) if num_classes > 0 else (4,)
        assert max(out_indices) <= num_stages
        self.out_indices = out_indices
        self.in_channels = 3
        start_idx = 0
        vgg_layers = []
        self.range_sub_modules = []
        for i, num_blocks in enumerate(self.stage_blocks):
            num_modules = num_blocks + 1
            end_idx = start_idx + num_modules
            dilation = dilations[i]
            out_channels = 64 * 2 ** i if i < 4 else 512
            vgg_layer = make_vgg_layer(self.in_channels, out_channels, num_blocks, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, dilation=dilation, with_norm=with_norm, ceil_mode=ceil_mode)
            vgg_layers.extend(vgg_layer)
            self.in_channels = out_channels
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
        super().init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)

    def forward(self, x):
        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i in range(len(self.stage_blocks)):
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

    def _freeze_stages(self):
        vgg_layers = getattr(self, self.module_name)
        for i in range(self.frozen_stages):
            for j in range(*self.range_sub_modules[i]):
                m = vgg_layers[j]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class ViPNAS_MobileNetV3(BaseBackbone):
    """ViPNAS_MobileNetV3 backbone.

    "ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search"
    More details can be found in the `paper
    <https://arxiv.org/abs/2105.10154>`__ .

    Args:
        wid (list(int)): Searched width config for each stage.
        expan (list(int)): Searched expansion ratio config for each stage.
        dep (list(int)): Searched depth config for each stage.
        ks (list(int)): Searched kernel size config for each stage.
        group (list(int)): Searched group number config for each stage.
        att (list(bool)): Searched attention config for each stage.
        stride (list(int)): Stride config for each stage.
        act (list(dict)): Activation config for each stage.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed.
            Default: False.
    """

    def __init__(self, wid=[16, 16, 24, 40, 80, 112, 160], expan=[None, 1, 5, 4, 5, 5, 6], dep=[None, 1, 4, 4, 4, 4, 4], ks=[3, 3, 7, 7, 5, 7, 5], group=[None, 8, 120, 20, 100, 280, 240], att=[None, True, True, False, True, True, True], stride=[2, 1, 2, 2, 2, 1, 2], act=['HSwish', 'ReLU', 'ReLU', 'ReLU', 'HSwish', 'HSwish', 'HSwish'], conv_cfg=None, norm_cfg=dict(type='BN'), frozen_stages=-1, norm_eval=False, with_cp=False):
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.wid = wid
        self.expan = expan
        self.dep = dep
        self.ks = ks
        self.group = group
        self.att = att
        self.stride = stride
        self.act = act
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.conv1 = ConvModule(in_channels=3, out_channels=self.wid[0], kernel_size=self.ks[0], stride=self.stride[0], padding=self.ks[0] // 2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=dict(type=self.act[0]))
        self.layers = self._make_layer()

    def _make_layer(self):
        layers = []
        layer_index = 0
        for i, dep in enumerate(self.dep[1:]):
            mid_channels = self.wid[i + 1] * self.expan[i + 1]
            if self.att[i + 1]:
                se_cfg = dict(channels=mid_channels, ratio=4, act_cfg=(dict(type='ReLU'), dict(type='HSigmoid')))
            else:
                se_cfg = None
            if self.expan[i + 1] == 1:
                with_expand_conv = False
            else:
                with_expand_conv = True
            for j in range(dep):
                if j == 0:
                    stride = self.stride[i + 1]
                    in_channels = self.wid[i]
                else:
                    stride = 1
                    in_channels = self.wid[i + 1]
                layer = InvertedResidual(in_channels=in_channels, out_channels=self.wid[i + 1], mid_channels=mid_channels, kernel_size=self.ks[i + 1], groups=self.group[i + 1], stride=stride, se_cfg=se_cfg, with_expand_conv=with_expand_conv, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=dict(type=self.act[i + 1]), with_cp=self.with_cp)
                layer_index += 1
                layer_name = f'layer{layer_index}'
                self.add_module(layer_name, layer)
                layers.append(layer_name)
        return layers

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    for name, _ in m.named_parameters():
                        if name in ['bias']:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
        return x

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class ViPNAS_ResLayer(nn.Sequential):
    """ViPNAS_ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ViPNAS ResLayer.
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
        kernel_size (int): Kernel Size of the corresponding convolution layer
            searched in the block.
        groups (int): Group number of the corresponding convolution layer
            searched in the block.
        attention (bool): Whether to use attention module in the end of the
            block.
    """

    def __init__(self, block, num_blocks, in_channels, out_channels, expansion=None, stride=1, avg_down=False, conv_cfg=None, norm_cfg=dict(type='BN'), downsample_first=True, kernel_size=3, groups=1, attention=False, **kwargs):
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
            layers.append(block(in_channels=in_channels, out_channels=out_channels, expansion=self.expansion, stride=stride, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, kernel_size=kernel_size, groups=groups, attention=attention, **kwargs))
            in_channels = out_channels
            for _ in range(1, num_blocks):
                layers.append(block(in_channels=in_channels, out_channels=out_channels, expansion=self.expansion, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, kernel_size=kernel_size, groups=groups, attention=attention, **kwargs))
        else:
            for i in range(0, num_blocks - 1):
                layers.append(block(in_channels=in_channels, out_channels=in_channels, expansion=self.expansion, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, kernel_size=kernel_size, groups=groups, attention=attention, **kwargs))
            layers.append(block(in_channels=in_channels, out_channels=out_channels, expansion=self.expansion, stride=stride, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, kernel_size=kernel_size, groups=groups, attention=attention, **kwargs))
        super().__init__(*layers)


class ViPNAS_ResNet(BaseBackbone):
    """ViPNAS_ResNet backbone.

    "ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search"
    More details can be found in the `paper
    <https://arxiv.org/abs/2105.10154>`__ .

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
        wid (list(int)): Searched width config for each stage.
        expan (list(int)): Searched expansion ratio config for each stage.
        dep (list(int)): Searched depth config for each stage.
        ks (list(int)): Searched kernel size config for each stage.
        group (list(int)): Searched group number config for each stage.
        att (list(bool)): Searched attention config for each stage.
    """
    arch_settings = {(50): ViPNAS_Bottleneck}

    def __init__(self, depth, in_channels=3, num_stages=4, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1), out_indices=(3,), style='pytorch', deep_stem=False, avg_down=False, frozen_stages=-1, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), norm_eval=False, with_cp=False, zero_init_residual=True, wid=[48, 80, 160, 304, 608], expan=[None, 1, 1, 1, 1], dep=[None, 4, 6, 7, 3], ks=[7, 3, 5, 5, 5], group=[None, 16, 16, 16, 16], att=[None, True, False, True, True]):
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = dep[0]
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block = self.arch_settings[depth]
        self.stage_blocks = dep[1:1 + num_stages]
        self._make_stem_layer(in_channels, wid[0], ks[0])
        self.res_layers = []
        _in_channels = wid[0]
        for i, num_blocks in enumerate(self.stage_blocks):
            expansion = get_expansion(self.block, expan[i + 1])
            _out_channels = wid[i + 1] * expansion
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(block=self.block, num_blocks=num_blocks, in_channels=_in_channels, out_channels=_out_channels, expansion=expansion, stride=stride, dilation=dilation, style=self.style, avg_down=self.avg_down, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, kernel_size=ks[i + 1], groups=group[i + 1], attention=att[i + 1])
            _in_channels = _out_channels
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self._freeze_stages()
        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        """Make a ViPNAS ResLayer."""
        return ViPNAS_ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels, kernel_size):
        """Make stem layer."""
        if self.deep_stem:
            self.stem = nn.Sequential(ConvModule(in_channels, stem_channels // 2, kernel_size=3, stride=2, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, inplace=True), ConvModule(stem_channels // 2, stem_channels // 2, kernel_size=3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, inplace=True), ConvModule(stem_channels // 2, stem_channels, kernel_size=3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, inplace=True))
        else:
            self.conv1 = build_conv_layer(self.conv_cfg, in_channels, stem_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, bias=False)
            self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super().init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    for name, _ in m.named_parameters():
                        if name in ['bias']:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):

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
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MoEMlp(nn.Module):

    def __init__(self, num_expert=1, in_features=1024, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, part_features=256):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.part_features = part_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features - part_features)
        self.drop = nn.Dropout(drop)
        self.num_expert = num_expert
        experts = []
        for i in range(num_expert):
            experts.append(nn.Linear(hidden_features, part_features))
        self.experts = nn.ModuleList(experts)

    def forward(self, x, indices):
        expert_x = torch.zeros_like(x[:, :, -self.part_features:], device=x.device, dtype=x.dtype)
        x = self.fc1(x)
        x = self.act(x)
        shared_x = self.fc2(x)
        indices = indices.view(-1, 1, 1)
        for i in range(self.num_expert):
            selectedIndex = indices == i
            current_x = self.experts[i](x) * selectedIndex
            expert_x = expert_x + current_x
        x = torch.cat([shared_x, expert_x], dim=-1)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None, num_expert=1, part_features=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MoEMlp(num_expert=num_expert, in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, part_features=part_features)

    def forward(self, x, indices=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), indices))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0]) * ratio ** 2
        self.patch_shape = int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio)
        self.origin_patch_shape = int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size[0] // ratio, padding=4 + 2 * (ratio // 2 - 1))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ViT(BaseBackbone):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=None, use_checkpoint=False, frozen_stages=-1, ratio=1, last_norm=True, patch_padding='pad', freeze_attn=False, freeze_ffn=False):
        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False
        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super().init_weights(pretrained, patch_padding=self.patch_padding)
        if pretrained is None:

            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
            self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.last_norm(x)
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        return xp

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()


class ViTMoE(BaseBackbone):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=None, use_checkpoint=False, frozen_stages=-1, ratio=1, last_norm=True, patch_padding='pad', freeze_attn=False, freeze_ffn=False, num_expert=1, part_features=None):
        super(ViTMoE, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches
        self.part_features = part_features
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_expert=num_expert, part_features=part_features) for i in range(depth)])
        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False
        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super().init_weights(pretrained, patch_padding=self.patch_padding, part_features=self.part_features)
        if pretrained is None:

            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
            self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, dataset_source=None):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, dataset_source)
            else:
                x = blk(x, dataset_source)
        x = self.last_norm(x)
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        return xp

    def forward(self, x, dataset_source=None):
        x = self.forward_features(x, dataset_source)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()


class BasePose(nn.Module, metaclass=ABCMeta):
    """Base class for pose detectors.

    All recognizers should subclass it.
    All subclass should overwrite:
        Methods:`forward_train`, supporting to forward when training.
        Methods:`forward_test`, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        head (dict): Head modules to give output.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    @abstractmethod
    def forward_train(self, img, img_metas, **kwargs):
        """Defines the computation performed at training."""

    @abstractmethod
    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at testing."""

    @abstractmethod
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Forward function."""

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor                 which may be a weighted sum of all losses, log_vars                 contains all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, float):
                log_vars[loss_name] = loss_value
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors or float')
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if not isinstance(loss_value, float):
                if dist.is_available() and dist.is_initialized():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
                log_vars[loss_name] = loss_value.item()
            else:
                log_vars[loss_name] = loss_value
        return loss, log_vars

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self.forward(**data_batch)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(next(iter(data_batch.values()))))
        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        results = self.forward(return_loss=False, **data_batch)
        outputs = dict(results=results)
        return outputs

    @abstractmethod
    def show_result(self, **kwargs):
        """Visualize the results."""
        raise NotImplementedError


class BaseDiscriminator(nn.Module):
    """Base linear module for SMPL parameter discriminator.

    Args:
        fc_layers (Tuple): Tuple of neuron count,
            such as (9, 32, 32, 1)
        use_dropout (Tuple): Tuple of bool define use dropout or not
            for each layer, such as (True, True, False)
        drop_prob (Tuple): Tuple of float defined the drop prob,
            such as (0.5, 0.5, 0)
        use_activation(Tuple): Tuple of bool define use active function
            or not, such as (True, True, False)
    """

    def __init__(self, fc_layers, use_dropout, drop_prob, use_activation):
        super().__init__()
        self.fc_layers = fc_layers
        self.use_dropout = use_dropout
        self.drop_prob = drop_prob
        self.use_activation = use_activation
        self._check()
        self.create_layers()

    def _check(self):
        """Check input to avoid ValueError."""
        if not isinstance(self.fc_layers, tuple):
            raise TypeError(f'fc_layers require tuple, get {type(self.fc_layers)}')
        if not isinstance(self.use_dropout, tuple):
            raise TypeError(f'use_dropout require tuple, get {type(self.use_dropout)}')
        if not isinstance(self.drop_prob, tuple):
            raise TypeError(f'drop_prob require tuple, get {type(self.drop_prob)}')
        if not isinstance(self.use_activation, tuple):
            raise TypeError(f'use_activation require tuple, get {type(self.use_activation)}')
        l_fc_layer = len(self.fc_layers)
        l_use_drop = len(self.use_dropout)
        l_drop_prob = len(self.drop_prob)
        l_use_activation = len(self.use_activation)
        pass_check = l_fc_layer >= 2 and l_use_drop < l_fc_layer and l_drop_prob < l_fc_layer and l_use_activation < l_fc_layer and l_drop_prob == l_use_drop
        if not pass_check:
            msg = 'Wrong BaseDiscriminator parameters!'
            raise ValueError(msg)

    def create_layers(self):
        """Create layers."""
        l_fc_layer = len(self.fc_layers)
        l_use_drop = len(self.use_dropout)
        l_use_activation = len(self.use_activation)
        self.fc_blocks = nn.Sequential()
        for i in range(l_fc_layer - 1):
            self.fc_blocks.add_module(name=f'regressor_fc_{i}', module=nn.Linear(in_features=self.fc_layers[i], out_features=self.fc_layers[i + 1]))
            if i < l_use_activation and self.use_activation[i]:
                self.fc_blocks.add_module(name=f'regressor_af_{i}', module=nn.ReLU())
            if i < l_use_drop and self.use_dropout[i]:
                self.fc_blocks.add_module(name=f'regressor_fc_dropout_{i}', module=nn.Dropout(p=self.drop_prob[i]))

    @abstractmethod
    def forward(self, inputs):
        """Forward function."""
        msg = 'the base class [BaseDiscriminator] is not callable!'
        raise NotImplementedError(msg)

    def init_weights(self):
        """Initialize model weights."""
        for m in self.fc_blocks.named_modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, gain=0.01)


class FullPoseDiscriminator(BaseDiscriminator):
    """Discriminator for SMPL pose parameters of all joints.

    Args:
        fc_layers (Tuple): Tuple of neuron count,
            such as (736, 1024, 1024, 1)
        use_dropout (Tuple): Tuple of bool define use dropout or not
            for each layer, such as (True, True, False)
        drop_prob (Tuple): Tuple of float defined the drop prob,
            such as (0.5, 0.5, 0)
        use_activation(Tuple): Tuple of bool define use active
            function or not, such as (True, True, False)
    """

    def __init__(self, fc_layers, use_dropout, drop_prob, use_activation):
        if fc_layers[-1] != 1:
            msg = f'the neuron count of the last layer must be 1, but got {fc_layers[-1]}'
            raise ValueError(msg)
        super().__init__(fc_layers, use_dropout, drop_prob, use_activation)

    def forward(self, inputs):
        """Forward function."""
        return self.fc_blocks(inputs)


class PoseDiscriminator(nn.Module):
    """Discriminator for SMPL pose parameters of each joint. It is composed of
    discriminators for each joints. The inputs is (batch_size x joint_count x
    9)

    Args:
        channels (Tuple): Tuple of channel number,
            such as (9, 32, 32, 1)
        joint_count (int): Joint number, such as 23
    """

    def __init__(self, channels, joint_count):
        super().__init__()
        if channels[-1] != 1:
            msg = f'the neuron count of the last layer must be 1, but got {channels[-1]}'
            raise ValueError(msg)
        self.joint_count = joint_count
        self.conv_blocks = nn.Sequential()
        len_channels = len(channels)
        for idx in range(len_channels - 2):
            self.conv_blocks.add_module(name=f'conv_{idx}', module=nn.Conv2d(in_channels=channels[idx], out_channels=channels[idx + 1], kernel_size=1, stride=1))
        self.fc_layer = nn.ModuleList()
        for idx in range(joint_count):
            self.fc_layer.append(nn.Linear(in_features=channels[len_channels - 2], out_features=1))

    def forward(self, inputs):
        """Forward function.

        The input is (batch_size x joint_count x 9).
        """
        inputs = inputs.transpose(1, 2).unsqueeze(2).contiguous()
        internal_outputs = self.conv_blocks(inputs)
        outputs = []
        for idx in range(self.joint_count):
            outputs.append(self.fc_layer[idx](internal_outputs[:, :, 0, idx]))
        return torch.cat(outputs, 1), internal_outputs

    def init_weights(self):
        """Initialize model weights."""
        for m in self.conv_blocks:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
        for m in self.fc_layer.named_modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, gain=0.01)


class ShapeDiscriminator(BaseDiscriminator):
    """Discriminator for SMPL shape parameters, the inputs is (batch_size x 10)

    Args:
        fc_layers (Tuple): Tuple of neuron count, such as (10, 5, 1)
        use_dropout (Tuple): Tuple of bool define use dropout or
            not for each layer, such as (True, True, False)
        drop_prob (Tuple): Tuple of float defined the drop prob,
            such as (0.5, 0)
        use_activation(Tuple): Tuple of bool define use active
            function or not, such as (True, False)
    """

    def __init__(self, fc_layers, use_dropout, drop_prob, use_activation):
        if fc_layers[-1] != 1:
            msg = f'the neuron count of the last layer must be 1, but got {fc_layers[-1]}'
            raise ValueError(msg)
        super().__init__(fc_layers, use_dropout, drop_prob, use_activation)

    def forward(self, inputs):
        """Forward function."""
        return self.fc_blocks(inputs)


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion
            -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion
            -- size = [B, 3, 3]
    """
    l2norm = torch.norm(theta + 1e-08, p=2, dim=1)
    angle = torch.unsqueeze(l2norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


class SMPLDiscriminator(nn.Module):
    """Discriminator for SMPL pose and shape parameters. It is composed of a
    discriminator for SMPL shape parameters, a discriminator for SMPL pose
    parameters of all joints  and a discriminator for SMPL pose parameters of
    each joint.

    Args:
        beta_channel (tuple of int): Tuple of neuron count of the
            discriminator of shape parameters. Defaults to (10, 5, 1)
        per_joint_channel (tuple of int): Tuple of neuron count of the
            discriminator of each joint. Defaults to (9, 32, 32, 1)
        full_pose_channel (tuple of int): Tuple of neuron count of the
            discriminator of full pose. Defaults to (23*32, 1024, 1024, 1)
    """

    def __init__(self, beta_channel=(10, 5, 1), per_joint_channel=(9, 32, 32, 1), full_pose_channel=(23 * 32, 1024, 1024, 1)):
        super().__init__()
        self.joint_count = 23
        assert beta_channel[0] == 10
        assert per_joint_channel[0] == 9
        assert self.joint_count * per_joint_channel[-2] == full_pose_channel[0]
        self.beta_channel = beta_channel
        self.per_joint_channel = per_joint_channel
        self.full_pose_channel = full_pose_channel
        self._create_sub_modules()

    def _create_sub_modules(self):
        """Create sub discriminators."""
        self.pose_discriminator = PoseDiscriminator(self.per_joint_channel, self.joint_count)
        fc_layers = self.full_pose_channel
        use_dropout = tuple([False] * (len(fc_layers) - 1))
        drop_prob = tuple([0.5] * (len(fc_layers) - 1))
        use_activation = tuple([True] * (len(fc_layers) - 2) + [False])
        self.full_pose_discriminator = FullPoseDiscriminator(fc_layers, use_dropout, drop_prob, use_activation)
        fc_layers = self.beta_channel
        use_dropout = tuple([False] * (len(fc_layers) - 1))
        drop_prob = tuple([0.5] * (len(fc_layers) - 1))
        use_activation = tuple([True] * (len(fc_layers) - 2) + [False])
        self.shape_discriminator = ShapeDiscriminator(fc_layers, use_dropout, drop_prob, use_activation)

    def forward(self, thetas):
        """Forward function."""
        _, poses, shapes = thetas
        batch_size = poses.shape[0]
        shape_disc_value = self.shape_discriminator(shapes)
        if poses.dim() == 2:
            rotate_matrixs = batch_rodrigues(poses.contiguous().view(-1, 3)).view(batch_size, 24, 9)[:, 1:, :]
        else:
            rotate_matrixs = poses.contiguous().view(batch_size, 24, 9)[:, 1:, :].contiguous()
        pose_disc_value, pose_inter_disc_value = self.pose_discriminator(rotate_matrixs)
        full_pose_disc_value = self.full_pose_discriminator(pose_inter_disc_value.contiguous().view(batch_size, -1))
        return torch.cat((pose_disc_value, full_pose_disc_value, shape_disc_value), 1)

    def init_weights(self):
        """Initialize model weights."""
        self.full_pose_discriminator.init_weights()
        self.pose_discriminator.init_weights()
        self.shape_discriminator.init_weights()


def imshow_mesh_3d(img, vertices, faces, camera_center, focal_length, colors=(76, 76, 204)):
    """Render 3D meshes on background image.

    Args:
        img(np.ndarray): Background image.
        vertices (list of np.ndarray): Vetrex coordinates in camera space.
        faces (list of np.ndarray): Faces of meshes.
        camera_center ([2]): Center pixel.
        focal_length ([2]): Focal length of camera.
        colors (list[str or tuple or Color]): A list of mesh colors.
    """
    H, W, C = img.shape
    if not has_pyrender:
        warnings.warn('pyrender package is not installed.')
        return img
    if not has_trimesh:
        warnings.warn('trimesh package is not installed.')
        return img
    try:
        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    except (ImportError, RuntimeError):
        warnings.warn('pyrender package is not installed correctly.')
        return img
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(vertices))]
    colors = [color_val(c) for c in colors]
    depth_map = np.ones([H, W]) * np.inf
    output_img = img
    for idx in range(len(vertices)):
        color = colors[idx]
        color = [(c / 255.0) for c in color]
        color.append(1.0)
        vert = vertices[idx]
        face = faces[idx]
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.2, alphaMode='OPAQUE', baseColorFactor=color)
        mesh = trimesh.Trimesh(vert, face)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')
        camera_pose = np.eye(4)
        camera = pyrender.IntrinsicsCamera(fx=focal_length[0], fy=focal_length[1], cx=camera_center[0], cy=camera_center[1], zfar=100000.0)
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)
        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        valid_mask = (rend_depth < depth_map) * (rend_depth > 0)
        depth_map[valid_mask] = rend_depth[valid_mask]
        valid_mask = valid_mask[:, :, None]
        output_img = valid_mask * color[:, :, :3] + (1 - valid_mask) * output_img
    return output_img


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class ParametricMesh(BasePose):
    """Model-based 3D human mesh detector. Take a single color image as input
    and output 3D joints, SMPL parameters and camera parameters.

    Args:
        backbone (dict): Backbone modules to extract feature.
        mesh_head (dict): Mesh head to process feature.
        smpl (dict): Config for SMPL model.
        disc (dict): Discriminator for SMPL parameters. Default: None.
        loss_gan (dict): Config for adversarial loss. Default: None.
        loss_mesh (dict): Config for mesh loss. Default: None.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
    """

    def __init__(self, backbone, mesh_head, smpl, disc=None, loss_gan=None, loss_mesh=None, train_cfg=None, test_cfg=None, pretrained=None):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        self.mesh_head = builder.build_head(mesh_head)
        self.generator = torch.nn.Sequential(self.backbone, self.mesh_head)
        self.smpl = builder.build_mesh_model(smpl)
        self.with_gan = disc is not None and loss_gan is not None
        if self.with_gan:
            self.discriminator = SMPLDiscriminator(**disc)
            self.loss_gan = builder.build_loss(loss_gan)
        self.disc_step_count = 0
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_mesh = builder.build_loss(loss_mesh)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        self.mesh_head.init_weights()
        if self.with_gan:
            self.discriminator.init_weights()

    def train_step(self, data_batch, optimizer, **kwargs):
        """Train step function.

        In this function, the detector will finish the train step following
        the pipeline:

            1. get fake and real SMPL parameters
            2. optimize discriminator (if have)
            3. optimize generator

        If `self.train_cfg.disc_step > 1`, the train step will contain multiple
        iterations for optimizing discriminator with different input data and
        only one iteration for optimizing generator after `disc_step`
        iterations for discriminator.

        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (dict[torch.optim.Optimizer]): Dict with optimizers for
                generator and discriminator (if have).

        Returns:
            outputs (dict): Dict with loss, information for logger,
            the number of samples.
        """
        img = data_batch['img']
        pred_smpl = self.generator(img)
        pred_pose, pred_beta, pred_camera = pred_smpl
        if self.train_cfg['disc_step'] > 0 and self.with_gan:
            set_requires_grad(self.discriminator, True)
            fake_data = pred_camera.detach(), pred_pose.detach(), pred_beta.detach()
            mosh_theta = data_batch['mosh_theta']
            real_data = mosh_theta[:, :3], mosh_theta[:, 3:75], mosh_theta[:, 75:]
            fake_score = self.discriminator(fake_data)
            real_score = self.discriminator(real_data)
            disc_losses = {}
            disc_losses['real_loss'] = self.loss_gan(real_score, target_is_real=True, is_disc=True)
            disc_losses['fake_loss'] = self.loss_gan(fake_score, target_is_real=False, is_disc=True)
            loss_disc, log_vars_d = self._parse_losses(disc_losses)
            optimizer['discriminator'].zero_grad()
            loss_disc.backward()
            optimizer['discriminator'].step()
            self.disc_step_count = (self.disc_step_count + 1) % self.train_cfg['disc_step']
            if self.disc_step_count != 0:
                outputs = dict(loss=loss_disc, log_vars=log_vars_d, num_samples=len(next(iter(data_batch.values()))))
                return outputs
        pred_out = self.smpl(betas=pred_beta, body_pose=pred_pose[:, 1:], global_orient=pred_pose[:, :1])
        pred_vertices, pred_joints_3d = pred_out['vertices'], pred_out['joints']
        gt_beta = data_batch['beta']
        gt_pose = data_batch['pose']
        gt_vertices = self.smpl(betas=gt_beta, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])['vertices']
        pred = dict(pose=pred_pose, beta=pred_beta, camera=pred_camera, vertices=pred_vertices, joints_3d=pred_joints_3d)
        target = {key: data_batch[key] for key in ['pose', 'beta', 'has_smpl', 'joints_3d', 'joints_2d', 'joints_3d_visible', 'joints_2d_visible']}
        target['vertices'] = gt_vertices
        losses = self.loss_mesh(pred, target)
        if self.with_gan:
            set_requires_grad(self.discriminator, False)
            pred_theta = pred_camera, pred_pose, pred_beta
            pred_score = self.discriminator(pred_theta)
            loss_adv = self.loss_gan(pred_score, target_is_real=True, is_disc=False)
            losses['adv_loss'] = loss_adv
        loss, log_vars = self._parse_losses(losses)
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(next(iter(data_batch.values()))))
        return outputs

    def forward_train(self, *args, **kwargs):
        """Forward function for training.

        For ParametricMesh, we do not use this interface.
        """
        raise NotImplementedError('This interface should not be used in current training schedule. Please use `train_step` for training.')

    def val_step(self, data_batch, **kwargs):
        """Forward function for evaluation.

        Args:
            data_batch (dict): Contain data for forward.

        Returns:
            dict: Contain the results from model.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Outputs.
        """
        output = self.generator(img)
        return output

    def forward_test(self, img, img_metas, return_vertices=False, return_faces=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        pred_smpl = self.generator(img)
        pred_pose, pred_beta, pred_camera = pred_smpl
        pred_out = self.smpl(betas=pred_beta, body_pose=pred_pose[:, 1:], global_orient=pred_pose[:, :1])
        pred_vertices, pred_joints_3d = pred_out['vertices'], pred_out['joints']
        all_preds = {}
        all_preds['keypoints_3d'] = pred_joints_3d.detach().cpu().numpy()
        all_preds['smpl_pose'] = pred_pose.detach().cpu().numpy()
        all_preds['smpl_beta'] = pred_beta.detach().cpu().numpy()
        all_preds['camera'] = pred_camera.detach().cpu().numpy()
        if return_vertices:
            all_preds['vertices'] = pred_vertices.detach().cpu().numpy()
        if return_faces:
            all_preds['faces'] = self.smpl.get_faces()
        all_boxes = []
        image_path = []
        for img_meta in img_metas:
            box = np.zeros(6, dtype=np.float32)
            c = img_meta['center']
            s = img_meta['scale']
            if 'bbox_score' in img_metas:
                score = np.array(img_metas['bbox_score']).reshape(-1)
            else:
                score = 1.0
            box[0:2] = c
            box[2:4] = s
            box[4] = np.prod(s * 200.0, axis=0)
            box[5] = score
            all_boxes.append(box)
            image_path.append(img_meta['image_file'])
        all_preds['bboxes'] = np.stack(all_boxes, axis=0)
        all_preds['image_path'] = image_path
        return all_preds

    def get_3d_joints_from_mesh(self, vertices):
        """Get 3D joints from 3D mesh using predefined joints regressor."""
        return torch.matmul(self.joints_regressor, vertices)

    def forward(self, img, img_metas=None, return_loss=False, **kwargs):
        """Forward function.

        Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note:
            - batch_size: N
            - num_img_channel: C (Default: 3)
            - img height: imgH
            - img width: imgW

        Args:
            img (torch.Tensor[N x C x imgH x imgW]): Input images.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.

        Returns:
            Return predicted 3D joints, SMPL parameters, boxes and image paths.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        return self.forward_test(img, img_metas, **kwargs)

    def show_result(self, result, img, show=False, out_file=None, win_name='', wait_time=0, bbox_color='green', mesh_color=(76, 76, 204), **kwargs):
        """Visualize 3D mesh estimation results.

        Args:
            result (list[dict]): The mesh estimation results containing:

               - "bbox" (ndarray[4]): instance bounding bbox
               - "center" (ndarray[2]): bbox center
               - "scale" (ndarray[2]): bbox scale
               - "keypoints_3d" (ndarray[K,3]): predicted 3D keypoints
               - "camera" (ndarray[3]): camera parameters
               - "vertices" (ndarray[V, 3]): predicted 3D vertices
               - "faces" (ndarray[F, 3]): mesh faces
            img (str or Tensor): Optional. The image to visualize 2D inputs on.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            wait_time (int): Value of waitKey param. Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            mesh_color (str or tuple or :obj:`Color`): Color of mesh surface.

        Returns:
            ndarray: Visualized img, only if not `show` or `out_file`.
        """
        if img is not None:
            img = mmcv.imread(img)
        focal_length = self.loss_mesh.focal_length
        H, W, C = img.shape
        img_center = np.array([[0.5 * W], [0.5 * H]])
        bboxes = [res['bbox'] for res in result]
        bboxes = np.vstack(bboxes)
        mmcv.imshow_bboxes(img, bboxes, colors=bbox_color, top_k=-1, thickness=2, show=False)
        vertex_list = []
        face_list = []
        for res in result:
            vertices = res['vertices']
            faces = res['faces']
            camera = res['camera']
            camera_center = res['center']
            scale = res['scale']
            translation = np.array([camera[1], camera[2], 2 * focal_length / (scale[0] * 200.0 * camera[0] + 1e-09)])
            mean_depth = vertices[:, -1].mean() + translation[-1]
            translation[:2] += (camera_center - img_center[:, 0]) / focal_length * mean_depth
            vertices += translation[None, :]
            vertex_list.append(vertices)
            face_list.append(faces)
        img_vis = imshow_mesh_3d(img, vertex_list, face_list, img_center, [focal_length, focal_length], colors=mesh_color)
        R = cv2.Rodrigues(np.array([0, np.radians(90.0), 0]))[0]
        rot_vertex_list = [np.dot(vert, R) for vert in vertex_list]
        rot_vertices = np.concatenate(rot_vertex_list, axis=0)
        min_corner = rot_vertices.min(0)
        max_corner = rot_vertices.max(0)
        center_3d = 0.5 * (min_corner + max_corner)
        ratio = 0.8
        bbox3d_size = max_corner - min_corner
        z_x = bbox3d_size[0] * focal_length / (ratio * W) - min_corner[2]
        z_y = bbox3d_size[1] * focal_length / (ratio * H) - min_corner[2]
        z = max(z_x, z_y)
        translation = -center_3d
        translation[2] = z
        translation = translation[None, :]
        rot_vertex_list = [(rot_vert + translation) for rot_vert in rot_vertex_list]
        img_side = imshow_mesh_3d(np.ones_like(img) * 255, rot_vertex_list, face_list, img_center, [focal_length, focal_length])
        img_vis = np.concatenate([img_vis, img_side], axis=1)
        if show:
            mmcv.visualization.imshow(img_vis, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img_vis, out_file)
        return img_vis


class MultiTask(nn.Module):
    """Multi-task detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        heads (list[dict]): heads to output predictions.
        necks (list[dict] | None): necks to process feature.
        head2neck (dict{int:int}): head index to neck index.
        pretrained (str): Path to the pretrained models.
    """

    def __init__(self, backbone, heads, necks=None, head2neck=None, pretrained=None):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        if head2neck is None:
            assert necks is None
            head2neck = {}
        self.head2neck = {}
        for i in range(len(heads)):
            self.head2neck[i] = head2neck[i] if i in head2neck else -1
        self.necks = nn.ModuleList([])
        if necks is not None:
            for neck in necks:
                self.necks.append(builder.build_neck(neck))
        self.necks.append(nn.Identity())
        self.heads = nn.ModuleList([])
        assert heads is not None
        for head in heads:
            assert head is not None
            self.heads.append(builder.build_head(head))
        self.init_weights(pretrained=pretrained)

    @property
    def with_necks(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'necks')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        if self.with_necks:
            for neck in self.necks:
                if hasattr(neck, 'init_weights'):
                    neck.init_weights()
        for head in self.heads:
            if hasattr(head, 'init_weights'):
                head.init_weights()

    def forward(self, img, target=None, target_weight=None, img_metas=None, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_img_channel: C (Default: 3)
            - img height: imgH
            - img weight: imgW
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            img (torch.Tensor[N,C,imgH,imgW]): Input images.
            target (list[torch.Tensor]): Targets.
            target_weight (List[torch.Tensor]): Weights.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.

        Returns:
            dict|tuple: if `return loss` is true, then return losses.                 Otherwise, return predicted poses, boxes, image paths                 and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas, **kwargs)
        return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        features = self.backbone(img)
        outputs = []
        for head_id, head in enumerate(self.heads):
            neck_id = self.head2neck[head_id]
            outputs.append(head(self.necks[neck_id](features)))
        losses = dict()
        for head, output, gt, gt_weight in zip(self.heads, outputs, target, target_weight):
            loss = head.get_loss(output, gt, gt_weight)
            assert len(set(losses.keys()).intersection(set(loss.keys()))) == 0
            losses.update(loss)
            if hasattr(head, 'get_accuracy'):
                acc = head.get_accuracy(output, gt, gt_weight)
                assert len(set(losses.keys()).intersection(set(acc.keys()))) == 0
                losses.update(acc)
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]
        results = {}
        features = self.backbone(img)
        outputs = []
        for head_id, head in enumerate(self.heads):
            neck_id = self.head2neck[head_id]
            if hasattr(head, 'inference_model'):
                head_output = head.inference_model(self.necks[neck_id](features), flip_pairs=None)
            else:
                head_output = head(self.necks[neck_id](features)).detach().cpu().numpy()
            outputs.append(head_output)
        for head, output in zip(self.heads, outputs):
            result = head.decode(img_metas, output, img_size=[img_width, img_height])
            results.update(result)
        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            list[Tensor]: Outputs.
        """
        features = self.backbone(img)
        outputs = []
        for head_id, head in enumerate(self.heads):
            neck_id = self.head2neck[head_id]
            outputs.append(head(self.necks[neck_id](features)))
        return outputs


class SingleCameraBase(metaclass=ABCMeta):
    """Base class for single camera model.

    Args:
        param (dict): Camera parameters

    Methods:
        world_to_camera: Project points from world coordinates to camera
            coordinates
        camera_to_world: Project points from camera coordinates to world
            coordinates
        camera_to_pixel: Project points from camera coordinates to pixel
            coordinates
        world_to_pixel: Project points from world coordinates to pixel
            coordinates
    """

    @abstractmethod
    def __init__(self, param):
        """Load camera parameters and check validity."""

    def world_to_camera(self, X):
        """Project points from world coordinates to camera coordinates."""
        raise NotImplementedError

    def camera_to_world(self, X):
        """Project points from camera coordinates to world coordinates."""
        raise NotImplementedError

    def camera_to_pixel(self, X):
        """Project points from camera coordinates to pixel coordinates."""
        raise NotImplementedError

    def world_to_pixel(self, X):
        """Project points from world coordinates to pixel coordinates."""
        _X = self.world_to_camera(X)
        return self.camera_to_pixel(_X)


class SimpleCameraTorch(SingleCameraBase):
    """Camera model to calculate coordinate transformation with given
    intrinsic/extrinsic camera parameters.

    Notes:
        The keypoint coordinate should be an np.ndarray with a shape of
    [...,J, C] where J is the keypoint number of an instance, and C is
    the coordinate dimension. For example:

        [J, C]: shape of joint coordinates of a person with J joints.
        [N, J, C]: shape of a batch of person joint coordinates.
        [N, T, J, C]: shape of a batch of pose sequences.

    Args:
        param (dict): camera parameters including:
            - R: 3x3, camera rotation matrix (camera-to-world)
            - T: 3x1, camera translation (camera-to-world)
            - K: (optional) 2x3, camera intrinsic matrix
            - k: (optional) nx1, camera radial distortion coefficients
            - p: (optional) mx1, camera tangential distortion coefficients
            - f: (optional) 2x1, camera focal length
            - c: (optional) 2x1, camera center
        if K is not provided, it will be calculated from f and c.

    Methods:
        world_to_camera: Project points from world coordinates to camera
            coordinates
        camera_to_pixel: Project points from camera coordinates to pixel
            coordinates
        world_to_pixel: Project points from world coordinates to pixel
            coordinates
    """

    def __init__(self, param, device):
        self.param = {}
        R = torch.tensor(param['R'], device=device)
        T = torch.tensor(param['T'], device=device)
        assert R.shape == (3, 3)
        assert T.shape == (3, 1)
        self.param['R_c2w'] = R.T
        self.param['T_c2w'] = T.T
        self.param['R_w2c'] = R
        self.param['T_w2c'] = -self.param['T_c2w'] @ self.param['R_w2c']
        if 'K' in param:
            K = torch.tensor(param['K'], device=device)
            assert K.shape == (2, 3)
            self.param['K'] = K.T
            self.param['f'] = torch.tensor([[K[0, 0]], [K[1, 1]]], device=device)
            self.param['c'] = torch.tensor([[K[0, 2]], [K[1, 2]]], device=device)
        elif 'f' in param and 'c' in param:
            f = torch.tensor(param['f'], device=device)
            c = torch.tensor(param['c'], device=device)
            assert f.shape == (2, 1)
            assert c.shape == (2, 1)
            self.param['K'] = torch.cat([torch.diagflat(f), c], dim=-1).T
            self.param['f'] = f
            self.param['c'] = c
        else:
            raise ValueError('Camera intrinsic parameters are missing. Either "K" or "f"&"c" should be provided.')
        if 'k' in param and 'p' in param:
            self.undistortion = True
            self.param['k'] = torch.tensor(param['k'], device=device).view(-1)
            self.param['p'] = torch.tensor(param['p'], device=device).view(-1)
            assert len(self.param['k']) in {3, 6}
            assert len(self.param['p']) == 2
        else:
            self.undistortion = False

    def world_to_camera(self, X):
        assert isinstance(X, torch.Tensor)
        assert X.ndim >= 2 and X.shape[-1] == 3
        return X @ self.param['R_w2c'] + self.param['T_w2c']

    def camera_to_world(self, X):
        assert isinstance(X, torch.Tensor)
        assert X.ndim >= 2 and X.shape[-1] == 3
        return X @ self.param['R_c2w'] + self.param['T_c2w']

    def camera_to_pixel(self, X):
        assert isinstance(X, torch.Tensor)
        assert X.ndim >= 2 and X.shape[-1] == 3
        _X = X / X[..., 2:]
        if self.undistortion:
            k = self.param['k']
            p = self.param['p']
            _X_2d = _X[..., :2]
            r2 = (_X_2d ** 2).sum(-1)
            radial = 1 + sum(ki * r2 ** (i + 1) for i, ki in enumerate(k[:3]))
            if k.size == 6:
                radial /= 1 + sum(ki * r2 ** (i + 1) for i, ki in enumerate(k[3:]))
            tangential = 2 * (p[1] * _X[..., 0] + p[0] * _X[..., 1])
            _X[..., :2] = _X_2d * (radial + tangential)[..., None] + torch.ger(r2, p.flip([0])).reshape(_X_2d.shape)
        return _X @ self.param['K']


def affine_transform_torch(pts, t):
    npts = pts.shape[0]
    pts_homo = torch.cat([pts, torch.ones(npts, 1, device=pts.device)], dim=1)
    out = torch.mm(t, torch.t(pts_homo))
    return torch.t(out[:2, :])


def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)
    return third_pt


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]
    return rotated_pt


def get_affine_transform(center, scale, rot, output_size, shift=(0.0, 0.0), inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2
    scale_tmp = scale * 200.0
    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0.0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


class ProjectLayer(nn.Module):

    def __init__(self, image_size, heatmap_size):
        """Project layer to get voxel feature. Adapted from
        https://github.com/microsoft/voxelpose-
        pytorch/blob/main/lib/models/project_layer.py.

        Args:
            image_size (int or list): input size of the 2D model
            heatmap_size (int or list): output size of the 2D model
        """
        super(ProjectLayer, self).__init__()
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        if isinstance(self.heatmap_size, int):
            self.heatmap_size = [self.heatmap_size, self.heatmap_size]

    def compute_grid(self, box_size, box_center, num_bins, device=None):
        if isinstance(box_size, int) or isinstance(box_size, float):
            box_size = [box_size, box_size, box_size]
        if isinstance(num_bins, int):
            num_bins = [num_bins, num_bins, num_bins]
        grid_1D_x = torch.linspace(-box_size[0] / 2, box_size[0] / 2, num_bins[0], device=device)
        grid_1D_y = torch.linspace(-box_size[1] / 2, box_size[1] / 2, num_bins[1], device=device)
        grid_1D_z = torch.linspace(-box_size[2] / 2, box_size[2] / 2, num_bins[2], device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_1D_x + box_center[0], grid_1D_y + box_center[1], grid_1D_z + box_center[2])
        grid_x = grid_x.contiguous().view(-1, 1)
        grid_y = grid_y.contiguous().view(-1, 1)
        grid_z = grid_z.contiguous().view(-1, 1)
        grid = torch.cat([grid_x, grid_y, grid_z], dim=1)
        return grid

    def get_voxel(self, feature_maps, meta, grid_size, grid_center, cube_size):
        device = feature_maps[0].device
        batch_size = feature_maps[0].shape[0]
        num_channels = feature_maps[0].shape[1]
        num_bins = cube_size[0] * cube_size[1] * cube_size[2]
        n = len(feature_maps)
        cubes = torch.zeros(batch_size, num_channels, 1, num_bins, n, device=device)
        w, h = self.heatmap_size
        grids = torch.zeros(batch_size, num_bins, 3, device=device)
        bounding = torch.zeros(batch_size, 1, 1, num_bins, n, device=device)
        for i in range(batch_size):
            if len(grid_center[0]) == 3 or grid_center[i][3] >= 0:
                if len(grid_center) == 1:
                    grid = self.compute_grid(grid_size, grid_center[0], cube_size, device=device)
                else:
                    grid = self.compute_grid(grid_size, grid_center[i], cube_size, device=device)
                grids[i:i + 1] = grid
                for c in range(n):
                    center = meta[i]['center'][c]
                    scale = meta[i]['scale'][c]
                    width, height = center * 2
                    trans = torch.as_tensor(get_affine_transform(center, scale / 200.0, 0, self.image_size), dtype=torch.float, device=device)
                    cam_param = meta[i]['camera'][c].copy()
                    single_view_camera = SimpleCameraTorch(param=cam_param, device=device)
                    xy = single_view_camera.world_to_pixel(grid)
                    bounding[i, 0, 0, :, c] = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < width) & (xy[:, 1] < height)
                    xy = torch.clamp(xy, -1.0, max(width, height))
                    xy = affine_transform_torch(xy, trans)
                    xy = xy * torch.tensor([w, h], dtype=torch.float, device=device) / torch.tensor(self.image_size, dtype=torch.float, device=device)
                    sample_grid = xy / torch.tensor([w - 1, h - 1], dtype=torch.float, device=device) * 2.0 - 1.0
                    sample_grid = torch.clamp(sample_grid.view(1, 1, num_bins, 2), -1.1, 1.1)
                    cubes[i:i + 1, :, :, :, c] += F.grid_sample(feature_maps[c][i:i + 1, :, :, :], sample_grid, align_corners=True)
        cubes = torch.sum(torch.mul(cubes, bounding), dim=-1) / (torch.sum(bounding, dim=-1) + 1e-06)
        cubes[cubes != cubes] = 0.0
        cubes = cubes.clamp(0.0, 1.0)
        cubes = cubes.view(batch_size, num_channels, cube_size[0], cube_size[1], cube_size[2])
        return cubes, grids

    def forward(self, feature_maps, meta, grid_size, grid_center, cube_size):
        cubes, grids = self.get_voxel(feature_maps, meta, grid_size, grid_center, cube_size)
        return cubes, grids


class DetectAndRegress(BasePose):
    """DetectAndRegress approach for multiview human pose detection.

    Args:
        backbone (ConfigDict): Dictionary to construct the 2D pose detector
        human_detector (ConfigDict): dictionary to construct human detector
        pose_regressor (ConfigDict): dictionary to construct pose regressor
        train_cfg (ConfigDict): Config for training. Default: None.
        test_cfg (ConfigDict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained 2D model. Default: None.
        freeze_2d (bool): Whether to freeze the 2D model in training.
            Default: True.
    """

    def __init__(self, backbone, human_detector, pose_regressor, train_cfg=None, test_cfg=None, pretrained=None, freeze_2d=True):
        super(DetectAndRegress, self).__init__()
        if backbone is not None:
            self.backbone = builder.build_posenet(backbone)
            if self.training and pretrained is not None:
                load_checkpoint(self.backbone, pretrained)
        else:
            self.backbone = None
        self.freeze_2d = freeze_2d
        self.human_detector = builder.MODELS.build(human_detector)
        self.pose_regressor = builder.MODELS.build(pose_regressor)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @staticmethod
    def _freeze(model):
        """Freeze parameters."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Sets the module in training mode.
        Args:
            mode (bool): whether to set training mode (``True``)
                or evaluation mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        super().train(mode)
        if mode and self.freeze_2d and self.backbone is not None:
            self._freeze(self.backbone)
        return self

    def forward(self, img=None, img_metas=None, return_loss=True, targets=None, masks=None, targets_3d=None, input_heatmaps=None, **kwargs):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            feature_maps width: W
            feature_maps height: H
            volume_length: cubeL
            volume_width: cubeW
            volume_height: cubeH

        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            return_loss: Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            targets (list(torch.Tensor[NxKxHxW])):
                Multi-camera target feature_maps of the 2D model.
            masks (list(torch.Tensor[NxHxW])):
                Multi-camera masks of the input to the 2D model.
            targets_3d (torch.Tensor[NxcubeLxcubeWxcubeH]):
                Ground-truth 3D heatmap of human centers.
            input_heatmaps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps when the 2D model is not available.
                 Default: None.
            **kwargs:

        Returns:
            dict: if 'return_loss' is true, then return losses.
              Otherwise, return predicted poses, human centers and sample_id

        """
        if return_loss:
            return self.forward_train(img, img_metas, targets, masks, targets_3d, input_heatmaps)
        else:
            return self.forward_test(img, img_metas, input_heatmaps)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self.forward(**data_batch)
        loss, log_vars = self._parse_losses(losses)
        if 'img' in data_batch:
            batch_size = data_batch['img'][0].shape[0]
        else:
            assert 'input_heatmaps' in data_batch
            batch_size = data_batch['input_heatmaps'][0][0].shape[0]
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=batch_size)
        return outputs

    def forward_train(self, img, img_metas, targets=None, masks=None, targets_3d=None, input_heatmaps=None):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            feature_maps width: W
            feature_maps height: H
            volume_length: cubeL
            volume_width: cubeW
            volume_height: cubeH

        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            targets (list(torch.Tensor[NxKxHxW])):
                Multi-camera target feature_maps of the 2D model.
            masks (list(torch.Tensor[NxHxW])):
                Multi-camera masks of the input to the 2D model.
            targets_3d (torch.Tensor[NxcubeLxcubeWxcubeH]):
                Ground-truth 3D heatmap of human centers.
            input_heatmaps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps when the 2D model is not available.
                 Default: None.

        Returns:
            dict: losses.

        """
        if self.backbone is None:
            assert input_heatmaps is not None
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list)
            for img_ in img:
                feature_maps.append(self.backbone.forward_dummy(img_)[0])
        losses = dict()
        human_candidates, human_loss = self.human_detector.forward_train(None, img_metas, feature_maps, targets_3d, return_preds=True)
        losses.update(human_loss)
        pose_loss = self.pose_regressor(None, img_metas, return_loss=True, feature_maps=feature_maps, human_candidates=human_candidates)
        losses.update(pose_loss)
        if not self.freeze_2d:
            losses_2d = {}
            heatmaps_tensor = torch.cat(feature_maps, dim=0)
            targets_tensor = torch.cat(targets, dim=0)
            masks_tensor = torch.cat(masks, dim=0)
            losses_2d_ = self.backbone.get_loss(heatmaps_tensor, targets_tensor, masks_tensor)
            for k, v in losses_2d_.items():
                losses_2d[k + '_2d'] = v
            losses.update(losses_2d)
        return losses

    def forward_test(self, img, img_metas, input_heatmaps=None):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            feature_maps width: W
            feature_maps height: H
            volume_length: cubeL
            volume_width: cubeW
            volume_height: cubeH

        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            input_heatmaps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps when the 2D model is not available.
                 Default: None.

        Returns:
            dict: predicted poses, human centers and sample_id

        """
        if self.backbone is None:
            assert input_heatmaps is not None
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list)
            for img_ in img:
                feature_maps.append(self.backbone.forward_dummy(img_)[0])
        human_candidates = self.human_detector.forward_test(None, img_metas, feature_maps)
        human_poses = self.pose_regressor(None, img_metas, return_loss=False, feature_maps=feature_maps, human_candidates=human_candidates)
        result = {}
        result['pose_3d'] = human_poses.cpu().numpy()
        result['human_detection_3d'] = human_candidates.cpu().numpy()
        result['sample_id'] = [img_meta['sample_id'] for img_meta in img_metas]
        return result

    def show_result(self, **kwargs):
        """Visualize the results."""
        raise NotImplementedError

    def forward_dummy(self, img, input_heatmaps=None, num_candidates=5):
        """Used for computing network FLOPs."""
        if self.backbone is None:
            assert input_heatmaps is not None
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list)
            for img_ in img:
                feature_maps.append(self.backbone.forward_dummy(img_)[0])
        _ = self.human_detector.forward_dummy(feature_maps)
        _ = self.pose_regressor.forward_dummy(feature_maps, num_candidates)


class VoxelSinglePose(BasePose):
    """VoxelPose Please refer to the `paper <https://arxiv.org/abs/2004.06239>`
    for details.

    Args:
        image_size (list): input size of the 2D model.
        heatmap_size (list): output size of the 2D model.
        sub_space_size (list): Size of the cuboid human proposal.
        sub_cube_size (list): Size of the input volume to the pose net.
        pose_net (ConfigDict): Dictionary to construct the pose net.
        pose_head (ConfigDict): Dictionary to construct the pose head.
        train_cfg (ConfigDict): Config for training. Default: None.
        test_cfg (ConfigDict): Config for testing. Default: None.
    """

    def __init__(self, image_size, heatmap_size, sub_space_size, sub_cube_size, num_joints, pose_net, pose_head, train_cfg=None, test_cfg=None):
        super(VoxelSinglePose, self).__init__()
        self.project_layer = ProjectLayer(image_size, heatmap_size)
        self.pose_net = builder.build_backbone(pose_net)
        self.pose_head = builder.build_head(pose_head)
        self.sub_space_size = sub_space_size
        self.sub_cube_size = sub_cube_size
        self.num_joints = num_joints
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(self, img, img_metas, return_loss=True, feature_maps=None, human_candidates=None, **kwargs):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            feature_maps width: W
            feature_maps height: H
            volume_length: cubeL
            volume_width: cubeW
            volume_height: cubeH

        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            feature_maps (list(torch.Tensor[NxCxHxW])):
                Multi-camera input feature_maps.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            human_candidates (torch.Tensor[NxPx5]):
                Human candidates.
            return_loss: Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.

        """
        if return_loss:
            return self.forward_train(img, img_metas, feature_maps, human_candidates)
        else:
            return self.forward_test(img, img_metas, feature_maps, human_candidates)

    def forward_train(self, img, img_metas, feature_maps=None, human_candidates=None, return_preds=False, **kwargs):
        """Defines the computation performed at training.
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            feature_maps width: W
            feature_maps height: H
            volume_length: cubeL
            volume_width: cubeW
            volume_height: cubeH

        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            feature_maps (list(torch.Tensor[NxCxHxW])):
                Multi-camera input feature_maps.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            human_candidates (torch.Tensor[NxPx5]):
                Human candidates.
            return_preds (bool): Whether to return prediction results

        Returns:
            dict: losses.

        """
        batch_size, num_candidates, _ = human_candidates.shape
        pred = human_candidates.new_zeros(batch_size, num_candidates, self.num_joints, 5)
        pred[:, :, :, 3:] = human_candidates[:, :, None, 3:]
        device = feature_maps[0].device
        gt_3d = torch.stack([torch.tensor(img_meta['joints_3d'], device=device) for img_meta in img_metas])
        gt_3d_vis = torch.stack([torch.tensor(img_meta['joints_3d_visible'], device=device) for img_meta in img_metas])
        valid_preds = []
        valid_targets = []
        valid_weights = []
        for n in range(num_candidates):
            index = pred[:, n, 0, 3] >= 0
            num_valid = index.sum()
            if num_valid > 0:
                pose_input_cube, coordinates = self.project_layer(feature_maps, img_metas, self.sub_space_size, human_candidates[:, n, :3], self.sub_cube_size)
                pose_heatmaps_3d = self.pose_net(pose_input_cube)
                pose_3d = self.pose_head(pose_heatmaps_3d[index], coordinates[index])
                pred[index, n, :, 0:3] = pose_3d.detach()
                valid_targets.append(gt_3d[index, pred[index, n, 0, 3].long()])
                valid_weights.append(gt_3d_vis[index, pred[index, n, 0, 3].long(), :, 0:1].float())
                valid_preds.append(pose_3d)
        losses = dict()
        if len(valid_preds) > 0:
            valid_targets = torch.cat(valid_targets, dim=0)
            valid_weights = torch.cat(valid_weights, dim=0)
            valid_preds = torch.cat(valid_preds, dim=0)
            losses.update(self.pose_head.get_loss(valid_preds, valid_targets, valid_weights))
        else:
            pose_input_cube = feature_maps[0].new_zeros(batch_size, self.num_joints, *self.sub_cube_size)
            coordinates = feature_maps[0].new_zeros(batch_size, *self.sub_cube_size, 3).view(batch_size, -1, 3)
            pseudo_targets = feature_maps[0].new_zeros(batch_size, self.num_joints, 3)
            pseudo_weights = feature_maps[0].new_zeros(batch_size, self.num_joints, 1)
            pose_heatmaps_3d = self.pose_net(pose_input_cube)
            pose_3d = self.pose_head(pose_heatmaps_3d, coordinates)
            losses.update(self.pose_head.get_loss(pose_3d, pseudo_targets, pseudo_weights))
        if return_preds:
            return pred, losses
        else:
            return losses

    def forward_test(self, img, img_metas, feature_maps=None, human_candidates=None, **kwargs):
        """Defines the computation performed at training.
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            feature_maps width: W
            feature_maps height: H
            volume_length: cubeL
            volume_width: cubeW
            volume_height: cubeH

        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            feature_maps (list(torch.Tensor[NxCxHxW])):
                Multi-camera input feature_maps.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            human_candidates (torch.Tensor[NxPx5]):
                Human candidates.

        Returns:
            dict: predicted poses, human centers and sample_id

        """
        batch_size, num_candidates, _ = human_candidates.shape
        pred = human_candidates.new_zeros(batch_size, num_candidates, self.num_joints, 5)
        pred[:, :, :, 3:] = human_candidates[:, :, None, 3:]
        for n in range(num_candidates):
            index = pred[:, n, 0, 3] >= 0
            num_valid = index.sum()
            if num_valid > 0:
                pose_input_cube, coordinates = self.project_layer(feature_maps, img_metas, self.sub_space_size, human_candidates[:, n, :3], self.sub_cube_size)
                pose_heatmaps_3d = self.pose_net(pose_input_cube)
                pose_3d = self.pose_head(pose_heatmaps_3d[index], coordinates[index])
                pred[index, n, :, 0:3] = pose_3d.detach()
        return pred

    def show_result(self, **kwargs):
        """Visualize the results."""
        raise NotImplementedError

    def forward_dummy(self, feature_maps, num_candidates=5):
        """Used for computing network FLOPs."""
        batch_size, num_channels = feature_maps[0].shape
        pose_input_cube = feature_maps[0].new_zeros(batch_size, num_channels, *self.sub_cube_size)
        for n in range(num_candidates):
            _ = self.pose_net(pose_input_cube)


class VoxelCenterDetector(BasePose):
    """Detect human center by 3D CNN on voxels.

    Please refer to the
    `paper <https://arxiv.org/abs/2004.06239>` for details.
    Args:
        image_size (list): input size of the 2D model.
        heatmap_size (list): output size of the 2D model.
        space_size (list): Size of the 3D space.
        cube_size (list): Size of the input volume to the 3D CNN.
        space_center (list): Coordinate of the center of the 3D space.
        center_net (ConfigDict): Dictionary to construct the center net.
        center_head (ConfigDict): Dictionary to construct the center head.
        train_cfg (ConfigDict): Config for training. Default: None.
        test_cfg (ConfigDict): Config for testing. Default: None.
    """

    def __init__(self, image_size, heatmap_size, space_size, cube_size, space_center, center_net, center_head, train_cfg=None, test_cfg=None):
        super(VoxelCenterDetector, self).__init__()
        self.project_layer = ProjectLayer(image_size, heatmap_size)
        self.center_net = builder.build_backbone(center_net)
        self.center_head = builder.build_head(center_head)
        self.space_size = space_size
        self.cube_size = cube_size
        self.space_center = space_center
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def assign2gt(self, center_candidates, gt_centers, gt_num_persons):
        """"Assign gt id to each valid human center candidate."""
        det_centers = center_candidates[..., :3]
        batch_size = center_candidates.shape[0]
        cand_num = center_candidates.shape[1]
        cand2gt = torch.zeros(batch_size, cand_num)
        for i in range(batch_size):
            cand = det_centers[i].view(cand_num, 1, -1)
            gt = gt_centers[None, i, :gt_num_persons[i]]
            dist = torch.sqrt(torch.sum((cand - gt) ** 2, dim=-1))
            min_dist, min_gt = torch.min(dist, dim=-1)
            cand2gt[i] = min_gt
            cand2gt[i][min_dist > self.train_cfg['dist_threshold']] = -1.0
        center_candidates[:, :, 3] = cand2gt
        return center_candidates

    def forward(self, img, img_metas, return_loss=True, feature_maps=None, targets_3d=None):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            heatmaps width: W
            heatmaps height: H
        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            return_loss: Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            targets_3d (torch.Tensor[NxcubeLxcubeWxcubeH]):
                Ground-truth 3D heatmap of human centers.
            feature_maps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps.
        Returns:
            dict: if 'return_loss' is true, then return losses.
                Otherwise, return predicted poses
        """
        if return_loss:
            return self.forward_train(img, img_metas, feature_maps, targets_3d)
        else:
            return self.forward_test(img, img_metas, feature_maps)

    def forward_train(self, img, img_metas, feature_maps=None, targets_3d=None, return_preds=False):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            heatmaps width: W
            heatmaps height: H
        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            targets_3d (torch.Tensor[NxcubeLxcubeWxcubeH]):
                Ground-truth 3D heatmap of human centers.
            feature_maps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps.
            return_preds (bool): Whether to return prediction results
        Returns:
            dict: if 'return_pred' is true, then return losses
                and human centers. Otherwise, return losses only
        """
        initial_cubes, _ = self.project_layer(feature_maps, img_metas, self.space_size, [self.space_center], self.cube_size)
        center_heatmaps_3d = self.center_net(initial_cubes)
        center_heatmaps_3d = center_heatmaps_3d.squeeze(1)
        center_candidates = self.center_head(center_heatmaps_3d)
        device = center_candidates.device
        gt_centers = torch.stack([torch.tensor(img_meta['roots_3d'], device=device) for img_meta in img_metas])
        gt_num_persons = torch.stack([torch.tensor(img_meta['num_persons'], device=device) for img_meta in img_metas])
        center_candidates = self.assign2gt(center_candidates, gt_centers, gt_num_persons)
        losses = dict()
        losses.update(self.center_head.get_loss(center_heatmaps_3d, targets_3d))
        if return_preds:
            return center_candidates, losses
        else:
            return losses

    def forward_test(self, img, img_metas, feature_maps=None):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            heatmaps width: W
            heatmaps height: H
        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            feature_maps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps.
        Returns:
            human centers
        """
        initial_cubes, _ = self.project_layer(feature_maps, img_metas, self.space_size, [self.space_center], self.cube_size)
        center_heatmaps_3d = self.center_net(initial_cubes)
        center_heatmaps_3d = center_heatmaps_3d.squeeze(1)
        center_candidates = self.center_head(center_heatmaps_3d)
        center_candidates[..., 3] = (center_candidates[..., 4] > self.test_cfg['center_threshold']).float() - 1.0
        return center_candidates

    def show_result(self, **kwargs):
        """Visualize the results."""
        raise NotImplementedError

    def forward_dummy(self, feature_maps):
        """Used for computing network FLOPs."""
        batch_size, num_channels, _, _ = feature_maps[0].shape
        initial_cubes = feature_maps[0].new_zeros(batch_size, num_channels, *self.cube_size)
        _ = self.center_net(initial_cubes)


def cast_tensor_type(inputs, src_type, dst_type):
    """Recursively convert Tensor in inputs from src_type to dst_type.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype): Source type.
        dst_type (torch.dtype): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({k: cast_tensor_type(v, src_type, dst_type) for k, v in inputs.items()})
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(cast_tensor_type(item, src_type, dst_type) for item in inputs)
    return inputs


def auto_fp16(apply_to=None, out_fp32=False):
    """Decorator to enable fp16 training automatically.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If inputs arguments are fp32 tensors, they will
    be converted to fp16 automatically. Arguments other than fp32 tensors are
    ignored.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp32 (bool): Whether to convert the output back to fp32.

    Example:

        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp16
        >>>     @auto_fp16()
        >>>     def forward(self, x, y):
        >>>         pass

        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp16
        >>>     @auto_fp16(apply_to=('pred', ))
        >>>     def do_something(self, pred, others):
        >>>         pass
    """
    warnings.warn('auto_fp16 in mmpose will be deprecated in the next release.Please use mmcv.runner.auto_fp16 instead (mmcv>=1.3.1).', DeprecationWarning)

    def auto_fp16_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@auto_fp16 can only be used to decorate the method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            args_info = getfullargspec(old_func)
            args_to_cast = args_info.args if apply_to is None else apply_to
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(cast_tensor_type(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(arg_value, torch.float, torch.half)
                    else:
                        new_kwargs[arg_name] = arg_value
            output = old_func(*new_args, **new_kwargs)
            if out_fp32:
                output = cast_tensor_type(output, torch.half, torch.float)
            return output
        return new_func
    return auto_fp16_wrapper


def imshow_bboxes(img, bboxes, labels=None, colors='green', text_color='white', thickness=1, font_scale=0.5, show=True, win_name='', wait_time=0, out_file=None):
    """Draw bboxes with labels (optional) on an image. This is a wrapper of
    mmcv.imshow_bboxes.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): ndarray of shape (k, 4), each row is a bbox in
            format [x1, y1, x2, y2].
        labels (str or list[str], optional): labels of each bbox.
        colors (list[str or tuple or :obj:`Color`]): A list of colors.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    bboxes = np.split(bboxes, bboxes.shape[0], axis=0) if bboxes.shape[0] > 0 else []
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [mmcv.color_val(c) for c in colors]
    assert len(bboxes) == len(colors)
    img = mmcv.imshow_bboxes(img, bboxes, colors, top_k=-1, thickness=thickness, show=False, out_file=None)
    if labels is not None:
        if not isinstance(labels, list):
            labels = [labels for _ in range(len(bboxes))]
        assert len(labels) == len(bboxes)
        for bbox, label, color in zip(bboxes, labels, colors):
            if label is None:
                continue
            bbox_int = bbox[0, :4].astype(np.int32)
            text_size, text_baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
            text_x1 = bbox_int[0]
            text_y1 = max(0, bbox_int[1] - text_size[1] - text_baseline)
            text_x2 = bbox_int[0] + text_size[0]
            text_y2 = text_y1 + text_size[1] + text_baseline
            cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), color, cv2.FILLED)
            cv2.putText(img, label, (text_x1, text_y2 - text_baseline), cv2.FONT_HERSHEY_DUPLEX, font_scale, mmcv.color_val(text_color), thickness)
    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
    return img


def imshow_keypoints_3d(pose_result, img=None, skeleton=None, pose_kpt_color=None, pose_link_color=None, vis_height=400, kpt_score_thr=0.3, num_instances=-1, *, axis_azimuth=70, axis_limit=1.7, axis_dist=10.0, axis_elev=15.0):
    """Draw 3D keypoints and links in 3D coordinates.

    Args:
        pose_result (list[dict]): 3D pose results containing:
            - "keypoints_3d" ([K,4]): 3D keypoints
            - "title" (str): Optional. A string to specify the title of the
                visualization of this pose result
        img (str|np.ndarray): Opptional. The image or image path to show input
            image and/or 2D pose. Note that the image should be given in BGR
            channel order.
        skeleton (list of [idx_i,idx_j]): Skeleton described by a list of
            links, each is a pair of joint indices.
        pose_kpt_color (np.ndarray[Nx3]`): Color of N keypoints. If None, do
            not nddraw keypoints.
        pose_link_color (np.array[Mx3]): Color of M links. If None, do not
            draw links.
        vis_height (int): The image height of the visualization. The width
                will be N*vis_height depending on the number of visualized
                items.
        kpt_score_thr (float): Minimum score of keypoints to be shown.
            Default: 0.3.
        num_instances (int): Number of instances to be shown in 3D. If smaller
            than 0, all the instances in the pose_result will be shown.
            Otherwise, pad or truncate the pose_result to a length of
            num_instances.
        axis_azimuth (float): axis azimuth angle for 3D visualizations.
        axis_dist (float): axis distance for 3D visualizations.
        axis_elev (float): axis elevation view angle for 3D visualizations.
        axis_limit (float): The axis limit to visualize 3d pose. The xyz
            range will be set as:
            - x: [x_c - axis_limit/2, x_c + axis_limit/2]
            - y: [y_c - axis_limit/2, y_c + axis_limit/2]
            - z: [0, axis_limit]
            Where x_c, y_c is the mean value of x and y coordinates
        figsize: (float): figure size in inch.
    """
    show_img = img is not None
    if num_instances < 0:
        num_instances = len(pose_result)
    elif len(pose_result) > num_instances:
        pose_result = pose_result[:num_instances]
    elif len(pose_result) < num_instances:
        pose_result += [dict()] * (num_instances - len(pose_result))
    num_axis = num_instances + 1 if show_img else num_instances
    plt.ioff()
    fig = plt.figure(figsize=(vis_height * num_axis * 0.01, vis_height * 0.01))
    if show_img:
        img = mmcv.imread(img, channel_order='bgr')
        img = mmcv.bgr2rgb(img)
        img = mmcv.imrescale(img, scale=vis_height / img.shape[0])
        ax_img = fig.add_subplot(1, num_axis, 1)
        ax_img.get_xaxis().set_visible(False)
        ax_img.get_yaxis().set_visible(False)
        ax_img.set_axis_off()
        ax_img.set_title('Input')
        ax_img.imshow(img, aspect='equal')
    for idx, res in enumerate(pose_result):
        dummy = len(res) == 0
        kpts = np.zeros((1, 3)) if dummy else res['keypoints_3d']
        if kpts.shape[1] == 3:
            kpts = np.concatenate([kpts, np.ones((kpts.shape[0], 1))], axis=1)
        valid = kpts[:, 3] >= kpt_score_thr
        ax_idx = idx + 2 if show_img else idx + 1
        ax = fig.add_subplot(1, num_axis, ax_idx, projection='3d')
        ax.view_init(elev=axis_elev, azim=axis_azimuth)
        x_c = np.mean(kpts[valid, 0]) if sum(valid) > 0 else 0
        y_c = np.mean(kpts[valid, 1]) if sum(valid) > 0 else 0
        ax.set_xlim3d([x_c - axis_limit / 2, x_c + axis_limit / 2])
        ax.set_ylim3d([y_c - axis_limit / 2, y_c + axis_limit / 2])
        ax.set_zlim3d([0, axis_limit])
        ax.set_aspect('auto')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = axis_dist
        if not dummy and pose_kpt_color is not None:
            pose_kpt_color = np.array(pose_kpt_color)
            assert len(pose_kpt_color) == len(kpts)
            x_3d, y_3d, z_3d = np.split(kpts[:, :3], [1, 2], axis=1)
            _color = pose_kpt_color[..., ::-1] / 255.0
            ax.scatter(x_3d[valid], y_3d[valid], z_3d[valid], marker='o', color=_color[valid])
        if not dummy and skeleton is not None and pose_link_color is not None:
            pose_link_color = np.array(pose_link_color)
            assert len(pose_link_color) == len(skeleton)
            for link, link_color in zip(skeleton, pose_link_color):
                link_indices = [_i for _i in link]
                xs_3d = kpts[link_indices, 0]
                ys_3d = kpts[link_indices, 1]
                zs_3d = kpts[link_indices, 2]
                kpt_score = kpts[link_indices, 3]
                if kpt_score.min() > kpt_score_thr:
                    _color = link_color[::-1] / 255.0
                    ax.plot(xs_3d, ys_3d, zs_3d, color=_color, zdir='z')
        if 'title' in res:
            ax.set_title(res['title'])
    fig.tight_layout()
    fig.canvas.draw()
    img_w, img_h = fig.canvas.get_width_height()
    img_vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(img_h, img_w, -1)
    img_vis = mmcv.rgb2bgr(img_vis)
    plt.close(fig)
    return img_vis


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


class AEHigherResolutionHead(nn.Module):
    """Associative embedding with higher resolution head. paper ref: Bowen
    Cheng et al. "HigherHRNet: Scale-Aware Representation Learning for Bottom-
    Up Human Pose Estimation".

    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joints
        tag_per_joint (bool): If tag_per_joint is True,
            the dimension of tags equals to num_joints,
            else the dimension of tags is 1. Default: True
        extra (dict): Configs for extra conv layers. Default: None
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        cat_output (list[bool]): Option to concat outputs.
        with_ae_loss (list[bool]): Option to use ae loss.
        loss_keypoint (dict): Config for loss. Default: None.
    """

    def __init__(self, in_channels, num_joints, tag_per_joint=True, extra=None, num_deconv_layers=1, num_deconv_filters=(32,), num_deconv_kernels=(4,), num_basic_blocks=4, cat_output=None, with_ae_loss=None, loss_keypoint=None):
        super().__init__()
        self.loss = build_loss(loss_keypoint)
        dim_tag = num_joints if tag_per_joint else 1
        self.num_deconvs = num_deconv_layers
        self.cat_output = cat_output
        final_layer_output_channels = []
        if with_ae_loss[0]:
            out_channels = num_joints + dim_tag
        else:
            out_channels = num_joints
        final_layer_output_channels.append(out_channels)
        for i in range(num_deconv_layers):
            if with_ae_loss[i + 1]:
                out_channels = num_joints + dim_tag
            else:
                out_channels = num_joints
            final_layer_output_channels.append(out_channels)
        deconv_layer_output_channels = []
        for i in range(num_deconv_layers):
            if with_ae_loss[i]:
                out_channels = num_joints + dim_tag
            else:
                out_channels = num_joints
            deconv_layer_output_channels.append(out_channels)
        self.final_layers = self._make_final_layers(in_channels, final_layer_output_channels, extra, num_deconv_layers, num_deconv_filters)
        self.deconv_layers = self._make_deconv_layers(in_channels, deconv_layer_output_channels, num_deconv_layers, num_deconv_filters, num_deconv_kernels, num_basic_blocks, cat_output)

    @staticmethod
    def _make_final_layers(in_channels, final_layer_output_channels, extra, num_deconv_layers, num_deconv_filters):
        """Make final layers."""
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            else:
                padding = 0
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0
        final_layers = []
        final_layers.append(build_conv_layer(cfg=dict(type='Conv2d'), in_channels=in_channels, out_channels=final_layer_output_channels[0], kernel_size=kernel_size, stride=1, padding=padding))
        for i in range(num_deconv_layers):
            in_channels = num_deconv_filters[i]
            final_layers.append(build_conv_layer(cfg=dict(type='Conv2d'), in_channels=in_channels, out_channels=final_layer_output_channels[i + 1], kernel_size=kernel_size, stride=1, padding=padding))
        return nn.ModuleList(final_layers)

    def _make_deconv_layers(self, in_channels, deconv_layer_output_channels, num_deconv_layers, num_deconv_filters, num_deconv_kernels, num_basic_blocks, cat_output):
        """Make deconv layers."""
        deconv_layers = []
        for i in range(num_deconv_layers):
            if cat_output[i]:
                in_channels += deconv_layer_output_channels[i]
            planes = num_deconv_filters[i]
            deconv_kernel, padding, output_padding = self._get_deconv_cfg(num_deconv_kernels[i])
            layers = []
            layers.append(nn.Sequential(build_upsample_layer(dict(type='deconv'), in_channels=in_channels, out_channels=planes, kernel_size=deconv_kernel, stride=2, padding=padding, output_padding=output_padding, bias=False), nn.BatchNorm2d(planes, momentum=0.1), nn.ReLU(inplace=True)))
            for _ in range(num_basic_blocks):
                layers.append(nn.Sequential(BasicBlock(planes, planes)))
            deconv_layers.append(nn.Sequential(*layers))
            in_channels = planes
        return nn.ModuleList(deconv_layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')
        return deconv_kernel, padding, output_padding

    def get_loss(self, outputs, targets, masks, joints):
        """Calculate bottom-up keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_outputs: O
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            outputs (list(torch.Tensor[N,K,H,W])): Multi-scale output heatmaps.
            targets (List(torch.Tensor[N,K,H,W])): Multi-scale target heatmaps.
            masks (List(torch.Tensor[N,H,W])): Masks of multi-scale target
                heatmaps
            joints (List(torch.Tensor[N,M,K,2])): Joints of multi-scale target
                heatmaps for ae loss
        """
        losses = dict()
        heatmaps_losses, push_losses, pull_losses = self.loss(outputs, targets, masks, joints)
        for idx in range(len(targets)):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                if 'heatmap_loss' not in losses:
                    losses['heatmap_loss'] = heatmaps_loss
                else:
                    losses['heatmap_loss'] += heatmaps_loss
            if push_losses[idx] is not None:
                push_loss = push_losses[idx].mean(dim=0)
                if 'push_loss' not in losses:
                    losses['push_loss'] = push_loss
                else:
                    losses['push_loss'] += push_loss
            if pull_losses[idx] is not None:
                pull_loss = pull_losses[idx].mean(dim=0)
                if 'pull_loss' not in losses:
                    losses['pull_loss'] = pull_loss
                else:
                    losses['pull_loss'] += pull_loss
        return losses

    def forward(self, x):
        """Forward function."""
        if isinstance(x, list):
            x = x[0]
        final_outputs = []
        y = self.final_layers[0](x)
        final_outputs.append(y)
        for i in range(self.num_deconvs):
            if self.cat_output[i]:
                x = torch.cat((x, y), 1)
            x = self.deconv_layers[i](x)
            y = self.final_layers[i + 1](x)
            final_outputs.append(y)
        return final_outputs

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for _, m in self.final_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)


class AEMultiStageHead(nn.Module):
    """Associative embedding multi-stage head.
    paper ref: Alejandro Newell et al. "Associative
    Embedding: End-to-end Learning for Joint Detection
    and Grouping"

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        loss_keypoint (dict): Config for loss. Default: None.
    """

    def __init__(self, in_channels, out_channels, num_stages=1, num_deconv_layers=3, num_deconv_filters=(256, 256, 256), num_deconv_kernels=(4, 4, 4), extra=None, loss_keypoint=None):
        super().__init__()
        self.loss = build_loss(loss_keypoint)
        self.in_channels = in_channels
        self.num_stages = num_stages
        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')
        self.multi_deconv_layers = nn.ModuleList([])
        for _ in range(self.num_stages):
            if num_deconv_layers > 0:
                deconv_layers = self._make_deconv_layer(num_deconv_layers, num_deconv_filters, num_deconv_kernels)
            elif num_deconv_layers == 0:
                deconv_layers = nn.Identity()
            else:
                raise ValueError(f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
            self.multi_deconv_layers.append(deconv_layers)
        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0
        self.multi_final_layers = nn.ModuleList([])
        for i in range(self.num_stages):
            if identity_final_layer:
                final_layer = nn.Identity()
            else:
                final_layer = build_conv_layer(cfg=dict(type='Conv2d'), in_channels=num_deconv_filters[-1] if num_deconv_layers > 0 else in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)
            self.multi_final_layers.append(final_layer)

    def get_loss(self, output, targets, masks, joints):
        """Calculate bottom-up keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (List(torch.Tensor[NxKxHxW])): Output heatmaps.
            targets(List(List(torch.Tensor[NxKxHxW]))):
                Multi-stage and multi-scale target heatmaps.
            masks(List(List(torch.Tensor[NxHxW]))):
                Masks of multi-stage and multi-scale target heatmaps
            joints(List(List(torch.Tensor[NxMxKx2]))):
                Joints of multi-stage multi-scale target heatmaps for ae loss
        """
        losses = dict()
        targets = [target for _targets in targets for target in _targets]
        masks = [mask for _masks in masks for mask in _masks]
        joints = [joint for _joints in joints for joint in _joints]
        heatmaps_losses, push_losses, pull_losses = self.loss(output, targets, masks, joints)
        for idx in range(len(targets)):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                if 'heatmap_loss' not in losses:
                    losses['heatmap_loss'] = heatmaps_loss
                else:
                    losses['heatmap_loss'] += heatmaps_loss
            if push_losses[idx] is not None:
                push_loss = push_losses[idx].mean(dim=0)
                if 'push_loss' not in losses:
                    losses['push_loss'] = push_loss
                else:
                    losses['push_loss'] += push_loss
            if pull_losses[idx] is not None:
                pull_loss = pull_losses[idx].mean(dim=0)
                if 'pull_loss' not in losses:
                    losses['pull_loss'] = pull_loss
                else:
                    losses['pull_loss'] += pull_loss
        return losses

    def forward(self, x):
        """Forward function.

        Returns:
            out (list[Tensor]): a list of heatmaps from multiple stages.
        """
        out = []
        assert isinstance(x, list)
        for i in range(self.num_stages):
            y = self.multi_deconv_layers[i](x[i])
            y = self.multi_final_layers[i](y)
            out.append(y)
        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) != length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) != length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.append(build_upsample_layer(dict(type='deconv'), in_channels=self.in_channels, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes
        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')
        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.multi_deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.multi_final_layers.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1) and (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1):
                    warnings.warn(f'When align_corners={align_corners}, the output would more aligned if input size {input_h, input_w} is `x+1` and out size {output_h, output_w} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class DeconvHead(nn.Module):
    """Simple deconv head.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for loss. Default: None.
    """

    def __init__(self, in_channels=3, out_channels=17, num_deconv_layers=3, num_deconv_filters=(256, 256, 256), num_deconv_kernels=(4, 4, 4), extra=None, in_index=0, input_transform=None, align_corners=False, loss_keypoint=None):
        super().__init__()
        self.in_channels = in_channels
        self.loss = build_loss(loss_keypoint)
        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners
        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')
        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(num_deconv_layers, num_deconv_filters, num_deconv_kernels)
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0
        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[-1] if num_deconv_layers > 0 else self.in_channels
            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels', [1] * num_conv_layers)
                for i in range(num_conv_layers):
                    layers.append(build_conv_layer(dict(type='Conv2d'), in_channels=conv_channels, out_channels=conv_channels, kernel_size=num_conv_kernels[i], stride=1, padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(nn.ReLU(inplace=True))
            layers.append(build_conv_layer(cfg=dict(type='Conv2d'), in_channels=conv_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding))
            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [resize(input=x, size=inputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners) for x in inputs]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) != length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) != length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.append(build_upsample_layer(dict(type='deconv'), in_channels=self.in_channels, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes
        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')
        return deconv_kernel, padding, output_padding

    def get_loss(self, outputs, targets, masks):
        """Calculate bottom-up masked mse loss.

        Note:
            - batch_size: N
            - num_channels: C
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            outputs (List(torch.Tensor[N,C,H,W])): Multi-scale outputs.
            targets (List(torch.Tensor[N,C,H,W])): Multi-scale targets.
            masks (List(torch.Tensor[N,H,W])): Masks of multi-scale targets.
        """
        losses = dict()
        for idx in range(len(targets)):
            if 'loss' not in losses:
                losses['loss'] = self.loss(outputs[idx], targets[idx], masks[idx])
            else:
                losses['loss'] += self.loss(outputs[idx], targets[idx], masks[idx])
        return losses

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        final_outputs = []
        x = self.deconv_layers(x)
        y = self.final_layer(x)
        final_outputs.append(y)
        return final_outputs

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)


def fliplr_regression(regression, flip_pairs, center_mode='static', center_x=0.5, center_index=0):
    """Flip human joints horizontally.

    Note:
        - batch_size: N
        - num_keypoint: K

    Args:
        regression (np.ndarray([..., K, C])): Coordinates of keypoints, where K
            is the joint number and C is the dimension. Example shapes are:

            - [N, K, C]: a batch of keypoints where N is the batch size.
            - [N, T, K, C]: a batch of pose sequences, where T is the frame
                number.
        flip_pairs (list[tuple()]): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).
        center_mode (str): The mode to set the center location on the x-axis
            to flip around. Options are:

            - static: use a static x value (see center_x also)
            - root: use a root joint (see center_index also)
        center_x (float): Set the x-axis location of the flip center. Only used
            when center_mode=static.
        center_index (int): Set the index of the root joint, whose x location
            will be used as the flip center. Only used when center_mode=root.

    Returns:
        np.ndarray([..., K, C]): Flipped joints.
    """
    assert regression.ndim >= 2, f'Invalid pose shape {regression.shape}'
    allowed_center_mode = {'static', 'root'}
    assert center_mode in allowed_center_mode, f'Get invalid center_mode {center_mode}, allowed choices are {allowed_center_mode}'
    if center_mode == 'static':
        x_c = center_x
    elif center_mode == 'root':
        assert regression.shape[-2] > center_index
        x_c = regression[..., center_index:center_index + 1, 0]
    regression_flipped = regression.copy()
    for left, right in flip_pairs:
        regression_flipped[..., left, :] = regression[..., right, :]
        regression_flipped[..., right, :] = regression[..., left, :]
    regression_flipped[..., 0] = x_c * 2 - regression_flipped[..., 0]
    return regression_flipped


def _calc_distances(preds, targets, mask, normalize):
    """Calculate the normalized distances between preds and target.

    Note:
        batch_size: N
        num_keypoints: K
        dimension of keypoints: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        targets (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize (np.ndarray[N, D]): Typical value is heatmap_size

    Returns:
        np.ndarray[K, N]: The normalized distances.             If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    _mask = mask.copy()
    _mask[np.where((normalize == 0).sum(1))[0], :] = False
    distances = np.full((N, K), -1, dtype=np.float32)
    normalize[np.where(normalize <= 0)] = 1000000.0
    distances[_mask] = np.linalg.norm(((preds - targets) / normalize[:, None, :])[_mask], axis=-1)
    return distances.T


def _distance_acc(distances, thr=0.5):
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        batch_size: N
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


def keypoint_pck_accuracy(pred, gt, mask, thr, normalize):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - batch_size: N
        - num_keypoints: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, normalize)
    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0
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
    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    return target_coords


def keypoints_from_regression(regression_preds, center, scale, img_size):
    """Get final keypoint predictions from regression vectors and transform
    them back to the image.

    Note:
        - batch_size: N
        - num_keypoints: K

    Args:
        regression_preds (np.ndarray[N, K, 2]): model prediction.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        img_size (list(img_width, img_height)): model input image size.

    Returns:
        tuple:

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    N, K, _ = regression_preds.shape
    preds, maxvals = regression_preds, np.ones((N, K, 1), dtype=np.float32)
    preds = preds * img_size
    for i in range(N):
        preds[i] = transform_preds(preds[i], center[i], scale[i], img_size)
    return preds, maxvals


class DeepposeRegressionHead(nn.Module):
    """Deeppose regression head with fully connected layers.

    "DeepPose: Human Pose Estimation via Deep Neural Networks".

    Args:
        in_channels (int): Number of input channels
        num_joints (int): Number of joints
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self, in_channels, num_joints, loss_keypoint=None, train_cfg=None, test_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.loss = build_loss(loss_keypoint)
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.fc = nn.Linear(self.in_channels, self.num_joints * 2)

    def forward(self, x):
        """Forward function."""
        output = self.fc(x)
        N, C = output.shape
        return output.reshape([N, C // 2, 2])

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3
        losses['reg_loss'] = self.loss(output, target, target_weight)
        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        accuracy = dict()
        N = output.shape[0]
        _, avg_acc, cnt = keypoint_pck_accuracy(output.detach().cpu().numpy(), target.detach().cpu().numpy(), target_weight[:, :, 0].detach().cpu().numpy() > 0, thr=0.05, normalize=np.ones((N, 2), dtype=np.float32))
        accuracy['acc_pose'] = avg_acc
        return accuracy

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_regression (np.ndarray): Output regression.

        Args:
            x (torch.Tensor[N, K, 2]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        if flip_pairs is not None:
            output_regression = fliplr_regression(output.detach().cpu().numpy(), flip_pairs)
        else:
            output_regression = output.detach().cpu().numpy()
        return output_regression

    def decode(self, img_metas, output, **kwargs):
        """Decode the keypoints from output regression.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, 2]): predicted regression vector.
            kwargs: dict contains 'img_size'.
                img_size (tuple(img_width, img_height)): input image size.
        """
        batch_size = len(img_metas)
        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None
        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])
            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])
        preds, maxvals = keypoints_from_regression(output, c, s, kwargs['img_size'])
        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
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

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Based on Zhou et al., "On the Continuity of Rotation
    Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


class HMRMeshHead(nn.Module):
    """SMPL parameters regressor head of simple baseline. "End-to-end Recovery
    of Human Shape and Pose", CVPR'2018.

    Args:
        in_channels (int): Number of input channels
        smpl_mean_params (str): The file name of the mean SMPL parameters
        n_iter (int): The iterations of estimating delta parameters
    """

    def __init__(self, in_channels, smpl_mean_params=None, n_iter=3):
        super().__init__()
        self.in_channels = in_channels
        self.n_iter = n_iter
        npose = 24 * 6
        nbeta = 10
        ncam = 3
        hidden_dim = 1024
        self.fc1 = nn.Linear(in_channels + npose + nbeta + ncam, hidden_dim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(hidden_dim, npose)
        self.decshape = nn.Linear(hidden_dim, nbeta)
        self.deccam = nn.Linear(hidden_dim, ncam)
        if smpl_mean_params is None:
            init_pose = torch.zeros([1, npose])
            init_shape = torch.zeros([1, nbeta])
            init_cam = torch.FloatTensor([[1, 0, 0]])
        else:
            mean_params = np.load(smpl_mean_params)
            init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0).float()
            init_shape = torch.from_numpy(mean_params['shape'][:]).unsqueeze(0).float()
            init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0).float()
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x):
        """Forward function.

        x is the image feature map and is expected to be in shape (batch size x
        channel number x height x width)
        """
        batch_size = x.shape[0]
        x = x.mean(dim=-1).mean(dim=-1)
        init_pose = self.init_pose.expand(batch_size, -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for _ in range(self.n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        out = pred_rotmat, pred_shape, pred_cam
        return out

    def init_weights(self):
        """Initialize model weights."""
        xavier_init(self.decpose, gain=0.01)
        xavier_init(self.decshape, gain=0.01)
        xavier_init(self.deccam, gain=0.01)


class Heatmap3DHead(nn.Module):
    """Heatmap3DHead is a sub-module of Interhand3DHead, and outputs 3D
    heatmaps. Heatmap3DHead is composed of (>=0) number of deconv layers and a
    simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        depth_size (int): Number of depth discretization size
        num_deconv_layers (int): Number of deconv layers.
        num_deconv_layers should >= 0. Note that 0 means no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
        num_deconv_kernels (list|tuple): Kernel sizes.
        extra (dict): Configs for extra conv layers. Default: None
    """

    def __init__(self, in_channels, out_channels, depth_size=64, num_deconv_layers=3, num_deconv_filters=(256, 256, 256), num_deconv_kernels=(4, 4, 4), extra=None):
        super().__init__()
        assert out_channels % depth_size == 0
        self.depth_size = depth_size
        self.in_channels = in_channels
        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')
        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(num_deconv_layers, num_deconv_filters, num_deconv_kernels)
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0
        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[-1] if num_deconv_layers > 0 else self.in_channels
            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels', [1] * num_conv_layers)
                for i in range(num_conv_layers):
                    layers.append(build_conv_layer(dict(type='Conv2d'), in_channels=conv_channels, out_channels=conv_channels, kernel_size=num_conv_kernels[i], stride=1, padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(nn.ReLU(inplace=True))
            layers.append(build_conv_layer(cfg=dict(type='Conv2d'), in_channels=conv_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding))
            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) != length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) != length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.append(build_upsample_layer(dict(type='deconv'), in_channels=self.in_channels, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes
        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')
        return deconv_kernel, padding, output_padding

    def forward(self, x):
        """Forward function."""
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        N, C, H, W = x.shape
        x = x.reshape(N, C // self.depth_size, self.depth_size, H, W)
        return x

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)


class Heatmap1DHead(nn.Module):
    """Heatmap1DHead is a sub-module of Interhand3DHead, and outputs 1D
    heatmaps.

    Args:
        in_channels (int): Number of input channels
        heatmap_size (int): Heatmap size
        hidden_dims (list|tuple): Number of feature dimension of FC layers.
    """

    def __init__(self, in_channels=2048, heatmap_size=64, hidden_dims=(512,)):
        super().__init__()
        self.in_channels = in_channels
        self.heatmap_size = heatmap_size
        feature_dims = [in_channels, *hidden_dims, heatmap_size]
        self.fc = self._make_linear_layers(feature_dims, relu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)
        accu = heatmap1d * torch.arange(self.heatmap_size, dtype=heatmap1d.dtype, device=heatmap1d.device)[None, :]
        coord = accu.sum(dim=1)
        return coord

    def _make_linear_layers(self, feat_dims, relu_final=False):
        """Make linear layers."""
        layers = []
        for i in range(len(feat_dims) - 1):
            layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
            if i < len(feat_dims) - 2 or i == len(feat_dims) - 2 and relu_final:
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        heatmap1d = self.fc(x)
        value = self.soft_argmax_1d(heatmap1d).view(-1, 1)
        return value

    def init_weights(self):
        """Initialize model weights."""
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)


class MultilabelClassificationHead(nn.Module):
    """MultilabelClassificationHead is a sub-module of Interhand3DHead, and
    outputs hand type classification.

    Args:
        in_channels (int): Number of input channels
        num_labels (int): Number of labels
        hidden_dims (list|tuple): Number of hidden dimension of FC layers.
    """

    def __init__(self, in_channels=2048, num_labels=2, hidden_dims=(512,)):
        super().__init__()
        self.in_channels = in_channels
        self.num_labesl = num_labels
        feature_dims = [in_channels, *hidden_dims, num_labels]
        self.fc = self._make_linear_layers(feature_dims, relu_final=False)

    def _make_linear_layers(self, feat_dims, relu_final=False):
        """Make linear layers."""
        layers = []
        for i in range(len(feat_dims) - 1):
            layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
            if i < len(feat_dims) - 2 or i == len(feat_dims) - 2 and relu_final:
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        labels = torch.sigmoid(self.fc(x))
        return labels

    def init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)


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


def flip_back(output_flipped, flip_pairs, target_type='GaussianHeatmap'):
    """Flip the flipped heatmaps back to the original form.

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        output_flipped (np.ndarray[N, K, H, W]): The output heatmaps obtained
            from the flipped images.
        flip_pairs (list[tuple()): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).
        target_type (str): GaussianHeatmap or CombinedTarget

    Returns:
        np.ndarray: heatmaps that flipped back to the original image
    """
    assert output_flipped.ndim == 4, 'output_flipped should be [batch_size, num_keypoints, height, width]'
    shape_ori = output_flipped.shape
    channels = 1
    if target_type.lower() == 'CombinedTarget'.lower():
        channels = 3
        output_flipped[:, 1::3, ...] = -output_flipped[:, 1::3, ...]
    output_flipped = output_flipped.reshape(shape_ori[0], -1, channels, shape_ori[2], shape_ori[3])
    output_flipped_back = output_flipped.copy()
    for left, right in flip_pairs:
        output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
        output_flipped_back[:, right, ...] = output_flipped[:, left, ...]
    output_flipped_back = output_flipped_back.reshape(shape_ori)
    output_flipped_back = output_flipped_back[..., ::-1]
    return output_flipped_back


def _get_max_preds_3d(heatmaps):
    """Get keypoint predictions from 3D score maps.

    Note:
        batch size: N
        num keypoints: K
        heatmap depth size: D
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, D, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 3]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps, np.ndarray), 'heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 5, 'heatmaps should be 5-ndim'
    N, K, D, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))
    preds = np.zeros((N, K, 3), dtype=np.float32)
    _idx = idx[..., 0]
    preds[..., 2] = _idx // (H * W)
    preds[..., 1] = _idx // W % H
    preds[..., 0] = _idx % W
    preds = np.where(maxvals > 0.0, preds, -1)
    return preds, maxvals


def keypoints_from_heatmaps3d(heatmaps, center, scale):
    """Get final keypoint predictions from 3d heatmaps and transform them back
    to the image.

    Note:
        - batch size: N
        - num keypoints: K
        - heatmap depth size: D
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, D, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 3]): Predicted 3d keypoint location             in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    N, K, D, H, W = heatmaps.shape
    preds, maxvals = _get_max_preds_3d(heatmaps)
    for i in range(N):
        preds[i, :, :2] = transform_preds(preds[i, :, :2], center[i], scale[i], [W, H])
    return preds, maxvals


def multilabel_classification_accuracy(pred, gt, mask, thr=0.5):
    """Get multi-label classification accuracy.

    Note:
        - batch size: N
        - label number: L

    Args:
        pred (np.ndarray[N, L, 2]): model predicted labels.
        gt (np.ndarray[N, L, 2]): ground-truth labels.
        mask (np.ndarray[N, 1] or np.ndarray[N, L] ): reliability of
        ground-truth labels.

    Returns:
        float: multi-label classification accuracy.
    """
    valid = (mask > 0).min(axis=1) if mask.ndim == 2 else mask > 0
    pred, gt = pred[valid], gt[valid]
    if pred.shape[0] == 0:
        acc = 0.0
    else:
        acc = ((pred - thr) * (gt - thr) > 0).all(axis=1).mean()
    return acc


class Interhand3DHead(nn.Module):
    """Interhand 3D head of paper ref: Gyeongsik Moon. "InterHand2.6M: A
    Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single
    RGB Image".

    Args:
        keypoint_head_cfg (dict): Configs of Heatmap3DHead for hand
            keypoint estimation.
        root_head_cfg (dict): Configs of Heatmap1DHead for relative
            hand root depth estimation.
        hand_type_head_cfg (dict): Configs of MultilabelClassificationHead
            for hand type classification.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
        loss_root_depth (dict): Config for relative root depth loss.
            Default: None.
        loss_hand_type (dict): Config for hand type classification
            loss. Default: None.
    """

    def __init__(self, keypoint_head_cfg, root_head_cfg, hand_type_head_cfg, loss_keypoint=None, loss_root_depth=None, loss_hand_type=None, train_cfg=None, test_cfg=None):
        super().__init__()
        self.right_hand_head = Heatmap3DHead(**keypoint_head_cfg)
        self.left_hand_head = Heatmap3DHead(**keypoint_head_cfg)
        self.root_head = Heatmap1DHead(**root_head_cfg)
        self.hand_type_head = MultilabelClassificationHead(**hand_type_head_cfg)
        self.neck = GlobalAveragePooling()
        self.keypoint_loss = build_loss(loss_keypoint)
        self.root_depth_loss = build_loss(loss_root_depth)
        self.hand_type_loss = build_loss(loss_hand_type)
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

    def init_weights(self):
        self.left_hand_head.init_weights()
        self.right_hand_head.init_weights()
        self.root_head.init_weights()
        self.hand_type_head.init_weights()

    def get_loss(self, output, target, target_weight):
        """Calculate loss for hand keypoint heatmaps, relative root depth and
        hand type.

        Args:
            output (list[Tensor]): a list of outputs from multiple heads.
            target (list[Tensor]): a list of targets for multiple heads.
            target_weight (list[Tensor]): a list of targets weight for
                multiple heads.
        """
        losses = dict()
        assert not isinstance(self.keypoint_loss, nn.Sequential)
        out, tar, tar_weight = output[0], target[0], target_weight[0]
        assert tar.dim() == 5 and tar_weight.dim() == 3
        losses['hand_loss'] = self.keypoint_loss(out, tar, tar_weight)
        assert not isinstance(self.root_depth_loss, nn.Sequential)
        out, tar, tar_weight = output[1], target[1], target_weight[1]
        assert tar.dim() == 2 and tar_weight.dim() == 2
        losses['rel_root_loss'] = self.root_depth_loss(out, tar, tar_weight)
        assert not isinstance(self.hand_type_loss, nn.Sequential)
        out, tar, tar_weight = output[2], target[2], target_weight[2]
        assert tar.dim() == 2 and tar_weight.dim() in [1, 2]
        losses['hand_type_loss'] = self.hand_type_loss(out, tar, tar_weight)
        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for hand type.

        Args:
            output (list[Tensor]): a list of outputs from multiple heads.
            target (list[Tensor]): a list of targets for multiple heads.
            target_weight (list[Tensor]): a list of targets weight for
                multiple heads.
        """
        accuracy = dict()
        avg_acc = multilabel_classification_accuracy(output[2].detach().cpu().numpy(), target[2].detach().cpu().numpy(), target_weight[2].detach().cpu().numpy())
        accuracy['acc_classification'] = float(avg_acc)
        return accuracy

    def forward(self, x):
        """Forward function."""
        outputs = []
        outputs.append(torch.cat([self.right_hand_head(x), self.left_hand_head(x)], dim=1))
        x = self.neck(x)
        outputs.append(self.root_head(x))
        outputs.append(self.hand_type_head(x))
        return outputs

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output (list[np.ndarray]): list of output hand keypoint
            heatmaps, relative root depth and hand type.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        if flip_pairs is not None:
            heatmap_3d = output[0]
            N, K, D, H, W = heatmap_3d.shape
            heatmap_3d = heatmap_3d.reshape(N, K * D, H, W)
            heatmap_3d_flipped_back = flip_back(heatmap_3d.detach().cpu().numpy(), flip_pairs, target_type=self.target_type)
            heatmap_3d_flipped_back = heatmap_3d_flipped_back.reshape(N, K, D, H, W)
            if self.test_cfg.get('shift_heatmap', False):
                heatmap_3d_flipped_back[..., 1:] = heatmap_3d_flipped_back[..., :-1]
            output[0] = heatmap_3d_flipped_back
            output[1] = -output[1].detach().cpu().numpy()
            hand_type = output[2].detach().cpu().numpy()
            hand_type_flipped_back = hand_type.copy()
            hand_type_flipped_back[:, 0] = hand_type[:, 1]
            hand_type_flipped_back[:, 1] = hand_type[:, 0]
            output[2] = hand_type_flipped_back
        else:
            output = [out.detach().cpu().numpy() for out in output]
        return output

    def decode(self, img_metas, output, **kwargs):
        """Decode hand keypoint, relative root depth and hand type.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
                - "heatmap3d_depth_bound": depth bound of hand keypoint
                    3D heatmap
                - "root_depth_bound": depth bound of relative root depth
                    1D heatmap
            output (list[np.ndarray]): model predicted 3D heatmaps, relative
                root depth and hand type.
        """
        batch_size = len(img_metas)
        result = {}
        heatmap3d_depth_bound = np.ones(batch_size, dtype=np.float32)
        root_depth_bound = np.ones(batch_size, dtype=np.float32)
        center = np.zeros((batch_size, 2), dtype=np.float32)
        scale = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size, dtype=np.float32)
        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None
        for i in range(batch_size):
            heatmap3d_depth_bound[i] = img_metas[i]['heatmap3d_depth_bound']
            root_depth_bound[i] = img_metas[i]['root_depth_bound']
            center[i, :] = img_metas[i]['center']
            scale[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])
            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_boxes[:, 0:2] = center[:, 0:2]
        all_boxes[:, 2:4] = scale[:, 0:2]
        all_boxes[:, 4] = np.prod(scale * 200.0, axis=1)
        all_boxes[:, 5] = score
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids
        heatmap3d = output[0]
        preds, maxvals = keypoints_from_heatmaps3d(heatmap3d, center, scale)
        keypoints_3d = np.zeros((batch_size, preds.shape[1], 4), dtype=np.float32)
        keypoints_3d[:, :, 0:3] = preds[:, :, 0:3]
        keypoints_3d[:, :, 3:4] = maxvals
        keypoints_3d[:, :, 2] = (keypoints_3d[:, :, 2] / self.right_hand_head.depth_size - 0.5) * heatmap3d_depth_bound[:, np.newaxis]
        result['preds'] = keypoints_3d
        result['rel_root_depth'] = (output[1] / self.root_head.heatmap_size - 0.5) * root_depth_bound
        result['hand_type'] = output[2] > 0.5
        return result


def compute_similarity_transform(source_points, target_points):
    """Computes a similarity transform (sR, t) that takes a set of 3D points
    source_points (N x 3) closest to a set of 3D points target_points, where R
    is an 3x3 rotation matrix, t 3x1 translation, s scale. And return the
    transformed 3D points source_points_hat (N x 3). i.e. solves the orthogonal
    Procrutes problem.

    Note:
        Points number: N

    Args:
        source_points (np.ndarray): Source point set with shape [N, 3].
        target_points (np.ndarray): Target point set with shape [N, 3].

    Returns:
        np.ndarray: Transformed source point set with shape [N, 3].
    """
    assert target_points.shape[0] == source_points.shape[0]
    assert target_points.shape[1] == 3 and source_points.shape[1] == 3
    source_points = source_points.T
    target_points = target_points.T
    mu1 = source_points.mean(axis=1, keepdims=True)
    mu2 = target_points.mean(axis=1, keepdims=True)
    X1 = source_points - mu1
    X2 = target_points - mu2
    var1 = np.sum(X1 ** 2)
    K = X1.dot(X2.T)
    U, _, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R = V.dot(Z.dot(U.T))
    scale = np.trace(R.dot(K)) / var1
    t = mu2 - scale * R.dot(mu1)
    source_points_hat = scale * R.dot(source_points) + t
    source_points_hat = source_points_hat.T
    return source_points_hat


class TemporalRegressionHead(nn.Module):
    """Regression head of VideoPose3D.

    "3D human pose estimation in video with temporal convolutions and
    semi-supervised training", CVPR'2019.

    Args:
        in_channels (int): Number of input channels
        num_joints (int): Number of joints
        loss_keypoint (dict): Config for keypoint loss. Default: None.
        max_norm (float|None): if not None, the weight of convolution layers
            will be clipped to have a maximum norm of max_norm.
        is_trajectory (bool): If the model only predicts root joint
            position, then this arg should be set to True. In this case,
            traj_loss will be calculated. Otherwise, it should be set to
            False. Default: False.
    """

    def __init__(self, in_channels, num_joints, max_norm=None, loss_keypoint=None, is_trajectory=False, train_cfg=None, test_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.max_norm = max_norm
        self.loss = build_loss(loss_keypoint)
        self.is_trajectory = is_trajectory
        if self.is_trajectory:
            assert self.num_joints == 1
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.conv = build_conv_layer(dict(type='Conv1d'), in_channels, num_joints * 3, 1)
        if self.max_norm is not None:
            weight_clip = WeightNormClipHook(self.max_norm)
            for module in self.modules():
                if isinstance(module, nn.modules.conv._ConvNd):
                    weight_clip.register(module)

    @staticmethod
    def _transform_inputs(x):
        """Transform inputs for decoder.

        Args:
            inputs (tuple or list of Tensor | Tensor): multi-level features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(x, (list, tuple)):
            return x
        assert len(x) > 0
        return x[-1]

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        assert x.ndim == 3 and x.shape[2] == 1, f'Invalid shape {x.shape}'
        output = self.conv(x)
        N = output.shape[0]
        return output.reshape(N, self.num_joints, 3)

    def get_loss(self, output, target, target_weight):
        """Calculate keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 3]): Output keypoints.
            target (torch.Tensor[N, K, 3]): Target keypoints.
            target_weight (torch.Tensor[N, K, 3]):
                Weights across different joint types.
                If self.is_trajectory is True and target_weight is None,
                target_weight will be set inversely proportional to joint
                depth.
        """
        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        if self.is_trajectory:
            if target.dim() == 2:
                target.unsqueeze_(1)
            if target_weight is None:
                target_weight = (1 / target[:, :, 2:]).expand(target.shape)
            assert target.dim() == 3 and target_weight.dim() == 3
            losses['traj_loss'] = self.loss(output, target, target_weight)
        else:
            if target_weight is None:
                target_weight = target.new_ones(target.shape)
            assert target.dim() == 3 and target_weight.dim() == 3
            losses['reg_loss'] = self.loss(output, target, target_weight)
        return losses

    def get_accuracy(self, output, target, target_weight, metas):
        """Calculate accuracy for keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 3]): Output keypoints.
            target (torch.Tensor[N, K, 3]): Target keypoints.
            target_weight (torch.Tensor[N, K, 3]):
                Weights across different joint types.
            metas (list(dict)): Information about data augmentation including:

                - target_image_path (str): Optional, path to the image file
                - target_mean (float): Optional, normalization parameter of
                    the target pose.
                - target_std (float): Optional, normalization parameter of the
                    target pose.
                - root_position (np.ndarray[3,1]): Optional, global
                    position of the root joint.
                - root_index (torch.ndarray[1,]): Optional, original index of
                    the root joint before root-centering.
        """
        accuracy = dict()
        N = output.shape[0]
        output_ = output.detach().cpu().numpy()
        target_ = target.detach().cpu().numpy()
        if 'target_mean' in metas[0] and 'target_std' in metas[0]:
            target_mean = np.stack([m['target_mean'] for m in metas])
            target_std = np.stack([m['target_std'] for m in metas])
            output_ = self._denormalize_joints(output_, target_mean, target_std)
            target_ = self._denormalize_joints(target_, target_mean, target_std)
        if self.test_cfg.get('restore_global_position', False):
            root_pos = np.stack([m['root_position'] for m in metas])
            root_idx = metas[0].get('root_position_index', None)
            output_ = self._restore_global_position(output_, root_pos, root_idx)
            target_ = self._restore_global_position(target_, root_pos, root_idx)
        if target_weight is None:
            target_weight_ = np.ones_like(target_)
        else:
            target_weight_ = target_weight.detach().cpu().numpy()
            if self.test_cfg.get('restore_global_position', False):
                root_idx = metas[0].get('root_position_index', None)
                root_weight = metas[0].get('root_joint_weight', 1.0)
                target_weight_ = self._restore_root_target_weight(target_weight_, root_weight, root_idx)
        mpjpe = np.mean(np.linalg.norm((output_ - target_) * target_weight_, axis=-1))
        transformed_output = np.zeros_like(output_)
        for i in range(N):
            transformed_output[i, :, :] = compute_similarity_transform(output_[i, :, :], target_[i, :, :])
        p_mpjpe = np.mean(np.linalg.norm((transformed_output - target_) * target_weight_, axis=-1))
        accuracy['mpjpe'] = output.new_tensor(mpjpe)
        accuracy['p_mpjpe'] = output.new_tensor(p_mpjpe)
        return accuracy

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_regression (np.ndarray): Output regression.

        Args:
            x (torch.Tensor[N, K, 2]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        if flip_pairs is not None:
            output_regression = fliplr_regression(output.detach().cpu().numpy(), flip_pairs, center_mode='static', center_x=0)
        else:
            output_regression = output.detach().cpu().numpy()
        return output_regression

    def decode(self, metas, output):
        """Decode the keypoints from output regression.

        Args:
            metas (list(dict)): Information about data augmentation.
                By default this includes:

                - "target_image_path": path to the image file
            output (np.ndarray[N, K, 3]): predicted regression vector.
            metas (list(dict)): Information about data augmentation including:

                - target_image_path (str): Optional, path to the image file
                - target_mean (float): Optional, normalization parameter of
                    the target pose.
                - target_std (float): Optional, normalization parameter of the
                    target pose.
                - root_position (np.ndarray[3,1]): Optional, global
                    position of the root joint.
                - root_index (torch.ndarray[1,]): Optional, original index of
                    the root joint before root-centering.
        """
        if 'target_mean' in metas[0] and 'target_std' in metas[0]:
            target_mean = np.stack([m['target_mean'] for m in metas])
            target_std = np.stack([m['target_std'] for m in metas])
            output = self._denormalize_joints(output, target_mean, target_std)
        if self.test_cfg.get('restore_global_position', False):
            root_pos = np.stack([m['root_position'] for m in metas])
            root_idx = metas[0].get('root_position_index', None)
            output = self._restore_global_position(output, root_pos, root_idx)
        target_image_paths = [m.get('target_image_path', None) for m in metas]
        result = {'preds': output, 'target_image_paths': target_image_paths}
        return result

    @staticmethod
    def _denormalize_joints(x, mean, std):
        """Denormalize joint coordinates with given statistics mean and std.

        Args:
            x (np.ndarray[N, K, 3]): Normalized joint coordinates.
            mean (np.ndarray[K, 3]): Mean value.
            std (np.ndarray[K, 3]): Std value.
        """
        assert x.ndim == 3
        assert x.shape == mean.shape == std.shape
        return x * std + mean

    @staticmethod
    def _restore_global_position(x, root_pos, root_idx=None):
        """Restore global position of the root-centered joints.

        Args:
            x (np.ndarray[N, K, 3]): root-centered joint coordinates
            root_pos (np.ndarray[N,1,3]): The global position of the
                root joint.
            root_idx (int|None): If not none, the root joint will be inserted
                back to the pose at the given index.
        """
        x = x + root_pos
        if root_idx is not None:
            x = np.insert(x, root_idx, root_pos.squeeze(1), axis=1)
        return x

    @staticmethod
    def _restore_root_target_weight(target_weight, root_weight, root_idx=None):
        """Restore the target weight of the root joint after the restoration of
        the global position.

        Args:
            target_weight (np.ndarray[N, K, 1]): Target weight of relativized
                joints.
            root_weight (float): The target weight value of the root joint.
            root_idx (int|None): If not none, the root joint weight will be
                inserted back to the target weight at the given index.
        """
        if root_idx is not None:
            root_weight = np.full(target_weight.shape[0], root_weight, dtype=target_weight.dtype)
            target_weight = np.insert(target_weight, root_idx, root_weight[:, None], axis=1)
        return target_weight

    def init_weights(self):
        """Initialize the weights."""
        for m in self.modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)


def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([N, K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1
    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border), dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps, np.ndarray), 'heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'
    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W
    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] - heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] + heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        - batch size: B
        - num keypoints: K
        - num persons: N
        - height of heatmaps: H
        - width of heatmaps: W

        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        np.ndarray([N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert B == 1 or B == N
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)
    batch_heatmaps_pad = np.pad(batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='edge').flatten()
    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]
    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords


def keypoints_from_heatmaps(heatmaps, center, scale, unbiased=False, post_process='default', kernel=11, valid_radius_factor=0.0546875, use_udp=False, target_type='GaussianHeatmap'):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        - batch size: N
        - num keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        post_process (str/None): Choice of methods to post-process
            heatmaps. Currently supported: None, 'default', 'unbiased',
            'megvii'.
        unbiased (bool): Option to use unbiased decoding. Mutually
            exclusive with megvii.
            Note: this arg is deprecated and unbiased=True can be replaced
            by post_process='unbiased'
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.
        valid_radius_factor (float): The radius factor of the positive area
            in classification heatmap for UDP.
        use_udp (bool): Use unbiased data processing.
        target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
            GaussianHeatmap: Classification target with gaussian distribution.
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    heatmaps = heatmaps.copy()
    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert not post_process == 'megvii'
    if post_process is False:
        warnings.warn('post_process=False is deprecated, please use post_process=None instead', DeprecationWarning)
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn("post_process=True, unbiased=True is deprecated, please use post_process='unbiased' instead", DeprecationWarning)
            post_process = 'unbiased'
        else:
            warnings.warn("post_process=True, unbiased=False is deprecated, please use post_process='default' instead", DeprecationWarning)
            post_process = 'default'
    elif post_process == 'default':
        if unbiased is True:
            warnings.warn("unbiased=True is deprecated, please use post_process='unbiased' instead", DeprecationWarning)
            post_process = 'unbiased'
    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)
    N, K, H, W = heatmaps.shape
    if use_udp:
        if target_type.lower() == 'GaussianHeatMap'.lower():
            preds, maxvals = _get_max_preds(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type.lower() == 'CombinedTarget'.lower():
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError("target_type should be either 'GaussianHeatmap' or 'CombinedTarget'")
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == 'unbiased':
            heatmaps = np.log(np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([heatmap[py][px + 1] - heatmap[py][px - 1], heatmap[py + 1][px] - heatmap[py - 1][px]])
                        preds[n][k] += np.sign(diff) * 0.25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5
    for i in range(N):
        preds[i] = transform_preds(preds[i], center[i], scale[i], [W, H], use_udp=use_udp)
    if post_process == 'megvii':
        maxvals = maxvals / 255.0 + 0.5
    return preds, maxvals


class TopdownHeatmapBaseHead(nn.Module):
    """Base class for top-down heatmap heads.

    All top-down heatmap heads should subclass it.
    All subclass should overwrite:

    Methods:`get_loss`, supporting to calculate loss.
    Methods:`get_accuracy`, supporting to calculate accuracy.
    Methods:`forward`, supporting to forward model.
    Methods:`inference_model`, supporting to inference model.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_loss(self, **kwargs):
        """Gets the loss."""

    @abstractmethod
    def get_accuracy(self, **kwargs):
        """Gets the accuracy."""

    @abstractmethod
    def forward(self, **kwargs):
        """Forward function."""

    @abstractmethod
    def inference_model(self, **kwargs):
        """Inference function."""

    def decode(self, img_metas, output, **kwargs):
        """Decode keypoints from heatmaps.

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
        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None
        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])
            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])
        preds, maxvals = keypoints_from_heatmaps(output, c, s, unbiased=self.test_cfg.get('unbiased_decoding', False), post_process=self.test_cfg.get('post_process', 'default'), kernel=self.test_cfg.get('modulate_kernel', 11), valid_radius_factor=self.test_cfg.get('valid_radius_factor', 0.0546875), use_udp=self.test_cfg.get('use_udp', False), target_type=self.test_cfg.get('target_type', 'GaussianHeatmap'))
        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
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

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')
        return deconv_kernel, padding, output_padding


def pose_pck_accuracy(output, target, mask, thr=0.05, normalize=None):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints from heatmaps.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        output (np.ndarray[N, K, H, W]): Model output heatmaps.
        target (np.ndarray[N, K, H, W]): Groundtruth heatmaps.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation. Default 0.05.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - np.ndarray[K]: Accuracy of each keypoint.
        - float: Averaged accuracy across all keypoints.
        - int: Number of valid keypoints.
    """
    N, K, H, W = output.shape
    if K == 0:
        return None, 0, 0
    if normalize is None:
        normalize = np.tile(np.array([[H, W]]), (N, 1))
    pred, _ = _get_max_preds(output)
    gt, _ = _get_max_preds(target)
    return keypoint_pck_accuracy(pred, gt, mask, thr, normalize)


class TopdownHeatmapMultiStageHead(TopdownHeatmapBaseHead):
    """Top-down heatmap multi-stage head.

    TopdownHeatmapMultiStageHead is consisted of multiple branches,
    each of which has num_deconv_layers(>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_stages (int): Number of stages.
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self, in_channels=512, out_channels=17, num_stages=1, num_deconv_layers=3, num_deconv_filters=(256, 256, 256), num_deconv_kernels=(4, 4, 4), extra=None, loss_keypoint=None, train_cfg=None, test_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_stages = num_stages
        self.loss = build_loss(loss_keypoint)
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')
        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')
        self.multi_deconv_layers = nn.ModuleList([])
        for _ in range(self.num_stages):
            if num_deconv_layers > 0:
                deconv_layers = self._make_deconv_layer(num_deconv_layers, num_deconv_filters, num_deconv_kernels)
            elif num_deconv_layers == 0:
                deconv_layers = nn.Identity()
            else:
                raise ValueError(f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
            self.multi_deconv_layers.append(deconv_layers)
        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0
        self.multi_final_layers = nn.ModuleList([])
        for i in range(self.num_stages):
            if identity_final_layer:
                final_layer = nn.Identity()
            else:
                final_layer = build_conv_layer(cfg=dict(type='Conv2d'), in_channels=num_deconv_filters[-1] if num_deconv_layers > 0 else in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)
            self.multi_final_layers.append(final_layer)

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_outputs: O
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]):
                Output heatmaps.
            target (torch.Tensor[N,K,H,W]):
                Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """
        losses = dict()
        assert isinstance(output, list)
        assert target.dim() == 4 and target_weight.dim() == 3
        if isinstance(self.loss, nn.Sequential):
            assert len(self.loss) == len(output)
        for i in range(len(output)):
            target_i = target
            target_weight_i = target_weight
            if isinstance(self.loss, nn.Sequential):
                loss_func = self.loss[i]
            else:
                loss_func = self.loss
            loss_i = loss_func(output[i], target_i, target_weight_i)
            if 'heatmap_loss' not in losses:
                losses['heatmap_loss'] = loss_i
            else:
                losses['heatmap_loss'] += loss_i
        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """
        accuracy = dict()
        if self.target_type == 'GaussianHeatmap':
            _, avg_acc, _ = pose_pck_accuracy(output[-1].detach().cpu().numpy(), target.detach().cpu().numpy(), target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['acc_pose'] = float(avg_acc)
        return accuracy

    def forward(self, x):
        """Forward function.

        Returns:
            out (list[Tensor]): a list of heatmaps from multiple stages.
        """
        out = []
        assert isinstance(x, list)
        for i in range(self.num_stages):
            y = self.multi_deconv_layers[i](x[i])
            y = self.multi_final_layers[i](y)
            out.append(y)
        return out

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (List[torch.Tensor[NxKxHxW]]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        assert isinstance(output, list)
        output = output[-1]
        if flip_pairs is not None:
            output_heatmap = flip_back(output.detach().cpu().numpy(), flip_pairs, target_type=self.target_type)
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) != length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) != length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.append(build_upsample_layer(dict(type='deconv'), in_channels=self.in_channels, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes
        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.multi_deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.multi_final_layers.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)


class PRM(nn.Module):
    """Pose Refine Machine.

    Please refer to "Learning Delicate Local Representations
    for Multi-Person Pose Estimation" (ECCV 2020).

    Args:
        out_channels (int): Channel number of the output. Equals to
            the number of key points.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self, out_channels, norm_cfg=dict(type='BN')):
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.out_channels = out_channels
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_path = nn.Sequential(Linear(self.out_channels, self.out_channels), build_norm_layer(dict(type='BN1d'), out_channels)[1], build_activation_layer(dict(type='ReLU')), Linear(self.out_channels, self.out_channels), build_norm_layer(dict(type='BN1d'), out_channels)[1], build_activation_layer(dict(type='ReLU')), build_activation_layer(dict(type='Sigmoid')))
        self.bottom_path = nn.Sequential(ConvModule(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0, norm_cfg=norm_cfg, inplace=False), DepthwiseSeparableConvModule(self.out_channels, 1, kernel_size=9, stride=1, padding=4, norm_cfg=norm_cfg, inplace=False), build_activation_layer(dict(type='Sigmoid')))
        self.conv_bn_relu_prm_1 = ConvModule(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg, inplace=False)

    def forward(self, x):
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
    """Predict the heat map for an input feature.

    Args:
        unit_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        out_shape (tuple): Shape of the output heatmap.
        use_prm (bool): Whether to use pose refine machine. Default: False.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self, unit_channels, out_channels, out_shape, use_prm=False, norm_cfg=dict(type='BN')):
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.unit_channels = unit_channels
        self.out_channels = out_channels
        self.out_shape = out_shape
        self.use_prm = use_prm
        if use_prm:
            self.prm = PRM(out_channels, norm_cfg=norm_cfg)
        self.conv_layers = nn.Sequential(ConvModule(unit_channels, unit_channels, kernel_size=1, stride=1, padding=0, norm_cfg=norm_cfg, inplace=False), ConvModule(unit_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=None, inplace=False))

    def forward(self, feature):
        feature = self.conv_layers(feature)
        output = nn.functional.interpolate(feature, size=self.out_shape, mode='bilinear', align_corners=True)
        if self.use_prm:
            output = self.prm(output)
        return output


class TopdownHeatmapMSMUHead(TopdownHeatmapBaseHead):
    """Heads for multi-stage multi-unit heads used in Multi-Stage Pose
    estimation Network (MSPN), and Residual Steps Networks (RSN).

    Args:
        unit_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        out_shape (tuple): Shape of the output heatmap.
        num_stages (int): Number of stages.
        num_units (int): Number of units in each stage.
        use_prm (bool): Whether to use pose refine machine (PRM).
            Default: False.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self, out_shape, unit_channels=256, out_channels=17, num_stages=4, num_units=4, use_prm=False, norm_cfg=dict(type='BN'), loss_keypoint=None, train_cfg=None, test_cfg=None):
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')
        self.out_shape = out_shape
        self.unit_channels = unit_channels
        self.out_channels = out_channels
        self.num_stages = num_stages
        self.num_units = num_units
        self.loss = build_loss(loss_keypoint)
        self.predict_layers = nn.ModuleList([])
        for i in range(self.num_stages):
            for j in range(self.num_units):
                self.predict_layers.append(PredictHeatmap(unit_channels, out_channels, out_shape, use_prm, norm_cfg=norm_cfg))

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_outputs: O
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,O,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,O,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,O,K,1]):
                Weights across different joint types.
        """
        losses = dict()
        assert isinstance(output, list)
        assert target.dim() == 5 and target_weight.dim() == 4
        assert target.size(1) == len(output)
        if isinstance(self.loss, nn.Sequential):
            assert len(self.loss) == len(output)
        for i in range(len(output)):
            target_i = target[:, i, :, :, :]
            target_weight_i = target_weight[:, i, :, :]
            if isinstance(self.loss, nn.Sequential):
                loss_func = self.loss[i]
            else:
                loss_func = self.loss
            loss_i = loss_func(output[i], target_i, target_weight_i)
            if 'heatmap_loss' not in losses:
                losses['heatmap_loss'] = loss_i
            else:
                losses['heatmap_loss'] += loss_i
        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """
        accuracy = dict()
        if self.target_type == 'GaussianHeatmap':
            assert isinstance(output, list)
            assert target.dim() == 5 and target_weight.dim() == 4
            _, avg_acc, _ = pose_pck_accuracy(output[-1].detach().cpu().numpy(), target[:, -1, ...].detach().cpu().numpy(), target_weight[:, -1, ...].detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['acc_pose'] = float(avg_acc)
        return accuracy

    def forward(self, x):
        """Forward function.

        Returns:
            out (list[Tensor]): a list of heatmaps from multiple stages
                                and units.
        """
        out = []
        assert isinstance(x, list)
        assert len(x) == self.num_stages
        assert isinstance(x[0], list)
        assert len(x[0]) == self.num_units
        assert x[0][0].shape[1] == self.unit_channels
        for i in range(self.num_stages):
            for j in range(self.num_units):
                y = self.predict_layers[i * self.num_units + j](x[i][j])
                out.append(y)
        return out

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (list[torch.Tensor[N,K,H,W]]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        assert isinstance(output, list)
        output = output[-1]
        if flip_pairs is not None:
            output_heatmap = flip_back(output.detach().cpu().numpy(), flip_pairs, target_type=self.target_type)
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap

    def init_weights(self):
        """Initialize model weights."""
        for m in self.predict_layers.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)


class TopdownHeatmapSimpleHead(TopdownHeatmapBaseHead):
    """Top-down heatmap simple head. paper ref: Bin Xiao et al. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self, in_channels, out_channels, num_deconv_layers=3, num_deconv_filters=(256, 256, 256), num_deconv_kernels=(4, 4, 4), extra=None, in_index=0, input_transform=None, align_corners=False, loss_keypoint=None, train_cfg=None, test_cfg=None, upsample=0):
        super().__init__()
        self.in_channels = in_channels
        self.loss = build_loss(loss_keypoint)
        self.upsample = upsample
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')
        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners
        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')
        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(num_deconv_layers, num_deconv_filters, num_deconv_kernels)
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0
        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[-1] if num_deconv_layers > 0 else self.in_channels
            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels', [1] * num_conv_layers)
                for i in range(num_conv_layers):
                    layers.append(build_conv_layer(dict(type='Conv2d'), in_channels=conv_channels, out_channels=conv_channels, kernel_size=num_conv_kernels[i], stride=1, padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(nn.ReLU(inplace=True))
            layers.append(build_conv_layer(cfg=dict(type='Conv2d'), in_channels=conv_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding))
            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """
        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 4 and target_weight.dim() == 3
        losses['heatmap_loss'] = self.loss(output, target, target_weight)
        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """
        accuracy = dict()
        if self.target_type == 'GaussianHeatmap':
            _, avg_acc, _ = pose_pck_accuracy(output.detach().cpu().numpy(), target.detach().cpu().numpy(), target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['acc_pose'] = float(avg_acc)
        return accuracy

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        if flip_pairs is not None:
            output_heatmap = flip_back(output.detach().cpu().numpy(), flip_pairs, target_type=self.target_type)
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            if not isinstance(inputs, list):
                if self.upsample > 0:
                    inputs = resize(input=F.relu(inputs), scale_factor=self.upsample, mode='bilinear', align_corners=self.align_corners)
            return inputs
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [resize(input=x, size=inputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners) for x in inputs]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) != length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) != length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.append(build_upsample_layer(dict(type='deconv'), in_channels=self.in_channels, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes
        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)


class ViPNASHeatmapSimpleHead(TopdownHeatmapBaseHead):
    """ViPNAS heatmap simple head.

    ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search.
    More details can be found in the `paper
    <https://arxiv.org/abs/2105.10154>`__ .

    TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        num_deconv_groups (list|tuple): Group number.
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self, in_channels, out_channels, num_deconv_layers=3, num_deconv_filters=(144, 144, 144), num_deconv_kernels=(4, 4, 4), num_deconv_groups=(16, 16, 16), extra=None, in_index=0, input_transform=None, align_corners=False, loss_keypoint=None, train_cfg=None, test_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.loss = build_loss(loss_keypoint)
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')
        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners
        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')
        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(num_deconv_layers, num_deconv_filters, num_deconv_kernels, num_deconv_groups)
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0
        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[-1] if num_deconv_layers > 0 else self.in_channels
            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels', [1] * num_conv_layers)
                for i in range(num_conv_layers):
                    layers.append(build_conv_layer(dict(type='Conv2d'), in_channels=conv_channels, out_channels=conv_channels, kernel_size=num_conv_kernels[i], stride=1, padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(nn.ReLU(inplace=True))
            layers.append(build_conv_layer(cfg=dict(type='Conv2d'), in_channels=conv_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding))
            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """
        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 4 and target_weight.dim() == 3
        losses['heatmap_loss'] = self.loss(output, target, target_weight)
        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """
        accuracy = dict()
        if self.target_type.lower() == 'GaussianHeatmap'.lower():
            _, avg_acc, _ = pose_pck_accuracy(output.detach().cpu().numpy(), target.detach().cpu().numpy(), target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['acc_pose'] = float(avg_acc)
        return accuracy

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        if flip_pairs is not None:
            output_heatmap = flip_back(output.detach().cpu().numpy(), flip_pairs, target_type=self.target_type)
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [resize(input=x, size=inputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners) for x in inputs]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, num_groups):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) != length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) != length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)
        if num_layers != len(num_groups):
            error_msg = f'num_layers({num_layers}) != length of num_groups({len(num_groups)})'
            raise ValueError(error_msg)
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            groups = num_groups[i]
            layers.append(build_upsample_layer(dict(type='deconv'), in_channels=self.in_channels, out_channels=planes, kernel_size=kernel, groups=groups, stride=2, padding=padding, output_padding=output_padding, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes
        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)


class CuboidCenterHead(nn.Module):
    """Get results from the 3D human center heatmap. In this module, human 3D
    centers are local maximums obtained from the 3D heatmap via NMS (max-
    pooling).

    Args:
        space_size (list[3]): The size of the 3D space.
        cube_size (list[3]): The size of the heatmap volume.
        space_center (list[3]): The coordinate of space center.
        max_num (int): Maximum of human center detections.
        max_pool_kernel (int): Kernel size of the max-pool kernel in nms.
    """

    def __init__(self, space_size, space_center, cube_size, max_num=10, max_pool_kernel=3):
        super(CuboidCenterHead, self).__init__()
        self.register_buffer('grid_size', torch.tensor(space_size))
        self.register_buffer('cube_size', torch.tensor(cube_size))
        self.register_buffer('grid_center', torch.tensor(space_center))
        self.num_candidates = max_num
        self.max_pool_kernel = max_pool_kernel
        self.loss = nn.MSELoss()

    def _get_real_locations(self, indices):
        """
        Args:
            indices (torch.Tensor(NXP)): Indices of points in the 3D tensor

        Returns:
            real_locations (torch.Tensor(NXPx3)): Locations of points
                in the world coordinate system
        """
        real_locations = indices.float() / (self.cube_size - 1) * self.grid_size + self.grid_center - self.grid_size / 2.0
        return real_locations

    def _nms_by_max_pool(self, heatmap_volumes):
        max_num = self.num_candidates
        batch_size = heatmap_volumes.shape[0]
        root_cubes_nms = self._max_pool(heatmap_volumes)
        root_cubes_nms_reshape = root_cubes_nms.reshape(batch_size, -1)
        topk_values, topk_index = root_cubes_nms_reshape.topk(max_num)
        topk_unravel_index = self._get_3d_indices(topk_index, heatmap_volumes[0].shape)
        return topk_values, topk_unravel_index

    def _max_pool(self, inputs):
        kernel = self.max_pool_kernel
        padding = (kernel - 1) // 2
        max = F.max_pool3d(inputs, kernel_size=kernel, stride=1, padding=padding)
        keep = (inputs == max).float()
        return keep * inputs

    @staticmethod
    def _get_3d_indices(indices, shape):
        """Get indices in the 3-D tensor.

        Args:
            indices (torch.Tensor(NXp)): Indices of points in the 1D tensor
            shape (torch.Size(3)): The shape of the original 3D tensor

        Returns:
            indices: Indices of points in the original 3D tensor
        """
        batch_size = indices.shape[0]
        num_people = indices.shape[1]
        indices_x = (indices // (shape[1] * shape[2])).reshape(batch_size, num_people, -1)
        indices_y = (indices % (shape[1] * shape[2]) // shape[2]).reshape(batch_size, num_people, -1)
        indices_z = (indices % shape[2]).reshape(batch_size, num_people, -1)
        indices = torch.cat([indices_x, indices_y, indices_z], dim=2)
        return indices

    def forward(self, heatmap_volumes):
        """

        Args:
            heatmap_volumes (torch.Tensor(NXLXWXH)):
                3D human center heatmaps predicted by the network.
        Returns:
            human_centers (torch.Tensor(NXPX5)):
                Coordinates of human centers.
        """
        batch_size = heatmap_volumes.shape[0]
        topk_values, topk_unravel_index = self._nms_by_max_pool(heatmap_volumes.detach())
        topk_unravel_index = self._get_real_locations(topk_unravel_index)
        human_centers = torch.zeros(batch_size, self.num_candidates, 5, device=heatmap_volumes.device)
        human_centers[:, :, 0:3] = topk_unravel_index
        human_centers[:, :, 4] = topk_values
        return human_centers

    def get_loss(self, pred_cubes, gt):
        return dict(loss_center=self.loss(pred_cubes, gt))


class CuboidPoseHead(nn.Module):

    def __init__(self, beta):
        """Get results from the 3D human pose heatmap. Instead of obtaining
        maximums on the heatmap, this module regresses the coordinates of
        keypoints via integral pose regression. Refer to `paper.

        <https://arxiv.org/abs/2004.06239>` for more details.

        Args:
            beta: Constant to adjust the magnification of soft-maxed heatmap.
        """
        super(CuboidPoseHead, self).__init__()
        self.beta = beta
        self.loss = nn.L1Loss()

    def forward(self, heatmap_volumes, grid_coordinates):
        """

        Args:
            heatmap_volumes (torch.Tensor(NxKxLxWxH)):
                3D human pose heatmaps predicted by the network.
            grid_coordinates (torch.Tensor(Nx(LxWxH)x3)):
                Coordinates of the grids in the heatmap volumes.
        Returns:
            human_poses (torch.Tensor(NxKx3)): Coordinates of human poses.
        """
        batch_size = heatmap_volumes.size(0)
        channel = heatmap_volumes.size(1)
        x = heatmap_volumes.reshape(batch_size, channel, -1, 1)
        x = F.softmax(self.beta * x, dim=2)
        grid_coordinates = grid_coordinates.unsqueeze(1)
        x = torch.mul(x, grid_coordinates)
        human_poses = torch.sum(x, dim=2)
        return human_poses

    def get_loss(self, preds, targets, weights):
        return dict(loss_pose=self.loss(preds * weights, targets * weights))


class BCELoss(nn.Module):
    """Binary Cross Entropy loss."""

    def __init__(self, use_target_weight=False, loss_weight=1.0):
        super().__init__()
        self.criterion = F.binary_cross_entropy
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
            loss = self.criterion(output, target, reduction='none')
            if target_weight.dim() == 1:
                target_weight = target_weight[:, None]
            loss = (loss * target_weight).mean()
        else:
            loss = self.criterion(output, target)
        return loss * self.loss_weight


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

    def forward(self, output, target, target_weight):
        """Forward function.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            loss = self.criterion(output * target_weight.unsqueeze(-1), target * target_weight.unsqueeze(-1))
        else:
            loss = self.criterion(output, target)
        return loss * self.loss_weight


def perspective_projection(points, rotation, translation, focal_length, camera_center):
    """This function computes the perspective projection of a set of 3D points.

    Note:
        - batch size: B
        - point number: N

    Args:
        points (Tensor([B, N, 3])): A set of 3D points
        rotation (Tensor([B, 3, 3])): Camera rotation matrix
        translation (Tensor([B, 3])): Camera translation
        focal_length (Tensor([B,])): Focal length
        camera_center (Tensor([B, 2])): Camera center

    Returns:
        projected_points (Tensor([B, N, 2])): Projected 2D
            points in image space.
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)
    projected_points = points / points[:, :, -1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)
    projected_points = projected_points[:, :, :-1]
    return projected_points


class MeshLoss(nn.Module):
    """Mix loss for 3D human mesh. It is composed of loss on 2D joints, 3D
    joints, mesh vertices and smpl parameters (if any).

    Args:
        joints_2d_loss_weight (float): Weight for loss on 2D joints.
        joints_3d_loss_weight (float): Weight for loss on 3D joints.
        vertex_loss_weight (float): Weight for loss on 3D verteices.
        smpl_pose_loss_weight (float): Weight for loss on SMPL
            pose parameters.
        smpl_beta_loss_weight (float): Weight for loss on SMPL
            shape parameters.
        img_res (int): Input image resolution.
        focal_length (float): Focal length of camera model. Default=5000.
    """

    def __init__(self, joints_2d_loss_weight, joints_3d_loss_weight, vertex_loss_weight, smpl_pose_loss_weight, smpl_beta_loss_weight, img_res, focal_length=5000):
        super().__init__()
        self.criterion_vertex = nn.L1Loss(reduction='none')
        self.criterion_joints_2d = nn.SmoothL1Loss(reduction='none')
        self.criterion_joints_3d = nn.SmoothL1Loss(reduction='none')
        self.criterion_regr = nn.MSELoss(reduction='none')
        self.joints_2d_loss_weight = joints_2d_loss_weight
        self.joints_3d_loss_weight = joints_3d_loss_weight
        self.vertex_loss_weight = vertex_loss_weight
        self.smpl_pose_loss_weight = smpl_pose_loss_weight
        self.smpl_beta_loss_weight = smpl_beta_loss_weight
        self.focal_length = focal_length
        self.img_res = img_res

    def joints_2d_loss(self, pred_joints_2d, gt_joints_2d, joints_2d_visible):
        """Compute 2D reprojection loss on the joints.

        The loss is weighted by joints_2d_visible.
        """
        conf = joints_2d_visible.float()
        loss = (conf * self.criterion_joints_2d(pred_joints_2d, gt_joints_2d)).mean()
        return loss

    def joints_3d_loss(self, pred_joints_3d, gt_joints_3d, joints_3d_visible):
        """Compute 3D joints loss for the examples that 3D joint annotations
        are available.

        The loss is weighted by joints_3d_visible.
        """
        conf = joints_3d_visible.float()
        if len(gt_joints_3d) > 0:
            gt_pelvis = (gt_joints_3d[:, 2, :] + gt_joints_3d[:, 3, :]) / 2
            gt_joints_3d = gt_joints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_joints_3d[:, 2, :] + pred_joints_3d[:, 3, :]) / 2
            pred_joints_3d = pred_joints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_joints_3d(pred_joints_3d, gt_joints_3d)).mean()
        return pred_joints_3d.sum() * 0

    def vertex_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute 3D vertex loss for the examples that 3D human mesh
        annotations are available.

        The loss is weighted by the has_smpl.
        """
        conf = has_smpl.float()
        loss_vertex = self.criterion_vertex(pred_vertices, gt_vertices)
        loss_vertex = (conf[:, None, None] * loss_vertex).mean()
        return loss_vertex

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        """Compute SMPL parameters loss for the examples that SMPL parameter
        annotations are available.

        The loss is weighted by has_smpl.
        """
        conf = has_smpl.float()
        gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
        loss_regr_pose = self.criterion_regr(pred_rotmat, gt_rotmat)
        loss_regr_betas = self.criterion_regr(pred_betas, gt_betas)
        loss_regr_pose = (conf[:, None, None, None] * loss_regr_pose).mean()
        loss_regr_betas = (conf[:, None] * loss_regr_betas).mean()
        return loss_regr_pose, loss_regr_betas

    def project_points(self, points_3d, camera):
        """Perform orthographic projection of 3D points using the camera
        parameters, return projected 2D points in image plane.

        Note:
            - batch size: B
            - point number: N

        Args:
            points_3d (Tensor([B, N, 3])): 3D points.
            camera (Tensor([B, 3])): camera parameters with the
                3 channel as (scale, translation_x, translation_y)

        Returns:
            Tensor([B, N, 2]): projected 2D points                 in image space.
        """
        batch_size = points_3d.shape[0]
        device = points_3d.device
        cam_t = torch.stack([camera[:, 1], camera[:, 2], 2 * self.focal_length / (self.img_res * camera[:, 0] + 1e-09)], dim=-1)
        camera_center = camera.new_zeros([batch_size, 2])
        rot_t = torch.eye(3, device=device, dtype=points_3d.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        joints_2d = perspective_projection(points_3d, rotation=rot_t, translation=cam_t, focal_length=self.focal_length, camera_center=camera_center)
        return joints_2d

    def forward(self, output, target):
        """Forward function.

        Args:
            output (dict): dict of network predicted results.
                Keys: 'vertices', 'joints_3d', 'camera',
                'pose'(optional), 'beta'(optional)
            target (dict): dict of ground-truth labels.
                Keys: 'vertices', 'joints_3d', 'joints_3d_visible',
                'joints_2d', 'joints_2d_visible', 'pose', 'beta',
                'has_smpl'

        Returns:
            dict: dict of losses.
        """
        losses = {}
        pred_vertices = output['vertices']
        gt_vertices = target['vertices']
        has_smpl = target['has_smpl']
        loss_vertex = self.vertex_loss(pred_vertices, gt_vertices, has_smpl)
        losses['vertex_loss'] = loss_vertex * self.vertex_loss_weight
        if 'pose' in output.keys() and 'beta' in output.keys():
            pred_rotmat = output['pose']
            pred_betas = output['beta']
            gt_pose = target['pose']
            gt_betas = target['beta']
            loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl)
            losses['smpl_pose_loss'] = loss_regr_pose * self.smpl_pose_loss_weight
            losses['smpl_beta_loss'] = loss_regr_betas * self.smpl_beta_loss_weight
        pred_joints_3d = output['joints_3d']
        gt_joints_3d = target['joints_3d']
        joints_3d_visible = target['joints_3d_visible']
        loss_joints_3d = self.joints_3d_loss(pred_joints_3d, gt_joints_3d, joints_3d_visible)
        losses['joints_3d_loss'] = loss_joints_3d * self.joints_3d_loss_weight
        pred_camera = output['camera']
        gt_joints_2d = target['joints_2d']
        joints_2d_visible = target['joints_2d_visible']
        pred_joints_2d = self.project_points(pred_joints_3d, pred_camera)
        pred_joints_2d = 2 * pred_joints_2d / (self.img_res - 1)
        gt_joints_2d = 2 * gt_joints_2d / (self.img_res - 1) - 1
        loss_joints_2d = self.joints_2d_loss(pred_joints_2d, gt_joints_2d, joints_2d_visible)
        losses['joints_2d_loss'] = loss_joints_2d * self.joints_2d_loss_weight
        return losses


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super().__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    @staticmethod
    def _wgan_loss(input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan,                 otherwise, return Tensor.
        """
        if self.gan_type == 'wgan':
            return target_is_real
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:
                loss = -input.mean()
        else:
            loss = self.loss(input, target_label)
        return loss if is_disc else loss * self.loss_weight


class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.0):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0.0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred * target_weight[:, idx], heatmap_gt * target_weight[:, idx])
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)
        return loss / num_joints * self.loss_weight


class CombinedTargetMSELoss(nn.Module):
    """MSE loss for combined target.
        CombinedTarget: The combination of classification target
        (response map) and regression target (offset map).
        Paper ref: Huang et al. The Devil is in the Details: Delving into
        Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight, loss_weight=1.0):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
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
                heatmap_pred = heatmap_pred * target_weight[:, idx]
                heatmap_gt = heatmap_gt * target_weight[:, idx]
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_x_pred, heatmap_gt * offset_x_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_y_pred, heatmap_gt * offset_y_gt)
        return loss / num_joints * self.loss_weight


class JointsOHKMMSELoss(nn.Module):
    """MSE loss with online hard keypoint mining.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        topk (int): Only top k joint losses are kept.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, topk=8, loss_weight=1.0):
        super().__init__()
        assert topk > 0
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.loss_weight = loss_weight

    def _ohkm(self, loss):
        """Online hard keypoint mining."""
        ohkm_loss = 0.0
        N = len(loss)
        for i in range(N):
            sub_loss = loss[i]
            _, topk_idx = torch.topk(sub_loss, k=self.topk, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= N
        return ohkm_loss

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)
        if num_joints < self.topk:
            raise ValueError(f'topk ({self.topk}) should not larger than num_joints ({num_joints}).')
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        losses = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                losses.append(self.criterion(heatmap_pred * target_weight[:, idx], heatmap_gt * target_weight[:, idx]))
            else:
                losses.append(self.criterion(heatmap_pred, heatmap_gt))
        losses = [loss.mean(dim=1).unsqueeze(dim=1) for loss in losses]
        losses = torch.cat(losses, dim=1)
        return self._ohkm(losses) * self.loss_weight


class HeatmapLoss(nn.Module):
    """Accumulate the heatmap loss for each image in the batch.

    Args:
        supervise_empty (bool): Whether to supervise empty channels.
    """

    def __init__(self, supervise_empty=True):
        super().__init__()
        self.supervise_empty = supervise_empty

    def forward(self, pred, gt, mask):
        """Forward function.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            pred (torch.Tensor[N,K,H,W]):heatmap of output.
            gt (torch.Tensor[N,K,H,W]): target heatmap.
            mask (torch.Tensor[N,H,W]): mask of target.
        """
        assert pred.size() == gt.size(), f'pred.size() is {pred.size()}, gt.size() is {gt.size()}'
        if not self.supervise_empty:
            empty_mask = (gt.sum(dim=[2, 3], keepdim=True) > 0).float()
            loss = (pred - gt) ** 2 * empty_mask.expand_as(pred) * mask[:, None, :, :].expand_as(pred)
        else:
            loss = (pred - gt) ** 2 * mask[:, None, :, :].expand_as(pred)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss


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


class AELoss(nn.Module):
    """Associative Embedding loss.

    `Associative Embedding: End-to-End Learning for Joint Detection and
    Grouping <https://arxiv.org/abs/1611.05424v2>`_.
    """

    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

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
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp)) ** 2)
        num_tags = len(tags)
        if num_tags == 0:
            return _make_input(torch.zeros(1).float(), device=pred_tag.device), _make_input(torch.zeros(1).float(), device=pred_tag.device)
        elif num_tags == 1:
            return _make_input(torch.zeros(1).float(), device=pred_tag.device), pull
        tags = torch.stack(tags)
        size = num_tags, num_tags
        A = tags.expand(*size)
        B = A.permute(1, 0)
        diff = A - B
        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unknown ae loss type')
        push_loss = push / ((num_tags - 1) * num_tags) * 0.5
        pull_loss = pull / num_tags
        return push_loss, pull_loss

    def forward(self, tags, joints):
        """Accumulate the tag loss for each image in the batch.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            tags (torch.Tensor[N,KxHxW,1]): tag channels of output.
            joints (torch.Tensor[N,M,K,2]): joints information.
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)


class MultiLossFactory(nn.Module):
    """Loss for bottom-up models.

    Args:
        num_joints (int): Number of keypoints.
        num_stages (int): Number of stages.
        ae_loss_type (str): Type of ae loss.
        with_ae_loss (list[bool]): Use ae loss or not in multi-heatmap.
        push_loss_factor (list[float]):
            Parameter of push loss in multi-heatmap.
        pull_loss_factor (list[float]):
            Parameter of pull loss in multi-heatmap.
        with_heatmap_loss (list[bool]):
            Use heatmap loss or not in multi-heatmap.
        heatmaps_loss_factor (list[float]):
            Parameter of heatmap loss in multi-heatmap.
        supervise_empty (bool): Whether to supervise empty channels.
    """

    def __init__(self, num_joints, num_stages, ae_loss_type, with_ae_loss, push_loss_factor, pull_loss_factor, with_heatmaps_loss, heatmaps_loss_factor, supervise_empty=True):
        super().__init__()
        assert isinstance(with_heatmaps_loss, (list, tuple)), 'with_heatmaps_loss should be a list or tuple'
        assert isinstance(heatmaps_loss_factor, (list, tuple)), 'heatmaps_loss_factor should be a list or tuple'
        assert isinstance(with_ae_loss, (list, tuple)), 'with_ae_loss should be a list or tuple'
        assert isinstance(push_loss_factor, (list, tuple)), 'push_loss_factor should be a list or tuple'
        assert isinstance(pull_loss_factor, (list, tuple)), 'pull_loss_factor should be a list or tuple'
        self.num_joints = num_joints
        self.num_stages = num_stages
        self.ae_loss_type = ae_loss_type
        self.with_ae_loss = with_ae_loss
        self.push_loss_factor = push_loss_factor
        self.pull_loss_factor = pull_loss_factor
        self.with_heatmaps_loss = with_heatmaps_loss
        self.heatmaps_loss_factor = heatmaps_loss_factor
        self.heatmaps_loss = nn.ModuleList([(HeatmapLoss(supervise_empty) if with_heatmaps_loss else None) for with_heatmaps_loss in self.with_heatmaps_loss])
        self.ae_loss = nn.ModuleList([(AELoss(self.ae_loss_type) if with_ae_loss else None) for with_ae_loss in self.with_ae_loss])

    def forward(self, outputs, heatmaps, masks, joints):
        """Forward function to calculate losses.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K
            - output_channel: C C=2K if use ae loss else K

        Args:
            outputs (list(torch.Tensor[N,C,H,W])): outputs of stages.
            heatmaps (list(torch.Tensor[N,K,H,W])): target of heatmaps.
            masks (list(torch.Tensor[N,H,W])): masks of heatmaps.
            joints (list(torch.Tensor[N,M,K,2])): joints of ae loss.
        """
        heatmaps_losses = []
        push_losses = []
        pull_losses = []
        for idx in range(len(outputs)):
            offset_feat = 0
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs[idx][:, :self.num_joints]
                offset_feat = self.num_joints
                heatmaps_loss = self.heatmaps_loss[idx](heatmaps_pred, heatmaps[idx], masks[idx])
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmaps_losses.append(heatmaps_loss)
            else:
                heatmaps_losses.append(None)
            if self.ae_loss[idx]:
                tags_pred = outputs[idx][:, offset_feat:]
                batch_size = tags_pred.size()[0]
                tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)
                push_loss, pull_loss = self.ae_loss[idx](tags_pred, joints[idx])
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]
                push_losses.append(push_loss)
                pull_losses.append(pull_loss)
            else:
                push_losses.append(None)
                pull_losses.append(None)
        return heatmaps_losses, push_losses, pull_losses


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
            loss = self.criterion(output * target_weight, target * target_weight)
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
    """L1Loss loss ."""

    def __init__(self, use_target_weight=False, loss_weight=1.0):
        super().__init__()
        self.criterion = F.l1_loss
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

    def __init__(self, joint_parents, use_target_weight=False, loss_weight=1.0):
        super().__init__()
        self.joint_parents = joint_parents
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.non_root_indices = []
        for i in range(len(self.joint_parents)):
            if i != self.joint_parents[i]:
                self.non_root_indices.append(i)

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
            loss = torch.mean(torch.abs((output_bone * target_weight).mean(dim=0) - (target_bone * target_weight).mean(dim=0)))
        else:
            loss = torch.mean(torch.abs(output_bone.mean(dim=0) - target_bone.mean(dim=0)))
        return loss * self.loss_weight


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


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


class SMPL(nn.Module):
    """SMPL 3d human mesh model of paper ref: Matthew Loper. ``SMPL: A skinned
    multi-person linear model''. This module is based on the smplx project
    (https://github.com/vchoutas/smplx).

    Args:
        smpl_path (str): The path to the folder where the model weights are
            stored.
        joints_regressor (str): The path to the file where the joints
            regressor weight are stored.
    """

    def __init__(self, smpl_path, joints_regressor):
        super().__init__()
        assert has_smpl, 'Please install smplx to use SMPL.'
        self.smpl_neutral = SMPL_(model_path=smpl_path, create_global_orient=False, create_body_pose=False, create_transl=False, gender='neutral')
        self.smpl_male = SMPL_(model_path=smpl_path, create_betas=False, create_global_orient=False, create_body_pose=False, create_transl=False, gender='male')
        self.smpl_female = SMPL_(model_path=smpl_path, create_betas=False, create_global_orient=False, create_body_pose=False, create_transl=False, gender='female')
        joints_regressor = torch.tensor(np.load(joints_regressor), dtype=torch.float)[None, ...]
        self.register_buffer('joints_regressor', joints_regressor)
        self.num_verts = self.smpl_neutral.get_num_verts()
        self.num_joints = self.joints_regressor.shape[1]

    def smpl_forward(self, model, **kwargs):
        """Apply a specific SMPL model with given model parameters.

        Note:
            B: batch size
            V: number of vertices
            K: number of joints

        Returns:
            outputs (dict): Dict with mesh vertices and joints.
                - vertices: Tensor([B, V, 3]), mesh vertices
                - joints: Tensor([B, K, 3]), 3d joints regressed
                    from mesh vertices.
        """
        betas = kwargs['betas']
        batch_size = betas.shape[0]
        device = betas.device
        output = {}
        if batch_size == 0:
            output['vertices'] = betas.new_zeros([0, self.num_verts, 3])
            output['joints'] = betas.new_zeros([0, self.num_joints, 3])
        else:
            smpl_out = model(**kwargs)
            output['vertices'] = smpl_out.vertices
            output['joints'] = torch.matmul(self.joints_regressor, output['vertices'])
        return output

    def get_faces(self):
        """Return mesh faces.

        Note:
            F: number of faces

        Returns:
            faces: np.ndarray([F, 3]), mesh faces
        """
        return self.smpl_neutral.faces

    def forward(self, betas, body_pose, global_orient, transl=None, gender=None):
        """Forward function.

        Note:
            B: batch size
            J: number of controllable joints of model, for smpl model J=23
            K: number of joints

        Args:
            betas: Tensor([B, 10]), human body shape parameters of SMPL model.
            body_pose: Tensor([B, J*3] or [B, J, 3, 3]), human body pose
                parameters of SMPL model. It should be axis-angle vector
                ([B, J*3]) or rotation matrix ([B, J, 3, 3)].
            global_orient: Tensor([B, 3] or [B, 1, 3, 3]), global orientation
                of human body. It should be axis-angle vector ([B, 3]) or
                rotation matrix ([B, 1, 3, 3)].
            transl: Tensor([B, 3]), global translation of human body.
            gender: Tensor([B]), gender parameters of human body. -1 for
                neutral, 0 for male , 1 for female.

        Returns:
            outputs (dict): Dict with mesh vertices and joints.
                - vertices: Tensor([B, V, 3]), mesh vertices
                - joints: Tensor([B, K, 3]), 3d joints regressed from
                    mesh vertices.
        """
        batch_size = betas.shape[0]
        pose2rot = True if body_pose.dim() == 2 else False
        if batch_size > 0 and gender is not None:
            output = {'vertices': betas.new_zeros([batch_size, self.num_verts, 3]), 'joints': betas.new_zeros([batch_size, self.num_joints, 3])}
            mask = gender < 0
            _out = self.smpl_forward(self.smpl_neutral, betas=betas[mask], body_pose=body_pose[mask], global_orient=global_orient[mask], transl=transl[mask] if transl is not None else None, pose2rot=pose2rot)
            output['vertices'][mask] = _out['vertices']
            output['joints'][mask] = _out['joints']
            mask = gender == 0
            _out = self.smpl_forward(self.smpl_male, betas=betas[mask], body_pose=body_pose[mask], global_orient=global_orient[mask], transl=transl[mask] if transl is not None else None, pose2rot=pose2rot)
            output['vertices'][mask] = _out['vertices']
            output['joints'][mask] = _out['joints']
            mask = gender == 1
            _out = self.smpl_forward(self.smpl_male, betas=betas[mask], body_pose=body_pose[mask], global_orient=global_orient[mask], transl=transl[mask] if transl is not None else None, pose2rot=pose2rot)
            output['vertices'][mask] = _out['vertices']
            output['joints'][mask] = _out['joints']
        else:
            return self.smpl_forward(self.smpl_neutral, betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl, pose2rot=pose2rot)
        return output


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model1 = nn.Conv2d(3, 8, kernel_size=3)
        self.model2 = nn.Conv2d(3, 4, kernel_size=3)

    def forward(self, x):
        return x


class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, 1)
        self.bn = nn.SyncBatchNorm(2)

    def forward(self, x):
        return self.bn(self.conv(x))

    def forward_dummy(self, x):
        return self.forward(x),


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AdaptiveWingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (CombinedTargetMSELoss,
     lambda: ([], {'use_target_weight': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (CuboidCenterHead,
     lambda: ([], {'space_size': 4, 'space_center': 4, 'cube_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CuboidPoseHead,
     lambda: ([], {'beta': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (DummyModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {})),
    (ExampleModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GlobalAveragePooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HMRMeshHead,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (JointsMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (L1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MPJPELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Pool3DBlock,
     lambda: ([], {'pool_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RSoftmax,
     lambda: ([], {'radix': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SmoothL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SoftWingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TopdownHeatmapBaseHead,
     lambda: ([], {}),
     lambda: ([], {})),
    (Upsample3DBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (WingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

