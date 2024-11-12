
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


import torch.distributed as dist


import random


import warnings


import numpy as np


import torch.nn as nn


from torch.cuda._utils import _get_device_index


from torch.utils.data import DataLoader


from torch import nn


from copy import deepcopy


from functools import partial


from torchvision.utils import save_image


import math


from torchvision.utils import make_grid


import copy


from abc import ABCMeta


from abc import abstractmethod


from torch.utils.data import Dataset


from torch.utils.data import ConcatDataset


import numbers


import torchvision.transforms as transforms


from torch.nn.modules.utils import _pair


from collections.abc import Sequence


from torch.nn import functional as F


import logging


from torch.utils.data import DistributedSampler as _DistributedSampler


import torch.nn.functional as F


from collections import OrderedDict


from torch.nn import init


from torch.nn.init import _calculate_correct_fan


import torch.autograd as autograd


from torch.nn.functional import conv2d


import torchvision.models.vgg as vgg


import functools


import torch.multiprocessing as mp


import time


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torchvision.datasets as datasets


import torchvision.models as models


from typing import Any


import string


from typing import Iterable


from typing import Optional


class DistributedDataParallelWrapper(nn.Module):
    """A DistributedDataParallel wrapper for models in MMediting.

    In MMedting, there is a need to wrap different modules in the models
    with separate DistributedDataParallel. Otherwise, it will cause
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


def inference_with_session(sess, io_binding, output_names, input_tensor):
    device_type = input_tensor.device.type
    device_id = input_tensor.device.index
    device_id = 0 if device_id is None else device_id
    io_binding.bind_input(name='input', device_type=device_type, device_id=device_id, element_type=np.float32, shape=input_tensor.shape, buffer_ptr=input_tensor.data_ptr())
    for name in output_names:
        io_binding.bind_output(name)
    sess.run_with_iobinding(io_binding)
    pred = io_binding.copy_outputs_to_cpu()
    return pred


class ONNXRuntimeMattor(nn.Module):

    def __init__(self, sess, io_binding, output_names, base_model):
        super(ONNXRuntimeMattor, self).__init__()
        self.sess = sess
        self.io_binding = io_binding
        self.output_names = output_names
        self.base_model = base_model

    def forward(self, merged, trimap, meta, test_mode=False, save_image=False, save_path=None, iteration=None):
        input_tensor = torch.cat((merged, trimap), 1).contiguous()
        pred_alpha = inference_with_session(self.sess, self.io_binding, self.output_names, input_tensor)[0]
        pred_alpha = pred_alpha.squeeze()
        pred_alpha = self.base_model.restore_shape(pred_alpha, meta)
        eval_result = self.base_model.evaluate(pred_alpha, meta)
        if save_image:
            self.base_model.save_image(pred_alpha, meta, save_path, iteration)
        return {'pred_alpha': pred_alpha, 'eval_result': eval_result}


class RestorerGenerator(nn.Module):

    def __init__(self, sess, io_binding, output_names):
        super(RestorerGenerator, self).__init__()
        self.sess = sess
        self.io_binding = io_binding
        self.output_names = output_names

    def forward(self, x):
        pred = inference_with_session(self.sess, self.io_binding, self.output_names, x)[0]
        pred = torch.from_numpy(pred)
        return pred


class ONNXRuntimeRestorer(nn.Module):

    def __init__(self, sess, io_binding, output_names, base_model):
        super(ONNXRuntimeRestorer, self).__init__()
        self.sess = sess
        self.io_binding = io_binding
        self.output_names = output_names
        self.base_model = base_model
        restorer_generator = RestorerGenerator(self.sess, self.io_binding, self.output_names)
        base_model.generator = restorer_generator

    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        return self.base_model(lq, gt=gt, test_mode=test_mode, **kwargs)


class BaseModel(nn.Module, metaclass=ABCMeta):
    """Base model.

    All models should subclass it.
    All subclass should overwrite:

        ``init_weights``, supporting to initialize models.

        ``forward_train``, supporting to forward when training.

        ``forward_test``, supporting to forward when testing.

        ``train_step``, supporting to train one step when training.
    """

    @abstractmethod
    def init_weights(self):
        """Abstract method for initializing weight.

        All subclass should overwrite it.
        """

    @abstractmethod
    def forward_train(self, imgs, labels):
        """Abstract method for training forward.

        All subclass should overwrite it.
        """

    @abstractmethod
    def forward_test(self, imgs):
        """Abstract method for testing forward.

        All subclass should overwrite it.
        """

    def forward(self, imgs, labels, test_mode, **kwargs):
        """Forward function for base model.

        Args:
            imgs (Tensor): Input image(s).
            labels (Tensor): Ground-truth label(s).
            test_mode (bool): Whether in test mode.
            kwargs (dict): Other arguments.

        Returns:
            Tensor: Forward results.
        """
        if test_mode:
            return self.forward_test(imgs, **kwargs)
        return self.forward_train(imgs, labels, **kwargs)

    @abstractmethod
    def train_step(self, data_batch, optimizer):
        """Abstract method for one training step.

        All subclass should overwrite it.
        """

    def val_step(self, data_batch, **kwargs):
        """Abstract method for one validation step.

        All subclass should overwrite it.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def parse_losses(self, losses):
        """Parse losses dict for different loss variants.

        Args:
            losses (dict): Loss dict.

        Returns:
            loss (float): Sum of the total loss.
            log_vars (dict): loss dict for different variants.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = loss
        for name in log_vars:
            log_vars[name] = log_vars[name].item()
        return loss, log_vars


def build(cfg, registry, default_args=None):
    """Build module function.

    Args:
        cfg (dict): Configuration for building modules.
        registry (obj): ``registry`` object.
        default_args (dict, optional): Default arguments. Defaults to None.
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone.

    Args:
        cfg (dict): Configuration for building backbone.
    """
    return build(cfg, BACKBONES)


def build_loss(cfg):
    """Build loss.

    Args:
        cfg (dict): Configuration for building loss.
    """
    return build(cfg, LOSSES)


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def psnr(img1, img2, crop_border=0, input_order='HWC', convert_to=None):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.

    Returns:
        float: psnr result.
    """
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    if isinstance(convert_to, str) and convert_to.lower() == 'y':
        img1 = mmcv.bgr2ycbcr(img1 / 255.0, y_only=True) * 255.0
        img2 = mmcv.bgr2ycbcr(img2 / 255.0, y_only=True) * 255.0
    elif convert_to is not None:
        raise ValueError('Wrong color model. Supported values are "Y" and None.')
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]
    mse_value = np.mean((img1 - img2) ** 2)
    if mse_value == 0:
        return float('inf')
    return 20.0 * np.log10(255.0 / np.sqrt(mse_value))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1, img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(img1, img2, crop_border=0, input_order='HWC', convert_to=None):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SSIM calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.

    Returns:
        float: ssim result.
    """
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    if isinstance(convert_to, str) and convert_to.lower() == 'y':
        img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
        img1 = mmcv.bgr2ycbcr(img1 / 255.0, y_only=True) * 255.0
        img2 = mmcv.bgr2ycbcr(img2 / 255.0, y_only=True) * 255.0
        img1 = np.expand_dims(img1, axis=2)
        img2 = np.expand_dims(img2, axis=2)
    elif convert_to is not None:
        raise ValueError('Wrong color model. Supported values are "Y" and None')
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]
    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to (min, max), image values will be normalized to [0, 1].

    For different tensor shapes, this function will have different behaviors:

        1. 4D mini-batch Tensor of shape (N x 3/1 x H x W):
            Use `make_grid` to stitch images in the batch dimension, and then
            convert it to numpy array.
        2. 3D Tensor of shape (3/1 x H x W) and 2D Tensor of shape (H x W):
            Directly change to numpy array.

    Note that the image channel in input tensors should be RGB order. This
    function will convert it to cv2 convention, i.e., (H x W x C) with BGR
    order.

    Args:
        tensor (Tensor | list[Tensor]): Input tensors.
        out_type (numpy type): Output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple): min and max values for clamp.

    Returns:
        (Tensor | list[Tensor]): 3D ndarray of shape (H x W x C) or 2D ndarray
        of shape (H x W).
    """
    if not (torch.is_tensor(tensor) or isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor)):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')
    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).squeeze(0)
        _tensor = _tensor.float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise ValueError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    result = result[0] if len(result) == 1 else result
    return result


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model.

    Args:
        cfg (dict): Configuration for building model.
        train_cfg (dict): Training configuration. Default: None.
        test_cfg (dict): Testing configuration. Default: None.
    """
    return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


class ONNXRuntimeEditing(nn.Module):

    def __init__(self, onnx_file, cfg, device_id):
        super(ONNXRuntimeEditing, self).__init__()
        ort_custom_op_path = ''
        try:
            ort_custom_op_path = get_onnxruntime_op_path()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv,                 you may have to build mmcv with ONNXRuntime from source.')
        session_options = ort.SessionOptions()
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        sess = ort.InferenceSession(onnx_file, session_options)
        providers = ['CPUExecutionProvider']
        options = [{}]
        is_cuda_available = ort.get_device() == 'GPU'
        if is_cuda_available:
            providers.insert(0, 'CUDAExecutionProvider')
            options.insert(0, {'device_id': device_id})
        sess.set_providers(providers, options)
        self.sess = sess
        self.device_id = device_id
        self.io_binding = sess.io_binding()
        self.output_names = [_.name for _ in sess.get_outputs()]
        base_model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        if isinstance(base_model, BaseMattor):
            WrapperClass = ONNXRuntimeMattor
        elif isinstance(base_model, BasicRestorer):
            WrapperClass = ONNXRuntimeRestorer
        self.wrapper = WrapperClass(self.sess, self.io_binding, self.output_names, base_model)

    def forward(self, **kwargs):
        return self.wrapper(**kwargs)


class AdaptiveFeatureSelect_maskgen(nn.Module):

    def __init__(self, outchannel=512 * 5, stylechannel=256, hard_fusion=False):
        super(AdaptiveFeatureSelect_maskgen, self).__init__()
        self.hardfusion = hard_fusion
        self.outchannel = outchannel
        self.mapping = nn.Sequential(nn.Linear(stylechannel, stylechannel), nn.LeakyReLU(0.1, True), nn.Linear(stylechannel, outchannel * 2))

    def forward(self, style):
        mask = self.mapping(style)
        mask = torch.unsqueeze(mask, 1).view(style.shape[0], 5, 2, 512, 1, 1)
        mask = torch.softmax(mask, 2)
        return mask[:, :, 0, :, :, :].squeeze(1), mask[:, :, 1, :, :, :].squeeze(1)


class AdaptiveFeatureSelect(nn.Module):

    def __init__(self, outchannel=512, stylechannel=256, hard_fusion=False):
        super(AdaptiveFeatureSelect, self).__init__()
        self.hardfusion = hard_fusion
        self.outchannel = outchannel
        self.mapping = nn.Sequential(nn.Linear(stylechannel, stylechannel), nn.LeakyReLU(0.1, True), nn.Linear(stylechannel, outchannel * 2))

    def forward(self, fea1, fea2, style):
        mask = self.mapping(style)
        mask = torch.unsqueeze(mask, 1).view(-1, 2, self.outchannel, 1, 1)
        mask = torch.softmax(mask, 1)
        fea1 = fea1 * mask[:, 0, :, :, :].squeeze(1)
        fea2 = fea2 * mask[:, 1, :, :, :].squeeze(1)
        return fea1 + fea2


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(), nn.Linear(gate_channels, gate_channels // reduction_ratio), nn.ReLU(), nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, fea_preprocess):
        x = fea_preprocess(x)
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = self.avgpool(x)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = self.maxpool(x)
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = self.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, fea_preprocess):
        x = fea_preprocess(x)
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.fea_preprocess = nn.Conv2d(gate_channels, gate_channels, 3, 1, 1)
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x, self.fea_preprocess)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out, self.fea_preprocess)
        return x_out


class SpatialGate_interpolation(nn.Module):

    def __init__(self):
        super(SpatialGate_interpolation, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, fea_preprocess):
        x = fea_preprocess(x)
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        mask = self.sigmoid(x_out)
        return x * mask + y * (1 - mask)


class CBAM_spatial_interpolation(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_spatial_interpolation, self).__init__()
        self.fea_preprocess = nn.Conv2d(gate_channels, gate_channels, 3, 1, 1)
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate_interpolation = SpatialGate_interpolation()

    def forward(self, x, y):
        x_out = self.SpatialGate_interpolation(x, y, self.fea_preprocess)
        return x_out


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.E = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, True), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, True), nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1, True), nn.AdaptiveAvgPool2d(1))
        self.mlp = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(0.1, True), nn.Linear(256, 256))

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)
        return fea, out


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels=64, growth_channels=32):
        super().__init__()
        for i in range(5):
            out_channels = mid_channels if i == 4 else growth_channels
            self.add_module(f'conv{i + 1}', nn.Conv2d(mid_channels + i * growth_channels, out_channels, 3, 1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.init_weights()

    def init_weights(self):
        """Init weights for ResidualDenseBlock.

        Use smaller std for better stability and performance. We empirically
        use 0.1. See more details in "ESRGAN: Enhanced Super-Resolution
        Generative Adversarial Networks"
        """
        for i in range(5):
            default_init_weights(getattr(self, f'conv{i + 1}'), 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels, growth_channels=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(mid_channels, growth_channels)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class RRDBFeatureExtractor(nn.Module):
    """Feature extractor composed of Residual-in-Residual Dense Blocks (RRDBs).

    It is equivalent to ESRGAN with the upsampling module removed.

    Args:
        in_channels (int): Channel number of inputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Default: 23
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self, in_channels=3, mid_channels=64, num_blocks=23, growth_channels=32):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(RRDB, num_blocks, mid_channels=mid_channels, growth_channels=growth_channels)
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        feat = self.conv_first(x)
        return feat + self.conv_body(self.body(feat))


def build_component(cfg):
    """Build component.

    Args:
        cfg (dict): Configuration for building component.
    """
    return build(cfg, COMPONENTS)


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmedit".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = get_logger(__name__.split('.')[0], log_file, log_level)
    return logger


class Panini_MFR(nn.Module):

    def __init__(self, in_size, out_size, img_channels=3, rrdb_channels=8, num_rrdbs=2, style_channels=512, num_mlps=8, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, default_style_mode='mix', eval_style_mode='single', mix_prob=0.9, pretrained=None, bgr2rgb=False):
        super().__init__()
        if in_size >= out_size:
            raise ValueError(f'in_size must be smaller than out_size, but got {in_size} and {out_size}.')
        self.generator = build_component(dict(type='StyleGANv2Generator', out_size=out_size, style_channels=style_channels, num_mlps=num_mlps, channel_multiplier=channel_multiplier, blur_kernel=blur_kernel, lr_mlp=lr_mlp, default_style_mode=default_style_mode, eval_style_mode=eval_style_mode, mix_prob=mix_prob, pretrained=pretrained, bgr2rgb=bgr2rgb))
        self.generator.requires_grad_(True)
        self.in_size = in_size
        self.style_channels = style_channels
        channels = self.generator.channels
        self.resize = nn.AdaptiveAvgPool2d((256, 256))
        self.deg_encoder = Encoder()
        with torch.no_grad():
            model_dict = self.deg_encoder.state_dict()
            pretrained_dict = torch.load('checkpoint/moco_checkpoint_0199.pth.tar', map_location='cpu')
            keys = []
            for k, v in pretrained_dict['state_dict'].items():
                keys.append(k)
            i = 2
            for k, v in model_dict.items():
                if v.size() == pretrained_dict['state_dict'][keys[i]].size():
                    model_dict[k] = pretrained_dict['state_dict'][keys[i]]
                    i = i + 1
                else:
                    None
            None
            self.deg_encoder.load_state_dict(model_dict)
            self.deg_encoder.requires_grad_(True)
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential(RRDBFeatureExtractor(img_channels, rrdb_channels, num_blocks=num_rrdbs), CBAM(rrdb_channels, reduction_ratio=2), nn.Conv2d(rrdb_channels, 32, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(32, 64, 3, 2, 1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(64, 128, 3, 2, 1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(128, 512, 3, 2, 1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        for i in range(4):
            self.encoder.append(nn.Sequential(nn.Conv2d(512, 512, 3, 2, 1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        self.encoder.append(nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Flatten(), nn.Linear(16 * 256, 18 * style_channels)))
        self.AFS_maskGen = AdaptiveFeatureSelect_maskgen()
        self.fusion_out = nn.ModuleList()
        for res in [64, 32, 16, 8, 4]:
            num_channels = channels[res]
            self.fusion_out.append(nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=True))

    def forward(self, lq):
        """Forward function.

        Args:
            lq (Tensor): Input LR image with shape (n, c, h, w).

        Returns:
            Tensor: Output HR image.
        """
        h, w = lq.shape[2:]
        if h != self.in_size or w != self.in_size:
            raise AssertionError(f'Spatial resolution must equal in_size ({self.in_size}). Got ({h}, {w}).')
        img_deg = self.resize(lq)
        deg_style = self.deg_encoder(img_deg)
        afs_mask1, afs_mask2 = self.AFS_maskGen(deg_style)
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]
        latent = encoder_features[0].view(lq.size(0), -1, self.style_channels)
        encoder_features = encoder_features[1:]
        injected_noise = [getattr(self.generator, f'injected_noise_{i}') for i in range(self.generator.num_injected_noises)]
        out = self.generator.constant_input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.generator.to_rgb1(out, latent[:, 1])
        _index = 1
        for up_conv, conv, noise1, noise2, to_rgb in zip(self.generator.convs[::2], self.generator.convs[1::2], injected_noise[1::2], injected_noise[2::2], self.generator.to_rgbs):
            if out.size(2) <= 64:
                fusion_index = (_index - 1) // 2
                feat = encoder_features[fusion_index]
                feat = self.fusion_out[fusion_index](feat)
                out = afs_mask1[:, fusion_index, :, :, :].squeeze(1) * out + afs_mask2[:, fusion_index, :, :, :].squeeze(1) * feat
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)
            _index += 2
        return skip

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. But received {type(pretrained)}.')


class Panini_SR(nn.Module):

    def __init__(self, in_size, out_size, img_channels=3, rrdb_channels=64, num_rrdbs=23, style_channels=512, num_mlps=8, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, default_style_mode='mix', eval_style_mode='single', mix_prob=0.9, pretrained=None, bgr2rgb=False):
        super().__init__()
        if in_size >= out_size:
            raise ValueError(f'in_size must be smaller than out_size, but got {in_size} and {out_size}.')
        self.generator = build_component(dict(type='StyleGANv2Generator', out_size=out_size, style_channels=style_channels, num_mlps=num_mlps, channel_multiplier=channel_multiplier, blur_kernel=blur_kernel, lr_mlp=lr_mlp, default_style_mode=default_style_mode, eval_style_mode=eval_style_mode, mix_prob=mix_prob, pretrained=pretrained, bgr2rgb=bgr2rgb))
        self.generator.requires_grad_(True)
        self.in_size = in_size
        self.style_channels = style_channels
        channels = self.generator.channels
        self.choice_vector = nn.Parameter(torch.randn(1, 256))
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential(RRDBFeatureExtractor(img_channels, rrdb_channels, num_blocks=num_rrdbs), CBAM(rrdb_channels, reduction_ratio=8), nn.Conv2d(rrdb_channels, 512, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        for i in range(4):
            None
            self.encoder.append(nn.Sequential(nn.Conv2d(512, 512, 3, 2, 1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        self.encoder.append(nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Flatten(), nn.Linear(16 * 256, 18 * style_channels)))
        self.AFS_maskGen = AdaptiveFeatureSelect_maskgen()
        self.fusion_out = nn.ModuleList()
        for res in [64, 32, 16, 8, 4]:
            num_channels = channels[res]
            self.fusion_out.append(nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=True))

    def forward(self, lq):
        """Forward function.

        Args:
            lq (Tensor): Input LR image with shape (n, c, h, w).

        Returns:
            Tensor: Output HR image.
        """
        h, w = lq.shape[2:]
        if h != self.in_size or w != self.in_size:
            raise AssertionError(f'Spatial resolution must equal in_size ({self.in_size}). Got ({h}, {w}).')
        deg_style = self.choice_vector.repeat(lq.shape[0], 1, 1)
        afs_mask1, afs_mask2 = self.AFS_maskGen(deg_style)
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]
        latent = encoder_features[0].view(lq.size(0), -1, self.style_channels)
        encoder_features = encoder_features[1:]
        injected_noise = [getattr(self.generator, f'injected_noise_{i}') for i in range(self.generator.num_injected_noises)]
        out = self.generator.constant_input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.generator.to_rgb1(out, latent[:, 1])
        _index = 1
        for up_conv, conv, noise1, noise2, to_rgb in zip(self.generator.convs[::2], self.generator.convs[1::2], injected_noise[1::2], injected_noise[2::2], self.generator.to_rgbs):
            if out.size(2) <= 64:
                fusion_index = (_index - 1) // 2
                feat = encoder_features[fusion_index]
                feat = self.fusion_out[fusion_index](feat)
                out = afs_mask1[:, fusion_index, :, :, :].squeeze(1) * out + afs_mask2[:, fusion_index, :, :, :].squeeze(1) * feat
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)
            _index += 2
        return skip

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. But received {type(pretrained)}.')


def pixel_unshuffle(x, scale):
    """Down-sample by pixel unshuffle.

    Args:
        x (Tensor): Input tensor.
        scale (int): Scale factor.

    Returns:
        Tensor: Output tensor.
    """
    b, c, h, w = x.shape
    if h % scale != 0 or w % scale != 0:
        raise AssertionError(f'Invalid scale ({scale}) of pixel unshuffle for tensor with shape: {x.shape}')
    h = int(h / scale)
    w = int(w / scale)
    x = x.view(b, c, h, scale, w, scale)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, -1, h, w)


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN and Real-ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data. # noqa: E501
    Currently, it supports [x1/x2/x4] upsampling scale factor.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Defaults: 23
        growth_channels (int): Channels for each growth. Default: 32.
        upscale_factor (int): Upsampling factor. Support x1, x2 and x4.
            Default: 4.
    """
    _supported_upscale_factors = [1, 2, 4]

    def __init__(self, in_channels, out_channels, mid_channels=64, num_blocks=23, growth_channels=32, upscale_factor=4):
        super().__init__()
        if upscale_factor in self._supported_upscale_factors:
            in_channels = in_channels * (4 // upscale_factor) ** 2
        else:
            raise ValueError(f'Unsupported scale factor {upscale_factor}. Currently supported ones are {self._supported_upscale_factors}.')
        self.upscale_factor = upscale_factor
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(RRDB, num_blocks, mid_channels=mid_channels, growth_channels=growth_channels)
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.upscale_factor in [1, 2]:
            feat = pixel_unshuffle(x, scale=4 // self.upscale_factor)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            for m in [self.conv_first, self.conv_body, self.conv_up1, self.conv_up2, self.conv_hr, self.conv_last]:
                default_init_weights(m, 0.1)
        else:
            raise TypeError(f'"pretrained" must be a str or None. But received {type(pretrained)}.')


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg):
        super().__init__(nn.AdaptiveAvgPool2d(1), ConvModule(in_channels, out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module.

    See https://arxiv.org/pdf/1704.04861.pdf for details.

    This module can replace a ConvModule with the conv block replaced by two
    conv block: depthwise conv block and pointwise conv block. The depthwise
    conv block contains depthwise-conv/norm/activation layers. The pointwise
    conv block contains pointwise-conv/norm/activation layers. It should be
    noted that there will be norm/activation layer in the depthwise conv block
    if ``norm_cfg`` and ``act_cfg`` are specified.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d. Default: 1.
        padding (int or tuple[int]): Same as nn.Conv2d. Default: 0.
        dilation (int or tuple[int]): Same as nn.Conv2d. Default: 1.
        norm_cfg (dict): Default norm config for both depthwise ConvModule and
            pointwise ConvModule. Default: None.
        act_cfg (dict): Default activation config for both depthwise ConvModule
            and pointwise ConvModule. Default: dict(type='ReLU').
        dw_norm_cfg (dict): Norm config of depthwise ConvModule. If it is
            'default', it will be the same as ``norm_cfg``. Default: 'default'.
        dw_act_cfg (dict): Activation config of depthwise ConvModule. If it is
            'default', it will be the same as ``act_cfg``. Default: 'default'.
        pw_norm_cfg (dict): Norm config of pointwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        pw_act_cfg (dict): Activation config of pointwise ConvModule. If it is
            'default', it will be the same as ``act_cfg``. Default: 'default'.
        kwargs (optional): Other shared arguments for depthwise and pointwise
            ConvModule. See ConvModule for ref.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_cfg=None, act_cfg=dict(type='ReLU'), dw_norm_cfg='default', dw_act_cfg='default', pw_norm_cfg='default', pw_act_cfg='default', **kwargs):
        super().__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'default' else norm_cfg
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'default' else norm_cfg
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'default' else act_cfg
        self.depthwise_conv = ConvModule(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, norm_cfg=dw_norm_cfg, act_cfg=dw_act_cfg, **kwargs)
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1, norm_cfg=pw_norm_cfg, act_cfg=pw_act_cfg, **kwargs)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            Tensor: Output tensor.
        """
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class ASPP(nn.Module):
    """ASPP module from DeepLabV3.

    The code is adopted from
    https://github.com/pytorch/vision/blob/master/torchvision/models/
    segmentation/deeplabv3.py

    For more information about the module:
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        in_channels (int): Input channels of the module.
        out_channels (int): Output channels of the module.
        mid_channels (int): Output channels of the intermediate ASPP conv
            modules.
        dilations (Sequence[int]): Dilation rate of three ASPP conv module.
            Default: [12, 24, 36].
        conv_cfg (dict): Config dict for convolution layer. If "None",
            nn.Conv2d will be applied. Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        separable_conv (bool): Whether replace normal conv with depthwise
            separable conv which is faster. Default: False.
    """

    def __init__(self, in_channels, out_channels=256, mid_channels=256, dilations=(12, 24, 36), conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), separable_conv=False):
        super().__init__()
        if separable_conv:
            conv_module = DepthwiseSeparableConvModule
        else:
            conv_module = ConvModule
        modules = []
        modules.append(ConvModule(in_channels, mid_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        for dilation in dilations:
            modules.append(conv_module(in_channels, mid_channels, 3, padding=dilation, dilation=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        modules.append(ASPPPooling(in_channels, mid_channels, conv_cfg, norm_cfg, act_cfg))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(ConvModule(5 * mid_channels, out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg), nn.Dropout(0.5))

    def forward(self, x):
        """Forward function for ASPP module.

        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            Tensor: Output tensor.
        """
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ContextualAttentionModule(nn.Module):
    """Contexture attention module.

    The details of this module can be found in:
    Generative Image Inpainting with Contextual Attention

    Args:
        unfold_raw_kernel_size (int): Kernel size used in unfolding raw
            feature. Default: 4.
        unfold_raw_stride (int): Stride used in unfolding raw feature. Default:
            2.
        unfold_raw_padding (int): Padding used in unfolding raw feature.
            Default: 1.
        unfold_corr_kernel_size (int): Kernel size used in unfolding
            context for computing correlation maps. Default: 3.
        unfold_corr_stride (int): Stride used in unfolding context for
            computing correlation maps. Default: 1.
        unfold_corr_dilation (int): Dilation used in unfolding context for
            computing correlation maps. Default: 1.
        unfold_corr_padding (int): Padding used in unfolding context for
            computing correlation maps. Default: 1.
        scale (float): The resale factor used in resize input features.
            Default: 0.5.
        fuse_kernel_size (int): The kernel size used in fusion module.
            Default: 3.
        softmax_scale (float): The scale factor for softmax function.
            Default: 10.
        return_attention_score (bool): If True, the attention score will be
            returned. Default: True.
    """

    def __init__(self, unfold_raw_kernel_size=4, unfold_raw_stride=2, unfold_raw_padding=1, unfold_corr_kernel_size=3, unfold_corr_stride=1, unfold_corr_dilation=1, unfold_corr_padding=1, scale=0.5, fuse_kernel_size=3, softmax_scale=10, return_attention_score=True):
        super().__init__()
        self.unfold_raw_kernel_size = unfold_raw_kernel_size
        self.unfold_raw_stride = unfold_raw_stride
        self.unfold_raw_padding = unfold_raw_padding
        self.unfold_corr_kernel_size = unfold_corr_kernel_size
        self.unfold_corr_stride = unfold_corr_stride
        self.unfold_corr_dilation = unfold_corr_dilation
        self.unfold_corr_padding = unfold_corr_padding
        self.scale = scale
        self.fuse_kernel_size = fuse_kernel_size
        self.with_fuse_correlation = fuse_kernel_size > 1
        self.softmax_scale = softmax_scale
        self.return_attention_score = return_attention_score
        if self.with_fuse_correlation:
            assert fuse_kernel_size % 2 == 1
            fuse_kernel = torch.eye(fuse_kernel_size).view(1, 1, fuse_kernel_size, fuse_kernel_size)
            self.register_buffer('fuse_kernel', fuse_kernel)
            padding = int((fuse_kernel_size - 1) // 2)
            self.fuse_conv = partial(F.conv2d, padding=padding, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, context, mask=None):
        """Forward Function.

        Args:
            x (torch.Tensor): Tensor with shape (n, c, h, w).
            context (torch.Tensor): Tensor with shape (n, c, h, w).
            mask (torch.Tensor): Tensor with shape (n, 1, h, w). Default: None.

        Returns:
            tuple(torch.Tensor): Features after contextural attention.
        """
        raw_context = context
        raw_context_cols = self.im2col(raw_context, kernel_size=self.unfold_raw_kernel_size, stride=self.unfold_raw_stride, padding=self.unfold_raw_padding, normalize=False, return_cols=True)
        x = F.interpolate(x, scale_factor=self.scale)
        context = F.interpolate(context, scale_factor=self.scale)
        context_cols = self.im2col(context, kernel_size=self.unfold_corr_kernel_size, stride=self.unfold_corr_stride, padding=self.unfold_corr_padding, dilation=self.unfold_corr_dilation, normalize=True, return_cols=True)
        h_unfold, w_unfold = self.calculate_unfold_hw(context.size()[-2:], kernel_size=self.unfold_corr_kernel_size, stride=self.unfold_corr_stride, padding=self.unfold_corr_padding, dilation=self.unfold_corr_dilation)
        context_cols = context_cols.reshape(-1, *context_cols.shape[2:])
        correlation_map = self.patch_correlation(x, context_cols)
        if self.with_fuse_correlation:
            correlation_map = self.fuse_correlation_map(correlation_map, h_unfold, w_unfold)
        correlation_map = self.mask_correlation_map(correlation_map, mask=mask)
        attention_score = self.softmax(correlation_map * self.softmax_scale)
        raw_context_filter = raw_context_cols.reshape(-1, *raw_context_cols.shape[2:])
        output = self.patch_copy_deconv(attention_score, raw_context_filter)
        overlap_factor = self.calculate_overlap_factor(attention_score)
        output /= overlap_factor
        if self.return_attention_score:
            n, _, h_s, w_s = attention_score.size()
            attention_score = attention_score.view(n, h_unfold, w_unfold, h_s, w_s)
            return output, attention_score
        return output

    def patch_correlation(self, x, kernel):
        """Calculate patch correlation.

        Args:
            x (torch.Tensor): Input tensor.
            kernel (torch.Tensor): Kernel tensor.

        Returns:
            torch.Tensor: Tensor with shape of (n, l, h, w).
        """
        n, _, h_in, w_in = x.size()
        patch_corr = F.conv2d(x.view(1, -1, h_in, w_in), kernel, stride=self.unfold_corr_stride, padding=self.unfold_corr_padding, dilation=self.unfold_corr_dilation, groups=n)
        h_out, w_out = patch_corr.size()[-2:]
        return patch_corr.view(n, -1, h_out, w_out)

    def patch_copy_deconv(self, attention_score, context_filter):
        """Copy patches using deconv.

        Args:
            attention_score (torch.Tensor): Tensor with shape of (n, l , h, w).
            context_filter (torch.Tensor): Filter kernel.

        Returns:
            torch.Tensor: Tensor with shape of (n, c, h, w).
        """
        n, _, h, w = attention_score.size()
        attention_score = attention_score.view(1, -1, h, w)
        output = F.conv_transpose2d(attention_score, context_filter, stride=self.unfold_raw_stride, padding=self.unfold_raw_padding, groups=n)
        h_out, w_out = output.size()[-2:]
        return output.view(n, -1, h_out, w_out)

    def fuse_correlation_map(self, correlation_map, h_unfold, w_unfold):
        """Fuse correlation map.

        This operation is to fuse correlation map for increasing large
        consistent correlation regions.

        The mechanism behind this op is simple and easy to understand. A
        standard 'Eye' matrix will be applied as a filter on the correlation
        map in horizontal and vertical direction.

        The shape of input correlation map is (n, h_unfold*w_unfold, h, w).
        When adopting fusing, we will apply convolutional filter in the
        reshaped feature map with shape of (n, 1, h_unfold*w_fold, h*w).

        A simple specification for horizontal direction is shown below:

        .. code-block:: python

                   (h, (h, (h, (h,
                    0)  1)  2)  3)  ...
            (h, 0)
            (h, 1)      1
            (h, 2)          1
            (h, 3)              1
            ...

        """
        n, _, h_map, w_map = correlation_map.size()
        map_ = correlation_map.permute(0, 2, 3, 1)
        map_ = map_.reshape(n, h_map * w_map, h_unfold * w_unfold, 1)
        map_ = map_.permute(0, 3, 1, 2).contiguous()
        map_ = self.fuse_conv(map_, self.fuse_kernel)
        correlation_map = map_.view(n, h_unfold, w_unfold, h_map, w_map)
        map_ = correlation_map.permute(0, 2, 1, 4, 3).reshape(n, 1, h_unfold * w_unfold, h_map * w_map)
        map_ = self.fuse_conv(map_, self.fuse_kernel)
        correlation_map = map_.view(n, w_unfold, h_unfold, w_map, h_map).permute(0, 4, 3, 2, 1)
        correlation_map = correlation_map.reshape(n, -1, h_unfold, w_unfold)
        return correlation_map

    def calculate_unfold_hw(self, input_size, kernel_size=3, stride=1, dilation=1, padding=0):
        """Calculate (h, w) after unfolding

        The official implementation of `unfold` in pytorch will put the
        dimension (h, w) into `L`. Thus, this function is just to calculate the
        (h, w) according to the equation in:
        https://pytorch.org/docs/stable/nn.html#torch.nn.Unfold
        """
        h_in, w_in = input_size
        h_unfold = int((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        w_unfold = int((w_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        return h_unfold, w_unfold

    def calculate_overlap_factor(self, attention_score):
        """Calculate the overlap factor after applying deconv.

        Args:
            attention_score (torch.Tensor): The attention score with shape of
                (n, c, h, w).

        Returns:
            torch.Tensor: The overlap factor will be returned.
        """
        h, w = attention_score.shape[-2:]
        kernel_size = self.unfold_raw_kernel_size
        ones_input = torch.ones(1, 1, h, w)
        ones_filter = torch.ones(1, 1, kernel_size, kernel_size)
        overlap = F.conv_transpose2d(ones_input, ones_filter, stride=self.unfold_raw_stride, padding=self.unfold_raw_padding)
        overlap[overlap == 0] = 1.0
        return overlap

    def mask_correlation_map(self, correlation_map, mask):
        """Add mask weight for correlation map.

        Add a negative infinity number to the masked regions so that softmax
        function will result in 'zero' in those regions.

        Args:
            correlation_map (torch.Tensor): Correlation map with shape of
                (n, h_unfold*w_unfold, h_map, w_map).
            mask (torch.Tensor): Mask tensor with shape of (n, c, h, w). '1'
                in the mask indicates masked region while '0' indicates valid
                region.

        Returns:
            torch.Tensor: Updated correlation map with mask.
        """
        if mask is not None:
            mask = F.interpolate(mask, scale_factor=self.scale)
            mask_cols = self.im2col(mask, kernel_size=self.unfold_corr_kernel_size, stride=self.unfold_corr_stride, padding=self.unfold_corr_padding, dilation=self.unfold_corr_dilation)
            mask_cols = (mask_cols.sum(dim=1, keepdim=True) > 0).float()
            mask_cols = mask_cols.permute(0, 2, 1).reshape(mask.size(0), -1, 1, 1)
            mask_cols[mask_cols == 1] = -float('inf')
            correlation_map += mask_cols
        return correlation_map

    def im2col(self, img, kernel_size, stride=1, padding=0, dilation=1, normalize=False, return_cols=False):
        """Reshape image-style feature to columns.

        This function is used for unfold feature maps to columns. The
        details of this function can be found in:
        https://pytorch.org/docs/1.1.0/nn.html?highlight=unfold#torch.nn.Unfold

        Args:
            img (torch.Tensor): Features to be unfolded. The shape of this
                feature should be (n, c, h, w).
            kernel_size (int): In this function, we only support square kernel
                with same height and width.
            stride (int): Stride number in unfolding. Default: 1.
            padding (int): Padding number in unfolding. Default: 0.
            dilation (int): Dilation number in unfolding. Default: 1.
            normalize (bool): If True, the unfolded feature will be normalized.
                Default: False.
            return_cols (bool): The official implementation in PyTorch of
                unfolding will return features with shape of
                (n, c*$prod{kernel_size}$, L). If True, the features will be
                reshaped to (n, L, c, kernel_size, kernel_size). Otherwise,
                the results will maintain the shape as the official
                implementation.

        Returns:
            torch.Tensor: Unfolded columns. If `return_cols` is True, the                 shape of output tensor is                 `(n, L, c, kernel_size, kernel_size)`. Otherwise, the shape                 will be `(n, c*$prod{kernel_size}$, L)`.
        """
        img_unfold = F.unfold(img, kernel_size, stride=stride, padding=padding, dilation=dilation)
        if normalize:
            norm = torch.sqrt((img_unfold ** 2).sum(dim=1, keepdim=True))
            eps = torch.tensor([0.0001])
            img_unfold = img_unfold / torch.max(norm, eps)
        if return_cols:
            img_unfold_ = img_unfold.permute(0, 2, 1)
            n, num_cols = img_unfold_.size()[:2]
            img_cols = img_unfold_.view(n, num_cols, img.size(1), kernel_size, kernel_size)
            return img_cols
        return img_unfold


class SpatialTemporalEnsemble(nn.Module):
    """ Apply spatial and temporal ensemble and compute outputs

    Args:
        is_temporal_ensemble (bool, optional): Whether to apply ensemble
            temporally. If True, the sequence will also be flipped temporally.
            If the input is an image, this argument must be set to False.
            Default: False.

    """

    def __init__(self, is_temporal_ensemble=False):
        super().__init__()
        self.is_temporal_ensemble = is_temporal_ensemble

    def _transform(self, imgs, mode):
        """Apply spatial transform (flip, rotate) to the images.

        Args:
            imgs (torch.Tensor): The images to be transformed/
            mode (str): The mode of transform. Supported values are 'vertical',
                'horizontal', and 'transpose', corresponding to vertical flip,
                horizontal flip, and rotation, respectively.

        Returns:
            torch.Tensor: Output of the model with spatial ensemble applied.

        """
        is_single_image = False
        if imgs.ndim == 4:
            if self.is_temporal_ensemble:
                raise ValueError('"is_temporal_ensemble" must be False if the input is an image.')
            is_single_image = True
            imgs = imgs.unsqueeze(1)
        if mode == 'vertical':
            imgs = imgs.flip(4).clone()
        elif mode == 'horizontal':
            imgs = imgs.flip(3).clone()
        elif mode == 'transpose':
            imgs = imgs.permute(0, 1, 2, 4, 3).clone()
        if is_single_image:
            imgs = imgs.squeeze(1)
        return imgs

    def spatial_ensemble(self, imgs, model):
        """Apply spatial ensemble.

        Args:
            imgs (torch.Tensor): The images to be processed by the model. Its
                size should be either (n, t, c, h, w) or (n, c, h, w).
            model (nn.Module): The model to process the images.

        Returns:
            torch.Tensor: Output of the model with spatial ensemble applied.

        """
        img_list = [imgs.cpu()]
        for mode in ['vertical', 'horizontal', 'transpose']:
            img_list.extend([self._transform(t, mode) for t in img_list])
        output_list = [model(t).cpu() for t in img_list]
        for i in range(len(output_list)):
            if i > 3:
                output_list[i] = self._transform(output_list[i], 'transpose')
            if i % 4 > 1:
                output_list[i] = self._transform(output_list[i], 'horizontal')
            if i % 4 % 2 == 1:
                output_list[i] = self._transform(output_list[i], 'vertical')
        outputs = torch.stack(output_list, dim=0)
        outputs = outputs.mean(dim=0, keepdim=False)
        return outputs

    def forward(self, imgs, model):
        """Apply spatial and temporal ensemble.

        Args:
            imgs (torch.Tensor): The images to be processed by the model. Its
                size should be either (n, t, c, h, w) or (n, c, h, w).
            model (nn.Module): The model to process the images.

        Returns:
            torch.Tensor: Output of the model with spatial ensemble applied.

        """
        outputs = self.spatial_ensemble(imgs, model)
        if self.is_temporal_ensemble:
            outputs += self.spatial_ensemble(imgs.flip(1), model).flip(1)
            outputs *= 0.5
        return outputs


class SimpleGatedConvModule(nn.Module):
    """Simple Gated Convolutional Module.

    This module is a simple gated convolutional module. The detailed formula
    is:

    .. math::
        y = \\phi(conv1(x)) * \\sigma(conv2(x)),

    where `phi` is the feature activation function and `sigma` is the gate
    activation function. In default, the gate activation function is sigmoid.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): The number of channels of the output feature. Note
            that `out_channels` in the conv module is doubled since this module
            contains two convolutions for feature and gate separately.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        feat_act_cfg (dict): Config dict for feature activation layer.
        gate_act_cfg (dict): Config dict for gate activation layer.
        kwargs (keyword arguments): Same as `ConvModule`.
    """

    def __init__(self, in_channels, out_channels, kernel_size, feat_act_cfg=dict(type='ELU'), gate_act_cfg=dict(type='Sigmoid'), **kwargs):
        super().__init__()
        kwargs_ = copy.deepcopy(kwargs)
        kwargs_['act_cfg'] = None
        self.with_feat_act = feat_act_cfg is not None
        self.with_gate_act = gate_act_cfg is not None
        self.conv = ConvModule(in_channels, out_channels * 2, kernel_size, **kwargs_)
        if self.with_feat_act:
            self.feat_act = build_activation_layer(feat_act_cfg)
        if self.with_gate_act:
            self.gate_act = build_activation_layer(gate_act_cfg)

    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        x = self.conv(x)
        x, gate = torch.split(x, x.size(1) // 2, dim=1)
        if self.with_feat_act:
            x = self.feat_act(x)
        if self.with_gate_act:
            gate = self.gate_act(gate)
        x = x * gate
        return x


class GCAModule(nn.Module):
    """Guided Contextual Attention Module.

    From https://arxiv.org/pdf/2001.04069.pdf.
    Based on https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting.
    This module use image feature map to augment the alpha feature map with
    guided contextual attention score.

    Image feature and alpha feature are unfolded to small patches and later
    used as conv kernel. Thus, we refer the unfolding size as kernel size.
    Image feature patches have a default kernel size 3 while the kernel size of
    alpha feature patches could be specified by `rate` (see `rate` below). The
    image feature patches are used to convolve with the image feature itself
    to calculate the contextual attention. Then the attention feature map is
    convolved by alpha feature patches to obtain the attention alpha feature.
    At last, the attention alpha feature is added to the input alpha feature.

    Args:
        in_channels (int): Input channels of the guided contextual attention
            module.
        out_channels (int): Output channels of the guided contextual attention
            module.
        kernel_size (int): Kernel size of image feature patches. Default 3.
        stride (int): Stride when unfolding the image feature. Default 1.
        rate (int): The downsample rate of image feature map. The corresponding
            kernel size and stride of alpha feature patches will be `rate x 2`
            and `rate`. It could be regarded as the granularity of the gca
            module. Default: 2.
        pad_args (dict): Parameters of padding when convolve image feature with
            image feature patches or alpha feature patches. Allowed keys are
            `mode` and `value`. See torch.nn.functional.pad() for more
            information. Default: dict(mode='reflect').
        interpolation (str): Interpolation method in upsampling and
            downsampling.
        penalty (float): Punishment hyperparameter to avoid a large correlation
            between each unknown patch and itself.
        eps (float): A small number to avoid dividing by 0 when calculating
            the normed image feature patch. Default: 1e-4.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, rate=2, pad_args=dict(mode='reflect'), interpolation='nearest', penalty=-10000.0, eps=0.0001):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.pad_args = pad_args
        self.interpolation = interpolation
        self.penalty = penalty
        self.eps = eps
        self.guidance_conv = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.out_conv = ConvModule(out_channels, out_channels, 1, norm_cfg=dict(type='BN'), act_cfg=None)
        self.init_weights()

    def init_weights(self):
        xavier_init(self.guidance_conv, distribution='uniform')
        xavier_init(self.out_conv.conv, distribution='uniform')
        constant_init(self.out_conv.norm, 0.001)

    def forward(self, img_feat, alpha_feat, unknown=None, softmax_scale=1.0):
        """Forward function of GCAModule.

        Args:
            img_feat (Tensor): Image feature map of shape
                (N, ori_c, ori_h, ori_w).
            alpha_feat (Tensor): Alpha feature map of shape
                (N, alpha_c, ori_h, ori_w).
            unknown (Tensor, optional): Unknown area map generated by trimap.
                If specified, this tensor should have shape
                (N, 1, ori_h, ori_w).
            softmax_scale (float, optional): The softmax scale of the attention
                if unknown area is not provided in forward. Default: 1.

        Returns:
            Tensor: The augmented alpha feature.
        """
        if alpha_feat.shape[2:4] != img_feat.shape[2:4]:
            raise ValueError(f'image feature size does not align with alpha feature size: image feature size {img_feat.shape[2:4]}, alpha feature size {alpha_feat.shape[2:4]}')
        if unknown is not None and unknown.shape[2:4] != img_feat.shape[2:4]:
            raise ValueError(f'image feature size does not align with unknown mask size: image feature size {img_feat.shape[2:4]}, unknown mask size {unknown.shape[2:4]}')
        img_feat = self.guidance_conv(img_feat)
        img_feat = F.interpolate(img_feat, scale_factor=1 / self.rate, mode=self.interpolation)
        unknown, softmax_scale = self.process_unknown_mask(unknown, img_feat, softmax_scale)
        img_ps, alpha_ps, unknown_ps = self.extract_feature_maps_patches(img_feat, alpha_feat, unknown)
        self_mask = self.get_self_correlation_mask(img_feat)
        img_groups = torch.split(img_feat, 1, dim=0)
        img_ps_groups = torch.split(img_ps, 1, dim=0)
        alpha_ps_groups = torch.split(alpha_ps, 1, dim=0)
        unknown_ps_groups = torch.split(unknown_ps, 1, dim=0)
        scale_groups = torch.split(softmax_scale, 1, dim=0)
        groups = img_groups, img_ps_groups, alpha_ps_groups, unknown_ps_groups, scale_groups
        out = []
        for img_i, img_ps_i, alpha_ps_i, unknown_ps_i, scale_i in zip(*groups):
            similarity_map = self.compute_similarity_map(img_i, img_ps_i)
            gca_score = self.compute_guided_attention_score(similarity_map, unknown_ps_i, scale_i, self_mask)
            out_i = self.propagate_alpha_feature(gca_score, alpha_ps_i)
            out.append(out_i)
        out = torch.cat(out, dim=0)
        out.reshape_as(alpha_feat)
        out = self.out_conv(out) + alpha_feat
        return out

    def extract_feature_maps_patches(self, img_feat, alpha_feat, unknown):
        """Extract image feature, alpha feature unknown patches.

        Args:
            img_feat (Tensor): Image feature map of shape
                (N, img_c, img_h, img_w).
            alpha_feat (Tensor): Alpha feature map of shape
                (N, alpha_c, ori_h, ori_w).
            unknown (Tensor, optional): Unknown area map generated by trimap of
                shape (N, 1, img_h, img_w).

        Returns:
            tuple: 3-tuple of

                ``Tensor``: Image feature patches of shape                     (N, img_h*img_w, img_c, img_ks, img_ks).

                ``Tensor``: Guided contextual attention alpha feature map.                     (N, img_h*img_w, alpha_c, alpha_ks, alpha_ks).

                ``Tensor``: Unknown mask of shape (N, img_h*img_w, 1, 1).
        """
        img_ks = self.kernel_size
        img_ps = self.extract_patches(img_feat, img_ks, self.stride)
        alpha_ps = self.extract_patches(alpha_feat, self.rate * 2, self.rate)
        unknown_ps = self.extract_patches(unknown, img_ks, self.stride)
        unknown_ps = unknown_ps.squeeze(dim=2)
        unknown_ps = unknown_ps.mean(dim=[2, 3], keepdim=True)
        return img_ps, alpha_ps, unknown_ps

    def compute_similarity_map(self, img_feat, img_ps):
        """Compute similarity between image feature patches.

        Args:
            img_feat (Tensor): Image feature map of shape
                (1, img_c, img_h, img_w).
            img_ps (Tensor): Image feature patches tensor of shape
                (1, img_h*img_w, img_c, img_ks, img_ks).

        Returns:
            Tensor: Similarity map between image feature patches with shape                 (1, img_h*img_w, img_h, img_w).
        """
        img_ps = img_ps[0]
        escape_NaN = torch.FloatTensor([self.eps])
        img_ps_normed = img_ps / torch.max(self.l2_norm(img_ps), escape_NaN)
        img_feat = self.pad(img_feat, self.kernel_size, self.stride)
        similarity_map = F.conv2d(img_feat, img_ps_normed)
        return similarity_map

    def compute_guided_attention_score(self, similarity_map, unknown_ps, scale, self_mask):
        """Compute guided attention score.

        Args:
            similarity_map (Tensor): Similarity map of image feature with shape
                (1, img_h*img_w, img_h, img_w).
            unknown_ps (Tensor): Unknown area patches tensor of shape
                (1, img_h*img_w, 1, 1).
            scale (Tensor): Softmax scale of known and unknown area:
                [unknown_scale, known_scale].
            self_mask (Tensor): Self correlation mask of shape
                (1, img_h*img_w, img_h, img_w). At (1, i*i, i, i) mask value
                equals -1e4 for i in [1, img_h*img_w] and other area is all
                zero.

        Returns:
            Tensor: Similarity map between image feature patches with shape                 (1, img_h*img_w, img_h, img_w).
        """
        unknown_scale, known_scale = scale[0]
        out = similarity_map * (unknown_scale * unknown_ps.gt(0.0).float() + known_scale * unknown_ps.le(0.0).float())
        out = out + self_mask * unknown_ps
        gca_score = F.softmax(out, dim=1)
        return gca_score

    def propagate_alpha_feature(self, gca_score, alpha_ps):
        """Propagate alpha feature based on guided attention score.

        Args:
            gca_score (Tensor): Guided attention score map of shape
                (1, img_h*img_w, img_h, img_w).
            alpha_ps (Tensor): Alpha feature patches tensor of shape
                (1, img_h*img_w, alpha_c, alpha_ks, alpha_ks).

        Returns:
            Tensor: Propagated alpha feature map of shape                 (1, alpha_c, alpha_h, alpha_w).
        """
        alpha_ps = alpha_ps[0]
        if self.rate == 1:
            gca_score = self.pad(gca_score, kernel_size=2, stride=1)
            alpha_ps = alpha_ps.permute(1, 0, 2, 3)
            out = F.conv2d(gca_score, alpha_ps) / 4.0
        else:
            out = F.conv_transpose2d(gca_score, alpha_ps, stride=self.rate, padding=1) / 4.0
        return out

    def process_unknown_mask(self, unknown, img_feat, softmax_scale):
        """Process unknown mask.

        Args:
            unknown (Tensor, optional): Unknown area map generated by trimap of
                shape (N, 1, ori_h, ori_w)
            img_feat (Tensor): The interpolated image feature map of shape
                (N, img_c, img_h, img_w).
            softmax_scale (float, optional): The softmax scale of the attention
                if unknown area is not provided in forward. Default: 1.

        Returns:
            tuple: 2-tuple of

                ``Tensor``: Interpolated unknown area map of shape                     (N, img_h*img_w, img_h, img_w).

                ``Tensor``: Softmax scale tensor of known and unknown area of                     shape (N, 2).
        """
        n, _, h, w = img_feat.shape
        if unknown is not None:
            unknown = unknown.clone()
            unknown = F.interpolate(unknown, scale_factor=1 / self.rate, mode=self.interpolation)
            unknown_mean = unknown.mean(dim=[2, 3])
            known_mean = 1 - unknown_mean
            unknown_scale = torch.clamp(torch.sqrt(unknown_mean / known_mean), 0.1, 10)
            known_scale = torch.clamp(torch.sqrt(known_mean / unknown_mean), 0.1, 10)
            softmax_scale = torch.cat([unknown_scale, known_scale], dim=1)
        else:
            unknown = torch.ones((n, 1, h, w))
            softmax_scale = torch.FloatTensor([softmax_scale, softmax_scale]).view(1, 2).repeat(n, 1)
        return unknown, softmax_scale

    def extract_patches(self, x, kernel_size, stride):
        """Extract feature patches.

        The feature map will be padded automatically to make sure the number of
        patches is equal to `(H / stride) * (W / stride)`.

        Args:
            x (Tensor): Feature map of shape (N, C, H, W).
            kernel_size (int): Size of each patches.
            stride (int): Stride between patches.

        Returns:
            Tensor: Extracted patches of shape                 (N, (H / stride) * (W / stride) , C, kernel_size, kernel_size).
        """
        n, c, _, _ = x.shape
        x = self.pad(x, kernel_size, stride)
        x = F.unfold(x, (kernel_size, kernel_size), stride=(stride, stride))
        x = x.permute(0, 2, 1)
        x = x.reshape(n, -1, c, kernel_size, kernel_size)
        return x

    def pad(self, x, kernel_size, stride):
        left = (kernel_size - stride + 1) // 2
        right = (kernel_size - stride) // 2
        pad = left, right, left, right
        return F.pad(x, pad, **self.pad_args)

    def get_self_correlation_mask(self, img_feat):
        _, _, h, w = img_feat.shape
        self_mask = F.one_hot(torch.arange(h * w).view(h, w), num_classes=int(h * w))
        self_mask = self_mask.permute(2, 0, 1).view(1, h * w, h, w)
        self_mask = self_mask * self.penalty
        return self_mask

    @staticmethod
    def l2_norm(x):
        x = x ** 2
        x = x.sum(dim=[1, 2, 3], keepdim=True)
        return torch.sqrt(x)


class UnetSkipConnectionBlock(nn.Module):
    """Construct a Unet submodule with skip connections, with the following
    structure: downsampling - `submodule` - upsampling.

    Args:
        outer_channels (int): Number of channels at the outer conv layer.
        inner_channels (int): Number of channels at the inner conv layer.
        in_channels (int): Number of channels in input images/features. If is
            None, equals to `outer_channels`. Default: None.
        submodule (UnetSkipConnectionBlock): Previously constructed submodule.
            Default: None.
        is_outermost (bool): Whether this module is the outermost module.
            Default: False.
        is_innermost (bool): Whether this module is the innermost module.
            Default: False.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='BN')`.
        use_dropout (bool): Whether to use dropout layers. Default: False.
    """

    def __init__(self, outer_channels, inner_channels, in_channels=None, submodule=None, is_outermost=False, is_innermost=False, norm_cfg=dict(type='BN'), use_dropout=False):
        super().__init__()
        assert not (is_outermost and is_innermost), "'is_outermost' and 'is_innermost' cannot be Trueat the same time."
        self.is_outermost = is_outermost
        assert isinstance(norm_cfg, dict), f"'norm_cfg' should be dict, butgot {type(norm_cfg)}"
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        use_bias = norm_cfg['type'] == 'IN'
        kernel_size = 4
        stride = 2
        padding = 1
        if in_channels is None:
            in_channels = outer_channels
        down_conv_cfg = dict(type='Conv2d')
        down_norm_cfg = norm_cfg
        down_act_cfg = dict(type='LeakyReLU', negative_slope=0.2)
        up_conv_cfg = dict(type='Deconv')
        up_norm_cfg = norm_cfg
        up_act_cfg = dict(type='ReLU')
        up_in_channels = inner_channels * 2
        up_bias = use_bias
        middle = [submodule]
        upper = []
        if is_outermost:
            down_act_cfg = None
            down_norm_cfg = None
            up_bias = True
            up_norm_cfg = None
            upper = [nn.Tanh()]
        elif is_innermost:
            down_norm_cfg = None
            up_in_channels = inner_channels
            middle = []
        else:
            upper = [nn.Dropout(0.5)] if use_dropout else []
        down = [ConvModule(in_channels=in_channels, out_channels=inner_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias, conv_cfg=down_conv_cfg, norm_cfg=down_norm_cfg, act_cfg=down_act_cfg, order=('act', 'conv', 'norm'))]
        up = [ConvModule(in_channels=up_in_channels, out_channels=outer_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=up_bias, conv_cfg=up_conv_cfg, norm_cfg=up_norm_cfg, act_cfg=up_act_cfg, order=('act', 'conv', 'norm'))]
        model = down + middle + up + upper
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.is_outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class ResidualBlockWithDropout(nn.Module):
    """Define a Residual Block with dropout layers.

    Ref:
    Deep Residual Learning for Image Recognition

    A residual block is a conv block with skip connections. A dropout layer is
    added between two common conv modules.

    Args:
        channels (int): Number of channels in the conv layer.
        padding_mode (str): The name of padding layer:
            'reflect' | 'replicate' | 'zeros'.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='IN')`.
        use_dropout (bool): Whether to use dropout layers. Default: True.
    """

    def __init__(self, channels, padding_mode, norm_cfg=dict(type='BN'), use_dropout=True):
        super().__init__()
        assert isinstance(norm_cfg, dict), f"'norm_cfg' should be dict, butgot {type(norm_cfg)}"
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        use_bias = norm_cfg['type'] == 'IN'
        block = [ConvModule(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=use_bias, norm_cfg=norm_cfg, padding_mode=padding_mode)]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        block += [ConvModule(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=use_bias, norm_cfg=norm_cfg, act_cfg=None, padding_mode=padding_mode)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        """Forward function. Add skip connections without final ReLU.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = x + self.block(x)
        return out


class ImgNormalize(nn.Conv2d):
    """Normalize images with the given mean and std value.

    Based on Conv2d layer, can work in GPU.

    Args:
        pixel_range (float): Pixel range of feature.
        img_mean (Tuple[float]): Image mean of each channel.
        img_std (Tuple[float]): Image std of each channel.
        sign (int): Sign of bias. Default -1.
    """

    def __init__(self, pixel_range, img_mean, img_std, sign=-1):
        assert len(img_mean) == len(img_std)
        num_channels = len(img_mean)
        super().__init__(num_channels, num_channels, kernel_size=1)
        std = torch.Tensor(img_std)
        self.weight.data = torch.eye(num_channels).view(num_channels, num_channels, 1, 1)
        self.weight.data.div_(std.view(num_channels, 1, 1, 1))
        self.bias.data = sign * pixel_range * torch.Tensor(img_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class LinearModule(nn.Module):
    """A linear block that contains linear/norm/activation layers.

    For low level vision, we add spectral norm and padding layer.

    Args:
        in_features (int): Same as nn.Linear.
        out_features (int): Same as nn.Linear.
        bias (bool): Same as nn.Linear.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
        inplace (bool): Whether to use inplace mode for activation.
        with_spectral_norm (bool): Whether use spectral norm in linear module.
        order (tuple[str]): The order of linear/activation layers. It is a
            sequence of "linear", "norm" and "act". Examples are
            ("linear", "act") and ("act", "linear").
    """

    def __init__(self, in_features, out_features, bias=True, act_cfg=dict(type='ReLU'), inplace=True, with_spectral_norm=False, order=('linear', 'act')):
        super().__init__()
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 2
        assert set(order) == set(['linear', 'act'])
        self.with_activation = act_cfg is not None
        self.with_bias = bias
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features
        if self.with_spectral_norm:
            self.linear = nn.utils.spectral_norm(self.linear)
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)
        self.init_weights()

    def init_weights(self):
        if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
            nonlinearity = 'leaky_relu'
            a = self.act_cfg.get('negative_slope', 0.01)
        else:
            nonlinearity = 'relu'
            a = 0
        kaiming_init(self.linear, a=a, nonlinearity=nonlinearity)

    def forward(self, x, activate=True):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of :math:`(n, *, c)`.
                Same as ``torch.nn.Linear``.
            activate (bool, optional): Whether to use activation layer.
                Defaults to True.

        Returns:
            torch.Tensor: Same as ``torch.nn.Linear``.
        """
        for layer in self.order:
            if layer == 'linear':
                x = self.linear(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x


class PartialConv2d(nn.Conv2d):
    """Implementation for partial convolution.

    Image Inpainting for Irregular Holes Using Partial Convolutions
    [https://arxiv.org/abs/1804.07723]

    Args:
        multi_channel (bool): If True, the mask is multi-channel. Otherwise,
            the mask is single-channel.
        eps (float): Need to be changed for mixed precision training.
            For mixed precision training, you need change 1e-8 to 1e-6.
    """

    def __init__(self, *args, multi_channel=False, eps=1e-08, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_channel = multi_channel
        self.eps = eps
        if self.multi_channel:
            out_channels, in_channels = self.out_channels, self.in_channels
        else:
            out_channels, in_channels = 1, 1
        self.register_buffer('weight_mask_updater', torch.ones(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.mask_kernel_numel = np.prod(self.weight_mask_updater.shape[1:4])
        self.mask_kernel_numel = self.mask_kernel_numel.item()

    def forward(self, input, mask=None, return_mask=True):
        """Forward function for partial conv2d.

        Args:
            input (torch.Tensor): Tensor with shape of (n, c, h, w).
            mask (torch.Tensor): Tensor with shape of (n, c, h, w) or
                (n, 1, h, w). If mask is not given, the function will
                work as standard conv2d. Default: None.
            return_mask (bool): If True and mask is not None, the updated
                mask will be returned. Default: True.

        Returns:
            torch.Tensor : Results after partial conv.            torch.Tensor : Updated mask will be returned if mask is given and                 ``return_mask`` is True.
        """
        assert input.dim() == 4
        if mask is not None:
            assert mask.dim() == 4
            if self.multi_channel:
                assert mask.shape[1] == input.shape[1]
            else:
                assert mask.shape[1] == 1
        if mask is not None:
            with torch.no_grad():
                updated_mask = F.conv2d(mask, self.weight_mask_updater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation)
                mask_ratio = self.mask_kernel_numel / (updated_mask + self.eps)
                updated_mask = torch.clamp(updated_mask, 0, 1)
                mask_ratio = mask_ratio * updated_mask
        if mask is not None:
            input = input * mask
        raw_out = super().forward(input)
        if mask is not None:
            if self.bias is None:
                output = raw_out * mask_ratio
            else:
                bias_view = self.bias.view(1, self.out_channels, 1, 1)
                output = (raw_out - bias_view) * mask_ratio + bias_view
                output = output * updated_mask
        else:
            output = raw_out
        if return_mask and mask is not None:
            return output, updated_mask
        return output


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """
        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(self.in_channels, self.out_channels * scale_factor * scale_factor, self.upsample_kernel, padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class ConstantInput(nn.Module):

    def __init__(self, channel, size=4):
        super().__init__()
        if isinstance(size, int):
            size = [size, size]
        elif mmcv.is_seq_of(size, int):
            assert len(size) == 2, f'The length of size should be 2 but got {len(size)}'
        else:
            raise ValueError(f'Got invalid value in size, {size}')
        self.input = nn.Parameter(torch.randn(1, channel, *size))

    def forward(self, x):
        batch = x.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


class EqualizedLR:
    """Equalized Learning Rate.

    This trick is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    The general idea is to dynamically rescale the weight in training instead
    of in initializing so that the variance of the responses in each layer is
    guaranteed with some statistical properties.

    Note that this function is always combined with a convolution module which
    is initialized with :math:`\\mathcal{N}(0, 1)`.

    Args:
        name (str | optional): The name of weights. Defaults to 'weight'.
        mode (str, optional): The mode of computing ``fan`` which is the
            same as ``kaiming_init`` in pytorch. You can choose one from
            ['fan_in', 'fan_out']. Defaults to 'fan_in'.
    """

    def __init__(self, name='weight', gain=2 ** 0.5, mode='fan_in', lr_mul=1.0):
        self.name = name
        self.mode = mode
        self.gain = gain
        self.lr_mul = lr_mul

    def compute_weight(self, module):
        """Compute weight with equalized learning rate.

        Args:
            module (nn.Module): A module that is wrapped with equalized lr.

        Returns:
            torch.Tensor: Updated weight.
        """
        weight = getattr(module, self.name + '_orig')
        if weight.ndim == 5:
            fan = _calculate_correct_fan(weight[0], self.mode)
        else:
            assert weight.ndim <= 4
            fan = _calculate_correct_fan(weight, self.mode)
        weight = weight * torch.tensor(self.gain, device=weight.device) * torch.sqrt(torch.tensor(1.0 / fan, device=weight.device)) * self.lr_mul
        return weight

    def __call__(self, module, inputs):
        """Standard interface for forward pre hooks."""
        setattr(module, self.name, self.compute_weight(module))

    @staticmethod
    def apply(module, name, gain=2 ** 0.5, mode='fan_in', lr_mul=1.0):
        """Apply function.

        This function is to register an equalized learning rate hook in an
        ``nn.Module``.

        Args:
            module (nn.Module): Module to be wrapped.
            name (str | optional): The name of weights. Defaults to 'weight'.
            mode (str, optional): The mode of computing ``fan`` which is the
                same as ``kaiming_init`` in pytorch. You can choose one from
                ['fan_in', 'fan_out']. Defaults to 'fan_in'.

        Returns:
            nn.Module: Module that is registered with equalized lr hook.
        """
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, EqualizedLR):
                raise RuntimeError(f'Cannot register two equalized_lr hooks on the same parameter {name} in {module} module.')
        fn = EqualizedLR(name, gain=gain, mode=mode, lr_mul=lr_mul)
        weight = module._parameters[name]
        delattr(module, name)
        module.register_parameter(name + '_orig', weight)
        setattr(module, name, weight.data)
        module.register_forward_pre_hook(fn)
        return fn


def equalized_lr(module, name='weight', gain=2 ** 0.5, mode='fan_in', lr_mul=1.0):
    """Equalized Learning Rate.

    This trick is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    The general idea is to dynamically rescale the weight in training instead
    of in initializing so that the variance of the responses in each layer is
    guaranteed with some statistical properties.

    Note that this function is always combined with a convolution module which
    is initialized with :math:`\\mathcal{N}(0, 1)`.

    Args:
        module (nn.Module): Module to be wrapped.
        name (str | optional): The name of weights. Defaults to 'weight'.
        mode (str, optional): The mode of computing ``fan`` which is the
            same as ``kaiming_init`` in pytorch. You can choose one from
            ['fan_in', 'fan_out']. Defaults to 'fan_in'.

    Returns:
        nn.Module: Module that is registered with equalized lr hook.
    """
    EqualizedLR.apply(module, name, gain=gain, mode=mode, lr_mul=lr_mul)
    return module


class EqualizedLRLinearModule(nn.Linear):
    """Equalized LR LinearModule.

    In this module, we adopt equalized lr in ``nn.Linear``. The equalized
    learning rate is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Note that, the initialization of ``self.weight`` will be overwritten as
    :math:`\\mathcal{N}(0, 1)`.

    Args:
        equalized_lr_cfg (dict | None, optional): Config for ``EqualizedLR``.
            If ``None``, equalized learning rate is ignored. Defaults to
            dict(mode='fan_in').
    """

    def __init__(self, *args, equalized_lr_cfg=dict(mode='fan_in'), **kwargs):
        super(EqualizedLRLinearModule, self).__init__(*args, **kwargs)
        self.with_equlized_lr = equalized_lr_cfg is not None
        if self.with_equlized_lr:
            self.lr_mul = equalized_lr_cfg.get('lr_mul', 1.0)
        else:
            self.lr_mul = 1.0
        if self.with_equlized_lr:
            equalized_lr(self, **equalized_lr_cfg)
            self._init_linear_weights()

    def _init_linear_weights(self):
        """Initialize linear weights as described in PGGAN."""
        nn.init.normal_(self.weight, 0, 1.0 / self.lr_mul)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class EqualLinearActModule(nn.Module):
    """Equalized LR Linear Module with Activation Layer.

    Args:
        nn ([type]): [description]
    """

    def __init__(self, *args, equalized_lr_cfg=dict(gain=1.0, lr_mul=1.0), bias=True, bias_init=0.0, act_cfg=None, **kwargs):
        super(EqualLinearActModule, self).__init__()
        self.with_activation = act_cfg is not None
        self.linear = EqualizedLRLinearModule(*args, bias=False, equalized_lr_cfg=equalized_lr_cfg, **kwargs)
        if equalized_lr_cfg is not None:
            self.lr_mul = equalized_lr_cfg.get('lr_mul', 1.0)
        else:
            self.lr_mul = 1.0
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.linear.out_features).fill_(bias_init))
        else:
            self.bias = None
        if self.with_activation:
            act_cfg = deepcopy(act_cfg)
            if act_cfg['type'] == 'fused_bias':
                self.act_type = act_cfg.pop('type')
                assert self.bias is not None
                self.activate = partial(fused_bias_leakyrelu, **act_cfg)
            else:
                self.act_type = 'normal'
                self.activate = build_activation_layer(act_cfg)
        else:
            self.act_type = None

    def forward(self, x):
        if x.ndim >= 3:
            x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        if self.with_activation and self.act_type == 'fused_bias':
            x = self.activate(x, self.bias * self.lr_mul)
        elif self.bias is not None and self.with_activation:
            x = self.activate(x + self.bias * self.lr_mul)
        elif self.bias is not None:
            x = x + self.bias * self.lr_mul
        elif self.with_activation:
            x = self.activate(x)
        return x


def _make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1):
        super(Blur, self).__init__()
        kernel = _make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * upsample_factor ** 2
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, x):
        return upfirdn2d(x, self.kernel, pad=self.pad)


class ModulatedConv2d(nn.Module):
    """Modulated Conv2d in StyleGANv2.

    Attention:

    #. ``style_bias`` is provided to check the difference between official TF
       implementation and other PyTorch implementation.
       In TF, Tero explicitly add the ``1.`` after style code, while unofficial
       implementation adopts bias initialization with ``1.``.
       Details can be found in:
       https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py#L214
       https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py#L99
    """

    def __init__(self, in_channels, out_channels, kernel_size, style_channels, demodulate=True, upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1], equalized_lr_cfg=dict(mode='fan_in', lr_mul=1.0, gain=1.0), style_mod_cfg=dict(bias_init=1.0), style_bias=0.0, eps=1e-08):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.style_channels = style_channels
        self.demodulate = demodulate
        assert isinstance(self.kernel_size, int) and (self.kernel_size >= 1 and self.kernel_size % 2 == 1)
        self.upsample = upsample
        self.downsample = downsample
        self.style_bias = style_bias
        self.eps = eps
        style_mod_cfg = dict() if style_mod_cfg is None else style_mod_cfg
        self.style_modulation = EqualLinearActModule(style_channels, in_channels, **style_mod_cfg)
        lr_mul_ = 1.0
        if equalized_lr_cfg is not None:
            lr_mul_ = equalized_lr_cfg.get('lr_mul', 1.0)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size).div_(lr_mul_))
        if upsample:
            factor = 2
            p = len(blur_kernel) - factor - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, (pad0, pad1), upsample_factor=factor)
        if downsample:
            factor = 2
            p = len(blur_kernel) - factor + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))
        if equalized_lr_cfg is not None:
            equalized_lr(self, **equalized_lr_cfg)
        self.padding = kernel_size // 2

    def forward(self, x, style):
        n, c, h, w = x.shape
        style = self.style_modulation(style).view(n, 1, c, 1, 1) + self.style_bias
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(n, self.out_channels, 1, 1, 1)
        weight = weight.view(n * self.out_channels, c, self.kernel_size, self.kernel_size)
        if self.upsample:
            x = x.reshape(1, n * c, h, w)
            weight = weight.view(n, self.out_channels, c, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(n * c, self.out_channels, self.kernel_size, self.kernel_size)
            x = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=n)
            x = x.reshape(n, self.out_channels, *x.shape[-2:])
            x = self.blur(x)
        elif self.downsample:
            x = self.blur(x)
            x = x.view(1, n * self.in_channels, *x.shape[-2:])
            x = F.conv2d(x, weight, stride=2, padding=0, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        else:
            x = x.view(1, n * c, h, w)
            x = F.conv2d(x, weight, stride=1, padding=self.padding, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        return x


class NoiseInjection(nn.Module):

    def __init__(self, noise_weight_init=0.0):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1).fill_(noise_weight_init))

    def forward(self, image, noise=None, return_noise=False):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        if return_noise:
            return image + self.weight * noise, noise
        return image + self.weight * noise


class ModulatedStyleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, style_channels, upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True, style_mod_cfg=dict(bias_init=1.0), style_bias=0.0):
        super(ModulatedStyleConv, self).__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size, style_channels, demodulate=demodulate, upsample=upsample, blur_kernel=blur_kernel, style_mod_cfg=style_mod_cfg, style_bias=style_bias)
        self.noise_injector = NoiseInjection()
        self.activate = FusedBiasLeakyReLU(out_channels)

    def forward(self, x, style, noise=None, return_noise=False):
        out = self.conv(x, style)
        if return_noise:
            out, noise = self.noise_injector(out, noise=noise, return_noise=return_noise)
        else:
            out = self.noise_injector(out, noise=noise, return_noise=return_noise)
        out = self.activate(out)
        if return_noise:
            return out, noise
        else:
            return out


class UpsampleUpFIRDn(nn.Module):

    def __init__(self, kernel, factor=2):
        super(UpsampleUpFIRDn, self).__init__()
        self.factor = factor
        kernel = _make_kernel(kernel) * factor ** 2
        self.register_buffer('kernel', kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = pad0, pad1

    def forward(self, x):
        out = upfirdn2d(x, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out


class ModulatedToRGB(nn.Module):

    def __init__(self, in_channels, style_channels, out_channels=3, upsample=True, blur_kernel=[1, 3, 3, 1], style_mod_cfg=dict(bias_init=1.0), style_bias=0.0):
        super(ModulatedToRGB, self).__init__()
        if upsample:
            self.upsample = UpsampleUpFIRDn(blur_kernel)
        self.conv = ModulatedConv2d(in_channels, out_channels=out_channels, kernel_size=1, style_channels=style_channels, demodulate=False, style_mod_cfg=style_mod_cfg, style_bias=style_bias)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.conv(x, style)
        out = out + self.bias
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


def pixel_norm(x, eps=1e-06):
    """Pixel Normalization.

    This normalization is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Args:
        x (torch.Tensor): Tensor to be normalized.
        eps (float, optional): Epsilon to avoid dividing zero.
            Defaults to 1e-6.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    if version.parse(torch.__version__) >= version.parse('1.7.0'):
        norm = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)
    else:
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
    norm = norm / torch.sqrt(torch.tensor(x.shape[1]))
    return x / (norm + eps)


class PixelNorm(nn.Module):
    """Pixel Normalization.

    This module is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Args:
        eps (float, optional): Epsilon value. Defaults to 1e-6.
    """
    _abbr_ = 'pn'

    def __init__(self, in_channels=None, eps=1e-06):
        super(PixelNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        return pixel_norm(x, self.eps)


def get_module_device(module):
    """Get the device of a module.

    Args:
        module (nn.Module): A module contains the parameters.

    Returns:
        torch.device: The device of the module.
    """
    try:
        next(module.parameters())
    except StopIteration:
        raise ValueError('The input module should contain parameters.')
    if next(module.parameters()).is_cuda:
        return next(module.parameters()).get_device()
    else:
        return torch.device('cpu')


class StyleGANv2Generator(nn.Module):
    """StyleGAN2 Generator.

    This module comes from MMGeneration. In the future, this code will be
    removed and StyleGANv2 will be directly imported from mmgeneration.

    In StyleGAN2, we use a static architecture composing of a style mapping
    module and number of convolutional style blocks. More details can be found
    in: Analyzing and Improving the Image Quality of StyleGAN CVPR2020.

    You can load pretrained model through passing information into
    ``pretrained`` argument. We have already offered official weights as
    follows:

    - styelgan2-ffhq-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth  # noqa
    - stylegan2-horse-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-horse-config-f-official_20210327_173203-ef3e69ca.pth  # noqa
    - stylegan2-car-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-car-config-f-official_20210327_172340-8cfe053c.pth  # noqa
    - styelgan2-cat-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-cat-config-f-official_20210327_172444-15bc485b.pth  # noqa
    - stylegan2-church-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth  # noqa

    If you want to load the ema model, you can just use following codes:

    .. code-block:: python

        # ckpt_http is one of the valid path from http source
        generator = StyleGANv2Generator(1024, 512,
                                        pretrained=dict(
                                            ckpt_path=ckpt_http,
                                            prefix='generator_ema'))

    Of course, you can also download the checkpoint in advance and set
    ``ckpt_path`` with local path. If you just want to load the original
    generator (not the ema model), please set the prefix with 'generator'.

    Note that our implementation allows to generate BGR image, while the
    original StyleGAN2 outputs RGB images by default. Thus, we provide
    ``bgr2rgb`` argument to convert the image space.

    Args:
        out_size (int): The output size of the StyleGAN2 generator.
        style_channels (int): The number of channels for style code.
        num_mlps (int, optional): The number of MLP layers. Defaults to 8.
        channel_multiplier (int, optional): The multiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        lr_mlp (float, optional): The learning rate for the style mapping
            layer. Defaults to 0.01.
        default_style_mode (str, optional): The default mode of style mixing.
            In training, we defaultly adopt mixing style mode. However, in the
            evaluation, we use 'single' style mode. `['mix', 'single']` are
            currently supported. Defaults to 'mix'.
        eval_style_mode (str, optional): The evaluation mode of style mixing.
            Defaults to 'single'.
        mix_prob (float, optional): Mixing probability. The value should be
            in range of [0, 1]. Defaults to 0.9.
        pretrained (dict | None, optional): Information for pretained models.
            The necessary key is 'ckpt_path'. Besides, you can also provide
            'prefix' to load the generator part from the whole state dict.
            Defaults to None.
        bgr2rgb (bool, optional): Whether to flip the image channel dimension.
            Defaults to False.
    """

    def __init__(self, out_size, style_channels, num_mlps=8, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, default_style_mode='mix', eval_style_mode='single', mix_prob=0.9, pretrained=None, bgr2rgb=False):
        super(StyleGANv2Generator, self).__init__()
        self.out_size = out_size
        self.style_channels = style_channels
        self.num_mlps = num_mlps
        self.channel_multiplier = channel_multiplier
        self.lr_mlp = lr_mlp
        self._default_style_mode = default_style_mode
        self.default_style_mode = default_style_mode
        self.eval_style_mode = eval_style_mode
        self.mix_prob = mix_prob
        self.bgr2rgb = bgr2rgb
        mapping_layers = [PixelNorm()]
        for _ in range(num_mlps):
            mapping_layers.append(EqualLinearActModule(style_channels, style_channels, equalized_lr_cfg=dict(lr_mul=lr_mlp, gain=1.0), act_cfg=dict(type='fused_bias')))
        self.style_mapping = nn.Sequential(*mapping_layers)
        self.channels = {(4): 512, (8): 512, (16): 512, (32): 512, (64): 256 * channel_multiplier, (128): 128 * channel_multiplier, (256): 64 * channel_multiplier, (512): 32 * channel_multiplier, (1024): 16 * channel_multiplier}
        self.constant_input = ConstantInput(self.channels[4])
        self.conv1 = ModulatedStyleConv(self.channels[4], self.channels[4], kernel_size=3, style_channels=style_channels, blur_kernel=blur_kernel)
        self.to_rgb1 = ModulatedToRGB(self.channels[4], style_channels, upsample=False)
        self.log_size = int(np.log2(self.out_size))
        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        in_channels_ = self.channels[4]
        for i in range(3, self.log_size + 1):
            out_channels_ = self.channels[2 ** i]
            self.convs.append(ModulatedStyleConv(in_channels_, out_channels_, 3, style_channels, upsample=True, blur_kernel=blur_kernel))
            self.convs.append(ModulatedStyleConv(out_channels_, out_channels_, 3, style_channels, upsample=False, blur_kernel=blur_kernel))
            self.to_rgbs.append(ModulatedToRGB(out_channels_, style_channels, upsample=True))
            in_channels_ = out_channels_
        self.num_latents = self.log_size * 2 - 2
        self.num_injected_noises = self.num_latents - 1
        for layer_idx in range(self.num_injected_noises):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.register_buffer(f'injected_noise_{layer_idx}', torch.randn(*shape))
        if pretrained is not None:
            self._load_pretrained_model(**pretrained)

    def _load_pretrained_model(self, ckpt_path, prefix='', map_location='cpu', strict=True):
        state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path, map_location)
        self.load_state_dict(state_dict, strict=strict)
        mmcv.print_log(f'Load pretrained model from {ckpt_path}', 'mmedit')

    def train(self, mode=True):
        if mode:
            if self.default_style_mode != self._default_style_mode:
                mmcv.print_log(f'Switch to train style mode: {self._default_style_mode}', 'mmgen')
            self.default_style_mode = self._default_style_mode
        else:
            if self.default_style_mode != self.eval_style_mode:
                mmcv.print_log(f'Switch to evaluation style mode: {self.eval_style_mode}', 'mmgen')
            self.default_style_mode = self.eval_style_mode
        return super(StyleGANv2Generator, self).train(mode)

    def make_injected_noise(self):
        device = get_module_device(self)
        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
        return noises

    def get_mean_latent(self, num_samples=4096, **kwargs):
        return get_mean_latent(self, num_samples, **kwargs)

    def style_mixing(self, n_source, n_target, inject_index=1, truncation_latent=None, truncation=0.7):
        return style_mixing(self, n_source=n_source, n_target=n_target, inject_index=inject_index, truncation=truncation, truncation_latent=truncation_latent, style_channels=self.style_channels)

    def forward(self, styles, num_batches=-1, return_noise=False, return_latents=False, inject_index=None, truncation=1, truncation_latent=None, input_is_latent=False, injected_noise=None, randomize_noise=True):
        """Forward function.

        This function has been integrated with the truncation trick. Please
        refer to the usage of `truncation` and `truncation_latent`.

        Args:
            styles (torch.Tensor | list[torch.Tensor] | callable | None): In
                StyleGAN2, you can provide noise tensor or latent tensor. Given
                a list containing more than one noise or latent tensors, style
                mixing trick will be used in training. Of course, You can
                directly give a batch of noise through a ``torch.Tensor`` or
                offer a callable function to sample a batch of noise data.
                Otherwise, the ``None`` indicates to use the default noise
                sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            return_latents (bool, optional): If True, ``latent`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            inject_index (int | None, optional): The index number for mixing
                style codes. Defaults to None.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncation trick will be adopted. Defaults to 1.
            truncation_latent (torch.Tensor, optional): Mean truncation latent.
                Defaults to None.
            input_is_latent (bool, optional): If `True`, the input tensor is
                the latent tensor. Defaults to False.
            injected_noise (torch.Tensor | None, optional): Given a tensor, the
                random noise will be fixed as this input injected noise.
                Defaults to None.
            randomize_noise (bool, optional): If `False`, images are sampled
                with the buffered noise tensor injected to the style conv
                block. Defaults to True.

        Returns:
            torch.Tensor | dict: Generated image tensor or dictionary
                containing more data.
        """
        if isinstance(styles, torch.Tensor):
            assert styles.shape[1] == self.style_channels
            styles = [styles]
        elif mmcv.is_seq_of(styles, torch.Tensor):
            for t in styles:
                assert t.shape[-1] == self.style_channels
        elif callable(styles):
            device = get_module_device(self)
            noise_generator = styles
            assert num_batches > 0
            if self.default_style_mode == 'mix' and random.random() < self.mix_prob:
                styles = [noise_generator((num_batches, self.style_channels)) for _ in range(2)]
            else:
                styles = [noise_generator((num_batches, self.style_channels))]
            styles = [s for s in styles]
        else:
            device = get_module_device(self)
            assert num_batches > 0 and not input_is_latent
            if self.default_style_mode == 'mix' and random.random() < self.mix_prob:
                styles = [torch.randn((num_batches, self.style_channels)) for _ in range(2)]
            else:
                styles = [torch.randn((num_batches, self.style_channels))]
            styles = [s for s in styles]
        if not input_is_latent:
            noise_batch = styles
            styles = [self.style_mapping(s) for s in styles]
        else:
            noise_batch = None
        if injected_noise is None:
            if randomize_noise:
                injected_noise = [None] * self.num_injected_noises
            else:
                injected_noise = [getattr(self, f'injected_noise_{i}') for i in range(self.num_injected_noises)]
        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_t
        if len(styles) < 2:
            inject_index = self.num_latents
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latents - 1)
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latents - inject_index, 1)
            latent = torch.cat([latent, latent2], 1)
        out = self.constant_input(latent)
        out = self.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        _index = 1
        for up_conv, conv, noise1, noise2, to_rgb in zip(self.convs[::2], self.convs[1::2], injected_noise[1::2], injected_noise[2::2], self.to_rgbs):
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)
            _index += 2
        img = skip
        if self.bgr2rgb:
            img = torch.flip(img, dims=1)
        if return_latents or return_noise:
            output_dict = dict(fake_img=img, latent=latent, inject_index=inject_index, noise_batch=noise_batch)
            return output_dict
        else:
            return img


class ConvDownLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False, blur_kernel=[1, 3, 3, 1], bias=True, act_cfg=dict(type='fused_bias')):
        layers = []
        if downsample:
            factor = 2
            p = len(blur_kernel) - factor + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2
        self.with_fused_bias = act_cfg is not None and act_cfg.get('type') == 'fused_bias'
        if self.with_fused_bias:
            conv_act_cfg = None
        else:
            conv_act_cfg = act_cfg
        layers.append(EqualizedLRConvModule(in_channels, out_channels, kernel_size, padding=self.padding, stride=stride, bias=bias and not self.with_fused_bias, norm_cfg=None, act_cfg=conv_act_cfg, equalized_lr_cfg=dict(mode='fan_in', gain=1.0)))
        if self.with_fused_bias:
            layers.append(FusedBiasLeakyReLU(out_channels))
        super(ConvDownLayer, self).__init__(*layers)


class ModMBStddevLayer(nn.Module):
    """Modified MiniBatch Stddev Layer.

    This layer is modified from ``MiniBatchStddevLayer`` used in PGGAN. In
    StyleGAN2, the authors add a new feature, `channel_groups`, into this
    layer.
    """

    def __init__(self, group_size=4, channel_groups=1, sync_groups=None, eps=1e-08):
        super(ModMBStddevLayer, self).__init__()
        self.group_size = group_size
        self.eps = eps
        self.channel_groups = channel_groups
        self.sync_groups = group_size if sync_groups is None else sync_groups

    def forward(self, x):
        assert x.shape[0] <= self.group_size or x.shape[0] % self.group_size == 0, f'Batch size be smaller than or equal to group size. Otherwise, batch size should be divisible by the group size.But got batch size {x.shape[0]}, group size {self.group_size}'
        assert x.shape[1] % self.channel_groups == 0, f'"channel_groups" must be divided by the feature channels. channel_groups: {self.channel_groups}, feature channels: {x.shape[1]}'
        n, c, h, w = x.shape
        group_size = min(n, self.group_size)
        y = torch.reshape(x, (group_size, -1, self.channel_groups, c // self.channel_groups, h, w))
        y = torch.var(y, dim=0, unbiased=False)
        y = torch.sqrt(y + self.eps)
        y = y.mean(dim=(2, 3, 4), keepdim=True).squeeze(2)
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, blur_kernel=[1, 3, 3, 1]):
        super(ResBlock, self).__init__()
        self.conv1 = ConvDownLayer(in_channels, in_channels, 3, blur_kernel=blur_kernel)
        self.conv2 = ConvDownLayer(in_channels, out_channels, 3, downsample=True, blur_kernel=blur_kernel)
        self.skip = ConvDownLayer(in_channels, out_channels, 1, downsample=True, act_cfg=None, bias=False, blur_kernel=blur_kernel)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        return out


class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 Discriminator.

    This module comes from MMGeneration. In the future, this code will be
    removed and StyleGANv2 will be directly imported from mmgeneration.

    The architecture of this discriminator is proposed in StyleGAN2. More
    details can be found in: Analyzing and Improving the Image Quality of
    StyleGAN CVPR2020.

    You can load pretrained model through passing information into
    ``pretrained`` argument. We have already offered official weights as
    follows:

    - styelgan2-ffhq-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth  # noqa
    - stylegan2-horse-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-horse-config-f-official_20210327_173203-ef3e69ca.pth  # noqa
    - stylegan2-car-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-car-config-f-official_20210327_172340-8cfe053c.pth  # noqa
    - styelgan2-cat-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-cat-config-f-official_20210327_172444-15bc485b.pth  # noqa
    - stylegan2-church-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth  # noqa

    If you want to load the ema model, you can just use following codes:

    .. code-block:: python

        # ckpt_http is one of the valid path from http source
        discriminator = StyleGAN2Discriminator(1024, 512,
                                               pretrained=dict(
                                                   ckpt_path=ckpt_http,
                                                   prefix='discriminator'))

    Of course, you can also download the checkpoint in advance and set
    ``ckpt_path`` with local path.

    Note that our implementation adopts BGR image as input, while the
    original StyleGAN2 provides RGB images to the discriminator. Thus, we
    provide ``bgr2rgb`` argument to convert the image space. If your images
    follow the RGB order, please set it to ``True`` accordingly.

    Args:
        in_size (int): The input size of images.
        channel_multiplier (int, optional): The multiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        mbstd_cfg (dict, optional): Configs for minibatch-stddev layer.
            Defaults to dict(group_size=4, channel_groups=1).
        pretrained (dict | None, optional): Information for pretained models.
            The necessary key is 'ckpt_path'. Besides, you can also provide
            'prefix' to load the generator part from the whole state dict.
            Defaults to None.
        bgr2rgb (bool, optional): Whether to flip the image channel dimension.
            Defaults to False.
    """

    def __init__(self, in_size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], mbstd_cfg=dict(group_size=4, channel_groups=1), pretrained=None, bgr2rgb=False):
        super(StyleGAN2Discriminator, self).__init__()
        self.bgr2rgb = bgr2rgb
        channels = {(4): 512, (8): 512, (16): 512, (32): 512, (64): 256 * channel_multiplier, (128): 128 * channel_multiplier, (256): 64 * channel_multiplier, (512): 32 * channel_multiplier, (1024): 16 * channel_multiplier}
        log_size = int(np.log2(in_size))
        in_channels = channels[in_size]
        convs = [ConvDownLayer(3, channels[in_size], 1)]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channels, out_channel, blur_kernel))
            in_channels = out_channel
        self.convs = nn.Sequential(*convs)
        self.mbstd_layer = ModMBStddevLayer(**mbstd_cfg)
        self.final_conv = ConvDownLayer(in_channels + 1, channels[4], 3)
        self.final_linear = nn.Sequential(EqualLinearActModule(channels[4] * 4 * 4, channels[4], act_cfg=dict(type='fused_bias')), EqualLinearActModule(channels[4], 1))
        if pretrained is not None:
            self._load_pretrained_model(**pretrained)

    def _load_pretrained_model(self, ckpt_path, prefix='', map_location='cpu', strict=True):
        state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path, map_location)
        self.load_state_dict(state_dict, strict=strict)
        mmcv.print_log(f'Load pretrained model from {ckpt_path}', 'mmedit')

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Predict score for the input image.
        """
        if self.bgr2rgb:
            x = torch.flip(x, dims=1)
        x = self.convs(x)
        x = self.mbstd_layer(x)
        x = self.final_conv(x)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)
        return x


class DonwsampleUpFIRDn(nn.Module):

    def __init__(self, kernel, factor=2):
        super(DonwsampleUpFIRDn, self).__init__()
        self.factor = factor
        kernel = _make_kernel(kernel)
        self.register_buffer('kernel', kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2
        self.pad = pad0, pad1

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out


class ModulatedPEConv2d(nn.Module):
    """Modulated Conv2d in StyleGANv2.

    Attention:

    #. ``style_bias`` is provided to check the difference between official TF
       implementation and other PyTorch implementation.
       In TF, Tero explicitly add the ``1.`` after style code, while unofficial
       implementation adopts bias initialization with ``1.``.
       Details can be found in:
       https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py#L214
       https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py#L99
    """

    def __init__(self, in_channels, out_channels, kernel_size, style_channels, demodulate=True, upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1], equalized_lr_cfg=dict(mode='fan_in', lr_mul=1.0, gain=1.0), style_mod_cfg=dict(bias_init=1.0), style_bias=0.0, eps=1e-08, no_pad=False, deconv2conv=False, interp_pad=None, up_config=dict(scale_factor=2, mode='nearest'), up_after_conv=False):
        super(ModulatedPEConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.style_channels = style_channels
        self.demodulate = demodulate
        assert isinstance(self.kernel_size, int) and (self.kernel_size >= 1 and self.kernel_size % 2 == 1)
        self.upsample = upsample
        self.downsample = downsample
        self.style_bias = style_bias
        self.eps = eps
        self.no_pad = no_pad
        self.deconv2conv = deconv2conv
        self.interp_pad = interp_pad
        self.with_interp_pad = interp_pad is not None
        self.up_config = deepcopy(up_config)
        self.up_after_conv = up_after_conv
        style_mod_cfg = dict() if style_mod_cfg is None else style_mod_cfg
        self.style_modulation = EqualLinearActModule(style_channels, in_channels, **style_mod_cfg)
        lr_mul_ = 1.0
        if equalized_lr_cfg is not None:
            lr_mul_ = equalized_lr_cfg.get('lr_mul', 1.0)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size).div_(lr_mul_))
        if upsample and not self.deconv2conv:
            factor = 2
            p = len(blur_kernel) - factor - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, (pad0, pad1), upsample_factor=factor)
        if downsample:
            factor = 2
            p = len(blur_kernel) - factor + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))
        if equalized_lr_cfg is not None:
            equalized_lr(self, **equalized_lr_cfg)
        self.padding = kernel_size // 2 if not no_pad else 0

    def forward(self, x, style):
        n, c, h, w = x.shape
        style = self.style_modulation(style).view(n, 1, c, 1, 1) + self.style_bias
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(n, self.out_channels, 1, 1, 1)
        weight = weight.view(n * self.out_channels, c, self.kernel_size, self.kernel_size)
        if self.upsample and not self.deconv2conv:
            x = x.reshape(1, n * c, h, w)
            weight = weight.view(n, self.out_channels, c, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(n * c, self.out_channels, self.kernel_size, self.kernel_size)
            x = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=n)
            x = x.reshape(n, self.out_channels, *x.shape[-2:])
            x = self.blur(x)
        elif self.upsample and self.deconv2conv:
            if self.up_after_conv:
                x = x.reshape(1, n * c, h, w)
                x = F.conv2d(x, weight, padding=self.padding, groups=n)
                x = x.view(n, self.out_channels, *x.shape[2:4])
            if self.with_interp_pad:
                h_, w_ = x.shape[-2:]
                up_cfg_ = deepcopy(self.up_config)
                up_scale = up_cfg_.pop('scale_factor')
                size_ = h_ * up_scale + self.interp_pad, w_ * up_scale + self.interp_pad
                x = F.interpolate(x, size=size_, **up_cfg_)
            else:
                x = F.interpolate(x, **self.up_config)
            if not self.up_after_conv:
                h_, w_ = x.shape[-2:]
                x = x.view(1, n * c, h_, w_)
                x = F.conv2d(x, weight, padding=self.padding, groups=n)
                x = x.view(n, self.out_channels, *x.shape[2:4])
        elif self.downsample:
            x = self.blur(x)
            x = x.view(1, n * self.in_channels, *x.shape[-2:])
            x = F.conv2d(x, weight, stride=2, padding=0, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        else:
            x = x.view(1, n * c, h, w)
            x = F.conv2d(x, weight, stride=1, padding=self.padding, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        return x


class ModulatedPEStyleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, style_channels, upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True, style_mod_cfg=dict(bias_init=1.0), style_bias=0.0, **kwargs):
        super(ModulatedPEStyleConv, self).__init__()
        self.conv = ModulatedPEConv2d(in_channels, out_channels, kernel_size, style_channels, demodulate=demodulate, upsample=upsample, blur_kernel=blur_kernel, style_mod_cfg=style_mod_cfg, style_bias=style_bias, **kwargs)
        self.noise_injector = NoiseInjection()
        self.activate = FusedBiasLeakyReLU(out_channels)

    def forward(self, x, style, noise=None, return_noise=False):
        out = self.conv(x, style)
        if return_noise:
            out, noise = self.noise_injector(out, noise=noise, return_noise=return_noise)
        else:
            out = self.noise_injector(out, noise=noise, return_noise=return_noise)
        out = self.activate(out)
        if return_noise:
            return out, noise
        else:
            return out


class GaussianBlur(nn.Module):
    """A Gaussian filter which blurs a given tensor with a two-dimensional
    gaussian kernel by convolving it along each channel. Batch operation
    is supported.

    This function is modified from kornia.filters.gaussian:
    `<https://kornia.readthedocs.io/en/latest/_modules/kornia/filters/gaussian.html>`.

    Args:
        kernel_size (tuple[int]): The size of the kernel. Default: (71, 71).
        sigma (tuple[float]): The standard deviation of the kernel.
        Default (10.0, 10.0)

    Returns:
        Tensor: The Gaussian-blurred tensor.

    Shape:
        - input: Tensor with shape of (n, c, h, w)
        - output: Tensor with shape of (n, c, h, w)
    """

    def __init__(self, kernel_size=(71, 71), sigma=(10.0, 10.0)):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = self.compute_zero_padding(kernel_size)
        self.kernel = self.get_2d_gaussian_kernel(kernel_size, sigma)

    @staticmethod
    def compute_zero_padding(kernel_size):
        """Compute zero padding tuple."""
        padding = [((ks - 1) // 2) for ks in kernel_size]
        return padding[0], padding[1]

    def get_2d_gaussian_kernel(self, kernel_size, sigma):
        """Get the two-dimensional Gaussian filter matrix coefficients.

        Args:
            kernel_size (tuple[int]): Kernel filter size in the x and y
                                      direction. The kernel sizes
                                      should be odd and positive.
            sigma (tuple[int]): Gaussian standard deviation in
                                the x and y direction.

        Returns:
            kernel_2d (Tensor): A 2D torch tensor with gaussian filter
                                matrix coefficients.
        """
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise TypeError('kernel_size must be a tuple of length two. Got {}'.format(kernel_size))
        if not isinstance(sigma, tuple) or len(sigma) != 2:
            raise TypeError('sigma must be a tuple of length two. Got {}'.format(sigma))
        kernel_size_x, kernel_size_y = kernel_size
        sigma_x, sigma_y = sigma
        kernel_x = self.get_1d_gaussian_kernel(kernel_size_x, sigma_x)
        kernel_y = self.get_1d_gaussian_kernel(kernel_size_y, sigma_y)
        kernel_2d = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
        return kernel_2d

    def get_1d_gaussian_kernel(self, kernel_size, sigma):
        """Get the Gaussian filter coefficients in one dimension (x or y direction).

        Args:
            kernel_size (int): Kernel filter size in x or y direction.
                               Should be odd and positive.
            sigma (float): Gaussian standard deviation in x or y direction.

        Returns:
            kernel_1d (Tensor): A 1D torch tensor with gaussian filter
                                coefficients in x or y direction.
        """
        if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size <= 0:
            raise TypeError('kernel_size must be an odd positive integer. Got {}'.format(kernel_size))
        kernel_1d = self.gaussian(kernel_size, sigma)
        return kernel_1d

    def gaussian(self, kernel_size, sigma):

        def gauss_arg(x):
            return -(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)
        gauss = torch.stack([torch.exp(torch.tensor(gauss_arg(x))) for x in range(kernel_size)])
        return gauss / gauss.sum()

    def forward(self, x):
        if not torch.is_tensor(x):
            raise TypeError('Input x type is not a torch.Tensor. Got {}'.format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.format(x.shape))
        _, c, _, _ = x.shape
        tmp_kernel = self.kernel.to(x.device)
        kernel = tmp_kernel.repeat(c, 1, 1, 1)
        return conv2d(x, kernel, padding=self.padding, stride=1, groups=c)


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
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight
        if self.gan_type == 'smgan':
            self.gaussian_blur = GaussianBlur()
        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan' or self.gan_type == 'smgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
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
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """
        if self.gan_type == 'wgan':
            return target_is_real
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False, mask=None):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the target is real or fake.
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
        elif self.gan_type == 'smgan':
            input_height, input_width = input.shape[2:]
            mask_height, mask_width = mask.shape[2:]
            if input_height != mask_height or input_width != mask_width:
                input = F.interpolate(input, size=(mask_height, mask_width), mode='bilinear', align_corners=True)
                target_label = self.get_target_label(input, target_is_real)
            if is_disc:
                if target_is_real:
                    target_label = target_label
                else:
                    target_label = self.gaussian_blur(mask).detach() if mask.is_cuda else self.gaussian_blur(mask).detach().cpu()
                loss = self.loss(input, target_label)
            else:
                loss = self.loss(input, target_label) * mask / mask.mean()
                loss = loss.mean()
        else:
            loss = self.loss(input, target_label)
        return loss if is_disc else loss * self.loss_weight


def gradient_penalty_loss(discriminator, real_data, fake_data, mask=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        mask (Tensor): Masks for inpainting. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1)
    interpolates = alpha * real_data + (1.0 - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones_like(disc_interpolates), create_graph=True, retain_graph=True, only_inputs=True)[0]
    if mask is not None:
        gradients = gradients * mask
    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    if mask is not None:
        gradients_penalty /= torch.mean(mask)
    return gradients_penalty


class GradientPenaltyLoss(nn.Module):
    """Gradient penalty loss for wgan-gp.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, discriminator, real_data, fake_data, mask=None):
        """Forward function.

        Args:
            discriminator (nn.Module): Network for the discriminator.
            real_data (Tensor): Real input data.
            fake_data (Tensor): Fake input data.
            mask (Tensor): Masks for inpainting. Default: None.

        Returns:
            Tensor: Loss.
        """
        loss = gradient_penalty_loss(discriminator, real_data, fake_data, mask=mask)
        return loss * self.loss_weight


class DiscShiftLoss(nn.Module):
    """Disc shift loss.

        Args:
            loss_weight (float, optional): Loss weight. Defaults to 1.0.
    """

    def __init__(self, loss_weight=0.1):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Tensor with shape (n, c, h, w)

        Returns:
            Tensor: Loss.
        """
        loss = torch.mean(x ** 2)
        return loss * self.loss_weight


class PerceptualVGG(nn.Module):
    """VGG network used in calculating perceptual loss.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): According to the name in this list,
            forward function will return the corresponding features. This
            list contains the name each layer in `vgg.feature`. An example
            of this list is ['4', '10'].
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image.
            Importantly, the input feature must in the range [0, 1].
            Default: True.
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'
    """

    def __init__(self, layer_name_list, vgg_type='vgg19', use_input_norm=True, pretrained='torchvision://vgg19'):
        super().__init__()
        if pretrained.startswith('torchvision://'):
            assert vgg_type in pretrained
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        _vgg = getattr(vgg, vgg_type)()
        self.init_weights(_vgg, pretrained)
        num_layers = max(map(int, layer_name_list)) + 1
        assert len(_vgg.features) >= num_layers
        self.vgg_layers = _vgg.features[:num_layers]
        if self.use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for v in self.vgg_layers.parameters():
            v.requires_grad = False

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}
        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_list:
                output[name] = x.clone()
        return output

    def init_weights(self, model, pretrained):
        """Init weights.

        Args:
            model (nn.Module): Models to be inited.
            pretrained (str): Path for pretrained weights.
        """
        logger = get_root_logger()
        load_checkpoint(model, pretrained, logger=logger)


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layers_weights (dict): The weight for each layer of vgg feature for
            perceptual loss. Here is an example: {'4': 1., '9': 1., '18': 1.},
            which means the 5th, 10th and 18th feature layer will be
            extracted with weight 1.0 in calculating losses.
        layers_weights_style (dict): The weight for each layer of vgg feature
            for style loss. If set to 'None', the weights are set equal to
            the weights for perceptual loss. Default: None.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 1.0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward function of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'.
        criterion (str): Criterion type. Options are 'l1' and 'mse'.
            Default: 'l1'.
    """

    def __init__(self, layer_weights, layer_weights_style=None, vgg_type='vgg19', use_input_norm=True, perceptual_weight=1.0, style_weight=1.0, norm_img=True, pretrained='torchvision://vgg19', criterion='l1'):
        super().__init__()
        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.layer_weights_style = layer_weights_style
        self.vgg = PerceptualVGG(layer_name_list=list(self.layer_weights.keys()), vgg_type=vgg_type, use_input_norm=use_input_norm, pretrained=pretrained)
        if self.layer_weights_style is not None and self.layer_weights_style != self.layer_weights:
            self.vgg_style = PerceptualVGG(layer_name_list=list(self.layer_weights_style.keys()), vgg_type=vgg_type, use_input_norm=use_input_norm, pretrained=pretrained)
        else:
            self.layer_weights_style = self.layer_weights
            self.vgg_style = None
        criterion = criterion.lower()
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported in this version.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.norm_img:
            x = (x + 1.0) * 0.5
            gt = (gt + 1.0) * 0.5
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None
        if self.style_weight > 0:
            if self.vgg_style is not None:
                x_features = self.vgg_style(x)
                gt_features = self.vgg_style(gt.detach())
            style_loss = 0
            for k in x_features.keys():
                style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(gt_features[k])) * self.layer_weights_style[k]
            style_loss *= self.style_weight
        else:
            style_loss = None
        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


_reduction_modes = ['none', 'mean', 'sum']


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()
    return loss.sum()


def mask_reduce_loss(loss, weight=None, reduction='mean', sample_wise=False):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            "none", "mean" and "sum". Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.

    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        if weight.size(1) == 1:
            weight = weight.expand_as(loss)
        eps = 1e-12
        if sample_wise:
            weight = weight.sum(dim=[1, 2, 3], keepdim=True)
            loss = (loss / (weight + eps)).sum() / weight.size(0)
        else:
            loss = loss.sum() / (weight.sum() + eps)
    return loss


def masked_loss(loss_func):
    """Create a masked version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @masked_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', sample_wise=False, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = mask_reduce_loss(loss, weight, reduction, sample_wise)
        return loss
    return wrapper


@masked_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated L1 loss.
    """
    return F.l1_loss(pred, target, reduction='none')


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction, sample_wise=self.sample_wise)


@masked_loss
def mse_loss(pred, target):
    """MSE loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated MSE loss.
    """
    return F.mse_loss(pred, target, reduction='none')


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction, sample_wise=self.sample_wise)


@masked_loss
def charbonnier_loss(pred, target, eps=1e-12):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    return torch.sqrt((pred - target) ** 2 + eps)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False, eps=1e-12):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction, sample_wise=self.sample_wise)


class MaskedTVLoss(L1Loss):
    """Masked TV loss.

        Args:
            loss_weight (float, optional): Loss weight. Defaults to 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def forward(self, pred, mask=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Tensor with shape of (n, c, h, w).
            mask (torch.Tensor, optional): Tensor with shape of (n, 1, h, w).
                Defaults to None.

        Returns:
            [type]: [description]
        """
        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=mask[:, :, :-1, :])
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=mask[:, :, :, :-1])
        loss = x_diff + y_diff
        return loss


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=256, K=32 * 256, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        idx_shuffle = torch.randperm(batch_size_all)
        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        _, q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        self._dequeue_and_enqueue(k)
        return logits, labels


class TensorRTRestorerGenerator(nn.Module):
    """Inner class for tensorrt restorer model inference

    Args:
        trt_file (str): The path to the tensorrt file.
        device_id (int): Which device to place the model.
    """

    def __init__(self, trt_file: 'str', device_id: 'int'):
        super().__init__()
        try:
            load_tensorrt_plugin()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv,                 you may have to build mmcv with TensorRT from source.')
        model = TRTWrapper(trt_file, input_names=['input'], output_names=['output'])
        self.device_id = device_id
        self.model = model

    def forward(self, x):
        with torch.device(self.device_id), torch.no_grad():
            seg_pred = self.model({'input': x})['output']
        seg_pred = seg_pred.detach().cpu()
        return seg_pred


class TensorRTRestorer(nn.Module):
    """A warper class for tensorrt restorer

    Args:
        base_model (Any): The base model build from config.
        trt_file (str): The path to the tensorrt file.
        device_id (int): Which device to place the model.
    """

    def __init__(self, base_model: 'Any', trt_file: 'str', device_id: 'int'):
        super().__init__()
        self.base_model = base_model
        restorer_generator = TensorRTRestorerGenerator(trt_file=trt_file, device_id=device_id)
        base_model.generator = restorer_generator

    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        return self.base_model(lq, gt=gt, test_mode=test_mode, **kwargs)


class TensorRTEditing(nn.Module):
    """A class for testing tensorrt deployment

    Args:
        trt_file (str): The path to the tensorrt file.
        cfg (Any): The configuration of the testing,             decided by the config file.
        device_id (int): Which device to place the model.
    """

    def __init__(self, trt_file: 'str', cfg: 'Any', device_id: 'int'):
        super().__init__()
        base_model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        if isinstance(base_model, BasicRestorer):
            WrapperClass = TensorRTRestorer
        self.wrapper = WrapperClass(base_model, trt_file, device_id)

    def forward(self, **kwargs):
        return self.wrapper(**kwargs)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CBAM,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CBAM_spatial_interpolation,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ChannelPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CharbonnierLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ConstantInput,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ContextualAttentionModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DiscShiftLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (EqualLinearActModule,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (EqualizedLRLinearModule,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GaussianBlur,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ImgNormalize,
     lambda: ([], {'pixel_range': 4, 'img_mean': [4, 4], 'img_std': [4, 4]}),
     lambda: ([torch.rand([4, 2, 64, 64])], {})),
    (L1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ModMBStddevLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ModulatedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 1, 'style_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {})),
    (ModulatedPEConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 1, 'style_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {})),
    (ModulatedToRGB,
     lambda: ([], {'in_channels': 4, 'style_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {})),
    (NoiseInjection,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PartialConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

