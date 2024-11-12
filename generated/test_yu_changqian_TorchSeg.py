
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


import functools


import torch.nn as nn


import time


import torch


import numpy as np


import torch.utils.data as data


import scipy.io as sio


import torch.distributed as dist


import torch.nn.functional as F


import torch.multiprocessing as mp


from torch.autograd import Variable


from torch.utils import data


from torch.cuda._utils import _get_device_index


from torch.autograd import Function


from torch.nn.parallel.data_parallel import DataParallel


import torch.cuda.comm as comm


from torch.nn.parallel._functions import Broadcast


import queue


from torch.utils.cpp_extension import load


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import collections


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.functional import batch_norm


from torch.nn.parallel._functions import ReduceAddCoalesced


import scipy.ndimage as nd


from collections import OrderedDict


from torch.optim.sgd import SGD


from torch.autograd.function import once_differentiable


from torch import nn


from collections import defaultdict


import torch.utils.model_zoo as model_zoo


from torch.utils.checkpoint import checkpoint


import torch.backends.cudnn as cudnn


import random


import math


from torch.nn.parallel import DistributedDataParallel


from functools import partial


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None, bn_eps=1e-05, bn_momentum=0.1, downsample=None, inplace=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm_layer=None, bn_eps=1e-05, bn_momentum=0.1, downsample=None, inplace=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

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
        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, norm_layer=nn.BatchNorm2d, bn_eps=1e-05, bn_momentum=0.1, deep_stem=False, stem_width=32, inplace=True):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNet, self).__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False), norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum), nn.ReLU(inplace=inplace), nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum), nn.ReLU(inplace=inplace), nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(stem_width * 2 if deep_stem else 64, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, 64, layers[0], inplace, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, 128, layers[1], inplace, stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, 256, layers[2], inplace, stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, norm_layer, 512, layers[3], inplace, stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)

    def _make_layer(self, block, norm_layer, planes, blocks, inplace=True, stride=1, bn_eps=1e-05, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), norm_layer(planes * block.expansion, eps=bn_eps, momentum=bn_momentum))
        layers = []
        layers.append(block(self.inplanes, planes, stride, norm_layer, bn_eps, bn_momentum, downsample, inplace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, bn_eps=bn_eps, bn_momentum=bn_momentum, inplace=inplace))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        blocks = []
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        x = self.layer4(x)
        blocks.append(x)
        return blocks


class ConvBnRelu(nn.Module):

    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1, groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-05, has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class SeparableConvBnRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, has_relu=True, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBnRelu, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.bn = norm_layer(in_channels)
        self.point_wise_cbr = ConvBnRelu(in_channels, out_channels, 1, 1, 0, has_bn=True, norm_layer=norm_layer, has_relu=has_relu, has_bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.point_wise_cbr(x)
        return x


class Block(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_out_channels, has_proj, stride, dilation=1, norm_layer=nn.BatchNorm2d):
        super(Block, self).__init__()
        self.has_proj = has_proj
        if has_proj:
            self.proj = SeparableConvBnRelu(in_channels, mid_out_channels * self.expansion, 3, stride, 1, has_relu=False, norm_layer=norm_layer)
        self.residual_branch = nn.Sequential(SeparableConvBnRelu(in_channels, mid_out_channels, 3, stride, dilation, dilation, has_relu=True, norm_layer=norm_layer), SeparableConvBnRelu(mid_out_channels, mid_out_channels, 3, 1, 1, has_relu=True, norm_layer=norm_layer), SeparableConvBnRelu(mid_out_channels, mid_out_channels * self.expansion, 3, 1, 1, has_relu=False, norm_layer=norm_layer))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        if self.has_proj:
            shortcut = self.proj(x)
        residual = self.residual_branch(x)
        output = self.relu(shortcut + residual)
        return output


class Xception(nn.Module):

    def __init__(self, block, layers, channels, norm_layer=nn.BatchNorm2d):
        super(Xception, self).__init__()
        self.in_channels = 8
        self.conv1 = ConvBnRelu(3, self.in_channels, 3, 2, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, layers[0], channels[0], stride=2)
        self.layer2 = self._make_layer(block, norm_layer, layers[1], channels[1], stride=2)
        self.layer3 = self._make_layer(block, norm_layer, layers[2], channels[2], stride=2)

    def _make_layer(self, block, norm_layer, blocks, mid_out_channels, stride=1):
        layers = []
        has_proj = True if stride > 1 else False
        layers.append(block(self.in_channels, mid_out_channels, has_proj, stride=stride, norm_layer=norm_layer))
        self.in_channels = mid_out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, mid_out_channels, has_proj=False, stride=1, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        blocks = []
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        return blocks


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelModel(DataParallel):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.DataParallelCriterion`.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = DataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """

    def gather(self, outputs, output_device):
        return outputs

    def replicate(self, module, device_ids):
        modules = super(DataParallelModel, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, "Previous result has't been fetched."
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()
            res = self._result
            self._result = None
            return res


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True
        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())
        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'
        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)
        for i in range(self.nr_slaves):
            assert self._queue.get() is True
        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


_ChildMessage = collections.namedtuple('Message', ['sum', 'ssum', 'sum_size'])


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class _batchnormtrain(Function):

    @staticmethod
    def forward(ctx, input, mean, std, gamma, beta):
        ctx.save_for_backward(input, mean, std, gamma, beta)
        if input.is_cuda:
            output = gpu.batchnorm_forward(input, mean, std, gamma, beta)
        else:
            output = cpu.batchnorm_forward(input, mean, std, gamma, beta)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        input, mean, std, gamma, beta = ctx.saved_variables
        if gradOutput.is_cuda:
            gradInput, gradMean, gradStd, gradGamma, gradBeta = gpu.batchnorm_backward(gradOutput, input, mean, std, gamma, beta, True)
        else:
            raise NotImplemented
        return gradInput, gradMean, gradStd, gradGamma, gradBeta


def batchnormtrain(input, mean, std, gamma, beta):
    """Applies Batch Normalization over a 3d input that is seen as a
    mini-batch.

    .. _encoding.batchnormtrain:

    .. math::

        y = \\frac{x - \\mu[x]}{ \\sqrt{var[x] + \\epsilon}} * \\gamma + \\beta

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    """
    return _batchnormtrain.apply(input, mean, std, gamma, beta)


class _sum_square(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input.is_cuda:
            xsum, xsqusum = gpu.sumsquare_forward(input)
        else:
            xsum, xsqusum = cpu.sumsquare_forward(input)
        return xsum, xsqusum

    @staticmethod
    def backward(ctx, gradSum, gradSquare):
        input, = ctx.saved_variables
        if input.is_cuda:
            gradInput = gpu.sumsquare_backward(input, gradSum, gradSquare)
        else:
            raise NotImplemented
        return gradInput


def sum_square(input):
    """Calculate sum of elements and sum of squares for Batch Normalization"""
    return _sum_square.apply(input)


class _SyncBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(_SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not self.training:
            return batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input_shape[0], self.num_features, -1)
        N = input.size(0) * input.size(2)
        xsum, xsqsum = sum_square(input)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(xsum, xsqsum, N))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(xsum, xsqsum, N))
        return batchnormtrain(input, mean, 1.0 / inv_std, self.weight, self.bias).view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._parallel_id = copy_id
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 + 2])))
        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        return mean, (bias_var + self.eps) ** -0.5


class BatchNorm2d(_SyncBatchNorm):
    """Cross-GPU Synchronized Batch normalization (SyncBN)

    Standard BN [1]_ implementation only normalize the data within each device (GPU).
    SyncBN normalizes the input within the whole mini-batch.
    We follow the sync-onece implmentation described in the paper [2]_ .
    Please see the design idea in the `notes <./notes/syncbn.html>`_.

    .. note::
        We adapt the awesome python API from another `PyTorch SyncBN Implementation
        <https://github.com/vacancy/Synchronized-BatchNorm-PyTorch>`_ and provide
        efficient CUDA backend.

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    The mean and standard-deviation are calculated per-channel over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Reference:
        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *ICML 2015*
        .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*

    Examples:
        >>> m = BatchNorm2d(100)
        >>> net = torch.nn.DataParallel(m)
        >>> encoding.parallel.patch_replication_callback(net)
        >>> output = net(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
        super(BatchNorm2d, self)._check_input_dim(input)


class BatchNorm1d(_SyncBatchNorm):
    """Please see the docs in :class:`encoding.nn.BatchNorm2d`"""

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))
        super(BatchNorm2d, self)._check_input_dim(input)


class BatchNorm3d(_SyncBatchNorm):
    """Please see the docs in :class:`encoding.nn.BatchNorm2d`"""

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        super(BatchNorm3d, self)._check_input_dim(input)


class SigmoidFocalLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target, gamma=2.0, alpha=0.25, reduction='mean'):
        ctx.save_for_backward(input, target)
        num_classes = input.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha
        loss = sigmoid_focal_loss_cuda.forward(input, target, num_classes, gamma, alpha)
        reduction_enum = F._Reduction.get_enum(reduction)
        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        input, target = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_input = sigmoid_focal_loss_cuda.backward(input, target, d_loss, num_classes, gamma, alpha)
        return d_input, None, None, None, None


sigmoid_focal_loss = SigmoidFocalLossFunction.apply


class SigmoidFocalLoss(nn.Module):

    def __init__(self, ignore_label, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        b, h, w = targets.size()
        logits = logits.view(b, -1)
        targets = targets.view(b, -1)
        mask = targets.ne(self.ignore_label)
        targets = mask.long() * targets
        target_mask = targets > 0
        loss = sigmoid_focal_loss(logits, targets, self.gamma, self.alpha, 'none')
        loss = loss * mask.float()
        return loss.sum() / target_mask.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'gamma=' + str(self.gamma)
        tmpstr += ', alpha=' + str(self.alpha)
        tmpstr += ')'
        return tmpstr


class ProbOhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256, down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor([1.4297, 1.4805, 1.4363, 3.365, 2.6635, 1.4311, 2.1943, 1.4817, 1.4513, 2.1984, 1.5295, 1.6892, 3.2224, 1.4727, 7.5978, 9.4117, 15.2588, 5.6818, 2.2067])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction, weight=weight, ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()
        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)
        if self.min_kept > num_valid:
            logger.info('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = torch.sort(mask_prob)
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)
        return self.criterion(pred, target)


class DeConvBnRelu(nn.Module):

    def __init__(self, in_planes, out_planes, ksize, stride, pad, output_pad, dilation=1, groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-05, has_relu=True, inplace=True, has_bias=False):
        super(DeConvBnRelu, self).__init__()
        self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=ksize, stride=stride, padding=pad, output_padding=output_pad, dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)
        return inputs


class SELayer(nn.Module):

    def __init__(self, in_planes, out_planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_planes, out_planes // reduction), nn.ReLU(inplace=True), nn.Linear(out_planes // reduction, out_planes), nn.Sigmoid())
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, out_planes, reduction):
        super(ChannelAttention, self).__init__()
        self.channel_attention = SELayer(in_planes, out_planes, reduction)

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], 1)
        channel_attetion = self.channel_attention(fm)
        fm = x1 * channel_attetion + x2
        return fm


class BNRefine(nn.Module):

    def __init__(self, in_planes, out_planes, ksize, has_bias=False, has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-05):
        super(BNRefine, self).__init__()
        self.conv_bn_relu = ConvBnRelu(in_planes, out_planes, ksize, 1, ksize // 2, has_bias=has_bias, norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize, stride=1, padding=ksize // 2, dilation=1, bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        t = self.conv_bn_relu(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x


class RefineResidual(nn.Module):

    def __init__(self, in_planes, out_planes, ksize, has_bias=False, has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-05):
        super(RefineResidual, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=has_bias)
        self.cbr = ConvBnRelu(out_planes, out_planes, ksize, 1, ksize // 2, has_bias=has_bias, norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize, stride=1, padding=ksize // 2, dilation=1, bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv_1x1(x)
        t = self.cbr(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x


class AttentionRefinement(nn.Module):

    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBnRelu(out_planes, out_planes, 1, 1, 0, has_bn=True, norm_layer=norm_layer, has_relu=False, has_bias=False), nn.Sigmoid())

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.channel_attention(fm)
        fm = fm * fm_se
        return fm


class FeatureFusion(nn.Module):

    def __init__(self, in_planes, out_planes, reduction=1, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0, has_bn=False, norm_layer=norm_layer, has_relu=True, has_bias=False), ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0, has_bn=False, norm_layer=norm_layer, has_relu=False, has_bias=False), nn.Sigmoid())

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se
        return output


class BiSeNetHead(nn.Module):

    def __init__(self, in_planes, out_planes, scale, is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 128, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(128, out_planes, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1, stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return output


class SpatialPath(nn.Module):

    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)
        return output


def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()
    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys
    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}'.format(', '.join('{}'.format(k) for k in missing_keys)))
    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(', '.join('{}'.format(k) for k in unexpected_keys)))
    del state_dict
    t_end = time.time()
    logger.info('Load model, Time usage:\n\tIO: {}, initialize parameters: {}'.format(t_ioend - t_start, t_end - t_ioend))
    return model


def xception39(pretrained_model=None, **kwargs):
    model = Xception(Block, [4, 8, 4], [16, 32, 64], **kwargs)
    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


class BiSeNet(nn.Module):

    def __init__(self, out_planes, is_training, criterion, ohem_criterion, pretrained_model=None, norm_layer=nn.BatchNorm2d):
        super(BiSeNet, self).__init__()
        self.context_path = xception39(pretrained_model, norm_layer=norm_layer)
        self.business_layer = []
        self.is_training = is_training
        self.spatial_path = SpatialPath(3, 128, norm_layer)
        conv_channel = 128
        self.global_context = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBnRelu(256, conv_channel, 1, 1, 0, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer))
        arms = [AttentionRefinement(256, conv_channel, norm_layer), AttentionRefinement(128, conv_channel, norm_layer)]
        refines = [ConvBnRelu(conv_channel, conv_channel, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False), ConvBnRelu(conv_channel, conv_channel, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)]
        heads = [BiSeNetHead(conv_channel, out_planes, 16, True, norm_layer), BiSeNetHead(conv_channel, out_planes, 8, True, norm_layer), BiSeNetHead(conv_channel * 2, out_planes, 8, False, norm_layer)]
        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2, 1, norm_layer)
        self.arms = nn.ModuleList(arms)
        self.refines = nn.ModuleList(refines)
        self.heads = nn.ModuleList(heads)
        self.business_layer.append(self.spatial_path)
        self.business_layer.append(self.global_context)
        self.business_layer.append(self.arms)
        self.business_layer.append(self.refines)
        self.business_layer.append(self.heads)
        self.business_layer.append(self.ffm)
        if is_training:
            self.criterion = criterion
            self.ohem_criterion = ohem_criterion

    def forward(self, data, label=None):
        spatial_out = self.spatial_path(data)
        context_blocks = self.context_path(data)
        context_blocks.reverse()
        global_context = self.global_context(context_blocks[0])
        global_context = F.interpolate(global_context, size=context_blocks[0].size()[2:], mode='bilinear', align_corners=True)
        last_fm = global_context
        pred_out = []
        for i, (fm, arm, refine) in enumerate(zip(context_blocks[:2], self.arms, self.refines)):
            fm = arm(fm)
            fm += last_fm
            last_fm = F.interpolate(fm, size=context_blocks[i + 1].size()[2:], mode='bilinear', align_corners=True)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm
        concate_fm = self.ffm(spatial_out, context_out)
        pred_out.append(concate_fm)
        if self.is_training:
            aux_loss0 = self.ohem_criterion(self.heads[0](pred_out[0]), label)
            aux_loss1 = self.ohem_criterion(self.heads[1](pred_out[1]), label)
            main_loss = self.ohem_criterion(self.heads[-1](pred_out[2]), label)
            loss = main_loss + aux_loss0 + aux_loss1
            return loss
        return F.log_softmax(self.heads[-1](pred_out[-1]), dim=1)


class DFNHead(nn.Module):

    def __init__(self, in_planes, out_planes, scale, norm_layer=nn.BatchNorm2d):
        super(DFNHead, self).__init__()
        self.rrb = RefineResidual(in_planes, out_planes * 9, 3, has_bias=False, has_relu=False, norm_layer=norm_layer)
        self.conv = nn.Conv2d(out_planes * 9, out_planes, kernel_size=1, stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        x = self.rrb(x)
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return x


def resnet101(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


class DFN(nn.Module):

    def __init__(self, out_planes, criterion, aux_criterion, alpha, pretrained_model=None, norm_layer=nn.BatchNorm2d):
        super(DFN, self).__init__()
        self.backbone = resnet101(pretrained_model, norm_layer=norm_layer, bn_eps=config.bn_eps, bn_momentum=config.bn_momentum, deep_stem=True, stem_width=64)
        self.business_layer = []
        smooth_inner_channel = 512
        self.global_context = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBnRelu(2048, smooth_inner_channel, 1, 1, 0, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer))
        self.business_layer.append(self.global_context)
        stage = [2048, 1024, 512, 256]
        self.smooth_pre_rrbs = []
        self.cabs = []
        self.smooth_aft_rrbs = []
        self.smooth_heads = []
        for i, channel in enumerate(stage):
            self.smooth_pre_rrbs.append(RefineResidual(channel, smooth_inner_channel, 3, has_bias=False, has_relu=True, norm_layer=norm_layer))
            self.cabs.append(ChannelAttention(smooth_inner_channel * 2, smooth_inner_channel, 1))
            self.smooth_aft_rrbs.append(RefineResidual(smooth_inner_channel, smooth_inner_channel, 3, has_bias=False, has_relu=True, norm_layer=norm_layer))
            self.smooth_heads.append(DFNHead(smooth_inner_channel, out_planes, 2 ** (5 - i), norm_layer=norm_layer))
        stage.reverse()
        border_inner_channel = 21
        self.border_pre_rrbs = []
        self.border_aft_rrbs = []
        self.border_heads = []
        for i, channel in enumerate(stage):
            self.border_pre_rrbs.append(RefineResidual(channel, border_inner_channel, 3, has_bias=False, has_relu=True, norm_layer=norm_layer))
            self.border_aft_rrbs.append(RefineResidual(border_inner_channel, border_inner_channel, 3, has_bias=False, has_relu=True, norm_layer=norm_layer))
            self.border_heads.append(DFNHead(border_inner_channel, 1, 4, norm_layer=norm_layer))
        self.smooth_pre_rrbs = nn.ModuleList(self.smooth_pre_rrbs)
        self.cabs = nn.ModuleList(self.cabs)
        self.smooth_aft_rrbs = nn.ModuleList(self.smooth_aft_rrbs)
        self.smooth_heads = nn.ModuleList(self.smooth_heads)
        self.border_pre_rrbs = nn.ModuleList(self.border_pre_rrbs)
        self.border_aft_rrbs = nn.ModuleList(self.border_aft_rrbs)
        self.border_heads = nn.ModuleList(self.border_heads)
        self.business_layer.append(self.smooth_pre_rrbs)
        self.business_layer.append(self.cabs)
        self.business_layer.append(self.smooth_aft_rrbs)
        self.business_layer.append(self.smooth_heads)
        self.business_layer.append(self.border_pre_rrbs)
        self.business_layer.append(self.border_aft_rrbs)
        self.business_layer.append(self.border_heads)
        self.criterion = criterion
        self.aux_criterion = aux_criterion
        self.alpha = alpha

    def forward(self, data, label=None, aux_label=None):
        blocks = self.backbone(data)
        blocks.reverse()
        global_context = self.global_context(blocks[0])
        global_context = F.interpolate(global_context, size=blocks[0].size()[2:], mode='bilinear', align_corners=True)
        last_fm = global_context
        pred_out = []
        for i, (fm, pre_rrb, cab, aft_rrb, head) in enumerate(zip(blocks, self.smooth_pre_rrbs, self.cabs, self.smooth_aft_rrbs, self.smooth_heads)):
            fm = pre_rrb(fm)
            fm = cab(fm, last_fm)
            fm = aft_rrb(fm)
            pred_out.append(head(fm))
            if i != 3:
                last_fm = F.interpolate(fm, scale_factor=2, mode='bilinear', align_corners=True)
        blocks.reverse()
        last_fm = None
        boder_out = []
        for i, (fm, pre_rrb, aft_rrb, head) in enumerate(zip(blocks, self.border_pre_rrbs, self.border_aft_rrbs, self.border_heads)):
            fm = pre_rrb(fm)
            if last_fm is not None:
                fm = F.interpolate(fm, scale_factor=2 ** i, mode='bilinear', align_corners=True)
                last_fm = last_fm + fm
                last_fm = aft_rrb(last_fm)
            else:
                last_fm = fm
            boder_out.append(head(last_fm))
        if label is not None and aux_label is not None:
            loss0 = self.criterion(pred_out[0], label)
            loss1 = self.criterion(pred_out[1], label)
            loss2 = self.criterion(pred_out[2], label)
            loss3 = self.criterion(pred_out[3], label)
            aux_loss0 = self.aux_criterion(boder_out[0], aux_label)
            aux_loss1 = self.aux_criterion(boder_out[1], aux_label)
            aux_loss2 = self.aux_criterion(boder_out[2], aux_label)
            aux_loss3 = self.aux_criterion(boder_out[3], aux_label)
            loss = loss0 + loss1 + loss2 + loss3
            aux_loss = aux_loss0 + aux_loss1 + aux_loss2 + aux_loss3
            return loss + self.alpha * aux_loss
        return F.log_softmax(pred_out[-1], dim=1)


class _FCNHead(nn.Module):

    def __init__(self, in_planes, out_planes, inplace=True, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_planes = in_planes // 4
        self.cbr = ConvBnRelu(in_planes, inter_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, inplace=inplace, has_bias=False)
        self.dropout = nn.Dropout2d(0.1)
        self.conv1x1 = nn.Conv2d(inter_planes, out_planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.conv1x1(x)
        return x


class FCN(nn.Module):

    def __init__(self, out_planes, criterion, inplace=True, pretrained_model=None, norm_layer=nn.BatchNorm2d):
        super(FCN, self).__init__()
        self.backbone = resnet101(pretrained_model, inplace=inplace, norm_layer=norm_layer, bn_eps=config.bn_eps, bn_momentum=config.bn_momentum, deep_stem=True, stem_width=64)
        self.business_layer = []
        self.head = _FCNHead(2048, out_planes, inplace, norm_layer=norm_layer)
        self.aux_head = _FCNHead(1024, out_planes, inplace, norm_layer=norm_layer)
        self.business_layer.append(self.head)
        self.business_layer.append(self.aux_head)
        self.criterion = criterion

    def forward(self, data, label=None):
        blocks = self.backbone(data)
        fm = self.head(blocks[-1])
        pred = F.interpolate(fm, scale_factor=32, mode='bilinear', align_corners=True)
        aux_fm = self.aux_head(blocks[-2])
        aux_pred = F.interpolate(aux_fm, scale_factor=16, mode='bilinear', align_corners=True)
        if label is not None:
            loss = self.criterion(pred, label)
            aux_loss = self.criterion(aux_pred, label)
            loss = loss + config.aux_loss_ratio * aux_loss
            return loss
        return pred


class PyramidPooling(nn.Module):

    def __init__(self, name, out_planes, fc_dim=4096, pool_scales=[1, 2, 3, 6], norm_layer=nn.BatchNorm2d):
        super(PyramidPooling, self).__init__()
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(OrderedDict([('{}/pool_1'.format(name), nn.AdaptiveAvgPool2d(scale)), ('{}/cbr'.format(name), ConvBnRelu(fc_dim, 512, 1, 1, 0, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer))])))
        self.ppm = nn.ModuleList(self.ppm)
        self.conv6 = nn.Sequential(ConvBnRelu(fc_dim + len(pool_scales) * 512, 512, 3, 1, 1, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer), nn.Dropout2d(0.1, inplace=False), nn.Conv2d(512, out_planes, kernel_size=1))

    def forward(self, x):
        input_size = x.size()
        ppm_out = [x]
        for pooling in self.ppm:
            ppm_out.append(F.interpolate(pooling(x), size=(input_size[2], input_size[3]), mode='bilinear', align_corners=True))
        ppm_out = torch.cat(ppm_out, 1)
        ppm_out = self.conv6(ppm_out)
        return ppm_out


def resnet50(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


class PSPNet(nn.Module):

    def __init__(self, out_planes, criterion, pretrained_model=None, norm_layer=nn.BatchNorm2d):
        super(PSPNet, self).__init__()
        self.backbone = resnet50(pretrained_model, norm_layer=norm_layer, bn_eps=config.bn_eps, bn_momentum=config.bn_momentum, deep_stem=True, stem_width=64)
        self.backbone.layer3.apply(partial(self._nostride_dilate, dilate=2))
        self.backbone.layer4.apply(partial(self._nostride_dilate, dilate=4))
        self.business_layer = []
        self.psp_layer = PyramidPooling('psp', out_planes, 2048, norm_layer=norm_layer)
        self.aux_layer = nn.Sequential(ConvBnRelu(1024, 1024, 3, 1, 1, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer), nn.Dropout2d(0.1, inplace=False), nn.Conv2d(1024, out_planes, kernel_size=1))
        self.business_layer.append(self.psp_layer)
        self.business_layer.append(self.aux_layer)
        self.criterion = criterion

    def forward(self, data, label=None):
        blocks = self.backbone(data)
        psp_fm = self.psp_layer(blocks[-1])
        aux_fm = self.aux_layer(blocks[-2])
        psp_fm = F.interpolate(psp_fm, scale_factor=8, mode='bilinear', align_corners=True)
        aux_fm = F.interpolate(aux_fm, scale_factor=8, mode='bilinear', align_corners=True)
        psp_fm = F.log_softmax(psp_fm, dim=1)
        aux_fm = F.log_softmax(aux_fm, dim=1)
        if label is not None:
            loss = self.criterion(psp_fm, label)
            aux_loss = self.criterion(aux_fm, label)
            loss = loss + 0.4 * aux_loss
            return loss
        return psp_fm

    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = 1, 1
                if m.kernel_size == (3, 3):
                    m.dilation = dilate // 2, dilate // 2
                    m.padding = dilate // 2, dilate // 2
            elif m.kernel_size == (3, 3):
                m.dilation = dilate, dilate
                m.padding = dilate, dilate


class PointwiseSpatialAttention(nn.Module):

    def __init__(self, name, out_planes, fc_dim=4096, pool_scales=[1, 2, 3, 6], norm_layer=nn.BatchNorm2d):
        super(PointwiseSpatialAttention, self).__init__()
        self.inner_channel = 512
        self.collect_reduction = ConvBnRelu(fc_dim, 512, 1, 1, 0, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer)
        self.collect_attention = nn.Sequential(ConvBnRelu(512, 512, 1, 1, 0, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer), ConvBnRelu(512, 3600, 1, 1, 0, has_bn=False, has_relu=False, has_bias=False, norm_layer=norm_layer))
        self.distribute_reduction = ConvBnRelu(fc_dim, 512, 1, 1, 0, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer)
        self.distribute_attention = nn.Sequential(ConvBnRelu(512, 512, 1, 1, 0, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer), ConvBnRelu(512, 3600, 1, 1, 0, has_bn=False, has_relu=False, has_bias=False, norm_layer=norm_layer))
        self.proj = ConvBnRelu(1024, 2048, 1, 1, 0, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(ConvBnRelu(fc_dim + len(pool_scales) * 512, 512, 3, 1, 1, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer), nn.Dropout2d(0.1, inplace=False), nn.Conv2d(512, out_planes, kernel_size=1))

    def forward(self, x):
        collect_reduce_x = self.collect_reduction(x)
        collect_attention = self.collect_attention(collect_reduce_x)
        b, c, h, w = collect_attention.size()
        collect_attention = collect_attention.view(b, c, -1)
        collect_reduce_x = collect_reduce_x.view(b, self.inner_channel, -1)
        collect_fm = torch.bmm(collect_reduce_x, torch.softmax(collect_attention, dim=1))
        collect_fm = collect_fm.view(b, self.inner_channel, h, w)
        distribute_reduce_x = self.distribute_reduction(x)
        distribute_attention = self.distribute_attention(distribute_reduce_x)
        b, c, h, w = distribute_attention.size()
        distribute_attention = distribute_attention.view(b, c, -1)
        distribute_reduce_x = distribute_reduce_x.view(b, self.inner_channel, -1)
        distribute_fm = torch.bmm(distribute_reduce_x, torch.softmax(distribute_attention, dim=1))
        distribute_fm = distribute_fm.view(b, self.inner_channel, h, w)
        psa_fm = torch.cat([collect_fm, distribute_fm], dim=1)
        psa_fm = self.proj(psa_fm)
        fm = torch.cat([x, psa_fm], dim=1)
        out = self.conv6(fm)
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentionRefinement,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BiSeNet,
     lambda: ([], {'out_planes': 4, 'is_training': False, 'criterion': 4, 'ohem_criterion': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (BiSeNetHead,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Block,
     lambda: ([], {'in_channels': 4, 'mid_out_channels': 4, 'has_proj': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBnRelu,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'ksize': 4, 'stride': 1, 'pad': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DFNHead,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DataParallelModel,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (FeatureFusion,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {})),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PyramidPooling,
     lambda: ([], {'name': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4096, 4, 4])], {})),
    (SELayer,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SeparableConvBnRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpatialPath,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_FCNHead,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_SyncBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

