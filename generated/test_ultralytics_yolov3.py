
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


import time


from copy import deepcopy


import torch.distributed as dist


import torch.hub as hub


import torch.optim.lr_scheduler as lr_scheduler


import torchvision


from torch.cuda import amp


import re


import warnings


import pandas as pd


from torch.utils.mobile_optimizer import optimize_for_mobile


import math


from collections import OrderedDict


from collections import namedtuple


from copy import copy


import numpy as np


import torch.nn as nn


import tensorflow as tf


from tensorflow import keras


import random


from torch.optim import lr_scheduler


import torchvision.transforms as T


import torchvision.transforms.functional as TF


from itertools import repeat


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import dataloader


from torch.utils.data import distributed


import logging


import inspect


import logging.config


from typing import Optional


import matplotlib.pyplot as plt


import matplotlib


from scipy.ndimage.filters import gaussian_filter1d


from torch.nn.parallel import DistributedDataParallel as DDP


import typing


class iOSModel(torch.nn.Module):
    """Exports a PyTorch model to an iOS-compatible format with normalized input dimensions and class configurations."""

    def __init__(self, model, im):
        """
        Initializes an iOSModel with normalized input dimensions and number of classes from a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model from which to initialize the iOS model. This should include attributes
                like `nc` (number of classes) which will be used to configure the iOS model.
            im (torch.Tensor): A Tensor representing a sample input image. The shape of this tensor should be
                (batch_size, channels, height, width). This is used to extract dimensions for input normalization.

        Returns:
            None

        Notes:
            - This class is specifically designed for use in exporting a PyTorch model for deployment on iOS platforms, optimizing
              input dimensions and class configurations to suit mobile requirements.
            - Normalization factor is derived from the input image dimensions, which impacts the model's performance during
              inference on iOS devices.
            - Ensure the sample input image `im` provided has correct dimensions and shape for accurate model configuration.
        """
        super().__init__()
        b, c, h, w = im.shape
        self.model = model
        self.nc = model.nc
        if w == h:
            self.normalize = 1.0 / w
        else:
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])

    def forward(self, x):
        """
        Performs a forward pass, returning scaled confidences and normalized coordinates given an input tensor.

        Args:
            x (torch.Tensor): Input tensor representing a batch of images, with dimensions [batch_size, channels,
                height, width].

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three elements:
                - xywh (torch.Tensor): Tensor of shape [batch_size, num_detections, 4] containing normalized x, y, width,
                  and height coordinates.
                - conf (torch.Tensor): Tensor of shape [batch_size, num_detections, 1] containing confidence scores for
                  each detection.
                - cls (torch.Tensor): Tensor of shape [batch_size, num_detections, num_classes] containing class
                  probabilities.

        Notes:
            The dimensions of `x` should match the input dimensions used during the model's initialization to ensure
            proper scaling and normalization.

        Examples:
            ```python
            model = iOSModel(trained_model, input_image_tensor)
            detection_results = model.forward(input_tensor)
            xywh, conf, cls = detection_results
            ```

        Further reading on exporting models to different formats:
        https://github.com/ultralytics/ultralytics

        See Also:
            `export.py` for exporting a YOLOv3 PyTorch model to various formats.
            https://github.com/zldrobit for TensorFlow export scripts.
        """
        xywh, conf, cls = self.model(x)[0].squeeze().split((4, 1, self.nc), 1)
        return cls * conf, xywh * self.normalize


def autopad(k, p=None, d=1):
    """Automatically calculates same shape padding for convolutional layers, optionally adjusts for dilation."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [(d * (x - 1) + 1) for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class Conv(nn.Module):
    """A standard Conv2D layer with batch normalization and optional activation for neural networks."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard Conv2D layer with batch normalization and optional activation; args are channel_in,
        channel_out, kernel_size, stride, padding, groups, dilation, and activation.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies convolution, batch normalization, and activation to input `x`; `x` shape: [N, C_in, H, W] -> [N,
        C_out, H_out, W_out].
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies fused convolution and activation to input `x`; input shape: [N, C_in, H, W] -> [N, C_out, H_out,
        W_out].
        """
        return self.act(self.conv(x))


class DWConv(Conv):
    """Implements depth-wise convolution for efficient spatial feature extraction in neural networks."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initializes depth-wise convolution with optional activation; parameters are channel in/out, kernel, stride,
        dilation.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Implements a depth-wise transpose convolution layer with specified channels, kernel size, stride, and padding."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initializes a depth-wise or transpose convolution layer with specified in/out channels, kernel size, stride,
        and padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    """Transformer layer with multi-head attention and feed-forward network, optimized by removing LayerNorm."""

    def __init__(self, c, num_heads):
        """Initializes a Transformer layer as per https://arxiv.org/abs/2010.11929, sans LayerNorm, with specified
        embedding dimension and number of heads.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Performs forward pass with multi-head attention and residual connections on input tensor 'x' [batch, seq_len,
        features].
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    """Implements a Vision Transformer block with transformer layers; https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initializes a Transformer block with optional convolution, linear, and transformer layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Applies an optional convolution, transforms features, and reshapes output matching input dimensions."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    """Implements a bottleneck layer with optional shortcut for efficient feature extraction in neural networks."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut; args: input channels (c1), output channels
        (c2), shortcut (bool), groups (g), expansion factor (e).
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Executes forward pass, performing convolutional ops and optional shortcut addition; expects input tensor
        x.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """Implements a CSP Bottleneck layer for feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes CSP Bottleneck with channel in/out, optional shortcut, groups, expansion; see
        https://github.com/WongKinYiu/CrossStagePartialNetworks.
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Processes input through layers, combining outputs with activation and normalization for feature
        extraction.
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    """Implements Cross Convolution Downsample with 1D and 2D convolutions and optional shortcut."""

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """Initializes CrossConv with downsample options, combining 1D and 2D convolutions, optional shortcut if
        input/output channels match.
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Performs forward pass using sequential 1D and 2D convolutions with optional shortcut addition."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """Implements a CSP Bottleneck with 3 convolutions, optional shortcuts, group convolutions, and expansion factor."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes CSP Bottleneck with 3 convolutions, optional shortcuts, group convolutions, and expansion
        factor.
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Processes input tensor `x` through convolutions and bottlenecks, returning the concatenated output tensor."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """Extends the C3 module with cross-convolutions for enhanced feature extraction and flexibility."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a C3x module with cross-convolutions, extending the C3 module with customizable parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    """C3 module with TransformerBlock for integrating attention mechanisms in CNNs."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a C3 module with TransformerBlock, extending C3 for attention mechanisms."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    """Implements Spatial Pyramid Pooling (SPP) for enhanced feature extraction; see https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """
        Initializes SPP layer with specified channels and kernels.

        More at https://arxiv.org/abs/1406.4729
        """
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """
        Applies convolution and max pooling layers to the input tensor `x`, concatenates results for feature extraction.

        `x` is a tensor of shape [N, C, H, W]. See https://arxiv.org/abs/1406.4729 for more details.
        """
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class C3SPP(C3):
    """Extends C3 with Spatial Pyramid Pooling (SPP) for enhanced feature extraction in CNNs."""

    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3SPP module, extending C3 with Spatial Pyramid Pooling for enhanced feature extraction."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class GhostConv(nn.Module):
    """Implements Ghost Convolution for efficient feature extraction; see github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes GhostConv with in/out channels, kernel size, stride, groups; see
        https://github.com/huawei-noah/ghostnet.
        """
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Executes forward pass, applying convolutions and concatenating results; input `x` is a tensor."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    """Implements a Ghost Bottleneck layer for efficient feature extraction from GhostNet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with in/out channels, kernel size, and stride; see
        https://github.com/huawei-noah/ghostnet.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1), DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(), GhostConv(c_, c2, 1, 1, act=False))
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Performs a forward pass through the network, returning the sum of convolution and shortcut outputs."""
        return self.conv(x) + self.shortcut(x)


class C3Ghost(C3):
    """Implements a C3 module with Ghost Bottlenecks for efficient feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3Ghost module with Ghost Bottlenecks for efficient feature extraction."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPPF(nn.Module):
    """Implements a fast Spatial Pyramid Pooling (SPPF) layer for efficient feature extraction in YOLOv3 models."""

    def __init__(self, c1, c2, k=5):
        """Initializes the SPPF layer with specified input/output channels and kernel size for YOLOv3."""
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Performs forward pass combining convolutions and max pooling on input `x` of shape [N, C, H, W] to produce
        feature map.
        """
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    """Focuses spatial information into channel space using configurable convolution."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus module to focus width and height information into channel space with configurable
        convolution parameters.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)

    def forward(self, x):
        """Applies focused downsampling to input tensor, returning a convolved output with increased channel depth."""
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))


class Contract(nn.Module):
    """Contracts spatial dimensions into channels, e.g., (1,64,80,80) to (1,256,40,40) with a specified gain."""

    def __init__(self, gain=2):
        """Initializes Contract module to refine input dimensions, e.g., from (1,64,80,80) to (1,256,40,40) with a
        default gain of 2.
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor (b,c,h,w) to contracted shape (b,c*s^2,h/s,w/s) with default gain s=2, e.g.,
        (1,64,80,80) to (1,256,40,40).
        """
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(b, c * s * s, h // s, w // s)


class Expand(nn.Module):
    """Expands spatial dimensions of input tensor by a factor while reducing channels correspondingly."""

    def __init__(self, gain=2):
        """Initializes Expand module to increase spatial dimensions by factor `gain` while reducing channels
        correspondingly.
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Expands spatial dimensions of input tensor `x` by factor `gain` while reducing channels, transforming shape
        `(B,C,H,W)` to `(B,C/gain^2,H*gain,W*gain)`.
        """
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(b, c // s ** 2, h * s, w * s)


class Concat(nn.Module):
    """Concatenates a list of tensors along a specified dimension for efficient feature aggregation."""

    def __init__(self, dimension=1):
        """Initializes a module to concatenate tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Concatenates a list of tensors along a specified dimension; x is a list of tensors to concatenate, dimension
        defaults to 1.
        """
        return torch.cat(x, self.d)


LOGGING_NAME = 'yolov5'


LOGGER = logging.getLogger(LOGGING_NAME)


class TritonRemoteModel:
    """
    A wrapper over a model served by the Triton Inference Server.

    It can be configured to communicate over GRPC or HTTP. It accepts Torch Tensors as input and returns them as
    outputs.
    """

    def __init__(self, url: 'str'):
        """
        Keyword Arguments:
        url: Fully qualified address of the Triton server - for e.g. grpc://localhost:8000.
        """
        parsed_url = urlparse(url)
        if parsed_url.scheme == 'grpc':
            self.client = InferenceServerClient(parsed_url.netloc)
            model_repository = self.client.get_model_repository_index()
            self.model_name = model_repository.models[0].name
            self.metadata = self.client.get_model_metadata(self.model_name, as_json=True)

            def create_input_placeholders() ->typing.List[InferInput]:
                return [InferInput(i['name'], [int(s) for s in i['shape']], i['datatype']) for i in self.metadata['inputs']]
        else:
            self.client = InferenceServerClient(parsed_url.netloc)
            model_repository = self.client.get_model_repository_index()
            self.model_name = model_repository[0]['name']
            self.metadata = self.client.get_model_metadata(self.model_name)

            def create_input_placeholders() ->typing.List[InferInput]:
                return [InferInput(i['name'], [int(s) for s in i['shape']], i['datatype']) for i in self.metadata['inputs']]
        self._create_input_placeholders_fn = create_input_placeholders

    @property
    def runtime(self):
        """Returns the model runtime."""
        return self.metadata.get('backend', self.metadata.get('platform'))

    def __call__(self, *args, **kwargs) ->typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]]:
        """
        Invokes the model.

        Parameters can be provided via args or kwargs. args, if provided, are assumed to match the order of inputs of
        the model. kwargs are matched with the model input names.
        """
        inputs = self._create_inputs(*args, **kwargs)
        response = self.client.infer(model_name=self.model_name, inputs=inputs)
        result = []
        for output in self.metadata['outputs']:
            tensor = torch.as_tensor(response.as_numpy(output['name']))
            result.append(tensor)
        return result[0] if len(result) == 1 else result

    def _create_inputs(self, *args, **kwargs):
        """Generates model inputs from args or kwargs, not allowing both; raises error if neither or both are
        provided.
        """
        args_len, kwargs_len = len(args), len(kwargs)
        if not args_len and not kwargs_len:
            raise RuntimeError('No inputs provided.')
        if args_len and kwargs_len:
            raise RuntimeError('Cannot specify args and kwargs at the same time')
        placeholders = self._create_input_placeholders_fn()
        if args_len:
            if args_len != len(placeholders):
                raise RuntimeError(f'Expected {len(placeholders)} inputs, got {args_len}.')
            for input, value in zip(placeholders, args):
                input.set_data_from_numpy(value.cpu().numpy())
        else:
            for input in placeholders:
                value = kwargs[input.name]
                input.set_data_from_numpy(value.cpu().numpy())
        return placeholders


def curl_download(url, filename, *, silent: bool=False) ->bool:
    """Download a file from a url to a filename using curl."""
    silent_option = 'sS' if silent else ''
    proc = subprocess.run(['curl', '-#', f'-{silent_option}L', url, '--output', filename, '--retry', '9', '-C', '-'])
    return proc.returncode == 0


def safe_download(file, url, url2=None, min_bytes=1.0, error_msg=''):
    """Downloads a file from 'url' or 'url2' to 'file', ensuring size > 'min_bytes'; removes incomplete downloads."""
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg
    except Exception as e:
        if file.exists():
            file.unlink()
        LOGGER.info(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        curl_download(url2 or url, file)
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:
            if file.exists():
                file.unlink()
            LOGGER.info(f'ERROR: {assert_msg}\n{error_msg}')
        LOGGER.info('')


def attempt_download(file, repo='ultralytics/yolov5', release='v7.0'):
    """Attempts to download a file from a specified URL or GitHub release, ensuring file integrity with a minimum size
    check.
    """

    def github_assets(repository, version='latest'):
        """Returns GitHub tag and assets for a given repository and version from the GitHub API."""
        if version != 'latest':
            version = f'tags/{version}'
        response = requests.get(f'https://api.github.com/repos/{repository}/releases/{version}').json()
        return response['tag_name'], [x['name'] for x in response['assets']]
    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        name = Path(urllib.parse.unquote(str(file))).name
        if str(file).startswith(('http:/', 'https:/')):
            url = str(file).replace(':/', '://')
            file = name.split('?')[0]
            if Path(file).is_file():
                LOGGER.info(f'Found {url} locally at {file}')
            else:
                safe_download(file=file, url=url, min_bytes=100000.0)
            return file
        assets = [f'yolov5{size}{suffix}.pt' for size in 'nsmlx' for suffix in ('', '6', '-cls', '-seg')]
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)
            except Exception:
                try:
                    tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release
        if name in assets:
            file.parent.mkdir(parents=True, exist_ok=True)
            safe_download(file, url=f'https://github.com/{repo}/releases/download/{tag}/{name}', min_bytes=100000.0, error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/{tag}')
    return str(file)

