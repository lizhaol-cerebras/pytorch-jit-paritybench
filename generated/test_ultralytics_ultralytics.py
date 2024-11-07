import sys
_module = sys.modules[__name__]
del sys
build_docs = _module
build_reference = _module
action_recognition = _module
main = _module
yolov8_region_counter = _module
yolov8_sahi = _module
tests = _module
conftest = _module
test_cli = _module
test_cuda = _module
test_engine = _module
test_exports = _module
test_integrations = _module
test_python = _module
test_solutions = _module
ultralytics = _module
cfg = _module
data = _module
annotator = _module
augment = _module
base = _module
build = _module
converter = _module
dataset = _module
loaders = _module
split_dota = _module
utils = _module
engine = _module
exporter = _module
model = _module
predictor = _module
results = _module
trainer = _module
tuner = _module
validator = _module
hub = _module
auth = _module
google = _module
session = _module
models = _module
fastsam = _module
predict = _module
utils = _module
val = _module
nas = _module
model = _module
predict = _module
val = _module
rtdetr = _module
predict = _module
train = _module
val = _module
sam = _module
amg = _module
build = _module
model = _module
modules = _module
blocks = _module
decoders = _module
encoders = _module
memory_attention = _module
sam = _module
tiny_encoder = _module
transformer = _module
utils = _module
predict = _module
loss = _module
ops = _module
yolo = _module
classify = _module
predict = _module
train = _module
val = _module
detect = _module
predict = _module
train = _module
val = _module
obb = _module
predict = _module
val = _module
pose = _module
predict = _module
val = _module
segment = _module
predict = _module
val = _module
world = _module
train = _module
train_world = _module
nn = _module
autobackend = _module
modules = _module
activation = _module
block = _module
conv = _module
head = _module
transformer = _module
utils = _module
tasks = _module
solutions = _module
ai_gym = _module
analytics = _module
distance_calculation = _module
heatmap = _module
object_counter = _module
parking_management = _module
queue_management = _module
speed_estimation = _module
streamlit_inference = _module
trackers = _module
basetrack = _module
bot_sort = _module
byte_tracker = _module
track = _module
gmc = _module
kalman_filter = _module
matching = _module
utils = _module
autobatch = _module
benchmarks = _module
callbacks = _module
clearml = _module
comet = _module
dvc = _module
mlflow = _module
neptune = _module
raytune = _module
tensorboard = _module
wb = _module
checks = _module
dist = _module
downloads = _module
errors = _module
files = _module
instance = _module
loss = _module
metrics = _module
ops = _module
patches = _module
plotting = _module
tal = _module
torch_utils = _module
triton = _module

from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import time


from collections import defaultdict


from typing import List


from typing import Optional


from typing import Tuple


import numpy as np


import torch


from itertools import product


import uuid


from copy import copy


import math


import random


from copy import deepcopy


from typing import Union


from torch.utils.data import Dataset


from torch.utils.data import dataloader


from torch.utils.data import distributed


from itertools import repeat


from torch.utils.data import ConcatDataset


import warnings


import inspect


import re


from functools import lru_cache


from torch import distributed as dist


from torch import nn


from torch import optim


from typing import Any


from typing import Generator


from functools import partial


import copy


from typing import Type


import torch.nn.functional as F


from torch import Tensor


import torch.nn as nn


from torch.nn.init import trunc_normal_


import itertools


import torch.utils.checkpoint as checkpoint


from scipy.optimize import linear_sum_assignment


from collections import OrderedDict


from collections import namedtuple


from torch.nn.init import constant_


from torch.nn.init import xavier_uniform_


from torch.nn.init import uniform_


import types


import logging.config


from types import SimpleNamespace


import matplotlib.pyplot as plt


import torch.cuda


from typing import Callable


from typing import Dict


import torch.distributed as dist


class IOSDetectModel(torch.nn.Module):
    """Wrap an Ultralytics YOLO model for Apple iOS CoreML export."""

    def __init__(self, model, im):
        """Initialize the IOSDetectModel class with a YOLO model and example image."""
        super().__init__()
        _, _, h, w = im.shape
        self.model = model
        self.nc = len(model.names)
        if w == h:
            self.normalize = 1.0 / w
        else:
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])

    def forward(self, x):
        """Normalize predictions of object detection model with input size-dependent factors."""
        xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
        return cls, xywh * self.normalize


def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {'.yaml', '.yml'}, f'Attempting to load non-YAML file {file} with yaml_load()'
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()
        if not s.isprintable():
            s = re.sub('[^\\x09\\x0A\\x0D\\x20-\\x7E\\x85\\xA0-\\uD7FF\\uE000-\\uFFFD\\U00010000-\\U0010ffff]+', '', s)
        data = yaml.safe_load(s) or {}
        if append_filename:
            data['yaml_file'] = str(file)
        return data


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [(d * (x - 1) + 1) for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class IterableSimpleNamespace(SimpleNamespace):
    """
    An iterable SimpleNamespace class that provides enhanced functionality for attribute access and iteration.

    This class extends the SimpleNamespace class with additional methods for iteration, string representation,
    and attribute access. It is designed to be used as a convenient container for storing and accessing
    configuration parameters.

    Methods:
        __iter__: Returns an iterator of key-value pairs from the namespace's attributes.
        __str__: Returns a human-readable string representation of the object.
        __getattr__: Provides a custom attribute access error message with helpful information.
        get: Retrieves the value of a specified key, or a default value if the key doesn't exist.

    Examples:
        >>> cfg = IterableSimpleNamespace(a=1, b=2, c=3)
        >>> for k, v in cfg:
        ...     print(f"{k}: {v}")
        a: 1
        b: 2
        c: 3
        >>> print(cfg)
        a=1
        b=2
        c=3
        >>> cfg.get("b")
        2
        >>> cfg.get("d", "default")
        'default'

    Notes:
        This class is particularly useful for storing configuration parameters in a more accessible
        and iterable format compared to a standard dictionary.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return '\n'.join(f'{k}={v}' for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"\n            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics\n            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace\n            {DEFAULT_CFG_PATH} with the latest version from\n            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml\n            ")

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


LOGGING_NAME = 'ultralytics'


def emojis(string=''):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string


def set_logging(name='LOGGING_NAME', verbose=True):
    """
    Sets up logging with UTF-8 encoding and configurable verbosity.

    This function configures logging for the Ultralytics library, setting the appropriate logging level and
    formatter based on the verbosity flag and the current process rank. It handles special cases for Windows
    environments where UTF-8 encoding might not be the default.

    Args:
        name (str): Name of the logger. Defaults to "LOGGING_NAME".
        verbose (bool): Flag to set logging level to INFO if True, ERROR otherwise. Defaults to True.

    Examples:
        >>> set_logging(name="ultralytics", verbose=True)
        >>> logger = logging.getLogger("ultralytics")
        >>> logger.info("This is an info message")

    Notes:
        - On Windows, this function attempts to reconfigure stdout to use UTF-8 encoding if possible.
        - If reconfiguration is not possible, it falls back to a custom formatter that handles non-UTF-8 environments.
        - The function sets up a StreamHandler with the appropriate formatter and level.
        - The logger's propagate flag is set to False to prevent duplicate logging in parent loggers.
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR
    formatter = logging.Formatter('%(message)s')
    if WINDOWS and hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':


        class CustomFormatter(logging.Formatter):

            def format(self, record):
                """Sets up logging with UTF-8 encoding and configurable verbosity."""
                return emojis(super().format(record))
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            elif hasattr(sys.stdout, 'buffer'):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            else:
                formatter = CustomFormatter('%(message)s')
        except Exception as e:
            None
            formatter = CustomFormatter('%(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def parse_version(version='0.0.0') ->tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version. This
    function replaces deprecated 'pkg_resources.parse_version(v)'.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version and the extra string, i.e. (2, 0, 1)
    """
    try:
        return tuple(map(int, re.findall('\\d+', version)[:3]))
    except Exception as e:
        LOGGER.warning(f'WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}')
        return 0, 0, 0


def check_version(current: 'str'='0.0.0', required: 'str'='0.0.0', name: 'str'='version', hard: 'bool'=False, verbose: 'bool'=False, msg: 'str'='') ->bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str, optional): Name to be used in warning message.
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.
        verbose (bool, optional): If True, print warning message if requirement is not met.
        msg (str, optional): Extra message to display if verbose.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Example:
        ```python
        # Check if current version is exactly 22.04
        check_version(current="22.04", required="==22.04")

        # Check if current version is greater than or equal to 22.04
        check_version(current="22.10", required="22.04")  # assumes '>=' inequality if none passed

        # Check if current version is less than or equal to 22.04
        check_version(current="22.04", required="<=22.04")

        # Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current="21.10", required=">20.04,<22.04")
        ```
    """
    if not current:
        LOGGER.warning(f'WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.')
        return True
    elif not current[0].isdigit():
        try:
            name = current
            current = metadata.version(current)
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(emojis(f'WARNING ⚠️ {current} package is required but not installed')) from e
            else:
                return False
    if not required:
        return True
    if 'sys_platform' in required and (WINDOWS and 'win32' not in required or LINUX and 'linux' not in required or MACOS and 'macos' not in required and 'darwin' not in required):
        return True
    op = ''
    version = ''
    result = True
    c = parse_version(current)
    for r in required.strip(',').split(','):
        op, version = re.match('([^0-9]*)([\\d.]+)', r).groups()
        if not op:
            op = '>='
        v = parse_version(version)
        if op == '==' and c != v:
            result = False
        elif op == '!=' and c == v:
            result = False
        elif op == '>=' and not c >= v:
            result = False
        elif op == '<=' and not c <= v:
            result = False
        elif op == '>' and not c > v:
            result = False
        elif op == '<' and not c < v:
            result = False
    if not result:
        warning = f'WARNING ⚠️ {name}{op}{version} is required, but {name}=={current} is currently installed {msg}'
        if hard:
            raise ModuleNotFoundError(emojis(warning))
        if verbose:
            LOGGER.warning(warning)
    return result


TORCH_1_10 = check_version(torch.__version__, '1.10.0')


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class Detect(nn.Module):
    """YOLO Detect head for detection models."""
    dynamic = False
    export = False
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch) if self.legacy else nn.ModuleList(nn.Sequential(nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)), nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return {'one2many': x, 'one2one': one2one}
        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {'one2many': x, 'one2one': one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        if self.export and self.format in {'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'}:
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        if self.export and self.format in {'tflite', 'edgetpu'}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):
                a[-1].bias.data[:] = 1.0
                b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: 'torch.Tensor', max_det: 'int', nc: 'int'=80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [(x // 2) for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')
        self.forward = self.forward_fuse


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=p - k // 2, g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels, out_channels=self.conv1.conv.out_channels, kernel_size=self.conv1.conv.kernel_size, stride=self.conv1.conv.stride, padding=self.conv1.conv.padding, dilation=self.conv1.conv.dilation, groups=self.conv1.conv.groups, bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True).requires_grad_(False)
    w_conv = conv.weight.view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) ->None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])
        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b
        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        self.conv = conv
        del self.conv1


def fuse_deconv_and_bn(deconv, bn):
    """Fuse ConvTranspose2d() and BatchNorm2d() layers."""
    fuseddconv = nn.ConvTranspose2d(deconv.in_channels, deconv.out_channels, kernel_size=deconv.kernel_size, stride=deconv.stride, padding=deconv.padding, output_padding=deconv.output_padding, dilation=deconv.dilation, groups=deconv.groups, bias=True).requires_grad_(False)
    w_deconv = deconv.weight.view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fuseddconv.weight.copy_(torch.mm(w_bn, w_deconv).view(fuseddconv.weight.shape))
    b_conv = torch.zeros(deconv.weight.shape[1], device=deconv.weight.device) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fuseddconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fuseddconv


def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model


def get_flops(model, imgsz=640):
    """Return a YOLO model's FLOPs."""
    if not thop:
        return 0.0
    try:
        model = de_parallel(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]
        try:
            stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
            im = torch.empty((1, p.shape[1], stride, stride), device=p.device)
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1000000000.0 * 2
            return flops * imgsz[0] / stride * imgsz[1] / stride
        except Exception:
            im = torch.empty((1, p.shape[1], *imgsz), device=p.device)
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1000000000.0 * 2
    except Exception:
        return 0.0


def get_num_gradients(model):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())


def model_info(model, detailed=False, verbose=True, imgsz=640):
    """
    Model information.

    imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320].
    """
    if not verbose:
        return
    n_p = get_num_params(model)
    n_g = get_num_gradients(model)
    n_l = len(list(model.modules()))
    if detailed:
        LOGGER.info(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            LOGGER.info('%5g %40s %9s %12g %20s %10.3g %10.3g %10s' % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std(), p.dtype))
    flops = get_flops(model, imgsz)
    fused = ' (fused)' if getattr(model, 'is_fused', lambda : False)() else ''
    fs = f', {flops:.1f} GFLOPs' if flops else ''
    yaml_file = getattr(model, 'yaml_file', '') or getattr(model, 'yaml', {}).get('yaml_file', '')
    model_name = Path(yaml_file).stem.replace('yolo', 'YOLO') or 'Model'
    LOGGER.info(f'{model_name} summary{fused}: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients{fs}')
    return n_l, n_p, n_g, flops


def time_sync():
    """PyTorch-accurate time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward(self, x, *args, **kwargs):
        """
        Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j]) for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(f"WARNING ⚠️ {self.__class__.__name__} does not support 'augment=True' prediction. Reverting to single-scale prediction.")
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to
        the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1] and isinstance(x, list)
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1000000000.0 * 2 if thop else 0
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, 'bn'):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse
                if isinstance(m, ConvTranspose) and hasattr(m, 'bn'):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
            self.info(verbose=verbose)
        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        return sum(isinstance(v, bn) for v in self.modules()) < thresh

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights['model'] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        csd = intersect_dicts(csd, self.state_dict())
        self.load_state_dict(csd, strict=False)
        if verbose:
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights')

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if getattr(self, 'criterion', None) is None:
            self.criterion = self.init_criterion()
        preds = self.forward(batch['img']) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError('compute_loss() needs to be implemented by task heads')


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) ->None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl + F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-07):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw.pow(2) + ch.pow(2) + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
            if CIoU:
                v = 4 / math.pi ** 2 * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0)
        return loss_iou, loss_dfl


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-09):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return torch.full_like(pd_scores[..., 0], self.bg_idx), torch.zeros_like(pd_bboxes), torch.zeros_like(pd_scores), torch.zeros_like(pd_scores[..., 0]), torch.zeros_like(pd_scores[..., 0])
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for horizontal bounding boxes."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
        target_labels.clamp_(0)
        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes), dtype=torch.int64, device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-09):
        """
        Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability. Defaults to 1e-9.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).

        Note:
            b: batch size, h: height, w: width.
        """
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores).sum() / target_scores_sum
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds['one2many']
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds['one2one']
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted rotated bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance, shape (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle, shape (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points, shape (h*w, 2).
        dim (int, optional): Dimension along which to split. Defaults to -1.

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, shape (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)
        angle = (angle.sigmoid() - 0.25) * math.pi
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLO Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {'tflite', 'edgetpu'}:
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 0.001
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


TORCH_1_9 = check_version(torch.__version__, '1.9.0')


class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        if not TORCH_1_9:
            raise ModuleNotFoundError('TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).')
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / temperature ** omega
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
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
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = q.transpose(-2, -1) @ k * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) ->None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape
        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)
        aw = torch.einsum('bmchw,bnmc->bmhwn', embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / self.hc ** 0.5
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale
        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(Conv(c1, c1, 3, g=c1), Conv(c1, 2 * c_, 1), RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_), Conv(2 * c_, c2, 1), Conv(c2, c2, 3, g=c2))
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1), DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(), GhostConv(c_, c2, 1, 1, act=False))
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + 2 * c4, c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + 2 * c4, c2, 1, 1)


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()
        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k ** 2
        x = [pool(proj(x)).view(bs, -1, num_patches) for x, proj, pool in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)
        aw = torch.einsum('bnmc,bkmc->bmnk', q, k)
        aw = aw / self.hc ** 0.5
        aw = F.softmax(aw, dim=-1)
        x = torch.einsum('bmnk,bkmc->bnmc', aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


def _get_clones(module, n):
    """Create a list of cloned modules from the given module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def inverse_sigmoid(x, eps=1e-05):
    """Calculate the inverse sigmoid function for a tensor."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class DeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self, embed, refer_bbox, feats, shapes, bbox_head, score_head, pos_mlp, attn_mask=None, padding_mask=None):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))
            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))
            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break
            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox
        return torch.stack(dec_bboxes), torch.stack(dec_cls)


def multi_scale_deformable_attn_pytorch(value: 'torch.Tensor', value_spatial_shapes: 'torch.Tensor', sampling_locations: 'torch.Tensor', attention_weights: 'torch.Tensor') ->torch.Tensor:
    """
    Multiscale deformable attention.

    https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([(H_ * W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    """
    Multiscale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Initialize MSDeformAttn with the given parameters."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        assert _d_per_head * n_heads == d_model, '`d_model` must be divisible by `n_heads`'
        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        Perform forward pass for multiscale deformable attention.

        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py

        Args:
            query (torch.Tensor): [bs, query_length, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v
        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0.0, act=nn.ReLU(), n_levels=4, n_points=4):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        """Perform the forward pass through the entire decoder layer."""
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[0].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask)
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)
        return self.forward_ffn(embed)


class MLP(nn.Module):
    """Implements a simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act=nn.ReLU, sigmoid=False):
        """Initialize the MLP with specified input, hidden, output dimensions and number of layers."""
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid = sigmoid
        self.act = act()

    def forward(self, x):
        """Forward pass for the entire MLP."""
        for i, layer in enumerate(self.layers):
            x = getattr(self, 'act', nn.ReLU())(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x.sigmoid() if getattr(self, 'sigmoid', False) else x


def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias value according to a given probability value."""
    return float(-np.log((1 - prior_prob) / prior_prob))


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def get_cdn_group(batch, num_classes, num_queries, class_embed, num_dn=100, cls_noise_ratio=0.5, box_noise_scale=1.0, training=False):
    """
    Get contrastive denoising training group. This function creates a contrastive denoising training group with positive
    and negative samples from the ground truths (gt). It applies noise to the class labels and bounding box coordinates,
    and returns the modified labels, bounding boxes, attention mask and meta information.

    Args:
        batch (dict): A dict that includes 'gt_cls' (torch.Tensor with shape [num_gts, ]), 'gt_bboxes'
            (torch.Tensor with shape [num_gts, 4]), 'gt_groups' (List(int)) which is a list of batch size length
            indicating the number of gts of each image.
        num_classes (int): Number of classes.
        num_queries (int): Number of queries.
        class_embed (torch.Tensor): Embedding weights to map class labels to embedding space.
        num_dn (int, optional): Number of denoising. Defaults to 100.
        cls_noise_ratio (float, optional): Noise ratio for class labels. Defaults to 0.5.
        box_noise_scale (float, optional): Noise scale for bounding box coordinates. Defaults to 1.0.
        training (bool, optional): If it's in training mode. Defaults to False.

    Returns:
        (Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): The modified class embeddings,
            bounding boxes, attention mask and meta information for denoising. If not in training mode or 'num_dn'
            is less than or equal to 0, the function returns None for all elements in the tuple.
    """
    if not training or num_dn <= 0:
        return None, None, None, None
    gt_groups = batch['gt_groups']
    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0:
        return None, None, None, None
    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    bs = len(gt_groups)
    gt_cls = batch['cls']
    gt_bbox = batch['bboxes']
    b_idx = batch['batch_idx']
    dn_cls = gt_cls.repeat(2 * num_group)
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)
    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num
    if cls_noise_ratio > 0:
        mask = torch.rand(dn_cls.shape) < cls_noise_ratio * 0.5
        idx = torch.nonzero(mask).squeeze(-1)
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device)
        dn_cls[idx] = new_label
    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(dn_bbox)
        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale
        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = torch.logit(dn_bbox, eps=1e-06)
    num_dn = int(max_nums * 2 * num_group)
    dn_cls_embed = class_embed[dn_cls]
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)
    map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
    pos_idx = torch.stack([(map_indices + max_nums * i) for i in range(num_group)], dim=0)
    map_indices = torch.cat([(map_indices + max_nums * i) for i in range(2 * num_group)])
    padding_cls[dn_b_idx, map_indices] = dn_cls_embed
    padding_bbox[dn_b_idx, map_indices] = dn_bbox
    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    attn_mask[num_dn:, :num_dn] = True
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 * (i + 1):num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 * (i + 1):num_dn] = True
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * 2 * i] = True
    dn_meta = {'dn_pos_idx': [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)], 'dn_num_group': num_group, 'dn_num_split': [num_dn, num_queries]}
    return padding_cls, padding_bbox, attn_mask, dn_meta


def linear_init(module):
    """Initialize the weights and biases of a linear module."""
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, 'bias') and module.bias is not None:
        uniform_(module.bias, -bound, bound)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """
    export = False

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4, nh=8, ndl=6, d_ffn=1024, dropout=0.0, act=nn.ReLU(), eval_idx=-1, nd=100, label_noise_ratio=0.5, box_noise_scale=1.0, learnt_init_query=False):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])
        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        feats, shapes = self._get_encoder_input(x)
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(batch, self.nc, self.num_queries, self.denoising_class_embed.weight, self.num_denoising, self.label_noise_ratio, self.box_noise_scale, self.training)
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        dec_bboxes, dec_scores = self.decoder(embed, refer_bbox, feats, shapes, self.dec_bbox_head, self.dec_score_head, self.query_pos_head, attn_mask=attn_mask)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=0.01):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * 2.0 ** i
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))
        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            shapes.append([h, w])
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)
        enc_outputs_scores = self.enc_score_head(features)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)
        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)
        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first
        if self.is_first:
            self.layer = nn.Sequential(Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: 'int'):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum('bchw,bkc->bkhw', x, w)
        return x * self.logit_scale.exp() + self.bias


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum('bchw,bkc->bkhw', x, w)
        return x * self.logit_scale.exp() + self.bias


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLO detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        if self.export and self.format in {'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'}:
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        if self.export and self.format in {'tflite', 'edgetpu'}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0


def colorstr(*input):
    """
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str | Path): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr("blue", "bold", "hello world")
        >>> "\\033[34m\\033[1mhello world\\033[0m"
    """
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {'black': '\x1b[30m', 'red': '\x1b[31m', 'green': '\x1b[32m', 'yellow': '\x1b[33m', 'blue': '\x1b[34m', 'magenta': '\x1b[35m', 'cyan': '\x1b[36m', 'white': '\x1b[37m', 'bright_black': '\x1b[90m', 'bright_red': '\x1b[91m', 'bright_green': '\x1b[92m', 'bright_yellow': '\x1b[93m', 'bright_blue': '\x1b[94m', 'bright_magenta': '\x1b[95m', 'bright_cyan': '\x1b[96m', 'bright_white': '\x1b[97m', 'end': '\x1b[0m', 'bold': '\x1b[1m', 'underline': '\x1b[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def make_divisible(x, divisor):
    """
    Returns the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """
    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)), nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.one2one_cv3 = copy.deepcopy(self.cv3)


def parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    legacy = True
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]
    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                except ValueError:
                    pass
        n = n_ = max(round(n * depth), 1) if n > 1 else n
        if m in {Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP, C1, C2, C2f, C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN, C2fAttn, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3, PSA, SCDown, C2fCIB}:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3, C2fPSA, C2fCIB, C2PSA}:
                args.insert(2, n)
                n = 1
            if m is C3k2:
                legacy = False
                if scale in 'mlx':
                    args[3] = True
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """Scales and pads an image tensor, optionally maintaining aspect ratio and padding to gs multiple."""
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = int(h * ratio), int(w * ratio)
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)
    if not same_shape:
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


def check_suffix(file='yolo11n.pt', suffix='.pt', msg=''):
    """Check file(s) for acceptable suffix."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = suffix,
        for f in (file if isinstance(file, (list, tuple)) else [file]):
            s = Path(f).suffix.lower().strip()
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}, not {s}'


def check_yolov5u_filename(file: 'str', verbose: 'bool'=True):
    """Replace legacy YOLOv5 filenames with updated YOLOv5u filenames."""
    if 'yolov3' in file or 'yolov5' in file:
        if 'u.yaml' in file:
            file = file.replace('u.yaml', '.yaml')
        elif '.pt' in file and 'u' not in file:
            original_file = file
            file = re.sub('(.*yolov5([nsmlx]))\\.pt', '\\1u.pt', file)
            file = re.sub('(.*yolov5([nsmlx])6)\\.pt', '\\1u.pt', file)
            file = re.sub('(.*yolov3(|-tiny|-spp))\\.pt', '\\1u.pt', file)
            if file != original_file and verbose:
                LOGGER.info(f"PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.\nYOLOv5 'u' models are trained with https://github.com/ultralytics/ultralytics and feature improved performance vs standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.\n")
    return file


def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = Path(url).as_posix().replace(':/', '://')
    return urllib.parse.unquote(url).split('?')[0]


def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name


def check_file(file, suffix='', download=True, download_dir='.', hard=True):
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix)
    file = str(file).strip()
    file = check_yolov5u_filename(file)
    if not file or '://' not in file and Path(file).exists() or file.lower().startswith('grpc://'):
        return file
    elif download and file.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://', 'tcp://')):
        url = file
        file = Path(download_dir) / url2file(file)
        if file.exists():
            LOGGER.info(f'Found {clean_url(url)} locally at {file}')
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return str(file)
    else:
        files = glob.glob(str(ROOT / '**' / file), recursive=True) or glob.glob(str(ROOT.parent / file))
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []


def check_yaml(file, suffix=('.yaml', '.yml'), hard=True):
    """Search/download YAML file (if necessary) and return path, checking suffix."""
    return check_file(file, suffix, hard=hard)


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    try:
        return re.search('yolo[v]?\\d+([nslmx])', Path(model_path).stem).group(1)
    except AttributeError:
        return ''


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub('(\\d+)([nslmx])6(.+)?$', '\\1\\2-p6\\3', path.stem)
        LOGGER.warning(f'WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.')
        path = path.with_name(new_stem + path.suffix)
    unified_path = re.sub('(\\d+)([nslmx])(.+)?$', '\\1\\3', str(path))
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)
    d['scale'] = guess_model_scale(path)
    d['yaml_file'] = str(path)
    return d


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        if self.yaml['backbone'][0][2] == 'Silence':
            LOGGER.warning('WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of nn.Identity. Please delete local *.pt file and re-download the latest model checkpoint.')
            self.yaml['backbone'][0][2] = 'nn.Identity'
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        self.inplace = self.yaml.get('inplace', True)
        self.end2end = getattr(self.model[-1], 'end2end', False)
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace

            def _forward(x):
                """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)['one2many']
                return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)
            m.stride = torch.tensor([(s / x.shape[-2]) for x in _forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        if getattr(self, 'end2end', False) or self.__class__.__name__ != 'DetectionModel':
            LOGGER.warning("WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, -1), None

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y
        elif flips == 3:
            x = img_size[1] - x
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl
        g = sum(4 ** x for x in range(nl))
        e = 1
        i = y[0].shape[-1] // g * sum(4 ** x for x in range(e))
        y[0] = y[0][..., :-i]
        i = y[-1].shape[-1] // g * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][..., i:]
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, 'end2end', False) else v8DetectionLoss(self)


def read_device_model() ->str:
    """
    Reads the device model information from the system and caches it for quick access. Used by is_jetson() and
    is_raspberrypi().

    Returns:
        (str): Model file contents if read successfully or empty string otherwise.
    """
    try:
        with open('/proc/device-tree/model') as f:
            return f.read()
    except Exception:
        return ''


PROC_DEVICE_MODEL = read_device_model()


def is_jetson() ->bool:
    """
    Determines if the Python environment is running on a Jetson Nano or Jetson Orin device by checking the device model
    information.

    Returns:
        (bool): True if running on a Jetson Nano or Jetson Orin, False otherwise.
    """
    return 'NVIDIA' in PROC_DEVICE_MODEL


IS_JETSON = is_jetson()


def is_raspberrypi() ->bool:
    """
    Determines if the Python environment is running on a Raspberry Pi by checking the device model information.

    Returns:
        (bool): True if running on a Raspberry Pi, False otherwise.
    """
    return 'Raspberry Pi' in PROC_DEVICE_MODEL


IS_RASPBERRYPI = is_raspberrypi()


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

    Args:
        masks (torch.Tensor): [n, h, w] tensor of masks
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
        (torch.Tensor): The masks are being cropped to the bounding box.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\nThis error can occur when incorrectly training a 'segment' model on a 'detect' dataset, i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help.") from e
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = max(target_scores.sum(), 1)
        loss[2] = self.bce(pred_scores, target_scores).sum() / target_scores_sum
        if fg_mask.sum():
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor, target_scores, target_scores_sum, fg_mask)
            masks = batch['masks'].float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]
            loss[1] = self.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap)
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.box
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def single_mask_loss(gt_mask: 'torch.Tensor', pred: 'torch.Tensor', proto: 'torch.Tensor', xyxy: 'torch.Tensor', area: 'torch.Tensor') ->torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum('in,nhw->ihw', pred, proto)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(self, fg_mask: 'torch.Tensor', masks: 'torch.Tensor', target_gt_idx: 'torch.Tensor', target_bboxes: 'torch.Tensor', batch_idx: 'torch.Tensor', proto: 'torch.Tensor', pred_masks: 'torch.Tensor', imgsz: 'torch.Tensor', overlap: 'bool') ->torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                loss += self.single_mask_loss(gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i])
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()
        return loss / fg_mask.sum()


class SegmentationModel(DetectionModel):
    """YOLOv8 segmentation model."""

    def __init__(self, cfg='yolov8n-seg.yaml', ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)


TASK2DATA = {'detect': 'coco8.yaml', 'segment': 'coco8-seg.yaml', 'classify': 'imagenet10', 'pose': 'coco8-pose.yaml', 'obb': 'dota8.yaml'}


TORCH_1_13 = check_version(torch.__version__, '1.13.0')


def is_online() ->bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    try:
        assert str(os.getenv('YOLO_OFFLINE', '')).lower() != 'true'
        for dns in ('1.1.1.1', '8.8.8.8'):
            socket.create_connection(address=(dns, 80), timeout=2.0).close()
            return True
    except Exception:
        return False


ONLINE = is_online()


IMG_FORMATS = {'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm', 'heic'}


VID_FORMATS = {'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'}


FORMATS_HELP_MSG = f"""Supported formats are:
images: {IMG_FORMATS}
videos: {VID_FORMATS}"""


HELP_URL = 'See https://docs.ultralytics.com/datasets for dataset formatting guidance.'


class Compose:
    """
    A class for composing multiple image transformations.

    Attributes:
        transforms (List[Callable]): A list of transformation functions to be applied sequentially.

    Methods:
        __call__: Applies a series of transformations to input data.
        append: Appends a new transform to the existing list of transforms.
        insert: Inserts a new transform at a specified index in the list of transforms.
        __getitem__: Retrieves a specific transform or a set of transforms using indexing.
        __setitem__: Sets a specific transform or a set of transforms using indexing.
        tolist: Converts the list of transforms to a standard Python list.

    Examples:
        >>> transforms = [RandomFlip(), RandomPerspective(30)]
        >>> compose = Compose(transforms)
        >>> transformed_data = compose(data)
        >>> compose.append(CenterCrop((224, 224)))
        >>> compose.insert(0, RandomFlip())
    """

    def __init__(self, transforms):
        """
        Initializes the Compose object with a list of transforms.

        Args:
            transforms (List[Callable]): A list of callable transform objects to be applied sequentially.

        Examples:
            >>> from ultralytics.data.augment import Compose, RandomHSV, RandomFlip
            >>> transforms = [RandomHSV(), RandomFlip()]
            >>> compose = Compose(transforms)
        """
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """
        Applies a series of transformations to input data. This method sequentially applies each transformation in the
        Compose object's list of transforms to the input data.

        Args:
            data (Any): The input data to be transformed. This can be of any type, depending on the
                transformations in the list.

        Returns:
            (Any): The transformed data after applying all transformations in sequence.

        Examples:
            >>> transforms = [Transform1(), Transform2(), Transform3()]
            >>> compose = Compose(transforms)
            >>> transformed_data = compose(input_data)
        """
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """
        Appends a new transform to the existing list of transforms.

        Args:
            transform (BaseTransform): The transformation to be added to the composition.

        Examples:
            >>> compose = Compose([RandomFlip(), RandomPerspective()])
            >>> compose.append(RandomHSV())
        """
        self.transforms.append(transform)

    def insert(self, index, transform):
        """
        Inserts a new transform at a specified index in the existing list of transforms.

        Args:
            index (int): The index at which to insert the new transform.
            transform (BaseTransform): The transform object to be inserted.

        Examples:
            >>> compose = Compose([Transform1(), Transform2()])
            >>> compose.insert(1, Transform3())
            >>> len(compose.transforms)
            3
        """
        self.transforms.insert(index, transform)

    def __getitem__(self, index: 'Union[list, int]') ->'Compose':
        """
        Retrieves a specific transform or a set of transforms using indexing.

        Args:
            index (int | List[int]): Index or list of indices of the transforms to retrieve.

        Returns:
            (Compose): A new Compose object containing the selected transform(s).

        Raises:
            AssertionError: If the index is not of type int or list.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), RandomHSV(0.5, 0.5, 0.5)]
            >>> compose = Compose(transforms)
            >>> single_transform = compose[1]  # Returns a Compose object with only RandomPerspective
            >>> multiple_transforms = compose[0:2]  # Returns a Compose object with RandomFlip and RandomPerspective
        """
        assert isinstance(index, (int, list)), f'The indices should be either list or int type but got {type(index)}'
        index = [index] if isinstance(index, int) else index
        return Compose([self.transforms[i] for i in index])

    def __setitem__(self, index: 'Union[list, int]', value: 'Union[list, int]') ->None:
        """
        Sets one or more transforms in the composition using indexing.

        Args:
            index (int | List[int]): Index or list of indices to set transforms at.
            value (Any | List[Any]): Transform or list of transforms to set at the specified index(es).

        Raises:
            AssertionError: If index type is invalid, value type doesn't match index type, or index is out of range.

        Examples:
            >>> compose = Compose([Transform1(), Transform2(), Transform3()])
            >>> compose[1] = NewTransform()  # Replace second transform
            >>> compose[0:2] = [NewTransform1(), NewTransform2()]  # Replace first two transforms
        """
        assert isinstance(index, (int, list)), f'The indices should be either list or int type but got {type(index)}'
        if isinstance(index, list):
            assert isinstance(value, list), f'The indices should be the same type as values, but got {type(index)} and {type(value)}'
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f'list index {i} out of range {len(self.transforms)}.'
            self.transforms[i] = v

    def tolist(self):
        """
        Converts the list of transforms to a standard Python list.

        Returns:
            (List): A list containing all the transform objects in the Compose instance.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), CenterCrop()]
            >>> compose = Compose(transforms)
            >>> transform_list = compose.tolist()
            >>> print(len(transform_list))
            3
        """
        return self.transforms

    def __repr__(self):
        """
        Returns a string representation of the Compose object.

        Returns:
            (str): A string representation of the Compose object, including the list of transforms.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(degrees=10, translate=0.1, scale=0.1)]
            >>> compose = Compose(transforms)
            >>> print(compose)
            Compose([
                RandomFlip(),
                RandomPerspective(degrees=10, translate=0.1, scale=0.1)
            ])
        """
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"


DATASET_CACHE_VERSION = '1.0.3'


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Convert a list of polygons to a binary mask of the specified image size.

    Args:
        imgsz (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int, optional): The color value to fill in the polygons on the mask. Defaults to 1.
        downsample_ratio (int, optional): Factor by which to downsample the mask. Defaults to 1.

    Returns:
        (np.ndarray): A binary mask of the specified image size with the polygons filled in.
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio
    return cv2.resize(mask, (nw, nh))


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Convert a list of polygons to a set of binary masks of the specified image size.

    Args:
        imgsz (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int): The color value to fill in the polygons on the masks.
        downsample_ratio (int, optional): Factor by which to downsample each mask. Defaults to 1.

    Returns:
        (np.ndarray): A set of binary masks of the specified image size with the polygons filled in.
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio), dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask.astype(masks.dtype))
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def xyxyxyxy2xywhr(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation]. Rotation values are
    returned in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray | torch.Tensor): Input box corners [xy1, xy2, xy3, xy4] of shape (n, 8).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted data in [cx, cy, w, h, rotation] format of shape (n, 5).
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


class Format:
    """
    A class for formatting image annotations for object detection, instance segmentation, and pose estimation tasks.

    This class standardizes image and instance annotations to be used by the `collate_fn` in PyTorch DataLoader.

    Attributes:
        bbox_format (str): Format for bounding boxes. Options are 'xywh' or 'xyxy'.
        normalize (bool): Whether to normalize bounding boxes.
        return_mask (bool): Whether to return instance masks for segmentation.
        return_keypoint (bool): Whether to return keypoints for pose estimation.
        return_obb (bool): Whether to return oriented bounding boxes.
        mask_ratio (int): Downsample ratio for masks.
        mask_overlap (bool): Whether to overlap masks.
        batch_idx (bool): Whether to keep batch indexes.
        bgr (float): The probability to return BGR images.

    Methods:
        __call__: Formats labels dictionary with image, classes, bounding boxes, and optionally masks and keypoints.
        _format_img: Converts image from Numpy array to PyTorch tensor.
        _format_segments: Converts polygon points to bitmap masks.

    Examples:
        >>> formatter = Format(bbox_format="xywh", normalize=True, return_mask=True)
        >>> formatted_labels = formatter(labels)
        >>> img = formatted_labels["img"]
        >>> bboxes = formatted_labels["bboxes"]
        >>> masks = formatted_labels["masks"]
    """

    def __init__(self, bbox_format='xywh', normalize=True, return_mask=False, return_keypoint=False, return_obb=False, mask_ratio=4, mask_overlap=True, batch_idx=True, bgr=0.0):
        """
        Initializes the Format class with given parameters for image and instance annotation formatting.

        This class standardizes image and instance annotations for object detection, instance segmentation, and pose
        estimation tasks, preparing them for use in PyTorch DataLoader's `collate_fn`.

        Args:
            bbox_format (str): Format for bounding boxes. Options are 'xywh', 'xyxy', etc.
            normalize (bool): Whether to normalize bounding boxes to [0,1].
            return_mask (bool): If True, returns instance masks for segmentation tasks.
            return_keypoint (bool): If True, returns keypoints for pose estimation tasks.
            return_obb (bool): If True, returns oriented bounding boxes.
            mask_ratio (int): Downsample ratio for masks.
            mask_overlap (bool): If True, allows mask overlap.
            batch_idx (bool): If True, keeps batch indexes.
            bgr (float): Probability of returning BGR images instead of RGB.

        Attributes:
            bbox_format (str): Format for bounding boxes.
            normalize (bool): Whether bounding boxes are normalized.
            return_mask (bool): Whether to return instance masks.
            return_keypoint (bool): Whether to return keypoints.
            return_obb (bool): Whether to return oriented bounding boxes.
            mask_ratio (int): Downsample ratio for masks.
            mask_overlap (bool): Whether masks can overlap.
            batch_idx (bool): Whether to keep batch indexes.
            bgr (float): The probability to return BGR images.

        Examples:
            >>> format = Format(bbox_format="xyxy", return_mask=True, return_keypoint=False)
            >>> print(format.bbox_format)
            xyxy
        """
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx
        self.bgr = bgr

    def __call__(self, labels):
        """
        Formats image annotations for object detection, instance segmentation, and pose estimation tasks.

        This method standardizes the image and instance annotations to be used by the `collate_fn` in PyTorch
        DataLoader. It processes the input labels dictionary, converting annotations to the specified format and
        applying normalization if required.

        Args:
            labels (Dict): A dictionary containing image and annotation data with the following keys:
                - 'img': The input image as a numpy array.
                - 'cls': Class labels for instances.
                - 'instances': An Instances object containing bounding boxes, segments, and keypoints.

        Returns:
            (Dict): A dictionary with formatted data, including:
                - 'img': Formatted image tensor.
                - 'cls': Class labels tensor.
                - 'bboxes': Bounding boxes tensor in the specified format.
                - 'masks': Instance masks tensor (if return_mask is True).
                - 'keypoints': Keypoints tensor (if return_keypoint is True).
                - 'batch_idx': Batch index tensor (if batch_idx is True).

        Examples:
            >>> formatter = Format(bbox_format="xywh", normalize=True, return_mask=True)
            >>> labels = {"img": np.random.rand(640, 640, 3), "cls": np.array([0, 1]), "instances": Instances(...)}
            >>> formatted_labels = formatter(labels)
            >>> print(formatted_labels.keys())
        """
        img = labels.pop('img')
        h, w = img.shape[:2]
        cls = labels.pop('cls')
        instances = labels.pop('instances')
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)
        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio)
            labels['masks'] = masks
        labels['img'] = self._format_img(img)
        labels['cls'] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels['bboxes'] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        if self.return_keypoint:
            labels['keypoints'] = torch.from_numpy(instances.keypoints)
            if self.normalize:
                labels['keypoints'][..., 0] /= w
                labels['keypoints'][..., 1] /= h
        if self.return_obb:
            labels['bboxes'] = xyxyxyxy2xywhr(torch.from_numpy(instances.segments)) if len(instances.segments) else torch.zeros((0, 5))
        if self.normalize:
            labels['bboxes'][:, [0, 2]] /= w
            labels['bboxes'][:, [1, 3]] /= h
        if self.batch_idx:
            labels['batch_idx'] = torch.zeros(nl)
        return labels

    def _format_img(self, img):
        """
        Formats an image for YOLO from a Numpy array to a PyTorch tensor.

        This function performs the following operations:
        1. Ensures the image has 3 dimensions (adds a channel dimension if needed).
        2. Transposes the image from HWC to CHW format.
        3. Optionally flips the color channels from RGB to BGR.
        4. Converts the image to a contiguous array.
        5. Converts the Numpy array to a PyTorch tensor.

        Args:
            img (np.ndarray): Input image as a Numpy array with shape (H, W, C) or (H, W).

        Returns:
            (torch.Tensor): Formatted image as a PyTorch tensor with shape (C, H, W).

        Examples:
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> formatted_img = self._format_img(img)
            >>> print(formatted_img.shape)
            torch.Size([3, 100, 100])
        """
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr else img)
        img = torch.from_numpy(img)
        return img

    def _format_segments(self, instances, cls, w, h):
        """
        Converts polygon segments to bitmap masks.

        Args:
            instances (Instances): Object containing segment information.
            cls (numpy.ndarray): Class labels for each instance.
            w (int): Width of the image.
            h (int): Height of the image.

        Returns:
            (tuple): Tuple containing:
                masks (numpy.ndarray): Bitmap masks with shape (N, H, W) or (1, H, W) if mask_overlap is True.
                instances (Instances): Updated instances object with sorted segments if mask_overlap is True.
                cls (numpy.ndarray): Updated class labels, sorted if mask_overlap is True.

        Notes:
            - If self.mask_overlap is True, masks are overlapped and sorted by area.
            - If self.mask_overlap is False, each mask is represented separately.
            - Masks are downsampled according to self.mask_ratio.
        """
        segments = instances.segments
        if self.mask_overlap:
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
            masks = masks[None]
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)
        return masks, instances, cls


_formats = ['xyxy', 'xywh', 'ltwh']


def ltwh2xywh(x):
    """
    Convert nx4 boxes from [x1, y1, w, h] to [x, y, w, h] where xy1=top-left, xy=center.

    Args:
        x (torch.Tensor): the input tensor

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xywh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2
    y[..., 1] = x[..., 1] + x[..., 3] / 2
    return y


def ltwh2xyxy(x):
    """
    It converts the bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.

    Args:
        x (np.ndarray | torch.Tensor): the input image

    Returns:
        y (np.ndarray | torch.Tensor): the xyxy coordinates of the bounding boxes.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]
    y[..., 3] = x[..., 3] + x[..., 1]
    return y


def _ntuple(n):
    """From PyTorch internals."""

    def parse(x):
        """Parse bounding boxes format between XYWH and LTWH."""
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))
    return parse


to_4tuple = _ntuple(4)


def xywh2ltwh(x):
    """
    Convert the bounding box format from [x, y, w, h] to [x1, y1, w, h], where x1, y1 are the top-left coordinates.

    Args:
        x (np.ndarray | torch.Tensor): The input tensor with the bounding box coordinates in the xywh format

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xyltwh format
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    return y


def xyxy2ltwh(x):
    """
    Convert nx4 bounding boxes from [x1, y1, x2, y2] to [x1, y1, w, h], where xy1=top-left, xy2=bottom-right.

    Args:
        x (np.ndarray | torch.Tensor): The input tensor with the bounding boxes coordinates in the xyxy format

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xyltwh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


class Bboxes:
    """
    A class for handling bounding boxes.

    The class supports various bounding box formats like 'xyxy', 'xywh', and 'ltwh'.
    Bounding box data should be provided in numpy arrays.

    Attributes:
        bboxes (numpy.ndarray): The bounding boxes stored in a 2D numpy array.
        format (str): The format of the bounding boxes ('xyxy', 'xywh', or 'ltwh').

    Note:
        This class does not handle normalization or denormalization of bounding boxes.
    """

    def __init__(self, bboxes, format='xyxy') ->None:
        """Initializes the Bboxes class with bounding box data in a specified format."""
        assert format in _formats, f'Invalid bounding box format: {format}, format must be one of {_formats}'
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format

    def convert(self, format):
        """Converts bounding box format from one type to another."""
        assert format in _formats, f'Invalid bounding box format: {format}, format must be one of {_formats}'
        if self.format == format:
            return
        elif self.format == 'xyxy':
            func = xyxy2xywh if format == 'xywh' else xyxy2ltwh
        elif self.format == 'xywh':
            func = xywh2xyxy if format == 'xyxy' else xywh2ltwh
        else:
            func = ltwh2xyxy if format == 'xyxy' else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self):
        """Return box areas."""
        return (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1]) if self.format == 'xyxy' else self.bboxes[:, 3] * self.bboxes[:, 2]

    def mul(self, scale):
        """
        Multiply bounding box coordinates by scale factor(s).

        Args:
            scale (int | tuple | list): Scale factor(s) for four coordinates.
                If int, the same scale is applied to all coordinates.
        """
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset):
        """
        Add offset to bounding box coordinates.

        Args:
            offset (int | tuple | list): Offset(s) for four coordinates.
                If int, the same offset is applied to all coordinates.
        """
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self):
        """Return the number of boxes."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: "List['Bboxes']", axis=0) ->'Bboxes':
        """
        Concatenate a list of Bboxes objects into a single Bboxes object.

        Args:
            boxes_list (List[Bboxes]): A list of Bboxes objects to concatenate.
            axis (int, optional): The axis along which to concatenate the bounding boxes.
                                   Defaults to 0.

        Returns:
            Bboxes: A new Bboxes object containing the concatenated bounding boxes.

        Note:
            The input should be a list or tuple of Bboxes objects.
        """
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty(0))
        assert all(isinstance(box, Bboxes) for box in boxes_list)
        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))

    def __getitem__(self, index) ->'Bboxes':
        """
        Retrieve a specific bounding box or a set of bounding boxes using indexing.

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired bounding boxes.

        Returns:
            Bboxes: A new Bboxes object containing the selected bounding boxes.

        Raises:
            AssertionError: If the indexed bounding boxes do not form a 2-dimensional matrix.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of bounding boxes.
        """
        if isinstance(index, int):
            return Bboxes(self.bboxes[index].reshape(1, -1))
        b = self.bboxes[index]
        assert b.ndim == 2, f'Indexing on Bboxes with {index} failed to return a matrix!'
        return Bboxes(b)


class Instances:
    """
    Container for bounding boxes, segments, and keypoints of detected objects in an image.

    Attributes:
        _bboxes (Bboxes): Internal object for handling bounding box operations.
        keypoints (ndarray): keypoints(x, y, visible) with shape [N, 17, 3]. Default is None.
        normalized (bool): Flag indicating whether the bounding box coordinates are normalized.
        segments (ndarray): Segments array with shape [N, 1000, 2] after resampling.

    Args:
        bboxes (ndarray): An array of bounding boxes with shape [N, 4].
        segments (list | ndarray, optional): A list or array of object segments. Default is None.
        keypoints (ndarray, optional): An array of keypoints with shape [N, 17, 3]. Default is None.
        bbox_format (str, optional): The format of bounding boxes ('xywh' or 'xyxy'). Default is 'xywh'.
        normalized (bool, optional): Whether the bounding box coordinates are normalized. Default is True.

    Examples:
        ```python
        # Create an Instances object
        instances = Instances(
            bboxes=np.array([[10, 10, 30, 30], [20, 20, 40, 40]]),
            segments=[np.array([[5, 5], [10, 10]]), np.array([[15, 15], [20, 20]])],
            keypoints=np.array([[[5, 5, 1], [10, 10, 1]], [[15, 15, 1], [20, 20, 1]]]),
        )
        ```

    Note:
        The bounding box format is either 'xywh' or 'xyxy', and is determined by the `bbox_format` argument.
        This class does not perform input validation, and it assumes the inputs are well-formed.
    """

    def __init__(self, bboxes, segments=None, keypoints=None, bbox_format='xywh', normalized=True) ->None:
        """
        Initialize the object with bounding boxes, segments, and keypoints.

        Args:
            bboxes (np.ndarray): Bounding boxes, shape [N, 4].
            segments (list | np.ndarray, optional): Segmentation masks. Defaults to None.
            keypoints (np.ndarray, optional): Keypoints, shape [N, 17, 3] and format (x, y, visible). Defaults to None.
            bbox_format (str, optional): Format of bboxes. Defaults to "xywh".
            normalized (bool, optional): Whether the coordinates are normalized. Defaults to True.
        """
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints
        self.normalized = normalized
        self.segments = segments

    def convert_bbox(self, format):
        """Convert bounding box format."""
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self):
        """Calculate the area of bounding boxes."""
        return self._bboxes.areas()

    def scale(self, scale_w, scale_h, bbox_only=False):
        """Similar to denormalize func but without normalized sign."""
        self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))
        if bbox_only:
            return
        self.segments[..., 0] *= scale_w
        self.segments[..., 1] *= scale_h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h

    def denormalize(self, w, h):
        """Denormalizes boxes, segments, and keypoints from normalized coordinates."""
        if not self.normalized:
            return
        self._bboxes.mul(scale=(w, h, w, h))
        self.segments[..., 0] *= w
        self.segments[..., 1] *= h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h
        self.normalized = False

    def normalize(self, w, h):
        """Normalize bounding boxes, segments, and keypoints to image dimensions."""
        if self.normalized:
            return
        self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
        self.segments[..., 0] /= w
        self.segments[..., 1] /= h
        if self.keypoints is not None:
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h
        self.normalized = True

    def add_padding(self, padw, padh):
        """Handle rect and mosaic situation."""
        assert not self.normalized, 'you should add padding with absolute coordinates.'
        self._bboxes.add(offset=(padw, padh, padw, padh))
        self.segments[..., 0] += padw
        self.segments[..., 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh

    def __getitem__(self, index) ->'Instances':
        """
        Retrieve a specific instance or a set of instances using indexing.

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired instances.

        Returns:
            Instances: A new Instances object containing the selected bounding boxes,
                       segments, and keypoints if present.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of instances.
        """
        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]
        bbox_format = self._bboxes.format
        return Instances(bboxes=bboxes, segments=segments, keypoints=keypoints, bbox_format=bbox_format, normalized=self.normalized)

    def flipud(self, h):
        """Flips the coordinates of bounding boxes, segments, and keypoints vertically."""
        if self._bboxes.format == 'xyxy':
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        else:
            self.bboxes[:, 1] = h - self.bboxes[:, 1]
        self.segments[..., 1] = h - self.segments[..., 1]
        if self.keypoints is not None:
            self.keypoints[..., 1] = h - self.keypoints[..., 1]

    def fliplr(self, w):
        """Reverses the order of the bounding boxes and segments horizontally."""
        if self._bboxes.format == 'xyxy':
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        else:
            self.bboxes[:, 0] = w - self.bboxes[:, 0]
        self.segments[..., 0] = w - self.segments[..., 0]
        if self.keypoints is not None:
            self.keypoints[..., 0] = w - self.keypoints[..., 0]

    def clip(self, w, h):
        """Clips bounding boxes, segments, and keypoints values to stay within image boundaries."""
        ori_format = self._bboxes.format
        self.convert_bbox(format='xyxy')
        self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
        self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
        if ori_format != 'xyxy':
            self.convert_bbox(format=ori_format)
        self.segments[..., 0] = self.segments[..., 0].clip(0, w)
        self.segments[..., 1] = self.segments[..., 1].clip(0, h)
        if self.keypoints is not None:
            self.keypoints[..., 0] = self.keypoints[..., 0].clip(0, w)
            self.keypoints[..., 1] = self.keypoints[..., 1].clip(0, h)

    def remove_zero_area_boxes(self):
        """Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height."""
        good = self.bbox_areas > 0
        if not all(good):
            self._bboxes = self._bboxes[good]
            if len(self.segments):
                self.segments = self.segments[good]
            if self.keypoints is not None:
                self.keypoints = self.keypoints[good]
        return good

    def update(self, bboxes, segments=None, keypoints=None):
        """Updates instance variables."""
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
        if segments is not None:
            self.segments = segments
        if keypoints is not None:
            self.keypoints = keypoints

    def __len__(self):
        """Return the length of the instance list."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: "List['Instances']", axis=0) ->'Instances':
        """
        Concatenates a list of Instances objects into a single Instances object.

        Args:
            instances_list (List[Instances]): A list of Instances objects to concatenate.
            axis (int, optional): The axis along which the arrays will be concatenated. Defaults to 0.

        Returns:
            Instances: A new Instances object containing the concatenated bounding boxes,
                       segments, and keypoints if present.

        Note:
            The `Instances` objects in the list should have the same properties, such as
            the format of the bounding boxes, whether keypoints are present, and if the
            coordinates are normalized.
        """
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, Instances) for instance in instances_list)
        if len(instances_list) == 1:
            return instances_list[0]
        use_keypoint = instances_list[0].keypoints is not None
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized
        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis)
        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
        return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized)

    @property
    def bboxes(self):
        """Return bounding boxes."""
        return self._bboxes.bboxes


class LetterBox:
    """
    Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates
    corresponding labels and bounding boxes.

    Attributes:
        new_shape (tuple): Target shape (height, width) for resizing.
        auto (bool): Whether to use minimum rectangle.
        scaleFill (bool): Whether to stretch the image to new_shape.
        scaleup (bool): Whether to allow scaling up. If False, only scale down.
        stride (int): Stride for rounding padding.
        center (bool): Whether to center the image or align to top-left.

    Methods:
        __call__: Resize and pad image, update labels and bounding boxes.

    Examples:
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result["img"]
        >>> updated_instances = result["instances"]
    """

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """
        Initialize LetterBox object for resizing and padding images.

        This class is designed to resize and pad images for object detection, instance segmentation, and pose estimation
        tasks. It supports various resizing modes including auto-sizing, scale-fill, and letterboxing.

        Args:
            new_shape (Tuple[int, int]): Target size (height, width) for the resized image.
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly.
            scaleFill (bool): If True, stretch the image to new_shape without padding.
            scaleup (bool): If True, allow scaling up. If False, only scale down.
            center (bool): If True, center the placed image. If False, place image in top-left corner.
            stride (int): Stride of the model (e.g., 32 for YOLOv5).

        Attributes:
            new_shape (Tuple[int, int]): Target size for the resized image.
            auto (bool): Flag for using minimum rectangle resizing.
            scaleFill (bool): Flag for stretching image without padding.
            scaleup (bool): Flag for allowing upscaling.
            stride (int): Stride value for ensuring image size is divisible by stride.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32)
            >>> resized_img = letterbox(original_img)
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center

    def __call__(self, labels=None, image=None):
        """
        Resizes and pads an image for object detection, instance segmentation, or pose estimation tasks.

        This method applies letterboxing to the input image, which involves resizing the image while maintaining its
        aspect ratio and adding padding to fit the new shape. It also updates any associated labels accordingly.

        Args:
            labels (Dict | None): A dictionary containing image data and associated labels, or empty dict if None.
            image (np.ndarray | None): The input image as a numpy array. If None, the image is taken from 'labels'.

        Returns:
            (Dict | Tuple): If 'labels' is provided, returns an updated dictionary with the resized and padded image,
                updated labels, and additional metadata. If 'labels' is empty, returns a tuple containing the resized
                and padded image, and a tuple of (ratio, (left_pad, top_pad)).

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> result = letterbox(labels={"img": np.zeros((480, 640, 3)), "instances": Instances(...)})
            >>> resized_img = result["img"]
            >>> updated_instances = result["instances"]
        """
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = new_shape, new_shape
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = new_shape[1], new_shape[0]
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        if self.center:
            dw /= 2
            dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = labels['ratio_pad'], (left, top)
        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """
        Updates labels after applying letterboxing to an image.

        This method modifies the bounding box coordinates of instances in the labels
        to account for resizing and padding applied during letterboxing.

        Args:
            labels (Dict): A dictionary containing image labels and instances.
            ratio (Tuple[float, float]): Scaling ratios (width, height) applied to the image.
            padw (float): Padding width added to the image.
            padh (float): Padding height added to the image.

        Returns:
            (Dict): Updated labels dictionary with modified instance coordinates.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> labels = {"instances": Instances(...)}
            >>> ratio = (0.5, 0.5)
            >>> padw, padh = 10, 20
            >>> updated_labels = letterbox._update_labels(labels, ratio, padw, padh)
        """
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    h = hashlib.sha256(str(size).encode())
    h.update(''.join(paths).encode())
    return h.hexdigest()


def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
    return [(sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt') for x in img_paths]


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    gc.disable()
    cache = np.load(str(path), allow_pickle=True).item()
    gc.enable()
    return cache


def resample_segments(segments, n=1000):
    """
    Inputs a list of segments (n,2) and returns a list of segments (n,2) up-sampled to n points each.

    Args:
        segments (list): a list of (n,2) arrays, where n is the number of points in the segment.
        n (int): number of points to resample the segment to. Defaults to 1000

    Returns:
        segments (list): the resampled segments.
    """
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
    return segments


def is_dir_writeable(dir_path: 'Union[str, Path]') ->bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)


def save_dataset_cache_file(prefix, path, x, version):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x['version'] = version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()
        np.save(str(path), x)
        path.with_suffix('.cache.npy').rename(path)
        LOGGER.info(f'{prefix}New cache created: {path}')
    else:
        LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')


class Albumentations:
    """
    Albumentations transformations for image augmentation.

    This class applies various image transformations using the Albumentations library. It includes operations such as
    Blur, Median Blur, conversion to grayscale, Contrast Limited Adaptive Histogram Equalization (CLAHE), random changes
    in brightness and contrast, RandomGamma, and image quality reduction through compression.

    Attributes:
        p (float): Probability of applying the transformations.
        transform (albumentations.Compose): Composed Albumentations transforms.
        contains_spatial (bool): Indicates if the transforms include spatial operations.

    Methods:
        __call__: Applies the Albumentations transformations to the input labels.

    Examples:
        >>> transform = Albumentations(p=0.5)
        >>> augmented_labels = transform(labels)

    Notes:
        - The Albumentations package must be installed to use this class.
        - If the package is not installed or an error occurs during initialization, the transform will be set to None.
        - Spatial transforms are handled differently and require special processing for bounding boxes.
    """

    def __init__(self, p=1.0):
        """
        Initialize the Albumentations transform object for YOLO bbox formatted parameters.

        This class applies various image augmentations using the Albumentations library, including Blur, Median Blur,
        conversion to grayscale, Contrast Limited Adaptive Histogram Equalization, random changes of brightness and
        contrast, RandomGamma, and image quality reduction through compression.

        Args:
            p (float): Probability of applying the augmentations. Must be between 0 and 1.

        Attributes:
            p (float): Probability of applying the augmentations.
            transform (albumentations.Compose): Composed Albumentations transforms.
            contains_spatial (bool): Indicates if the transforms include spatial transformations.

        Raises:
            ImportError: If the Albumentations package is not installed.
            Exception: For any other errors during initialization.

        Examples:
            >>> transform = Albumentations(p=0.5)
            >>> augmented = transform(image=image, bboxes=bboxes, class_labels=classes)
            >>> augmented_image = augmented["image"]
            >>> augmented_bboxes = augmented["bboxes"]

        Notes:
            - Requires Albumentations version 1.0.3 or higher.
            - Spatial transforms are handled differently to ensure bbox compatibility.
            - Some transforms are applied with very low probability (0.01) by default.
        """
        self.p = p
        self.transform = None
        prefix = colorstr('albumentations: ')
        try:
            check_version(A.__version__, '1.0.3', hard=True)
            spatial_transforms = {'Affine', 'BBoxSafeRandomCrop', 'CenterCrop', 'CoarseDropout', 'Crop', 'CropAndPad', 'CropNonEmptyMaskIfExists', 'D4', 'ElasticTransform', 'Flip', 'GridDistortion', 'GridDropout', 'HorizontalFlip', 'Lambda', 'LongestMaxSize', 'MaskDropout', 'MixUp', 'Morphological', 'NoOp', 'OpticalDistortion', 'PadIfNeeded', 'Perspective', 'PiecewiseAffine', 'PixelDropout', 'RandomCrop', 'RandomCropFromBorders', 'RandomGridShuffle', 'RandomResizedCrop', 'RandomRotate90', 'RandomScale', 'RandomSizedBBoxSafeCrop', 'RandomSizedCrop', 'Resize', 'Rotate', 'SafeRotate', 'ShiftScaleRotate', 'SmallestMaxSize', 'Transpose', 'VerticalFlip', 'XYMasking'}
            T = [A.Blur(p=0.01), A.MedianBlur(p=0.01), A.ToGray(p=0.01), A.CLAHE(p=0.01), A.RandomBrightnessContrast(p=0.0), A.RandomGamma(p=0.0), A.ImageCompression(quality_lower=75, p=0.0)]
            self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])) if self.contains_spatial else A.Compose(T)
            LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        except ImportError:
            pass
        except Exception as e:
            LOGGER.info(f'{prefix}{e}')

    def __call__(self, labels):
        """
        Applies Albumentations transformations to input labels.

        This method applies a series of image augmentations using the Albumentations library. It can perform both
        spatial and non-spatial transformations on the input image and its corresponding labels.

        Args:
            labels (Dict): A dictionary containing image data and annotations. Expected keys are:
                - 'img': numpy.ndarray representing the image
                - 'cls': numpy.ndarray of class labels
                - 'instances': object containing bounding boxes and other instance information

        Returns:
            (Dict): The input dictionary with augmented image and updated annotations.

        Examples:
            >>> transform = Albumentations(p=0.5)
            >>> labels = {
            ...     "img": np.random.rand(640, 640, 3),
            ...     "cls": np.array([0, 1]),
            ...     "instances": Instances(bboxes=np.array([[0, 0, 1, 1], [0.5, 0.5, 0.8, 0.8]])),
            ... }
            >>> augmented = transform(labels)
            >>> assert augmented["img"].shape == (640, 640, 3)

        Notes:
            - The method applies transformations with probability self.p.
            - Spatial transforms update bounding boxes, while non-spatial transforms only modify the image.
            - Requires the Albumentations library to be installed.
        """
        if self.transform is None or random.random() > self.p:
            return labels
        if self.contains_spatial:
            cls = labels['cls']
            if len(cls):
                im = labels['img']
                labels['instances'].convert_bbox('xywh')
                labels['instances'].normalize(*im.shape[:2][::-1])
                bboxes = labels['instances'].bboxes
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)
                if len(new['class_labels']) > 0:
                    labels['img'] = new['image']
                    labels['cls'] = np.array(new['class_labels'])
                    bboxes = np.array(new['bboxes'], dtype=np.float32)
                labels['instances'].update(bboxes=bboxes)
        else:
            labels['img'] = self.transform(image=labels['img'])['image']
        return labels


class BaseMixTransform:
    """
    Base class for mix transformations like MixUp and Mosaic.

    This class provides a foundation for implementing mix transformations on datasets. It handles the
    probability-based application of transforms and manages the mixing of multiple images and labels.

    Attributes:
        dataset (Any): The dataset object containing images and labels.
        pre_transform (Callable | None): Optional transform to apply before mixing.
        p (float): Probability of applying the mix transformation.

    Methods:
        __call__: Applies the mix transformation to the input labels.
        _mix_transform: Abstract method to be implemented by subclasses for specific mix operations.
        get_indexes: Abstract method to get indexes of images to be mixed.
        _update_label_text: Updates label text for mixed images.

    Examples:
        >>> class CustomMixTransform(BaseMixTransform):
        ...     def _mix_transform(self, labels):
        ...         # Implement custom mix logic here
        ...         return labels
        ...
        ...     def get_indexes(self):
        ...         return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        >>> dataset = YourDataset()
        >>> transform = CustomMixTransform(dataset, p=0.5)
        >>> mixed_labels = transform(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) ->None:
        """
        Initializes the BaseMixTransform object for mix transformations like MixUp and Mosaic.

        This class serves as a base for implementing mix transformations in image processing pipelines.

        Args:
            dataset (Any): The dataset object containing images and labels for mixing.
            pre_transform (Callable | None): Optional transform to apply before mixing.
            p (float): Probability of applying the mix transformation. Should be in the range [0.0, 1.0].

        Examples:
            >>> dataset = YOLODataset("path/to/data")
            >>> pre_transform = Compose([RandomFlip(), RandomPerspective()])
            >>> mix_transform = BaseMixTransform(dataset, pre_transform, p=0.5)
        """
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        """
        Applies pre-processing transforms and mixup/mosaic transforms to labels data.

        This method determines whether to apply the mix transform based on a probability factor. If applied, it
        selects additional images, applies pre-transforms if specified, and then performs the mix transform.

        Args:
            labels (Dict): A dictionary containing label data for an image.

        Returns:
            (Dict): The transformed labels dictionary, which may include mixed data from other images.

        Examples:
            >>> transform = BaseMixTransform(dataset, pre_transform=None, p=0.5)
            >>> result = transform({"image": img, "bboxes": boxes, "cls": classes})
        """
        if random.uniform(0, 1) > self.p:
            return labels
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels['mix_labels'] = mix_labels
        labels = self._update_label_text(labels)
        labels = self._mix_transform(labels)
        labels.pop('mix_labels', None)
        return labels

    def _mix_transform(self, labels):
        """
        Applies MixUp or Mosaic augmentation to the label dictionary.

        This method should be implemented by subclasses to perform specific mix transformations like MixUp or
        Mosaic. It modifies the input label dictionary in-place with the augmented data.

        Args:
            labels (Dict): A dictionary containing image and label data. Expected to have a 'mix_labels' key
                with a list of additional image and label data for mixing.

        Returns:
            (Dict): The modified labels dictionary with augmented data after applying the mix transform.

        Examples:
            >>> transform = BaseMixTransform(dataset)
            >>> labels = {"image": img, "bboxes": boxes, "mix_labels": [{"image": img2, "bboxes": boxes2}]}
            >>> augmented_labels = transform._mix_transform(labels)
        """
        raise NotImplementedError

    def get_indexes(self):
        """
        Gets a list of shuffled indexes for mosaic augmentation.

        Returns:
            (List[int]): A list of shuffled indexes from the dataset.

        Examples:
            >>> transform = BaseMixTransform(dataset)
            >>> indexes = transform.get_indexes()
            >>> print(indexes)  # [3, 18, 7, 2]
        """
        raise NotImplementedError

    def _update_label_text(self, labels):
        """
        Updates label text and class IDs for mixed labels in image augmentation.

        This method processes the 'texts' and 'cls' fields of the input labels dictionary and any mixed labels,
        creating a unified set of text labels and updating class IDs accordingly.

        Args:
            labels (Dict): A dictionary containing label information, including 'texts' and 'cls' fields,
                and optionally a 'mix_labels' field with additional label dictionaries.

        Returns:
            (Dict): The updated labels dictionary with unified text labels and updated class IDs.

        Examples:
            >>> labels = {
            ...     "texts": [["cat"], ["dog"]],
            ...     "cls": torch.tensor([[0], [1]]),
            ...     "mix_labels": [{"texts": [["bird"], ["fish"]], "cls": torch.tensor([[0], [1]])}],
            ... }
            >>> updated_labels = self._update_label_text(labels)
            >>> print(updated_labels["texts"])
            [['cat'], ['dog'], ['bird'], ['fish']]
            >>> print(updated_labels["cls"])
            tensor([[0],
                    [1]])
            >>> print(updated_labels["mix_labels"][0]["cls"])
            tensor([[2],
                    [3]])
        """
        if 'texts' not in labels:
            return labels
        mix_texts = sum([labels['texts']] + [x['texts'] for x in labels['mix_labels']], [])
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for i, text in enumerate(mix_texts)}
        for label in ([labels] + labels['mix_labels']):
            for i, cls in enumerate(label['cls'].squeeze(-1).tolist()):
                text = label['texts'][int(cls)]
                label['cls'][i] = text2id[tuple(text)]
            label['texts'] = mix_texts
        return labels


def bbox_ioa(box1, box2, iou=False, eps=1e-07):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.

    Args:
        box1 (np.ndarray): A numpy array of shape (n, 4) representing n bounding boxes.
        box2 (np.ndarray): A numpy array of shape (m, 4) representing m bounding boxes.
        iou (bool): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.ndarray): A numpy array of shape (n, m) representing the intersection over box2 area.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area
    return inter_area / (area + eps)


class CopyPaste(BaseMixTransform):
    """
    CopyPaste class for applying Copy-Paste augmentation to image datasets.

    This class implements the Copy-Paste augmentation technique as described in the paper "Simple Copy-Paste is a Strong
    Data Augmentation Method for Instance Segmentation" (https://arxiv.org/abs/2012.07177). It combines objects from
    different images to create new training samples.

    Attributes:
        dataset (Any): The dataset to which Copy-Paste augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before Copy-Paste.
        p (float): Probability of applying Copy-Paste augmentation.

    Methods:
        get_indexes: Returns a random index from the dataset.
        _mix_transform: Applies Copy-Paste augmentation to the input labels.
        __call__: Applies the Copy-Paste transformation to images and annotations.

    Examples:
        >>> from ultralytics.data.augment import CopyPaste
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> copypaste = CopyPaste(dataset, p=0.5)
        >>> augmented_labels = copypaste(original_labels)
    """

    def __init__(self, dataset=None, pre_transform=None, p=0.5, mode='flip') ->None:
        """Initializes CopyPaste object with dataset, pre_transform, and probability of applying MixUp."""
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        assert mode in {'flip', 'mixup'}, f'Expected `mode` to be `flip` or `mixup`, but got {mode}.'
        self.mode = mode

    def get_indexes(self):
        """Returns a list of random indexes from the dataset for CopyPaste augmentation."""
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        """Applies Copy-Paste augmentation to combine objects from another image into the current image."""
        labels2 = labels['mix_labels'][0]
        return self._transform(labels, labels2)

    def __call__(self, labels):
        """Applies Copy-Paste augmentation to an image and its labels."""
        if len(labels['instances'].segments) == 0 or self.p == 0:
            return labels
        if self.mode == 'flip':
            return self._transform(labels)
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels['mix_labels'] = mix_labels
        labels = self._update_label_text(labels)
        labels = self._mix_transform(labels)
        labels.pop('mix_labels', None)
        return labels

    def _transform(self, labels1, labels2={}):
        """Applies Copy-Paste augmentation to combine objects from another image into the current image."""
        im = labels1['img']
        cls = labels1['cls']
        h, w = im.shape[:2]
        instances = labels1.pop('instances')
        instances.convert_bbox(format='xyxy')
        instances.denormalize(w, h)
        im_new = np.zeros(im.shape, np.uint8)
        instances2 = labels2.pop('instances', None)
        if instances2 is None:
            instances2 = deepcopy(instances)
            instances2.fliplr(w)
        ioa = bbox_ioa(instances2.bboxes, instances.bboxes)
        indexes = np.nonzero((ioa < 0.3).all(1))[0]
        n = len(indexes)
        sorted_idx = np.argsort(ioa.max(1)[indexes])
        indexes = indexes[sorted_idx]
        for j in indexes[:round(self.p * n)]:
            cls = np.concatenate((cls, labels2.get('cls', cls)[[j]]), axis=0)
            instances = Instances.concatenate((instances, instances2[[j]]), axis=0)
            cv2.drawContours(im_new, instances2.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)
        result = labels2.get('img', cv2.flip(im, 1))
        i = im_new.astype(bool)
        im[i] = result[i]
        labels1['img'] = im
        labels1['cls'] = cls
        labels1['instances'] = instances
        return labels1


class MixUp(BaseMixTransform):
    """
    Applies MixUp augmentation to image datasets.

    This class implements the MixUp augmentation technique as described in the paper "mixup: Beyond Empirical Risk
    Minimization" (https://arxiv.org/abs/1710.09412). MixUp combines two images and their labels using a random weight.

    Attributes:
        dataset (Any): The dataset to which MixUp augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before MixUp.
        p (float): Probability of applying MixUp augmentation.

    Methods:
        get_indexes: Returns a random index from the dataset.
        _mix_transform: Applies MixUp augmentation to the input labels.

    Examples:
        >>> from ultralytics.data.augment import MixUp
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> mixup = MixUp(dataset, p=0.5)
        >>> augmented_labels = mixup(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) ->None:
        """
        Initializes the MixUp augmentation object.

        MixUp is an image augmentation technique that combines two images by taking a weighted sum of their pixel
        values and labels. This implementation is designed for use with the Ultralytics YOLO framework.

        Args:
            dataset (Any): The dataset to which MixUp augmentation will be applied.
            pre_transform (Callable | None): Optional transform to apply to images before MixUp.
            p (float): Probability of applying MixUp augmentation to an image. Must be in the range [0, 1].

        Examples:
            >>> from ultralytics.data.dataset import YOLODataset
            >>> dataset = YOLODataset("path/to/data.yaml")
            >>> mixup = MixUp(dataset, pre_transform=None, p=0.5)
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_indexes(self):
        """
        Get a random index from the dataset.

        This method returns a single random index from the dataset, which is used to select an image for MixUp
        augmentation.

        Returns:
            (int): A random integer index within the range of the dataset length.

        Examples:
            >>> mixup = MixUp(dataset)
            >>> index = mixup.get_indexes()
            >>> print(index)
            42
        """
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        """
        Applies MixUp augmentation to the input labels.

        This method implements the MixUp augmentation technique as described in the paper
        "mixup: Beyond Empirical Risk Minimization" (https://arxiv.org/abs/1710.09412).

        Args:
            labels (Dict): A dictionary containing the original image and label information.

        Returns:
            (Dict): A dictionary containing the mixed-up image and combined label information.

        Examples:
            >>> mixer = MixUp(dataset)
            >>> mixed_labels = mixer._mix_transform(labels)
        """
        r = np.random.beta(32.0, 32.0)
        labels2 = labels['mix_labels'][0]
        labels['img'] = (labels['img'] * r + labels2['img'] * (1 - r)).astype(np.uint8)
        labels['instances'] = Instances.concatenate([labels['instances'], labels2['instances']], axis=0)
        labels['cls'] = np.concatenate([labels['cls'], labels2['cls']], 0)
        return labels


class Mosaic(BaseMixTransform):
    """
    Mosaic augmentation for image datasets.

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
    The augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
        p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
        n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
        border (Tuple[int, int]): Border size for width and height.

    Methods:
        get_indexes: Returns a list of random indexes from the dataset.
        _mix_transform: Applies mixup transformation to the input image and labels.
        _mosaic3: Creates a 1x3 image mosaic.
        _mosaic4: Creates a 2x2 image mosaic.
        _mosaic9: Creates a 3x3 image mosaic.
        _update_labels: Updates labels with padding.
        _cat_labels: Concatenates labels and clips mosaic border instances.

    Examples:
        >>> from ultralytics.data.augment import Mosaic
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        >>> augmented_labels = mosaic_aug(original_labels)
    """

    def __init__(self, dataset, imgsz=640, p=1.0, n=4):
        """
        Initializes the Mosaic augmentation object.

        This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
        The augmentation is applied to a dataset with a given probability.

        Args:
            dataset (Any): The dataset on which the mosaic augmentation is applied.
            imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
            p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
            n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).

        Examples:
            >>> from ultralytics.data.augment import Mosaic
            >>> dataset = YourDataset(...)
            >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        """
        assert 0 <= p <= 1.0, f'The probability should be in range [0, 1], but got {p}.'
        assert n in {4, 9}, 'grid must be equal to 4 or 9.'
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = -imgsz // 2, -imgsz // 2
        self.n = n

    def get_indexes(self, buffer=True):
        """
        Returns a list of random indexes from the dataset for mosaic augmentation.

        This method selects random image indexes either from a buffer or from the entire dataset, depending on
        the 'buffer' parameter. It is used to choose images for creating mosaic augmentations.

        Args:
            buffer (bool): If True, selects images from the dataset buffer. If False, selects from the entire
                dataset.

        Returns:
            (List[int]): A list of random image indexes. The length of the list is n-1, where n is the number
                of images used in the mosaic (either 3 or 8, depending on whether n is 4 or 9).

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> indexes = mosaic.get_indexes()
            >>> print(len(indexes))  # Output: 3
        """
        if buffer:
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels):
        """
        Applies mosaic augmentation to the input image and labels.

        This method combines multiple images (3, 4, or 9) into a single mosaic image based on the 'n' attribute.
        It ensures that rectangular annotations are not present and that there are other images available for
        mosaic augmentation.

        Args:
            labels (Dict): A dictionary containing image data and annotations. Expected keys include:
                - 'rect_shape': Should be None as rect and mosaic are mutually exclusive.
                - 'mix_labels': A list of dictionaries containing data for other images to be used in the mosaic.

        Returns:
            (Dict): A dictionary containing the mosaic-augmented image and updated annotations.

        Raises:
            AssertionError: If 'rect_shape' is not None or if 'mix_labels' is empty.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> augmented_data = mosaic._mix_transform(labels)
        """
        assert labels.get('rect_shape', None) is None, 'rect and mosaic are mutually exclusive.'
        assert len(labels.get('mix_labels', [])), 'There are no other images for mosaic augment.'
        return self._mosaic3(labels) if self.n == 3 else self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)

    def _mosaic3(self, labels):
        """
        Creates a 1x3 image mosaic by combining three images.

        This method arranges three images in a horizontal layout, with the main image in the center and two
        additional images on either side. It's part of the Mosaic augmentation technique used in object detection.

        Args:
            labels (Dict): A dictionary containing image and label information for the main (center) image.
                Must include 'img' key with the image array, and 'mix_labels' key with a list of two
                dictionaries containing information for the side images.

        Returns:
            (Dict): A dictionary with the mosaic image and updated labels. Keys include:
                - 'img' (np.ndarray): The mosaic image array with shape (H, W, C).
                - Other keys from the input labels, updated to reflect the new image dimensions.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=3)
            >>> labels = {
            ...     "img": np.random.rand(480, 640, 3),
            ...     "mix_labels": [{"img": np.random.rand(480, 640, 3)} for _ in range(2)],
            ... }
            >>> result = mosaic._mosaic3(labels)
            >>> print(result["img"].shape)
            (640, 640, 3)
        """
        mosaic_labels = []
        s = self.imgsz
        for i in range(3):
            labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
            img = labels_patch['img']
            h, w = labels_patch.pop('resized_shape')
            if i == 0:
                img3 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)
                h0, w0 = h, w
                c = s, s, s + w, s + h
            elif i == 1:
                c = s + w0, s, s + w0 + w, s + h
            elif i == 2:
                c = s - w, s + h0 - h, s, s + h0
            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)
            img3[y1:y2, x1:x2] = img[y1 - padh:, x1 - padw:]
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels['img'] = img3[-self.border[0]:self.border[0], -self.border[1]:self.border[1]]
        return final_labels

    def _mosaic4(self, labels):
        """
        Creates a 2x2 image mosaic from four input images.

        This method combines four images into a single mosaic image by placing them in a 2x2 grid. It also
        updates the corresponding labels for each image in the mosaic.

        Args:
            labels (Dict): A dictionary containing image data and labels for the base image (index 0) and three
                additional images (indices 1-3) in the 'mix_labels' key.

        Returns:
            (Dict): A dictionary containing the mosaic image and updated labels. The 'img' key contains the mosaic
                image as a numpy array, and other keys contain the combined and adjusted labels for all four images.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> labels = {
            ...     "img": np.random.rand(480, 640, 3),
            ...     "mix_labels": [{"img": np.random.rand(480, 640, 3)} for _ in range(3)],
            ... }
            >>> result = mosaic._mosaic4(labels)
            >>> assert result["img"].shape == (1280, 1280, 3)
        """
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
        for i in range(4):
            labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
            img = labels_patch['img']
            h, w = labels_patch.pop('resized_shape')
            if i == 0:
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels['img'] = img4
        return final_labels

    def _mosaic9(self, labels):
        """
        Creates a 3x3 image mosaic from the input image and eight additional images.

        This method combines nine images into a single mosaic image. The input image is placed at the center,
        and eight additional images from the dataset are placed around it in a 3x3 grid pattern.

        Args:
            labels (Dict): A dictionary containing the input image and its associated labels. It should have
                the following keys:
                - 'img' (numpy.ndarray): The input image.
                - 'resized_shape' (Tuple[int, int]): The shape of the resized image (height, width).
                - 'mix_labels' (List[Dict]): A list of dictionaries containing information for the additional
                  eight images, each with the same structure as the input labels.

        Returns:
            (Dict): A dictionary containing the mosaic image and updated labels. It includes the following keys:
                - 'img' (numpy.ndarray): The final mosaic image.
                - Other keys from the input labels, updated to reflect the new mosaic arrangement.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=9)
            >>> input_labels = dataset[0]
            >>> mosaic_result = mosaic._mosaic9(input_labels)
            >>> mosaic_image = mosaic_result["img"]
        """
        mosaic_labels = []
        s = self.imgsz
        hp, wp = -1, -1
        for i in range(9):
            labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
            img = labels_patch['img']
            h, w = labels_patch.pop('resized_shape')
            if i == 0:
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)
                h0, w0 = h, w
                c = s, s, s + w, s + h
            elif i == 1:
                c = s, s - h, s + w, s
            elif i == 2:
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:
                c = s - w, s + h0 - hp - h, s, s + h0 - hp
            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)
            img9[y1:y2, x1:x2] = img[y1 - padh:, x1 - padw:]
            hp, wp = h, w
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels['img'] = img9[-self.border[0]:self.border[0], -self.border[1]:self.border[1]]
        return final_labels

    @staticmethod
    def _update_labels(labels, padw, padh):
        """
        Updates label coordinates with padding values.

        This method adjusts the bounding box coordinates of object instances in the labels by adding padding
        values. It also denormalizes the coordinates if they were previously normalized.

        Args:
            labels (Dict): A dictionary containing image and instance information.
            padw (int): Padding width to be added to the x-coordinates.
            padh (int): Padding height to be added to the y-coordinates.

        Returns:
            (Dict): Updated labels dictionary with adjusted instance coordinates.

        Examples:
            >>> labels = {"img": np.zeros((100, 100, 3)), "instances": Instances(...)}
            >>> padw, padh = 50, 50
            >>> updated_labels = Mosaic._update_labels(labels, padw, padh)
        """
        nh, nw = labels['img'].shape[:2]
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(nw, nh)
        labels['instances'].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels):
        """
        Concatenates and processes labels for mosaic augmentation.

        This method combines labels from multiple images used in mosaic augmentation, clips instances to the
        mosaic border, and removes zero-area boxes.

        Args:
            mosaic_labels (List[Dict]): A list of label dictionaries for each image in the mosaic.

        Returns:
            (Dict): A dictionary containing concatenated and processed labels for the mosaic image, including:
                - im_file (str): File path of the first image in the mosaic.
                - ori_shape (Tuple[int, int]): Original shape of the first image.
                - resized_shape (Tuple[int, int]): Shape of the mosaic image (imgsz * 2, imgsz * 2).
                - cls (np.ndarray): Concatenated class labels.
                - instances (Instances): Concatenated instance annotations.
                - mosaic_border (Tuple[int, int]): Mosaic border size.
                - texts (List[str], optional): Text labels if present in the original labels.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640)
            >>> mosaic_labels = [{"cls": np.array([0, 1]), "instances": Instances(...)} for _ in range(4)]
            >>> result = mosaic._cat_labels(mosaic_labels)
            >>> print(result.keys())
            dict_keys(['im_file', 'ori_shape', 'resized_shape', 'cls', 'instances', 'mosaic_border'])
        """
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2
        for labels in mosaic_labels:
            cls.append(labels['cls'])
            instances.append(labels['instances'])
        final_labels = {'im_file': mosaic_labels[0]['im_file'], 'ori_shape': mosaic_labels[0]['ori_shape'], 'resized_shape': (imgsz, imgsz), 'cls': np.concatenate(cls, 0), 'instances': Instances.concatenate(instances, axis=0), 'mosaic_border': self.border}
        final_labels['instances'].clip(imgsz, imgsz)
        good = final_labels['instances'].remove_zero_area_boxes()
        final_labels['cls'] = final_labels['cls'][good]
        if 'texts' in mosaic_labels[0]:
            final_labels['texts'] = mosaic_labels[0]['texts']
        return final_labels


class RandomFlip:
    """
    Applies a random horizontal or vertical flip to an image with a given probability.

    This class performs random image flipping and updates corresponding instance annotations such as
    bounding boxes and keypoints.

    Attributes:
        p (float): Probability of applying the flip. Must be between 0 and 1.
        direction (str): Direction of flip, either 'horizontal' or 'vertical'.
        flip_idx (array-like): Index mapping for flipping keypoints, if applicable.

    Methods:
        __call__: Applies the random flip transformation to an image and its annotations.

    Examples:
        >>> transform = RandomFlip(p=0.5, direction="horizontal")
        >>> result = transform({"img": image, "instances": instances})
        >>> flipped_image = result["img"]
        >>> flipped_instances = result["instances"]
    """

    def __init__(self, p=0.5, direction='horizontal', flip_idx=None) ->None:
        """
        Initializes the RandomFlip class with probability and direction.

        This class applies a random horizontal or vertical flip to an image with a given probability.
        It also updates any instances (bounding boxes, keypoints, etc.) accordingly.

        Args:
            p (float): The probability of applying the flip. Must be between 0 and 1.
            direction (str): The direction to apply the flip. Must be 'horizontal' or 'vertical'.
            flip_idx (List[int] | None): Index mapping for flipping keypoints, if any.

        Raises:
            AssertionError: If direction is not 'horizontal' or 'vertical', or if p is not between 0 and 1.

        Examples:
            >>> flip = RandomFlip(p=0.5, direction="horizontal")
            >>> flip = RandomFlip(p=0.7, direction="vertical", flip_idx=[1, 0, 3, 2, 5, 4])
        """
        assert direction in {'horizontal', 'vertical'}, f'Support direction `horizontal` or `vertical`, got {direction}'
        assert 0 <= p <= 1.0, f'The probability should be in range [0, 1], but got {p}.'
        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels):
        """
        Applies random flip to an image and updates any instances like bounding boxes or keypoints accordingly.

        This method randomly flips the input image either horizontally or vertically based on the initialized
        probability and direction. It also updates the corresponding instances (bounding boxes, keypoints) to
        match the flipped image.

        Args:
            labels (Dict): A dictionary containing the following keys:
                'img' (numpy.ndarray): The image to be flipped.
                'instances' (ultralytics.utils.instance.Instances): An object containing bounding boxes and
                    optionally keypoints.

        Returns:
            (Dict): The same dictionary with the flipped image and updated instances:
                'img' (numpy.ndarray): The flipped image.
                'instances' (ultralytics.utils.instance.Instances): Updated instances matching the flipped image.

        Examples:
            >>> labels = {"img": np.random.rand(640, 640, 3), "instances": Instances(...)}
            >>> random_flip = RandomFlip(p=0.5, direction="horizontal")
            >>> flipped_labels = random_flip(labels)
        """
        img = labels['img']
        instances = labels.pop('instances')
        instances.convert_bbox(format='xywh')
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w
        if self.direction == 'vertical' and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
        if self.direction == 'horizontal' and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        labels['img'] = np.ascontiguousarray(img)
        labels['instances'] = instances
        return labels


class RandomHSV:
    """
    Randomly adjusts the Hue, Saturation, and Value (HSV) channels of an image.

    This class applies random HSV augmentation to images within predefined limits set by hgain, sgain, and vgain.

    Attributes:
        hgain (float): Maximum variation for hue. Range is typically [0, 1].
        sgain (float): Maximum variation for saturation. Range is typically [0, 1].
        vgain (float): Maximum variation for value. Range is typically [0, 1].

    Methods:
        __call__: Applies random HSV augmentation to an image.

    Examples:
        >>> import numpy as np
        >>> from ultralytics.data.augment import RandomHSV
        >>> augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> labels = {"img": image}
        >>> augmented_labels = augmenter(labels)
        >>> augmented_image = augmented_labels["img"]
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) ->None:
        """
        Initializes the RandomHSV object for random HSV (Hue, Saturation, Value) augmentation.

        This class applies random adjustments to the HSV channels of an image within specified limits.

        Args:
            hgain (float): Maximum variation for hue. Should be in the range [0, 1].
            sgain (float): Maximum variation for saturation. Should be in the range [0, 1].
            vgain (float): Maximum variation for value. Should be in the range [0, 1].

        Examples:
            >>> hsv_aug = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> augmented_image = hsv_aug(image)
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels):
        """
        Applies random HSV augmentation to an image within predefined limits.

        This method modifies the input image by randomly adjusting its Hue, Saturation, and Value (HSV) channels.
        The adjustments are made within the limits set by hgain, sgain, and vgain during initialization.

        Args:
            labels (Dict): A dictionary containing image data and metadata. Must include an 'img' key with
                the image as a numpy array.

        Returns:
            (None): The function modifies the input 'labels' dictionary in-place, updating the 'img' key
                with the HSV-augmented image.

        Examples:
            >>> hsv_augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> labels = {"img": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)}
            >>> hsv_augmenter(labels)
            >>> augmented_img = labels["img"]
        """
        img = labels['img']
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = (x * r[0] % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return labels


def segment2box(segment, width=640, height=640):
    """
    Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy).

    Args:
        segment (torch.Tensor): the segment label
        width (int): the width of the image. Defaults to 640
        height (int): The height of the image. Defaults to 640

    Returns:
        (np.ndarray): the minimum and maximum x and y values of the segment.
    """
    x, y = segment.T
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x = x[inside]
    y = y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype) if any(x) else np.zeros(4, dtype=segment.dtype)


class RandomPerspective:
    """
    Implements random perspective and affine transformations on images and corresponding annotations.

    This class applies random rotations, translations, scaling, shearing, and perspective transformations
    to images and their associated bounding boxes, segments, and keypoints. It can be used as part of an
    augmentation pipeline for object detection and instance segmentation tasks.

    Attributes:
        degrees (float): Maximum absolute degree range for random rotations.
        translate (float): Maximum translation as a fraction of the image size.
        scale (float): Scaling factor range, e.g., scale=0.1 means 0.9-1.1.
        shear (float): Maximum shear angle in degrees.
        perspective (float): Perspective distortion factor.
        border (Tuple[int, int]): Mosaic border size as (x, y).
        pre_transform (Callable | None): Optional transform to apply before the random perspective.

    Methods:
        affine_transform: Applies affine transformations to the input image.
        apply_bboxes: Transforms bounding boxes using the affine matrix.
        apply_segments: Transforms segments and generates new bounding boxes.
        apply_keypoints: Transforms keypoints using the affine matrix.
        __call__: Applies the random perspective transformation to images and annotations.
        box_candidates: Filters transformed bounding boxes based on size and aspect ratio.

    Examples:
        >>> transform = RandomPerspective(degrees=10, translate=0.1, scale=0.1, shear=10)
        >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> labels = {"img": image, "cls": np.array([0, 1]), "instances": Instances(...)}
        >>> result = transform(labels)
        >>> transformed_image = result["img"]
        >>> transformed_instances = result["instances"]
    """

    def __init__(self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, border=(0, 0), pre_transform=None):
        """
        Initializes RandomPerspective object with transformation parameters.

        This class implements random perspective and affine transformations on images and corresponding bounding boxes,
        segments, and keypoints. Transformations include rotation, translation, scaling, and shearing.

        Args:
            degrees (float): Degree range for random rotations.
            translate (float): Fraction of total width and height for random translation.
            scale (float): Scaling factor interval, e.g., a scale factor of 0.5 allows a resize between 50%-150%.
            shear (float): Shear intensity (angle in degrees).
            perspective (float): Perspective distortion factor.
            border (Tuple[int, int]): Tuple specifying mosaic border (top/bottom, left/right).
            pre_transform (Callable | None): Function/transform to apply to the image before starting the random
                transformation.

        Examples:
            >>> transform = RandomPerspective(degrees=10.0, translate=0.1, scale=0.5, shear=5.0)
            >>> result = transform(labels)  # Apply random perspective to labels
        """
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.pre_transform = pre_transform

    def affine_transform(self, img, border):
        """
        Applies a sequence of affine transformations centered around the image center.

        This function performs a series of geometric transformations on the input image, including
        translation, perspective change, rotation, scaling, and shearing. The transformations are
        applied in a specific order to maintain consistency.

        Args:
            img (np.ndarray): Input image to be transformed.
            border (Tuple[int, int]): Border dimensions for the transformed image.

        Returns:
            (Tuple[np.ndarray, np.ndarray, float]): A tuple containing:
                - np.ndarray: Transformed image.
                - np.ndarray: 3x3 transformation matrix.
                - float: Scale factor applied during the transformation.

        Examples:
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> border = (10, 10)
            >>> transformed_img, matrix, scale = affine_transform(img, border)
        """
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -img.shape[1] / 2
        C[1, 2] = -img.shape[0] / 2
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]
        M = T @ S @ R @ P @ C
        if border[0] != 0 or border[1] != 0 or (M != np.eye(3)).any():
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, M, s

    def apply_bboxes(self, bboxes, M):
        """
        Apply affine transformation to bounding boxes.

        This function applies an affine transformation to a set of bounding boxes using the provided
        transformation matrix.

        Args:
            bboxes (torch.Tensor): Bounding boxes in xyxy format with shape (N, 4), where N is the number
                of bounding boxes.
            M (torch.Tensor): Affine transformation matrix with shape (3, 3).

        Returns:
            (torch.Tensor): Transformed bounding boxes in xyxy format with shape (N, 4).

        Examples:
            >>> bboxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])
            >>> M = torch.eye(3)
            >>> transformed_bboxes = apply_bboxes(bboxes, M)
        """
        n = len(bboxes)
        if n == 0:
            return bboxes
        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def apply_segments(self, segments, M):
        """
        Apply affine transformations to segments and generate new bounding boxes.

        This function applies affine transformations to input segments and generates new bounding boxes based on
        the transformed segments. It clips the transformed segments to fit within the new bounding boxes.

        Args:
            segments (np.ndarray): Input segments with shape (N, M, 2), where N is the number of segments and M is the
                number of points in each segment.
            M (np.ndarray): Affine transformation matrix with shape (3, 3).

        Returns:
            (Tuple[np.ndarray, np.ndarray]): A tuple containing:
                - New bounding boxes with shape (N, 4) in xyxy format.
                - Transformed and clipped segments with shape (N, M, 2).

        Examples:
            >>> segments = np.random.rand(10, 500, 2)  # 10 segments with 500 points each
            >>> M = np.eye(3)  # Identity transformation matrix
            >>> new_bboxes, new_segments = apply_segments(segments, M)
        """
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments
        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0)
        segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        return bboxes, segments

    def apply_keypoints(self, keypoints, M):
        """
        Applies affine transformation to keypoints.

        This method transforms the input keypoints using the provided affine transformation matrix. It handles
        perspective rescaling if necessary and updates the visibility of keypoints that fall outside the image
        boundaries after transformation.

        Args:
            keypoints (np.ndarray): Array of keypoints with shape (N, 17, 3), where N is the number of instances,
                17 is the number of keypoints per instance, and 3 represents (x, y, visibility).
            M (np.ndarray): 3x3 affine transformation matrix.

        Returns:
            (np.ndarray): Transformed keypoints array with the same shape as input (N, 17, 3).

        Examples:
            >>> random_perspective = RandomPerspective()
            >>> keypoints = np.random.rand(5, 17, 3)  # 5 instances, 17 keypoints each
            >>> M = np.eye(3)  # Identity transformation
            >>> transformed_keypoints = random_perspective.apply_keypoints(keypoints, M)
        """
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T
        xy = xy[:, :2] / xy[:, 2:3]
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > self.size[0]) | (xy[:, 1] > self.size[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def __call__(self, labels):
        """
        Applies random perspective and affine transformations to an image and its associated labels.

        This method performs a series of transformations including rotation, translation, scaling, shearing,
        and perspective distortion on the input image and adjusts the corresponding bounding boxes, segments,
        and keypoints accordingly.

        Args:
            labels (Dict): A dictionary containing image data and annotations.
                Must include:
                    'img' (ndarray): The input image.
                    'cls' (ndarray): Class labels.
                    'instances' (Instances): Object instances with bounding boxes, segments, and keypoints.
                May include:
                    'mosaic_border' (Tuple[int, int]): Border size for mosaic augmentation.

        Returns:
            (Dict): Transformed labels dictionary containing:
                - 'img' (np.ndarray): The transformed image.
                - 'cls' (np.ndarray): Updated class labels.
                - 'instances' (Instances): Updated object instances.
                - 'resized_shape' (Tuple[int, int]): New image shape after transformation.

        Examples:
            >>> transform = RandomPerspective()
            >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> labels = {
            ...     "img": image,
            ...     "cls": np.array([0, 1, 2]),
            ...     "instances": Instances(bboxes=np.array([[10, 10, 50, 50], [100, 100, 150, 150]])),
            ... }
            >>> result = transform(labels)
            >>> assert result["img"].shape[:2] == result["resized_shape"]
        """
        if self.pre_transform and 'mosaic_border' not in labels:
            labels = self.pre_transform(labels)
        labels.pop('ratio_pad', None)
        img = labels['img']
        cls = labels['cls']
        instances = labels.pop('instances')
        instances.convert_bbox(format='xyxy')
        instances.denormalize(*img.shape[:2][::-1])
        border = labels.pop('mosaic_border', self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2
        img, M, scale = self.affine_transform(img, border)
        bboxes = self.apply_bboxes(instances.bboxes, M)
        segments = instances.segments
        keypoints = instances.keypoints
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)
        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format='xyxy', normalized=False)
        new_instances.clip(*self.size)
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        i = self.box_candidates(box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.1)
        labels['instances'] = new_instances[i]
        labels['cls'] = cls[i]
        labels['img'] = img
        labels['resized_shape'] = img.shape[:2]
        return labels

    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        """
        Compute candidate boxes for further processing based on size and aspect ratio criteria.

        This method compares boxes before and after augmentation to determine if they meet specified
        thresholds for width, height, aspect ratio, and area. It's used to filter out boxes that have
        been overly distorted or reduced by the augmentation process.

        Args:
            box1 (numpy.ndarray): Original boxes before augmentation, shape (4, N) where n is the
                number of boxes. Format is [x1, y1, x2, y2] in absolute coordinates.
            box2 (numpy.ndarray): Augmented boxes after transformation, shape (4, N). Format is
                [x1, y1, x2, y2] in absolute coordinates.
            wh_thr (float): Width and height threshold in pixels. Boxes smaller than this in either
                dimension are rejected.
            ar_thr (float): Aspect ratio threshold. Boxes with an aspect ratio greater than this
                value are rejected.
            area_thr (float): Area ratio threshold. Boxes with an area ratio (new/old) less than
                this value are rejected.
            eps (float): Small epsilon value to prevent division by zero.

        Returns:
            (numpy.ndarray): Boolean array of shape (n,) indicating which boxes are candidates.
                True values correspond to boxes that meet all criteria.

        Examples:
            >>> random_perspective = RandomPerspective()
            >>> box1 = np.array([[0, 0, 100, 100], [0, 0, 50, 50]]).T
            >>> box2 = np.array([[10, 10, 90, 90], [5, 5, 45, 45]]).T
            >>> candidates = random_perspective.box_candidates(box1, box2)
            >>> print(candidates)
            [True True]
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def v8_transforms(dataset, imgsz, hyp, stretch=False):
    """
    Applies a series of image transformations for training.

    This function creates a composition of image augmentation techniques to prepare images for YOLO training.
    It includes operations such as mosaic, copy-paste, random perspective, mixup, and various color adjustments.

    Args:
        dataset (Dataset): The dataset object containing image data and annotations.
        imgsz (int): The target image size for resizing.
        hyp (Dict): A dictionary of hyperparameters controlling various aspects of the transformations.
        stretch (bool): If True, applies stretching to the image. If False, uses LetterBox resizing.

    Returns:
        (Compose): A composition of image transformations to be applied to the dataset.

    Examples:
        >>> from ultralytics.data.dataset import YOLODataset
        >>> dataset = YOLODataset(img_path="path/to/images", imgsz=640)
        >>> hyp = {"mosaic": 1.0, "copy_paste": 0.5, "degrees": 10.0, "translate": 0.2, "scale": 0.9}
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data = transforms(dataset[0])
    """
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(degrees=hyp.degrees, translate=hyp.translate, scale=hyp.scale, shear=hyp.shear, perspective=hyp.perspective, pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)))
    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == 'flip':
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(CopyPaste(dataset, pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]), p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    flip_idx = dataset.data.get('flip_idx', [])
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get('kpt_shape', None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and len(flip_idx) != kpt_shape[0]:
            raise ValueError(f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}')
    return Compose([pre_transform, MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup), Albumentations(p=1.0), RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v), RandomFlip(direction='vertical', p=hyp.flipud), RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx)])


def exif_size(img: 'Image.Image'):
    """Returns exif-corrected PIL size."""
    s = img.size
    if img.format == 'JPEG':
        try:
            exif = img.getexif()
            if exif:
                rotation = exif.get(274, None)
                if rotation in {6, 8}:
                    s = s[1], s[0]
        except Exception:
            pass
    return s


def segments2boxes(segments):
    """
    It converts segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh).

    Args:
        segments (list): list of segments, each segment is a list of points, each point is a list of x, y coordinates

    Returns:
        (np.ndarray): the xywh coordinates of the bounding boxes.
    """
    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xyxy2xywh(np.array(boxes))


def verify_image_label(args):
    """Verify one image-label pair."""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, '', [], None
    try:
        im = Image.open(im_file)
        im.verify()
        shape = exif_size(im)
        shape = shape[1], shape[0]
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}. {FORMATS_HELP_MSG}'
        if im.format.lower() in {'jpg', 'jpeg'}:
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'
        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and not keypoint:
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                if keypoint:
                    assert lb.shape[1] == 5 + nkpt * ndim, f'labels require {5 + nkpt * ndim} columns each'
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    points = lb[:, 1:]
                assert points.max() <= 1, f'non-normalized or out of bounds coordinates {points[points > 1]}'
                assert lb.min() >= 0, f'negative label values {lb[lb < 0]}'
                max_cls = lb[:, 0].max()
                assert max_cls <= num_cls, f'Label class {int(max_cls)} exceeds dataset class count {num_cls}. Possible class labels are 0-{num_cls - 1}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1
                lb = np.zeros((0, 5 + nkpt * ndim if keypoint else 5), dtype=np.float32)
        else:
            nm = 1
            lb = np.zeros((0, 5 + nkpt * ndim if keypoints else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


GITHUB_ASSETS_NAMES = [f'yolov8{k}{suffix}.pt' for k in 'nsmlx' for suffix in ('', '-cls', '-seg', '-pose', '-obb', '-oiv7')] + [f'yolo11{k}{suffix}.pt' for k in 'nsmlx' for suffix in ('', '-cls', '-seg', '-pose', '-obb')] + [f'yolov5{k}{resolution}u.pt' for k in 'nsmlx' for resolution in ('', '6')] + [f'yolov3{k}u.pt' for k in ('', '-spp', '-tiny')] + [f'yolov8{k}-world.pt' for k in 'smlx'] + [f'yolov8{k}-worldv2.pt' for k in 'smlx'] + [f'yolov9{k}.pt' for k in 'tsmce'] + [f'yolov10{k}.pt' for k in 'nsmblx'] + [f'yolo_nas_{k}.pt' for k in 'sml'] + [f'sam_{k}.pt' for k in 'bl'] + [f'FastSAM-{k}.pt' for k in 'sx'] + [f'rtdetr-{k}.pt' for k in 'lx'] + ['mobile_sam.pt'] + ['calibration_image_sample_data_20x128x128x3_float32.npy.zip']


GITHUB_ASSETS_REPO = 'ultralytics/assets'


def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory. If
    the current file is not part of a git repository, returns None.

    Returns:
        (Path | None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / '.git').is_dir():
            return d


class JSONDict(dict):
    """
    A dictionary-like class that provides JSON persistence for its contents.

    This class extends the built-in dictionary to automatically save its contents to a JSON file whenever they are
    modified. It ensures thread-safe operations using a lock.

    Attributes:
        file_path (Path): The path to the JSON file used for persistence.
        lock (threading.Lock): A lock object to ensure thread-safe operations.

    Methods:
        _load: Loads the data from the JSON file into the dictionary.
        _save: Saves the current state of the dictionary to the JSON file.
        __setitem__: Stores a key-value pair and persists it to disk.
        __delitem__: Removes an item and updates the persistent storage.
        update: Updates the dictionary and persists changes.
        clear: Clears all entries and updates the persistent storage.

    Examples:
        >>> json_dict = JSONDict("data.json")
        >>> json_dict["key"] = "value"
        >>> print(json_dict["key"])
        value
        >>> del json_dict["key"]
        >>> json_dict.update({"new_key": "new_value"})
        >>> json_dict.clear()
    """

    def __init__(self, file_path: 'Union[str, Path]'='data.json'):
        """Initialize a JSONDict object with a specified file path for JSON persistence."""
        super().__init__()
        self.file_path = Path(file_path)
        self.lock = Lock()
        self._load()

    def _load(self):
        """Load the data from the JSON file into the dictionary."""
        try:
            if self.file_path.exists():
                with open(self.file_path) as f:
                    self.update(json.load(f))
        except json.JSONDecodeError:
            None
        except Exception as e:
            None

    def _save(self):
        """Save the current state of the dictionary to the JSON file."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, 'w') as f:
                json.dump(dict(self), f, indent=2, default=self._json_default)
        except Exception as e:
            None

    @staticmethod
    def _json_default(obj):
        """Handle JSON serialization of Path objects."""
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

    def __setitem__(self, key, value):
        """Store a key-value pair and persist to disk."""
        with self.lock:
            super().__setitem__(key, value)
            self._save()

    def __delitem__(self, key):
        """Remove an item and update the persistent storage."""
        with self.lock:
            super().__delitem__(key)
            self._save()

    def __str__(self):
        """Return a pretty-printed JSON string representation of the dictionary."""
        return f'JSONDict("{self.file_path}"):\n{json.dumps(dict(self), indent=2, ensure_ascii=False, default=self._json_default)}'

    def update(self, *args, **kwargs):
        """Update the dictionary and persist changes."""
        with self.lock:
            super().update(*args, **kwargs)
            self._save()

    def clear(self):
        """Clear all entries and update the persistent storage."""
        with self.lock:
            super().clear()
            self._save()


def get_user_config_dir(sub_dir='Ultralytics'):
    """
    Return the appropriate config directory based on the environment operating system.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    if WINDOWS:
        path = Path.home() / 'AppData' / 'Roaming' / sub_dir
    elif MACOS:
        path = Path.home() / 'Library' / 'Application Support' / sub_dir
    elif LINUX:
        path = Path.home() / '.config' / sub_dir
    else:
        raise ValueError(f'Unsupported operating system: {platform.system()}')
    if not is_dir_writeable(path.parent):
        LOGGER.warning(f"WARNING ⚠️ user config directory '{path}' is not writeable, defaulting to '/tmp' or CWD.Alternatively you can define a YOLO_CONFIG_DIR environment variable for this path.")
        path = Path('/tmp') / sub_dir if is_dir_writeable('/tmp') else Path().cwd() / sub_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_github_assets(repo='ultralytics/assets', version='latest', retry=False):
    """
    Retrieve the specified version's tag and assets from a GitHub repository. If the version is not specified, the
    function fetches the latest release assets.

    Args:
        repo (str, optional): The GitHub repository in the format 'owner/repo'. Defaults to 'ultralytics/assets'.
        version (str, optional): The release version to fetch assets from. Defaults to 'latest'.
        retry (bool, optional): Flag to retry the request in case of a failure. Defaults to False.

    Returns:
        (tuple): A tuple containing the release tag and a list of asset names.

    Example:
        ```python
        tag, assets = get_github_assets(repo="ultralytics/assets", version="latest")
        ```
    """
    if version != 'latest':
        version = f'tags/{version}'
    url = f'https://api.github.com/repos/{repo}/releases/{version}'
    r = requests.get(url)
    if r.status_code != 200 and r.reason != 'rate limit exceeded' and retry:
        r = requests.get(url)
    if r.status_code != 200:
        LOGGER.warning(f'⚠️ GitHub assets check failure for {url}: {r.status_code} {r.reason}')
        return '', []
    data = r.json()
    return data['tag_name'], [x['name'] for x in data['assets']]


def get_google_drive_file_info(link):
    """
    Retrieves the direct download link and filename for a shareable Google Drive file link.

    Args:
        link (str): The shareable link of the Google Drive file.

    Returns:
        (str): Direct download URL for the Google Drive file.
        (str): Original filename of the Google Drive file. If filename extraction fails, returns None.

    Example:
        ```python
        from ultralytics.utils.downloads import get_google_drive_file_info

        link = "https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link"
        url, filename = get_google_drive_file_info(link)
        ```
    """
    file_id = link.split('/d/')[1].split('/view')[0]
    drive_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    filename = None
    with requests.Session() as session:
        response = session.get(drive_url, stream=True)
        if 'quota exceeded' in str(response.content.lower()):
            raise ConnectionError(emojis(f'❌  Google Drive file download quota exceeded. Please try again later or download this file manually at {link}.'))
        for k, v in response.cookies.items():
            if k.startswith('download_warning'):
                drive_url += f'&confirm={v}'
        cd = response.headers.get('content-disposition')
        if cd:
            filename = re.findall('filename="(.+)"', cd)[0]
    return drive_url, filename


def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX'), exist_ok=False, progress=True):
    """
    Unzips a *.zip file to the specified path, excluding files containing strings in the exclude list.

    If the zipfile does not contain a single top-level directory, the function will create a new
    directory with the same name as the zipfile (without the extension) to extract its contents.
    If a path is not provided, the function will use the parent directory of the zipfile as the default path.

    Args:
        file (str): The path to the zipfile to be extracted.
        path (str, optional): The path to extract the zipfile to. Defaults to None.
        exclude (tuple, optional): A tuple of filename strings to be excluded. Defaults to ('.DS_Store', '__MACOSX').
        exist_ok (bool, optional): Whether to overwrite existing contents if they exist. Defaults to False.
        progress (bool, optional): Whether to display a progress bar. Defaults to True.

    Raises:
        BadZipFile: If the provided file does not exist or is not a valid zipfile.

    Returns:
        (Path): The path to the directory where the zipfile was extracted.

    Example:
        ```python
        from ultralytics.utils.downloads import unzip_file

        dir = unzip_file("path/to/file.zip")
        ```
    """
    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent
    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        top_level_dirs = {Path(f).parts[0] for f in files}
        unzip_as_dir = len(top_level_dirs) == 1
        if unzip_as_dir:
            extract_path = path
            path = Path(path) / list(top_level_dirs)[0]
        else:
            path = extract_path = Path(path) / Path(file).stem
        if path.exists() and any(path.iterdir()) and not exist_ok:
            LOGGER.warning(f'WARNING ⚠️ Skipping {file} unzip as destination directory {path} is not empty.')
            return path
        for f in TQDM(files, desc=f'Unzipping {file} to {Path(path).resolve()}...', unit='file', disable=not progress):
            if '..' in Path(f).parts:
                LOGGER.warning(f'Potentially insecure file path: {f}, skipping extraction.')
                continue
            zipObj.extract(f, extract_path)
    return path


def safe_download(url, file=None, dir=None, unzip=True, delete=False, curl=False, retry=3, min_bytes=1.0, exist_ok=False, progress=True):
    """
    Downloads files from a URL, with options for retrying, unzipping, and deleting the downloaded file.

    Args:
        url (str): The URL of the file to be downloaded.
        file (str, optional): The filename of the downloaded file.
            If not provided, the file will be saved with the same name as the URL.
        dir (str, optional): The directory to save the downloaded file.
            If not provided, the file will be saved in the current working directory.
        unzip (bool, optional): Whether to unzip the downloaded file. Default: True.
        delete (bool, optional): Whether to delete the downloaded file after unzipping. Default: False.
        curl (bool, optional): Whether to use curl command line tool for downloading. Default: False.
        retry (int, optional): The number of times to retry the download in case of failure. Default: 3.
        min_bytes (float, optional): The minimum number of bytes that the downloaded file should have, to be considered
            a successful download. Default: 1E0.
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping. Defaults to False.
        progress (bool, optional): Whether to display a progress bar during the download. Default: True.

    Example:
        ```python
        from ultralytics.utils.downloads import safe_download

        link = "https://ultralytics.com/assets/bus.jpg"
        path = safe_download(link)
        ```
    """
    gdrive = url.startswith('https://drive.google.com/')
    if gdrive:
        url, file = get_google_drive_file_info(url)
    f = Path(dir or '.') / (file or url2file(url))
    if '://' not in str(url) and Path(url).is_file():
        f = Path(url)
    elif not f.is_file():
        uri = (url if gdrive else clean_url(url)).replace('https://github.com/ultralytics/assets/releases/download/v0.0.0/', 'https://ultralytics.com/assets/')
        desc = f"Downloading {uri} to '{f}'"
        LOGGER.info(f'{desc}...')
        f.parent.mkdir(parents=True, exist_ok=True)
        check_disk_space(url, path=f.parent)
        for i in range(retry + 1):
            try:
                if curl or i > 0:
                    s = 'sS' * (not progress)
                    r = subprocess.run(['curl', '-#', f'-{s}L', url, '-o', f, '--retry', '3', '-C', '-']).returncode
                    assert r == 0, f'Curl return value {r}'
                else:
                    method = 'torch'
                    if method == 'torch':
                        torch.hub.download_url_to_file(url, f, progress=progress)
                    else:
                        with request.urlopen(url) as response, TQDM(total=int(response.getheader('Content-Length', 0)), desc=desc, disable=not progress, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                            with open(f, 'wb') as f_opened:
                                for data in response:
                                    f_opened.write(data)
                                    pbar.update(len(data))
                if f.exists():
                    if f.stat().st_size > min_bytes:
                        break
                    f.unlink()
            except Exception as e:
                if i == 0 and not is_online():
                    raise ConnectionError(emojis(f'❌  Download failure for {uri}. Environment is not online.')) from e
                elif i >= retry:
                    raise ConnectionError(emojis(f'❌  Download failure for {uri}. Retry limit reached.')) from e
                LOGGER.warning(f'⚠️ Download failure, retrying {i + 1}/{retry} {uri}...')
    if unzip and f.exists() and f.suffix in {'', '.zip', '.tar', '.gz'}:
        unzip_dir = (dir or f.parent).resolve()
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir, exist_ok=exist_ok, progress=progress)
        elif f.suffix in {'.tar', '.gz'}:
            LOGGER.info(f'Unzipping {f} to {unzip_dir}...')
            subprocess.run(['tar', 'xf' if f.suffix == '.tar' else 'xfz', f, '--directory', unzip_dir], check=True)
        if delete:
            f.unlink()
        return unzip_dir


def attempt_download_asset(file, repo='ultralytics/assets', release='v8.3.0', **kwargs):
    """
    Attempt to download a file from GitHub release assets if it is not found locally. The function checks for the file
    locally first, then tries to download it from the specified GitHub repository release.

    Args:
        file (str | Path): The filename or file path to be downloaded.
        repo (str, optional): The GitHub repository in the format 'owner/repo'. Defaults to 'ultralytics/assets'.
        release (str, optional): The specific release version to be downloaded. Defaults to 'v8.3.0'.
        **kwargs (any): Additional keyword arguments for the download process.

    Returns:
        (str): The path to the downloaded file.

    Example:
        ```python
        file_path = attempt_download_asset("yolo11n.pt", repo="ultralytics/assets", release="latest")
        ```
    """
    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ''))
    if file.exists():
        return str(file)
    elif (SETTINGS['weights_dir'] / file).exists():
        return str(SETTINGS['weights_dir'] / file)
    else:
        name = Path(parse.unquote(str(file))).name
        download_url = f'https://github.com/{repo}/releases/download'
        if str(file).startswith(('http:/', 'https:/')):
            url = str(file).replace(':/', '://')
            file = url2file(name)
            if Path(file).is_file():
                LOGGER.info(f'Found {clean_url(url)} locally at {file}')
            else:
                safe_download(url=url, file=file, min_bytes=100000.0, **kwargs)
        elif repo == GITHUB_ASSETS_REPO and name in GITHUB_ASSETS_NAMES:
            safe_download(url=f'{download_url}/{release}/{name}', file=file, min_bytes=100000.0, **kwargs)
        else:
            tag, assets = get_github_assets(repo, release)
            if not assets:
                tag, assets = get_github_assets(repo)
            if name in assets:
                safe_download(url=f'{download_url}/{tag}/{name}', file=file, min_bytes=100000.0, **kwargs)
        return str(file)


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()


def seed_worker(worker_id):
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()
    nw = min(os.cpu_count() // max(nd, 1), workers)
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(dataset=dataset, batch_size=batch, shuffle=shuffle and sampler is None, num_workers=nw, sampler=sampler, pin_memory=PIN_MEMORY, collate_fn=getattr(dataset, 'collate_fn', None), worker_init_fn=seed_worker, generator=generator)


def _log_scalars(scalars, step=0):
    """Logs scalar values to TensorBoard."""
    if WRITER:
        for k, v in scalars.items():
            WRITER.add_scalar(k, v, step)


def on_fit_epoch_end(trainer):
    """Logs epoch metrics at end of training epoch."""
    _log_scalars(trainer.metrics, trainer.epoch + 1)


PREFIX = colorstr('Ultralytics: ')


def on_pretrain_routine_start(trainer):
    """Initialize TensorBoard logging with SummaryWriter."""
    if SummaryWriter:
        try:
            global WRITER
            WRITER = SummaryWriter(str(trainer.save_dir))
            LOGGER.info(f"{PREFIX}Start with 'tensorboard --logdir {trainer.save_dir}', view at http://localhost:6006/")
        except Exception as e:
            LOGGER.warning(f'{PREFIX}WARNING ⚠️ TensorBoard not initialized correctly, not logging this run. {e}')


def on_train_epoch_end(trainer):
    """Logs scalar statistics at the end of a training epoch."""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix='train'), trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)


def _log_tensorboard_graph(trainer):
    """Log model graph to TensorBoard."""
    imgsz = trainer.args.imgsz
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    p = next(trainer.model.parameters())
    im = torch.zeros((1, 3, *imgsz), device=p.device, dtype=p.dtype)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        warnings.simplefilter('ignore', category=torch.jit.TracerWarning)
        try:
            trainer.model.eval()
            WRITER.add_graph(torch.jit.trace(de_parallel(trainer.model), im, strict=False), [])
            LOGGER.info(f'{PREFIX}model graph visualization added ✅')
            return
        except Exception:
            try:
                model = deepcopy(de_parallel(trainer.model))
                model.eval()
                model = model.fuse(verbose=False)
                for m in model.modules():
                    if hasattr(m, 'export'):
                        m.export = True
                        m.format = 'torchscript'
                model(im)
                WRITER.add_graph(torch.jit.trace(model, im, strict=False), [])
                LOGGER.info(f'{PREFIX}model graph visualization added ✅')
            except Exception as e:
                LOGGER.warning(f'{PREFIX}WARNING ⚠️ TensorBoard graph visualization failure {e}')


def on_train_start(trainer):
    """Log TensorBoard graph."""
    if WRITER:
        _log_tensorboard_graph(trainer)


def check_class_names(names):
    """
    Check class names.

    Map imagenet class codes to human-readable names if required. Convert lists to dicts.
    """
    if isinstance(names, list):
        names = dict(enumerate(names))
    if isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(f'{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices {min(names.keys())}-{max(names.keys())} defined in your dataset YAML.')
        if isinstance(names[0], str) and names[0].startswith('n0'):
            names_map = yaml_load(ROOT / 'cfg/datasets/ImageNet.yaml')['map']
            names = {k: names_map[v] for k, v in names.items()}
    return names


def check_cls_dataset(dataset, split=''):
    """
    Checks a classification dataset such as Imagenet.

    This function accepts a `dataset` name and attempts to retrieve the corresponding dataset information.
    If the dataset is not found locally, it attempts to download the dataset from the internet and save it locally.

    Args:
        dataset (str | Path): The name of the dataset.
        split (str, optional): The split of the dataset. Either 'val', 'test', or ''. Defaults to ''.

    Returns:
        (dict): A dictionary containing the following keys:
            - 'train' (Path): The directory path containing the training set of the dataset.
            - 'val' (Path): The directory path containing the validation set of the dataset.
            - 'test' (Path): The directory path containing the test set of the dataset.
            - 'nc' (int): The number of classes in the dataset.
            - 'names' (dict): A dictionary of class names in the dataset.
    """
    if str(dataset).startswith(('http:/', 'https:/')):
        dataset = safe_download(dataset, dir=DATASETS_DIR, unzip=True, delete=False)
    elif Path(dataset).suffix in {'.zip', '.tar', '.gz'}:
        file = check_file(dataset)
        dataset = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
    dataset = Path(dataset)
    data_dir = (dataset if dataset.is_dir() else DATASETS_DIR / dataset).resolve()
    if not data_dir.is_dir():
        LOGGER.warning(f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...')
        t = time.time()
        if str(dataset) == 'imagenet':
            subprocess.run(f"bash {ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
        else:
            url = f'https://github.com/ultralytics/assets/releases/download/v0.0.0/{dataset}.zip'
            download(url, dir=data_dir.parent)
        s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
        LOGGER.info(s)
    train_set = data_dir / 'train'
    val_set = data_dir / 'val' if (data_dir / 'val').exists() else data_dir / 'validation' if (data_dir / 'validation').exists() else None
    test_set = data_dir / 'test' if (data_dir / 'test').exists() else None
    if split == 'val' and not val_set:
        LOGGER.warning("WARNING ⚠️ Dataset 'split=val' not found, using 'split=test' instead.")
    elif split == 'test' and not test_set:
        LOGGER.warning("WARNING ⚠️ Dataset 'split=test' not found, using 'split=val' instead.")
    nc = len([x for x in (data_dir / 'train').glob('*') if x.is_dir()])
    names = [x.name for x in (data_dir / 'train').iterdir() if x.is_dir()]
    names = dict(enumerate(sorted(names)))
    for k, v in {'train': train_set, 'val': val_set, 'test': test_set}.items():
        prefix = f"{colorstr(f'{k}:')} {v}..."
        if v is None:
            LOGGER.info(prefix)
        else:
            files = [path for path in v.rglob('*.*') if path.suffix[1:].lower() in IMG_FORMATS]
            nf = len(files)
            nd = len({file.parent for file in files})
            if nf == 0:
                if k == 'train':
                    raise FileNotFoundError(emojis(f"{dataset} '{k}:' no training images found ❌ "))
                else:
                    LOGGER.warning(f'{prefix} found {nf} images in {nd} classes: WARNING ⚠️ no images found')
            elif nd != nc:
                LOGGER.warning(f'{prefix} found {nf} images in {nd} classes: ERROR ❌️ requires {nc} classes, not {nd}')
            else:
                LOGGER.info(f'{prefix} found {nf} images in {nd} classes ✅ ')
    return {'train': train_set, 'val': val_set, 'test': test_set, 'nc': nc, 'names': names}


class ThreadingLocked:
    """
    A decorator class for ensuring thread-safe execution of a function or method. This class can be used as a decorator
    to make sure that if the decorated function is called from multiple threads, only one thread at a time will be able
    to execute the function.

    Attributes:
        lock (threading.Lock): A lock object used to manage access to the decorated function.

    Example:
        ```python
        from ultralytics.utils import ThreadingLocked

        @ThreadingLocked()
        def my_function():
            # Your code here
        ```
    """

    def __init__(self):
        """Initializes the decorator class for thread-safe execution of a function or method."""
        self.lock = threading.Lock()

    def __call__(self, f):
        """Run thread-safe execution of function or method."""
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            """Applies thread-safety to the decorated function or method."""
            with self.lock:
                return f(*args, **kwargs)
        return decorated


def is_ascii(s) ->bool:
    """
    Check if a string is composed of only ASCII characters.

    Args:
        s (str): String to be checked.

    Returns:
        (bool): True if the string is composed only of ASCII characters, False otherwise.
    """
    s = str(s)
    return all(ord(c) < 128 for c in s)


def check_det_dataset(dataset, autodownload=True):
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        autodownload (bool, optional): Whether to automatically download the dataset if not found. Defaults to True.

    Returns:
        (dict): Parsed dataset information and paths.
    """
    file = check_file(dataset)
    extract_dir = ''
    if zipfile.is_zipfile(file) or is_tarfile(file):
        new_dir = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
        file = find_dataset_yaml(DATASETS_DIR / new_dir)
        extract_dir, autodownload = file.parent, False
    data = yaml_load(file, append_filename=True)
    for k in ('train', 'val'):
        if k not in data:
            if k != 'val' or 'validation' not in data:
                raise SyntaxError(emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs."))
            LOGGER.info("WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format.")
            data['val'] = data.pop('validation')
    if 'names' not in data and 'nc' not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if 'names' in data and 'nc' in data and len(data['names']) != data['nc']:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if 'names' not in data:
        data['names'] = [f'class_{i}' for i in range(data['nc'])]
    else:
        data['nc'] = len(data['names'])
    data['names'] = check_class_names(data['names'])
    path = Path(extract_dir or data.get('path') or Path(data.get('yaml_file', '')).parent)
    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()
    data['path'] = path
    for k in ('train', 'val', 'test', 'minival'):
        if data.get(k):
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]
    val, s = (data.get(x) for x in ('val', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]
        if not all(x.exists() for x in val):
            name = clean_url(dataset)
            m = f"\nDataset '{name}' images not found ⚠️, missing path '{[x for x in val if not x.exists()][0]}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_FILE}'"
                raise FileNotFoundError(m)
            t = time.time()
            r = None
            if s.startswith('http') and s.endswith('.zip'):
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith('bash '):
                LOGGER.info(f'Running {s} ...')
                r = os.system(s)
            else:
                exec(s, {'yaml': data})
            dt = f'({round(time.time() - t, 1)}s)'
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in {0, None} else f'failure {dt} ❌'
            LOGGER.info(f'Dataset download {s}\n')
    check_font('Arial.ttf' if is_ascii(data['names']) else 'Arial.Unicode.ttf')
    return data


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """
    Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int | cList[int]): Image size.
        stride (int): Stride value.
        min_dim (int): Minimum number of dimensions.
        max_dim (int): Maximum number of dimensions.
        floor (int): Minimum allowed value for image size.

    Returns:
        (List[int]): Updated image size.
    """
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):
        imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
    else:
        raise TypeError(f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'")
    if len(imgsz) > max_dim:
        msg = "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        if max_dim != 1:
            raise ValueError(f'imgsz={imgsz} is not a valid image size. {msg}')
        LOGGER.warning(f"WARNING ⚠️ updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]
    if sz != imgsz:
        LOGGER.warning(f'WARNING ⚠️ imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}')
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz
    return sz


def check_is_path_safe(basedir, path):
    """
    Check if the resolved path is under the intended directory to prevent path traversal.

    Args:
        basedir (Path | str): The intended directory.
        path (Path | str): The path to check.

    Returns:
        (bool): True if the path is safe, False otherwise.
    """
    base_dir_resolved = Path(basedir).resolve()
    path_resolved = Path(path).resolve()
    return path_resolved.exists() and path_resolved.parts[:len(base_dir_resolved.parts)] == base_dir_resolved.parts


def default_class_names(data=None):
    """Applies default class names to an input YAML file or returns numerical class names."""
    if data:
        try:
            return yaml_load(check_yaml(data))['names']
        except Exception:
            pass
    return {i: f'class{i}' for i in range(999)}


def export_formats():
    """Ultralytics YOLO export formats."""
    x = [['PyTorch', '-', '.pt', True, True], ['TorchScript', 'torchscript', '.torchscript', True, True], ['ONNX', 'onnx', '.onnx', True, True], ['OpenVINO', 'openvino', '_openvino_model', True, False], ['TensorRT', 'engine', '.engine', False, True], ['CoreML', 'coreml', '.mlpackage', True, False], ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True], ['TensorFlow GraphDef', 'pb', '.pb', True, True], ['TensorFlow Lite', 'tflite', '.tflite', True, False], ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', True, False], ['TensorFlow.js', 'tfjs', '_web_model', True, False], ['PaddlePaddle', 'paddle', '_paddle_model', True, True], ['MNN', 'mnn', '.mnn', True, True], ['NCNN', 'ncnn', '_ncnn_model', True, True]]
    return dict(zip(['Format', 'Argument', 'Suffix', 'CPU', 'GPU'], zip(*x)))


def file_size(path):
    """Returns the size of a file or directory in megabytes (MB)."""
    if isinstance(path, (str, Path)):
        mb = 1 << 20
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / mb
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    return 0.0


def gd_outputs(gd):
    """TensorFlow GraphDef model output node names."""
    name_list, input_list = [], []
    for node in gd.node:
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(f'{x}:0' for x in list(set(name_list) - set(input_list)) if not x.startswith('NoOp'))


def cfg2dict(cfg):
    """
    Converts a configuration object to a dictionary.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration object to be converted. Can be a file path,
            a string, a dictionary, or a SimpleNamespace object.

    Returns:
        (Dict): Configuration object in dictionary format.

    Examples:
        Convert a YAML file path to a dictionary:
        >>> config_dict = cfg2dict("config.yaml")

        Convert a SimpleNamespace to a dictionary:
        >>> from types import SimpleNamespace
        >>> config_sn = SimpleNamespace(param1="value1", param2="value2")
        >>> config_dict = cfg2dict(config_sn)

        Pass through an already existing dictionary:
        >>> config_dict = cfg2dict({"param1": "value1", "param2": "value2"})

    Notes:
        - If cfg is a path or string, it's loaded as YAML and converted to a dictionary.
        - If cfg is a SimpleNamespace object, it's converted to a dictionary using vars().
        - If cfg is already a dictionary, it's returned unchanged.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    return cfg


CFG_BOOL_KEYS = {'save', 'exist_ok', 'verbose', 'deterministic', 'single_cls', 'rect', 'cos_lr', 'overlap_mask', 'val', 'save_json', 'save_hybrid', 'half', 'dnn', 'plots', 'show', 'save_txt', 'save_conf', 'save_crop', 'save_frames', 'show_labels', 'show_conf', 'visualize', 'augment', 'agnostic_nms', 'retina_masks', 'show_boxes', 'keras', 'optimize', 'int8', 'dynamic', 'simplify', 'nms', 'profile', 'multi_scale'}


CFG_FLOAT_KEYS = {'warmup_epochs', 'box', 'cls', 'dfl', 'degrees', 'shear', 'time', 'workspace', 'batch'}


CFG_FRACTION_KEYS = {'dropout', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_momentum', 'warmup_bias_lr', 'label_smoothing', 'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale', 'perspective', 'flipud', 'fliplr', 'bgr', 'mosaic', 'mixup', 'copy_paste', 'conf', 'iou', 'fraction'}


CFG_INT_KEYS = {'epochs', 'patience', 'workers', 'seed', 'close_mosaic', 'mask_ratio', 'max_det', 'vid_stride', 'line_width', 'nbs', 'save_period'}


def check_cfg(cfg, hard=True):
    """
    Checks configuration argument types and values for the Ultralytics library.

    This function validates the types and values of configuration arguments, ensuring correctness and converting
    them if necessary. It checks for specific key types defined in global variables such as CFG_FLOAT_KEYS,
    CFG_FRACTION_KEYS, CFG_INT_KEYS, and CFG_BOOL_KEYS.

    Args:
        cfg (Dict): Configuration dictionary to validate.
        hard (bool): If True, raises exceptions for invalid types and values; if False, attempts to convert them.

    Examples:
        >>> config = {
        ...     "epochs": 50,  # valid integer
        ...     "lr0": 0.01,  # valid float
        ...     "momentum": 1.2,  # invalid float (out of 0.0-1.0 range)
        ...     "save": "true",  # invalid bool
        ... }
        >>> check_cfg(config, hard=False)
        >>> print(config)
        {'epochs': 50, 'lr0': 0.01, 'momentum': 1.2, 'save': False}  # corrected 'save' key

    Notes:
        - The function modifies the input dictionary in-place.
        - None values are ignored as they may be from optional arguments.
        - Fraction keys are checked to be within the range [0.0, 1.0].
    """
    for k, v in cfg.items():
        if v is not None:
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                if hard:
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
                cfg[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    if hard:
                        raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
                    cfg[k] = v = float(v)
                if not 0.0 <= v <= 1.0:
                    raise ValueError(f"'{k}={v}' is an invalid value. Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be an int (i.e. '{k}=8')")
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be a bool (i.e. '{k}=True' or '{k}=False')")
                cfg[k] = bool(v)


MODES = {'train', 'val', 'predict', 'export', 'track', 'benchmark'}


SOLUTION_MAP = {'count': ('ObjectCounter', 'count'), 'heatmap': ('Heatmap', 'generate_heatmap'), 'queue': ('QueueManager', 'process_queue'), 'speed': ('SpeedEstimator', 'estimate_speed'), 'workout': ('AIGym', 'monitor'), 'analytics': ('Analytics', 'process_data'), 'help': None}


TASKS = {'detect', 'segment', 'classify', 'pose', 'obb'}


def deprecation_warn(arg, new_arg):
    """Issue a deprecation warning when a deprecated argument is used, suggesting an updated argument."""
    LOGGER.warning(f"WARNING ⚠️ '{arg}' is deprecated and will be removed in in the future. Use '{new_arg}' instead.")


def _handle_deprecation(custom):
    """
    Handles deprecated configuration keys by mapping them to current equivalents with deprecation warnings.

    Args:
        custom (Dict): Configuration dictionary potentially containing deprecated keys.

    Examples:
        >>> custom_config = {"boxes": True, "hide_labels": "False", "line_thickness": 2}
        >>> _handle_deprecation(custom_config)
        >>> print(custom_config)
        {'show_boxes': True, 'show_labels': True, 'line_width': 2}

    Notes:
        This function modifies the input dictionary in-place, replacing deprecated keys with their current
        equivalents. It also handles value conversions where necessary, such as inverting boolean values for
        'hide_labels' and 'hide_conf'.
    """
    for key in custom.copy().keys():
        if key == 'boxes':
            deprecation_warn(key, 'show_boxes')
            custom['show_boxes'] = custom.pop('boxes')
        if key == 'hide_labels':
            deprecation_warn(key, 'show_labels')
            custom['show_labels'] = custom.pop('hide_labels') == 'False'
        if key == 'hide_conf':
            deprecation_warn(key, 'show_conf')
            custom['show_conf'] = custom.pop('hide_conf') == 'False'
        if key == 'line_thickness':
            deprecation_warn(key, 'line_width')
            custom['line_width'] = custom.pop('line_thickness')
    return custom


def check_dict_alignment(base: 'Dict', custom: 'Dict', e=None):
    """
    Checks alignment between custom and base configuration dictionaries, handling deprecated keys and providing error
    messages for mismatched keys.

    Args:
        base (Dict): The base configuration dictionary containing valid keys.
        custom (Dict): The custom configuration dictionary to be checked for alignment.
        e (Exception | None): Optional error instance passed by the calling function.

    Raises:
        SystemExit: If mismatched keys are found between the custom and base dictionaries.

    Examples:
        >>> base_cfg = {"epochs": 50, "lr0": 0.01, "batch_size": 16}
        >>> custom_cfg = {"epoch": 100, "lr": 0.02, "batch_size": 32}
        >>> try:
        ...     check_dict_alignment(base_cfg, custom_cfg)
        ... except SystemExit:
        ...     print("Mismatched keys found")

    Notes:
        - Suggests corrections for mismatched keys based on similarity to valid keys.
        - Automatically replaces deprecated keys in the custom configuration with updated equivalents.
        - Prints detailed error messages for each mismatched key to help users correct their configurations.
    """
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))
    mismatched = [k for k in custom_keys if k not in base_keys]
    if mismatched:
        string = ''
        for x in mismatched:
            matches = get_close_matches(x, base_keys)
            matches = [(f'{k}={base[k]}' if base.get(k) is not None else k) for k in matches]
            match_str = f'Similar arguments are i.e. {matches}.' if matches else ''
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e


def get_latest_opset():
    """Return the second-most recent ONNX opset version supported by this version of PyTorch, adjusted for maturity."""
    if TORCH_1_13:
        return max(int(k[14:]) for k in vars(torch.onnx) if 'symbolic_opset' in k) - 1
    version = torch.onnx.producer_version.rsplit('.', 1)[0]
    return {'1.12': 15, '1.11': 14, '1.10': 13, '1.9': 12, '1.8': 12}.get(version, 12)


TORCH_2_0 = check_version(torch.__version__, '2.0.0')


def get_cpu_info():
    """Return a string with system CPU information, i.e. 'Apple M2'."""
    if 'cpu_info' not in PERSISTENT_CACHE:
        try:
            k = 'brand_raw', 'hardware_raw', 'arch_string_raw'
            info = cpuinfo.get_cpu_info()
            string = info.get(k[0] if k[0] in info else k[1] if k[1] in info else k[2], 'unknown')
            PERSISTENT_CACHE['cpu_info'] = string.replace('(R)', '').replace('CPU ', '').replace('@ ', '')
        except Exception:
            pass
    return PERSISTENT_CACHE.get('cpu_info', 'unknown')


def get_gpu_info(index):
    """Return a string with system GPU information, i.e. 'Tesla T4, 15102MiB'."""
    properties = torch.cuda.get_device_properties(index)
    return f'{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB'


def select_device(device='', batch=0, newline=False, verbose=True):
    """
    Selects the appropriate PyTorch device based on the provided arguments.

    The function takes a string specifying the device or a torch.device object and returns a torch.device object
    representing the selected device. The function also validates the number of available devices and raises an
    exception if the requested device(s) are not available.

    Args:
        device (str | torch.device, optional): Device string or torch.device object.
            Options are 'None', 'cpu', or 'cuda', or '0' or '0,1,2,3'. Defaults to an empty string, which auto-selects
            the first available GPU, or CPU if no GPU is available.
        batch (int, optional): Batch size being used in your model. Defaults to 0.
        newline (bool, optional): If True, adds a newline at the end of the log string. Defaults to False.
        verbose (bool, optional): If True, logs the device information. Defaults to True.

    Returns:
        (torch.device): Selected device.

    Raises:
        ValueError: If the specified device is not available or if the batch size is not a multiple of the number of
            devices when using multiple GPUs.

    Examples:
        >>> select_device("cuda:0")
        device(type='cuda', index=0)

        >>> select_device("cpu")
        device(type='cpu')

    Note:
        Sets the 'CUDA_VISIBLE_DEVICES' environment variable for specifying which GPUs to use.
    """
    if isinstance(device, torch.device) or str(device).startswith('tpu'):
        return device
    s = f'Ultralytics {__version__} 🚀 Python-{PYTHON_VERSION} torch-{torch.__version__} '
    device = str(device).lower()
    for remove in ('cuda:', 'none', '(', ')', '[', ']', "'", ' '):
        device = device.replace(remove, '')
    cpu = device == 'cpu'
    mps = device in {'mps', 'mps:0'}
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        if device == 'cuda':
            device = '0'
        if ',' in device:
            device = ','.join([x for x in device.split(',') if x])
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(','))):
            LOGGER.info(s)
            install = 'See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no CUDA devices are seen by torch.\n' if torch.cuda.device_count() == 0 else ''
            raise ValueError(f"Invalid CUDA 'device={device}' requested. Use 'device=cpu' or pass valid CUDA device(s) if available, i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n\ntorch.cuda.is_available(): {torch.cuda.is_available()}\ntorch.cuda.device_count(): {torch.cuda.device_count()}\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n{install}")
    if not cpu and not mps and torch.cuda.is_available():
        devices = device.split(',') if device else '0'
        n = len(devices)
        if n > 1:
            if batch < 1:
                raise ValueError('AutoBatch with batch<1 not supported for Multi-GPU training, please specify a valid batch size, i.e. batch=16.')
            if batch >= 0 and batch % n != 0:
                raise ValueError(f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or 'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}.")
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            s += f"{'' if i == 0 else space}CUDA:{d} ({get_gpu_info(i)})\n"
        arg = 'cuda:0'
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():
        s += f'MPS ({get_cpu_info()})\n'
        arg = 'mps'
    else:
        s += f'CPU ({get_cpu_info()})\n'
        arg = 'cpu'
    if arg in {'cpu', 'mps'}:
        torch.set_num_threads(NUM_THREADS)
    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)


def smart_inference_mode():
    """Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator."""

    def decorate(fn):
        """Applies appropriate torch decorator for inference mode based on torch version."""
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn
        else:
            return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)
    return decorate


def get_default_args(func):
    """
    Returns a dictionary of default arguments for a function.

    Args:
        func (callable): The function to inspect.

    Returns:
        (dict): A dictionary where each key is a parameter name, and each value is the default value of that parameter.
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def try_export(inner_func):
    """YOLO export decorator, i.e. @try_export."""
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """Export a model."""
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as '{f}' ({file_size(f):.1f} MB)")
            return f, model
        except Exception as e:
            LOGGER.error(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
            raise e
    return outer_func


def yaml_save(file='data.yaml', data=None, header=''):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.
        header (str, optional): YAML header to add.

    Returns:
        (None): Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)
    with open(file, 'w', errors='ignore', encoding='utf-8') as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


HELP_MSG = """
    Examples for running Ultralytics:

    1. Install the ultralytics package:

        pip install ultralytics

    2. Use the Python SDK:

        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.yaml")  # build a new model from scratch
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Use the model
        results = model.train(data="coco8.yaml", epochs=3)  # train the model
        results = model.val()  # evaluate model performance on the validation set
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        success = model.export(format="onnx")  # export the model to ONNX format

    3. Use the command line interface (CLI):

        Ultralytics 'yolo' CLI commands use the following syntax:

            yolo TASK MODE ARGS

            Where   TASK (optional) is one of [detect, segment, classify, pose, obb]
                    MODE (required) is one of [train, val, predict, export, track, benchmark]
                    ARGS (optional) are any number of custom "arg=value" pairs like "imgsz=320" that override defaults.
                        See all ARGS at https://docs.ultralytics.com/usage/cfg or with "yolo cfg"

        - Train a detection model for 10 epochs with an initial learning_rate of 0.01
            yolo detect train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01

        - Predict a YouTube video using a pretrained segmentation model at image size 320:
            yolo segment predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

        - Val a pretrained detection model at batch-size 1 and image size 640:
            yolo detect val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640

        - Export a YOLO11n classification model to ONNX format at image size 224 by 128 (no TASK required)
            yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128

        - Run special commands:
            yolo help
            yolo checks
            yolo version
            yolo settings
            yolo copy-cfg
            yolo cfg

    Docs: https://docs.ultralytics.com
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """


class HUBModelError(Exception):
    """
    Custom exception class for handling errors related to model fetching in Ultralytics YOLO.

    This exception is raised when a requested model is not found or cannot be retrieved.
    The message is also processed to include emojis for better user experience.

    Attributes:
        message (str): The error message displayed when the exception is raised.

    Note:
        The message is automatically processed through the 'emojis' function from the 'ultralytics.utils' package.
    """

    def __init__(self, message='Model not found. Please check model URL and try again.'):
        """Create an exception for when a model is not found."""
        super().__init__(emojis(message))


class HUBTrainingSession:
    """
    HUB training session for Ultralytics HUB YOLO models. Handles model initialization, heartbeats, and checkpointing.

    Attributes:
        model_id (str): Identifier for the YOLO model being trained.
        model_url (str): URL for the model in Ultralytics HUB.
        rate_limits (dict): Rate limits for different API calls (in seconds).
        timers (dict): Timers for rate limiting.
        metrics_queue (dict): Queue for the model's metrics.
        model (dict): Model data fetched from Ultralytics HUB.
    """

    def __init__(self, identifier):
        """
        Initialize the HUBTrainingSession with the provided model identifier.

        Args:
            identifier (str): Model identifier used to initialize the HUB training session.
                It can be a URL string or a model key with specific format.

        Raises:
            ValueError: If the provided model identifier is invalid.
            ConnectionError: If connecting with global API key is not supported.
            ModuleNotFoundError: If hub-sdk package is not installed.
        """
        self.rate_limits = {'metrics': 3, 'ckpt': 900, 'heartbeat': 300}
        self.metrics_queue = {}
        self.metrics_upload_failed_queue = {}
        self.timers = {}
        self.model = None
        self.model_url = None
        self.model_file = None
        self.train_args = None
        api_key, model_id, self.filename = self._parse_identifier(identifier)
        active_key = api_key or SETTINGS.get('api_key')
        credentials = {'api_key': active_key} if active_key else None
        self.client = HUBClient(credentials)
        try:
            if model_id:
                self.load_model(model_id)
            else:
                self.model = self.client.model()
        except Exception:
            if identifier.startswith(f'{HUB_WEB_ROOT}/models/') and not self.client.authenticated:
                LOGGER.warning(f"{PREFIX}WARNING ⚠️ Please log in using 'yolo login API_KEY'. You can find your API Key at: https://hub.ultralytics.com/settings?tab=api+keys.")

    @classmethod
    def create_session(cls, identifier, args=None):
        """Class method to create an authenticated HUBTrainingSession or return None."""
        try:
            session = cls(identifier)
            if args and not identifier.startswith(f'{HUB_WEB_ROOT}/models/'):
                session.create_model(args)
                assert session.model.id, 'HUB model not loaded correctly'
            return session
        except (PermissionError, ModuleNotFoundError, AssertionError):
            return None

    def load_model(self, model_id):
        """Loads an existing model from Ultralytics HUB using the provided model identifier."""
        self.model = self.client.model(model_id)
        if not self.model.data:
            raise ValueError(emojis('❌ The specified HUB model does not exist'))
        self.model_url = f'{HUB_WEB_ROOT}/models/{self.model.id}'
        if self.model.is_trained():
            None
            url = self.model.get_weights_url('best')
            self.model_file = checks.check_file(url, download_dir=Path(SETTINGS['weights_dir']) / 'hub' / self.model.id)
            return
        self._set_train_args()
        self.model.start_heartbeat(self.rate_limits['heartbeat'])
        LOGGER.info(f'{PREFIX}View model at {self.model_url} 🚀')

    def create_model(self, model_args):
        """Initializes a HUB training session with the specified model identifier."""
        payload = {'config': {'batchSize': model_args.get('batch', -1), 'epochs': model_args.get('epochs', 300), 'imageSize': model_args.get('imgsz', 640), 'patience': model_args.get('patience', 100), 'device': str(model_args.get('device', '')), 'cache': str(model_args.get('cache', 'ram'))}, 'dataset': {'name': model_args.get('data')}, 'lineage': {'architecture': {'name': self.filename.replace('.pt', '').replace('.yaml', '')}, 'parent': {}}, 'meta': {'name': self.filename}}
        if self.filename.endswith('.pt'):
            payload['lineage']['parent']['name'] = self.filename
        self.model.create_model(payload)
        if not self.model.id:
            return None
        self.model_url = f'{HUB_WEB_ROOT}/models/{self.model.id}'
        self.model.start_heartbeat(self.rate_limits['heartbeat'])
        LOGGER.info(f'{PREFIX}View model at {self.model_url} 🚀')

    @staticmethod
    def _parse_identifier(identifier):
        """
        Parses the given identifier to determine the type of identifier and extract relevant components.

        The method supports different identifier formats:
            - A HUB model URL https://hub.ultralytics.com/models/MODEL
            - A HUB model URL with API Key https://hub.ultralytics.com/models/MODEL?api_key=APIKEY
            - A local filename that ends with '.pt' or '.yaml'

        Args:
            identifier (str): The identifier string to be parsed.

        Returns:
            (tuple): A tuple containing the API key, model ID, and filename as applicable.

        Raises:
            HUBModelError: If the identifier format is not recognized.
        """
        api_key, model_id, filename = None, None, None
        if Path(identifier).suffix in {'.pt', '.yaml'}:
            filename = identifier
        elif identifier.startswith(f'{HUB_WEB_ROOT}/models/'):
            parsed_url = urlparse(identifier)
            model_id = Path(parsed_url.path).stem
            query_params = parse_qs(parsed_url.query)
            api_key = query_params.get('api_key', [None])[0]
        else:
            raise HUBModelError(f"model='{identifier} invalid, correct format is {HUB_WEB_ROOT}/models/MODEL_ID")
        return api_key, model_id, filename

    def _set_train_args(self):
        """
        Initializes training arguments and creates a model entry on the Ultralytics HUB.

        This method sets up training arguments based on the model's state and updates them with any additional
        arguments provided. It handles different states of the model, such as whether it's resumable, pretrained,
        or requires specific file setup.

        Raises:
            ValueError: If the model is already trained, if required dataset information is missing, or if there are
                issues with the provided training arguments.
        """
        if self.model.is_resumable():
            self.train_args = {'data': self.model.get_dataset_url(), 'resume': True}
            self.model_file = self.model.get_weights_url('last')
        else:
            self.train_args = self.model.data.get('train_args')
            self.model_file = self.model.get_weights_url('parent') if self.model.is_pretrained() else self.model.get_architecture()
        if 'data' not in self.train_args:
            raise ValueError('Dataset may still be processing. Please wait a minute and try again.')
        self.model_file = checks.check_yolov5u_filename(self.model_file, verbose=False)
        self.model_id = self.model.id

    def request_queue(self, request_func, retry=3, timeout=30, thread=True, verbose=True, progress_total=None, stream_response=None, *args, **kwargs):
        """Attempts to execute `request_func` with retries, timeout handling, optional threading, and progress."""

        def retry_request():
            """Attempts to call `request_func` with retries, timeout, and optional threading."""
            t0 = time.time()
            response = None
            for i in range(retry + 1):
                if time.time() - t0 > timeout:
                    LOGGER.warning(f'{PREFIX}Timeout for request reached. {HELP_MSG}')
                    break
                response = request_func(*args, **kwargs)
                if response is None:
                    LOGGER.warning(f'{PREFIX}Received no response from the request. {HELP_MSG}')
                    time.sleep(2 ** i)
                    continue
                if progress_total:
                    self._show_upload_progress(progress_total, response)
                elif stream_response:
                    self._iterate_content(response)
                if HTTPStatus.OK <= response.status_code < HTTPStatus.MULTIPLE_CHOICES:
                    if kwargs.get('metrics'):
                        self.metrics_upload_failed_queue = {}
                    return response
                if i == 0:
                    message = self._get_failure_message(response, retry, timeout)
                    if verbose:
                        LOGGER.warning(f'{PREFIX}{message} {HELP_MSG} ({response.status_code})')
                if not self._should_retry(response.status_code):
                    LOGGER.warning(f'{PREFIX}Request failed. {HELP_MSG} ({response.status_code}')
                    break
                time.sleep(2 ** i)
            if response is None and kwargs.get('metrics'):
                self.metrics_upload_failed_queue.update(kwargs.get('metrics'))
            return response
        if thread:
            threading.Thread(target=retry_request, daemon=True).start()
        else:
            return retry_request()

    @staticmethod
    def _should_retry(status_code):
        """Determines if a request should be retried based on the HTTP status code."""
        retry_codes = {HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.BAD_GATEWAY, HTTPStatus.GATEWAY_TIMEOUT}
        return status_code in retry_codes

    def _get_failure_message(self, response: 'requests.Response', retry: 'int', timeout: 'int'):
        """
        Generate a retry message based on the response status code.

        Args:
            response: The HTTP response object.
            retry: The number of retry attempts allowed.
            timeout: The maximum timeout duration.

        Returns:
            (str): The retry message.
        """
        if self._should_retry(response.status_code):
            return f'Retrying {retry}x for {timeout}s.' if retry else ''
        elif response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            headers = response.headers
            return f"Rate limit reached ({headers['X-RateLimit-Remaining']}/{headers['X-RateLimit-Limit']}). Please retry after {headers['Retry-After']}s."
        else:
            try:
                return response.json().get('message', 'No JSON message.')
            except AttributeError:
                return 'Unable to read JSON.'

    def upload_metrics(self):
        """Upload model metrics to Ultralytics HUB."""
        return self.request_queue(self.model.upload_metrics, metrics=self.metrics_queue.copy(), thread=True)

    def upload_model(self, epoch: 'int', weights: 'str', is_best: 'bool'=False, map: 'float'=0.0, final: 'bool'=False) ->None:
        """
        Upload a model checkpoint to Ultralytics HUB.

        Args:
            epoch (int): The current training epoch.
            weights (str): Path to the model weights file.
            is_best (bool): Indicates if the current model is the best one so far.
            map (float): Mean average precision of the model.
            final (bool): Indicates if the model is the final model after training.
        """
        weights = Path(weights)
        if not weights.is_file():
            last = weights.with_name(f'last{weights.suffix}')
            if final and last.is_file():
                LOGGER.warning(f"{PREFIX} WARNING ⚠️ Model 'best.pt' not found, copying 'last.pt' to 'best.pt' and uploading. This often happens when resuming training in transient environments like Google Colab. For more reliable training, consider using Ultralytics HUB Cloud. Learn more at https://docs.ultralytics.com/hub/cloud-training.")
                shutil.copy(last, weights)
            else:
                LOGGER.warning(f'{PREFIX} WARNING ⚠️ Model upload issue. Missing model {weights}.')
                return
        self.request_queue(self.model.upload_model, epoch=epoch, weights=str(weights), is_best=is_best, map=map, final=final, retry=10, timeout=3600, thread=not final, progress_total=weights.stat().st_size if final else None, stream_response=True)

    @staticmethod
    def _show_upload_progress(content_length: 'int', response: 'requests.Response') ->None:
        """
        Display a progress bar to track the upload progress of a file download.

        Args:
            content_length (int): The total size of the content to be downloaded in bytes.
            response (requests.Response): The response object from the file download request.

        Returns:
            None
        """
        with TQDM(total=content_length, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for data in response.iter_content(chunk_size=1024):
                pbar.update(len(data))

    @staticmethod
    def _iterate_content(response: 'requests.Response') ->None:
        """
        Process the streamed HTTP response data.

        Args:
            response (requests.Response): The response object from the file download request.

        Returns:
            None
        """
        for _ in response.iter_content(chunk_size=1024):
            pass


def is_colab():
    """
    Check if the current script is running inside a Google Colab notebook.

    Returns:
        (bool): True if running inside a Colab notebook, False otherwise.
    """
    return 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_BACKEND_VERSION' in os.environ


def is_kaggle():
    """
    Check if the current script is running inside a Kaggle kernel.

    Returns:
        (bool): True if running inside a Kaggle kernel, False otherwise.
    """
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'


class Colors:
    """
    Ultralytics color palette https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Colors.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.

    ## Ultralytics Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #042aff;"></i> | `#042aff` | (4, 42, 255)      |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #0bdbeb;"></i> | `#0bdbeb` | (11, 219, 235)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #f3f3f3;"></i> | `#f3f3f3` | (243, 243, 243)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #00dfb7;"></i> | `#00dfb7` | (0, 223, 183)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #111f68;"></i> | `#111f68` | (17, 31, 104)     |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #ff6fdd;"></i> | `#ff6fdd` | (255, 111, 221)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff444f;"></i> | `#ff444f` | (255, 68, 79)     |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #cced00;"></i> | `#cced00` | (204, 237, 0)     |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #00f344;"></i> | `#00f344` | (0, 243, 68)      |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #bd00ff;"></i> | `#bd00ff` | (189, 0, 255)     |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #00b4ff;"></i> | `#00b4ff` | (0, 180, 255)     |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #dd00ba;"></i> | `#dd00ba` | (221, 0, 186)     |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #00ffff;"></i> | `#00ffff` | (0, 255, 255)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #26c000;"></i> | `#26c000` | (38, 192, 0)      |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #01ffb3;"></i> | `#01ffb3` | (1, 255, 179)     |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #7d24ff;"></i> | `#7d24ff` | (125, 36, 255)    |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #7b0068;"></i> | `#7b0068` | (123, 0, 104)     |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #ff1b6c;"></i> | `#ff1b6c` | (255, 27, 108)    |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #fc6d2f;"></i> | `#fc6d2f` | (252, 109, 47)    |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #a2ff0b;"></i> | `#a2ff0b` | (162, 255, 11)    |

    ## Pose Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #ff8000;"></i> | `#ff8000` | (255, 128, 0)     |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #ff9933;"></i> | `#ff9933` | (255, 153, 51)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #ffb266;"></i> | `#ffb266` | (255, 178, 102)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #e6e600;"></i> | `#e6e600` | (230, 230, 0)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #ff99ff;"></i> | `#ff99ff` | (255, 153, 255)   |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #99ccff;"></i> | `#99ccff` | (153, 204, 255)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff66ff;"></i> | `#ff66ff` | (255, 102, 255)   |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #ff33ff;"></i> | `#ff33ff` | (255, 51, 255)    |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #66b2ff;"></i> | `#66b2ff` | (102, 178, 255)   |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #3399ff;"></i> | `#3399ff` | (51, 153, 255)    |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #ff9999;"></i> | `#ff9999` | (255, 153, 153)   |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #ff6666;"></i> | `#ff6666` | (255, 102, 102)   |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #ff3333;"></i> | `#ff3333` | (255, 51, 51)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #99ff99;"></i> | `#99ff99` | (153, 255, 153)   |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #66ff66;"></i> | `#66ff66` | (102, 255, 102)   |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #33ff33;"></i> | `#33ff33` | (51, 255, 51)     |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #00ff00;"></i> | `#00ff00` | (0, 255, 0)       |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #0000ff;"></i> | `#0000ff` | (0, 0, 255)       |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #ff0000;"></i> | `#ff0000` | (255, 0, 0)       |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #ffffff;"></i> | `#ffffff` | (255, 255, 255)   |

    !!! note "Ultralytics Brand Colors"

        For Ultralytics brand colors see [https://www.ultralytics.com/brand](https://www.ultralytics.com/brand). Please use the official Ultralytics colors for all marketing materials.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = '042AFF', '0BDBEB', 'F3F3F3', '00DFB7', '111F68', 'FF6FDD', 'FF444F', 'CCED00', '00F344', 'BD00FF', '00B4FF', 'DD00BA', '00FFFF', '26C000', '01FFB3', '7D24FF', '7B0068', 'FF1B6C', 'FC6D2F', 'A2FF0B'
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


class Annotator:
    """
    Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        im (Image.Image or numpy array): The image to annotate.
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype or ImageFont.load_default): Font used for text annotations.
        lw (float): Line width for drawing.
        skeleton (List[List[int]]): Skeleton structure for keypoints.
        limb_color (List[int]): Color palette for limbs.
        kpt_color (List[int]): Color palette for keypoints.
    """

    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        non_ascii = not is_ascii(example)
        input_is_pil = isinstance(im, Image.Image)
        self.pil = pil or non_ascii or input_is_pil
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        if self.pil:
            self.im = im if input_is_pil else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            try:
                font = check_font('Arial.Unicode.ttf' if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            if check_version(pil_version, '9.2.0'):
                self.font.getsize = lambda x: self.font.getbbox(x)[2:4]
        else:
            assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images.'
            self.im = im if im.flags.writeable else im.copy()
            self.tf = max(self.lw - 1, 1)
            self.sf = self.lw / 3
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.dark_colors = {(235, 219, 11), (243, 243, 243), (183, 223, 0), (221, 111, 255), (0, 237, 204), (68, 243, 0), (255, 255, 0), (179, 255, 1), (11, 255, 162)}
        self.light_colors = {(255, 42, 4), (79, 68, 255), (255, 0, 189), (255, 180, 0), (186, 0, 221), (0, 192, 38), (255, 36, 125), (104, 0, 123), (108, 27, 255), (47, 109, 252), (104, 31, 17)}

    def get_txt_color(self, color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Assign text color based on background color."""
        if color in self.dark_colors:
            return 104, 31, 17
        elif color in self.light_colors:
            return 255, 255, 255
        else:
            return txt_color

    def circle_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), margin=2):
        """
        Draws a label with a background circle centered within a given bounding box.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            margin (int, optional): The margin between the text and the rectangle border.
        """
        if len(label) > 3:
            None
            label = label[:3]
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        text_size = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.tf)[0]
        required_radius = int((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5 / 2) + margin
        cv2.circle(self.im, (x_center, y_center), required_radius, color, -1)
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        cv2.putText(self.im, str(label), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.get_txt_color(color, txt_color), self.tf, lineType=cv2.LINE_AA)

    def text_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), margin=5):
        """
        Draws a label with a background rectangle centered within a given bounding box.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            margin (int, optional): The margin between the text and the rectangle border.
        """
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.1, self.tf)[0]
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        cv2.rectangle(self.im, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
        cv2.putText(self.im, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.1, self.get_txt_color(color, txt_color), self.tf, lineType=cv2.LINE_AA)

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False):
        """
        Draws a bounding box to image with label.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            rotated (bool, optional): Variable used to check if task is OBB
        """
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        if self.pil or not is_ascii(label):
            if rotated:
                p1 = box[0]
                self.draw.polygon([tuple(b) for b in box], width=self.lw, outline=color)
            else:
                p1 = box[0], box[1]
                self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)
                outside = p1[1] >= h
                if p1[0] > self.im.size[0] - w:
                    p1 = self.im.size[0] - w, p1[1]
                self.draw.rectangle((p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1), fill=color)
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else:
            if rotated:
                p1 = [int(b) for b in box[0]]
                cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)
            else:
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
                h += 3
                outside = p1[1] >= h
                if p1[0] > self.im.shape[1] - w:
                    p1 = self.im.shape[1] - w, p1[1]
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h - 1), 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def masks(self, masks, colors, im_gpu, alpha=0.5, retina_masks=False):
        """
        Plot masks on image.

        Args:
            masks (tensor): Predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): Colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): Image is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): Mask transparency: 0.0 fully transparent, 1.0 opaque
            retina_masks (bool): Whether to use high resolution masks or not. Defaults to False.
        """
        if self.pil:
            self.im = np.asarray(self.im).copy()
        if len(masks) == 0:
            self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
        if im_gpu.device != masks.device:
            im_gpu = im_gpu
        colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0
        colors = colors[:, None, None]
        masks = masks.unsqueeze(3)
        masks_color = masks * (colors * alpha)
        inv_alpha_masks = (1 - masks * alpha).cumprod(0)
        mcs = masks_color.max(dim=0).values
        im_gpu = im_gpu.flip(dims=[0])
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()
        im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
        im_mask = im_gpu * 255
        im_mask_np = im_mask.byte().cpu().numpy()
        self.im[:] = im_mask_np if retina_masks else ops.scale_image(im_mask_np, self.im.shape)
        if self.pil:
            self.fromarray(self.im)

    def kpts(self, kpts, shape=(640, 640), radius=None, kpt_line=True, conf_thres=0.25, kpt_color=None):
        """
        Plot keypoints on the image.

        Args:
            kpts (torch.Tensor): Keypoints, shape [17, 3] (x, y, confidence).
            shape (tuple, optional): Image shape (h, w). Defaults to (640, 640).
            radius (int, optional): Keypoint radius. Defaults to 5.
            kpt_line (bool, optional): Draw lines between keypoints. Defaults to True.
            conf_thres (float, optional): Confidence threshold. Defaults to 0.25.
            kpt_color (tuple, optional): Keypoint color (B, G, R). Defaults to None.

        Note:
            - `kpt_line=True` currently only supports human pose plotting.
            - Modifies self.im in-place.
            - If self.pil is True, converts image to numpy array and back to PIL.
        """
        radius = radius if radius is not None else self.lw
        if self.pil:
            self.im = np.asarray(self.im).copy()
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose
        for i, k in enumerate(kpts):
            color_k = kpt_color or (self.kpt_color[i].tolist() if is_pose else colors(i))
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < conf_thres:
                        continue
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1])
                pos2 = int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1])
                if ndim == 3:
                    conf1 = kpts[sk[0] - 1, 2]
                    conf2 = kpts[sk[1] - 1, 2]
                    if conf1 < conf_thres or conf2 < conf_thres:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(self.im, pos1, pos2, kpt_color or self.limb_color[i].tolist(), thickness=int(np.ceil(self.lw / 2)), lineType=cv2.LINE_AA)
        if self.pil:
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        """Add rectangle to image (PIL-only)."""
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top', box_style=False):
        """Adds text to an image using PIL or cv2."""
        if anchor == 'bottom':
            w, h = self.font.getsize(text)
            xy[1] += 1 - h
        if self.pil:
            if box_style:
                w, h = self.font.getsize(text)
                self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=txt_color)
                txt_color = 255, 255, 255
            if '\n' in text:
                lines = text.split('\n')
                _, h = self.font.getsize(text)
                for line in lines:
                    self.draw.text(xy, line, fill=txt_color, font=self.font)
                    xy[1] += h
            else:
                self.draw.text(xy, text, fill=txt_color, font=self.font)
        else:
            if box_style:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
                h += 3
                outside = xy[1] >= h
                p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
                cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)
                txt_color = 255, 255, 255
            cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def fromarray(self, im):
        """Update self.im from a numpy array."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        """Return annotated image as array."""
        return np.asarray(self.im)

    def show(self, title=None):
        """Show the annotated image."""
        im = Image.fromarray(np.asarray(self.im)[..., ::-1])
        if IS_COLAB or IS_KAGGLE:
            try:
                display(im)
            except ImportError as e:
                LOGGER.warning(f'Unable to display image in Jupyter notebooks: {e}')
        else:
            im.show(title=title)

    def save(self, filename='image.jpg'):
        """Save the annotated image to 'filename'."""
        cv2.imwrite(filename, np.asarray(self.im))

    def get_bbox_dimension(self, bbox=None):
        """
        Calculate the area of a bounding box.

        Args:
            bbox (tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).

        Returns:
            angle (degree): Degree value of angle between three points
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height, width * height

    def draw_region(self, reg_pts=None, color=(0, 255, 0), thickness=5):
        """
        Draw region line.

        Args:
            reg_pts (list): Region Points (for line 2 points, for region 4 points)
            color (tuple): Region Color value
            thickness (int): Region area thickness value
        """
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)
        for point in reg_pts:
            cv2.circle(self.im, (point[0], point[1]), thickness * 2, color, -1)

    def draw_centroid_and_tracks(self, track, color=(255, 0, 255), track_thickness=2):
        """
        Draw centroid point and track trails.

        Args:
            track (list): object tracking points for trails display
            color (tuple): tracks line color
            track_thickness (int): track line thickness value
        """
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.im, [points], isClosed=False, color=color, thickness=track_thickness)
        cv2.circle(self.im, (int(track[-1][0]), int(track[-1][1])), track_thickness * 2, color, -1)

    def queue_counts_display(self, label, points=None, region_color=(255, 255, 255), txt_color=(0, 0, 0)):
        """
        Displays queue counts on an image centered at the points with customizable font size and colors.

        Args:
            label (str): queue counts label
            points (tuple): region points for center point calculation to display text
            region_color (tuple): RGB queue region color.
            txt_color (tuple): RGB text display color.
        """
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        center_x = sum(x_values) // len(points)
        center_y = sum(y_values) // len(points)
        text_size = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_width = text_size[0]
        text_height = text_size[1]
        rect_width = text_width + 20
        rect_height = text_height + 20
        rect_top_left = center_x - rect_width // 2, center_y - rect_height // 2
        rect_bottom_right = center_x + rect_width // 2, center_y + rect_height // 2
        cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)
        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2
        cv2.putText(self.im, label, (text_x, text_y), 0, fontScale=self.sf, color=txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def display_objects_labels(self, im0, text, txt_color, bg_color, x_center, y_center, margin):
        """
        Display the bounding boxes labels in parking management app.

        Args:
            im0 (ndarray): inference image
            text (str): object/class name
            txt_color (tuple): display color for text foreground
            bg_color (tuple): display color for text background
            x_center (float): x position center point for bounding box
            y_center (float): y position center point for bounding box
            margin (int): gap between text and rectangle for better display
        """
        text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
        cv2.putText(im0, text, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)

    def display_analytics(self, im0, text, txt_color, bg_color, margin):
        """
        Display the overall statistics for parking lots.

        Args:
            im0 (ndarray): inference image
            text (dict): labels dictionary
            txt_color (tuple): display color for text foreground
            bg_color (tuple): display color for text background
            margin (int): gap between text and rectangle for better display
        """
        horizontal_gap = int(im0.shape[1] * 0.02)
        vertical_gap = int(im0.shape[0] * 0.01)
        text_y_offset = 0
        for label, value in text.items():
            txt = f'{label}: {value}'
            text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
            if text_size[0] < 5 or text_size[1] < 5:
                text_size = 5, 5
            text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
            text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
            rect_x1 = text_x - margin * 2
            rect_y1 = text_y - text_size[1] - margin * 2
            rect_x2 = text_x + text_size[0] + margin * 2
            rect_y2 = text_y + margin * 2
            cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
            cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)
            text_y_offset = rect_y2

    @staticmethod
    def estimate_pose_angle(a, b, c):
        """
        Calculate the pose angle for object.

        Args:
            a (float) : The value of pose point a
            b (float): The value of pose point b
            c (float): The value o pose point c

        Returns:
            angle (degree): Degree value of angle between three points
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def draw_specific_points(self, keypoints, indices=None, radius=2, conf_thres=0.25):
        """
        Draw specific keypoints for gym steps counting.

        Args:
            keypoints (list): Keypoints data to be plotted.
            indices (list, optional): Keypoint indices to be plotted. Defaults to [2, 5, 7].
            radius (int, optional): Keypoint radius. Defaults to 2.
            conf_thres (float, optional): Confidence threshold for keypoints. Defaults to 0.25.

        Returns:
            (numpy.ndarray): Image with drawn keypoints.

        Note:
            Keypoint format: [x, y] or [x, y, confidence].
            Modifies self.im in-place.
        """
        indices = indices or [2, 5, 7]
        points = [(int(k[0]), int(k[1])) for i, k in enumerate(keypoints) if i in indices and k[2] >= conf_thres]
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(self.im, start, end, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        for pt in points:
            cv2.circle(self.im, pt, radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        return self.im

    def plot_workout_information(self, display_text, position, color=(104, 31, 17), txt_color=(255, 255, 255)):
        """
        Draw text with a background on the image.

        Args:
            display_text (str): The text to be displayed.
            position (tuple): Coordinates (x, y) on the image where the text will be placed.
            color (tuple, optional): Text background color
            txt_color (tuple, optional): Text foreground color
        """
        (text_width, text_height), _ = cv2.getTextSize(display_text, 0, self.sf, self.tf)
        cv2.rectangle(self.im, (position[0], position[1] - text_height - 5), (position[0] + text_width + 10, position[1] - text_height - 5 + text_height + 10 + self.tf), color, -1)
        cv2.putText(self.im, display_text, position, 0, self.sf, txt_color, self.tf)
        return text_height

    def plot_angle_and_count_and_stage(self, angle_text, count_text, stage_text, center_kpt, color=(104, 31, 17), txt_color=(255, 255, 255)):
        """
        Plot the pose angle, count value, and step stage.

        Args:
            angle_text (str): Angle value for workout monitoring
            count_text (str): Counts value for workout monitoring
            stage_text (str): Stage decision for workout monitoring
            center_kpt (list): Centroid pose index for workout monitoring
            color (tuple, optional): Text background color
            txt_color (tuple, optional): Text foreground color
        """
        angle_text, count_text, stage_text = f' {angle_text:.2f}', f'Steps : {count_text}', f' {stage_text}'
        angle_height = self.plot_workout_information(angle_text, (int(center_kpt[0]), int(center_kpt[1])), color, txt_color)
        count_height = self.plot_workout_information(count_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + 20), color, txt_color)
        self.plot_workout_information(stage_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + count_height + 40), color, txt_color)

    def seg_bbox(self, mask, mask_color=(255, 0, 255), label=None, txt_color=(255, 255, 255)):
        """
        Function for drawing segmented object in bounding box shape.

        Args:
            mask (np.ndarray): A 2D array of shape (N, 2) containing the contour points of the segmented object.
            mask_color (tuple): RGB color for the contour and label background.
            label (str, optional): Text label for the object. If None, no label is drawn.
            txt_color (tuple): RGB color for the label text.
        """
        if mask.size == 0:
            return
        cv2.polylines(self.im, [np.int32([mask])], isClosed=True, color=mask_color, thickness=2)
        text_size, _ = cv2.getTextSize(label, 0, self.sf, self.tf)
        cv2.rectangle(self.im, (int(mask[0][0]) - text_size[0] // 2 - 10, int(mask[0][1]) - text_size[1] - 10), (int(mask[0][0]) + text_size[0] // 2 + 10, int(mask[0][1] + 10)), mask_color, -1)
        if label:
            cv2.putText(self.im, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1])), 0, self.sf, txt_color, self.tf)

    def plot_distance_and_line(self, pixels_distance, centroids, line_color=(104, 31, 17), centroid_color=(255, 0, 255)):
        """
        Plot the distance and line on frame.

        Args:
            pixels_distance (float): Pixels distance between two bbox centroids.
            centroids (list): Bounding box centroids data.
            line_color (tuple, optional): Distance line color.
            centroid_color (tuple, optional): Bounding box centroid color.
        """
        text = f'Pixels Distance: {pixels_distance:.2f}'
        (text_width_m, text_height_m), _ = cv2.getTextSize(text, 0, self.sf, self.tf)
        cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 20, 25 + text_height_m + 20), line_color, -1)
        text_position = 25, 25 + text_height_m + 10
        cv2.putText(self.im, text, text_position, 0, self.sf, (255, 255, 255), self.tf, cv2.LINE_AA)
        cv2.line(self.im, centroids[0], centroids[1], line_color, 3)
        cv2.circle(self.im, centroids[0], 6, centroid_color, -1)
        cv2.circle(self.im, centroids[1], 6, centroid_color, -1)

    def visioneye(self, box, center_point, color=(235, 219, 11), pin_color=(255, 0, 255)):
        """
        Function for pinpoint human-vision eye mapping and plotting.

        Args:
            box (list): Bounding box coordinates
            center_point (tuple): center point for vision eye view
            color (tuple): object centroid and line color value
            pin_color (tuple): visioneye point color value
        """
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)
        cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)
        cv2.line(self.im, center_point, center_bbox, color, self.tf)


class SimpleClass:
    """
    A simple base class for creating objects with string representations of their attributes.

    This class provides a foundation for creating objects that can be easily printed or represented as strings,
    showing all their non-callable attributes. It's useful for debugging and introspection of object states.

    Methods:
        __str__: Returns a human-readable string representation of the object.
        __repr__: Returns a machine-readable string representation of the object.
        __getattr__: Provides a custom attribute access error message with helpful information.

    Examples:
        >>> class MyClass(SimpleClass):
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = "hello"
        >>> obj = MyClass()
        >>> print(obj)
        __main__.MyClass object with attributes:

        x: 10
        y: 'hello'

    Notes:
        - This class is designed to be subclassed. It provides a convenient way to inspect object attributes.
        - The string representation includes the module and class name of the object.
        - Callable attributes and attributes starting with an underscore are excluded from the string representation.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith('_'):
                if isinstance(v, SimpleClass):
                    s = f'{a}: {v.__module__}.{v.__class__.__name__} object'
                else:
                    s = f'{a}: {repr(v)}'
                attr.append(s)
        return f'{self.__module__}.{self.__class__.__name__} object with attributes:\n\n' + '\n'.join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class BaseTensor(SimpleClass):
    """
    Base tensor class with additional methods for easy manipulation and device handling.

    Attributes:
        data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
        orig_shape (Tuple[int, int]): Original shape of the image, typically in the format (height, width).

    Methods:
        cpu: Return a copy of the tensor stored in CPU memory.
        numpy: Returns a copy of the tensor as a numpy array.
        cuda: Moves the tensor to GPU memory, returning a new instance if necessary.
        to: Return a copy of the tensor with the specified device and dtype.

    Examples:
        >>> import torch
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> orig_shape = (720, 1280)
        >>> base_tensor = BaseTensor(data, orig_shape)
        >>> cpu_tensor = base_tensor.cpu()
        >>> numpy_array = base_tensor.numpy()
        >>> gpu_tensor = base_tensor.cuda()
    """

    def __init__(self, data, orig_shape) ->None:
        """
        Initialize BaseTensor with prediction data and the original shape of the image.

        Args:
            data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
            orig_shape (Tuple[int, int]): Original shape of the image in (height, width) format.

        Examples:
            >>> import torch
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
        """
        assert isinstance(data, (torch.Tensor, np.ndarray)), 'data must be torch.Tensor or np.ndarray'
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        """
        Returns the shape of the underlying data tensor.

        Returns:
            (Tuple[int, ...]): The shape of the data tensor.

        Examples:
            >>> data = torch.rand(100, 4)
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> print(base_tensor.shape)
            (100, 4)
        """
        return self.data.shape

    def cpu(self):
        """
        Returns a copy of the tensor stored in CPU memory.

        Returns:
            (BaseTensor): A new BaseTensor object with the data tensor moved to CPU memory.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> cpu_tensor = base_tensor.cpu()
            >>> isinstance(cpu_tensor, BaseTensor)
            True
            >>> cpu_tensor.data.device
            device(type='cpu')
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """
        Returns a copy of the tensor as a numpy array.

        Returns:
            (np.ndarray): A numpy array containing the same data as the original tensor.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> numpy_array = base_tensor.numpy()
            >>> print(type(numpy_array))
            <class 'numpy.ndarray'>
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """
        Moves the tensor to GPU memory.

        Returns:
            (BaseTensor): A new BaseTensor instance with the data moved to GPU memory if it's not already a
                numpy array, otherwise returns self.

        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> gpu_tensor = base_tensor.cuda()
            >>> print(gpu_tensor.data.device)
            cuda:0
        """
        return self.__class__(torch.as_tensor(self.data), self.orig_shape)

    def to(self, *args, **kwargs):
        """
        Return a copy of the tensor with the specified device and dtype.

        Args:
            *args (Any): Variable length argument list to be passed to torch.Tensor.to().
            **kwargs (Any): Arbitrary keyword arguments to be passed to torch.Tensor.to().

        Returns:
            (BaseTensor): A new BaseTensor instance with the data moved to the specified device and/or dtype.

        Examples:
            >>> base_tensor = BaseTensor(torch.randn(3, 4), orig_shape=(480, 640))
            >>> cuda_tensor = base_tensor.to("cuda")
            >>> float16_tensor = base_tensor.to(dtype=torch.float16)
        """
        return self.__class__(torch.as_tensor(self.data), self.orig_shape)

    def __len__(self):
        """
        Returns the length of the underlying data tensor.

        Returns:
            (int): The number of elements in the first dimension of the data tensor.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> len(base_tensor)
            2
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a new BaseTensor instance containing the specified indexed elements of the data tensor.

        Args:
            idx (int | List[int] | torch.Tensor): Index or indices to select from the data tensor.

        Returns:
            (BaseTensor): A new BaseTensor instance containing the indexed data.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> result = base_tensor[0]  # Select the first row
            >>> print(result.data)
            tensor([1, 2, 3])
        """
        return self.__class__(self.data[idx], self.orig_shape)


class Boxes(BaseTensor):
    """
    A class for managing and manipulating detection boxes.

    This class provides functionality for handling detection boxes, including their coordinates, confidence scores,
    class labels, and optional tracking IDs. It supports various box formats and offers methods for easy manipulation
    and conversion between different coordinate systems.

    Attributes:
        data (torch.Tensor | numpy.ndarray): The raw tensor containing detection boxes and associated data.
        orig_shape (Tuple[int, int]): The original image dimensions (height, width).
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xyxy (torch.Tensor | numpy.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | numpy.ndarray): Confidence scores for each box.
        cls (torch.Tensor | numpy.ndarray): Class labels for each box.
        id (torch.Tensor | numpy.ndarray): Tracking IDs for each box (if available).
        xywh (torch.Tensor | numpy.ndarray): Boxes in [x, y, width, height] format.
        xyxyn (torch.Tensor | numpy.ndarray): Normalized [x1, y1, x2, y2] boxes relative to orig_shape.
        xywhn (torch.Tensor | numpy.ndarray): Normalized [x, y, width, height] boxes relative to orig_shape.

    Methods:
        cpu(): Returns a copy of the object with all tensors on CPU memory.
        numpy(): Returns a copy of the object with all tensors as numpy arrays.
        cuda(): Returns a copy of the object with all tensors on GPU memory.
        to(*args, **kwargs): Returns a copy of the object with tensors on specified device and dtype.

    Examples:
        >>> import torch
        >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape = (480, 640)  # height, width
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf)
        >>> print(boxes.cls)
        >>> print(boxes.xywhn)
    """

    def __init__(self, boxes, orig_shape) ->None:
        """
        Initialize the Boxes class with detection box data and the original image shape.

        This class manages detection boxes, providing easy access and manipulation of box coordinates,
        confidence scores, class identifiers, and optional tracking IDs. It supports multiple formats
        for box coordinates, including both absolute and normalized forms.

        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array with detection boxes of shape
                (num_boxes, 6) or (num_boxes, 7). Columns should contain
                [x1, y1, x2, y2, confidence, class, (optional) track_id].
            orig_shape (Tuple[int, int]): The original image shape as (height, width). Used for normalization.

        Attributes:
            data (torch.Tensor): The raw tensor containing detection boxes and their associated data.
            orig_shape (Tuple[int, int]): The original image size, used for normalization.
            is_track (bool): Indicates whether tracking IDs are included in the box data.

        Examples:
            >>> import torch
            >>> boxes = torch.tensor([[100, 50, 150, 100, 0.9, 0]])
            >>> orig_shape = (480, 640)
            >>> detection_boxes = Boxes(boxes, orig_shape)
            >>> print(detection_boxes.xyxy)
            tensor([[100.,  50., 150., 100.]])
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f'expected 6 or 7 values but got {n}'
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        """
        Returns bounding boxes in [x1, y1, x2, y2] format.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array of shape (n, 4) containing bounding box
                coordinates in [x1, y1, x2, y2] format, where n is the number of boxes.

        Examples:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> xyxy = boxes.xyxy
            >>> print(xyxy)
        """
        return self.data[:, :4]

    @property
    def conf(self):
        """
        Returns the confidence scores for each detection box.

        Returns:
            (torch.Tensor | numpy.ndarray): A 1D tensor or array containing confidence scores for each detection,
                with shape (N,) where N is the number of detections.

        Examples:
            >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 0]]), orig_shape=(100, 100))
            >>> conf_scores = boxes.conf
            >>> print(conf_scores)
            tensor([0.9000])
        """
        return self.data[:, -2]

    @property
    def cls(self):
        """
        Returns the class ID tensor representing category predictions for each bounding box.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the class IDs for each detection box.
                The shape is (N,), where N is the number of boxes.

        Examples:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> class_ids = boxes.cls
            >>> print(class_ids)  # tensor([0., 2., 1.])
        """
        return self.data[:, -1]

    @property
    def id(self):
        """
        Returns the tracking IDs for each detection box if available.

        Returns:
            (torch.Tensor | None): A tensor containing tracking IDs for each box if tracking is enabled,
                otherwise None. Shape is (N,) where N is the number of boxes.

        Examples:
            >>> results = model.track("path/to/video.mp4")
            >>> for result in results:
            ...     boxes = result.boxes
            ...     if boxes.is_track:
            ...         track_ids = boxes.id
            ...         print(f"Tracking IDs: {track_ids}")
            ...     else:
            ...         print("Tracking is not enabled for these boxes.")

        Notes:
            - This property is only available when tracking is enabled (i.e., when `is_track` is True).
            - The tracking IDs are typically used to associate detections across multiple frames in video analysis.
        """
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xywh(self):
        """
        Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format.

        Returns:
            (torch.Tensor | numpy.ndarray): Boxes in [x_center, y_center, width, height] format, where x_center, y_center are the coordinates of
                the center point of the bounding box, width, height are the dimensions of the bounding box and the
                shape of the returned tensor is (N, 4), where N is the number of boxes.

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100], [200, 150, 300, 250]]), orig_shape=(480, 640))
            >>> xywh = boxes.xywh
            >>> print(xywh)
            tensor([[100.0000,  50.0000,  50.0000,  50.0000],
                    [200.0000, 150.0000, 100.0000, 100.0000]])
        """
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        """
        Returns normalized bounding box coordinates relative to the original image size.

        This property calculates and returns the bounding box coordinates in [x1, y1, x2, y2] format,
        normalized to the range [0, 1] based on the original image dimensions.

        Returns:
            (torch.Tensor | numpy.ndarray): Normalized bounding box coordinates with shape (N, 4), where N is
                the number of boxes. Each row contains [x1, y1, x2, y2] values normalized to [0, 1].

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 300, 400, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xyxyn
            >>> print(normalized)
            tensor([[0.1562, 0.1042, 0.4688, 0.8333]])
        """
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        """
        Returns normalized bounding boxes in [x, y, width, height] format.

        This property calculates and returns the normalized bounding box coordinates in the format
        [x_center, y_center, width, height], where all values are relative to the original image dimensions.

        Returns:
            (torch.Tensor | numpy.ndarray): Normalized bounding boxes with shape (N, 4), where N is the
                number of boxes. Each row contains [x_center, y_center, width, height] values normalized
                to [0, 1] based on the original image dimensions.

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xywhn
            >>> print(normalized)
            tensor([[0.1953, 0.1562, 0.0781, 0.1042]])
        """
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Keypoints(BaseTensor):
    """
    A class for storing and manipulating detection keypoints.

    This class encapsulates functionality for handling keypoint data, including coordinate manipulation,
    normalization, and confidence values.

    Attributes:
        data (torch.Tensor): The raw tensor containing keypoint data.
        orig_shape (Tuple[int, int]): The original image dimensions (height, width).
        has_visible (bool): Indicates whether visibility information is available for keypoints.
        xy (torch.Tensor): Keypoint coordinates in [x, y] format.
        xyn (torch.Tensor): Normalized keypoint coordinates in [x, y] format, relative to orig_shape.
        conf (torch.Tensor): Confidence values for each keypoint, if available.

    Methods:
        cpu(): Returns a copy of the keypoints tensor on CPU memory.
        numpy(): Returns a copy of the keypoints tensor as a numpy array.
        cuda(): Returns a copy of the keypoints tensor on GPU memory.
        to(*args, **kwargs): Returns a copy of the keypoints tensor with specified device and dtype.

    Examples:
        >>> import torch
        >>> from ultralytics.engine.results import Keypoints
        >>> keypoints_data = torch.rand(1, 17, 3)  # 1 detection, 17 keypoints, (x, y, conf)
        >>> orig_shape = (480, 640)  # Original image shape (height, width)
        >>> keypoints = Keypoints(keypoints_data, orig_shape)
        >>> print(keypoints.xy.shape)  # Access xy coordinates
        >>> print(keypoints.conf)  # Access confidence values
        >>> keypoints_cpu = keypoints.cpu()  # Move keypoints to CPU
    """

    @smart_inference_mode()
    def __init__(self, keypoints, orig_shape) ->None:
        """
        Initializes the Keypoints object with detection keypoints and original image dimensions.

        This method processes the input keypoints tensor, handling both 2D and 3D formats. For 3D tensors
        (x, y, confidence), it masks out low-confidence keypoints by setting their coordinates to zero.

        Args:
            keypoints (torch.Tensor): A tensor containing keypoint data. Shape can be either:
                - (num_objects, num_keypoints, 2) for x, y coordinates only
                - (num_objects, num_keypoints, 3) for x, y coordinates and confidence scores
            orig_shape (Tuple[int, int]): The original image dimensions (height, width).

        Examples:
            >>> kpts = torch.rand(1, 17, 3)  # 1 object, 17 keypoints (COCO format), x,y,conf
            >>> orig_shape = (720, 1280)  # Original image height, width
            >>> keypoints = Keypoints(kpts, orig_shape)
        """
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        if keypoints.shape[2] == 3:
            mask = keypoints[..., 2] < 0.5
            keypoints[..., :2][mask] = 0
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """
        Returns x, y coordinates of keypoints.

        Returns:
            (torch.Tensor): A tensor containing the x, y coordinates of keypoints with shape (N, K, 2), where N is
                the number of detections and K is the number of keypoints per detection.

        Examples:
            >>> results = model("image.jpg")
            >>> keypoints = results[0].keypoints
            >>> xy = keypoints.xy
            >>> print(xy.shape)  # (N, K, 2)
            >>> print(xy[0])  # x, y coordinates of keypoints for first detection

        Notes:
            - The returned coordinates are in pixel units relative to the original image dimensions.
            - If keypoints were initialized with confidence values, only keypoints with confidence >= 0.5 are returned.
            - This property uses LRU caching to improve performance on repeated access.
        """
        return self.data[..., :2]

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """
        Returns normalized coordinates (x, y) of keypoints relative to the original image size.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or array of shape (N, K, 2) containing normalized keypoint
                coordinates, where N is the number of instances, K is the number of keypoints, and the last
                dimension contains [x, y] values in the range [0, 1].

        Examples:
            >>> keypoints = Keypoints(torch.rand(1, 17, 2), orig_shape=(480, 640))
            >>> normalized_kpts = keypoints.xyn
            >>> print(normalized_kpts.shape)
            torch.Size([1, 17, 2])
        """
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]
        xy[..., 1] /= self.orig_shape[0]
        return xy

    @property
    @lru_cache(maxsize=1)
    def conf(self):
        """
        Returns confidence values for each keypoint.

        Returns:
            (torch.Tensor | None): A tensor containing confidence scores for each keypoint if available,
                otherwise None. Shape is (num_detections, num_keypoints) for batched data or (num_keypoints,)
                for single detection.

        Examples:
            >>> keypoints = Keypoints(torch.rand(1, 17, 3), orig_shape=(640, 640))  # 1 detection, 17 keypoints
            >>> conf = keypoints.conf
            >>> print(conf.shape)  # torch.Size([1, 17])
        """
        return self.data[..., 2] if self.has_visible else None


class Masks(BaseTensor):
    """
    A class for storing and manipulating detection masks.

    This class extends BaseTensor and provides functionality for handling segmentation masks,
    including methods for converting between pixel and normalized coordinates.

    Attributes:
        data (torch.Tensor | numpy.ndarray): The raw tensor or array containing mask data.
        orig_shape (tuple): Original image shape in (height, width) format.
        xy (List[numpy.ndarray]): A list of segments in pixel coordinates.
        xyn (List[numpy.ndarray]): A list of normalized segments.

    Methods:
        cpu(): Returns a copy of the Masks object with the mask tensor on CPU memory.
        numpy(): Returns a copy of the Masks object with the mask tensor as a numpy array.
        cuda(): Returns a copy of the Masks object with the mask tensor on GPU memory.
        to(*args, **kwargs): Returns a copy of the Masks object with the mask tensor on specified device and dtype.

    Examples:
        >>> masks_data = torch.rand(1, 160, 160)
        >>> orig_shape = (720, 1280)
        >>> masks = Masks(masks_data, orig_shape)
        >>> pixel_coords = masks.xy
        >>> normalized_coords = masks.xyn
    """

    def __init__(self, masks, orig_shape) ->None:
        """
        Initialize the Masks class with detection mask data and the original image shape.

        Args:
            masks (torch.Tensor | np.ndarray): Detection masks with shape (num_masks, height, width).
            orig_shape (tuple): The original image shape as (height, width). Used for normalization.

        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import Masks
            >>> masks = torch.rand(10, 160, 160)  # 10 masks of 160x160 resolution
            >>> orig_shape = (720, 1280)  # Original image shape
            >>> mask_obj = Masks(masks, orig_shape)
        """
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """
        Returns normalized xy-coordinates of the segmentation masks.

        This property calculates and caches the normalized xy-coordinates of the segmentation masks. The coordinates
        are normalized relative to the original image shape.

        Returns:
            (List[numpy.ndarray]): A list of numpy arrays, where each array contains the normalized xy-coordinates
                of a single segmentation mask. Each array has shape (N, 2), where N is the number of points in the
                mask contour.

        Examples:
            >>> results = model("image.jpg")
            >>> masks = results[0].masks
            >>> normalized_coords = masks.xyn
            >>> print(normalized_coords[0])  # Normalized coordinates of the first mask
        """
        return [ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True) for x in ops.masks2segments(self.data)]

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """
        Returns the [x, y] pixel coordinates for each segment in the mask tensor.

        This property calculates and returns a list of pixel coordinates for each segmentation mask in the
        Masks object. The coordinates are scaled to match the original image dimensions.

        Returns:
            (List[numpy.ndarray]): A list of numpy arrays, where each array contains the [x, y] pixel
                coordinates for a single segmentation mask. Each array has shape (N, 2), where N is the
                number of points in the segment.

        Examples:
            >>> results = model("image.jpg")
            >>> masks = results[0].masks
            >>> xy_coords = masks.xy
            >>> print(len(xy_coords))  # Number of masks
            >>> print(xy_coords[0].shape)  # Shape of first mask's coordinates
        """
        return [ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False) for x in ops.masks2segments(self.data)]


class Probs(BaseTensor):
    """
    A class for storing and manipulating classification probabilities.

    This class extends BaseTensor and provides methods for accessing and manipulating
    classification probabilities, including top-1 and top-5 predictions.

    Attributes:
        data (torch.Tensor | numpy.ndarray): The raw tensor or array containing classification probabilities.
        orig_shape (tuple | None): The original image shape as (height, width). Not used in this class.
        top1 (int): Index of the class with the highest probability.
        top5 (List[int]): Indices of the top 5 classes by probability.
        top1conf (torch.Tensor | numpy.ndarray): Confidence score of the top 1 class.
        top5conf (torch.Tensor | numpy.ndarray): Confidence scores of the top 5 classes.

    Methods:
        cpu(): Returns a copy of the probabilities tensor on CPU memory.
        numpy(): Returns a copy of the probabilities tensor as a numpy array.
        cuda(): Returns a copy of the probabilities tensor on GPU memory.
        to(*args, **kwargs): Returns a copy of the probabilities tensor with specified device and dtype.

    Examples:
        >>> probs = torch.tensor([0.1, 0.3, 0.6])
        >>> p = Probs(probs)
        >>> print(p.top1)
        2
        >>> print(p.top5)
        [2, 1, 0]
        >>> print(p.top1conf)
        tensor(0.6000)
        >>> print(p.top5conf)
        tensor([0.6000, 0.3000, 0.1000])
    """

    def __init__(self, probs, orig_shape=None) ->None:
        """
        Initialize the Probs class with classification probabilities.

        This class stores and manages classification probabilities, providing easy access to top predictions and their
        confidences.

        Args:
            probs (torch.Tensor | np.ndarray): A 1D tensor or array of classification probabilities.
            orig_shape (tuple | None): The original image shape as (height, width). Not used in this class but kept for
                consistency with other result classes.

        Attributes:
            data (torch.Tensor | np.ndarray): The raw tensor or array containing classification probabilities.
            top1 (int): Index of the top 1 class.
            top5 (List[int]): Indices of the top 5 classes.
            top1conf (torch.Tensor | np.ndarray): Confidence of the top 1 class.
            top5conf (torch.Tensor | np.ndarray): Confidences of the top 5 classes.

        Examples:
            >>> import torch
            >>> probs = torch.tensor([0.1, 0.3, 0.2, 0.4])
            >>> p = Probs(probs)
            >>> print(p.top1)
            3
            >>> print(p.top1conf)
            tensor(0.4000)
            >>> print(p.top5)
            [3, 1, 2, 0]
        """
        super().__init__(probs, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def top1(self):
        """
        Returns the index of the class with the highest probability.

        Returns:
            (int): Index of the class with the highest probability.

        Examples:
            >>> probs = Probs(torch.tensor([0.1, 0.3, 0.6]))
            >>> probs.top1
            2
        """
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5(self):
        """
        Returns the indices of the top 5 class probabilities.

        Returns:
            (List[int]): A list containing the indices of the top 5 class probabilities, sorted in descending order.

        Examples:
            >>> probs = Probs(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
            >>> print(probs.top5)
            [4, 3, 2, 1, 0]
        """
        return (-self.data).argsort(0)[:5].tolist()

    @property
    @lru_cache(maxsize=1)
    def top1conf(self):
        """
        Returns the confidence score of the highest probability class.

        This property retrieves the confidence score (probability) of the class with the highest predicted probability
        from the classification results.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor containing the confidence score of the top 1 class.

        Examples:
            >>> results = model("image.jpg")  # classify an image
            >>> probs = results[0].probs  # get classification probabilities
            >>> top1_confidence = probs.top1conf  # get confidence of top 1 class
            >>> print(f"Top 1 class confidence: {top1_confidence.item():.4f}")
        """
        return self.data[self.top1]

    @property
    @lru_cache(maxsize=1)
    def top5conf(self):
        """
        Returns confidence scores for the top 5 classification predictions.

        This property retrieves the confidence scores corresponding to the top 5 class probabilities
        predicted by the model. It provides a quick way to access the most likely class predictions
        along with their associated confidence levels.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or array containing the confidence scores for the
                top 5 predicted classes, sorted in descending order of probability.

        Examples:
            >>> results = model("image.jpg")
            >>> probs = results[0].probs
            >>> top5_conf = probs.top5conf
            >>> print(top5_conf)  # Prints confidence scores for top 5 classes
        """
        return self.data[self.top5]


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    Increments a file or directory path, i.e., runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and `exist_ok` is not True, the path will be incremented by appending a number and `sep` to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If `mkdir` is set to True, the path will be created as a
    directory if it does not already exist.

    Args:
        path (str | pathlib.Path): Path to increment.
        exist_ok (bool): If True, the path will not be incremented and returned as-is.
        sep (str): Separator to use between the path and the incrementation number.
        mkdir (bool): Create a directory if it does not exist.

    Returns:
        (pathlib.Path): Incremented path.

    Examples:
        Increment a directory path:
        >>> from pathlib import Path
        >>> path = Path("runs/exp")
        >>> new_path = increment_path(path)
        >>> print(new_path)
        runs/exp2

        Increment a file path:
        >>> path = Path("runs/exp/results.txt")
        >>> new_path = increment_path(path)
        >>> print(new_path)
        runs/exp/results2.txt
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def is_github_action_running() ->bool:
    """
    Determine if the current environment is a GitHub Actions runner.

    Returns:
        (bool): True if the current environment is a GitHub Actions runner, False otherwise.
    """
    return 'GITHUB_ACTIONS' in os.environ and 'GITHUB_WORKFLOW' in os.environ and 'RUNNER_OS' in os.environ


def is_pytest_running():
    """
    Determines whether pytest is currently running or not.

    Returns:
        (bool): True if pytest is running, False otherwise.
    """
    return 'PYTEST_CURRENT_TEST' in os.environ or 'pytest' in sys.modules or 'pytest' in Path(ARGV[0]).stem


def get_save_dir(args, name=None):
    """
    Returns the directory path for saving outputs, derived from arguments or default settings.

    Args:
        args (SimpleNamespace): Namespace object containing configurations such as 'project', 'name', 'task',
            'mode', and 'save_dir'.
        name (str | None): Optional name for the output directory. If not provided, it defaults to 'args.name'
            or the 'args.mode'.

    Returns:
        (Path): Directory path where outputs should be saved.

    Examples:
        >>> from types import SimpleNamespace
        >>> args = SimpleNamespace(project="my_project", task="detect", mode="train", exist_ok=True)
        >>> save_dir = get_save_dir(args)
        >>> print(save_dir)
        my_project/detect/train
    """
    if getattr(args, 'save_dir', None):
        save_dir = args.save_dir
    else:
        project = args.project or (ROOT.parent / 'tests/tmp/runs' if TESTS_RUNNING else RUNS_DIR) / args.task
        name = name or args.name or f'{args.mode}'
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)
    return Path(save_dir)


def plt_color_scatter(v, f, bins=20, cmap='viridis', alpha=0.8, edgecolors='none'):
    """
    Plots a scatter plot with points colored based on a 2D histogram.

    Args:
        v (array-like): Values for the x-axis.
        f (array-like): Values for the y-axis.
        bins (int, optional): Number of bins for the histogram. Defaults to 20.
        cmap (str, optional): Colormap for the scatter plot. Defaults to 'viridis'.
        alpha (float, optional): Alpha for the scatter plot. Defaults to 0.8.
        edgecolors (str, optional): Edge colors for the scatter plot. Defaults to 'none'.

    Examples:
        >>> v = np.random.rand(100)
        >>> f = np.random.rand(100)
        >>> plt_color_scatter(v, f)
    """
    hist, xedges, yedges = np.histogram2d(v, f, bins=bins)
    colors = [hist[min(np.digitize(v[i], xedges, right=True) - 1, hist.shape[0] - 1), min(np.digitize(f[i], yedges, right=True) - 1, hist.shape[1] - 1)] for i in range(len(v))]
    plt.scatter(v, f, c=colors, cmap=cmap, alpha=alpha, edgecolors=edgecolors)


def plot_tune_results(csv_file='tune_results.csv'):
    """
    Plot the evolution results stored in an 'tune_results.csv' file. The function generates a scatter plot for each key
    in the CSV, color-coded based on fitness scores. The best-performing configurations are highlighted on the plots.

    Args:
        csv_file (str, optional): Path to the CSV file containing the tuning results. Defaults to 'tune_results.csv'.

    Examples:
        >>> plot_tune_results("path/to/tune_results.csv")
    """
    import pandas as pd
    from scipy.ndimage import gaussian_filter1d

    def _save_one_file(file):
        """Save one matplotlib plot to 'file'."""
        plt.savefig(file, dpi=200)
        plt.close()
        LOGGER.info(f'Saved {file}')
    csv_file = Path(csv_file)
    data = pd.read_csv(csv_file)
    num_metrics_columns = 1
    keys = [x.strip() for x in data.columns][num_metrics_columns:]
    x = data.values
    fitness = x[:, 0]
    j = np.argmax(fitness)
    n = math.ceil(len(keys) ** 0.5)
    plt.figure(figsize=(10, 10), tight_layout=True)
    for i, k in enumerate(keys):
        v = x[:, i + num_metrics_columns]
        mu = v[j]
        plt.subplot(n, n, i + 1)
        plt_color_scatter(v, fitness, cmap='viridis', alpha=0.8, edgecolors='none')
        plt.plot(mu, fitness.max(), 'k+', markersize=15)
        plt.title(f'{k} = {mu:.3g}', fontdict={'size': 9})
        plt.tick_params(axis='both', labelsize=8)
        if i % n != 0:
            plt.yticks([])
    _save_one_file(csv_file.with_name('tune_scatter_plots.png'))
    x = range(1, len(fitness) + 1)
    plt.figure(figsize=(10, 6), tight_layout=True)
    plt.plot(x, fitness, marker='o', linestyle='none', label='fitness')
    plt.plot(x, gaussian_filter1d(fitness, sigma=3), ':', label='smoothed', linewidth=2)
    plt.title('Fitness vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    _save_one_file(csv_file.with_name('tune_fitness.png'))


def remove_colorstr(input_string):
    """
    Removes ANSI escape codes from a string, effectively un-coloring it.

    Args:
        input_string (str): The string to remove color and style from.

    Returns:
        (str): A new string with all ANSI escape codes removed.

    Examples:
        >>> remove_colorstr(colorstr("blue", "bold", "hello world"))
        >>> "hello world"
    """
    ansi_escape = re.compile('\\x1B\\[[0-9;]*[A-Za-z]')
    return ansi_escape.sub('', input_string)


def yaml_print(yaml_file: 'Union[str, Path, dict]') ->None:
    """
    Pretty prints a YAML file or a YAML-formatted dictionary.

    Args:
        yaml_file: The file path of the YAML file or a YAML-formatted dictionary.

    Returns:
        (None)
    """
    yaml_dict = yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    dump = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True, width=float('inf'))
    LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg['head'][-1][-2].lower()
        if m in {'classify', 'classifier', 'cls', 'fc'}:
            return 'classify'
        if 'detect' in m:
            return 'detect'
        if m == 'segment':
            return 'segment'
        if m == 'pose':
            return 'pose'
        if m == 'obb':
            return 'obb'
    if isinstance(model, dict):
        try:
            return cfg2task(model)
        except Exception:
            pass
    if isinstance(model, nn.Module):
        for x in ('model.args', 'model.model.args', 'model.model.model.args'):
            try:
                return eval(x)['task']
            except Exception:
                pass
        for x in ('model.yaml', 'model.model.yaml', 'model.model.model.yaml'):
            try:
                return cfg2task(eval(x))
            except Exception:
                pass
        for m in model.modules():
            if isinstance(m, Segment):
                return 'segment'
            elif isinstance(m, Classify):
                return 'classify'
            elif isinstance(m, Pose):
                return 'pose'
            elif isinstance(m, OBB):
                return 'obb'
            elif isinstance(m, (Detect, WorldDetect, v10Detect)):
                return 'detect'
    if isinstance(model, (str, Path)):
        model = Path(model)
        if '-seg' in model.stem or 'segment' in model.parts:
            return 'segment'
        elif '-cls' in model.stem or 'classify' in model.parts:
            return 'classify'
        elif '-pose' in model.stem or 'pose' in model.parts:
            return 'pose'
        elif '-obb' in model.stem or 'obb' in model.parts:
            return 'obb'
        elif 'detect' in model.parts:
            return 'detect'
    LOGGER.warning("WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'.")
    return 'detect'


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


def torch_safe_load(weight, safe_only=False):
    """
    Attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Example:
    ```python
    from ultralytics.nn.tasks import torch_safe_load

    ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    ```

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename
    """
    check_suffix(file=weight, suffix='.pt')
    file = attempt_download_asset(weight)
    try:
        with temporary_modules(modules={'ultralytics.yolo.utils': 'ultralytics.utils', 'ultralytics.yolo.v8': 'ultralytics.models.yolo', 'ultralytics.yolo.data': 'ultralytics.data'}, attributes={'ultralytics.nn.modules.block.Silence': 'torch.nn.Identity', 'ultralytics.nn.tasks.YOLOv10DetectionModel': 'ultralytics.nn.tasks.DetectionModel', 'ultralytics.utils.loss.v10DetectLoss': 'ultralytics.utils.loss.E2EDetectLoss'}):
            if safe_only:
                safe_pickle = types.ModuleType('safe_pickle')
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, 'rb') as f:
                    ckpt = torch.load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch.load(file, map_location='cpu')
    except ModuleNotFoundError as e:
        if e.name == 'models':
            raise TypeError(emojis(f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with YOLOv8 at https://github.com/ultralytics/ultralytics.\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'")) from e
        LOGGER.warning(f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in Ultralytics requirements.\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future.\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'")
        check_requirements(e.name)
        ckpt = torch.load(file, map_location='cpu')
    if not isinstance(ckpt, dict):
        LOGGER.warning(f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. For optimal results, use model.save('filename.pt') to correctly save YOLO models.")
        ckpt = {'model': ckpt.model}
    return ckpt, file


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)
    args = {**DEFAULT_CFG_DICT, **ckpt.get('train_args', {})}
    model = (ckpt.get('ema') or ckpt['model']).float()
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}
    model.pt_path = weight
    model.task = guess_model_task(model)
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.0])
    model = model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval()
    for m in model.modules():
        if hasattr(m, 'inplace'):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None
    return model, ckpt


def on_predict_postprocess_end(predictor: 'object', persist: 'bool'=False) ->None:
    """
    Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Postprocess predictions and update with tracking
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor, persist=True)
    """
    path, im0s = predictor.batch[:2]
    is_obb = predictor.args.task == 'obb'
    is_stream = predictor.dataset.mode == 'stream'
    for i in range(len(im0s)):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(path[i]).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path
        det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes).cpu().numpy()
        if len(det) == 0:
            continue
        tracks = tracker.update(det, im0s[i])
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]
        update_args = {('obb' if is_obb else 'boxes'): torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)


class KalmanFilterXYAH:
    """
    A KalmanFilterXYAH class for tracking bounding boxes in image space using a Kalman filter.

    Implements a simple Kalman filter for tracking bounding boxes in image space. The 8-dimensional state space
    (x, y, a, h, vx, vy, va, vh) contains the bounding box center position (x, y), aspect ratio a, height h, and their
    respective velocities. Object motion follows a constant velocity model, and bounding box location (x, y, a, h) is
    taken as a direct observation of the state space (linear observation model).

    Attributes:
        _motion_mat (np.ndarray): The motion matrix for the Kalman filter.
        _update_mat (np.ndarray): The update matrix for the Kalman filter.
        _std_weight_position (float): Standard deviation weight for position.
        _std_weight_velocity (float): Standard deviation weight for velocity.

    Methods:
        initiate: Creates a track from an unassociated measurement.
        predict: Runs the Kalman filter prediction step.
        project: Projects the state distribution to measurement space.
        multi_predict: Runs the Kalman filter prediction step (vectorized version).
        update: Runs the Kalman filter correction step.
        gating_distance: Computes the gating distance between state distribution and measurements.

    Examples:
        Initialize the Kalman filter and create a track from a measurement
        >>> kf = KalmanFilterXYAH()
        >>> measurement = np.array([100, 200, 1.5, 50])
        >>> mean, covariance = kf.initiate(measurement)
        >>> print(mean)
        >>> print(covariance)
    """

    def __init__(self):
        """
        Initialize Kalman filter model matrices with motion and observation uncertainty weights.

        The Kalman filter is initialized with an 8-dimensional state space (x, y, a, h, vx, vy, va, vh), where (x, y)
        represents the bounding box center position, 'a' is the aspect ratio, 'h' is the height, and their respective
        velocities are (vx, vy, va, vh). The filter uses a constant velocity model for object motion and a linear
        observation model for bounding box location.

        Examples:
            Initialize a Kalman filter for tracking:
            >>> kf = KalmanFilterXYAH()
        """
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: 'np.ndarray') ->tuple:
        """
        Create a track from an unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, a, h) with center position (x, y), aspect ratio a,
                and height h.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector (8-dimensional) and covariance matrix (8x8 dimensional)
                of the new track. Unobserved velocities are initialized to 0 mean.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> measurement = np.array([100, 50, 1.5, 200])
            >>> mean, covariance = kf.initiate(measurement)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [2 * self._std_weight_position * measurement[3], 2 * self._std_weight_position * measurement[3], 0.01, 2 * self._std_weight_position * measurement[3], 10 * self._std_weight_velocity * measurement[3], 10 * self._std_weight_velocity * measurement[3], 1e-05, 10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: 'np.ndarray', covariance: 'np.ndarray') ->tuple:
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8-dimensional mean vector of the object state at the previous time step.
            covariance (ndarray): The 8x8-dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        """
        std_pos = [self._std_weight_position * mean[3], self._std_weight_position * mean[3], 0.01, self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3], self._std_weight_velocity * mean[3], 1e-05, self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: 'np.ndarray', covariance: 'np.ndarray') ->tuple:
        """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (tuple[ndarray, ndarray]): Returns the projected mean and covariance matrix of the given state estimate.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> projected_mean, projected_covariance = kf.project(mean, covariance)
        """
        std = [self._std_weight_position * mean[3], self._std_weight_position * mean[3], 0.1, self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: 'np.ndarray', covariance: 'np.ndarray') ->tuple:
        """
        Run Kalman filter prediction step for multiple object states (Vectorized version).

        Args:
            mean (ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean matrix and covariance matrix of the predicted states.
                The mean matrix has shape (N, 8) and the covariance matrix has shape (N, 8, 8). Unobserved velocities
                are initialized to 0 mean.

        Examples:
            >>> mean = np.random.rand(10, 8)  # 10 object states
            >>> covariance = np.random.rand(10, 8, 8)  # Covariance matrices for 10 object states
            >>> predicted_mean, predicted_covariance = kalman_filter.multi_predict(mean, covariance)
        """
        std_pos = [self._std_weight_position * mean[:, 3], self._std_weight_position * mean[:, 3], 0.01 * np.ones_like(mean[:, 3]), self._std_weight_position * mean[:, 3]]
        std_vel = [self._std_weight_velocity * mean[:, 3], self._std_weight_velocity * mean[:, 3], 1e-05 * np.ones_like(mean[:, 3]), self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(self, mean: 'np.ndarray', covariance: 'np.ndarray', measurement: 'np.ndarray') ->tuple:
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector (x, y, a, h), where (x, y) is the center
                position, a the aspect ratio, and h the height of the bounding box.

        Returns:
            (tuple[ndarray, ndarray]): Returns the measurement-corrected state distribution.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurement = np.array([1, 1, 1, 1])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean: 'np.ndarray', covariance: 'np.ndarray', measurements: 'np.ndarray', only_position: 'bool'=False, metric: 'str'='maha') ->np.ndarray:
        """
        Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If `only_position` is False, the chi-square
        distribution has 4 degrees of freedom, otherwise 2.

        Args:
            mean (ndarray): Mean vector over the state distribution (8 dimensional).
            covariance (ndarray): Covariance of the state distribution (8x8 dimensional).
            measurements (ndarray): An (N, 4) matrix of N measurements, each in format (x, y, a, h) where (x, y) is the
                bounding box center position, a the aspect ratio, and h the height.
            only_position (bool): If True, distance computation is done with respect to box center position only.
            metric (str): The metric to use for calculating the distance. Options are 'gaussian' for the squared
                Euclidean distance and 'maha' for the squared Mahalanobis distance.

        Returns:
            (np.ndarray): Returns an array of length N, where the i-th element contains the squared distance between
                (mean, covariance) and `measurements[i]`.

        Examples:
            Compute gating distance using Mahalanobis metric:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurements = np.array([[1, 1, 1, 1], [2, 2, 1, 1]])
            >>> distances = kf.gating_distance(mean, covariance, measurements, only_position=False, metric="maha")
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)
        else:
            raise ValueError('Invalid distance metric')


class KalmanFilterXYWH(KalmanFilterXYAH):
    """
    A KalmanFilterXYWH class for tracking bounding boxes in image space using a Kalman filter.

    Implements a Kalman filter for tracking bounding boxes with state space (x, y, w, h, vx, vy, vw, vh), where
    (x, y) is the center position, w is the width, h is the height, and vx, vy, vw, vh are their respective velocities.
    The object motion follows a constant velocity model, and the bounding box location (x, y, w, h) is taken as a direct
    observation of the state space (linear observation model).

    Attributes:
        _motion_mat (np.ndarray): The motion matrix for the Kalman filter.
        _update_mat (np.ndarray): The update matrix for the Kalman filter.
        _std_weight_position (float): Standard deviation weight for position.
        _std_weight_velocity (float): Standard deviation weight for velocity.

    Methods:
        initiate: Creates a track from an unassociated measurement.
        predict: Runs the Kalman filter prediction step.
        project: Projects the state distribution to measurement space.
        multi_predict: Runs the Kalman filter prediction step in a vectorized manner.
        update: Runs the Kalman filter correction step.

    Examples:
        Create a Kalman filter and initialize a track
        >>> kf = KalmanFilterXYWH()
        >>> measurement = np.array([100, 50, 20, 40])
        >>> mean, covariance = kf.initiate(measurement)
        >>> print(mean)
        >>> print(covariance)
    """

    def initiate(self, measurement: 'np.ndarray') ->tuple:
        """
        Create track from unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, w, h) with center position (x, y), width, and height.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector (8 dimensional) and covariance matrix (8x8 dimensional)
                of the new track. Unobserved velocities are initialized to 0 mean.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> measurement = np.array([100, 50, 20, 40])
            >>> mean, covariance = kf.initiate(measurement)
            >>> print(mean)
            [100.  50.  20.  40.   0.   0.   0.   0.]
            >>> print(covariance)
            [[ 4.  0.  0.  0.  0.  0.  0.  0.]
             [ 0.  4.  0.  0.  0.  0.  0.  0.]
             [ 0.  0.  4.  0.  0.  0.  0.  0.]
             [ 0.  0.  0.  4.  0.  0.  0.  0.]
             [ 0.  0.  0.  0.  0.25  0.  0.  0.]
             [ 0.  0.  0.  0.  0.  0.25  0.  0.]
             [ 0.  0.  0.  0.  0.  0.  0.25  0.]
             [ 0.  0.  0.  0.  0.  0.  0.  0.25]]
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [2 * self._std_weight_position * measurement[2], 2 * self._std_weight_position * measurement[3], 2 * self._std_weight_position * measurement[2], 2 * self._std_weight_position * measurement[3], 10 * self._std_weight_velocity * measurement[2], 10 * self._std_weight_velocity * measurement[3], 10 * self._std_weight_velocity * measurement[2], 10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance) ->tuple:
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8-dimensional mean vector of the object state at the previous time step.
            covariance (ndarray): The 8x8-dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        """
        std_pos = [self._std_weight_position * mean[2], self._std_weight_position * mean[3], self._std_weight_position * mean[2], self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[2], self._std_weight_velocity * mean[3], self._std_weight_velocity * mean[2], self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance) ->tuple:
        """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (tuple[ndarray, ndarray]): Returns the projected mean and covariance matrix of the given state estimate.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> projected_mean, projected_cov = kf.project(mean, covariance)
        """
        std = [self._std_weight_position * mean[2], self._std_weight_position * mean[3], self._std_weight_position * mean[2], self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance) ->tuple:
        """
        Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.

        Examples:
            >>> mean = np.random.rand(5, 8)  # 5 objects with 8-dimensional state vectors
            >>> covariance = np.random.rand(5, 8, 8)  # 5 objects with 8x8 covariance matrices
            >>> kf = KalmanFilterXYWH()
            >>> predicted_mean, predicted_covariance = kf.multi_predict(mean, covariance)
        """
        std_pos = [self._std_weight_position * mean[:, 2], self._std_weight_position * mean[:, 3], self._std_weight_position * mean[:, 2], self._std_weight_position * mean[:, 3]]
        std_vel = [self._std_weight_velocity * mean[:, 2], self._std_weight_velocity * mean[:, 3], self._std_weight_velocity * mean[:, 2], self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement) ->tuple:
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector (x, y, w, h), where (x, y) is the center
                position, w the width, and h the height of the bounding box.

        Returns:
            (tuple[ndarray, ndarray]): Returns the measurement-corrected state distribution.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurement = np.array([0.5, 0.5, 1.2, 1.2])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)
        """
        return super().update(mean, covariance, measurement)


class TrackState:
    """
    Enumeration class representing the possible states of an object being tracked.

    Attributes:
        New (int): State when the object is newly detected.
        Tracked (int): State when the object is successfully tracked in subsequent frames.
        Lost (int): State when the object is no longer tracked.
        Removed (int): State when the object is removed from tracking.

    Examples:
        >>> state = TrackState.New
        >>> if state == TrackState.New:
        >>>     print("Object is newly detected.")
    """
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """
    Base class for object tracking, providing foundational attributes and methods.

    Attributes:
        _count (int): Class-level counter for unique track IDs.
        track_id (int): Unique identifier for the track.
        is_activated (bool): Flag indicating whether the track is currently active.
        state (TrackState): Current state of the track.
        history (OrderedDict): Ordered history of the track's states.
        features (List): List of features extracted from the object for tracking.
        curr_feature (Any): The current feature of the object being tracked.
        score (float): The confidence score of the tracking.
        start_frame (int): The frame number where tracking started.
        frame_id (int): The most recent frame ID processed by the track.
        time_since_update (int): Frames passed since the last update.
        location (Tuple): The location of the object in the context of multi-camera tracking.

    Methods:
        end_frame: Returns the ID of the last frame where the object was tracked.
        next_id: Increments and returns the next global track ID.
        activate: Abstract method to activate the track.
        predict: Abstract method to predict the next state of the track.
        update: Abstract method to update the track with new data.
        mark_lost: Marks the track as lost.
        mark_removed: Marks the track as removed.
        reset_id: Resets the global track ID counter.

    Examples:
        Initialize a new track and mark it as lost:
        >>> track = BaseTrack()
        >>> track.mark_lost()
        >>> print(track.state)  # Output: 2 (TrackState.Lost)
    """
    _count = 0

    def __init__(self):
        """
        Initializes a new track with a unique ID and foundational tracking attributes.

        Examples:
            Initialize a new track
            >>> track = BaseTrack()
            >>> print(track.track_id)
            0
        """
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = np.inf, np.inf

    @property
    def end_frame(self):
        """Returns the ID of the most recent frame where the object was tracked."""
        return self.frame_id

    @staticmethod
    def next_id():
        """Increment and return the next unique global track ID for object tracking."""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """Activates the track with provided arguments, initializing necessary attributes for tracking."""
        raise NotImplementedError

    def predict(self):
        """Predicts the next state of the track based on the current state and tracking model."""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """Updates the track with new observations and data, modifying its state and attributes accordingly."""
        raise NotImplementedError

    def mark_lost(self):
        """Marks the track as lost by updating its state to TrackState.Lost."""
        self.state = TrackState.Lost

    def mark_removed(self):
        """Marks the track as removed by setting its state to TrackState.Removed."""
        self.state = TrackState.Removed

    @staticmethod
    def reset_id():
        """Reset the global track ID counter to its initial value."""
        BaseTrack._count = 0


class STrack(BaseTrack):
    """
    Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter that is used across all STrack instances for prediction.
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        score (float): Confidence score of the track.
        tracklet_len (int): Length of the tracklet.
        cls (Any): Class label for the object.
        idx (int): Index or identifier for the object.
        frame_id (int): Current frame ID.
        start_frame (int): Frame where the object was first detected.

    Methods:
        predict(): Predict the next state of the object using Kalman filter.
        multi_predict(stracks): Predict the next states for multiple tracks.
        multi_gmc(stracks, H): Update multiple track states using a homography matrix.
        activate(kalman_filter, frame_id): Activate a new tracklet.
        re_activate(new_track, frame_id, new_id): Reactivate a previously lost tracklet.
        update(new_track, frame_id): Update the state of a matched track.
        convert_coords(tlwh): Convert bounding box to x-y-aspect-height format.
        tlwh_to_xyah(tlwh): Convert tlwh bounding box to xyah format.

    Examples:
        Initialize and activate a new track
        >>> track = STrack(xywh=[100, 200, 50, 80, 0], score=0.9, cls="person")
        >>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)
    """
    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh, score, cls):
        """
        Initialize a new STrack instance.

        Args:
            xywh (List[float]): Bounding box coordinates and dimensions in the format (x, y, w, h, [a], idx), where
                (x, y) is the center, (w, h) are width and height, [a] is optional aspect ratio, and idx is the id.
            score (float): Confidence score of the detection.
            cls (Any): Class label for the detected object.

        Examples:
            >>> xywh = [100.0, 150.0, 50.0, 75.0, 1]
            >>> score = 0.9
            >>> cls = "person"
            >>> track = STrack(xywh, score, cls)
        """
        super().__init__()
        assert len(xywh) in {5, 6}, f'expected 5 or 6 values but got {len(xywh)}'
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

    def predict(self):
        """Predicts the next state (mean and covariance) of the object using the Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Update state tracks positions and covariances using a homography matrix for multiple tracks."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Activate a new tracklet using the provided Kalman filter and initialize its state and covariance."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a previously lost track using new detection data and updates its state and attributes."""
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.convert_coords(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track, frame_id):
        """
        Update the state of a matched track.

        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.

        Examples:
            Update the state of a track with new detection information
            >>> track = STrack([100, 200, 50, 80, 0.9, 1])
            >>> new_track = STrack([105, 205, 55, 85, 0.95, 1])
            >>> track.update(new_track, 2)
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.convert_coords(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh):
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        """Returns the bounding box in top-left-width-height format from the current state estimate."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self):
        """Converts bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self):
        """Returns the current position of the bounding box in (center x, center y, width, height) format."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self):
        """Returns position in (center x, center y, width, height, angle) format, warning if angle is missing."""
        if self.angle is None:
            LOGGER.warning('WARNING ⚠️ `angle` attr not found, returning `xywh` instead.')
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self):
        """Returns the current tracking results in the appropriate bounding box format."""
        coords = self.xyxy if self.angle is None else self.xywha
        return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]

    def __repr__(self):
        """Returns a string representation of the STrack object including start frame, end frame, and track ID."""
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


class BOTrack(STrack):
    """
    An extended version of the STrack class for YOLOv8, adding object tracking features.

    This class extends the STrack class to include additional functionalities for object tracking, such as feature
    smoothing, Kalman filter prediction, and reactivation of tracks.

    Attributes:
        shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of BOTrack.
        smooth_feat (np.ndarray): Smoothed feature vector.
        curr_feat (np.ndarray): Current feature vector.
        features (deque): A deque to store feature vectors with a maximum length defined by `feat_history`.
        alpha (float): Smoothing factor for the exponential moving average of features.
        mean (np.ndarray): The mean state of the Kalman filter.
        covariance (np.ndarray): The covariance matrix of the Kalman filter.

    Methods:
        update_features(feat): Update features vector and smooth it using exponential moving average.
        predict(): Predicts the mean and covariance using Kalman filter.
        re_activate(new_track, frame_id, new_id): Reactivates a track with updated features and optionally new ID.
        update(new_track, frame_id): Update the YOLOv8 instance with new track and frame ID.
        tlwh: Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
        multi_predict(stracks): Predicts the mean and covariance of multiple object tracks using shared Kalman filter.
        convert_coords(tlwh): Converts tlwh bounding box coordinates to xywh format.
        tlwh_to_xywh(tlwh): Convert bounding box to xywh format `(center x, center y, width, height)`.

    Examples:
        Create a BOTrack instance and update its features
        >>> bo_track = BOTrack(tlwh=[100, 50, 80, 40], score=0.9, cls=1, feat=np.random.rand(128))
        >>> bo_track.predict()
        >>> new_track = BOTrack(tlwh=[110, 60, 80, 40], score=0.85, cls=1, feat=np.random.rand(128))
        >>> bo_track.update(new_track, frame_id=2)
    """
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):
        """
        Initialize a BOTrack object with temporal parameters, such as feature history, alpha, and current features.

        Args:
            tlwh (np.ndarray): Bounding box coordinates in tlwh format (top left x, top left y, width, height).
            score (float): Confidence score of the detection.
            cls (int): Class ID of the detected object.
            feat (np.ndarray | None): Feature vector associated with the detection.
            feat_history (int): Maximum length of the feature history deque.

        Examples:
            Initialize a BOTrack object with bounding box, score, class ID, and feature vector
            >>> tlwh = np.array([100, 50, 80, 120])
            >>> score = 0.9
            >>> cls = 1
            >>> feat = np.random.rand(128)
            >>> bo_track = BOTrack(tlwh, score, cls, feat)
        """
        super().__init__(tlwh, score, cls)
        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        """Update the feature vector and apply exponential moving average smoothing."""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        """Predicts the object's future state using the Kalman filter to update its mean and covariance."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a track with updated features and optionally assigns a new ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track, frame_id):
        """Updates the YOLOv8 instance with new track information and the current frame ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    @property
    def tlwh(self):
        """Returns the current bounding box position in `(top left x, top left y, width, height)` format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
        """Predicts the mean and covariance for multiple object tracks using a shared Kalman filter."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh):
        """Converts tlwh bounding box coordinates to xywh format."""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box from tlwh (top-left-width-height) to xywh (center-x-center-y-width-height) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BYTETracker:
    """
    BYTETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking.

    Responsible for initializing, updating, and managing the tracks for detected objects in a video sequence.
    It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman filtering for predicting
    the new object locations, and performs data association.

    Attributes:
        tracked_stracks (List[STrack]): List of successfully activated tracks.
        lost_stracks (List[STrack]): List of lost tracks.
        removed_stracks (List[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (Namespace): Command-line arguments.
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
        kalman_filter (KalmanFilterXYAH): Kalman Filter object.

    Methods:
        update(results, img=None): Updates object tracker with new detections.
        get_kalmanfilter(): Returns a Kalman filter object for tracking bounding boxes.
        init_track(dets, scores, cls, img=None): Initialize object tracking with detections.
        get_dists(tracks, detections): Calculates the distance between tracks and detections.
        multi_predict(tracks): Predicts the location of tracks.
        reset_id(): Resets the ID counter of STrack.
        joint_stracks(tlista, tlistb): Combines two lists of stracks.
        sub_stracks(tlista, tlistb): Filters out the stracks present in the second list from the first list.
        remove_duplicate_stracks(stracksa, stracksb): Removes duplicate stracks based on IoU.

    Examples:
        Initialize BYTETracker and update with detection results
        >>> tracker = BYTETracker(args, frame_rate=30)
        >>> results = yolo_model.detect(image)
        >>> tracked_objects = tracker.update(results)
    """

    def __init__(self, args, frame_rate=30):
        """
        Initialize a BYTETracker instance for object tracking.

        Args:
            args (Namespace): Command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video sequence.

        Examples:
            Initialize BYTETracker with command-line arguments and a frame rate of 30
            >>> args = Namespace(track_buffer=30)
            >>> tracker = BYTETracker(args, frame_rate=30)
        """
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def update(self, results, img=None):
        """Updates the tracker with new detections and returns the current list of tracked objects."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        scores = results.conf
        bboxes = results.xywhr if hasattr(results, 'xywhr') else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh
        inds_second = inds_low & inds_high
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]
        detections = self.init_track(dets, scores_keep, cls_keep, img)
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)
        if hasattr(self, 'gmc') and img is not None:
            warp = self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)
        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        detections_second = self.init_track(dets_second, scores_second, cls_second, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]
        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_kalmanfilter(self):
        """Returns a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH."""
        return KalmanFilterXYAH()

    def init_track(self, dets, scores, cls, img=None):
        """Initializes object tracking with given detections, scores, and class labels using the STrack algorithm."""
        return [STrack(xyxy, s, c) for xyxy, s, c in zip(dets, scores, cls)] if len(dets) else []

    def get_dists(self, tracks, detections):
        """Calculates the distance between tracks and detections using IoU and optionally fuses scores."""
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks):
        """Predict the next states for multiple tracks using Kalman filter."""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Resets the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
        STrack.reset_id()

    def reset(self):
        """Resets the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """Combines two lists of STrack objects into a single list, ensuring no duplicates based on track IDs."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """Filters out the stracks present in the second list from the first list."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Removes duplicate stracks from two lists based on Intersection over Union (IoU) distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb


class GMC:
    """
    Generalized Motion Compensation (GMC) class for tracking and object detection in video frames.

    This class provides methods for tracking and detecting objects based on several tracking algorithms including ORB,
    SIFT, ECC, and Sparse Optical Flow. It also supports downscaling of frames for computational efficiency.

    Attributes:
        method (str): The method used for tracking. Options include 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'.
        downscale (int): Factor by which to downscale the frames for processing.
        prevFrame (np.ndarray): Stores the previous frame for tracking.
        prevKeyPoints (List): Stores the keypoints from the previous frame.
        prevDescriptors (np.ndarray): Stores the descriptors from the previous frame.
        initializedFirstFrame (bool): Flag to indicate if the first frame has been processed.

    Methods:
        __init__: Initializes a GMC object with the specified method and downscale factor.
        apply: Applies the chosen method to a raw frame and optionally uses provided detections.
        applyEcc: Applies the ECC algorithm to a raw frame.
        applyFeatures: Applies feature-based methods like ORB or SIFT to a raw frame.
        applySparseOptFlow: Applies the Sparse Optical Flow method to a raw frame.
        reset_params: Resets the internal parameters of the GMC object.

    Examples:
        Create a GMC object and apply it to a frame
        >>> gmc = GMC(method="sparseOptFlow", downscale=2)
        >>> frame = np.array([[1, 2, 3], [4, 5, 6]])
        >>> processed_frame = gmc.apply(frame)
        >>> print(processed_frame)
        array([[1, 2, 3],
               [4, 5, 6]])
    """

    def __init__(self, method: 'str'='sparseOptFlow', downscale: 'int'=2) ->None:
        """
        Initialize a Generalized Motion Compensation (GMC) object with tracking method and downscale factor.

        Args:
            method (str): The method used for tracking. Options include 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'.
            downscale (int): Downscale factor for processing frames.

        Examples:
            Initialize a GMC object with the 'sparseOptFlow' method and a downscale factor of 2
            >>> gmc = GMC(method="sparseOptFlow", downscale=2)
        """
        super().__init__()
        self.method = method
        self.downscale = max(1, downscale)
        if self.method == 'orb':
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif self.method == 'ecc':
            number_of_iterations = 5000
            termination_eps = 1e-06
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps
        elif self.method == 'sparseOptFlow':
            self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04)
        elif self.method in {'none', 'None', None}:
            self.method = None
        else:
            raise ValueError(f'Error: Unknown GMC method:{method}')
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False

    def apply(self, raw_frame: 'np.array', detections: 'list'=None) ->np.array:
        """
        Apply object detection on a raw frame using the specified method.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed, with shape (H, W, C).
            detections (List | None): List of detections to be used in the processing.

        Returns:
            (np.ndarray): Processed frame with applied object detection.

        Examples:
            >>> gmc = GMC(method="sparseOptFlow")
            >>> raw_frame = np.random.rand(480, 640, 3)
            >>> processed_frame = gmc.apply(raw_frame)
            >>> print(processed_frame.shape)
            (480, 640, 3)
        """
        if self.method in {'orb', 'sift'}:
            return self.applyFeatures(raw_frame, detections)
        elif self.method == 'ecc':
            return self.applyEcc(raw_frame)
        elif self.method == 'sparseOptFlow':
            return self.applySparseOptFlow(raw_frame)
        else:
            return np.eye(2, 3)

    def applyEcc(self, raw_frame: 'np.array') ->np.array:
        """
        Apply the ECC (Enhanced Correlation Coefficient) algorithm to a raw frame for motion compensation.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed, with shape (H, W, C).

        Returns:
            (np.ndarray): The processed frame with the applied ECC transformation.

        Examples:
            >>> gmc = GMC(method="ecc")
            >>> processed_frame = gmc.applyEcc(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
            >>> print(processed_frame)
            [[1. 0. 0.]
             [0. 1. 0.]]
        """
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.initializedFirstFrame = True
            return H
        try:
            _, H = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except Exception as e:
            LOGGER.warning(f'WARNING: find transform failed. Set warp as identity {e}')
        return H

    def applyFeatures(self, raw_frame: 'np.array', detections: 'list'=None) ->np.array:
        """
        Apply feature-based methods like ORB or SIFT to a raw frame.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed, with shape (H, W, C).
            detections (List | None): List of detections to be used in the processing.

        Returns:
            (np.ndarray): Processed frame.

        Examples:
            >>> gmc = GMC(method="orb")
            >>> raw_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> processed_frame = gmc.applyFeatures(raw_frame)
            >>> print(processed_frame.shape)
            (2, 3)
        """
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale
        mask = np.zeros_like(frame)
        mask[int(0.02 * height):int(0.98 * height), int(0.02 * width):int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0
        keypoints = self.detector.detect(frame, mask)
        keypoints, descriptors = self.extractor.compute(frame, keypoints)
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            self.initializedFirstFrame = True
            return H
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)
        matches = []
        spatialDistances = []
        maxSpatialDistance = 0.25 * np.array([width, height])
        if len(knnMatches) == 0:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H
        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt
                spatialDistance = prevKeyPointLocation[0] - currKeyPointLocation[0], prevKeyPointLocation[1] - currKeyPointLocation[1]
                if np.abs(spatialDistance[0]) < maxSpatialDistance[0] and np.abs(spatialDistance[1]) < maxSpatialDistance[1]:
                    spatialDistances.append(spatialDistance)
                    matches.append(m)
        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)
        inliers = spatialDistances - meanSpatialDistances < 2.5 * stdSpatialDistances
        goodMatches = []
        prevPoints = []
        currPoints = []
        for i in range(len(matches)):
            if inliers[i, 0] and inliers[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)
        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)
        if prevPoints.shape[0] > 4:
            H, inliers = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            LOGGER.warning('WARNING: not enough matching points')
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)
        return H

    def applySparseOptFlow(self, raw_frame: 'np.array') ->np.array:
        """
        Apply Sparse Optical Flow method to a raw frame.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed, with shape (H, W, C).

        Returns:
            (np.ndarray): Processed frame with shape (2, 3).

        Examples:
            >>> gmc = GMC()
            >>> result = gmc.applySparseOptFlow(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
            >>> print(result)
            [[1. 0. 0.]
             [0. 1. 0.]]
        """
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
        if not self.initializedFirstFrame or self.prevKeyPoints is None:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.initializedFirstFrame = True
            return H
        matchedKeypoints, status, _ = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)
        prevPoints = []
        currPoints = []
        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])
        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)
        if prevPoints.shape[0] > 4 and prevPoints.shape[0] == prevPoints.shape[0]:
            H, _ = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            LOGGER.warning('WARNING: not enough matching points')
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        return H

    def reset_params(self) ->None:
        """Reset the internal parameters including previous frame, keypoints, and descriptors."""
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False


class BOTSORT(BYTETracker):
    """
    An extended version of the BYTETracker class for YOLOv8, designed for object tracking with ReID and GMC algorithm.

    Attributes:
        proximity_thresh (float): Threshold for spatial proximity (IoU) between tracks and detections.
        appearance_thresh (float): Threshold for appearance similarity (ReID embeddings) between tracks and detections.
        encoder (Any): Object to handle ReID embeddings, set to None if ReID is not enabled.
        gmc (GMC): An instance of the GMC algorithm for data association.
        args (Any): Parsed command-line arguments containing tracking parameters.

    Methods:
        get_kalmanfilter(): Returns an instance of KalmanFilterXYWH for object tracking.
        init_track(dets, scores, cls, img): Initialize track with detections, scores, and classes.
        get_dists(tracks, detections): Get distances between tracks and detections using IoU and (optionally) ReID.
        multi_predict(tracks): Predict and track multiple objects with YOLOv8 model.

    Examples:
        Initialize BOTSORT and process detections
        >>> bot_sort = BOTSORT(args, frame_rate=30)
        >>> bot_sort.init_track(dets, scores, cls, img)
        >>> bot_sort.multi_predict(tracks)

    Note:
        The class is designed to work with the YOLOv8 object detection model and supports ReID only if enabled via args.
    """

    def __init__(self, args, frame_rate=30):
        """
        Initialize YOLOv8 object with ReID module and GMC algorithm.

        Args:
            args (object): Parsed command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video being processed.

        Examples:
            Initialize BOTSORT with command-line arguments and a specified frame rate:
            >>> args = parse_args()
            >>> bot_sort = BOTSORT(args, frame_rate=30)
        """
        super().__init__(args, frame_rate)
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        if args.with_reid:
            self.encoder = None
        self.gmc = GMC(method=args.gmc_method)

    def get_kalmanfilter(self):
        """Returns an instance of KalmanFilterXYWH for predicting and updating object states in the tracking process."""
        return KalmanFilterXYWH()

    def init_track(self, dets, scores, cls, img=None):
        """Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features."""
        if len(dets) == 0:
            return []
        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder.inference(img, dets)
            return [BOTrack(xyxy, s, c, f) for xyxy, s, c, f in zip(dets, scores, cls, features_keep)]
        else:
            return [BOTrack(xyxy, s, c) for xyxy, s, c in zip(dets, scores, cls)]

    def get_dists(self, tracks, detections):
        """Calculates distances between tracks and detections using IoU and optionally ReID embeddings."""
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > self.proximity_thresh
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists

    def multi_predict(self, tracks):
        """Predicts the mean and covariance of multiple object tracks using a shared Kalman filter."""
        BOTrack.multi_predict(tracks)

    def reset(self):
        """Resets the BOTSORT tracker to its initial state, clearing all tracked objects and internal states."""
        super().reset()
        self.gmc.reset_params()


TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}


def on_predict_start(predictor: 'object', persist: 'bool'=False) ->None:
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool): Whether to persist the trackers if they already exist.

    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.

    Examples:
        Initialize trackers for a predictor object:
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    if hasattr(predictor, 'trackers') and persist:
        return
    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))
    if cfg.tracker_type not in {'bytetrack', 'botsort'}:
        raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")
    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        if predictor.dataset.mode != 'stream':
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs


def register_tracker(model: 'object', persist: 'bool') ->None:
    """
    Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Register tracking callbacks to a YOLO model
        >>> model = YOLOModel()
        >>> register_tracker(model, persist=True)
    """
    model.add_callback('on_predict_start', partial(on_predict_start, persist=persist))
    model.add_callback('on_predict_postprocess_end', partial(on_predict_postprocess_end, persist=persist))


TASK2METRIC = {'detect': 'metrics/mAP50-95(B)', 'segment': 'metrics/mAP50-95(M)', 'classify': 'metrics/accuracy_top1', 'pose': 'metrics/mAP50-95(P)', 'obb': 'metrics/mAP50-95(B)'}


def run_ray_tune(model, space: 'dict'=None, grace_period: 'int'=10, gpu_per_trial: 'int'=None, max_samples: 'int'=10, **train_args):
    """
    Runs hyperparameter tuning using Ray Tune.

    Args:
        model (YOLO): Model to run the tuner on.
        space (dict, optional): The hyperparameter search space. Defaults to None.
        grace_period (int, optional): The grace period in epochs of the ASHA scheduler. Defaults to 10.
        gpu_per_trial (int, optional): The number of GPUs to allocate per trial. Defaults to None.
        max_samples (int, optional): The maximum number of trials to run. Defaults to 10.
        train_args (dict, optional): Additional arguments to pass to the `train()` method. Defaults to {}.

    Returns:
        (dict): A dictionary containing the results of the hyperparameter search.

    Example:
        ```python
        from ultralytics import YOLO

        # Load a YOLOv8n model
        model = YOLO("yolo11n.pt")

        # Start tuning hyperparameters for YOLOv8n training on the COCO8 dataset
        result_grid = model.tune(data="coco8.yaml", use_ray=True)
        ```
    """
    LOGGER.info('💡 Learn about RayTune at https://docs.ultralytics.com/integrations/ray-tune')
    if train_args is None:
        train_args = {}
    try:
        checks.check_requirements('ray[tune]')
    except ImportError:
        raise ModuleNotFoundError('Ray Tune required but not found. To install run: pip install "ray[tune]"')
    try:
        assert hasattr(wandb, '__version__')
    except (ImportError, AssertionError):
        wandb = False
    checks.check_version(ray.__version__, '>=2.0.0', 'ray')
    default_space = {'lr0': tune.uniform(1e-05, 0.1), 'lrf': tune.uniform(0.01, 1.0), 'momentum': tune.uniform(0.6, 0.98), 'weight_decay': tune.uniform(0.0, 0.001), 'warmup_epochs': tune.uniform(0.0, 5.0), 'warmup_momentum': tune.uniform(0.0, 0.95), 'box': tune.uniform(0.02, 0.2), 'cls': tune.uniform(0.2, 4.0), 'hsv_h': tune.uniform(0.0, 0.1), 'hsv_s': tune.uniform(0.0, 0.9), 'hsv_v': tune.uniform(0.0, 0.9), 'degrees': tune.uniform(0.0, 45.0), 'translate': tune.uniform(0.0, 0.9), 'scale': tune.uniform(0.0, 0.9), 'shear': tune.uniform(0.0, 10.0), 'perspective': tune.uniform(0.0, 0.001), 'flipud': tune.uniform(0.0, 1.0), 'fliplr': tune.uniform(0.0, 1.0), 'bgr': tune.uniform(0.0, 1.0), 'mosaic': tune.uniform(0.0, 1.0), 'mixup': tune.uniform(0.0, 1.0), 'copy_paste': tune.uniform(0.0, 1.0)}
    task = model.task
    model_in_store = ray.put(model)

    def _tune(config):
        """
        Trains the YOLO model with the specified hyperparameters and additional arguments.

        Args:
            config (dict): A dictionary of hyperparameters to use for training.

        Returns:
            None
        """
        model_to_train = ray.get(model_in_store)
        model_to_train.reset_callbacks()
        config.update(train_args)
        results = model_to_train.train(**config)
        return results.results_dict
    if not space:
        space = default_space
        LOGGER.warning('WARNING ⚠️ search space not provided, using default search space.')
    data = train_args.get('data', TASK2DATA[task])
    space['data'] = data
    if 'data' not in train_args:
        LOGGER.warning(f'WARNING ⚠️ data not provided, using default "data={data}".')
    trainable_with_resources = tune.with_resources(_tune, {'cpu': NUM_THREADS, 'gpu': gpu_per_trial or 0})
    asha_scheduler = ASHAScheduler(time_attr='epoch', metric=TASK2METRIC[task], mode='max', max_t=train_args.get('epochs') or DEFAULT_CFG_DICT['epochs'] or 100, grace_period=grace_period, reduction_factor=3)
    tuner_callbacks = [WandbLoggerCallback(project='YOLOv8-tune')] if wandb else []
    tune_dir = get_save_dir(DEFAULT_CFG, name='tune').resolve()
    tune_dir.mkdir(parents=True, exist_ok=True)
    tuner = tune.Tuner(trainable_with_resources, param_space=space, tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=max_samples), run_config=RunConfig(callbacks=tuner_callbacks, storage_path=tune_dir))
    tuner.fit()
    results = tuner.get_results()
    ray.shutdown()
    return results


class TritonRemoteModel:
    """
    Client for interacting with a remote Triton Inference Server model.

    Attributes:
        endpoint (str): The name of the model on the Triton server.
        url (str): The URL of the Triton server.
        triton_client: The Triton client (either HTTP or gRPC).
        InferInput: The input class for the Triton client.
        InferRequestedOutput: The output request class for the Triton client.
        input_formats (List[str]): The data types of the model inputs.
        np_input_formats (List[type]): The numpy data types of the model inputs.
        input_names (List[str]): The names of the model inputs.
        output_names (List[str]): The names of the model outputs.
    """

    def __init__(self, url: 'str', endpoint: 'str'='', scheme: 'str'=''):
        """
        Initialize the TritonRemoteModel.

        Arguments may be provided individually or parsed from a collective 'url' argument of the form
            <scheme>://<netloc>/<endpoint>/<task_name>

        Args:
            url (str): The URL of the Triton server.
            endpoint (str): The name of the model on the Triton server.
            scheme (str): The communication scheme ('http' or 'grpc').
        """
        if not endpoint and not scheme:
            splits = urlsplit(url)
            endpoint = splits.path.strip('/').split('/')[0]
            scheme = splits.scheme
            url = splits.netloc
        self.endpoint = endpoint
        self.url = url
        if scheme == 'http':
            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)
            config = self.triton_client.get_model_config(endpoint)
        else:
            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)
            config = self.triton_client.get_model_config(endpoint, as_json=True)['config']
        config['output'] = sorted(config['output'], key=lambda x: x.get('name'))
        type_map = {'TYPE_FP32': np.float32, 'TYPE_FP16': np.float16, 'TYPE_UINT8': np.uint8}
        self.InferRequestedOutput = client.InferRequestedOutput
        self.InferInput = client.InferInput
        self.input_formats = [x['data_type'] for x in config['input']]
        self.np_input_formats = [type_map[x] for x in self.input_formats]
        self.input_names = [x['name'] for x in config['input']]
        self.output_names = [x['name'] for x in config['output']]

    def __call__(self, *inputs: np.ndarray) ->List[np.ndarray]:
        """
        Call the model with the given inputs.

        Args:
            *inputs (List[np.ndarray]): Input data to the model.

        Returns:
            (List[np.ndarray]): Model outputs.
        """
        infer_inputs = []
        input_format = inputs[0].dtype
        for i, x in enumerate(inputs):
            if x.dtype != self.np_input_formats[i]:
                x = x.astype(self.np_input_formats[i])
            infer_input = self.InferInput(self.input_names[i], [*x.shape], self.input_formats[i].replace('TYPE_', ''))
            infer_input.set_data_from_numpy(x)
            infer_inputs.append(infer_input)
        infer_outputs = [self.InferRequestedOutput(output_name) for output_name in self.output_names]
        outputs = self.triton_client.infer(model_name=self.endpoint, inputs=infer_inputs, outputs=infer_outputs)
        return [outputs.as_numpy(output_name).astype(input_format) for output_name in self.output_names]


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 2)
        return y, None


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""
    ensemble = Ensemble()
    for w in (weights if isinstance(weights, list) else [weights]):
        ckpt, w = torch_safe_load(w)
        args = {**DEFAULT_CFG_DICT, **ckpt['train_args']} if 'train_args' in ckpt else None
        model = (ckpt.get('ema') or ckpt['model']).float()
        model.args = args
        model.pt_path = w
        model.task = guess_model_task(model)
        if not hasattr(model, 'stride'):
            model.stride = torch.tensor([32.0])
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval())
    for m in ensemble.modules():
        if hasattr(m, 'inplace'):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None
    if len(ensemble) == 1:
        return ensemble[-1]
    LOGGER.info(f'Ensemble created with {weights}\n')
    for k in ('names', 'nc', 'yaml'):
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f'Models differ in class counts {[m.nc for m in ensemble]}'
    return ensemble


def is_url(url, check=False):
    """
    Validates if the given string is a URL and optionally checks if the URL exists online.

    Args:
        url (str): The string to be validated as a URL.
        check (bool, optional): If True, performs an additional check to see if the URL exists online.
            Defaults to False.

    Returns:
        (bool): Returns True for a valid URL. If 'check' is True, also returns True if the URL exists online.
            Returns False otherwise.

    Example:
        ```python
        valid = is_url("https://www.example.com")
        ```
    """
    try:
        url = str(url)
        result = parse.urlparse(url)
        assert all([result.scheme, result.netloc])
        if check:
            with request.urlopen(url) as response:
                return response.getcode() == 200
        return True
    except Exception:
        return False


STREAM_WARNING = """
WARNING ⚠️ inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
"""


def check_imshow(warn=False):
    """Check if environment supports image displays."""
    try:
        if LINUX:
            assert not IS_COLAB and not IS_KAGGLE
            assert 'DISPLAY' in os.environ, "The DISPLAY environment variable isn't set."
        cv2.imshow('test', np.zeros((8, 8, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f'WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}')
        return False


DEFAULT_CROP_FRACTION = 1.0


DEFAULT_MEAN = 0.0, 0.0, 0.0


DEFAULT_STD = 1.0, 1.0, 1.0


def classify_transforms(size=224, mean=DEFAULT_MEAN, std=DEFAULT_STD, interpolation='BILINEAR', crop_fraction: 'float'=DEFAULT_CROP_FRACTION):
    """
    Creates a composition of image transforms for classification tasks.

    This function generates a sequence of torchvision transforms suitable for preprocessing images
    for classification models during evaluation or inference. The transforms include resizing,
    center cropping, conversion to tensor, and normalization.

    Args:
        size (int | tuple): The target size for the transformed image. If an int, it defines the shortest edge. If a
            tuple, it defines (height, width).
        mean (tuple): Mean values for each RGB channel used in normalization.
        std (tuple): Standard deviation values for each RGB channel used in normalization.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.
        crop_fraction (float): Fraction of the image to be cropped.

    Returns:
        (torchvision.transforms.Compose): A composition of torchvision transforms.

    Examples:
        >>> transforms = classify_transforms(size=224)
        >>> img = Image.open("path/to/image.jpg")
        >>> transformed_img = transforms(img)
    """
    import torchvision.transforms as T
    if isinstance(size, (tuple, list)):
        assert len(size) == 2, f"'size' tuples must be length 2, not length {len(size)}"
        scale_size = tuple(math.floor(x / crop_fraction) for x in size)
    else:
        scale_size = math.floor(size / crop_fraction)
        scale_size = scale_size, scale_size
    if scale_size[0] == scale_size[1]:
        tfl = [T.Resize(scale_size[0], interpolation=getattr(T.InterpolationMode, interpolation))]
    else:
        tfl = [T.Resize(scale_size)]
    tfl.extend([T.CenterCrop(size), T.ToTensor(), T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))])
    return T.Compose(tfl)


class LoadImagesAndVideos:
    """
    A class for loading and processing images and videos for YOLO object detection.

    This class manages the loading and pre-processing of image and video data from various sources, including
    single image files, video files, and lists of image and video paths.

    Attributes:
        files (List[str]): List of image and video file paths.
        nf (int): Total number of files (images and videos).
        video_flag (List[bool]): Flags indicating whether a file is a video (True) or an image (False).
        mode (str): Current mode, 'image' or 'video'.
        vid_stride (int): Stride for video frame-rate.
        bs (int): Batch size.
        cap (cv2.VideoCapture): Video capture object for OpenCV.
        frame (int): Frame counter for video.
        frames (int): Total number of frames in the video.
        count (int): Counter for iteration, initialized at 0 during __iter__().
        ni (int): Number of images.

    Methods:
        __init__: Initialize the LoadImagesAndVideos object.
        __iter__: Returns an iterator object for VideoStream or ImageFolder.
        __next__: Returns the next batch of images or video frames along with their paths and metadata.
        _new_video: Creates a new video capture object for the given path.
        __len__: Returns the number of batches in the object.

    Examples:
        >>> loader = LoadImagesAndVideos("path/to/data", batch=32, vid_stride=1)
        >>> for paths, imgs, info in loader:
        ...     # Process batch of images or video frames
        ...     pass

    Notes:
        - Supports various image formats including HEIC.
        - Handles both local files and directories.
        - Can read from a text file containing paths to images and videos.
    """

    def __init__(self, path, batch=1, vid_stride=1):
        """Initialize dataloader for images and videos, supporting various input formats."""
        parent = None
        if isinstance(path, str) and Path(path).suffix == '.txt':
            parent = Path(path).parent
            path = Path(path).read_text().splitlines()
        files = []
        for p in (sorted(path) if isinstance(path, (list, tuple)) else [path]):
            a = str(Path(p).absolute())
            if '*' in a:
                files.extend(sorted(glob.glob(a, recursive=True)))
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, '*.*'))))
            elif os.path.isfile(a):
                files.append(a)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))
            else:
                raise FileNotFoundError(f'{p} does not exist')
        images, videos = [], []
        for f in files:
            suffix = f.split('.')[-1].lower()
            if suffix in IMG_FORMATS:
                images.append(f)
            elif suffix in VID_FORMATS:
                videos.append(f)
        ni, nv = len(images), len(videos)
        self.files = images + videos
        self.nf = ni + nv
        self.ni = ni
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.vid_stride = vid_stride
        self.bs = batch
        if any(videos):
            self._new_video(videos[0])
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f'No images or videos found in {p}. {FORMATS_HELP_MSG}')

    def __iter__(self):
        """Iterates through image/video files, yielding source paths, images, and metadata."""
        self.count = 0
        return self

    def __next__(self):
        """Returns the next batch of images or video frames with their paths and metadata."""
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:
                if imgs:
                    return paths, imgs, info
                else:
                    raise StopIteration
            path = self.files[self.count]
            if self.video_flag[self.count]:
                self.mode = 'video'
                if not self.cap or not self.cap.isOpened():
                    self._new_video(path)
                success = False
                for _ in range(self.vid_stride):
                    success = self.cap.grab()
                    if not success:
                        break
                if success:
                    success, im0 = self.cap.retrieve()
                    if success:
                        self.frame += 1
                        paths.append(path)
                        imgs.append(im0)
                        info.append(f'video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ')
                        if self.frame == self.frames:
                            self.count += 1
                            self.cap.release()
                else:
                    self.count += 1
                    if self.cap:
                        self.cap.release()
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])
            else:
                self.mode = 'image'
                if path.split('.')[-1].lower() == 'heic':
                    check_requirements('pillow-heif')
                    register_heif_opener()
                    with Image.open(path) as img:
                        im0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                else:
                    im0 = imread(path)
                if im0 is None:
                    LOGGER.warning(f'WARNING ⚠️ Image Read Error {path}')
                else:
                    paths.append(path)
                    imgs.append(im0)
                    info.append(f'image {self.count + 1}/{self.nf} {path}: ')
                self.count += 1
                if self.count >= self.ni:
                    break
        return paths, imgs, info

    def _new_video(self, path):
        """Creates a new video capture object for the given path and initializes video-related attributes."""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f'Failed to open video {path}')
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self):
        """Returns the number of files (images and videos) in the dataset."""
        return math.ceil(self.nf / self.bs)


class LoadPilAndNumpy:
    """
    Load images from PIL and Numpy arrays for batch processing.

    This class manages loading and pre-processing of image data from both PIL and Numpy formats. It performs basic
    validation and format conversion to ensure that the images are in the required format for downstream processing.

    Attributes:
        paths (List[str]): List of image paths or autogenerated filenames.
        im0 (List[np.ndarray]): List of images stored as Numpy arrays.
        mode (str): Type of data being processed, set to 'image'.
        bs (int): Batch size, equivalent to the length of `im0`.

    Methods:
        _single_check: Validate and format a single image to a Numpy array.

    Examples:
        >>> from PIL import Image
        >>> import numpy as np
        >>> pil_img = Image.new("RGB", (100, 100))
        >>> np_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> loader = LoadPilAndNumpy([pil_img, np_img])
        >>> paths, images, _ = next(iter(loader))
        >>> print(f"Loaded {len(images)} images")
        Loaded 2 images
    """

    def __init__(self, im0):
        """Initializes a loader for PIL and Numpy images, converting inputs to a standardized format."""
        if not isinstance(im0, list):
            im0 = [im0]
        self.paths = [(getattr(im, 'filename', '') or f'image{i}.jpg') for i, im in enumerate(im0)]
        self.im0 = [self._single_check(im) for im in im0]
        self.mode = 'image'
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im):
        """Validate and format an image to numpy array, ensuring RGB order and contiguous memory."""
        assert isinstance(im, (Image.Image, np.ndarray)), f'Expected PIL/np.ndarray image type, but got {type(im)}'
        if isinstance(im, Image.Image):
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)
        return im

    def __len__(self):
        """Returns the length of the 'im0' attribute, representing the number of loaded images."""
        return len(self.im0)

    def __next__(self):
        """Returns the next batch of images, paths, and metadata for processing."""
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [''] * self.bs

    def __iter__(self):
        """Iterates through PIL/numpy images, yielding paths, raw images, and metadata for processing."""
        self.count = 0
        return self


class LoadScreenshots:
    """
    Ultralytics screenshot dataloader for capturing and processing screen images.

    This class manages the loading of screenshot images for processing with YOLO. It is suitable for use with
    `yolo predict source=screen`.

    Attributes:
        source (str): The source input indicating which screen to capture.
        screen (int): The screen number to capture.
        left (int): The left coordinate for screen capture area.
        top (int): The top coordinate for screen capture area.
        width (int): The width of the screen capture area.
        height (int): The height of the screen capture area.
        mode (str): Set to 'stream' indicating real-time capture.
        frame (int): Counter for captured frames.
        sct (mss.mss): Screen capture object from `mss` library.
        bs (int): Batch size, set to 1.
        fps (int): Frames per second, set to 30.
        monitor (Dict[str, int]): Monitor configuration details.

    Methods:
        __iter__: Returns an iterator object.
        __next__: Captures the next screenshot and returns it.

    Examples:
        >>> loader = LoadScreenshots("0 100 100 640 480")  # screen 0, top-left (100,100), 640x480
        >>> for source, im, im0s, vid_cap, s in loader:
        ...     print(f"Captured frame: {im.shape}")
    """

    def __init__(self, source):
        """Initialize screenshot capture with specified screen and region parameters."""
        check_requirements('mss')
        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.mode = 'stream'
        self.frame = 0
        self.sct = mss.mss()
        self.bs = 1
        self.fps = 30
        monitor = self.sct.monitors[self.screen]
        self.top = monitor['top'] if top is None else monitor['top'] + top
        self.left = monitor['left'] if left is None else monitor['left'] + left
        self.width = width or monitor['width']
        self.height = height or monitor['height']
        self.monitor = {'left': self.left, 'top': self.top, 'width': self.width, 'height': self.height}

    def __iter__(self):
        """Yields the next screenshot image from the specified screen or region for processing."""
        return self

    def __next__(self):
        """Captures and returns the next screenshot as a numpy array using the mss library."""
        im0 = np.asarray(self.sct.grab(self.monitor))[:, :, :3]
        s = f'screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: '
        self.frame += 1
        return [str(self.screen)], [im0], [s]


def get_best_youtube_url(url, method='pytube'):
    """
    Retrieves the URL of the best quality MP4 video stream from a given YouTube video.

    Args:
        url (str): The URL of the YouTube video.
        method (str): The method to use for extracting video info. Options are "pytube", "pafy", and "yt-dlp".
            Defaults to "pytube".

    Returns:
        (str | None): The URL of the best quality MP4 video stream, or None if no suitable stream is found.

    Examples:
        >>> url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        >>> best_url = get_best_youtube_url(url)
        >>> print(best_url)
        https://rr4---sn-q4flrnek.googlevideo.com/videoplayback?expire=...

    Notes:
        - Requires additional libraries based on the chosen method: pytubefix, pafy, or yt-dlp.
        - The function prioritizes streams with at least 1080p resolution when available.
        - For the "yt-dlp" method, it looks for formats with video codec, no audio, and *.mp4 extension.
    """
    if method == 'pytube':
        check_requirements('pytubefix>=6.5.2')
        streams = YouTube(url).streams.filter(file_extension='mp4', only_video=True)
        streams = sorted(streams, key=lambda s: s.resolution, reverse=True)
        for stream in streams:
            if stream.resolution and int(stream.resolution[:-1]) >= 1080:
                return stream.url
    elif method == 'pafy':
        check_requirements(('pafy', 'youtube_dl==2020.12.2'))
        return pafy.new(url).getbestvideo(preftype='mp4').url
    elif method == 'yt-dlp':
        check_requirements('yt-dlp')
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
        for f in reversed(info_dict.get('formats', [])):
            good_size = (f.get('width') or 0) >= 1920 or (f.get('height') or 0) >= 1080
            if good_size and f['vcodec'] != 'none' and f['acodec'] == 'none' and f['ext'] == 'mp4':
                return f.get('url')


class LoadStreams:
    """
    Stream Loader for various types of video streams.

    Supports RTSP, RTMP, HTTP, and TCP streams. This class handles the loading and processing of multiple video
    streams simultaneously, making it suitable for real-time video analysis tasks.

    Attributes:
        sources (List[str]): The source input paths or URLs for the video streams.
        vid_stride (int): Video frame-rate stride.
        buffer (bool): Whether to buffer input streams.
        running (bool): Flag to indicate if the streaming thread is running.
        mode (str): Set to 'stream' indicating real-time capture.
        imgs (List[List[np.ndarray]]): List of image frames for each stream.
        fps (List[float]): List of FPS for each stream.
        frames (List[int]): List of total frames for each stream.
        threads (List[Thread]): List of threads for each stream.
        shape (List[Tuple[int, int, int]]): List of shapes for each stream.
        caps (List[cv2.VideoCapture]): List of cv2.VideoCapture objects for each stream.
        bs (int): Batch size for processing.

    Methods:
        update: Read stream frames in daemon thread.
        close: Close stream loader and release resources.
        __iter__: Returns an iterator object for the class.
        __next__: Returns source paths, transformed, and original images for processing.
        __len__: Return the length of the sources object.

    Examples:
        >>> stream_loader = LoadStreams("rtsp://example.com/stream1.mp4")
        >>> for sources, imgs, _ in stream_loader:
        ...     # Process the images
        ...     pass
        >>> stream_loader.close()

    Notes:
        - The class uses threading to efficiently load frames from multiple streams simultaneously.
        - It automatically handles YouTube links, converting them to the best available stream URL.
        - The class implements a buffer system to manage frame storage and retrieval.
    """

    def __init__(self, sources='file.streams', vid_stride=1, buffer=False):
        """Initialize stream loader for multiple video sources, supporting various stream types."""
        torch.backends.cudnn.benchmark = True
        self.buffer = buffer
        self.running = True
        self.mode = 'stream'
        self.vid_stride = vid_stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.bs = n
        self.fps = [0] * n
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n
        self.imgs = [[] for _ in range(n)]
        self.shape = [[] for _ in range(n)]
        self.sources = [ops.clean_str(x) for x in sources]
        for i, s in enumerate(sources):
            st = f'{i + 1}/{n}: {s}... '
            if urlparse(s).hostname in {'www.youtube.com', 'youtube.com', 'youtu.be'}:
                s = get_best_youtube_url(s)
            s = eval(s) if s.isnumeric() else s
            if s == 0 and (IS_COLAB or IS_KAGGLE):
                raise NotImplementedError("'source=0' webcam not supported in Colab and Kaggle notebooks. Try running 'source=0' in a local environment.")
            self.caps[i] = cv2.VideoCapture(s)
            if not self.caps[i].isOpened():
                raise ConnectionError(f'{st}Failed to open {s}')
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30
            success, im = self.caps[i].read()
            if not success or im is None:
                raise ConnectionError(f'{st}Failed to read images from {s}')
            self.imgs[i].append(im)
            self.shape[i] = im.shape
            self.threads[i] = Thread(target=self.update, args=[i, self.caps[i], s], daemon=True)
            LOGGER.info(f'{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)')
            self.threads[i].start()
        LOGGER.info('')

    def update(self, i, cap, stream):
        """Read stream frames in daemon thread and update image buffer."""
        n, f = 0, self.frames[i]
        while self.running and cap.isOpened() and n < f - 1:
            if len(self.imgs[i]) < 30:
                n += 1
                cap.grab()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                        cap.open(stream)
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)

    def close(self):
        """Terminates stream loader, stops threads, and releases video capture resources."""
        self.running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)
        for cap in self.caps:
            try:
                cap.release()
            except Exception as e:
                LOGGER.warning(f'WARNING ⚠️ Could not release VideoCapture object: {e}')
        cv2.destroyAllWindows()

    def __iter__(self):
        """Iterates through YOLO image feed and re-opens unresponsive streams."""
        self.count = -1
        return self

    def __next__(self):
        """Returns the next batch of frames from multiple video streams for processing."""
        self.count += 1
        images = []
        for i, x in enumerate(self.imgs):
            while not x:
                if not self.threads[i].is_alive() or cv2.waitKey(1) == ord('q'):
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    LOGGER.warning(f'WARNING ⚠️ Waiting for stream {i}')
            if self.buffer:
                images.append(x.pop(0))
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()
        return self.sources, images, [''] * self.bs

    def __len__(self):
        """Return the number of video streams in the LoadStreams object."""
        return self.bs


class LoadTensor:
    """
    A class for loading and processing tensor data for object detection tasks.

    This class handles the loading and pre-processing of image data from PyTorch tensors, preparing them for
    further processing in object detection pipelines.

    Attributes:
        im0 (torch.Tensor): The input tensor containing the image(s) with shape (B, C, H, W).
        bs (int): Batch size, inferred from the shape of `im0`.
        mode (str): Current processing mode, set to 'image'.
        paths (List[str]): List of image paths or auto-generated filenames.

    Methods:
        _single_check: Validates and formats an input tensor.

    Examples:
        >>> import torch
        >>> tensor = torch.rand(1, 3, 640, 640)
        >>> loader = LoadTensor(tensor)
        >>> paths, images, info = next(iter(loader))
        >>> print(f"Processed {len(images)} images")
    """

    def __init__(self, im0) ->None:
        """Initialize LoadTensor object for processing torch.Tensor image data."""
        self.im0 = self._single_check(im0)
        self.bs = self.im0.shape[0]
        self.mode = 'image'
        self.paths = [getattr(im, 'filename', f'image{i}.jpg') for i, im in enumerate(im0)]

    @staticmethod
    def _single_check(im, stride=32):
        """Validates and formats a single image tensor, ensuring correct shape and normalization."""
        s = f'WARNING ⚠️ torch.Tensor inputs should be BCHW i.e. shape(1, 3, 640, 640) divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible.'
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            LOGGER.warning(s)
            im = im.unsqueeze(0)
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)
        if im.max() > 1.0 + torch.finfo(im.dtype).eps:
            LOGGER.warning(f'WARNING ⚠️ torch.Tensor inputs should be normalized 0.0-1.0 but max value is {im.max()}. Dividing input by 255.')
            im = im.float() / 255.0
        return im

    def __iter__(self):
        """Yields an iterator object for iterating through tensor image data."""
        self.count = 0
        return self

    def __next__(self):
        """Yields the next batch of tensor images and metadata for processing."""
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [''] * self.bs

    def __len__(self):
        """Returns the batch size of the tensor input."""
        return self.bs


LOADERS = LoadStreams, LoadPilAndNumpy, LoadImagesAndVideos, LoadScreenshots


def autocast_list(source):
    """Merges a list of sources into a list of numpy arrays or PIL images for Ultralytics prediction."""
    files = []
    for im in source:
        if isinstance(im, (str, Path)):
            files.append(Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im))
        elif isinstance(im, (Image.Image, np.ndarray)):
            files.append(im)
        else:
            raise TypeError(f'type {type(im).__name__} is not a supported Ultralytics prediction source type. \nSee https://docs.ultralytics.com/modes/predict for supported source types.')
    return files


def check_source(source):
    """Check source type and return corresponding flag values."""
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):
        source = str(source)
        is_file = Path(source).suffix[1:] in IMG_FORMATS | VID_FORMATS
        is_url = source.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://', 'tcp://'))
        webcam = source.isnumeric() or source.endswith('.streams') or is_url and not is_file
        screenshot = source.lower() == 'screen'
        if is_url and is_file:
            source = check_file(source)
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError('Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict')
    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False):
    """
    Loads an inference source for object detection and applies necessary transformations.

    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference.
        batch (int, optional): Batch size for dataloaders. Default is 1.
        vid_stride (int, optional): The frame interval for video sources. Default is 1.
        buffer (bool, optional): Determined whether stream frames will be buffered. Default is False.

    Returns:
        dataset (Dataset): A dataset object for the specified input source.
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
    elif screenshot:
        dataset = LoadScreenshots(source)
    elif from_img:
        dataset = LoadPilAndNumpy(source)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride)
    setattr(dataset, 'source_type', source_type)
    return dataset


class BaseValidator:
    """
    BaseValidator.

    A base class for creating validators.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names.
        seen: Records the number of images seen so far during validation.
        stats: Placeholder for statistics during validation.
        confusion_matrix: Placeholder for a confusion matrix.
        nc: Number of classes.
        iouv: (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (dict): Dictionary to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective
                      batch processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (dict): Dictionary to store various callback functions.
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}
        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)
        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Executes validation process, running inference on dataloader and computing performance metrics."""
        self.training = trainer is not None
        augment = self.args.augment and not self.training
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != 'cpu' and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or trainer.epoch == trainer.epochs - 1
            model.eval()
        else:
            if str(self.args.model).endswith('.yaml'):
                LOGGER.warning('WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.')
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(weights=model or self.args.model, device=select_device(self.args.device, self.args.batch), dnn=self.args.dnn, data=self.args.data, fp16=self.args.half)
            self.device = model.device
            self.args.half = model.fp16
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get('batch', 1)
                LOGGER.info(f'Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})')
            if str(self.args.data).split('.')[-1] in {'yaml', 'yml'}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))
            if self.device.type in {'cpu', 'mps'}:
                self.args.workers = 0
            if not pt:
                self.args.rect = False
            self.stride = model.stride
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))
        self.run_callbacks('on_val_start')
        dt = Profile(device=self.device), Profile(device=self.device), Profile(device=self.device), Profile(device=self.device)
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []
        for batch_i, batch in enumerate(bar):
            self.run_callbacks('on_val_batch_start')
            self.batch_i = batch_i
            with dt[0]:
                batch = self.preprocess(batch)
            with dt[1]:
                preds = model(batch['img'], augment=augment)
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]
            with dt[3]:
                preds = self.postprocess(preds)
            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
            self.run_callbacks('on_val_batch_end')
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1000.0 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks('on_val_end')
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
            return {k: round(float(v), 5) for k, v in results.items()}
        else:
            LOGGER.info('Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image'.format(*tuple(self.speed.values())))
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                    LOGGER.info(f'Saving {f.name}...')
                    json.dump(self.jdict, f)
                stats = self.eval_json(stats)
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                import scipy
                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def add_callback(self, event: 'str', callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: 'str'):
        """Runs all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError('get_dataloader function not implemented for this validator')

    def build_dataset(self, img_path):
        """Build dataset."""
        raise NotImplementedError('build_dataset function not implemented in validator')

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        return batch

    def postprocess(self, preds):
        """Preprocesses the predictions."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        pass

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes and returns all metrics."""
        pass

    def get_stats(self):
        """Returns statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Checks statistics."""
        pass

    def print_results(self):
        """Prints the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Returns the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        self.plots[Path(name)] = {'data': data, 'timestamp': time.time()}

    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plots YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(obb1, obb2, eps=1e-07):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))
    t1 = ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps) * 0.25
    t2 = (c1 + c2) * (x2 - x1) * (y1 - y2) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps) * 0.5
    t3 = (((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps) + eps).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def box_iou(box1, box2, eps=1e-07):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def plt_settings(rcparams=None, backend='Agg'):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Example:
        decorator: @plt_settings({"font.size": 12})
        context manager: with plt_settings({"font.size": 12}):

    Args:
        rcparams (dict): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend. This decorator can be
            applied to any function that needs to have specific matplotlib rc parameters and backend for its execution.
    """
    if rcparams is None:
        rcparams = {'font.size': 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Sets rc parameters and backend, calls the original function, and restores the settings."""
            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close('all')
                plt.switch_backend(backend)
            try:
                with plt.rc_context(rcparams):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close('all')
                    plt.switch_backend(original_backend)
            return result
        return wrapper
    return decorator


class Metric(SimpleClass):
    """
    Class for computing evaluation metrics for YOLOv8 model.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.

    Methods:
        ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        mp(): Mean precision of all classes. Returns: Float.
        mr(): Mean recall of all classes. Returns: Float.
        map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
        map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
        map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
        mean_results(): Mean of results, returns mp, mr, map50, map.
        class_result(i): Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
        fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
        update(results): Update metric attributes with new evaluation results.
    """

    def __init__(self) ->None:
        """Initializes a Metric instance for computing evaluation metrics for the YOLOv8 model."""
        self.p = []
        self.r = []
        self.f1 = []
        self.all_ap = []
        self.ap_class_index = []
        self.nc = 0

    @property
    def ap50(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """
        Returns the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """
        Returns the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """
        Returns the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i):
        """Class-aware result, return p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        """MAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        """Model fitness as a weighted combination of metrics."""
        w = [0.0, 0.0, 0.1, 0.9]
        return (np.array(self.mean_results()) * w).sum()

    def update(self, results):
        """
        Updates the evaluation metrics of the model with a new set of results.

        Args:
            results (tuple): A tuple containing the following evaluation metrics:
                - p (list): Precision for each class. Shape: (nc,).
                - r (list): Recall for each class. Shape: (nc,).
                - f1 (list): F1 score for each class. Shape: (nc,).
                - all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
                - ap_class_index (list): Index of class for each AP score. Shape: (nc,).

        Side Effects:
            Updates the class attributes `self.p`, `self.r`, `self.f1`, `self.all_ap`, and `self.ap_class_index` based
            on the values provided in the `results` tuple.
        """
        self.p, self.r, self.f1, self.all_ap, self.ap_class_index, self.p_curve, self.r_curve, self.f1_curve, self.px, self.prec_values = results

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [[self.px, self.prec_values, 'Recall', 'Precision'], [self.px, self.f1_curve, 'Confidence', 'F1'], [self.px, self.p_curve, 'Confidence', 'Precision'], [self.px, self.r_curve, 'Confidence', 'Recall']]


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    method = 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, mpre, mrec


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')


class RandomLoadText:
    """
    Randomly samples positive and negative texts and updates class indices accordingly.

    This class is responsible for sampling texts from a given set of class texts, including both positive
    (present in the image) and negative (not present in the image) samples. It updates the class indices
    to reflect the sampled texts and can optionally pad the text list to a fixed length.

    Attributes:
        prompt_format (str): Format string for text prompts.
        neg_samples (Tuple[int, int]): Range for randomly sampling negative texts.
        max_samples (int): Maximum number of different text samples in one image.
        padding (bool): Whether to pad texts to max_samples.
        padding_value (str): The text used for padding when padding is True.

    Methods:
        __call__: Processes the input labels and returns updated classes and texts.

    Examples:
        >>> loader = RandomLoadText(prompt_format="Object: {}", neg_samples=(5, 10), max_samples=20)
        >>> labels = {"cls": [0, 1, 2], "texts": [["cat"], ["dog"], ["bird"]], "instances": [...]}
        >>> updated_labels = loader(labels)
        >>> print(updated_labels["texts"])
        ['Object: cat', 'Object: dog', 'Object: bird', 'Object: elephant', 'Object: car']
    """

    def __init__(self, prompt_format: 'str'='{}', neg_samples: 'Tuple[int, int]'=(80, 80), max_samples: 'int'=80, padding: 'bool'=False, padding_value: 'str'='') ->None:
        """
        Initializes the RandomLoadText class for randomly sampling positive and negative texts.

        This class is designed to randomly sample positive texts and negative texts, and update the class
        indices accordingly to the number of samples. It can be used for text-based object detection tasks.

        Args:
            prompt_format (str): Format string for the prompt. Default is '{}'. The format string should
                contain a single pair of curly braces {} where the text will be inserted.
            neg_samples (Tuple[int, int]): A range to randomly sample negative texts. The first integer
                specifies the minimum number of negative samples, and the second integer specifies the
                maximum. Default is (80, 80).
            max_samples (int): The maximum number of different text samples in one image. Default is 80.
            padding (bool): Whether to pad texts to max_samples. If True, the number of texts will always
                be equal to max_samples. Default is False.
            padding_value (str): The padding text to use when padding is True. Default is an empty string.

        Attributes:
            prompt_format (str): The format string for the prompt.
            neg_samples (Tuple[int, int]): The range for sampling negative texts.
            max_samples (int): The maximum number of text samples.
            padding (bool): Whether padding is enabled.
            padding_value (str): The value used for padding.

        Examples:
            >>> random_load_text = RandomLoadText(prompt_format="Object: {}", neg_samples=(50, 100), max_samples=120)
            >>> random_load_text.prompt_format
            'Object: {}'
            >>> random_load_text.neg_samples
            (50, 100)
            >>> random_load_text.max_samples
            120
        """
        self.prompt_format = prompt_format
        self.neg_samples = neg_samples
        self.max_samples = max_samples
        self.padding = padding
        self.padding_value = padding_value

    def __call__(self, labels: 'dict') ->dict:
        """
        Randomly samples positive and negative texts and updates class indices accordingly.

        This method samples positive texts based on the existing class labels in the image, and randomly
        selects negative texts from the remaining classes. It then updates the class indices to match the
        new sampled text order.

        Args:
            labels (Dict): A dictionary containing image labels and metadata. Must include 'texts' and 'cls' keys.

        Returns:
            (Dict): Updated labels dictionary with new 'cls' and 'texts' entries.

        Examples:
            >>> loader = RandomLoadText(prompt_format="A photo of {}", neg_samples=(5, 10), max_samples=20)
            >>> labels = {"cls": np.array([[0], [1], [2]]), "texts": [["dog"], ["cat"], ["bird"]]}
            >>> updated_labels = loader(labels)
        """
        assert 'texts' in labels, 'No texts found in labels.'
        class_texts = labels['texts']
        num_classes = len(class_texts)
        cls = np.asarray(labels.pop('cls'), dtype=int)
        pos_labels = np.unique(cls).tolist()
        if len(pos_labels) > self.max_samples:
            pos_labels = random.sample(pos_labels, k=self.max_samples)
        neg_samples = min(min(num_classes, self.max_samples) - len(pos_labels), random.randint(*self.neg_samples))
        neg_labels = [i for i in range(num_classes) if i not in pos_labels]
        neg_labels = random.sample(neg_labels, k=neg_samples)
        sampled_labels = pos_labels + neg_labels
        random.shuffle(sampled_labels)
        label2ids = {label: i for i, label in enumerate(sampled_labels)}
        valid_idx = np.zeros(len(labels['instances']), dtype=bool)
        new_cls = []
        for i, label in enumerate(cls.squeeze(-1).tolist()):
            if label not in label2ids:
                continue
            valid_idx[i] = True
            new_cls.append([label2ids[label]])
        labels['instances'] = labels['instances'][valid_idx]
        labels['cls'] = np.array(new_cls)
        texts = []
        for label in sampled_labels:
            prompts = class_texts[label]
            assert len(prompts) > 0
            prompt = self.prompt_format.format(prompts[random.randrange(len(prompts))])
            texts.append(prompt)
        if self.padding:
            valid_labels = len(pos_labels) + len(neg_labels)
            num_padding = self.max_samples - valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding
        labels['texts'] = texts
        return labels


def build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32, multi_modal=False):
    """Build YOLO Dataset."""
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    return dataset(img_path=img_path, imgsz=cfg.imgsz, batch_size=batch, augment=mode == 'train', hyp=cfg, rect=cfg.rect or rect, cache=cfg.cache or None, single_cls=cfg.single_cls or False, stride=int(stride), pad=0.0 if mode == 'train' else 0.5, prefix=colorstr(f'{mode}: '), task=cfg.task, classes=cfg.classes, data=data, fraction=cfg.fraction if mode == 'train' else 1.0)


def output_to_target(output, max_det=300):
    """Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting."""
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, ops.xyxy2xywh(box), conf), 1))
    targets = torch.cat(targets, 0).numpy()
    return targets[:, 0], targets[:, 1], targets[:, 2:-1], targets[:, -1]


def threaded(func):
    """
    Multi-threads a target function by default and returns the thread or function result.

    Use as @threaded decorator. The function runs in a separate thread unless 'threaded=False' is passed.
    """

    def wrapper(*args, **kwargs):
        """Multi-threads a given function based on 'threaded' kwarg and returns the thread or function result."""
        if kwargs.pop('threaded', True):
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)
    return wrapper


@threaded
def plot_images(images: 'Union[torch.Tensor, np.ndarray]', batch_idx: 'Union[torch.Tensor, np.ndarray]', cls: 'Union[torch.Tensor, np.ndarray]', bboxes: 'Union[torch.Tensor, np.ndarray]'=np.zeros(0, dtype=np.float32), confs: 'Optional[Union[torch.Tensor, np.ndarray]]'=None, masks: 'Union[torch.Tensor, np.ndarray]'=np.zeros(0, dtype=np.uint8), kpts: 'Union[torch.Tensor, np.ndarray]'=np.zeros((0, 51), dtype=np.float32), paths: 'Optional[List[str]]'=None, fname: 'str'='images.jpg', names: 'Optional[Dict[int, str]]'=None, on_plot: 'Optional[Callable]'=None, max_size: 'int'=1920, max_subplots: 'int'=16, save: 'bool'=True, conf_thres: 'float'=0.25) ->Optional[np.ndarray]:
    """
    Plot image grid with labels, bounding boxes, masks, and keypoints.

    Args:
        images: Batch of images to plot. Shape: (batch_size, channels, height, width).
        batch_idx: Batch indices for each detection. Shape: (num_detections,).
        cls: Class labels for each detection. Shape: (num_detections,).
        bboxes: Bounding boxes for each detection. Shape: (num_detections, 4) or (num_detections, 5) for rotated boxes.
        confs: Confidence scores for each detection. Shape: (num_detections,).
        masks: Instance segmentation masks. Shape: (num_detections, height, width) or (1, height, width).
        kpts: Keypoints for each detection. Shape: (num_detections, 51).
        paths: List of file paths for each image in the batch.
        fname: Output filename for the plotted image grid.
        names: Dictionary mapping class indices to class names.
        on_plot: Optional callback function to be called after saving the plot.
        max_size: Maximum size of the output image grid.
        max_subplots: Maximum number of subplots in the image grid.
        save: Whether to save the plotted image grid to a file.
        conf_thres: Confidence threshold for displaying detections.

    Returns:
        np.ndarray: Plotted image grid as a numpy array if save is False, None otherwise.

    Note:
        This function supports both tensor and numpy array inputs. It will automatically
        convert tensor inputs to numpy arrays for processing.
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs ** 0.5)
    if np.max(images[0]) <= 1:
        images *= 255
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        mosaic[y:y + h, x:x + w, :] = images[i].transpose(1, 2, 0)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))
    fs = int((h + w) * ns * 0.01)
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype('int')
            labels = confs is None
            if len(bboxes):
                boxes = bboxes[idx]
                conf = confs[idx] if confs is not None else None
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:
                        boxes[..., [0, 2]] *= w
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:
                        boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                is_obb = boxes.shape[-1] == 5
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = f'{c}' if labels else f'{c} {conf[j]:.1f}'
                        annotator.box_label(box, label, color=color, rotated=is_obb)
            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    annotator.text((x, y), f'{c}', txt_color=color, box_style=True)
            if len(kpts):
                kpts_ = kpts[idx].copy()
                if len(kpts_):
                    if kpts_[..., 0].max() <= 1.01 or kpts_[..., 1].max() <= 1.01:
                        kpts_[..., 0] *= w
                        kpts_[..., 1] *= h
                    elif scale < 1:
                        kpts_ *= scale
                kpts_[..., 0] += x
                kpts_[..., 1] += y
                for j in range(len(kpts_)):
                    if labels or conf[j] > conf_thres:
                        annotator.kpts(kpts_[j], conf_thres=conf_thres)
            if len(masks):
                if idx.shape[0] == masks.shape[0]:
                    image_masks = masks[idx]
                else:
                    image_masks = masks[[i]]
                    nl = idx.sum()
                    index = np.arange(nl).reshape((nl, 1, 1)) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)
                im = np.asarray(annotator.im).copy()
                for j in range(len(image_masks)):
                    if labels or conf[j] > conf_thres:
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        try:
                            im[y:y + h, x:x + w, :][mask] = im[y:y + h, x:x + w, :][mask] * 0.4 + np.array(color) * 0.6
                        except Exception:
                            pass
                annotator.fromarray(im)
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)
    if on_plot:
        on_plot(fname)


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model="yolo11n.pt", data="coco8.yaml")
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = 'detect'
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()
        self.lb = []
        if self.args.save_hybrid:
            LOGGER.warning("WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\nWARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n")

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch['img'] = batch['img']
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k]
        if self.args.save_hybrid:
            height, width = batch['img'].shape[2:]
            nb = len(batch['img'])
            bboxes = batch['bboxes'] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [torch.cat([batch['cls'][batch['batch_idx'] == i], bboxes[batch['batch_idx'] == i]], dim=-1) for i in range(nb)]
        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, '')
        self.is_coco = isinstance(val, str) and 'coco' in val and (val.endswith(f'{os.sep}val2017.txt') or val.endswith(f'{os.sep}test-dev2017.txt'))
        self.is_lvis = isinstance(val, str) and 'lvis' in val and not self.is_coco
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(len(model.names)))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)')

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(preds, self.args.conf, self.args.iou, labels=self.lb, multi_label=True, agnostic=self.args.single_cls or self.args.agnostic_nms, max_det=self.args.max_det)

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch['batch_idx'] == si
        cls = batch['cls'][idx].squeeze(-1)
        bbox = batch['bboxes'][idx]
        ori_shape = batch['ori_shape'][si]
        imgsz = batch['img'].shape[2:]
        ratio_pad = batch['ratio_pad'][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)
        return {'cls': cls, 'bbox': bbox, 'ori_shape': ori_shape, 'imgsz': imgsz, 'ratio_pad': ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(pbatch['imgsz'], predn[:, :4], pbatch['ori_shape'], ratio_pad=pbatch['ratio_pad'])
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(conf=torch.zeros(0, device=self.device), pred_cls=torch.zeros(0, device=self.device), tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device))
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop('cls'), pbatch.pop('bbox')
            nl = len(cls)
            stat['target_cls'] = cls
            stat['target_img'] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat['conf'] = predn[:, 4]
            stat['pred_cls'] = predn[:, 5]
            if nl:
                stat['tp'] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                self.save_one_txt(predn, self.args.save_conf, pbatch['ori_shape'], self.save_dir / 'labels' / f"{Path(batch['im_file'][si]).stem}.txt")

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}
        self.nt_per_class = np.bincount(stats['target_cls'].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats['target_img'].astype(int), minlength=self.nc)
        stats.pop('target_img', None)
        if len(stats) and stats['tp'].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics.keys)
        LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f'WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels')
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i)))
        if self.args.plots:
            for normalize in (True, False):
                self.confusion_matrix.plot(save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot)

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode='val', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode='val')
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(batch['img'], batch['batch_idx'], batch['cls'].squeeze(-1), batch['bboxes'], paths=batch['im_file'], fname=self.save_dir / f'val_batch{ni}_labels.jpg', names=self.names, on_plot=self.on_plot)

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(batch['img'], *output_to_target(preds, max_det=self.args.max_det), paths=batch['im_file'], fname=self.save_dir / f'val_batch{ni}_pred.jpg', names=self.names, on_plot=self.on_plot)

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        Results(np.zeros((shape[0], shape[1]), dtype=np.uint8), path=None, names=self.names, boxes=predn[:, :6]).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append({'image_id': image_id, 'category_id': self.class_map[int(p[5])] + (1 if self.is_lvis else 0), 'bbox': [round(x, 3) for x in b], 'score': round(p[4], 5)})

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / 'predictions.json'
            anno_json = self.data['path'] / 'annotations' / ('instances_val2017.json' if self.is_coco else f'lvis_v1_{self.args.split}.json')
            pkg = 'pycocotools' if self.is_coco else 'lvis'
            LOGGER.info(f'\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...')
            try:
                for x in (pred_json, anno_json):
                    assert x.is_file(), f'{x} file not found'
                check_requirements('pycocotools>=2.0.6' if self.is_coco else 'lvis>=0.5.3')
                if self.is_coco:
                    anno = COCO(str(anno_json))
                    pred = anno.loadRes(str(pred_json))
                    val = COCOeval(anno, pred, 'bbox')
                else:
                    anno = LVIS(str(anno_json))
                    pred = anno._load_json(str(pred_json))
                    val = LVISEval(anno, pred, 'bbox')
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = val.stats[:2] if self.is_coco else [val.results['AP50'], val.results['AP']]
            except Exception as e:
                LOGGER.warning(f'{pkg} unable to run: {e}')
        return stats


class NASValidator(DetectionValidator):
    """
    Ultralytics YOLO NAS Validator for object detection.

    Extends `DetectionValidator` from the Ultralytics models package and is designed to post-process the raw predictions
    generated by YOLO NAS models. It performs non-maximum suppression to remove overlapping and low-confidence boxes,
    ultimately producing the final detections.

    Attributes:
        args (Namespace): Namespace containing various configurations for post-processing, such as confidence and IoU.
        lb (torch.Tensor): Optional tensor for multilabel NMS.

    Example:
        ```python
        from ultralytics import NAS

        model = NAS("yolo_nas_s")
        validator = model.validator
        # Assumes that raw_preds are available
        final_preds = validator.postprocess(raw_preds)
        ```

    Note:
        This class is generally not instantiated directly but is used internally within the `NAS` class.
    """

    def postprocess(self, preds_in):
        """Apply Non-maximum suppression to prediction outputs."""
        boxes = ops.xyxy2xywh(preds_in[0][0])
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)
        return ops.non_max_suppression(preds, self.args.conf, self.args.iou, labels=self.lb, multi_label=False, agnostic=self.args.single_cls or self.args.agnostic_nms, max_det=self.args.max_det, max_time_img=0.5)


def batch_iterator(batch_size: 'int', *args) ->Generator[List[Any], None, None]:
    """Yields batches of data from input arguments with specified batch size for efficient processing."""
    assert args and all(len(a) == len(args[0]) for a in args), 'Batched iteration must have same-size inputs.'
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size:(b + 1) * batch_size] for arg in args]


def batched_mask_to_box(masks: 'torch.Tensor') ->torch.Tensor:
    """Calculates bounding boxes in XYXY format around binary masks, handling empty masks and various input shapes."""
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)
    shape = masks.shape
    h, w = shape[-2:]
    masks = masks.flatten(0, -3) if len(shape) > 2 else masks.unsqueeze(0)
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * ~in_height
    top_edges, _ = torch.min(in_height_coords, dim=-1)
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * ~in_width
    left_edges, _ = torch.min(in_width_coords, dim=-1)
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)
    return out.reshape(*shape[:-2], 4) if len(shape) > 2 else out[0]


def build_point_grid(n_per_side: 'int') ->np.ndarray:
    """Generate a 2D grid of evenly spaced points in the range [0,1]x[0,1] for image segmentation tasks."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    return np.stack([points_x, points_y], axis=-1).reshape(-1, 2)


def build_all_layer_point_grids(n_per_side: 'int', n_layers: 'int', scale_per_layer: 'int') ->List[np.ndarray]:
    """Generates point grids for multiple crop layers with varying scales and densities."""
    return [build_point_grid(int(n_per_side / scale_per_layer ** i)) for i in range(n_layers + 1)]


class MLPBlock(nn.Module):
    """Implements a single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim, mlp_dim, act=nn.GELU):
        """Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function."""
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward pass for the MLPBlock."""
        return self.lin2(self.act(self.lin1(x)))


def get_rel_pos(q_size: 'int', k_size: 'int', rel_pos: 'torch.Tensor') ->torch.Tensor:
    """
    Extracts relative positional embeddings based on query and key sizes.

    Args:
        q_size (int): Size of the query.
        k_size (int): Size of the key.
        rel_pos (torch.Tensor): Relative position embeddings with shape (L, C), where L is the maximum relative
            distance and C is the embedding dimension.

    Returns:
        (torch.Tensor): Extracted positional embeddings according to relative positions, with shape (q_size,
            k_size, C).

    Examples:
        >>> q_size, k_size = 8, 16
        >>> rel_pos = torch.randn(31, 64)  # 31 = 2 * max(8, 16) - 1
        >>> extracted_pos = get_rel_pos(q_size, k_size, rel_pos)
        >>> print(extracted_pos.shape)
        torch.Size([8, 16, 64])
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1), size=max_rel_dist, mode='linear')
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = q_coords - k_coords + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn: 'torch.Tensor', q: 'torch.Tensor', rel_pos_h: 'torch.Tensor', rel_pos_w: 'torch.Tensor', q_size: 'Tuple[int, int]', k_size: 'Tuple[int, int]') ->torch.Tensor:
    """
    Adds decomposed Relative Positional Embeddings to the attention map.

    This function calculates and applies decomposed Relative Positional Embeddings as described in the MVITv2
    paper. It enhances the attention mechanism by incorporating spatial relationships between query and key
    positions.

    Args:
        attn (torch.Tensor): Attention map with shape (B, q_h * q_w, k_h * k_w).
        q (torch.Tensor): Query tensor in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (torch.Tensor): Relative position embeddings for height axis with shape (Lh, C).
        rel_pos_w (torch.Tensor): Relative position embeddings for width axis with shape (Lw, C).
        q_size (Tuple[int, int]): Spatial sequence size of query q as (q_h, q_w).
        k_size (Tuple[int, int]): Spatial sequence size of key k as (k_h, k_w).

    Returns:
        (torch.Tensor): Updated attention map with added relative positional embeddings, shape
            (B, q_h * q_w, k_h * k_w).

    Examples:
        >>> B, C, q_h, q_w, k_h, k_w = 1, 64, 8, 8, 8, 8
        >>> attn = torch.rand(B, q_h * q_w, k_h * k_w)
        >>> q = torch.rand(B, q_h * q_w, C)
        >>> rel_pos_h = torch.rand(2 * max(q_h, k_h) - 1, C)
        >>> rel_pos_w = torch.rand(2 * max(q_w, k_w) - 1, C)
        >>> q_size, k_size = (q_h, q_w), (k_h, k_w)
        >>> updated_attn = add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size)
        >>> print(updated_attn.shape)
        torch.Size([1, 64, 64])

    References:
        https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum('bhwc,hkc->bhwk', r_q, Rh)
    rel_w = torch.einsum('bhwc,wkc->bhwk', r_q, Rw)
    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)
    return attn


class REAttention(nn.Module):
    """
    Rotary Embedding Attention module for efficient self-attention in transformer architectures.

    This class implements a multi-head attention mechanism with rotary positional embeddings, designed
    for use in vision transformer models. It supports optional query pooling and window partitioning
    for efficient processing of large inputs.

    Attributes:
        compute_cis (Callable): Function to compute axial complex numbers for rotary encoding.
        freqs_cis (Tensor): Precomputed frequency tensor for rotary encoding.
        rope_k_repeat (bool): Flag to repeat query RoPE to match key length for cross-attention to memories.
        q_proj (nn.Linear): Linear projection for query.
        k_proj (nn.Linear): Linear projection for key.
        v_proj (nn.Linear): Linear projection for value.
        out_proj (nn.Linear): Output projection.
        num_heads (int): Number of attention heads.
        internal_dim (int): Internal dimension for attention computation.

    Methods:
        forward: Applies rotary position encoding and computes attention between query, key, and value tensors.

    Examples:
        >>> rope_attn = REAttention(embedding_dim=256, num_heads=8, rope_theta=10000.0, feat_sizes=(32, 32))
        >>> q = torch.randn(1, 1024, 256)
        >>> k = torch.randn(1, 1024, 256)
        >>> v = torch.randn(1, 1024, 256)
        >>> output = rope_attn(q, k, v)
        >>> print(output.shape)
        torch.Size([1, 1024, 256])
    """

    def __init__(self, dim: 'int', num_heads: 'int'=8, qkv_bias: 'bool'=True, use_rel_pos: 'bool'=False, rel_pos_zero_init: 'bool'=True, input_size: 'Optional[Tuple[int, int]]'=None) ->None:
        """
        Initializes a Relative Position Attention module for transformer-based architectures.

        This module implements multi-head attention with optional relative positional encodings, designed
        specifically for vision tasks in transformer models.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads. Default is 8.
            qkv_bias (bool): If True, adds a learnable bias to query, key, value projections. Default is True.
            use_rel_pos (bool): If True, uses relative positional encodings. Default is False.
            rel_pos_zero_init (bool): If True, initializes relative positional parameters to zero. Default is True.
            input_size (Tuple[int, int] | None): Input resolution for calculating relative positional parameter size.
                Required if use_rel_pos is True. Default is None.

        Examples:
            >>> attention = REAttention(dim=256, num_heads=8, input_size=(32, 32))
            >>> x = torch.randn(1, 32, 32, 256)
            >>> output = attention(x)
            >>> print(output.shape)
            torch.Size([1, 32, 32, 256])
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, 'Input size must be provided if using relative positional encoding.'
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Applies multi-head attention with optional relative positional encoding to input tensor."""
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = q * self.scale @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)


def window_partition(x, window_size):
    """
    Partitions input tensor into non-overlapping windows with padding if needed.

    Args:
        x (torch.Tensor): Input tensor with shape (B, H, W, C).
        window_size (int): Size of each window.

    Returns:
        (Tuple[torch.Tensor, Tuple[int, int]]): A tuple containing:
            - windows (torch.Tensor): Partitioned windows with shape (B * num_windows, window_size, window_size, C).
            - (Hp, Wp) (Tuple[int, int]): Padded height and width before partition.

    Examples:
        >>> x = torch.randn(1, 16, 16, 3)
        >>> windows, (Hp, Wp) = window_partition(x, window_size=4)
        >>> print(windows.shape, Hp, Wp)
        torch.Size([16, 4, 4, 3]) 16 16
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Unpartitions windowed sequences into original sequences and removes padding.

    This function reverses the windowing process, reconstructing the original input from windowed segments
    and removing any padding that was added during the windowing process.

    Args:
        windows (torch.Tensor): Input tensor of windowed sequences with shape (B * num_windows, window_size,
            window_size, C), where B is the batch size, num_windows is the number of windows, window_size is
            the size of each window, and C is the number of channels.
        window_size (int): Size of each window.
        pad_hw (Tuple[int, int]): Padded height and width (Hp, Wp) of the input before windowing.
        hw (Tuple[int, int]): Original height and width (H, W) of the input before padding and windowing.

    Returns:
        (torch.Tensor): Unpartitioned sequences with shape (B, H, W, C), where B is the batch size, H and W
            are the original height and width, and C is the number of channels.

    Examples:
        >>> windows = torch.rand(32, 8, 8, 64)  # 32 windows of size 8x8 with 64 channels
        >>> pad_hw = (16, 16)  # Padded height and width
        >>> hw = (15, 14)  # Original height and width
        >>> x = window_unpartition(windows, window_size=8, pad_hw=pad_hw, hw=hw)
        >>> print(x.shape)
        torch.Size([1, 15, 14, 64])
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class Block(nn.Module):
    """
    Transformer block with support for window attention and residual propagation.

    This class implements a transformer block that can use either global or windowed self-attention,
    followed by a feed-forward network. It supports relative positional embeddings and is designed
    for use in vision transformer architectures.

    Attributes:
        norm1 (nn.Module): First normalization layer.
        attn (REAttention): Self-attention layer with optional relative positional encoding.
        norm2 (nn.Module): Second normalization layer.
        mlp (MLPBlock): Multi-layer perceptron block.
        window_size (int): Size of attention window. If 0, global attention is used.

    Methods:
        forward: Processes input through the transformer block.

    Examples:
        >>> import torch
        >>> block = Block(dim=256, num_heads=8, window_size=7)
        >>> x = torch.randn(1, 56, 56, 256)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 56, 56, 256])
    """

    def __init__(self, dim: 'int', num_heads: 'int', mlp_ratio: 'float'=4.0, qkv_bias: 'bool'=True, norm_layer: 'Type[nn.Module]'=nn.LayerNorm, act_layer: 'Type[nn.Module]'=nn.GELU, use_rel_pos: 'bool'=False, rel_pos_zero_init: 'bool'=True, window_size: 'int'=0, input_size: 'Optional[Tuple[int, int]]'=None) ->None:
        """
        Initializes a transformer block with optional window attention and relative positional embeddings.

        This constructor sets up a transformer block that can use either global or windowed self-attention,
        followed by a feed-forward network. It supports relative positional embeddings and is designed
        for use in vision transformer architectures.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in the self-attention layer.
            mlp_ratio (float): Ratio of mlp hidden dimension to embedding dimension.
            qkv_bias (bool): If True, adds a learnable bias to query, key, value projections.
            norm_layer (Type[nn.Module]): Type of normalization layer to use.
            act_layer (Type[nn.Module]): Type of activation function to use in the MLP block.
            use_rel_pos (bool): If True, uses relative positional embeddings in attention.
            rel_pos_zero_init (bool): If True, initializes relative positional parameters to zero.
            window_size (int): Size of attention window. If 0, uses global attention.
            input_size (Optional[Tuple[int, int]]): Input resolution for calculating relative positional parameter size.

        Examples:
            >>> block = Block(dim=256, num_heads=8, window_size=7)
            >>> x = torch.randn(1, 56, 56, 256)
            >>> output = block(x)
            >>> print(output.shape)
            torch.Size([1, 56, 56, 256])
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = REAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init, input_size=input_size if window_size == 0 else (window_size, window_size))
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Processes input through transformer block with optional windowed self-attention and residual connection."""
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.

    Original implementations in
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    and
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py.
    """

    def __init__(self, num_channels, eps=1e-06):
        """Initialize LayerNorm2d with the given parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """Perform forward pass for 2D layer normalization."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Conv2d_BN(torch.nn.Sequential):
    """
    A sequential container that performs 2D convolution followed by batch normalization.

    Attributes:
        c (torch.nn.Conv2d): 2D convolution layer.
        1 (torch.nn.BatchNorm2d): Batch normalization layer.

    Methods:
        __init__: Initializes the Conv2d_BN with specified parameters.

    Args:
        a (int): Number of input channels.
        b (int): Number of output channels.
        ks (int): Kernel size for the convolution. Defaults to 1.
        stride (int): Stride for the convolution. Defaults to 1.
        pad (int): Padding for the convolution. Defaults to 0.
        dilation (int): Dilation factor for the convolution. Defaults to 1.
        groups (int): Number of groups for the convolution. Defaults to 1.
        bn_weight_init (float): Initial value for batch normalization weight. Defaults to 1.

    Examples:
        >>> conv_bn = Conv2d_BN(3, 64, ks=3, stride=1, pad=1)
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output = conv_bn(input_tensor)
        >>> print(output.shape)
    """

    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        """Initializes a sequential container with 2D convolution followed by batch normalization."""
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """
    Embeds images into patches and projects them into a specified embedding dimension.

    Attributes:
        patches_resolution (Tuple[int, int]): Resolution of the patches after embedding.
        num_patches (int): Total number of patches.
        in_chans (int): Number of input channels.
        embed_dim (int): Dimension of the embedding.
        seq (nn.Sequential): Sequence of convolutional and activation layers for patch embedding.

    Methods:
        forward: Processes the input tensor through the patch embedding sequence.

    Examples:
        >>> import torch
        >>> patch_embed = PatchEmbed(in_chans=3, embed_dim=96, resolution=224, activation=nn.GELU)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = patch_embed(x)
        >>> print(output.shape)
    """

    def __init__(self, in_chans, embed_dim, resolution, activation):
        """Initializes patch embedding with convolutional layers for image-to-patch conversion and projection."""
        super().__init__()
        img_size: 'Tuple[int, int]' = to_2tuple(resolution)
        self.patches_resolution = img_size[0] // 4, img_size[1] // 4
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(Conv2d_BN(in_chans, n // 2, 3, 2, 1), activation(), Conv2d_BN(n // 2, n, 3, 2, 1))

    def forward(self, x):
        """Processes input tensor through patch embedding sequence, converting images to patch embeddings."""
        return self.seq(x)


class ImageEncoderViT(nn.Module):
    """
    An image encoder using Vision Transformer (ViT) architecture for encoding images into a compact latent space.

    This class processes images by splitting them into patches, applying transformer blocks, and generating a final
    encoded representation through a neck module.

    Attributes:
        img_size (int): Dimension of input images, assumed to be square.
        patch_embed (PatchEmbed): Module for patch embedding.
        pos_embed (nn.Parameter | None): Absolute positional embedding for patches.
        blocks (nn.ModuleList): List of transformer blocks for processing patch embeddings.
        neck (nn.Sequential): Neck module to further process the output.

    Methods:
        forward: Processes input through patch embedding, positional embedding, blocks, and neck.

    Examples:
        >>> import torch
        >>> encoder = ImageEncoderViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
        >>> input_image = torch.randn(1, 3, 224, 224)
        >>> output = encoder(input_image)
        >>> print(output.shape)
    """

    def __init__(self, img_size: 'int'=1024, patch_size: 'int'=16, in_chans: 'int'=3, embed_dim: 'int'=768, depth: 'int'=12, num_heads: 'int'=12, mlp_ratio: 'float'=4.0, out_chans: 'int'=256, qkv_bias: 'bool'=True, norm_layer: 'Type[nn.Module]'=nn.LayerNorm, act_layer: 'Type[nn.Module]'=nn.GELU, use_abs_pos: 'bool'=True, use_rel_pos: 'bool'=False, rel_pos_zero_init: 'bool'=True, window_size: 'int'=0, global_attn_indexes: 'Tuple[int, ...]'=()) ->None:
        """
        Initializes an ImageEncoderViT instance for encoding images using Vision Transformer architecture.

        Args:
            img_size (int): Input image size, assumed to be square.
            patch_size (int): Size of image patches.
            in_chans (int): Number of input image channels.
            embed_dim (int): Dimension of patch embeddings.
            depth (int): Number of transformer blocks.
            num_heads (int): Number of attention heads in each block.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
            out_chans (int): Number of output channels from the neck module.
            qkv_bias (bool): If True, adds learnable bias to query, key, value projections.
            norm_layer (Type[nn.Module]): Type of normalization layer to use.
            act_layer (Type[nn.Module]): Type of activation layer to use.
            use_abs_pos (bool): If True, uses absolute positional embeddings.
            use_rel_pos (bool): If True, adds relative positional embeddings to attention maps.
            rel_pos_zero_init (bool): If True, initializes relative positional parameters to zero.
            window_size (int): Size of attention window for windowed attention blocks.
            global_attn_indexes (Tuple[int, ...]): Indices of blocks that use global attention.

        Attributes:
            img_size (int): Dimension of input images.
            patch_embed (PatchEmbed): Module for patch embedding.
            pos_embed (nn.Parameter | None): Absolute positional embedding for patches.
            blocks (nn.ModuleList): List of transformer blocks.
            neck (nn.Sequential): Neck module for final processing.

        Examples:
            >>> encoder = ImageEncoderViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
            >>> input_image = torch.randn(1, 3, 224, 224)
            >>> output = encoder(input_image)
            >>> print(output.shape)
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
        """Processes input through patch embedding, positional embedding, transformer blocks, and neck module."""
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = F.interpolate(self.pos_embed.permute(0, 3, 1, 2), scale_factor=self.img_size / 1024).permute(0, 2, 3, 1) if self.img_size != 1024 else self.pos_embed
            x = x + pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.neck(x.permute(0, 3, 1, 2))


class MaskDecoder(nn.Module):
    """
    Decoder module for generating masks and their associated quality scores using a transformer architecture.

    This class predicts masks given image and prompt embeddings, utilizing a transformer to process the inputs and
    generate mask predictions along with their quality scores.

    Attributes:
        transformer_dim (int): Channel dimension for the transformer module.
        transformer (nn.Module): Transformer module used for mask prediction.
        num_multimask_outputs (int): Number of masks to predict for disambiguating masks.
        iou_token (nn.Embedding): Embedding for the IoU token.
        num_mask_tokens (int): Number of mask tokens.
        mask_tokens (nn.Embedding): Embedding for the mask tokens.
        output_upscaling (nn.Sequential): Neural network sequence for upscaling the output.
        output_hypernetworks_mlps (nn.ModuleList): Hypernetwork MLPs for generating masks.
        iou_prediction_head (nn.Module): MLP for predicting mask quality.

    Methods:
        forward: Predicts masks given image and prompt embeddings.
        predict_masks: Internal method for mask prediction.

    Examples:
        >>> decoder = MaskDecoder(transformer_dim=256, transformer=transformer_module)
        >>> masks, iou_pred = decoder(
        ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output=True
        ... )
        >>> print(f"Predicted masks shape: {masks.shape}, IoU predictions shape: {iou_pred.shape}")
    """

    def __init__(self, transformer_dim: 'int', transformer: 'nn.Module', num_multimask_outputs: 'int'=3, activation: 'Type[nn.Module]'=nn.GELU, iou_head_depth: 'int'=3, iou_head_hidden_dim: 'int'=256) ->None:
        """
        Initializes the MaskDecoder module for generating masks and their quality scores.

        Args:
            transformer_dim (int): Channel dimension for the transformer module.
            transformer (nn.Module): Transformer module used for mask prediction.
            num_multimask_outputs (int): Number of masks to predict for disambiguating masks.
            activation (Type[nn.Module]): Type of activation to use when upscaling masks.
            iou_head_depth (int): Depth of the MLP used to predict mask quality.
            iou_head_hidden_dim (int): Hidden dimension of the MLP used to predict mask quality.

        Examples:
            >>> transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=8), num_layers=6)
            >>> decoder = MaskDecoder(transformer_dim=256, transformer=transformer)
            >>> print(decoder)
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2), LayerNorm2d(transformer_dim // 4), activation(), nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2), activation())
        self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)])
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def forward(self, image_embeddings: 'torch.Tensor', image_pe: 'torch.Tensor', sparse_prompt_embeddings: 'torch.Tensor', dense_prompt_embeddings: 'torch.Tensor', multimask_output: 'bool') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks given image and prompt embeddings.

        Args:
            image_embeddings (torch.Tensor): Embeddings from the image encoder.
            image_pe (torch.Tensor): Positional encoding with the shape of image_embeddings.
            sparse_prompt_embeddings (torch.Tensor): Embeddings of the points and boxes.
            dense_prompt_embeddings (torch.Tensor): Embeddings of the mask inputs.
            multimask_output (bool): Whether to return multiple masks or a single mask.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): A tuple containing:
                - masks (torch.Tensor): Batched predicted masks.
                - iou_pred (torch.Tensor): Batched predictions of mask quality.

        Examples:
            >>> decoder = MaskDecoder(transformer_dim=256, transformer=transformer_module)
            >>> image_emb = torch.rand(1, 256, 64, 64)
            >>> image_pe = torch.rand(1, 256, 64, 64)
            >>> sparse_emb = torch.rand(1, 2, 256)
            >>> dense_emb = torch.rand(1, 256, 64, 64)
            >>> masks, iou_pred = decoder(image_emb, image_pe, sparse_emb, dense_emb, multimask_output=True)
            >>> print(f"Masks shape: {masks.shape}, IoU predictions shape: {iou_pred.shape}")
        """
        masks, iou_pred = self.predict_masks(image_embeddings=image_embeddings, image_pe=image_pe, sparse_prompt_embeddings=sparse_prompt_embeddings, dense_prompt_embeddings=dense_prompt_embeddings)
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        return masks, iou_pred

    def predict_masks(self, image_embeddings: 'torch.Tensor', image_pe: 'torch.Tensor', sparse_prompt_embeddings: 'torch.Tensor', dense_prompt_embeddings: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks and quality scores using image and prompt embeddings via transformer architecture."""
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.shape[0], -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:1 + self.num_mask_tokens, :]
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: 'List[torch.Tensor]' = [self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(self.num_mask_tokens)]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.

    This class generates positional embeddings for input coordinates using random spatial frequencies. It is
    particularly useful for transformer-based models that require position information.

    Attributes:
        positional_encoding_gaussian_matrix (torch.Tensor): A buffer containing random values for encoding.

    Methods:
        _pe_encoding: Positionally encodes points that are normalized to [0,1].
        forward: Generates positional encoding for a grid of the specified size.
        forward_with_coords: Positionally encodes points that are not normalized to [0,1].

    Examples:
        >>> pe = PositionEmbeddingRandom(num_pos_feats=64)
        >>> size = (32, 32)
        >>> encoding = pe(size)
        >>> print(encoding.shape)
        torch.Size([128, 32, 32])
    """

    def __init__(self, num_pos_feats: 'int'=64, scale: 'Optional[float]'=None) ->None:
        """Initializes random spatial frequency position embedding for transformers."""
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer('positional_encoding_gaussian_matrix', scale * torch.randn((2, num_pos_feats)))
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

    def _pe_encoding(self, coords: 'torch.Tensor') ->torch.Tensor:
        """Encodes normalized [0,1] coordinates using random spatial frequencies."""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: 'Tuple[int, int]') ->torch.Tensor:
        """Generates positional encoding for a grid using random spatial frequencies."""
        h, w = size
        device: 'Any' = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(self, coords_input: 'torch.Tensor', image_size: 'Tuple[int, int]') ->torch.Tensor:
        """Positionally encodes input coordinates, normalizing them to [0,1] based on the given image size."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)


class PromptEncoder(nn.Module):
    """
    Encodes different types of prompts for input to SAM's mask decoder, producing sparse and dense embeddings.

    Attributes:
        embed_dim (int): Dimension of the embeddings.
        input_image_size (Tuple[int, int]): Size of the input image as (H, W).
        image_embedding_size (Tuple[int, int]): Spatial size of the image embedding as (H, W).
        pe_layer (PositionEmbeddingRandom): Module for random position embedding.
        num_point_embeddings (int): Number of point embeddings for different types of points.
        point_embeddings (nn.ModuleList): List of point embeddings.
        not_a_point_embed (nn.Embedding): Embedding for points that are not part of any label.
        mask_input_size (Tuple[int, int]): Size of the input mask.
        mask_downscaling (nn.Sequential): Neural network for downscaling the mask.
        no_mask_embed (nn.Embedding): Embedding for cases where no mask is provided.

    Methods:
        get_dense_pe: Returns the positional encoding used to encode point prompts.
        forward: Embeds different types of prompts, returning both sparse and dense embeddings.

    Examples:
        >>> prompt_encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
        >>> points = (torch.rand(1, 5, 2), torch.randint(0, 4, (1, 5)))
        >>> boxes = torch.rand(1, 2, 2)
        >>> masks = torch.rand(1, 1, 256, 256)
        >>> sparse_embeddings, dense_embeddings = prompt_encoder(points, boxes, masks)
        >>> print(sparse_embeddings.shape, dense_embeddings.shape)
        torch.Size([1, 7, 256]) torch.Size([1, 256, 64, 64])
    """

    def __init__(self, embed_dim: 'int', image_embedding_size: 'Tuple[int, int]', input_image_size: 'Tuple[int, int]', mask_in_chans: 'int', activation: 'Type[nn.Module]'=nn.GELU) ->None:
        """
        Initializes the PromptEncoder module for encoding various types of prompts.

        This module encodes different types of prompts (points, boxes, masks) for input to SAM's mask decoder,
        producing both sparse and dense embeddings.

        Args:
            embed_dim (int): The dimension of the embeddings.
            image_embedding_size (Tuple[int, int]): The spatial size of the image embedding as (H, W).
            input_image_size (Tuple[int, int]): The padded size of the input image as (H, W).
            mask_in_chans (int): The number of hidden channels used for encoding input masks.
            activation (Type[nn.Module]): The activation function to use when encoding input masks.

        Attributes:
            embed_dim (int): Dimension of the embeddings.
            input_image_size (Tuple[int, int]): Size of the input image as (H, W).
            image_embedding_size (Tuple[int, int]): Spatial size of the image embedding as (H, W).
            pe_layer (PositionEmbeddingRandom): Module for random position embedding.
            num_point_embeddings (int): Number of point embeddings for different types of points.
            point_embeddings (nn.ModuleList): List of point embeddings.
            not_a_point_embed (nn.Embedding): Embedding for points that are not part of any label.
            mask_input_size (Tuple[int, int]): Size of the input mask.
            mask_downscaling (nn.Sequential): Neural network for downscaling the mask.

        Examples:
            >>> prompt_encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
            >>> points = (torch.rand(1, 5, 2), torch.randint(0, 4, (1, 5)))
            >>> boxes = torch.rand(1, 2, 2)
            >>> masks = torch.rand(1, 1, 256, 256)
            >>> sparse_embeddings, dense_embeddings = prompt_encoder(points, boxes, masks)
            >>> print(sparse_embeddings.shape, dense_embeddings.shape)
            torch.Size([1, 7, 256]) torch.Size([1, 256, 64, 64])
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_point_embeddings: 'int' = 4
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = 4 * image_embedding_size[0], 4 * image_embedding_size[1]
        self.mask_downscaling = nn.Sequential(nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2), LayerNorm2d(mask_in_chans // 4), activation(), nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2), LayerNorm2d(mask_in_chans), activation(), nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1))
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) ->torch.Tensor:
        """
        Returns the dense positional encoding used for encoding point prompts.

        This method generates a positional encoding for a dense set of points matching the shape of the image
        encoding. The encoding is used to provide spatial information to the model when processing point prompts.

        Returns:
            (torch.Tensor): Positional encoding tensor with shape (1, embed_dim, H, W), where H and W are the
                height and width of the image embedding size, respectively.

        Examples:
            >>> prompt_encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
            >>> dense_pe = prompt_encoder.get_dense_pe()
            >>> print(dense_pe.shape)
            torch.Size([1, 256, 64, 64])
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: 'torch.Tensor', labels: 'torch.Tensor', pad: 'bool') ->torch.Tensor:
        """Embeds point prompts by applying positional encoding and label-specific embeddings."""
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == 2] += self.point_embeddings[2].weight
        point_embedding[labels == 3] += self.point_embeddings[3].weight
        return point_embedding

    def _embed_boxes(self, boxes: 'torch.Tensor') ->torch.Tensor:
        """Embeds box prompts by applying positional encoding and adding corner embeddings."""
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: 'torch.Tensor') ->torch.Tensor:
        """Embeds mask inputs by downscaling and processing through convolutional layers."""
        return self.mask_downscaling(masks)

    @staticmethod
    def _get_batch_size(points: 'Optional[Tuple[torch.Tensor, torch.Tensor]]', boxes: 'Optional[torch.Tensor]', masks: 'Optional[torch.Tensor]') ->int:
        """Gets the batch size of the output given the batch size of the input prompts."""
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) ->torch.device:
        """Returns the device of the first point embedding's weight tensor."""
        return self.point_embeddings[0].weight.device

    def forward(self, points: 'Optional[Tuple[torch.Tensor, torch.Tensor]]', boxes: 'Optional[torch.Tensor]', masks: 'Optional[torch.Tensor]') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (Tuple[torch.Tensor, torch.Tensor] | None): Point coordinates and labels to embed. The first
                tensor contains coordinates with shape (B, N, 2), and the second tensor contains labels with
                shape (B, N).
            boxes (torch.Tensor | None): Boxes to embed with shape (B, M, 2, 2), where M is the number of boxes.
            masks (torch.Tensor | None): Masks to embed with shape (B, 1, H, W).

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): A tuple containing:
                - sparse_embeddings (torch.Tensor): Sparse embeddings for points and boxes with shape (B, N, embed_dim).
                - dense_embeddings (torch.Tensor): Dense embeddings for masks of shape (B, embed_dim, embed_H, embed_W).

        Examples:
            >>> encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
            >>> points = (torch.rand(1, 5, 2), torch.randint(0, 4, (1, 5)))
            >>> boxes = torch.rand(1, 2, 2, 2)
            >>> masks = torch.rand(1, 1, 256, 256)
            >>> sparse_emb, dense_emb = encoder(points, boxes, masks)
            >>> print(sparse_emb.shape, dense_emb.shape)
            torch.Size([1, 7, 256]) torch.Size([1, 256, 64, 64])
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=boxes is None)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(bs, -1, self.image_embedding_size[0], self.image_embedding_size[1])
        return sparse_embeddings, dense_embeddings


class SAMModel(nn.Module):
    """
    Segment Anything Model (SAM) for object segmentation tasks.

    This class combines image encoders, prompt encoders, and mask decoders to predict object masks from images
    and input prompts.

    Attributes:
        mask_threshold (float): Threshold value for mask prediction.
        image_encoder (ImageEncoderViT): Backbone for encoding images into embeddings.
        prompt_encoder (PromptEncoder): Encoder for various types of input prompts.
        mask_decoder (MaskDecoder): Predicts object masks from image and prompt embeddings.
        pixel_mean (torch.Tensor): Mean pixel values for image normalization, shape (3, 1, 1).
        pixel_std (torch.Tensor): Standard deviation values for image normalization, shape (3, 1, 1).

    Methods:
        __init__: Initializes the SAMModel with encoders, decoder, and normalization parameters.

    Examples:
        >>> image_encoder = ImageEncoderViT(...)
        >>> prompt_encoder = PromptEncoder(...)
        >>> mask_decoder = MaskDecoder(...)
        >>> sam_model = SAMModel(image_encoder, prompt_encoder, mask_decoder)
        >>> # Further usage depends on SAMPredictor class

    Notes:
        All forward() operations are implemented in the SAMPredictor class.
    """
    mask_threshold: 'float' = 0.0

    def __init__(self, image_encoder: 'ImageEncoderViT', prompt_encoder: 'PromptEncoder', mask_decoder: 'MaskDecoder', pixel_mean: 'List[float]'=(123.675, 116.28, 103.53), pixel_std: 'List[float]'=(58.395, 57.12, 57.375)) ->None:
        """
        Initialize the SAMModel class to predict object masks from an image and input prompts.

        Args:
            image_encoder (ImageEncoderViT): The backbone used to encode the image into image embeddings.
            prompt_encoder (PromptEncoder): Encodes various types of input prompts.
            mask_decoder (MaskDecoder): Predicts masks from the image embeddings and encoded prompts.
            pixel_mean (List[float]): Mean values for normalizing pixels in the input image.
            pixel_std (List[float]): Std values for normalizing pixels in the input image.

        Examples:
            >>> image_encoder = ImageEncoderViT(...)
            >>> prompt_encoder = PromptEncoder(...)
            >>> mask_decoder = MaskDecoder(...)
            >>> sam_model = SAMModel(image_encoder, prompt_encoder, mask_decoder)
            >>> # Further usage depends on SAMPredictor class

        Notes:
            All forward() operations moved to SAMPredictor.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(-1, 1, 1), False)

    def set_imgsz(self, imgsz):
        """
        Set image size to make model compatible with different image sizes.

        Args:
            imgsz (Tuple[int, int]): The size of the input image.
        """
        if hasattr(self.image_encoder, 'set_imgsz'):
            self.image_encoder.set_imgsz(imgsz)
        self.prompt_encoder.input_image_size = imgsz
        self.prompt_encoder.image_embedding_size = [(x // 16) for x in imgsz]
        self.image_encoder.img_size = imgsz[0]


class Mlp(nn.Module):
    """
    Multi-layer Perceptron (MLP) module for transformer architectures.

    This module applies layer normalization, two fully-connected layers with an activation function in between,
    and dropout. It is commonly used in transformer-based architectures.

    Attributes:
        norm (nn.LayerNorm): Layer normalization applied to the input.
        fc1 (nn.Linear): First fully-connected layer.
        fc2 (nn.Linear): Second fully-connected layer.
        act (nn.Module): Activation function applied after the first fully-connected layer.
        drop (nn.Dropout): Dropout layer applied after the activation function.

    Methods:
        forward: Applies the MLP operations on the input tensor.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> mlp = Mlp(in_features=256, hidden_features=512, out_features=256, act_layer=nn.GELU, drop=0.1)
        >>> x = torch.randn(32, 100, 256)
        >>> output = mlp(x)
        >>> print(output.shape)
        torch.Size([32, 100, 256])
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        """Initializes a multi-layer perceptron with configurable input, hidden, and output dimensions."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Applies MLP operations: layer norm, FC layers, activation, and dropout to the input tensor."""
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class TinyViTBlock(nn.Module):
    """
    TinyViT Block that applies self-attention and a local convolution to the input.

    This block is a key component of the TinyViT architecture, combining self-attention mechanisms with
    local convolutions to process input features efficiently.

    Attributes:
        dim (int): The dimensionality of the input and output.
        input_resolution (Tuple[int, int]): Spatial resolution of the input feature map.
        num_heads (int): Number of attention heads.
        window_size (int): Size of the attention window.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        drop_path (nn.Module): Stochastic depth layer, identity function during inference.
        attn (Attention): Self-attention module.
        mlp (Mlp): Multi-layer perceptron module.
        local_conv (Conv2d_BN): Depth-wise local convolution layer.

    Methods:
        forward: Processes the input through the TinyViT block.
        extra_repr: Returns a string with extra information about the block's parameters.

    Examples:
        >>> input_tensor = torch.randn(1, 196, 192)
        >>> block = TinyViTBlock(dim=192, input_resolution=(14, 14), num_heads=3)
        >>> output = block(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 196, 192])
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4.0, drop=0.0, drop_path=0.0, local_conv_size=3, activation=nn.GELU):
        """
        Initializes a TinyViT block with self-attention and local convolution.

        This block is a key component of the TinyViT architecture, combining self-attention mechanisms with
        local convolutions to process input features efficiently.

        Args:
            dim (int): Dimensionality of the input and output features.
            input_resolution (Tuple[int, int]): Spatial resolution of the input feature map (height, width).
            num_heads (int): Number of attention heads.
            window_size (int): Size of the attention window. Must be greater than 0.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
            drop (float): Dropout rate.
            drop_path (float): Stochastic depth rate.
            local_conv_size (int): Kernel size of the local convolution.
            activation (torch.nn.Module): Activation function for MLP.

        Raises:
            AssertionError: If window_size is not greater than 0.
            AssertionError: If dim is not divisible by num_heads.

        Examples:
            >>> block = TinyViTBlock(dim=192, input_resolution=(14, 14), num_heads=3)
            >>> input_tensor = torch.randn(1, 196, 192)
            >>> output = block(input_tensor)
            >>> print(output.shape)
            torch.Size([1, 196, 192])
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_path = nn.Identity()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads
        window_resolution = window_size, window_size
        self.attn = Attention(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=mlp_activation, drop=drop)
        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        """Applies self-attention, local convolution, and MLP operations to the input tensor."""
        h, w = self.input_resolution
        b, hw, c = x.shape
        assert hw == h * w, 'input feature has wrong size'
        res_x = x
        if h == self.window_size and w == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(b, h, w, c)
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            pH, pW = h + pad_b, w + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            x = x.view(b, nH, self.window_size, nW, self.window_size, c).transpose(2, 3).reshape(b * nH * nW, self.window_size * self.window_size, c)
            x = self.attn(x)
            x = x.view(b, nH, nW, self.window_size, self.window_size, c).transpose(2, 3).reshape(b, pH, pW, c)
            if padding:
                x = x[:, :h, :w].contiguous()
            x = x.view(b, hw, c)
        x = res_x + self.drop_path(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.local_conv(x)
        x = x.view(b, c, hw).transpose(1, 2)
        return x + self.drop_path(self.mlp(x))

    def extra_repr(self) ->str:
        """
        Returns a string representation of the TinyViTBlock's parameters.

        This method provides a formatted string containing key information about the TinyViTBlock, including its
        dimension, input resolution, number of attention heads, window size, and MLP ratio.

        Returns:
            (str): A formatted string containing the block's parameters.

        Examples:
            >>> block = TinyViTBlock(dim=192, input_resolution=(14, 14), num_heads=3, window_size=7, mlp_ratio=4.0)
            >>> print(block.extra_repr())
            dim=192, input_resolution=(14, 14), num_heads=3, window_size=7, mlp_ratio=4.0
        """
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, mlp_ratio={self.mlp_ratio}'


class BasicLayer(nn.Module):
    """
    A basic TinyViT layer for one stage in a TinyViT architecture.

    This class represents a single layer in the TinyViT model, consisting of multiple TinyViT blocks
    and an optional downsampling operation.

    Attributes:
        dim (int): The dimensionality of the input and output features.
        input_resolution (Tuple[int, int]): Spatial resolution of the input feature map.
        depth (int): Number of TinyViT blocks in this layer.
        use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
        blocks (nn.ModuleList): List of TinyViT blocks that make up this layer.
        downsample (nn.Module | None): Downsample layer at the end of the layer, if specified.

    Methods:
        forward: Processes the input through the layer's blocks and optional downsampling.
        extra_repr: Returns a string with the layer's parameters for printing.

    Examples:
        >>> input_tensor = torch.randn(1, 3136, 192)
        >>> layer = BasicLayer(dim=192, input_resolution=(56, 56), depth=2, num_heads=3, window_size=7)
        >>> output = layer(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 784, 384])
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, drop=0.0, drop_path=0.0, downsample=None, use_checkpoint=False, local_conv_size=3, activation=nn.GELU, out_dim=None):
        """
        Initializes a BasicLayer in the TinyViT architecture.

        This layer consists of multiple TinyViT blocks and an optional downsampling operation. It is designed to
        process feature maps at a specific resolution and dimensionality within the TinyViT model.

        Args:
            dim (int): Dimensionality of the input and output features.
            input_resolution (Tuple[int, int]): Spatial resolution of the input feature map (height, width).
            depth (int): Number of TinyViT blocks in this layer.
            num_heads (int): Number of attention heads in each TinyViT block.
            window_size (int): Size of the local window for attention computation.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
            drop (float): Dropout rate.
            drop_path (float | List[float]): Stochastic depth rate. Can be a float or a list of floats for each block.
            downsample (nn.Module | None): Downsampling layer at the end of the layer. None to skip downsampling.
            use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
            local_conv_size (int): Kernel size for the local convolution in each TinyViT block.
            activation (nn.Module): Activation function used in the MLP.
            out_dim (int | None): Output dimension after downsampling. None means it will be the same as `dim`.

        Raises:
            ValueError: If `drop_path` is a list and its length doesn't match `depth`.

        Examples:
            >>> layer = BasicLayer(dim=96, input_resolution=(56, 56), depth=2, num_heads=3, window_size=7)
            >>> x = torch.randn(1, 56 * 56, 96)
            >>> output = layer(x)
            >>> print(output.shape)
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([TinyViTBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, local_conv_size=local_conv_size, activation=activation) for i in range(depth)])
        self.downsample = None if downsample is None else downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)

    def forward(self, x):
        """Processes input through TinyViT blocks and optional downsampling."""
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x if self.downsample is None else self.downsample(x)

    def extra_repr(self) ->str:
        """Returns a string with the layer's parameters for printing."""
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'


class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Conv (MBConv) layer, part of the EfficientNet architecture.

    Attributes:
        in_chans (int): Number of input channels.
        hidden_chans (int): Number of hidden channels.
        out_chans (int): Number of output channels.
        conv1 (Conv2d_BN): First convolutional layer.
        act1 (nn.Module): First activation function.
        conv2 (Conv2d_BN): Depthwise convolutional layer.
        act2 (nn.Module): Second activation function.
        conv3 (Conv2d_BN): Final convolutional layer.
        act3 (nn.Module): Third activation function.
        drop_path (nn.Module): Drop path layer (Identity for inference).

    Methods:
        forward: Performs the forward pass through the MBConv layer.

    Examples:
        >>> in_chans, out_chans = 32, 64
        >>> mbconv = MBConv(in_chans, out_chans, expand_ratio=4, activation=nn.ReLU, drop_path=0.1)
        >>> x = torch.randn(1, in_chans, 56, 56)
        >>> output = mbconv(x)
        >>> print(output.shape)
        torch.Size([1, 64, 56, 56])
    """

    def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
        """Initializes the MBConv layer with specified input/output channels, expansion ratio, and activation."""
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans
        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()
        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans, ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()
        self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()
        self.drop_path = nn.Identity()

    def forward(self, x):
        """Implements the forward pass of MBConv, applying convolutions and skip connection."""
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        return self.act3(x)


class ConvLayer(nn.Module):
    """
    Convolutional Layer featuring multiple MobileNetV3-style inverted bottleneck convolutions (MBConv).

    This layer optionally applies downsample operations to the output and supports gradient checkpointing.

    Attributes:
        dim (int): Dimensionality of the input and output.
        input_resolution (Tuple[int, int]): Resolution of the input image.
        depth (int): Number of MBConv layers in the block.
        use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
        blocks (nn.ModuleList): List of MBConv layers.
        downsample (Optional[Callable]): Function for downsampling the output.

    Methods:
        forward: Processes the input through the convolutional layers.

    Examples:
        >>> input_tensor = torch.randn(1, 64, 56, 56)
        >>> conv_layer = ConvLayer(64, (56, 56), depth=3, activation=nn.ReLU)
        >>> output = conv_layer(input_tensor)
        >>> print(output.shape)
    """

    def __init__(self, dim, input_resolution, depth, activation, drop_path=0.0, downsample=None, use_checkpoint=False, out_dim=None, conv_expand_ratio=4.0):
        """
        Initializes the ConvLayer with the given dimensions and settings.

        This layer consists of multiple MobileNetV3-style inverted bottleneck convolutions (MBConv) and
        optionally applies downsampling to the output.

        Args:
            dim (int): The dimensionality of the input and output.
            input_resolution (Tuple[int, int]): The resolution of the input image.
            depth (int): The number of MBConv layers in the block.
            activation (Callable): Activation function applied after each convolution.
            drop_path (float | List[float]): Drop path rate. Single float or a list of floats for each MBConv.
            downsample (Optional[Callable]): Function for downsampling the output. None to skip downsampling.
            use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
            out_dim (Optional[int]): The dimensionality of the output. None means it will be the same as `dim`.
            conv_expand_ratio (float): Expansion ratio for the MBConv layers.

        Examples:
            >>> input_tensor = torch.randn(1, 64, 56, 56)
            >>> conv_layer = ConvLayer(64, (56, 56), depth=3, activation=nn.ReLU)
            >>> output = conv_layer(input_tensor)
            >>> print(output.shape)
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([MBConv(dim, dim, conv_expand_ratio, activation, drop_path[i] if isinstance(drop_path, list) else drop_path) for i in range(depth)])
        self.downsample = None if downsample is None else downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)

    def forward(self, x):
        """Processes input through convolutional layers, applying MBConv blocks and optional downsampling."""
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x if self.downsample is None else self.downsample(x)


class PatchMerging(nn.Module):
    """
    Merges neighboring patches in the feature map and projects to a new dimension.

    This class implements a patch merging operation that combines spatial information and adjusts the feature
    dimension. It uses a series of convolutional layers with batch normalization to achieve this.

    Attributes:
        input_resolution (Tuple[int, int]): The input resolution (height, width) of the feature map.
        dim (int): The input dimension of the feature map.
        out_dim (int): The output dimension after merging and projection.
        act (nn.Module): The activation function used between convolutions.
        conv1 (Conv2d_BN): The first convolutional layer for dimension projection.
        conv2 (Conv2d_BN): The second convolutional layer for spatial merging.
        conv3 (Conv2d_BN): The third convolutional layer for final projection.

    Methods:
        forward: Applies the patch merging operation to the input tensor.

    Examples:
        >>> input_resolution = (56, 56)
        >>> patch_merging = PatchMerging(input_resolution, dim=64, out_dim=128, activation=nn.ReLU)
        >>> x = torch.randn(4, 64, 56, 56)
        >>> output = patch_merging(x)
        >>> print(output.shape)
    """

    def __init__(self, input_resolution, dim, out_dim, activation):
        """Initializes the PatchMerging module for merging and projecting neighboring patches in feature maps."""
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c = 1 if out_dim in {320, 448, 576} else 2
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        """Applies patch merging and dimension projection to the input feature map."""
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x.flatten(2).transpose(1, 2)


class TinyViT(nn.Module):
    """
    TinyViT: A compact vision transformer architecture for efficient image classification and feature extraction.

    This class implements the TinyViT model, which combines elements of vision transformers and convolutional
    neural networks for improved efficiency and performance on vision tasks.

    Attributes:
        img_size (int): Input image size.
        num_classes (int): Number of classification classes.
        depths (List[int]): Number of blocks in each stage.
        num_layers (int): Total number of layers in the network.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        patch_embed (PatchEmbed): Module for patch embedding.
        patches_resolution (Tuple[int, int]): Resolution of embedded patches.
        layers (nn.ModuleList): List of network layers.
        norm_head (nn.LayerNorm): Layer normalization for the classifier head.
        head (nn.Linear): Linear layer for final classification.
        neck (nn.Sequential): Neck module for feature refinement.

    Methods:
        set_layer_lr_decay: Sets layer-wise learning rate decay.
        _init_weights: Initializes weights for linear and normalization layers.
        no_weight_decay_keywords: Returns keywords for parameters that should not use weight decay.
        forward_features: Processes input through the feature extraction layers.
        forward: Performs a forward pass through the entire network.

    Examples:
        >>> model = TinyViT(img_size=224, num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> features = model.forward_features(x)
        >>> print(features.shape)
        torch.Size([1, 256, 64, 64])
    """

    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=(96, 192, 384, 768), depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_sizes=(7, 7, 14, 7), mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.1, use_checkpoint=False, mbconv_expand_ratio=4.0, local_conv_size=3, layer_lr_decay=1.0):
        """
        Initializes the TinyViT model.

        This constructor sets up the TinyViT architecture, including patch embedding, multiple layers of
        attention and convolution blocks, and a classification head.

        Args:
            img_size (int): Size of the input image. Default is 224.
            in_chans (int): Number of input channels. Default is 3.
            num_classes (int): Number of classes for classification. Default is 1000.
            embed_dims (Tuple[int, int, int, int]): Embedding dimensions for each stage.
                Default is (96, 192, 384, 768).
            depths (Tuple[int, int, int, int]): Number of blocks in each stage. Default is (2, 2, 6, 2).
            num_heads (Tuple[int, int, int, int]): Number of attention heads in each stage.
                Default is (3, 6, 12, 24).
            window_sizes (Tuple[int, int, int, int]): Window sizes for each stage. Default is (7, 7, 14, 7).
            mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Default is 4.0.
            drop_rate (float): Dropout rate. Default is 0.0.
            drop_path_rate (float): Stochastic depth rate. Default is 0.1.
            use_checkpoint (bool): Whether to use checkpointing to save memory. Default is False.
            mbconv_expand_ratio (float): Expansion ratio for MBConv layer. Default is 4.0.
            local_conv_size (int): Kernel size for local convolutions. Default is 3.
            layer_lr_decay (float): Layer-wise learning rate decay factor. Default is 1.0.

        Examples:
            >>> model = TinyViT(img_size=224, num_classes=1000)
            >>> x = torch.randn(1, 3, 224, 224)
            >>> output = model(x)
            >>> print(output.shape)
            torch.Size([1, 1000])
        """
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        activation = nn.GELU
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0], resolution=img_size, activation=activation)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(dim=embed_dims[i_layer], input_resolution=(patches_resolution[0] // 2 ** (i_layer - 1 if i_layer == 3 else i_layer), patches_resolution[1] // 2 ** (i_layer - 1 if i_layer == 3 else i_layer)), depth=depths[i_layer], drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint, out_dim=embed_dims[min(i_layer + 1, len(embed_dims) - 1)], activation=activation)
            if i_layer == 0:
                layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                layer = BasicLayer(num_heads=num_heads[i_layer], window_size=window_sizes[i_layer], mlp_ratio=self.mlp_ratio, drop=drop_rate, local_conv_size=local_conv_size, **kwargs)
            self.layers.append(layer)
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(nn.Conv2d(embed_dims[-1], 256, kernel_size=1, bias=False), LayerNorm2d(256), nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), LayerNorm2d(256))

    def set_layer_lr_decay(self, layer_lr_decay):
        """Sets layer-wise learning rate decay for the TinyViT model based on depth."""
        decay_rate = layer_lr_decay
        depth = sum(self.depths)
        lr_scales = [(decay_rate ** (depth - i - 1)) for i in range(depth)]

        def _set_lr_scale(m, scale):
            """Sets the learning rate scale for each layer in the model based on the layer's depth."""
            for p in m.parameters():
                p.lr_scale = scale
        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))
        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            """Checks if the learning rate scale attribute is present in module's parameters."""
            for p in m.parameters():
                assert hasattr(p, 'lr_scale'), p.param_name
        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        """Initializes weights for linear and normalization layers in the TinyViT model."""
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Returns a set of keywords for parameters that should not use weight decay."""
        return {'attention_biases'}

    def forward_features(self, x):
        """Processes input through feature extraction layers, returning spatial features."""
        x = self.patch_embed(x)
        x = self.layers[0](x)
        start_i = 1
        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        batch, _, channel = x.shape
        x = x.view(batch, self.patches_resolution[0] // 4, self.patches_resolution[1] // 4, channel)
        x = x.permute(0, 3, 1, 2)
        return self.neck(x)

    def forward(self, x):
        """Performs the forward pass through the TinyViT model, extracting features from the input image."""
        return self.forward_features(x)

    def set_imgsz(self, imgsz=[1024, 1024]):
        """
        Set image size to make model compatible with different image sizes.

        Args:
            imgsz (Tuple[int, int]): The size of the input image.
        """
        imgsz = [(s // 4) for s in imgsz]
        self.patches_resolution = imgsz
        for i, layer in enumerate(self.layers):
            input_resolution = imgsz[0] // 2 ** (i - 1 if i == 3 else i), imgsz[1] // 2 ** (i - 1 if i == 3 else i)
            layer.input_resolution = input_resolution
            if layer.downsample is not None:
                layer.downsample.input_resolution = input_resolution
            if isinstance(layer, BasicLayer):
                for b in layer.blocks:
                    b.input_resolution = input_resolution


class TwoWayAttentionBlock(nn.Module):
    """
    A two-way attention block for simultaneous attention to image and query points.

    This class implements a specialized transformer block with four main layers: self-attention on sparse inputs,
    cross-attention of sparse inputs to dense inputs, MLP block on sparse inputs, and cross-attention of dense
    inputs to sparse inputs.

    Attributes:
        self_attn (Attention): Self-attention layer for queries.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        cross_attn_token_to_image (Attention): Cross-attention layer from queries to keys.
        norm2 (nn.LayerNorm): Layer normalization after token-to-image attention.
        mlp (MLPBlock): MLP block for transforming query embeddings.
        norm3 (nn.LayerNorm): Layer normalization after MLP block.
        norm4 (nn.LayerNorm): Layer normalization after image-to-token attention.
        cross_attn_image_to_token (Attention): Cross-attention layer from keys to queries.
        skip_first_layer_pe (bool): Whether to skip positional encoding in the first layer.

    Methods:
        forward: Applies self-attention and cross-attention to queries and keys.

    Examples:
        >>> embedding_dim, num_heads = 256, 8
        >>> block = TwoWayAttentionBlock(embedding_dim, num_heads)
        >>> queries = torch.randn(1, 100, embedding_dim)
        >>> keys = torch.randn(1, 1000, embedding_dim)
        >>> query_pe = torch.randn(1, 100, embedding_dim)
        >>> key_pe = torch.randn(1, 1000, embedding_dim)
        >>> processed_queries, processed_keys = block(queries, keys, query_pe, key_pe)
    """

    def __init__(self, embedding_dim: 'int', num_heads: 'int', mlp_dim: 'int'=2048, activation: 'Type[nn.Module]'=nn.ReLU, attention_downsample_rate: 'int'=2, skip_first_layer_pe: 'bool'=False) ->None:
        """
        Initializes a TwoWayAttentionBlock for simultaneous attention to image and query points.

        This block implements a specialized transformer layer with four main components: self-attention on sparse
        inputs, cross-attention of sparse inputs to dense inputs, MLP block on sparse inputs, and cross-attention
        of dense inputs to sparse inputs.

        Args:
            embedding_dim (int): Channel dimension of the embeddings.
            num_heads (int): Number of attention heads in the attention layers.
            mlp_dim (int): Hidden dimension of the MLP block.
            activation (Type[nn.Module]): Activation function for the MLP block.
            attention_downsample_rate (int): Downsampling rate for the attention mechanism.
            skip_first_layer_pe (bool): Whether to skip positional encoding in the first layer.

        Examples:
            >>> embedding_dim, num_heads = 256, 8
            >>> block = TwoWayAttentionBlock(embedding_dim, num_heads)
            >>> queries = torch.randn(1, 100, embedding_dim)
            >>> keys = torch.randn(1, 1000, embedding_dim)
            >>> query_pe = torch.randn(1, 100, embedding_dim)
            >>> key_pe = torch.randn(1, 1000, embedding_dim)
            >>> processed_queries, processed_keys = block(queries, keys, query_pe, key_pe)
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: 'Tensor', keys: 'Tensor', query_pe: 'Tensor', key_pe: 'Tensor') ->Tuple[Tensor, Tensor]:
        """Applies two-way attention to process query and key embeddings in a transformer block."""
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class TwoWayTransformer(nn.Module):
    """
    A Two-Way Transformer module for simultaneous attention to image and query points.

    This class implements a specialized transformer decoder that attends to an input image using queries with
    supplied positional embeddings. It's useful for tasks like object detection, image segmentation, and point
    cloud processing.

    Attributes:
        depth (int): Number of layers in the transformer.
        embedding_dim (int): Channel dimension for input embeddings.
        num_heads (int): Number of heads for multihead attention.
        mlp_dim (int): Internal channel dimension for the MLP block.
        layers (nn.ModuleList): List of TwoWayAttentionBlock layers composing the transformer.
        final_attn_token_to_image (Attention): Final attention layer from queries to image.
        norm_final_attn (nn.LayerNorm): Layer normalization applied to final queries.

    Methods:
        forward: Processes image and point embeddings through the transformer.

    Examples:
        >>> transformer = TwoWayTransformer(depth=6, embedding_dim=256, num_heads=8, mlp_dim=2048)
        >>> image_embedding = torch.randn(1, 256, 32, 32)
        >>> image_pe = torch.randn(1, 256, 32, 32)
        >>> point_embedding = torch.randn(1, 100, 256)
        >>> output_queries, output_image = transformer(image_embedding, image_pe, point_embedding)
        >>> print(output_queries.shape, output_image.shape)
    """

    def __init__(self, depth: 'int', embedding_dim: 'int', num_heads: 'int', mlp_dim: 'int', activation: 'Type[nn.Module]'=nn.ReLU, attention_downsample_rate: 'int'=2) ->None:
        """
        Initialize a Two-Way Transformer for simultaneous attention to image and query points.

        Args:
            depth (int): Number of layers in the transformer.
            embedding_dim (int): Channel dimension for input embeddings.
            num_heads (int): Number of heads for multihead attention. Must divide embedding_dim.
            mlp_dim (int): Internal channel dimension for the MLP block.
            activation (Type[nn.Module]): Activation function to use in the MLP block.
            attention_downsample_rate (int): Downsampling rate for attention mechanism.

        Attributes:
            depth (int): Number of layers in the transformer.
            embedding_dim (int): Channel dimension for input embeddings.
            num_heads (int): Number of heads for multihead attention.
            mlp_dim (int): Internal channel dimension for the MLP block.
            layers (nn.ModuleList): List of TwoWayAttentionBlock layers.
            final_attn_token_to_image (Attention): Final attention layer from queries to image.
            norm_final_attn (nn.LayerNorm): Layer normalization applied to final queries.

        Examples:
            >>> transformer = TwoWayTransformer(depth=6, embedding_dim=256, num_heads=8, mlp_dim=2048)
            >>> image_embedding = torch.randn(1, 256, 32, 32)
            >>> image_pe = torch.randn(1, 256, 32, 32)
            >>> point_embedding = torch.randn(1, 100, 256)
            >>> output_queries, output_image = transformer(image_embedding, image_pe, point_embedding)
            >>> print(output_queries.shape, output_image.shape)
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_dim=mlp_dim, activation=activation, attention_downsample_rate=attention_downsample_rate, skip_first_layer_pe=i == 0))
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: 'Tensor', image_pe: 'Tensor', point_embedding: 'Tensor') ->Tuple[Tensor, Tensor]:
        """
        Processes image and point embeddings through the Two-Way Transformer.

        Args:
            image_embedding (torch.Tensor): Image to attend to, with shape (B, embedding_dim, H, W).
            image_pe (torch.Tensor): Positional encoding to add to the image, with same shape as image_embedding.
            point_embedding (torch.Tensor): Embedding to add to query points, with shape (B, N_points, embedding_dim).

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): Processed point_embedding and image_embedding.

        Examples:
            >>> transformer = TwoWayTransformer(depth=6, embedding_dim=256, num_heads=8, mlp_dim=2048)
            >>> image_embedding = torch.randn(1, 256, 32, 32)
            >>> image_pe = torch.randn(1, 256, 32, 32)
            >>> point_embedding = torch.randn(1, 100, 256)
            >>> output_queries, output_image = transformer(image_embedding, image_pe, point_embedding)
            >>> print(output_queries.shape, output_image.shape)
        """
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


def _build_sam(encoder_embed_dim, encoder_depth, encoder_num_heads, encoder_global_attn_indexes, checkpoint=None, mobile_sam=False):
    """
    Builds a Segment Anything Model (SAM) with specified encoder parameters.

    Args:
        encoder_embed_dim (int | List[int]): Embedding dimension for the encoder.
        encoder_depth (int | List[int]): Depth of the encoder.
        encoder_num_heads (int | List[int]): Number of attention heads in the encoder.
        encoder_global_attn_indexes (List[int] | None): Indexes for global attention in the encoder.
        checkpoint (str | None): Path to the model checkpoint file.
        mobile_sam (bool): Whether to build a Mobile-SAM model.

    Returns:
        (SAMModel): A Segment Anything Model instance with the specified architecture.

    Examples:
        >>> sam = _build_sam(768, 12, 12, [2, 5, 8, 11])
        >>> sam = _build_sam([64, 128, 160, 320], [2, 2, 6, 2], [2, 4, 5, 10], None, mobile_sam=True)
    """
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    image_encoder = TinyViT(img_size=1024, in_chans=3, num_classes=1000, embed_dims=encoder_embed_dim, depths=encoder_depth, num_heads=encoder_num_heads, window_sizes=[7, 7, 14, 7], mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.0, use_checkpoint=False, mbconv_expand_ratio=4.0, local_conv_size=3, layer_lr_decay=0.8) if mobile_sam else ImageEncoderViT(depth=encoder_depth, embed_dim=encoder_embed_dim, img_size=image_size, mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-06), num_heads=encoder_num_heads, patch_size=vit_patch_size, qkv_bias=True, use_rel_pos=True, global_attn_indexes=encoder_global_attn_indexes, window_size=14, out_chans=prompt_embed_dim)
    sam = SAMModel(image_encoder=image_encoder, prompt_encoder=PromptEncoder(embed_dim=prompt_embed_dim, image_embedding_size=(image_embedding_size, image_embedding_size), input_image_size=(image_size, image_size), mask_in_chans=16), mask_decoder=MaskDecoder(num_multimask_outputs=3, transformer=TwoWayTransformer(depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8), transformer_dim=prompt_embed_dim, iou_head_depth=3, iou_head_hidden_dim=256), pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375])
    if checkpoint is not None:
        checkpoint = attempt_download_asset(checkpoint)
        with open(checkpoint, 'rb') as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    sam.eval()
    return sam


def build_mobile_sam(checkpoint=None):
    """Builds and returns a Mobile Segment Anything Model (Mobile-SAM) for efficient image segmentation."""
    return _build_sam(encoder_embed_dim=[64, 128, 160, 320], encoder_depth=[2, 2, 6, 2], encoder_num_heads=[2, 4, 5, 10], encoder_global_attn_indexes=None, mobile_sam=True, checkpoint=checkpoint)


class PositionEmbeddingSine(nn.Module):
    """
    A module for generating sinusoidal positional embeddings for 2D inputs like images.

    This class implements sinusoidal position encoding for 2D spatial positions, which can be used in
    transformer-based models for computer vision tasks.

    Attributes:
        num_pos_feats (int): Number of positional features (half of the embedding dimension).
        temperature (int): Temperature parameter for the sinusoidal functions.
        normalize (bool): Whether to normalize the positional embeddings.
        scale (float): Scaling factor for the embeddings when normalize is True.
        cache (Dict): Cache for storing precomputed embeddings.

    Methods:
        _encode_xy: Encodes 2D positions using sine and cosine functions.
        encode_boxes: Encodes box coordinates and dimensions into positional embeddings.
        encode_points: Encodes 2D point coordinates with sinusoidal positional embeddings.
        forward: Generates sinusoidal position embeddings for 2D inputs.

    Examples:
        >>> pos_emb = PositionEmbeddingSine(num_pos_feats=128)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> embeddings = pos_emb(x)
        >>> print(embeddings.shape)
        torch.Size([1, 256, 224, 224])
    """

    def __init__(self, num_pos_feats, temperature: 'int'=10000, normalize: 'bool'=True, scale: 'Optional[float]'=None):
        """Initializes sinusoidal position embeddings for 2D image inputs."""
        super().__init__()
        assert num_pos_feats % 2 == 0, 'Expecting even model width'
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.cache = {}

    def _encode_xy(self, x, y):
        """Encodes 2D positions using sine/cosine functions for transformer positional embeddings."""
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        """Encodes box coordinates and dimensions into positional embeddings for detection."""
        pos_x, pos_y = self._encode_xy(x, y)
        return torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
    encode = encode_boxes

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        """Encodes 2D points with sinusoidal embeddings and appends labels."""
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        return torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)

    @torch.no_grad()
    def forward(self, x: 'torch.Tensor'):
        """Generates sinusoidal position embeddings for 2D inputs like images."""
        cache_key = x.shape[-2], x.shape[-1]
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device).view(1, -1, 1).repeat(x.shape[0], 1, x.shape[-1])
        x_embed = torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device).view(1, 1, -1).repeat(x.shape[0], x.shape[-2], 1)
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos


class FpnNeck(nn.Module):
    """
    A Feature Pyramid Network (FPN) neck variant for multiscale feature fusion in object detection models.

    This FPN variant removes the output convolution and uses bicubic interpolation for feature resizing,
    similar to ViT positional embedding interpolation.

    Attributes:
        position_encoding (PositionEmbeddingSine): Sinusoidal positional encoding module.
        convs (nn.ModuleList): List of convolutional layers for each backbone level.
        backbone_channel_list (List[int]): List of channel dimensions from the backbone.
        fpn_interp_model (str): Interpolation mode for FPN feature resizing.
        fuse_type (str): Type of feature fusion, either 'sum' or 'avg'.
        fpn_top_down_levels (List[int]): Levels to have top-down features in outputs.

    Methods:
        forward: Performs forward pass through the FPN neck.

    Examples:
        >>> backbone_channels = [64, 128, 256, 512]
        >>> fpn_neck = FpnNeck(256, backbone_channels)
        >>> inputs = [torch.rand(1, c, 32, 32) for c in backbone_channels]
        >>> outputs, positions = fpn_neck(inputs)
        >>> print(len(outputs), len(positions))
        4 4
    """

    def __init__(self, d_model: 'int', backbone_channel_list: 'List[int]', kernel_size: 'int'=1, stride: 'int'=1, padding: 'int'=0, fpn_interp_model: 'str'='bilinear', fuse_type: 'str'='sum', fpn_top_down_levels: 'Optional[List[int]]'=None):
        """
        Initializes a modified Feature Pyramid Network (FPN) neck.

        This FPN variant removes the output convolution and uses bicubic interpolation for feature resizing,
        similar to ViT positional embedding interpolation.

        Args:
            d_model (int): Dimension of the model.
            backbone_channel_list (List[int]): List of channel dimensions from the backbone.
            kernel_size (int): Kernel size for the convolutional layers.
            stride (int): Stride for the convolutional layers.
            padding (int): Padding for the convolutional layers.
            fpn_interp_model (str): Interpolation mode for FPN feature resizing.
            fuse_type (str): Type of feature fusion, either 'sum' or 'avg'.
            fpn_top_down_levels (Optional[List[int]]): Levels to have top-down features in outputs.

        Examples:
            >>> backbone_channels = [64, 128, 256, 512]
            >>> fpn_neck = FpnNeck(256, backbone_channels)
            >>> print(fpn_neck)
        """
        super().__init__()
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=256)
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module('conv', nn.Conv2d(in_channels=dim, out_channels=d_model, kernel_size=kernel_size, stride=stride, padding=padding))
            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in {'sum', 'avg'}
        self.fuse_type = fuse_type
        if fpn_top_down_levels is None:
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: 'List[torch.Tensor]'):
        """
        Performs forward pass through the Feature Pyramid Network (FPN) neck.

        This method processes a list of input tensors from the backbone through the FPN, applying lateral connections
        and top-down feature fusion. It generates output feature maps and corresponding positional encodings.

        Args:
            xs (List[torch.Tensor]): List of input tensors from the backbone, each with shape (B, C, H, W).

        Returns:
            (Tuple[List[torch.Tensor], List[torch.Tensor]]): A tuple containing:
                - out (List[torch.Tensor]): List of output feature maps after FPN processing, each with shape
                  (B, d_model, H, W).
                - pos (List[torch.Tensor]): List of positional encodings corresponding to each output feature map.

        Examples:
            >>> fpn_neck = FpnNeck(d_model=256, backbone_channel_list=[64, 128, 256, 512])
            >>> inputs = [torch.rand(1, c, 32, 32) for c in [64, 128, 256, 512]]
            >>> outputs, positions = fpn_neck(inputs)
            >>> print(len(outputs), len(positions))
            4 4
        """
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        prev_features = None
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode=self.fpn_interp_model, align_corners=None if self.fpn_interp_model == 'nearest' else False, antialias=False)
                prev_features = lateral_features + top_down_features
                if self.fuse_type == 'avg':
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out)
        return out, pos


class DropPath(nn.Module):
    """
    Implements stochastic depth regularization for neural networks during training.

    Attributes:
        drop_prob (float): Probability of dropping a path during training.
        scale_by_keep (bool): Whether to scale the output by the keep probability.

    Methods:
        forward: Applies stochastic depth to input tensor during training, with optional scaling.

    Examples:
        >>> drop_path = DropPath(drop_prob=0.2, scale_by_keep=True)
        >>> x = torch.randn(32, 64, 224, 224)
        >>> output = drop_path(x)
    """

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        """Initialize DropPath module for stochastic depth regularization during training."""
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Applies stochastic depth to input tensor during training, with optional scaling."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


def do_pool(x: 'torch.Tensor', pool: 'nn.Module', norm: 'nn.Module'=None) ->torch.Tensor:
    """Applies pooling and optional normalization to a tensor, handling spatial dimension permutations."""
    if pool is None:
        return x
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)
    return x


class MultiScaleAttention(nn.Module):
    """
    Implements multi-scale self-attention with optional query pooling for efficient feature extraction.

    This class provides a flexible implementation of multi-scale attention, allowing for optional
    downsampling of query features through pooling. It's designed to enhance the model's ability to
    capture multi-scale information in visual tasks.

    Attributes:
        dim (int): Input dimension of the feature map.
        dim_out (int): Output dimension of the attention module.
        num_heads (int): Number of attention heads.
        scale (float): Scaling factor for dot-product attention.
        q_pool (nn.Module | None): Optional pooling module for query features.
        qkv (nn.Linear): Linear projection for query, key, and value.
        proj (nn.Linear): Output projection.

    Methods:
        forward: Applies multi-scale attention to the input tensor.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> x = torch.randn(1, 64, 64, 256)
        >>> msa = MultiScaleAttention(dim=256, dim_out=256, num_heads=8)
        >>> output = msa(x)
        >>> print(output.shape)
        torch.Size([1, 64, 64, 256])
    """

    def __init__(self, dim: 'int', dim_out: 'int', num_heads: 'int', q_pool: 'nn.Module'=None):
        """Initializes multi-scale attention with optional query pooling for efficient feature extraction."""
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Applies multi-scale attention with optional query pooling to extract multi-scale features."""
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class MultiScaleBlock(nn.Module):
    """
    A multi-scale attention block with window partitioning and query pooling for efficient vision transformers.

    This class implements a multi-scale attention mechanism with optional window partitioning and downsampling,
    designed for use in vision transformer architectures.

    Attributes:
        dim (int): Input dimension of the block.
        dim_out (int): Output dimension of the block.
        norm1 (nn.Module): First normalization layer.
        window_size (int): Size of the window for partitioning.
        pool (nn.Module | None): Pooling layer for query downsampling.
        q_stride (Tuple[int, int] | None): Stride for query pooling.
        attn (MultiScaleAttention): Multi-scale attention module.
        drop_path (nn.Module): Drop path layer for regularization.
        norm2 (nn.Module): Second normalization layer.
        mlp (MLP): Multi-layer perceptron module.
        proj (nn.Linear | None): Projection layer for dimension mismatch.

    Methods:
        forward: Processes input tensor through the multi-scale block.

    Examples:
        >>> block = MultiScaleBlock(dim=256, dim_out=512, num_heads=8, window_size=7)
        >>> x = torch.randn(1, 56, 56, 256)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 28, 28, 512])
    """

    def __init__(self, dim: 'int', dim_out: 'int', num_heads: 'int', mlp_ratio: 'float'=4.0, drop_path: 'float'=0.0, norm_layer: 'Union[nn.Module, str]'='LayerNorm', q_stride: 'Tuple[int, int]'=None, act_layer: 'nn.Module'=nn.GELU, window_size: 'int'=0):
        """Initializes a multi-scale attention block with window partitioning and optional query pooling."""
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-06)
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.window_size = window_size
        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)
        self.attn = MultiScaleAttention(dim, dim_out, num_heads=num_heads, q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2, act=act_layer)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Processes input through multi-scale attention and MLP, with optional windowing and downsampling."""
        shortcut = x
        x = self.norm1(x)
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)
        x = self.attn(x)
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = H + pad_h, W + pad_w
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    """
    Hierarchical vision transformer for efficient multiscale feature extraction in image processing tasks.

    This class implements a Hiera model, which is a hierarchical vision transformer architecture designed for
    efficient multiscale feature extraction. It uses a series of transformer blocks organized into stages,
    with optional pooling and global attention mechanisms.

    Attributes:
        window_spec (Tuple[int, ...]): Window sizes for each stage.
        q_stride (Tuple[int, int]): Downsampling stride between stages.
        stage_ends (List[int]): Indices of the last block in each stage.
        q_pool_blocks (List[int]): Indices of blocks where pooling is applied.
        return_interm_layers (bool): Whether to return intermediate layer outputs.
        patch_embed (PatchEmbed): Module for patch embedding.
        global_att_blocks (Tuple[int, ...]): Indices of blocks with global attention.
        window_pos_embed_bkg_spatial_size (Tuple[int, int]): Spatial size for window positional embedding background.
        pos_embed (nn.Parameter): Positional embedding for the background.
        pos_embed_window (nn.Parameter): Positional embedding for the window.
        blocks (nn.ModuleList): List of MultiScaleBlock modules.
        channel_list (List[int]): List of output channel dimensions for each stage.

    Methods:
        _get_pos_embed: Generates positional embeddings by interpolating and combining window and background embeddings.
        forward: Performs the forward pass through the Hiera model.

    Examples:
        >>> model = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3))
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output_features = model(input_tensor)
        >>> for feat in output_features:
        ...     print(feat.shape)
    """

    def __init__(self, embed_dim: 'int'=96, num_heads: 'int'=1, drop_path_rate: 'float'=0.0, q_pool: 'int'=3, q_stride: 'Tuple[int, int]'=(2, 2), stages: 'Tuple[int, ...]'=(2, 3, 16, 3), dim_mul: 'float'=2.0, head_mul: 'float'=2.0, window_pos_embed_bkg_spatial_size: 'Tuple[int, int]'=(14, 14), window_spec: 'Tuple[int, ...]'=(8, 4, 14, 7), global_att_blocks: 'Tuple[int, ...]'=(12, 16, 20), return_interm_layers=True):
        """Initializes the Hiera model, configuring its hierarchical vision transformer architecture."""
        super().__init__()
        assert len(stages) == len(window_spec)
        self.window_spec = window_spec
        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [(sum(stages[:i]) - 1) for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [(x + 1) for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers
        self.patch_embed = PatchEmbed(embed_dim=embed_dim, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        self.global_att_blocks = global_att_blocks
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0]))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        cur_stage = 1
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
            block = MultiScaleBlock(dim=embed_dim, dim_out=dim_out, num_heads=num_heads, drop_path=dpr[i], q_stride=self.q_stride if i in self.q_pool_blocks else None, window_size=window_size)
            embed_dim = dim_out
            self.blocks.append(block)
        self.channel_list = [self.blocks[i].dim_out for i in self.stage_ends[::-1]] if return_interm_layers else [self.blocks[-1].dim_out]

    def _get_pos_embed(self, hw: 'Tuple[int, int]') ->torch.Tensor:
        """Generates positional embeddings by interpolating and combining window and background embeddings."""
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode='bicubic')
        pos_embed = pos_embed + window_embed.tile([(x // y) for x, y in zip(pos_embed.shape, window_embed.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: 'torch.Tensor') ->List[torch.Tensor]:
        """Performs forward pass through Hiera model, extracting multiscale features from input images."""
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.stage_ends[-1] or i in self.stage_ends and self.return_interm_layers:
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        return outputs


class ImageEncoder(nn.Module):
    """
    Encodes images using a trunk-neck architecture, producing multiscale features and positional encodings.

    This class combines a trunk network for feature extraction with a neck network for feature refinement
    and positional encoding generation. It can optionally discard the lowest resolution features.

    Attributes:
        trunk (nn.Module): The trunk network for initial feature extraction.
        neck (nn.Module): The neck network for feature refinement and positional encoding generation.
        scalp (int): Number of lowest resolution feature levels to discard.

    Methods:
        forward: Processes the input image through the trunk and neck networks.

    Examples:
        >>> trunk = SomeTrunkNetwork()
        >>> neck = SomeNeckNetwork()
        >>> encoder = ImageEncoder(trunk, neck, scalp=1)
        >>> image = torch.randn(1, 3, 224, 224)
        >>> output = encoder(image)
        >>> print(output.keys())
        dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
    """

    def __init__(self, trunk: 'nn.Module', neck: 'nn.Module', scalp: 'int'=0):
        """Initializes the ImageEncoder with trunk and neck networks for feature extraction and refinement."""
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert self.trunk.channel_list == self.neck.backbone_channel_list, f'Channel dims of trunk {self.trunk.channel_list} and neck {self.neck.backbone_channel_list} do not match.'

    def forward(self, sample: 'torch.Tensor'):
        """Encodes input through patch embedding, positional embedding, transformer blocks, and neck module."""
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            features, pos = features[:-self.scalp], pos[:-self.scalp]
        src = features[-1]
        return {'vision_features': src, 'vision_pos_enc': pos, 'backbone_fpn': features}


def reshape_for_broadcast(freqs_cis: 'torch.Tensor', x: 'torch.Tensor'):
    """Reshapes frequency tensor for broadcasting with input tensor, ensuring dimensional compatibility."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [(d if i >= ndim - 2 else 1) for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(xq: 'torch.Tensor', xk: 'torch.Tensor', freqs_cis: 'torch.Tensor', repeat_freqs_k: 'bool'=False):
    """Applies rotary positional encoding to query and key tensors using complex-valued frequency components."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.type_as(xq), xk
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def init_t_xy(end_x: 'int', end_y: 'int'):
    """Initializes 1D and 2D coordinate tensors for a grid of specified dimensions."""
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y


def compute_axial_cis(dim: 'int', end_x: 'int', end_y: 'int', theta: 'float'=10000.0):
    """Computes axial complex exponential positional encodings for 2D spatial positions in a grid."""
    freqs_x = 1.0 / theta ** (torch.arange(0, dim, 4)[:dim // 4].float() / dim)
    freqs_y = 1.0 / theta ** (torch.arange(0, dim, 4)[:dim // 4].float() / dim)
    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


class RoPEAttention(Attention):
    """
    Implements rotary position encoding for attention mechanisms in transformer architectures.

    This class extends the base Attention class by incorporating Rotary Position Encoding (RoPE) to enhance
    the positional awareness of the attention mechanism.

    Attributes:
        compute_cis (Callable): Function to compute axial complex numbers for rotary encoding.
        freqs_cis (Tensor): Precomputed frequency tensor for rotary encoding.
        rope_k_repeat (bool): Flag to repeat query RoPE to match key length for cross-attention to memories.

    Methods:
        forward: Applies rotary position encoding and computes attention between query, key, and value tensors.

    Examples:
        >>> rope_attn = RoPEAttention(embedding_dim=256, num_heads=8, rope_theta=10000.0, feat_sizes=(32, 32))
        >>> q = torch.randn(1, 1024, 256)
        >>> k = torch.randn(1, 1024, 256)
        >>> v = torch.randn(1, 1024, 256)
        >>> output = rope_attn(q, k, v)
        >>> print(output.shape)
        torch.Size([1, 1024, 256])
    """

    def __init__(self, *args, rope_theta=10000.0, rope_k_repeat=False, feat_sizes=(32, 32), **kwargs):
        """Initializes RoPEAttention with rotary position encoding for enhanced positional awareness."""
        super().__init__(*args, **kwargs)
        self.compute_cis = partial(compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta)
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat

    def forward(self, q: 'Tensor', k: 'Tensor', v: 'Tensor', num_k_exclude_rope: 'int'=0) ->Tensor:
        """Applies rotary position encoding and computes attention between query, key, and value tensors."""
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat
        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(q, k[:, :, :num_k_rope], freqs_cis=self.freqs_cis, repeat_freqs_k=self.rope_k_repeat)
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class MemoryAttention(nn.Module):
    """
    Memory attention module for processing sequential data with self and cross-attention mechanisms.

    This class implements a multi-layer attention mechanism that combines self-attention and cross-attention
    for processing sequential data, particularly useful in transformer-like architectures.

    Attributes:
        d_model (int): The dimension of the model's hidden state.
        layers (nn.ModuleList): A list of MemoryAttentionLayer modules.
        num_layers (int): The number of attention layers.
        norm (nn.LayerNorm): Layer normalization applied to the output.
        pos_enc_at_input (bool): Whether to apply positional encoding at the input.
        batch_first (bool): Whether the input tensors are in batch-first format.

    Methods:
        forward: Processes input tensors through the attention layers.

    Examples:
        >>> d_model = 256
        >>> layer = MemoryAttentionLayer(d_model)
        >>> attention = MemoryAttention(d_model, pos_enc_at_input=True, layer=layer, num_layers=3)
        >>> curr = torch.randn(10, 32, d_model)  # (seq_len, batch_size, d_model)
        >>> memory = torch.randn(20, 32, d_model)  # (mem_len, batch_size, d_model)
        >>> curr_pos = torch.randn(10, 32, d_model)
        >>> memory_pos = torch.randn(20, 32, d_model)
        >>> output = attention(curr, memory, curr_pos, memory_pos)
        >>> print(output.shape)
        torch.Size([10, 32, 256])
    """

    def __init__(self, d_model: 'int', pos_enc_at_input: 'bool', layer: 'nn.Module', num_layers: 'int', batch_first: 'bool'=True):
        """Initializes MemoryAttention module with layers and normalization for attention processing."""
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(self, curr: 'torch.Tensor', memory: 'torch.Tensor', curr_pos: 'Optional[Tensor]'=None, memory_pos: 'Optional[Tensor]'=None, num_obj_ptr_tokens: 'int'=0):
        """Processes input tensors through multiple attention layers, applying self and cross-attention mechanisms."""
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]
        assert curr.shape[1] == memory.shape[1], 'Batch size must be the same for curr and memory'
        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos
        if self.batch_first:
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)
        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {'num_k_exclude_rope': num_obj_ptr_tokens}
            output = layer(tgt=output, memory=memory, pos=memory_pos, query_pos=curr_pos, **kwds)
        normed_output = self.norm(output)
        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
        return normed_output


class MemoryAttentionLayer(nn.Module):
    """
    Implements a memory attention layer with self-attention and cross-attention mechanisms for neural networks.

    This class combines self-attention, cross-attention, and feedforward components to process input tensors and
    generate memory-based attention outputs.

    Attributes:
        d_model (int): Dimensionality of the model.
        dim_feedforward (int): Dimensionality of the feedforward network.
        dropout_value (float): Dropout rate for regularization.
        self_attn (RoPEAttention): Self-attention mechanism using RoPE (Rotary Position Embedding).
        cross_attn_image (RoPEAttention): Cross-attention mechanism for image processing.
        linear1 (nn.Linear): First linear layer of the feedforward network.
        linear2 (nn.Linear): Second linear layer of the feedforward network.
        norm1 (nn.LayerNorm): Layer normalization for self-attention output.
        norm2 (nn.LayerNorm): Layer normalization for cross-attention output.
        norm3 (nn.LayerNorm): Layer normalization for feedforward network output.
        dropout1 (nn.Dropout): Dropout layer after self-attention.
        dropout2 (nn.Dropout): Dropout layer after cross-attention.
        dropout3 (nn.Dropout): Dropout layer after feedforward network.
        activation (nn.ReLU): Activation function for the feedforward network.
        pos_enc_at_attn (bool): Flag to add positional encoding at attention.
        pos_enc_at_cross_attn_queries (bool): Flag to add positional encoding to cross-attention queries.
        pos_enc_at_cross_attn_keys (bool): Flag to add positional encoding to cross-attention keys.

    Methods:
        forward: Performs the full memory attention operation on input tensors.
        _forward_sa: Performs self-attention on input tensor.
        _forward_ca: Performs cross-attention between target and memory tensors.

    Examples:
        >>> layer = MemoryAttentionLayer(d_model=256, dim_feedforward=2048, dropout=0.1)
        >>> tgt = torch.randn(1, 100, 256)
        >>> memory = torch.randn(1, 100, 64)
        >>> pos = torch.randn(1, 100, 256)
        >>> query_pos = torch.randn(1, 100, 256)
        >>> output = layer(tgt, memory, pos, query_pos)
        >>> print(output.shape)
        torch.Size([1, 100, 256])
    """

    def __init__(self, d_model: 'int'=256, dim_feedforward: 'int'=2048, dropout: 'float'=0.1, pos_enc_at_attn: 'bool'=False, pos_enc_at_cross_attn_keys: 'bool'=True, pos_enc_at_cross_attn_queries: 'bool'=False):
        """Initializes a memory attention layer with self-attention, cross-attention, and feedforward components."""
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = RoPEAttention(embedding_dim=256, num_heads=1, downsample_rate=1)
        self.cross_attn_image = RoPEAttention(rope_k_repeat=True, embedding_dim=256, num_heads=1, downsample_rate=1, kv_in_dim=64)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        """Performs self-attention on input tensor using positional encoding and RoPE attention mechanism."""
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        """Performs cross-attention between target and memory tensors using RoPEAttention mechanism."""
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {'num_k_exclude_rope': num_k_exclude_rope}
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2, k=memory + pos if self.pos_enc_at_cross_attn_keys else memory, v=memory, **kwds)
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(self, tgt, memory, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None, num_k_exclude_rope: 'int'=0) ->torch.Tensor:
        """Processes input tensors using self-attention, cross-attention, and MLP for memory-based attention."""
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class CXBlock(nn.Module):
    """
    ConvNeXt Block for efficient feature extraction in convolutional neural networks.

    This block implements a modified version of the ConvNeXt architecture, offering improved performance and
    flexibility in feature extraction.

    Attributes:
        dwconv (nn.Conv2d): Depthwise or standard 2D convolution layer.
        norm (LayerNorm2d): Layer normalization applied to channels.
        pwconv1 (nn.Linear): First pointwise convolution implemented as a linear layer.
        act (nn.GELU): GELU activation function.
        pwconv2 (nn.Linear): Second pointwise convolution implemented as a linear layer.
        gamma (nn.Parameter | None): Learnable scale parameter for layer scaling.
        drop_path (nn.Module): DropPath layer for stochastic depth regularization.

    Methods:
        forward: Processes the input tensor through the ConvNeXt block.

    Examples:
        >>> import torch
        >>> x = torch.randn(1, 64, 56, 56)
        >>> block = CXBlock(dim=64, kernel_size=7, padding=3)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 64, 56, 56])
    """

    def __init__(self, dim, kernel_size=7, padding=3, drop_path=0.0, layer_scale_init_value=1e-06, use_dwconv=True):
        """
        Initialize a ConvNeXt Block for efficient feature extraction in convolutional neural networks.

        This block implements a modified version of the ConvNeXt architecture, offering improved performance and
        flexibility in feature extraction.

        Args:
            dim (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Padding size for the convolution.
            drop_path (float): Stochastic depth rate.
            layer_scale_init_value (float): Initial value for Layer Scale.
            use_dwconv (bool): Whether to use depthwise convolution.

        Examples:
            >>> block = CXBlock(dim=64, kernel_size=7, padding=3)
            >>> x = torch.randn(1, 64, 32, 32)
            >>> output = block(x)
            >>> print(output.shape)
            torch.Size([1, 64, 32, 32])
        """
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim if use_dwconv else 1)
        self.norm = LayerNorm2d(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """Applies ConvNeXt block operations to input tensor, including convolutions and residual connection."""
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class Fuser(nn.Module):
    """
    A module for fusing features through multiple layers of a neural network.

    This class applies a series of identical layers to an input tensor, optionally projecting the input first.

    Attributes:
        proj (nn.Module): An optional input projection layer. Identity if no projection is needed.
        layers (nn.ModuleList): A list of identical layers to be applied sequentially.

    Methods:
        forward: Applies the fuser to an input tensor.

    Examples:
        >>> layer = CXBlock(dim=256)
        >>> fuser = Fuser(layer, num_layers=3, dim=256, input_projection=True)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = fuser(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        """
        Initializes the Fuser module for feature fusion through multiple layers.

        This module creates a sequence of identical layers and optionally applies an input projection.

        Args:
            layer (nn.Module): The layer to be replicated in the fuser.
            num_layers (int): The number of times to replicate the layer.
            dim (int | None): The dimension for input projection, if used.
            input_projection (bool): Whether to use input projection.

        Examples:
            >>> layer = nn.Linear(64, 64)
            >>> fuser = Fuser(layer, num_layers=3, dim=64, input_projection=True)
            >>> input_tensor = torch.randn(1, 64)
            >>> output = fuser(input_tensor)
        """
        super().__init__()
        self.proj = nn.Identity()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        """Applies a series of layers to the input tensor, optionally projecting it first."""
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MaskDownSampler(nn.Module):
    """
    A mask downsampling and embedding module for efficient processing of input masks.

    This class implements a mask downsampler that progressively reduces the spatial dimensions of input masks
    while expanding their channel dimensions using convolutional layers, layer normalization, and activation
    functions.

    Attributes:
        encoder (nn.Sequential): A sequential container of convolutional layers, layer normalization, and
            activation functions for downsampling and embedding masks.

    Methods:
        forward: Downsamples and encodes input mask to embed_dim channels.

    Examples:
        >>> mask_downsampler = MaskDownSampler(embed_dim=256, kernel_size=4, stride=4, padding=0, total_stride=16)
        >>> input_mask = torch.randn(1, 1, 256, 256)
        >>> output = mask_downsampler(input_mask)
        >>> print(output.shape)
        torch.Size([1, 256, 16, 16])
    """

    def __init__(self, embed_dim=256, kernel_size=4, stride=4, padding=0, total_stride=16, activation=nn.GELU):
        """Initializes a mask downsampler module for progressive downsampling and channel expansion."""
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride ** num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * stride ** 2
            self.encoder.append(nn.Conv2d(mask_in_chans, mask_out_chans, kernel_size=kernel_size, stride=stride, padding=padding))
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans
        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))

    def forward(self, x):
        """Downsamples and encodes input mask to embed_dim channels using convolutional layers and LayerNorm2d."""
        return self.encoder(x)


class MemoryEncoder(nn.Module):
    """
    Encodes pixel features and masks into a memory representation for efficient image segmentation.

    This class processes pixel-level features and masks, fusing them to generate encoded memory representations
    suitable for downstream tasks in image segmentation models like SAM (Segment Anything Model).

    Attributes:
        mask_downsampler (MaskDownSampler): Module for downsampling input masks.
        pix_feat_proj (nn.Conv2d): Convolutional layer for projecting pixel features.
        fuser (Fuser): Module for fusing pixel features and masks.
        position_encoding (PositionEmbeddingSine): Module for adding positional encoding to features.
        out_proj (nn.Module): Output projection layer, either nn.Identity or nn.Conv2d.

    Methods:
        forward: Processes input pixel features and masks to generate encoded memory representations.

    Examples:
        >>> import torch
        >>> encoder = MemoryEncoder(out_dim=256, in_dim=256)
        >>> pix_feat = torch.randn(1, 256, 64, 64)
        >>> masks = torch.randn(1, 1, 64, 64)
        >>> encoded_feat, pos = encoder(pix_feat, masks)
        >>> print(encoded_feat.shape, pos.shape)
        torch.Size([1, 256, 64, 64]) torch.Size([1, 128, 64, 64])
    """

    def __init__(self, out_dim, in_dim=256):
        """Initializes the MemoryEncoder for encoding pixel features and masks into memory representations."""
        super().__init__()
        self.mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1)
        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = Fuser(CXBlock(dim=256), num_layers=2)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=64)
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, pix_feat: 'torch.Tensor', masks: 'torch.Tensor', skip_mask_sigmoid: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor]:
        """Processes pixel features and masks to generate encoded memory representations for segmentation."""
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)
        pix_feat = pix_feat
        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)
        pos = self.position_encoding(x)
        return {'vision_features': x, 'vision_pos_enc': [pos]}


NO_OBJ_SCORE = -1024.0


class SAM2MaskDecoder(nn.Module):
    """
    Transformer-based decoder for predicting instance segmentation masks from image and prompt embeddings.

    This class extends the functionality of the MaskDecoder, incorporating additional features such as
    high-resolution feature processing, dynamic multimask output, and object score prediction.

    Attributes:
        transformer_dim (int): Channel dimension of the transformer.
        transformer (nn.Module): Transformer used to predict masks.
        num_multimask_outputs (int): Number of masks to predict when disambiguating masks.
        iou_token (nn.Embedding): Embedding for IOU token.
        num_mask_tokens (int): Total number of mask tokens.
        mask_tokens (nn.Embedding): Embedding for mask tokens.
        pred_obj_scores (bool): Whether to predict object scores.
        obj_score_token (nn.Embedding): Embedding for object score token.
        use_multimask_token_for_obj_ptr (bool): Whether to use multimask token for object pointer.
        output_upscaling (nn.Sequential): Upscaling layers for output.
        use_high_res_features (bool): Whether to use high-resolution features.
        conv_s0 (nn.Conv2d): Convolutional layer for high-resolution features (s0).
        conv_s1 (nn.Conv2d): Convolutional layer for high-resolution features (s1).
        output_hypernetworks_mlps (nn.ModuleList): List of MLPs for output hypernetworks.
        iou_prediction_head (MLP): MLP for IOU prediction.
        pred_obj_score_head (nn.Linear | MLP): Linear layer or MLP for object score prediction.
        dynamic_multimask_via_stability (bool): Whether to use dynamic multimask via stability.
        dynamic_multimask_stability_delta (float): Delta value for dynamic multimask stability.
        dynamic_multimask_stability_thresh (float): Threshold for dynamic multimask stability.

    Methods:
        forward: Predicts masks given image and prompt embeddings.
        predict_masks: Predicts instance segmentation masks from image and prompt embeddings.
        _get_stability_scores: Computes mask stability scores based on IoU between thresholds.
        _dynamic_multimask_via_stability: Dynamically selects the most stable mask output.

    Examples:
        >>> image_embeddings = torch.rand(1, 256, 64, 64)
        >>> image_pe = torch.rand(1, 256, 64, 64)
        >>> sparse_prompt_embeddings = torch.rand(1, 2, 256)
        >>> dense_prompt_embeddings = torch.rand(1, 256, 64, 64)
        >>> decoder = SAM2MaskDecoder(256, transformer)
        >>> masks, iou_pred, sam_tokens_out, obj_score_logits = decoder.forward(
        ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, True, False
        ... )
    """

    def __init__(self, transformer_dim: 'int', transformer: 'nn.Module', num_multimask_outputs: 'int'=3, activation: 'Type[nn.Module]'=nn.GELU, iou_head_depth: 'int'=3, iou_head_hidden_dim: 'int'=256, use_high_res_features: 'bool'=False, iou_prediction_use_sigmoid=False, dynamic_multimask_via_stability=False, dynamic_multimask_stability_delta=0.05, dynamic_multimask_stability_thresh=0.98, pred_obj_scores: 'bool'=False, pred_obj_scores_mlp: 'bool'=False, use_multimask_token_for_obj_ptr: 'bool'=False) ->None:
        """
        Initializes the SAM2MaskDecoder module for predicting instance segmentation masks.

        This decoder extends the functionality of MaskDecoder, incorporating additional features such as
        high-resolution feature processing, dynamic multimask output, and object score prediction.

        Args:
            transformer_dim (int): Channel dimension of the transformer.
            transformer (nn.Module): Transformer used to predict masks.
            num_multimask_outputs (int): Number of masks to predict when disambiguating masks.
            activation (Type[nn.Module]): Type of activation to use when upscaling masks.
            iou_head_depth (int): Depth of the MLP used to predict mask quality.
            iou_head_hidden_dim (int): Hidden dimension of the MLP used to predict mask quality.
            use_high_res_features (bool): Whether to use high-resolution features.
            iou_prediction_use_sigmoid (bool): Whether to use sigmoid for IOU prediction.
            dynamic_multimask_via_stability (bool): Whether to use dynamic multimask via stability.
            dynamic_multimask_stability_delta (float): Delta value for dynamic multimask stability.
            dynamic_multimask_stability_thresh (float): Threshold for dynamic multimask stability.
            pred_obj_scores (bool): Whether to predict object scores.
            pred_obj_scores_mlp (bool): Whether to use MLP for object score prediction.
            use_multimask_token_for_obj_ptr (bool): Whether to use multimask token for object pointer.

        Examples:
            >>> transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=8), num_layers=6)
            >>> decoder = SAM2MaskDecoder(transformer_dim=256, transformer=transformer)
            >>> print(decoder)
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.output_upscaling = nn.Sequential(nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2), LayerNorm2d(transformer_dim // 4), activation(), nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2), activation())
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, stride=1)
            self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1)
        self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)])
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth, sigmoid=iou_prediction_use_sigmoid)
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(self, image_embeddings: 'torch.Tensor', image_pe: 'torch.Tensor', sparse_prompt_embeddings: 'torch.Tensor', dense_prompt_embeddings: 'torch.Tensor', multimask_output: 'bool', repeat_image: 'bool', high_res_features: 'Optional[List[torch.Tensor]]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks given image and prompt embeddings.

        Args:
            image_embeddings (torch.Tensor): Embeddings from the image encoder with shape (B, C, H, W).
            image_pe (torch.Tensor): Positional encoding with the shape of image_embeddings (B, C, H, W).
            sparse_prompt_embeddings (torch.Tensor): Embeddings of the points and boxes with shape (B, N, C).
            dense_prompt_embeddings (torch.Tensor): Embeddings of the mask inputs with shape (B, C, H, W).
            multimask_output (bool): Whether to return multiple masks or a single mask.
            repeat_image (bool): Flag to repeat the image embeddings.
            high_res_features (List[torch.Tensor] | None): Optional high-resolution features.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing:
                - masks (torch.Tensor): Batched predicted masks with shape (B, N, H, W).
                - iou_pred (torch.Tensor): Batched predictions of mask quality with shape (B, N).
                - sam_tokens_out (torch.Tensor): Batched SAM token for mask output with shape (B, N, C).
                - object_score_logits (torch.Tensor): Batched object score logits with shape (B, 1).

        Examples:
            >>> image_embeddings = torch.rand(1, 256, 64, 64)
            >>> image_pe = torch.rand(1, 256, 64, 64)
            >>> sparse_prompt_embeddings = torch.rand(1, 2, 256)
            >>> dense_prompt_embeddings = torch.rand(1, 256, 64, 64)
            >>> decoder = SAM2MaskDecoder(256, transformer)
            >>> masks, iou_pred, sam_tokens_out, obj_score_logits = decoder.forward(
            ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, True, False
            ... )
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(image_embeddings=image_embeddings, image_pe=image_pe, sparse_prompt_embeddings=sparse_prompt_embeddings, dense_prompt_embeddings=dense_prompt_embeddings, repeat_image=repeat_image, high_res_features=high_res_features)
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]
        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(self, image_embeddings: 'torch.Tensor', image_pe: 'torch.Tensor', sparse_prompt_embeddings: 'torch.Tensor', dense_prompt_embeddings: 'torch.Tensor', repeat_image: 'bool', high_res_features: 'Optional[List[torch.Tensor]]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """Predicts instance segmentation masks from image and prompt embeddings using a transformer."""
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat([self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0)
            s = 1
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert image_pe.size(0) == 1, 'image_pe should have size 1 in batch dim (from `get_dense_pe()`)'
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1:s + 1 + self.num_mask_tokens, :]
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)
        hyper_in_list: 'List[torch.Tensor]' = [self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(self.num_mask_tokens)]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)
        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """Computes mask stability scores based on IoU between upper and lower thresholds."""
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        return torch.where(area_u > 0, area_i / area_u, 1.0)

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        Dynamically selects the most stable mask output based on stability scores and IoU predictions.

        This method is used when outputting a single mask. If the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, it instead selects from multi-mask outputs
        (based on output tokens 1-3) the mask with the highest predicted IoU score. This ensures a valid mask
        for both clicking and tracking scenarios.

        Args:
            all_mask_logits (torch.Tensor): Logits for all predicted masks, shape (B, N, H, W) where B is
                batch size, N is number of masks (typically 4), and H, W are mask dimensions.
            all_iou_scores (torch.Tensor): Predicted IoU scores for all masks, shape (B, N).

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]):
                - mask_logits_out (torch.Tensor): Selected mask logits, shape (B, 1, H, W).
                - iou_scores_out (torch.Tensor): Selected IoU scores, shape (B, 1).

        Examples:
            >>> decoder = SAM2MaskDecoder(...)
            >>> all_mask_logits = torch.rand(2, 4, 256, 256)  # 2 images, 4 masks each
            >>> all_iou_scores = torch.rand(2, 4)
            >>> mask_logits, iou_scores = decoder._dynamic_multimask_via_stability(all_mask_logits, all_iou_scores)
            >>> print(mask_logits.shape, iou_scores.shape)
            torch.Size([2, 1, 256, 256]) torch.Size([2, 1])
        """
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(multimask_iou_scores.size(0), device=all_iou_scores.device)
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh
        mask_logits_out = torch.where(is_stable[..., None, None].expand_as(singlemask_logits), singlemask_logits, best_multimask_logits)
        iou_scores_out = torch.where(is_stable.expand_as(singlemask_iou_scores), singlemask_iou_scores, best_multimask_iou_scores)
        return mask_logits_out, iou_scores_out


class SAM2TwoWayAttentionBlock(TwoWayAttentionBlock):
    """
    A two-way attention block for performing self-attention and cross-attention in both directions.

    This block extends the TwoWayAttentionBlock and consists of four main components: self-attention on
    sparse inputs, cross-attention from sparse to dense inputs, an MLP block on sparse inputs, and
    cross-attention from dense to sparse inputs.

    Attributes:
        self_attn (Attention): Self-attention layer for queries.
        norm1 (nn.LayerNorm): Layer normalization after the first attention block.
        cross_attn_token_to_image (Attention): Cross-attention layer from queries to keys.
        norm2 (nn.LayerNorm): Layer normalization after the second attention block.
        mlp (MLP): MLP block for transforming query embeddings.
        norm3 (nn.LayerNorm): Layer normalization after the MLP block.
        norm4 (nn.LayerNorm): Layer normalization after the third attention block.
        cross_attn_image_to_token (Attention): Cross-attention layer from keys to queries.
        skip_first_layer_pe (bool): Flag to skip positional encoding in the first layer.

    Methods:
        forward: Processes input through the attention blocks and MLP.

    Examples:
        >>> block = SAM2TwoWayAttentionBlock(embedding_dim=256, num_heads=8)
        >>> sparse_input = torch.randn(1, 100, 256)
        >>> dense_input = torch.randn(1, 256, 16, 16)
        >>> sparse_output, dense_output = block(sparse_input, dense_input)
    """

    def __init__(self, embedding_dim: 'int', num_heads: 'int', mlp_dim: 'int'=2048, activation: 'Type[nn.Module]'=nn.ReLU, attention_downsample_rate: 'int'=2, skip_first_layer_pe: 'bool'=False) ->None:
        """
        Initializes a SAM2TwoWayAttentionBlock for performing self-attention and cross-attention in two directions.

        This block extends the TwoWayAttentionBlock and consists of four main components: self-attention on sparse
        inputs, cross-attention from sparse to dense inputs, an MLP block on sparse inputs, and cross-attention
        from dense to sparse inputs.

        Args:
            embedding_dim (int): The channel dimension of the embeddings.
            num_heads (int): The number of heads in the attention layers.
            mlp_dim (int): The hidden dimension of the MLP block.
            activation (Type[nn.Module]): The activation function of the MLP block.
            attention_downsample_rate (int): The downsample rate for attention computations.
            skip_first_layer_pe (bool): Whether to skip the positional encoding in the first layer.

        Examples:
            >>> block = SAM2TwoWayAttentionBlock(embedding_dim=256, num_heads=8, mlp_dim=2048)
            >>> sparse_inputs = torch.randn(1, 100, 256)
            >>> dense_inputs = torch.randn(1, 256, 32, 32)
            >>> sparse_outputs, dense_outputs = block(sparse_inputs, dense_inputs)
        """
        super().__init__(embedding_dim, num_heads, mlp_dim, activation, attention_downsample_rate, skip_first_layer_pe)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2, act=activation)


class SAM2TwoWayTransformer(TwoWayTransformer):
    """
    A Two-Way Transformer module for simultaneous attention to image and query points.

    This class extends the TwoWayTransformer, implementing a specialized transformer decoder that attends to an
    input image using queries with supplied positional embeddings. It is particularly useful for tasks like
    object detection, image segmentation, and point cloud processing.

    Attributes:
        depth (int): Number of layers in the transformer.
        embedding_dim (int): Channel dimension for input embeddings.
        num_heads (int): Number of heads for multihead attention.
        mlp_dim (int): Internal channel dimension for the MLP block.
        layers (nn.ModuleList): List of SAM2TwoWayAttentionBlock layers comprising the transformer.
        final_attn_token_to_image (Attention): Final attention layer from queries to image.
        norm_final_attn (nn.LayerNorm): Layer normalization applied to final queries.

    Methods:
        forward: Processes input image embeddings and query embeddings through the transformer.

    Examples:
        >>> transformer = SAM2TwoWayTransformer(depth=5, embedding_dim=256, num_heads=8, mlp_dim=2048)
        >>> image_embedding = torch.randn(1, 256, 64, 64)
        >>> query_embedding = torch.randn(1, 100, 256)
        >>> output = transformer(image_embedding, query_embedding)
        >>> print(output[0].shape, output[1].shape)
        torch.Size([1, 100, 256]) torch.Size([1, 256, 64, 64])
    """

    def __init__(self, depth: 'int', embedding_dim: 'int', num_heads: 'int', mlp_dim: 'int', activation: 'Type[nn.Module]'=nn.ReLU, attention_downsample_rate: 'int'=2) ->None:
        """
        Initializes a SAM2TwoWayTransformer instance.

        This transformer decoder attends to an input image using queries with supplied positional embeddings.
        It is designed for tasks like object detection, image segmentation, and point cloud processing.

        Args:
            depth (int): Number of layers in the transformer.
            embedding_dim (int): Channel dimension for the input embeddings.
            num_heads (int): Number of heads for multihead attention. Must divide embedding_dim.
            mlp_dim (int): Channel dimension internal to the MLP block.
            activation (Type[nn.Module]): Activation function to use in the MLP block.
            attention_downsample_rate (int): Downsampling rate for attention computations.

        Examples:
            >>> transformer = SAM2TwoWayTransformer(depth=5, embedding_dim=256, num_heads=8, mlp_dim=2048)
            >>> transformer
            SAM2TwoWayTransformer(
              (layers): ModuleList(
                (0-4): 5 x SAM2TwoWayAttentionBlock(...)
              )
              (final_attn_token_to_image): Attention(...)
              (norm_final_attn): LayerNorm(...)
            )
        """
        super().__init__(depth, embedding_dim, num_heads, mlp_dim, activation, attention_downsample_rate)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(SAM2TwoWayAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_dim=mlp_dim, activation=activation, attention_downsample_rate=attention_downsample_rate, skip_first_layer_pe=i == 0))


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """Generates 1D sinusoidal positional embeddings for given positions and dimensions."""
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Selects the closest conditioning frames to a given frame index.

    Args:
        frame_idx (int): Current frame index.
        cond_frame_outputs (Dict[int, Any]): Dictionary of conditioning frame outputs keyed by frame indices.
        max_cond_frame_num (int): Maximum number of conditioning frames to select.

    Returns:
        (Tuple[Dict[int, Any], Dict[int, Any]]): A tuple containing two dictionaries:
            - selected_outputs: Selected items from cond_frame_outputs.
            - unselected_outputs: Items not selected from cond_frame_outputs.

    Examples:
        >>> frame_idx = 5
        >>> cond_frame_outputs = {1: "a", 3: "b", 7: "c", 9: "d"}
        >>> max_cond_frame_num = 2
        >>> selected, unselected = select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num)
        >>> print(selected)
        {3: 'b', 7: 'c'}
        >>> print(unselected)
        {1: 'a', 9: 'd'}
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, 'we should allow using 2+ conditioning frames'
        selected_outputs = {}
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted((t for t in cond_frame_outputs if t not in selected_outputs), key=lambda x: abs(x - frame_idx))[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs}
    return selected_outputs, unselected_outputs


class SAM2Model(torch.nn.Module):
    """
    SAM2Model class for Segment Anything Model 2 with memory-based video object segmentation capabilities.

    This class extends the functionality of SAM to handle video sequences, incorporating memory mechanisms
    for temporal consistency and efficient tracking of objects across frames.

    Attributes:
        mask_threshold (float): Threshold value for mask prediction.
        image_encoder (ImageEncoderViT): Visual encoder for extracting image features.
        memory_attention (nn.Module): Module for attending to memory features.
        memory_encoder (nn.Module): Encoder for generating memory representations.
        num_maskmem (int): Number of accessible memory frames.
        image_size (int): Size of input images.
        backbone_stride (int): Stride of the backbone network output.
        sam_prompt_embed_dim (int): Dimension of SAM prompt embeddings.
        sam_image_embedding_size (int): Size of SAM image embeddings.
        sam_prompt_encoder (PromptEncoder): Encoder for processing input prompts.
        sam_mask_decoder (SAM2MaskDecoder): Decoder for generating object masks.
        obj_ptr_proj (nn.Module): Projection layer for object pointers.
        obj_ptr_tpos_proj (nn.Module): Projection for temporal positional encoding in object pointers.

    Methods:
        forward_image: Processes image batch through encoder to extract multi-level features.
        track_step: Performs a single tracking step, updating object masks and memory features.

    Examples:
        >>> model = SAM2Model(image_encoder, memory_attention, memory_encoder)
        >>> image_batch = torch.rand(1, 3, 512, 512)
        >>> features = model.forward_image(image_batch)
        >>> track_results = model.track_step(0, True, features, None, None, None, {})
    """
    mask_threshold: 'float' = 0.0

    def __init__(self, image_encoder, memory_attention, memory_encoder, num_maskmem=7, image_size=512, backbone_stride=16, sigmoid_scale_for_mem_enc=1.0, sigmoid_bias_for_mem_enc=0.0, binarize_mask_from_pts_for_mem_enc=False, use_mask_input_as_output_without_sam=False, max_cond_frames_in_attn=-1, directly_add_no_mem_embed=False, use_high_res_features_in_sam=False, multimask_output_in_sam=False, multimask_min_pt_num=1, multimask_max_pt_num=1, multimask_output_for_tracking=False, use_multimask_token_for_obj_ptr: 'bool'=False, iou_prediction_use_sigmoid=False, memory_temporal_stride_for_eval=1, non_overlap_masks_for_mem_enc=False, use_obj_ptrs_in_encoder=False, max_obj_ptrs_in_encoder=16, add_tpos_enc_to_obj_ptrs=True, proj_tpos_enc_in_obj_ptrs=False, use_signed_tpos_enc_to_obj_ptrs=False, only_obj_ptrs_in_the_past_for_eval=False, pred_obj_scores: 'bool'=False, pred_obj_scores_mlp: 'bool'=False, fixed_no_obj_ptr: 'bool'=False, soft_no_obj_ptr: 'bool'=False, use_mlp_for_obj_ptr_proj: 'bool'=False, no_obj_embed_spatial: 'bool'=False, sam_mask_decoder_extra_args=None, compile_image_encoder: 'bool'=False):
        """
        Initializes the SAM2Model for video object segmentation with memory-based tracking.

        Args:
            image_encoder (nn.Module): Visual encoder for extracting image features.
            memory_attention (nn.Module): Module for attending to memory features.
            memory_encoder (nn.Module): Encoder for generating memory representations.
            num_maskmem (int): Number of accessible memory frames. Default is 7 (1 input frame + 6 previous frames).
            image_size (int): Size of input images.
            backbone_stride (int): Stride of the image backbone output.
            sigmoid_scale_for_mem_enc (float): Scale factor for mask sigmoid probability.
            sigmoid_bias_for_mem_enc (float): Bias factor for mask sigmoid probability.
            binarize_mask_from_pts_for_mem_enc (bool): Whether to binarize sigmoid mask logits on interacted frames
                with clicks during evaluation.
            use_mask_input_as_output_without_sam (bool): Whether to directly output the input mask without using SAM
                prompt encoder and mask decoder on frames with mask input.
            max_cond_frames_in_attn (int): Maximum number of conditioning frames to participate in memory attention.
                -1 means no limit.
            directly_add_no_mem_embed (bool): Whether to directly add no-memory embedding to image feature on the
                first frame.
            use_high_res_features_in_sam (bool): Whether to use high-resolution feature maps in the SAM mask decoder.
            multimask_output_in_sam (bool): Whether to output multiple (3) masks for the first click on initial
                conditioning frames.
            multimask_min_pt_num (int): Minimum number of clicks to use multimask output in SAM.
            multimask_max_pt_num (int): Maximum number of clicks to use multimask output in SAM.
            multimask_output_for_tracking (bool): Whether to use multimask output for tracking.
            use_multimask_token_for_obj_ptr (bool): Whether to use multimask tokens for object pointers.
            iou_prediction_use_sigmoid (bool): Whether to use sigmoid to restrict IoU prediction to [0-1].
            memory_temporal_stride_for_eval (int): Memory bank's temporal stride during evaluation.
            non_overlap_masks_for_mem_enc (bool): Whether to apply non-overlapping constraints on object masks in
                memory encoder during evaluation.
            use_obj_ptrs_in_encoder (bool): Whether to cross-attend to object pointers from other frames in the encoder.
            max_obj_ptrs_in_encoder (int): Maximum number of object pointers from other frames in encoder
                cross-attention.
            add_tpos_enc_to_obj_ptrs (bool): Whether to add temporal positional encoding to object pointers in
                the encoder.
            proj_tpos_enc_in_obj_ptrs (bool): Whether to add an extra linear projection layer for temporal positional
                encoding in object pointers.
            use_signed_tpos_enc_to_obj_ptrs (bool): whether to use signed distance (instead of unsigned absolute distance)
                in the temporal positional encoding in the object pointers, only relevant when both `use_obj_ptrs_in_encoder=True`
                and `add_tpos_enc_to_obj_ptrs=True`.
            only_obj_ptrs_in_the_past_for_eval (bool): Whether to only attend to object pointers in the past
                during evaluation.
            pred_obj_scores (bool): Whether to predict if there is an object in the frame.
            pred_obj_scores_mlp (bool): Whether to use an MLP to predict object scores.
            fixed_no_obj_ptr (bool): Whether to have a fixed no-object pointer when there is no object present.
            soft_no_obj_ptr (bool): Whether to mix in no-object pointer softly for easier recovery and error mitigation.
            use_mlp_for_obj_ptr_proj (bool): Whether to use MLP for object pointer projection.
            no_obj_embed_spatial (bool): Whether add no obj embedding to spatial frames.
            sam_mask_decoder_extra_args (Dict | None): Extra arguments for constructing the SAM mask decoder.
            compile_image_encoder (bool): Whether to compile the image encoder for faster inference.

        Examples:
            >>> image_encoder = ImageEncoderViT(...)
            >>> memory_attention = SAM2TwoWayTransformer(...)
            >>> memory_encoder = nn.Sequential(...)
            >>> model = SAM2Model(image_encoder, memory_attention, memory_encoder)
            >>> image_batch = torch.rand(1, 3, 512, 512)
            >>> features = model.forward_image(image_batch)
            >>> track_results = model.track_step(0, True, features, None, None, None, {})
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval
        self.memory_attention = memory_attention
        self.hidden_dim = memory_attention.d_model
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, 'out_proj') and hasattr(self.memory_encoder.out_proj, 'weight'):
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem
        self.maskmem_tpos_enc = torch.nn.Parameter(torch.zeros(num_maskmem, 1, 1, self.mem_dim))
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)
        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        if compile_image_encoder:
            None
            self.image_encoder.forward = torch.compile(self.image_encoder.forward, mode='max-autotune', fullgraph=True, dynamic=False)

    @property
    def device(self):
        """Returns the device on which the model's parameters are stored."""
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        """Processes image and prompt inputs to generate object masks and scores in video sequences."""
        raise NotImplementedError('Please use the corresponding methods in SAM2VideoPredictor for inference.See notebooks/video_predictor_example.ipynb for an example.')

    def _build_sam_heads(self):
        """Builds SAM-style prompt encoder and mask decoder for image segmentation tasks."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        self.sam_prompt_encoder = PromptEncoder(embed_dim=self.sam_prompt_embed_dim, image_embedding_size=(self.sam_image_embedding_size, self.sam_image_embedding_size), input_image_size=(self.image_size, self.image_size), mask_in_chans=16)
        self.sam_mask_decoder = SAM2MaskDecoder(num_multimask_outputs=3, transformer=SAM2TwoWayTransformer(depth=2, embedding_dim=self.sam_prompt_embed_dim, mlp_dim=2048, num_heads=8), transformer_dim=self.sam_prompt_embed_dim, iou_head_depth=3, iou_head_hidden_dim=256, use_high_res_features=self.use_high_res_features_in_sam, iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid, pred_obj_scores=self.pred_obj_scores, pred_obj_scores_mlp=self.pred_obj_scores_mlp, use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr, **self.sam_mask_decoder_extra_args or {})
        if self.use_obj_ptrs_in_encoder:
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(self, backbone_features, point_inputs=None, mask_inputs=None, high_res_features=None, multimask_output=False):
        """
        Forward pass through SAM prompt encoders and mask heads.

        This method processes image features and optional point/mask inputs to generate object masks and scores.

        Args:
            backbone_features (torch.Tensor): Image features with shape (B, C, H, W).
            point_inputs (Dict[str, torch.Tensor] | None): Dictionary containing point prompts.
                'point_coords': Tensor of shape (B, P, 2) with float32 dtype, containing absolute
                    pixel-unit coordinates in (x, y) format for P input points.
                'point_labels': Tensor of shape (B, P) with int32 dtype, where 1 means positive clicks,
                    0 means negative clicks, and -1 means padding.
            mask_inputs (torch.Tensor | None): Mask of shape (B, 1, H*16, W*16), float or bool, with the
                same spatial size as the image.
            high_res_features (List[torch.Tensor] | None): List of two feature maps with shapes
                (B, C, 4*H, 4*W) and (B, C, 2*H, 2*W) respectively, used as high-resolution feature maps
                for SAM decoder.
            multimask_output (bool): If True, output 3 candidate masks and their IoU estimates; if False,
                output only 1 mask and its IoU estimate.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
                low_res_multimasks: Tensor of shape (B, M, H*4, W*4) with SAM output mask logits.
                high_res_multimasks: Tensor of shape (B, M, H*16, W*16) with upsampled mask logits.
                ious: Tensor of shape (B, M) with estimated IoU for each output mask.
                low_res_masks: Tensor of shape (B, 1, H*4, W*4) with best low-resolution mask.
                high_res_masks: Tensor of shape (B, 1, H*16, W*16) with best high-resolution mask.
                obj_ptr: Tensor of shape (B, C) with object pointer vector for the output mask.
                object_score_logits: Tensor of shape (B,) with object score logits.

            Where M is 3 if multimask_output=True, and 1 if multimask_output=False.

        Examples:
            >>> backbone_features = torch.rand(1, 256, 32, 32)
            >>> point_inputs = {"point_coords": torch.rand(1, 2, 2), "point_labels": torch.tensor([[1, 0]])}
            >>> mask_inputs = torch.rand(1, 1, 512, 512)
            >>> results = model._forward_sam_heads(backbone_features, point_inputs, mask_inputs)
            >>> (
            ...     low_res_multimasks,
            ...     high_res_multimasks,
            ...     ious,
            ...     low_res_masks,
            ...     high_res_masks,
            ...     obj_ptr,
            ...     object_score_logits,
            ... ) = results
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size
        if point_inputs is not None:
            sam_point_coords = point_inputs['point_coords']
            sam_point_labels = point_inputs['point_labels']
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)
        if mask_inputs is not None:
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(mask_inputs.float(), size=self.sam_prompt_encoder.mask_input_size, align_corners=False, mode='bilinear', antialias=True)
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt)
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.sam_mask_decoder(image_embeddings=backbone_features, image_pe=self.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output, repeat_image=False, high_res_features=high_res_features)
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(low_res_multimasks, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        return low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """Processes mask inputs directly as output, bypassing SAM encoder/decoder."""
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(high_res_masks, size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4), align_corners=False, mode='bilinear', antialias=True)
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            obj_ptr = torch.zeros(mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device)
        else:
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(backbone_features=backbone_features, mask_inputs=self.mask_downsample(mask_inputs_float), high_res_features=high_res_features)
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        return low_res_masks, high_res_masks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits

    def forward_image(self, img_batch: 'torch.Tensor'):
        """Processes image batch through encoder to extract multi-level features for SAM model."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            backbone_out['backbone_fpn'][0] = self.sam_mask_decoder.conv_s0(backbone_out['backbone_fpn'][0])
            backbone_out['backbone_fpn'][1] = self.sam_mask_decoder.conv_s1(backbone_out['backbone_fpn'][1])
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepares and flattens visual features from the image backbone output for further processing."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out['backbone_fpn']) == len(backbone_out['vision_pos_enc'])
        assert len(backbone_out['backbone_fpn']) >= self.num_feature_levels
        feature_maps = backbone_out['backbone_fpn'][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out['vision_pos_enc'][-self.num_feature_levels:]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, output_dict, num_frames, track_in_reverse=False):
        """Prepares memory-conditioned features by fusing current frame's visual features with previous memories."""
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device
        if self.num_maskmem == 0:
            return current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        if not is_init_cond_frame:
            to_cat_memory, to_cat_memory_pos_embed = [], []
            assert len(output_dict['cond_frame_outputs']) > 0
            cond_outputs = output_dict['cond_frame_outputs']
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(frame_idx, cond_outputs, self.max_cond_frames_in_attn)
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            r = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                if t_rel == 1:
                    prev_frame_idx = frame_idx + t_rel if track_in_reverse else frame_idx - t_rel
                elif not track_in_reverse:
                    prev_frame_idx = (frame_idx - 2) // r * r
                    prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                else:
                    prev_frame_idx = -(-(frame_idx + 2) // r) * r
                    prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
                out = output_dict['non_cond_frame_outputs'].get(prev_frame_idx, None)
                if out is None:
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))
            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue
                feats = prev['maskmem_features']
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                maskmem_enc = prev['maskmem_pos_enc'][-1]
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                to_cat_memory_pos_embed.append(maskmem_enc)
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {t: out for t, out in selected_cond_outputs.items() if (t >= frame_idx if track_in_reverse else t <= frame_idx)}
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [((frame_idx - t) * tpos_sign_mul if self.use_signed_tpos_enc_to_obj_ptrs else abs(frame_idx - t), out['obj_ptr']) for t, out in ptr_cond_outputs.items()]
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or num_frames is not None and t >= num_frames:
                        break
                    out = output_dict['non_cond_frame_outputs'].get(t, unselected_cond_outputs.get(t, None))
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out['obj_ptr']))
                if pos_and_ptrs:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            if self.directly_add_no_mem_embed:
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        pix_feat_with_mem = self.memory_attention(curr=current_vision_feats, curr_pos=current_vision_pos_embeds, memory=memory, memory_pos=memory_pos_embed, num_obj_ptr_tokens=num_obj_ptr_tokens)
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(self, current_vision_feats, feat_sizes, pred_masks_high_res, object_score_logits, is_mask_from_pts):
        """Encodes frame features and masks into a new memory representation for video segmentation."""
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            pred_masks_high_res = self._apply_non_overlapping_constraints(pred_masks_high_res)
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out['vision_features']
        maskmem_pos_enc = maskmem_out['vision_pos_enc']
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (1 - is_obj_appearing[..., None, None]) * self.no_obj_embed_spatial[..., None, None].expand(*maskmem_features.shape)
        return maskmem_features, maskmem_pos_enc

    def _track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs, output_dict, num_frames, track_in_reverse, prev_sam_mask_logits):
        """Performs a single tracking step, updating object masks and memory features based on current frame inputs."""
        current_out = {'point_inputs': point_inputs, 'mask_inputs': mask_inputs}
        if len(current_vision_feats) > 1:
            high_res_features = [x.permute(1, 2, 0).view(x.size(1), x.size(2), *s) for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(pix_feat, high_res_features, mask_inputs)
        else:
            pix_feat = self._prepare_memory_conditioned_features(frame_idx=frame_idx, is_init_cond_frame=is_init_cond_frame, current_vision_feats=current_vision_feats[-1:], current_vision_pos_embeds=current_vision_pos_embeds[-1:], feat_sizes=feat_sizes[-1:], output_dict=output_dict, num_frames=num_frames, track_in_reverse=track_in_reverse)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(backbone_features=pix_feat, point_inputs=point_inputs, mask_inputs=mask_inputs, high_res_features=high_res_features, multimask_output=multimask_output)
        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(self, current_vision_feats, feat_sizes, point_inputs, run_mem_encoder, high_res_masks, object_score_logits, current_out):
        """Finally run the memory encoder on the predicted mask to encode, it into a new memory feature (that can be
        used in future frames).
        """
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(current_vision_feats=current_vision_feats, feat_sizes=feat_sizes, pred_masks_high_res=high_res_masks_for_mem_enc, object_score_logits=object_score_logits, is_mask_from_pts=point_inputs is not None)
            current_out['maskmem_features'] = maskmem_features
            current_out['maskmem_pos_enc'] = maskmem_pos_enc
        else:
            current_out['maskmem_features'] = None
            current_out['maskmem_pos_enc'] = None

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs, output_dict, num_frames, track_in_reverse=False, run_mem_encoder=True, prev_sam_mask_logits=None):
        """Performs a single tracking step, updating object masks and memory features based on current frame inputs."""
        current_out, sam_outputs, _, _ = self._track_step(frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs, output_dict, num_frames, track_in_reverse, prev_sam_mask_logits)
        _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = sam_outputs
        current_out['pred_masks'] = low_res_masks
        current_out['pred_masks_high_res'] = high_res_masks
        current_out['obj_ptr'] = obj_ptr
        if not self.training:
            current_out['object_score_logits'] = object_score_logits
        self._encode_memory_in_output(current_vision_feats, feat_sizes, point_inputs, run_mem_encoder, high_res_masks, object_score_logits, current_out)
        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Determines whether to use multiple mask outputs in the SAM head based on configuration and inputs."""
        num_pts = 0 if point_inputs is None else point_inputs['point_labels'].size(1)
        return self.multimask_output_in_sam and (is_init_cond_frame or self.multimask_output_for_tracking) and self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num

    def _apply_non_overlapping_constraints(self, pred_masks):
        """Applies non-overlapping constraints to masks, keeping highest scoring object per location."""
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks
        device = pred_masks.device
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    def set_imgsz(self, imgsz):
        """
        Set image size to make model compatible with different image sizes.

        Args:
            imgsz (Tuple[int, int]): The size of the input image.
        """
        self.image_size = imgsz[0]
        self.sam_prompt_encoder.input_image_size = imgsz
        self.sam_prompt_encoder.image_embedding_size = [(x // 16) for x in imgsz]


def _build_sam2(encoder_embed_dim=1280, encoder_stages=[2, 6, 36, 4], encoder_num_heads=2, encoder_global_att_blocks=[7, 15, 23, 31], encoder_backbone_channel_list=[1152, 576, 288, 144], encoder_window_spatial_size=[7, 7], encoder_window_spec=[8, 4, 16, 8], checkpoint=None):
    """
    Builds and returns a Segment Anything Model 2 (SAM2) with specified architecture parameters.

    Args:
        encoder_embed_dim (int): Embedding dimension for the encoder.
        encoder_stages (List[int]): Number of blocks in each stage of the encoder.
        encoder_num_heads (int): Number of attention heads in the encoder.
        encoder_global_att_blocks (List[int]): Indices of global attention blocks in the encoder.
        encoder_backbone_channel_list (List[int]): Channel dimensions for each level of the encoder backbone.
        encoder_window_spatial_size (List[int]): Spatial size of the window for position embeddings.
        encoder_window_spec (List[int]): Window specifications for each stage of the encoder.
        checkpoint (str | None): Path to the checkpoint file for loading pre-trained weights.

    Returns:
        (SAM2Model): A configured and initialized SAM2 model.

    Examples:
        >>> sam2_model = _build_sam2(encoder_embed_dim=96, encoder_stages=[1, 2, 7, 2])
        >>> sam2_model.eval()
    """
    image_encoder = ImageEncoder(trunk=Hiera(embed_dim=encoder_embed_dim, num_heads=encoder_num_heads, stages=encoder_stages, global_att_blocks=encoder_global_att_blocks, window_pos_embed_bkg_spatial_size=encoder_window_spatial_size, window_spec=encoder_window_spec), neck=FpnNeck(d_model=256, backbone_channel_list=encoder_backbone_channel_list, fpn_top_down_levels=[2, 3], fpn_interp_model='nearest'), scalp=1)
    memory_attention = MemoryAttention(d_model=256, pos_enc_at_input=True, num_layers=4, layer=MemoryAttentionLayer())
    memory_encoder = MemoryEncoder(out_dim=64)
    is_sam2_1 = checkpoint is not None and 'sam2.1' in checkpoint
    sam2 = SAM2Model(image_encoder=image_encoder, memory_attention=memory_attention, memory_encoder=memory_encoder, num_maskmem=7, image_size=1024, sigmoid_scale_for_mem_enc=20.0, sigmoid_bias_for_mem_enc=-10.0, use_mask_input_as_output_without_sam=True, directly_add_no_mem_embed=True, use_high_res_features_in_sam=True, multimask_output_in_sam=True, iou_prediction_use_sigmoid=True, use_obj_ptrs_in_encoder=True, add_tpos_enc_to_obj_ptrs=True, only_obj_ptrs_in_the_past_for_eval=True, pred_obj_scores=True, pred_obj_scores_mlp=True, fixed_no_obj_ptr=True, multimask_output_for_tracking=True, use_multimask_token_for_obj_ptr=True, multimask_min_pt_num=0, multimask_max_pt_num=1, use_mlp_for_obj_ptr_proj=True, compile_image_encoder=False, no_obj_embed_spatial=is_sam2_1, proj_tpos_enc_in_obj_ptrs=is_sam2_1, use_signed_tpos_enc_to_obj_ptrs=is_sam2_1, sam_mask_decoder_extra_args=dict(dynamic_multimask_via_stability=True, dynamic_multimask_stability_delta=0.05, dynamic_multimask_stability_thresh=0.98))
    if checkpoint is not None:
        checkpoint = attempt_download_asset(checkpoint)
        with open(checkpoint, 'rb') as f:
            state_dict = torch.load(f)['model']
        sam2.load_state_dict(state_dict)
    sam2.eval()
    return sam2


def build_sam2_b(checkpoint=None):
    """Builds and returns a SAM2 base-size model with specified architecture parameters."""
    return _build_sam2(encoder_embed_dim=112, encoder_stages=[2, 3, 16, 3], encoder_num_heads=2, encoder_global_att_blocks=[12, 16, 20], encoder_window_spec=[8, 4, 14, 7], encoder_window_spatial_size=[14, 14], encoder_backbone_channel_list=[896, 448, 224, 112], checkpoint=checkpoint)


def build_sam2_l(checkpoint=None):
    """Builds and returns a large-size Segment Anything Model (SAM2) with specified architecture parameters."""
    return _build_sam2(encoder_embed_dim=144, encoder_stages=[2, 6, 36, 4], encoder_num_heads=2, encoder_global_att_blocks=[23, 33, 43], encoder_window_spec=[8, 4, 16, 8], encoder_backbone_channel_list=[1152, 576, 288, 144], checkpoint=checkpoint)


def build_sam2_s(checkpoint=None):
    """Builds and returns a small-size Segment Anything Model (SAM2) with specified architecture parameters."""
    return _build_sam2(encoder_embed_dim=96, encoder_stages=[1, 2, 11, 2], encoder_num_heads=1, encoder_global_att_blocks=[7, 10, 13], encoder_window_spec=[8, 4, 14, 7], encoder_backbone_channel_list=[768, 384, 192, 96], checkpoint=checkpoint)


def build_sam2_t(checkpoint=None):
    """Builds and returns a Segment Anything Model 2 (SAM2) tiny-size model with specified architecture parameters."""
    return _build_sam2(encoder_embed_dim=96, encoder_stages=[1, 2, 7, 2], encoder_num_heads=1, encoder_global_att_blocks=[5, 7, 9], encoder_window_spec=[8, 4, 14, 7], encoder_backbone_channel_list=[768, 384, 192, 96], checkpoint=checkpoint)


def build_sam_vit_b(checkpoint=None):
    """Constructs and returns a Segment Anything Model (SAM) with b-size architecture and optional checkpoint."""
    return _build_sam(encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12, encoder_global_attn_indexes=[2, 5, 8, 11], checkpoint=checkpoint)


def build_sam_vit_h(checkpoint=None):
    """Builds and returns a Segment Anything Model (SAM) h-size model with specified encoder parameters."""
    return _build_sam(encoder_embed_dim=1280, encoder_depth=32, encoder_num_heads=16, encoder_global_attn_indexes=[7, 15, 23, 31], checkpoint=checkpoint)


def build_sam_vit_l(checkpoint=None):
    """Builds and returns a Segment Anything Model (SAM) l-size model with specified encoder parameters."""
    return _build_sam(encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16, encoder_global_attn_indexes=[5, 11, 17, 23], checkpoint=checkpoint)


sam_model_map = {'sam_h.pt': build_sam_vit_h, 'sam_l.pt': build_sam_vit_l, 'sam_b.pt': build_sam_vit_b, 'mobile_sam.pt': build_mobile_sam, 'sam2_t.pt': build_sam2_t, 'sam2_s.pt': build_sam2_s, 'sam2_b.pt': build_sam2_b, 'sam2_l.pt': build_sam2_l, 'sam2.1_t.pt': build_sam2_t, 'sam2.1_s.pt': build_sam2_s, 'sam2.1_b.pt': build_sam2_b, 'sam2.1_l.pt': build_sam2_l}


def build_sam(ckpt='sam_b.pt'):
    """
    Builds and returns a Segment Anything Model (SAM) based on the provided checkpoint.

    Args:
        ckpt (str | Path): Path to the checkpoint file or name of a pre-defined SAM model.

    Returns:
        (SAMModel | SAM2Model): A configured and initialized SAM or SAM2 model instance.

    Raises:
        FileNotFoundError: If the provided checkpoint is not a supported SAM model.

    Examples:
        >>> sam_model = build_sam("sam_b.pt")
        >>> sam_model = build_sam("path/to/custom_checkpoint.pt")

    Notes:
        Supported pre-defined models include:
        - SAM: 'sam_h.pt', 'sam_l.pt', 'sam_b.pt', 'mobile_sam.pt'
        - SAM2: 'sam2_t.pt', 'sam2_s.pt', 'sam2_b.pt', 'sam2_l.pt'
    """
    model_builder = None
    ckpt = str(ckpt)
    for k in sam_model_map.keys():
        if ckpt.endswith(k):
            model_builder = sam_model_map.get(k)
    if not model_builder:
        raise FileNotFoundError(f'{ckpt} is not a supported SAM model. Available models are: \n {sam_model_map.keys()}')
    return model_builder(ckpt)


def calculate_stability_score(masks: 'torch.Tensor', mask_threshold: 'float', threshold_offset: 'float') ->torch.Tensor:
    """
    Computes the stability score for a batch of masks.

    The stability score is the IoU between binary masks obtained by thresholding the predicted mask logits at
    high and low values.

    Args:
        masks (torch.Tensor): Batch of predicted mask logits.
        mask_threshold (float): Threshold value for creating binary masks.
        threshold_offset (float): Offset applied to the threshold for creating high and low binary masks.

    Returns:
        (torch.Tensor): Stability scores for each mask in the batch.

    Notes:
        - One mask is always contained inside the other.
        - Memory is saved by preventing unnecessary cast to torch.int64.

    Examples:
        >>> masks = torch.rand(10, 256, 256)  # Batch of 10 masks
        >>> mask_threshold = 0.5
        >>> threshold_offset = 0.1
        >>> stability_scores = calculate_stability_score(masks, mask_threshold, threshold_offset)
    """
    intersections = (masks > mask_threshold + threshold_offset).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    unions = (masks > mask_threshold - threshold_offset).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    return intersections / unions


def generate_crop_boxes(im_size: 'Tuple[int, ...]', n_layers: 'int', overlap_ratio: 'float') ->Tuple[List[List[int]], List[int]]:
    """Generates crop boxes of varying sizes for multi-scale image processing, with layered overlapping regions."""
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        """Crops bounding boxes to the size of the input image."""
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))
    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))
        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)
        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)
    return crop_boxes, layer_idxs


def uncrop_boxes_xyxy(boxes: 'torch.Tensor', crop_box: 'List[int]') ->torch.Tensor:
    """Uncrop bounding boxes by adding the crop box offset to their coordinates."""
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset


def is_box_near_crop_edge(boxes: 'torch.Tensor', crop_box: 'List[int]', orig_box: 'List[int]', atol: 'float'=20.0) ->torch.Tensor:
    """Determines if bounding boxes are near the edge of a cropped image region using a specified tolerance."""
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


def uncrop_masks(masks: 'torch.Tensor', crop_box: 'List[int]', orig_h: 'int', orig_w: 'int') ->torch.Tensor:
    """Uncrop masks by padding them to the original image size, handling coordinate transformations."""
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = x0, pad_x - x0, y0, pad_y - y0
    return torch.nn.functional.pad(masks, pad, value=0)


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class HungarianMatcher(nn.Module):
    """
    A module implementing the HungarianMatcher, which is a differentiable module to solve the assignment problem in an
    end-to-end fashion.

    HungarianMatcher performs optimal assignment over the predicted and ground truth bounding boxes using a cost
    function that considers classification scores, bounding box coordinates, and optionally, mask predictions.

    Attributes:
        cost_gain (dict): Dictionary of cost coefficients: 'class', 'bbox', 'giou', 'mask', and 'dice'.
        use_fl (bool): Indicates whether to use Focal Loss for the classification cost calculation.
        with_mask (bool): Indicates whether the model makes mask predictions.
        num_sample_points (int): The number of sample points used in mask cost calculation.
        alpha (float): The alpha factor in Focal Loss calculation.
        gamma (float): The gamma factor in Focal Loss calculation.

    Methods:
        forward(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None): Computes the
            assignment between predictions and ground truths for a batch.
        _cost_mask(bs, num_gts, masks=None, gt_mask=None): Computes the mask cost and dice cost if masks are predicted.
    """

    def __init__(self, cost_gain=None, use_fl=True, with_mask=False, num_sample_points=12544, alpha=0.25, gamma=2.0):
        """Initializes a HungarianMatcher module for optimal assignment of predicted and ground truth bounding boxes."""
        super().__init__()
        if cost_gain is None:
            cost_gain = {'class': 1, 'bbox': 5, 'giou': 2, 'mask': 1, 'dice': 1}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None):
        """
        Forward pass for HungarianMatcher. This function computes costs based on prediction and ground truth
        (classification cost, L1 cost between boxes and GIoU cost between boxes) and finds the optimal matching between
        predictions and ground truth based on these costs.

        Args:
            pred_bboxes (Tensor): Predicted bounding boxes with shape [batch_size, num_queries, 4].
            pred_scores (Tensor): Predicted scores with shape [batch_size, num_queries, num_classes].
            gt_cls (torch.Tensor): Ground truth classes with shape [num_gts, ].
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape [num_gts, 4].
            gt_groups (List[int]): List of length equal to batch size, containing the number of ground truths for
                each image.
            masks (Tensor, optional): Predicted masks with shape [batch_size, num_queries, height, width].
                Defaults to None.
            gt_mask (List[Tensor], optional): List of ground truth masks, each with shape [num_masks, Height, Width].
                Defaults to None.

        Returns:
            (List[Tuple[Tensor, Tensor]]): A list of size batch_size, each element is a tuple (index_i, index_j), where:
                - index_i is the tensor of indices of the selected predictions (in order)
                - index_j is the tensor of indices of the corresponding selected ground truth targets (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, nq, nc = pred_scores.shape
        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]
        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        pred_bboxes = pred_bboxes.detach().view(-1, 4)
        pred_scores = pred_scores[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (1 - self.alpha) * pred_scores ** self.gamma * -(1 - pred_scores + 1e-08).log()
            pos_cost_class = self.alpha * (1 - pred_scores) ** self.gamma * -(pred_scores + 1e-08).log()
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores
        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)
        cost_giou = 1.0 - bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, GIoU=True).squeeze(-1)
        C = self.cost_gain['class'] * cost_class + self.cost_gain['bbox'] * cost_bbox + self.cost_gain['giou'] * cost_giou
        if self.with_mask:
            C += self._cost_mask(bs, gt_groups, masks, gt_mask)
        C[C.isnan() | C.isinf()] = 0.0
        C = C.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))]
        gt_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        return [(torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_groups[k]) for k, (i, j) in enumerate(indices)]


def autocast(enabled: 'bool', device: 'str'='cuda'):
    """
    Get the appropriate autocast context manager based on PyTorch version and AMP setting.

    This function returns a context manager for automatic mixed precision (AMP) training that is compatible with both
    older and newer versions of PyTorch. It handles the differences in the autocast API between PyTorch versions.

    Args:
        enabled (bool): Whether to enable automatic mixed precision.
        device (str, optional): The device to use for autocast. Defaults to 'cuda'.

    Returns:
        (torch.amp.autocast): The appropriate autocast context manager.

    Note:
        - For PyTorch versions 1.13 and newer, it uses `torch.amp.autocast`.
        - For older versions, it uses `torch.cuda.autocast`.

    Example:
        ```python
        with autocast(amp=True):
            # Your mixed precision operations here
            pass
        ```
    """
    if TORCH_1_13:
        return torch.amp.autocast(device, enabled=enabled)
    else:
        return torch.amp.autocast(enabled)


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') * weight).mean(1).sum()
        return loss


class DETRLoss(nn.Module):
    """
    DETR (DEtection TRansformer) Loss class. This class calculates and returns the different loss components for the
    DETR object detection model. It computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary
    losses.

    Attributes:
        nc (int): The number of classes.
        loss_gain (dict): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary losses.
        use_fl (bool): Use FocalLoss or not.
        use_vfl (bool): Use VarifocalLoss or not.
        use_uni_match (bool): Whether to use a fixed layer to assign labels for the auxiliary branch.
        uni_match_ind (int): The fixed indices of a layer to use if `use_uni_match` is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss or None): Focal Loss object if `use_fl` is True, otherwise None.
        vfl (VarifocalLoss or None): Varifocal Loss object if `use_vfl` is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True, use_vfl=False, use_uni_match=False, uni_match_ind=0):
        """
        Initialize DETR loss function with customizable components and gains.

        Uses default loss_gain if not provided. Initializes HungarianMatcher with
        preset cost gains. Supports auxiliary losses and various loss types.

        Args:
            nc (int): Number of classes.
            loss_gain (dict): Coefficients for different loss components.
            aux_loss (bool): Use auxiliary losses from each decoder layer.
            use_fl (bool): Use FocalLoss.
            use_vfl (bool): Use VarifocalLoss.
            use_uni_match (bool): Use fixed layer for auxiliary branch label assignment.
            uni_match_ind (int): Index of fixed layer for uni_match.
        """
        super().__init__()
        if loss_gain is None:
            loss_gain = {'class': 1, 'bbox': 5, 'giou': 2, 'no_object': 0.1, 'mask': 1, 'dice': 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={'class': 2, 'bbox': 5, 'giou': 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss() if use_fl else None
        self.vfl = VarifocalLoss() if use_vfl else None
        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        """Computes the classification loss based on predictions, target values, and ground truth scores."""
        name_class = f'loss_class{postfix}'
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot
        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction='none')(pred_scores, gt_scores).mean(1).sum()
        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        """Computes bounding box and GIoU losses for predicted and ground truth bounding boxes."""
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'
        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss
        loss[name_bbox] = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / len(gt_bboxes)
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    def _get_loss_aux(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, match_indices=None, postfix='', masks=None, gt_mask=None):
        """Get auxiliary losses."""
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(pred_bboxes[self.uni_match_ind], pred_scores[self.uni_match_ind], gt_bboxes, gt_cls, gt_groups, masks=masks[self.uni_match_ind] if masks is not None else None, gt_mask=gt_mask)
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(aux_bboxes, aux_scores, gt_bboxes, gt_cls, gt_groups, masks=aux_masks, gt_mask=gt_mask, postfix=postfix, match_indices=match_indices)
            loss[0] += loss_[f'loss_class{postfix}']
            loss[1] += loss_[f'loss_bbox{postfix}']
            loss[2] += loss_[f'loss_giou{postfix}']
        loss = {f'loss_class_aux{postfix}': loss[0], f'loss_bbox_aux{postfix}': loss[1], f'loss_giou_aux{postfix}': loss[2]}
        return loss

    @staticmethod
    def _get_index(match_indices):
        """Returns batch indices, source indices, and destination indices from provided match indices."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for src, _ in match_indices])
        dst_idx = torch.cat([dst for _, dst in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(self, pred_bboxes, gt_bboxes, match_indices):
        """Assigns predicted bounding boxes to ground truth bounding boxes based on the match indices."""
        pred_assigned = torch.cat([(t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)) for t, (i, _) in zip(pred_bboxes, match_indices)])
        gt_assigned = torch.cat([(t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)) for t, (_, j) in zip(gt_bboxes, match_indices)])
        return pred_assigned, gt_assigned

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None, postfix='', match_indices=None):
        """Get losses."""
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask)
        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]
        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]
        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)
        loss = {}
        loss.update(self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix))
        loss.update(self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix))
        return loss

    def forward(self, pred_bboxes, pred_scores, batch, postfix='', **kwargs):
        """
        Calculate loss for predicted bounding boxes and scores.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape [l, b, query, 4].
            pred_scores (torch.Tensor): Predicted class scores, shape [l, b, query, num_classes].
            batch (dict): Batch information containing:
                cls (torch.Tensor): Ground truth classes, shape [num_gts].
                bboxes (torch.Tensor): Ground truth bounding boxes, shape [num_gts, 4].
                gt_groups (List[int]): Number of ground truths for each image in the batch.
            postfix (str): Postfix for loss names.
            **kwargs (Any): Additional arguments, may include 'match_indices'.

        Returns:
            (dict): Computed losses, including main and auxiliary (if enabled).

        Note:
            Uses last elements of pred_bboxes and pred_scores for main loss, and the rest for auxiliary losses if
            self.aux_loss is True.
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get('match_indices', None)
        gt_cls, gt_bboxes, gt_groups = batch['cls'], batch['bboxes'], batch['gt_groups']
        total_loss = self._get_loss(pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices)
        if self.aux_loss:
            total_loss.update(self._get_loss_aux(pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix))
        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    """
    Real-Time DeepTracker (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        """
        Forward pass to compute the detection loss.

        Args:
            preds (tuple): Predicted bounding boxes and scores.
            batch (dict): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes. Default is None.
            dn_scores (torch.Tensor, optional): Denoising scores. Default is None.
            dn_meta (dict, optional): Metadata for denoising. Default is None.

        Returns:
            (dict): Dictionary containing the total loss and, if applicable, the denoising loss.
        """
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta['dn_pos_idx'], dn_meta['dn_num_group']
            assert len(batch['gt_groups']) == len(dn_pos_idx)
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch['gt_groups'])
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix='_dn', match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            total_loss.update({f'{k}_dn': torch.tensor(0.0, device=self.device) for k in total_loss.keys()})
        return total_loss

    @staticmethod
    def get_dn_match_indices(dn_pos_idx, dn_num_group, gt_groups):
        """
        Get the match indices for denoising.

        Args:
            dn_pos_idx (List[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (List[int]): List of integers representing the number of ground truths for each image.

        Returns:
            (List[tuple]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), 'Expected the same length, '
                f"""but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."""
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices


class AGLU(nn.Module):
    """Unified activation function module from https://github.com/kostas1515/AGLU."""

    def __init__(self, device=None, dtype=None) ->None:
        """Initialize the Unified activation function."""
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Compute the forward pass of the Unified activation function."""
        lam = torch.clamp(self.lambd, min=0.0001)
        return torch.exp(1 / lam * self.act(self.kappa * x - torch.log(lam)))


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: 'int') ->None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


def probiou(obb1, obb2, CIoU=False, eps=1e-07):
    """
    Calculate probabilistic IoU between oriented bounding boxes.

    Implements the algorithm from https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
        CIoU (bool, optional): If True, calculate CIoU. Defaults to False.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): OBB similarities, shape (N,).

    Note:
        OBB format: [center_x, center_y, width, height, rotation_angle].
        If CIoU is True, returns CIoU instead of IoU.
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)
    t1 = ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps) * 0.25
    t2 = (c1 + c2) * (x2 - x1) * (y1 - y2) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps) * 0.5
    t3 = (((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps) + eps).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = 4 / math.pi ** 2 * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha
    return iou


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0)
        return loss_iou, loss_dfl


def xywhr2xyxyxyxy(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin, cat, stack = (torch.cos, torch.sin, torch.cat, torch.stack) if isinstance(x, torch.Tensor) else (np.cos, np.sin, np.concatenate, np.stack)
    ctr = x[..., :2]
    w, h, angle = (x[..., i:i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to rotated bounding boxes using a task-aligned metric."""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 5)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        corners = xywhr2xyxyxyxy(gt_bboxes)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes'].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\nThis error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a correctly formatted 'OBB' dataset using 'data=dota8.yaml' as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help.") from e
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)
        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(pred_scores.detach().sigmoid(), bboxes_for_assigner.type(gt_bboxes.dtype), anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores).sum() / target_scores_sum
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
        else:
            loss[0] += (pred_angle * 0).sum()
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class OBBModel(DetectionModel):
    """YOLOv8 Oriented Bounding Box (OBB) model."""

    def __init__(self, cfg='yolov8n-obb.yaml', ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 OBB model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) ->None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-09)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-09) * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


OKS_SIGMA = np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]) / 10.0


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        sigmas = torch.from_numpy(OKS_SIGMA) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = max(target_scores.sum(), 1)
        loss[3] = self.bce(pred_scores, target_scores).sum() / target_scores_sum
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            loss[1], loss[2] = self.calculate_keypoints_loss(fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        loss[3] *= self.hyp.cls
        loss[4] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()
        batched_keypoints = torch.zeros((batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device)
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, :keypoints_i.shape[0]] = keypoints_i
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)
        selected_keypoints = batched_keypoints.gather(1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2]))
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)
        kpts_loss = 0
        kpts_obj_loss = 0
        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
        return kpts_loss, kpts_obj_loss


class PoseModel(DetectionModel):
    """YOLOv8 pose model."""

    def __init__(self, cfg='yolov8n-pose.yaml', ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg['kpt_shape']):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg['kpt_shape'] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = F.cross_entropy(preds, batch['cls'], reduction='mean')
        loss_items = loss.detach()
        return loss, loss_items


class ClassificationModel(BaseModel):
    """YOLOv8 classification model."""

    def __init__(self, cfg='yolov8n-cls.yaml', ch=3, nc=None, verbose=True):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        elif not nc and not self.yaml.get('nc', None):
            raise ValueError('nc not specified. Must specify nc in model.yaml or function arguments.')
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.stride = torch.Tensor([1])
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]
        if isinstance(m, Classify):
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(nn.Linear)
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(nn.Conv2d)
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        cfg (str): The configuration file path or preset string. Default is 'rtdetr-l.yaml'.
        ch (int): Number of input channels. Default is 3 (RGB).
        nc (int, optional): Number of classes for object detection. Default is None.
        verbose (bool): Specifies if summary statistics are shown during initialization. Default is True.

    Methods:
        init_criterion: Initializes the criterion used for loss calculation.
        loss: Computes and returns the loss during training.
        predict: Performs a forward pass through the network and returns the output.
    """

    def __init__(self, cfg='rtdetr-l.yaml', ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes. Defaults to None.
            verbose (bool, optional): Print additional information during initialization. Defaults to True.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions. Defaults to None.

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor.
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()
        img = batch['img']
        bs = len(img)
        batch_idx = batch['batch_idx']
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {'cls': batch['cls'].view(-1), 'bboxes': batch['bboxes'], 'batch_idx': batch_idx.view(-1), 'gt_groups': gt_groups}
        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta['dn_num_split'], dim=2)
        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
        loss = self.criterion((dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta)
        return sum(loss.values()), torch.as_tensor([loss[k].detach() for k in ['loss_giou', 'loss_class', 'loss_bbox']], device=img.device)

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            batch (dict, optional): Ground truth data for evaluation. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []
        for m in self.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j]) for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ADown,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BNContrastiveHead,
     lambda: ([], {'embed_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (Block,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Bottleneck,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BottleneckCSP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (C1,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (C2,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (C2PSA,
     lambda: ([], {'c1': 128, 'c2': 128}),
     lambda: ([torch.rand([4, 128, 64, 64])], {})),
    (C2f,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (C2fCIB,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (C2fPSA,
     lambda: ([], {'c1': 128, 'c2': 128}),
     lambda: ([torch.rand([4, 128, 64, 64])], {})),
    (C3,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (C3f,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (C3k,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (C3k2,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (C3x,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CBAM,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CBLinear,
     lambda: ([], {'c1': 4, 'c2s': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CIB,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CXBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ChannelAttention,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Classify,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2d_BN,
     lambda: ([], {'a': 4, 'b': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvLayer,
     lambda: ([], {'dim': 4, 'input_resolution': 4, 'depth': 1, 'activation': torch.nn.ReLU}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvTranspose,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DWConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DWConvTranspose2d,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ELAN1,
     lambda: ([], {'c1': 4, 'c2': 4, 'c3': 4, 'c4': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Focus,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Fuser,
     lambda: ([], {'layer': torch.nn.ReLU(), 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GhostBottleneck,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GhostConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HGBlock,
     lambda: ([], {'c1': 4, 'cm': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HGStem,
     lambda: ([], {'c1': 4, 'cm': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm2d,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LightConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MBConv,
     lambda: ([], {'in_chans': 4, 'out_chans': 4, 'expand_ratio': 4, 'activation': torch.nn.ReLU, 'drop_path': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLPBlock,
     lambda: ([], {'embedding_dim': 4, 'mlp_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaskDownSampler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {})),
    (MemoryEncoder,
     lambda: ([], {'out_dim': 4}),
     lambda: ([torch.rand([4, 256, 64, 64]), torch.rand([4, 1, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiScaleAttention,
     lambda: ([], {'dim': 4, 'dim_out': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiScaleBlock,
     lambda: ([], {'dim': 4, 'dim_out': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PSA,
     lambda: ([], {'c1': 128, 'c2': 128}),
     lambda: ([torch.rand([4, 128, 64, 64])], {})),
    (PatchMerging,
     lambda: ([], {'input_resolution': 4, 'dim': 4, 'out_dim': 4, 'activation': torch.nn.ReLU}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionEmbeddingSine,
     lambda: ([], {'num_pos_feats': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Proto,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepBottleneck,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepC3,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepCSP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepNCSPELAN4,
     lambda: ([], {'c1': 4, 'c2': 4, 'c3': 4, 'c4': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepVGGDW,
     lambda: ([], {'ed': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNetBlock,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNetLayer,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SCDown,
     lambda: ([], {'c1': 4, 'c2': 4, 'k': 4, 's': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SPP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SPPELAN,
     lambda: ([], {'c1': 4, 'c2': 4, 'c3': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SPPF,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpatialAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerBlock,
     lambda: ([], {'c1': 4, 'c2': 4, 'num_heads': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerLayer,
     lambda: ([], {'c': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (VarifocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

