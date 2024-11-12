
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


from torchvision.models import resnet18


import inspect


import logging


from functools import wraps


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Sequence


from typing import Union


import numpy as np


from copy import deepcopy


from functools import partial


from typing import Tuple


from torch.utils.data import DataLoader


from typing import NamedTuple


from abc import ABCMeta


from abc import abstractmethod


import abc


import time


from random import randint


import re


from queue import Queue


from typing import Generator


from typing import Iterable


from torch import nn


from torch.utils.data import Dataset


from torch import Tensor


import copy


import math


import torch.nn.functional as F


from torch.nn import functional as F


from torch.autograd import Function


from torch.onnx import symbolic_helper


from collections import defaultdict


from itertools import zip_longest


import torch.nn as nn


import types


from typing import MutableSequence


import functools


import warnings


from torch.onnx.symbolic_helper import parse_args


from torch.onnx import symbolic_helper as sym_help


from torch.nn.modules.utils import _pair


from torch.types import Number


import torch.onnx.symbolic_helper as sym_help


from torch.onnx.symbolic_helper import _get_tensor_dim_size


from torch.onnx.symbolic_helper import _get_tensor_rank


from torch.onnx.symbolic_helper import _unimplemented


from torch.onnx.symbolic_helper import _unsqueeze_helper


from torch.onnx.symbolic_opset9 import unused


from torch.onnx.symbolic_helper import _slice_helper


import random


import string


from logging import Logger


from torch.utils.data.dataset import Dataset


import numpy


import torch.multiprocessing as mp


from torch.multiprocessing import Process


from torch.multiprocessing import set_start_method


import pandas as pd


from torch.hub import download_url_to_file


class BaseWrapper(torch.nn.Module, metaclass=ABCMeta):
    """Abstract base class for backend wrappers.

    Args:
        output_names (Sequence[str]): Names to model outputs in order, which is
        useful when converting the output dict to a ordered list or converting
        the output ordered list to a key-value dict.
    """

    def __init__(self, output_names: 'Sequence[str]'):
        super().__init__()
        self._output_names = output_names

    @staticmethod
    def get_backend_file_count() ->int:
        """Return the count of backend file(s)

        Each backend has its own requirement on backend files (e.g., TensorRT
        requires 1 .engine file and ncnn requires 2 files (.param, .bin)). This
        interface allow developers to get the count of these required files.

        Returns:
            int: The count of required backend file(s).
        """
        return 1

    @abstractmethod
    def forward(self, inputs: 'Dict[str, torch.Tensor]') ->Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): Key-value pairs of model inputs.

        Returns:
            Dict[str, torch.Tensor]: Key-value pairs of model outputs.
        """
        pass

    @property
    def output_names(self):
        """Return the output names."""
        return self._output_names

    @output_names.setter
    def output_names(self, value):
        """Set the output names."""
        self._output_names = value

    def output_to_list(self, output_dict: 'Dict[str, torch.Tensor]') ->List[torch.Tensor]:
        """Convert the output dict of forward() to a tensor list.

        Args:
            output_dict (Dict[str, torch.Tensor]): Key-value pairs of model
                outputs.

        Returns:
            List[torch.Tensor]: An output value list whose order is determined
                by the ouput_names list.
        """
        outputs = [output_dict[name] for name in self._output_names]
        return outputs


logger_initialized = {}


def get_logger(name: 'str', log_file: 'Optional[str]'=None, log_level: 'int'=logging.INFO, file_mode: 'str'='w'):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified, a FileHandler will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    try:
        if MMLogger.check_instance_created(name):
            logger = MMLogger.get_instance(name)
        else:
            logger = MMLogger.get_instance(name, log_file=log_file, log_level=log_level, file_mode=file_mode)
        return logger
    except Exception:
        pass
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    logger_initialized[name] = True
    return logger


def get_root_logger(log_file=None, log_level=logging.INFO) ->logging.Logger:
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
    Returns:
        logging.Logger: The obtained logger
    """
    logger = get_logger(name='mmdeploy', log_file=log_file, log_level=log_level)
    return logger


def get_file_path(prefix, candidates) ->str:
    """Search for file in candidates.

    Args:
        prefix (str): Prefix of the paths.
        candidates (str): Candidate paths
    Returns:
        str: file path or '' if not found
    """
    for candidate in candidates:
        wildcard = os.path.abspath(os.path.join(prefix, candidate))
        paths = glob.glob(wildcard)
        if paths:
            lib_path = paths[0]
            return lib_path
    return ''


def get_lib_path() ->str:
    """Get the library path of onnxruntime.

    Returns:
        str: The library path to onnxruntime.
    """
    candidates = ['../../lib/libonnxruntime.so*', '../../lib/onnxruntime.dll']
    return get_file_path(os.path.dirname(__file__), candidates)


def get_ops_path() ->str:
    """Get ncnn custom ops library path.

    Returns:
        str: The library path of ncnn custom ops.
    """
    candidates = ['../../lib/libmmdeploy_ncnn_ops.so', '../../lib/mmdeploy_ncnn_ops.dll']
    return get_file_path(os.path.dirname(__file__), candidates)


def parse_cuda_device_id(device: 'str') ->int:
    """Parse cuda device index from a string.

    Args:
        device (str): The typical style of string specifying cuda device,
            e.g.: 'cuda:0'.

    Returns:
        int: The parsed device id, defaults to `0`.
    """
    match_result = re.match('([^:]+)(:[0-9]+)?$', device)
    assert match_result is not None, f'Can not parse device {device}.'
    assert match_result.group(1).lower() == 'cuda', 'Not cuda device.'
    device_id = 0 if match_result.lastindex == 1 else int(match_result.group(2)[1:])
    return device_id


def parse_device_id(device: 'str') ->Optional[int]:
    """Parse device index from a string.

    Args:
        device (str): The typical style of string specifying device,
            e.g.: 'cuda:0', 'cpu'.

    Returns:
        Optional[int]: The return value depends on the type of device.
            If device is 'cuda': cuda device index, defaults to `0`.
            If device is 'cpu': `-1`.
            Otherwise, `None` will be returned.
    """
    if device == 'cpu':
        return -1
    if 'cuda' in device:
        return parse_cuda_device_id(device)
    return None


class SleepingPolicy(abc.ABC):

    @abc.abstractmethod
    def sleep(self, try_i: 'int'):
        """How long to sleep in milliseconds.

        :param try_i: the number of retry (starting from zero)
        """
        assert try_i >= 0


class ExponentialBackoff(SleepingPolicy):

    def __init__(self, *, init_backoff_ms: int, max_backoff_ms: int, multiplier: int):
        self.init_backoff = randint(0, init_backoff_ms)
        self.max_backoff = max_backoff_ms
        self.multiplier = multiplier

    def sleep(self, try_i: 'int'):
        sleep_range = min(self.init_backoff * self.multiplier ** try_i, self.max_backoff)
        sleep_ms = randint(0, sleep_range)
        logger = get_root_logger()
        logger.debug(f'Sleeping for {sleep_ms}')
        time.sleep(sleep_ms / 1000)


def get_env_key() ->str:
    """Return environment key str.

    Returns:
        str: The string to find SNPE service URI
    """
    return '__MMDEPLOY_SNPE_URI'


def load_tensorrt_plugin() ->bool:
    """Load TensorRT plugins library.

    Returns:
        bool: True if TensorRT plugin library is successfully loaded.
    """
    lib_path = get_ops_path()
    success = False
    logger = get_root_logger()
    if os.path.exists(lib_path):
        ctypes.CDLL(lib_path)
        logger.info(f'Successfully loaded tensorrt plugins from {lib_path}')
        success = True
    else:
        logger.warning(f'Could not load the library of tensorrt plugins.             Because the file does not exist: {lib_path}')
    return success


def torch_device_from_trt(device: 'trt.TensorLocation'):
    """Convert pytorch device to TensorRT device.

    Args:
        device (trt.TensorLocation): The device in tensorrt.
    Returns:
        torch.device: The corresponding device in torch.
    """
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError(f'{device} is not supported by torch')


def torch_dtype_from_trt(dtype: 'trt.DataType') ->torch.dtype:
    """Convert pytorch dtype to TensorRT dtype.

    Args:
        dtype (str.DataType): The data type in tensorrt.

    Returns:
        torch.dtype: The corresponding data type in torch.
    """
    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f'{dtype} is not supported by torch')


class VACCForward:

    def __init__(self, model_info: 'Union[str, Dict[str, str]]', vdsp_params_info: 'str', device_id: 'int'=0, batch_size: 'int'=1) ->None:
        if isinstance(model_info, str):
            with open(model_info) as f:
                model_info = json.load(f)
        self.device_id = device_id
        self.input_id = 0
        self.vast_stream = vaststream.vast_stream()
        self.input_dict = {}
        self.event_dict = {}
        self.result_dict = {}
        balance_mode = 0

        def callback(output_description, ulOutPointerArray, ulArraySize, user_data_ptr):
            user_data = ctypes.cast(user_data_ptr, ctypes.POINTER(vacl_stream.StreamInfo))
            input_id = output_description.contents.input_id
            device_ddr = self.input_dict.pop(input_id)
            self.vast_stream.free_data_on_device(device_ddr, self.device_id)
            model_name = user_data.contents.model_name
            stream_output_list = self.vast_stream.stream_get_stream_output(model_name, ulOutPointerArray, ulArraySize)
            heatmap = np.squeeze(stream_output_list[0])
            num_outputs = self.vast_stream.get_output_num_per_batch(model_name)
            heatmap_shape = []
            for i in range(num_outputs):
                _, shape = self.vast_stream.get_output_shape_by_index(model_name, i)
                ndims = shape.ndims
                _shape = []
                for i in range(ndims):
                    _shape.append(shape.shapes[i])
                heatmap_shape.append(_shape)
            self.result_dict[input_id] = num_outputs, heatmap_shape, heatmap
            self.event_dict[input_id].set()
        self.callback = vaststream.output_callback_type(callback)
        self.stream = vacl_stream.create_vaststream(device_id, vdsp_params_info, model_info, self.callback, balance_mode, batch_size)

    def __start_extract(self, image: 'Union[str, np.ndarray]') ->int:
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        image_size = int(height * width * c)
        device_ddr = self.stream.copy_data_to_device(image, image_size)
        input_id = self.input_id
        self.input_dict[input_id] = device_ddr
        self.event_dict[input_id] = Event()
        self.stream.run_stream_dynamic([device_ddr], (height, width), input_id)
        self.input_id += 1
        return input_id

    def get_output_num(self):
        num_outputs = self.vast_stream.get_output_num_per_batch(self.vast_stream.model_name)
        return num_outputs

    def extract(self, image: 'Union[str, np.ndarray]') ->str:
        input_id = self.__start_extract(image)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]
        return result

    def extract_batch(self, images: 'Iterable[Union[str, np.ndarray]]') ->Generator[str, None, None]:
        queue = Queue(20)

        def input_thread():
            for image in images:
                input_id = self.__start_extract(image)
                queue.put(input_id)
            queue.put(None)
        thread = Thread(target=input_thread)
        thread.start()
        while True:
            input_id = queue.get()
            if input_id is None:
                break
            self.event_dict[input_id].wait()
            result = self.result_dict.pop(input_id)
            del self.event_dict[input_id]
            yield result


class VACCWrapper(BaseWrapper):
    """vacc wrapper class for inference.

    Args:
        lib_file (str): Path of a model lib file.
        graph_file (str): Path of a model graph file.
        param_file (str): Path of a model param file.
        vdsp_params_info_json (str): Path of a vdsp params info json file.
        output_names (Sequence[str] | None): Names of model outputs in order.
            Defaults to `None` and the wrapper will load the output names from
            vacc model.
    """

    def __init__(self, lib_file: 'str', graph_file: 'str', param_file: 'str', vdsp_params_info: 'dict', output_names: 'Optional[Sequence[str]]'=None, **kwargs):
        parent_path = os.path.abspath(os.path.dirname(lib_file) + os.path.sep + '.')
        model_info = {'model_name': 'model', 'model_lib_path': lib_file, 'model_graph_path': graph_file, 'model_params_path': param_file, 'hw_config_file': None}
        model_info_json = json.dumps(model_info)
        with open(os.path.join(parent_path, 'model_info.json'), 'w') as json_file:
            json_file.write(model_info_json)
        vdsp_params_info_json = json.dumps(vdsp_params_info)
        with open(os.path.join(parent_path, 'vdsp_param_info.json'), 'w') as json_file:
            json_file.write(vdsp_params_info_json)
        self.model = VACCForward(os.path.join(parent_path, 'model_info.json'), os.path.join(parent_path, 'vdsp_param_info.json'))
        super().__init__(output_names)

    @staticmethod
    def get_backend_file_count() ->int:
        """Return the count of backend file(s)

        vacc needs a .params file/a .json file/a .so file. So the count is 3.

        Returns:
            int: The count of required backend file(s).
        """
        return 3

    def forward(self, inputs: 'Dict[str, torch.Tensor]') ->Dict[str, torch.Tensor]:
        input_list = list(inputs.values())
        batch_size = input_list[0].size(0)
        logger = get_root_logger()
        if batch_size > 1:
            logger.warning(f'vacc only support batch_size = 1, but given {batch_size}')
        outputs = dict([name, [None] * batch_size] for name in self.output_names)
        for batch_id in range(batch_size):
            output = []
            for name, input_tensor in inputs.items():
                data = input_tensor[batch_id].contiguous()
                data = data.detach().cpu().numpy()
                results = self.model.extract_batch([data])
                for result in results:
                    output_num = result[0]
                    if output_num == 1:
                        output.append(np.reshape(np.array(result[2]).astype(np.float32), result[1][0])[0])
                    else:
                        outputs_ = []
                        outputs = {}
                        for index in range(output_num):
                            out = np.reshape(result[2][index].astype(np.float32), result[1][index])
                            outputs_.append(torch.from_numpy(out))
                        outputs['output'] = outputs_
                        return outputs
            output = np.array(output)
            for name in self.output_names:
                outputs[name][batch_id] = torch.from_numpy(output[0])
        for name, output_tensor in outputs.items():
            outputs[name] = torch.stack(output_tensor)
        return outputs


class PositionalEncoding(nn.Module):
    """Rewrite Position Encoding module in `ABINet."""

    def __init__(self, module, deploy_cfg, **kwargs):
        super(PositionalEncoding, self).__init__()
        self._module = module
        self.deploy_cfg = deploy_cfg
        self.n_position = module.position_table.size(1)
        self.d_hid = module.position_table.size(2)

    def _get_sinusoid_encoding_table(self, n_position, d_hid, device):
        """Sinusoid position encoding table."""
        denominator = torch.Tensor([(1.0 / torch.tensor(10000).to(device).pow(torch.tensor(2 * (hid_j // 2) / d_hid)).to(device)) for hid_j in range(d_hid)])
        denominator = denominator.view(1, -1)
        pos_tensor = torch.arange(n_position).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
        return sinusoid_table.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor of shape (batch_size, pos_len, d_hid, ...)
        """
        device = x.device
        position_table = self._get_sinusoid_encoding_table(self.n_position, self.d_hid, device)
        x = x + position_table[:, :x.size(1), ...]
        return x


class WrapFunction(nn.Module):
    """Wrap a pytorch function to nn.Module.

    It serves as partial function and can be exportable to ONNX.

    Args:
        wrapped_function (Callable): Input function to be wrapped.

    Examples:
        >>> from mmdeploy.utils.test import WrapFunction
        >>> import torch
        >>>
        >>> def clip(x, min, max):
        >>>     return torch.clamp(x, min, max)
        >>>
        >>> wrapped_model = WrapFunction(clip)
    """

    def __init__(self, wrapped_function: 'Callable', **kwargs):
        super(WrapFunction, self).__init__()
        self.wrapped_function = wrapped_function
        self.kwargs = kwargs

    def forward(self, *args, **kwargs) ->Any:
        """Call the wrapped function."""
        kwargs.update(self.kwargs)
        return self.wrapped_function(*args, **kwargs)


class WrapModel(nn.Module):
    """A wrapper class for rewrite unittests.

    It serves as partial function but can be rewritten with RewriterContext.

    Args:
        model (nn.Module): A pytorch module.
        func_name (str): Which function to use as forward function.

    Examples:
        >>> from mmdeploy.utils.test import WrapModel
        >>> from mmdet.models import AnchorHead
        >>>
        >>> model = AnchorHead(num_classes=4, in_channels=1)
        >>> wrapped_model = WrapModel(anchor_head,
        >>>                           'get_bboxes',
        >>>                           with_nms=True)
    """

    def __init__(self, model: 'nn.Module', func_name: 'str', **kwargs):
        super(WrapModel, self).__init__()
        assert hasattr(model, func_name), f'Got unexpected func name: {func_name}'
        self.model = model
        self.kwargs = kwargs
        self.func_name = func_name

    def forward(self, *args, **kwargs):
        """Run forward of the model."""
        kwargs.update(self.kwargs)
        func = getattr(self.model, self.func_name)
        return func(*args, **kwargs)


test_img = torch.rand(1, 3, 8, 8)


class DummyWrapper(torch.nn.Module):

    def __init__(self, outputs):
        self.outputs = outputs

    def __call__(self, *arg, **kwargs):
        return 0

    def output_to_list(self, *arg, **kwargs):
        return self.outputs


class ONNXNMSMatchOp(torch.autograd.Function):
    """Create onnx::NonMaxSuppressionMatch op.

    NMS_Match in mmcv only supports one class with no batch info. This class
    assists in exporting NMS_Match of ONNX's definition.
    """

    @staticmethod
    def forward(ctx, boxes: 'Tensor', scores: 'Tensor', iou_threshold: 'float', score_threshold: 'float') ->Tensor:
        """Get NMS_Match_Fake output indices.

        Args:
            ctx (Context): The context with meta information.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): score threshold of nms.

        Returns:
            Tensor: Selected indices of boxes. 2-D tensor of shape
            (num_selected_indices, 4) with each row of
            [batch_index, class_index, box_index, suppresion_index].
        """
        batch_size, num_class, _ = scores.shape
        indices = []
        score_threshold = float(score_threshold)
        iou_threshold = float(iou_threshold)
        for batch_id in range(batch_size):
            for cls_id in range(num_class):
                _boxes = boxes[batch_id, ...]
                _scores = scores[batch_id, cls_id, ...].contiguous()
                _dets = torch.cat((_boxes, _scores.unsqueeze(1)), dim=1)
                box_inds = nms_match(_dets, iou_threshold)
                batch_inds = torch.zeros(1) + batch_id
                cls_inds = torch.zeros(1) + cls_id
                both_inds = torch.cat([batch_inds, cls_inds])
                for box in box_inds:
                    if box.size() == 1:
                        continue
                    keep = box[0]
                    box = box[1:]
                    if _dets[keep][-1] < score_threshold:
                        continue
                    for supp in box:
                        indices.append(torch.cat((both_inds, keep.unsqueeze(0), supp.unsqueeze(0))))
        return torch.stack(indices)

    @staticmethod
    def symbolic(g, boxes: 'Tensor', scores: 'Tensor', iou_threshold: 'float', score_threshold: 'float'):
        """Symbolic function for mmdeploy::NMSMatch.

        Args:
            g (Graph): The traced onnx graph.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): score threshold of nms.

        Returns:
            NonMaxSuppressionMatch op for onnx.
        """
        if not sym_help._is_value(iou_threshold):
            iou_threshold = g.op('Constant', value_t=torch.tensor([iou_threshold], dtype=torch.float))
        if not sym_help._is_value(score_threshold):
            score_threshold = g.op('Constant', value_t=torch.tensor([score_threshold], dtype=torch.float))
        return g.op('mmdeploy::NMSMatch', boxes, scores, iou_threshold, score_threshold)


match_op = ONNXNMSMatchOp.apply


class test_ONNX_Match(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, boxes, scores, iou_threshold, score_threshold):
        return match_op(boxes, scores, iou_threshold, score_threshold)


class OpModel(torch.nn.Module):

    def __init__(self, func, *args):
        super().__init__()
        self._func = func
        self._arg_tuple = args

    def forward(self, x):
        return self._func(x, *self._arg_tuple)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (DummyWrapper,
     lambda: ([], {'outputs': 4}),
     lambda: ([], {})),
    (OpModel,
     lambda: ([], {'func': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (WrapFunction,
     lambda: ([], {'wrapped_function': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
]

