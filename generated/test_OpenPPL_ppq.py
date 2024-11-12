
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


import numpy as np


import torch


import torchvision


from enum import Enum


from abc import abstractmethod


from collections import deque


from typing import Any


from typing import Dict


from typing import List


from typing import Text


from typing import Union


from typing import Tuple


from typing import Callable


from typing import Iterable


from torch.utils.data import DataLoader


from numpy import dtype as np_type


from numpy import ndarray


from torch import Tensor


from torch import dtype as torch_type


from torch.cuda import empty_cache


from torch.cuda import synchronize


from torch.utils.cpp_extension import load


import time


from abc import ABCMeta


from functools import reduce


import torch.nn.functional as F


from torch import _VF


from copy import deepcopy


from typing import Optional


from typing import Type


import torch.nn as nn


from math import sqrt


import random


from random import randint


from torch.autograd import Function


from typing import Iterator


from collections import defaultdict


from functools import partial


import math


from numpy import dot


from numpy.linalg import norm


from math import ceil


from typing import Set


from abc import abstractproperty


from torchvision import models


import pandas as pd


import torch.utils.data


import torchvision.datasets as datasets


import torchvision.transforms as transforms


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.dataset import Subset


import torchvision.models


from torch import nn


from torch.nn import functional as F


import logging


from math import floor


from math import log2


class BaseGraphExecutor(Callable, metaclass=ABCMeta):
    """PPQ Base Graph Executor.

    Args:
        Callable ([type]): [description]
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """

    def __init__(self, graph: 'BaseGraph') ->dict:
        self._graph = None
        self._graph_input_dictionary = None
        self._graph_output_dictionary = None
        self._executing_order = None
        self.load_graph(graph=graph)

    def load_graph(self, graph: 'BaseGraph') ->dict:
        self._graph = graph
        self._graph_input_dictionary = self._graph.inputs
        self._graph_output_dictionary = self._graph.outputs
        self._executing_order = self._graph.topological_sort()

    def prepare_input(self, inputs: 'Union[dict, list, torch.Tensor]'):
        assert type(inputs) in (dict, list, torch.Tensor), f'Input format misunderstood. Except either dict, list or tensor; while {type(inputs)} was given.'
        inputs_dictionary = self._graph.inputs
        if len(inputs_dictionary) == 0:
            assert inputs is None, 'Graph do not need any inputs. please set your inputs to be None.'
            return None
        if isinstance(inputs, torch.Tensor):
            assert len(inputs_dictionary) == 1, 'Graph needs more than one input, while only one tensor was given.'
            return {list(inputs_dictionary.keys())[0]: inputs}
        elif isinstance(inputs, list):
            assert len(inputs_dictionary) == len(inputs), f'Inputs format misunderstood. Given inputs has {len(inputs)} elements, while graph needs {len(inputs_dictionary)}'
            return {key: inputs[idx] for idx, key in enumerate(inputs_dictionary)}
        elif isinstance(inputs, dict):
            assert len(inputs_dictionary) == len(inputs), f'Inputs format misunderstood. Given inputs has {len(inputs)} elements, while graph needs {len(inputs_dictionary)}'
            return inputs
        else:
            raise Exception('Oops, you can never reach here.')

    @abstractmethod
    def forward(self, inputs: 'Union[dict, list, torch.Tensor]', output_names: 'List[str]'=None, hooks: 'Dict[str, RuntimeHook]'=None) ->List[torch.Tensor]:
        raise NotImplementedError('Please implement this function first.')

    @abstractmethod
    def tracing_operation_meta(self, inputs: 'Union[dict, list, torch.Tensor]', output_names: 'List[str]'=None) ->None:
        raise NotImplementedError('Please implement this function first.')

    def __call__(self, inputs: 'Union[dict, torch.Tensor]', output_names: 'List[str]'=None) ->List[torch.Tensor]:
        return self.forward(inputs=inputs, output_names=output_names)

    def __str__(self) ->str:
        return f'PPQ GraphExecuter Object: {self.__hash__()}'


class DataType(Enum):
    """
        DataType defines all PPQ internal data type and its enumeration value.
            ATTENTION: PPQ shares same data type enumeration value with Onnx.

        System maintainers and modifier are supposed to keep this corresponding.
        Cause OnnxExporter directly use this value to export PPQ graph towards Onnx.
    """
    INT4 = -1
    UINT4 = -2
    INT8 = 3
    UINT8 = 2
    INT16 = 5
    UINT16 = 4
    INT32 = 6
    UINT32 = 12
    INT64 = 7
    UINT64 = 13
    FP16 = 10
    FP32 = 1
    FP64 = 11
    BOOL = 9
    COMPLEX128 = 15
    COMPLEX64 = 14
    NONETYPE = 0

    @classmethod
    def convert_from_numpy(cls, dtype: 'np_type'):
        numpy_converting_dict = {np_type('bool'): DataType.BOOL, np_type('uint8'): DataType.UINT8, np_type('int8'): DataType.INT8, np_type('int16'): DataType.INT16, np_type('int32'): DataType.INT32, np_type('int64'): DataType.INT64, np_type('float16'): DataType.FP16, np_type('float32'): DataType.FP32, np_type('float64'): DataType.FP64}
        if dtype not in numpy_converting_dict:
            raise TypeError(f'Numpy type {dtype} is not included in ppq now. please contact with system developer.')
        else:
            return numpy_converting_dict[dtype]

    @classmethod
    def convert_from_torch(cls, dtype: 'torch_type'):
        torch_converting_dict = {torch.bool: DataType.BOOL, torch.uint8: DataType.UINT8, torch.int8: DataType.INT8, torch.int16: DataType.INT16, torch.int32: DataType.INT32, torch.int64: DataType.INT64, torch.float16: DataType.FP16, torch.float32: DataType.FP32, torch.float64: DataType.FP64}
        if dtype not in torch_converting_dict:
            raise TypeError(f'Torch dtype {dtype} is not included in ppq now. please contact with system developer.')
        else:
            return torch_converting_dict[dtype]

    @classmethod
    def to_numpy(cls, dtype) ->np_type:
        numpy_converting_dict = {DataType.BOOL: np_type('bool'), DataType.UINT8: np_type('uint8'), DataType.INT8: np_type('int8'), DataType.INT16: np_type('int16'), DataType.INT32: np_type('int32'), DataType.INT64: np_type('int64'), DataType.FP16: np_type('float16'), DataType.FP32: np_type('float32'), DataType.FP64: np_type('float64')}
        assert isinstance(dtype, DataType)
        return numpy_converting_dict[dtype]

    @classmethod
    def to_torch(cls, dtype) ->torch_type:
        torch_converting_dict = {DataType.BOOL: torch.bool, DataType.UINT8: torch.uint8, DataType.INT8: torch.int8, DataType.INT16: torch.int16, DataType.INT32: torch.int32, DataType.INT64: torch.int64, DataType.FP16: torch.float16, DataType.FP32: torch.float32, DataType.FP64: torch.float64}
        assert isinstance(dtype, DataType)
        return torch_converting_dict[dtype]


class GraphCommandType(Enum):
    CONVERT_TO_TENSOR = 0
    DEPLOY_TO_CUDA = 1
    DEPLOY_TO_CPU = 2
    DEPLOY_TO_NUMPY = 3
    QUANTIZE_OPERATION = 5
    DISABLE_OPERATION_QUANTIZATION = 6
    RESTORE_OPERATION_QUANTIZATION = 7
    FORMAT_CLIP = 9
    FORMAT_PAD = 10
    FORMAT_GATHER = 11
    FORMAT_CAST = 12
    FORMAT_INT64_CONSTANT = 13
    DELETE_ISOLATED = 14
    FORMAT_PARAMETERS = 16
    REPLACE_OP = 17
    REPLACE_VAR = 18
    REMOVE_INPUT = 19
    TRAVERSAL_PATTERN_MATCHING = 20
    TRAVERSAL_OPSET_MATCHING = 21
    ACTIVATION_MATCHING = 22
    CONCAT_MATCHING = 23
    INSERT_SWITCHER = 25
    REMOVE_SWITCHER = 26
    FUSE_BN = 27
    FORMAT_CONSTANT_INPUT = 28
    FORMAT_SLICE = 29
    TRUNCATE_ON_VAR = 30
    FORMAT_RESIZE = 31
    REPLACE_BATCHNORM_TO_CONV = 32
    REPLACE_BATCHNORM_TO_SCALE = 33
    FUSE_BIAS_ADD = 34
    REMOVE_IDENTITY = 35


class GraphCommand:

    def __init__(self, command_type: 'GraphCommandType', **kwargs) ->None:
        assert isinstance(command_type, GraphCommandType), f'Command Type must be a GraphCommandType object, but {type(command_type)} received.'
        self.command_type = command_type
        self.kwargs = kwargs

    def __str__(self) ->str:
        return f'GraphCommand object {self.__hash__()},\t Command type: {self.command_type},\t Args:{self.kwargs}'


class GraphDeployCommand(GraphCommand):

    def __init__(self, device: 'str') ->None:
        if device.startswith('cuda'):
            super().__init__(GraphCommandType.DEPLOY_TO_CUDA)
        elif device.startswith('cpu'):
            super().__init__(GraphCommandType.DEPLOY_TO_CPU)
        else:
            raise ValueError(f'Device type {device} not understand.')
        self._device = device

    def __str__(self) ->str:
        return super().__str__()


def ASSERT_NUM_OF_INPUT(op: 'Operation', values: 'List[torch.Tensor]', min_num_of_input: 'int'=-1, max_num_of_input: 'int'=99):
    if min_num_of_input == max_num_of_input:
        if len(values) != min_num_of_input:
            raise ValueError(f'Can not feed value to operation {op.name}, expects exact {min_num_of_input} inputs, however {len(values)} was given')
    elif len(values) > max_num_of_input:
        raise ValueError(f'Too many input value for {op.name}, expects {max_num_of_input} inputs at most, however {len(values)} was given')
    elif len(values) < min_num_of_input:
        raise ValueError(f'Too few input value for {op.name}, expects {min_num_of_input} inputs at least, however {len(values)} was given')


def Abs_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Absolute takes one input data (Tensor) and produces one output data (Tensor) 
    where the absolute is, y = abs(x), is applied to the tensor elementwise.

    Inputs
        X (differentiable) : T
            Input tensor
    
    Outputs
        Y (differentiable) : T
            Output tensor

    Args:
        op (Operation): _description_
        values (List[torch.Tensor]): _description_
        ctx (TorchBackendContext, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return x.abs()


def AdaptiveAvgPool2d_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    input_value, output_size = values
    output = F.adaptive_avg_pool2d(input_value, output_size)
    return output


class TargetPlatform(Enum):
    """TargetPlatform is a core abstraction of PPQ framework, it defines
    "platform" as an attribute of an operation. Platform attribute of an
    operation indicates where this operation is going to be deployed. This
    feature enables PPQ to simulate inter-device computing.

    Platform attribute also tells PPQ how to quantize an operation, and how to execute it.
        ATTENTION: Different platform might bring different behaviour of a same operation.
        ATTENTION: Operation which is assigned to an non-quantizible platform will never be quantized.

    There are several supported platforms for PPQ now,
        however you are supposed to be aware of some particular platforms here:

    SHAPE_OR_INDEX is a virtual platform, however it is an EXTREMELY IMPORTANT components in PPQ.
        Dispatch an operation to platform SHAPE_OR_INDEX means this operation is SOI-related,
        it processes a SOI tensor and gives a processed SOI, all calculation of this operation must be sent to CPU
            (or any platform capable for calculating this.) when deploy.

        An operation with SHAPE_OR_INDEX platform assigned will never be quantized regardless of its type.
        It is a crucial feature for quantizing network that contains SOI-related operation. (Shufflenet etc.)

        By default, PPQ automatically detects all SOI-related operations, and dispatch them to SHAPE_OR_INDEX platform.
        To understand how this feature works, see also: ppq.sche

    UNSPECIFIED is a virtual platform, all operations are sent to this platform once they were created.
        Quantizer then dispatches them towards desired platform through its quantization logic.
    """
    MNN_INT8 = 100
    TRT_INT8 = 101
    TRT_FP8 = 105
    NCNN_INT8 = 102
    OPENVINO_INT8 = 103
    TENGINE_INT8 = 104
    ASC_INT8 = 106
    PPL_CUDA_INT8 = 201
    PPL_CUDA_INT4 = 202
    PPL_CUDA_FP16 = 203
    PPL_CUDA_MIX = 204
    PPL_DSP_INT8 = 301
    SNPE_INT8 = 302
    PPL_DSP_TI_INT8 = 303
    QNN_DSP_INT8 = 304
    HOST_INT8 = 401
    NXP_INT8 = 501
    FPGA_INT8 = 502
    RKNN_INT8 = 601
    METAX_INT8_C = 701
    METAX_INT8_T = 702
    HEXAGON_INT8 = 801
    GRAPHCORE_FP8 = 901
    FP32 = 0
    FP16 = 1
    BF16 = 2
    FP8 = 3
    INT8 = 4
    SOI = -1
    UNSPECIFIED = -2
    BOUNDARY = -3
    ONNX = -4
    CAFFE = -5
    NATIVE = -6
    ONNXRUNTIME = -7
    EXTENSION = -10086

    @classmethod
    def is_quantized_platform(cls, platform) ->bool:
        return platform in {cls.PPL_DSP_INT8, cls.PPL_DSP_TI_INT8, cls.QNN_DSP_INT8, cls.TRT_INT8, cls.NCNN_INT8, cls.NXP_INT8, cls.SNPE_INT8, cls.PPL_CUDA_INT8, cls.PPL_CUDA_INT4, cls.EXTENSION, cls.PPL_CUDA_MIX, cls.RKNN_INT8, cls.METAX_INT8_C, cls.METAX_INT8_T, cls.OPENVINO_INT8, cls.FPGA_INT8, cls.TENGINE_INT8, cls.FP8, cls.GRAPHCORE_FP8, cls.TRT_FP8, cls.ASC_INT8, cls.UNSPECIFIED, cls.INT8, cls.MNN_INT8}


def VALUE_TO_EXECUTING_DEVICE(op: 'Operation', ctx: 'TorchBackendContext', values: 'List[torch.Tensor]') ->List[torch.Tensor]:
    if ctx is None:
        device = values[0].device
    else:
        device = ctx.executing_device
    for idx, (plat, value) in enumerate(zip(op.socket.in_plat, values)):
        if value is None:
            continue
        if plat == TargetPlatform.SOI or op.platform == TargetPlatform.SOI:
            values[idx] = value.cpu()
        else:
            values[idx] = value
    return values


def Add_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Performs element-wise binary addition (with Numpy-style broadcasting
    support).

    This operator supports multidirectional (i.e., Numpy-style) broadcasting;
        for more details please check the doc.

    (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

    Inputs
        A (differentiable) : T
            First operand.
        B (differentiable) : T
            Second operand.

    Outputs
        C (differentiable) : T
            Result, has same element type as two inputs

    Args:
        op (Operation): [description]
        input_values (list): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    a, b = values
    return a + b


def And_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    a, b = values
    return a & b


def ArgMax_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    [input_value] = values
    dim = op.attributes.get('axis', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    output = torch.argmax(input_value, dim=dim, keepdim=keepdim)
    return output


def GET_ATTRIBUTE_FROM_OPERATION(op: 'Operation', attribute: 'str', compulsive: 'bool'=False, default: 'Any'=None):
    """Try to get an attribute from operation. If an attribute is compulsive,
    then operation must give a value of it, otherwise an error will be thrown.
    If an attribute is not compulsive, a default value will be given if
    operation.attributes do not holds a value of requesting attribute.

    Args:
        op (Operation): Operation instance.
        attribute (str): Attribute name.
        compulsive (bool): Whether is a compulsive attribute.
        default (Any, optional): [description]. default value of attribute.
    """
    if attribute in op.attributes:
        return op.attributes[attribute]
    elif compulsive:
        raise KeyError(f'Operation {op.name} is supposed to have a value of attribute {attribute}. ', 'However this value is missing from currecnt operation.')
    else:
        return default


def Attention_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """
    com.microsoft.Attention
        Multi-Head Self Attention that can be either unidirectional (like GPT-2) or bidirectional (like BERT). 
        The mask_index input is optional. Besides raw attention mask with shape 
        (batch_size, past_sequence_length + sequence_length) or 
        (batch_size, sequence_length, past_sequence_length + sequence_length) with value 0 for masked and 1 otherwise, 
        
        we also support other two formats: When input has right-side padding, 
        mask_index is one dimension with shape (batch_size), where value of each element is the end position, 
        or valid length of actual sequence excluding padding. When input has left-side padding, mask_index has shape (2 * batch_size), 
        where the values are the exclusive end positions followed by the inclusive start positions. When unidirectional is 1, 
        and each token only attend to previous tokens. For GPT-2, both past and present state are optional. 
        
        Present state could appear in output even when past state is not in input.

    Version
        This version of the operator has been available since version 1 of the 'com.microsoft' operator set.

    Attributes
        num_heads : int (required)
            Number of attention heads
    
        qkv_hidden_sizes : list of ints
            Hidden layer sizes of Q, K, V paths in Attention
        
        unidirectional : int
            Whether every token can only attend to previous tokens. Default value is 0.
    
    Inputs (3 - 6)
        input : T
            3D input tensor with shape (batch_size, sequence_length, input_hidden_size)
    
        weight : T
            2D input tensor with shape (input_hidden_size, 3 * hidden_size), where hidden_size = num_heads * head_size
    
        bias : T
            1D input tensor with shape (3 * hidden_size)
    
        mask_index (optional) : M
            Attention mask with shape (batch_size, 1, max_sequence_length, max_sequence_length), 
            (batch_size, past_sequence_length + sequence_length)  
            or (batch_size, sequence_length, past_sequence_length + sequence_length), 
            or index with shape (batch_size) or (2 * batch_size).

        past (optional) : T
            past state for key and value with shape (2, batch_size, num_heads, past_sequence_length, head_size).
    
        extra_add (optional) : T
            additional add to QxK' with shape (batch_size, num_heads, sequence_length, sequence_length).
    
    Outputs (1 - 2)
        output : T
            3D output tensor with shape (batch_size, sequence_length, hidden_size)
    
        present (optional) : T
            present state for key and value with shape 
            (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=6)
    num_heads = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='num_heads', compulsive=True)
    qkv_hidden_sizes = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='qkv_hidden_sizes', default=0)
    unidirectional = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='unidirectional', default=0)
    if unidirectional != 0:
        raise NotImplementedError('Attention Layer with unidirectional != 0 is not implemented.')
    input, weight, bias = values[:3]
    mask_index = values[3] if len(values) >= 4 else None
    past = values[4] if len(values) >= 5 else None
    extra_add = values[5] if len(values) >= 6 else None
    if mask_index is not None or past is not None:
        raise NotImplementedError('Attention Layer with mask_index and past != None is not implemented.')
    BATCHSIZE, NUM_OF_HEADS, HIDDEN_SIZE = input.shape[0], num_heads, qkv_hidden_sizes
    HEAD_SIZE = HIDDEN_SIZE // NUM_OF_HEADS
    qkv = torch.matmul(input, weight) + bias
    q = qkv[:, :, HIDDEN_SIZE * 0:HIDDEN_SIZE * 1]
    k = qkv[:, :, HIDDEN_SIZE * 1:HIDDEN_SIZE * 2]
    v = qkv[:, :, HIDDEN_SIZE * 2:HIDDEN_SIZE * 3]
    q = q.reshape(BATCHSIZE, -1, NUM_OF_HEADS, HEAD_SIZE).permute(0, 2, 1, 3)
    k = k.reshape(BATCHSIZE, -1, NUM_OF_HEADS, HEAD_SIZE).permute(0, 2, 1, 3)
    v = v.reshape(BATCHSIZE, -1, NUM_OF_HEADS, HEAD_SIZE).permute(0, 2, 1, 3)
    attn_score = q @ k.transpose(-2, -1) * HEAD_SIZE ** -0.5 + extra_add
    attn_score = torch.softmax(attn_score, dim=-1)
    feat = (attn_score @ v).transpose(1, 2).reshape(BATCHSIZE, -1, HIDDEN_SIZE)
    return feat


def convert_onnx_pads_to_torch(onnx_pads: 'List[int]', mode: 'str'=None) ->List[int]:
    if onnx_pads is None:
        return 0
    if isinstance(onnx_pads, int):
        return onnx_pads
    if mode is not None:
        if mode == '1d':
            assert len(onnx_pads) == 2, f'1d Operation needs 2-d padding value, while your padding value is {onnx_pads}'
        elif mode == '2d':
            assert len(onnx_pads) == 4, f'2d Operation needs 4-d padding value, while your padding value is {onnx_pads}'
        elif mode == '3d':
            assert len(onnx_pads) == 6, f'3d Operation needs 6-d padding value, while your padding value is {onnx_pads}'
    middle = len(onnx_pads) // 2
    onnx_pad_begin, onnx_pad_end = onnx_pads[:middle], onnx_pads[middle:]
    onnx_pad_begin, onnx_pad_end = onnx_pad_begin[::-1], onnx_pad_end[::-1]
    torch_pads = []
    for begin, end in zip(onnx_pad_begin, onnx_pad_end):
        torch_pads.extend([begin, end])
    if mode is None:
        return torch_pads
    if len(torch_pads) == 2:
        p1, p2 = torch_pads
        if p1 == p2:
            torch_pads = [p1]
    if len(torch_pads) == 4:
        p1, p2, p3, p4 = torch_pads
        if p1 == p2 and p3 == p4:
            torch_pads = [p1, p3]
    if len(torch_pads) == 6:
        p1, p2, p3, p4, p5, p6 = torch_pads
        if p1 == p2 and p3 == p4 and p5 == p6:
            torch_pads = [p1, p3, p5]
    return torch_pads


COLOR_END = ' \x1b[m'


G_BEGIN = '\x1b[38;5;2m'


class LEVEL(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    def __le__(self, other):
        return self.value <= other.value

    @staticmethod
    def convert(level: 'str'):
        level = level.lower()
        if level == 'debug':
            return LEVEL.DEBUG
        elif level == 'info':
            return LEVEL.INFO
        elif level == 'warning':
            return LEVEL.WARNING
        elif level == 'error':
            return LEVEL.ERROR
        else:
            raise ValueError(f'the lowercase of param level should be one of debug/info/warning/errorhowever {level} is given')


class Handler:

    def __init__(self, file_name: 'str'=None, level: 'LEVEL'=LEVEL.INFO) ->None:
        self._file_name = file_name
        self._level = level
        self._fd = sys.stdout if file_name is None else open(file_name, 'w+', encoding='utf-8')

    def process(self, msg: 'str', level: 'LEVEL') ->None:
        if self._level <= level:
            self._fd.write('\r' + msg + '\n')
            self._fd.flush()

    def set_level(self, level: 'LEVEL') ->None:
        self._level = level


R_BEGIN = '\x1b[38;5;1m'


Y_BEGIN = '\x1b[38;5;3m'


def get_current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


initialized_loggers = {}


class NaiveLogger(object):
    """A very naive implementation of colored logger, but would be suffice in
    single-process situation here where no race condition would happen."""
    __create_key = object()

    def __init__(self, create_key: 'object', name: 'str', level: 'LEVEL', file_name: 'str') ->None:
        assert create_key == NaiveLogger.__create_key, 'logger instance must be created using NaiveLogger.get_logger'
        self._name = name
        self._level = level
        self._handlers = {'stdout': Handler(level=level)}
        if file_name is not None:
            self.register_handler(file_name)

    def set_level(self, level: 'Union[str, LEVEL]'):
        if isinstance(level, str):
            level = LEVEL.convert(level)
        assert isinstance(level, LEVEL), 'level should be given by either str or LEVEL instances'
        self._level = level
        for handler in self._handlers.values():
            handler.set_level(level)

    @classmethod
    def get_logger(cls, name: 'str', level: 'Union[str, LEVEL]'=LEVEL.INFO, file_name: 'str'=None):
        if name in initialized_loggers:
            return initialized_loggers[name]
        if isinstance(level, str):
            level = LEVEL.convert(level)
        assert isinstance(level, LEVEL), 'level should be given by either str or LEVEL instances'
        logger = NaiveLogger(cls.__create_key, name, level, file_name)
        initialized_loggers[name] = logger
        return logger

    def wrap_header(self, type: 'str') ->str:
        cur_time = get_current_time()
        return '[{}][{}][{}]: '.format(type, self._name, cur_time)

    def info(self, msg: 'str'):
        header = self.wrap_header('INFO')
        print_msg = G_BEGIN + header + COLOR_END + msg
        for handler in self._handlers.values():
            if handler._file_name is not None:
                handler.process(msg, LEVEL.INFO)
            else:
                handler.process(print_msg, LEVEL.INFO)

    def warning(self, msg: 'str'):
        header = self.wrap_header('WARNING')
        print_msg = Y_BEGIN + header + COLOR_END + msg
        for handler in self._handlers.values():
            if handler._file_name is not None:
                handler.process(msg, LEVEL.WARNING)
            else:
                handler.process(print_msg, LEVEL.WARNING)

    def error(self, msg: 'str'):
        header = self.wrap_header('ERROR')
        print_msg = R_BEGIN + header + COLOR_END + msg
        for handler in self._handlers.values():
            if handler._file_name is not None:
                handler.process(msg, LEVEL.ERROR)
            else:
                handler.process(print_msg, LEVEL.ERROR)

    def debug(self, msg: 'str'):
        header = self.wrap_header('DEBUG')
        print_msg = G_BEGIN + header + COLOR_END + msg
        for handler in self._handlers.values():
            if handler._file_name is not None:
                handler.process(msg, LEVEL.DEBUG)
            else:
                handler.process(print_msg, LEVEL.DEBUG)

    def register_handler(self, file_name: 'str', level: 'Union[str, LEVEL]'=LEVEL.INFO):
        if file_name not in self._handlers:
            if isinstance(level, str):
                level = LEVEL.convert(level)
            assert isinstance(level, LEVEL), 'level should be given by either str or LEVEL instances'
            self._handlers[file_name] = Handler(file_name, level)

    def remove_handler(self, file_name: 'str'):
        if file_name in self._handlers:
            handler = self._handlers[file_name]
            if handler._file_name is not None:
                handler._fd.close()
            self._handlers.pop(file_name)


def process_attribute(attr, input_shape, kernel_shape=None, op_type=None):
    auto_pad = attr.get('auto_pad', 'NOTSET')
    strides = attr.get('strides', [1, 1])
    dilations = attr.get('dilations', [1, 1])
    kernels = attr.get('kernel_shape', kernel_shape)
    pad_needed = None
    if op_type == 'ConvTranspose' and 'output_shape' in attr:
        output_shape = attr['output_shape']
        out_pad = [0, 1] if output_shape % 2 != 0 else [0, 0]
        pad_needed = [((input_shape[i] - 1) * strides[i] + dilations[i] * (kernels[i] - 1) + 1 + out_pad[i] - output_shape[i]) for i in range(len(input_shape))]
    if auto_pad != 'NOTSET':
        if 'pads' in attr:
            logger.warning('auto_pad is conflict with pads attribute. Use pads here.')
        elif auto_pad == 'VALID':
            attr['pads'] = [0, 0, 0, 0]
        elif auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            if op_type == 'ConvTranspose':
                out_pad = attr.get('output_padding', [0, 0])
                output_shape = [(input_shape[i] * strides[i]) for i in range(len(input_shape))]
                pad_needed = [((input_shape[i] - 1) * strides[i] + dilations[i] * (kernels[i] - 1) + 1 + out_pad[i] - output_shape[i]) for i in range(len(input_shape))]
            else:
                output_shape = [((input_shape[i] + strides[i] - 1) // strides[i]) for i in range(len(input_shape))]
                pad_needed = [((output_shape[i] - 1) * strides[i] + dilations[i] * (kernels[i] - 1) + 1 - input_shape[i]) for i in range(len(input_shape))]
        else:
            raise ValueError(f'Invalid auto_pad value {auto_pad}')
    if pad_needed is not None:
        pads = []
        for item in pad_needed:
            pads.append((item if auto_pad == 'SAME_UPPER' else item + 1) // 2)
        pads = pads + [(pad_needed[i] - p) for i, p in enumerate(pads)]
        attr['pads'] = pads
        attr.pop('auto_pad')


def AveragePool_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    process_attribute(op.attributes, values[0].shape[2:])
    [x] = values
    onnx_pads = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    stride = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=None)
    ceil_mode = bool(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='ceil_mode', default=False))
    if op.type == 'GlobalAveragePool':
        kernel_size = x.size()[2:]
    else:
        kernel_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='kernel_shape', compulsive=True)
    ndim = x.ndim
    if ndim == 3:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(torch_pads, list) and len(torch_pads) != 1:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.avg_pool1d(x, kernel_size=kernel_size, padding=torch_pads, stride=stride, ceil_mode=ceil_mode)
        return output
    if ndim == 4:
        if isinstance(onnx_pads, list) and len(onnx_pads) == 4:
            p_left, p_right, p_top, p_bottom = onnx_pads[1], onnx_pads[3], onnx_pads[0], onnx_pads[2]
            if p_left == p_right and p_top == p_bottom:
                onnx_pads = [p_top, p_left]
            else:
                x = F.pad(x, pad=[p_left, p_right, p_top, p_bottom])
                onnx_pads = 0
        output = F.avg_pool2d(x, kernel_size=kernel_size, padding=onnx_pads, stride=stride, ceil_mode=ceil_mode)
        return output
    elif ndim == 5:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='3d')
        if isinstance(torch_pads, list) and len(torch_pads) != 3:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.avg_pool3d(x, kernel_size=kernel_size, padding=torch_pads, stride=stride, ceil_mode=ceil_mode)
        return output
    else:
        raise ValueError(f'Operation {op.name} is invalid, {ndim}-d input is not supported.')
    return output


def BatchNormalization_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=5, max_num_of_input=5)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_data, weight, bias, running_mean, running_var = values
    op_attr = {'eps': op.attributes.get('epsilon', 1e-05), 'momentum': 1 - op.attributes.get('momentum', 0.9)}
    output = F.batch_norm(input_data, running_mean, running_var, weight=weight, bias=bias, **op_attr)
    return output


def CaffeArgMax_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    assert len(values) == 1
    input_data = values[0]
    dim = op.attributes.get('axis', None)
    output = input_data.topk(op.attributes.get('top_k', 1), dim=dim)
    return output
    """
    # There are some gaps between ppl-argmax and standard argmax
    # If out_max_val is true, produce pairs (argmax, maxval)
    output = (output[1], output[0])
    if op.attributes.get('out_max_val', False):
        _update_output(op, output, 1)
    else:
        _update_output(op, output[0], 1)
    """


def Cast_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    [input_value] = values
    new_data_type = DataType.to_torch(op.attributes['to'])
    output = input_value
    return output


def ChannelShuffle_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    input_data = values[0]
    group = op.attributes.get('group', 1)
    assert input_data.shape[1] % group == 0
    n, c, h, w = input_data.shape
    input_data = input_data.view(n, group, c // group, h, w)
    input_data = input_data.permute(0, 2, 1, 3, 4)
    output = input_data.contiguous().view(n, c, h, w)
    return output


def Clip_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    if len(values) == 1:
        values.append(op.attributes.get('min', float('-inf')))
        values.append(op.attributes.get('max', float('+inf')))
    output = torch.clamp(*values)
    return output


def Concat_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Concatenate a list of tensors into a single tensor. All input tensors
    must have the same shape, except for the dimension size of the axis to
    concatenate on.

    Attributes
        axis : int (required)
            Which axis to concat on.
            A negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(inputs)..

    Inputs (1 - ∞)
        inputs (variadic, differentiable) : T
            List of tensors for concatenation

    Outputs
        concat_result (differentiable) : T
            Concatenated tensor

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1)
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', compulsive=True)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    concat_view = []
    for value in values:
        if value.ndim == 0:
            value = value.unsqueeze(0)
        concat_view.append(value)
    return torch.cat(concat_view, axis=axis)


def convert_any_to_python_primary_type(x: 'Union[torch.Tensor, np.ndarray, int, float, list, str]', accept_none: 'bool'=True) ->Union[int, float, list, str]:
    if x is None and accept_none:
        return None
    if x is None and not accept_none:
        raise ValueError('Trying to convert an empty value.')
    if isinstance(x, list) or isinstance(x, tuple):
        return list(x)
    elif isinstance(x, int) or isinstance(x, float):
        return x
    elif isinstance(x, torch.Tensor):
        if x.numel() == 0 and accept_none:
            return None
        if x.numel() == 0 and not accept_none:
            raise ValueError('Trying to convert an empty value.')
        if str(x.device) != 'cpu':
            x = x.cpu()
        if x.numel() == 1:
            return x.item()
        if x.numel() > 1:
            return x.tolist()
    elif isinstance(x, np.ndarray):
        if x.size == 0 and accept_none:
            return None
        if x.size == 0 and not accept_none:
            raise ValueError('Trying to convert an empty value.')
        if x.size == 1:
            return x.reshape((1,)).tolist()[0]
        if x.size > 1:
            return x.tolist()
    elif isinstance(x, str):
        return x
    else:
        raise TypeError(f'input value {x}({type(x)}) can not be converted as python primary type.')


def ConstantOfShape_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Generate a tensor with given value and shape.

    Attributes
    value : tensor
    (Optional) The value of the output elements.Should be a one-element tensor.
        If not specified, it defaults to a tensor of value 0 and datatype float32

    Inputs
        input : T1
            1D tensor. The shape of the expected output tensor.
            If empty tensor is given, the output would be a scalar. All values must be >= 0.

    Outputs
        output : T2
        Output tensor of shape specified by 'input'.If attribute 'value' is specified,
            the value and datatype of the output tensor is taken from 'value'. If attribute 'value' is not specified,
            the value in the output defaults to 0, and the datatype defaults to float32.

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    value = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='value', compulsive=False, default=0.0)
    [shape], fill_value = values, convert_any_to_python_primary_type(value)
    output = torch.Tensor().new_full(size=shape.tolist(), fill_value=fill_value)
    if isinstance(fill_value, int):
        output = output.long()
    elif isinstance(fill_value, float):
        output = output.float()
    else:
        raise TypeError(f'Can not parse value type{type(value)}.')
    return output


def Constant_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """A constant tensor. Exactly one of the two attributes, either value or
    sparse_value, must be specified.

    Version
    This version of the operator has been available since version 11 of the default ONNX operator set.

    Attributes
        sparse_value : sparse_tensor
            The value for the elements of the output tensor in sparse format.

        value : tensor
            The value for the elements of the output tensor.
    Inputs

    Outputs
        output : T
            Output tensor containing the same value of the provided tensor.

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=0, max_num_of_input=0)
    return op.attributes['value']


def ConvTranspose_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """The convolution transpose operator consumes an input tensor and a
    filter, and computes the output.

    If the pads parameter is provided the shape of the output is calculated via the following equation:

        output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]
        output_shape can also be explicitly specified in which case pads values are
            auto generated using these equations:

    total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]

    If (auto_pads == SAME_UPPER):
        pads[start_i] = total_padding[i]/2;
        pads[end_i] = total_padding[i] - (total_padding[i]/2)
    Else:
        pads[start_i] = total_padding[i] - (total_padding[i]/2);
        pads[end_i] = (total_padding[i]/2).

    Attributes
        auto_pad : string (default is NOTSET)
        auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET,
            which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that
            `output_shape[i] = input_shape[i] * strides[i]` for each axis `i`.
        The padding is split between the two sides equally or almost equally
            (depending on whether it is even or odd).
        In case the padding is an odd number,
            the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.

        dilations : list of ints
        dilation value along each spatial axis of the filter.
            If not present, the dilation defaults to 1 along each spatial axis.

        group : int (default is 1)
        number of groups input channels and output channels are divided into.

        kernel_shape : list of ints
        The shape of the convolution kernel. If not present, should be inferred from input W.

        output_padding : list of ints
        Additional elements added to the side with higher coordinate indices in the output.
            Each padding value in "output_padding" must be less than the corresponding stride/dilation dimension.
            By default, this attribute is a zero vector.
        Note that this attribute doesn't directly affect the computed output values.
            It only controls the selection of the computed values,
            so changing this attribute only adds or removes output elements.
        If "output_shape" is explicitly provided,
            "output_padding" does not contribute additional size to "output_shape"
            but participates in the computation of the needed padding amount.
            This is also called adjs or adjustment in some frameworks.

        output_shape : list of ints
        The shape of the output can be explicitly set which will cause pads values to be auto generated.
        If output_shape is specified pads values are ignored. See doc for details for equations to generate pads

        pads : list of ints
        Padding for the beginning and ending along each spatial axis,
            it can take any value greater than or equal to 0.
        The value represent the number of pixels added to the beginning and end part of the corresponding axis.
        `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...],
            where xi_begin the number of pixels added at the beginning of axis `i` and xi_end,
            the number of pixels added at the end of axis `i`.
        This attribute cannot be used simultaneously with auto_pad attribute.
        If not present, the padding defaults to 0 along start and end of each spatial axis.

        strides : list of ints
        Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.

    Inputs (2 - 3)
        X (differentiable) : T
        Input data tensor from previous layer; has size (N x C x H x W),
            where N is the batch size, C is the number of channels, and H and W are the height and width.
            Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn)

        W (differentiable) : T
        The weight tensor that will be used in the convolutions; has size (C x M/group x kH x kW),
        where C is the number of channels, and kH and kW are the height and width of the kernel,
            and M is the number of feature maps.
        For more than 2 dimensions, the weight shape will be (C x M/group x k1 x k2 x ... x kn),
            where (k1 x k2 x ... x kn) is the dimension of the kernel.
        The number of channels in the output should be equal to
            W.shape[1] * group (assuming zero based indices of the shape array)

        B (optional, differentiable) : T
        Optional 1D bias to be added to the convolution, has size of M.

    Outputs
        Y (differentiable) : T
        Output data tensor that contains the result of the convolution.
        The output dimensions are functions of the kernel size, stride size, pad lengths and group count.
        The number of channels in the output should be equal to
            W.shape[1] * group (assuming zero based indices of the shape array)

    Type Constraints
        T : tensor(float16), tensor(float), tensor(double)
        Constrain input and output types to float tensors.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    process_attribute(op.attributes, values[0].shape[2:], values[1].shape[2:], 'ConvTranspose')
    groups = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='group', default=1)
    onnx_pads = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    dilation = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='dilations', default=1)
    stride = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=1)
    output_padding = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='output_padding', default=0)
    x, w = values[:2]
    b = values[2] if len(values) > 2 else None
    ndim = x.ndim
    if ndim == 4:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='2d')
        if isinstance(torch_pads, list) and len(torch_pads) == 2:
            output = F.conv_transpose2d(input=x, weight=w, bias=b, groups=groups, padding=torch_pads, dilation=dilation, stride=stride, output_padding=output_padding)
        else:
            output = F.conv_transpose2d(input=x, weight=w, bias=b, groups=groups, padding=0, dilation=dilation, stride=stride, output_padding=output_padding)
            p1, p2, p3, p4 = torch_pads
            _, _, h, w = output.shape
            output = output[:, :, 0 + p1:h - p2, 0 + p3:w - p4]
    elif ndim in {2, 3}:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(torch_pads, list) and len(torch_pads) == 1:
            output = F.conv_transpose1d(input=x, weight=w, bias=b, groups=groups, padding=torch_pads, dilation=dilation, stride=stride, output_padding=output_padding)
        else:
            output = F.conv_transpose1d(input=x, weight=w, bias=b, groups=groups, padding=0, dilation=dilation, stride=stride, output_padding=output_padding)
            p1, p2 = torch_pads
            _, _, h = output.shape
            output = output[:, :, 0 + p1:h - p2]
    elif ndim == 5:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='3d')
        if isinstance(torch_pads, list) and len(torch_pads) == 3:
            output = F.conv_transpose3d(input=x, weight=w, bias=b, groups=groups, padding=torch_pads, dilation=dilation, stride=stride, output_padding=output_padding)
        else:
            output = F.conv_transpose3d(input=x, weight=w, bias=b, groups=groups, padding=0, dilation=dilation, stride=stride, output_padding=output_padding)
            p1, p2, p3, p4, p5, p6 = torch_pads
            _, _, d, h, w = output.shape
            output = output[:, :, 0 + p1:d - p2, 0 + p3:h - p4, 0 + p5:w - p6]
    else:
        raise ValueError(f'Operation {op.name} is invalid, {ndim}-d input is not supported.')
    return output


def Conv_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """The convolution operator consumes an input tensor and a filter, and
    computes the output.

    Attributes
        auto_pad : string (default is NOTSET)
            auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET,
            which means explicit padding is used.

            SAME_UPPER or SAME_LOWER mean pad the input so that
                `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`.
            The padding is split between the two sides equally or almost equally
                (depending on whether it is even or odd).
            In case the padding is an odd number,
                the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.

        dilations : list of ints
            dilation value along each spatial axis of the filter.
            If not present, the dilation defaults is 1 along each spatial axis.

        group : int (default is 1)
            number of groups input channels and output channels are divided into.

        kernel_shape : list of ints
            The shape of the convolution kernel. If not present, should be inferred from input W.

        pads : list of ints
            Padding for the beginning and ending along each spatial axis,
            it can take any value greater than or equal to 0.

            The value represent the number of pixels added to the beginning and end part of the corresponding axis.
            `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...],
            where xi_begin the number of pixels added at the beginning of axis `i` and xi_end,
            the number of pixels added at the end of axis `i`.

            This attribute cannot be used simultaneously with auto_pad attribute.
            If not present, the padding defaults to 0 along start and end of each spatial axis.

        strides : list of ints
            Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.

    Inputs (2 - 3)
        X (differentiable) : T
            Input data tensor from previous layer;
            has size (N x C x H x W), where N is the batch size,
            C is the number of channels, and H and W are the height and width.
            Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn).

            Optionally, if dimension denotation is in effect,
            the operation expects input data tensor to arrive
                with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].

        W (differentiable) : T
            The weight tensor that will be used in the convolutions;
            has size (M x C/group x kH x kW), where C is the number of channels,
            and kH and kW are the height and width of the kernel,
            and M is the number of feature maps. For more than 2 dimensions,
            the kernel shape will be (M x C/group x k1 x k2 x ... x kn),
                where (k1 x k2 x ... kn) is the dimension of the kernel.
            Optionally, if dimension denotation is in effect,
                the operation expects the weight tensor to arrive with the dimension
                denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...].

            Assuming zero based indices for the shape array,
            X.shape[1] == (W.shape[1] * group) == C and W.shape[0] mod G == 0.

            Or in other words FILTER_IN_CHANNEL multiplied by the number of groups should be
            equal to DATA_CHANNEL and the number of feature maps M should be a multiple of the number of groups G.

        B (optional, differentiable) : T
            Optional 1D bias to be added to the convolution, has size of M.

    Outputs
        Y (differentiable) : T
            Output data tensor that contains the result of the convolution.
            The output dimensions are functions of the kernel size, stride size, and pad lengths.

    Args:
        op ([type]): [description]
        input_value ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    groups = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='group', default=1)
    onnx_pads = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    dilation = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='dilations', default=1)
    stride = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=1)
    auto_pad = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='auto_pad', default='NOTSET')
    x, w = values[:2]
    b = values[2] if len(values) > 2 else None
    ndim = w.ndim
    if ndim in {2, 3}:
        if auto_pad != 'NOTSET':
            raise NotImplementedError(f'auto_pad must be "NOTSET" with 1-d conv {op.name}')
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(torch_pads, list) and len(torch_pads) == 2:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.conv1d(input=x, weight=w, bias=b, groups=groups, padding=torch_pads, dilation=dilation, stride=stride)
    elif ndim == 4:
        process_attribute(op.attributes, values[0].shape[2:], values[1].shape[2:])
        if isinstance(onnx_pads, list) and len(onnx_pads) == 4:
            p_left, p_right, p_top, p_bottom = onnx_pads[1], onnx_pads[3], onnx_pads[0], onnx_pads[2]
            if p_left == p_right and p_top == p_bottom:
                onnx_pads = [p_top, p_left]
            else:
                x = F.pad(x, pad=[p_left, p_right, p_top, p_bottom])
                onnx_pads = 0
        output = F.conv2d(input=x, weight=w, bias=b, groups=groups, padding=onnx_pads, dilation=dilation, stride=stride)
    elif ndim == 5:
        if auto_pad != 'NOTSET':
            raise NotImplementedError(f'auto_pad must be "NOTSET" with 3-d conv {op.name}')
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='3d')
        if isinstance(torch_pads, list) and len(torch_pads) == 6:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.conv3d(input=x, weight=w, bias=b, groups=groups, padding=torch_pads, dilation=dilation, stride=stride)
    else:
        raise ValueError(f'Operation {op.name} is invalid, {ndim}-d input is not supported.')
    return output


def Cos_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Calculates the cosine of the given input tensor, element-wise.

    Inputs
        input (differentiable) : T
            Input tensor
    
    Outputs
        output (differentiable) : T
            The cosine of the input tensor computed element-wise
    
    Type Constraints
    T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.

    Args:
        op (Operation): _description_
        values (List[torch.Tensor]): _description_
        ctx (TorchBackendContext, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return torch.cos(x)


def DepthToSpace_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_data = values[0]
    upsample = op.attributes.get('blocksize', 1)
    mode = op.attributes.get('mode', 'DCR')
    if mode == 'DCR':
        output = F.pixel_shuffle(input_data, upsample)
    else:
        output = F.pixel_shuffle(input_data, upsample)
    return output


def Eltwise_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    if op.type == 'Add':
        assert len(values) == 2
        output = torch.add(*values).float()
    elif op.type == 'Sub':
        assert len(values) == 2
        output = torch.sub(*values).float()
    elif op.type == 'Mul':
        assert len(values) == 2
        output = torch.mul(*values).float()
    elif op.type == 'Div':
        assert len(values) == 2
        version = torch.__version__
        if version < '1.5.0' or version >= '1.7.0':
            if values[0].dtype in [torch.int32, torch.int64] and values[1].dtype in [torch.int32, torch.int64]:
                if values[0].dtype == torch.int64 or values[1].dtype[1] == torch.int64:
                    output = torch.floor_divide(*values).long()
                else:
                    output = torch.floor_divide(*values).int()
            else:
                output = torch.div(*values)
        elif values[0].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            output = torch.floor_divide(*values).int()
        else:
            output = torch.div(*values)
    elif op.type == 'Max':
        output = torch.max(*values)
    elif op.type == 'Min':
        output = torch.min(*values)
    else:
        logger.warning('Not Eltwise op, return input as output')
        output = values
    return output


def Elu_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """
    Elu takes one input data (Tensor) and produces one output data (Tensor) 
    where the function f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0., 
    is applied to the tensor elementwise.

    Version
    This version of the operator has been available since version 6 of the default ONNX operator set.

    Other versions of this operator: 1

    Attributes
        alpha : float (default is 1.0)
            Coefficient of ELU.
    
    Inputs
        X (differentiable) : T
            1D input tensor
    
    Outputs
        Y (differentiable) : T
            1D output tensor
    
    Type Constraints
    T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    alpha = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='alpha', default=1.0)
    return F.elu(x, alpha=alpha)


def Equal_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_a, input_b = values
    output = torch.eq(input_a, input_b)
    return output


def Erf_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """
    Elu takes one input data (Tensor) and produces one output data (Tensor) 
    where the function f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0., 
    is applied to the tensor elementwise.

    Version
    This version of the operator has been available since version 6 of the default ONNX operator set.

    Other versions of this operator: 1

    Attributes
        alpha : float (default is 1.0)
            Coefficient of ELU.
    
    Inputs
        X (differentiable) : T
            1D input tensor
    
    Outputs
        Y (differentiable) : T
            1D output tensor
    
    Type Constraints
    T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return torch.erf(x)


def Expand_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    """Broadcast the input tensor following the given shape and the broadcast
    rule.

    The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
        Dimensions are right alignment;
        Two corresponding dimension must have the same value, or one of them is equal to 1.

    Also, this operator is similar to numpy.broadcast_to(input, shape),
    but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
    It is possible that the output.shape is not equal to shape,
    when some dimensions in shape is equal to 1, or the shape.ndim < input.shape.ndim.

    Inputs
        input (differentiable) : T
        Input tensor

        shape (non-differentiable) : tensor(int64)
        A 1-D tensor indicates the shape you want to expand to, following the broadcast rule

    Outputs
        output (differentiable) : T
        Output tensor

    Args:
        op ([type]): [description]
        input_value ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    tensor, shape = values
    output = tensor * torch.ones(convert_any_to_python_primary_type(shape), dtype=tensor.dtype, device=tensor.device)
    return output


def Flatten_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    [input_value] = values
    dim = op.attributes.get('axis', 1)
    shape = list(input_value.shape)
    new_shape = [1, -1] if dim == 0 else [reduce(operator.mul, shape[:dim], 1), -1]
    output = input_value.reshape(new_shape)
    return output


def Floor_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    input_data = values[0]
    output = torch.floor(input_data)
    return output


def GET_VALUE_FROM_INPUTS(values: 'list', idx: 'int') ->torch.Tensor:
    assert isinstance(idx, int)
    assert idx > 0
    if len(values) > idx:
        return values[idx]
    else:
        return None


GRU_FLATTEN_WEIGHT_ATTRIB = 'GRU_FLATTEN_WEIGHT_ATTRIB'


def GRU_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Computes an one-layer GRU. This operator is usually supported via some
    custom implementation such as CuDNN.
    只支持 pytorch 导出来的 GRU 啊亲; 必须要 6 个输入 Variable
    Notations:
        X - input tensor
        z - update gate
        r - reset gate
        h - hidden gate
        t - time step (t-1 means previous time step)
        W[zrh] - W parameter weight matrix for update, reset, and hidden gates
        R[zrh] - R recurrence weight matrix for update, reset, and hidden gates
        Wb[zrh] - W bias vectors for update, reset, and hidden gates
        Rb[zrh] - R bias vectors for update, reset, and hidden gates
        WB[zrh] - W parameter weight matrix for backward update, reset, and hidden gates
        RB[zrh] - R recurrence weight matrix for backward update, reset, and hidden gates
        WBb[zrh] - W bias vectors for backward update, reset, and hidden gates
        RBb[zrh] - R bias vectors for backward update, reset, and hidden gates
        H - Hidden state
        num_directions - 2 if direction == bidirectional else 1
    Activation functions:
        Relu(x)                - max(0, x)
        Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
        Sigmoid(x)             - 1/(1 + e^{-x})
    (NOTE: Below are optional)
        Affine(x)              - alpha*x + beta
        LeakyRelu(x)           - x if x >= 0 else alpha * x
        ThresholdedRelu(x)     - x if x >= alpha else 0
        ScaledTanh(x)          - alpha*Tanh(beta*x)
        HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
        Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
        Softsign(x)            - x/(1 + |x|)
        Softplus(x)            - log(1 + e^x)
    Equations (Default: f=Sigmoid, g=Tanh):
        - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
        - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
        - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
        - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
        - Ht = (1 - zt) (.) ht + zt (.) Ht-1
    This operator has optional inputs/outputs. See the doc for more details about the representation of optional arguments.
    An empty string may be used in the place of an actual argument's name to indicate a missing argument.
    Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
    Version
    This version of the operator has been available since version 14 of the default ONNX operator set.
    Other versions of this operator: 1, 3, 7
    Attributes
        activation_alpha : list of floats
            Optional scaling values used by some activation functions.
            The values are consumed in the order of activation functions,
            for example (f, g, h) in LSTM.
            Default values are the same as of corresponding ONNX operators.For example with LeakyRelu,
            the default alpha is 0.01.
        activation_beta : list of floats
            Optional scaling values used by some activation functions.
            The values are consumed in the order of activation functions,
            for example (f, g, h) in LSTM.
            Default values are the same as of corresponding ONNX operators.
        activations : list of strings
            A list of 2 (or 4 if bidirectional) activation functions for update, reset, and hidden gates.
            The activation functions must be one of the activation functions specified above.
            Optional: See the equations for default if not specified.
        clip : float
            Cell clip threshold.
            Clipping bounds the elements of a tensor in the range of [-threshold, +threshold]
            and is applied to the input of activations. No clip if not specified.
        direction : string (default is forward)
            Specify if the RNN is forward, reverse, or bidirectional.
            Must be one of forward (default), reverse, or bidirectional.
        hidden_size : int
            Number of neurons in the hidden layer
        layout : int (default is 0)
            The shape format of inputs X, initial_h and outputs Y, Y_h.
            If 0, the following shapes are expected:
                X.shape = [seq_length, batch_size, input_size],
                Y.shape = [seq_length, num_directions, batch_size, hidden_size],
                initial_h.shape = Y_h.shape = [num_directions, batch_size, hidden_size].
            If 1, the following shapes are expected:
                X.shape = [batch_size, seq_length, input_size],
                Y.shape = [batch_size, seq_length, num_directions, hidden_size],
                initial_h.shape = Y_h.shape = [batch_size, num_directions, hidden_size].
        linear_before_reset : int (default is 0)
            When computing the output of the hidden gate,
            apply the linear transformation before multiplying by the output of the reset gate.
    Inputs (3 - 6)
        X (differentiable) : T
            The input sequences packed (and potentially padded) into one 3-D tensor with the shape of
            `[seq_length, batch_size, input_size]`.
        W (differentiable) : T
            The weight tensor for the gates.
            Concatenation of `W[zrh]` and `WB[zrh]` (if bidirectional) along dimension 0.
            This tensor has shape `[num_directions, 3*hidden_size, input_size]`.
        R (differentiable) : T
            The recurrence weight tensor.
            Concatenation of `R[zrh]` and `RB[zrh]` (if bidirectional) along dimension 0.
            This tensor has shape `[num_directions, 3*hidden_size, hidden_size]`.
        B (optional, differentiable) : T
            The bias tensor for the gates.
            Concatenation of `[Wb[zrh], Rb[zrh]]` and `[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0.
            This tensor has shape `[num_directions, 6*hidden_size]`. Optional: If not specified - assumed to be 0
        sequence_lens (optional, non-differentiable) : T1
            Optional tensor specifying lengths of the sequences in a batch.
            If not specified - assumed all sequences in the batch to have length `seq_length`.
            It has shape `[batch_size]`.
        initial_h (optional, non-differentiable) : T
            Optional initial value of the hidden.
            If not specified - assumed to be 0.
            It has shape `[num_directions, batch_size, hidden_size]`.
    Outputs (0 - 2)
        Y (optional, differentiable) : T
            A tensor that concats all the intermediate output values of the hidden.
            It has shape `[seq_length, num_directions, batch_size, hidden_size]`.
        Y_h (optional, differentiable) : T
            The last output value of the hidden.
            It has shape `[num_directions, batch_size, hidden_size]`.
    Type Constraints
        T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.
        T1 : tensor(int32)
    Constrain seq_lens to integer tensor.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=6)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    x, w, r = values[:3]
    b = GET_VALUE_FROM_INPUTS(values, 3)
    seq_len = GET_VALUE_FROM_INPUTS(values, 4)
    initial_h = GET_VALUE_FROM_INPUTS(values, 5)
    activation_alpha = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_alpha', default=None)
    activation_beta = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_beta', default=None)
    activations = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activations', default=None)
    clip = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='clip', default=None)
    direction = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='direction', default='forward')
    hidden_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='hidden_size', compulsive=True)
    linear_before_reset = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='layout', default=0)
    if linear_before_reset != 0:
        raise NotImplementedError('PPQ do not support LSTM with linear_before_reset != 1.')
    if activation_alpha is not None:
        raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    if activation_beta is not None:
        raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    if activations is not None:
        raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    bidirectional = direction == 'bidirectional'
    has_bias = b is not None
    if GRU_FLATTEN_WEIGHT_ATTRIB not in op.attributes:
        forward_w = torch.cat([w[0][hidden_size * 1:hidden_size * 2], w[0][hidden_size * 0:hidden_size * 1], w[0][hidden_size * 2:hidden_size * 3]], dim=0).contiguous()
        forward_r = torch.cat([r[0][hidden_size * 1:hidden_size * 2], r[0][hidden_size * 0:hidden_size * 1], r[0][hidden_size * 2:hidden_size * 3]], dim=0).contiguous()
        if has_bias:
            forward_bias_1 = torch.cat([b[0, hidden_size * 1:hidden_size * 2], b[0, hidden_size * 0:hidden_size * 1], b[0, hidden_size * 2:hidden_size * 3]]).contiguous()
            forward_bias_2 = torch.cat([b[0, hidden_size * 4:hidden_size * 5], b[0, hidden_size * 3:hidden_size * 4], b[0, hidden_size * 5:hidden_size * 6]]).contiguous()
        if bidirectional == True:
            reverse_w = torch.cat([w[1][hidden_size * 1:hidden_size * 2], w[1][hidden_size * 0:hidden_size * 1], w[1][hidden_size * 2:hidden_size * 3]], dim=0).contiguous()
            reverse_r = torch.cat([r[1][hidden_size * 1:hidden_size * 2], r[1][hidden_size * 0:hidden_size * 1], r[1][hidden_size * 2:hidden_size * 3]], dim=0).contiguous()
            if has_bias:
                reverse_bias_1 = torch.cat([b[1, hidden_size * 1:hidden_size * 2], b[1, hidden_size * 0:hidden_size * 1], b[1, hidden_size * 2:hidden_size * 3]]).contiguous()
                reverse_bias_2 = torch.cat([b[1, hidden_size * 4:hidden_size * 5], b[1, hidden_size * 3:hidden_size * 4], b[1, hidden_size * 5:hidden_size * 6]]).contiguous()
        flatten_weight = [forward_w, forward_r]
        if has_bias:
            flatten_weight = [forward_w, forward_r, forward_bias_1, forward_bias_2]
        if bidirectional:
            flatten_weight = [forward_w, forward_r, reverse_w, reverse_r]
        if bidirectional and has_bias:
            flatten_weight = [forward_w, forward_r, forward_bias_1, forward_bias_2, reverse_w, reverse_r, reverse_bias_1, reverse_bias_2]
        op.set_extension_attrib(GRU_FLATTEN_WEIGHT_ATTRIB, flatten_weight)
    s = 2 if bidirectional else 1
    if initial_h is None:
        initial_h = torch.zeros(size=[s, x.shape[1], x.shape[2]], device=x.device, dtype=torch.float32)
    result = _VF.gru(x, initial_h, op._detail[GRU_FLATTEN_WEIGHT_ATTRIB], has_bias, 1, 0.0, False, bidirectional, False)
    hidden_vector, last_state = result
    return hidden_vector.unsqueeze(1), last_state


def GatherND_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_data, indices = values
    batch_dims = op.attributes.get('batch_dims', 0)
    data_rank = len(input_data.shape)
    assert indices.shape[-1] <= data_rank
    num_i = batch_dims
    num_k = len(input_data.shape) - num_i - indices.shape[-1]
    num_idx = indices.shape[-1]
    shape_i = indices.shape[:num_i]
    shape_j = indices.shape[num_i:-1]
    shape_k = input_data.shape[num_i + num_idx:]
    shape_idx = input_data.shape[num_i:num_i + num_idx]
    reshaped_indices = indices.reshape(*shape_i, -1, num_idx)
    strides = torch.tensor([reduce(operator.mul, shape_idx[i + 1:], 1) for i in range(num_idx)], device=input_data.device, dtype=torch.float)
    merged_indices = torch.tensordot(reshaped_indices.float(), strides, 1)
    expanded_indices = merged_indices.reshape(*merged_indices.shape, *([1] * num_k)).expand(*merged_indices.shape, *shape_k).long()
    reshaped_input = input_data.reshape(*shape_i, -1, *shape_k)
    output = reshaped_input.gather(batch_dims, expanded_indices)
    reshaped_output = output.reshape(*shape_i, *shape_j, *shape_k)
    return reshaped_output


def Gather_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Given data tensor of rank r >= 1, and indices tensor of rank q, 
    gather entries of the axis dimension of data (by default outer-most one as axis=0) indexed by indices, 
    and concatenates them in an output tensor of rank q + (r - 1).

    axis = 0 :

    Let k = indices[i_{0}, ..., i_{q-1}] 
        Then output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]

        data = [
            [1.0, 1.2],
            [2.3, 3.4],
            [4.5, 5.7],
        ]
        indices = [
            [0, 1],
            [1, 2],
        ]
        output = [
            [
                [1.0, 1.2],
                [2.3, 3.4],
            ],
            [
                [2.3, 3.4],
                [4.5, 5.7],
            ],
        ]
    
    axis = 1 :

    Let k = indices[i_{0}, ..., i_{q-1}] 
        Then output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]

        data = [
            [1.0, 1.2, 1.9],
            [2.3, 3.4, 3.9],
            [4.5, 5.7, 5.9],
        ]
        indices = [
            [0, 2],
        ]
        axis = 1,
        output = [
                [[1.0, 1.9]],
                [[2.3, 3.9]],
                [[4.5, 5.9]],
        ]
    
    Version
        This version of the operator has been available since version 13 of the default ONNX operator set.

        Other versions of this operator: 1, 11

    Attributes
        axis : int (default is 0)
            Which axis to gather on. Negative value means counting dimensions from the back. 
            Accepted range is [-r, r-1] where r = rank(data).
    
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.
    
        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, of any rank q. All index values are expected to be 
            within bounds [-s, s-1] along axis of size s. 
            
            It is an error if any of the index values are out of bounds.
    
    Outputs
        output (differentiable) : T
            Tensor of rank q + (r - 1).
    """
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_data, indices = values
    indices = indices.long()
    axis = op.attributes.get('axis', 0)
    if op.type == 'Gather':
        array_idx = [(indices if axis == i else slice(dim)) for i, dim in enumerate(input_data.shape)]
        output = input_data[array_idx]
    elif op.type == 'GatherElements':
        output = torch.gather(input_data, axis, indices)
    else:
        logger.warning('Not Gather op, return input as output')
        output = values
    return output


def Gelu_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input_value] = values
    return F.gelu(input_value)


def Gemm_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    A, B = values[:2]
    if op.opset.is_caffe() and A.ndim > 2:
        axis = op.attributes.get('axis', 1)
        A = A.flatten(start_dim=axis)
    C = values[2] if len(values) > 2 else 0
    alpha = op.attributes.get('alpha', 1.0)
    beta = op.attributes.get('beta', 1.0)
    transA = op.attributes.get('transA', 0)
    transB = op.attributes.get('transB', 0)
    A = A.transpose(0, 1) if transA else A
    B = B.transpose(0, 1) if transB else B
    output = alpha * torch.matmul(A, B) + beta * C
    return output


def Greater_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_a, input_b = values
    if input_a.dim() >= input_b.dim() or input_a.shape > input_b.shape:
        output = torch.gt(input_a, input_b)
    else:
        output = torch.lt(input_b, input_a)
    return output


def GridSampler_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """
    Given an input and a flow-field grid, computes the output using input values and pixel locations from grid. 
    Currently, only spatial (4-D) inputs are supported. 
    For input with shape (N, C, H, W) and grid with shape (N, H_out, W_out, 2), 
    the output will have shape (N, C, H_out, W_out). 
    For each output location output[N, C, H_out, W_out], 
    the size-2 vector grid[N, H_out, W_out] specifies input pixel locations x and y, 
    which are used to interpolate the output value output[N, C, H_out, W_out].

    The GridSample operator is often used in doing grid generator and sampler in the Spatial Transformer Networks. 
    See also in torch.nn.functional.grid_sample.

    Attributes
        align_corners : int (default is 0)
            If align_corners=1, the extrema (-1 and 1) are considered as referring to the center points of the input's corner pixels. 
            If align_corners=0, they are instead considered as referring to the corner points of the input's corner pixels,
            making the sampling more resolution agnostic.

        mode : string (default is bilinear)
            Three interpolation modes: bilinear (default), nearest and bicubic.
    
        padding_mode : string (default is zeros)
            Support padding modes for outside grid values: `zeros`(default), `border`, `reflection`. 
            zeros: use 0 for out-of-bound grid locations, border: use border values for out-of-bound grid locations, 
            reflection: use values at locations reflected by the border for out-of-bound grid locations.
            If index 0 represents the margin pixel, the reflected value at index -1 will be the same as the value at index 1. 
            For location far away from the border, it will keep being reflected until becoming in bound. 
            If pixel location x = -3.5 reflects by border -1 and becomes x' = 1.5, then reflects by border 1 and becomes x'' = 0.5.
    
    Inputs
        X (differentiable) : T1
            4-D tensor of shape (N, C, H, W), where N is the batch size, C is the numbers of channels, 
            H and W are the height and width of the input data.
        
        grid (non-differentiable) : T1
            Input offset, 4-D tensor of shape (N, H_out, W_out, 2), 
            where H_out and W_out are the height and width of grid and output, 
            Grid specifies the sampling pixel locations normalized by the input spatial dimensions. Therefore,
            it should have most values in the range of [-1, 1]. 
            If grid has values outside the range of [-1, 1], 
            the corresponding outputs will be handled as defined by padding_mode.
    
    Outputs
        Y (differentiable) : T2
            4-D tensor of shape (N, C, H_out, W_out).
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    value, grid = values
    output = F.grid_sample(value, grid, align_corners=False)
    return output


def HardSigmoid_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """
    HardSigmoid takes one input data (Tensor) and produces one output data (Tensor) where the HardSigmoid function, 
    y = max(0, min(1, alpha * x + beta)), is applied to the tensor elementwise.

    Attributes
        alpha : float (default is 0.2)
            Value of alpha.
    
        beta : float (default is 0.5)
            Value of beta.
    
    Inputs
        X (differentiable) : T
            Input tensor
    
    Outputs
        Y (differentiable) : T
            Output tensor
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    alpha = GET_ATTRIBUTE_FROM_OPERATION(op, 'alpha', default=0.2)
    beta = GET_ATTRIBUTE_FROM_OPERATION(op, 'beta', default=0.5)
    [value] = values
    value = alpha * value + beta
    return torch.clip(value, 0, 1)


def HardSwish_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """
    HardSwish takes one input data (Tensor) and produces one output data (Tensor) where the HardSwish function, 
        y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x), 
        where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.

    Inputs
        X (differentiable) : T
            Input tensor
    
    Outputs
        Y (differentiable) : T
        Output tensor
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [value] = values
    return F.hardswish(value)


def Identity_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    return values[0]


def InstanceNormalization_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    num_features = op.attributes.get('num_features', 1)
    eps = op.attributes.get('eps', 1e-05)
    affine = op.attributes.get('affine', False)
    if len(values) == 3:
        input_data, weight, bias = values
        running_mean, running_var = None, None
    elif len(values) == 1:
        input_data = values[0]
        running_mean, running_var, weight, bias = None, None, None, None
    else:
        raise ValueError(f'The number of input data in InstanceNom is {len(values)}')
    if affine:
        assert num_features == input_data.shape[1]
    output = F.instance_norm(input_data, running_mean, running_var, weight, bias, eps=eps)
    return output


def Interp_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    input_data = values[0]
    mode = op.attributes.get('mode', 'nearest')
    linear_mode_map = {(1): 'linear', (2): 'bilinear', (3): 'trilinear'}
    input_shape = input_data.shape
    align_corners = False if not op.attributes.get('align_corners') else True
    zoom_factor = op.attributes.get('zoom_factor', 1)
    shrink_factor = op.attributes.get('shrink_factor', 1)
    pad_beg = op.attributes.get('pad_beg', 0)
    pad_end = op.attributes.get('pad_end', 0)
    height_in_eff = input_shape[-2] + pad_beg + pad_end
    width_in_eff = input_shape[-1] + pad_beg + pad_end
    height_out = height_in_eff
    width_out = width_in_eff
    if zoom_factor != 1:
        height_out = height_in_eff + (height_in_eff - 1) * (zoom_factor - 1)
        width_out = width_in_eff + (width_in_eff - 1) * (zoom_factor - 1)
    if shrink_factor != 1:
        height_out = (height_in_eff - 1) // shrink_factor + 1
        width_out = (width_in_eff - 1) // shrink_factor + 1
    if bool(op.attributes.get('height', None)):
        height_out = op.attributes.get('height')
        width_out = op.attributes.get('width')
    if len(values) == 2:
        height_out, width_out = values[1].shape[-2:]
    sizes = list(input_shape[:2]) + [height_out, width_out]
    scales = None
    assert sizes[:2] == list(input_data.shape[:2])
    sizes = sizes[2:]
    mode = linear_mode_map[len(sizes)] if mode == 'linear' else mode
    output = F.interpolate(input_data, sizes, scales, mode, align_corners)
    return output


LSTM_FLATTEN_WEIGHT_ATTRIB = 'LSTM_FLATTEN_WEIGHT_ATTRIB'


def LSTM_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Computes an one-layer LSTM. This operator is usually supported via some
    custom implementation such as CuDNN.

    只支持 pytorch 导出来的 LSTM 啊亲; 必须要 7 个输入 Variable

    Computes an one-layer LSTM. This operator is usually supported via some custom implementation such as CuDNN.

    Notations:

    X - input tensor

    i - input gate

    o - output gate

    f - forget gate

    c - cell gate

    t - time step (t-1 means previous time step)

    W[iofc] - W parameter weight matrix for input, output, forget, and cell gates

    R[iofc] - R recurrence weight matrix for input, output, forget, and cell gates

    Wb[iofc] - W bias vectors for input, output, forget, and cell gates

    Rb[iofc] - R bias vectors for input, output, forget, and cell gates

    P[iof] - P peephole weight vector for input, output, and forget gates

    WB[iofc] - W parameter weight matrix for backward input, output, forget, and cell gates

    RB[iofc] - R recurrence weight matrix for backward input, output, forget, and cell gates

    WBb[iofc] - W bias vectors for backward input, output, forget, and cell gates

    RBb[iofc] - R bias vectors for backward input, output, forget, and cell gates

    PB[iof] - P peephole weight vector for backward input, output, and forget gates

    H - Hidden state

    num_directions - 2 if direction == bidirectional else 1

    Activation functions:

    Relu(x)                - max(0, x)

    Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

    Sigmoid(x)             - 1/(1 + e^{-x})

    (NOTE: Below are optional)

    Affine(x)              - alpha*x + beta

    LeakyRelu(x)           - x if x >= 0 else alpha * x

    ThresholdedRelu(x)     - x if x >= alpha else 0

    ScaledTanh(x)          - alpha*Tanh(beta*x)

    HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

    Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

    Softsign(x)            - x/(1 + |x|)

    Softplus(x)            - log(1 + e^x)
    Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

    - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)

    - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)

    - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

    - Ct = ft (.) Ct-1 + it (.) ct

    - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)

    - Ht = ot (.) h(Ct)
    This operator has optional inputs/outputs. 
    See the doc for more details about the representation of optional arguments. 
    An empty string may be used in the place of an actual argument's name to indicate a missing argument. 
    Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

    Version
    This version of the operator has been available since version 7 of the default ONNX operator set.

    Attributes
        activation_alpha : list of floats
            Optional scaling values used by some activation functions. 
            The values are consumed in the order of activation functions, 
                for example (f, g, h) in LSTM. 
    
            Default values are the same as of corresponding ONNX operators.For example with LeakyRelu, the default alpha is 0.01.
    
        activation_beta : list of floats
            Optional scaling values used by some activation functions. 
            The values are consumed in the order of activation functions, 
            for example (f, g, h) in LSTM. 
            
            Default values are the same as of corresponding ONNX operators.
    
        activations : list of strings
            A list of 3 (or 6 if bidirectional) activation functions for input, 
            output, forget, cell, and hidden. 
            
            The activation functions must be one of the activation functions specified above. 
            Optional: See the equations for default if not specified.
    
        clip : float
            Cell clip threshold. Clipping bounds the elements of a tensor in the range of 
            [-threshold, +threshold] and is applied to the input of activations.
            No clip if not specified.
        
        direction : string (default is forward)
            Specify if the RNN is forward, reverse, or bidirectional. 
            Must be one of forward (default), reverse, or bidirectional.

        hidden_size : int
            Number of neurons in the hidden layer
    
        input_forget : int (default is 0)
            Couple the input and forget gates if 1.
    
    Inputs (3 - 8)
        X : T
            The input sequences packed (and potentially padded) into one 3-D tensor 
                with the shape of `[seq_length, batch_size, input_size]`.
   
        W : T
            The weight tensor for the gates. Concatenation of `W[iofc]` and `WB[iofc]` 
            (if bidirectional) along dimension 0. The tensor has shape `[num_directions, 4*hidden_size, input_size]`.
    
        R : T
            The recurrence weight tensor. Concatenation of `R[iofc]` and `RB[iofc]` (if bidirectional) along dimension 0. 
            This tensor has shape `[num_directions, 4*hidden_size, hidden_size]`.
    
        B (optional) : T
            The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, 
            and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. 
            
            This tensor has shape `[num_directions, 8*hidden_size]`. 
            Optional: If not specified - assumed to be 0.
    
        sequence_lens (optional) : T1
            Optional tensor specifying lengths of the sequences in a batch. 
            If not specified - assumed all sequences in the batch to have length `seq_length`. 
            It has shape `[batch_size]`.
        
        initial_h (optional) : T
            Optional initial value of the hidden. 
            If not specified - assumed to be 0. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    
        initial_c (optional) : T
            Optional initial value of the cell. 
            If not specified - assumed to be 0. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    
        P (optional) : T
            The weight tensor for peepholes.
            Concatenation of `P[iof]` and `PB[iof]` (if bidirectional) along dimension 0. 
            It has shape `[num_directions, 3*hidde_size]`. Optional: If not specified - assumed to be 0.
    
    Outputs (0 - 3)
        Y (optional) : T
            A tensor that concats all the intermediate output values of the hidden. 
            It has shape `[seq_length, num_directions, batch_size, hidden_size]`.
    
        Y_h (optional) : T
            The last output value of the hidden. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    
        Y_c (optional) : T
            The last output value of the cell. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=8)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    x, w, r = values[:3]
    b = GET_VALUE_FROM_INPUTS(values, 3)
    seq_len = GET_VALUE_FROM_INPUTS(values, 4)
    initial_h = GET_VALUE_FROM_INPUTS(values, 5)
    initial_c = GET_VALUE_FROM_INPUTS(values, 6)
    p = GET_VALUE_FROM_INPUTS(values, 7)
    if p is not None:
        raise NotImplementedError('PPQ do not support LSTM with peepholes.')
    activation_alpha = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_alpha', default=None)
    activation_beta = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_beta', default=None)
    activations = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activations', default=None)
    clip = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='clip', default=None)
    direction = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='direction', default='forward')
    hidden_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='hidden_size', compulsive=True)
    input_forget = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='input_forget', default=0)
    layout = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='layout', default=0)
    if layout != 0:
        raise NotImplementedError('PPQ do not support LSTM with layout != 1.')
    if activation_alpha is not None:
        raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    if activation_beta is not None:
        raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    if activations is not None:
        raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    bidirectional = direction == 'bidirectional'
    has_bias = b is not None
    if direction == 'reverse':
        raise NotImplementedError('GRU do not support reverse mode now.')
    if LSTM_FLATTEN_WEIGHT_ATTRIB not in op.attributes:
        forward_w = torch.cat([w[0][hidden_size * 0:hidden_size * 1], w[0][hidden_size * 2:hidden_size * 3], w[0][hidden_size * 3:hidden_size * 4], w[0][hidden_size * 1:hidden_size * 2]], dim=0).contiguous()
        forward_r = torch.cat([r[0][hidden_size * 0:hidden_size * 1], r[0][hidden_size * 2:hidden_size * 3], r[0][hidden_size * 3:hidden_size * 4], r[0][hidden_size * 1:hidden_size * 2]], dim=0).contiguous()
        if has_bias:
            forward_bias_1 = torch.cat([b[0, hidden_size * 0:hidden_size * 1], b[0, hidden_size * 2:hidden_size * 3], b[0, hidden_size * 3:hidden_size * 4], b[0, hidden_size * 1:hidden_size * 2]]).contiguous()
            forward_bias_2 = torch.cat([b[0, hidden_size * 4:hidden_size * 5], b[0, hidden_size * 6:hidden_size * 7], b[0, hidden_size * 7:hidden_size * 8], b[0, hidden_size * 5:hidden_size * 6]]).contiguous()
        if bidirectional == True:
            reverse_w = torch.cat([w[1][hidden_size * 0:hidden_size * 1], w[1][hidden_size * 2:hidden_size * 3], w[1][hidden_size * 3:hidden_size * 4], w[1][hidden_size * 1:hidden_size * 2]], dim=0).contiguous()
            reverse_r = torch.cat([r[1][hidden_size * 0:hidden_size * 1], r[1][hidden_size * 2:hidden_size * 3], r[1][hidden_size * 3:hidden_size * 4], r[1][hidden_size * 1:hidden_size * 2]], dim=0).contiguous()
            if has_bias:
                reverse_bias_1 = torch.cat([b[1, hidden_size * 0:hidden_size * 1], b[1, hidden_size * 2:hidden_size * 3], b[1, hidden_size * 3:hidden_size * 4], b[1, hidden_size * 1:hidden_size * 2]]).contiguous()
                reverse_bias_2 = torch.cat([b[1, hidden_size * 4:hidden_size * 5], b[1, hidden_size * 6:hidden_size * 7], b[1, hidden_size * 7:hidden_size * 8], b[1, hidden_size * 5:hidden_size * 6]]).contiguous()
        flatten_weight = [forward_w, forward_r]
        if has_bias:
            flatten_weight = [forward_w, forward_r, forward_bias_1, forward_bias_2]
        if bidirectional:
            flatten_weight = [forward_w, forward_r, reverse_w, reverse_r]
        if bidirectional and has_bias:
            flatten_weight = [forward_w, forward_r, forward_bias_1, forward_bias_2, reverse_w, reverse_r, reverse_bias_1, reverse_bias_2]
        op.set_extension_attrib(LSTM_FLATTEN_WEIGHT_ATTRIB, flatten_weight)
    s = 2 if bidirectional else 1
    if initial_h is None:
        initial_h = torch.zeros(size=[s, x.shape[1], hidden_size], device=x.device, dtype=torch.float32)
    if initial_c is None:
        initial_c = torch.zeros(size=[s, x.shape[1], hidden_size], device=x.device, dtype=torch.float32)
    result = _VF.lstm(x.contiguous(), (initial_h.contiguous(), initial_c.contiguous()), op._detail[LSTM_FLATTEN_WEIGHT_ATTRIB], has_bias, 1, 0.0, False, bidirectional, False)
    hs, h, c = result
    if bidirectional:
        hs = hs.reshape((hs.shape[0], hs.shape[1], 2, hs.shape[-1] // 2))
        hs = hs.permute((0, 2, 1, 3))
    else:
        hs = hs.reshape((hs.shape[0], hs.shape[1], 1, hs.shape[-1]))
        hs = hs.permute((0, 2, 1, 3))
    return hs, h, c


def LayerNorm_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    """
    This is layer normalization defined in ONNX as function. 
    The overall computation can be split into two stages. 
    The first stage is standardization, which makes the normalized elements have zero mean and unit variances. 
    The computation required by standardization can be described by the following equations. 
    Mean = ReduceMean<axes=normalized_axes>(X) D = Sub(X, Mean) DD = Mul(Diff, Diff) 
    Var = ReduceMean<axes=normalized_axes>(DD) VarEps = Add(Var, epsilon) 
    StdDev = Sqrt(VarEps) InvStdDev = Reciprocal(StdDev) 
    Normalized = Mul(D, InvStdDev) where normalized_axes is [axis, ..., rank of X - 1]. 
    
    The variables Var and StdDev stand for variance and standard deviation, respectively. 
    The second output is Mean and the last one is InvStdDev. Depending on stash_type attribute, 
    the actual computation must happen in different floating-point precision. 
    
    For example, if stash_type is 1, this operator casts all input variables to 32-bit float, perform the computation, 
    and finally cast Normalized back to the original type of X. 
    
    The second stage then scales and shifts the outcome of the first stage using
        NormalizedScaled = Mul(Normalized, Scale) 
        Y = Add(NormalizedScaled, B) 
    The second stage doesn't depends on stash_type. All equations are in this syntax. 
    
    The same variable (i.e., input, output, and attribute) uses the same name in the equations above and this operator's definition. 
    Let d[i] indicate the i-th dimension of X. If X's shape is [d[0], ..., d[axis-1], d[axis], ..., d[rank-1]], 
    the shape of Mean and InvStdDev is [d[0], ..., d[axis-1], 1, ..., 1]. Y and X have the same shape.

    Version
    This version of the operator has been available since version 17 of the default ONNX operator set.

    Attributes
        axis : int (default is -1)
            The first normalization dimension. If rank(X) is r, axis' allowed range is [-r, r]. 
            Negative value means counting dimensions from the back.

        epsilon : float (default is 1e-05)
            The epsilon value to use to avoid division by zero.

        stash_type : int (default is 1)
            Type of Mean and InvStdDev. 
            This also specifies stage one's computation precision.

    Inputs (2 - 3)
        X : T
            Tensor to be normalized.
        
        Scale : T
            Scale tensor.
        
        B (optional) : T
            Bias tensor.
    
    Outputs (1 - 3)
        Y : T
            Normalized tensor.
    
        Mean (optional) : U
            Saved mean used during training to speed up gradient computation
        
        InvStdDev (optional) : U
            Saved inverse standard deviation used during training to speed up gradient computation.

    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    x, weight = values[0], values[1]
    if len(values) == 3:
        bias = values[-1]
    else:
        bias = None
    eps = op.attributes.get('epsilon', 1e-05)
    axis = op.attributes.get('axis', -1)
    if axis != -1 and axis != x.ndim - 1:
        raise ValueError('Unsupported Layernorm axis. We will implement it soon.')
    normalized_shape = weight.shape
    output = F.layer_norm(x, normalized_shape, weight, bias, eps)
    return output


def LeakyRelu_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    [input_data] = values
    alpha = op.attributes.get('alpha', 0.01)
    output = F.leaky_relu(input_data, alpha)
    return output


def Less_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_a, input_b = values
    if input_a.dim() >= input_b.dim() or input_a.shape > input_b.shape:
        output = torch.lt(input_a, input_b)
    else:
        output = torch.gt(input_b, input_a)
    return output


def Log_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    input_data = values[0]
    output = torch.log(input_data)
    return output


def Softmax_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """The operator computes the normalized exponential values for the given
    input:

    Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)

    The "axis" attribute indicates the dimension along which Softmax will be performed.
    The output tensor has the same shape and contains the Softmax values of the corresponding input.

    Attributes
        axis : int (default is -1)
            Describes the dimension Softmax will be performed on.
            Negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(input).

    Inputs
        input (differentiable) : T
        The input tensor of rank >= axis.

    Outputs
        output (differentiable) : T
        The output values with the same shape as the input tensor.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    default_axis = -1 if op.opset.onnx_opset_version() >= 13 else 1
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input] = values
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', default=default_axis)
    output = F.softmax(input, axis)
    return output


def LogSoftmax_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    x = Softmax_forward(op=op, values=values, ctx=ctx, kwargs=kwargs)
    x = Log_forward(op=op, values=[x], ctx=ctx, kwargs=kwargs)
    return x


def FORCE_CONVERT_DEVICE(value: 'torch.Tensor', device: 'str') ->torch.Tensor:
    return value


def MMCVRoiAlign_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    data, rois = values
    rois = FORCE_CONVERT_DEVICE(rois, device=data.device)
    mode = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='mode', default='avg')
    aligned = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='aligned', default=True)
    output_height = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='output_height', default=1)
    output_width = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='output_width', default=1)
    sampling_ratio = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='sampling_ratio', default=0)
    spatial_scale = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='spatial_scale', default=1.0)
    output_size = output_height, output_width
    if rois.shape[0] == 0:
        output = torch.empty([0, data.shape[1], 14, 14])
    else:
        output = mmcv_roi_align(data, rois, output_size, spatial_scale, sampling_ratio, mode, aligned)
    return output


def MatMul_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """
    Matrix product that behaves like numpy.matmul: 
        https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html

    Version
        This version of the operator has been available since version 13 of the default ONNX operator set.

        Other versions of this operator: 1, 9

    Inputs
        A (differentiable) : T
            N-dimensional matrix A
    
        B (differentiable) : T
            N-dimensional matrix B
    
    Outputs
        Y (differentiable) : T
            Matrix multiply results from A * B
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    output = torch.matmul(values[0], values[1])
    return output


def MaxPool2d_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    process_attribute(op.attributes, values[0].shape[2:])
    [x] = values
    onnx_pads = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    dilation = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='dilations', default=1)
    stride = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=None)
    ceil_mode = bool(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='ceil_mode', default=False))
    if op.type == 'GlobalMaxPool':
        kernel_size = x.size()[2:]
    else:
        kernel_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='kernel_shape', compulsive=True)
    ndim = x.ndim
    if ndim == 5:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='3d')
        if isinstance(torch_pads, list) and len(torch_pads) != 3:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.max_pool3d(x, kernel_size=kernel_size, padding=torch_pads, dilation=dilation, stride=stride, ceil_mode=ceil_mode)
    elif ndim == 4:
        if isinstance(onnx_pads, list) and len(onnx_pads) == 4:
            p_left, p_right, p_top, p_bottom = onnx_pads[1], onnx_pads[3], onnx_pads[0], onnx_pads[2]
            if p_left == p_right and p_top == p_bottom:
                onnx_pads = [p_top, p_left]
            else:
                x = F.pad(x, pad=convert_onnx_pads_to_torch(onnx_pads), value=float('-inf'))
                onnx_pads = 0
        output = F.max_pool2d(x, kernel_size=kernel_size, padding=onnx_pads, dilation=dilation, stride=stride, ceil_mode=ceil_mode)
    elif ndim in {2, 3}:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(torch_pads, list) and len(torch_pads) != 1:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.max_pool1d(x, kernel_size=kernel_size, padding=torch_pads, dilation=dilation, stride=stride, ceil_mode=ceil_mode)
    else:
        raise ValueError(f'Operation {op.name} is invalid, {ndim}-d input is not supported.')
    return output


def Mul_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Performs element-wise binary multiplication (with Numpy-style
    broadcasting support).

    This operator supports multidirectional (i.e., Numpy-style) broadcasting; for more details please check the doc.

    (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

    Inputs
        A (differentiable) : T
            First operand.

        B (differentiable) : T
            Second operand.

    Outputs
        C (differentiable) : T
            Result, has same element type as two inputs

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    multiplicand, multiplier = values
    return multiplicand * multiplier


def MultiHeadAttention_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Perform MultiHeadAttetion opr forward.

    Args:
        op (Operation): MultiHeadAttention
        values (List[torch.Tensor]): opr inputs
        ctx (TorchBackendContext, optional): Context. Defaults to None.

    Raises:
        NotImplementedError: In [Vit Paper](https://arxiv.org/abs/2010.11929), MultiHeadAttention inputs are actually the same tensor, we suppose that this would **not** be simplified.
        ValueError: MultiHeadAttention contains `embed_dim` and `num_heads`.

    Returns:
        list: opr output and internal result for quantization.
    """
    if len(values) != 11:
        raise NotImplementedError('Not implement simplified MultiHeadAttention')
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    q_in, k_in, v_in, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b = values
    embed_dim = op.attributes.get('embed_dim')
    num_heads = op.attributes.get('num_heads')
    if embed_dim is None or num_heads is None:
        raise ValueError('Cannot fetch embed_dim or num_heads')
    batch_size = q_in.shape[0]
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5
    xq = F.linear(q_in, q_w, q_b)
    xk = F.linear(k_in, k_w, k_b)
    xv = F.linear(v_in, v_w, v_b)
    B, N, _ = xq.shape
    q = xq.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    k = xk.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    v = xv.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    energy = q @ k.transpose(-2, -1) * scale
    attn = energy.softmax(dim=-1)
    feat = (attn @ v).transpose(1, 2).reshape(batch_size, -1, embed_dim)
    out = F.linear(feat, o_w, o_b)
    return out


def Neg_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """
    Neg takes one input data (Tensor) and produces one output data (Tensor)
    where each element flipped sign, y = -x, is applied to the tensor elementwise.

    Inputs
        X (differentiable) : T
        Input tensor

    Outputs
        Y (differentiable) : T
        Output tensor

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return -x


def NonZero_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    [input_value] = values
    output = torch.nonzero(input_value, as_tuple=True)
    output = torch.stack(output)
    return output


def Not_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [value] = values
    return ~value


def Onehot_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Produces a one-hot tensor based on inputs. The locations represented by
    the index values in the 'indices' input tensor will have 'on_value' and the
    other locations will have 'off_value' in the output tensor,

    where 'on_value' and 'off_value' are specified as part of required input argument 'values',
    which is a two-element tensor of format [off_value, on_value].

    The rank of the output tensor will be one greater than the rank of the input tensor.
    The additional dimension is for one-hot representation. The additional dimension will be inserted at the position specified by 'axis'.
    If 'axis' is not specified then then additional dimension will be inserted as the innermost dimension,
    i.e. axis=-1. The size of the additional dimension is specified by required scalar input 'depth'.

    The type of the output tensor is the same as the type of the 'values' input. Any entries in the 'indices'
    input tensor with values outside the range [-depth, depth-1] will result in one-hot representation
    with all 'off_value' values in the output tensor.

    when axis = 0:
    output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.

    when axis = -1:
    output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.
    Version
    This version of the operator has been available since version 11 of the default ONNX operator set.

    Attributes
    axis : int (default is -1)
    (Optional) Axis along which one-hot representation in added. Default: axis=-1. axis=-1 means that
        the additional dimension will be inserted as the innermost/last dimension in the output tensor.
    Negative value means counting dimensions from the back. Accepted range is [-r-1, r] where r = rank(indices).

    Inputs
    indices (non-differentiable) : T1
        Input tensor containing indices. Any entries in the 'indices' input tensor with values outside the range [-depth, depth-1]
            will result in one-hot representation with all 'off_value' values in the output tensor.In case 'indices' is of non-integer type,
            the values will be casted to int64 before use.

    depth (non-differentiable) : T2
        Scalar specifying the number of classes in one-hot tensor.
        This is also the size of the one-hot dimension (specified by 'axis' attribute) added on in the output tensor.
            The values in the 'indices' input tensor are expected to be in the range [-depth, depth-1].
            In case 'depth' is of non-integer type, it will be casted to int64 before use.

    values (non-differentiable) : T3
        Rank 1 tensor containing exactly two elements,
        in the format [off_value, on_value], where 'on_value' is the value used for filling locations specified in 'indices' input tensor,
        and 'off_value' is the value used for filling locations other than those specified in 'indices' input tensor.

    Outputs
    output (non-differentiable) : T3
        Tensor of rank one greater than input tensor 'indices', i.e. rank(output) = rank(indices) + 1.
        The data type for the elements of the output tensor is the same as the type of input 'values' is used.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', default=-1)
    indices, depth, values = values
    off_value, on_value = values
    out = F.one_hot(indices, depth.item())
    out = out * (on_value - off_value) + off_value
    rank = len(indices.shape)
    if axis < 0:
        axis += rank + 1
    if not rank == axis:
        order = list(range(len(indices.shape)))
        order.insert(axis, -1)
        out = out.permute(order)
    return out


def PPQBiasFusedMatMul_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """
    PPQ Special Edition of MatMul
        Matrix product that behaves like numpy.matmul: 
        https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html

    Version
        This version of the operator has been available since version 13 of the default ONNX operator set.

        Other versions of this operator: 1, 9

    Inputs
        A (differentiable) : T
            N-dimensional matrix A
    
        B (differentiable) : T
            N-dimensional matrix B
            
        C (Optional) (differentiable) : T
            Bias Tensor Of MatMul
    
    Outputs
        Y (differentiable) : T
            Matrix multiply results from A * B

    Args:
        op (Operation): _description_
        values (List[torch.Tensor]): _description_
        ctx (TorchBackendContext, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    output = torch.matmul(values[0], values[1])
    if len(values) == 3:
        output += values[-1]
    return output


def PPQDeviceSwitch_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    [value] = values
    return value


def PRelu_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    input_data, weight = values
    output = F.prelu(input_data, weight.squeeze())
    return output


def Pad_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    """
    Given a tensor containing the data to be padded (data), 
        a tensor containing the number of start and end pad values for axis (pads), 
        (optionally) a mode, and (optionally) constant_value, a padded tensor (output) is generated.

    The three supported modes are (similar to corresponding modes supported by numpy.pad):

        constant(default) - pads with a given constant value as specified by constant_value 
            (which defaults to 0, empty string, or False)

        reflect - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

        edge - pads with the edge values of array

    Version
        This version of the operator has been available since version 18 of the default ONNX operator set.

        Other versions of this operator: 1, 2, 11, 13

    Attributes
        mode : string (default is constant)
        Supported modes: `constant`(default), `reflect`, `edge`
    
    Inputs (2 - 4)
        data (differentiable) : T
            Input tensor.
    
        pads (non-differentiable) : tensor(int64)
            Tensor of integers indicating the number of padding elements to add or remove 
            (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. 
            `pads` should be a 1D tensor of shape [2 * num_axes] where `num_axes` refers to the number of elements 
            in the `axes` input or the input rank if `axes` are not provided explicitly. 
            
            `pads` format should be: [x1_begin, x2_begin, ..., x1_end, x2_end,...], 
            where xi_begin is the number of pad values added at the beginning of 
            axis `axes[i]` and xi_end, the number of pad values added at the end of axis `axes[i]`.
    
        constant_value (optional, non-differentiable) : T
            (Optional) A scalar value to be used if the mode chosen is `constant` (by default it is 0, empty string or False).
        
        axes (optional, non-differentiable) : Tind
            1-D tensor of axes that `pads` apply to. Negative value means counting dimensions from the back. 
            Accepted range is [-r, r-1] where r = rank(data). Behavior is undefined if an axis is repeated. 
            If not provided, all axes are assumed (`[0, 1, ..., input_rank-1]`).
    
    Outputs
        output (differentiable) : T
            Tensor after padding.

    """
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    mode = op.attributes.get('mode', 'constant')
    value, pads = values[0], values[1]
    if len(values) > 3:
        raise ValueError('Unsupported pad version, except at most 2 input variable.')
    pads = pads.tolist()
    pads = convert_onnx_pads_to_torch(pads)
    if mode == 'constant':
        if len(values) == 3 and values[-1] is None:
            constant_value = 0
        else:
            constant_value = values[-1].item() if len(values) == 3 else 0
        output = F.pad(value, pads, mode, constant_value)
    elif mode == 'reflect':
        output = value
        while len(pads) > 4:
            output = F.pad(value, pads[-4:], mode)
            pads = pads[:-4]
        output = F.pad(value, pads, mode)
    elif mode == 'edge':
        output = F.pad(value, pads, 'replicate')
    else:
        raise TypeError(f'Unsupported mode {mode} in Pad op')
    return output


def Parameter_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    input_data = values[0]
    m = op.attributes.get('m', -1)
    n = op.attributes.get('n', -1)
    output = input_data.reshape(m, n)
    return output


def Pow_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_data = values[0]
    power = op.attributes.get('power', 1)
    scale = op.attributes.get('scale', 1)
    shift = op.attributes.get('shift', 0)
    if len(values) == 2:
        power = values[1]
        scale, shift = 1.0, 0.0
    output = torch.pow(input_data * scale + shift, power)
    return output


def Range_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    """
    Generate a tensor containing a sequence of numbers that 
        begin at start and extends by increments of delta up to limit (exclusive).

    The number of elements in the output of range is computed as below-

    number_of_elements = max( ceil( (limit - start) / delta ) , 0 )

    The pseudocode determining the contents of the output is shown below-

    for(int i=0; i<number_of_elements; ++i)

    {

        output[i] = start + (i * delta);

    }

    Example 1 Inputs: start = 3, limit = 9, delta = 3 Output: [3, 6]

    Example 2 Inputs: start = 10, limit = 4, delta = -2 Output: [10, 8, 6]

    Inputs
        start : T
            Scalar. First entry for the range of output values.
    
        limit : T
            Scalar. Exclusive upper limit for the range of output values.
    
        delta : T
            Scalar. Value to step by.
    
    Outputs
        output : T
            A 1-D tensor with same type as the inputs containing generated range of values.
    """
    start, limit, delta = values
    start = start.item()
    limit = limit.item()
    delta = delta.item()
    output = torch.arange(start, limit, delta, device=ctx.executing_device)
    return output


def Reciprocal_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """
    Reciprocal takes one input data (Tensor) and produces one output data (Tensor) where the reciprocal is,
        y = 1/x, is applied to the tensor elementwise.

    Version
        This version of the operator has been available since version 13 of the default ONNX operator set.

    Inputs
        X (differentiable) : T Input tensor
    Outputs
        Y (differentiable) : T Output tensor

    Constrain input and output types to float tensors.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return 1 / x


def ReduceL2_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    [input_value] = values
    axis = op.attributes['axes']
    keepdim = bool(op.attributes.get('keepdims', 1))
    output = torch.norm(input_value, dim=axis, keepdim=keepdim)
    if axis is None and keepdim:
        output = output.reshape([1] * input_value.dim())
    return output


def ReduceMax_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    [input_value] = values
    dim = op.attributes.get('axes', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    if len(input_value) == 0:
        output = input_value
    elif dim is None:
        output = torch.max(input_value)
        if keepdim:
            output = output.reshape([1] * input_value.dim())
    else:
        output, _ = torch.max(input_value, dim=dim[0], keepdim=keepdim)
    return output


def ReduceMean_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    [input_value] = values
    dim = op.attributes.get('axes', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    if len(input_value) == 0:
        output = input_value
    elif dim is None:
        output = torch.mean(input_value)
        if keepdim:
            output = output.reshape([1] * input_value.dim())
    else:
        output = torch.mean(input_value, dim=dim, keepdim=keepdim)
    return output


def ReduceSum_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    if op.opset.onnx_opset_version() >= 13:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=2)
        values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
        input_value, dim = values[0], None
        if len(values) > 1:
            dim = values[1]
        keepdim, noop_with_empty_axes = bool(op.attributes.get('keepdims', 1)), op.attributes.get('noop_with_empty_axes', 0)
        if dim is None:
            if noop_with_empty_axes:
                return input_value
            else:
                output = torch.sum(input_value)
                if keepdim:
                    output = output.reshape([1] * input_value.dim())
                return output
        else:
            dim = dim.tolist()
            if isinstance(dim, int):
                dim = [dim]
            output = torch.sum(input_value, dim=dim, keepdim=keepdim)
            return output
    [input_value] = values
    dim = op.attributes.get('axes', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    if dim is None:
        output = torch.sum(input_value)
        if keepdim:
            output = output.reshape([1] * input_value.dim())
    else:
        output = torch.sum(input_value, dim=dim, keepdim=keepdim)
    return output


def Reshape_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Reshape the input tensor similar to numpy.reshape. First input is the
    data tensor, second input is a shape tensor which specifies the output
    shape. It outputs the reshaped tensor.

    At most one dimension of the new shape can be -1.
    In this case, the value is inferred from the size of the tensor and the remaining dimensions.
    A dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor).
    If 'allowzero' is set, and the new shape includes 0,
        the dimension will be set explicitly to zero (i.e. not taken from input tensor)

    Attributes
        allowzero : int (default is 0)
        (Optional) By default, when any value in the 'shape' input is equal to zero
            the corresponding dimension value is copied from the input tensor dynamically.

        allowzero=1 indicates that if any value in the 'shape' input is set to zero,
            the zero value is honored, similar to NumPy.
    Inputs
        data (differentiable) : T
            An input tensor.

        shape (non-differentiable) : tensor(int64)
            Specified shape for output.

    Outputs
        reshaped (differentiable) : T
            Reshaped data.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    data, shape = values
    shape = shape.cpu()
    shape = [(shape[i] if shape[i] != 0 else data.shape[i]) for i in range(len(shape))]
    shape = [(shape[i].item() if hasattr(shape[i], 'item') else shape[i]) for i in range(len(shape))]
    return data.reshape(shape)


def Resize_forward(op, input_value, device=None):
    """NXP Platform has a custimized resize operation implementation, which
    gives different result from torch.nn.Resize. To correctly simulate hardware
    beviour and have the same result with NXP, it is necessary to force resize
    to run with nearest mode. Any other mode of resize will be ignored by this
    function.

    Args:
        op ([type]): [description]
        input_value ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    input_data = input_value[0]
    scales = input_value[2] if len(input_value) > 2 else None
    sizes = input_value[-1].tolist() if len(input_value) == 4 else None
    mode = 'nearest'
    if sizes is None or len(sizes) == 0:
        sizes = None
        if scales.numel() == 1:
            scales = scales.item()
        else:
            assert scales.numel() % 2 == 0
            scales = scales[-2].cpu().numpy().tolist()
    else:
        scales = None
        assert sizes[:2] == list(input_data.shape[:2])
        sizes = sizes[2:]
    trans_mode = op.attributes.get('coordinate_transformation_mode', 'half_pixel')
    if trans_mode == 'align_corners':
        output = F.interpolate(input_data, sizes, scales, mode, align_corners=True)
    else:
        output = F.interpolate(input_data, sizes, scales, mode)
    return output


def RoiAlign_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    from torchvision.ops import roi_align as torch_roi_align
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    value = values[0]
    rois = values[1]
    rois = rois
    output_height = op.attributes.get('output_height', 1)
    output_width = op.attributes.get('output_width', 1)
    sampling_ratio = op.attributes.get('sampling_ratio', 0)
    spatial_scale = op.attributes.get('spatial_scale', 1.0)
    if isinstance(rois, torch.Tensor):
        if rois.shape[1] == 5:
            boxes = rois
        elif rois.shape[1] == 4:
            boxes = [rois]
        else:
            raise ValueError(f'Unsupported rois shape {rois.shape}')
    else:
        raise TypeError('Unsupported rois type')
    output_size = output_height, output_width
    output = torch_roi_align(value, boxes, output_size, spatial_scale, sampling_ratio)
    return output


def Scale_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    assert len(values) >= 2
    input_data = values[0]
    scale = values[1]
    bias_term = op.attributes.get('bias_term', False)
    axis = op.attributes.get('axis', 1)
    scale_shape = list(scale.shape)
    for i in range(axis):
        scale_shape.insert(0, 1)
    for i in range(input_data.dim() - scale.dim() - axis):
        scale_shape.append(1)
    scale = scale.reshape(scale_shape)
    scale = scale.expand_as(input_data)
    if bias_term:
        bias = values[2]
        bias = bias.reshape(scale_shape).expand_as(input_data)
        output = input_data * scale + bias
    else:
        output = input_data * scale
    return output


def ScatterElements_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    """
    ScatterElements takes three inputs data, updates, 
    and indices of the same rank r >= 1 and an optional attribute axis that identifies an axis of data 
    (by default, the outer-most axis, that is axis 0). 
    
    The output of the operation is produced by creating a copy of the input data, 
    and then updating its value to values specified by updates at specific index positions specified by indices. 
    Its output shape is the same as the shape of data.

    For each entry in updates, 
    the target index in data is obtained by combining the corresponding entry in indices with the index of the entry itself: 
        the index-value for dimension = axis is obtained from the value of the corresponding entry in indices and the index-value 
    for dimension != axis is obtained from the index of the entry itself.

    reduction allows specification of an optional reduction operation, 
        which is applied to all values in updates tensor into output at the specified indices.
    In cases where reduction is set to "none", indices should not have duplicate entries: that is, 
        if idx1 != idx2, then indices[idx1] != indices[idx2]. 
    
    For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry is performed as below:

    output[indices[i][j]][j] = updates[i][j] if axis = 0,
    output[i][indices[i][j]] = updates[i][j] if axis = 1,
    When reduction is set to "add", the update corresponding to the [i][j] entry is performed as below:

    output[indices[i][j]][j] += updates[i][j] if axis = 0,
    output[i][indices[i][j]] += updates[i][j] if axis = 1,
    When reduction is set to "mul", the update corresponding to the [i][j] entry is performed as below:

    output[indices[i][j]][j] *= updates[i][j] if axis = 0,
    output[i][indices[i][j]] *= updates[i][j] if axis = 1,
    This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

    Attributes
        axis : int (default is 0)
            Which axis to scatter on. Negative value means counting dimensions from the back.
                Accepted range is [-r, r-1] where r = rank(data).
            reduction : string (default is none)
            Type of reduction to apply: none (default), add, mul. 

            'none': no reduction applied. 
            'add': reduction using the addition operation. 
            'mul': reduction using the multiplication operation.

    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, of r >= 1 (same rank as input). 
            All index values are expected to be within bounds [-s, s-1] along axis of size s. 
            It is an error if any of the index values are out of bounds.

        updates (differentiable) : T
            Tensor of rank r >=1 (same rank and shape as indices)

    Outputs
        output (differentiable) : T
            Tensor of rank r >= 1 (same rank as input).
    """
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    value, indices, updates = values
    dim = op.attributes.get('axis', 0)
    indices[indices < 0] += value.shape[dim]
    output = value.scatter(dim, indices, updates)
    return output


def ScatterND_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """OPSET 11: ScatterND takes three inputs data tensor of rank r >= 1,
    indices tensor of rank q >= 1,

        and updates tensor of rank q + r - indices.shape[-1] - 1.

    The output of the operation is produced by creating a copy of the input data,
    and then updating its value to values specified by updates at specific index positions specified by indices.
    Its output shape is the same as the shape of data. Note that indices should not have duplicate entries.
    That is, two or more updates for the same index-location is not supported.

    indices is an integer tensor. Let k denote indices.shape[-1], the last dimension in the shape of indices.
    indices is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into data.
    Hence, k can be a value at most the rank of data. When k equals rank(data),
    each update entry specifies an update to a single element of the tensor.
    When k is less than rank(data) each update entry specifies an update to a slice of the tensor.

    updates is treated as a (q-1)-dimensional tensor of replacement-slice-values.
    Thus, the first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
    The remaining dimensions of updates correspond to the dimensions of the replacement-slice-values.
    Each replacement-slice-value is a (r-k) dimensional tensor, corresponding to the trailing (r-k) dimensions of data.
    Thus, the shape of updates must equal indices.shape[0:q-1] ++ data.shape[k:r-1],
    where ++ denotes the concatenation of shapes.

    Inputs

        data : T
            Tensor of rank r >= 1.

        indices : tensor(int64)
            Tensor of rank q >= 1.

        updates : T
            Tensor of rank q + r - indices_shape[-1] - 1.

    Outputs

        output : T
            Tensor of rank r >= 1.

    Args:
        op ([type]): [description]
        values ([type]): [description]

    Returns:
        [type]: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    value, indices, updates = values
    output = value.clone()
    ind_dim = indices.dim()
    indices = indices.reshape((-1, indices.shape[-1])).T.tolist()
    updates = updates.reshape((-1, *updates.shape[ind_dim - 1:]))
    output[indices] = updates
    return output


def Shape_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Takes a tensor as input and outputs an 1D int64 tensor containing the
    shape of the input tensor.

    Version
        This version of the operator has been available since version 1 of the default ONNX operator set.

    Inputs
        data : T
        An input tensor.

    Outputs
        shape : T1
        Shape of the input tensor

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [value] = values
    shape_tensor = torch.Tensor([k for k in value.shape]).long()
    return shape_tensor


def Sin_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Calculates the sine of the given input tensor, element-wise.

    Inputs
        input (differentiable) : T
            Input tensor
    
    Outputs
        output (differentiable) : T
            The sine of the input tensor computed element-wise
    
    Type Constraints
    T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.

    Args:
        op (Operation): _description_
        values (List[torch.Tensor]): _description_
        ctx (TorchBackendContext, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return torch.sin(x)


def Slice_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Produces a slice of the input tensor along multiple axes. Similar to
    numpy: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    Slices uses starts, ends, axes and steps inputs to specify the start and
    end dimension and step for each axis in the list of axes,

    it uses this information to slice the input data tensor.
    If a negative value is passed for any of the start or end indices,
        it represents number of elements before the end of that dimension.

    If the value passed to start or end is larger than the n (the number of elements in this dimension),
        it represents n. For slicing to the end of a dimension with unknown size,
        it is recommended to pass in INT_MAX when sclicing forward and 'INT_MIN' when slicing backward.

    If a negative value is passed for step, it represents slicing backward.
    However step value cannot be 0. If axes are omitted, they are set to [0, ..., ndim-1].

    If steps are omitted, they are set to [1, ..., 1] of length len(starts)

    Example 1: data = [ [1, 2, 3, 4], [5, 6, 7, 8], ] axes = [0, 1] starts = [1, 0] ends = [2, 3] steps = [1, 2] result = [ [5, 7], ]
    Example 2: data = [ [1, 2, 3, 4], [5, 6, 7, 8], ] starts = [0, 1] ends = [-1, 1000] result = [ [2, 3, 4], ]

    Inputs (3 - 5)
        data : T
            Tensor of data to extract slices from.

        starts : Tind
            1-D tensor of starting indices of corresponding axis in `axes`

        ends : Tind
            1-D tensor of ending indices (exclusive) of corresponding axis in `axes`

        axes (optional) : Tind
            1-D tensor of axes that `starts` and `ends` apply to.
            Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).

        steps (optional) : Tind
            1-D tensor of slice step of corresponding axis in `axes`.
            Negative value means slicing backward. 'steps' cannot be 0. Defaults to 1.

    Outputs
        output : T
            Sliced data tensor.

    Args:
        op ([type]): [description]
        input_value ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=5)
    data, starts, ends = values[:3]
    axes = values[3] if len(values) > 3 else torch.tensor([int(_) for idx, _ in enumerate(starts.tolist())])
    steps = values[4] if len(values) > 4 else torch.ones_like(starts)
    if axes is not None:
        axes = axes.tolist()
    starts, ends, steps = starts.tolist(), ends.tolist(), steps.tolist()
    slices, flip_dims = {}, []
    for start, end, axis, step in zip(starts, ends, axes, steps):
        if step < 0:
            flip_dims.append(axis)
            start, end, step = -start - 1, -end - 1, -step
        slices[axis] = slice(int(start), int(end), int(step))
    pos_axes_slices = list(slices.get(a, slice(None, None)) for a in range(max(axes) + 1))
    neg_axes_slices = list(slices.get(a, slice(None, None)) for a in range(min(axes), 0))
    if neg_axes_slices:
        neg_axes_slices = [Ellipsis] + neg_axes_slices
    if flip_dims:
        data = torch.flip(data, dims=flip_dims)
    if pos_axes_slices:
        data = data[pos_axes_slices]
    if neg_axes_slices:
        data = data[neg_axes_slices]
    return data


def Softplus_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    input_data = values[0]
    output = torch.log(torch.exp(input_data) + 1)
    return output


def SpaceToDepth_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_data = values[0]
    downsample = op.attributes.get('blocksize', 1)
    output = F.pixel_unshuffle(input_data, downsample)
    return output


def Split_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    if op.opset.onnx_opset_version() >= 13:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=2)
        values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
        value = values[0]
        axis = op.attributes.get('axis', 0)
        split = value.shape[axis] // len(op.outputs)
        if len(values) > 1:
            split = values[1]
            split = split.tolist()
    else:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
        axis = op.attributes.get('axis', 0)
        split = op.attributes.get('split', 0)
        [value] = values
        if 'split' not in op.attributes:
            split = value.shape[axis] // len(op.outputs)
    outputs = torch.split(value, split, axis)
    return outputs


def Sqrt_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    input_data = values[0]
    output = torch.sqrt(input_data)
    return output


def Squeeze_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Remove single-dimensional entries from the shape of a tensor. Takes an
    input axes with a list of axes to squeeze. If axes is not provided, all the
    single dimensions will be removed from the shape. If an axis is selected
    with shape entry not equal to one, an error is raised.

    Inputs (1 - 2)
        data (differentiable) : T
        Tensors with at least max(dims) dimensions.

        axes (optional, non-differentiable) : tensor(int64)
        List of integers indicating the dimensions to squeeze.
        Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).

    Outputs
        squeezed (differentiable) : T
        Reshaped tensor with same data as input.

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    if op.opset.onnx_opset_version() >= 13:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=2)
        squeezing_tensor = values[0]
        axes = [axis for axis in range(squeezing_tensor.ndim) if squeezing_tensor.shape[axis] == 1]
        if len(values) > 1:
            axes = values[1]
            axes = axes.tolist()
    else:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
        [squeezing_tensor], axes = values, GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axes', compulsive=False, default=None)
    if axes is None:
        axes = []
        shape = squeezing_tensor.shape
        for dim, s in enumerate(shape):
            if s == 1:
                axes.append(dim)
    if isinstance(axes, list):
        for squeezing_dim in sorted(axes, reverse=True):
            squeezing_tensor = torch.squeeze(squeezing_tensor, squeezing_dim)
    elif isinstance(axes, int):
        squeezing_tensor = torch.squeeze(squeezing_tensor, axes)
    else:
        raise TypeError(f'Parameter axes of operation {op.name} misunderstood, expect int value of list of int, while {type(axes)} was given.')
    return squeezing_tensor


def Sum_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """
    Element-wise sum of each of the input tensors (with Numpy-style broadcasting support). 
    All inputs and outputs must have the same data type. 
    This operator supports multidirectional (i.e., Numpy-style) broadcasting; 
    for more details please check the doc.

    Version
    This version of the operator has been available since version 13 of the default ONNX operator set.

    Other versions of this operator: 1, 6, 8

    Inputs (1 - ∞)
        data_0 (variadic, differentiable) : T
            List of tensors for sum.
    Outputs
        sum (differentiable) : Tq
            Output tensor.
    
    Type Constraints
        T : tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
    Constrain input and output types to float tensors.

    Args:
        op (Operation): _description_
        values (List[torch.Tensor]): _description_
        ctx (TorchBackendContext, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    if op.platform != TargetPlatform.SOI:
        target_device = ctx.executing_device
    else:
        target_device = 'cpu'
    output = torch.zeros_like(values[0])
    for value in values:
        output += value
    return output


def Tan_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    input_data = values[0]
    output = torch.tan(input_data)
    return output


def Tanh_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    input_data = values[0]
    output = torch.tanh(input_data)
    return output


def Tile_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Constructs a tensor by tiling a given tensor. This is the same as
    function tile in Numpy, but no broadcast.

    For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]

    Version
        This version of the operator has been available since version 6 of the default ONNX operator set.

    Inputs
        input : T
            Input tensor of any shape.

        repeats : T1
            1D int64 tensor of the same length as input's dimension number,
            includes numbers of repeated copies along input's dimensions.

    Outputs
        output : T
            Output tensor of the same dimension and type as tensor input.
            output_dim[i] = input_dim[i] * repeats[i]

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input, repeats = values
    axis = op.attributes.get('axis', None)
    tiles = op.attributes.get('tiles', None)
    if axis is not None:
        repeats = [(1) for _ in range(input.ndim)]
        repeats[axis] = tiles
    else:
        repeats = convert_any_to_python_primary_type(values[-1])
        if not isinstance(repeats, list):
            repeats = [repeats]
    assert input.ndim == len(repeats)
    output = input.repeat(tuple(repeats))
    return output


def TopK_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Retrieve the top-K largest or smallest elements along a specified axis.
    Given an input tensor of shape [a_1, a_2, ..., a_n, r] and integer argument
    k, return two outputs: -Value tensor of shape [a_1, a_2, ..., a_{axis-1},
    k, a_{axis+1}, ... a_n] which contains the values of the top k elements.

    along the specified axis - Index tensor of shape [a_1, a_2, ...,
    a_{axis-1}, k, a_{axis+1}, ... a_n] which contains the indices of the top k
    elements (original indices from the input tensor).

    If "largest" is 1 (the default value) then the k largest elements are returned.
    If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
    If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.

    Given two equivalent values, this operator uses the indices along the axis as a tiebreaker.
    That is, the element with the lower index will appear first.

    Attributes
        axis : int (default is -1)
            Dimension on which to do the sort. Negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(input).

        largest : int (default is 1)
            Whether to return the top-K largest or smallest elements.

        sorted : int (default is 1)
            Whether to return the elements in sorted order.

    Inputs
        X (differentiable) : T
            Tensor of shape [a_1, a_2, ..., a_n, r]

        K (non-differentiable) : tensor(int64)
            A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve

    Outputs
        Values (differentiable) : T
            Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
            containing top K values from the input tensor

        Indices (non-differentiable) : I
            Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
            containing the corresponding input tensor indices for the top K values.

    Args:
        op (Operation): [description]
        input_value ([type]): [description]

    Returns:
        [type]: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', default=-1)
    largest = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='largest', default=1)
    sorted = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='sorted', default=1)
    largest, sorted = bool(largest), bool(sorted)
    x, k = values
    k = convert_any_to_python_primary_type(k)
    values, indices = torch.topk(input=x, k=k, dim=axis, largest=largest, sorted=sorted)
    return values, indices


def Transpose_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Transpose the input tensor similar to numpy.transpose. For example, when
    perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
    will be (2, 1, 3).

    Attributes
        perm : list of ints
            A list of integers. By default, reverse the dimensions,
            otherwise permute the axes according to the values given.

    Inputs
        data (differentiable) : T
            An input tensor.

    Outputs
        transposed (differentiable) : T
            Transposed output.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    perm = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='perm', compulsive=True)
    [data] = values
    output = data.permute(perm)
    return output


def UnaryEltwise_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    [input_value] = values
    if op.type == 'Exp':
        output = torch.exp(input_value)
    elif op.type == 'Sigmoid':
        output = torch.sigmoid(input_value)
    elif op.type == 'Relu':
        output = F.relu(input_value)
    else:
        logger.warning('Not UnaryEltwise op, return input as output')
        output = input_value
    return output


def Unsqueeze_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Insert single-dimensional entries to the shape of an input tensor
    (data).

    Takes one required argument axes - which contains a list of dimension indices and
        this operator will insert a dimension of value 1 into the corresponding
        index of the output tensor (expanded).

    For example: Given an input tensor (data) of shape [3, 4, 5],
        then Unsqueeze(data, axes=[0, 4]) outputs a tensor (expanded)
        containing same data as data but with shape [1, 3, 4, 5, 1].

    The attribute axes should not contain any duplicate entries.
    It is an error if it contains duplicates.

    The rank of the output tensor (output_rank) is the rank of the input tensor (data)
        plus the number of values in axes.

    Each value in axes should be within the (inclusive) range [-output_rank , output_rank - 1].
    The order of values in axes does not matter and can come in any order.

    Attributes
        axes : list of ints (required)
            List of integers indicating the dimensions to be inserted.
            Negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(expanded).

    Inputs
        data : T
            Original tensor

    Outputs
        expanded : T
            Reshaped tensor with same data as input.

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    if op.opset.onnx_opset_version() >= 13:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
        unsqueezing_tensor, axes = values
        axes = axes.tolist()
    else:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
        [unsqueezing_tensor] = values
        axes = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axes', compulsive=True)
    if isinstance(axes, list):
        for squeezing_dim in sorted(axes):
            unsqueezing_tensor = torch.unsqueeze(unsqueezing_tensor, squeezing_dim)
    elif isinstance(axes, int):
        unsqueezing_tensor = torch.unsqueeze(unsqueezing_tensor, axes)
    else:
        raise TypeError(f'Parameter axes of operation {op.name} misunderstood, expect int value of list of int, while {type(axes)} was given.')
    return unsqueezing_tensor


def Where_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    condition, x, y = values
    output = torch.where(condition, x, y)
    return output


def _NMS_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs) ->torch.Tensor:
    """Filter out boxes that have high intersection-over-union (IOU) overlap
    with previously selected boxes. Bounding boxes with score less than
    score_threshold are removed. Bounding box format is indicated by attribute
    center_point_box.

    Note that this algorithm is agnostic to where the origin is in the coordinate system and
    more generally is invariant to orthogonal transformations and translations of the coordinate system;

    thus translating or reflections of the coordinate system result in the same boxes being selected by the algorithm.
    The selected_indices output is a set of integers indexing into the input collection of
        bounding boxes representing the selected boxes.
    The bounding box coordinates corresponding to the selected indices
        can then be obtained using the Gather or GatherND operation.

    Attributes
        center_point_box : int (default is 0)
        Integer indicate the format of the box data.
        The default is 0. 0 - the box data is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2)
            are the coordinates of any diagonal pair of box corners and the coordinates can be provided as normalized
            (i.e., lying in the interval [0, 1]) or absolute.
            Mostly used for TF models. 1 - the box data is supplied as
                [x_center, y_center, width, height]. Mostly used for Pytorch models.

    Inputs (2 - 5)
        boxes : tensor(float)
        An input tensor with shape [num_batches, spatial_dimension, 4].
        The single box data format is indicated by center_point_box.

        scores : tensor(float)
        An input tensor with shape [num_batches, num_classes, spatial_dimension]

        max_output_boxes_per_class (optional) : tensor(int64)
        Integer representing the maximum number of boxes to be selected per batch per class.
        It is a scalar. Default to 0, which means no output.

        iou_threshold (optional) : tensor(float)
        Float representing the threshold for deciding whether boxes overlap too much with respect to IOU.
            It is scalar. Value range [0, 1]. Default to 0.

        score_threshold (optional) : tensor(float)
        Float representing the threshold for deciding when to remove boxes based on score. It is a scalar.

    Outputs
        selected_indices : tensor(int64)
        selected indices from the boxes tensor. [num_selected_indices, 3],
            the selected index format is [batch_index, class_index, box_index].

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
    """
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    boxes, scores = values[:2]
    max_output_boxes_per_class = values[2].item() if len(values) > 2 else 0
    iou_threshold = values[3].item() if len(values) > 3 else 0
    score_threshold = values[4].item() if len(values) > 4 else 0
    center_point_box = op.attributes.get('center_point_box', 0)
    batch, num_classes = boxes.shape[0], scores.shape[1]
    output = []
    for i in range(batch):
        sub_boxes = boxes[i]
        sub_scores = scores[i]
        if center_point_box:
            sub_boxes = torch.stack((sub_boxes[:, 0] - sub_boxes[:, 2] / 2, sub_boxes[:, 1] - sub_boxes[:, 3] / 2, sub_boxes[:, 0] + sub_boxes[:, 2] / 2, sub_boxes[:, 1] + sub_boxes[:, 3] / 2), dim=1)
        for j in range(num_classes):
            """# Strange speed, Revert back to original implementation.

            # May lead to error if boxes are not in the order given above

            sorted_boxes = torch.tensor([[min(item[0], item[2]), min(item[1], item[3]), max(item[0], item[2]),
                                          max(item[1], item[3])] for item in sub_boxes], device=device)
            keep = mmcv.ops.nms(sorted_boxes, sub_scores[j].contiguous(), iou_threshold)[1]
            """
            keep = mmcv.ops.nms(sub_boxes, sub_scores[j].contiguous(), iou_threshold)[1]
            keep = keep[sub_scores[j][keep] > score_threshold]
            keep = keep[:max_output_boxes_per_class]
            keep = torch.stack((torch.full_like(keep, i), torch.full_like(keep, j), keep), dim=1)
            output.append(keep)
    output = torch.cat(output)
    return output


def skipLayerNormPlugin_forward(op: 'Operation', values: 'List[torch.Tensor]', ctx: 'TorchBackendContext'=None, **kwargs):
    """
    User should check this: http://nvidia.zhidx.com/content-6-1543-1.html
    and this: https://github.com/microsoft/onnxruntime/blob/rel-1.13.1/docs/ContribOperators.md

    Skip Layernorm Plugin is one of Onnx Contrib Operators.
    It is supported by TensorRT, Openvino and Onnxruntime.

    com.microsoft.SkipLayerNormalization
        Skip and Layer Normalization Fusion

    Version
        This version of the operator has been available since version 1 of the 'com.microsoft' operator set.

    Attributes
        epsilon : float
            The epsilon value to use to avoid division by zero.

    Inputs (3 - 5)
        input : T
            3D input tensor with shape (batch_size, sequence_length, hidden_size)

        skip : T
            3D skip tensor with shape (batch_size, sequence_length, hidden_size)

        gamma : T
            1D input tensor with shape (hidden_size)

        beta (optional) : T
            1D skip tensor with shape (hidden_size)

        bias (optional) : T
            1D bias tensor with shape (hidden_size)

    Outputs (1 - 3)
        output : T
            3D output tensor with shape (batch_size, sequence_length, hidden_size)

        mean (optional) : U
            Saved mean used during training to speed up gradient computation

        inv_std_var (optional) : U
            Saved inverse standard variance used during training to speed up gradient computation.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=4)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    x, skip, gamma = values[0], values[1], values[2]
    None
    if len(values) >= 4:
        bias = values[3]
    else:
        bias = None
    eps = op.attributes.get('epsilon', 1e-05)
    axis = op.attributes.get('axis', -1)
    if axis != -1 and axis != x.ndim - 1:
        raise ValueError('Unsupported Layernorm axis. We will implement it soon.')
    normalized_shape = gamma.shape
    output = F.layer_norm(x + skip, normalized_shape, gamma, bias, eps)
    return output


DEFAULT_BACKEND_TABLE = {'Abs': Abs_forward, 'AdaptiveAvgPool2d': AdaptiveAvgPool2d_forward, 'And': And_forward, 'Add': Add_forward, 'ArgMax': ArgMax_forward, 'Attention': Attention_forward, 'AveragePool': AveragePool_forward, 'BatchNormalization': BatchNormalization_forward, 'Cast': Cast_forward, 'Clip': Clip_forward, 'Concat': Concat_forward, 'Constant': Constant_forward, 'ConstantOfShape': ConstantOfShape_forward, 'Conv': Conv_forward, 'ConvTranspose': ConvTranspose_forward, 'Cos': Cos_forward, 'Div': Eltwise_forward, 'Equal': Equal_forward, 'Exp': UnaryEltwise_forward, 'Expand': Expand_forward, 'Flatten': Flatten_forward, 'Gather': Gather_forward, 'GatherElements': Gather_forward, 'GatherND': GatherND_forward, 'Gelu': Gelu_forward, 'Gemm': Gemm_forward, 'grid_sampler': GridSampler_forward, 'GlobalAveragePool': AveragePool_forward, 'GlobalMaxPool': MaxPool2d_forward, 'Greater': Greater_forward, 'LayerNorm': LayerNorm_forward, 'LayerNormalization': LayerNorm_forward, 'LeakyRelu': LeakyRelu_forward, 'Less': Less_forward, 'LogSoftmax': LogSoftmax_forward, 'MatMul': MatMul_forward, 'Max': Eltwise_forward, 'MaxPool': MaxPool2d_forward, 'Min': Eltwise_forward, 'Mul': Mul_forward, 'MultiHeadAttention': MultiHeadAttention_forward, 'NonMaxSuppression': _NMS_forward, 'NonZero': NonZero_forward, 'Not': Not_forward, 'Pad': Pad_forward, 'PPQBiasFusedMatMul': PPQBiasFusedMatMul_forward, 'PRelu': PRelu_forward, 'Range': Range_forward, 'ReduceL2': ReduceL2_forward, 'ReduceMax': ReduceMax_forward, 'ReduceMean': ReduceMean_forward, 'ReduceSum': ReduceSum_forward, 'Relu': UnaryEltwise_forward, 'Reshape': Reshape_forward, 'Resize': Resize_forward, 'ScatterElements': ScatterElements_forward, 'ScatterND': ScatterND_forward, 'Shape': Shape_forward, 'Sigmoid': UnaryEltwise_forward, 'Sin': Sin_forward, 'Slice': Slice_forward, 'skipLayerNormPlugin': skipLayerNormPlugin_forward, 'Softmax': Softmax_forward, 'Softplus': Softplus_forward, 'Split': Split_forward, 'Squeeze': Squeeze_forward, 'Sub': Eltwise_forward, 'Tile': Tile_forward, 'TopK': TopK_forward, 'Transpose': Transpose_forward, 'Unsqueeze': Unsqueeze_forward, 'Where': Where_forward, 'Sqrt': Sqrt_forward, 'Log': Log_forward, 'Floor': Floor_forward, 'RoiAlign': RoiAlign_forward, 'MMCVRoiAlign': MMCVRoiAlign_forward, 'SpaceToDepth': SpaceToDepth_forward, 'DepthToSpace': DepthToSpace_forward, 'Scale': Scale_forward, 'Tanh': Tanh_forward, 'Tan': Tan_forward, 'Pow': Pow_forward, 'ChannelShuffle': ChannelShuffle_forward, 'InstanceNormalization': InstanceNormalization_forward, 'Parameter': Parameter_forward, 'Interp': Interp_forward, 'CaffeArgMax': CaffeArgMax_forward, 'HardSigmoid': HardSigmoid_forward, 'HardSwish': HardSwish_forward, 'Neg': Neg_forward, 'GRU': GRU_forward, 'PPQDeviceSwitch': PPQDeviceSwitch_forward, 'Identity': Identity_forward, 'OneHot': Onehot_forward, 'Reciprocal': Reciprocal_forward, 'LSTM': LSTM_forward, 'Sum': Sum_forward, 'Elu': Elu_forward, 'Erf': Erf_forward}


OPERATION_FORWARD_TABLE = {platform: DEFAULT_BACKEND_TABLE.copy() for platform in TargetPlatform}


class VLink:

    def __init__(self, in_idx: 'int', out_idx: 'int') ->None:
        if not isinstance(in_idx, int):
            raise TypeError(f'Can not create vlink with input_idx {in_idx}, only int value is acceptable here.')
        if not isinstance(out_idx, int):
            raise TypeError(f'Can not create vlink with output_idx {out_idx}, only int value is acceptable here.')
        self.in_idx = in_idx
        self.out_idx = out_idx


class OpSocket:

    def __init__(self, op: 'OperationBase', in_plat: 'List[TargetPlatform]'=None, out_plat: 'List[TargetPlatform]'=None, links: 'List[VLink]'=None) ->None:
        self.in_plat = in_plat
        if in_plat is None:
            self.in_plat = [TargetPlatform.UNSPECIFIED for _ in range(op.num_of_input)]
        self.out_plat = out_plat
        if out_plat is None:
            self.out_plat = [TargetPlatform.UNSPECIFIED for _ in range(op.num_of_output)]
        self.links = links
        if self.links is None:
            self.links = []
            for i in range(op.num_of_input):
                for j in range(op.num_of_output):
                    self.links.append(VLink(i, j))


def DEFAULT_SOCKET_CREATOR(op: 'OperationBase') ->OpSocket:
    return OpSocket(op=op)


def CEHCK_TYPE(op: 'OperationBase', t: 'str'):
    pass


ONNX_DOMAIN = 'ai.onnx'


STRICT_OPSET_CHECKING = False


def CHECK_OPSET(min_version_supported: 'int', max_version_supported: 'int', op: 'OperationBase', strict_check: 'bool'=STRICT_OPSET_CHECKING):
    if not strict_check:
        return
    opset = op.opset
    if opset.domain != ONNX_DOMAIN:
        raise TypeError(f'Unrecognizable opset was found, can not generate dispatching scheme with op {op.name}, cause it might not be a standard onnx operation.')
    if opset.version > max_version_supported or opset.version < min_version_supported:
        raise TypeError(f'opset version is not supported, can not generate dispatching scheme with op {op.name}({op.type}), currently we support only [{min_version_supported, max_version_supported}], however {opset.version} was given.')


def Clip_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 1 - 13:
        
    Inputs (1 - 3)
        input (differentiable) : T
            Input tensor whose elements to be clipped
    
        min (optional, non-differentiable) : T
            Minimum value, under which element is replaced by min. 
            It must be a scalar(tensor of empty shape).
    
        max (optional, non-differentiable) : T
            Maximum value, above which element is replaced by max. 
            It must be a scalar(tensor of empty shape).
    
    Outputs
        output (differentiable) : T
            Output tensor with clipped input elements
    """
    CEHCK_TYPE(op=op, t='Clip')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def ConstantOfShape_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 9 - 9:
        
    Inputs
        input : T1
            1D tensor. The shape of the expected output tensor. 
            If empty tensor is given, the output would be a scalar. All values must be >= 0.
    
    Outputs
        output : T2
            Output tensor of shape specified by 'input'.If attribute 'value' is specified, 
            the value and datatype of the output tensor is taken from 'value'.If attribute 'value' is not specified, 
            the value in the output defaults to 0, and the datatype defaults to float32.
    """
    CEHCK_TYPE(op=op, t='ConstantOfShape')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[])


def Expand_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 8 - 13:
        
    Inputs
        input (differentiable) : T
    
    Input tensor
        shape (non-differentiable) : tensor(int64)
            A 1-D tensor indicates the shape you want to expand to, following the broadcast rule
    
    Outputs
        output (differentiable) : T
            Output tensor
    """
    CEHCK_TYPE(op=op, t='Expand')
    CHECK_OPSET(op=op, min_version_supported=8, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def GatherElements_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 11 - 13:
        
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, with the same rank r as the input. 
            All index values are expected to be within bounds [-s, s-1] along axis of size s. 
            It is an error if any of the index values are out of bounds.

    Outputs
        output (differentiable) : T
            Tensor of the same shape as indices.
    """
    CEHCK_TYPE(op=op, t='GatherElements')
    CHECK_OPSET(op=op, min_version_supported=11, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def GatherND_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 11 - 13:
        
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : tensor(int64)
            Tensor of rank q >= 1. All index values are expected to be within bounds [-s, s-1] along axis of size s.
            It is an error if any of the index values are out of bounds.

    Outputs
        output (differentiable) : T
            Tensor of rank q + r - indices_shape[-1] - 1.
    """
    CEHCK_TYPE(op=op, t='GatherND')
    CHECK_OPSET(op=op, min_version_supported=11, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Gather_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 1 - 13:
    
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.
        
        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, of any rank q. 
            All index values are expected to be within bounds [-s, s-1] along axis of size s. 
            It is an error if any of the index values are out of bounds.
    
    Outputs
        output (differentiable) : T
            Tensor of rank q + (r - 1).
    """
    CEHCK_TYPE(op=op, t='Gather')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    return OpSocket(op=op, in_plat=[TargetPlatform.UNSPECIFIED, TargetPlatform.FP32], links=[VLink(in_idx=0, out_idx=0)])


def GridSampler_Socket(op: 'OperationBase') ->OpSocket:
    """
    From MMCV
    """
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Logical_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 1 - 16:

    Inputs
        A (non-differentiable) : T
            First input operand for the logical operator.
    
        B (non-differentiable) : T
            Second input operand for the logical operator.
    
    Outputs
        C (non-differentiable) : T1
            Result tensor.
    """
    CHECK_OPSET(op=op, min_version_supported=12, max_version_supported=16)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.UNSPECIFIED]
    out_plat = [TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], out_plat=out_plat, links=[])


def NonMaxSuppression_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 10 - 13:
        
    Inputs (2 - 5)
        boxes : tensor(float)
            An input tensor with shape [num_batches, spatial_dimension, 4]. 
            The single box data format is indicated by center_point_box.
    
        scores : tensor(float)
            An input tensor with shape [num_batches, num_classes, spatial_dimension]
    
        max_output_boxes_per_class (optional) : tensor(int64)
            Integer representing the maximum number of boxes to be selected per batch per class. 
            It is a scalar. Default to 0, which means no output.
    
        iou_threshold (optional) : tensor(float)
            Float representing the threshold for deciding whether boxes overlap too much with respect to IOU. 
            It is scalar. Value range [0, 1]. Default to 0.
    
        score_threshold (optional) : tensor(float)
            Float representing the threshold for deciding when to remove boxes based on score. It is a scalar.

    Outputs
        selected_indices : tensor(int64)
            selected indices from the boxes tensor. [num_selected_indices, 3], 
            the selected index format is [batch_index, class_index, box_index].
    """
    CEHCK_TYPE(op=op, t='NonMaxSuppression')
    CHECK_OPSET(op=op, min_version_supported=10, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[])


def NonZero_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 9 - 13:
        
    Inputs
        X (non-differentiable) : T
            input

    Outputs
        Y (non-differentiable) : tensor(int64)
            output
    """
    CHECK_OPSET(op=op, min_version_supported=9, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED]
    out_plat = [TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], out_plat=out_plat, links=[])


def Onehot_Socket(op: 'OperationBase') ->OpSocket:
    """
    Inputs
        indices (non-differentiable) : T1

        depth (non-differentiable) : T2
            
        values (non-differentiable) : T3
    
    Outputs
        output (non-differentiable) : T3

    Args:
        op (Operation): _description_

    Returns:
        OpSocket: _description_
    """
    CHECK_OPSET(op=op, min_version_supported=12, max_version_supported=16)
    in_plat = [TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    out_plat = [TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], out_plat=out_plat, links=[])


def Pad_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 1 - 13:
    
    Inputs (2 - 3)
        data (differentiable) : T
            Input tensor.
            
        pads (non-differentiable) : tensor(int64)
            Tensor of integers indicating the number of padding elements to add or remove 
            (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. 
            `pads` should be a 1D tensor of shape [2 * input_rank]. 
            `pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...], 
            where xi_begin is the number of pad values added at the beginning of axis `i` and xi_end, 
            the number of pad values added at the end of axis `i`.
            
        constant_value (optional, non-differentiable) : T
            (Optional) A scalar value to be used if the mode chosen is `constant`
            (by default it is 0, empty string or False).
        
    Outputs
        output (differentiable) : T
            Tensor after padding.
    """
    CEHCK_TYPE(op=op, t='Pad')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Range_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 11 - 13:
        
    Inputs
        start : T
            Scalar. First entry for the range of output values.

        limit : T
            Scalar. Exclusive upper limit for the range of output values.

        delta : T
            Scalar. Value to step by.

    Outputs
        output : T
            A 1-D tensor with same type as the inputs containing generated range of values.
    """
    CHECK_OPSET(op=op, min_version_supported=9, max_version_supported=13)
    in_plat = [TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[])


def Reshape_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 5 - 13:
    
    Inputs
        data (differentiable) : T
            An input tensor.
    
        shape (non-differentiable) : tensor(int64)
            Specified shape for output.
    
    Outputs
        reshaped (differentiable) : T
            Reshaped data.
    """
    CEHCK_TYPE(op=op, t='Reshape')
    CHECK_OPSET(op=op, min_version_supported=5, max_version_supported=13)
    return OpSocket(op=op, in_plat=[TargetPlatform.UNSPECIFIED, TargetPlatform.SOI], links=[VLink(in_idx=0, out_idx=0)])


def Resize_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 11 - 13:
    
    Inputs (1 - 4)
        X (differentiable) : T1
            N-D tensor
   
        roi (optional, non-differentiable) : T2
            1-D tensor given as [start1, ..., startN, end1, ..., endN], 
            where N is the rank of X. The RoIs' coordinates are normalized in the coordinate system of the input image. 
            It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"
    
        scales (optional, non-differentiable) : tensor(float)
            The scale array along each dimension. It takes value greater than 0. 
            If it's less than 1, it's sampling down, otherwise, it's upsampling. 
            The number of elements of 'scales' should be the same as the rank of input 'X'.
            One of 'scales' and 'sizes' MUST be specified and it is an error if both are specified.
            If 'sizes' is needed, the user can use an empty string as the name of 'scales' in this operator's input list.
    
        sizes (optional, non-differentiable) : tensor(int64)
            The size of the output tensor. The number of elements of 'sizes' should be the same as the rank of input 'X'.
            Only one of 'scales' and 'sizes' can be specified.
    
    Outputs
        Y (differentiable) : T1
            N-D tensor after resizing
    """
    CEHCK_TYPE(op=op, t='Resize')
    CHECK_OPSET(op=op, min_version_supported=10, max_version_supported=13, strict_check=True)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def RoiAlign_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 10 - 16:
        
    Inputs
        X : T1
            Input data tensor from the previous operator; 4-D feature map of shape (N, C, H, W), 
            where N is the batch size, C is the number of channels, 
            and H and W are the height and the width of the data.

        rois : T1
            RoIs (Regions of Interest) to pool over; rois is 2-D input of shape (num_rois, 4) 
            given as [[x1, y1, x2, y2], ...]. The RoIs' coordinates are in the coordinate system of the input image. 
            Each coordinate set has a 1:1 correspondence with the 'batch_indices' input.

        batch_indices : T2
            1-D tensor of shape (num_rois,) with each element denoting the index of the corresponding image in the batch.

    Outputs
        Y : T1
            RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width). 
            The r-th batch element Y[r-1] is a pooled feature map corresponding to the r-th RoI X[r-1].
    """
    CEHCK_TYPE(op=op, t='RoiAlign')
    CHECK_OPSET(op=op, min_version_supported=10, max_version_supported=16)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def ScatterElements_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 11 - 16:
        
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, of r >= 1 (same rank as input). 
            All index values are expected to be within bounds [-s, s-1] along axis of size s. 
            It is an error if any of the index values are out of bounds.

        updates (differentiable) : T
            Tensor of rank r >=1 (same rank and shape as indices)

    Outputs
        output (differentiable) : T
            Tensor of rank r >= 1 (same rank as input).
    """
    CHECK_OPSET(op=op, min_version_supported=11, max_version_supported=16)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.UNSPECIFIED]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0), VLink(in_idx=2, out_idx=0)])


def ScatterND_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 11 - 16:
        
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.
        
        indices (non-differentiable) : tensor(int64)
            Tensor of rank q >= 1.
        
        updates (differentiable) : T
            Tensor of rank q + r - indices_shape[-1] - 1.

    Outputs
        output (differentiable) : T
            Tensor of rank r >= 1.
    """
    CHECK_OPSET(op=op, min_version_supported=11, max_version_supported=16)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.UNSPECIFIED]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0), VLink(in_idx=2, out_idx=0)])


def Shape_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 1 - 16:
        
    Inputs
        data (non-differentiable) : T
            An input tensor.
    
    Outputs
        shape (non-differentiable) : T1
            Shape of the input tensor
    """
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=16)
    in_plat = [TargetPlatform.UNSPECIFIED]
    out_plat = [TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], out_plat=out_plat, links=[])


def Slice_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 1 - 16:

    Inputs (3 - 5)
        data : T
            Tensor of data to extract slices from.

        starts : Tind
            1-D tensor of starting indices of corresponding axis in `axes`

        ends : Tind
            1-D tensor of ending indices (exclusive) of corresponding axis in `axes`

        axes (optional) : Tind
            1-D tensor of axes that `starts` and `ends` apply to.
            Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).

        steps (optional) : Tind
            1-D tensor of slice step of corresponding axis in `axes`.
            Negative value means slicing backward. 'steps' cannot be 0. Defaults to 1.

    Outputs
        output : T
            Sliced data tensor.
    """
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Split_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 1 - 13:
    
    Inputs (1 - 2)
        input (differentiable) : T
            The tensor to split
    
        split (optional, non-differentiable) : tensor(int64)
            Optional length of each output. Values should be >= 0.Sum of the values 
            must be equal to the dim value at 'axis' specified.
    
    Outputs (1 - ∞)
        outputs (variadic, differentiable) : T
            One or more outputs forming list of tensors after splitting
    """
    CEHCK_TYPE(op=op, t='Split')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Squeeze_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 1 - 13:

    Inputs (1 - 2)
        data (differentiable) : T
            Tensors with at least max(dims) dimensions.

        axes (optional, non-differentiable) : tensor(int64)
            List of integers indicating the dimensions to squeeze. 
            Negative value means counting dimensions from the back. 
            Accepted range is [-r, r-1] where r = rank(data).
    
    Outputs
        squeezed (differentiable) : T
            Reshaped tensor with same data as input.
    """
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Tile_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 1 - 13:
        
    Inputs
        input (differentiable) : T
            Input tensor of any shape.
    
    repeats (non-differentiable) : T1
        1D int64 tensor of the same length as input's dimension number, 
        includes numbers of repeated copies along input's dimensions.

    Outputs
        output (differentiable) : T
            Output tensor of the same dimensions and type as tensor input. 
            output_dim[i] = input_dim[i] * repeats[i]
    """
    CEHCK_TYPE(op=op, t='Tile')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Topk_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 1 - 11:
    
    Inputs
        X (differentiable) : T
            Tensor of shape [a_1, a_2, ..., a_n, r]
    
        K (non-differentiable) : tensor(int64)
            A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve
    
    Outputs
        Values (differentiable) : T
            Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] 
            containing top K values from the input tensor
    
        Indices (non-differentiable) : I
            Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] 
            containing the corresponding input tensor indices for the top K values.
    """
    CEHCK_TYPE(op=op, t='TopK')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=11)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    out_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], out_plat=out_plat, links=[VLink(in_idx=0, out_idx=0)])


def Unsqueeze_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 1 - 13:

    Inputs
        data (differentiable) : T
            Original tensor

        axes (non-differentiable) : tensor(int64)
            List of integers indicating the dimensions to be inserted. 
            Negative value means counting dimensions from the back. 
            Accepted range is [-r, r-1] where r = rank(expanded).

    Outputs
        expanded (differentiable) : T
            Reshaped tensor with same data as input.
    """
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Where_Socket(op: 'OperationBase') ->OpSocket:
    """
    From Opset 9 - 16:

    Inputs
        condition (non-differentiable) : B
            When True (nonzero), yield X, otherwise yield Y
    
        X (differentiable) : T
            values selected at indices where condition is True
        
        Y (differentiable) : T
            values selected at indices where condition is False
    
    Outputs
        output (differentiable) : T
            Tensor of shape equal to the broadcasted shape of condition, X, and Y.
    """
    CHECK_OPSET(op=op, min_version_supported=9, max_version_supported=16)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


DEFAULT_SOCKET_TABLE = {'AdaptiveAvgPool2d': DEFAULT_SOCKET_CREATOR, 'Add': DEFAULT_SOCKET_CREATOR, 'ArgMax': DEFAULT_SOCKET_CREATOR, 'AveragePool': DEFAULT_SOCKET_CREATOR, 'BatchNormalization': DEFAULT_SOCKET_CREATOR, 'Cast': DEFAULT_SOCKET_CREATOR, 'Clip': Clip_Socket, 'Concat': DEFAULT_SOCKET_CREATOR, 'Constant': DEFAULT_SOCKET_CREATOR, 'ConstantOfShape': ConstantOfShape_Socket, 'Conv': DEFAULT_SOCKET_CREATOR, 'ConvTranspose': DEFAULT_SOCKET_CREATOR, 'Div': DEFAULT_SOCKET_CREATOR, 'Equal': Logical_Socket, 'Exp': DEFAULT_SOCKET_CREATOR, 'Expand': Expand_Socket, 'Flatten': DEFAULT_SOCKET_CREATOR, 'Gather': Gather_Socket, 'GatherElements': GatherElements_Socket, 'GatherND': GatherND_Socket, 'Gelu': DEFAULT_SOCKET_CREATOR, 'Gemm': DEFAULT_SOCKET_CREATOR, 'grid_sampler': GridSampler_Socket, 'GlobalAveragePool': DEFAULT_SOCKET_CREATOR, 'GlobalMaxPool': DEFAULT_SOCKET_CREATOR, 'Greater': Logical_Socket, 'LayerNorm': DEFAULT_SOCKET_CREATOR, 'LeakyRelu': DEFAULT_SOCKET_CREATOR, 'Less': Logical_Socket, 'LogSoftmax': DEFAULT_SOCKET_CREATOR, 'MatMul': DEFAULT_SOCKET_CREATOR, 'Max': DEFAULT_SOCKET_CREATOR, 'MaxPool': DEFAULT_SOCKET_CREATOR, 'Min': DEFAULT_SOCKET_CREATOR, 'Mul': DEFAULT_SOCKET_CREATOR, 'MultiHeadAttention': DEFAULT_SOCKET_CREATOR, 'NonMaxSuppression': NonMaxSuppression_Socket, 'NonZero': NonZero_Socket, 'Not': DEFAULT_SOCKET_CREATOR, 'Pad': Pad_Socket, 'PRelu': DEFAULT_SOCKET_CREATOR, 'Range': Range_Socket, 'ReduceL2': DEFAULT_SOCKET_CREATOR, 'ReduceMax': DEFAULT_SOCKET_CREATOR, 'ReduceMean': DEFAULT_SOCKET_CREATOR, 'ReduceSum': DEFAULT_SOCKET_CREATOR, 'Relu': DEFAULT_SOCKET_CREATOR, 'Reshape': Reshape_Socket, 'Resize': Resize_Socket, 'ScatterElements': ScatterElements_Socket, 'ScatterND': ScatterND_Socket, 'Shape': Shape_Socket, 'Sigmoid': DEFAULT_SOCKET_CREATOR, 'Slice': Slice_Socket, 'Softmax': DEFAULT_SOCKET_CREATOR, 'Softplus': DEFAULT_SOCKET_CREATOR, 'Split': Split_Socket, 'Squeeze': Squeeze_Socket, 'Sub': DEFAULT_SOCKET_CREATOR, 'Tile': Tile_Socket, 'TopK': Topk_Socket, 'Transpose': DEFAULT_SOCKET_CREATOR, 'Unsqueeze': Unsqueeze_Socket, 'Where': Where_Socket, 'Sqrt': DEFAULT_SOCKET_CREATOR, 'Log': DEFAULT_SOCKET_CREATOR, 'Floor': DEFAULT_SOCKET_CREATOR, 'RoiAlign': RoiAlign_Socket, 'MMCVRoiAlign': RoiAlign_Socket, 'SpaceToDepth': DEFAULT_SOCKET_CREATOR, 'DepthToSpace': DEFAULT_SOCKET_CREATOR, 'Scale': DEFAULT_SOCKET_CREATOR, 'Tanh': DEFAULT_SOCKET_CREATOR, 'Pow': DEFAULT_SOCKET_CREATOR, 'Crop': DEFAULT_SOCKET_CREATOR, 'ChannelShuffle': DEFAULT_SOCKET_CREATOR, 'InstanceNormalization': DEFAULT_SOCKET_CREATOR, 'Parameter': DEFAULT_SOCKET_CREATOR, 'Interp': DEFAULT_SOCKET_CREATOR, 'CaffeArgMax': DEFAULT_SOCKET_CREATOR, 'HardSigmoid': DEFAULT_SOCKET_CREATOR, 'HardSwish': DEFAULT_SOCKET_CREATOR, 'Neg': DEFAULT_SOCKET_CREATOR, 'GRU': DEFAULT_SOCKET_CREATOR, 'PPQDeviceSwitch': DEFAULT_SOCKET_CREATOR, 'Identity': DEFAULT_SOCKET_CREATOR, 'OneHot': Onehot_Socket, 'Reciprocal': DEFAULT_SOCKET_CREATOR, 'GreaterOrEqual': Logical_Socket, 'LessOrEqual': Logical_Socket, 'Xor': Logical_Socket, 'Or': Logical_Socket, 'And': Logical_Socket, 'Erf': DEFAULT_SOCKET_CREATOR}


LINEAR_ACTIVATIONS = {'Relu', 'Clip'}


COMPUTING_OP = {'Conv', 'Gemm', 'ConvTranspose', 'MatMul', 'Attention', 'PPQBiasFusedMatMul'}


class OperationMeta:

    def __init__(self, input_metas: 'List[TensorMeta]', output_metas: 'List[TensorMeta]', operation_name: 'str', operation_type: 'str', executing_order: 'int') ->None:
        """OperationMeta structure describes all related tensor metadata of an
        operation.

        It naturally is a collection of TensorMeta.
        Take a look at TensorMeta to get more information.
        Args:
            input_metas (List[TensorMeta]):
                A collection contains all input tensors' metadata.
                ATTENTION: All parameters are considered as input in PPQ.
            output_metas (List[TensorMeta]):
                A collection contains all output tensors' metadata.
            operation_name (str): Not yet used.
            operation_type (str): Not yet used.
            executing_order (int): a int value represents the executing order of this operation.
                (order 0 means this operation is the first operation to be executed)
        """
        assert isinstance(input_metas, list), 'can only accept list object here.'
        assert isinstance(output_metas, list), 'can only accept list object here.'
        self.input_metas = input_metas
        self.output_metas = output_metas
        self.operation_name = operation_name
        self.operation_type = operation_type
        self.executing_order = executing_order

    def __str__(self) ->str:
        return 'Inputs: '.join(str([_ for _ in self.input_metas])) + 'Outputs: '.join(str([_ for _ in self.input_metas]))

    @property
    def num_of_input(self):
        return len(self.input_metas)

    @property
    def num_of_output(self):
        return len(self.output_metas)

    def copy(self):
        return OperationMeta(input_metas=[meta.copy() for meta in self.input_metas], output_metas=[meta.copy() for meta in self.output_metas], operation_name=self.operation_name, operation_type=self.operation_type, executing_order=self.executing_order)


CAFFE_DOMAIN = 'ppq.caffe'


DEFAULT_OPSET_DOMAIN = 'ai.onnx'


DEFAULT_OPSET_VERSION = 11


class Opset:

    def __init__(self, domain: 'str'=DEFAULT_OPSET_DOMAIN, version: 'int'=DEFAULT_OPSET_VERSION) ->None:
        """Open Neural Network Exchange (ONNX) is an open ecosystem that empowers AI developers 
            to choose the right tools as their project evolves. 
        
        ONNX provides an open source format for AI models, both deep learning and traditional ML. 
        It defines an extensible computation graph model, as well as definitions of
            built-in operators and standard data types. 
        Currently we focus on the capabilities needed for inferencing (scoring).
        
        PPQ IR is built based on ONNX defination.

        Args:
            domain (str, optional): _description_. Defaults to DEFAULT_OPSET_DOMAIN.
            version (int, optional): _description_. Defaults to DEFAULT_OPSET_VERSION.
        """
        self.domain = domain
        self.version = version

    def is_onnx_v13(self):
        return self.domain == ONNX_DOMAIN and self.version == 13

    def is_onnx_v11(self):
        return self.domain == ONNX_DOMAIN and self.version == 11

    def onnx_opset_version(self) ->int:
        return self.version if self.domain == ONNX_DOMAIN else -1

    def is_onnx(self):
        return self.domain == ONNX_DOMAIN

    def is_caffe(self):
        return self.domain == CAFFE_DOMAIN


class OperationBase(metaclass=ABCMeta):

    def __init__(self, name: 'str', op_type: 'str', attributes: 'Dict[str, Any]', opset=None, platform: 'TargetPlatform'=TargetPlatform.UNSPECIFIED) ->None:
        self._name = name
        self._type = op_type
        self._attributes = attributes
        self._platform = platform
        self._meta = None
        self._input_vars = []
        self._output_vars = []
        self._detail = {}
        if opset is None:
            self._opset = Opset()
        else:
            self._opset = opset

    @abstractproperty
    def inputs(self) ->List[Any]:
        pass

    @abstractproperty
    def outputs(self) ->List[Any]:
        pass

    @abstractproperty
    def parameters(self) ->List[Any]:
        pass

    @property
    def name(self) ->str:
        return self._name

    @property
    def type(self) ->str:
        return self._type

    @type.setter
    def type(self, type: 'str'):
        self._type = type

    @property
    def opset(self) ->Opset:
        return self._opset

    @opset.setter
    def opset(self, opset: 'Opset'):
        self._opset = opset

    @property
    def attributes(self) ->Dict[str, Any]:
        return self._attributes

    @property
    def platform(self) ->TargetPlatform:
        return self._platform

    @platform.setter
    def platform(self, platform: 'TargetPlatform'):
        self._platform = platform

    @property
    def num_of_input(self) ->int:
        return len(self._input_vars)

    @property
    def num_of_output(self) ->int:
        return len(self._output_vars)

    @property
    def meta_data(self) ->OperationMeta:
        raise NotImplementedError('This property has been removed since PPQ 0.6.6')

    @property
    def inputs(self) ->List[object]:
        return self._input_vars

    @property
    def outputs(self) ->List[object]:
        return self._output_vars

    @property
    def is_computing_op(self) ->bool:
        return self.type in COMPUTING_OP

    def set_extension_attrib(self, attrib: 'str', value: 'Any'):
        self._detail[attrib] = value

    @property
    def extension_attrib(self) ->dict:
        return self._detail

    def __hash__(self) ->int:
        return self._name.__hash__()


class PPQ_GLOBAL_CONFIGURATION:

    def __init__(self) ->None:
        self.USING_CUDA_KERNEL = False
        self.NAME = 'PPL Quantization Tool'
        self.VERSION = '0.6.6'
        self.DUMP_VALUE_WHEN_EXPORT = True
        self.EXPORT_PPQ_INTERNAL_INFO = False
        self.PPQ_DEBUG = False


PPQ_CONFIG = PPQ_GLOBAL_CONFIGURATION()

