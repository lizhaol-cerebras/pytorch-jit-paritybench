
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


import time


import torch


from functools import reduce


from typing import Callable


from typing import Optional


from typing import Tuple


import warnings


from warnings import warn


import logging


import re


from typing import List


from typing import Dict


from typing import Iterable


from typing import Iterator


import itertools


from typing import Any


import numpy as np


from torch import Tensor


import copy


from typing import TypeVar


from typing import Union


from typing import overload


from torch import device


from torch import dtype


from torch import nn


import torch.nn.functional as F


from functools import partial


import torch.nn as nn


import math


import torch.distributed as dist


from typing import Literal


from torch.optim import Optimizer


from collections import abc as container_abcs


from collections import defaultdict


from copy import deepcopy


from itertools import chain


from itertools import product


import random


from scipy.stats import norm


import inspect


import uuid


class GlobalOptimManager:
    """
    A global optimizer manager for enabling custom optimizer configs.
    """
    _instance = None

    def __init__(self):
        raise RuntimeError('Call get_instance() instead')

    def initialize(self):
        self.pid2config = {}
        self.index2config = {}
        self.optimizer = None
        self.uses_config_override = False
        self.module_weight_config_triple = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def register_parameters(self, params):
        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for group_index, group in enumerate(param_groups):
            for p_index, p in enumerate(group['params']):
                if id(p) in self.pid2config:
                    self.index2config[group_index, p_index] = self.pid2config[id(p)]

    def override_config(self, parameters, key=None, value=None, key_value_dict=None):
        """
        Override initial optimizer config with specific hyperparameters.

        The key-values of the optimizer config for the input parameters are overridden
        This can be both, optimizer parameters like `betas` or `lr`, or it can be
        8-bit specific parameters like `optim_bits` or `percentile_clipping`.

        Arguments:
           parameters (`torch.Tensor` or `list(torch.Tensors)`):
             The input parameters.
           key (`str`):
             The hyperparamter to override.
           value:
             The hyperparameter values.
           key_value_dict (`dict`):
             A dictionary with multiple key-values to override.

        Example:

        ```py
        import torch
        import bitsandbytes as bnb

        mng = bnb.optim.GlobalOptimManager.get_instance()

        model = MyModel()
        mng.register_parameters(model.parameters()) # 1. register parameters while still on CPU

        model = model.cuda()
        # use 8-bit optimizer states for all parameters
        adam = bnb.optim.Adam(model.parameters(), lr=0.001, optim_bits=8)

        # 2. override: the parameter model.fc1.weight now uses 32-bit Adam
        mng.override_config(model.fc1.weight, 'optim_bits', 32)
        ```
        """
        self.uses_config_override = True
        if isinstance(parameters, torch.nn.Parameter):
            parameters = [parameters]
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        if key is not None and value is not None:
            assert key_value_dict is None
            key_value_dict = {key: value}
        if key_value_dict is not None:
            for p in parameters:
                if id(p) in self.pid2config:
                    self.pid2config[id(p)].update(key_value_dict)
                else:
                    self.pid2config[id(p)] = key_value_dict

    def register_module_override(self, module, param_name, config):
        self.module_weight_config_triple.append((module, param_name, config))


class StableEmbedding(torch.nn.Embedding):
    """
    Custom embedding layer designed to improve stability during training for NLP tasks by using 32-bit optimizer states. It is designed to reduce gradient variations that can result from quantization. This embedding layer is initialized with Xavier uniform initialization followed by layer normalization.

    Example:

    ```
    # Initialize StableEmbedding layer with vocabulary size 1000, embedding dimension 300
    embedding_layer = StableEmbedding(num_embeddings=1000, embedding_dim=300)

    # Reset embedding parameters
    embedding_layer.reset_parameters()

    # Perform a forward pass with input tensor
    input_tensor = torch.tensor([1, 2, 3])
    output_embedding = embedding_layer(input_tensor)
    ```

    Attributes:
        norm (`torch.nn.LayerNorm`): Layer normalization applied after the embedding.

    Methods:
        reset_parameters(): Reset embedding parameters using Xavier uniform initialization.
        forward(input: Tensor) -> Tensor: Forward pass through the stable embedding layer.
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', padding_idx: 'Optional[int]'=None, max_norm: 'Optional[float]'=None, norm_type: 'float'=2.0, scale_grad_by_freq: 'bool'=False, sparse: 'bool'=False, _weight: 'Optional[Tensor]'=None, device=None, dtype=None) ->None:
        """
        Args:
            num_embeddings (`int`):
                The number of unique embeddings (vocabulary size).
            embedding_dim (`int`):
                The dimensionality of the embedding.
            padding_idx (`Optional[int]`):
                Pads the output with zeros at the given index.
            max_norm (`Optional[float]`):
                Renormalizes embeddings to have a maximum L2 norm.
            norm_type (`float`, defaults to `2.0`):
                The p-norm to compute for the `max_norm` option.
            scale_grad_by_freq (`bool`, defaults to `False`):
                Scale gradient by frequency during backpropagation.
            sparse (`bool`, defaults to `False`):
                Computes dense gradients. Set to `True` to compute sparse gradients instead.
            _weight (`Optional[Tensor]`):
                Pretrained embeddings.
        """
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, device, dtype)
        self.norm = torch.nn.LayerNorm(embedding_dim, device=device)
        GlobalOptimManager.get_instance().register_module_override(self, 'weight', {'optim_bits': 32})

    def reset_parameters(self) ->None:
        torch.nn.init.xavier_uniform_(self.weight)
        self._fill_padding_idx_with_zero()
    """ !!! This is a redefinition of _fill_padding_idx_with_zero in torch.nn.Embedding
        to make the Layer compatible with Pytorch < 1.9.
        This means that if this changes in future PyTorch releases this need to change too
        which is cumbersome. However, with this we can ensure compatibility with previous
        PyTorch releases.
    """

    def _fill_padding_idx_with_zero(self) ->None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: 'Tensor') ->Tensor:
        emb = F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        emb = emb
        return self.norm(emb)


class Embedding(torch.nn.Embedding):
    """
    Embedding class to store and retrieve word embeddings from their indices.
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', padding_idx: 'Optional[int]'=None, max_norm: 'Optional[float]'=None, norm_type: 'float'=2.0, scale_grad_by_freq: 'bool'=False, sparse: 'bool'=False, _weight: 'Optional[Tensor]'=None, device: 'Optional[device]'=None) ->None:
        """
        Args:
            num_embeddings (`int`):
                The number of unique embeddings (vocabulary size).
            embedding_dim (`int`):
                The dimensionality of the embedding.
            padding_idx (`Optional[int]`):
                Pads the output with zeros at the given index.
            max_norm (`Optional[float]`):
                Renormalizes embeddings to have a maximum L2 norm.
            norm_type (`float`, defaults to `2.0`):
                The p-norm to compute for the `max_norm` option.
            scale_grad_by_freq (`bool`, defaults to `False`):
                Scale gradient by frequency during backpropagation.
            sparse (`bool`, defaults to `False`):
                Computes dense gradients. Set to `True` to compute sparse gradients instead.
            _weight (`Optional[Tensor]`):
                Pretrained embeddings.
        """
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, device=device)
        GlobalOptimManager.get_instance().register_module_override(self, 'weight', {'optim_bits': 32})

    def reset_parameters(self) ->None:
        torch.nn.init.xavier_uniform_(self.weight)
        self._fill_padding_idx_with_zero()
    """ !!! This is a redefinition of _fill_padding_idx_with_zero in torch.nn.Embedding
        to make the Layer compatible with Pytorch < 1.9.
        This means that if this changes in future PyTorch releases this need to change too
        which is cumbersome. However, with this we can ensure compatibility with previous
        PyTorch releases.
    """

    def _fill_padding_idx_with_zero(self) ->None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: 'Tensor') ->Tensor:
        emb = F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        return emb


def pack_dict_to_tensor(source_dict):
    """
    Pack a dictionary into a torch tensor for storing quant_state items in state_dict.

    Parameters:
    - source_dict: The dictionary to be packed.

    Returns:
    A torch tensor containing the packed data.
    """
    json_str = json.dumps(source_dict)
    json_bytes = json_str.encode('utf-8')
    tensor_data = torch.tensor(list(json_bytes), dtype=torch.uint8)
    return tensor_data


def unpack_tensor_to_dict(tensor_data):
    """
    Unpack a torch tensor into a Python dictionary.

    Parameters:
    - tensor_data: The torch tensor containing the packed data.

    Returns:
    A Python dictionary containing the unpacked data.
    """
    json_bytes = bytes(tensor_data.cpu().numpy())
    json_str = json_bytes.decode('utf-8')
    unpacked_dict = json.loads(json_str)
    return unpacked_dict


class QuantState:
    """container for quantization state components to work with Params4bit and similar classes"""
    valid_quant_types = 'fp4', 'nf4'
    valid_qs_type_keys = [f'bitsandbytes__{x}' for x in valid_quant_types]
    valid_qs_keys = ['absmax', 'quant_map', 'nested_absmax', 'nested_quant_map', 'quant_state', 'quant_type', 'blocksize', 'dtype', 'shape', 'nested_blocksize', 'nested_dtype', 'nested_offset']

    def __init__(self, absmax, shape=None, code=None, blocksize=None, quant_type=None, dtype=None, offset=None, state2=None):
        self.absmax = absmax
        self.shape = shape
        self.code = code
        self.dtype = dtype
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.offset = offset
        self.state2 = state2
        self.nested = state2 is not None

    def __get_item__(self, idx):
        """
        ensures compatibility with older quant state scheme with nested lists.
        assumes the following layout:
        state = [qabsmax, input_shape, A.dtype, blocksize, [offset, state2], quant_type]
        state2 = [absmax, input_shape, A.dtype, blocksize, None, quant_type]
        """
        if self.nested:
            list_repr = [self.absmax, self.shape, self.dtype, self.blocksize, [self.offset, self.state2], self.quant_type]
        else:
            list_repr = [self.absmax, self.shape, self.dtype, self.blocksize, None, self.quant_type]
        return list_repr[idx]

    @classmethod
    def from_dict(cls, qs_dict: 'Dict[str, Any]', device: 'torch.device') ->'QuantState':
        """
        unpacks components of state_dict into QuantState
        where necessary, convert into strings, torch.dtype, ints, etc.

        qs_dict: based on state_dict, with only relevant keys, striped of prefixes.

        item with key `quant_state.bitsandbytes__[nf4/fp4]` may contain minor and non-tensor quant state items.
        """
        qs_key = [k for k, v in qs_dict.items() if 'quant_state' in k and isinstance(v, torch.Tensor)]
        if not len(qs_key) and 'quant_type' not in qs_dict:
            raise ValueError('Expected packed or unpacked quant_state items, found neither')
        elif len(qs_key) != 1 or qs_key[0].split('.')[-1] not in cls.valid_qs_type_keys:
            raise ValueError(f'There should be exactly one `quant_state` item with ending from {cls.valid_qs_type_keys}.\nDetected {qs_key}.')
        if len(qs_key) == 1:
            first_qs_key = qs_key[0]
            qs_dict.update(unpack_tensor_to_dict(qs_dict.pop(first_qs_key)))
        qs_dict = {k.split('.')[-1]: v for k, v in qs_dict.items()}
        assert set(qs_dict.keys()).issubset(cls.valid_qs_keys)
        if 'nested_absmax' in qs_dict:
            offset = torch.tensor(float(qs_dict['nested_offset']))
            state2 = cls(absmax=qs_dict['nested_absmax'], blocksize=qs_dict['nested_blocksize'], code=qs_dict['nested_quant_map'], dtype=getattr(torch, qs_dict['nested_dtype']))
        else:
            offset, state2 = None, None
        quant_state = cls(quant_type=qs_dict['quant_type'], absmax=qs_dict['absmax'], blocksize=qs_dict['blocksize'], code=qs_dict['quant_map'], dtype=getattr(torch, qs_dict['dtype']), shape=torch.Size(qs_dict['shape']) if qs_dict['shape'] is not None else None, offset=offset, state2=state2)
        return quant_state

    def as_dict(self, packed=False):
        """
        returns dict of tensors and strings to use in serialization via _save_to_state_dict()
        param: packed -- returns dict[str, torch.Tensor] for state_dict fit for safetensors saving
        """
        qs_dict = {'quant_type': self.quant_type, 'absmax': self.absmax, 'blocksize': self.blocksize, 'quant_map': self.code, 'dtype': str(self.dtype).strip('torch.'), 'shape': tuple(self.shape)}
        if self.nested:
            qs_dict.update({'nested_absmax': self.state2.absmax, 'nested_blocksize': self.state2.blocksize, 'nested_quant_map': self.state2.code.clone(), 'nested_dtype': str(self.state2.dtype).strip('torch.'), 'nested_offset': self.offset.item()})
        if not packed:
            return qs_dict
        qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
        non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
        qs_packed_dict['quant_state.' + 'bitsandbytes__' + self.quant_type] = pack_dict_to_tensor(non_tensor_dict)
        return qs_packed_dict

    def to(self, device):
        self.absmax = self.absmax
        if self.nested:
            self.offset = self.offset
            self.state2.absmax = self.state2.absmax
            self.state2.code = self.state2.code

    def __eq__(self, other):
        if not isinstance(other, QuantState):
            return False
        return torch.allclose(self.absmax, other.absmax, atol=1e-06) and self.shape == other.shape and torch.allclose(self.code, other.code, atol=1e-06) and self.dtype == other.dtype and self.blocksize == other.blocksize and self.quant_type == other.quant_type and (self.offset == other.offset if self.offset is not None and other.offset is not None else self.offset is other.offset) and (self.state2 == other.state2 if self.state2 is not None and other.state2 is not None else self.state2 is other.state2)


T = TypeVar('T', bound='torch.nn.Module')


class Params4bit(torch.nn.Parameter):

    def __new__(cls, data: 'Optional[torch.Tensor]'=None, requires_grad=False, quant_state: 'Optional[QuantState]'=None, blocksize: 'int'=64, compress_statistics: 'bool'=True, quant_type: 'str'='fp4', quant_storage: 'torch.dtype'=torch.uint8, module: "Optional['Linear4bit']"=None, bnb_quantized: 'bool'=False) ->'Params4bit':
        if data is None:
            data = torch.empty(0)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.quant_storage = quant_storage
        self.bnb_quantized = bnb_quantized
        self.data = data
        self.module = module
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state['data'] = self.data
        state['requires_grad'] = self.requires_grad
        return state

    def __setstate__(self, state):
        self.requires_grad = state['requires_grad']
        self.blocksize = state['blocksize']
        self.compress_statistics = state['compress_statistics']
        self.quant_type = state['quant_type']
        self.quant_state = state['quant_state']
        self.data = state['data']
        self.quant_storage = state['quant_storage']
        self.bnb_quantized = state['bnb_quantized']
        self.module = state['module']

    def __deepcopy__(self, memo):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        new_instance.quant_state = copy.deepcopy(state['quant_state'])
        new_instance.data = copy.deepcopy(state['data'])
        return new_instance

    def __copy__(self):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        return new_instance

    @classmethod
    def from_prequantized(cls, data: 'torch.Tensor', quantized_stats: 'Dict[str, Any]', requires_grad: 'bool'=False, device='cuda', module: "Optional['Linear4bit']"=None, **kwargs) ->'Params4bit':
        self = torch.Tensor._make_subclass(cls, data)
        self.requires_grad = requires_grad
        self.quant_state = QuantState.from_dict(qs_dict=quantized_stats, device=device)
        self.blocksize = self.quant_state.blocksize
        self.compress_statistics = self.quant_state.nested
        self.quant_type = self.quant_state.quant_type
        self.bnb_quantized = True
        self.quant_storage = data.dtype
        self.module = module
        if self.module is not None:
            self.module.quant_state = self.quant_state
        return self

    def _quantize(self, device):
        w = self.data.contiguous()
        w_4bit, quant_state = bnb.functional.quantize_4bit(w, blocksize=self.blocksize, compress_statistics=self.compress_statistics, quant_type=self.quant_type, quant_storage=self.quant_storage)
        self.data = w_4bit
        self.quant_state = quant_state
        if self.module is not None:
            self.module.quant_state = quant_state
        self.bnb_quantized = True
        return self

    def cuda(self, device: 'Optional[Union[int, device, str]]'=None, non_blocking: 'bool'=False):
        return self

    @overload
    def to(self: 'T', device: 'Optional[Union[int, device]]'=..., dtype: 'Optional[Union[dtype, str]]'=..., non_blocking: 'bool'=...) ->T:
        ...

    @overload
    def to(self: 'T', dtype: 'Union[dtype, str]', non_blocking: 'bool'=...) ->T:
        ...

    @overload
    def to(self: 'T', tensor: 'Tensor', non_blocking: 'bool'=...) ->T:
        ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type == 'cuda' and not self.bnb_quantized:
            return self._quantize(device)
        else:
            if self.quant_state is not None:
                self.quant_state
            new_param = Params4bit(super(), requires_grad=self.requires_grad, quant_state=self.quant_state, blocksize=self.blocksize, compress_statistics=self.compress_statistics, quant_type=self.quant_type, quant_storage=self.quant_storage)
            return new_param


def fix_4bit_weight_quant_state_from_module(module: "Union['Embedding4bit', 'Linear4bit']"):
    if getattr(module.weight, 'quant_state', None) is not None:
        return
    if getattr(module, 'quant_state', None) is None:
        warnings.warn('FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.')
    assert module.weight.shape[1] == 1
    if not isinstance(module.weight, Params4bit):
        module.weight = Params4bit(module.weight, quant_storage=module.quant_storage, bnb_quantized=True)
    module.weight.quant_state = module.quant_state


class Linear4bit(nn.Linear):
    """
    This class is the base module for the 4-bit quantization algorithm presented in [QLoRA](https://arxiv.org/abs/2305.14314).
    QLoRA 4-bit linear layers uses blockwise k-bit quantization under the hood, with the possibility of selecting various
    compute datatypes such as FP4 and NF4.

    In order to quantize a linear layer one should first load the original fp16 / bf16 weights into
    the Linear4bit module, then call `quantized_module.to("cuda")` to quantize the fp16 / bf16 weights.

    Example:

    ```python
    import torch
    import torch.nn as nn

    import bitsandbytes as bnb
    from bnb.nn import Linear4bit

    fp16_model = nn.Sequential(
        nn.Linear(64, 64),
        nn.Linear(64, 64)
    )

    quantized_model = nn.Sequential(
        Linear4bit(64, 64),
        Linear4bit(64, 64)
    )

    quantized_model.load_state_dict(fp16_model.state_dict())
    quantized_model = quantized_model.to(0) # Quantization happens here
    ```
    """

    def __init__(self, input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True, quant_type='fp4', quant_storage=torch.uint8, device=None):
        """
        Initialize Linear4bit class.

        Args:
            input_features (`str`):
                Number of input features of the linear layer.
            output_features (`str`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        super().__init__(input_features, output_features, bias, device)
        self.weight = Params4bit(self.weight.data, requires_grad=False, compress_statistics=compress_statistics, quant_type=quant_type, quant_storage=quant_storage, module=self)
        self.compute_dtype = compute_dtype
        self.compute_type_is_set = False
        self.quant_state = None
        self.quant_storage = quant_storage

    def set_compute_type(self, x):
        if x.dtype in [torch.float32, torch.bfloat16]:
            self.compute_dtype = x.dtype
        elif x.dtype == torch.float16:
            if self.compute_dtype == torch.float32 and x.numel() == x.shape[-1]:
                warnings.warn('Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference.')
                warnings.filterwarnings('ignore', message='.*inference.')
            if self.compute_dtype == torch.float32 and x.numel() != x.shape[-1]:
                warnings.warn('Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference or training speed.')
                warnings.filterwarnings('ignore', message='.*inference or training')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        save weight and bias,
        then fill state_dict with components of quant_state
        """
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if getattr(self.weight, 'quant_state', None) is not None:
            for k, v in self.weight.quant_state.as_dict(packed=True).items():
                destination[prefix + 'weight.' + k] = v if keep_vars else v.detach()

    def forward(self, x: 'torch.Tensor'):
        fix_4bit_weight_quant_state_from_module(self)
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data
        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True
        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x
        bias = None if self.bias is None else self.bias
        out = bnb.matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state)
        out = out
        return out


class LinearFP4(Linear4bit):
    """
    Implements the FP4 data type.
    """

    def __init__(self, input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True, quant_storage=torch.uint8, device=None):
        """
        Args:
            input_features (`str`):
                Number of input features of the linear layer.
            output_features (`str`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        super().__init__(input_features, output_features, bias, compute_dtype, compress_statistics, 'fp4', quant_storage, device)


class LinearNF4(Linear4bit):
    """Implements the NF4 data type.

    Constructs a quantization data type where each bin has equal area under a standard normal distribution N(0, 1) that
    is normalized into the range [-1, 1].

    For more information read the paper: QLoRA: Efficient Finetuning of Quantized LLMs (https://arxiv.org/abs/2305.14314)

    Implementation of the NF4 data type in bitsandbytes can be found in the `create_normal_map` function in
    the `functional.py` file: https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L236.
    """

    def __init__(self, input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True, quant_storage=torch.uint8, device=None):
        """
        Args:
            input_features (`str`):
                Number of input features of the linear layer.
            output_features (`str`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        super().__init__(input_features, output_features, bias, compute_dtype, compress_statistics, 'nf4', quant_storage, device)


class Int8Params(torch.nn.Parameter):

    def __new__(cls, data=None, requires_grad=True, has_fp16_weights=False, CB=None, SCB=None):
        if data is None:
            data = torch.empty(0)
        obj = torch.Tensor._make_subclass(cls, data, requires_grad)
        obj.CB = CB
        obj.SCB = SCB
        obj.has_fp16_weights = has_fp16_weights
        return obj

    def cuda(self, device):
        if self.has_fp16_weights:
            return super()
        else:
            B = self.data.contiguous().half()
            CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
            del CBt
            del SCBt
            self.data = CB
            self.CB = CB
            self.SCB = SCB
        return self

    def __deepcopy__(self, memo):
        new_instance = type(self).__new__(type(self), data=copy.deepcopy(self.data, memo), requires_grad=self.requires_grad, has_fp16_weights=self.has_fp16_weights, CB=copy.deepcopy(self.CB, memo), SCB=copy.deepcopy(self.SCB, memo))
        return new_instance

    @overload
    def to(self: 'T', device: 'Optional[Union[int, device]]'=..., dtype: 'Optional[Union[dtype, str]]'=..., non_blocking: 'bool'=...) ->T:
        ...

    @overload
    def to(self: 'T', dtype: 'Union[dtype, str]', non_blocking: 'bool'=...) ->T:
        ...

    @overload
    def to(self: 'T', tensor: 'Tensor', non_blocking: 'bool'=...) ->T:
        ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type == 'cuda' and self.data.device.type == 'cpu':
            return self
        else:
            new_param = Int8Params(super(), requires_grad=self.requires_grad, has_fp16_weights=self.has_fp16_weights)
            new_param.CB = self.CB
            new_param.SCB = self.SCB
            return new_param


class Embedding8bit(nn.Embedding):
    """
    This class implements [LLM.int8()](https://arxiv.org/abs/2208.07339) algorithm for embedding layer

    Quantization API is similar to Linear8bitLt:
    ```python
    import torch
    import torch.nn as nn

    from bitsandbytes.nn import Embedding8bit

    fp16_module = nn.Embedding(128, 64)
    int8_module = Embedding8bit(128, 64)

    int8_module.load_state_dict(fp16_module.state_dict())

    int8_module = int8_module.to(0) # Quantization happens here
    ```
    """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.dtype = self.weight.data.dtype
        self.weight = Int8Params(self.weight.data, has_fp16_weights=False, requires_grad=False)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        raise NotImplementedError('Saving Embedding8bit module is not implemented')

    def forward(self, input: 'Tensor') ->Tensor:
        if not hasattr(self.weight, 'SCB'):
            raise RuntimeError('Embedding layer is not quantized. Please call .cuda() or .to(device) first.')
        rows = self.weight.data
        row_stats = self.weight.SCB
        assert rows.shape == (self.num_embeddings, self.embedding_dim)
        assert row_stats.shape == (self.num_embeddings,)
        compressed_output = F.embedding(input, rows)
        compressed_output_stats = F.embedding(input, row_stats.view(self.num_embeddings, 1))
        output = compressed_output * (compressed_output_stats / 127.0)
        return output


class Embedding4bit(nn.Embedding):
    """
    This is the base class similar to Linear4bit. It implements the 4-bit quantization algorithm presented in
    [QLoRA](https://arxiv.org/abs/2305.14314) for embeddings.

    Quantization API is similar to Linear4bit:
    ```python
    import torch
    import torch.nn as nn

    from bitsandbytes.nn import Embedding4bit

    fp16_module = nn.Embedding(128, 64)
    quantized_module = Embedding4bit(128, 64)

    quantized_module.load_state_dict(fp16_module.state_dict())

    quantized_module = quantized_module.to(0) # Quantization happens here
    ```
    """

    def __init__(self, num_embeddings, embedding_dim, dtype=None, quant_type='fp4', quant_storage=torch.uint8, device=None):
        super().__init__(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.dtype = self.weight.data.dtype
        self.weight = Params4bit(self.weight.data, requires_grad=False, compress_statistics=None, quant_type=quant_type, quant_storage=quant_storage, module=self)
        blocksize = self.weight.blocksize
        if embedding_dim % blocksize != 0:
            warnings.warn(f'Embedding size {embedding_dim} is not divisible by block size {blocksize}. This will lead to slow inference.')

    def _forward_with_partial_dequantize(self, input: 'Tensor'):
        assert self.embedding_dim % self.weight.quant_state.blocksize == 0
        w_4bit_uint8 = self.weight.data.view(torch.uint8).view(self.num_embeddings * self.embedding_dim // 2, 1)
        output_4bit = torch.nn.functional.embedding(weight=w_4bit_uint8.view(self.num_embeddings, self.embedding_dim // 2), input=input).view(-1, 1)
        assert output_4bit.shape == (input.numel() * self.embedding_dim // 2, 1)
        blocks_per_emb = self.embedding_dim // self.weight.blocksize
        absmax = self.weight.quant_state.absmax
        assert absmax.shape == (self.num_embeddings * blocks_per_emb,)
        output_absmax = torch.nn.functional.embedding(weight=absmax.view(self.num_embeddings, blocks_per_emb), input=input).view(-1)
        assert output_absmax.shape == (input.numel() * blocks_per_emb,)
        output_quant_state = copy.deepcopy(self.weight.quant_state)
        output_quant_state.absmax = output_absmax
        output_quant_state.shape = torch.Size((*input.shape, self.embedding_dim))
        output = bnb.functional.dequantize_4bit(output_4bit, output_quant_state)
        assert output.shape == (*input.shape, self.embedding_dim)
        return output

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        raise NotImplementedError('Saving Embedding4bit module is not implemented')

    def forward(self, input: 'Tensor') ->Tensor:
        fix_4bit_weight_quant_state_from_module(self)
        if self.embedding_dim % self.weight.quant_state.blocksize == 0:
            return self._forward_with_partial_dequantize(input)
        dequantized_weight = bnb.functional.dequantize_4bit(self.weight.data, self.weight.quant_state)
        return torch.nn.functional.embedding(weight=dequantized_weight, input=input)


class EmbeddingFP4(Embedding4bit):

    def __init__(self, num_embeddings, embedding_dim, dtype=None, quant_storage=torch.uint8, device=None):
        super().__init__(num_embeddings, embedding_dim, dtype=dtype, quant_type='fp4', quant_storage=quant_storage, device=device)


class EmbeddingNF4(Embedding4bit):

    def __init__(self, num_embeddings, embedding_dim, dtype=None, quant_storage=torch.uint8, device=None):
        super().__init__(num_embeddings, embedding_dim, dtype=dtype, quant_type='nf4', quant_storage=quant_storage, device=device)


LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING = {'row': 0, 'col32': 1, 'col_turing': 2, 'col_ampere': 3}


INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING = {val: name for name, val in LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING.items()}


def _get_tile_size(format):
    assert format in ('col_turing', 'col_ampere'), f'please find this assert and manually enter tile size for {format}'
    return (8, 32) if format == 'col_turing' else (32, 32)


def get_inverse_transform_indices(transform_tile: 'Callable[[torch.Tensor], torch.Tensor]', tile_size: 'Tuple[int, int]'):
    """
    Compute a permutation of indices that invert the specified (tiled) matrix transformation

    :param transform_tile: a function that applies forward transform to a tensor of shape [dim1, dim2]
    :param tile_size: higher-level tile dimensions, i.e. (8, 32) for Turing and (32, 32) for Ampere
    :note: we assume that tile_transform applies to a cpu-based int8 tensor of shape tile_size
    :example: transform_tile function for the turing layout (bitsandbytes.functional as F)
    :returns: indices
    """
    d1, d2 = tile_size
    assert 0 < d1 * d2 < 2 ** 64
    tile_indices = torch.arange(d1 * d2, dtype=torch.int64).view(d1, d2)
    permuted_tile_indices = torch.zeros_like(tile_indices)
    for i in range(8):
        ith_dim_indices = torch.div(tile_indices, 256 ** i, rounding_mode='trunc') % 256
        sample_tile_i = (ith_dim_indices - 128).contiguous()
        assert torch.all(sample_tile_i.int() + 128 == ith_dim_indices), 'int overflow'
        permuted_tile_i = transform_tile(sample_tile_i)
        ith_permuted_indices = permuted_tile_i + 128
        permuted_tile_indices += ith_permuted_indices * 256 ** i
        if d1 * d2 < 256 ** i:
            break
    return permuted_tile_indices


def get_tile_inds(format, device):
    transform = lambda x: F.transform(x.to(device), from_order='row', to_order=format)[0]
    with torch.no_grad():
        return get_inverse_transform_indices(transform, _get_tile_size(format))


def undo_layout(permuted_tensor: 'torch.Tensor', tile_indices: 'torch.LongTensor') ->torch.Tensor:
    """
    Undo a tiled permutation such as turing or ampere layout

    :param permuted_tensor: torch tensor in a permuted layout
    :param tile_indices: reverse transformation indices, from get_inverse_transform_indices
    :return: contiguous row-major tensor
    """
    (rows, cols), (tile_rows, tile_cols) = permuted_tensor.shape, tile_indices.shape
    assert rows % tile_rows == cols % tile_cols == 0, 'tensor must contain a whole number of tiles'
    tensor = permuted_tensor.reshape(-1, tile_indices.numel()).t()
    outputs = torch.empty_like(tensor)
    outputs[tile_indices.flatten()] = tensor
    outputs = outputs.reshape(tile_rows, tile_cols, cols // tile_cols, rows // tile_rows)
    outputs = outputs.permute(3, 0, 2, 1)
    return outputs.reshape(rows, cols).contiguous()


def maybe_rearrange_weight(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    weight = state_dict.get(f'{prefix}weight')
    if weight is None:
        return
    weight_format = state_dict.pop(f'{prefix}weight_format', 'row')
    if isinstance(weight_format, torch.Tensor):
        weight_format = weight_format.item()
    if isinstance(weight_format, int) and weight_format not in INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING:
        raise ValueError(f'Expected supported weight format - got {weight_format}')
    elif isinstance(weight_format, int) and weight_format in INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING:
        weight_format = INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING[weight_format]
    if weight_format != 'row':
        tile_indices = get_tile_inds(weight_format, weight.device)
        state_dict[f'{prefix}weight'] = undo_layout(weight, tile_indices)


class Linear8bitLt(nn.Linear):
    """
    This class is the base module for the [LLM.int8()](https://arxiv.org/abs/2208.07339) algorithm.
    To read more about it, have a look at the paper.

    In order to quantize a linear layer one should first load the original fp16 / bf16 weights into
    the Linear8bitLt module, then call `int8_module.to("cuda")` to quantize the fp16 weights.

    Example:

    ```python
    import torch
    import torch.nn as nn

    import bitsandbytes as bnb
    from bnb.nn import Linear8bitLt

    fp16_model = nn.Sequential(
        nn.Linear(64, 64),
        nn.Linear(64, 64)
    )

    int8_model = nn.Sequential(
        Linear8bitLt(64, 64, has_fp16_weights=False),
        Linear8bitLt(64, 64, has_fp16_weights=False)
    )

    int8_model.load_state_dict(fp16_model.state_dict())
    int8_model = int8_model.to(0) # Quantization happens here
    ```
    """

    def __init__(self, input_features: 'int', output_features: 'int', bias=True, has_fp16_weights=True, memory_efficient_backward=False, threshold=0.0, index=None, device=None):
        """
        Initialize Linear8bitLt class.

        Args:
            input_features (`int`):
                Number of input features of the linear layer.
            output_features (`int`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        super().__init__(input_features, output_features, bias, device)
        assert not memory_efficient_backward, 'memory_efficient_backward is no longer required and the argument is deprecated in 0.37.0 and will be removed in 0.39.0'
        self.state = bnb.MatmulLtState()
        self.index = index
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True
        self.weight = Int8Params(self.weight.data, has_fp16_weights=has_fp16_weights, requires_grad=has_fp16_weights)
        self._register_load_state_dict_pre_hook(maybe_rearrange_weight)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        scb_name = 'SCB'
        param_from_weight = getattr(self.weight, scb_name)
        param_from_state = getattr(self.state, scb_name)
        layout_reordered = self.state.CxB is not None
        key_name = prefix + f'{scb_name}'
        format_name = prefix + 'weight_format'
        if not self.state.has_fp16_weights:
            if param_from_weight is not None:
                destination[key_name] = param_from_weight if keep_vars else param_from_weight.detach()
                destination[format_name] = torch.tensor(0, dtype=torch.uint8)
            elif param_from_state is not None and not layout_reordered:
                destination[key_name] = param_from_state if keep_vars else param_from_state.detach()
                destination[format_name] = torch.tensor(0, dtype=torch.uint8)
            elif param_from_state is not None:
                destination[key_name] = param_from_state if keep_vars else param_from_state.detach()
                weights_format = self.state.formatB
                if weights_format not in LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING:
                    raise ValueError(f'Unrecognized weights format {weights_format}')
                weights_format = LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING[weights_format]
                destination[format_name] = torch.tensor(weights_format, dtype=torch.uint8)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        unexpected_copy = list(unexpected_keys)
        for key in unexpected_copy:
            input_name = key[len(prefix):]
            if input_name == 'SCB':
                if self.weight.SCB is None:
                    raise RuntimeError('Loading a quantized checkpoint into non-quantized Linear8bitLt is not supported. Please call module.cuda() before module.load_state_dict()')
                input_param = state_dict[key]
                self.weight.SCB.copy_(input_param)
                if self.state.SCB is not None:
                    self.state.SCB = self.weight.SCB
                unexpected_keys.remove(key)

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x: 'torch.Tensor'):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data
        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)
        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                del self.state.CB
                self.weight.data = self.state.CxB
        return out

