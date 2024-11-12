
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


import torch


from typing import Dict


from typing import Sequence


from typing import Tuple


from typing import Union


from torch.fx.node import Argument


from torch.fx.node import Node


from torch.fx.node import Target


from typing import Optional


import numpy as np


import torchvision.models as models


import torch.nn as nn


import torch.nn.functional as F


import torchvision.datasets as datasets


import torchvision.transforms as transforms


import copy


from typing import Any


from torch.library import custom_op


from torch import nn


from typing import List


from torch.distributed._tensor import Replicate


from torch.distributed._tensor import Shard


from torch.distributed.device_mesh import DeviceMesh


from torch.distributed.tensor.parallel import ColwiseParallel


from torch.distributed.tensor.parallel import PrepareModuleInput


from torch.distributed.tensor.parallel import RowwiseParallel


from torch.distributed.tensor.parallel import SequenceParallel


from torch.distributed.tensor.parallel import parallelize_module


import time


from torch.distributed._composable.fsdp import MixedPrecisionPolicy


from torch.distributed._composable.fsdp.fully_shard import fully_shard


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


from torch.distributed.device_mesh import init_device_mesh


from torch.distributed._tensor.device_mesh import init_device_mesh


import torch.fx


from functools import partial


import pandas as pd


import torch._dynamo as torchdynamo


from torch._dynamo.testing import collect_results


from torch._dynamo.testing import same


import typing as t


from copy import deepcopy


import torchvision


from torch.ao.quantization.quantize_fx import convert_fx


from torch.ao.quantization.quantize_fx import convert_to_reference_fx


from torch.ao.quantization.quantize_fx import prepare_fx


from torch.fx.experimental.normalize import NormalizeArgs


from torch.fx.passes import shape_prop


import torch.utils.data as data


import random


import torch.optim as optim


from torch.utils.tensorboard import SummaryWriter


from functools import reduce


from enum import Enum


import collections.abc


from typing import Callable


from typing import Set


from enum import auto


from typing import Type


import math


import warnings


from typing import Collection


from torch.export import ExportedProgram


from abc import ABC


from abc import abstractmethod


from typing import cast


from torch._inductor.codecache import FxGraphCachePickler


from torch._inductor.codecache import sha256_hash


from torch.fx.experimental.proxy_tensor import unset_fake_temporarily


from torch._guards import detect_fake_mode


from torch._subclasses.fake_tensor import FakeTensor


from torch.export import ExportGraphSignature


from torch.export.exported_program import CustomObjArgument


from torch.export.exported_program import InputKind


from torch.export.exported_program import InputSpec


from torch.export.exported_program import ModuleCallEntry


from torch.export.exported_program import ModuleCallSignature


from torch.export.exported_program import OutputKind


from torch.export.exported_program import OutputSpec


from torch.export.exported_program import TensorArgument


from inspect import signature


from torch.export import Dim


from torch.export import export


import torch._dynamo as td


from torch._dynamo.utils import detect_fake_mode


from torch._functorch.aot_autograd import aot_export_joint_simple


import functools


from torch import SymBool


from torch import SymFloat


from torch import SymInt


from torch._ops import OpOverloadPacket


from torch.fx.node import _get_qualified_name


from typing import NamedTuple


from torch.fx.passes.shape_prop import TensorMetadata


from torch.utils._python_dispatch import _disable_current_modes


import collections


from typing import overload


from torch._decomp import core_aten_decompositions


from torch._decomp import get_decompositions as get_torch_decompositions


from torch._ops import OpOverload


from torch._decomp import register_decomposition


from torch.fx.passes.pass_manager import PassManager


import torch.fx.passes.operator_support as ops


from torch.fx.passes.splitter_base import FxNetAccFusionsFinder


from torch.fx.passes.splitter_base import FxNetAccNodesFinder


from torch.fx.passes.splitter_base import Subgraph


from torch.fx.passes.splitter_base import _SplitterBase


from torch.fx.passes.splitter_base import _SplitterSettingBase


from torch.fx.passes.tools_common import CALLABLE_NODE_OPS


from torch.fx.passes.tools_common import NodeSet


from typing import Mapping


from torch.fx.graph_module import GraphModule


from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner


from torch.fx.passes.infra.partitioner import Partition


from torch.fx.passes.operator_support import OperatorSupport


from torch.fx.passes.operator_support import SupportDict


from typing import Iterator


from torch.nn import Module


from collections import OrderedDict


import torch._prims as prims


from torch._dynamo.variables import BuiltinVariable


from torch.fx.immutable_collections import immutable_list


import numpy


from typing import Iterable


import torch.fx as fx


from torch.fx.passes.splitter_base import SplitResult


from collections.abc import Sequence


from torch.fx.experimental.const_fold import split_const_subgraphs


from torch.fx.passes.infra.pass_base import PassResult


from functools import wraps


from torch.fx.passes.pass_manager import inplace_wrapper


from torch.fx.passes.shape_prop import ShapeProp


from torch.fx.passes.splitter_base import generate_inputs_for_submodules


from torch import fx


from torch.testing._internal.common_utils import run_tests


from torch.testing._internal.common_utils import TestCase


from collections import Counter


import itertools


import torch.nn.quantized._reference as nnqr


from torch.ao.quantization import default_qconfig


from torch.ao.quantization.backend_config import get_tensorrt_backend_config_dict


from torch.ao.quantization.backend_config import ObservationType


from torch.ao.quantization.fx.match_utils import MatchAllNode


from torch.ao.quantization.quantize_fx import prepare_qat_fx


from torch.testing._internal.common_cuda import TEST_CUDA


from torch.testing._internal.common_quantization import NodeSpec as ns


from torch.testing._internal.common_quantization import QuantizationTestCase


from torch.package import PackageImporter


import torch._dynamo.config


from torch.library import Library


import torch.fx.passes.operator_support as op_support


import torch.fx.passes.shape_prop as shape_prop


from torch.fx.passes import splitter_base


from typing import BinaryIO


from typing import TextIO


from torch.fx.passes.split_utils import getattr_recursive


from torch.package import PackageExporter


import torch.fx.passes.net_min_base as net_min_base


from torch.fx.passes.tools_common import Tensors


import torch.fx.passes.splitter_base as splitter_base


from torch.fx.passes.tools_common import get_acc_ops_name


import inspect


import re


from collections import defaultdict


from enum import Flag


from typing import DefaultDict


from torch.fx.passes.shape_prop import _extract_tensor_metadata


from types import FunctionType


import torch.jit as jit


from torch._sources import normalize_source_lines


from torch.fx import Graph


from torch.fx import Tracer


from torch.fx.immutable_collections import immutable_dict


from torch.fx.passes import graph_drawer


from typing import Generator


import torch.utils._pytree as pytree


from torch._C import _disabled_torch_function_impl


from torch.fx import GraphModule


from torch.utils.cpp_extension import IS_WINDOWS


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from math import exp


from torch.nn.parameter import Parameter


from torch.nn.parameter import UninitializedParameter


from torch.nn import functional as F


class GeLU(torch.nn.Module):

    def __init__(self, mode='tanh'):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate=self.mode)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = torch.linalg.norm(x, ord=2, dim=1)
        x = self.relu(x)
        return x


class VGG(nn.Module):

    def __init__(self, layer_spec, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        layers = []
        in_channels = 3
        for l in layer_spec:
            if l == 'pool':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [nn.Conv2d(in_channels, l, kernel_size=3, padding=1), nn.BatchNorm2d(l), nn.ReLU()]
                in_channels = l
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(512 * 1 * 1, 4096), nn.ReLU(), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MyModel(nn.Module):

    def __init__(self, padding: 'Sequence[int]'):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(1, 5, kernel_size=3)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        padded_x = torch.ops.torchtrt_ex.triton_circular_pad(x, self.padding)
        y = self.conv(padded_x)
        return y


class Elu(torch.nn.Module):

    def __init__(self):
        super(Elu, self).__init__()
        self.elu = torch.nn.ELU()

    def forward(self, x):
        return self.elu(x)


class RMSNorm(nn.Module):
    """Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: 'int', eps: 'float'=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: 'torch.Tensor'):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: 'torch.Tensor'):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


def reshape_for_broadcast(freqs_cis: 'torch.Tensor', x: 'torch.Tensor') ->torch.Tensor:
    """Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [(d if i == 1 or i == ndim - 1 else 1) for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: 'torch.Tensor', xk: 'torch.Tensor', freqs_cis: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: 'torch.Tensor', n_rep: 'int') ->torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    """Multi-head attention module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: 'ModelArgs'):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads
        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)

    def init_weights(self, init_std: 'float') ->None:
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(self, x: 'torch.Tensor', freqs_cis: 'torch.Tensor') ->Any:
        """Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bs, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bs, seqlen, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)
        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """FeedForward module.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(self, dim: 'int', hidden_dim: 'int', multiple_of: 'int', ffn_dim_multiplier: 'Optional[float]'):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x) ->Any:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: 'float') ->None:
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """TransformerBlock Module.

    Args:
        layer_id (int): Identifier for the layer.
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: 'int', model_args: 'ModelArgs'):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(dim=model_args.dim, hidden_dim=4 * model_args.dim, multiple_of=model_args.multiple_of, ffn_dim_multiplier=model_args.ffn_dim_multiplier)
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers
        self.attention_norm = RMSNorm(dim=model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = RMSNorm(dim=model_args.dim, eps=model_args.norm_eps)
        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

    def forward(self, x: 'torch.Tensor', freqs_cis: 'torch.Tensor'):
        """Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


def precompute_freqs_cis(dim: 'int', end: 'int', theta: 'float'=10000.0) ->torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    """
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


class ParallelTransformer(nn.Module):
    """Transformer Module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        model_args (ModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: 'ModelArgs', tp_mesh: 'DeviceMesh'=None):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.tok_embeddings
        self.tok_embeddings = self.parallel_embeddings(self.tok_embeddings, tp_mesh)
        self.register_buffer('freqs_cis', self._precompute_freqs_cis(), persistent=True)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            block = TransformerBlock(layer_id, model_args)
            self.layers[str(layer_id)] = block
            self.parallel_transformer_block(self.layers[str(layer_id)], tp_mesh)
        self.norm = RMSNorm(dim=model_args.dim, eps=model_args.norm_eps)
        self.norm = self.parallel_norm(self.norm, tp_mesh)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.output = self.parallel_output(self.output, tp_mesh)
        self.init_weights()

    def parallel_transformer_block(self, transformer_block, tp_mesh):
        if tp_mesh.size() <= 1:
            return
        plan = {'attention': PrepareModuleInput(input_layouts=(Shard(1), None), desired_input_layouts=(Replicate(), None)), 'attention.wq': ColwiseParallel(), 'attention.wk': ColwiseParallel(), 'attention.wv': ColwiseParallel(), 'attention.wo': RowwiseParallel(output_layouts=Shard(1)), 'attention_norm': SequenceParallel(), 'feed_forward': PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)), 'feed_forward.w1': ColwiseParallel(), 'feed_forward.w2': RowwiseParallel(output_layouts=Shard(1)), 'feed_forward.w3': ColwiseParallel(), 'ffn_norm': SequenceParallel()}
        attn_layer = transformer_block.attention
        attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()
        parallelize_module(transformer_block, tp_mesh, plan)

    def parallel_embeddings(self, embedding, tp_mesh):
        plan = {'tok_embeddings': RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1))}
        return parallelize_module(embedding, tp_mesh, plan)

    def parallel_output(self, output, tp_mesh):
        plan = {'output': ColwiseParallel(input_layouts=Shard(1))}
        return parallelize_module(output, tp_mesh, plan)

    def parallel_norm(self, norm, tp_mesh):
        plan = {'norm': SequenceParallel()}
        return parallelize_module(norm, tp_mesh, plan)

    def reset_parameters(self):
        with torch.device(self.freqs_cis.device):
            self.freqs_cis = self._precompute_freqs_cis()

    def init_weights(self):
        """[Note: On ``init_weights`` vs.

        ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.

        """
        with torch.device(self.freqs_cis.device):
            self.freqs_cis = self._precompute_freqs_cis()
        nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            layer.init_weights()
        self.norm.reset_parameters()
        final_out_std = self.model_args.dim ** -0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(self.output.weight, mean=0.0, std=final_out_std, a=-cutoff_factor * final_out_std, b=cutoff_factor * final_out_std)

    def _precompute_freqs_cis(self) ->torch.Tensor:
        return precompute_freqs_cis(self.model_args.dim // self.model_args.n_heads, self.model_args.max_seq_len * 2, self.model_args.rope_theta)

    def forward(self, tokens: 'torch.Tensor'):
        """Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        h = self.norm(h) if self.norm else h
        return self.output(h).float() if self.output else h

    @classmethod
    def from_model_args(cls, model_args: 'ModelArgs') ->'Transformer':
        """Initialize a Transformer model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)


class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 3200)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(3200, 1600)
        self.in_proj2 = nn.Linear(1600, 500)
        self.out_proj2 = nn.Linear(500, 100)

    def forward(self, x):
        x = self.out_proj(self.relu(self.in_proj(x)))
        x = self.relu(x)
        x = self.out_proj2(self.relu(self.in_proj2(x)))
        return x


class ArgsToKwargsWrapper(torch.nn.Module):

    def __init__(self, model):
        super(ArgsToKwargsWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, decoder_input_ids):
        return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)


class M(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 46)

    def forward(self, x):
        out = self.linear(x)
        return out


class Norm(torch.nn.Module):

    def __init__(self):
        super(Norm, self).__init__()

    def forward(self, x):
        return torch.norm(x, 2, None, False)


class ConvGelu(torch.nn.Module):

    def __init__(self):
        super(ConvGelu, self).__init__()
        self.conv = nn.Conv2d(3, 32, 3, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu(x)
        return x


ASSUME_DYNAMIC_SHAPE_SUPPORT = False


CACHE_BUILT_ENGINES = False


DEBUG = False


DISABLE_TF32 = False


DLA_GLOBAL_DRAM_SIZE = 536870912


DLA_LOCAL_DRAM_SIZE = 1073741824


DLA_SRAM_SIZE = 1048576


DRYRUN = False


class AccOpProperty(Flag):
    """
    A collection of static properties for acc_ops.

    * pointwise - op commutes with data restructuring ops such as reshape,
        transpose, permute. e.g. op(reshape(x)) == reshape(op(x)).
        Alternatively, for tensor x = (x1, x2, ...), there exists a scalar
        function f such that op(x) = (f(x1), f(x2), ...).
    * quantized - op expects quantized inputs and return quantized outputs
    * unary - op has exactly one graph dependent input. e.g. relu,
        dequantize, sum
    """
    pointwise = auto()
    quantized = auto()
    unary = auto()


def register_acc_op(acc_op: 'Callable'):
    """
    For a new acc op, add this as decorator to register it.
    """
    _acc_ops.add(acc_op)
    return acc_op


def register_acc_op_properties(*properties: AccOpProperty):
    """
    Attach properties to acc_op to inform optimization
    """

    def decorator(acc_op: 'Callable'):
        acc_op_properties[acc_op] |= set(properties)
        for prop in properties:
            acc_ops_with_property[prop].add(acc_op)
        return acc_op
    return decorator


ENABLE_CROSS_COMPILE_FOR_WINDOWS = False


ENABLE_EXPERIMENTAL_DECOMPOSITIONS = False


ENABLE_WEIGHT_STREAMING = False


def sanitized_torch_version() ->Any:
    return torch.__version__ if '.nv' not in torch.__version__ else torch.__version__.split('.nv')[0]


_FX_FE_AVAIL = True


HARDWARE_COMPATIBLE = False


LAZY_ENGINE_INIT = False


MAKE_REFITTABLE = False


MAX_AUX_STREAMS = None


MIN_BLOCK_SIZE = 5


NUM_AVG_TIMING_ITERS = 1


OPTIMIZATION_LEVEL = None


PASS_THROUGH_BUILD_FAILURES = False


REQUIRE_FULL_COMPILATION = False


REUSE_CACHED_ENGINES = False


SPARSE_WEIGHTS = False


TRUNCATE_DOUBLE = False


USE_EXPLICIT_TYPING = False


USE_FAST_PARTITIONER = True


USE_FP32_ACC = False


USE_PYTHON_RUNTIME = False


VERSION_COMPATIBLE = False


WORKSPACE_SIZE = 0


def needs_torch_tensorrt_runtime(f: 'Callable[..., Any]') ->Callable[..., Any]:

    def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) ->Any:
        if ENABLED_FEATURES.torch_tensorrt_runtime:
            return f(*args, **kwargs)
        else:

            def not_implemented(*args: List[Any], **kwargs: Dict[str, Any]) ->Any:
                raise NotImplementedError('Torch-TensorRT Runtime is not available')
            return not_implemented(*args, **kwargs)
    return wrapper


DYNAMIC_DIM = -1


_LOGGER = logging.getLogger('torch_tensorrt [TensorRT Conversion Context]')


logger = logging.getLogger(__name__)


def _is_switch_required(curr_device_id: 'int', engine_device_id: 'int', curr_device_properties: 'torch._C._CudaDeviceProperties', engine_device_properties: 'torch._C._CudaDeviceProperties') ->bool:
    """Determines whether a device switch is required based on input device parameters"""
    if (curr_device_properties.major, curr_device_properties.minor) != (engine_device_properties.major, engine_device_properties.minor):
        logger.warning(f'Configured SM capability {engine_device_properties.major, engine_device_properties.minor} does not match with current device SM capability {curr_device_properties.major, curr_device_properties.minor}. Switching device context.')
        return True
    if curr_device_properties.name != engine_device_properties.name:
        logger.warning(f'Program compiled for {engine_device_properties.name} but current CUDA device is current device SM capability {curr_device_properties.name}. Attempting to switch device context for better compatibility.')
        return True
    if curr_device_id != engine_device_id:
        logger.warning(f'Configured Device ID: {engine_device_id} is different than current device ID: {curr_device_id}. Attempting to switch device context for better compatibility.')
        return True
    return False


def _get_most_compatible_device(curr_device_id: 'int', engine_device_id: 'int', engine_device_properties: 'torch._C._CudaDeviceProperties') ->Optional[Tuple[int, torch._C._CudaDeviceProperties]]:
    """Selects a runtime device based on compatibility checks"""
    all_devices = [(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())]
    logger.debug(f'All available devices: {all_devices}')
    target_device_sm = engine_device_properties.major, engine_device_properties.minor
    candidate_devices = [(i, device_properties) for i, device_properties in all_devices if (device_properties.major, device_properties.minor) == target_device_sm]
    logger.debug(f'Found candidate devices: {candidate_devices}')
    if len(candidate_devices) <= 1:
        return candidate_devices[0] if candidate_devices else None
    best_match = None
    for candidate in candidate_devices:
        i, device_properties = candidate
        if device_properties.name == engine_device_properties.name:
            if i == curr_device_id:
                best_match = candidate
                break
            elif i == engine_device_id:
                best_match = candidate
            elif best_match is None:
                best_match = candidate
    return best_match


def _select_rt_device(curr_device_id: 'int', engine_device_id: 'int', engine_device_properties: 'torch._C._CudaDeviceProperties') ->Tuple[int, torch._C._CudaDeviceProperties]:
    """Wraps compatible device check and raises error if none are found"""
    new_target_device_opt = _get_most_compatible_device(curr_device_id, engine_device_id, engine_device_properties)
    assert new_target_device_opt is not None, 'Could not find a compatible device on the system to run TRT Engine'
    return new_target_device_opt


def multi_gpu_device_check() ->None:
    if not torch_tensorrt.runtime._multi_device_safe_mode._PY_RT_MULTI_DEVICE_SAFE_MODE and torch.cuda.device_count() > 1:
        logger.warning(f'Detected this engine is being instantitated in a multi-GPU system with multi-device safe mode disabled. For more on the implications of this as well as workarounds, see the linked documentation (https://pytorch.org/TensorRT/user_guide/runtime.html#multi-device-safe-mode). The engine is set to be instantiated on the current default cuda device, cuda:{torch.cuda.current_device()}. If this is incorrect, please set the desired cuda device via torch.cuda.set_device(...) and retry.')


ABI_TARGET_IDX = -1


DEVICE_IDX = -1


ENGINE_IDX = -1


HW_COMPATIBLE_IDX = -1


INPUT_BINDING_NAMES_IDX = -1


NAME_IDX = -1


OUTPUT_BINDING_NAMES_IDX = -1


SERIALIZATION_LEN = -1


SERIALIZED_METADATA_IDX = -1


SerializedTensorRTEngineFmt = List[Union[str, bytes]]


SerializedTorchTensorRTModuleFmt = Tuple[str, Optional[SerializedTensorRTEngineFmt], List[str], List[str]]


TARGET_PLATFORM_IDX = -1


def for_all_methods(decorator: 'Callable[..., Any]', exclude: 'Optional[List[str]]'=None) ->Callable[..., Any]:
    exclude_list: 'List[str]' = []
    if exclude:
        exclude_list = exclude

    def decorate(cls: 'Type[T]') ->Type[T]:
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)) and attr not in exclude_list:
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


class TestModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Module()
        self.b = torch.nn.Module()
        self.a.weights = torch.nn.Parameter(torch.randn(1, 2))
        self.b.weights = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x + self.a.weights + self.b.weights


class ConditionalExceptionWrapper(nn.Module):
    """
    This wrapper class is used to wrap conditional raising of exceptions during
    rewriting. For example:

    .. code-block:: python

        if self.name != "x":
            raise AssertionError(f"Name was not x: {self.name}")

    Is rewritten into

    .. code-block:: python

        self._conditional_exception_wrapper_AssertionError(
            self.name != "x", f"Name was not x: {self.name}"
        )

    Note that __init__ takes the Exception class that it is wrapping, while
    forward takes the condition to check and the message for the exception.

    """
    _is_impure = True

    def __init__(self, exc: 'Type[Exception]'):
        super().__init__()
        self.exc = exc

    def forward(self, cond: 'bool', msg: 'str'):
        if cond:
            raise (self.exc if msg is None else self.exc(msg))


class ConditionalExceptionBoolCondWrapper(nn.Module):
    """
    This is a wrapper class to for boolean ops used inside conditionals
    raising exceptions.
    This currently only handles binary input cases for the `and` operator
    at one level of depth
    For example:

    .. code-block:: python

    if self.name != "x" and self.name != "y":
        raise AssertionError(f"Name was not x: {self.name}")

    rewrites the `self.name != "x" and self.name != "y"` with
    a `_conditional_exception_wrapper_AssertionError_bool` as follows:

    .. code-block:: python

        self._conditional_exception_wrapper_AssertionError(
            self._conditional_exception_wrapper_AssertionError_bool(self.name != "x" and self.name != "y"), f"Name was not x: {self.name}"
        )
    """
    _is_impure = True

    def __init__(self, op):
        super().__init__()

    def forward(self, *conds: Iterable):
        return all(conds)


class Frameworks(Enum):
    NUMPY = 'numpy'
    TORCH = 'torch'
    TRT = 'trt'


class TRTModule(torch.nn.Module):

    def __init__(self, engine=None, input_names=None, output_names=None, cuda_graph_batch_size=-1):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)
        self.engine = engine
        self.input_names = input_names
        self.output_names = output_names
        self.cuda_graph_batch_size = cuda_graph_batch_size
        self.initialized = False
        if engine:
            self._initialize()

    def _initialize(self):
        self.initialized = True
        self.context = self.engine.create_execution_context()
        self.input_binding_indices_in_order: 'Sequence[int]' = [self.engine.get_binding_index(name) for name in self.input_names]
        self.output_binding_indices_in_order: 'Sequence[int]' = [self.engine.get_binding_index(name) for name in self.output_names]
        primary_input_outputs = set()
        primary_input_outputs.update(self.input_binding_indices_in_order)
        primary_input_outputs.update(self.output_binding_indices_in_order)
        self.hidden_output_binding_indices_in_order: 'Sequence[int]' = []
        self.hidden_output_names: 'Sequence[str]' = []
        for i in range(self.engine.num_bindings // self.engine.num_optimization_profiles):
            if i not in primary_input_outputs:
                self.hidden_output_binding_indices_in_order.append(i)
                self.hidden_output_names.append(self.engine.get_binding_name(i))
        assert self.engine.num_bindings // self.engine.num_optimization_profiles == len(self.input_names) + len(self.output_names) + len(self.hidden_output_names)
        self.input_dtypes: 'Sequence[torch.dtype]' = [unified_dtype_converter(self.engine.get_binding_dtype(idx), Frameworks.TORCH) for idx in self.input_binding_indices_in_order]
        self.input_shapes: 'Sequence[Sequence[int]]' = [tuple(self.engine.get_binding_shape(idx)) for idx in self.input_binding_indices_in_order]
        self.output_dtypes: 'Sequence[torch.dtype]' = [unified_dtype_converter(self.engine.get_binding_dtype(idx), Frameworks.TORCH) for idx in self.output_binding_indices_in_order]
        self.output_shapes = [(tuple(self.engine.get_binding_shape(idx)) if self.engine.has_implicit_batch_dimension else tuple()) for idx in self.output_binding_indices_in_order]
        self.hidden_output_dtypes: 'Sequence[torch.dtype]' = [unified_dtype_converter(self.engine.get_binding_dtype(idx), Frameworks.TORCH) for idx in self.hidden_output_binding_indices_in_order]
        self.hidden_output_shapes = [(tuple(self.engine.get_binding_shape(idx)) if self.engine.has_implicit_batch_dimension else tuple()) for idx in self.hidden_output_binding_indices_in_order]

    def _check_initialized(self):
        if not self.initialized:
            raise RuntimeError('TRTModule is not initialized.')

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        self._check_initialized()
        state_dict[prefix + 'engine'] = bytearray(self.engine.serialize())
        state_dict[prefix + 'input_names'] = self.input_names
        state_dict[prefix + 'output_names'] = self.output_names
        state_dict[prefix + 'cuda_graph_batch_size'] = self.cuda_graph_batch_size

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        engine_bytes = state_dict[prefix + 'engine']
        logger = trt.Logger()
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.input_names = state_dict[prefix + 'input_names']
        self.output_names = state_dict[prefix + 'output_names']
        self._initialize()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['engine'] = bytearray(self.engine.serialize())
        state.pop('context', None)
        return state

    def __setstate__(self, state):
        logger = trt.Logger()
        runtime = trt.Runtime(logger)
        state['engine'] = runtime.deserialize_cuda_engine(state['engine'])
        self.__dict__.update(state)
        if self.engine:
            self.context = self.engine.create_execution_context()

    def forward(self, *inputs):
        with torch.autograd.profiler.record_function('TRTModule:Forward'):
            self._check_initialized()
            with torch.autograd.profiler.record_function('TRTModule:ProcessInputs'):
                assert len(inputs) == len(self.input_names), f'Wrong number of inputs, expect {len(self.input_names)} get {len(inputs)}.'
                batch_size = inputs[0].shape[0]
                contiguous_inputs: 'List[torch.Tensor]' = [i.contiguous() for i in inputs]
                bindings: 'List[Any]' = [None] * (len(self.input_names) + len(self.output_names) + len(self.hidden_output_names))
                for i, input_name in enumerate(self.input_names):
                    assert inputs[i].dtype == self.input_dtypes[i], f'Dtype mismatch for {i}th input({input_name}). Expect {self.input_dtypes[i]}, got {inputs[i].dtype}.'
                    idx = self.input_binding_indices_in_order[i]
                    bindings[idx] = contiguous_inputs[i].data_ptr()
                    if not self.engine.has_implicit_batch_dimension:
                        self.context.set_binding_shape(idx, tuple(contiguous_inputs[i].shape))
                    else:
                        assert inputs[i].size()[1:] == self.input_shapes[i], f'Shape mismatch for {i}th input({input_name}). Expect {self.input_shapes[i]}, got {inputs[i].size()[1:]}.'
            with torch.autograd.profiler.record_function('TRTModule:ProcessOutputs'):
                outputs: 'List[torch.Tensor]' = []
                for i, idx in enumerate(self.output_binding_indices_in_order):
                    if self.engine.has_implicit_batch_dimension:
                        shape = (batch_size,) + self.output_shapes[i]
                    else:
                        shape = tuple(self.context.get_binding_shape(idx))
                    output = torch.empty(size=shape, dtype=self.output_dtypes[i], device=torch.cuda.current_device())
                    outputs.append(output)
                    bindings[idx] = output.data_ptr()
                for i, idx in enumerate(self.hidden_output_binding_indices_in_order):
                    if self.engine.has_implicit_batch_dimension:
                        shape = (batch_size,) + self.hidden_output_shapes[i]
                    else:
                        shape = tuple(self.context.get_binding_shape(idx))
                    output = torch.empty(size=shape, dtype=self.hidden_output_dtypes[i], device=torch.cuda.current_device())
                    bindings[idx] = output.data_ptr()
            with torch.autograd.profiler.record_function('TRTModule:TensorRTRuntime'):
                if self.engine.has_implicit_batch_dimension:
                    self.context.execute_async(batch_size, bindings, torch.cuda.current_stream().cuda_stream)
                else:
                    self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)

    def enable_profiling(self, profiler: "'trt.IProfiler'"=None):
        """
        Enable TensorRT profiling. After calling this function, TensorRT will report
        time spent on each layer in stdout for each forward run.
        """
        self._check_initialized()
        if not self.context.profiler:
            self.context.profiler = trt.Profiler() if profiler is None else profiler

    def disable_profiling(self):
        """
        Disable TensorRT profiling.
        """
        self._check_initialized()
        torch.cuda.synchronize()
        del self.context
        self.context = self.engine.create_execution_context()

    def get_layer_info(self) ->str:
        """
        Get layer info of the engine. Only support for TRT > 8.2.
        """
        inspector = self.engine.create_engine_inspector()
        return inspector.get_engine_information(trt.LayerInformationFormat.JSON)


class Pool(nn.Module):

    def __init__(self):
        super(Pool, self).__init__()

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (5, 5))


class ModuleFallbackSub(nn.Module):

    def __init__(self):
        super(ModuleFallbackSub, self).__init__()
        self.conv = nn.Conv2d(1, 3, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class ModuleFallbackMain(nn.Module):

    def __init__(self):
        super(ModuleFallbackMain, self).__init__()
        self.layer1 = ModuleFallbackSub()
        self.conv = nn.Conv2d(3, 6, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(self.layer1(x)))


class LoopFallbackEval(nn.Module):

    def __init__(self):
        super(LoopFallbackEval, self).__init__()

    def forward(self, x):
        add_list = torch.empty(0)
        for i in range(x.shape[1]):
            add_list = torch.cat((add_list, torch.tensor([x.shape[1]])), 0)
        return x + add_list


class LoopFallbackNoEval(nn.Module):

    def __init__(self):
        super(LoopFallbackNoEval, self).__init__()

    def forward(self, x):
        for _ in range(x.shape[1]):
            x = x + torch.ones_like(x)
        return x


class FallbackIf(torch.nn.Module):

    def __init__(self):
        super(FallbackIf, self).__init__()
        self.relu1 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.log_sig = torch.nn.LogSigmoid()
        self.conv2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
        x = self.relu1(x)
        x_first = x[0][0][0][0].item()
        if x_first > 0:
            x = self.conv1(x)
            x1 = self.log_sig(x)
            x2 = self.conv2(x)
            x = self.conv3(x1 + x2)
        else:
            x = self.log_sig(x)
        x = self.conv1(x)
        return x


class FallbackInplaceOPIf(nn.Module):

    def __init__(self):
        super(FallbackInplaceOPIf, self).__init__()

    def forward(self, x, y):
        mod_list = [x]
        if x.sum() > y.sum():
            mod_list.append(y)
        z = torch.cat(mod_list)
        return z


class StandardTensorInput(nn.Module):

    def __init__(self):
        super(StandardTensorInput, self).__init__()

    def forward(self, x, y):
        r = x + y
        return r


class TupleInput(nn.Module):

    def __init__(self):
        super(TupleInput, self).__init__()

    def forward(self, z: 'Tuple[torch.Tensor, torch.Tensor]'):
        r = z[0] + z[1]
        return r


class ListInput(nn.Module):

    def __init__(self):
        super(ListInput, self).__init__()

    def forward(self, z: 'List[torch.Tensor]'):
        r = z[0] + z[1]
        return r


class TupleInputOutput(nn.Module):

    def __init__(self):
        super(TupleInputOutput, self).__init__()

    def forward(self, z: 'Tuple[torch.Tensor, torch.Tensor]'):
        r1 = z[0] + z[1]
        r2 = z[0] - z[1]
        r1 = r1 * 10
        r = r1, r2
        return r


class ListInputOutput(nn.Module):

    def __init__(self):
        super(ListInputOutput, self).__init__()

    def forward(self, z: 'List[torch.Tensor]'):
        r1 = z[0] + z[1]
        r2 = z[0] - z[1]
        r = [r1, r2]
        return r


class ListInputTupleOutput(nn.Module):

    def __init__(self):
        super(ListInputTupleOutput, self).__init__()
        self.list_model = ListInputOutput()
        self.tuple_model = TupleInputOutput()

    def forward(self, z: 'List[torch.Tensor]'):
        r1 = z[0] + z[1]
        r2 = z[0] - z[1]
        r3 = r1, r2
        r4 = [r2, r1]
        tuple_out = self.tuple_model(r3)
        list_out = self.list_model(r4)
        r = tuple_out[1], list_out[0]
        return r


class MyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out


class SampleModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(100, 128)
        self.layer2 = torch.nn.Linear(30, 64)
        self.mat1 = torch.randn((128, 32))
        self.mat2 = torch.randn((64, 512))
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv1d(64, 6, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.matmul(out, self.mat1)
        out = self.relu((out + 2.0) * 0.05)
        out = self.conv(out)
        out = self.layer2(out)
        out = torch.matmul(out, self.mat2)
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ConditionalExceptionBoolCondWrapper,
     lambda: ([], {'op': 4}),
     lambda: ([], {})),
    (ConditionalExceptionWrapper,
     lambda: ([], {'exc': torch.nn.ReLU()}),
     lambda: ([0, torch.rand([4, 4, 4, 4])], {})),
    (ConvGelu,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Elu,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FallbackIf,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (FallbackInplaceOPIf,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4, 'multiple_of': 4, 'ffn_dim_multiplier': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GeLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ListInput,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ListInputOutput,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ListInputTupleOutput,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LoopFallbackEval,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LoopFallbackNoEval,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ModuleFallbackMain,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {})),
    (ModuleFallbackSub,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {})),
    (MyModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Norm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Pool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StandardTensorInput,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TestModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 2])], {})),
    (TupleInput,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TupleInputOutput,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

