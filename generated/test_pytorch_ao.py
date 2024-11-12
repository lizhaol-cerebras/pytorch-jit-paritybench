
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


import copy


import pandas as pd


import torch.nn.functional as F


import random


import torch.utils.benchmark as benchmark


from torch import nn


from torch.sparse import SparseSemiStructuredTensor


from torch.sparse import to_sparse_semi_structured


from torch.sparse._triton_ops_meta import optimize_bsr_dense_addmm


import math


import time


from functools import partial


from torch.utils.data import DataLoader


from torchvision.transforms import v2


import itertools


from typing import Tuple


from torch.utils import benchmark


from math import log


from copy import deepcopy


from itertools import product


from typing import Callable


from typing import List


from typing import Optional


import torch.nn as nn


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch._inductor.utils import do_bench_using_profiling


from torch.profiler import profile


from torch.profiler import ProfilerActivity


from torch.profiler import record_function


import collections


import re


import numpy as np


from torch.utils.checkpoint import checkpoint


import matplotlib.pyplot as plt


from torch._inductor import config as inductorconfig


import logging


import torch._dynamo.config


import torch._inductor.config


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import IS_WINDOWS


from torch.testing._internal.common_utils import TestCase


from torch.testing._internal.common_utils import run_tests


from torch.testing._internal import common_utils


from torch._inductor.test_case import TestCase as InductorTestCase


from torch.distributed._tensor import DTensor


from torch.distributed._tensor import Replicate


from torch.distributed._tensor import Shard


from torch.distributed._tensor import DeviceMesh


from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase


from torch.testing._internal.distributed._tensor.common_dtensor import with_comms


from torch.testing._internal.distributed._tensor.common_dtensor import NUM_DEVICES


from torch.utils._triton import has_triton


from torch.testing._internal.common_utils import instantiate_parametrized_tests


from torch.testing._internal.common_utils import parametrize


from collections import OrderedDict


from typing import Union


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing


from torch.distributed.fsdp.wrap import ModuleWrapPolicy


from torch.testing._internal.common_distributed import skip_if_lt_x_gpu


from torch.testing._internal.common_fsdp import FSDPTest


from torch.ao.quantization.quantize_pt2e import prepare_pt2e


from torch.ao.quantization.quantize_pt2e import convert_pt2e


from torch.ao.quantization.quantizer import QuantizationSpec


from torch.ao.quantization.quantizer import Quantizer


from torch.testing._internal.common_quantization import NodeSpec as ns


from torch.testing._internal.common_quantization import QuantizationTestCase


from torch.ao.quantization.observer import ObserverBase


from torch.fx import Node


from torch.fx import GraphModule


from torch.ao.quantization.quantizer import QuantizationAnnotation


import warnings


from torch._dynamo.test_case import TestCase as DynamoTestCase


from torch._dynamo.testing import CompileCounterWithBackend


from torch.distributed._tensor import distribute_tensor


from torch.distributed.device_mesh import DeviceMesh


from torch.distributed.device_mesh import init_device_mesh


from torch.distributed.tensor.parallel import parallelize_module


from torch.testing._internal.distributed._tensor.common_dtensor import ModelArgs


from torch.testing._internal.distributed._tensor.common_dtensor import Transformer


from torch.distributed.fsdp import FullStateDictConfig


from torch.distributed.fsdp import StateDictType


from typing import Any


import torch._dynamo.testing


from torch.distributed._composable.fsdp import fully_shard


from torch.distributed._composable.fsdp import MixedPrecisionPolicy


from torch.distributed._tensor import init_device_mesh


from torch.testing._internal.common_cuda import TEST_CUDA


from torch.testing._internal.common_fsdp import FSDPTestMultiThread


from torch.testing._internal.common_fsdp import MLP


from torch.testing._internal.common_fsdp import patch_all_gather


from torch.testing._internal.distributed._tensor.common_dtensor import TransformerBlock


import torch.utils.data


from torch._inductor.utils import run_and_get_code


from torch._dynamo import config


from torch.ao.quantization import MinMaxObserver


from torch.ao.quantization import QConfigMapping


from torch.ao.quantization.quantize_fx import convert_to_reference_fx


from torch.ao.quantization.quantize_fx import prepare_fx


from torch.nn.utils import parametrize


from torch.nn.utils.parametrize import is_parametrized


from torch.testing._internal.common_pruning import ImplementedSparsifier


from torch.testing._internal.common_pruning import MockSparseLinear


from torch.testing._internal.common_pruning import SimpleLinear


from torch.testing._internal.common_quantization import ConvBnReLUModel


from torch.testing._internal.common_quantization import ConvModel


from torch.testing._internal.common_quantization import FunctionalLinear


from torch.testing._internal.common_quantization import LinearAddModel


from torch.testing._internal.common_quantization import ManualEmbeddingBagLinear


from torch.testing._internal.common_quantization import SingleLayerLinearModel


from torch.testing._internal.common_quantization import TwoLayerLinearModel


from torch.testing._internal.common_pruning import Conv2dActivation


from torch.testing._internal.common_pruning import Conv2dBias


from torch.testing._internal.common_pruning import Conv2dPadBias


from torch.testing._internal.common_pruning import Conv2dPool


from torch.testing._internal.common_pruning import Conv2dPoolFlatten


from torch.testing._internal.common_pruning import Conv2dPoolFlattenFunctional


from torch.testing._internal.common_pruning import LinearActivation


from torch.testing._internal.common_pruning import LinearActivationFunctional


from torch.testing._internal.common_pruning import LinearBias


from torch.testing._internal.common_pruning import LSTMLayerNormLinearModel


from torch.testing._internal.common_pruning import LSTMLinearModel


from torch.testing._internal.common_pruning import rows_are_subset


from torch.testing._internal.common_pruning import SimpleConv2d


from torch.testing._internal.common_utils import skipIfTorchDynamo


from torch.ao.quantization.observer import MinMaxObserver


from torch.ao.quantization.observer import PerChannelMinMaxObserver


from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib


from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer


from torch.ao.quantization.quantizer.xnnpack_quantizer import get_symmetric_quantization_config


from torch.ao.pruning import FakeSparsity


from torch.testing._internal.optests import opcheck


from torch import Tensor


from torch.nn import functional as F


from torch.nn.attention import SDPBackend


from scipy import ndimage


from typing import Dict


from torchvision.ops.boxes import batched_nms


from torchvision.ops.boxes import box_area


from typing import Type


import torch.distributed


from torch.nn.init import trunc_normal_


from typing import Generator


from typing import ItemsView


from torchvision.transforms import Normalize


from torchvision.transforms import Resize


from torchvision.transforms import ToTensor


from torch.utils._python_dispatch import is_traceable_wrapper_subclass


from torch.utils._python_dispatch import return_and_correct_aliasing


from functools import reduce


import functools


from enum import Enum


from enum import auto


from torch._prims_common import make_contiguous_strides_for


import torch._prims_common as utils


import torch.utils._pytree as pytree


from torch.library import Library


from torch.library import impl


from torch.ao.quantization.fx._decomposed import dequantize_per_channel_group


from torch.ao.quantization.fx._decomposed import quantize_per_channel_group


import enum


import torch.utils.checkpoint as checkpoint


from torch.distributed._functional_collectives import all_reduce


from torch.distributed._functional_collectives import AsyncCollectiveTensor


from torch.utils._pytree import tree_map


from typing import NamedTuple


import torch.distributed._functional_collectives as funcol


from torch.distributed.tensor.parallel import ColwiseParallel


from torch.distributed.tensor.parallel import PrepareModuleInput


from torch.distributed.tensor.parallel import RowwiseParallel


from typing import Iterable


from typing import Literal


from torch._prims_common import suggest_memory_format


import inspect


from collections import defaultdict


from torch.utils.flop_counter import FlopCounterMode


from torch.utils._pytree import tree_flatten


from torch.utils._pytree import tree_unflatten


import types


import torch.autograd.profiler_util


from torch.autograd.profiler import record_function


from torch.cuda.nvtx import range as nvtx_range


from enum import unique


from torch.optim import Optimizer


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import ParamsT


from torch.nn.attention import sdpa_kernel


from torch.autograd.functional import hvp


import torchvision.models as models


from torch.autograd.functional import vhp


from itertools import chain


from typing import Set


from torch.fx import symbolic_trace


from typing import cast


from torch.ao.quantization.utils import MatchAllNode


from torch.nn.utils.parametrize import ParametrizationList


from functools import wraps


import abc


from torch.nn.utils.parametrize import type_before_parametrizations


import torchvision


from torch.sparse._triton_ops_meta import dump as store_tuned_kernel_params


from torch.sparse._triton_ops import broadcast_batch_dims


from torch.sparse._triton_ops import bsr_dense_addmm


from torch.sparse._triton_ops import bsr_dense_mm


from torch.autograd import Variable


from torch.utils.data.dataloader import default_collate


from torchvision.transforms.functional import InterpolationMode


from collections import deque


from torchvision.transforms import autoaugment


from torchvision.transforms import functional as F


from torchvision.transforms import transforms


import typing


import torch.fx as fx


from typing import Sequence


from abc import ABCMeta


from abc import abstractmethod


import torch.nn.utils.parametrize as parametrize


from abc import ABC


from torch.utils._python_dispatch import TorchDispatchMode


from torch.ao.pruning import WeightNormSparsifier


from torch.ao.quantization.observer import UniformQuantizationObserverBase


from torch.ao.pruning import BaseSparsifier


from torch.ao.quantization import default_placeholder_observer


from torch.ao.quantization import QConfig


from torch.ao.quantization.quantize import _remove_qconfig


from math import gcd


from torch.overrides import TorchFunctionMode


from torchvision import models


from torch.distributed import DeviceMesh


from torch.distributed.tensor import DTensor


from torch.distributed.tensor import Replicate


from torch.distributed.tensor import Shard


from torch.distributed.tensor import Placement


from math import inf


from math import nan


from torch._inductor.hooks import run_intermediate_hooks


from torch._inductor.utils import maybe_profile


from torch._inductor.codegen.memory_planning import _align as align


from torch import device


from torch import empty_strided


from torch._inductor.async_compile import AsyncCompile


from torch._inductor.select_algorithm import extern_kernels


from torch._inductor.codegen.multi_kernel import MultiKernelCall


from torch._C import _cuda_getCurrentRawStream as get_raw_stream


import torch._inductor.kernel.mm_common


class ToyLinearModel(torch.nn.Module):

    def __init__(self, m=64, n=32, k=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, k, bias=False)
        self.linear2 = torch.nn.Linear(k, n, bias=False)

    def example_inputs(self, batch_size=1, dtype=torch.float32, device='cpu'):
        return torch.randn(batch_size, self.linear1.in_features, dtype=dtype, device=device),

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class LinearTest(torch.nn.Module):

    def __init__(self, mkn):
        super().__init__()
        m, k, n = mkn
        self.model = torch.nn.Linear(k, n).half()
        self.input = torch.randn([m, k], device='cuda', dtype=torch.half, requires_grad=True)
        self.grad = torch.randn([m, n], device='cuda', dtype=torch.half)

    def fw(self):
        self.out = self.model(self.input)

    def bw(self):
        self.out.backward(self.grad, retain_graph=True)


class SemiSparseLinearOfflineCompressionTest(torch.nn.Module):

    def __init__(self, mkn):
        super().__init__()
        m, k, n = mkn
        self.model = torch.nn.Linear(k, n).half()
        self.model.weight = torch.nn.Parameter(to_sparse_semi_structured(self.model.weight))
        self.input = torch.randn([m, k], device='cuda', dtype=torch.half, requires_grad=True)
        self.grad = torch.randn([m, n], device='cuda', dtype=torch.half)

    def fw(self):
        self.out = self.model(self.input)


class _SparsifyFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: 'torch.Tensor', algo: 'str', backend: 'GRADIENT_TYPE'):
        use_cutlass = backend == 'cutlass'
        if not isinstance(x, SparseSemiStructuredTensor):
            packed, meta, packed_t, meta_t, bitmask = torch._sparse_semi_structured_tile(x, algorithm=algo, use_cutlass=use_cutlass)
            cls = SparseSemiStructuredTensorCUTLASS if use_cutlass else SparseSemiStructuredTensorCUSPARSELT
            out = cls(x.shape, packed=packed, meta=meta, packed_t=packed_t, meta_t=meta_t, compressed_swizzled_bitmask=bitmask, requires_grad=False, fuse_transpose_cusparselt=True)
        else:
            out = x.detach()
        return out

    @staticmethod
    def backward(ctx, grad_out: 'torch.Tensor'):
        return grad_out, None, None


@torch._dynamo.allow_in_graph
def semi_structured_sparsify(x: 'torch.Tensor', algo: 'str'='', backend: 'str'='cutlass') ->SparseSemiStructuredTensor:
    """
    Sparsifies a dense tensor into a semi-structured tensor, according to the algo and backend passed.
    """
    return _SparsifyFunc.apply(x, algo, backend)


class SemiSparseLinear(torch.nn.Linear):
    """
    Replacement nn.Linear that supports runtime weight sparsity
    """

    def forward(self, x):
        sparse_weight = semi_structured_sparsify(self.weight, backend='cusparselt')
        return torch.nn.functional.linear(x, sparse_weight, self.bias)

    @classmethod
    def from_dense(cls, linear):
        mod = cls(linear.in_features, linear.out_features)
        mod.weight = linear.weight
        mod.bias = linear.bias
        return mod

    @classmethod
    def to_dense(cls, semi_sparse_linear):
        mod = torch.nn.Linear(semi_sparse_linear.in_features, semi_sparse_linear.out_features)
        mod.weight = semi_sparse_linear.weight
        mod.bias = semi_sparse_linear.bias
        return mod


class SemiSparseLinearTest(LinearTest):

    def __init__(self, mkn):
        super().__init__(mkn)
        self.model = SemiSparseLinear.from_dense(self.model)


class SemiSparseKernelTest(LinearTest):

    def __init__(self, mkn):
        super().__init__(mkn)

    def fw(self):
        self.out = semi_structured_sparsify(self.input)

    def bw(self):
        pass


class SAMTest(torch.nn.Module):

    def __init__(self, model_type, batch_size):
        super().__init__()
        self.model = sam_model_registry[model_type]().image_encoder.half().train()
        self.input = torch.randn(batch_size, 3, 1024, 1024, device='cuda', dtype=torch.half, requires_grad=True)
        self.grad = torch.randn([batch_size, 256, 64, 64], device='cuda', dtype=torch.half)

    def fw(self):
        self.out = self.model(self.input)

    def bw(self):
        self.out.backward(self.grad, retain_graph=True)


def swap_linear_with_semi_sparse_linear(model, config, current=''):
    """
    Public API for replacing nn.Linear with SemiSparseLinear
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        fqn = f'{current}.{name}' if current else name
        if isinstance(child, torch.nn.Linear):
            if fqn in config:
                setattr(model, name, config[fqn].from_dense(child))
                del child
        else:
            swap_linear_with_semi_sparse_linear(child, config, current=fqn)


class SAM_W24_MLP_ONLY(SAMTest):

    def __init__(self, model_type, batch_size):
        super().__init__(model_type, batch_size)
        sparse_config = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Linear) and 'mlp' in name:
                sparse_config[name] = SemiSparseLinear
        swap_linear_with_semi_sparse_linear(self.model, sparse_config)


class SAM_W24_ALL(SAMTest):

    def __init__(self, model_type, batch_size):
        super().__init__(model_type, batch_size)
        sparse_config = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                sparse_config[name] = SemiSparseLinear
        swap_linear_with_semi_sparse_linear(self.model, sparse_config)


class Linear16(torch.nn.Module):

    def __init__(self, scale, device):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(scale * 2, scale, bias=False, dtype=torch.float16, device=device), torch.nn.Linear(scale, scale, bias=False, dtype=torch.float16, device=device), torch.nn.Linear(scale, scale // 2, bias=False, dtype=torch.float16, device=device))

    def forward(self, x):
        return self.net(x)


class LNLinearSigmoid(torch.nn.Module):

    def __init__(self, fc_dim1, fc_dim2):
        super().__init__()
        self.ln = torch.nn.LayerNorm(fc_dim1, elementwise_affine=False)
        self.fc = torch.nn.Linear(fc_dim1, fc_dim2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.ln(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class LNLinear(torch.nn.Module):

    def __init__(self, fc_dim1, fc_dim2):
        super().__init__()
        self.ln = torch.nn.LayerNorm(fc_dim1, elementwise_affine=False)
        self.fc = torch.nn.Linear(fc_dim1, fc_dim2, bias=False)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc(x)
        return x


class RMSNorm(nn.Module):

    def __init__(self, dim: 'int', eps: 'float'=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: 'Tensor') ->Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):

    def __init__(self, config: 'ModelArgs') ->None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: 'Tensor') ->Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class NormFFNResidualNorm(nn.Module):
    """
    A fragment representing the end of TransformerBlock n and the start
    of TransformerBlock n + 1, intended to include the fusions relevant
    to float8 gemms in the FFN module in forward and backward.
    """

    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super().__init__()
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
        self.attn_norm = RMSNorm(dim)

    def forward(self, h):
        x = self.ffn_norm(h)
        x = self.ffn(x)
        x = h + x
        x = self.attn_norm(x)
        return x


class LinearNF4(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: 'torch.Tensor', weight: 'NF4Tensor'):
        """Save the quantized nf4 weight for backward pass"""
        ctx.save_for_backward(weight)
        return F.linear(input, weight)

    @staticmethod
    def backward(ctx, grad_output):
        """The nf4 weight will never require grad so we can just return the grad_output @ weight.to(grad_output.dtype)"""
        weight: 'NF4Tensor' = ctx.saved_tensors[0]
        return grad_output @ weight, None


def linear_nf4(input: 'torch.Tensor', weight: 'NF4Tensor') ->torch.Tensor:
    """Apply a linear operation with the NF4Tensor weight

    Args:
        input: Input tensor
        weight: NF4Tensor weight
    """
    return LinearNF4.apply(input, weight)


CHUNK_SIZE = 1024 ** 2


NF4_TORCH_FUNCTIONS = {}


def get_block_absmax(input_tensor: 'torch.Tensor', block_size: 'int') ->torch.Tensor:
    """Iterate through a flattened tensor getting the absmax scalers for each block

    Args:
        input_tensor: Input tensor to get scalers for
        block_size: Block size for the scanning window
    Returns:
        torch.Tensor: Tensor of scalers for each block
    """
    assert input_tensor.dim() == 1, 'Input tensor must be flattened'
    assert input_tensor.numel() % block_size == 0, f'Input tensor must be divisible by block size, got {input_tensor.numel()} and {block_size}'
    n_blocks = input_tensor.numel() // block_size
    blocks = input_tensor.view(n_blocks, block_size)
    block_scalers = blocks.abs().max(dim=1).values
    return block_scalers

