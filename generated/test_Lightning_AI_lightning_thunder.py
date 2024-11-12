
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


import inspect


import torch


import math


from typing import List


import numpy as np


import collections


from enum import Enum


import functools


import numpy


from torch import nn


from typing import Any


from typing import Optional


from typing import Tuple


import torch.nn.functional as F


from functools import partial


import random


import torch.distributed as dist


import time


from torch.distributed import destroy_process_group


from torch.distributed import init_process_group


from torch.nn.parallel import DistributedDataParallel as DDP


from functools import wraps


from collections import defaultdict


from collections import namedtuple


from collections.abc import Callable


from collections.abc import Sequence


import warnings


import torch as pytorch


from collections import UserDict


from numbers import Number


import torch.multiprocessing as mp


import torch.nn as nn


from torch.testing import make_tensor


from typing import TYPE_CHECKING


from torch.utils.data import DataLoader


from torch.utils.data import IterableDataset


import torch.distributed as torch_dist


from torch.distributed.device_mesh import init_device_mesh


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper


import pandas as pd


from functools import reduce


from typing import Union


import copy


from collections.abc import Generator


from collections.abc import Hashable


from collections import deque


import torch as torch


import collections.abc


from typing import Type


from types import MappingProxyType


from types import ModuleType


from types import CodeType


from types import FunctionType


from types import MethodType


import re


from enum import auto


from collections.abc import Iterable


from typing import Dict


from typing import Literal


from collections.abc import ValuesView


from collections.abc import Iterator


from types import CellType


from types import ClassMethodDescriptorType


from types import CoroutineType


from types import FrameType


from types import MethodDescriptorType


from types import BuiltinFunctionType


from types import BuiltinMethodType


from types import MethodWrapperType


from types import WrapperDescriptorType


from types import TracebackType


from types import GetSetDescriptorType


import torch.utils.checkpoint


import itertools


from torch.utils.weak import WeakTensorKeyDictionary


from itertools import chain


import string


from abc import ABC


from itertools import filterfalse


from typing import overload


from typing import Generic


from typing import TypeVar


from inspect import Parameter


from inspect import Signature


from torch import Tensor


from torch.distributed._tensor import DTensor


from torch.distributed._tensor import Shard


from torch.nn import Module


import torch.distributed


from abc import abstractmethod


import torch.distributed as tdist


from torch.fx.passes.split_module import split_module


from warnings import warn


from functools import lru_cache


from collections import OrderedDict


from typing import Set


from collections.abc import Mapping


from copy import copy


from collections.abc import Collection


from collections.abc import MutableSet


from collections.abc import MutableMapping


from collections.abc import MutableSequence


from torch._subclasses.fake_tensor import FakeTensor


from torch._subclasses.fake_tensor import FakeTensorMode


import enum


from functools import cache


import torch.cuda


from typing import ClassVar


from torch.testing._internal import common_distributed


from torch.testing._internal import common_utils


from torch.distributed import distributed_c10d as c10d


from itertools import product


from torch.testing import assert_close


from torch.distributed.fsdp import FullyShardedDataParallel


from torch.distributed.fsdp.wrap import always_wrap_policy


from functools import singledispatchmethod


from torch._dynamo import is_inductor_supported


from typing import cast


from torch.nn import functional as F


from torch.distributed import is_available


from itertools import islice


from torch.testing._internal.common_device_type import skipCPUIfNoLapack


from torch.testing._internal.common_device_type import skipCUDAIfNoMagma


from torch.testing._internal.common_methods_invocations import op_db


from torch._dynamo.eval_frame import is_inductor_supported


import torch.fx


import torch.testing


import torch._higher_order_ops.wrap


class RMSNorm(torch.nn.Module):

    def __init__(self, dim: 'int', eps: 'float'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_rotary_emb(xq: 'torch.Tensor', xk: 'torch.Tensor', freqs_cos: 'torch.Tensor', freqs_sin: 'torch.Tensor') ->tuple[torch.Tensor, torch.Tensor]:
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
    a, b = freqs_cos.shape
    freqs_cos = freqs_cos.view(1, a, 1, b)
    freqs_sin = freqs_sin.view(1, a, 1, b)
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: 'torch.Tensor', n_rep: 'int') ->torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)


class Attention(nn.Module):

    def __init__(self, args: 'ModelArgs'):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

    def forward(self, x: 'torch.Tensor', freqs_cos: 'torch.Tensor', freqs_sin: 'torch.Tensor'):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):

    def __init__(self, dim: 'int', hidden_dim: 'int', multiple_of: 'int', dropout: 'float'):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):

    def __init__(self, layer_id: 'int', args: 'ModelArgs'):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=args.hidden_dim, multiple_of=args.multiple_of, dropout=args.dropout)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


def precompute_freqs_cis(dim: 'int', end: 'int', theta: 'float'=10000.0):
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


class REDUCTION_OUTPUT_TYPE_KIND(Enum):
    SAME = 0,
    COMPLEX_TO_FLOAT = 1,
    KEEP_PROMOTED_TYPE = 2,
    ALWAYS_BOOL = 3,


class DistParallelType(Enum):
    NONE = auto()
    REPLICATED = auto()
    FULLY_SHARDED = auto()
    COLUMN_WISE = auto()
    ROW_WISE = auto()

