
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


import uuid


from typing import Literal


from typing import Optional


from typing import Tuple


from typing import Union


import torch


import torch.nn as nn


from torch.distributed import launcher as pet


from torch.utils.data.dataset import Dataset


from torch.utils.data.dataset import TensorDataset


from typing import Dict


from torch.utils.data import Dataset


from torch.utils.data import random_split


from torch.utils.data.dataset import Subset


import math


from torch.nn import functional as F


from typing import List


import torch.nn.functional as F


from torch.optim import Adadelta


from torch.optim.lr_scheduler import StepLR


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision import transforms


from typing import Iterator


from torch import distributed as dist


from torch.distributed import launcher


from torch.distributed.optim.apply_optimizer_in_backward import _apply_optimizer_in_backward


import time


from typing import Any


from typing import cast


from typing import Iterable


import torch.distributed as dist


from torch import nn


from torch.distributed.checkpoint import FileSystemReader


from torch.distributed.checkpoint import FileSystemWriter


from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader


from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner


from torch.distributed.checkpoint.default_planner import DefaultSavePlanner


from torch.distributed.checkpoint.metadata import Metadata


from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE


from torch.distributed.checkpoint.metadata import StorageMeta


from torch.utils.tensorboard import SummaryWriter


from functools import partial


from inspect import getmembers


from inspect import isfunction


from typing import Set


from typing import Type


from typing import TypeVar


from torch.distributed import GradBucket


from copy import deepcopy


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


from typing import Mapping


from torch.ao.quantization.pt2e.export_utils import model_is_exported


from torch.utils.data.distributed import DistributedSampler


from torch.optim import Optimizer


from torch.utils.data import TensorDataset


import random


from collections import Counter


from torch.profiler import ProfilerActivity


import numpy as np


from collections import defaultdict


from collections import namedtuple


from typing import Callable


import torch.distributed.launcher as launcher


from torch.distributed import ProcessGroup


import torch.distributed.launcher as pet


from collections import deque


from typing import Deque


from torch.amp.grad_scaler import GradScaler


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.distributed._composable import fully_shard


from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision


import itertools


from random import random


from typing import Protocol


from typing import runtime_checkable


from torch.nn.parallel.distributed import DistributedDataParallel


from torch import Tensor


from torch.utils.data import IterableDataset


import collections


import inspect


from abc import ABCMeta


from abc import abstractmethod


from typing import ContextManager


from typing import Generic


from torch.optim.swa_utils import SWALR


import abc


from torch.distributed import checkpoint as dcp


from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader as Reader


from torch.distributed.checkpoint._fsspec_filesystem import FsspecWriter as Writer


from torch.distributed.checkpoint.planner import LoadPlanner


from torch.distributed.checkpoint.planner import SavePlanner


from torch.distributed.checkpoint.storage import StorageReader


from torch.distributed.checkpoint.storage import StorageWriter


from typing import Generator


from abc import ABC


from torch.profiler import record_function


import re


from enum import Enum


from functools import total_ordering


from typing import Pattern


from itertools import cycle


from typing import MutableMapping


from typing import TYPE_CHECKING


from functools import wraps


from torch import multiprocessing


from torch.distributed.elastic.utils.distributed import get_free_port


from torch.distributed.constants import default_pg_timeout


from functools import reduce


from numbers import Number


from typing import DefaultDict


from torch.utils._python_dispatch import TorchDispatchMode


from torch.utils._pytree import PyTree


from torch.utils._pytree import tree_map


from typing import Sequence


from torch.distributed.fsdp import StateDictType as _StateDictType


from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch as _BackwardPrefetch


from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision as _MixedPrecision


from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy as _ShardingStrategy


from numpy import ndarray


import torch.optim.lr_scheduler


from types import TracebackType


import copy


import warnings


from time import perf_counter


from torch.nn.parameter import UninitializedParameter


from torch.utils._pytree import tree_flatten


from torch.utils.hooks import RemovableHandle


from torch.distributed._composable_state import _get_module_state


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl


from torch.distributed.fsdp._common_utils import _FSDPState


from torch.distributed.fsdp.api import OptimStateDictConfig


from torch.distributed.fsdp.api import StateDictConfig


from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload


from typing import TextIO


from torch.distributed.distributed_c10d import Work


class MultiheadAttentionLayer(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config: 'GPTConfig', dtype: 'torch.dtype'=torch.float32) ->None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, device=config.device, dtype=dtype)
        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.attn = torch.nn.MultiheadAttention(embed_dim=config.n_embd, num_heads=config.n_head, dropout=config.attn_pdrop, batch_first=True, device=config.device, dtype=dtype)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        _, seq_size, _ = x.size()
        y = self.attn(x, x, x, attn_mask=self.mask[0, 0, :seq_size, :seq_size])[0]
        y = self.resid_drop(self.c_proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config: 'GPTConfig') ->None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadAttentionLayer(config)
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(), nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.resid_pdrop))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class EmbeddingStem(nn.Module):

    def __init__(self, config: 'GPTConfig', dtype: 'torch.dtype'=torch.float32) ->None:
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, device=config.device, dtype=dtype)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd, device=config.device, dtype=dtype))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.block_size: 'int' = config.block_size

    def reset_parameters(self) ->None:
        self.tok_emb.reset_parameters()

    def forward(self, idx: 'torch.Tensor') ->torch.Tensor:
        b, t = idx.size()
        assert t <= self.block_size, f'Cannot forward sequence of length {t}, block size is only {self.block_size}'
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        return self.drop(token_embeddings + position_embeddings)


class Net(nn.Module):

    def __init__(self) ->None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class _BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    """
    The only difference between :class:`torch.nn.BatchNorm1d`, :class:`torch.nn.BatchNorm2d`,
    :class:`torch.nn.BatchNorm3d`, etc is this method that is overwritten by the sub-class.
    This method is used when calling forward as a sanity check.
    When using :function:`revert_sync_batchnorm` this sanity check is lost.
    """

    def _check_input_dim(self, input: 'Tensor') ->None:
        return


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (_BatchNormXd,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

