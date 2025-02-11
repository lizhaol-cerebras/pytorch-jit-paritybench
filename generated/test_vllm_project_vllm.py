
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


from typing import List


from typing import Optional


import numpy as np


import torch


import random


import copy


import itertools


from typing import Callable


from typing import Iterable


from typing import Tuple


import torch.utils.benchmark as TBenchmark


from torch.utils.benchmark import Measurement as TMeasurement


import torch.nn.functional as F


import math


from itertools import product


import pandas as pd


import torch.utils.benchmark as benchmark


from typing import Any


from typing import Dict


from typing import TypedDict


from itertools import accumulate


import re


from collections import defaultdict


import matplotlib.pyplot as plt


from torch.utils.hipify.hipify_python import hipify


from collections import namedtuple


from collections.abc import Iterable


from typing import Union


import logging


from torch.utils.data import DataLoader


import inspect


from torch.utils.cpp_extension import CUDA_HOME


import uuid


from copy import copy


from copy import deepcopy


from torch import nn


from torch.library import Library


from collections import UserList


from enum import Enum


from typing import Type


from typing import TypeVar


import torch.nn as nn


from collections import deque


from typing import Set


from torch import Use


import torch.distributed as dist


import torch.distributed


from typing import NamedTuple


from numbers import Number


from typing import Sequence


from torch._prims_common import TensorLikeType


from collections import OrderedDict


import types


import warnings


import torch.cuda


from typing import Mapping


from types import SimpleNamespace


from itertools import count


from typing import Sequence as GenericSequence


import functools


from typing import TYPE_CHECKING


import torch.library


from abc import ABC


from abc import abstractmethod


from typing import Hashable


from typing import Literal


from enum import auto


from typing import Generic


from torch.nn.functional import scaled_dot_product_attention


from functools import lru_cache


import enum


from typing import Generator


import torch.fx as fx


from torch._higher_order_ops.auto_functionalize import auto_functionalized


from torch._inductor.pattern_matcher import Match


from torch._inductor.pattern_matcher import PatternMatcherPass


from torch._inductor.pattern_matcher import fwd_only


from torch._inductor.pattern_matcher import register_replacement


import torch.fx


from torch import SymInt


from types import CodeType


from typing import ClassVar


from typing import Final


from torch.distributed import ProcessGroup


import torch.multiprocessing as mp


from torch.distributed import ReduceOp


from torch.distributed import Backend


from torch.distributed.distributed_c10d import Backend


from torch.distributed.distributed_c10d import PrefixStore


from torch.distributed.distributed_c10d import _get_default_timeout


from torch.distributed.distributed_c10d import is_nccl_available


from torch.distributed.rendezvous import rendezvous


from typing import cast


from typing import get_args


from functools import partial


from typing import AsyncGenerator


from typing import Coroutine


from typing import overload


from collections import Counter as collectionsCounter


from typing import Deque


from typing import FrozenSet


from typing import Awaitable


from collections import UserDict


from typing import Protocol


import torch.types


from typing import DefaultDict


from torch.nn.parameter import Parameter


from torch.nn.parameter import UninitializedParameter


from enum import IntEnum


from torch.nn import Parameter


from torch.nn import Module


import numpy


from functools import cached_property


import torch.jit


from torch.nn.init import trunc_normal_


from math import inf


from typing import Iterator


import collections


from typing import BinaryIO


from torch.nn import LayerNorm


import torch.utils.checkpoint


from typing import runtime_checkable


import torchvision.transforms as T


from torch.nn import functional as F


from itertools import tee


from torchvision import transforms


from torchvision.transforms import InterpolationMode


from torch.func import functional_call


from typing import final


from functools import wraps


from torch._C._autograd import DeviceType


from torch._C._autograd import _KinetoEvent


from torch._C._autograd import _ProfilerResult


from torch._C._profiler import _EventType


from torch._C._profiler import _ExperimentalConfig


from torch._C._profiler import _ProfilerEvent


from torch.autograd.profiler import FunctionEvent


from torch.profiler import ProfilerActivity


from torch.profiler import profile


from torch._C._profiler import _TensorMetadata


from functools import reduce


from itertools import chain


from uuid import uuid4


from collections.abc import Mapping


from typing import OrderedDict


import numpy.typing as npt


from torch import is_tensor


class CompilationLevel:
    NO_COMPILATION = 0
    DYNAMO_AS_IS = 1
    DYNAMO_ONCE = 2
    PIECEWISE = 3


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_torch_compile_backend() ->Optional[Union[Callable, str]]:
    return _torch_compile_backend

