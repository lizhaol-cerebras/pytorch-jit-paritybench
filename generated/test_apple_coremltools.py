
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


import re as _re


import collections


from typing import List


from typing import Optional


from typing import Text


from typing import Union


import warnings as _warnings


from typing import Tuple


import itertools


import numpy as np


import math


from collections import OrderedDict


from enum import Enum


from typing import Dict


import torch as torch


from torch.jit._script import RecursiveScriptModule


import torch


from typing import Any


import torch as _torch


import math as _math


import numbers


import re


from collections.abc import Iterable


import numpy as _np


import torch.nn as nn


import torch.nn.functional as F


from torch._export import capture_pre_autograd_graph


from torch.ao.quantization.quantize_pt2e import convert_pt2e


from torch.ao.quantization.quantize_pt2e import prepare_pt2e


from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e


from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer


from torch.ao.quantization.quantizer.xnnpack_quantizer import get_symmetric_quantization_config


import numpy.testing


import torchvision


from typing import Callable


from collections import defaultdict


import scipy


import functools


from typing import Set


import copy


from copy import deepcopy as _deepcopy


from typing import Optional as _Optional


import numpy as _numpy


from abc import ABC


from abc import abstractmethod


from typing import IO


import logging


from typing import Any as _Any


from typing import Callable as _Callable


from typing import Dict as _Dict


import torch.distributed as _dist


from abc import ABC as _ABC


from abc import abstractmethod as _abstractmethod


from functools import partial as _partial


from typing import Iterable as _Iterable


from typing import Type as _Type


from torch.distributed.fsdp.wrap import ModuleWrapPolicy as _TorchModuleWrapPolicy


from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy as _size_based_auto_wrap_policy


from typing import List as _List


from typing import Tuple as _Tuple


import logging as _logging


import queue as _queue


from typing import Union as _Union


import torch.multiprocessing as _mp


from collections import OrderedDict as _OrderedDict


from typing import IO as _IO


from typing import Type


from typing import Mapping


from typing import NamedTuple


import torch.nn as _nn


from enum import Enum as _Enum


import copy as _copy


from collections import UserDict as _UserDict


import time as _time


from typing import NewType as _NewType


import torch.nn.qat as _nnqat


import torch.nn.functional as _F


from torch.ao.quantization.observer import ObserverBase as _ObserverBase


from torch.quantization import FakeQuantize as _FakeQuantize


from torch.ao.quantization import FakeQuantize as _FakeQuantize


from torch.distributed.fsdp import FullStateDictConfig as _FullStateDictConfig


from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP


from torch.distributed.fsdp import ShardingStrategy as _ShardingStrategy


from torch.distributed.fsdp import StateDictType as _StateDictType


import types as _types


from typing import NamedTuple as _NamedTuple


from typing import cast as _cast


import torch.nn.utils.prune as _prune


import torch.utils.hooks as _hooks


import torch.ao.quantization as _aoquant


from torch.ao.quantization.quantizer.quantizer import QuantizationSpec as _TorchQuantizationSpec


from typing import Set as _Set


import torch.ao.nn.qat as _nnq


import torch.ao.nn.quantized.reference as _nnr


import torch.nn.intrinsic as _nni


import torch.nn.intrinsic.qat as _nniq


from torch.ao.quantization.backend_config import BackendConfig as _BackendConfig


from torch.ao.quantization.backend_config import BackendPatternConfig as _BackendPatternConfig


from torch.ao.quantization.backend_config import DTypeWithConstraints as _DTypeWithConstraints


from torch.ao.quantization.backend_config import DTypeConfig as _DTypeConfig


from torch.ao.quantization.backend_config import ObservationType as _ObservationType


from collections import defaultdict as _defaultdict


import torch.fx as _fx


import torch.nn.intrinsic.qat as _nniqat


from torch.ao.quantization.fx.custom_config import PrepareCustomConfig as _PrepareCustomConfig


from torch.quantization.quantize_fx import prepare_qat_fx as _prepare_qat_fx


from torch.ao.quantization.quantizer.quantizer import Quantizer as _TorchQuantizer


from torch.ao.quantization.quantizer.xnnpack_quantizer import _get_module_name_filter


from torch.fx import Node as _Node


import itertools as _itertools


from torch.ao.quantization.quantizer.quantizer import FixedQParamsQuantizationSpec as _FixedQParamsQuantizationSpec


from torch.ao.quantization.quantizer.quantizer import QuantizationAnnotation as _QuantizationAnnotation


from torch.ao.quantization.quantizer.quantizer import QuantizationSpecBase as _TorchQuantizationSpecBase


from torch.ao.quantization.quantizer.quantizer import SharedQuantizationSpec as _SharedQuantizationSpec


from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import _is_annotated


from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import _mark_nodes_as_annotated


from torch.fx.passes.utils.matcher_with_name_node_map_utils import SubgraphMatcherWithNameNodeMap as _SubgraphMatcherWithNameNodeMap


from torch.fx.passes.utils.source_matcher_utils import get_source_partitions as _get_source_partitions


from torch.ao.nn.quantized.reference.modules.utils import _quantize_and_dequantize_weight_decomposed


from typing import TypeVar as _TypeVar


from torch import Tensor as _Tensor


from torch.ao.nn.intrinsic import _FusedModule


from torch.nn.common_types import _size_1_t


from torch.nn.common_types import _size_2_t


from torch.nn.common_types import _size_3_t


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _triple


import torch.ao.nn.intrinsic as nni


from torch import Tensor


from torch.nn.utils import fuse_conv_bn_weights


import torch.ao.nn.intrinsic as _nni


import torch.ao.nn.qat as _nnqat


import torch.ao.nn.quantized.reference as _reference


from enum import unique as _unique


from torch.ao.quantization.fx.custom_config import ConvertCustomConfig as _ConvertCustomConfig


from torch.ao.quantization.quantize_fx import convert_to_reference_fx as _convert_to_reference_fx


from collections import Counter


import random


from torchvision import datasets


from torchvision import transforms


from torch.ao.quantization import quantization_mappings


import torch.functional as F


import torch.ao.nn.quantized.reference


import torch.ao.quantization


import torch.nn.intrinsic


import torch.nn.intrinsic.qat


import torch.nn.qat


import torch.nn.quantized


from torch.fx import Node


import torch.nn.quantized.modules.utils


import torch.utils.data

