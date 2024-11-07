import sys
_module = sys.modules[__name__]
del sys
benchmark_forward = _module
benchmark_inference = _module
benchmark_training = _module
petals = _module
cli = _module
run_dht = _module
run_server = _module
client = _module
config = _module
from_pretrained = _module
inference_session = _module
lm_head = _module
ptune = _module
remote_forward_backward = _module
remote_generation = _module
remote_sequential = _module
routing = _module
sequence_info = _module
sequence_manager = _module
spending_policy = _module
sequential_autograd = _module
constants = _module
data_structures = _module
dht_utils = _module
models = _module
bloom = _module
block = _module
model = _module
falcon = _module
block = _module
model = _module
llama = _module
block = _module
model = _module
speculative_model = _module
mixtral = _module
block = _module
model = _module
server = _module
backend = _module
block_functions = _module
block_selection = _module
block_utils = _module
from_pretrained = _module
handler = _module
memory_cache = _module
reachability = _module
server = _module
task_pool = _module
task_prioritizer = _module
throughput = _module
utils = _module
asyncio = _module
auto_config = _module
convert_block = _module
cuda_graphs = _module
dht = _module
disk_cache = _module
hf_auth = _module
logging = _module
misc = _module
packaging = _module
peft = _module
ping = _module
random = _module
version = _module
conftest = _module
test_aux_functions = _module
test_block_exact_match = _module
test_cache = _module
test_chained_calls = _module
test_dtype = _module
test_full_model = _module
test_optimized_layers = _module
test_peft = _module
test_priority_pool = _module
test_remote_sequential = _module
test_sequence_manager = _module
test_server_stats = _module
test_speculative_generation = _module
test_tensor_parallel = _module
test_utils = _module

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


from time import perf_counter


import numpy as np


import torch


import logging


import re


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import itertools


import time


import uuid


from typing import AsyncIterator


import torch.nn.functional as F


import torch.utils.checkpoint


from torch import nn


import torch.nn as nn


from typing import Iterable


from typing import Sequence


from typing import Any


from typing import ContextManager


from typing import Dict


from torch import Tensor


from collections import deque


import math


from functools import partial


from collections import Counter


from itertools import chain


from enum import Enum


from typing import AsyncContextManager


import random


import torch.mps


from queue import PriorityQueue


from abc import ABC


from abc import abstractmethod


from torch.utils._pytree import tree_flatten as _tree_flatten


from torch.utils._pytree import tree_unflatten as _tree_unflatten


class LMHead(nn.Module):

    def __init__(self, config: 'PretrainedConfig'):
        super().__init__()
        if not config.tie_word_embeddings:
            self.weight = nn.Parameter(torch.zeros(config.vocab_size, config.hidden_size))
            self.weight.requires_grad = False
        else:
            self.weight = None
        self.bias = None
        self.in_features = config.hidden_size
        self.out_features = config.vocab_size
        self.use_chunked_forward = config.use_chunked_forward
        if self.use_chunked_forward == 'auto':
            if platform.machine() == 'x86_64':
                self.use_chunked_forward = not (CPUFeature['AVX512f'] and CPUFeature['OS_AVX512'])
            else:
                self.use_chunked_forward = True
        self.chunked_forward_step = config.chunked_forward_step
        self._bf16_warning_shown = False

    def forward(self, hidden_states):
        if self.weight.dtype in [torch.float16, torch.bfloat16] and self.weight.device.type == 'cpu' and self.use_chunked_forward:
            lm_logits = self.chunked_forward(hidden_states)
        else:
            hidden_states = hidden_states
            lm_logits = F.linear(hidden_states, self.weight)
        return lm_logits

    def chunked_forward(self, hidden_states):
        """Splits word embeddings on chunks and iteratively casts them into fp32 to perform matmul more efficiently on CPU.
        chunked_forward_step: provides trade-off between efficiency and extra memory consumption.
        """
        assert self.chunked_forward_step > 0, 'Chunk size for chunked forward must be positive'
        if not self._bf16_warning_shown:
            logger.warning('Running the model in bfloat16 on CPU will be slow since your CPU does not support AVX512. To speed it up, load the model in float32 using .from_pretrained(..., torch_dtype=torch.float32)')
            self._bf16_warning_shown = True
        hidden_states = hidden_states.float()
        output = torch.empty(*hidden_states.shape[:-1], self.out_features)
        for i in range(0, self.out_features, self.chunked_forward_step):
            chunk = self.weight[i:i + self.chunked_forward_step].float()
            output[..., i:i + self.chunked_forward_step] = F.linear(hidden_states, chunk)
        return output


CHAIN_DELIMITER = ' '


DUMMY = torch.empty(0)


DUMMY_INT64 = torch.empty(0, dtype=torch.int64)


PUBLIC_INITIAL_PEERS = ['/dns/bootstrap1.petals.dev/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY', '/dns/bootstrap2.petals.dev/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5', '/dns6/bootstrap1.petals.dev/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY', '/dns6/bootstrap2.petals.dev/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5', '/ip4/159.89.214.152/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY', '/ip4/159.203.156.48/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5']


ModuleUID = str


RPCInfo = Dict[str, Any]


class ServerState(Enum):
    OFFLINE = 0
    JOINING = 1
    ONLINE = 2


CACHE_TOKENS_AVAILABLE = 'cache_tokens_available'


class TaskPrioritizerBase(ABC):
    """Abstract class for TaskPrioritizer whose responsibility is to evaluate task priority"""

    @abstractmethod
    def prioritize(self, *input: torch.Tensor, points: float=0.0, **kwargs) ->float:
        """Evaluates task value by the amount of points given, task input and additional kwargs. Lower priority is better"""
        pass


class DummyTaskPrioritizer(TaskPrioritizerBase):

    def prioritize(self, *input: torch.Tensor, points: float=0.0, **kwargs) ->float:
        if kwargs.get('type') == 'inference':
            return 1.0
        return 2.0


class Event(Enum):
    NEW_SESSION = 0
    END_SESSION = 1
    PUSH = 2
    SHUTDOWN = 3


Handle = int


class QuantType(Enum):
    NONE = 0
    INT8 = 1
    NF4 = 2


class AllocationFailed(Exception):
    pass


SPECIAL_DTYPE_SIZES = {torch.bool: 1, torch.qint8: 1, torch.qint32: 4}


def get_size_in_bytes(dtype: 'torch.dtype') ->int:
    if dtype in SPECIAL_DTYPE_SIZES:
        return SPECIAL_DTYPE_SIZES[dtype]
    get_info = torch.finfo if dtype.is_floating_point else torch.iinfo
    return get_info(dtype).bits * (1 + dtype.is_complex) // 8


async def shield_and_wait(task):
    """
    Works like asyncio.shield(), but waits for the task to finish before raising CancelledError to the caller.
    """
    if not isinstance(task, asyncio.Task):
        task = asyncio.create_task(task)
    cancel_exc = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as e:
            cancel_exc = e
    if cancel_exc is not None:
        raise cancel_exc
    return result


def _move_to_device_if_tensor(arg: 'Any', device: 'Union[torch.device, str]', share_memory: 'bool'=False):
    if isinstance(arg, torch.Tensor):
        arg = arg.detach().requires_grad_(arg.requires_grad)
        if share_memory:
            arg = arg.share_memory_()
    return arg


def is_dummy(tensor: 'torch.Tensor') ->bool:
    return tensor.numel() == 0


UID_DELIMITER = '.'


MAX_NF4_SHORT_INFERENCE_TOKENS = 1


MAX_SHORT_INFERENCE_TOKENS = 128


def _get_tensor_index(item: 'bytes') ->int:
    return int(item[3:])


def _is_masked_tensor(item: 'Any') ->bool:
    return isinstance(item, bytes) and item.startswith(b'__T')


def unpack_args_kwargs(flat_tensors: 'List[torch.Tensor]', args_structure: 'Any'):
    """
    Restore arguments after `pack_args_kwargs` function.
    :returns: list of args and dict of kwargs
    """
    return nested_pack((value if not _is_masked_tensor(value) else flat_tensors[_get_tensor_index(value)] for value in nested_flatten(args_structure)), args_structure)


def _mark_masked_tensor(index: 'int') ->bytes:
    return b'__T' + str(index).encode()


def pack_args_kwargs(*args, **kwargs) ->Tuple[List[torch.Tensor], Any]:
    """
    Check the function's arguments and pack all tensors into different flattened lists.
    :returns: a flattened list of tensors and args and kwargs, where tensors were masked
    """
    masked_flat_values, flat_tensors, tensor_to_index = [], [], {}
    for value in nested_flatten((args, kwargs)):
        if isinstance(value, torch.Tensor):
            tensor_index = tensor_to_index.setdefault(value, len(flat_tensors))
            if tensor_index == len(flat_tensors):
                flat_tensors.append(value)
            masked_flat_values.append(_mark_masked_tensor(tensor_index))
        else:
            masked_flat_values.append(value)
    return flat_tensors, nested_pack(masked_flat_values, (args, kwargs))


def maybe_log_traceback(exc: 'Exception'):
    traceback_level = logging.DEBUG if str(exc) or isinstance(exc, asyncio.TimeoutError) else logging.WARNING
    logger.log(traceback_level, 'See detailed traceback below:', exc_info=True)


class MissingBlocksError(RuntimeError):

    def __init__(self, block_indices: 'Union[int, Sequence[int]]'):
        super().__init__(f"No servers holding blocks {block_indices} are online. You can check the public swarm's state at https://health.petals.dev If there are not enough servers, please connect your GPU: https://github.com/bigscience-workshop/petals#connect-your-gpu-and-increase-petals-capacity ")


class SpendingPolicyBase(ABC):

    @abstractmethod
    def get_points(self, protocol: 'str', *args, **kwargs) ->float:
        pass


class NoSpendingPolicy(SpendingPolicyBase):

    def get_points(self, protocol: 'str', *args, **kwargs) ->float:
        return 0.0


def parse_uid(uid: 'ModuleUID') ->Tuple[str, int]:
    assert CHAIN_DELIMITER not in uid, 'parse_uid() does not support chained UIDs'
    dht_prefix, index = uid.split(UID_DELIMITER)
    return dht_prefix, int(index)


MAX_TOKENS_IN_BATCH = 1024


async def _gather_backward(grad_output_batches, intermediate_input_batches, prompt_batches, forward_sequences, sequence_manager):
    """Wrapper for asyncio.gather to perform parallel sequential backwards"""
    return await asyncio.gather(*[sequential_backward((grad_output,), input_batch, prompt_batch, spans, sequence_manager) for grad_output, input_batch, prompt_batch, spans in zip(grad_output_batches, intermediate_input_batches, prompt_batches, forward_sequences)])


async def _gather_forward(input_batches, prompt_batches, sequence_manager):
    """Wrapper for asyncio.gather to perform parallel sequential forwards"""
    return await asyncio.gather(*[sequential_forward(input_batch, prompt_batch, sequence_manager) for input_batch, prompt_batch in zip(input_batches, prompt_batches)])


class _RemoteSequentialAutogradFunction(torch.autograd.Function):
    """
    PyTorch autograd function that provides forward and backward calls for the entire sequence of remote transformer blocks.
    This function splits input data into batches with <MAX_TOKENS_IN_BATCH> and performs efficient parallel processing.
    """

    @staticmethod
    def forward(ctx, inputs: 'torch.Tensor', prompts: 'torch.Tensor', sequence_manager: 'RemoteSequenceManager'):
        batch_size = max(MAX_TOKENS_IN_BATCH // inputs.shape[1], 1)
        input_batches: 'Sequence[torch.Tensor]' = inputs.detach().split(batch_size)
        if prompts is None or is_dummy(prompts):
            prompt_batches = [DUMMY] * len(input_batches)
        else:
            prompt_batches: 'Sequence[torch.Tensor]' = prompts.detach().split(batch_size, dim=1)
        sequence_manager.rpc_info
        outputs = RemoteExpertWorker.run_coroutine(_gather_forward(input_batches, prompt_batches, sequence_manager))
        assert len(outputs) == len(input_batches)
        output_batches = [output[0] for output in outputs]
        intemediate_input_batches = [output[1] for output in outputs]
        sequences_for_batches = [output[2] for output in outputs]
        ctx.prompt_batches = prompt_batches
        ctx.sequence_manager = sequence_manager
        ctx.intemediate_input_batches = intemediate_input_batches
        ctx.sequences_for_batches = sequences_for_batches
        return torch.cat(output_batches, dim=0)

    @staticmethod
    def backward(ctx, grad_outputs: 'torch.Tensor'):
        intermediate_input_batches: 'List[Sequence[torch.Tensor]]' = ctx.intemediate_input_batches
        forward_sequences: 'List[Sequence[RemoteSpanInfo]]' = ctx.sequences_for_batches
        ctx.sequence_manager.rpc_info
        batch_size = max(MAX_TOKENS_IN_BATCH // grad_outputs.shape[1], 1)
        grad_output_batches: 'Sequence[torch.Tensor]' = grad_outputs.split(batch_size)
        assert len(intermediate_input_batches) == len(grad_output_batches) == len(forward_sequences)
        outputs = RemoteExpertWorker.run_coroutine(_gather_backward(grad_output_batches, intermediate_input_batches, ctx.prompt_batches, forward_sequences, ctx.sequence_manager))
        grad_input_batches = [output[0][0] for output in outputs]
        grad_prompt_batches = [output[1] for output in outputs]
        grad_inputs = torch.cat(grad_input_batches, dim=0)
        dummy_grad_prompts = [(grad_prompt is None) for grad_prompt in grad_prompt_batches]
        grad_prompts = torch.cat(grad_prompt_batches, dim=1) if not any(dummy_grad_prompts) else None
        return grad_inputs, grad_prompts, None


INFERENCE_MAX_LENGTH = 8192


def apply_rotary(query, key, cos, sin):
    return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin


class OptimizedFalconRotaryEmbedding(nn.Module):

    def __init__(self, head_dim: 'int', base=10000):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.head_dim = head_dim
        self.seq_len_cached = -1
        self.cuda_graph = None
        self.input_surface = None
        self.static_outputs = None

    def _optimized_apply_rotary(self, query, key, cos, sin):
        if self.cuda_graph is None:
            self.cuda_graph = torch.cuda.CUDAGraph()
            self.input_surface = query, key, cos, sin
            s = torch.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    apply_rotary(*self.input_surface)
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(self.cuda_graph):
                self.static_outputs = apply_rotary(*self.input_surface)
        inputs = query, key, cos, sin
        for static_input, data in zip(self.input_surface, inputs):
            static_input.copy_(data)
        self.cuda_graph.replay()
        return tuple(o.detach() for o in self.static_outputs)

    def cos_sin(self, seq_len: 'int', past_key_values_length: 'int', device='cpu', dtype=torch.bfloat16) ->torch.Tensor:
        total_length = seq_len + past_key_values_length
        if self.seq_len_cached == -1:
            total_length = max(INFERENCE_MAX_LENGTH, total_length)
        if total_length > self.seq_len_cached:
            with torch.inference_mode(False):
                self.seq_len_cached = total_length
                t = torch.arange(total_length, device=device, dtype=self.inv_freq.dtype)
                freqs = torch.einsum('i,j->ij', t, self.inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                if dtype in [torch.float16, torch.bfloat16]:
                    emb = emb.float()
                self.register_buffer('cos_cached', emb.cos()[None, :, :].type(dtype), persistent=False)
                self.register_buffer('sin_cached', emb.sin()[None, :, :].type(dtype), persistent=False)
        return self.cos_cached[:, past_key_values_length:seq_len + past_key_values_length].type(dtype), self.sin_cached[:, past_key_values_length:seq_len + past_key_values_length].type(dtype)

    def forward(self, query, key, past_key_values_length=0):
        batch, seq_len, head_dim = query.shape
        cos, sin = self.cos_sin(seq_len, past_key_values_length, query.device, query.dtype)
        if seq_len == 1 and torch.is_inference_mode_enabled() and query.device.type == 'cuda':
            return self._optimized_apply_rotary(query, key, cos, sin)
        else:
            return apply_rotary(query, key, cos, sin)

