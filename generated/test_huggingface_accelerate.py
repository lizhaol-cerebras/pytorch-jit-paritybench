
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


from torch.nn.parallel import DistributedDataParallel as DDP


from functools import partial


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.distributed.fsdp import MixedPrecision


from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


from torch.optim import AdamW


from torch.utils.data import DataLoader


from typing import List


import numpy as np


from sklearn.model_selection import StratifiedKFold


import logging


import math


import random


from itertools import chain


from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig


from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig


import re


from torch.optim.lr_scheduler import OneCycleLR


from torch.utils.data import Dataset


from torchvision.transforms import Compose


from torchvision.transforms import RandomResizedCrop


from torchvision.transforms import Resize


from torchvision.transforms import ToTensor


import queue


from typing import Union


import scipy.io.wavfile


import functools


import warnings


from collections import OrderedDict


from types import MethodType


from typing import Any


from typing import Callable


import torch.utils.hooks as hooks


from torch.distributed.algorithms.join import Join


from functools import wraps


from typing import Dict


from typing import Optional


import torch.nn as nn


from torch.cuda.amp import GradScaler


from torch.utils.data import BatchSampler


from torch.utils.data import IterableDataset


from torch.utils.data import RandomSampler


from typing import Mapping


from typing import Tuple


import inspect


from copy import deepcopy


import torch.distributed


from torch.utils.data import TensorDataset


from torch.utils.data import default_collate


from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy


from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


from torch.distributed.elastic.multiprocessing.errors import ChildFailedError


import torch.nn.functional as F


from torch.optim.lr_scheduler import LambdaLR


import copy


import enum


from typing import Iterable


from typing import Literal


from typing import get_args


from functools import lru_cache


from collections import defaultdict


from abc import ABC


from torch.nn import BCEWithLogitsLoss


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP


from typing import Set


from collections.abc import Mapping


from functools import update_wrapper


import collections


from functools import reduce


from typing import OrderedDict


import itertools


from torch.utils.data import SequentialSampler


from torch.fx import symbolic_trace


from torch import nn


import uuid


from collections import UserDict


from collections import namedtuple


from typing import NamedTuple


class NoiseModel(torch.nn.Module):

    def __init__(self, noise_factor=0.1):
        super().__init__()
        self.noise_factor = torch.nn.Parameter(torch.tensor(noise_factor, dtype=torch.float32))

    def forward(self, loss):
        return loss * self.noise_factor


class MockModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.p = torch.nn.Parameter(torch.randn(40, 20))

    def forward(self, x, rank):
        return self.p * x ** (1 + rank)


class TinyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 16)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 16)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class RegressionModel4XPU(torch.nn.Module):

    def __init__(self, a=0, b=0, double_output=False):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([2, 3]).float())
        self.b = torch.nn.Parameter(torch.tensor([2, 3]).float())
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            None
            self.first_batch = False
        return x * self.a[0] + self.b[0]


class RegressionModel(torch.nn.Module):

    def __init__(self, a=0, b=0, double_output=False):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a).float())
        self.b = torch.nn.Parameter(torch.tensor(b).float())
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            None
            self.first_batch = False
        return x * self.a + self.b


class AbstractTrainStep(ABC):
    """Abstract class for batching, forward pass and loss handler."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def get_batch_func(self, accelerator, megatron_dataset_flag):
        pass

    def get_forward_step_func(self):
        pass

    def get_loss_func(self, accelerator):
        pass


def is_namedtuple(data):
    """
    Checks if `data` is a `namedtuple` or not. Can have false positives, but only if a user is trying to mimic a
    `namedtuple` perfectly.
    """
    return isinstance(data, tuple) and hasattr(data, '_asdict') and hasattr(data, '_fields')


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple, or namedtuple)
    """
    if is_namedtuple(obj):
        return type(obj)(*list(generator))
    else:
        return type(obj)(generator)


@lru_cache
def is_npu_available(check_device=False):
    """Checks if `torch_npu` is installed and potentially if a NPU is in the environment"""
    if importlib.util.find_spec('torch_npu') is None:
        return False
    if check_device:
        try:
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, 'npu') and torch.npu.is_available()


def is_torch_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def is_ipex_available():
    """Checks if ipex is installed."""

    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + '.' + str(version.parse(full_version).minor)
    _torch_version = importlib.metadata.version('torch')
    if importlib.util.find_spec('intel_extension_for_pytorch') is None:
        return False
    _ipex_version = 'N/A'
    try:
        _ipex_version = importlib.metadata.version('intel_extension_for_pytorch')
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(f'Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*, but PyTorch {_torch_version} is found. Please switch to the matching version and run again.')
        return False
    return True


STR_OPERATION_TO_FUNC = {'>': op.gt, '>=': op.ge, '==': op.eq, '!=': op.ne, '<=': op.le, '<': op.lt}


def compare_versions(library_or_version: 'Union[str, Version]', operation: 'str', requirement_version: 'str'):
    """
    Compares a library version to some requirement using a given operation.

    Args:
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f'`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}')
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib.metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


def is_torch_version(operation: 'str', version: 'str'):
    """
    Compares the current PyTorch version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of PyTorch
    """
    return compare_versions(torch_version, operation, version)


def str_to_bool(value) ->int:
    """
    Converts a string representation of truth to `True` (1) or `False` (0).

    True values are `y`, `yes`, `t`, `true`, `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
    """
    value = value.lower()
    if value in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif value in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError(f'invalid truth value {value}')


def parse_flag_from_env(key, default=False):
    """Returns truthy value for `key` from the env if available else the default."""
    value = os.environ.get(key, str(default))
    return str_to_bool(value) == 1


@lru_cache
def is_xpu_available(check_device=False):
    """
    Checks if XPU acceleration is available either via `intel_extension_for_pytorch` or via stock PyTorch (>=2.4) and
    potentially if a XPU is in the environment
    """
    """check if user disables it explicitly"""
    if not parse_flag_from_env('ACCELERATE_USE_XPU', default=True):
        return False
    if is_ipex_available():
        if is_torch_version('<=', '1.12'):
            return False
    elif is_torch_version('<=', '2.3'):
        return False
    if check_device:
        try:
            _ = torch.xpu.device_count()
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, 'xpu') and torch.xpu.is_available()


def send_to_device(tensor, device, non_blocking=False, skip_keys=None):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to send to a given device.
        device (`torch.device`):
            The device to send the data to.

    Returns:
        The same data structure as `tensor` with all tensors sent to the proper device.
    """
    if is_torch_tensor(tensor) or hasattr(tensor, 'to'):
        if device == 'npu':
            device = 'npu:0'
        if device == 'xpu':
            device = 'xpu:0'
        try:
            return tensor
        except TypeError:
            return tensor
        except AssertionError as error:
            if is_npu_available():
                if isinstance(device, int):
                    device = f'npu:{device}'
            elif is_xpu_available():
                if isinstance(device, int):
                    device = f'xpu:{device}'
            else:
                raise error
        try:
            return tensor
        except TypeError:
            return tensor
    elif isinstance(tensor, (tuple, list)):
        return honor_type(tensor, (send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys) for t in tensor))
    elif isinstance(tensor, Mapping):
        if isinstance(skip_keys, str):
            skip_keys = [skip_keys]
        elif skip_keys is None:
            skip_keys = []
        return type(tensor)({k: (t if k in skip_keys else send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys)) for k, t in tensor.items()})
    else:
        return tensor


class BertTrainStep(AbstractTrainStep):
    """
    Bert train step class.

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, accelerator, args):
        super().__init__('BertTrainStep')
        self.get_batch = self.get_batch_func(accelerator, args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func(accelerator, args.pretraining_flag, args.num_labels)
        self.forward_step = self.get_forward_step_func(args.pretraining_flag, args.bert_binary_head)
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            self.model_output_class = SequenceClassifierOutput

    def get_batch_func(self, accelerator, megatron_dataset_flag):

        def get_batch_megatron(data_iterator):
            """Build the batch."""
            keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask']
            datatype = torch.int64
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = tensor_parallel.broadcast_data(keys, data, datatype)
            tokens = data_b['text'].long()
            types = data_b['types'].long()
            sentence_order = data_b['is_random'].long()
            loss_mask = data_b['loss_mask'].float()
            lm_labels = data_b['labels'].long()
            padding_mask = data_b['padding_mask'].long()
            return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask

        def get_batch_transformer(data_iterator):
            """Build the batch."""
            data = next(data_iterator)
            data = send_to_device(data, torch.cuda.current_device())
            tokens = data['input_ids'].long()
            padding_mask = data['attention_mask'].long()
            if 'token_type_ids' in data:
                types = data['token_type_ids'].long()
            else:
                types = None
            if 'labels' in data:
                lm_labels = data['labels'].long()
                loss_mask = data['labels'] != -100
            else:
                lm_labels = None
                loss_mask = None
            if 'next_sentence_label' in data:
                sentence_order = data['next_sentence_label'].long()
            else:
                sentence_order = None
            return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask
        if accelerator.state.megatron_lm_plugin.custom_get_batch_function is not None:
            return accelerator.state.megatron_lm_plugin.custom_get_batch_function
        if megatron_dataset_flag:
            try:
                return get_batch
            except ImportError:
                pass
            return get_batch_megatron
        else:
            return get_batch_transformer

    def get_loss_func(self, accelerator, pretraining_flag, num_labels):

        def loss_func_pretrain(loss_mask, sentence_order, output_tensor):
            lm_loss_, sop_logits = output_tensor
            lm_loss_ = lm_loss_.float()
            loss_mask = loss_mask.float()
            lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
            if sop_logits is not None:
                sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(), sentence_order.view(-1), ignore_index=-1)
                sop_loss = sop_loss.float()
                loss = lm_loss + sop_loss
                averaged_losses = average_losses_across_data_parallel_group([lm_loss, sop_loss])
                return loss, {'lm loss': averaged_losses[0], 'sop loss': averaged_losses[1]}
            else:
                loss = lm_loss
                averaged_losses = average_losses_across_data_parallel_group([lm_loss])
                return loss, {'lm loss': averaged_losses[0]}

        def loss_func_finetune(labels, logits):
            if num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            averaged_losses = average_losses_across_data_parallel_group([loss])
            return loss, {'loss': averaged_losses[0]}
        if accelerator.state.megatron_lm_plugin.custom_loss_function is not None:
            return accelerator.state.megatron_lm_plugin.custom_loss_function
        if pretraining_flag:
            return loss_func_pretrain
        else:
            return loss_func_finetune

    def get_forward_step_func(self, pretraining_flag, bert_binary_head):

        def forward_step(data_iterator, model):
            """Forward step."""
            tokens, types, sentence_order, loss_mask, labels, padding_mask = self.get_batch(data_iterator)
            if not bert_binary_head:
                types = None
            if pretraining_flag:
                output_tensor = model(tokens, padding_mask, tokentype_ids=types, lm_labels=labels)
                return output_tensor, partial(self.loss_func, loss_mask, sentence_order)
            else:
                logits = model(tokens, padding_mask, tokentype_ids=types)
                return logits, partial(self.loss_func, labels)
        return forward_step


class GPTTrainStep(AbstractTrainStep):
    """
    GPT train step class.

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, accelerator, args):
        super().__init__('GPTTrainStep')
        self.get_batch = self.get_batch_func(accelerator, args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func(accelerator)
        self.forward_step = self.get_forward_step_func()
        self.eod_token = args.padded_vocab_size - 1
        if args.vocab_file is not None:
            tokenizer = get_tokenizer()
            self.eod_token = tokenizer.eod
        self.reset_position_ids = args.reset_position_ids
        self.reset_attention_mask = args.reset_attention_mask
        self.eod_mask_loss = args.eod_mask_loss
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            self.model_output_class = CausalLMOutputWithCrossAttentions

    def get_batch_func(self, accelerator, megatron_dataset_flag):

        def get_batch_megatron(data_iterator):
            """Generate a batch"""
            keys = ['text']
            datatype = torch.int64
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = tensor_parallel.broadcast_data(keys, data, datatype)
            tokens_ = data_b['text'].long()
            labels = tokens_[:, 1:].contiguous()
            tokens = tokens_[:, :-1].contiguous()
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(tokens, self.eod_token, self.reset_position_ids, self.reset_attention_mask, self.eod_mask_loss)
            return tokens, labels, loss_mask, attention_mask, position_ids

        def get_batch_transformer(data_iterator):
            data = next(data_iterator)
            data = {'input_ids': data['input_ids']}
            data = send_to_device(data, torch.cuda.current_device())
            tokens_ = data['input_ids'].long()
            padding = torch.zeros((tokens_.shape[0], 1), dtype=tokens_.dtype, device=tokens_.device) + self.eod_token
            tokens_ = torch.concat([tokens_, padding], dim=1)
            labels = tokens_[:, 1:].contiguous()
            tokens = tokens_[:, :-1].contiguous()
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(tokens, self.eod_token, self.reset_position_ids, self.reset_attention_mask, True)
            return tokens, labels, loss_mask, attention_mask, position_ids
        if accelerator.state.megatron_lm_plugin.custom_get_batch_function is not None:
            return accelerator.state.megatron_lm_plugin.custom_get_batch_function
        if megatron_dataset_flag:
            try:
                return get_batch
            except ImportError:
                pass
            return get_batch_megatron
        else:
            return get_batch_transformer

    def get_loss_func(self, accelerator):
        args = get_args()

        def loss_func(loss_mask, output_tensor):
            if args.return_logits:
                losses, logits = output_tensor
            else:
                losses = output_tensor
            losses = losses.float()
            loss_mask = loss_mask.view(-1).float()
            if args.context_parallel_size > 1:
                loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
                torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
                loss = loss[0] / loss[1]
            else:
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
            if args.check_for_nan_in_loss_and_grad:
                global_rank = torch.distributed.get_rank()
                assert not loss.isnan(), f'Rank {global_rank}: found NaN in local forward loss calculation. Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
            averaged_loss = average_losses_across_data_parallel_group([loss])
            output_dict = {'lm loss': averaged_loss[0]}
            if args.return_logits:
                output_dict.update({'logits': logits})
            return loss, output_dict
        if accelerator.state.megatron_lm_plugin.custom_loss_function is not None:
            return accelerator.state.megatron_lm_plugin.custom_loss_function
        return loss_func

    def get_forward_step_func(self):

        def forward_step(data_iterator, model):
            """Forward step."""
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(data_iterator)
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
            return output_tensor, partial(self.loss_func, loss_mask)
        return forward_step


class T5TrainStep(AbstractTrainStep):
    """
    T5 train step class.

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, accelerator, args):
        super().__init__('T5TrainStep')
        self.get_batch = self.get_batch_func(accelerator, args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func(accelerator)
        self.forward_step = self.get_forward_step_func()
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            self.model_output_class = Seq2SeqLMOutput

    @staticmethod
    def attn_mask_postprocess(attention_mask):
        attention_mask_b1s = attention_mask.unsqueeze(1)
        attention_mask_bs1 = attention_mask.unsqueeze(2)
        attention_mask_bss = attention_mask_b1s * attention_mask_bs1
        extended_attention_mask = attention_mask_bss < 0.5
        return extended_attention_mask

    @staticmethod
    def get_decoder_mask(seq_length, device):
        attention_mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device))
        attention_mask = attention_mask < 0.5
        return attention_mask

    @staticmethod
    def get_enc_dec_mask(attention_mask, dec_seq_length, device):
        batch_size, _ = attention_mask.shape
        attention_mask_b1s = attention_mask.unsqueeze(1)
        attention_mask_bs1 = torch.ones((batch_size, dec_seq_length, 1), device=device)
        attention_mask_bss = attention_mask_bs1 * attention_mask_b1s
        extended_attention_mask = attention_mask_bss < 0.5
        return extended_attention_mask

    def get_batch_func(self, accelerator, megatron_dataset_flag):

        def get_batch_megatron(data_iterator):
            """Build the batch."""
            keys = ['text_enc', 'text_dec', 'labels', 'loss_mask', 'enc_mask', 'dec_mask', 'enc_dec_mask']
            datatype = torch.int64
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = tensor_parallel.broadcast_data(keys, data, datatype)
            tokens_enc = data_b['text_enc'].long()
            tokens_dec = data_b['text_dec'].long()
            labels = data_b['labels'].long()
            loss_mask = data_b['loss_mask'].float()
            enc_mask = data_b['enc_mask'] < 0.5
            dec_mask = data_b['dec_mask'] < 0.5
            enc_dec_mask = data_b['enc_dec_mask'] < 0.5
            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask

        def get_batch_transformer(data_iterator):
            """Build the batch."""
            data = next(data_iterator)
            data = send_to_device(data, torch.cuda.current_device())
            tokens_enc = data['input_ids'].long()
            labels = data['labels'].long()
            loss_mask = labels != -100
            if 'decoder_input_ids' in data:
                tokens_dec = data['decoder_input_ids'].long()
            else:
                tokens_dec = labels.new_zeros(labels.shape, device=labels.device, dtype=torch.long)
                tokens_dec[..., 1:] = labels[..., :-1].clone()
                tokens_dec[..., 0] = 0
                tokens_dec.masked_fill_(tokens_dec == -100, 0)
            enc_mask = T5TrainStep.attn_mask_postprocess(data['attention_mask'].long())
            dec_mask = T5TrainStep.get_decoder_mask(tokens_dec.shape[1], tokens_dec.device)
            enc_dec_mask = T5TrainStep.get_enc_dec_mask(data['attention_mask'].long(), tokens_dec.shape[1], tokens_dec.device)
            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask
        if accelerator.state.megatron_lm_plugin.custom_get_batch_function is not None:
            return accelerator.state.megatron_lm_plugin.custom_get_batch_function
        if megatron_dataset_flag:
            try:
                return get_batch
            except ImportError:
                pass
            return get_batch_megatron
        else:
            return get_batch_transformer

    def get_loss_func(self, accelerator):

        def loss_func(loss_mask, output_tensor):
            lm_loss_ = output_tensor.float()
            lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
            loss = lm_loss
            averaged_losses = average_losses_across_data_parallel_group([lm_loss])
            return loss, {'lm loss': averaged_losses[0]}
        if accelerator.state.megatron_lm_plugin.custom_loss_function is not None:
            return accelerator.state.megatron_lm_plugin.custom_loss_function
        return loss_func

    def get_forward_step_func(self):

        def forward_step(data_iterator, model):
            """Forward step."""
            tokens_enc, tokens_dec, loss_mask, lm_labels, enc_mask, dec_mask, enc_dec_mask = self.get_batch(data_iterator)
            output_tensor = model(tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask, tokentype_ids=None, lm_labels=lm_labels)
            return output_tensor, partial(self.loss_func, loss_mask)
        return forward_step


class MegatronEngine(torch.nn.Module):
    """
    Megatron-LM model wrapper

    Args:
        accelerator (:class:`~accelerate.Accelerator`): The accelerator object to use.
        model: Megatron-LM model
        optimizer: Megatron-LM optimizer
        lr_scheduler: Megatron-LM lr scheduler
    """

    def __init__(self, accelerator, model, optimizer, scheduler):
        super().__init__()
        self.module = model
        self.base_model = model[0]
        self.optimizer = optimizer
        self.scheduler = scheduler
        args = get_args()
        if accelerator.state.megatron_lm_plugin.custom_train_step_class is not None:
            self.train_step_handler = accelerator.state.megatron_lm_plugin.custom_train_step_class(args, **accelerator.state.megatron_lm_plugin.custom_train_step_kwargs)
        elif args.model_type_name == 'bert':
            self.train_step_handler = BertTrainStep(accelerator, args)
        elif args.model_type_name == 'gpt':
            self.train_step_handler = GPTTrainStep(accelerator, args)
        elif args.model_type_name == 't5':
            self.train_step_handler = T5TrainStep(accelerator, args)
        else:
            raise ValueError(f'Unsupported model type: {args.model_type_name}')
        self.optimizer.skipped_iter = False
        self.total_loss_dict = {}
        self.eval_total_loss_dict = {}
        self.iteration = 0
        self.report_memory_flag = True
        self.num_floating_point_operations_so_far = 0
        self.module_config = None
        if args.tensorboard_dir is not None:
            write_args_to_tensorboard()

    def get_module_config(self):
        args = get_args()
        config = get_model_config(self.module[0])
        config.grad_scale_func = self.optimizer.scale_loss
        if isinstance(self.module[0], LocalDDP) and args.overlap_grad_reduce:
            assert config.no_sync_func is None, 'When overlap_grad_reduce is True, config.no_sync_func must be None; a custom no_sync_func is not supported when overlapping grad-reduce'
            config.no_sync_func = [model_chunk.no_sync for model_chunk in self.module]
            if len(self.module) == 1:
                config.no_sync_func = config.no_sync_func[0]
            if args.delay_grad_reduce:
                config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in self.module]
                if len(self.module) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        if args.overlap_param_gather and args.delay_param_gather:
            config.param_sync_func = [(lambda x: self.optimizer.finish_param_sync(model_index, x)) for model_index in range(len(self.module))]
            if len(self.module) == 1:
                config.param_sync_func = config.param_sync_func[0]
        config.finalize_model_grads_func = finalize_model_grads
        return config

    def train(self):
        for model_module in self.module:
            model_module.train()
        if self.module_config is None:
            self.module_config = self.get_module_config()
        self.log_eval_results()

    def eval(self):
        for model_module in self.module:
            model_module.eval()
        if self.module_config is None:
            self.module_config = self.get_module_config()

    def get_batch_data_iterator(self, batch_data):
        args = get_args()
        data_chunks = []
        if len(batch_data) > 0:
            if args.num_micro_batches > 1:
                for i in range(0, args.num_micro_batches):
                    data_chunks.append({k: v[i * args.micro_batch_size:(i + 1) * args.micro_batch_size] for k, v in batch_data.items()})
            else:
                data_chunks = [batch_data]
        if len(self.module) > 1:
            batch_data_iterator = [iter(data_chunks) for _ in range(len(self.module))] if len(batch_data) > 0 else [None] * len(self.module)
        else:
            batch_data_iterator = iter(data_chunks) if len(batch_data) > 0 else None
        return batch_data_iterator

    def train_step(self, **batch_data):
        """
        Training step for Megatron-LM

        Args:
            batch_data (:obj:`dict`): The batch data to train on.
        """
        batch_data_iterator = self.get_batch_data_iterator(batch_data)
        loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad = train_step(forward_step_func=self.train_step_handler.forward_step, data_iterator=batch_data_iterator, model=self.module, optimizer=self.optimizer, opt_param_scheduler=self.scheduler, config=self.module_config)
        self.optimizer.skipped_iter = skipped_iter == 1
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad

    def eval_step(self, **batch_data):
        """
        Evaluation step for Megatron-LM

        Args:
            batch_data (:obj:`dict`): The batch data to evaluate on.
        """
        args = get_args()
        batch_data_iterator = self.get_batch_data_iterator(batch_data)
        forward_backward_func = get_forward_backward_func()
        loss_dicts = forward_backward_func(forward_step_func=self.train_step_handler.forward_step, data_iterator=batch_data_iterator, model=self.module, num_microbatches=get_num_microbatches(), seq_length=args.seq_length, micro_batch_size=args.micro_batch_size, forward_only=True)
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()
        args.consumed_valid_samples += mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            loss_reduced = {}
            for key in loss_dicts[0]:
                losses_reduced_for_key = [x[key] for x in loss_dicts]
                if len(losses_reduced_for_key[0].shape) == 0:
                    loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
                else:
                    loss_reduced[key] = torch.concat(losses_reduced_for_key)
            return loss_reduced
        return {}

    def forward(self, **batch_data):
        args = get_args()
        if self.module[0].training:
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = self.train_step(**batch_data)
            self.iteration += 1
            batch_size = mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
            args.consumed_train_samples += batch_size
            self.num_floating_point_operations_so_far += num_floating_point_operations(args, batch_size)
            if args.tensorboard_dir is not None:
                loss_scale = self.optimizer.get_loss_scale().item()
                params_norm = None
                if args.log_params_norm:
                    params_norm = calc_params_l2_norm(self.model)
                self.report_memory_flag = training_log(loss_dict, self.total_loss_dict, self.optimizer.param_groups[0]['lr'], self.iteration, loss_scale, self.report_memory_flag, skipped_iter, grad_norm, params_norm, num_zeros_in_grad)
        else:
            loss_dict = self.eval_step(**batch_data)
            if args.tensorboard_dir is not None:
                for key in loss_dict:
                    self.eval_total_loss_dict[key] = self.eval_total_loss_dict.get(key, torch.FloatTensor([0.0])) + loss_dict[key]
                    self.eval_total_loss_dict[key + '_num_iters'] = self.eval_total_loss_dict.get(key + '_num_iters', torch.FloatTensor([0.0])) + torch.FloatTensor([1.0])
        loss = torch.tensor(0.0, device=torch.cuda.current_device())
        for key in loss_dict:
            if len(loss_dict[key].shape) == 0:
                loss += loss_dict[key]
        logits = None
        if 'logits' in loss_dict:
            logits = loss_dict['logits']
        if self.train_step_handler.model_output_class is not None:
            return self.train_step_handler.model_output_class(loss=loss, logits=logits)
        return loss

    def log_eval_results(self):
        args = get_args()
        if args.tensorboard_dir is None or self.iteration == 0:
            return
        args = get_args()
        writer = get_tensorboard_writer()
        string = f'validation loss at iteration {self.iteration} | '
        for key in self.eval_total_loss_dict:
            if key.endswith('_num_iters'):
                continue
            value = self.eval_total_loss_dict[key] / self.eval_total_loss_dict[key + '_num_iters']
            string += f'{key} value: {value} | '
            ppl = math.exp(min(20, value.item()))
            if args.pretraining_flag:
                string += f'{key} PPL: {ppl} | '
            if writer:
                writer.add_scalar(f'{key} validation', value.item(), self.iteration)
                if args.pretraining_flag:
                    writer.add_scalar(f'{key} validation ppl', ppl, self.iteration)
        length = len(string) + 1
        print_rank_last('-' * length)
        print_rank_last(string)
        print_rank_last('-' * length)
        self.eval_total_loss_dict = {}

    def save_checkpoint(self, output_dir):
        self.log_eval_results()
        args = get_args()
        args.save = output_dir
        torch.distributed.barrier()
        save_checkpoint(self.iteration, self.module, self.optimizer, self.scheduler, num_floating_point_operations_so_far=self.num_floating_point_operations_so_far)
        torch.distributed.barrier()

    def load_checkpoint(self, input_dir):
        args = get_args()
        args.load = input_dir
        args.consumed_train_samples = 0
        args.consumed_valid_samples = 0
        torch.distributed.barrier()
        iteration, num_floating_point_operations_so_far = load_checkpoint(self.module, self.optimizer, self.scheduler)
        torch.distributed.barrier()
        self.iteration = iteration
        self.num_floating_point_operations_so_far = num_floating_point_operations_so_far
        if args.fp16 and self.iteration == 0:
            self.optimizer.reload_model_params()

    def megatron_generate(self, inputs, attention_mask=None, max_length=None, max_new_tokens=None, num_beams=None, temperature=None, top_k=None, top_p=None, length_penalty=None, **kwargs):
        """
        Generate method for GPT2 model. This method is used for inference. Supports both greedy and beam search along
        with sampling. Refer the Megatron-LM repo for more details

        Args:
            inputs (torch.Tensor): input ids
            attention_mask (torch.Tensor, optional): attention mask. Defaults to None.
            max_length (int, optional): max length of the generated sequence. Defaults to None.
            Either this or max_new_tokens should be provided.
            max_new_tokens (int, optional): max number of tokens to be generated. Defaults to None.
            Either this or max_length should be provided.
            num_beams (int, optional): number of beams to use for beam search. Defaults to None.
            temperature (float, optional): temperature for sampling. Defaults to 1.0.
            top_k (int, optional): top k tokens to consider for sampling. Defaults to 0.0.
            top_p (float, optional): tokens in top p probability are considered for sampling. Defaults to 0.0.
            length_penalty (float, optional): length penalty for beam search. Defaults to None.
            kwargs: additional key-value arguments
        """
        args = get_args()
        if args.model_type_name != 'gpt':
            raise NotImplementedError('Generate method is not implemented for this model')
        if args.data_parallel_size > 1:
            raise ValueError('Generate method requires data parallelism to be 1')
        if args.sequence_parallel:
            raise ValueError('Generate method requires sequence parallelism to be False')
        if args.recompute_granularity is not None:
            raise ValueError('Checkpoint activations cannot be set for inference')
        if args.vocab_file is None:
            raise ValueError('Vocab file is required for inference')
        if max_length is None and max_new_tokens is None:
            raise ValueError('`max_length` or `max_new_tokens` are required for inference')
        if temperature is None:
            temperature = 1.0
        elif not 0.0 < temperature <= 100.0:
            raise ValueError('temperature must be a positive number less than or equal to 100.0')
        if top_k is None:
            top_k = 0
        elif not 0 <= top_k <= 1000:
            raise ValueError('top_k must be a positive number less than or equal to 1000')
        if top_p is None:
            top_p = 0.0
        elif top_p > 0.0 and top_k > 0.0:
            raise ValueError('top_p and top_k sampling cannot be set together')
        elif not 0.0 <= top_p <= 1.0:
            raise ValueError('top_p must be less than or equal to 1.0')
        top_p_decay = kwargs.get('top_p_decay', 0.0)
        if not 0.0 <= top_p_decay <= 1.0:
            raise ValueError('top_p_decay must be less than or equal to 1.0')
        top_p_bound = kwargs.get('top_p_bound', 0.0)
        if not 0.0 <= top_p_bound <= 1.0:
            raise ValueError('top_p_bound must be less than or equal to 1.0')
        add_BOS = kwargs.get('add_BOS', False)
        if not isinstance(add_BOS, bool):
            raise ValueError('add_BOS must be a boolean')
        beam_width = num_beams
        if beam_width is not None:
            if not isinstance(beam_width, int):
                raise ValueError('beam_width must be an integer')
            if beam_width < 1:
                raise ValueError('beam_width must be greater than 0')
            if inputs.shape[0] > 1:
                return 'When doing beam_search, batch size must be 1'
        tokenizer = get_tokenizer()
        stop_token = kwargs.get('stop_token', tokenizer.eod)
        if stop_token is not None:
            if not isinstance(stop_token, int):
                raise ValueError('stop_token must be an integer')
        if length_penalty is None:
            length_penalty = 1.0
        sizes_list = None
        prompts_tokens_tensor = None
        prompts_length_tensor = None
        if torch.distributed.get_rank() == 0:
            if attention_mask is None:
                prompts_length_tensor = torch.LongTensor([inputs.shape[1]] * inputs.shape[0])
            else:
                prompts_length_tensor = attention_mask.sum(axis=-1)
            if max_new_tokens is None:
                max_new_tokens = max_length - inputs.shape[1]
            if max_new_tokens <= 0:
                raise ValueError('max_new_tokens must be greater than 0')
            if add_BOS:
                max_length = max_new_tokens + inputs.shape[1] + 1
                max_length = 4 * math.ceil(max_length / 4)
                max_new_tokens = max_length - (inputs.shape[1] + 1)
                padding = torch.LongTensor([[tokenizer.eod] * max_new_tokens] * inputs.shape[0])
                prompts_tokens_tensor = torch.concat([torch.unsqueeze(padding[:, 0], axis=-1), inputs, padding], axis=-1)
            else:
                max_length = max_new_tokens + inputs.shape[1]
                max_length = 4 * math.ceil(max_length / 4)
                max_new_tokens = max_length - inputs.shape[1]
                padding = torch.LongTensor([[tokenizer.eod] * max_new_tokens] * inputs.shape[0])
                prompts_tokens_tensor = torch.concat([inputs, padding], axis=-1)
            sizes_list = [prompts_tokens_tensor.size(0), prompts_tokens_tensor.size(1)]
        sizes_tensor = broadcast_int_list(2, int_list=sizes_list, rank=0)
        sizes = sizes_tensor.tolist()
        context_tokens_tensor = broadcast_tensor(sizes, torch.int64, tensor=prompts_tokens_tensor, rank=0)
        context_length_tensor = broadcast_tensor(sizes[0], torch.int64, tensor=prompts_length_tensor, rank=0)
        random_seed = kwargs.get('random_seed', 0)
        torch.random.manual_seed(random_seed)
        unwrapped_model = unwrap_model(self.base_model, (torchDDP, LocalDDP, Float16Module))
        if beam_width is not None:
            tokens, _ = beam_search_and_return_on_first_stage(unwrapped_model, context_tokens_tensor, context_length_tensor, beam_width, stop_token=stop_token, num_return_gen=1, length_penalty=length_penalty)
        else:
            tokens, _, _ = generate_tokens_probs_and_return_on_first_stage(unwrapped_model, context_tokens_tensor, context_length_tensor, return_output_log_probs=False, top_k=top_k, top_p=top_p, top_p_decay=top_p_decay, top_p_bound=top_p_bound, temperature=temperature, use_eod_token_for_early_termination=True)
        return tokens


class ModelWithTiedWeights(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.linear2 = torch.nn.Linear(4, 2)
        self.linear2.weight = self.linear1.weight
        self.linear2.bias = self.linear1.bias

    def forward(self, x):
        return self.linear2(self.linear1(x))


class ModelForTest(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class LinearWithNonPersistentBuffers(nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int', bias: 'bool'=True, device=None, dtype=None) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight', torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.register_buffer('bias', torch.empty(out_features, **factory_kwargs), persistent=False)
        else:
            self.register_buffer('bias', None)

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return torch.nn.functional.linear(input, self.weight, self.bias)


class ModelForTestNonPersistentBuffers(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = LinearWithNonPersistentBuffers(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = LinearWithNonPersistentBuffers(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class ModelForTestCopy(nn.Module):

    def __init__(self, id: 'int'):
        super().__init__()
        self.id = id
        self.linear1 = nn.Linear(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x))), self.id


class ModelForTestTiedWeights(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class BiggerModelForTest(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 5)
        self.batchnorm = nn.BatchNorm1d(5)
        self.linear3 = nn.Linear(5, 6)
        self.linear4 = nn.Linear(6, 5)

    def forward(self, x):
        return self.linear4(self.linear3(self.batchnorm(self.linear2(self.linear1(x)))))


class ModuleWithUnusedSubModules(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return x @ self.linear.weight.t() + self.linear.bias


class ModelWithUnusedSubModulesForTest(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = ModuleWithUnusedSubModules(3, 4)
        self.linear2 = ModuleWithUnusedSubModules(4, 5)
        self.batchnorm = nn.BatchNorm1d(5)
        self.linear3 = ModuleWithUnusedSubModules(5, 6)
        self.linear4 = ModuleWithUnusedSubModules(6, 5)

    def forward(self, x):
        return self.linear4(self.linear3(self.batchnorm(self.linear2(self.linear1(x)))))


class NestedModelForTest(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = ModelForTest()

    def forward(self, x):
        return self.model(x)


class ModelSeveralDtypes(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('int_param', torch.randint(high=10, size=(15, 30)))
        self.register_parameter('float_param', torch.nn.Parameter(torch.rand(10, 5)))

    def forward(self, x):
        return x + 2


class DummyModel(nn.Module):
    """Simple model to do y=mx+b"""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x * self.a + self.b


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (DummyModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LinearWithNonPersistentBuffers,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MockModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 40, 20]), torch.rand([4, 4, 40, 20])], {})),
    (ModelForTestTiedWeights,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ModelSeveralDtypes,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ModuleWithUnusedSubModules,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NoiseModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

