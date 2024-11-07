
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


from torchvision.utils import save_image


import time


import torch.utils.checkpoint


import itertools


import logging


import math


import torch.nn.functional as F


from typing import Optional


import random


from torchvision import transforms


from typing import Dict


from typing import List


from typing import Tuple


from typing import Union


from torch import nn


from torch.nn import functional as F


from typing import Any


from typing import Callable


import warnings


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.nn as nn


from random import randint


import copy


import re


from collections import Counter


from torch.optim import AdamW


from enum import Enum


from typing import TYPE_CHECKING


from torch.optim import Optimizer


import collections


import inspect


from copy import deepcopy


from typing import Literal


from torch.nn import BCEWithLogitsLoss


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from collections import OrderedDict


from torch.nn import Module


from torch.autograd import Function


from itertools import chain


from torch.nn.modules import Module


from typing import Set


from typing import Type


from torch import svd_lowrank


from functools import partial


from functools import reduce


import torch.nn.init as init


from abc import abstractmethod


from abc import ABC


from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


from torch.nn.init import _calculate_correct_fan


from torch import Tensor


from torch.testing import assert_close


from torch.distributed import init_process_group


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from scipy import stats


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(self, conditioning_embedding_channels: 'int', conditioning_channels: 'int'=3, block_out_channels: 'Tuple[int]'=(16, 32, 96, 256)):
        super().__init__()
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
        self.conv_out = zero_module(nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1))

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        embedding = self.conv_out(embedding)
        return embedding


class AutoModelForSentenceEmbedding(nn.Module):

    def __init__(self, model_name, tokenizer, normalize=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.tokenizer = tokenizer

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-09)

    def __getattr__(self, name: 'str'):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == 'model':
                raise
            return getattr(self.model, name)


class CastOutputToFloat(nn.Sequential):

    def forward(self, x):
        return super().forward(x)


class Shell(nn.Module):

    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


DUMMY_MODEL_CONFIG = {'model_type': 'custom'}


class BufferDict(Module):
    """
    Holds buffers in a dictionary.

    BufferDict can be indexed like a regular Python dictionary, but buffers it contains are properly registered, and
    will be visible by all Module methods. `torch.nn.BufferDict` is an **ordered** dictionary that respects

    * the order of insertion, and
    * in `torch.nn.BufferDict.update`, the order of the merged `OrderedDict` or another `torch.nn.BufferDict` (the
      argument to `torch.nn.BufferDict.update`).

    Note that `torch.nn.BufferDict.update` with other unordered mapping types (e.g., Python's plain `dict`) does not
    preserve the order of the merged mapping.

    Args:
        buffers (iterable, optional):
            a mapping (dictionary) of (string : `torch.Tensor`) or an iterable of key-value pairs of type (string,
            `torch.Tensor`)

    ```python
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.buffers = nn.BufferDict({"left": torch.randn(5, 10), "right": torch.randn(5, 10)})

        def forward(self, x, choice):
            x = self.buffers[choice].mm(x)
            return x
    ```
    """

    def __init__(self, buffers=None, persistent: 'bool'=False):
        """
        Args:
            buffers (`dict`):
                A mapping (dictionary) from string to `torch.Tensor`, or an iterable of key-value pairs of type
                (string, `torch.Tensor`).
        """
        super().__init__()
        if buffers is not None:
            self.update(buffers)
        self.persistent = persistent

    def __getitem__(self, key):
        return self._buffers[key]

    def __setitem__(self, key, buffer):
        self.register_buffer(key, buffer, persistent=self.persistent)

    def __delitem__(self, key):
        del self._buffers[key]

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.keys())

    def __contains__(self, key):
        return key in self._buffers

    def clear(self):
        """Remove all items from the BufferDict."""
        self._buffers.clear()

    def pop(self, key):
        """Remove key from the BufferDict and return its buffer.

        Args:
            key (`str`):
                Key to pop from the BufferDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        """Return an iterable of the BufferDict keys."""
        return self._buffers.keys()

    def items(self):
        """Return an iterable of the BufferDict key/value pairs."""
        return self._buffers.items()

    def values(self):
        """Return an iterable of the BufferDict values."""
        return self._buffers.values()

    def update(self, buffers):
        """
        Update the `torch.nn.BufferDict` with the key-value pairs from a mapping or an iterable, overwriting existing
        keys.

        Note:
            If `buffers` is an `OrderedDict`, a `torch.nn.BufferDict`, or an iterable of key-value pairs, the order of
            new elements in it is preserved.

        Args:
            buffers (iterable):
                a mapping (dictionary) from string to `torch.Tensor`, or an iterable of key-value pairs of type
                (string, `torch.Tensor`).
        """
        if not isinstance(buffers, collections.abc.Iterable):
            raise TypeError('BuffersDict.update should be called with an iterable of key/value pairs, but got ' + type(buffers).__name__)
        if isinstance(buffers, collections.abc.Mapping):
            if isinstance(buffers, (OrderedDict, BufferDict)):
                for key, buffer in buffers.items():
                    self[key] = buffer
            else:
                for key, buffer in sorted(buffers.items()):
                    self[key] = buffer
        else:
            for j, p in enumerate(buffers):
                if not isinstance(p, collections.abc.Iterable):
                    raise TypeError('BufferDict update sequence element #' + str(j) + ' should be Iterable; is' + type(p).__name__)
                if not len(p) == 2:
                    raise ValueError('BufferDict update sequence element #' + str(j) + ' has length ' + str(len(p)) + '; 2 is required')
                self[p[0]] = p[1]

    def extra_repr(self):
        child_lines = []
        for k, p in self._buffers.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else f' (GPU {p.get_device()})'
            parastr = f'Buffer containing: [{torch.typename(p)} of size {size_str}{device_str}]'
            child_lines.append('  (' + k + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError('BufferDict should not be called.')


DUMMY_TARGET_MODULES = 'dummy-target-modules'


EMBEDDING_LAYER_NAMES = ['embed_tokens', 'lm_head']


MIN_TARGET_MODULES_FOR_OPTIMIZATION = 20


class ModulesToSaveWrapper(torch.nn.Module):

    def __init__(self, module_to_save, adapter_name):
        super().__init__()
        self.original_module = module_to_save
        self.modules_to_save = torch.nn.ModuleDict({})
        self._active_adapter = adapter_name
        self._disable_adapters = False
        self.update(adapter_name)
        self.check_module()

    def check_module(self):
        """Perform some sanity checks on the module to ensure that it works"""
        forbidden_classes = torch.nn.ModuleDict, torch.nn.ModuleList, torch.nn.ParameterDict, torch.nn.ParameterList
        if isinstance(self.original_module, forbidden_classes):
            cls_name = self.original_module.__class__
            raise TypeError(f'modules_to_save cannot be applied to modules of type {cls_name}')
        if isinstance(self.original_module, BaseTunerLayer):
            cls_name = self.original_module.__class__
            raise TypeError(f'modules_to_save cannot be applied to modules of type {cls_name}')

    @property
    def disable_adapters(self) ->bool:
        return self._disable_adapters

    @property
    def active_adapter(self) ->str:
        return self._active_adapter

    def __getattr__(self, name: 'str'):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        if '_modules' not in self.__dict__:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        modules = self.__dict__['_modules']
        if self.disable_adapters:
            module = modules['original_module']
        elif self.active_adapter in modules['modules_to_save']:
            module = modules['modules_to_save'][self.active_adapter]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(module, name)

    def update(self, adapter_name):
        context_manager = nullcontext()
        for _, param in self.original_module.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, 'ds_numel'):
                context_manager = deepspeed.zero.GatheredParameters(self.original_module.parameters(), modifier_rank=0)
                break
        with context_manager:
            self.modules_to_save.update(torch.nn.ModuleDict({adapter_name: copy.deepcopy(self.original_module)}))
        if hasattr(self.modules_to_save[adapter_name], '_hf_hook'):
            old_hook = self.modules_to_save[adapter_name]._hf_hook
            new_hook = self._create_new_hook(old_hook)
            remove_hook_from_module(self.modules_to_save[adapter_name])
            add_hook_to_module(self.modules_to_save[adapter_name], new_hook)
        self.original_module.requires_grad_(False)
        if adapter_name == self.active_adapter:
            self.modules_to_save[adapter_name].requires_grad_(True)

    def _create_new_hook(self, old_hook):
        """
        Creates a new hook based on the old hook. Use it only if you know what you are doing !
        """
        old_hook_cls = getattr(accelerate.hooks, old_hook.__class__.__name__)
        old_hook_attr = old_hook.__dict__
        filtered_old_hook_attr = {}
        old_hook_init_signature = inspect.signature(old_hook_cls.__init__)
        for k in old_hook_attr.keys():
            if k in old_hook_init_signature.parameters:
                filtered_old_hook_attr[k] = old_hook_attr[k]
        new_hook = old_hook_cls(**filtered_old_hook_attr)
        return new_hook

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get('adapter_names', None)
        if adapter_names is None:
            return
        if len(x) != len(adapter_names):
            msg = f'Length of `adapter_names` should be the same as the number of inputs, but got {len(adapter_names)} and {len(x)} respectively.'
            raise ValueError(msg)

    def _mixed_batch_forward(self, input: 'torch.Tensor', *args: Any, adapter_names: list[str], **kwargs: Any) ->torch.Tensor:
        SUPPORTED_MODULES = torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d
        module_names = ', '.join([module.__name__ for module in SUPPORTED_MODULES])
        if not isinstance(self.original_module, SUPPORTED_MODULES):
            raise TypeError(f'Mixed batching is only supported for the following modules: {module_names}.')
        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])
        results = [(0) for _ in range(len(input))]
        for i, active_adapter in enumerate(unique_adapters):
            sub_batch = input[sub_batch_indices_list[i]]
            if active_adapter == '__base__':
                output = self.original_module(sub_batch, *args, **kwargs)
            else:
                output = self.modules_to_save[active_adapter](sub_batch, *args, **kwargs)
            for index, j in enumerate(sub_batch_indices_list[i]):
                results[j] = output[index]
        return torch.stack(results)

    def forward(self, x: 'torch.Tensor', *args, **kwargs):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop('adapter_names', None)
        if self.disable_adapters or self.active_adapter not in self.modules_to_save:
            return self.original_module(x, *args, **kwargs)
        if adapter_names is None:
            return self.modules_to_save[self.active_adapter](x, *args, **kwargs)
        return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)

    def enable_adapters(self, enabled: 'bool'):
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if self._disable_adapters is not enabled:
            return
        if enabled:
            self.original_module.requires_grad_(False)
            self.modules_to_save[self.active_adapter].requires_grad_(True)
            self._disable_adapters = False
        else:
            self.original_module.requires_grad_(True)
            self.modules_to_save.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_name: 'str'):
        """Set the active adapter

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (str): The name of the adapter to set as active
        """
        if adapter_name not in self.modules_to_save:
            raise ValueError(f'Adapter {adapter_name} not found in {self.modules_to_save.keys()}')
        self.modules_to_save[self.active_adapter].requires_grad_(False)
        self.modules_to_save[adapter_name].requires_grad_(True)
        self._active_adapter = adapter_name


CONFIG_NAME = 'adapter_config.json'


MIN_EXPECTED_CONFIG_KEYS = {'peft_type'}


def _check_and_remove_unused_kwargs(cls, kwargs):
    """Make PEFT configs forward-compatible by removing unused kwargs that were added in later PEFT versions.

    This assumes that removing the unused kwargs will not affect the default behavior.

    Returns the filtered kwargs and the set of removed keys.
    """
    signature_parameters = inspect.signature(cls.__init__).parameters
    unexpected_kwargs = set(kwargs.keys()) - set(signature_parameters.keys())
    for key in unexpected_kwargs:
        del kwargs[key]
    return kwargs, unexpected_kwargs


class PeftType(str, enum.Enum):
    """
    Enum class for the different types of adapters in PEFT.

    Supported PEFT types:
    - PROMPT_TUNING
    - MULTITASK_PROMPT_TUNING
    - P_TUNING
    - PREFIX_TUNING
    - LORA
    - ADALORA
    - BOFT
    - ADAPTION_PROMPT
    - IA3
    - LOHA
    - LOKR
    - OFT
    - XLORA
    - POLY
    - LN_TUNING
    - VERA
    - FOURIERFT
    - HRA
    - BONE
    """
    PROMPT_TUNING = 'PROMPT_TUNING'
    MULTITASK_PROMPT_TUNING = 'MULTITASK_PROMPT_TUNING'
    P_TUNING = 'P_TUNING'
    PREFIX_TUNING = 'PREFIX_TUNING'
    LORA = 'LORA'
    ADALORA = 'ADALORA'
    BOFT = 'BOFT'
    ADAPTION_PROMPT = 'ADAPTION_PROMPT'
    IA3 = 'IA3'
    LOHA = 'LOHA'
    LOKR = 'LOKR'
    OFT = 'OFT'
    POLY = 'POLY'
    LN_TUNING = 'LN_TUNING'
    VERA = 'VERA'
    FOURIERFT = 'FOURIERFT'
    XLORA = 'XLORA'
    HRA = 'HRA'
    VBLORA = 'VBLORA'
    BONE = 'BONE'


class _ExcludedModule:
    """
    A private helper method used to represent excluded modules in the check_target_module_exists function.
    """

    def __bool__(self):
        return False


def _find_minimal_target_modules(target_modules: 'list[str] | set[str]', other_module_names: 'list[str] | set[str]') ->set[str]:
    """Find the minimal set of target modules that is sufficient to separate them from the other modules.

    Sometimes, a very large list of target_modules could be passed, which can slow down loading of adapters (e.g. when
    loaded from diffusers). It may be possible to condense this list from hundreds of items to just a handful of
    suffixes that are sufficient to distinguish the target modules from the other modules.

    Example:
        ```py
        >>> from peft.tuners.tuners_utils import _find_minimal_target_modules

        >>> target_modules = [f"model.decoder.layers.{i}.self_attn.q_proj" for i in range(100)]
        >>> target_modules += [f"model.decoder.layers.{i}.self_attn.v_proj" for i in range(100)]
        >>> other_module_names = [f"model.encoder.layers.{i}.self_attn.k_proj" for i in range(100)]
        >>> _find_minimal_target_modules(target_modules, other_module_names)
        {"q_proj", "v_proj"}
        ```

    Args:
        target_modules (`list[str]` | `set[str]`):
            The list of target modules.
        other_module_names (`list[str]` | `set[str]`):
            The list of other module names. They must not overlap with the target modules.

    Returns:
        `set[str]`:
            The minimal set of target modules that is sufficient to separate them from the other modules.

    Raises:
        ValueError:
            If `target_modules` is not a list or set of strings or if it contains an empty string. Also raises an error
            if `target_modules` and `other_module_names` contain common elements.
    """
    if isinstance(target_modules, str) or not target_modules:
        raise ValueError('target_modules should be a list or set of strings.')
    target_modules = set(target_modules)
    if '' in target_modules:
        raise ValueError('target_modules should not contain an empty string.')
    other_module_names = set(other_module_names)
    if not target_modules.isdisjoint(other_module_names):
        msg = 'target_modules and other_module_names contain common elements, this should not happen, please open a GitHub issue at https://github.com/huggingface/peft/issues with the code to reproduce this issue'
        raise ValueError(msg)

    def generate_suffixes(s):
        parts = s.split('.')
        return ['.'.join(parts[i:]) for i in range(len(parts))][::-1]
    other_module_suffixes = {suffix for item in other_module_names for suffix in generate_suffixes(item)}
    target_modules_suffix_map = {item: generate_suffixes(item) for item in target_modules}
    required_suffixes = set()
    for item, suffixes in sorted(target_modules_suffix_map.items(), key=lambda tup: tup[1]):
        for suffix in suffixes:
            if suffix in required_suffixes or suffix in other_module_suffixes:
                continue
            if not any(item.endswith('.' + req_suffix) for req_suffix in required_suffixes):
                required_suffixes.add(suffix)
                break
    if not required_suffixes:
        return set(target_modules)
    return required_suffixes


def _get_submodules(model, key):
    parent = model.get_submodule('.'.join(key.split('.')[:-1]))
    target_name = key.split('.')[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


INCLUDE_LINEAR_LAYERS_SHORTHAND = 'all-linear'


SEQ_CLS_HEAD_NAMES = ['score', 'classifier']


class TaskType(str, enum.Enum):
    """
    Enum class for the different types of tasks supported by PEFT.

    Overview of the supported task types:
    - SEQ_CLS: Text classification.
    - SEQ_2_SEQ_LM: Sequence-to-sequence language modeling.
    - CAUSAL_LM: Causal language modeling.
    - TOKEN_CLS: Token classification.
    - QUESTION_ANS: Question answering.
    - FEATURE_EXTRACTION: Feature extraction. Provides the hidden states which can be used as embeddings or features
      for downstream tasks.
    """
    SEQ_CLS = 'SEQ_CLS'
    SEQ_2_SEQ_LM = 'SEQ_2_SEQ_LM'
    CAUSAL_LM = 'CAUSAL_LM'
    TOKEN_CLS = 'TOKEN_CLS'
    QUESTION_ANS = 'QUESTION_ANS'
    FEATURE_EXTRACTION = 'FEATURE_EXTRACTION'


logger = logging.getLogger(__name__)


COMPATIBLE_TUNER_TYPES = PeftType.LORA, PeftType.LOHA, PeftType.LOKR, PeftType.ADALORA, PeftType.OFT


TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {'t5': ['q', 'v'], 'mt5': ['q', 'v'], 'bart': ['q_proj', 'v_proj'], 'gpt2': ['c_attn'], 'bloom': ['query_key_value'], 'blip-2': ['q', 'v', 'q_proj', 'v_proj'], 'opt': ['q_proj', 'v_proj'], 'gptj': ['q_proj', 'v_proj'], 'gpt_neox': ['query_key_value'], 'gpt_neo': ['q_proj', 'v_proj'], 'bert': ['query', 'value'], 'roberta': ['query', 'value'], 'xlm-roberta': ['query', 'value'], 'electra': ['query', 'value'], 'deberta-v2': ['query_proj', 'value_proj'], 'deberta': ['in_proj'], 'layoutlm': ['query', 'value'], 'llama': ['q_proj', 'v_proj'], 'chatglm': ['query_key_value'], 'gpt_bigcode': ['c_attn'], 'mpt': ['Wqkv'], 'RefinedWebModel': ['query_key_value'], 'RefinedWeb': ['query_key_value'], 'falcon': ['query_key_value'], 'btlm': ['c_proj', 'c_attn'], 'codegen': ['qkv_proj'], 'mistral': ['q_proj', 'v_proj'], 'mixtral': ['q_proj', 'v_proj'], 'stablelm': ['q_proj', 'v_proj'], 'phi': ['q_proj', 'v_proj', 'fc1', 'fc2'], 'gemma': ['q_proj', 'v_proj'], 'gemma2': ['q_proj', 'v_proj'], 'qwen2': ['q_proj', 'v_proj']}


def get_auto_gptq_quant_linear(gptq_quantization_config):
    """
    Get the right AutoGPTQQuantLinear class based on the quantization config file
    """
    if gptq_quantization_config is not None and is_auto_gptq_available():
        desc_act = gptq_quantization_config.desc_act
        group_size = gptq_quantization_config.group_size
        bits = gptq_quantization_config.bits
        if hasattr(gptq_quantization_config, 'use_exllama'):
            use_exllama = gptq_quantization_config.use_exllama
        else:
            use_exllama = not gptq_quantization_config.disable_exllama
        if hasattr(gptq_quantization_config, 'exllama_config'):
            exllama_version = gptq_quantization_config.exllama_config['version']
        else:
            exllama_version = 1
        AutoGPTQQuantLinear = dynamically_import_QuantLinear(use_triton=False, desc_act=desc_act, group_size=group_size, bits=bits, disable_exllama=not (use_exllama and exllama_version == 1), disable_exllamav2=not (use_exllama and exllama_version == 2))
        return AutoGPTQQuantLinear
    return None


def dequantize_bnb_weight(weight: 'torch.nn.Parameter', state=None):
    """Helper function to dequantize 4bit or 8bit bnb weights.

    Since dequantization is not supported on CPU, the weight will be temporarily moved to CUDA if necessary.
    """
    device = weight.device
    is_cpu = device.type == torch.device('cpu').type
    if is_cpu:
        weight = weight
    cls_name = weight.__class__.__name__
    if cls_name == 'Params4bit':
        dequantized = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        if is_cpu:
            dequantized = dequantized
        return dequantized
    if state.SCB is None:
        state.SCB = weight.SCB
    im = torch.eye(weight.data.shape[-1]).contiguous().half()
    im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
    im, Sim = bnb.functional.transform(im, 'col32')
    if state.CxB is None:
        state.CxB, state.SB = bnb.functional.transform(weight.data, to_order=state.formatB)
    out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
    dequantized = bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()
    if is_cpu:
        dequantized = dequantized
    return dequantized


def dequantize_module_weight(module: 'torch.nn.Module') ->torch.nn.Parameter:
    """
    Helper function to dequantize a quantized weight.

    This function should be extended if more quantization schemes are added to the library.

    If the weight is not quantized, it will be returned as is.
    """
    if hasattr(module, 'W_q'):
        weight = module.dequantize()
        return weight
    elif type(module.weight).__module__.startswith('torchao.'):
        weight = module.weight.dequantize()
        return weight
    weight = module.weight
    if not isinstance(weight, torch.nn.Parameter):
        if isinstance(weight, torch.Tensor):
            return weight
        raise TypeError(f'Input weight should be of type nn.Parameter, got {type(weight)} instead')
    cls_name = weight.__class__.__name__
    if cls_name not in ('Params4bit', 'Int8Params'):
        return weight
    quant_state = getattr(module, 'state', None)
    device = weight.device
    is_cpu = device.type == torch.device('cpu').type
    weight = dequantize_bnb_weight(weight, state=quant_state)
    if is_cpu:
        module.weight = module.weight
    return weight


def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight
    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T


class DoraLinearLayer(nn.Module):

    def __init__(self, fan_in_fan_out):
        super().__init__()
        self.fan_in_fan_out = fan_in_fan_out

    def get_weight_norm(self, weight, lora_weight, scaling) ->torch.Tensor:
        weight = transpose(weight, self.fan_in_fan_out)
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1)
        return weight_norm

    def update_layer(self, *, base_layer, lora_A, lora_B, scaling, place_on_cpu=False) ->None:
        dtype_is_fp16 = lora_A.dtype == torch.float16
        if dtype_is_fp16:
            lora_A = lora_A.float()
            lora_B = lora_B.float()
        with gather_params_ctx(base_layer.parameters()):
            if base_layer.__class__.__name__ == 'Linear4bit':
                base_layer = deepcopy(base_layer)
            weight = dequantize_module_weight(base_layer)
            if weight.data.ndim >= 4:
                lora_weight = torch.mm(lora_B.flatten(start_dim=1), lora_A.flatten(start_dim=1))
                lora_weight = lora_weight.reshape(weight.shape)
            else:
                lora_weight = lora_B @ lora_A
            if dtype_is_fp16:
                lora_weight = lora_weight.half()
            weight_norm = self.get_weight_norm(weight, lora_weight, scaling)
        if place_on_cpu:
            weight_norm = weight_norm
        self.weight = nn.Parameter(weight_norm, requires_grad=True)

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        x_eye = torch.eye(lora_A.weight.shape[1], device=lora_A.weight.device, dtype=x.dtype)
        lora_weight = lora_B(lora_A(x_eye)).T
        magnitude = self.weight
        weight = dequantize_module_weight(base_layer)
        weight = weight
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling)
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        lora_result = lora_B(lora_A(x))
        bias = None
        if base_result is not None:
            bias = base_layer.bias
            if bias is not None:
                base_result = base_result - bias
        else:
            base_result = F.linear(x, transpose(weight, self.fan_in_fan_out))
        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scaling
        return result_dora

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'lora.dora.' + rep


def get_bnb_param_type(param: 'torch.nn.Parameter') ->Literal[False, '4bit', '8bit']:
    """Returns '4bit' or '8bit' if bitsandbytes parameter, else False"""
    if param.__class__.__name__ == 'Params4bit':
        return '4bit'
    if param.__class__.__name__ == 'Int8Params':
        return '8bit'
    return False


class MultiplicativeDropoutLayer(nn.Module):
    """
    Implements the multiplicative dropout layer for OFT.
    """

    def __init__(self, p=0.0):
        """
        Initializes the multiplicative dropout layer.

        Parameters:
        p (float): The probability of dropping out a block. Defaults to 0.0.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Applies multiplicative dropout to the input tensor.

        Parameters:
        x (Tensor): The input tensor of shape (D, H, H), where `D` represents
                    the number of OFT blocks, and `H` is the size of the square blocks along the last two dimensions,
                    the block size in OFT.
        """
        if self.training:
            if x.shape[-1] != x.shape[-2]:
                raise ValueError('The last two dimensions of input should be the same!')
            D, H, _ = x.shape
            if D == 1:
                return x
            num_to_replace = int(self.p * D)
            num_zeros = D - num_to_replace
            mask = torch.cat([torch.ones(num_to_replace, device=x.device), torch.zeros(num_zeros, device=x.device)])
            mask = mask[torch.randperm(D)].view(D, 1, 1)
            eye_matrix = torch.eye(H, device=x.device).repeat(D, 1, 1)
            x = (1 - mask) * x + mask * eye_matrix
        return x


def check_adapters_to_merge(module: 'BaseTunerLayer', adapter_names: 'Optional[list[str]]'=None) ->list[str]:
    """
    Helper function to check which adapters should be merged.

    Only return those adapters that are not already merged. Give a warning if some or all of the adapters are already
    merged.

    """
    if adapter_names is None:
        adapter_names = module.active_adapters
    if isinstance(adapter_names, str):
        raise ValueError(f'adapter_names should be a list of strings, got {adapter_names!r}.')
    if module.merged:
        merged_adapters = set(module.merged_adapters)
        adapter_names = [name for name in adapter_names if name not in merged_adapters]
        if adapter_names:
            warnings.warn(f"Already following adapters were merged {','.join(module.merged_adapters)}. You are now additionally merging {','.join(adapter_names)}.")
        else:
            warnings.warn('All adapters are already merged, nothing to do.')
    return adapter_names


def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    kwargs['adapter_names'] = adapter_names
    return args, kwargs


def _freeze_adapter(model, adapter_name):
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = False


def magnitude_based_pruning(tensor: 'torch.Tensor', density: 'float') ->torch.Tensor:
    """
    Prune the smallest values of the task tensors and retain the top-k values based on the specified fraction
    `density`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The tensor with the pruned weights.
    """
    mask = torch.zeros_like(tensor).reshape(-1)
    k = int(density * tensor.numel())
    top_k = torch.topk(tensor.abs().reshape(-1), k=k, largest=True)
    mask[top_k[1]] = 1
    return tensor * mask.reshape(tensor.shape)


def random_pruning(tensor: 'torch.Tensor', density: 'float', rescale: 'bool') ->torch.Tensor:
    """
    Prune random values based on the specified fraction `density`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.

    Returns:
        `torch.Tensor`: The pruned tensor.
    """
    mask = torch.bernoulli(torch.full_like(input=tensor, fill_value=density))
    pruned_tensor = tensor * mask
    if rescale:
        torch.div(input=pruned_tensor, other=density)
    return pruned_tensor


def prune(tensor: 'torch.Tensor', density: 'float', method: "Literal['magnitude', 'random']", rescale: 'bool'=False) ->torch.Tensor:
    """
    Prune the values of task tensors based on the `method`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        method (`str`):The method to use to prune. Should be one of ["magnitude", "random"].
        rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.

    Returns:
        `torch.Tensor`: The pruned tensor.
    """
    if density >= 1:
        warnings.warn(f'The density {density} is greater than or equal to 1, no pruning will be performed.')
        return tensor
    elif density < 0:
        raise ValueError(f'Density should be >= 0, got {density}')
    if method == 'magnitude':
        return magnitude_based_pruning(tensor, density)
    elif method == 'random':
        return random_pruning(tensor, density, rescale=rescale)
    else:
        raise ValueError(f'Unknown method {method}')


def reshape_weight_task_tensors(task_tensors, weights):
    """
    Reshapes `weights` to match the shape of `task_tensors` by unsqeezing in the remaining dimenions.

    Args:
        task_tensors (`torch.Tensor`): The tensors that will be used to reshape `weights`.
        weights (`torch.Tensor`): The tensor to be reshaped.

    Returns:
        `torch.Tensor`: The reshaped tensor.
    """
    new_shape = weights.shape + (1,) * (task_tensors.dim() - weights.dim())
    weights = weights.view(new_shape)
    return weights


def dare_linear(task_tensors: 'List[torch.Tensor]', weights: 'torch.Tensor', density: 'float') ->torch.Tensor:
    """
    Merge the task tensors using `dare linear`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = [prune(tensor, density, method='random', rescale=True) for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def calculate_majority_sign_mask(tensor: 'torch.Tensor', method: "Literal['total', 'frequency']"='total') ->torch.Tensor:
    """
    Get the mask of the majority sign across the task tensors. Task tensors are stacked on dimension 0.

    Args:
        tensor (`torch.Tensor`):The tensor to get the mask from.
        method (`str`):The method to use to get the mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The majority sign mask.
    """
    sign = tensor.sign()
    if method == 'total':
        sign_magnitude = tensor.sum(dim=0)
    elif method == 'frequency':
        sign_magnitude = sign.sum(dim=0)
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')
    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return sign == majority_sign


def disjoint_merge(task_tensors: 'torch.Tensor', majority_sign_mask: 'torch.Tensor') ->torch.Tensor:
    """
    Merge the task tensors using disjoint merge.

    Args:
        task_tensors (`torch.Tensor`):The task tensors to merge.
        majority_sign_mask (`torch.Tensor`):The mask of the majority sign across the task tensors.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    mixed_task_tensors = (task_tensors * majority_sign_mask).sum(dim=0)
    num_params_preserved = majority_sign_mask.sum(dim=0)
    return mixed_task_tensors / torch.clamp(num_params_preserved, min=1.0)


def dare_ties(task_tensors: 'List[torch.Tensor]', weights: 'torch.Tensor', density: 'float', majority_sign_method: "Literal['total', 'frequency']"='total') ->torch.Tensor:
    """
    Merge the task tensors using `dare ties`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        majority_sign_method (`str`):
            The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = [prune(tensor, density, method='random', rescale=True) for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors


def dispatch_aqlm(target: 'torch.nn.Module', adapter_name: 'str', **kwargs: Any) ->Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if is_aqlm_available() and isinstance(target_base_layer, QuantizedLinear):
        new_module = AqlmLoraLinear(target, adapter_name, **kwargs)
        target.qweight = target_base_layer.codes
    return new_module


def dispatch_awq(target: 'torch.nn.Module', adapter_name: 'str', **kwargs: Any) ->Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if is_auto_awq_available() and isinstance(target_base_layer, WQLinear_GEMM):
        AUTOAWQ_MINIMUM_VERSION = packaging.version.parse('0.2.0')
        version_autoawq = packaging.version.parse(importlib_metadata.version('autoawq'))
        if AUTOAWQ_MINIMUM_VERSION > version_autoawq:
            raise ImportError(f'Found an incompatible version of auto-awq. Found version {version_autoawq}, but only versions above {AUTOAWQ_MINIMUM_VERSION} are supported for PEFT.')
        new_module = AwqLoraLinear(target, adapter_name, **kwargs)
        target.qweight = target_base_layer.qweight
    return new_module


class _DoraConvNdLayer(DoraLinearLayer):

    def get_weight_norm(self, weight, lora_weight, scaling) ->torch.Tensor:
        weight = weight + scaling * lora_weight
        dim = tuple(range(1, weight.dim()))
        weight_norm = weight.norm(p=2, dim=dim, keepdim=True).transpose(1, 0)
        return weight_norm

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        weight = base_layer.weight
        lora_weight = torch.mm(lora_B.weight.flatten(start_dim=1), lora_A.weight.flatten(start_dim=1))
        lora_weight = lora_weight.reshape(weight.shape)
        magnitude = self.weight
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling)
        weight_norm = weight_norm.detach()
        mag_norm_scale = magnitude / weight_norm
        result_dora = (mag_norm_scale - 1) * self.conv_fn(x, weight, bias=None, stride=base_layer.stride, padding=base_layer.padding, dilation=base_layer.dilation, groups=base_layer.groups) + mag_norm_scale * lora_B(lora_A(x)) * scaling
        return result_dora

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'lora.dora.' + rep


class DoraConv3dLayer(_DoraConvNdLayer):

    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv3d


class DoraEmbeddingLayer(DoraLinearLayer):

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, embed_fn):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        lora_weight = (lora_A @ lora_B).T
        magnitude = self.weight
        weight = base_layer.weight
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling)
        weight_norm = weight_norm.detach()
        mag_norm_scale = magnitude / weight_norm
        result_dora = mag_norm_scale * (embed_fn(x, lora_A) @ lora_B) * scaling
        return mag_norm_scale, result_dora

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'lora.dora.' + rep


def dispatch_default(target: 'torch.nn.Module', adapter_name: 'str', lora_config: 'LoraConfig', **kwargs) ->Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop('fan_in_fan_out', None)
        embedding_kwargs.update(lora_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv3d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv3d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs['fan_in_fan_out']:
            warnings.warn('fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. Setting fan_in_fan_out to False.')
            kwargs['fan_in_fan_out'] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs['fan_in_fan_out']:
            warnings.warn('fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True.')
            kwargs['fan_in_fan_out'] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)
    return new_module


def dispatch_eetq(target: 'torch.nn.Module', adapter_name: 'str', **kwargs: Any) ->Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if is_eetq_available() and isinstance(target_base_layer, EetqLinear):
        new_module = EetqLoraLinear(target, adapter_name, **kwargs)
        target.weight = target_base_layer.weight
        if hasattr(target, 'bias'):
            target.bias = target_base_layer.bias
    return new_module


def dispatch_gptq(target: 'torch.nn.Module', adapter_name: 'str', **kwargs: Any) ->Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    gptq_quantization_config = kwargs.get('gptq_quantization_config', None)
    AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)
    if AutoGPTQQuantLinear is not None and isinstance(target_base_layer, AutoGPTQQuantLinear):
        new_module = QuantLinear(target, adapter_name, **kwargs)
        target.qweight = target_base_layer.qweight
    return new_module


def dispatch_hqq(target: 'torch.nn.Module', adapter_name: 'str', **kwargs):
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if is_hqq_available() and isinstance(target_base_layer, HQQLinear):
        new_module = HqqLoraLinear(target_base_layer, adapter_name, **kwargs)
    return new_module


def dispatch_megatron(target: 'torch.nn.Module', adapter_name: 'str', lora_config, **kwargs: Any) ->Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if lora_config.megatron_config:
        megatron_core = importlib.import_module(lora_config.megatron_core)
    else:
        megatron_core = None
    if megatron_core and isinstance(target_base_layer, (megatron_core.tensor_parallel.ColumnParallelLinear, megatron_core.tensor_parallel.RowParallelLinear)):
        megatron_kwargs = kwargs.copy()
        megatron_config = lora_config.megatron_config
        if isinstance(megatron_config, dict):
            transformer_config_class = megatron_core.transformer.transformer_config.TransformerConfig
            megatron_config = transformer_config_class(**lora_config.megatron_config)
        megatron_kwargs['megatron_config'] = megatron_config
        if megatron_kwargs['fan_in_fan_out']:
            warnings.warn('fan_in_fan_out is set to True but the target module is `ColumnParallelLinear` or `RowParallelLinear`. Setting fan_in_fan_out to False.')
            megatron_kwargs['fan_in_fan_out'] = lora_config.fan_in_fan_out = False
        new_module = LoraParallelLinear(base_layer=target, adapter_name=adapter_name, backend=megatron_core.tensor_parallel, **megatron_kwargs)
    return new_module


def dispatch_torchao(target: 'torch.nn.Module', adapter_name: 'str', lora_config: 'LoraConfig', **kwargs: Any) ->Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if not hasattr(target_base_layer, 'weight'):
        return new_module
    if not is_torchao_available():
        return new_module
    if isinstance(target_base_layer.weight, (AffineQuantizedTensor, LinearActivationQuantizedTensor)):
        new_module = TorchaoLoraLinear(target, adapter_name, **kwargs)
    return new_module


def get_pattern_key(pattern_keys, key_to_match):
    """Match a substring of key_to_match in pattern keys"""
    return next(filter(lambda key: re.match(f'.*\\.{key}$', key_to_match), pattern_keys), key_to_match)


def str_to_bool(value: 'str') ->int:
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


def check_file_exists_on_hf_hub(repo_id: 'str', filename: 'str', **kwargs) ->Optional[bool]:
    """Check if a file exists on HF Hub, if check was not successful returns None instead of erroring.

    Respect offline mode if set.

    """
    exists: 'Optional[bool]' = None
    if str_to_bool(os.environ.get('HF_HUB_OFFLINE', '0')):
        return exists
    try:
        exists = file_exists(repo_id, filename, **kwargs)
    except (HFValidationError, EntryNotFoundError):
        pass
    except Exception as e:
        warnings.warn(f'Unable to fetch remote file due to the following error {e} - silently ignoring the lookup for the file {filename} in {repo_id}.')
    return exists


def get_embedding_layer_name(model, layer, is_embedding_in_target_modules):
    """Get the name of the embedding module for a given layer."""
    for name, module in model.named_modules():
        if not is_embedding_in_target_modules and module == layer or module == getattr(layer, 'base_layer', None):
            return name
    return None


def has_valid_embedding_base_layer(layer):
    """Check if the layer has an embedding base layer"""
    return hasattr(layer, 'base_layer') and isinstance(layer.base_layer, (torch.nn.Linear, torch.nn.Embedding))


def get_peft_model_state_dict(model, state_dict=None, adapter_name='default', unwrap_compiled=False, save_embedding_layers='auto'):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
            the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the passed model will be used.
        adapter_name (`str`, *optional*, defaults to `"default"`):
            The name of the adapter whose state dict should be returned.
        unwrap_compiled (`bool`, *optional*, defaults to `False`):
            Whether to unwrap the model if torch.compile was used.
        save_embedding_layers (`Union[bool, str]`, , *optional*, defaults to `auto`):
            If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common embedding
            layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available. Based on it
            sets the boolean flag. This only works for 🤗 transformers models.
    """
    if unwrap_compiled:
        model = getattr(model, '_orig_mod', model)
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        bias = config.bias
        if bias == 'none':
            to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k}
        elif bias == 'all':
            to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k or 'bias' in k}
        elif bias == 'lora_only':
            to_return = {}
            for k in state_dict:
                if 'lora_' in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split('lora_')[0] + 'bias'
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {k: v for k, v in to_return.items() if 'lora_' in k and adapter_name in k or 'bias' in k}
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {k.replace(f'.{adapter_name}', ''): v for k, v in rank_pattern.items()}
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(rank_pattern, to_return, adapter_name)
        if config.use_dora:
            new_dora_suffix = f'lora_magnitude_vector.{adapter_name}.weight'

            def renamed_dora_weights(k):
                if k.endswith(new_dora_suffix):
                    k = k[:-7]
                return k
            to_return = {renamed_dora_weights(k): v for k, v in to_return.items()}
    elif config.peft_type == PeftType.BOFT:
        bias = config.bias
        if bias == 'none':
            to_return = {k: state_dict[k] for k in state_dict if 'boft_' in k}
        elif bias == 'all':
            to_return = {k: state_dict[k] for k in state_dict if 'boft_' in k or 'bias' in k}
        elif bias == 'boft_only':
            to_return = {}
            for k in state_dict:
                if 'boft_' in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split('boft_')[0] + 'bias'
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
    elif config.peft_type == PeftType.LOHA:
        to_return = {k: state_dict[k] for k in state_dict if 'hada_' in k}
    elif config.peft_type == PeftType.LOKR:
        to_return = {k: state_dict[k] for k in state_dict if 'lokr_' in k}
    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {k: state_dict[k] for k in state_dict if k.split('.')[-1].startswith('adaption_')}
    elif config.is_prompt_learning:
        to_return = {}
        if config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            to_return['prefix_task_cols'] = model.prompt_encoder[adapter_name].prefix_task_cols
            to_return['prefix_task_rows'] = model.prompt_encoder[adapter_name].prefix_task_rows
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        elif config.inference_mode:
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return['prompt_embeddings'] = prompt_embeddings
    elif config.peft_type == PeftType.IA3:
        to_return = {k: state_dict[k] for k in state_dict if 'ia3_' in k}
    elif config.peft_type == PeftType.OFT:
        to_return = {k: state_dict[k] for k in state_dict if 'oft_' in k}
    elif config.peft_type == PeftType.POLY:
        to_return = {k: state_dict[k] for k in state_dict if 'poly_' in k}
    elif config.peft_type == PeftType.LN_TUNING:
        to_return = {k: state_dict[k] for k in state_dict if 'ln_tuning_' in k}
    elif config.peft_type == PeftType.VERA:
        to_return = {k: state_dict[k] for k in state_dict if 'vera_lambda_' in k}
        if config.save_projection:
            if f'base_model.vera_A.{adapter_name}' not in state_dict:
                raise ValueError('Model was initialised to not save vera_A and vera_B but config now specifies to save projection! Set `config.save_projection` to `False`.')
            to_return['base_model.vera_A.' + adapter_name] = state_dict['base_model.vera_A.' + adapter_name]
            to_return['base_model.vera_B.' + adapter_name] = state_dict['base_model.vera_B.' + adapter_name]
    elif config.peft_type == PeftType.FOURIERFT:
        to_return = {k: state_dict[k] for k in state_dict if 'fourierft_' in k}
    elif config.peft_type == PeftType.XLORA:
        to_return = {k: state_dict[k] for k in state_dict if 'internal_xlora_classifier' in k}
    elif config.peft_type == PeftType.HRA:
        to_return = {k: state_dict[k] for k in state_dict if 'hra_' in k}
    elif config.peft_type == PeftType.VBLORA:
        to_return = {}
        if config.num_vectors < 2 ** 8:
            indices_dtype = torch.uint8
        elif config.num_vectors < 2 ** 15:
            indices_dtype = torch.int16
        elif config.num_vectors < 2 ** 31:
            indices_dtype = torch.int32
        else:
            indices_dtype = torch.int64
        if config.save_only_topk_weights:
            for k in state_dict:
                if 'vblora_logits' in k:
                    logits, indices = state_dict[k].topk(config.topk)
                    to_return.update({(k + '_topk_indices'): indices})
                    to_return.update({(k + '_topk_weights'): torch.softmax(logits, dim=-1)[:, :, :-1].contiguous()})
        else:
            to_return = {k: state_dict[k] for k in state_dict if 'vblora_logits' in k}
        to_return['base_model.vblora_vector_bank.' + adapter_name] = state_dict['base_model.vblora_vector_bank.' + adapter_name]
    elif config.peft_type == PeftType.BONE:
        to_return = {k: state_dict[k] for k in state_dict if 'bone_' in k}
    else:
        raise ValueError(f'Unknown PEFT type passed: {config.peft_type}')
    if getattr(model, 'modules_to_save', None) is not None:
        for key, value in state_dict.items():
            if any(f'{module_name}.modules_to_save.{adapter_name}' in key for module_name in model.modules_to_save):
                to_return[key.replace('modules_to_save.', '')] = value
    is_embedding_in_target_modules = False
    if save_embedding_layers == 'auto' and hasattr(config, 'target_modules') and any(k in config.target_modules for k in EMBEDDING_LAYER_NAMES):
        warnings.warn('Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.')
        save_embedding_layers = is_embedding_in_target_modules = True
    elif save_embedding_layers == 'auto':
        vocab_size = getattr(getattr(model, 'config', None), 'vocab_size', None)
        model_id = getattr(config, 'base_model_name_or_path', None)
        has_base_config = False
        if model_id is not None:
            local_config_exists = os.path.exists(os.path.join(model_id, 'config.json'))
            exists = local_config_exists or check_file_exists_on_hf_hub(model_id, 'config.json')
            if exists is None:
                warnings.warn(f'Could not find a config file in {model_id} - will assume that the vocabulary was not modified.')
                has_base_config = False
            else:
                has_base_config = exists
        if vocab_size and model_id and has_base_config and vocab_size != model.config.__class__.from_pretrained(model_id).vocab_size:
            warnings.warn('Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.')
            save_embedding_layers = True
        else:
            save_embedding_layers = False
    if save_embedding_layers and hasattr(model, 'get_input_embeddings'):
        for layer in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if not is_embedding_in_target_modules or has_valid_embedding_base_layer(layer):
                embedding_module_name = get_embedding_layer_name(model, layer, is_embedding_in_target_modules)
                if embedding_module_name:
                    to_return.update({k: v for k, v in state_dict.items() if embedding_module_name in k})
    elif save_embedding_layers:
        warnings.warn('Could not identify embedding layer(s) because the model is not a 🤗 transformers model.')
    to_return = {k.replace(f'.{adapter_name}', ''): v for k, v in to_return.items()}
    return to_return


def get_quantization_config(model: 'torch.nn.Module', method: 'str'):
    """
    Get the quantization config of the related quantization method
    """
    if hasattr(model, 'config') and hasattr(model.config, 'quantization_config') and getattr(model, 'quantization_method', None) == method:
        return model.config.quantization_config
    return None


def magnitude_prune(task_tensors: 'List[torch.Tensor]', weights: 'torch.Tensor', density: 'float') ->torch.Tensor:
    """
    Merge the task tensors using `task arithmetic`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`): The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = [prune(tensor, density, method='magnitude') for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def clone_module(module: 'nn.Module', share_weights=False):
    """Clone a module in a pytorch model.

    Clones a module of a model, optionally sharing all the parameters between the original and the clone. Simplifies
    reusing a module when manipulating the architecture of a model.
    """
    clone = copy.deepcopy(module)

    def _share_weights(src: 'nn.Module', dst: 'nn.Module'):
        for name, param in src.named_parameters(recurse=False):
            dst.register_parameter(name, param)
    if share_weights:
        for name, submodule in module.named_modules():
            _share_weights(submodule, clone.get_submodule(name))
    return clone


def replicate_layers(model: 'nn.Module', layer_map: 'list[tuple[int, int]]'):
    """Replicate layers in a transfomer model with weight sharing.

    This function looks for a module list attribute at model[(.model)*].layers and replicates the layers in the module
    list according to the layer map. For example the map `[[0, 4], [2, 5]]` will take the set of layers `[0, 1, 2, 3,
    4]` and replace them with a module list containing `[0, 1, 2, 3, 2, 3, 4]`.
    """
    while hasattr(model, 'model'):
        model = model.model
    if hasattr(model, 'bert'):
        model = model.bert
    model_type = None
    layers: 'nn.ModuleList' = None
    if hasattr(model, 'layers'):
        model_type = 'llama'
        layers = model.layers
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        model_type = 'bert'
        layers = model.encoder.layer
    elif hasattr(model, 'h'):
        model_type = 'falcon'
        layers = model.h
    if not model_type or not isinstance(layers, nn.ModuleList):
        raise ValueError('Could not locate the layers attribute in the model. Expected Llama, Bert or Falcon compatible architectures.')
    new_layers = []
    for start, end in layer_map:
        for i in range(start, end):
            current_idx = len(new_layers)
            new_layers.append(clone_module(layers[i], share_weights=True))
            for submodule in new_layers[-1].modules():
                if hasattr(submodule, 'layer_idx'):
                    submodule.layer_idx = current_idx
    layers = nn.ModuleList(new_layers)
    if model_type == 'llama':
        model.layers = layers
    elif model_type == 'bert':
        model.encoder.layer = layers
    elif model_type == 'falcon':
        model.h = layers
    else:
        raise ValueError('Unexpected model type, need to handle post-processing of layers.')
    if hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = len(new_layers)


def task_arithmetic(task_tensors: 'List[torch.Tensor]', weights: 'torch.Tensor') ->torch.Tensor:
    """
    Merge the task tensors using `task arithmetic`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = torch.stack(task_tensors, dim=0)
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def ties(task_tensors: 'List[torch.Tensor]', weights: 'torch.Tensor', density: 'float', majority_sign_method: "Literal['total', 'frequency']"='total') ->torch.Tensor:
    """
    Merge the task tensors using `ties`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        majority_sign_method (`str`):
            The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = [prune(tensor, density, method='magnitude') for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors


class RankAllocator:
    """
    The RankAllocator for AdaLoraModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.

    """

    def __init__(self, model, peft_config, adapter_name):
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1
        self.reset_ipt()
        self._set_budget_scheduler(model)

    def set_total_step(self, total_step):
        self.peft_config.total_step = total_step

    def reset_ipt(self):
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

    def _set_budget_scheduler(self, model):
        self.init_bgt = 0
        self.name_set = set()
        for n, p in model.named_parameters():
            if f'lora_A.{self.adapter_name}' in n:
                self.init_bgt += p.size(0)
                self.name_set.add(n.replace('lora_A', '%s'))
        self.name_set = sorted(self.name_set)
        self.target_bgt = self.peft_config.target_r * len(self.name_set)

    def budget_schedule(self, step: 'int'):
        tinit = self.peft_config.tinit
        tfinal = self.peft_config.tfinal
        total_step = self.peft_config.total_step
        if step <= tinit:
            budget = self.init_bgt
            mask_ind = False
        elif step > total_step - tfinal:
            budget = self.target_bgt
            mask_ind = True
        else:
            mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
            budget = int((self.init_bgt - self.target_bgt) * mul_coeff ** 3 + self.target_bgt)
            mask_ind = True if step % self.peft_config.deltaT == 0 else False
        return budget, mask_ind

    def update_ipt(self, model):
        for n, p in model.named_parameters():
            if 'lora_' in n and self.adapter_name in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    if deepspeed_config() is not None:
                        grad = deepspeed.utils.safe_get_full_grad(p)
                        self.ipt[n] = (p * grad).abs().detach()
                    else:
                        self.ipt[n] = (p * p.grad).abs().detach()
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()

    def _element_score(self, n):
        return self.exp_avg_ipt[n] * self.exp_avg_unc[n]

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_budget(self, model, budget):
        value_ipt = {}
        vector_ipt = {}
        triplet_ipt = {}
        for n, p in model.named_parameters():
            if f'lora_A.{self.adapter_name}' in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
                name_m = n.replace('lora_A', '%s')
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f'lora_B.{self.adapter_name}' in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                name_m = n.replace('lora_B', '%s')
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f'lora_E.{self.adapter_name}' in n:
                entry_ipt = self._element_score(n)
                name_m = n.replace('lora_E', '%s')
                value_ipt[name_m] = entry_ipt
        all_score = []
        for name_m in vector_ipt:
            ipt_E = value_ipt[name_m]
            ipt_AB = torch.cat(vector_ipt[name_m], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_m % 'lora_E'
            triplet_ipt[name_E] = sum_ipt.view(-1, 1)
            all_score.append(sum_ipt.view(-1))
        mask_threshold = torch.kthvalue(torch.cat(all_score), k=self.init_bgt - budget)[0].item()
        rank_pattern = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if f'lora_E.{self.adapter_name}' in n:
                    p.masked_fill_(triplet_ipt[n] <= mask_threshold, 0.0)
                    rank_pattern[n] = (~(triplet_ipt[n] <= mask_threshold)).view(-1).tolist()
        return rank_pattern

    def update_and_allocate(self, model, global_step, force_mask=False):
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            self.update_ipt(model)
        budget, mask_ind = self.budget_schedule(global_step)
        if mask_ind or force_mask:
            rank_pattern = self.mask_to_budget(model, budget)
        else:
            rank_pattern = None
        return budget, rank_pattern

    def mask_using_rank_pattern(self, model, rank_pattern):
        is_adapter_name_truncated = False
        if self.adapter_name not in next(iter(rank_pattern.keys())):
            is_adapter_name_truncated = True
        with torch.no_grad():
            for n, p in model.named_parameters():
                if f'lora_E.{self.adapter_name}' in n:
                    key = n if not is_adapter_name_truncated else n.replace(f'.{self.adapter_name}', '')
                    mask = torch.Tensor(rank_pattern[key]).unsqueeze(-1)
                    p.masked_fill_(~mask.bool(), 0.0)


TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING = {'t5': ['q', 'k', 'v', 'o', 'wi', 'wo'], 'mt5': ['q', 'k', 'v', 'o', 'wi_0', 'wi_1', 'wo'], 'bart': ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'], 'gpt2': ['c_attn'], 'bloom': ['query_key_value'], 'opt': ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'], 'gptj': ['q_proj', 'v_proj'], 'gpt_neox': ['query_key_value'], 'gpt_neo': ['q_proj', 'v_proj'], 'llama': ['q_proj', 'v_proj'], 'bert': ['query', 'value'], 'roberta': ['query', 'key', 'value', 'dense'], 'deberta-v2': ['query_proj', 'key_proj', 'value_proj', 'dense'], 'gpt_bigcode': ['c_attn'], 'deberta': ['in_proj'], 'qwen2': ['q_proj', 'v_proj']}


def llama_rotate_half(x: 'torch.Tensor') ->torch.Tensor:
    """
    Rotate half the hidden dims of the input.

    This function was duplicated verbatim from:
    https://github.com/huggingface/transformers/blob/1de8ce9ee1191ba761a593ac15d9ccbf5851bfc5/src/transformers/models/llama/modeling_llama.py#L126

    This was done to eliminate the Llama transformers implementation as a dependency of this file. Note that some other
    functions were also adapted from the transformers implementation but were modified.
    """
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def llama_apply_rotary_pos_emb(q, cos, sin, position_ids):
    """
    Apply rotary position embedding to query states in the Llama model.

    This function was adapted from:
    https://github.com/huggingface/transformers/blob/1de8ce9ee1191ba761a593ac15d9ccbf5851bfc5/src/transformers/models/llama/modeling_llama.py#L133

    It was modified to remove unnecessary processing of key states. The method is compatible with transformers <=
    4.34.2 and also with the latest version (>=4.35).
    """
    if len(cos.shape) == 4:
        gather_indices = position_ids[:, None, :, None]
        gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
        cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
        sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    q_embed = q * cos + llama_rotate_half(q) * sin
    return q_embed


def llama_compute_query_states(model: 'nn.Module', **kwargs) ->torch.Tensor:
    """
    Compute query states for Llama models specifically. They need to be recomputed as the forward() method of the
    original LlamaModel in the transformers library does not return them. See the related discussion in the PR:
    https://github.com/huggingface/peft/pull/268
    """
    hidden_states = kwargs.get('hidden_states')
    position_ids = kwargs.get('position_ids')
    past_key_value = kwargs.get('past_key_value')
    bsz, q_len, _ = hidden_states.size()
    query_states = model.q_proj(hidden_states).view(bsz, q_len, model.num_heads, model.head_dim).transpose(1, 2)
    factor = model.k_proj.in_features // model.k_proj.out_features
    value_states = model.v_proj(hidden_states).view(bsz, q_len, model.num_heads // factor, model.head_dim).transpose(1, 2)
    seq_len = q_len
    if past_key_value is not None:
        if isinstance(past_key_value, tuple):
            seq_len += past_key_value[0].shape[-2]
        else:
            seq_len += past_key_value.get_seq_length(model.layer_idx)
    if 'position_ids' not in inspect.signature(model.rotary_emb.forward).parameters:
        cos, sin = model.rotary_emb(value_states, seq_len=seq_len)
        return llama_apply_rotary_pos_emb(query_states, cos, sin, position_ids)
    past_seen_tokens = 0
    if position_ids is None:
        if past_key_value is None:
            new_cache_positions = torch.arange(q_len, q_len + q_len, device=value_states.device)
        else:
            past_seen_tokens = past_key_value.get_usable_length(q_len, model.layer_idx)
            new_cache_positions = torch.arange(past_seen_tokens, past_seen_tokens + q_len, device=value_states.device)
        position_ids = new_cache_positions.unsqueeze(0)
    rotary_emb_kwargs = {'position_ids': position_ids}
    if 'seq_len' in inspect.signature(model.rotary_emb.forward).parameters:
        rotary_emb_kwargs['seq_len'] = q_len + past_seen_tokens
    cos, sin = model.rotary_emb(value_states, **rotary_emb_kwargs)
    if len(cos.shape) == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return query_states * cos + llama_rotate_half(query_states) * sin


class AdaptedAttention(nn.Module):
    """This module wraps a LLamaAttention module and injects adaption prompts."""

    def __init__(self, model_type: 'str', adapter_len: 'int', model):
        """
        Initialize object.

        Args:
            model_type: The transformer model type. This is used to retrieve the right method to
                compute query states.
            adapter_len: The length of the adaption prompt to insert.
            model: The original transformer attention module that is being wrapped.
        """
        assert not isinstance(model, AdaptedAttention)
        super().__init__()
        self.model_type = model_type
        self.model = model
        self.adapter_len = adapter_len
        device = next(model.parameters()).device
        target_dtype = model.q_proj.weight.dtype if model.q_proj.weight.dtype not in [torch.int8, torch.uint8] else torch.float32
        self.adaption_prompt = nn.Parameter(torch.empty(1, adapter_len, self.model.hidden_size, device=device, dtype=target_dtype).normal_())
        self.adaption_gate = nn.Parameter(torch.zeros(1, device=device, dtype=target_dtype))

    def forward(self, **kwargs):
        """
        Forward pass for the adapter which wraps the original LlamaAttention module.

        "Official" paper implementation:
        https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L141

        Args:
            kwargs: See the original LlamaAttention module.
        """
        if kwargs.get('output_attention', False):
            raise NotImplementedError('output_attention is not currently supported.')
        output, _, past_key_value = self.model(**kwargs)
        bsz = output.shape[0]
        q_len = output.shape[1]
        embed_dim = output.shape[2]
        k_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].k_proj_layer
        v_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].v_proj_layer
        o_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].o_proj_layer
        factor = self.model.k_proj.in_features // self.model.k_proj.out_features
        if k_proj_layer == v_proj_layer:
            _, key, value = getattr(self.model, k_proj_layer)(self.adaption_prompt).split(embed_dim, dim=2)
        else:
            key = getattr(self.model, k_proj_layer)(self.adaption_prompt)
            value = getattr(self.model, v_proj_layer)(self.adaption_prompt)
        adapter_k = key.view(1, self.adapter_len, self.model.num_heads // factor, self.model.head_dim).repeat(bsz, 1, 1, 1).transpose(1, 2)
        adapter_v = value.view(1, self.adapter_len, self.model.num_heads // factor, self.model.head_dim).repeat(bsz, 1, 1, 1).transpose(1, 2)
        adapter_k = torch.repeat_interleave(adapter_k, repeats=factor, dim=1)
        adapter_v = torch.repeat_interleave(adapter_v, repeats=factor, dim=1)
        compute_query_states = TRANSFORMERS_MODEL_CONFIG[self.model_type].compute_query_states
        query_states = compute_query_states(model=self.model, **kwargs)
        previous_dtype = query_states.dtype
        scores = torch.matmul(query_states, adapter_k.transpose(2, 3)) / math.sqrt(self.model.head_dim)
        scores = self.adaption_gate * F.softmax(scores, dim=-1, dtype=torch.float32)
        adapter_output = torch.matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)
        if o_proj_layer is not None:
            adapter_output = getattr(self.model, o_proj_layer)(adapter_output)
        output = output + adapter_output
        output = output
        return output, None, past_key_value


def is_adaption_prompt_trainable(params: 'str') ->bool:
    """Return True if module is trainable under adaption prompt fine-tuning."""
    return params.split('.')[-1].startswith('adaption_')


class AdaptionPromptModel(nn.Module):
    """
    Implements adaption prompts as described in https://arxiv.org/pdf/2303.16199.pdf.

    The top L attention modules are replaced with AdaptedAttention modules that wrap the original ones, but insert
    trainable prompts with gates (for zero init).

    Notes on the multi-adapter pattern:
    - We store the states of different adapters by keeping a dictionary of AdaptedAttention modules indexed by adapter
      name.
    - Every time we switch adapters, we remove the modules of the currently active adapter from the model, store them
      in the dictionary, and replace them with the modules of the new adapter.
    - To avoid duplicated and potentially inconsistent state, the currently active adapter is always removed from the
      dictionary.
    - Disabling the adapter would also result in the modules being removed from the model.
    """

    def __init__(self, model, configs: 'Dict', adapter_name: 'str'):
        super().__init__()
        self.model = model
        self.peft_config: 'Dict[str, AdaptionPromptConfig]' = {}
        self._parents: 'Dict[str, List[nn.Module]]' = {}
        self._cached_adapters: 'Dict[str, List]' = {}
        self._active_adapter = None
        self._enabled = True
        self.forward = self.model.forward
        self.add_adapter(adapter_name, configs[adapter_name])
        self._mark_only_adaption_prompts_as_trainable(self.model)

    def add_adapter(self, adapter_name: 'str', config: 'AdaptionPromptConfig') ->None:
        """Add an adapter with the given name and config."""
        config = prepare_config(config, self.model)
        if adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
        parents = []
        for name, _ in self.model.named_modules():
            if name.endswith(config.target_modules):
                par, _, _ = _get_submodules(self.model, name)
                parents.append(par)
        if len(parents) < config.adapter_layers:
            raise ValueError(f"Config specifies more adapter layers '{config.adapter_layers}' than the model has '{len(parents)}'.")
        parents = parents[-config.adapter_layers:]
        self._parents[adapter_name] = parents
        if self._active_adapter is not None and self._enabled:
            self._remove_adapted_attentions(self._active_adapter)
        self._active_adapter = adapter_name
        self.peft_config[adapter_name] = config
        self._create_adapted_attentions(config, parents)
        if not self._enabled:
            self._remove_adapted_attentions(self._active_adapter)
        if config.inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def set_adapter(self, adapter_name: 'str') ->None:
        """Set the model to use the adapter with the given name."""
        if self._active_adapter == adapter_name:
            return
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter with name '{adapter_name}' does not exist.")
        if self._enabled:
            self._remove_adapted_attentions(self._active_adapter)
            self._set_adapted_attentions(adapter_name)
        self._active_adapter = adapter_name

    def enable_adapter_layers(self):
        """Enable adapter layers by swapping in cached AdaptedAttention modules."""
        self._enabled = True
        self._set_adapted_attentions(self._active_adapter)

    def disable_adapter_layers(self):
        """Disable adapter layers by swapping out AdaptedAttention modules."""
        self._enabled = False
        self._remove_adapted_attentions(self._active_adapter)

    def _create_adapted_attentions(self, config: 'AdaptionPromptConfig', parents: 'List[nn.Module]') ->None:
        """Wrap LlamaAttention modules with newly created AdaptedAttention modules."""
        for par in parents:
            attn = AdaptedAttention(model_type=self.model.config.model_type, adapter_len=config.adapter_len, model=getattr(par, config.target_modules))
            setattr(par, config.target_modules, attn)

    def _set_adapted_attentions(self, adapter_name: 'str') ->None:
        """Replace LlamaAttention modules with cached AdaptedAttention modules."""
        cached = self._cached_adapters[adapter_name]
        del self._cached_adapters[adapter_name]
        config = self.peft_config[adapter_name]
        for i, par in enumerate(self._parents[adapter_name]):
            setattr(par, config.target_modules, cached[i])

    def _remove_adapted_attentions(self, adapter_name: 'str') ->None:
        """Remove AdaptedAttention modules from the model and store them in the cache."""
        config = self.peft_config[adapter_name]
        adapted_attentions = []
        for par in self._parents[adapter_name]:
            attn = getattr(par, config.target_modules)
            adapted_attentions.append(attn)
            setattr(par, config.target_modules, attn.model)
        self._cached_adapters[adapter_name] = adapted_attentions

    def _mark_only_adaption_prompts_as_trainable(self, model: 'nn.Module') ->None:
        """Freeze all parameters of the model except the adaption prompts."""
        for n, p in model.named_parameters():
            if not is_adaption_prompt_trainable(n):
                p.requires_grad = False

    def __getattr__(self, name: 'str'):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == 'model':
                raise
            return getattr(self.model, name)


def get_fbd_cuda():
    global _FBD_CUDA
    if _FBD_CUDA is not None:
        return _FBD_CUDA
    from torch.utils.cpp_extension import load
    curr_dir = os.path.dirname(__file__)
    try:
        with patch_environment(CC='gcc', CXX='gcc'):
            fbd_cuda = load(name='fbd_cuda', sources=[f'{curr_dir}/fbd/fbd_cuda.cpp', f'{curr_dir}/fbd/fbd_cuda_kernel.cu'], verbose=True)
    except Exception as e:
        warnings.warn(f'Failed to load the CUDA extension: {e}, check if ninja is available.')
        warnings.warn('Setting boft_n_butterfly_factor to 1 to speed up the finetuning process.')
        fbd_cuda = None
    _FBD_CUDA = fbd_cuda
    return _FBD_CUDA


TRANSFORMERS_MODELS_TO_FOURIERFT_TARGET_MODULES_MAPPING = {'t5': ['q', 'v'], 'mt5': ['q', 'v'], 'bart': ['q_proj', 'v_proj'], 'gpt2': ['mlp.c_proj'], 'bloom': ['query_key_value'], 'blip-2': ['q', 'v', 'q_proj', 'v_proj'], 'opt': ['q_proj', 'v_proj'], 'gptj': ['q_proj', 'v_proj'], 'gpt_neox': ['query_key_value'], 'gpt_neo': ['q_proj', 'v_proj'], 'bert': ['query', 'value'], 'roberta': ['query', 'value'], 'xlm-roberta': ['query', 'value'], 'electra': ['query', 'value'], 'deberta-v2': ['query_proj', 'value_proj'], 'deberta': ['in_proj'], 'layoutlm': ['query', 'value'], 'llama': ['q_proj', 'v_proj'], 'chatglm': ['query_key_value'], 'gpt_bigcode': ['mlp.c_proj'], 'mpt': ['Wqkv'], 'RefinedWebModel': ['query_key_value'], 'RefinedWeb': ['query_key_value'], 'falcon': ['query_key_value'], 'codegen': ['qkv_proj'], 'mistral': ['q_proj', 'v_proj'], 'mixtral': ['q_proj', 'v_proj'], 'stablelm': ['q_proj', 'v_proj'], 'phi': ['q_proj', 'v_proj', 'fc1', 'fc2'], 'gemma': ['q_proj', 'v_proj'], 'gemma2': ['q_proj', 'v_proj'], 'qwen2': ['q_proj', 'v_proj']}


TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING = {'t5': ['wo'], 'mt5': [], 'gpt2': ['mlp.c_proj'], 'bloom': ['mlp.dense_4h_to_h'], 'roberta': ['output.dense'], 'opt': ['fc2'], 'gptj': ['fc_out'], 'gpt_neox': ['dense_4h_to_h'], 'gpt_neo': ['c_proj'], 'bart': ['fc2'], 'gpt_bigcode': ['mlp.c_proj'], 'llama': ['down_proj'], 'mistral': ['down_proj'], 'mixtral': ['w2'], 'bert': ['output.dense'], 'deberta-v2': ['output.dense'], 'deberta': ['output.dense'], 'RefinedWeb': ['dense_4h_to_h'], 'RefinedWebModel': ['dense_4h_to_h'], 'falcon': ['dense_4h_to_h'], 'phi': ['fc2'], 'gemma': ['down_proj'], 'gemma2': ['down_proj'], 'qwen2': ['down_proj']}


TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING = {'t5': ['k', 'v', 'wo'], 'mt5': ['k', 'v', 'wi_1'], 'gpt2': ['c_attn', 'mlp.c_proj'], 'bloom': ['query_key_value', 'mlp.dense_4h_to_h'], 'roberta': ['key', 'value', 'output.dense'], 'opt': ['q_proj', 'k_proj', 'fc2'], 'gptj': ['q_proj', 'v_proj', 'fc_out'], 'gpt_neox': ['query_key_value', 'dense_4h_to_h'], 'gpt_neo': ['q_proj', 'v_proj', 'c_proj'], 'bart': ['q_proj', 'v_proj', 'fc2'], 'gpt_bigcode': ['c_attn', 'mlp.c_proj'], 'llama': ['k_proj', 'v_proj', 'down_proj'], 'mistral': ['k_proj', 'v_proj', 'down_proj'], 'mixtral': ['k_proj', 'v_proj', 'w2'], 'bert': ['key', 'value', 'output.dense'], 'deberta-v2': ['key_proj', 'value_proj', 'output.dense'], 'deberta': ['in_proj', 'output.dense'], 'RefinedWebModel': ['query_key_value', 'dense_4h_to_h'], 'RefinedWeb': ['query_key_value', 'dense_4h_to_h'], 'falcon': ['query_key_value', 'dense_4h_to_h'], 'phi': ['q_proj', 'v_proj', 'fc2'], 'gemma': ['q_proj', 'v_proj', 'down_proj'], 'gemma2': ['q_proj', 'v_proj', 'down_proj'], 'qwen2': ['q_proj', 'v_proj', 'down_proj']}


TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING = {'llama': ['input_layernorm', 'post_attention_layernorm', 'norm'], 'bloom': ['input_layernorm', 'post_attention_layernorm', 'ln_f'], 'llava': ['multi_modal_projector', 'input_layernorm', 'post_attention_layernorm', 'norm', 'embed_tokens', 'lm_head'], 't5': ['layer_norm', 'final_layer_norm'], 'mt5': ['layer_norm', 'final_layer_norm'], 'bart': ['self_attn_layer_norm', 'encoder_attn_layer_norm', 'final_layer_norm'], 'gpt2': ['ln_1', 'ln_2', 'ln_f'], 'blip-2': ['layernorm', 'LayerNorm', 'final_layer_norm', 'self_attn_layer_norm'], 'gptj': ['ln_1', 'ln_f'], 'falcon': ['input_layernorm', 'post_attention_layernorm', 'ln_f'], 'mistral': ['input_layernorm', 'post_attention_layernorm', 'norm'], 'phi': ['input_layernorm', 'final_layernorm'], 'gemma': ['input_layernorm', 'post_attention_layernorm', 'norm'], 'gemma2': ['input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm', 'post_feedforward_layernorm', 'norm'], 'qwen2': ['post_attention_layernorm']}


class HadaWeight(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w1a, w1b, w2a, w2b, scale=torch.tensor(1)):
        ctx.save_for_backward(w1a, w1b, w2a, w2b, scale)
        diff_weight = w1a @ w1b * (w2a @ w2b) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        w1a, w1b, w2a, w2b, scale = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out * (w2a @ w2b)
        grad_w1a = temp @ w1b.T
        grad_w1b = w1a.T @ temp
        temp = grad_out * (w1a @ w1b)
        grad_w2a = temp @ w2b.T
        grad_w2b = w2a.T @ temp
        del temp
        return grad_w1a, grad_w1b, grad_w2a, grad_w2b, None


def make_weight(w1a, w1b, w2a, w2b, scale):
    return HadaWeight.apply(w1a, w1b, w2a, w2b, scale)


def make_weight_cp(t, wa, wb):
    rebuild2 = torch.einsum('i j k l, i p, j r -> p r k l', t, wa, wb)
    return rebuild2


def factorization(dimension: 'int', factor: 'int'=-1) ->Tuple[int, int]:
    """Factorizes the provided number into the product of two numbers

    Args:
        dimension (`int`): The number that needs to be factorized.
        factor (`int`, optional):
            Factorization divider. The algorithm will try to output two numbers, one of each will be as close to the
            factor as possible. If -1 is provided, the decomposition algorithm would try to search dividers near the
            square root of the dimension. Defaults to -1.

    Returns:
        Tuple[`int`, `int`]: A tuple of two numbers, whose product is equal to the provided number. The first number is
        always less than or equal to the second.

    Example:
        ```py
        >>> factorization(256, factor=-1)
        (16, 16)

        >>> factorization(128, factor=-1)
        (8, 16)

        >>> factorization(127, factor=-1)
        (1, 127)

        >>> factorization(128, factor=4)
        (4, 32)
        ```
    """
    if factor > 0 and dimension % factor == 0:
        m = factor
        n = dimension // factor
        return m, n
    if factor == -1:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_kron(w1, w2, scale=1.0):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    return rebuild * scale


EPS = 1e-12


class Router(nn.Module, ABC):

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def forward(self, task_ids: 'torch.Tensor', input_ids: 'torch.Tensor'):
        ...


class PolyRouter(Router):

    def __init__(self, poly_config: 'PolyConfig'):
        super().__init__()
        self.poly_type = poly_config.poly_type
        self.n_tasks = poly_config.n_tasks
        self.n_skills = poly_config.n_skills
        self.n_splits = poly_config.n_splits
        self.module_logits = nn.Parameter(torch.empty((self.n_tasks, self.n_splits * self.n_skills)))

    def reset(self):
        torch.nn.init.uniform_(self.module_logits, -0.001, 0.001)

    def forward(self, task_ids: 'torch.Tensor', input_ids: 'torch.Tensor'):
        if task_ids is None:
            raise ValueError('task_ids should not be None.')
        if task_ids.max().item() >= self.n_tasks:
            raise ValueError(f'Only {self.n_tasks} tasks available. Found task id = {task_ids.max().item()}')
        task_ids = task_ids
        module_logits = self.module_logits[task_ids]
        module_logits = module_logits.view(-1, self.n_splits, self.n_skills)
        if self.training:
            module_logits = RelaxedBernoulli(temperature=1.0, logits=module_logits).rsample()
        else:
            module_logits = torch.sigmoid(module_logits)
        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)
        return module_weights


def get_router(poly_config: 'PolyConfig') ->nn.Module:
    if poly_config.poly_type == 'poly':
        return PolyRouter(poly_config)
    else:
        raise ValueError(f'Unsupported poly_type: {poly_config.poly_type}. Currently, only the following types are supported: `poly`.')


class PrefixEncoder(torch.nn.Module):
    """
    The `torch.nn` model to encode the prefix.

    Args:
        config ([`PrefixTuningConfig`]): The configuration of the prefix encoder.

    Example:

    ```py
    >>> from peft import PrefixEncoder, PrefixTuningConfig

    >>> config = PrefixTuningConfig(
    ...     peft_type="PREFIX_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     encoder_hidden_size=768,
    ... )
    >>> prefix_encoder = PrefixEncoder(config)
    ```

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prefix encoder.
        - **transform** (`torch.nn.Sequential`) -- The two-layer MLP to transform the prefix embeddings if
          `prefix_projection` is `True`.
        - **prefix_projection** (`bool`) -- Whether to project the prefix embeddings.

    Input shape: (`batch_size`, `num_virtual_tokens`)

    Output shape: (`batch_size`, `num_virtual_tokens`, `2*layers*hidden`)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        num_layers = config.num_layers
        encoder_hidden_size = config.encoder_hidden_size
        num_virtual_tokens = config.num_virtual_tokens
        if self.prefix_projection and not config.inference_mode:
            self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
            self.transform = torch.nn.Sequential(torch.nn.Linear(token_dim, encoder_hidden_size), torch.nn.Tanh(), torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim))
        else:
            self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)

    def forward(self, prefix: 'torch.Tensor'):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.transform(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class PromptTuningInit(str, enum.Enum):
    TEXT = 'TEXT'
    RANDOM = 'RANDOM'


class PromptEmbedding(torch.nn.Module):
    """
    The model to encode virtual tokens into prompt embeddings.

    Args:
        config ([`PromptTuningConfig`]): The configuration of the prompt embedding.
        word_embeddings (`torch.nn.Module`): The word embeddings of the base transformer model.

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt embedding.

    Example:

    ```py
    >>> from peft import PromptEmbedding, PromptTuningConfig

    >>> config = PromptTuningConfig(
    ...     peft_type="PROMPT_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     prompt_tuning_init="TEXT",
    ...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
    ...     tokenizer_name_or_path="t5-base",
    ... )

    >>> # t5_model.shared is the word embeddings of the base model
    >>> prompt_embedding = PromptEmbedding(config, t5_model.shared)
    ```

    Input Shape: (`batch_size`, `total_virtual_tokens`)

    Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """

    def __init__(self, config, word_embeddings):
        super().__init__()
        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)
        if config.prompt_tuning_init == PromptTuningInit.TEXT and not config.inference_mode:
            tokenizer_kwargs = config.tokenizer_kwargs or {}
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, **tokenizer_kwargs)
            init_text = config.prompt_tuning_init_text
            init_token_ids = tokenizer(init_text)['input_ids']
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]
            init_token_ids = torch.LongTensor(init_token_ids)
            with gather_params_ctx(word_embeddings.parameters()):
                word_embedding_weights = word_embeddings(init_token_ids).detach().clone()
            word_embedding_weights = word_embedding_weights
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

    def forward(self, indices):
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings


class PromptEncoderReparameterizationType(str, enum.Enum):
    MLP = 'MLP'
    LSTM = 'LSTM'


class PromptEncoder(torch.nn.Module):
    """
    The prompt encoder network that is used to generate the virtual token embeddings for p-tuning.

    Args:
        config ([`PromptEncoderConfig`]): The configuration of the prompt encoder.

    Example:

    ```py
    >>> from peft import PromptEncoder, PromptEncoderConfig

    >>> config = PromptEncoderConfig(
    ...     peft_type="P_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     encoder_reparameterization_type="MLP",
    ...     encoder_hidden_size=768,
    ... )

    >>> prompt_encoder = PromptEncoder(config)
    ```

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt encoder.
        - **mlp_head** (`torch.nn.Sequential`) -- The MLP head of the prompt encoder if `inference_mode=False`.
        - **lstm_head** (`torch.nn.LSTM`) -- The LSTM head of the prompt encoder if `inference_mode=False` and
        `encoder_reparameterization_type="LSTM"`.
        - **token_dim** (`int`) -- The hidden embedding dimension of the base transformer model.
        - **input_size** (`int`) -- The input size of the prompt encoder.
        - **output_size** (`int`) -- The output size of the prompt encoder.
        - **hidden_size** (`int`) -- The hidden size of the prompt encoder.
        - **total_virtual_tokens** (`int`): The total number of virtual tokens of the
        prompt encoder.
        - **encoder_type** (Union[[`PromptEncoderReparameterizationType`], `str`]): The encoder type of the prompt
          encoder.


    Input shape: (`batch_size`, `total_virtual_tokens`)

    Output shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """

    def __init__(self, config):
        super().__init__()
        self.token_dim = config.token_dim
        self.input_size = self.token_dim
        self.output_size = self.token_dim
        self.hidden_size = config.encoder_hidden_size
        self.total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.encoder_type = config.encoder_reparameterization_type
        self.embedding = torch.nn.Embedding(self.total_virtual_tokens, self.token_dim)
        if not config.inference_mode:
            if self.encoder_type == PromptEncoderReparameterizationType.LSTM:
                lstm_dropout = config.encoder_dropout
                num_layers = config.encoder_num_layers
                self.lstm_head = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layers, dropout=lstm_dropout, bidirectional=True, batch_first=True)
                self.mlp_head = torch.nn.Sequential(torch.nn.Linear(self.hidden_size * 2, self.hidden_size * 2), torch.nn.ReLU(), torch.nn.Linear(self.hidden_size * 2, self.output_size))
            elif self.encoder_type == PromptEncoderReparameterizationType.MLP:
                encoder_num_layers_default = PromptEncoderConfig.encoder_num_layers
                if config.encoder_num_layers != encoder_num_layers_default:
                    warnings.warn(f'for {self.encoder_type.value}, the argument `encoder_num_layers` is ignored. Exactly {encoder_num_layers_default} MLP layers are used.')
                layers = [torch.nn.Linear(self.input_size, self.hidden_size), torch.nn.ReLU(), torch.nn.Linear(self.hidden_size, self.hidden_size), torch.nn.ReLU(), torch.nn.Linear(self.hidden_size, self.output_size)]
                self.mlp_head = torch.nn.Sequential(*layers)
            else:
                raise ValueError('Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.')

    def forward(self, indices):
        input_embeds = self.embedding(indices)
        if self.encoder_type == PromptEncoderReparameterizationType.LSTM:
            output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        elif self.encoder_type == PromptEncoderReparameterizationType.MLP:
            output_embeds = self.mlp_head(input_embeds)
        else:
            raise ValueError('Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.')
        return output_embeds


TRANSFORMERS_MODELS_TO_VBLORA_TARGET_MODULES_MAPPING = {'t5': ['q', 'k', 'v', 'o', 'wi', 'wo'], 'mt5': ['q', 'k', 'v', 'o', 'wi_0', 'wi_1', 'wo'], 'bart': ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'], 'gpt2': ['c_attn'], 'bloom': ['query_key_value'], 'opt': ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'], 'gptj': ['q_proj', 'v_proj'], 'gpt_neox': ['query_key_value'], 'gpt_neo': ['q_proj', 'v_proj'], 'llama': ['q_proj', 'v_proj'], 'bert': ['query', 'value'], 'roberta': ['query', 'value'], 'deberta-v2': ['query_proj', 'key_proj', 'value_proj', 'dense'], 'gpt_bigcode': ['c_attn'], 'deberta': ['in_proj'], 'qwen2': ['q_proj', 'v_proj']}


TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING = {'t5': ['q', 'v'], 'mt5': ['q', 'v'], 'bart': ['q_proj', 'v_proj'], 'gpt2': ['c_attn'], 'bloom': ['query_key_value'], 'blip-2': ['q', 'v', 'q_proj', 'v_proj'], 'opt': ['q_proj', 'v_proj'], 'gptj': ['q_proj', 'v_proj'], 'gpt_neox': ['query_key_value'], 'gpt_neo': ['q_proj', 'v_proj'], 'bert': ['query', 'value'], 'roberta': ['query', 'value'], 'xlm-roberta': ['query', 'value'], 'electra': ['query', 'value'], 'deberta-v2': ['query_proj', 'value_proj'], 'deberta': ['in_proj'], 'layoutlm': ['query', 'value'], 'llama': ['q_proj', 'v_proj'], 'chatglm': ['query_key_value'], 'gpt_bigcode': ['c_attn'], 'mpt': ['Wqkv'], 'RefinedWebModel': ['query_key_value'], 'RefinedWeb': ['query_key_value'], 'falcon': ['query_key_value'], 'btlm': ['c_proj', 'c_attn'], 'codegen': ['qkv_proj'], 'mistral': ['q_proj', 'v_proj'], 'mixtral': ['q_proj', 'v_proj'], 'stablelm': ['q_proj', 'v_proj'], 'phi': ['q_proj', 'v_proj'], 'gemma': ['q_proj', 'v_proj'], 'gemma2': ['q_proj', 'v_proj'], 'qwen2': ['q_proj', 'v_proj']}


def _kaiming_init(tensor_or_shape: 'Union[torch.Tensor, tuple[int, ...]]', generator: 'torch.Generator') ->torch.Tensor:
    """
    Kaiming Uniform Initialisation adapted to accept a `torch.Generator` object for PRNG.

    Args:
        tensor_or_shape (`Union[torch.Tensor, tuple[int, ...]]`):
            Tensor to initialise, or shape of new tensor to create and then initialise.
        generator: (`torch.Generator`):
            Generator object that manages the state of the PRNG algorithm in use.

    Returns:
        `torch.Tensor`: The initialised tensor.
    """
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape
    fan = _calculate_correct_fan(tensor, 'fan_in')
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


class TemperatureScaledSoftmax(nn.Module):

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, logits):
        scaled_logits = logits / self.temperature
        return self.softmax(scaled_logits)


class XLoraClassifier(nn.Module):
    """
    A classifier to select LoRA layers for XLora.
    """

    def __init__(self, model: 'nn.Module', config: 'XLoraConfig', n_classes: 'int', n_layers: 'int', device: 'torch.device'):
        """
        Construct an X-LoRA classifier from a model, config and some metadata. Note that n_layers is the number of LoRA
        adapter layers, not the number of model layers.
        """
        super().__init__()
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.config = config
        self.log_scalings = []
        self.softmax = TemperatureScaledSoftmax(temperature=self.config.softmax_temperature)
        self.override_scaling_pass_value: 'Number' = config.scaling_pass_value
        self.scalings_logging = False
        self.dtype = next(model.parameters()).dtype
        add_dropout = config.xlora_dropout_p > 0.0
        layers = []
        if self.config.xlora_depth == 1:
            if config.layerwise_scalings:
                last = nn.Linear(config.hidden_size, n_classes * n_layers, bias=True).to(device)
            else:
                last = nn.Linear(config.hidden_size, n_classes, bias=True).to(device)
        else:
            if self.config.xlora_depth <= 0:
                raise ValueError('X-LoRA depth must be strictly positive.')
            layers.append(nn.Linear(config.hidden_size, config.xlora_size, bias=True).to(device))
            layers.append(nn.ReLU())
            if add_dropout:
                layers.append(nn.Dropout(p=config.xlora_dropout_p))
            for _ in range(config.xlora_depth - 2):
                layers.append(nn.Linear(config.xlora_size, config.xlora_size, bias=True).to(device))
                layers.append(nn.ReLU())
                if add_dropout:
                    layers.append(nn.Dropout(p=config.xlora_dropout_p))
            if config.layerwise_scalings:
                last = nn.Linear(config.xlora_size, n_classes * n_layers, bias=True).to(device)
            else:
                last = nn.Linear(config.xlora_size, n_classes, bias=True).to(device)
        self.layers = nn.Sequential(*layers, last)

    def make_dummy_scalings(self, input_ids: 'Optional[torch.LongTensor]'=None, inputs_embeds: 'Optional[torch.FloatTensor]'=None, *args, **kwargs) ->torch.Tensor:
        """
        Make some dummy scalings for the scalings pass (the one to get the logits for the X-LoRA classifier). These are
        of shape (batch_size, seq_len, n_layers, n_classes) and filled with the override scalings pass value. Note that
        n_layers is the number of LoRA adapter layers, not the number of model layers.
        """
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            seq_len = input_ids.shape[1]
        else:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
            seq_len = inputs_embeds.shape[1]
        return torch.full((batch_size, seq_len, self.n_layers, self.n_classes), self.override_scaling_pass_value)

    def forward(self, result, input_ids: 'Optional[torch.LongTensor]'=None, inputs_embeds: 'Optional[torch.FloatTensor]'=None, *args, **kwargs) ->torch.Tensor:
        """
        Using the hidden states of the model, predict `n_classes` LoRA alpha values. Returns the scalings.
        """
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
        else:
            batch_size = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]
        hidden_states = result.hidden_states
        hidden_state = hidden_states[-1]
        logits = self.layers.forward(hidden_state)
        if not self.config.layerwise_scalings:
            logits = logits.unsqueeze(2)
            logits = logits.expand(-1, -1, self.n_layers, -1)
        scalings = logits.reshape(batch_size, seq_len, self.n_layers, self.n_classes)
        if self.config.enable_softmax:
            scalings = self.softmax(scalings)
        if self.scalings_logging:
            self.log_scalings.append(scalings)
        return scalings

    def _get_bucketed_scalings(self) ->dict[int, tuple[list[int], list[torch.Tensor]]]:
        """
        Returns bucketed scalings, bucketed by seq_len. Each value consists of the positions (the first) and the
        associated tensors. The positions are paired with the associated tensors and give the position in the scaling
        log. Each scaling is a tensor of shape (batch_size, seq_len, n_layers, n_classes)).
        """
        seqlens_map: 'dict[int, tuple[list[int], list[torch.Tensor]]]' = {}
        for i, scaling in enumerate(self.log_scalings):
            seq_len = scaling.shape[1]
            if seq_len not in seqlens_map:
                seqlens_map[seq_len] = [i], [scaling]
            else:
                seqlens_map[seq_len][0].append(i)
                seqlens_map[seq_len][1].append(scaling)
        return seqlens_map

    def _set_override_scaling_pass_value(self, value: 'Union[Number, None]'):
        if value is None:
            self.override_scaling_pass_value = 1 / self.n_classes
        else:
            self.override_scaling_pass_value = value
        self.config.scaling_pass_value = self.override_scaling_pass_value


class MultitaskPromptTuningInit(str, enum.Enum):
    TEXT = 'TEXT'
    RANDOM = 'RANDOM'
    AVERAGE_SOURCE_TASKS = 'AVERAGE_SOURCE_TASKS'
    EXACT_SOURCE_TASK = 'EXACT_SOURCE_TASK'
    ONLY_SOURCE_SHARED = 'ONLY_SOURCE_SHARED'


def torch_load(*args, weights_only=True, **kwargs):
    """Call torch.load and handle weights_only.

    Defaults to weights_only=True to anticipate upcoming switch on the PyTorch side.

    """
    if version.parse(torch.__version__) < version.parse('1.13'):
        return torch.load(*args, **kwargs)
    return torch.load(*args, weights_only=weights_only, **kwargs)


class MultitaskPromptEmbedding(PromptEmbedding):

    def __init__(self, config: 'MultitaskPromptTuningConfig', word_embeddings):
        super().__init__(config, word_embeddings)
        self.num_tasks = config.num_tasks
        self.num_ranks = config.num_ranks
        self.num_virtual_tokens = config.num_virtual_tokens
        self.num_transformer_submodules = config.num_transformer_submodules
        if self.num_transformer_submodules is None:
            self.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1
        self.token_dim = config.token_dim
        total_virtual_tokens = self.num_virtual_tokens * self.num_transformer_submodules
        self.prefix_task_cols = torch.nn.Parameter(torch.normal(mean=0, std=0.02, size=(self.num_tasks, total_virtual_tokens, self.num_ranks)))
        self.prefix_task_rows = torch.nn.Parameter(torch.normal(mean=0, std=0.02, size=(self.num_tasks, self.num_ranks, self.token_dim)))
        if config.prompt_tuning_init in [MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS, MultitaskPromptTuningInit.EXACT_SOURCE_TASK, MultitaskPromptTuningInit.ONLY_SOURCE_SHARED]:
            if config.prompt_tuning_init_state_dict_path is None:
                raise ValueError(f'prompt_tuning_init_state_dict_path needs to be specified with {config.prompt_tuning_init} init method')
            if config.prompt_tuning_init_state_dict_path.endswith('.safetensors'):
                state_dict: 'dict' = load_file(config.prompt_tuning_init_state_dict_path)
            else:
                state_dict: 'dict' = torch_load(config.prompt_tuning_init_state_dict_path, map_location=word_embeddings.weight.device)
        if config.prompt_tuning_init in [MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS, MultitaskPromptTuningInit.EXACT_SOURCE_TASK]:
            prefix_task_cols_: 'torch.Tensor' = state_dict['prefix_task_cols']
            prefix_task_rows_: 'torch.Tensor' = state_dict['prefix_task_rows']
            if config.prompt_tuning_init == MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS:
                prefix_task_cols_ = prefix_task_cols_.mean(0, keepdim=True)
                prefix_task_rows_ = prefix_task_rows_.mean(0, keepdim=True)
            elif config.prompt_tuning_init == MultitaskPromptTuningInit.EXACT_SOURCE_TASK:
                prefix_task_cols_ = prefix_task_cols_[config.prompt_tuning_init_task, ...].unsqueeze(0)
                prefix_task_rows_ = prefix_task_rows_[config.prompt_tuning_init_task, ...].unsqueeze(0)
            state_dict = {'embedding.weight': state_dict['prompt_embeddings'], 'prefix_task_cols': prefix_task_cols_, 'prefix_task_rows': prefix_task_rows_}
            self.load_state_dict(state_dict, strict=True)
        elif config.prompt_tuning_init == MultitaskPromptTuningInit.ONLY_SOURCE_SHARED:
            state_dict = {'embedding.weight': state_dict['prompt_embeddings']}
            self.load_state_dict(state_dict, strict=False)

    def forward(self, indices, task_ids):
        if task_ids is None:
            raise ValueError('task_ids cannot be None')
        prompt_embeddings = self.embedding(indices)
        task_cols = torch.index_select(self.prefix_task_cols, 0, task_ids)
        task_rows = torch.index_select(self.prefix_task_rows, 0, task_ids)
        task_prompts = torch.matmul(task_cols, task_rows)
        prompt_embeddings *= task_prompts
        return prompt_embeddings

