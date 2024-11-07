
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


from typing import List


import torch


import torch.nn.functional as F


from enum import Enum


from typing import Tuple


import collections


import enum


from typing import Any


from typing import Dict


from typing import Type


from typing import Union


from collections import OrderedDict


from typing import Optional


import math


from copy import deepcopy


from typing import Generator


from typing import Iterable


from typing import MutableMapping


from typing import Set


import copy


import itertools


from itertools import chain


import numpy as np


import pandas as pd


import re


from typing import NamedTuple


from collections import Counter


from logging import getLogger


from typing import Callable


from functools import partial


import torch.utils.data


import logging


import random


from collections import defaultdict


from torch import nn


import torch.nn as nn


from scipy.special import logsumexp


from torch.multiprocessing.spawn import spawn


from torch import Tensor


from numpy import linalg as LA


from torch.utils.tensorboard import SummaryWriter


from itertools import tee


from itertools import zip_longest


from typing import Sequence


import time


from torch import optim


import torch.jit as jit


from torch.serialization import default_restore_location


from torch import jit


import warnings


from torch.nn import Linear


import torch.onnx.operators


from torch.nn import ModuleList


import inspect


from inspect import signature


import torch.jit


from enum import IntEnum


from enum import unique


import functools


from scipy.special import comb


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.onnx


import torch.cuda


from torch.nn import Parameter


from torch.nn import functional as F


from typing import Sized


from torch.nn import LayerNorm


import string


from torch.onnx import ExportTypes


from torch.onnx import OperatorExportTypes


from torch.optim.optimizer import Optimizer as PT_Optimizer


from collections import namedtuple


from torch.optim import Optimizer as PT_Optimizer


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.lr_scheduler import CosineAnnealingLR as TorchCosineAnnealingLR


from torch.optim.lr_scheduler import CyclicLR as TorchCyclicLR


from torch.optim.lr_scheduler import ExponentialLR as TorchExponentialLR


from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchReduceLROnPlateau


from torch.optim.lr_scheduler import StepLR as TorchStepLR


from torch.autograd import Variable


from torch import sparse


from torch.utils import data


from torch import sort


import abc


import torch.jit.quantized


import torch.multiprocessing as mp


from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook


from collections.abc import Sequence


import torch.distributed as dist_c10d


from inspect import getmembers


from inspect import isclass


from inspect import isfunction


import numpy


from typing import get_type_hints


from typing import IO


from typing import Iterator


@torch.jit.script
def validate_padding_control(padding_control: 'Optional[List[int]]') ->bool:
    if padding_control is not None:
        if len(padding_control) < 2:
            return False
        elif padding_control[0] != 0:
            return False
    return True


class TensorizerScriptImpl(torch.nn.Module):
    device: 'str'
    seq_padding_control: 'Optional[List[int]]'
    batch_padding_control: 'Optional[List[int]]'

    def __init__(self):
        super().__init__()
        self.device: 'str' = ''
        self.seq_padding_control = None
        self.batch_padding_control = None

    @torch.jit.export
    def set_device(self, device: 'str'):
        self.device = device

    @torch.jit.export
    def set_padding_control(self, dimension: 'str', padding_control: 'Optional[List[int]]'):
        """
        This functions will be called to set a padding style.
        None - No padding
        List: first element 0, round seq length to the smallest list element larger than inputs
        """
        if not validate_padding_control(padding_control):
            raise RuntimeError('Malformed padding_control value')
        if dimension == 'sequence_length':
            self.seq_padding_control = padding_control
        elif dimension == 'batch_length':
            self.batch_padding_control = padding_control
        else:
            raise RuntimeError('Illegal padding dimension specified.')

    def batch_size(self, inputs: 'ScriptBatchInput') ->int:
        texts: 'Optional[List[List[str]]]' = inputs.texts
        tokens: 'Optional[List[List[List[str]]]]' = inputs.tokens
        if texts is not None:
            return len(texts)
        elif tokens is not None:
            return len(tokens)
        else:
            raise RuntimeError('Empty input for both texts and tokens.')

    def row_size(self, inputs: 'ScriptBatchInput') ->int:
        texts: 'Optional[List[List[str]]]' = inputs.texts
        tokens: 'Optional[List[List[List[str]]]]' = inputs.tokens
        if texts is not None:
            return len(texts[0])
        elif tokens is not None:
            return len(tokens[0])
        else:
            raise RuntimeError('Empty input for both texts and tokens.')

    def get_texts_by_index(self, texts: 'Optional[List[List[str]]]', index: 'int') ->Optional[List[str]]:
        if texts is None or len(texts) == 0:
            return None
        return texts[index]

    def get_tokens_by_index(self, tokens: 'Optional[List[List[List[str]]]]', index: 'int') ->Optional[List[List[str]]]:
        if tokens is None or len(tokens) == 0:
            return None
        return tokens[index]

    def tokenize(self, *args, **kwargs):
        """
        This functions will receive the inputs from Clients, usually there are
        two possible inputs
        1) a row of texts: List[str]
        2) a row of pre-processed tokens: List[List[str]]

        Override this function to be TorchScriptable, e.g you need to declare
        concrete input arguments with type hints.
        """
        raise NotImplementedError

    def numberize(self, *args, **kwargs):
        """
        This functions will receive the outputs from function: tokenize() or
        will be called directly from PyTextTensorizer function: numberize().

        Override this function to be TorchScriptable, e.g you need to declare
        concrete input arguments with type hints.
        """
        raise NotImplementedError

    def tensorize(self, *args, **kwargs):
        """
        This functions will receive a list(e.g a batch) of outputs
        from function numberize(), padding and convert to output tensors.

        Override this function to be TorchScriptable, e.g you need to declare
        concrete input arguments with type hints.
        """
        raise NotImplementedError

    @torch.jit.ignore
    def tensorize_wrapper(self, *args, **kwargs):
        """
        This functions will receive a list(e.g a batch) of outputs
        from function numberize(), padding and convert to output tensors.

        It will be called in PyText Tensorizer during training time, this
        function is not torchscriptiable because it depends on cuda.device().
        """
        with to_device(self, cuda.device()):
            return self.tensorize(*args, **kwargs)

    @torch.jit.ignore
    def torchscriptify(self):
        return torch.jit.script(self)


class ScriptTokenizerBase(torch.jit.ScriptModule):

    @torch.jit.script_method
    def tokenize(self, input: 'str') ->List[Tuple[str, int, int]]:
        """
        Process a single line of raw inputs into tokens, it supports
        two input formats:
        1) a single text
        2) a token

        Returns a list of tokens with start and end indices in original input.
        """
        raise NotImplementedError


class ScriptDoNothingTokenizer(ScriptTokenizerBase):

    @torch.jit.script_method
    def tokenize(self, raw_token: 'str') ->List[Tuple[str, int, int]]:
        return [(raw_token, -1, -1)]


class Token(NamedTuple):
    value: 'str'
    start: 'int'
    end: 'int'


class ScriptVocabulary(nn.Module):
    idx: 'Dict[str, int]'

    def __init__(self, vocab_list, unk_idx: 'int'=0, pad_idx: 'int'=-1, bos_idx: 'int'=-1, eos_idx: 'int'=-1, mask_idx: 'int'=-1, unk_token: 'Optional[str]'=None):
        super().__init__()
        self.vocab: 'List[str]' = vocab_list
        self.unk_idx: 'int' = unk_idx
        self.pad_idx: 'int' = pad_idx
        self.eos_idx: 'int' = eos_idx
        self.bos_idx: 'int' = bos_idx
        self.mask_idx: 'int' = mask_idx
        self.idx: 'Dict[str, int]' = {word: i for i, word in enumerate(vocab_list)}
        pad_token = vocab_list[pad_idx] if pad_idx >= 0 else SpecialTokens.PAD
        self.pad_token: 'str' = pad_token
        self.unk_token = unk_token

    def get_pad_index(self):
        return self.pad_idx

    def get_unk_index(self):
        return self.unk_idx

    def lookup_indices_1d(self, values: 'List[str]') ->List[int]:
        result: 'List[int]' = []
        for value in values:
            result.append(self.idx.get(value, self.unk_idx))
        return result

    def lookup_indices_2d(self, values: 'List[List[str]]') ->List[List[int]]:
        result: 'List[List[int]]' = []
        for value in values:
            result.append(self.lookup_indices_1d(value))
        return result

    def lookup_indices_3d(self, values: 'List[List[List[str]]]') ->List[List[List[int]]]:
        result: 'List[List[List[int]]]' = []
        for value in values:
            result.append(self.lookup_indices_2d(value))
        return result

    def lookup_words_1d(self, values: 'torch.Tensor', filter_token_list: 'List[int]'=(), possible_unk_token: 'Optional[str]'=None) ->List[str]:
        """If possible_unk_token is not None, then all UNK id's will be replaced
        by possible_unk_token instead of the default UNK string which is <UNK>.
        This is a simple way to resolve UNK's when there's a correspondence
        between source and target translations.
        """
        result: 'List[str]' = []
        for idx in range(values.size(0)):
            value = int(values[idx])
            if not value in filter_token_list:
                result.append(self.lookup_word(value, possible_unk_token))
        return result

    def lookup_words_1d_cycle_heuristic(self, values: 'torch.Tensor', filter_token_list: 'List[int]', ordered_unks_token: 'List[str]') ->List[str]:
        """This function is a extension of the possible_unk_token heuristic
        in lookup_words_1d, which fails in the case when multiple unks are
        available. The way we deal with this is we increment every unk token in
        ordered_unks_token everytime we substitute an unk token. This solves a
        substantial amount of queries with multiple unk tokens.
        """
        unk_idx = 0
        unk_idx_length: 'int' = len(ordered_unks_token)
        unk_copy: 'bool' = unk_idx_length != 0
        vocab_length: 'int' = len(self.vocab)
        result: 'List[str]' = []
        for idx in range(values.size(0)):
            value = int(values[idx])
            if not value in filter_token_list:
                if value < vocab_length and value != self.unk_idx:
                    result.append(self.vocab[value])
                elif not unk_copy:
                    result.append(self.vocab[self.unk_idx])
                else:
                    unk_value = ordered_unks_token[unk_idx % unk_idx_length]
                    result.append(unk_value)
                    unk_idx += 1
        return result

    def lookup_word(self, idx: 'int', possible_unk_token: 'Optional[str]'=None):
        if idx < len(self.vocab) and idx != self.unk_idx:
            return self.vocab[idx]
        else:
            return self.vocab[self.unk_idx] if possible_unk_token is None else possible_unk_token

    def __len__(self):
        return len(self.vocab)


class CharacterVocabTokenTensorizerScriptImpl(TensorizerScriptImpl):

    def __init__(self, add_bos_token: 'bool', add_eos_token: 'bool', use_eos_token_for_bos: 'bool', max_seq_len: 'int', vocab: 'Vocabulary', tokenizer: 'Optional[Tokenizer]', lowercase_tokens: 'bool'=False, use_unk: 'bool'=True):
        super().__init__()
        if tokenizer is not None and hasattr(tokenizer, 'torchscriptify'):
            try:
                self.tokenizer = tokenizer.torchscriptify()
            except NotImplementedError:
                self.tokenizer = None
        else:
            self.tokenizer = None
        self.do_nothing_tokenizer = ScriptDoNothingTokenizer()
        self.vocab = ScriptVocabulary(list(vocab), pad_idx=vocab.get_pad_index(), bos_idx=vocab.get_bos_index() if add_bos_token else -1, eos_idx=vocab.get_eos_index() if add_eos_token else -1)
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_eos_token_for_bos = use_eos_token_for_bos
        self.max_seq_len = max_seq_len
        self.use_unk = use_unk
        self.lowercase_tokens = lowercase_tokens

    def tokenize(self, row_text: 'Optional[str]'=None, row_pre_tokenized: 'Optional[List[str]]'=None) ->Tuple[List[List[str]], List[int]]:
        tokens: 'List[Tuple[str, int, int]]' = []
        char_tokens: 'List[List[str]]' = []
        char_tokens_lengths: 'List[int]' = []
        if row_text is not None:
            assert self.tokenizer is not None
            tokens = self.tokenizer.tokenize(row_text)
        elif row_pre_tokenized is not None:
            for token in row_pre_tokenized:
                tokens.extend(self.do_nothing_tokenizer.tokenize(token))
        for token in tokens:
            chars: 'List[str]' = []
            for char in token[0]:
                chars.append(char)
            char_tokens.append(chars)
            char_tokens_lengths.append(len(chars))
        return char_tokens, char_tokens_lengths

    def numberize(self, char_tokens: 'List[List[str]]', char_tokens_lengths: 'List[int]') ->Tuple[List[List[int]], List[int]]:
        tokens: 'List[List[int]]' = []
        tokens = self.vocab.lookup_indices_2d(char_tokens)
        return tokens, char_tokens_lengths

    def tensorize(self, tokens: 'List[List[List[int]]]', tokens_lengths: 'List[List[int]]') ->Tuple[torch.Tensor, torch.Tensor]:
        tokens_padded: 'List[List[List[int]]]' = []
        tokens_lengths_padded: 'List[List[int]]' = []
        tokens_padded, tokens_lengths_padded = pad_3d(tokens, tokens_lengths, self.vocab.get_pad_index())
        tokens_tensor: 'torch.Tensor' = torch.tensor(tokens_padded, dtype=torch.long)
        tokens_lengths_tensor: 'torch.Tensor' = torch.tensor(tokens_lengths_padded, dtype=torch.long)
        return tokens_tensor, tokens_lengths_tensor

    def get_texts_by_index(self, texts: 'Optional[List[List[str]]]', index: 'int') ->Optional[str]:
        if texts is None or len(texts) == 0:
            return None
        return texts[index][0]

    def get_tokens_by_index(self, tokens: 'Optional[List[List[List[str]]]]', index: 'int') ->Optional[List[str]]:
        if tokens is None or len(tokens) == 0:
            return None
        return tokens[index][0]

    def forward(self, inputs: 'ScriptBatchInput') ->Tuple[torch.Tensor, torch.Tensor]:
        tokens_3d: 'List[List[List[int]]]' = []
        seq_lens_2d: 'List[List[int]]' = []
        for idx in range(self.batch_size(inputs)):
            char_tokens: 'List[List[int]]' = []
            char_tokens_lengths: 'List[int]' = []
            char_tokens, char_tokens_lengths = self.tokenize(self.get_texts_by_index(inputs.texts, idx), self.get_tokens_by_index(inputs.tokens, idx))
            numberized: 'Tuple[List[List[int]], List[int]]' = self.numberize(char_tokens, char_tokens_lengths)
            tokens_3d.append(numberized[0])
            seq_lens_2d.append(numberized[1])
        return self.tensorize(tokens_3d, seq_lens_2d)


class String2DListTensorizerScriptImpl(TensorizerScriptImpl):

    def __init__(self, vocab: 'Vocabulary'):
        super().__init__()
        self.vocab = ScriptVocabulary(list(vocab), pad_idx=vocab.get_pad_index())

    def numberize(self, tokens: 'List[List[str]]') ->Tuple[List[List[int]], List[int], int]:
        token_indices: 'List[List[int]]' = self.vocab.lookup_indices_2d(tokens)
        token_lengths: 'List[int]' = []
        for idx in range(len(token_indices)):
            token_lengths.append(len(token_indices[idx]))
        return token_indices, token_lengths, len(token_indices)

    def tensorize(self, tokens_3d: 'List[List[List[int]]]', seq_lens_2d: 'List[List[int]]', seq_lens_1d: 'List[int]') ->Tuple[torch.Tensor, torch.Tensor]:
        padded_batch, _ = pad_3d(batch=tokens_3d, tokens_lengths=seq_lens_2d, pad_idx=self.vocab.pad_idx)
        return torch.tensor(padded_batch, dtype=torch.long), torch.tensor(seq_lens_1d, dtype=torch.long)

    def forward(self, inputs: 'List[List[List[str]]]') ->Tuple[torch.Tensor, torch.Tensor]:
        tokens_3d: 'List[List[List[int]]]' = []
        seq_lens_2d: 'List[List[int]]' = []
        seq_lens_1d: 'List[int]' = []
        for idx in range(len(inputs)):
            numberized: 'Tuple[List[List[int]], List[int], int]' = self.numberize(inputs[idx])
            tokens_3d.append(numberized[0])
            seq_lens_2d.append(numberized[1])
            seq_lens_1d.append(numberized[2])
        return self.tensorize(tokens_3d, seq_lens_2d, seq_lens_1d)


class VocabLookup(torch.jit.ScriptModule):
    """
    TorchScript implementation of lookup_tokens() in pytext/data/tensorizers.py
    """

    def __init__(self, vocab: 'ScriptVocabulary'):
        super().__init__()
        self.vocab = vocab

    @torch.jit.script_method
    def forward(self, tokens: 'List[Tuple[str, int, int]]', bos_idx: 'Optional[int]'=None, eos_idx: 'Optional[int]'=None, use_eos_token_for_bos: 'bool'=False, max_seq_len: 'int'=2 ** 30) ->Tuple[List[int], List[int], List[int]]:
        """Convert tokens into ids by doing vocab look-up.

        Convert tokens into ids by doing vocab look-up. It will also append
        bos & eos index into token_ids if needed. A token is represented by
        a Tuple[str, int, int], which is [token, start_index, end_index].

        Args:
            tokens: List of tokens with start and end position in the original
                text. start and end index could be optional (e.g value is -1)
            bos_idx: index of begin of sentence, optional.
            eos_idx: index of end of sentence, optional.
            use_eos_token_for_bos: use eos index as bos.
            max_seq_len: maximum tokens length.
        """
        if bos_idx is None:
            bos_idx = -1
        if eos_idx is None:
            eos_idx = -1
        text_tokens: 'List[str]' = []
        start_idxs: 'List[int]' = []
        end_idxs: 'List[int]' = []
        max_seq_len = max_seq_len - (1 if bos_idx >= 0 else 0) - (1 if eos_idx >= 0 else 0)
        for i in range(min(len(tokens), max_seq_len)):
            token: 'Tuple[str, int, int]' = tokens[i]
            text_tokens.append(token[0])
            start_idxs.append(token[1])
            end_idxs.append(token[2])
        token_ids: 'List[int]' = self.vocab.lookup_indices_1d(text_tokens)
        if bos_idx >= 0:
            if use_eos_token_for_bos:
                bos_idx = eos_idx
            token_ids = [bos_idx] + token_ids
            start_idxs = [-1] + start_idxs
            end_idxs = [-1] + end_idxs
        if eos_idx >= 0:
            token_ids.append(eos_idx)
            start_idxs.append(-1)
            end_idxs.append(-1)
        return token_ids, start_idxs, end_idxs


class TokenTensorizerScriptImpl(TensorizerScriptImpl):

    def __init__(self, add_bos_token: 'bool', add_eos_token: 'bool', use_eos_token_for_bos: 'bool', max_seq_len: 'int', vocab: 'Vocabulary', tokenizer: 'Optional[Tokenizer]'):
        super().__init__()
        if tokenizer is not None and hasattr(tokenizer, 'torchscriptify'):
            try:
                self.tokenizer = tokenizer.torchscriptify()
            except NotImplementedError:
                self.tokenizer = None
        else:
            self.tokenizer = None
        self.do_nothing_tokenizer = ScriptDoNothingTokenizer()
        self.vocab = ScriptVocabulary(list(vocab), pad_idx=vocab.get_pad_index(), bos_idx=vocab.get_bos_index() if add_bos_token else -1, eos_idx=vocab.get_eos_index() if add_eos_token else -1)
        self.vocab_lookup_1d = VocabLookup(self.vocab)
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_eos_token_for_bos = use_eos_token_for_bos
        self.max_seq_len = max_seq_len

    def get_texts_by_index(self, texts: 'Optional[List[List[str]]]', index: 'int') ->Optional[str]:
        if texts is None or len(texts) == 0:
            return None
        return texts[index][0]

    def get_tokens_by_index(self, tokens: 'Optional[List[List[List[str]]]]', index: 'int') ->Optional[List[str]]:
        if tokens is None or len(tokens) == 0:
            return None
        return tokens[index][0]

    def _lookup_tokens_1d(self, tokens: 'List[Tuple[str, int, int]]') ->Tuple[List[int], List[int], List[int]]:
        return self.vocab_lookup_1d(tokens, bos_idx=self.vocab.bos_idx if self.add_bos_token else None, eos_idx=self.vocab.eos_idx if self.add_eos_token else None, use_eos_token_for_bos=self.use_eos_token_for_bos, max_seq_len=self.max_seq_len)

    def tokenize(self, row_text: 'Optional[str]', row_pre_tokenized: 'Optional[List[str]]') ->List[Tuple[str, int, int]]:
        tokens: 'List[Tuple[str, int, int]]' = []
        if row_text is not None:
            if self.tokenizer is not None:
                tokens = self.tokenizer.tokenize(row_text)
        elif row_pre_tokenized is not None:
            for token in row_pre_tokenized:
                tokens.extend(self.do_nothing_tokenizer.tokenize(token))
        return tokens

    def numberize(self, text_tokens: 'List[Tuple[str, int, int]]') ->Tuple[List[int], int, List[Tuple[int, int]]]:
        token_indices: 'List[int]' = []
        token_starts: 'List[int]' = []
        token_ends: 'List[int]' = []
        token_indices, token_starts, token_ends = self._lookup_tokens_1d(text_tokens)
        token_ranges: 'List[Tuple[int, int]]' = []
        for s, e in zip(token_starts, token_ends):
            token_ranges.append((s, e))
        return token_indices, len(token_indices), token_ranges

    def tensorize(self, tokens_2d: 'List[List[int]]', seq_lens_1d: 'List[int]', positions_2d: 'List[List[Tuple[int, int]]]') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_indices_tensor: 'torch.Tensor' = torch.tensor(pad_2d(tokens_2d, seq_lens=seq_lens_1d, pad_idx=self.vocab.pad_idx), dtype=torch.long)
        token_starts_2d: 'List[List[int]]' = []
        token_ends_2d: 'List[List[int]]' = []
        for position_list in positions_2d:
            token_starts_2d.append([x[0] for x in position_list])
            token_ends_2d.append([x[1] for x in position_list])
        token_positions_tensor = torch.stack([torch.tensor(pad_2d(token_starts_2d, seq_lens=seq_lens_1d, pad_idx=-1), dtype=torch.long), torch.tensor(pad_2d(token_ends_2d, seq_lens=seq_lens_1d, pad_idx=-1), dtype=torch.long)], dim=2)
        return token_indices_tensor, torch.tensor(seq_lens_1d, dtype=torch.long), token_positions_tensor

    def forward(self, inputs: 'ScriptBatchInput') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens_2d: 'List[List[int]]' = []
        seq_lens_1d: 'List[int]' = []
        positions_2d: 'List[List[Tuple[int, int]]]' = []
        for idx in range(self.batch_size(inputs)):
            tokens: 'List[Tuple[str, int, int]]' = self.tokenize(self.get_texts_by_index(inputs.texts, idx), self.get_tokens_by_index(inputs.tokens, idx))
            numberized: 'Tuple[List[int], int, List[Tuple[int, int]]]' = self.numberize(tokens)
            tokens_2d.append(numberized[0])
            seq_lens_1d.append(numberized[1])
            positions_2d.append(numberized[2])
        return self.tensorize(tokens_2d, seq_lens_1d, positions_2d)

