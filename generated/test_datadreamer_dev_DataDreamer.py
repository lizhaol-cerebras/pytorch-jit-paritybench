
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


import itertools


from functools import cached_property


from functools import partial


import logging


from abc import ABC


from abc import abstractmethod


from collections import Counter


from collections import defaultdict


from collections.abc import Iterator


from collections.abc import Sized


from itertools import chain


from itertools import islice


from itertools import tee


from logging import Logger


from math import ceil


from time import time


from typing import Any


from typing import Callable


from typing import DefaultDict


from typing import Generator


from typing import Iterable


from typing import cast


from uuid import uuid4


import torch


from collections import UserDict


from typing import TYPE_CHECKING


from collections.abc import Iterable


from pandas import DataFrame


import numpy as np


import torch._dynamo


from functools import lru_cache


import re


from types import MethodType


from functools import cache


from logging import root


import warnings


from typing import Sequence


import torch.nn.functional as F


import typing


import uuid


from math import floor


from random import Random


from time import sleep


from types import GeneratorType


from types import SimpleNamespace


from copy import copy


from typing import Type


from typing import Dict


from typing import List


from typing import Optional


from typing import Union


from torch.nn.utils.rnn import pad_sequence


import random


from copy import deepcopy


from functools import wraps


from typing import Literal


from typing import Tuple


import torch.nn as nn


from torch.utils.data import DataLoader


from torch.nn import functional as F


from torch.optim import AdamW


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LRScheduler


from functools import total_ordering


from collections import namedtuple


from logging import StreamHandler


import torch.cuda


from torch.optim.lr_scheduler import LambdaLR


from types import ModuleType


class SentenceTransformerLossWrapper(torch.nn.Module):

    def __init__(self, orig_model: 'SentenceTransformer', wrapped_model: 'SentenceTransformerWrapper', loss_module: 'torch.nn.Module', _is_peft: 'bool'):
        torch.nn.Module.__init__(self)
        self.orig_model = orig_model
        self.wrapped_model = wrapped_model
        self.loss_module = loss_module
        self._is_peft = _is_peft

    def __getattr__(self, name):
        if name == 'config':
            if self._is_peft:
                sentence_transformer_model = get_base_model_from_peft_model(self.orig_model)
            else:
                sentence_transformer_model = self.orig_model
            has_transformer_module = '0' in sentence_transformer_model._modules and isinstance(sentence_transformer_model._modules['0'], Transformer)
            if has_transformer_module:
                transformer_module = sentence_transformer_model._modules['0']
                has_auto_model = 'auto_model' in transformer_module._modules and isinstance(transformer_module._modules['auto_model'], PreTrainedModel)
                if has_auto_model:
                    return transformer_module._modules['auto_model'].config
        return super().__getattr__(name)

    def forward(self, anchor_input_ids: 'None | torch.Tensor'=None, anchor_attention_mask: 'None | torch.Tensor'=None, positive_input_ids: 'None | torch.Tensor'=None, positive_attention_mask: 'None | torch.Tensor'=None, negative_input_ids: 'None | torch.Tensor'=None, negative_attention_mask: 'None | torch.Tensor'=None, labels: 'None | torch.Tensor'=None):
        _uniq_ids = []
        sentence_features = []
        _uniq_ids.append(uuid4().hex)
        sentence_features.append({'_uniq_id': _uniq_ids[-1], 'input_ids': anchor_input_ids, 'attention_mask': anchor_attention_mask})
        if positive_input_ids is not None:
            _uniq_ids.append(uuid4().hex)
            sentence_features.append({'_uniq_id': _uniq_ids[-1], 'input_ids': positive_input_ids, 'attention_mask': positive_attention_mask})
        if negative_input_ids is not None:
            _uniq_ids.append(uuid4().hex)
            sentence_features.append({'_uniq_id': _uniq_ids[-1], 'input_ids': negative_input_ids, 'attention_mask': negative_attention_mask})
        loss = self.loss_module(sentence_features=sentence_features, labels=labels)
        return {'loss': loss, 'embeddings': [self.wrapped_model.results[_uniq_id]['sentence_embedding'].detach() for _uniq_id in _uniq_ids], 'loss_for_joint_metric': loss}

