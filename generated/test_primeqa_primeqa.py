
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


import random


from collections import defaultdict


from collections import OrderedDict


import time


import torch.multiprocessing as mp


import numpy as np


from itertools import product


from torch.utils.cpp_extension import load


from typing import Any


import string


import torch.nn as nn


from math import ceil


from torch._C import device


from typing import Union


import itertools


from itertools import chain


import pandas as pd


from random import randrange


import re


from torch.optim import Adam


from torch.utils.data import DataLoader


from sklearn.metrics import ndcg_score


from typing import List


from typing import Dict


import logging


import warnings


import torch.nn.functional as F


import math


import copy


from queue import Empty


from typing import Iterable


from torch.utils.checkpoint import checkpoint


from typing import Callable


from typing import Optional


from torch import nn


from torch.utils.data import Dataset


from torch.autograd import Variable


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


import torch.optim as optim


import collections


from torch.utils.data.distributed import DistributedSampler


from functools import partial


from torch.utils.data import TensorDataset


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from torch.utils.data import WeightedRandomSampler


from sklearn.metrics import accuracy_score


from typing import Tuple


from typing import Type


from abc import ABCMeta


from abc import abstractmethod


from copy import deepcopy


from torch.nn.functional import normalize


from itertools import groupby


import sklearn


from sklearn.neural_network import MLPClassifier


from torch.utils.data import ConcatDataset


import inspect


from torch.utils.data import Subset


from torch.utils.data import IterableDataset


from torch import cuda


from sklearn.metrics.pairwise import linear_kernel


from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class CoreConfig:

    def __post_init__(self):
        """
        Source: https://stackoverflow.com/a/58081120/1493011
        """
        self.assigned = {}
        for field in fields(self):
            field_val = getattr(self, field.name)
            if isinstance(field_val, DefaultVal) or field_val is None:
                setattr(self, field.name, field.default.val)
            if not isinstance(field_val, DefaultVal):
                self.assigned[field.name] = True

    def assign_defaults(self):
        for field in fields(self):
            setattr(self, field.name, field.default.val)
            self.assigned[field.name] = True

    def configure(self, ignore_unrecognized=True, **kw_args):
        ignored = set()
        for key, value in kw_args.items():
            self.set(key, value, ignore_unrecognized) or ignored.update({key})
        return ignored
        """
        # TODO: Take a config object, not kw_args.

        for key in config.assigned:
            value = getattr(config, key)
        """

    def set(self, key, value, ignore_unrecognized=False):
        if hasattr(self, key):
            setattr(self, key, value)
            self.assigned[key] = True
            return True
        if not ignore_unrecognized:
            raise Exception(f'Unrecognized key `{key}` for {type(self)}')

    def help(self):
        None

    def __export_value(self, v):
        v = v.provenance() if hasattr(v, 'provenance') else v
        if isinstance(v, list) and len(v) > 100:
            v = f'list with {len(v)} elements starting with...', v[:3]
        if isinstance(v, dict) and len(v) > 100:
            v = f'dict with {len(v)} keys starting with...', list(v.keys())[:3]
        return v

    def export(self):
        d = dataclasses.asdict(self)
        for k, v in d.items():
            d[k] = self.__export_value(v)
        return d


class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    Credit: derek73 @ https://stackoverflow.com/questions/2352181
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_metadata_only():
    args = dotdict()
    args.hostname = socket.gethostname()
    try:
        args.git_branch = git.Repo(search_parent_directories=True).active_branch.name
        args.git_hash = git.Repo(search_parent_directories=True).head.object.hexsha
        args.git_commit_datetime = str(git.Repo(search_parent_directories=True).head.object.committed_datetime)
    except git.exc.InvalidGitRepositoryError as e:
        pass
    args.current_datetime = time.strftime('%b %d, %Y ; %l:%M%p %Z (%z)')
    args.cmd = ' '.join(sys.argv)
    return args


def print_message(*s, condition=True, pad=False):
    s = ' '.join([str(x) for x in s])
    msg = '[{}] {}'.format(datetime.datetime.now().strftime('%b %d, %H:%M:%S'), s)
    if condition:
        msg = msg if not pad else f'\n{msg}\n'
        None
    return msg


def torch_load_dnn(path):
    if path.startswith('http:') or path.startswith('https:'):
        dnn = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        dnn = torch.load(path, map_location='cpu')
    return dnn


@dataclass
class BaseConfig(CoreConfig):

    @classmethod
    def from_existing(cls, *sources):
        kw_args = {}
        for source in sources:
            if source is None:
                continue
            local_kw_args = dataclasses.asdict(source)
            local_kw_args = {k: local_kw_args[k] for k in source.assigned}
            kw_args = {**kw_args, **local_kw_args}
        obj = cls(**kw_args)
        return obj

    @classmethod
    def from_deprecated_args(cls, args):
        obj = cls()
        ignored = obj.configure(ignore_unrecognized=True, **args)
        return obj, ignored

    @classmethod
    def from_path(cls, name):
        print_message(f'#> base_config.py from_path {name}')
        with open(name) as f:
            args = ujson.load(f)
            print_message(f'#> base_config.py from_path args loaded! ')
            if 'config' in args:
                args = args['config']
                print_message(f'#> base_config.py from_path args replaced ! ')
        return cls.from_deprecated_args(args)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        if checkpoint_path.endswith('.dnn') or checkpoint_path.endswith('.model'):
            dnn = torch_load_dnn(checkpoint_path)
            config, _ = cls.from_deprecated_args(dnn.get('arguments', {}))
            config.set('checkpoint', checkpoint_path)
            return config
        loaded_config_path = os.path.join(checkpoint_path, 'artifact.metadata')
        print_message(f'#> base_config.py load_from_checkpoint {checkpoint_path}')
        print_message(f'#> base_config.py load_from_checkpoint {loaded_config_path}')
        if os.path.exists(loaded_config_path):
            loaded_config, _ = cls.from_path(loaded_config_path)
            loaded_config.set('checkpoint', checkpoint_path)
            return loaded_config
        return None

    @classmethod
    def load_from_index(cls, index_path):
        try:
            metadata_path = os.path.join(index_path, 'metadata.json')
            loaded_config, _ = cls.from_path(metadata_path)
        except:
            metadata_path = os.path.join(index_path, 'plan.json')
            loaded_config, _ = cls.from_path(metadata_path)
        return loaded_config

    def save(self, path, overwrite=False):
        assert overwrite or not os.path.exists(path), path
        with open(path, 'w') as f:
            args = self.export()
            args['meta'] = get_metadata_only()
            args['meta']['version'] = 'colbert-v0.4'
            f.write(ujson.dumps(args, indent=4) + '\n')

    def save_for_checkpoint(self, checkpoint_path):
        assert not checkpoint_path.endswith('.dnn'), f'{checkpoint_path}: We reserve *.dnn names for the deprecated checkpoint format.'
        output_config_path = os.path.join(checkpoint_path, 'artifact.metadata')
        self.save(output_config_path, overwrite=True)


def timestamp(daydir=False):
    format_str = f"%Y-%m{'/' if daydir else '-'}%d{'/' if daydir else '_'}%H.%M.%S"
    result = datetime.datetime.now().strftime(format_str)
    return result


def get_model_type(name, return_config=False):
    if name.endswith('.dnn') or name.endswith('.model'):
        dnn_checkpoint = torch_load_dnn(name)
        config = dnn_checkpoint.get('config', None)
        if config:
            if hasattr(PretrainedConfig, 'model_type'):
                delattr(PretrainedConfig, 'model_type')
            config = PretrainedConfig.from_dict(config)
            if not hasattr(config, 'hidden_size'):
                config.hidden_size = config.d_model
            model_type = config.model_type
        else:
            state_dict = dnn_checkpoint['model_state_dict']
            oneparam = list(state_dict.keys())[0]
            model_type = oneparam.split('.')[1]
    else:
        checkpoint_config = AutoConfig.from_pretrained(name)
        model_type = checkpoint_config.model_type
        config = None
    if return_config:
        return model_type, config
    else:
        return model_type


def get_colbert_from_pretrained(name, colbert_config):
    model_type, config = get_model_type(name, return_config=True)
    print_message(f'factory model type: {model_type}')
    if model_type == 'bert':
        if config:
            colbert = HF_ColBERT(config, colbert_config)
            colbert.load_state_dict(name)
        else:
            colbert = HF_ColBERT.from_pretrained(name, colbert_config)
    elif model_type == 'xlm-roberta':
        if config:
            colbert = HF_ColBERT_XLMR(config, colbert_config)
            colbert.load_state_dict(name)
        else:
            colbert = HF_ColBERT_XLMR.from_pretrained(name, colbert_config)
    elif model_type == 'roberta':
        if config:
            colbert = HF_ColBERT_Roberta(config, colbert_config)
            colbert.load_state_dict(name)
        else:
            colbert = HF_ColBERT_Roberta.from_pretrained(name, colbert_config)
    else:
        raise NotImplementedError(f'Model type: {model_type} is not supported.')
    return colbert


class BaseColBERT(torch.nn.Module):
    """
    Shallow module that wraps the ColBERT parameters, custom configuration, and underlying tokenizer.
    This class provides direct instantiation and saving of the model/colbert_config/tokenizer package.

    Like HF, evaluation mode is the default.
    """

    def __init__(self, name, colbert_config=None):
        super().__init__()
        print_message(f'#>>>>> at BaseColBERT name (model name) : {name}')
        self.name = name
        self.colbert_config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint(name), colbert_config)
        self.model = get_colbert_from_pretrained(name, colbert_config=self.colbert_config)
        self.config = self.model.config
        self.raw_tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path)
        self.eval()

    @property
    def device(self):
        return self.model.device

    @property
    def bert(self):
        return self.model.bert

    @property
    def linear(self):
        return self.model.linear

    @property
    def score_scaler(self):
        return self.model.score_scaler

    def save(self, path):
        assert not path.endswith('.dnn'), f'{path}: We reserve *.dnn names for the deprecated checkpoint format.'
        self.model.save_pretrained(path)
        self.raw_tokenizer.save_pretrained(path)
        self.colbert_config.save_for_checkpoint(path)


DEVICE = torch.device('cuda')


def colbert_score_reduce(scores_padded, D_mask, config: 'ColBERTConfig'):
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values
    assert config.interaction in ['colbert', 'flipr'], config.interaction
    if config.interaction == 'flipr':
        assert config.query_maxlen == 64, ('for now', config)
        K1 = config.query_maxlen // 2
        K2 = 8
        A = scores[:, :config.query_maxlen].topk(K1, dim=-1).values.sum(-1)
        B = 0
        if K2 <= scores.size(1) - config.query_maxlen:
            B = scores[:, config.query_maxlen:].topk(K2, dim=-1).values.sum(1)
        return A + B
    return scores.sum(-1)


def flatten(L):
    result = []
    for _list in L:
        result += _list
    return result


def print_torch_extension_error_message():
    msg = """Troubleshooting possible causes for failed PyTorch extension compilation:

    - PyTorch is using the system CUDA installation instead of environment CUDA (possible fix: set CUDA_PATH environment variable)
    - Incompatible gcc and nvcc compiler versions (possible fix: manually install a different gcc/gxx version, e.g. 9.4.0)
    - Compilation hangs indefinitely (possible fix: remove /path/to/.cache/torch_extensions directory)
    """
    return print_message(msg, pad=True)


class ColBERT(BaseColBERT):
    """
        This class handles the basic encoding and scoring operations in ColBERT. It is used for training.
    """

    def __init__(self, name=None, colbert_config=None):
        print_message(f'#>>>>> at ColBERT name (model name) : {name}')
        super().__init__(name, colbert_config)
        self.use_gpu = torch.cuda.is_available()
        ColBERT.try_load_torch_extensions(self.use_gpu)
        if self.colbert_config.mask_punctuation:
            self.skiplist = {w: (True) for symbol in string.punctuation for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        self.query_used = False
        self.doc_used = False

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, 'loaded_extensions') or use_gpu:
            return
        verbose = os.getenv('COLBERT_LOAD_TORCH_EXTENSION_VERBOSE', 'False') == 'True'
        print_message(f'Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...')
        try:
            segmented_maxsim_cpp = load(name='segmented_maxsim_cpp', sources=[os.path.join(pathlib.Path(__file__).parent.resolve(), 'segmented_maxsim.cpp')], extra_cflags=['-O3'], verbose=verbose)
        except (RuntimeError, KeyboardInterrupt) as e:
            if not verbose:
                traceback.print_exc()
            print_torch_extension_error_message()
            sys.exit(1)
        cls.segmented_maxsim = segmented_maxsim_cpp.segmented_maxsim_cpp
        cls.loaded_extensions = True

    def forward(self, Q, D):
        Q = self.query(*Q)
        D, D_mask = self.doc(*D, keep_dims='return_mask')
        Q_duplicated = Q.repeat_interleave(self.colbert_config.nway, dim=0).contiguous()
        scores = self.score(Q_duplicated, D, D_mask)
        if self.colbert_config.distill_query_passage_separately:
            if self.colbert_config.query_only:
                return Q
            else:
                return scores, Q_duplicated, D
        if self.colbert_config.use_ib_negatives:
            ib_loss = self.compute_ib_loss(Q, D, D_mask)
            return scores, ib_loss
        return scores

    def compute_ib_loss(self, Q, D, D_mask):
        if DEVICE == torch.device('cuda'):
            scores = (D.unsqueeze(0) @ Q.permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)
        else:
            scores = (D.unsqueeze(0).float() @ Q.permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)
        scores = colbert_score_reduce(scores, D_mask.repeat(Q.size(0), 1, 1), self.colbert_config)
        nway = self.colbert_config.nway
        all_except_self_negatives = [(list(range(qidx * D.size(0), qidx * D.size(0) + nway * qidx + 1)) + list(range(qidx * D.size(0) + nway * (qidx + 1), qidx * D.size(0) + D.size(0)))) for qidx in range(Q.size(0))]
        scores = scores[flatten(all_except_self_negatives)]
        scores = scores.view(Q.size(0), -1)
        labels = torch.arange(0, Q.size(0), device=scores.device) * self.colbert_config.nway
        return torch.nn.CrossEntropyLoss()(scores, labels)

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids, attention_mask
        if not self.query_used:
            print_message('#>>>> colbert query ==')
            print_message(f'#>>>>> input_ids: {input_ids[0].size()}, {input_ids[0]}')
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        if not self.query_used:
            print_message('#>>>> before linear query ==')
            print_message(f'#>>>>> Q: {Q[0].size()}, {Q[0]}')
            print_message(f'#>>>>> self.linear query : {self.linear.weight}')
        Q = self.linear(Q)
        if not self.query_used:
            self.query_used = True
            print_message('#>>>> colbert query ==')
            print_message(f'#>>>>> Q: {Q[0].size()}, {Q[0]}')
        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask
        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']
        input_ids, attention_mask = input_ids, attention_mask
        if not self.doc_used:
            print_message('#>>>> colbert doc ==')
            print_message(f'#>>>>> input_ids: {input_ids[0].size()}, {input_ids[0]}')
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        if not self.doc_used:
            print_message('#>>>> before linear doc ==')
            print_message(f'#>>>>> D: {D[0].size()}, {D[0]}')
            print_message(f'#>>>>> self.linear doc : {self.linear.weight}')
        D = self.linear(D)
        if not self.doc_used:
            self.doc_used = True
            print_message('#>>>> colbert doc ==')
            print_message(f'#>>>>> D: {D[0].size()}, {D[0]}')
        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        D = D * mask
        D = torch.nn.functional.normalize(D, p=2, dim=2)
        if self.use_gpu:
            D = D.half()
        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]
        elif keep_dims == 'return_mask':
            return D, mask.bool()
        return D

    def score(self, Q, D_padded, D_mask):
        if self.colbert_config.similarity == 'l2':
            assert False, 'l2 similarity is not supported'
            assert self.colbert_config.interaction == 'colbert'
            return (-1.0 * ((Q.unsqueeze(2) - D_padded.unsqueeze(1)) ** 2).sum(-1)).max(-1).values.sum(-1)
        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config)

    def mask(self, input_ids, skiplist):
        mask = [[(x not in skiplist and x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask


class NullContextManager(object):

    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


class MixedPrecisionManager:

    def __init__(self, activated):
        self.activated = activated
        if self.activated:
            self.scaler = torch.amp.GradScaler()

    def context(self):
        return torch.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, colbert, optimizer, scheduler=None):
        if self.activated:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0, error_if_nonfinite=False)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)
    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)
    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos
    return output


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))
    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices
    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset + bsize], mask[offset:offset + bsize]))
    return batches


class DocTokenizer:

    def __init__(self, doc_maxlen, model_type):
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(model_type)
        self.doc_maxlen = doc_maxlen
        self.D_marker_token, self.D_marker_token_id = '[D]', self.tok.convert_tokens_to_ids('[unused1]')
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        assert self.D_marker_token_id == 2
        self.used = False

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)
        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]
        if not add_special_tokens:
            return tokens
        prefix, suffix = [self.cls_token, self.D_marker_token], [self.sep_token]
        tokens = [(prefix + lst + suffix) for lst in tokens]
        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)
        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']
        if not add_special_tokens:
            return ids
        prefix, suffix = [self.cls_token_id, self.D_marker_token_id], [self.sep_token_id]
        ids = [(prefix + lst + suffix) for lst in ids]
        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], type(batch_text)
        batch_text = [('. ' + x) for x in batch_text]
        obj = self.tok(batch_text, padding='longest', truncation='longest_first', return_tensors='pt', max_length=self.doc_maxlen)
        ids, mask = obj['input_ids'], obj['attention_mask']
        ids[:, 1] = self.D_marker_token_id
        if not self.used:
            self.used = True
            print_message('#> BERT DocTokenizer.tensorize(batch_text[0], batch_background[0], bsize) ==')
            print_message(f'#> Input: {batch_text[0]}, \t\t {bsize}')
            print_message(f'#> Output IDs: {ids[0].size()}, {ids[0]}')
            print_message(f'#> Output Mask: {mask[0].size()}, {mask[0]}')
        if bsize:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices
        return ids, mask


class DocTokenizerRoberta:

    def __init__(self, doc_maxlen, model_type):
        self.tok = HF_ColBERT_Roberta.raw_tokenizer_from_pretrained(model_type)
        self.doc_maxlen = doc_maxlen
        self.D_marker_token, self.D_marker_token_id = '[D]', self.tok.convert_tokens_to_ids('madeupword0001')
        self.used = False

    def tokenize(self, batch_text, add_special_tokens=False):
        raise NotImplementedError()

    def encode(self, batch_text, add_special_tokens=False):
        raise NotImplementedError()

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], type(batch_text)
        batch_text = [('$ ' + x) for x in batch_text]
        obj = self.tok(batch_text, padding='longest', truncation='longest_first', return_tensors='pt', max_length=self.doc_maxlen)
        ids, mask = obj['input_ids'], obj['attention_mask']
        ids[:, 1] = self.D_marker_token_id
        if self.used is False:
            self.used = True
            print_message('#> Roberta DocTokenizer.tensorize(batch_text[0], batch_background[0], bsize) ==')
            print_message(f'#> Input: {batch_text[0]}, \t\t {bsize}')
            print_message(f'#> Output IDs: {ids[0].size()}, {ids[0]}')
            print_message(f'#> Output Mask: {mask[0].size()}, {mask[0]}')
        if bsize:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices
        return ids, mask


class DocTokenizerXLMR:

    def __init__(self, doc_maxlen, model_type):
        self.tok = HF_ColBERT_XLMR.raw_tokenizer_from_pretrained(model_type)
        self.doc_maxlen = doc_maxlen
        self.Q_marker_token, self.D_marker_token_id = '?', 9749
        self.used = False

    def tokenize(self, batch_text, add_special_tokens=False):
        raise NotImplementedError()

    def encode(self, batch_text, add_special_tokens=False):
        raise NotImplementedError()

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], type(batch_text)
        batch_text = [('$ ' + x) for x in batch_text]
        obj = self.tok(batch_text, padding='longest', truncation='longest_first', return_tensors='pt', max_length=self.doc_maxlen)
        ids, mask = obj['input_ids'], obj['attention_mask']
        ids[:, 1] = self.D_marker_token_id
        if self.used is False:
            self.used = True
            print_message('#> XLMR DocTokenizer.tensorize(batch_text[0], batch_background[0], bsize) ==')
            print_message(f'#> Input: {batch_text[0]}, \t\t {bsize}')
            print_message(f'#> Output IDs: {ids[0].size()}, {ids[0]}')
            print_message(f'#> Output Mask: {mask[0].size()}, {mask[0]}')
        if bsize:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices
        return ids, mask


def get_doc_tokenizer(name, colbert_config, is_teacher=False):
    model_type = get_model_type(name)
    maxlen = colbert_config.teacher_doc_maxlen if is_teacher else colbert_config.doc_maxlen
    print_message(f'factory model type: {model_type}')
    if model_type == 'bert':
        return DocTokenizer(maxlen, name)
    elif model_type == 'xlm-roberta':
        return DocTokenizerXLMR(maxlen, name)
    elif model_type == 'roberta':
        return DocTokenizerRoberta(maxlen, name)
    else:
        raise NotImplementedError(f'Model type: {model_type} is not supported.')


class QueryTokenizer:

    def __init__(self, query_maxlen, model_type, attend_to_mask_tokens):
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(model_type)
        self.query_maxlen = query_maxlen
        self.background_maxlen = 512 - self.query_maxlen + 1
        self.attend_to_mask_tokens = attend_to_mask_tokens
        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.convert_tokens_to_ids('[unused0]')
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id
        assert self.Q_marker_token_id == 1 and self.mask_token_id == 103
        self.used = False

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)
        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]
        if not add_special_tokens:
            return tokens
        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [(prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst) + 3))) for lst in tokens]
        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)
        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']
        if not add_special_tokens:
            return ids
        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
        ids = [(prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst) + 3))) for lst in ids]
        return ids

    def tensorize(self, batch_text, bsize=None, context=None):
        assert type(batch_text) in [list, tuple], type(batch_text)
        batch_text = [('. ' + x) for x in batch_text]
        obj = self.tok(batch_text, padding='max_length', truncation=True, return_tensors='pt', max_length=self.query_maxlen)
        ids, mask = obj['input_ids'], obj['attention_mask']
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id
        if context is not None:
            assert len(context) == len(batch_text), (len(context), len(batch_text))
            obj_2 = self.tok(context, padding='longest', truncation=True, return_tensors='pt', max_length=self.background_maxlen)
            ids_2, mask_2 = obj_2['input_ids'][:, 1:], obj_2['attention_mask'][:, 1:]
            ids = torch.cat((ids, ids_2), dim=-1)
            mask = torch.cat((mask, mask_2), dim=-1)
        if self.attend_to_mask_tokens:
            mask[ids == self.mask_token_id] = 1
            assert mask.sum().item() == mask.size(0) * mask.size(1), mask
        if not self.used:
            self.used = True
            firstbg = context is None or context[0]
            print_message('#> BERT QueryTokenizer.tensorize(batch_text[0], batch_background[0], bsize) ==')
            print_message(f'#> Input: {batch_text[0]}, \t\t {firstbg}, \t\t {bsize}')
            print_message(f'#> Output IDs: {ids[0].size()}, {ids[0]}')
            print_message(f'#> Output Mask: {mask[0].size()}, {mask[0]}')
        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches
        return ids, mask


class QueryTokenizerRoberta:

    def __init__(self, query_maxlen, model_type):
        self.tok = HF_ColBERT_Roberta.raw_tokenizer_from_pretrained(model_type)
        self.query_maxlen = query_maxlen
        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.convert_tokens_to_ids('madeupword0000')
        self.mask_token, self.mask_token_id = self.tok.pad_token, self.tok.pad_token_id
        self.used = False

    def tokenize(self, batch_text, add_special_tokens=False):
        raise NotImplementedError()

    def encode(self, batch_text, add_special_tokens=False):
        raise NotImplementedError()

    def tensorize(self, batch_text, bsize=None, context=None):
        assert type(batch_text) in [list, tuple], type(batch_text)
        batch_text = [('$ ' + x) for x in batch_text]
        obj = self.tok(batch_text, padding='max_length', truncation=True, return_tensors='pt', max_length=self.query_maxlen)
        ids, mask = obj['input_ids'], obj['attention_mask']
        ids[:, 1] = self.Q_marker_token_id
        if context is not None:
            print_message(f'#> length of context: {len(context)}')
        if not self.used:
            self.used = True
            firstbg = context is None or context[0]
            print_message('#> Roberta QueryTokenizer.tensorize(batch_text[0], batch_background[0], bsize) ==')
            print_message(f'#> Input: {batch_text[0]}, \t\t {firstbg}, \t\t {bsize}')
            print_message(f'#> Output IDs: {ids[0].size()}, {ids[0]}')
            print_message(f'#> Output Mask: {mask[0].size()}, {mask[0]}')
        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches
        return ids, mask


class QueryTokenizerXLMR:

    def __init__(self, query_maxlen, model_type):
        self.tok = HF_ColBERT_XLMR.raw_tokenizer_from_pretrained(model_type)
        self.query_maxlen = query_maxlen
        self.Q_marker_token, self.Q_marker_token_id = '?', 9748
        self.mask_token, self.mask_token_id = self.tok.pad_token, self.tok.pad_token_id
        self.used = False

    def tokenize(self, batch_text, add_special_tokens=False):
        raise NotImplementedError()

    def encode(self, batch_text, add_special_tokens=False):
        raise NotImplementedError()

    def tensorize(self, batch_text, bsize=None, context=None):
        assert type(batch_text) in [list, tuple], type(batch_text)
        batch_text = [('$ ' + x) for x in batch_text]
        obj = self.tok(batch_text, padding='max_length', truncation=True, return_tensors='pt', max_length=self.query_maxlen)
        ids, mask = obj['input_ids'], obj['attention_mask']
        ids[:, 1] = self.Q_marker_token_id
        if context is not None:
            print_message(f'#> length of context: {len(context)}')
        if not self.used:
            self.used = True
            firstbg = context is None or context[0]
            print_message('#> XMLR QueryTokenizer.tensorize(batch_text[0], batch_background[0], bsize) ==')
            print_message(f'#> Input: {batch_text[0]}, \t\t {firstbg}, \t\t {bsize}')
            print_message(f'#> Output IDs: {ids[0].size()}, {ids[0]}')
            print_message(f'#> Output Mask: {mask[0].size()}, {mask[0]}')
        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches
        return ids, mask


def get_query_tokenizer(name, colbert_config):
    model_type = get_model_type(name)
    maxlen = colbert_config.query_maxlen
    attend_to_mask_tokens = colbert_config.attend_to_mask_tokens
    print_message(f'factory model type: {model_type}')
    if model_type == 'bert':
        return QueryTokenizer(maxlen, name, attend_to_mask_tokens)
    elif model_type == 'xlm-roberta':
        return QueryTokenizerXLMR(maxlen, name)
    elif model_type == 'roberta':
        return QueryTokenizerRoberta(maxlen, name)
    else:
        raise NotImplementedError(f'Model type: {model_type} is not supported.')


class Checkpoint(ColBERT):
    """
        Easy inference with ColBERT.

        TODO: Add .cast() accepting [also] an object instance-of(Checkpoint) as first argument.
    """

    def __init__(self, name, colbert_config=None):
        super().__init__(name, colbert_config)
        assert self.training is False
        self.query_tokenizer = get_query_tokenizer(name, colbert_config)
        self.doc_tokenizer = get_doc_tokenizer(name, colbert_config)
        self.amp_manager = MixedPrecisionManager(True)
        self.docFromText_used = False

    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = super().query(*args, **kw_args)
                return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = super().doc(*args, **kw_args)
                if to_cpu:
                    return (D[0].cpu(), *D[1:]) if isinstance(D, tuple) else D.cpu()
                return D

    def queryFromText(self, queries, bsize=None, to_cpu=False, context=None):
        if bsize:
            batches = self.query_tokenizer.tensorize(queries, context=context, bsize=bsize)
            batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
            return torch.cat(batches)
        input_ids, attention_mask = self.query_tokenizer.tensorize(queries, context=context)
        return self.query(input_ids, attention_mask)

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False, showprogress=False, return_tokens=False):
        assert keep_dims in [True, False, 'flatten']
        if not self.docFromText_used:
            print_message(f'#> checkpoint, docFromText, Input: {docs[0]}, \t\t {bsize}')
        if bsize:
            text_batches, reverse_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)
            if not self.docFromText_used:
                print_message(f'#> checkpoint, docFromText, Output IDs: {text_batches[0]}')
                self.docFromText_used = True
            returned_text = []
            if return_tokens:
                returned_text = [text for batch in text_batches for text in batch[0]]
                returned_text = [returned_text[idx] for idx in reverse_indices.tolist()]
                returned_text = [returned_text]
            keep_dims_ = 'return_mask' if keep_dims == 'flatten' else keep_dims
            batches = [self.doc(input_ids, attention_mask, keep_dims=keep_dims_, to_cpu=to_cpu) for input_ids, attention_mask in tqdm(text_batches, disable=not showprogress)]
            if keep_dims is True:
                D = _stack_3D_tensors(batches)
                return D[reverse_indices], *returned_text
            elif keep_dims == 'flatten':
                D, mask = [], []
                for D_, mask_ in batches:
                    D.append(D_)
                    mask.append(mask_)
                D, mask = torch.cat(D)[reverse_indices], torch.cat(mask)[reverse_indices]
                doclens = mask.squeeze(-1).sum(-1).tolist()
                D = D.view(-1, self.colbert_config.dim)
                D = D[mask.bool().flatten()].cpu()
                return D, doclens, *returned_text
            assert keep_dims is False
            D = [d for batch in batches for d in batch]
            return [D[idx] for idx in reverse_indices.tolist()], *returned_text
        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)

    def lazy_rank(self, queries, docs):
        Q = self.queryFromText(queries, bsize=128, to_cpu=True)
        D = self.docFromText(docs, bsize=128, to_cpu=True)
        assert False, 'Implement scoring'

    def score(self, Q, D, mask=None, lengths=None):
        assert False, 'Call colbert_score'
        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"
            mask = torch.arange(D.size(1), device=self.device) + 1
            mask = mask.unsqueeze(0) <= lengths.unsqueeze(-1)
        scores = D @ Q
        scores = scores if mask is None else scores * mask.unsqueeze(-1)
        scores = scores.max(1)
        return scores.values.sum(-1).cpu()


class EncoderWrapper(torch.nn.Module):
    """
    EncoderWrapper for the FiD model
    
    B - Batch size
    N the number of passages per example
    L the max seq length
    
    The EncoderWrapper transforms the input from  B * (N * L) to (B * N) * L
    Every passage of size L is encoded separatelly 
    After the encoder, concatenate encoder output for all N passages
    """

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()
        self.encoder = encoder
        self.main_input_name = encoder.main_input_name

    def forward(self, input_ids=None, attention_mask=None, return_dict=False, **kwargs):
        if input_ids.dim() == 3:
            input_ids = input_ids.view(input_ids.size(0), -1)
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        if not return_dict:
            return (outputs[0].view(bsz, self.n_passages * passage_length, -1),) + outputs[1:]
        return BaseModelOutput(last_hidden_state=outputs[0].view(bsz, self.n_passages * passage_length, -1))


logger = logging.getLogger(__name__)


class BiEncoder(torch.nn.Module):
    """
    This trains the DPR encoders to maximize dot product between queries and positive contexts.
    We only use this model during training.
    """

    def __init__(self, hypers: 'BiEncoderHypers'):
        super().__init__()
        self.hypers = hypers
        logger.info(f'BiEncoder: initializing from {hypers.qry_encoder_name_or_path} and {hypers.ctx_encoder_name_or_path}')
        self.qry_model = EncoderWrapper(DPRQuestionEncoder.from_pretrained(hypers.qry_encoder_name_or_path))
        self.ctx_model = EncoderWrapper(DPRContextEncoder.from_pretrained(hypers.ctx_encoder_name_or_path))
        self.saved_debug = False

    def encode(self, model, input_ids: 'torch.Tensor', attention_mask: 'torch.Tensor'):
        if 0 < self.hypers.encoder_gpu_train_limit:
            dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
            all_pooled_output = []
            for sub_bndx in range(0, input_ids.shape[0], self.hypers.encoder_gpu_train_limit):
                sub_input_ids = input_ids[sub_bndx:sub_bndx + self.hypers.encoder_gpu_train_limit]
                sub_attention_mask = attention_mask[sub_bndx:sub_bndx + self.hypers.encoder_gpu_train_limit]
                pooler_output = checkpoint(model, sub_input_ids, sub_attention_mask, dummy_tensor)
                all_pooled_output.append(pooler_output)
            return torch.cat(all_pooled_output, dim=0)
        else:
            return model(input_ids, attention_mask, None)

    def gather(self, tensor):
        dtensor = tensor.detach()
        gather_list = [torch.zeros_like(dtensor) for _ in range(self.hypers.world_size)]
        torch.distributed.all_gather(gather_list, dtensor)
        gather_list[self.hypers.global_rank] = tensor
        return torch.cat(gather_list, 0)

    def save_for_debug(self, qry_reps, ctx_reps, positive_indices):
        if self.hypers.global_rank == 0 and not self.saved_debug and self.hypers.debug_location and not os.path.exists(self.hypers.debug_location):
            os.makedirs(self.hypers.debug_location)
            torch.save(qry_reps, os.path.join(self.hypers.debug_location, 'qry_reps.bin'))
            torch.save(ctx_reps, os.path.join(self.hypers.debug_location, 'ctx_reps.bin'))
            torch.save(positive_indices, os.path.join(self.hypers.debug_location, 'positive_indices.bin'))
            self.saved_debug = True
            logger.warning(f'saved debug info at {self.hypers.debug_location}')

    def forward(self, input_ids_q: 'torch.Tensor', attention_mask_q: 'torch.Tensor', input_ids_c: 'torch.Tensor', attention_mask_c: 'torch.Tensor', positive_indices: 'torch.Tensor'):
        """
        All batches must be the same size (q and c are fixed during training)
        :param input_ids_q: q x seq_len_q [0, vocab_size)
        :param attention_mask_q: q x seq_len_q [0, 1]
        :param input_ids_c: c x seq_len_c
        :param attention_mask_c: c x seq_len_c
        :param positive_indices: q [0, c)
        :return:
        """
        qry_reps = self.encode(self.qry_model, input_ids_q, attention_mask_q)
        ctx_reps = self.encode(self.ctx_model, input_ids_c, attention_mask_c)
        if self.hypers.world_size > 1:
            positive_indices = self.gather(positive_indices + self.hypers.global_rank * ctx_reps.shape[0])
            qry_reps = self.gather(qry_reps)
            ctx_reps = self.gather(ctx_reps)
        self.save_for_debug(qry_reps, ctx_reps, positive_indices)
        dot_products = torch.matmul(qry_reps, ctx_reps.transpose(0, 1))
        probs = F.log_softmax(dot_products, dim=1)
        loss = F.nll_loss(probs, positive_indices)
        predictions = torch.max(probs, 1)[1]
        accuracy = (predictions == positive_indices).sum() / positive_indices.shape[0]
        return loss, accuracy

    def save(self, save_dir: 'Union[str, os.PathLike]'):
        self.qry_model.encoder.save_pretrained(os.path.join(save_dir, 'qry_encoder'))
        self.ctx_model.encoder.save_pretrained(os.path.join(save_dir, 'ctx_encoder'))


class RowClassifierSC(nn.Module):

    def __init__(self, bert_model, state_dict=None):
        super(RowClassifierSC, self).__init__()
        self.num_labels = 2
        if state_dict:
            self.model = BertForSequenceClassification.from_pretrained(bert_model, state_dict=state_dict)
        else:
            self.model = BertForSequenceClassification.from_pretrained(bert_model)

    def forward(self, q_r_input, labels=None):
        """
        The forward function is the core of a PyTorch module. It defines what happens when you call an instantiated
        object of this class with one or more tensors as arguments. In this case, it takes in two tensors: the raw
        question and passage text, and returns a single output (the un-normalized logits for each class). The forward function
        is where most of your code will go.
        
        Args:
            self: Access the attributes and methods of the class
            q_r_input: Pass the question and the passage to the model
            labels: Calculate the loss function
        
        Returns:
            The output of the model
        """
        outputs = self.model(**q_r_input, labels=labels)
        return outputs

