
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


import functools


import inspect


import re


import warnings


from enum import Enum


from types import FunctionType


from typing import TYPE_CHECKING


from typing import Any


from typing import Callable


from typing import Container


from typing import Dict


from typing import Iterable


from typing import List


from typing import Mapping


from typing import Optional


from typing import Sequence


from typing import Set


from typing import Tuple


from typing import Type


from typing import TypeVar


from typing import Union


import abc


import random


from collections import namedtuple


from copy import copy


from functools import wraps


from inspect import isgeneratorfunction


from inspect import signature


from typing import Generic


import torch


import torch.nn.functional as F


from collections import defaultdict


import logging


import math


import time


import torch.optim


from itertools import chain


from typing import Collection


import pandas as pd


import torch.nn


ALL_CACHES = object()


class CurriedFactory:

    def __init__(self, func, kwargs):
        self.kwargs = kwargs
        self.factory = func
        self.instantiated = None
        self.error = None

    def maybe_nlp(self) ->Union['CurriedFactory', Any]:
        """
        If the factory requires an nlp argument and the user has explicitly
        provided it (this is unusual, we usually expect the factory to be
        instantiated via add_pipe, or a config), then we should instantiate
        it.

        Returns
        -------
        Union["CurriedFactory", Any]
        """
        sig = inspect.signature(self.factory)
        if 'nlp' not in sig.parameters or 'nlp' in self.kwargs:
            return self.factory(**self.kwargs)
        return self

    def instantiate(obj: 'Any', nlp: "'edsnlp.Pipeline'", path: 'Optional[Sequence[str]]'=()):
        """
        To ensure compatibility with spaCy's API, we need to support
        passing in the nlp object and name to factories. Since they can be
        nested, we need to add them to every factory in the config.
        """
        if isinstance(obj, CurriedFactory):
            if obj.error is not None:
                raise obj.error
            if obj.instantiated is not None:
                return obj.instantiated
            name = path[0] if len(path) == 1 else None
            parameters = inspect.signature(obj.factory.__init__).parameters if isinstance(obj.factory, type) else inspect.signature(obj.factory).parameters
            kwargs = {key: CurriedFactory.instantiate(obj=value, nlp=nlp, path=(*path, key)) for key, value in obj.kwargs.items()}
            try:
                if nlp and 'nlp' in parameters:
                    kwargs['nlp'] = nlp
                if name and 'name' in parameters:
                    kwargs['name'] = name
                obj.instantiated = obj.factory(**kwargs)
            except ConfitValidationError as e:
                obj.error = e
                raise ConfitValidationError(patch_errors(e.raw_errors, path, model=e.model), model=e.model, name=obj.factory.__module__ + '.' + obj.factory.__qualname__)
            return obj.instantiated
        elif isinstance(obj, dict):
            instantiated = {}
            errors = []
            for key, value in obj.items():
                try:
                    instantiated[key] = CurriedFactory.instantiate(obj=value, nlp=nlp, path=(*path, key))
                except ConfitValidationError as e:
                    errors.extend(e.raw_errors)
            if errors:
                raise ConfitValidationError(errors)
            return instantiated
        elif isinstance(obj, (tuple, list)):
            instantiated = []
            errors = []
            for i, value in enumerate(obj):
                try:
                    instantiated.append(CurriedFactory.instantiate(value, nlp, (*path, str(i))))
                except ConfitValidationError as e:
                    errors.append(e.raw_errors)
            if errors:
                raise ConfitValidationError(errors)
            return type(obj)(instantiated)
        else:
            return obj

    def _raise_curried_factory_error(self):
        raise TypeError(f'This component CurriedFactory({self.factory}) has not been instantiated yet, likely because it was missing an `nlp` pipeline argument. You should either:\n- add it to a pipeline: `pipe = nlp.add_pipe(pipe)`\n- or fill its `nlp` argument: `pipe = factory(nlp=nlp, ...)`')

    def __call__(self, *args, **kwargs):
        self._raise_curried_factory_error()

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        self._raise_curried_factory_error()


class BaseComponentMeta(abc.ABCMeta):

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        sig = inspect.signature(cls.__init__)
        sig = sig.replace(parameters=tuple(sig.parameters.values())[1:])
        cls.__signature__ = sig

    def __call__(cls, nlp=inspect.Signature.empty, *args, **kwargs):
        sig = inspect.signature(cls.__init__)
        bound = sig.bind_partial(None, nlp, *args, **kwargs)
        bound.arguments.pop('self', None)
        if 'nlp' in sig.parameters and sig.parameters['nlp'].default is sig.empty and bound.arguments.get('nlp', sig.empty) is sig.empty:
            return CurriedFactory(cls, bound.arguments)
        if nlp is inspect.Signature.empty:
            bound.arguments.pop('nlp', None)
        return super().__call__(**bound.arguments)


def value_getter(span: 'Span'):
    key = span._._get_key('value')
    if key in span.doc.user_data:
        return span.doc.user_data[key]
    return span._.get(span.label_) if span._.has(span.label_) else None


class BaseComponent(abc.ABC, metaclass=BaseComponentMeta):
    """
    The `BaseComponent` adds a `set_extensions` method,
    called at the creation of the object.

    It helps decouple the initialisation of the pipeline from
    the creation of extensions, and is particularly usefull when
    distributing EDSNLP on a cluster, since the serialisation mechanism
    imposes that the extensions be reset.
    """

    def __init__(self, nlp: 'Optional[PipelineProtocol]'=None, name: 'Optional[str]'=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.set_extensions()

    def set_extensions(self):
        """
        Set `Doc`, `Span` and `Token` extensions.
        """
        if Span.has_extension('value'):
            if Span.get_extension('value')[2] is not value_getter:
                warnings.warn("A Span extension 'value' already exists with a different getter. Keeping the existing extension, but some components of edsnlp may not work as expected.")
            return
        Span.set_extension('value', getter=value_getter)

    def get_spans(self, doc: 'Doc'):
        """
        Returns sorted spans of interest according to the
        possible value of `on_ents_only`.
        Includes `doc.ents` by default, and adds eventual SpanGroups.
        """
        ents = list(doc.ents) + list(doc.spans.get('discarded', []))
        on_ents_only = getattr(self, 'on_ents_only', None)
        if isinstance(on_ents_only, str):
            on_ents_only = [on_ents_only]
        if isinstance(on_ents_only, (set, list)):
            for spankey in (set(on_ents_only) & set(doc.spans.keys())):
                ents.extend(doc.spans.get(spankey, []))
        return sorted(list(set(ents)), key=attrgetter('start', 'end'))

    def _boundaries(self, doc: 'Doc', terminations: 'Optional[List[Span]]'=None) ->List[Tuple[int, int]]:
        """
        Create sub sentences based sentences and terminations found in text.

        Parameters
        ----------
        doc:
            spaCy Doc object
        terminations:
            List of tuples with (match_id, start, end)

        Returns
        -------
        boundaries:
            List of tuples with (start, end) of spans
        """
        if terminations is None:
            terminations = []
        sent_starts = [sent.start for sent in doc.sents]
        termination_starts = [t.start for t in terminations]
        starts = sent_starts + termination_starts + [len(doc)]
        starts = list(set(starts))
        starts.sort()
        boundaries = [(start, end) for start, end in zip(starts[:-1], starts[1:])]
        return boundaries

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.set_extensions()


BatchInput = TypeVar('BatchInput', bound=Dict[str, Any])


BatchOutput = TypeVar('BatchOutput', bound=Dict[str, Any])


_caches = {}


def cached(key, store_key=False):

    def wrapper(fn):

        @wraps(fn)
        def wrapped(self: "'TorchComponent'", *args, **kwargs):
            if self._current_cache_id is None:
                return fn(self, *args, **kwargs)
            cache_key = fn.__name__, f'{self.__class__.__name__}<{id(self)}>', key(self, *args, **kwargs)
            cache = _caches[self._current_cache_id]
            if cache_key in cache:
                return cache[cache_key]
            res = fn(self, *args, **kwargs)
            cache[cache_key] = res
            if store_key:
                res['__cache_key__'] = cache_key
            return res
        wrapped._cached = fn
        return wrapped
    return wrapper


def hash_batch(batch):
    if isinstance(batch, list):
        return hash(tuple(id(item) for item in batch))
    elif not isinstance(batch, dict):
        return id(batch)
    if '__batch_hash__' in batch:
        return batch['__batch_hash__']
    batch_hash = hash((tuple(batch.keys()), tuple(map(hash_batch, batch.values()))))
    batch['__batch_hash__'] = batch_hash
    return batch_hash


def _cached_batch_to_device(fn):
    return cached(lambda self, batch, device: (hash_batch(batch), device))(fn)


def _cached_collate(fn):
    if hasattr(fn, '_cached'):
        return fn
    return cached(lambda self, batch, *args, **kwargs: hash_batch(batch), store_key=True)(fn)


def _cached_forward(fn):
    if hasattr(fn, '_cached'):
        return fn
    return cached(lambda self, batch, *args, **kwargs: hash_batch(batch))(fn)


def hash_inputs(inputs):
    res = io.BytesIO()
    p = pickle.Pickler(res)
    p.dispatch_table = dispatch_table
    hashed = xxhash.xxh3_64()
    p.dump(inputs)
    hashed.update(res.getvalue())
    return hashed.hexdigest()


def _cached_preprocess(fn):
    if hasattr(fn, '_cached'):
        return fn
    return cached(lambda self, *args, **kwargs: hash_inputs((*args, sorted(kwargs.items(), key=lambda x: x[0]))))(fn)


def _cached_preprocess_supervised(fn):
    if hasattr(fn, '_cached'):
        return fn
    return cached(lambda self, *args, **kwargs: hash_inputs((*args, sorted(kwargs.items(), key=lambda x: x[0]))))(fn)


class TorchComponentMeta(BaseComponentMeta):

    def __new__(mcs, name, bases, class_dict):
        if 'preprocess' in class_dict:
            class_dict['preprocess'] = _cached_preprocess(class_dict['preprocess'])
        if 'preprocess_supervised' in class_dict:
            class_dict['preprocess_supervised'] = _cached_preprocess_supervised(class_dict['preprocess_supervised'])
        if 'collate' in class_dict:
            class_dict['collate'] = _cached_collate(class_dict['collate'])
        if 'batch_to_device' in class_dict:
            class_dict['batch_to_device'] = _cached_batch_to_device(class_dict['batch_to_device'])
        if 'forward' in class_dict:
            class_dict['forward'] = _cached_forward(class_dict['forward'])
        return super().__new__(mcs, name, bases, class_dict)


FLATTEN_TEMPLATE = """def flatten(root):
    res={}
    return res
"""


def _discover_scheme(obj):
    keys = defaultdict(lambda : [])

    def rec(current, path):
        if not isinstance(current, dict):
            keys[id(current)].append(path)
            return
        for key, value in current.items():
            if not key.startswith('$'):
                rec(value, (*path, key))
    rec(obj, ())
    code = FLATTEN_TEMPLATE.format('{' + '\n'.join('{}: root{},'.format(repr('|'.join(map('/'.join, key_list))), ''.join(f'[{repr(k)}]' for k in key_list[0])) for key_list in keys.values()) + '}')
    return code


class batch_compress_dict:
    """
    Compress a sequence of dictionaries in which values that occur multiple times are
    deduplicated. The corresponding keys will be merged into a single string using
    the "|" character as a separator.
    This is useful to preserve referential identities when decompressing the dictionary
    after it has been serialized and deserialized.

    Parameters
    ----------
    seq: Iterable[Dict[str, Any]]
        Sequence of dictionaries to compress
    """
    __slots__ = 'flatten', 'seq'

    def __init__(self, seq: 'Optional[Iterable[Dict[str, Any]]]'=None):
        self.seq = seq
        self.flatten = None

    def __iter__(self):
        return batch_compress_dict(iter(self.seq))

    def __call__(self, item):
        exec_result = {}
        if self.flatten is None:
            exec(_discover_scheme(item), {}, exec_result)
            self.flatten = exec_result['flatten']
        return self.flatten(item)

    def __next__(self) ->Dict[str, List]:
        return self(next(self.seq))


class StreamSentinel:
    pass


T = TypeVar('T')


def batchify(iterable: 'Iterable[T]', batch_size: 'int', drop_last: 'bool'=False, sentinel_mode: "Literal['drop', 'keep', 'split']"='drop') ->Iterable[List[T]]:
    """
    Yields batch that contain at most `batch_size` elements.
    If an item contains more than `batch_size` elements, it will be yielded as a single
    batch.

    Parameters
    ----------
    iterable: Iterable[T]
        The iterable to batchify
    batch_size: int
        The maximum number of elements in a batch
    drop_last: bool
        Whether to drop the last batch if it is smaller than `batch_size`
    sentinel_mode: Literal["auto", "drop", "keep", "split"]
        How to handle the sentinel values in the iterable:

        - "drop": drop sentinel values
        - "keep": keep sentinel values inside the produced batches
        - "split": split batches at sentinel values and yield sentinel values
            separately
    """
    assert sentinel_mode in ('drop', 'keep', 'split')
    batch = []
    num_items = 0
    for item in iterable:
        if isinstance(item, StreamSentinel):
            if sentinel_mode == 'split':
                if num_items > 0:
                    yield batch
                yield item
                batch = []
                num_items = 0
            elif sentinel_mode == 'keep':
                batch.append(item)
            continue
        if num_items >= batch_size:
            yield batch
            batch = []
            num_items = 0
        batch.append(item)
        num_items += 1
    if num_items > 0 and not drop_last:
        yield batch


def ld_to_dl(ld: 'Iterable[Mapping[str, T]]') ->Dict[str, List[T]]:
    """
    Convert a list of dictionaries to a dictionary of lists

    Parameters
    ----------
    ld: Iterable[Mapping[str, T]]
        The list of dictionaries

    Returns
    -------
    Dict[str, List[T]]
        The dictionary of lists
    """
    ld = list(ld)
    return {k: [dic.get(k) for dic in ld] for k in (ld[0] if len(ld) else ())}


def decompress_dict(seq: 'Union[Iterable[Dict[str, Any]], Dict[str, Any]]'):
    """
    Decompress a dictionary of lists into a sequence of dictionaries.
    This function assumes that the dictionary structure was obtained using the
    `batch_compress_dict` class.
    Keys that were merged into a single string using the "|" character as a separator
    will be split into a nested dictionary structure.

    Parameters
    ----------
    seq: Union[Iterable[Dict[str, Any]], Dict[str, Any]]
        The dictionary to decompress or a sequence of dictionaries to decompress

    Returns
    -------

    """
    obj = ld_to_dl(seq) if isinstance(seq, Sequence) else seq
    res = {}
    for key, value in obj.items():
        for path in key.split('|'):
            current = res
            parts = path.split('/')
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
    return res


IMPOSSIBLE = -10000


@torch.jit.script
def logsumexp_reduce(log_A, log_B):
    return (log_A.unsqueeze(-1) + log_B.unsqueeze(-3)).logsumexp(-2)


def repeat_like(x, y):
    return x.repeat(tuple(a if b == 1 else 1 for a, b in zip(y.shape, x.shape)))


def masked_flip(x, mask, dim_x=-2):
    mask = repeat_like(mask, x)
    flipped_x = torch.zeros_like(x)
    flipped_x[mask] = x.flip(dim_x)[mask.flip(-1)]
    return flipped_x


@torch.jit.script
def max_reduce(log_A, log_B):
    return (log_A.unsqueeze(-1) + log_B.unsqueeze(-3)).max(-2)


class LinearChainCRF(torch.nn.Module):

    def __init__(self, forbidden_transitions, start_forbidden_transitions=None, end_forbidden_transitions=None, learnable_transitions=True, with_start_end_transitions=True):
        """
        A linear chain CRF in Pytorch

        Parameters
        ----------
        forbidden_transitions: torch.BoolTensor
            Shape: n_tags * n_tags
            Impossible transitions (1 means impossible) from position n to position n+1
        start_forbidden_transitions: Optional[torch.BoolTensor]
            Shape: n_tags
            Impossible transitions at the start of a sequence
        end_forbidden_transitions Optional[torch.BoolTensor]
            Shape: n_tags
            Impossible transitions at the end of a sequence
        learnable_transitions: bool
            Should we learn transition scores to complete the
            constraints ?
        with_start_end_transitions:
            Should we apply start-end transitions.
            If learnable_transitions is True, learn start/end transition scores
        """
        super().__init__()
        num_tags = forbidden_transitions.shape[0]
        self.with_start_end_transitions = with_start_end_transitions
        self.register_buffer('forbidden_transitions', forbidden_transitions.bool())
        self.register_buffer('start_forbidden_transitions', start_forbidden_transitions.bool() if start_forbidden_transitions is not None else torch.zeros(num_tags, dtype=torch.bool))
        self.register_buffer('end_forbidden_transitions', end_forbidden_transitions.bool() if end_forbidden_transitions is not None else torch.zeros(num_tags, dtype=torch.bool))
        if learnable_transitions:
            self.transitions = torch.nn.Parameter(torch.zeros_like(forbidden_transitions, dtype=torch.float))
        else:
            self.register_buffer('transitions', torch.zeros_like(forbidden_transitions, dtype=torch.float))
        if learnable_transitions and with_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.zeros(num_tags, dtype=torch.float))
        else:
            self.register_buffer('start_transitions', torch.zeros(num_tags, dtype=torch.float))
        if learnable_transitions and with_start_end_transitions:
            self.end_transitions = torch.nn.Parameter(torch.zeros(num_tags, dtype=torch.float))
        else:
            self.register_buffer('end_transitions', torch.zeros(num_tags, dtype=torch.float))

    def decode(self, emissions, mask):
        """
        Decodes a sequence of tag scores using the Viterbi algorithm

        Parameters
        ----------
        emissions: torch.FloatTensor
            Shape: ... * n_tokens * n_tags
        mask: torch.BoolTensor
            Shape: ... * n_tokens

        Returns
        -------
        torch.LongTensor
            Backtrack indices (= argmax), ie best tag sequence
        """
        transitions = self.transitions.masked_fill(self.forbidden_transitions, IMPOSSIBLE)
        start_transitions = end_transitions = torch.zeros_like(self.start_transitions)
        if self.with_start_end_transitions:
            start_transitions = self.start_transitions.masked_fill(self.start_forbidden_transitions, IMPOSSIBLE)
            end_transitions = self.end_transitions.masked_fill(self.end_forbidden_transitions, IMPOSSIBLE)
        path = torch.zeros(*emissions.shape[:-1], dtype=torch.long)
        if 0 not in emissions.shape:
            emissions[..., 1:][~mask] = IMPOSSIBLE
            emissions = emissions.unbind(1)
            out = [emissions[0] + start_transitions]
            backtrack = []
            for k in range(1, len(emissions)):
                res, indices = max_reduce(out[-1], transitions)
                backtrack.append(indices)
                out.append(res + emissions[k])
            res, indices = max_reduce(out[-1], end_transitions.unsqueeze(-1))
            path[:, -1] = indices.squeeze(-1)
            if len(backtrack) > 1:
                for k, b in enumerate(backtrack[::-1]):
                    path[:, -k - 2] = index_dim(b, path[:, -k - 1], dim=-1)
        return path

    def marginal(self, emissions, mask):
        """
        Compute the marginal log-probabilities of the tags
        given the emissions and the transition probabilities and
        constraints of the CRF

        We could use the `propagate` method but this implementation
        is faster.

        Parameters
        ----------
        emissions: torch.FloatTensor
            Shape: ... * n_tokens * n_tags
        mask: torch.BoolTensor
            Shape: ... * n_tokens

        Returns
        -------
        torch.FloatTensor
            Shape: ... * n_tokens * n_tags
        """
        device = emissions.device
        transitions = self.transitions.masked_fill(self.forbidden_transitions, IMPOSSIBLE)
        start_transitions = end_transitions = torch.zeros_like(self.start_transitions)
        if self.with_start_end_transitions:
            start_transitions = self.start_transitions.masked_fill(self.start_forbidden_transitions, IMPOSSIBLE)
            end_transitions = self.end_transitions.masked_fill(self.end_forbidden_transitions, IMPOSSIBLE)
        bi_transitions = torch.stack([transitions, transitions.t()], dim=0).unsqueeze(1)
        emissions[:, 0] = emissions[:, 0] + start_transitions
        emissions[torch.arange(mask.shape[0], device=device), mask.long().sum(1) - 1] = emissions[torch.arange(mask.shape[0], device=device), mask.long().sum(1) - 1] + end_transitions
        bi_emissions = torch.stack([emissions, masked_flip(emissions, mask, dim_x=1)], 0).unbind(2)
        out = [bi_emissions[0]]
        for word_bi_emissions in bi_emissions[1:]:
            res = logsumexp_reduce(out[-1], bi_transitions)
            out.append(res + word_bi_emissions)
        out = torch.stack(out, dim=2)
        forward = out[0]
        backward = masked_flip(out[1], mask, dim_x=1)
        backward_z = backward.logsumexp(-1)
        return forward + backward - emissions - backward_z[:, :, :, None]

    def forward(self, emissions, mask, target):
        """
        Compute the posterior reduced log-probabilities of the tags
        given the emissions and the transition probabilities and
        constraints of the CRF, ie the loss.


        We could use the `propagate` method but this implementation
        is faster.

        Parameters
        ----------
        emissions: torch.FloatTensor
            Shape: n_samples * n_tokens * ... * n_tags
        mask: torch.BoolTensor
            Shape: n_samples * n_tokens * ...
        target: torch.BoolTensor
            Shape: n_samples * n_tokens * ... * n_tags
            The target tags represented with 1-hot encoding
            We use 1-hot instead of long format to handle
            cases when multiple tags at a given position are
            allowed during training.

        Returns
        -------
        torch.FloatTensor
            Shape: ...
            The loss
        """
        transitions = self.transitions.masked_fill(self.forbidden_transitions, IMPOSSIBLE)
        start_transitions = end_transitions = torch.zeros_like(self.start_transitions)
        if self.with_start_end_transitions:
            start_transitions = self.start_transitions.masked_fill(self.start_forbidden_transitions, IMPOSSIBLE)
            end_transitions = self.end_transitions.masked_fill(self.end_forbidden_transitions, IMPOSSIBLE)
        bi_emissions = torch.stack([emissions.masked_fill(~target, IMPOSSIBLE), emissions], 0).unbind(2)
        out = [bi_emissions[0] + start_transitions]
        for word_bi_emissions in bi_emissions[1:]:
            res = logsumexp_reduce(out[-1], transitions)
            out.append(res + word_bi_emissions)
        last_out = torch.stack([out[length - 1][:, i] for i, length in enumerate(mask.long().sum(1).tolist())], dim=1) + end_transitions
        supervised_z, unsupervised_z = last_out.logsumexp(-1)
        return -(supervised_z - unsupervised_z)


class MultiLabelBIOULDecoder(LinearChainCRF):

    def __init__(self, num_labels, with_start_end_transitions=True, learnable_transitions=True):
        """
        Create a linear chain CRF with hard constraints to enforce the BIOUL tagging
        scheme

        Parameters
        ----------
        num_labels: int
        with_start_end_transitions: bool
        learnable_transitions: bool
        """
        O, I, B, L, U = 0, 1, 2, 3, 4
        num_tags = 1 + num_labels * 4
        self.num_tags = num_tags
        forbidden_transitions = torch.ones(num_tags, num_tags, dtype=torch.bool)
        forbidden_transitions[O, O] = 0
        for i in range(num_labels):
            STRIDE = 4 * i
            for j in range(num_labels):
                STRIDE_J = j * 4
                forbidden_transitions[L + STRIDE, B + STRIDE_J] = 0
                forbidden_transitions[L + STRIDE, U + STRIDE_J] = 0
                forbidden_transitions[U + STRIDE, B + STRIDE_J] = 0
                forbidden_transitions[U + STRIDE, U + STRIDE_J] = 0
            forbidden_transitions[O, B + STRIDE] = 0
            forbidden_transitions[B + STRIDE, I + STRIDE] = 0
            forbidden_transitions[I + STRIDE, I + STRIDE] = 0
            forbidden_transitions[I + STRIDE, L + STRIDE] = 0
            forbidden_transitions[B + STRIDE, L + STRIDE] = 0
            forbidden_transitions[L + STRIDE, O] = 0
            forbidden_transitions[O, U + STRIDE] = 0
            forbidden_transitions[U + STRIDE, O] = 0
        start_forbidden_transitions = torch.zeros(num_tags, dtype=torch.bool)
        for i in range(num_labels):
            STRIDE = 4 * i
            start_forbidden_transitions[I + STRIDE] = 1
            start_forbidden_transitions[L + STRIDE] = 1
        end_forbidden_transitions = torch.zeros(num_tags, dtype=torch.bool)
        for i in range(num_labels):
            STRIDE = 4 * i
            end_forbidden_transitions[I + STRIDE] = 1
            end_forbidden_transitions[B + STRIDE] = 1
        super().__init__(forbidden_transitions, start_forbidden_transitions, end_forbidden_transitions, with_start_end_transitions=with_start_end_transitions, learnable_transitions=learnable_transitions)

    @staticmethod
    def tags_to_spans(tags):
        """
        Convert a sequence of multiple label BIOUL tags to a sequence of spans

        Parameters
        ----------
        tags: torch.LongTensor
            Shape: n_samples * n_tokens * n_labels

        Returns
        -------
        torch.LongTensor
            Shape: n_spans *  4
            (doc_idx, begin, end, label_idx)
        """
        tags = tags.transpose(1, 2)
        tags_after = tags.roll(-1, 2)
        tags_before = tags.roll(1, 2)
        if 0 not in tags.shape:
            tags_after[..., -1] = 0
            tags_before[..., 0] = 0
        begins_indices = torch.nonzero((tags == 4) | (tags == 2) | ((tags == 1) | (tags == 3)) & (tags_before != 2) & (tags_before != 1))
        ends_indices = torch.nonzero((tags == 4) | (tags == 3) | ((tags == 1) | (tags == 2)) & (tags_after != 3) & (tags_after != 1))
        return torch.cat([begins_indices[..., :3], ends_indices[..., [2]] + 1], dim=-1)


class Metric(torch.nn.Module):
    """
    Metric layer, used for computing similarities between two sets of vectors. A typical
    use case is to compute the similarity between a set of query vectors (input
    embeddings) and a set of concept vectors (output embeddings).

    Parameters
    ----------
    in_features : int
        Size of the input embeddings
    out_features : int
        Size of the output embeddings
    num_groups : int
        Number of groups for the output embeddings, that can be used to filter out
        certain concepts that are not relevant for a given query (e.g. do not compare
        a drug with concepts for diseases)
    metric : Literal["cosine", "dot"]
        Whether to compute the cosine similarity between the input and output embeddings
        or the dot product.
    rescale: Optional[float]
        Rescale the output cosine similarities by a constant factor.
    """

    def __init__(self, in_features: 'int', out_features: 'int', num_groups: 'int'=0, metric: "Literal['cosine', 'dot']"='cosine', rescale: 'Optional[float]'=None, bias: 'bool'=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features))
        self.register_buffer('groups', torch.zeros(num_groups, out_features, dtype=torch.bool))
        self.rescale: 'float' = rescale if rescale is not None else 20.0 if metric == 'cosine' else 1.0
        self.metric = metric
        self.register_parameter('bias', torch.nn.Parameter(torch.tensor(-0.65 if metric == 'cosine' else 0.0)) if bias else None)
        self.reset_parameters()
        self._last_version = None
        self._normalized_weight = None

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def normalized_weight(self):
        if (self.weight._version, id(self.weight)) == self._last_version and not self.training and self._normalized_weight is not None:
            return self._normalized_weight
        normalized_weight = self.normalize_embedding(self.weight)
        if not self.training and normalized_weight is not self.weight:
            self._normalized_weight = normalized_weight
            self._last_version = self.weight._version, id(self.weight)
        return normalized_weight

    def normalize_embedding(self, inputs):
        if self.metric == 'cosine':
            inputs = F.normalize(inputs, dim=-1)
        return inputs

    def forward(self, inputs, group_indices=None, **kwargs):
        x = F.linear(self.normalize_embedding(inputs), self.normalized_weight())
        if self.bias is not None:
            x += self.bias
        if self.rescale != 1.0:
            x *= self.rescale
        if group_indices is not None and len(self.groups):
            x = x.masked_fill(~self.groups[group_indices], -10000)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, rescale={}, groups={}'.format(self.in_features, self.out_features, float(self.rescale or 1.0), self.groups.shape[0] if self.groups is not None else None)


class Residual(torch.nn.Module):

    def __init__(self, normalize: "Literal['pre', 'post', 'none']"='pre'):
        super().__init__()
        self.normalize = normalize

    def forward(self, before, after):
        return before + F.layer_norm(after, after.shape[-1:]) if self.normalize == 'pre' else F.layer_norm(before + after, after.shape[-1:]) if self.normalize == 'post' else before + after


def get_activation_function(activation: 'ActivationFunction'):
    return getattr(torch.nn.functional, activation)


class TextCnn(torch.nn.Module):

    def __init__(self, input_size: 'int', output_size: 'Optional[int]'=None, out_channels: 'Optional[int]'=None, kernel_sizes: 'Sequence[int]'=(3, 4, 5), activation: 'ActivationFunction'='relu', residual: 'bool'=True, normalize: "Literal['pre', 'post', 'none']"='pre'):
        """
        Parameters
        ----------
        input_size: int
            Size of the input embeddings
        output_size: Optional[int]
            Size of the output embeddings
            Defaults to the `input_size`
        out_channels: int
            Number of channels
        kernel_sizes: Sequence[int]
            Window size of each kernel
        activation: str
            Activation function to use
        residual: bool
            Whether to use residual connections
        normalize: Literal["pre", "post", "none"]
            Whether to normalize before or after the residual connection
        """
        super().__init__()
        if out_channels is None:
            out_channels = input_size
        output_size = input_size if output_size is None else output_size
        self.convolutions = torch.nn.ModuleList(torch.nn.Conv1d(in_channels=input_size, out_channels=out_channels, kernel_size=kernel_size, padding=0) for kernel_size in kernel_sizes)
        self.linear = torch.nn.Linear(in_features=out_channels * len(kernel_sizes), out_features=output_size)
        self.activation = get_activation_function(activation)
        self.residual = Residual(normalize=normalize) if residual else None

    def forward(self, embeddings: 'torch.FloatTensor', mask: 'torch.BoolTensor') ->torch.FloatTensor:
        if 0 in embeddings.shape:
            return embeddings.view((*embeddings.shape[:-1], self.linear.out_features))
        max_k = max(conv.kernel_size[0] for conv in self.convolutions)
        left_pad = max_k // 2
        right_pad = (max_k - 1) // 2
        n_samples, n_words, dim = embeddings.shape
        n_words_with_pad = n_words + left_pad + right_pad
        padded_x = F.pad(embeddings, pad=(0, 0, max_k // 2, (max_k - 1) // 2))
        padded_mask = F.pad(mask, pad=(max_k // 2 + (max_k - 1) // 2, 0), value=True)
        flat_x = padded_x[padded_mask]
        flat_x = flat_x.permute(1, 0).unsqueeze(0)
        conv_results = []
        for conv_idx, conv in enumerate(self.convolutions):
            k = conv.kernel_size[0]
            conv_x = conv(flat_x)
            offset_left = left_pad - k // 2
            offset_right = conv_x.size(2) - (right_pad - (k - 1) // 2)
            conv_results.append(conv_x[0, :, offset_left:offset_right])
        flat_x = torch.cat(conv_results, dim=0)
        flat_x = flat_x.transpose(1, 0)
        flat_x = torch.relu(flat_x)
        flat_x = self.linear(flat_x)
        new_dim = flat_x.size(-1)
        x = torch.empty(n_samples * n_words_with_pad, new_dim, device=flat_x.device, dtype=flat_x.dtype)
        flat_mask = padded_mask.clone()
        flat_mask[-1, padded_mask[-1].sum() - right_pad:] = False
        flat_mask[0, :left_pad] = False
        flat_mask = flat_mask.view(-1)
        x[flat_mask] = flat_x
        x = x.view(n_samples, n_words_with_pad, new_dim)
        x = x[:, left_pad:-right_pad]
        if self.residual is not None:
            x = self.residual(embeddings, x)
        return x.masked_fill_((~mask).unsqueeze(-1), 0)


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(10, 1)
        self.fc2 = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Metric,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Residual,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

