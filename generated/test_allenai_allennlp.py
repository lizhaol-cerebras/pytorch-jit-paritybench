
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


import warnings


import logging


from typing import Union


from typing import Dict


from typing import List


from typing import Tuple


from typing import NamedTuple


from typing import cast


import torch


from typing import Any


from typing import Optional


import torch.distributed as dist


import torch.multiprocessing as mp


import re


from torch import cuda


from abc import ABC


from collections import defaultdict


from typing import Callable


from typing import Set


from typing import Iterator


from typing import Iterable


from typing import MutableMapping


import time


import numpy as np


from torch import Tensor


from torch.testing import assert_allclose


import copy


import random


import numpy


from numpy.testing import assert_allclose


from itertools import islice


from itertools import zip_longest


from typing import Generator


from typing import TypeVar


from typing import Sequence


from torch import nn as nn


from abc import abstractmethod


from copy import deepcopy


import torch.nn as nn


from collections import Counter


from collections import deque


from queue import Full


import itertools


import math


from typing import Generic


import torch.nn.functional


import torchvision


from torch import FloatTensor


from torch import IntTensor


from typing import Mapping


import sklearn


import scipy


from scipy.stats import wasserstein_distance


from torch.distributions.categorical import Categorical


from torch.distributions.kl import kl_divergence


from torch import autograd


import torch.autograd as autograd


from torch.nn import Module


from typing import Type


import inspect


from torch.nn.modules.linear import Linear


import torch.nn.functional as F


from torch.nn.parameter import Parameter


from torch.nn import Parameter


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.modules import Dropout


from torch.nn import ParameterList


from torch import nn


import torch.nn


from torch.nn import Conv1d


from torch.nn import Linear


from typing import BinaryIO


from torch.nn.functional import embedding


from typing import TYPE_CHECKING


from torch.nn import CrossEntropyLoss


from collections import OrderedDict


import torchvision.ops.boxes as box_ops


from inspect import signature


import functools


from torch.utils.checkpoint import CheckpointFunction


import torch.nn.init


from typing import OrderedDict


from torch.cuda import amp


from torch.nn.utils import clip_grad_norm_


from itertools import chain


from torch.utils.hooks import RemovableHandle


from torch import backends


from typing import Deque


from torch.cuda.amp.grad_scaler import OptState


from sklearn import metrics


import scipy.stats as stats


import torch.optim.lr_scheduler


from torch import allclose


from numpy.testing import assert_almost_equal


from torch.autograd import Variable


from torch.nn.modules.rnn import LSTM


from torch.nn import LSTM


from torch.nn import RNN


from torch.nn import GRU


from torch.nn import Embedding


from numpy.testing import assert_array_almost_equal


from sklearn.metrics import precision_recall_fscore_support


from math import isclose


class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __reduce__(self) ->Union[str, Tuple[Any, ...]]:
        return type(self), (self.message,)

    def __init__(self, message: 'str'):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message


DataArray = TypeVar('DataArray', torch.Tensor, Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]])


class Field(Generic[DataArray]):
    """
    A `Field` is some piece of a data instance that ends up as an tensor in a model (either as an
    input or an output).  Data instances are just collections of fields.

    Fields go through up to two steps of processing: (1) tokenized fields are converted into token
    ids, (2) fields containing token ids (or any other numeric data) are padded (if necessary) and
    converted into tensors.  The `Field` API has methods around both of these steps, though they
    may not be needed for some concrete `Field` classes - if your field doesn't have any strings
    that need indexing, you don't need to implement `count_vocab_items` or `index`.  These
    methods `pass` by default.

    Once a vocabulary is computed and all fields are indexed, we will determine padding lengths,
    then intelligently batch together instances and pad them into actual tensors.
    """
    __slots__ = []

    def count_vocab_items(self, counter: 'Dict[str, Dict[str, int]]'):
        """
        If there are strings in this field that need to be converted into integers through a
        :class:`Vocabulary`, here is where we count them, to determine which tokens are in or out
        of the vocabulary.

        If your `Field` does not have any strings that need to be converted into indices, you do
        not need to implement this method.

        A note on this `counter`: because `Fields` can represent conceptually different things,
        we separate the vocabulary items by `namespaces`.  This way, we can use a single shared
        mechanism to handle all mappings from strings to integers in all fields, while keeping
        words in a `TextField` from sharing the same ids with labels in a `LabelField` (e.g.,
        "entailment" or "contradiction" are labels in an entailment task)

        Additionally, a single `Field` might want to use multiple namespaces - `TextFields` can
        be represented as a combination of word ids and character ids, and you don't want words and
        characters to share the same vocabulary - "a" as a word should get a different id from "a"
        as a character, and the vocabulary sizes of words and characters are very different.

        Because of this, the first key in the `counter` object is a `namespace`, like "tokens",
        "token_characters", "tags", or "labels", and the second key is the actual vocabulary item.
        """
        pass

    def human_readable_repr(self) ->Any:
        """
        This method should be implemented by subclasses to return a structured, yet human-readable
        representation of the field.

        !!! Note
            `human_readable_repr()` is not meant to be used as a method to serialize a `Field` since the return
            value does not necessarily contain all of the attributes of the `Field` instance. But the object
            returned should be JSON-serializable.
        """
        raise NotImplementedError

    def index(self, vocab: 'Vocabulary'):
        """
        Given a :class:`Vocabulary`, converts all strings in this field into (typically) integers.
        This `modifies` the `Field` object, it does not return anything.

        If your `Field` does not have any strings that need to be converted into indices, you do
        not need to implement this method.
        """
        pass

    def get_padding_lengths(self) ->Dict[str, int]:
        """
        If there are things in this field that need padding, note them here.  In order to pad a
        batch of instance, we get all of the lengths from the batch, take the max, and pad
        everything to that length (or use a pre-specified maximum length).  The return value is a
        dictionary mapping keys to lengths, like `{'num_tokens': 13}`.

        This is always called after :func:`index`.
        """
        raise NotImplementedError

    def as_tensor(self, padding_lengths: 'Dict[str, int]') ->DataArray:
        """
        Given a set of specified padding lengths, actually pad the data in this field and return a
        torch Tensor (or a more complex data structure) of the correct shape.  We also take a
        couple of parameters that are important when constructing torch Tensors.

        # Parameters

        padding_lengths : `Dict[str, int]`
            This dictionary will have the same keys that were produced in
            :func:`get_padding_lengths`.  The values specify the lengths to use when padding each
            relevant dimension, aggregated across all instances in a batch.
        """
        raise NotImplementedError

    def empty_field(self) ->'Field':
        """
        So that `ListField` can pad the number of fields in a list (e.g., the number of answer
        option `TextFields`), we need a representation of an empty field of each type.  This
        returns that.  This will only ever be called when we're to the point of calling
        :func:`as_tensor`, so you don't need to worry about `get_padding_lengths`,
        `count_vocab_items`, etc., being called on this empty field.

        We make this an instance method instead of a static method so that if there is any state
        in the Field, we can copy it over (e.g., the token indexers in `TextField`).
        """
        raise NotImplementedError

    def batch_tensors(self, tensor_list: 'List[DataArray]') ->DataArray:
        """
        Takes the output of `Field.as_tensor()` from a list of `Instances` and merges it into
        one batched tensor for this `Field`.  The default implementation here in the base class
        handles cases where `as_tensor` returns a single torch tensor per instance.  If your
        subclass returns something other than this, you need to override this method.

        This operation does not modify `self`, but in some cases we need the information
        contained in `self` in order to perform the batching, so this is an instance method, not
        a class method.
        """
        return torch.stack(tensor_list)

    def __eq__(self, other) ->bool:
        if isinstance(self, other.__class__):
            for class_ in self.__class__.mro():
                for attr in getattr(class_, '__slots__', []):
                    if getattr(self, attr) != getattr(other, attr):
                        return False
            if hasattr(self, '__dict__'):
                return self.__dict__ == other.__dict__
            return True
        return NotImplemented

    def __len__(self):
        raise NotImplementedError

    def duplicate(self):
        return deepcopy(self)


JsonDict = Dict[str, Any]


class Instance(Mapping[str, Field]):
    """
    An `Instance` is a collection of :class:`~allennlp.data.fields.field.Field` objects,
    specifying the inputs and outputs to
    some model.  We don't make a distinction between inputs and outputs here, though - all
    operations are done on all fields, and when we return arrays, we return them as dictionaries
    keyed by field name.  A model can then decide which fields it wants to use as inputs as which
    as outputs.

    The `Fields` in an `Instance` can start out either indexed or un-indexed.  During the data
    processing pipeline, all fields will be indexed, after which multiple instances can be combined
    into a `Batch` and then converted into padded arrays.

    # Parameters

    fields : `Dict[str, Field]`
        The `Field` objects that will be used to produce data arrays for this instance.
    """
    __slots__ = ['fields', 'indexed']

    def __init__(self, fields: 'MutableMapping[str, Field]') ->None:
        self.fields = fields
        self.indexed = False

    def __getitem__(self, key: 'str') ->Field:
        return self.fields[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) ->int:
        return len(self.fields)

    def add_field(self, field_name: 'str', field: 'Field', vocab: 'Vocabulary'=None) ->None:
        """
        Add the field to the existing fields mapping.
        If we have already indexed the Instance, then we also index `field`, so
        it is necessary to supply the vocab.
        """
        self.fields[field_name] = field
        if self.indexed and vocab is not None:
            field.index(vocab)

    def count_vocab_items(self, counter: 'Dict[str, Dict[str, int]]'):
        """
        Increments counts in the given `counter` for all of the vocabulary items in all of the
        `Fields` in this `Instance`.
        """
        for field in self.fields.values():
            field.count_vocab_items(counter)

    def index_fields(self, vocab: 'Vocabulary') ->None:
        """
        Indexes all fields in this `Instance` using the provided `Vocabulary`.
        This `mutates` the current object, it does not return a new `Instance`.
        A `DataLoader` will call this on each pass through a dataset; we use the `indexed`
        flag to make sure that indexing only happens once.

        This means that if for some reason you modify your vocabulary after you've
        indexed your instances, you might get unexpected behavior.
        """
        if not self.indexed:
            for field in self.fields.values():
                field.index(vocab)
            self.indexed = True

    def get_padding_lengths(self) ->Dict[str, Dict[str, int]]:
        """
        Returns a dictionary of padding lengths, keyed by field name.  Each `Field` returns a
        mapping from padding keys to actual lengths, and we just key that dictionary by field name.
        """
        lengths = {}
        for field_name, field in self.fields.items():
            lengths[field_name] = field.get_padding_lengths()
        return lengths

    def as_tensor_dict(self, padding_lengths: 'Dict[str, Dict[str, int]]'=None) ->Dict[str, DataArray]:
        """
        Pads each `Field` in this instance to the lengths given in `padding_lengths` (which is
        keyed by field name, then by padding key, the same as the return value in
        :func:`get_padding_lengths`), returning a list of torch tensors for each field.

        If `padding_lengths` is omitted, we will call `self.get_padding_lengths()` to get the
        sizes of the tensors to create.
        """
        padding_lengths = padding_lengths or self.get_padding_lengths()
        tensors = {}
        for field_name, field in self.fields.items():
            tensors[field_name] = field.as_tensor(padding_lengths[field_name])
        return tensors

    def __str__(self) ->str:
        base_string = 'Instance with fields:\n'
        return ' '.join([base_string] + [f'\t {name}: {field} \n' for name, field in self.fields.items()])

    def duplicate(self) ->'Instance':
        new = Instance({k: field.duplicate() for k, field in self.fields.items()})
        new.indexed = self.indexed
        return new

    def human_readable_dict(self) ->JsonDict:
        """
        This function help to output instances to json files or print for human readability.
        Use case includes example-based explanation, where it's better to have a output file or
        rather than printing or logging.
        """
        return {key: field.human_readable_repr() for key, field in self.fields.items()}


A = TypeVar('A')


def ensure_list(iterable: 'Iterable[A]') ->List[A]:
    """
    An Iterable may be a list or a generator.
    This ensures we get a list without making an unnecessary copy.
    """
    if isinstance(iterable, list):
        return iterable
    else:
        return list(iterable)


logger = logging.getLogger(__name__)


class Batch(Iterable):
    """
    A batch of Instances. In addition to containing the instances themselves,
    it contains helper functions for converting the data into tensors.

    A Batch just takes an iterable of instances in its constructor and hangs onto them
    in a list.
    """
    __slots__ = ['instances']

    def __init__(self, instances: 'Iterable[Instance]') ->None:
        super().__init__()
        self.instances = ensure_list(instances)
        self._check_types()

    def _check_types(self) ->None:
        """
        Check that all the instances have the same types.
        """
        field_name_to_type_counters: 'Dict[str, Counter]' = defaultdict(lambda : Counter())
        field_counts: 'Counter' = Counter()
        for instance in self.instances:
            for field_name, value in instance.fields.items():
                field_name_to_type_counters[field_name][value.__class__.__name__] += 1
                field_counts[field_name] += 1
        for field_name, type_counters in field_name_to_type_counters.items():
            if len(type_counters) > 1:
                raise ConfigurationError(f"You cannot construct a Batch with non-homogeneous Instances. Field '{field_name}' has {len(type_counters)} different types: {', '.join(type_counters.keys())}")
            if field_counts[field_name] != len(self.instances):
                raise ConfigurationError(f"You cannot construct a Batch with non-homogeneous Instances. Field '{field_name}' present in some Instances but not others.")

    def get_padding_lengths(self) ->Dict[str, Dict[str, int]]:
        """
        Gets the maximum padding lengths from all `Instances` in this batch.  Each `Instance`
        has multiple `Fields`, and each `Field` could have multiple things that need padding.
        We look at all fields in all instances, and find the max values for each (field_name,
        padding_key) pair, returning them in a dictionary.

        This can then be used to convert this batch into arrays of consistent length, or to set
        model parameters, etc.
        """
        padding_lengths: 'Dict[str, Dict[str, int]]' = defaultdict(dict)
        all_instance_lengths: 'List[Dict[str, Dict[str, int]]]' = [instance.get_padding_lengths() for instance in self.instances]
        all_field_lengths: 'Dict[str, List[Dict[str, int]]]' = defaultdict(list)
        for instance_lengths in all_instance_lengths:
            for field_name, instance_field_lengths in instance_lengths.items():
                all_field_lengths[field_name].append(instance_field_lengths)
        for field_name, field_lengths in all_field_lengths.items():
            for padding_key in field_lengths[0].keys():
                max_value = max(x.get(padding_key, 0) for x in field_lengths)
                padding_lengths[field_name][padding_key] = max_value
        return {**padding_lengths}

    def as_tensor_dict(self, padding_lengths: 'Dict[str, Dict[str, int]]'=None, verbose: 'bool'=False) ->Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        This method converts this `Batch` into a set of pytorch Tensors that can be passed
        through a model.  In order for the tensors to be valid tensors, all `Instances` in this
        batch need to be padded to the same lengths wherever padding is necessary, so we do that
        first, then we combine all of the tensors for each field in each instance into a set of
        batched tensors for each field.

        # Parameters

        padding_lengths : `Dict[str, Dict[str, int]]`
            If a key is present in this dictionary with a non-`None` value, we will pad to that
            length instead of the length calculated from the data.  This lets you, e.g., set a
            maximum value for sentence length if you want to throw out long sequences.

            Entries in this dictionary are keyed first by field name (e.g., "question"), then by
            padding key (e.g., "num_tokens").

        verbose : `bool`, optional (default=`False`)
            Should we output logging information when we're doing this padding?  If the batch is
            large, this is nice to have, because padding a large batch could take a long time.
            But if you're doing this inside of a data generator, having all of this output per
            batch is a bit obnoxious (and really slow).

        # Returns

        tensors : `Dict[str, DataArray]`
            A dictionary of tensors, keyed by field name, suitable for passing as input to a model.
            This is a `batch` of instances, so, e.g., if the instances have a "question" field and
            an "answer" field, the "question" fields for all of the instances will be grouped
            together into a single tensor, and the "answer" fields for all instances will be
            similarly grouped in a parallel set of tensors, for batched computation. Additionally,
            for complex `Fields`, the value of the dictionary key is not necessarily a single
            tensor.  For example, with the `TextField`, the output is a dictionary mapping
            `TokenIndexer` keys to tensors. The number of elements in this sub-dictionary
            therefore corresponds to the number of `TokenIndexers` used to index the
            `TextField`.  Each `Field` class is responsible for batching its own output.
        """
        padding_lengths = padding_lengths or defaultdict(dict)
        if verbose:
            logger.info(f'Padding batch of size {len(self.instances)} to lengths {padding_lengths}')
            logger.info('Getting max lengths from instances')
        instance_padding_lengths = self.get_padding_lengths()
        if verbose:
            logger.info(f'Instance max lengths: {instance_padding_lengths}')
        lengths_to_use: 'Dict[str, Dict[str, int]]' = defaultdict(dict)
        for field_name, instance_field_lengths in instance_padding_lengths.items():
            for padding_key in instance_field_lengths.keys():
                if padding_key in padding_lengths[field_name]:
                    lengths_to_use[field_name][padding_key] = padding_lengths[field_name][padding_key]
                else:
                    lengths_to_use[field_name][padding_key] = instance_field_lengths[padding_key]
        field_tensors: 'Dict[str, list]' = defaultdict(list)
        if verbose:
            logger.info(f'Now actually padding instances to length: {lengths_to_use}')
        for instance in self.instances:
            for field, tensors in instance.as_tensor_dict(lengths_to_use).items():
                field_tensors[field].append(tensors)
        field_classes = self.instances[0].fields
        return {field_name: field_classes[field_name].batch_tensors(field_tensor_list) for field_name, field_tensor_list in field_tensors.items()}

    def __iter__(self) ->Iterator[Instance]:
        return iter(self.instances)

    def index_instances(self, vocab: 'Vocabulary') ->None:
        for instance in self.instances:
            instance.index_fields(vocab)

    def print_statistics(self) ->None:
        sequence_field_lengths: 'Dict[str, List]' = defaultdict(list)
        for instance in self.instances:
            if not instance.indexed:
                raise ConfigurationError('Instances must be indexed with vocabulary before asking to print dataset statistics.')
            for field, field_padding_lengths in instance.get_padding_lengths().items():
                for key, value in field_padding_lengths.items():
                    sequence_field_lengths[f'{field}.{key}'].append(value)
        None
        for name, lengths in sequence_field_lengths.items():
            None
            None
        None
        for i in numpy.random.randint(len(self.instances), size=10):
            None
            None

    def __len__(self):
        return len(self.instances)


def _is_encodable(value: 'str') ->bool:
    """
    We need to filter out environment variables that can't
    be unicode-encoded to avoid a "surrogates not allowed"
    error in jsonnet.
    """
    return value == '' or value.encode('utf-8', 'ignore') != b''


def _environment_variables() ->Dict[str, str]:
    """
    Wraps `os.environ` to filter out non-encodable values.
    """
    return {key: value for key, value in os.environ.items() if _is_encodable(value)}


def _is_dict_free(obj: 'Any') ->bool:
    """
    Returns False if obj is a dict, or if it's a list with an element that _has_dict.
    """
    if isinstance(obj, dict):
        return False
    elif isinstance(obj, list):
        return all(_is_dict_free(item) for item in obj)
    else:
        return True


def _replace_none(params: 'Any') ->Any:
    if params == 'None':
        return None
    elif isinstance(params, dict):
        for key, value in params.items():
            params[key] = _replace_none(value)
        return params
    elif isinstance(params, list):
        return [_replace_none(value) for value in params]
    return params


class _CacheEntry(NamedTuple):
    regular_files: 'List[_Meta]'
    extraction_dirs: 'List[_Meta]'


def _find_entries(patterns: 'List[str]'=None, cache_dir: 'Union[str, Path]'=None) ->Tuple[int, Dict[str, _CacheEntry]]:
    """
    Find all cache entries, filtering ones that don't match any of the glob patterns given.

    Returns the total size of the matching entries and mapping or resource name to meta data.

    The values in the returned mapping are tuples because we seperate meta entries that
    correspond to extraction directories vs regular cache entries.
    """
    cache_dir = os.path.expanduser(cache_dir or CACHE_DIRECTORY)
    total_size: 'int' = 0
    cache_entries: 'Dict[str, _CacheEntry]' = defaultdict(lambda : _CacheEntry([], []))
    for meta_path in glob.glob(str(cache_dir) + '/*.json'):
        meta = _Meta.from_path(meta_path)
        if patterns and not any(fnmatch(meta.resource, p) for p in patterns):
            continue
        if meta.extraction_dir:
            cache_entries[meta.resource].extraction_dirs.append(meta)
        else:
            cache_entries[meta.resource].regular_files.append(meta)
        total_size += meta.size
    for entry in cache_entries.values():
        entry.regular_files.sort(key=lambda meta: meta.creation_time, reverse=True)
        entry.extraction_dirs.sort(key=lambda meta: meta.creation_time, reverse=True)
    return total_size, cache_entries


def inspect_cache(patterns: 'List[str]'=None, cache_dir: 'Union[str, Path]'=None):
    """
    Print out useful information about the cache directory.
    """
    cache_dir = os.path.expanduser(cache_dir or CACHE_DIRECTORY)
    total_size, cache_entries = _find_entries(patterns=patterns, cache_dir=cache_dir)
    if patterns:
        None
    else:
        None
    for resource, entry in sorted(cache_entries.items(), key=lambda x: max(0 if not x[1][0] else x[1][0][0].creation_time, 0 if not x[1][1] else x[1][1][0].creation_time), reverse=True):
        None
        if entry.regular_files:
            td = timedelta(seconds=time.time() - entry.regular_files[0].creation_time)
            n_versions = len(entry.regular_files)
            size = entry.regular_files[0].size
            None
        if entry.extraction_dirs:
            td = timedelta(seconds=time.time() - entry.extraction_dirs[0].creation_time)
            n_versions = len(entry.extraction_dirs)
            size = entry.extraction_dirs[0].size
            None
    None


def remove_cache_entries(patterns: 'List[str]', cache_dir: 'Union[str, Path]'=None) ->int:
    """
    Remove cache entries matching the given patterns.

    Returns the total reclaimed space in bytes.
    """
    total_size, cache_entries = _find_entries(patterns=patterns, cache_dir=cache_dir)
    for resource, entry in cache_entries.items():
        for meta in entry.regular_files:
            logger.info('Removing cached version of %s at %s', resource, meta.cached_path)
            os.remove(meta.cached_path)
            if os.path.exists(meta.cached_path + '.lock'):
                os.remove(meta.cached_path + '.lock')
            os.remove(meta.cached_path + '.json')
        for meta in entry.extraction_dirs:
            logger.info('Removing extracted version of %s at %s', resource, meta.cached_path)
            shutil.rmtree(meta.cached_path)
            if os.path.exists(meta.cached_path + '.lock'):
                os.remove(meta.cached_path + '.lock')
            os.remove(meta.cached_path + '.json')
    return total_size


def _cached_path(args: 'argparse.Namespace'):
    logger.info('Cache directory: %s', args.cache_dir)
    if args.inspect:
        if args.extract_archive or args.force_extract or args.remove:
            raise RuntimeError('cached-path cannot accept --extract-archive, --force-extract, or --remove options when --inspect flag is used.')
        inspect_cache(patterns=args.resources, cache_dir=args.cache_dir)
    elif args.remove:
        if args.extract_archive or args.force_extract or args.inspect:
            raise RuntimeError('cached-path cannot accept --extract-archive, --force-extract, or --inspect options when --remove flag is used.')
        if not args.resources:
            raise RuntimeError("Missing positional argument(s) 'resources'. 'resources' is required when using the --remove option. If you really want to remove everything, pass '*' for 'resources'.")
        reclaimed_space = remove_cache_entries(args.resources, cache_dir=args.cache_dir)
        None
    else:
        for resource in args.resources:
            None


def cached_path(url_or_filename: 'Union[str, PathLike]', cache_dir: 'Union[str, Path]'=None, extract_archive: 'bool'=False, force_extract: 'bool'=False) ->str:
    """
    Given something that might be a URL or local path, determine which.
    If it's a remote resource, download the file and cache it, and
    then return the path to the cached file. If it's already a local path,
    make sure the file exists and return the path.

    For URLs, "http://", "https://", "s3://", "gs://", and "hf://" are all supported.
    The latter corresponds to the HuggingFace Hub.

    For example, to download the PyTorch weights for the model `epwalsh/bert-xsmall-dummy`
    on HuggingFace, you could do:

    ```python
    cached_path("hf://epwalsh/bert-xsmall-dummy/pytorch_model.bin")
    ```

    For paths or URLs that point to a tarfile or zipfile, you can also add a path
    to a specific file to the `url_or_filename` preceeded by a "!", and the archive will
    be automatically extracted (provided you set `extract_archive` to `True`),
    returning the local path to the specific file. For example:

    ```python
    cached_path("model.tar.gz!weights.th", extract_archive=True)
    ```

    # Parameters

    url_or_filename : `Union[str, Path]`
        A URL or path to parse and possibly download.

    cache_dir : `Union[str, Path]`, optional (default = `None`)
        The directory to cache downloads.

    extract_archive : `bool`, optional (default = `False`)
        If `True`, then zip or tar.gz archives will be automatically extracted.
        In which case the directory is returned.

    force_extract : `bool`, optional (default = `False`)
        If `True` and the file is an archive file, it will be extracted regardless
        of whether or not the extracted directory already exists.

        !!! Warning
            Use this flag with caution! This can lead to race conditions if used
            from multiple processes on the same file.
    """
    return str(_cached_path.cached_path(url_or_filename, cache_dir=cache_dir or CACHE_DIRECTORY, extract_archive=extract_archive, force_extract=force_extract))


def infer_and_cast(value: 'Any'):
    """
    In some cases we'll be feeding params dicts to functions we don't own;
    for example, PyTorch optimizers. In that case we can't use `pop_int`
    or similar to force casts (which means you can't specify `int` parameters
    using environment variables). This function takes something that looks JSON-like
    and recursively casts things that look like (bool, int, float) to (bool, int, float).
    """
    if isinstance(value, (int, float, bool)):
        return value
    elif isinstance(value, list):
        return [infer_and_cast(item) for item in value]
    elif isinstance(value, dict):
        return {key: infer_and_cast(item) for key, item in value.items()}
    elif isinstance(value, str):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            try:
                return int(value)
            except ValueError:
                pass
            try:
                return float(value)
            except ValueError:
                return value
    else:
        raise ValueError(f'cannot infer type of {value}')


T = TypeVar('T')


def with_overrides(original: 'T', overrides_dict: 'Dict[str, Any]', prefix: 'str'='') ->T:
    merged: 'T'
    keys: 'Union[Iterable[str], Iterable[int]]'
    if isinstance(original, list):
        merged = [None] * len(original)
        keys = range(len(original))
    elif isinstance(original, dict):
        merged = {}
        keys = chain(original.keys(), (k for k in overrides_dict if '.' not in k and k not in original))
    elif prefix:
        raise ValueError(f"overrides for '{prefix[:-1]}.*' expected list or dict in original, found {type(original)} instead")
    else:
        raise ValueError(f'expected list or dict, found {type(original)} instead')
    used_override_keys: 'Set[str]' = set()
    for key in keys:
        if str(key) in overrides_dict:
            merged[key] = copy.deepcopy(overrides_dict[str(key)])
            used_override_keys.add(str(key))
        else:
            overrides_subdict = {}
            for o_key in overrides_dict:
                if o_key.startswith(f'{key}.'):
                    overrides_subdict[o_key[len(f'{key}.'):]] = overrides_dict[o_key]
                    used_override_keys.add(o_key)
            if overrides_subdict:
                merged[key] = with_overrides(original[key], overrides_subdict, prefix=prefix + f'{key}.')
            else:
                merged[key] = copy.deepcopy(original[key])
    unused_override_keys = [(prefix + key) for key in set(overrides_dict.keys()) - used_override_keys]
    if unused_override_keys:
        raise ValueError(f'overrides dict contains unused keys: {unused_override_keys}')
    return merged


class Params(MutableMapping):
    """
    Represents a parameter dictionary with a history, and contains other functionality around
    parameter passing and validation for AllenNLP.

    There are currently two benefits of a `Params` object over a plain dictionary for parameter
    passing:

    1. We handle a few kinds of parameter validation, including making sure that parameters
       representing discrete choices actually have acceptable values, and making sure no extra
       parameters are passed.
    2. We log all parameter reads, including default values.  This gives a more complete
       specification of the actual parameters used than is given in a JSON file, because
       those may not specify what default values were used, whereas this will log them.

    !!! Consumption
        The convention for using a `Params` object in AllenNLP is that you will consume the parameters
        as you read them, so that there are none left when you've read everything you expect.  This
        lets us easily validate that you didn't pass in any `extra` parameters, just by making sure
        that the parameter dictionary is empty.  You should do this when you're done handling
        parameters, by calling `Params.assert_empty`.
    """
    DEFAULT = object()

    def __init__(self, params: 'Dict[str, Any]', history: 'str'='') ->None:
        self.params = _replace_none(params)
        self.history = history

    def pop(self, key: 'str', default: 'Any'=DEFAULT, keep_as_dict: 'bool'=False) ->Any:
        """
        Performs the functionality associated with dict.pop(key), along with checking for
        returned dictionaries, replacing them with Param objects with an updated history
        (unless keep_as_dict is True, in which case we leave them as dictionaries).

        If `key` is not present in the dictionary, and no default was specified, we raise a
        `ConfigurationError`, instead of the typical `KeyError`.
        """
        if default is self.DEFAULT:
            try:
                value = self.params.pop(key)
            except KeyError:
                msg = f'key "{key}" is required'
                if self.history:
                    msg += f' at location "{self.history}"'
                raise ConfigurationError(msg)
        else:
            value = self.params.pop(key, default)
        if keep_as_dict or _is_dict_free(value):
            logger.info(f'{self.history}{key} = {value}')
            return value
        else:
            return self._check_is_dict(key, value)

    def pop_int(self, key: 'str', default: 'Any'=DEFAULT) ->Optional[int]:
        """
        Performs a pop and coerces to an int.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        else:
            return int(value)

    def pop_float(self, key: 'str', default: 'Any'=DEFAULT) ->Optional[float]:
        """
        Performs a pop and coerces to a float.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        else:
            return float(value)

    def pop_bool(self, key: 'str', default: 'Any'=DEFAULT) ->Optional[bool]:
        """
        Performs a pop and coerces to a bool.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        elif isinstance(value, bool):
            return value
        elif value == 'true':
            return True
        elif value == 'false':
            return False
        else:
            raise ValueError('Cannot convert variable to bool: ' + value)

    def get(self, key: 'str', default: 'Any'=DEFAULT):
        """
        Performs the functionality associated with dict.get(key) but also checks for returned
        dicts and returns a Params object in their place with an updated history.
        """
        default = None if default is self.DEFAULT else default
        value = self.params.get(key, default)
        return self._check_is_dict(key, value)

    def pop_choice(self, key: 'str', choices: 'List[Any]', default_to_first_choice: 'bool'=False, allow_class_names: 'bool'=True) ->Any:
        """
        Gets the value of `key` in the `params` dictionary, ensuring that the value is one of
        the given choices. Note that this `pops` the key from params, modifying the dictionary,
        consistent with how parameters are processed in this codebase.

        # Parameters

        key: `str`

            Key to get the value from in the param dictionary

        choices: `List[Any]`

            A list of valid options for values corresponding to `key`.  For example, if you're
            specifying the type of encoder to use for some part of your model, the choices might be
            the list of encoder classes we know about and can instantiate.  If the value we find in
            the param dictionary is not in `choices`, we raise a `ConfigurationError`, because
            the user specified an invalid value in their parameter file.

        default_to_first_choice: `bool`, optional (default = `False`)

            If this is `True`, we allow the `key` to not be present in the parameter
            dictionary.  If the key is not present, we will use the return as the value the first
            choice in the `choices` list.  If this is `False`, we raise a
            `ConfigurationError`, because specifying the `key` is required (e.g., you `have` to
            specify your model class when running an experiment, but you can feel free to use
            default settings for encoders if you want).

        allow_class_names: `bool`, optional (default = `True`)

            If this is `True`, then we allow unknown choices that look like fully-qualified class names.
            This is to allow e.g. specifying a model type as my_library.my_model.MyModel
            and importing it on the fly. Our check for "looks like" is extremely lenient
            and consists of checking that the value contains a '.'.
        """
        default = choices[0] if default_to_first_choice else self.DEFAULT
        value = self.pop(key, default)
        ok_because_class_name = allow_class_names and '.' in value
        if value not in choices and not ok_because_class_name:
            key_str = self.history + key
            message = f'{value} not in acceptable choices for {key_str}: {choices}. You should either use the --include-package flag to make sure the correct module is loaded, or use a fully qualified class name in your config file like {{"model": "my_module.models.MyModel"}} to have it imported automatically.'
            raise ConfigurationError(message)
        return value

    def as_dict(self, quiet: 'bool'=False, infer_type_and_cast: 'bool'=False):
        """
        Sometimes we need to just represent the parameters as a dict, for instance when we pass
        them to PyTorch code.

        # Parameters

        quiet: `bool`, optional (default = `False`)

            Whether to log the parameters before returning them as a dict.

        infer_type_and_cast: `bool`, optional (default = `False`)

            If True, we infer types and cast (e.g. things that look like floats to floats).
        """
        if infer_type_and_cast:
            params_as_dict = infer_and_cast(self.params)
        else:
            params_as_dict = self.params
        if quiet:
            return params_as_dict

        def log_recursively(parameters, history):
            for key, value in parameters.items():
                if isinstance(value, dict):
                    new_local_history = history + key + '.'
                    log_recursively(value, new_local_history)
                else:
                    logger.info(f'{history}{key} = {value}')
        log_recursively(self.params, self.history)
        return params_as_dict

    def as_flat_dict(self) ->Dict[str, Any]:
        """
        Returns the parameters of a flat dictionary from keys to values.
        Nested structure is collapsed with periods.
        """
        flat_params = {}

        def recurse(parameters, path):
            for key, value in parameters.items():
                newpath = path + [key]
                if isinstance(value, dict):
                    recurse(value, newpath)
                else:
                    flat_params['.'.join(newpath)] = value
        recurse(self.params, [])
        return flat_params

    def duplicate(self) ->'Params':
        """
        Uses `copy.deepcopy()` to create a duplicate (but fully distinct)
        copy of these Params.
        """
        return copy.deepcopy(self)

    def assert_empty(self, class_name: 'str'):
        """
        Raises a `ConfigurationError` if `self.params` is not empty.  We take `class_name` as
        an argument so that the error message gives some idea of where an error happened, if there
        was one.  `class_name` should be the name of the `calling` class, the one that got extra
        parameters (if there are any).
        """
        if self.params:
            raise ConfigurationError('Extra parameters passed to {}: {}'.format(class_name, self.params))

    def __getitem__(self, key):
        if key in self.params:
            return self._check_is_dict(key, self.params[key])
        else:
            raise KeyError(str(key))

    def __setitem__(self, key, value):
        self.params[key] = value

    def __delitem__(self, key):
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def _check_is_dict(self, new_history, value):
        if isinstance(value, dict):
            new_history = self.history + new_history + '.'
            return Params(value, history=new_history)
        if isinstance(value, list):
            value = [self._check_is_dict(f'{new_history}.{i}', v) for i, v in enumerate(value)]
        return value

    @classmethod
    def from_file(cls, params_file: 'Union[str, PathLike]', params_overrides: 'Union[str, Dict[str, Any]]'='', ext_vars: 'dict'=None) ->'Params':
        """
        Load a `Params` object from a configuration file.

        # Parameters

        params_file: `str`

            The path to the configuration file to load.

        params_overrides: `Union[str, Dict[str, Any]]`, optional (default = `""`)

            A dict of overrides that can be applied to final object.
            e.g. `{"model.embedding_dim": 10}` will change the value of "embedding_dim"
            within the "model" object of the config to 10. If you wanted to override the entire
            "model" object of the config, you could do `{"model": {"type": "other_type", ...}}`.

        ext_vars: `dict`, optional

            Our config files are Jsonnet, which allows specifying external variables
            for later substitution. Typically we substitute these using environment
            variables; however, you can also specify them here, in which case they
            take priority over environment variables.
            e.g. {"HOME_DIR": "/Users/allennlp/home"}
        """
        if ext_vars is None:
            ext_vars = {}
        params_file = cached_path(params_file)
        ext_vars = {**_environment_variables(), **ext_vars}
        file_dict = json.loads(evaluate_file(params_file, ext_vars=ext_vars))
        if isinstance(params_overrides, dict):
            params_overrides = json.dumps(params_overrides)
        overrides_dict = parse_overrides(params_overrides, ext_vars=ext_vars)
        if overrides_dict:
            param_dict = with_overrides(file_dict, overrides_dict)
        else:
            param_dict = file_dict
        return cls(param_dict)

    def to_file(self, params_file: 'str', preference_orders: 'List[List[str]]'=None) ->None:
        with open(params_file, 'w') as handle:
            json.dump(self.as_ordered_dict(preference_orders), handle, indent=4)

    def as_ordered_dict(self, preference_orders: 'List[List[str]]'=None) ->OrderedDict:
        """
        Returns Ordered Dict of Params from list of partial order preferences.

        # Parameters

        preference_orders: `List[List[str]]`, optional

            `preference_orders` is list of partial preference orders. ["A", "B", "C"] means
            "A" > "B" > "C". For multiple preference_orders first will be considered first.
            Keys not found, will have last but alphabetical preference. Default Preferences:
            `[["dataset_reader", "iterator", "model", "train_data_path", "validation_data_path",
            "test_data_path", "trainer", "vocabulary"], ["type"]]`
        """
        params_dict = self.as_dict(quiet=True)
        if not preference_orders:
            preference_orders = []
            preference_orders.append(['dataset_reader', 'iterator', 'model', 'train_data_path', 'validation_data_path', 'test_data_path', 'trainer', 'vocabulary'])
            preference_orders.append(['type'])

        def order_func(key):
            order_tuple = [(order.index(key) if key in order else len(order)) for order in preference_orders]
            return order_tuple + [key]

        def order_dict(dictionary, order_func):
            result = OrderedDict()
            for key, val in sorted(dictionary.items(), key=lambda item: order_func(item[0])):
                result[key] = order_dict(val, order_func) if isinstance(val, dict) else val
            return result
        return order_dict(params_dict, order_func)

    def get_hash(self) ->str:
        """
        Returns a hash code representing the current state of this `Params` object.  We don't
        want to implement `__hash__` because that has deeper python implications (and this is a
        mutable object), but this will give you a representation of the current state.
        We use `zlib.adler32` instead of Python's builtin `hash` because the random seed for the
        latter is reset on each new program invocation, as discussed here:
        https://stackoverflow.com/questions/27954892/deterministic-hashing-in-python-3.
        """
        dumped = json.dumps(self.params, sort_keys=True)
        hashed = zlib.adler32(dumped.encode())
        return str(hashed)

    def __str__(self) ->str:
        return f'{self.history}Params({self.params})'

