
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


import numpy as np


from scipy import signal


import random


import warnings


import torch.nn.functional as F


import math


from functools import partial


from collections import namedtuple


from enum import Enum


from scipy.signal import fftconvolve


from torch.nn import Conv1d


import logging


from torch.nn.parallel import DistributedDataParallel


from collections import defaultdict


from typing import Dict


import time


import pandas as pd


from scipy.io import wavfile


import torchvision


from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt


from torch.nn import functional as F


from torch.utils.data import IterableDataset


import re


import string


from typing import List


import copy


from enum import auto


from torch.utils.data import DataLoader


import itertools


from types import SimpleNamespace


from itertools import chain


import itertools as it


import functools


import torch.nn as nn


import numpy


from scipy.signal import resample_poly


from typing import Optional


from typing import Union


import inspect


from torch.nn import DataParallel as DP


from torch.nn import SyncBatchNorm


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data import DistributedSampler


import collections


from torch.utils.data._utils.collate import default_convert


from torch.utils.data._utils.pin_memory import pin_memory as recursive_pin_memory


from torch.utils.data.dataloader import _BaseDataLoaderIter


from types import MethodType


from torch.utils.data import Dataset


from collections import Counter


from scipy.stats import lognorm


from torch.utils.data import RandomSampler


from torch.utils.data import Sampler


from torch.utils.data import WeightedRandomSampler


from itertools import groupby


from typing import Any


from typing import Tuple


from functools import cached_property


from torch.distributions import Categorical


import abc


from collections import OrderedDict


from torch.nn import Dropout


from math import sqrt


from torch import nn


from torch.nn.modules.loss import _Loss


from torch.nn.utils import spectral_norm


from torch.autograd import Function


from functools import reduce


from math import floor


import torch.utils.data


from torch.nn import Parameter


from torch.nn import Module


from itertools import permutations


from torch.linalg import vector_norm


from scipy.stats import chi


from torch.autograd import Variable


from abc import abstractmethod


from scipy import linalg


from typing import Callable


from typing import Iterable


import collections.abc


from numbers import Number


from functools import wraps


import logging.config


from torch import profiler


from typing import Literal


import torch.multiprocessing as mp


from copy import deepcopy


import torch.nn


class BatchNorm1d(nn.Module):
    """Applies 1d batch normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    combine_batch_time : bool
        When true, it combines batch an time axis.
    skip_transpose : bool
        Whether to skip the transposition.


    Example
    -------
    >>> input = torch.randn(100, 10)
    >>> norm = BatchNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10])
    """

    def __init__(self, input_shape=None, input_size=None, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, combine_batch_time=False, skip_transpose=False):
        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose
        if input_size is None and skip_transpose:
            input_size = input_shape[1]
        elif input_size is None:
            input_size = input_shape[-1]
        self.norm = nn.BatchNorm1d(input_size, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, [channels])
            input to normalize. 2d or 3d tensors are expected in input
            4d tensors can be used when combine_dims=True.

        Returns
        -------
        x_n : torch.Tensor
            The normalized outputs.
        """
        shape_or = x.shape
        if self.combine_batch_time:
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[3], shape_or[2])
        elif not self.skip_transpose:
            x = x.transpose(-1, 1)
        x_n = self.norm(x)
        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)
        return x_n


class Linear(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape : tuple
        It is the shape of the input tensor.
    input_size : int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    max_norm : float
        weight max-norm.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(self, n_neurons, input_shape=None, input_size=None, bias=True, max_norm=None, combine_dims=False):
        super().__init__()
        self.max_norm = max_norm
        self.combine_dims = combine_dims
        if input_shape is None and input_size is None:
            raise ValueError('Expected one of input_shape or input_size')
        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]
        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.

        Returns
        -------
        wx : torch.Tensor
            The linearly transformed outputs.
        """
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        if self.max_norm is not None:
            self.w.weight.data = torch.renorm(self.w.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        wx = self.w(x)
        return wx


class StatisticsPooling(nn.Module):
    """This class implements a statistic pooling layer.

    It returns the mean and/or std of input tensor.

    Arguments
    ---------
    return_mean : bool
         If True, the average pooling will be returned.
    return_std : bool
         If True, the standard deviation will be returned.

    Example
    -------
    >>> inp_tensor = torch.rand([5, 100, 50])
    >>> sp_layer = StatisticsPooling()
    >>> out_tensor = sp_layer(inp_tensor)
    >>> out_tensor.shape
    torch.Size([5, 1, 100])
    """

    def __init__(self, return_mean=True, return_std=True):
        super().__init__()
        self.eps = 1e-05
        self.return_mean = return_mean
        self.return_std = return_std
        if not (self.return_mean or self.return_std):
            raise ValueError('both of statistics are equal to False \nconsider enabling mean and/or std statistic pooling')

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch.
        lengths : torch.Tensor
            The lengths of the samples in the input.

        Returns
        -------
        pooled_stats : torch.Tensor
            The mean and std for the input.
        """
        if lengths is None:
            if self.return_mean:
                mean = x.mean(dim=1)
            if self.return_std:
                std = x.std(dim=1)
        else:
            mean = []
            std = []
            for snt_id in range(x.shape[0]):
                actual_size = int(torch.round(lengths[snt_id] * x.shape[1]))
                if self.return_mean:
                    mean.append(torch.mean(x[snt_id, 0:actual_size, ...], dim=0))
                if self.return_std:
                    std.append(torch.std(x[snt_id, 0:actual_size, ...], dim=0))
            if self.return_mean:
                mean = torch.stack(mean)
            if self.return_std:
                std = torch.stack(std)
        if self.return_mean:
            gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
            gnoise = gnoise
            mean += gnoise
        if self.return_std:
            std = std + self.eps
        if self.return_mean and self.return_std:
            pooled_stats = torch.cat((mean, std), dim=1)
            pooled_stats = pooled_stats.unsqueeze(1)
        elif self.return_mean:
            pooled_stats = mean.unsqueeze(1)
        elif self.return_std:
            pooled_stats = std.unsqueeze(1)
        return pooled_stats

    def _get_gauss_noise(self, shape_of_tensor, device='cpu'):
        """Returns a tensor of epsilon Gaussian noise.

        Arguments
        ---------
        shape_of_tensor : torch.Tensor
            It represents the size of tensor for generating Gaussian noise.
        device : str
            Device on which to perform computations.

        Returns
        -------
        gnoise : torch.Tensor
            The Gaussian noise.
        """
        gnoise = torch.randn(shape_of_tensor, device=device)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)
        return gnoise


class Xvector(torch.nn.Module):
    """This model extracts X-vectors for speaker recognition

    Arguments
    ---------
    device : str
        The device to place this model on (e.g. "cpu" or "cuda")
    activation : torch class
        A class for constructing the activation layers.
    tdnn_blocks : int
        Number of time-delay neural (TDNN) layers.
    tdnn_channels : list of ints
        Output channels for TDNN layer.
    tdnn_kernel_sizes : list of ints
        List of kernel sizes for each TDNN layer.
    tdnn_dilations : list of ints
        List of dilations for kernels in each TDNN layer.
    lin_neurons : int
        Number of neurons in linear layers.
    in_channels : int
        Number of channels expected in the input.

    Example
    -------
    >>> compute_xvect = Xvector()
    >>> input_feats = torch.rand([5, 10, 40])
    >>> outputs = compute_xvect(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 512])
    """

    def __init__(self, device='cpu', activation=torch.nn.LeakyReLU, tdnn_blocks=5, tdnn_channels=[512, 512, 512, 512, 1500], tdnn_kernel_sizes=[5, 3, 3, 1, 1], tdnn_dilations=[1, 2, 3, 1, 1], lin_neurons=512, in_channels=40):
        super().__init__()
        self.blocks = nn.ModuleList()
        for block_index in range(tdnn_blocks):
            out_channels = tdnn_channels[block_index]
            self.blocks.extend([Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=tdnn_kernel_sizes[block_index], dilation=tdnn_dilations[block_index]), activation(), BatchNorm1d(input_size=out_channels)])
            in_channels = tdnn_channels[block_index]
        self.blocks.append(StatisticsPooling())
        self.blocks.append(Linear(input_size=out_channels * 2, n_neurons=lin_neurons, bias=True, combine_dims=False))

    def forward(self, x, lens=None):
        """Returns the x-vectors.

        Arguments
        ---------
        x : torch.Tensor
            The input features for computation.
        lens : torch.Tensor
            The length of the corresponding inputs.

        Returns
        -------
        The computed x-vectors
        """
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lens)
            except TypeError:
                x = layer(x)
        return x


def batch_log_matvecmul(A, b):
    """For each 'matrix' and 'vector' pair in the batch, do matrix-vector
    multiplication in the log domain, i.e., logsumexp instead of add,
    add instead of multiply.

    Arguments
    ---------
    A : torch.Tensor (batch, dim1, dim2)
        Tensor
    b : torch.Tensor (batch, dim1)
        Tensor.

    Returns
    -------
    x : torch.Tensor (batch, dim1)

    Example
    -------
    >>> A = torch.tensor([[[   0., 0.],
    ...                    [ -1e5, 0.]]])
    >>> b = torch.tensor([[0., 0.,]])
    >>> x = batch_log_matvecmul(A, b)
    >>> x
    tensor([[0.6931, 0.0000]])
    >>>
    >>> # non-log domain equivalent without batching functionality
    >>> A_ = torch.tensor([[1., 1.],
    ...                    [0., 1.]])
    >>> b_ = torch.tensor([1., 1.,])
    >>> x_ = torch.matmul(A_, b_)
    >>> x_
    tensor([2., 1.])
    """
    b = b.unsqueeze(1)
    x = torch.logsumexp(A + b, dim=2)
    return x


def batch_log_maxvecmul(A, b):
    """Similar to batch_log_matvecmul, but takes a maximum instead of
    logsumexp. Returns both the max and the argmax.

    Arguments
    ---------
    A : torch.Tensor (batch, dim1, dim2)
        Tensor.
    b : torch.Tensor (batch, dim1)
        Tensor

    Returns
    -------
    x : torch.Tensor (batch, dim1)
        Tensor.
    argmax : torch.Tensor (batch, dim1)
        Tensor.

    Example
    -------
    >>> A = torch.tensor([[[   0., -1.],
    ...                    [ -1e5,  0.]]])
    >>> b = torch.tensor([[0., 0.,]])
    >>> x, argmax = batch_log_maxvecmul(A, b)
    >>> x
    tensor([[0., 0.]])
    >>> argmax
    tensor([[0, 1]])
    """
    b = b.unsqueeze(1)
    x, argmax = torch.max(A + b, dim=2)
    return x, argmax


def map_inds_to_intersect(lists1, lists2, ind2labs):
    """Converts 2 lists containing indices for phonemes from different
    phoneme sets to a single phoneme so that comparing the equality
    of the indices of the resulting lists will yield the correct
    accuracy.

    Arguments
    ---------
    lists1 : list of lists of ints
        Contains the indices of the first sequence of phonemes.
    lists2 : list of lists of ints
        Contains the indices of the second sequence of phonemes.
    ind2labs : tuple (dict, dict)
        Contains the original index-to-label dicts for the first and second
        sequence of phonemes.

    Returns
    -------
    lists1_new : list of lists of ints
        Contains the indices of the first sequence of phonemes, mapped
        to the new phoneme set.
    lists2_new : list of lists of ints
        Contains the indices of the second sequence of phonemes, mapped
        to the new phoneme set.

    Example
    -------
    >>> lists1 = [[0, 1]]
    >>> lists2 = [[0, 1]]
    >>> ind2lab1 = {
    ...        0: "a",
    ...        1: "b",
    ...        }
    >>> ind2lab2 = {
    ...        0: "a",
    ...        1: "c",
    ...        }
    >>> ind2labs = (ind2lab1, ind2lab2)
    >>> out1, out2 = map_inds_to_intersect(lists1, lists2, ind2labs)
    >>> out1
    [[0, 1]]
    >>> out2
    [[0, 2]]
    """
    ind2lab1, ind2lab2 = ind2labs
    set1, set2 = set(ind2lab1.values()), set(ind2lab2.values())
    intersect = set1.intersection(set2)
    set1_only = set1.difference(set2)
    set2_only = set2.difference(set1)
    new_lab2ind = {lab: i for i, lab in enumerate(intersect)}
    new_lab2ind.update({lab: (len(new_lab2ind) + i) for i, lab in enumerate(set1_only)})
    new_lab2ind.update({lab: (len(new_lab2ind) + i) for i, lab in enumerate(set2_only)})
    lists1_lab = [[ind2lab1[ind] for ind in utt] for utt in lists1]
    lists2_lab = [[ind2lab2[ind] for ind in utt] for utt in lists2]
    lists1_new = [[new_lab2ind[lab] for lab in utt] for utt in lists1_lab]
    lists2_new = [[new_lab2ind[lab] for lab in utt] for utt in lists2_lab]
    return lists1_new, lists2_new


def mark_as_loader(method):
    """Method decorator which marks given method as checkpoint loading hook.

    Arguments
    ---------
    method : callable
        Method of the class to decorate. Must be callable with
        signature (instance, path, end_of_epoch) using positional
        arguments. This is satisfied by for example:
        `def loader(self, path, end_of_epoch):`

    Returns
    -------
    The decorated method, registered as a checkpoint loader.

    Note
    ----
    This will not add the hook (not possible via a method decorator),
    you must also decorate the class with @register_checkpoint_hooks
    Only one method can be added as the hook.
    """
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path('testpath'), True)
    except TypeError:
        MSG = 'Checkpoint loader must have signature (self, path, end_of_epoch)'
        raise TypeError(MSG)
    method._speechbrain_loader = True
    return method


def mark_as_saver(method):
    """Method decorator which marks given method as the checkpoint saving hook.

    See register_checkpoint_hooks for example.

    Arguments
    ---------
    method : callable
        Method of the class to decorate. Must be callable with
        signature (instance, path) using positional arguments. This is
        satisfied by for example: def saver(self, path):

    Returns
    -------
    The decorated method, marked as a checkpoint saver.

    Note
    ----
    This will not add the hook (not possible via a method decorator),
    you must also decorate the class with @register_checkpoint_hooks
    Only one method can be added as the hook.
    """
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path('testpath'))
    except TypeError:
        MSG = 'Checkpoint saver must match signature (instance, path)'
        raise TypeError(MSG)
    method._speechbrain_saver = True
    return method


def undo_padding(batch, lengths):
    """Produces Python lists given a batch of sentences with
    their corresponding relative lengths.

    Arguments
    ---------
    batch : torch.Tensor
        Batch of sentences gathered in a batch.
    lengths : torch.Tensor
        Relative length of each sentence in the batch.

    Returns
    -------
    as_list : list
        A python list of the corresponding input tensor.

    Example
    -------
    >>> batch=torch.rand([4,100])
    >>> lengths=torch.tensor([0.5,0.6,0.7,1.0])
    >>> snt_list=undo_padding(batch, lengths)
    >>> len(snt_list)
    4
    """
    batch_max_len = batch.shape[1]
    as_list = []
    for seq, seq_length in zip(batch, lengths):
        actual_size = int(torch.round(seq_length * batch_max_len))
        seq_true = seq.narrow(0, 0, actual_size)
        as_list.append(seq_true.tolist())
    return as_list


def lengths_arg_exists(func):
    """Check if func takes ``lengths`` keyword argument.

    Arguments
    ---------
    func : callable
        The function, method, or other callable to search for the lengths arg.

    Returns
    -------
    True if func takes ``lengths`` keyword argument.
    """
    spec = inspect.getfullargspec(func)
    return 'lengths' in spec.args + spec.kwonlyargs


def is_distributed_initialized() ->bool:
    """Returns whether the current system is distributed."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def if_main_process() ->bool:
    """Returns whether the current process is the main process."""
    if is_distributed_initialized():
        return torch.distributed.get_rank() == 0
    else:
        return True


class MultiProcessLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that handles multi-process logging, ensuring logs are written
    only on the main process if specified. This class extends `logging.LoggerAdapter`
    and provides additional functionality for controlling logging in multi-process
    environments, with the option to limit logs to the main process only.

    This class is heavily inspired by HuggingFace Accelerate toolkit:
    https://github.com/huggingface/accelerate/blob/85b1a03552cf8d58e036634e004220c189bfb247/src/accelerate/logging.py#L22
    """

    @staticmethod
    def _should_log(main_process_only: 'bool') ->bool:
        """
        Determines if logging should occur based on whether the code is running
        on the main process or not.

        Arguments
        ---------
        main_process_only : bool
            A flag indicating if logging should be restricted to the main process.

        Returns
        -------
        bool
            True if logging should be performed (based on the process and the flag),
            False otherwise.
        """
        return not main_process_only or main_process_only and if_main_process()

    def log(self, level: 'int', msg: 'str', *args: tuple, **kwargs: dict):
        """
        Logs a message with the specified log level, respecting the `main_process_only`
        flag to decide whether to log based on the current process.

        Arguments
        ---------
        level : int
            Logging level (e.g., logging.INFO, logging.WARNING).
        msg : str
            The message to log.
        *args : tuple
            Additional positional arguments passed to the logger.
        **kwargs : dict
            Additional keyword arguments passed to the logger, including:
            - main_process_only (bool): If True, log only from the main process (default: True).
            - stacklevel (int): The stack level to use when logging (default: 2).

        Notes
        -----
        If `main_process_only` is True, the log will only be written if the current process
        is the main process, as determined by `if_main_process()`.
        """
        main_process_only = kwargs.pop('main_process_only', True)
        kwargs.setdefault('stacklevel', 2)
        if self.isEnabledFor(level):
            if self._should_log(main_process_only):
                msg, kwargs = self.process(msg, kwargs)
                self.logger.log(level, msg, *args, **kwargs)

    @functools.lru_cache(None)
    def warning_once(self, *args: tuple, **kwargs: dict):
        """
        Logs a warning message only once by using caching to prevent duplicate warnings.

        Arguments
        ---------
        *args : tuple
            Positional arguments passed to the warning log.
        **kwargs : dict
            Keyword arguments passed to the warning log.

        Notes
        -----
        This method is decorated with `functools.lru_cache(None)`, ensuring that the warning
        message is logged only once regardless of how many times the method is called.
        """
        self.warning(*args, **kwargs)


def get_logger(name: 'str') ->MultiProcessLoggerAdapter:
    """
    Retrieves a logger with the specified name, applying a log level from the environment variable
    `SB_LOG_LEVEL` if set, or defaults to `INFO` level.

    If the environment variable `SB_LOG_LEVEL` is not defined, it defaults to `INFO` level and sets
    this level in the environment for future use. The environment variable can be set manually or
    automatically in `Brain` class following `setup_logging`.

    Arguments
    ---------
    name : str
        The name of the logger to retrieve.

    Returns
    -------
    MultiProcessLoggerAdapter
        An instance of `MultiProcessLoggerAdapter` wrapping the logger with the specified name.
    """
    logger = logging.getLogger(name)
    log_level = os.environ.get('SB_LOG_LEVEL', None)
    if log_level is None:
        log_level = logging.INFO
        os.environ['SB_LOG_LEVEL'] = str(log_level)
    logging.basicConfig(level=int(log_level))
    return MultiProcessLoggerAdapter(logger, {})


class Augmenter(torch.nn.Module):
    """Applies pipelines of data augmentation.

    Arguments
    ---------
    parallel_augment: bool
        If False, the augmentations are applied sequentially with
        the order specified in the pipeline argument.
        When True, all the N augmentations are concatenated in the output
        on the batch axis.
    parallel_augment_fixed_bs: bool
        If False, each augmenter (performed in parallel) generates a number of
        augmented examples equal to the batch size. Thus, overall, with this
        option N*batch size artificial data are
        generated, where N is the number of augmenters.
        When True, the number of total augmented examples is kept fixed at
        the batch size, thus, for each augmenter, fixed at batch size // N examples.
        This option is useful to keep controlled the number of synthetic examples
        with respect to the original data distribution, as it keep always
        50% of original data, and 50% of augmented data.
    concat_original: bool
        if True, the original input is concatenated with the
        augmented outputs (on the batch axis).
    min_augmentations: int
        The number of augmentations applied to the input signal is randomly
        sampled between min_augmentations and max_augmentations. For instance,
        if the augmentation dict contains N=6 augmentations and we set
        select min_augmentations=1 and max_augmentations=4 we apply up to
        M=4 augmentations. The selected augmentations are applied in the order
        specified in the augmentations dict. If shuffle_augmentations = True,
        a random set of M augmentations is selected.
    max_augmentations: int
        Maximum number of augmentations to apply. See min_augmentations for
        more details.
    shuffle_augmentations:  bool
        If True, it shuffles the entries of the augmentations dictionary.
        The effect is to randomply select the order of the augmentations
        to apply.
    repeat_augment: int
        Applies the augmentation algorithm N times. This can be used to
        perform more data augmentation.
    augment_start_index: int
        The index of the first element in the input batch from which data
        augmentation should begin.
        This argument allows you to specify the starting point for applying
        data augmentation.
    augment_end_index: int
        The index of the last element in the input batch at which data
        augmentation should stop.
        You can use this argument to define the endpoint for applying data
        augmentation within the batch.
    concat_start_index: int
        If `concat_original` is set to True, you can specify a subpart of the
        original batch to concatenate in the output.
        Use this argument to select the index of the first element from the
        original input batch to start copying from.
    concat_end_index: int
        If `concat_original` is set to True, you can specify a subpart of the
        original batch to concatenate in the output. Use this argument to select
        the index of the last element from the original input batch to end the
        copying process.
    augment_prob: float
        The probability (0.0 to 1.0) of applying data augmentation. When set to 0.0,
        the original signal is returned without any augmentation. When set to 1.0,
        augmentation is always applied. Values in between determine the likelihood
        of augmentation.
    augmentations: list
        List of augmentater objects to combine to perform data augmentation.
    enable_augmentations: list
        A list of booleans used to selectively enable or disable specific augmentation
        techniques within the 'augmentations' list.
        Each boolean corresponds to an augmentation object in the 'augmentations' list
        and should be of the same length and order.
        This feature is useful for performing ablations on augmentation techniques to
        tailor them for a specific task.

    Example
    -------
    >>> from speechbrain.augment.time_domain import DropFreq, DropChunk
    >>> freq_dropper = DropFreq()
    >>> chunk_dropper = DropChunk(drop_start=100, drop_end=16000)
    >>> augment = Augmenter(parallel_augment=False, concat_original=False, augmentations=[freq_dropper, chunk_dropper])
    >>> signal = torch.rand([4, 16000])
    >>> output_signal, lengths = augment(signal, lengths=torch.tensor([0.2,0.5,0.7,1.0]))
    """

    def __init__(self, parallel_augment=False, parallel_augment_fixed_bs=False, concat_original=False, min_augmentations=None, max_augmentations=None, shuffle_augmentations=False, repeat_augment=1, augment_start_index=0, augment_end_index=None, concat_start_index=0, concat_end_index=None, augment_prob=1.0, augmentations=list(), enable_augmentations=None):
        super().__init__()
        self.parallel_augment = parallel_augment
        self.parallel_augment_fixed_bs = parallel_augment_fixed_bs
        self.concat_original = concat_original
        self.augmentations = augmentations
        self.min_augmentations = min_augmentations
        self.max_augmentations = max_augmentations
        self.shuffle_augmentations = shuffle_augmentations
        self.augment_start_index = augment_start_index
        self.augment_end_index = augment_end_index
        self.concat_start_index = concat_start_index
        self.concat_end_index = concat_end_index
        self.repeat_augment = repeat_augment
        self.augment_prob = augment_prob
        self.check_min_max_augmentations()
        self.num_augmentations = None
        self.do_augment = True
        if not isinstance(self.repeat_augment, int):
            raise ValueError('repeat_augment must be an integer.')
        if self.repeat_augment < 0:
            raise ValueError('repeat_augment must be greater than 0.')
        if self.augment_end_index is not None:
            if self.augment_end_index < self.augment_start_index:
                raise ValueError('augment_end_index must be smaller or equal to augment_start_index.')
        if self.concat_end_index is not None:
            if self.concat_end_index < self.concat_start_index:
                raise ValueError('concat_end_index must be smaller or equal to concat_start_index.')
        if enable_augmentations is None:
            enable_augmentations = [True] * len(augmentations)
        elif not isinstance(enable_augmentations, list):
            raise ValueError('enable_augmentations must be a list.')
        elif len(enable_augmentations) != len(augmentations):
            raise ValueError('enable_augmentations must have the same length as augmentations.')
        else:
            augmentations = [aug for aug, enabled in zip(augmentations, enable_augmentations) if enabled]
        self.augmentations = {(augmentation.__class__.__name__ + str(i)): augmentation for i, augmentation in enumerate(augmentations)}
        if len(self.augmentations) == 0:
            logger.warning('No augmentation is applied because the augmentation list is empty.')
        if self.max_augmentations <= 0:
            logger.warning('No augmentations applied because max_augmentations is non-positive.')
        if self.min_augmentations < 0:
            self.min_augmentations = 0
            logger.warning('min_augmentations is negative. Modified to be non-negative.')
        if self.min_augmentations > self.max_augmentations:
            logger.warning('min_augmentations is greater than max_augmentations. min_augmentations set to max_augmentations.')
            self.max_augmentations = self.min_augmentations
        self.require_lengths = {}
        for aug_key, aug_fun in self.augmentations.items():
            self.require_lengths[aug_key] = lengths_arg_exists(aug_fun.forward)

    def augment(self, x, lengths, selected_augmentations):
        """Applies data augmentation on the selected augmentations.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to augment.
        lengths : torch.Tensor
            The length of each sequence in the batch.
        selected_augmentations: dict
            Dictionary containing the selected augmentation to apply.

        Returns
        -------
        output : torch.Tensor
            Augmented outputs.
        output_lengths : torch.Tensor
            The corresponding length of each output.
        """
        next_input = x
        next_lengths = lengths
        output = []
        output_lengths = []
        out_lengths = lengths
        for k, augment_name in enumerate(selected_augmentations):
            augment_fun = self.augmentations[augment_name]
            idx = torch.arange(x.shape[0])
            if self.parallel_augment and self.parallel_augment_fixed_bs:
                idx_startstop = torch.linspace(0, x.shape[0], len(selected_augmentations) + 1)
                idx_start = idx_startstop[k]
                idx_stop = idx_startstop[k + 1]
                idx = idx[idx_start:idx_stop]
            if self.require_lengths[augment_name]:
                out = augment_fun(next_input[idx, ...], lengths=next_lengths[idx])
            else:
                out = augment_fun(next_input[idx, ...])
            if isinstance(out, tuple):
                if len(out) == 2:
                    out, out_lengths = out
                else:
                    raise ValueError('The function must return max two arguments (Tensor, Length[optional])')
            if not self.parallel_augment:
                next_input = out
                next_lengths = out_lengths[idx]
            else:
                output.append(out)
                output_lengths.append(out_lengths)
        if self.parallel_augment:
            output, output_lengths = self.concatenate_outputs(output, output_lengths)
        else:
            output = out
            output_lengths = out_lengths
        return output, output_lengths

    def forward(self, x, lengths):
        """Applies data augmentation.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to augment.
        lengths : torch.Tensor
            The length of each sequence in the batch.

        Returns
        -------
        output : torch.Tensor
            Augmented outputs.
        output_lengths : torch.Tensor
            The corresponding length of each output.
        """
        self.do_augment = True
        if random.random() > self.augment_prob:
            self.do_augment = False
            return x, lengths
        x_original = x
        len_original = lengths
        self.augment_end_index_batch = min(self.augment_end_index, x.shape[0]) if self.augment_end_index is not None else x.shape[0]
        if self.augment_start_index >= x.shape[0]:
            self.do_augment = False
            logger.warning('No augmentation is applied because the augmentation start index is greater than or equal to the number of examples in the input batch.')
            return x, lengths
        self.N_augment = torch.randint(low=self.min_augmentations, high=self.max_augmentations + 1, size=(1,), device=x.device)
        augmentations_lst = list(self.augmentations.keys())
        if self.repeat_augment == 0 or self.N_augment == 0 or len(augmentations_lst) == 0:
            self.do_augment = False
            return x, lengths
        if self.shuffle_augmentations:
            random.shuffle(augmentations_lst)
        selected_augmentations = augmentations_lst[0:self.N_augment]
        x = x[self.augment_start_index:self.augment_end_index_batch]
        lengths = lengths[self.augment_start_index:self.augment_end_index_batch]
        output_lst = []
        output_len_lst = []
        self.skip_concat = not self.concat_original
        if self.concat_original:
            if self.concat_start_index >= x_original.shape[0]:
                self.skip_concat = True
                pass
            else:
                self.skip_concat = False
                self.concat_end_index_batch = min(self.concat_end_index, x_original.shape[0]) if self.concat_end_index is not None else x_original.shape[0]
                output_lst.append(x_original[self.concat_start_index:self.concat_end_index_batch])
                output_len_lst.append(len_original[self.concat_start_index:self.concat_end_index_batch])
        for i in range(self.repeat_augment):
            output, output_lengths = self.augment(x, lengths, selected_augmentations)
            output_lst.append(output)
            output_len_lst.append(output_lengths)
        output, output_lengths = self.concatenate_outputs(output_lst, output_len_lst)
        return output, output_lengths

    def concatenate_outputs(self, augment_lst, augment_len_lst):
        """
        Concatenate a list of augmented signals, accounting for varying temporal lengths.
        Padding is applied to ensure all signals can be concatenated.

        Arguments
        ---------
        augment_lst : List of torch.Tensor
            List of augmented signals to be concatenated.
        augment_len_lst : List of torch.Tensor
            List of lengths corresponding to the augmented signals.

        Returns
        -------
        concatenated_signals : torch.Tensor
            A tensor containing the concatenated signals.
        concatenated_lengths : torch.Tensor
            A tensor containing the concatenated signal lengths.

        Notes
        -----
        This function takes a list of augmented signals, which may have different temporal
        lengths due to variations such as speed changes. It pads the signals to match the
        maximum temporal dimension found among the input signals and rescales the lengths
        accordingly before concatenating them.
        """
        max_len = max(augment.shape[1] for augment in augment_lst)
        augment_len_lst = [(length * (output.shape[1] / max_len)) for length, output in zip(augment_len_lst, augment_lst)]
        augment_lst = [F.pad(output, (0, max_len - output.shape[1])) for output in augment_lst]
        output = torch.cat(augment_lst, dim=0)
        output_lengths = torch.cat(augment_len_lst, dim=0)
        return output, output_lengths

    def replicate_multiple_labels(self, *args):
        """
        Replicates the labels along the batch axis a number of times that
        corresponds to the number of augmentations. Indeed parallel and
        concatenation augmentations alter the time dimension.

        Arguments
        ---------
        *args : tuple
            Input label tensors to be replicated. Can be a uniq or a list of
            torch.Tensors.

        Returns
        -------
        augmented_labels: torch.Tensor
            Labels corresponding to the augmented input. Returns as many torch.Tensor
            as given in input.
        """
        if not self.do_augment:
            return args
        list_of_augmented_labels = []
        for labels in args:
            list_of_augmented_labels.append(self.replicate_labels(labels))
        return list_of_augmented_labels

    def replicate_labels(self, labels):
        """
        Replicates the labels along the batch axis a number of times that
        corresponds to the number of augmentations. Indeed parallel and
        concatenation augmentations alter the time dimension.

        Arguments
        ---------
        labels : torch.Tensor
            Input label tensors to be replicated.

        Returns
        -------
        augmented_labels: torch.Tensor
            Labels corresponding to the augmented input. Returns as many torch.Tensor
            as given in input.
        """
        if not self.do_augment:
            return labels
        augmented_labels = []
        if self.concat_original and not self.skip_concat:
            augmented_labels = [labels[self.concat_start_index:self.concat_end_index_batch]]
        selected_labels = labels[self.augment_start_index:self.augment_end_index_batch]
        if self.parallel_augment:
            selected_labels = torch.cat([selected_labels] * self.N_augment, dim=0)
        augmented_labels = augmented_labels + [selected_labels] * self.repeat_augment
        augmented_labels = torch.cat(augmented_labels, dim=0)
        return augmented_labels

    def check_min_max_augmentations(self):
        """Checks the min_augmentations and max_augmentations arguments."""
        if self.min_augmentations is None:
            self.min_augmentations = 1
        if self.max_augmentations is None:
            self.max_augmentations = len(self.augmentations)
        if self.max_augmentations > len(self.augmentations):
            self.max_augmentations = len(self.augmentations)
        if self.min_augmentations > len(self.augmentations):
            self.min_augmentations = len(self.augmentations)


class CodecAugment(torch.nn.Module):
    """
    Apply random audio codecs to input waveforms using torchaudio.

    This class provides an interface for applying codec augmentation techniques to audio data.

    Arguments
    ---------
    sample_rate: int
        The sample rate of the input waveform.

    Example
    -------
    >>> waveform = torch.rand(4, 16000)
    >>> if torchaudio.list_audio_backends()[0] == 'ffmpeg':
    ...     augmenter = CodecAugment(16000)
    ...     output_waveform = augmenter(waveform)
    """

    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.available_format_encoders = [('wav', 'pcm_mulaw'), ('mp3', None), ('g722', None)]

    def apply_codec(self, waveform, format=None, encoder=None):
        """
        Apply the selected audio codec.

        Arguments
        ----------
        waveform: torch.Tensor
            Input waveform of shape `[batch, time]`.
        format: str
            The audio format to use (e.g., "wav", "mp3"). Default is None.
        encoder: str
            The encoder to use for the format (e.g., "opus", "vorbis"). Default is None.

        Returns
        ---------
        torch.Tensor:
            Coded version of the input waveform of shape `[batch, time]`.
        """
        audio_effector = torchaudio.io.AudioEffector(format=format, encoder=encoder)
        waveform_aug = audio_effector.apply(waveform.transpose(0, 1), self.sample_rate)
        return waveform_aug.transpose(0, 1)

    def forward(self, waveform):
        """
        Apply a random audio codec from the available list.

        Arguments
        ---------
        waveform: torch.Tensor
            Input waveform of shape `[batch, time]`.

        Returns
        -------
        torch.Tensor
            Coded version of the input waveform of shape `[batch, time]`.
        """
        format, encoder = random.choice(self.available_format_encoders)
        return self.apply_codec(waveform, format=format, encoder=encoder)


class SpectrogramDrop(torch.nn.Module):
    """This class drops slices of the input spectrogram.

    Using `SpectrogramDrop` as an augmentation strategy helps a models learn to rely
    on all parts of the signal, since it can't expect a given part to be
    present.

    Reference:
        https://arxiv.org/abs/1904.08779

    Arguments
    ---------
    drop_length_low : int
        The low end of lengths for which to drop the
        spectrogram, in samples.
    drop_length_high : int
        The high end of lengths for which to drop the
        signal, in samples.
    drop_count_low : int
        The low end of number of times that the signal
        can be dropped.
    drop_count_high : int
        The high end of number of times that the signal
        can be dropped.
    replace: str
        - 'zeros': Masked values are replaced with zeros.
        - 'mean': Masked values are replaced with the mean value of the spectrogram.
        - 'rand': Masked values are replaced with random numbers ranging between
                  the maximum and minimum values of the spectrogram.
        - 'cutcat': Masked values are replaced with chunks from other signals in the batch.
        - 'swap': Masked values are replaced with other chunks from the same sentence.
        - 'random_selection': A random selection among the approaches above.
    dim : int
        Corresponding dimension to mask. If dim=1, we apply time masking.
        If dim=2, we apply frequency masking.

    Example
    -------
    >>> # time-masking
    >>> drop = SpectrogramDrop(dim=1)
    >>> spectrogram = torch.rand(4, 150, 40)
    >>> print(spectrogram.shape)
    torch.Size([4, 150, 40])
    >>> out = drop(spectrogram)
    >>> print(out.shape)
    torch.Size([4, 150, 40])
    >>> # frequency-masking
    >>> drop = SpectrogramDrop(dim=2)
    >>> spectrogram = torch.rand(4, 150, 40)
    >>> print(spectrogram.shape)
    torch.Size([4, 150, 40])
    >>> out = drop(spectrogram)
    >>> print(out.shape)
    torch.Size([4, 150, 40])
    """

    def __init__(self, drop_length_low=5, drop_length_high=15, drop_count_low=1, drop_count_high=3, replace='zeros', dim=1):
        super().__init__()
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.replace = replace
        self.dim = dim
        if drop_length_low > drop_length_high:
            raise ValueError('Low limit must not be more than high limit')
        if drop_count_low > drop_count_high:
            raise ValueError('Low limit must not be more than high limit')
        self.replace_opts = ['zeros', 'mean', 'rand', 'cutcat', 'swap', 'random_selection']
        if self.replace not in self.replace_opts:
            raise ValueError(f"Invalid 'replace' option. Select one of {', '.join(self.replace_opts)}")

    def forward(self, spectrogram):
        """
        Apply the DropChunk augmentation to the input spectrogram.

        This method randomly drops chunks of the input spectrogram to augment the data.

        Arguments
        ---------
        spectrogram : torch.Tensor
            Input spectrogram of shape `[batch, time, fea]`.

        Returns
        -------
        torch.Tensor
            Augmented spectrogram of shape `[batch, time, fea]`.
        """
        if spectrogram.dim() == 4:
            spectrogram = spectrogram.view(-1, spectrogram.shape[2], spectrogram.shape[3])
        batch_size, time_duration, fea_size = spectrogram.shape
        if self.dim == 1:
            D = time_duration
        else:
            D = fea_size
        n_masks = torch.randint(low=self.drop_count_low, high=self.drop_count_high + 1, size=(1,), device=spectrogram.device)
        if n_masks == 0:
            return spectrogram
        mask_len = torch.randint(low=self.drop_length_low, high=self.drop_length_high, size=(batch_size, n_masks), device=spectrogram.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D, -mask_len.max()), (batch_size, n_masks), device=spectrogram.device).unsqueeze(2)
        arange = torch.arange(D, device=spectrogram.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < mask_pos + mask_len)
        mask = mask.any(dim=1)
        mask = mask.unsqueeze(2) if self.dim == 1 else mask.unsqueeze(1)
        if self.replace == 'random_selection':
            self.replace = random.choice(self.replace_opts[:-1])
        if self.replace == 'zeros':
            spectrogram = spectrogram.masked_fill_(mask, 0.0)
        elif self.replace == 'mean':
            mean = spectrogram.mean().detach()
            spectrogram = spectrogram.masked_fill_(mask, mean)
        elif self.replace == 'rand':
            max_spectrogram = spectrogram.max().detach()
            min_spectrogram = spectrogram.min().detach()
            rand_spectrogram = torch.rand_like(spectrogram)
            rand_spectrogram = rand_spectrogram * (max_spectrogram - min_spectrogram) + min_spectrogram
            mask = mask.float()
            spectrogram = (1 - mask) * spectrogram + mask * rand_spectrogram
        elif self.replace == 'cutcat':
            rolled_spectrogram = torch.roll(spectrogram, shifts=1, dims=0)
            mask = mask.float()
            spectrogram = (1 - mask) * spectrogram + mask * rolled_spectrogram
        elif self.replace == 'swap':
            shift = torch.randint(low=1, high=spectrogram.shape[1], size=(1,), device=spectrogram.device)
            rolled_spectrogram = torch.roll(spectrogram, shifts=shift.item(), dims=1)
            mask = mask.float()
            spectrogram = (1 - mask) * spectrogram + mask * rolled_spectrogram
        return spectrogram.view(*spectrogram.shape)


class Warping(torch.nn.Module):
    """
    Apply time or frequency warping to a spectrogram.

    If `dim=1`, time warping is applied; if `dim=2`, frequency warping is applied.
    This implementation selects a center and a window length to perform warping.
    It ensures that the temporal dimension remains unchanged by upsampling or
    downsampling the affected regions accordingly.

    Reference:
        https://arxiv.org/abs/1904.08779

    Arguments
    ---------
    warp_window : int, optional
        The width of the warping window. Default is 5.
    warp_mode : str, optional
        The interpolation mode for time warping. Default is "bicubic."
    dim : int, optional
        Dimension along which to apply warping (1 for time, 2 for frequency).
        Default is 1.

    Example
    -------
    >>> # Time-warping
    >>> warp = Warping()
    >>> spectrogram = torch.rand(4, 150, 40)
    >>> print(spectrogram.shape)
    torch.Size([4, 150, 40])
    >>> out = warp(spectrogram)
    >>> print(out.shape)
    torch.Size([4, 150, 40])
    >>> # Frequency-warping
    >>> warp = Warping(dim=2)
    >>> spectrogram = torch.rand(4, 150, 40)
    >>> print(spectrogram.shape)
    torch.Size([4, 150, 40])
    >>> out = warp(spectrogram)
    >>> print(out.shape)
    torch.Size([4, 150, 40])
    """

    def __init__(self, warp_window=5, warp_mode='bicubic', dim=1):
        super().__init__()
        self.warp_window = warp_window
        self.warp_mode = warp_mode
        self.dim = dim

    def forward(self, spectrogram):
        """
        Apply warping to the input spectrogram.

        Arguments
        ---------
        spectrogram : torch.Tensor
            Input spectrogram with shape `[batch, time, fea]`.

        Returns
        -------
        torch.Tensor
            Augmented spectrogram with shape `[batch, time, fea]`.
        """
        if self.dim == 2:
            spectrogram = spectrogram.transpose(1, 2)
        original_size = spectrogram.shape
        window = self.warp_window
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(1)
        len_original = spectrogram.shape[2]
        if len_original - window <= window:
            return spectrogram.view(*original_size)
        c = torch.randint(window, len_original - window, (1,))[0]
        w = torch.randint(c - window, c + window, (1,))[0] + 1
        left = torch.nn.functional.interpolate(spectrogram[:, :, :c], (w, spectrogram.shape[3]), mode=self.warp_mode, align_corners=True)
        right = torch.nn.functional.interpolate(spectrogram[:, :, c:], (len_original - w, spectrogram.shape[3]), mode=self.warp_mode, align_corners=True)
        spectrogram[:, :, :w] = left
        spectrogram[:, :, w:] = right
        spectrogram = spectrogram.view(*original_size)
        if self.dim == 2:
            spectrogram = spectrogram.transpose(1, 2)
        return spectrogram


class RandomShift(torch.nn.Module):
    """Shifts the input tensor by a random amount, allowing for either a time
    or frequency (or channel) shift depending on the specified axis.
    It is crucial to calibrate the minimum and maximum shifts according to the
    requirements of your specific task.
    We recommend using small shifts to preserve information integrity.
    Using large shifts may result in the loss of significant data and could
    potentially lead to misalignments with corresponding labels.

    Arguments
    ---------
    min_shift : int
        The minimum channel shift.
    max_shift : int
        The maximum channel shift.
    dim: int
        The dimension to shift.

    Example
    -------
    >>> # time shift
    >>> signal = torch.zeros(4, 100, 80)
    >>> signal[0,50,:] = 1
    >>> rand_shift =  RandomShift(dim=1, min_shift=-10, max_shift=10)
    >>> lengths = torch.tensor([0.2, 0.8, 0.9,1.0])
    >>> output_signal, lengths = rand_shift(signal,lengths)

    >>> # frequency shift
    >>> signal = torch.zeros(4, 100, 80)
    >>> signal[0,:,40] = 1
    >>> rand_shift =  RandomShift(dim=2, min_shift=-10, max_shift=10)
    >>> lengths = torch.tensor([0.2, 0.8, 0.9,1.0])
    >>> output_signal, lengths = rand_shift(signal,lengths)
    """

    def __init__(self, min_shift=0, max_shift=0, dim=1):
        super().__init__()
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.dim = dim
        if self.max_shift < self.min_shift:
            raise ValueError('max_shift must be  >= min_shift')

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """
        N_shifts = torch.randint(low=self.min_shift, high=self.max_shift + 1, size=(1,), device=waveforms.device)
        waveforms = torch.roll(waveforms, shifts=N_shifts.item(), dims=self.dim)
        if self.dim == 1:
            lengths = lengths + N_shifts / waveforms.shape[self.dim]
            lengths = torch.clamp(lengths, min=0.0, max=1.0)
        return waveforms, lengths


class CircularDependencyError(ValueError):
    """
    An error caused by running into circular dependencies while searching for
    an evaluation order in a DependencyGraph.
    """
    pass


DGNode = collections.namedtuple('DGNode', ['key', 'edges', 'data'])


class DependencyGraph:
    """General-purpose dependency graph.

    Essentially a directed acyclic graph.
    Usually used to find an evaluation order for e.g. variable substitution
    The relation that an edge between A and B represents is:
    "A depends on B, i.e. B should be evaluated before A"

    Nodes can be added explicitly or they can be created implicitly
    while adding edges.
    Nodes have keys, which should be some hashable value that identifies
    the elements the graph represents in your use case. E.G. they can just
    be the variable name you want to substitute.
    However, if needed, more generally you can attach any data to a node
    (e.g. a path in your tree), and if so desired, a unique key can be
    created for you. You'll only need to know that key while adding edges
    to/from it.
    Implicit keys and explicit keys can also be mixed.
    """

    def __init__(self):
        self.digraph = []
        self.key2ind = {}
        self._manually_added_keys = []

    @staticmethod
    def get_unique_key():
        """Returns a unique hashable identifier."""
        return uuid.uuid4()

    def add_node(self, key=None, data=None):
        """Adds a node explicitly.

        Arguments
        ---------
        key : hashable, optional
            If not given, a key is created for you.
        data : Any, optional
            Any additional data you wish to attach to this node.

        Returns
        -------
        hashable
            The key that was used (either yours or generated).

        Raises
        ------
        ValueError
            If node with the given key has already been added explicitly
            (with this method, not "add_edge").
        """
        if key is None:
            key = self.get_unique_key()
        elif key in self._manually_added_keys:
            raise ValueError('Adding duplicate node: {key}'.format(key=key))
        else:
            self._manually_added_keys.append(key)
        if key in self.key2ind:
            ind = self.key2ind[key]
            node = self.digraph[ind]
            self.digraph[ind] = DGNode(node.key, node.edges, data)
            return key
        self.key2ind[key] = len(self.digraph)
        self.digraph.append(DGNode(key, [], data))
        return key

    def add_edge(self, from_key, to_key):
        """Adds an edge, and implicitly also creates nodes for keys which have
        not been seen before. This will not let you add data to your nodes.
        The relation encodes: "from_key depends on to_key"
        (to_key must be evaluated before from_key).

        Arguments
        ---------
        from_key : hashable
            The key which depends on.
        to_key : hashable
            The key which is depended on.
        """
        from_ind = self._get_ind_and_add_if_new(from_key)
        to_ind = self._get_ind_and_add_if_new(to_key)
        edges_list = self.digraph[from_ind].edges
        if to_ind not in edges_list:
            edges_list.append(to_ind)

    def _get_ind_and_add_if_new(self, key):
        if key not in self.key2ind:
            self.key2ind[key] = len(self.digraph)
            self.digraph.append(DGNode(key, [], None))
        return self.key2ind[key]

    def is_valid(self):
        """Checks if an evaluation order can be found.

        A dependency graph is evaluatable if there are no circular
        dependencies, i.e., the graph is acyclic.

        Returns
        -------
        bool
            Indicating if the graph is evaluatable.
        """
        return not self._find_first_cycle()

    def get_evaluation_order(self, selected_keys=None):
        """Finds one valid evaluation order.

        There can be many different valid
        orders.
        NOTE: Generates output one DGNode at a time. May generate DGNodes
        before it finds a circular dependency. If you really need to know
        whether an order can be found, check is_valid() first. However,
        the algorithm for finding cycles is essentially the same as the one
        used for finding an evaluation order, so for very large graphs...
        Ah well, but maybe then you should be using some other solution
        anyway.

        Arguments
        ---------
        selected_keys : list, None
            List of keys. If not None, only the selected keys are guaranteed
            in the evaluation order (along with the keys they depend on).

        Yields
        ------
        DGNode
            The added DGNodes in a valid evaluation order.
            See the DGNode namedtuple above.

        Raises
        ------
        CircularDependencyError
            If a circular dependency is found.
        """
        seen_ever = set()

        def toposort(root_ind, visited):
            """Implementation of toposort."""
            nonlocal seen_ever
            here = visited + [root_ind]
            if root_ind in visited:
                raise CircularDependencyError('{cycle}'.format(cycle=' -> '.join(str(self.digraph[i].key) for i in here)))
            if root_ind in seen_ever:
                return
            seen_ever = seen_ever.union(set([root_ind]))
            for to_ind in self.digraph[root_ind].edges:
                for ind in toposort(to_ind, visited=here):
                    yield ind
            yield root_ind
        if selected_keys is None:
            start_inds = range(len(self.digraph))
        else:
            start_inds = [self.key2ind[key] for key in selected_keys]
        for start_ind in start_inds:
            for ind in toposort(start_ind, []):
                yield self.digraph[ind]

    def _find_first_cycle(self):
        """Depth-first search based algorithm for finding cycles in the graph."""
        seen_ever = set()

        def cycle_dfs(root_ind, visited):
            """Implementation of cycle_dfs."""
            nonlocal seen_ever
            None
            here = visited + [root_ind]
            if root_ind in visited:
                return here
            if root_ind in seen_ever:
                return []
            seen_ever = seen_ever.union(set([root_ind]))
            for to_ind in self.digraph[root_ind].edges:
                cycle = cycle_dfs(to_ind, here)
                if cycle:
                    return cycle
            return []
        for ind in range(len(self.digraph)):
            if ind not in seen_ever:
                cycle = cycle_dfs(ind, [])
                if cycle:
                    return cycle
        return []

    def __contains__(self, key):
        return key in self.key2ind


class DynamicItem:
    """Essentially represents a data transformation function.

    A DynamicItem takes some arguments and computes its value dynamically when
    called. A straight-forward use-case is to load something from disk
    dynamically; take the path and provide the loaded data.

    Instances of this class are often created implicitly via the
    @takes and @provides decorators or otherwise from specifying the taken and
    provided arguments and the function.

    A counterpart is the GeneratorDynamicItem, which should be used for
    generator functions.

    Arguments
    ---------
    takes : list
        The keys of the items that this needs to compute its output.
    func : callable
        The function that is used to compute the output.
    provides : list
        The keys that this provides.
    """

    def __init__(self, takes=[], func=None, provides=[]):
        self.takes = takes
        self.func = func
        self.provides = provides

    def __call__(self, *args):
        return self.func(*args)

    def next_takes(self):
        """The next argkeys to provide to this, when called."""
        return self.takes

    def next_provides(self):
        """The next keys that this provides, when called."""
        return self.provides

    def provided_in_order(self):
        """Assuming that this may need to be called multiple times; which keys
        does it provide at that call. Returns a list, with len equal to the
        number of times that this may be called.
        """
        return [self.provides]

    def reset(self):
        """Signals that this will not be called any more times on this pipeline
        call.
        """
        pass


class GeneratorDynamicItem(DynamicItem):
    """Essentially represents a multi-step data transformation.

    This is the generator function counterpart for DynamicItem (which should be
    used for regular functions).

    A GeneratorDynamicItem first takes some arguments and then uses those in
    multiple steps to incrementally compute some values when called.

    A typical use-case is a pipeline of transformations on data: e.g. taking in
    text as a string, and first a tokenized version, and then on the second
    call providing an integer-encoded version. This can be used even though the
    integer-encoder needs to be trained on the first outputs.

    The main benefit is to be able to define the pipeline in a clear function,
    even if parts of the pipeline depend on others for their initialization.

    Arguments
    ---------
    *args : tuple
        Forwarded to parent class
    **kwargs : tuple
        Forwarded to parent class

    Example
    -------
    >>> lab2ind = {}
    >>> def text_pipeline(text):
    ...     text = text.lower().strip()
    ...     text = "".join(c for c in text if c.isalpha() or c == " ")
    ...     words = text.split()
    ...     yield words
    ...     encoded = [lab2ind[word] for word in words]
    ...     yield encoded
    >>> item = GeneratorDynamicItem(
    ...         func=text_pipeline,
    ...         takes=["text"],
    ...         provides=["words", "words_encoded"])
    >>> # First create the integer-encoding:
    >>> ind = 1
    >>> for token in item("Is this it? - This is it."):
    ...     if token not in lab2ind:
    ...         lab2ind[token] = ind
    ...         ind += 1
    >>> # Now the integers can be encoded!
    >>> item()
    [1, 2, 3, 2, 1, 3]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_generator = None
        self.num_provided_items = 0

    def __call__(self, *args):
        if self.num_provided_items == len(self.provides):
            raise RuntimeError('DynamicItemPipeline called too many times!')
        if not self.current_generator:
            self.current_generator = self.func(*args)
        out = next(self.current_generator)
        self.num_provided_items += 1
        return out

    def next_takes(self):
        """The next argkeys to provide to this, when called."""
        if not self.current_generator:
            return self.takes
        else:
            return []

    def next_provides(self):
        """The next keys that this provides, when called."""
        keys = self.provides[self.num_provided_items]
        if isinstance(keys, str):
            return [keys]
        else:
            return keys

    def provided_in_order(self):
        """Assuming that this may need to be called multiple times; which keys
        does it provide at that call. Returns a list, with len equal to the
        number of times that this may be called.
        """
        in_order = []
        for keys in self.provides:
            if isinstance(keys, str):
                in_order.append([keys])
            else:
                in_order.append(keys)
        return in_order

    def reset(self):
        """Signals that this will not be called any more times on this pipeline
        call.
        """
        if self.current_generator is not None:
            self.current_generator.close()
        self.current_generator = None
        self.num_provided_items = 0


def provides(*output_keys):
    """Decorator which makes a DynamicItem and specifies what keys it provides.

    If the wrapped object is a generator function (has a yield statement),
    Creates a GeneratorDynamicItem. If the object is already a DynamicItem,
    just specifies the provided keys for that. Otherwise creates a new regular
    DynamicItem, with provided keys specified.

    Arguments
    ---------
    *output_keys : tuple
        The data keys to be produced by this function

    Returns
    -------
    The decorated function, with output keys specified

    NOTE
    ----
    The behavior is slightly different for generators and regular functions, if
    many output keys are specified, e.g. @provides("signal", "mfcc"). Regular
    functions should return a tuple with len equal to len(output_keys), while
    generators should yield the items one by one.

    >>> @provides("signal", "feat")
    ... def read_feat():
    ...     wav = [.1,.2,-.1]
    ...     feat = [s**2 for s in wav]
    ...     return wav, feat
    >>> @provides("signal", "feat")
    ... def read_feat():
    ...     wav = [.1,.2,-.1]
    ...     yield wav
    ...     feat = [s**2 for s in wav]
    ...     yield feat

    If multiple keys are yielded at once, write e.g.,

    >>> @provides("wav_read", ["left_channel", "right_channel"])
    ... def read_multi_channel():
    ...     wav = [[.1,.2,-.1],[.2,.1,-.1]]
    ...     yield wav
    ...     yield wav[0], wav[1]

    """

    def decorator(obj):
        """Decorator definition."""
        if isinstance(obj, DynamicItem):
            if obj.provides:
                raise ValueError("Can't overwrite DynamicItem provides-list.")
            obj.provides = output_keys
            return obj
        elif inspect.isgeneratorfunction(obj):
            return GeneratorDynamicItem(func=obj, provides=output_keys)
        else:
            return DynamicItem(func=obj, provides=output_keys)
    return decorator


provides_decorator = provides


def takes(*argkeys):
    """Decorator which makes a DynamicItem and specifies its argkeys.

    If the wrapped object is a generator function (has a yield statement),
    Creates a GeneratorDynamicItem. If the object is already a DynamicItem,
    just specifies the argkeys for that. Otherwise creates a new regular
    DynamicItem, with argkeys specified.

    The args are always passed to the function at the start. Generators could
    support sending new arguments, but for such use cases, simply create a new
    dynamic item. The GeneratorDynamicItem class is meant for pipelines which
    take in an input and transform it in multiple ways, where the intermediate
    representations may be needed for e.g. fitting a BPE segmenter.

    Arguments
    ---------
    *argkeys : tuple
        The data keys expected as input

    Returns
    -------
    The decorated function, with input argkeys specified

    Example
    -------
    >>> @takes("text")
    ... def tokenize(text):
    ...     return text.strip().lower().split()
    >>> tokenize.provides = ["tokenized"]
    >>> tokenize('	This Example gets tokenized')
    ['this', 'example', 'gets', 'tokenized']
    """

    def decorator(obj):
        """Decorator definition."""
        if isinstance(obj, DynamicItem):
            if obj.takes:
                raise ValueError("Can't overwrite DynamicItem.takes")
            obj.takes = argkeys
            return obj
        elif inspect.isgeneratorfunction(obj):
            return GeneratorDynamicItem(takes=argkeys, func=obj)
        else:
            return DynamicItem(takes=argkeys, func=obj)
    return decorator


takes_decorator = takes


class DataPipeline:
    """Organises data transformations into a pipeline.

    Arguments
    ---------
    static_data_keys: list
        The keys which are provided as data
    dynamic_items: list
        A list of mappings with "func", "takes", and "provides"
    output_keys: list
        The keys to use as outputs

    Example
    -------
    >>> pipeline = DataPipeline(
    ...     static_data_keys=["text"],
    ...     dynamic_items=[
    ...     {"func": lambda x: x.lower(), "takes": "text", "provides": "foo"},
    ...     {"func": lambda x: x[::-1], "takes": "foo", "provides": "bar"},
    ...     ],
    ...     output_keys=["bar"],
    ... )
    >>> pipeline({"text": "Test"})
    {'bar': 'tset'}
    """

    def __init__(self, static_data_keys, dynamic_items=[], output_keys=[]):
        self.dg = DependencyGraph()
        self._exec_order = None
        self.key_to_node = {}
        self.unaccounted_keys = {}
        self.dynamic_items = []
        self.output_mapping = {}
        self.add_static_keys(static_data_keys)
        self.add_dynamic_items(dynamic_items)
        self.set_output_keys(output_keys)

    def add_static_keys(self, static_keys):
        """Informs the pipeline about static items.

        Static items are the ones provided to __call__ as data.
        """
        for key in static_keys:
            node_id = self.dg.add_node(data=StaticItem(key=key))
            self.key_to_node[key] = node_id

    def add_dynamic_items(self, dynamic_items):
        """Add multiple dynamic items at once."""
        for item in dynamic_items:
            try:
                self.add_dynamic_item(**item)
            except TypeError:
                self.add_dynamic_item(item)

    def add_dynamic_item(self, func, takes=None, provides=None):
        """Adds a dynamic item to the Pipeline.

        Two calling conventions. For DynamicItem objects, just use:
        add_dynamic_item(dynamic_item)
        But otherwise, should use:
        add_dynamic_item(func, takes, provides)

        Arguments
        ---------
        func : callable, DynamicItem
            If a DynamicItem is given, adds that directly. Otherwise a
            DynamicItem is created, and this specifies the callable to use. If
            a generator function is given, then create a GeneratorDynamicItem.
            Otherwise creates a normal DynamicItem.
        takes : list, str
            List of keys. When func is called, each key is resolved to
            either an entry in the data or the output of another dynamic_item.
            The func is then called with these as positional arguments,
            in the same order as specified here.
            A single key can be given as a bare string.
        provides : str, list
            For regular functions, the key or list of keys that it provides.
            If you give a generator function, key or list of keys that it
            yields, in order. Also see the provides decorator.
            A single key can be given as a bare string.

        Returns
        -------
        None
        """
        if isinstance(func, DynamicItem):
            if takes is not None or provides is not None:
                raise ValueError("If providing a DynamicItem directly, don't specify takes or provides")
            else:
                self._add_dynamic_item_object(func)
                return
        if isinstance(takes, str):
            takes = [takes]
        if isinstance(provides, str):
            provides = [provides]
        di = takes_decorator(*takes)(provides_decorator(*provides)(func))
        self._add_dynamic_item_object(di)

    def _add_dynamic_item_object(self, obj):
        """Internally adds the object.

        There is a node in the dependency graph for each call of the
        DynamicItem. Each call may return multiple keys and depend on multiple
        keys. An internal dict maps key to the id of the node that produces it.
        """
        if not obj.provides:
            raise ValueError("Won't add redundant dynamic item which doesn't provide anything.")
        depended = []
        for key in obj.takes:
            if key not in self.key_to_node:
                dependee_keys = self.unaccounted_keys.setdefault(key, [])
                dependee_keys.extend(obj.next_provides())
            else:
                depended.append(self.key_to_node[key])
        for provided in obj.provided_in_order():
            node_id = self.dg.add_node(data=obj)
            for key in provided:
                self.key_to_node[key] = node_id
                if key in self.unaccounted_keys:
                    for dependee_key in self.unaccounted_keys[key]:
                        dependee_node = self.key_to_node[dependee_key]
                        self.dg.add_edge(dependee_node, node_id)
                    del self.unaccounted_keys[key]
            for dep_id in depended:
                self.dg.add_edge(node_id, dep_id)
            depended = [node_id]
        self.dynamic_items.append(obj)

    def set_output_keys(self, keys):
        """Use this to change the output keys.

        Also re-evaluates execution order.
        So if you request different outputs, some parts of the
        data pipeline may be skipped.

        Arguments
        ---------
        keys : dict, list, None
            List of keys (str) to produce in output.

            If a dict is given; it is used to map internal keys to output keys.
            From the output_keys dict key:value pairs the key appears outside,
            and value is the internal key.
        """
        self.output_mapping = self._output_keys_to_mapping(keys)
        self._exec_order = None

    @staticmethod
    def _output_keys_to_mapping(keys):
        if keys is None:
            output_mapping = {}
        elif isinstance(keys, dict):
            output_mapping = keys
        else:
            output_mapping = {key: key for key in keys}
        return output_mapping

    def compute_outputs(self, data):
        """
        Arguments
        ---------
        data : dict
            Dictionary with data entries by key.

        Returns
        -------
        dict
            With the keys that were set.
        """
        if self._exec_order is None:
            self._prepare_run(data)
        return self._compute(data, self._exec_order, self.output_mapping)

    def compute_specific(self, keys, data):
        """Compute output of specific item, without changing output_keys."""
        output_mapping = self._output_keys_to_mapping(keys)
        order = self.dg.get_evaluation_order(selected_keys=self.get_selected_node_ids(keys))
        return self._compute(data, order, output_mapping)

    def _compute(self, data, order, output_mapping):
        if self.unaccounted_keys:
            MSG = 'These keys are still unaccounted for in the data pipeline: '
            MSG += ', '.join(self.unaccounted_keys)
            raise RuntimeError(MSG)
        intermediate = {}
        for node_id, edges, item in order:
            if isinstance(item, StaticItem):
                try:
                    data[item.key]
                    continue
                except KeyError:
                    raise KeyError(f'Expected key {item.key} in data!')
            args = [(data[argkey] if argkey in data else intermediate[argkey]) for argkey in item.next_takes()]
            provided_keys = item.next_provides()
            values = item(*args)
            if len(provided_keys) == 1:
                values = [values]
            intermediate.update(zip(provided_keys, values))
        for dynamic_item in self.dynamic_items:
            dynamic_item.reset()
        return {outkey: (data[inkey] if inkey in data else intermediate[inkey]) for outkey, inkey in output_mapping.items()}

    def get_selected_node_ids(self, selected_keys):
        """Translates selected keys to dependency graph keys."""
        return [self.key_to_node[key] for key in selected_keys]

    def __call__(self, data):
        return self.compute_outputs(data)

    def _prepare_run(self, data):
        self._exec_order = list(self.dg.get_evaluation_order(self.get_selected_node_ids(self.output_mapping.values())))

