
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


import torch


from itertools import chain


import numpy as np


from torchvision.datasets import CIFAR10


from typing import Tuple


from typing import Union


import torchvision.transforms as T


import sklearn.datasets


import matplotlib.pyplot as plt


import pandas as pd


from sklearn import datasets


from typing import Any


from typing import Callable


from typing import Collection


from typing import Dict


from typing import List


from typing import Optional


from typing import Sequence


from typing import Type


from torch import Tensor


from torch.utils.data import Dataset


from typing import Mapping


import torch.nn as nn


from abc import abstractmethod


import torch.jit


from torch import nn


from torch.utils.data import DataLoader


from torch.utils.data import Sampler


import torch.nn.functional as F


from typing import TYPE_CHECKING


from typing import Iterable


from torch.utils.data.dataset import IterableDataset


from torch.utils.data.sampler import Sampler


import functools


from enum import Enum


from typing import cast


from functools import reduce


from typing import ClassVar


from torch.utils.data._utils.collate import default_collate as torch_default_collate


import copy


import re


from functools import partial


from typing import Set


from torch.nn import Module


from torch.optim import Optimizer


from inspect import getmembers


from copy import deepcopy


from copy import copy


from torch.utils.data._utils.collate import default_collate


import inspect


from abc import ABCMeta


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.optimizer import Optimizer


import math


from torch.optim.optimizer import required


from inspect import isclass


from torch import optim


from torch.optim import lr_scheduler


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from torch.optim.lr_scheduler import CyclicLR


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import StepLR


from functools import wraps


from typing import Iterator


from typing import NamedTuple


from typing import TypeVar


import types


from types import MethodType


from torch.nn import Linear


from torch.nn import functional as F


from torch.utils.data.dataloader import default_collate


from typing import IO


from collections import defaultdict


from torch.utils.data import IterableDataset


from torch.hub import load_state_dict_from_url


from torch.utils.data import random_split


from torch.utils.data._utils.worker import get_worker_info


from types import FunctionType


from torch import tensor


import random


from typing import NoReturn


from torch.utils.data.dataset import Dataset


from abc import ABC


import logging


import collections


from torch.utils.data import DistributedSampler


from collections import namedtuple


import time


from torch.optim import Adam


from numbers import Number


from torch.nn import Flatten


from torch.nn import LogSoftmax


from torch.utils.data import Subset


from torch.utils.data import SequentialSampler


from pandas import DataFrame


@dataclass
class InputTransform:

    def __post_init__(self):
        self.callbacks: 'Optional[List]' = None
        self._transform: 'Dict[RunningStage, _InputTransformPerStage]' = {}
        for stage in RunningStage:
            if stage not in INVALID_STAGES_FOR_INPUT_TRANSFORMS:
                self._populate_transforms_for_stage(stage)

    def current_transform(self, stage: 'RunningStage', current_fn: 'str') ->Callable:
        return self._transform[stage].transforms.get(current_fn, self._identity)

    def per_sample_transform(self) ->Callable:
        """Defines the transform to be applied on a single sample on cpu for all stages stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_sample_transform(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        pass

    def train_per_sample_transform(self) ->Callable:
        """Defines the transform to be applied on a single sample on cpu for the training stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        """
        return self.per_sample_transform()

    def val_per_sample_transform(self) ->Callable:
        """Defines the transform to be applied on a single sample on cpu for the validating stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_sample_transform(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        return self.per_sample_transform()

    def test_per_sample_transform(self) ->Callable:
        """Defines the transform to be applied on a single sample on cpu for the testing stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        """
        return self.per_sample_transform()

    def predict_per_sample_transform(self) ->Callable:
        """Defines the transform to be applied on a single sample on cpu for the predicting stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_sample_transform(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        return self.per_sample_transform()

    def serve_per_sample_transform(self) ->Callable:
        """Defines the transform to be applied on a single sample on cpu for the serving stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_sample_transform(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        return self.per_sample_transform()

    def per_sample_transform_on_device(self) ->Callable:
        """Defines the transform to be applied on a single sample on device for all stages stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_sample_transform_on_device(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        pass

    def train_per_sample_transform_on_device(self) ->Callable:
        """Defines the transform to be applied on a single sample on device for the training stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        """
        return self.per_sample_transform_on_device()

    def val_per_sample_transform_on_device(self) ->Callable:
        """Defines the transform to be applied on a single sample on device for the validating stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_sample_transform_on_device(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        return self.per_sample_transform_on_device()

    def test_per_sample_transform_on_device(self) ->Callable:
        """Defines the transform to be applied on a single sample on device for the testing stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        """
        return self.per_sample_transform_on_device()

    def predict_per_sample_transform_on_device(self) ->Callable:
        """Defines the transform to be applied on a single sample on device for the predicting stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_sample_transform_on_device(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        return self.per_sample_transform_on_device()

    def serve_per_sample_transform_on_device(self) ->Callable:
        """Defines the transform to be applied on a single sample on device for the serving stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def serve_per_sample_transform_on_device(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        return self.per_sample_transform_on_device()

    def per_batch_transform(self) ->Callable:
        """Defines the transform to be applied on a batch of data on cpu for all stages stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_batch_transform(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        pass

    def train_per_batch_transform(self) ->Callable:
        """Defines the transform to be applied on a batch of data on cpu for the training stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        """
        return self.per_batch_transform()

    def val_per_batch_transform(self) ->Callable:
        """Defines the transform to be applied on a batch of data on cpu for the validating stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_batch_transform(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        return self.per_batch_transform()

    def test_per_batch_transform(self) ->Callable:
        """Defines the transform to be applied on a batch of data on cpu for the testing stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        """
        return self.per_batch_transform()

    def predict_per_batch_transform(self) ->Callable:
        """Defines the transform to be applied on a batch of data on cpu for the predicting stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_batch_transform(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        return self.per_batch_transform()

    def serve_per_batch_transform(self) ->Callable:
        """Defines the transform to be applied on a batch of data on cpu for the serving stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_batch_transform(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        return self.per_batch_transform()

    def per_batch_transform_on_device(self) ->Callable:
        """Defines the transform to be applied on a batch of data on device for all stages stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_batch_transform_on_device(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        pass

    def train_per_batch_transform_on_device(self) ->Callable:
        """Defines the transform to be applied on a batch of data on device for the training stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        """
        return self.per_batch_transform_on_device()

    def val_per_batch_transform_on_device(self) ->Callable:
        """Defines the transform to be applied on a batch of data on device for the validating stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_batch_transform_on_device(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        return self.per_batch_transform_on_device()

    def test_per_batch_transform_on_device(self) ->Callable:
        """Defines the transform to be applied on a batch of data on device for the testing stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        """
        return self.per_batch_transform_on_device()

    def predict_per_batch_transform_on_device(self) ->Callable:
        """Defines the transform to be applied on a batch of data on device for the predicting stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_batch_transform_on_device(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        return self.per_batch_transform_on_device()

    def serve_per_batch_transform_on_device(self) ->Callable:
        """Defines the transform to be applied on a batch of data on device for the serving stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def serve_per_batch_transform_on_device(self) -> Callable:
                    return ApplyToKeys("input", my_func)

        """
        return self.per_batch_transform_on_device()

    def collate(self) ->Callable:
        """Defines the transform to be applied on a list of sample to create a batch for all stages."""
        return default_collate

    def train_collate(self) ->Callable:
        """Defines the transform to be applied on a list of training sample to create a training batch."""
        return self.collate()

    def val_collate(self) ->Callable:
        """Defines the transform to be applied on a list of validating sample to create a validating batch."""
        return self.collate()

    def test_collate(self) ->Callable:
        """Defines the transform to be applied on a list of testing sample to create a testing batch."""
        return self.collate()

    def predict_collate(self) ->Callable:
        """Defines the transform to be applied on a list of predicting sample to create a predicting batch."""
        return self.collate()

    def serve_collate(self) ->Callable:
        """Defines the transform to be applied on a list of serving sample to create a serving batch."""
        return self.collate()

    def _per_sample_transform(self, sample: 'Any', stage: 'RunningStage') ->Any:
        fn = self.current_transform(stage=stage, current_fn='per_sample_transform')
        if isinstance(sample, list):
            return [fn(s) for s in sample]
        return fn(sample)

    def _per_batch_transform(self, batch: 'Any', stage: 'RunningStage') ->Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note:: This option is mutually exclusive with :meth:`per_sample_transform_on_device`, since if both are
        specified, uncollation has to be applied.

        """
        return self.current_transform(stage=stage, current_fn='per_batch_transform')(batch)

    def _collate(self, samples: 'Sequence', stage: 'RunningStage') ->Any:
        """Transform to convert a sequence of samples to a collated batch."""
        return self.current_transform(stage=stage, current_fn='collate')(samples)

    def _per_sample_transform_on_device(self, sample: 'Any', stage: 'RunningStage') ->Any:
        """Transforms to apply to the data before the collation (per-sample basis).

        .. note::     This option is mutually exclusive with :meth:`per_batch_transform`,     since if both are
        specified, uncollation has to be applied. .. note::     This function won't be called within the dataloader
        workers, since to make that happen     each of the workers would have to create it's own CUDA-context which
        would pollute GPU memory (if on GPU).

        """
        fn = self.current_transform(stage=stage, current_fn='per_sample_transform_on_device')
        if isinstance(sample, list):
            return [fn(s) for s in sample]
        return fn(sample)

    def _per_batch_transform_on_device(self, batch: 'Any', stage: 'RunningStage') ->Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::     This function won't be called within the dataloader workers, since to make that happen     each of
        the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).

        """
        return self.current_transform(stage=stage, current_fn='per_batch_transform_on_device')(batch)

    def inject_collate_fn(self, collate_fn: 'Callable'):
        if collate_fn is not default_collate:
            for stage in RunningStage:
                if stage not in [RunningStage.SANITY_CHECKING, RunningStage.TUNING]:
                    self._transform[stage].transforms[InputTransformPlacement.COLLATE.value] = collate_fn

    def _populate_transforms_for_stage(self, running_stage: 'RunningStage'):
        transform, collate_in_worker = self.__check_transforms(transform=self.__resolve_transforms(running_stage))
        self._transform[running_stage] = _InputTransformPerStage(collate_in_worker=collate_in_worker, transforms=transform)

    def __resolve_transforms(self, running_stage: 'RunningStage') ->Optional[Dict[str, Callable]]:
        transforms = {}
        stage = _STAGES_PREFIX[running_stage]
        for transform_name in InputTransformPlacement:
            transform_name = transform_name.value
            method_name = f'{stage}_{transform_name}'
            try:
                fn = getattr(self, method_name)()
            except AttributeError as e:
                raise AttributeError(str(e) + '. Make sure you include a call to super().__init__(...) in your __init__ after setting all attributes.')
            if fn is None:
                continue
            if not callable(fn):
                raise TypeError(f'The hook {method_name} should return a callable.')
            transforms[transform_name] = fn
        return transforms

    def __check_transforms(self, transform: 'Dict[str, Callable]') ->Tuple[Dict[str, Callable], Optional[bool]]:
        is_per_batch_transform_in = 'per_batch_transform' in transform
        is_per_sample_transform_on_device_in = 'per_sample_transform_on_device' in transform
        if is_per_batch_transform_in and is_per_sample_transform_on_device_in:
            raise TypeError(f'{transform}: `per_batch_transform` and `per_sample_transform_on_device` are mutually exclusive.')
        collate_in_worker: 'Optional[bool]' = not is_per_sample_transform_on_device_in
        return transform, collate_in_worker

    @staticmethod
    def _identity(x: 'Any') ->Any:
        return x

    def __str__(self) ->str:
        return f'{self.__class__.__name__}(' + f'transform={self._transform})'


def create_or_configure_input_transform(transform: 'INPUT_TRANSFORM_TYPE', transform_kwargs: 'Optional[Dict]'=None) ->Optional[InputTransform]:
    if not transform_kwargs:
        transform_kwargs = {}
    if isinstance(transform, InputTransform):
        return transform
    if inspect.isclass(transform) and issubclass(transform, InputTransform):
        rank_zero_warn('Please pass an instantiated object of the `InputTransform` class. Passing the Class and keyword arguments separately has been deprecated since v0.8.0 and will be removed in v0.9.0.', stacklevel=8, category=FutureWarning)
        return transform(**transform_kwargs)
    if isinstance(transform, partial):
        return transform(**transform_kwargs)
    if not transform:
        return None
    raise ValueError(f"The format for the transform isn't correct. Found {transform}")


class _InputTransformProcessor:
    """
    This class is used to encapsulate the following functions of an `InputTransform` Object:
    Inside a worker:
        per_sample_transform: Function to transform an individual sample
        collate: Function to merge sample into a batch
        per_batch_transform: Function to transform an individual batch

    Inside main process:
        per_sample_transform_on_device: Function to transform an individual sample
        collate: Function to merge sample into a batch
        per_batch_transform_on_device: Function to transform an individual batch
    """

    def __init__(self, input_transform: 'InputTransform', collate_fn: 'Callable', per_sample_transform: 'Callable', per_batch_transform: 'Callable', stage: 'RunningStage', apply_per_sample_transform: 'bool'=True, on_device: 'bool'=False):
        super().__init__()
        self.input_transform = input_transform
        self.callback = ControlFlow(self.input_transform.callbacks or [])
        self.collate_fn = collate_fn
        self.per_sample_transform = per_sample_transform
        self.per_batch_transform = per_batch_transform
        self.apply_per_sample_transform = apply_per_sample_transform
        self.stage = stage
        self.on_device = on_device

    def __call__(self, samples: 'Sequence[Any]') ->Any:
        if not self.on_device:
            for sample in samples:
                self.callback.on_load_sample(sample, self.stage)
        if self.apply_per_sample_transform:
            list_samples = [samples] if not isinstance(samples, list) else samples
            transformed_samples = [self.per_sample_transform(sample, self.stage) for sample in list_samples]
            for sample in transformed_samples:
                if self.on_device:
                    self.callback.on_per_sample_transform_on_device(sample, self.stage)
                else:
                    self.callback.on_per_sample_transform(sample, self.stage)
            collated_samples = self.collate_fn(transformed_samples, self.stage)
            self.callback.on_collate(collated_samples, self.stage)
        else:
            collated_samples = samples
        transformed_collated_samples = self.per_batch_transform(collated_samples, self.stage)
        if self.on_device:
            self.callback.on_per_batch_transform_on_device(transformed_collated_samples, self.stage)
        else:
            self.callback.on_per_batch_transform(transformed_collated_samples, self.stage)
        return transformed_collated_samples

    def __str__(self) ->str:
        return f'_InputTransformProcessor:\n\t(per_sample_transform): {str(self.per_sample_transform)}\n\t(collate_fn): {str(self.collate_fn)}\n\t(per_batch_transform): {str(self.per_batch_transform)}\n\t(apply_per_sample_transform): {str(self.apply_per_sample_transform)}\n\t(on_device): {str(self.on_device)}\n\t(stage): {str(self.stage)}'


def __make_collates(input_transform: 'InputTransform', on_device: 'bool', collate: 'Callable') ->Tuple[Callable, Callable]:
    """Returns the appropriate collate functions based on whether the transforms happen in a DataLoader worker or on the
    device (main process)."""
    if on_device:
        return input_transform._identity, collate
    return collate, input_transform._identity


def __configure_worker_and_device_collate_fn(running_stage: 'RunningStage', input_transform: 'InputTransform') ->Tuple[Callable, Callable]:
    transform_for_stage: '_InputTransformPerStage' = input_transform._transform[running_stage]
    worker_collate_fn, device_collate_fn = __make_collates(input_transform, not transform_for_stage.collate_in_worker, input_transform._collate)
    return worker_collate_fn, device_collate_fn


def create_worker_input_transform_processor(running_stage: 'RunningStage', input_transform: 'InputTransform') ->_InputTransformProcessor:
    """This utility is used to create the 2 `_InputTransformProcessor` objects which contain the transforms used as the
    DataLoader `collate_fn`."""
    worker_collate_fn, _ = __configure_worker_and_device_collate_fn(running_stage=running_stage, input_transform=input_transform)
    return _InputTransformProcessor(input_transform, worker_collate_fn, input_transform._per_sample_transform, input_transform._per_batch_transform, running_stage)


class DatasetProcessor:
    """The ``DatasetProcessor`` mixin provides hooks for classes which need custom logic for producing the data loaders
    for each running stage given the corresponding dataset."""

    def __init__(self):
        super().__init__()
        self._collate_fn = default_collate
        self._input_transform = None

    @torch.jit.unused
    @property
    def collate_fn(self) ->Callable:
        return self._collate_fn

    @collate_fn.setter
    def collate_fn(self, collate_fn: 'Callable') ->None:
        self._collate_fn = collate_fn

    @torch.jit.unused
    @property
    def input_transform(self) ->Optional[InputTransform]:
        if self._input_transform is not None:
            return create_or_configure_input_transform(self._input_transform)
        return None

    @input_transform.setter
    def input_transform(self, input_transform: 'InputTransform') ->None:
        self._input_transform = input_transform

    def process_train_dataset(self, dataset: 'InputBase', batch_size: 'int', num_workers: 'int'=0, pin_memory: 'bool'=False, shuffle: 'bool'=True, drop_last: 'bool'=True, sampler: 'Optional[Sampler]'=None, persistent_workers: 'bool'=False, input_transform: 'Optional[InputTransform]'=None, trainer: "Optional['flash.Trainer']"=None) ->DataLoader:
        input_transform = input_transform or self.input_transform
        collate_fn = self.collate_fn
        if input_transform is not None:
            input_transform.inject_collate_fn(self.collate_fn)
            collate_fn = create_worker_input_transform_processor(RunningStage.TRAINING, input_transform)
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, drop_last=drop_last, sampler=sampler, collate_fn=collate_fn, persistent_workers=persistent_workers)

    def process_val_dataset(self, dataset: 'InputBase', batch_size: 'int', num_workers: 'int'=0, pin_memory: 'bool'=False, shuffle: 'bool'=False, drop_last: 'bool'=False, sampler: 'Optional[Sampler]'=None, persistent_workers: 'bool'=False, input_transform: 'Optional[InputTransform]'=None, trainer: "Optional['flash.Trainer']"=None) ->DataLoader:
        input_transform = input_transform or self.input_transform
        collate_fn = self.collate_fn
        if input_transform is not None:
            input_transform.inject_collate_fn(self.collate_fn)
            collate_fn = create_worker_input_transform_processor(RunningStage.VALIDATING, input_transform)
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, drop_last=drop_last, sampler=sampler, collate_fn=collate_fn, persistent_workers=persistent_workers)

    def process_test_dataset(self, dataset: 'InputBase', batch_size: 'int', num_workers: 'int'=0, pin_memory: 'bool'=False, shuffle: 'bool'=False, drop_last: 'bool'=False, sampler: 'Optional[Sampler]'=None, persistent_workers: 'bool'=False, input_transform: 'Optional[InputTransform]'=None, trainer: "Optional['flash.Trainer']"=None) ->DataLoader:
        input_transform = input_transform or self.input_transform
        collate_fn = self.collate_fn
        if input_transform is not None:
            input_transform.inject_collate_fn(self.collate_fn)
            collate_fn = create_worker_input_transform_processor(RunningStage.TESTING, input_transform)
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, drop_last=drop_last, sampler=sampler, collate_fn=collate_fn, persistent_workers=persistent_workers)

    def process_predict_dataset(self, dataset: 'InputBase', batch_size: 'int', num_workers: 'int'=0, pin_memory: 'bool'=False, shuffle: 'bool'=False, drop_last: 'bool'=False, sampler: 'Optional[Sampler]'=None, persistent_workers: 'bool'=False, input_transform: 'Optional[InputTransform]'=None, trainer: "Optional['flash.Trainer']"=None) ->DataLoader:
        input_transform = input_transform or self.input_transform
        collate_fn = self.collate_fn
        if input_transform is not None:
            input_transform.inject_collate_fn(self.collate_fn)
            collate_fn = create_worker_input_transform_processor(RunningStage.PREDICTING, input_transform)
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, drop_last=drop_last, sampler=sampler, collate_fn=collate_fn, persistent_workers=persistent_workers)

