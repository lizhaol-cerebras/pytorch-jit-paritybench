
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


from typing import Type


from typing import Union


import torch


from torch import nn


from typing import Any


from typing import Dict


from typing import Optional


from typing import Tuple


import numpy as np


import math


from typing import List


from enum import Enum


from enum import auto


import torchvision


from torch.nn.functional import normalize


import random


from typing import Callable


from abc import abstractmethod


from torch.nn import Parameter


from torch.utils.data import Dataset


from torch.utils.data.distributed import DistributedSampler


from typing import Literal


from scipy.spatial.transform import Rotation


from copy import deepcopy


import numpy.typing as npt


from torch.utils.data.dataloader import DataLoader


import collections


import collections.abc


import re


import torch.utils.data


from torch.cuda.amp.grad_scaler import GradScaler


from torch.nn.parameter import Parameter


from torch.optim import Optimizer


from torch.optim import lr_scheduler


import time


import torch.nn.functional as F


from torch.autograd import Function


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from typing import Set


from torch.autograd import Variable


from math import exp


from collections import defaultdict


import typing


from time import time


from typing import cast


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from matplotlib import cm


from math import floor


from math import log


from typing import NoReturn


from typing import TypeVar


import enum


from torch.utils.tensorboard import SummaryWriter


import matplotlib.pyplot as plt


from torchvision import transforms


from scipy.interpolate import interp1d


from scipy.spatial.transform import Slerp


import torch.multiprocessing as mp


from itertools import product


class IterableWrapper:
    """A helper that will allow an instance of a class to return multiple kinds of iterables bound
    to different functions of that class.

    To use this, take an instance of a class. From that class, pass in the <instance>.<new_iter_function>
    and <instance>.<new_next_function> to the IterableWrapper constructor. By passing in the instance's
    functions instead of just the class's functions, the self argument should automatically be accounted
    for.

    Args:
        new_iter: function that will be called instead as the __iter__() function
        new_next: function that will be called instead as the __next__() function
        length: length of the iterable. If -1, the iterable will be infinite.


    Attributes:
        new_iter: object's pointer to the function we are calling for __iter__()
        new_next: object's pointer to the function we are calling for __next__()
        length: length of the iterable. If -1, the iterable will be infinite.
        i: current index of the iterable.

    """
    i: 'int'

    def __init__(self, new_iter: 'Callable', new_next: 'Callable', length: 'int'=-1):
        self.new_iter = new_iter
        self.new_next = new_next
        self.length = length

    def __next__(self):
        if self.length != -1 and self.i >= self.length:
            raise StopIteration
        self.i += 1
        return self.new_next()

    def __iter__(self):
        self.new_iter()
        self.i = 0
        return self


class TrainingCallback:
    """Callback class used during training.
    The function 'func' with 'args' and 'kwargs' will be called every 'update_every_num_iters' training iterations,
    including at iteration 0. The function is called after the training iteration.

    Args:
        where_to_run: List of locations for when to run callbak (before/after iteration)
        func: The function that will be called.
        update_every_num_iters: How often to call the function `func`.
        iters: Tuple of iteration steps to perform callback
        args: args for the function 'func'.
        kwargs: kwargs for the function 'func'.
    """

    def __init__(self, where_to_run: 'List[TrainingCallbackLocation]', func: 'Callable', update_every_num_iters: 'Optional[int]'=None, iters: 'Optional[Tuple[int, ...]]'=None, args: 'Optional[List]'=None, kwargs: 'Optional[Dict]'=None):
        assert 'step' in signature(func).parameters.keys(), f"'step: int' must be an argument in the callback function 'func': {func.__name__}"
        self.where_to_run = where_to_run
        self.update_every_num_iters = update_every_num_iters
        self.iters = iters
        self.func = func
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}

    def run_callback(self, step: 'int'):
        """Callback to run after training step

        Args:
            step: current iteration step
        """
        if self.update_every_num_iters is not None:
            if step % self.update_every_num_iters == 0:
                self.func(*self.args, **self.kwargs, step=step)
        elif self.iters is not None:
            if step in self.iters:
                self.func(*self.args, **self.kwargs, step=step)

    def run_callback_at_location(self, step: 'int', location: 'TrainingCallbackLocation'):
        """Runs the callback if it's supposed to be run at the given location.

        Args:
            step: current iteration step
            location: when to run callback (before/after iteration)
        """
        if location in self.where_to_run:
            self.run_callback(step=step)


class DataManager(nn.Module):
    """Generic data manager's abstract class

    This version of the data manager is designed be a monolithic way to load data and latents,
    especially since this may contain learnable parameters which need to be shared across the train
    and test data managers. The idea is that we have setup methods for train and eval separately and
    this can be a combined train/eval if you want.

    Usage:
    To get data, use the next_train and next_eval functions.
    This data manager's next_train and next_eval methods will return 2 things:
        1. A Raybundle: This will contain the rays we are sampling, with latents and
            conditionals attached (everything needed at inference)
        2. A "batch" of auxilury information: This will contain the mask, the ground truth
            pixels, etc needed to actually train, score, etc the model

    Rationale:
    Because of this abstraction we've added, we can support more NeRF paradigms beyond the
    vanilla nerf paradigm of single-scene, fixed-images, no-learnt-latents.
    We can now support variable scenes, variable number of images, and arbitrary latents.


    Train Methods:
        setup_train: sets up for being used as train
        iter_train: will be called on __iter__() for the train iterator
        next_train: will be called on __next__() for the training iterator
        get_train_iterable: utility that gets a clean pythonic iterator for your training data

    Eval Methods:
        setup_eval: sets up for being used as eval
        iter_eval: will be called on __iter__() for the eval iterator
        next_eval: will be called on __next__() for the eval iterator
        get_eval_iterable: utility that gets a clean pythonic iterator for your eval data


    Attributes:
        train_count (int): the step number of our train iteration, needs to be incremented manually
        eval_count (int): the step number of our eval iteration, needs to be incremented manually
        train_dataset (Dataset): the dataset for the train dataset
        eval_dataset (Dataset): the dataset for the eval dataset

        Additional attributes specific to each subclass are defined in the setup_train and setup_eval
        functions.

    """
    train_dataset: 'Optional[Dataset]' = None
    eval_dataset: 'Optional[Dataset]' = None
    train_sampler: 'Optional[DistributedSampler]' = None
    eval_sampler: 'Optional[DistributedSampler]' = None

    def __init__(self):
        """Constructor for the DataManager class.

        Subclassed DataManagers will likely need to override this constructor.

        If you aren't manually calling the setup_train and setup_eval functions from an overriden
        constructor, that you call super().__init__() BEFORE you initialize any
        nn.Modules or nn.Parameters, but AFTER you've already set all the attributes you need
        for the setup functions."""
        super().__init__()
        self.train_count = 0
        self.eval_count = 0
        if self.train_dataset and self.test_mode != 'inference':
            self.setup_train()
        if self.eval_dataset and self.test_mode != 'inference':
            self.setup_eval()

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    def iter_train(self):
        """The __iter__ function for the train iterator.

        This only exists to assist the get_train_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.train_count = 0

    def iter_eval(self):
        """The __iter__ function for the eval iterator.

        This only exists to assist the get_eval_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.eval_count = 0

    def get_train_iterable(self, length=-1) ->IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_train and next_train functions
        as __iter__ and __next__ methods respectivley.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_train_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_train, self.next_train, length)

    def get_eval_iterable(self, length=-1) ->IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_eval and next_eval functions
        as __iter__ and __next__ methods respectivley.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_eval_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_eval, self.next_eval, length)

    @abstractmethod
    def setup_train(self):
        """Sets up the data manager for training.

        Here you will define any subclass specific object attributes from the attribute"""
        raise NotImplementedError

    @abstractmethod
    def setup_eval(self):
        """Sets up the data manager for evaluation"""
        raise NotImplementedError

    @abstractmethod
    def next_train(self, step: 'int') ->Tuple:
        """Returns the next batch of data from the train data manager.

        This will be a tuple of all the information that this data manager outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval(self, step: 'int') ->Tuple:
        """Returns the next batch of data from the eval data manager.

        This will be a tuple of all the information that this data manager outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval_image(self, step: 'int') ->Tuple:
        """Returns the next eval image."""
        raise NotImplementedError

    def get_training_callbacks(self, training_callback_attributes: 'TrainingCallbackAttributes') ->List[TrainingCallback]:
        """Returns a list of callbacks to be used during training."""
        return []

    @abstractmethod
    def get_param_groups(self) ->Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}


class BasicImages:
    """This is a very primitive struct for holding images, especially for when these images
    are of different heights / widths.

    The purpose of this is to have a special struct wrapping around a list so that the
    nerfstudio_collate fn and other parts of the code recognise this as a struct to leave alone
    instead of reshaping or concatenating into a single tensor (since this will likely be used
    for cases where we have images of different sizes and shapes).

    This only has one batch dimension and will likely be replaced down the line with some
    TensorDataclass alternative that supports arbitrary batches.
    """

    def __init__(self, images: 'List'):
        assert isinstance(images, List)
        assert not images or isinstance(images[0], torch.Tensor), f'Input should be a list of tensors, not {type(images[0]) if isinstance(images, List) else type(images)}'
        self.images = images

    def to(self, device):
        """Move the images to the given device."""
        assert isinstance(device, torch.device)
        return BasicImages([image for image in self.images])


def get_dict_to_torch(stuff: 'Any', device: 'Union[torch.device, str]'='cpu', exclude: 'Optional[List[str]]'=None):
    """Set everything in the dict to the specified torch device.

    Args:
        stuff: things to convert to torch
        device: machine to put the "stuff" on
        exclude: list of keys to skip over transferring to device
    """
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            if exclude and k in exclude:
                stuff[k] = v
            else:
                stuff[k] = get_dict_to_torch(v, device)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff
    return stuff


class CameraType(Enum):
    """Supported camera types."""
    PERSPECTIVE = auto()
    FISHEYE = auto()
    EQUIRECTANGULAR = auto()


TensorDataclassT = TypeVar('TensorDataclassT', bound='TensorDataclass')


class TensorDataclass:
    """@dataclass of tensors with the same size batch. Allows indexing and standard tensor ops.
    Fields that are not Tensors will not be batched unless they are also a TensorDataclass.
    Any fields that are dictionaries will have their Tensors or TensorDataclasses batched, and
    dictionaries will have their tensors or TensorDataclasses considered in the initial broadcast.
    Tensor fields must have at least 1 dimension, meaning that you must convert a field like torch.Tensor(1)
    to torch.Tensor([1])

    Example:

    .. code-block:: python

        @dataclass
        class TestTensorDataclass(TensorDataclass):
            a: torch.Tensor
            b: torch.Tensor
            c: torch.Tensor = None

        # Create a new tensor dataclass with batch size of [2,3,4]
        test = TestTensorDataclass(a=torch.ones((2, 3, 4, 2)), b=torch.ones((4, 3)))

        test.shape  # [2, 3, 4]
        test.a.shape  # [2, 3, 4, 2]
        test.b.shape  # [2, 3, 4, 3]

        test.reshape((6,4)).shape  # [6, 4]
        test.flatten().shape  # [24,]

        test[..., 0].shape  # [2, 3]
        test[:, 0, :].shape  # [2, 4]
    """
    _shape: 'tuple'
    _field_custom_dimensions: 'Dict[str, int]' = {}

    def __post_init__(self) ->None:
        """Finishes setting up the TensorDataclass

        This will 1) find the broadcasted shape and 2) broadcast all fields to this shape 3)
        set _shape to be the broadcasted shape.
        """
        if self._field_custom_dimensions is not None:
            for k, v in self._field_custom_dimensions.items():
                assert isinstance(v, int) and v > 1, f'Custom dimensions must be an integer greater than 1, since 1 is the default, received {k}: {v}'
        if not dataclasses.is_dataclass(self):
            raise TypeError('TensorDataclass must be a dataclass')
        batch_shapes = self._get_dict_batch_shapes({f.name: self.__getattribute__(f.name) for f in dataclasses.fields(self)})
        if len(batch_shapes) == 0:
            raise ValueError('TensorDataclass must have at least one tensor')
        batch_shape = torch.broadcast_shapes(*batch_shapes)
        broadcasted_fields = self._broadcast_dict_fields({f.name: self.__getattribute__(f.name) for f in dataclasses.fields(self)}, batch_shape)
        for f, v in broadcasted_fields.items():
            self.__setattr__(f, v)
        self.__setattr__('_shape', batch_shape)

    def _get_dict_batch_shapes(self, dict_: 'Dict') ->List:
        """Returns batch shapes of all tensors in a dictionary

        Args:
            dict_: The dictionary to get the batch shapes of.

        Returns:
            The batch shapes of all tensors in the dictionary.
        """
        batch_shapes = []
        for k, v in dict_.items():
            if isinstance(v, torch.Tensor):
                if isinstance(self._field_custom_dimensions, dict) and k in self._field_custom_dimensions:
                    batch_shapes.append(v.shape[:-self._field_custom_dimensions[k]])
                else:
                    batch_shapes.append(v.shape[:-1])
            elif isinstance(v, TensorDataclass):
                batch_shapes.append(v.shape)
            elif isinstance(v, Dict):
                batch_shapes.extend(self._get_dict_batch_shapes(v))
        return batch_shapes

    def _broadcast_dict_fields(self, dict_: 'Dict', batch_shape) ->Dict:
        """Broadcasts all tensors in a dictionary according to batch_shape

        Args:
            dict_: The dictionary to broadcast.

        Returns:
            The broadcasted dictionary.
        """
        new_dict = {}
        for k, v in dict_.items():
            if isinstance(v, torch.Tensor):
                if isinstance(self._field_custom_dimensions, dict) and k in self._field_custom_dimensions:
                    new_dict[k] = v.broadcast_to((*batch_shape, *v.shape[-self._field_custom_dimensions[k]:]))
                else:
                    new_dict[k] = v.broadcast_to((*batch_shape, v.shape[-1]))
            elif isinstance(v, TensorDataclass):
                new_dict[k] = v.broadcast_to(batch_shape)
            elif isinstance(v, Dict):
                new_dict[k] = self._broadcast_dict_fields(v, batch_shape)
        return new_dict

    def __getitem__(self: 'TensorDataclassT', indices) ->TensorDataclassT:
        if isinstance(indices, torch.Tensor):
            return self._apply_fn_to_fields(lambda x: x[indices])
        if isinstance(indices, (int, slice, type(Ellipsis))):
            indices = indices,
        assert isinstance(indices, tuple)
        tensor_fn = lambda x: x[indices + (slice(None),)]
        dataclass_fn = lambda x: x[indices]

        def custom_tensor_dims_fn(k, v):
            custom_dims = self._field_custom_dimensions[k]
            return v[indices + (slice(None),) * custom_dims]
        return self._apply_fn_to_fields(tensor_fn, dataclass_fn, custom_tensor_dims_fn=custom_tensor_dims_fn)

    def __setitem__(self, indices, value) ->NoReturn:
        raise RuntimeError('Index assignment is not supported for TensorDataclass')

    def __len__(self) ->int:
        if len(self._shape) == 0:
            raise TypeError('len() of a 0-d tensor')
        return self.shape[0]

    def __bool__(self) ->bool:
        if len(self) == 0:
            raise ValueError(f'The truth value of {self.__class__.__name__} when `len(x) == 0` is ambiguous. Use `len(x)` or `x is not None`.')
        return True

    @property
    def shape(self) ->Tuple[int, ...]:
        """Returns the batch shape of the tensor dataclass."""
        return self._shape

    @property
    def size(self) ->int:
        """Returns the number of elements in the tensor dataclass batch dimension."""
        if len(self._shape) == 0:
            return 1
        return int(np.prod(self._shape))

    @property
    def ndim(self) ->int:
        """Returns the number of dimensions of the tensor dataclass."""
        return len(self._shape)

    def reshape(self: 'TensorDataclassT', shape: 'Tuple[int, ...]') ->TensorDataclassT:
        """Returns a new TensorDataclass with the same data but with a new shape.

        This should deepcopy as well.

        Args:
            shape: The new shape of the tensor dataclass.

        Returns:
            A new TensorDataclass with the same data but with a new shape.
        """
        if isinstance(shape, int):
            shape = shape,
        tensor_fn = lambda x: x.reshape((*shape, x.shape[-1]))
        dataclass_fn = lambda x: x.reshape(shape)

        def custom_tensor_dims_fn(k, v):
            custom_dims = self._field_custom_dimensions[k]
            return v.reshape((*shape, *v.shape[-custom_dims:]))
        return self._apply_fn_to_fields(tensor_fn, dataclass_fn, custom_tensor_dims_fn=custom_tensor_dims_fn)

    def flatten(self: 'TensorDataclassT') ->TensorDataclassT:
        """Returns a new TensorDataclass with flattened batch dimensions

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but with a new shape.
        """
        return self.reshape((-1,))

    def broadcast_to(self: 'TensorDataclassT', shape: 'Union[torch.Size, Tuple[int, ...]]') ->TensorDataclassT:
        """Returns a new TensorDataclass broadcast to new shape.

        Changes to the original tensor dataclass should effect the returned tensor dataclass,
        meaning it is NOT a deepcopy, and they are still linked.

        Args:
            shape: The new shape of the tensor dataclass.

        Returns:
            A new TensorDataclass with the same data but with a new shape.
        """

        def custom_tensor_dims_fn(k, v):
            custom_dims = self._field_custom_dimensions[k]
            return v.broadcast_to((*shape, *v.shape[-custom_dims:]))
        return self._apply_fn_to_fields(lambda x: x.broadcast_to((*shape, x.shape[-1])), custom_tensor_dims_fn=custom_tensor_dims_fn)

    def to(self: 'TensorDataclassT', device) ->TensorDataclassT:
        """Returns a new TensorDataclass with the same data but on the specified device.

        Args:
            device: The device to place the tensor dataclass.

        Returns:
            A new TensorDataclass with the same data but on the specified device.
        """
        return self._apply_fn_to_fields(lambda x: x)

    def _apply_fn_to_fields(self: 'TensorDataclassT', fn: 'Callable', dataclass_fn: 'Optional[Callable]'=None, custom_tensor_dims_fn: 'Optional[Callable]'=None) ->TensorDataclassT:
        """Applies a function to all fields of the tensor dataclass.

        TODO: Someone needs to make a high level design choice for whether not not we want this
        to apply the function to any fields in arbitray superclasses. This is an edge case until we
        upgrade to python 3.10 and dataclasses can actually be subclassed with vanilla python and no
        janking, but if people try to jank some subclasses that are grandchildren of TensorDataclass
        (imagine if someone tries to subclass the RayBundle) this will matter even before upgrading
        to 3.10 . Currently we aren't going to be able to work properly for grandchildren, but you
        want to use self.__dict__ if you want to apply this to grandchildren instead of our dictionary
        from dataclasses.fields(self) as we do below and in other places.

        Args:
            fn: The function to apply to tensor fields.
            dataclass_fn: The function to apply to TensorDataclass fields.

        Returns:
            A new TensorDataclass with the same data but with a new shape.
        """
        new_fields = self._apply_fn_to_dict({f.name: self.__getattribute__(f.name) for f in dataclasses.fields(self)}, fn, dataclass_fn, custom_tensor_dims_fn)
        return dataclasses.replace(self, **new_fields)

    def _apply_fn_to_dict(self, dict_: 'Dict', fn: 'Callable', dataclass_fn: 'Optional[Callable]'=None, custom_tensor_dims_fn: 'Optional[Callable]'=None) ->Dict:
        """A helper function for _apply_fn_to_fields, applying a function to all fields of dict_

        Args:
            dict_: The dictionary to apply the function to.
            fn: The function to apply to tensor fields.
            dataclass_fn: The function to apply to TensorDataclass fields.

        Returns:
            A new dictionary with the same data but with a new shape. Will deep copy"""
        field_names = dict_.keys()
        new_dict = {}
        for f in field_names:
            v = dict_[f]
            if v is not None:
                if isinstance(v, TensorDataclass) and dataclass_fn is not None:
                    new_dict[f] = dataclass_fn(v)
                elif isinstance(v, torch.Tensor) and isinstance(self._field_custom_dimensions, dict) and f in self._field_custom_dimensions and custom_tensor_dims_fn is not None:
                    new_dict[f] = custom_tensor_dims_fn(f, v)
                elif isinstance(v, (torch.Tensor, TensorDataclass)):
                    new_dict[f] = fn(v)
                elif isinstance(v, Dict):
                    new_dict[f] = self._apply_fn_to_dict(v, fn, dataclass_fn)
                else:
                    new_dict[f] = deepcopy(v)
        return new_dict


NERFSTUDIO_COLLATE_ERR_MSG_FORMAT = 'default_collate: batch must contain tensors, numpy arrays, numbers, dicts, lists or anything in {}; found {}'


np_str_obj_array_pattern = re.compile('[SaUO]')


def nerfstudio_collate(batch, extra_mappings: 'Union[Dict[type, Callable], None]'=None):
    """
    This is the default pytorch collate function, but with support for nerfstudio types. All documentation
    below is copied straight over from pytorch's default_collate function, python version 3.8.13,
    pytorch version '1.12.1+cu113'. Custom nerfstudio types are accounted for at the end, and extra
    mappings can be passed in to handle custom types. These mappings are from types: callable (types
    being like int or float or the return value of type(3.), etc). The only code before we parse for custom types that
    was changed from default pytorch was the addition of the extra_mappings argument, a find and replace operation
    from default_collate to nerfstudio_collate, and the addition of the nerfstudio_collate_err_msg_format variable.


    Function that takes in a batch of data and puts the elements within the batch
    into a tensor with an additional outer dimension - batch size. The exact output type can be
    a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
    Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

    Here is the general input type (based on the type of the element within the batch) to output type mapping:

        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, nerfstudio_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[nerfstudio_collate([V1_1, V1_2, ...]),
          nerfstudio_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[nerfstudio_collate([V1_1, V1_2, ...]),
          nerfstudio_collate([V2_1, V2_2, ...]), ...]`

    Args:
        batch: a single batch to be collated

    Examples:
        >>> # Example with a batch of `int`s:
        >>> nerfstudio_collate([0, 1, 2, 3])
        tensor([0, 1, 2, 3])
        >>> # Example with a batch of `str`s:
        >>> nerfstudio_collate(['a', 'b', 'c'])
        ['a', 'b', 'c']
        >>> # Example with `Map` inside the batch:
        >>> nerfstudio_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
        {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
        >>> # Example with `NamedTuple` inside the batch:
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> nerfstudio_collate([Point(0, 0), Point(1, 1)])
        Point(x=tensor([0, 1]), y=tensor([0, 1]))
        >>> # Example with `Tuple` inside the batch:
        >>> nerfstudio_collate([(0, 1), (2, 3)])
        [tensor([0, 2]), tensor([1, 3])]
        >>> # Example with `List` inside the batch:
        >>> nerfstudio_collate([[0, 1], [2, 3]])
        [tensor([0, 2]), tensor([1, 3])]
    """
    if extra_mappings is None:
        extra_mappings = {}
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(NERFSTUDIO_COLLATE_ERR_MSG_FORMAT.format(elem.dtype))
            return nerfstudio_collate([torch.as_tensor(b) for b in batch], extra_mappings=extra_mappings)
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: nerfstudio_collate([d[key] for d in batch], extra_mappings=extra_mappings) for key in elem})
        except TypeError:
            return {key: nerfstudio_collate([d[key] for d in batch], extra_mappings=extra_mappings) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return elem_type(*(nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))
        if isinstance(elem, tuple):
            return [nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in transposed]
        else:
            try:
                return elem_type([nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in transposed])
            except TypeError:
                return [nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in transposed]
    elif isinstance(elem, Cameras):
        assert all(isinstance(cam, Cameras) for cam in batch)
        assert all(cam.distortion_params is None for cam in batch) or all(cam.distortion_params is not None for cam in batch), 'All cameras must have distortion parameters or none of them should have distortion parameters.            Generalized batching will be supported in the future.'
        if elem.shape == ():
            op = torch.stack
        else:
            op = torch.cat
        return Cameras(op([cameras.camera_to_worlds for cameras in batch], dim=0), op([cameras.fx for cameras in batch], dim=0), op([cameras.fy for cameras in batch], dim=0), op([cameras.cx for cameras in batch], dim=0), op([cameras.cy for cameras in batch], dim=0), height=op([cameras.height for cameras in batch], dim=0), width=op([cameras.width for cameras in batch], dim=0), distortion_params=op([(cameras.distortion_params if cameras.distortion_params is not None else torch.zeros_like(cameras.distortion_params)) for cameras in batch], dim=0), camera_type=op([cameras.camera_type for cameras in batch], dim=0), times=torch.stack([(cameras.times if cameras.times is not None else -torch.ones_like(cameras.times)) for cameras in batch], dim=0))
    elif isinstance(elem, BasicImages):
        assert all(isinstance(elem, BasicImages) for elem in batch)
        all_images = []
        for images in batch:
            all_images.extend(images.images)
        return BasicImages(all_images)
    for type_key in extra_mappings:
        if isinstance(elem, type_key):
            return extra_mappings[type_key](batch)
    raise TypeError(NERFSTUDIO_COLLATE_ERR_MSG_FORMAT.format(elem_type))


class CacheDataloader(DataLoader):
    """Collated image dataset that implements caching of default-pytorch-collatable data.
    Creates batches of the InputDataset return type.

    Args:
        dataset: Dataset to sample from.
        num_samples_to_collate: How many images to sample rays for each batch. -1 for all images.
        num_times_to_repeat_images: How often to collate new images. -1 to never pick new images.
        device: Device to perform computation.
        collate_fn: The function we will use to collate our training data
    """

    def __init__(self, dataset: 'Dataset', num_images_to_sample_from: 'int'=-1, num_times_to_repeat_images: 'int'=-1, device: 'Union[torch.device, str]'='cpu', collate_fn=nerfstudio_collate, **kwargs):
        self.dataset = dataset
        super().__init__(dataset=dataset, **kwargs)
        self.num_times_to_repeat_images = num_times_to_repeat_images
        self.cache_all_images = num_images_to_sample_from == -1 or num_images_to_sample_from >= len(self.dataset)
        self.num_images_to_sample_from = len(self.dataset) if self.cache_all_images else num_images_to_sample_from
        self.device = device
        self.collate_fn = collate_fn
        self.num_workers = kwargs.get('num_workers', 0)
        self.num_repeated = self.num_times_to_repeat_images
        self.first_time = True
        self.cached_collated_batch = None
        if self.cache_all_images:
            CONSOLE.print(f'Caching all {len(self.dataset)} images.')
            if len(self.dataset) > 500:
                CONSOLE.print('[bold yellow]Warning: If you run out of memory, try reducing the number of images to sample from.')
            self.cached_collated_batch = self._get_collated_batch()
        elif self.num_times_to_repeat_images == -1:
            CONSOLE.print(f'Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, without resampling.')
        else:
            CONSOLE.print(f'Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, resampling every {self.num_times_to_repeat_images} iters.')

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""
        indices = random.sample(range(len(self.dataset)), k=self.num_images_to_sample_from)
        batch_list = []
        results = []
        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)
            for res in track(results, description='Loading data batch', transient=True, disable=self.num_images_to_sample_from == 1):
                batch_list.append(res.result())
        return batch_list

    def _get_collated_batch(self):
        """Returns a collated batch."""
        batch_list = self._get_batch_list()
        collated_batch = self.collate_fn(batch_list)
        collated_batch = get_dict_to_torch(collated_batch, device=self.device, exclude=['image'])
        return collated_batch

    def __iter__(self):
        while True:
            if self.cache_all_images:
                collated_batch = self.cached_collated_batch
            elif self.first_time or self.num_times_to_repeat_images != -1 and self.num_repeated >= self.num_times_to_repeat_images:
                self.num_repeated = 0
                collated_batch = self._get_collated_batch()
                self.cached_collated_batch = collated_batch if self.num_times_to_repeat_images != 0 else None
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
                self.num_repeated += 1
            yield collated_batch


def collate_image_dataset_batch(batch: 'Dict', num_rays_per_batch: 'int', keep_full_image: 'bool'=False):
    """
    Operates on a batch of images and samples pixels to use for generating rays.
    Returns a collated batch which is input to the Graph.
    It will sample only within the valid 'mask' if it's specified.

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """
    device = batch['image'].device
    num_images, image_height, image_width, _ = batch['image'].shape
    if 'mask' in batch:
        nonzero_indices = torch.nonzero(batch['mask'][..., 0], as_tuple=False)
        chosen_indices = random.sample(range(len(nonzero_indices)), k=num_rays_per_batch)
        indices = nonzero_indices[chosen_indices]
    else:
        indices = torch.floor(torch.rand((num_rays_per_batch, 3), device=device) * torch.tensor([num_images, image_height, image_width], device=device)).long()
    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    collated_batch = {key: value[c, y, x] for key, value in batch.items() if key not in ('image_idx', 'src_imgs', 'src_idxs', 'sparse_sfm_points') and value is not None}
    assert collated_batch['image'].shape == (num_rays_per_batch, 3), collated_batch['image'].shape
    if 'sparse_sfm_points' in batch:
        collated_batch['sparse_sfm_points'] = batch['sparse_sfm_points'].images[c[0]]
    indices[:, 0] = batch['image_idx'][c]
    collated_batch['indices'] = indices
    if keep_full_image:
        collated_batch['full_image'] = batch['image']
    return collated_batch


def collate_image_dataset_batch_list(batch: 'Dict', num_rays_per_batch: 'int', keep_full_image: 'bool'=False):
    """
    Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
    a list.

    We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
    The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
    since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """
    device = batch['image'][0].device
    num_images = len(batch['image'])
    all_indices = []
    all_images = []
    all_fg_masks = []
    if 'mask' in batch:
        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
            nonzero_indices = batch['mask'][i]
            chosen_indices = random.sample(range(len(nonzero_indices)), k=num_rays_in_batch)
            indices = nonzero_indices[chosen_indices]
            indices = torch.cat([torch.full((num_rays_in_batch, 1), i, device=device), indices], dim=-1)
            all_indices.append(indices)
            all_images.append(batch['image'][i][indices[:, 1], indices[:, 2]])
            if 'fg_mask' in batch:
                all_fg_masks.append(batch['fg_mask'][i][indices[:, 1], indices[:, 2]])
    else:
        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            image_height, image_width, _ = batch['image'][i].shape
            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
            indices = torch.floor(torch.rand((num_rays_in_batch, 3), device=device) * torch.tensor([1, image_height, image_width], device=device)).long()
            indices[:, 0] = i
            all_indices.append(indices)
            all_images.append(batch['image'][i][indices[:, 1], indices[:, 2]])
            if 'fg_mask' in batch:
                all_fg_masks.append(batch['fg_mask'][i][indices[:, 1], indices[:, 2]])
    indices = torch.cat(all_indices, dim=0)
    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    collated_batch = {key: value[c, y, x] for key, value in batch.items() if key != 'image_idx' and key != 'image' and key != 'mask' and key != 'fg_mask' and key != 'sparse_pts' and value is not None}
    collated_batch['image'] = torch.cat(all_images, dim=0)
    if len(all_fg_masks) > 0:
        collated_batch['fg_mask'] = torch.cat(all_fg_masks, dim=0)
    if 'sparse_pts' in batch:
        rand_idx = random.randint(0, num_images - 1)
        collated_batch['sparse_pts'] = batch['sparse_pts'][rand_idx]
    assert collated_batch['image'].shape == (num_rays_per_batch, 3), collated_batch['image'].shape
    indices[:, 0] = batch['image_idx'][c]
    collated_batch['indices'] = indices
    if keep_full_image:
        collated_batch['full_image'] = batch['image']
    return collated_batch


class PixelSampler:
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    def __init__(self, num_rays_per_batch: 'int', keep_full_image: 'bool'=False) ->None:
        self.num_rays_per_batch = num_rays_per_batch
        self.keep_full_image = keep_full_image

    def set_num_rays_per_batch(self, num_rays_per_batch: 'int'):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample(self, image_batch: 'Dict'):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch['image'], list):
            image_batch = dict(image_batch.items())
            pixel_batch = collate_image_dataset_batch_list(image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image)
        elif isinstance(image_batch['image'], BasicImages):
            image_batch = dict(image_batch.items())
            image_batch['image'] = image_batch['image'].images
            if 'mask' in image_batch:
                image_batch['mask'] = image_batch['mask'].images
            if 'fg_mask' in image_batch:
                image_batch['fg_mask'] = image_batch['fg_mask'].images
            if 'sparse_pts' in image_batch:
                image_batch['sparse_pts'] = image_batch['sparse_pts'].images
            pixel_batch = collate_image_dataset_batch_list(image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image)
        elif isinstance(image_batch['image'], torch.Tensor):
            pixel_batch = collate_image_dataset_batch(image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image)
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch


def collate_image_dataset_batch_equirectangular(batch: 'Dict', num_rays_per_batch: 'int', keep_full_image: 'bool'=False):
    """
    Operates on a batch of equirectangular images and samples pixels to use for
    generating rays. Rays will be generated uniformly on the sphere.
    Returns a collated batch which is input to the Graph.
    It will sample only within the valid 'mask' if it's specified.

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """
    device = batch['image'].device
    num_images, image_height, image_width, _ = batch['image'].shape
    if 'mask' in batch:
        raise NotImplementedError('Masking not implemented for equirectangular images.')
    num_images_rand = torch.rand(num_rays_per_batch, device=device)
    phi_rand = torch.acos(1 - 2 * torch.rand(num_rays_per_batch, device=device)) / torch.pi
    theta_rand = torch.rand(num_rays_per_batch, device=device)
    indices = torch.floor(torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1) * torch.tensor([num_images, image_height, image_width], device=device)).long()
    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    collated_batch = {key: value[c, y, x] for key, value in batch.items() if key != 'image_idx' and value is not None}
    assert collated_batch['image'].shape == (num_rays_per_batch, 3), collated_batch['image'].shape
    indices[:, 0] = batch['image_idx'][c]
    collated_batch['indices'] = indices
    if keep_full_image:
        collated_batch['full_image'] = batch['image']
    return collated_batch


class EquirectangularPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Assumes images are
    equirectangular and the sampling is done uniformly on the sphere.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    def sample(self, image_batch: 'Dict'):
        pixel_batch = collate_image_dataset_batch_equirectangular(image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image)
        return pixel_batch


BaseImage = collections.namedtuple('Image', ['id', 'qvec', 'tvec', 'camera_id', 'name', 'xys', 'point3D_ids'])


class Image(BaseImage):

    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def get_image_mask_tensor_from_path(filepath: 'Path', scale_factor: 'float'=1.0) ->torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = int(width * scale_factor), int(height * scale_factor)
        pil_mask = pil_mask.resize(newsize, resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(pil_mask)).unsqueeze(-1).bool()
    return mask_tensor


def expected_sin(x_means: 'torch.Tensor', x_vars: 'torch.Tensor') ->torch.Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        torch.Tensor: The expected value of sin.
    """
    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)


def print_tcnn_speed_warning(method_name: 'str'):
    """Prints a warning about the speed of the TCNN."""
    CONSOLE.line()
    CONSOLE.print(f'[bold yellow]WARNING: Using a slow implementation of {method_name}. ')
    CONSOLE.print('[bold yellow]:person_running: :person_running: ' + 'Install tcnn for speedups :person_running: :person_running:')
    CONSOLE.print('[yellow]pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch')
    CONSOLE.line()


class FieldHeadNames(Enum):
    """Possible field outputs"""
    RGB = 'rgb'
    SH = 'sh'
    DENSITY = 'density'
    NORMALS = 'normals'
    PRED_NORMALS = 'pred_normals'
    UNCERTAINTY = 'uncertainty'
    TRANSIENT_RGB = 'transient_rgb'
    TRANSIENT_DENSITY = 'transient_density'
    SEMANTICS = 'semantics'
    NORMAL = 'normal'
    SDF = 'sdf'
    ALPHA = 'alpha'
    GRADIENT = 'gradient'
    OCCUPANCY = 'occupancy'


class _TruncExp(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp = _TruncExp.apply


def get_normalized_directions(directions: "TensorType['bs':..., 3]"):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


class SingleVarianceNetwork(nn.Module):
    """Variance network in NeuS

    Args:
        nn (_type_): init value in NeuS variance network
    """

    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, x):
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)

    def get_variance(self):
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-06, 1000000.0)


def reduction_batch_based(image_loss, M):
    divisor = torch.sum(M)
    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))
    return reduction(image_loss, 2 * M)


def reduction_image_based(image_loss, M):
    valid = M.nonzero()
    image_loss[valid] = image_loss[valid] / M[valid]
    return torch.mean(image_loss)


class MiDaSMSELoss(nn.Module):

    def __init__(self, reduction='batch-based'):
        super().__init__()
        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    diff = prediction - target
    diff = torch.mul(mask, diff)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)
    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
    return reduction(image_loss, M)


class GradientLoss(nn.Module):

    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()
        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based
        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0
        for scale in range(self.__scales):
            step = pow(2, scale)
            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step], mask[:, ::step, ::step], reduction=self.__reduction)
        return total


def compute_scale_and_shift(prediction, target, mask):
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)
    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    return x_0, x_1


class ScaleAndShiftInvariantLoss(nn.Module):

    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()
        self.__data_loss = MiDaSMSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha
        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)
        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi
    prediction_ssi = property(__get_prediction_ssi)


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self, patch_size):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(patch_size, 1)
        self.mu_y_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_x_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_y_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_xy_pool = nn.AvgPool2d(patch_size, 1)
        self.refl = nn.ReflectionPad2d(patch_size // 2)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class NCC(nn.Module):
    """Layer to compute the normalization cross correlation (NCC) of patches"""

    def __init__(self, patch_size: 'int'=11, min_patch_variance: 'float'=0.01):
        super(NCC, self).__init__()
        self.patch_size = patch_size
        self.min_patch_variance = min_patch_variance

    def forward(self, x, y):
        x = torch.mean(x, dim=1)
        y = torch.mean(y, dim=1)
        x_mean = torch.mean(x, dim=(1, 2), keepdim=True)
        y_mean = torch.mean(y, dim=(1, 2), keepdim=True)
        x_normalized = x - x_mean
        y_normalized = y - y_mean
        norm = torch.sum(x_normalized * y_normalized, dim=(1, 2))
        var = torch.square(x_normalized).sum(dim=(1, 2)) * torch.square(y_normalized).sum(dim=(1, 2))
        denom = torch.sqrt(var + 1e-06)
        ncc = norm / (denom + 1e-06)
        not_valid = (torch.square(x_normalized).sum(dim=(1, 2)) < self.min_patch_variance) | (torch.square(y_normalized).sum(dim=(1, 2)) < self.min_patch_variance)
        ncc[not_valid] = 1.0
        score = 1 - ncc.clip(-1.0, 1.0)
        return score[:, None, None, None]


class MultiViewLoss(nn.Module):
    """compute multi-view consistency loss"""

    def __init__(self, patch_size: 'int'=11, topk: 'int'=4, min_patch_variance: 'float'=0.01):
        super(MultiViewLoss, self).__init__()
        self.patch_size = patch_size
        self.topk = topk
        self.min_patch_variance = min_patch_variance
        self.ssim = NCC(patch_size=patch_size, min_patch_variance=min_patch_variance)
        self.iter = 0

    def forward(self, patches: 'torch.Tensor', valid: 'torch.Tensor'):
        """take the mim

        Args:
            patches (torch.Tensor): _description_
            valid (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        num_imgs, num_rays, _, num_channels = patches.shape
        if num_rays <= 0:
            return torch.tensor(0.0)
        ref_patches = patches[:1, ...].reshape(1, num_rays, self.patch_size, self.patch_size, num_channels).expand(num_imgs - 1, num_rays, self.patch_size, self.patch_size, num_channels).reshape(-1, self.patch_size, self.patch_size, num_channels).permute(0, 3, 1, 2)
        src_patches = patches[1:, ...].reshape(num_imgs - 1, num_rays, self.patch_size, self.patch_size, num_channels).reshape(-1, self.patch_size, self.patch_size, num_channels).permute(0, 3, 1, 2)
        src_patches_valid = valid[1:, ...].reshape(num_imgs - 1, num_rays, self.patch_size, self.patch_size, 1).reshape(-1, self.patch_size, self.patch_size, 1).permute(0, 3, 1, 2)
        ssim = self.ssim(ref_patches.detach(), src_patches)
        ssim = torch.mean(ssim, dim=(1, 2, 3))
        ssim = ssim.reshape(num_imgs - 1, num_rays)
        ssim_valid = src_patches_valid.reshape(-1, self.patch_size * self.patch_size).all(dim=-1).reshape(num_imgs - 1, num_rays)
        min_ssim, idx = torch.topk(ssim, k=self.topk, largest=False, dim=0, sorted=True)
        min_ssim_valid = ssim_valid[idx, torch.arange(num_rays)[None].expand_as(idx)]
        min_ssim[torch.logical_not(min_ssim_valid)] = 0.0
        if False:
            import numpy as np
            vis_patch_num = num_rays
            K = min(100, vis_patch_num)
            image = patches[:, :vis_patch_num, :, :].reshape(-1, vis_patch_num, self.patch_size, self.patch_size, 3).permute(1, 2, 0, 3, 4).reshape(vis_patch_num * self.patch_size, -1, 3)
            src_patches_reshaped = src_patches.reshape(num_imgs - 1, num_rays, 3, self.patch_size, self.patch_size).permute(1, 0, 3, 4, 2)
            idx = idx.permute(1, 0)
            selected_patch = src_patches_reshaped[torch.arange(num_rays)[:, None].expand(idx.shape), idx].permute(0, 2, 1, 3, 4).reshape(num_rays, self.patch_size, self.topk * self.patch_size, 3)[:vis_patch_num].reshape(-1, self.topk * self.patch_size, 3)
            src_patches_valid_reshaped = src_patches_valid.reshape(num_imgs - 1, num_rays, 1, self.patch_size, self.patch_size).permute(1, 0, 3, 4, 2)
            selected_patch_valid = src_patches_valid_reshaped[torch.arange(num_rays)[:, None].expand(idx.shape), idx].permute(0, 2, 1, 3, 4).reshape(num_rays, self.patch_size, self.topk * self.patch_size, 1)[:vis_patch_num].reshape(-1, self.topk * self.patch_size, 1)
            selected_patch_valid = selected_patch_valid.expand_as(selected_patch).float()
            image = torch.cat([selected_patch_valid, selected_patch, image], dim=1)
            image = image.reshape(num_rays, self.patch_size, -1, 3)
            _, idx2 = torch.topk(torch.sum(min_ssim, dim=0) / (min_ssim_valid.float().sum(dim=0) + 1e-06), k=K, largest=True, dim=0, sorted=True)
            image = image[idx2].reshape(K * self.patch_size, -1, 3)
            cv2.imwrite(f'vis/{self.iter}.png', (image.detach().cpu().numpy() * 255).astype(np.uint8)[..., ::-1])
            self.iter += 1
            if self.iter == 9:
                breakpoint()
        return torch.sum(min_ssim) / (min_ssim_valid.float().sum() + 1e-06)


class SensorDepthLoss(nn.Module):
    """Sensor Depth loss"""

    def __init__(self, truncation: 'float'):
        super(SensorDepthLoss, self).__init__()
        self.truncation = truncation

    def forward(self, batch, outputs):
        """take the mim

        Args:
            batch (Dict): inputs
            outputs (Dict): outputs data from surface model

        Returns:
            l1_loss: l1 loss
            freespace_loss: free space loss
            sdf_loss: sdf loss
        """
        depth_pred = outputs['depth']
        depth_gt = batch['sensor_depth'][..., None]
        valid_gt_mask = depth_gt > 0.0
        l1_loss = torch.sum(valid_gt_mask * torch.abs(depth_gt - depth_pred)) / (valid_gt_mask.sum() + 1e-06)
        ray_samples = outputs['ray_samples']
        filed_outputs = outputs['field_outputs']
        pred_sdf = filed_outputs[FieldHeadNames.SDF][..., 0]
        directions_norm = outputs['directions_norm']
        z_vals = ray_samples.frustums.starts[..., 0] / directions_norm
        truncation = self.truncation
        front_mask = valid_gt_mask & (z_vals < depth_gt - truncation)
        back_mask = valid_gt_mask & (z_vals > depth_gt + truncation)
        sdf_mask = valid_gt_mask & ~front_mask & ~back_mask
        num_fs_samples = front_mask.sum()
        num_sdf_samples = sdf_mask.sum()
        num_samples = num_fs_samples + num_sdf_samples + 1e-06
        fs_weight = 1.0 - num_fs_samples / num_samples
        sdf_weight = 1.0 - num_sdf_samples / num_samples
        free_space_loss = torch.mean((F.relu(truncation - pred_sdf) * front_mask) ** 2) * fs_weight
        sdf_loss = torch.mean((z_vals + pred_sdf - depth_gt) ** 2 * sdf_mask) * sdf_weight
        return l1_loss, free_space_loss, sdf_loss


class S3IM(torch.nn.Module):

    def __init__(self, s3im_kernel_size=4, s3im_stride=4, s3im_repeat_time=10, s3im_patch_height=64, size_average=True):
        super(S3IM, self).__init__()
        self.s3im_kernel_size = s3im_kernel_size
        self.s3im_stride = s3im_stride
        self.s3im_repeat_time = s3im_repeat_time
        self.s3im_patch_height = s3im_patch_height
        self.size_average = size_average
        self.channel = 1
        self.s3im_kernel = self.create_kernel(s3im_kernel_size, self.channel)

    def gaussian(self, s3im_kernel_size, sigma):
        gauss = torch.Tensor([exp(-(x - s3im_kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(s3im_kernel_size)])
        return gauss / gauss.sum()

    def create_kernel(self, s3im_kernel_size, channel):
        _1D_window = self.gaussian(s3im_kernel_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        s3im_kernel = Variable(_2D_window.expand(channel, 1, s3im_kernel_size, s3im_kernel_size).contiguous())
        return s3im_kernel

    def _ssim(self, img1, img2, s3im_kernel, s3im_kernel_size, channel, size_average=True, s3im_stride=None):
        mu1 = F.conv2d(img1, s3im_kernel, padding=(s3im_kernel_size - 1) // 2, groups=channel, stride=s3im_stride)
        mu2 = F.conv2d(img2, s3im_kernel, padding=(s3im_kernel_size - 1) // 2, groups=channel, stride=s3im_stride)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, s3im_kernel, padding=(s3im_kernel_size - 1) // 2, groups=channel, stride=s3im_stride) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, s3im_kernel, padding=(s3im_kernel_size - 1) // 2, groups=channel, stride=s3im_stride) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, s3im_kernel, padding=(s3im_kernel_size - 1) // 2, groups=channel, stride=s3im_stride) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def ssim_loss(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.s3im_kernel.data.type() == img1.data.type():
            s3im_kernel = self.s3im_kernel
        else:
            s3im_kernel = self.create_kernel(self.s3im_kernel_size, channel)
            if img1.is_cuda:
                s3im_kernel = s3im_kernel
            s3im_kernel = s3im_kernel.type_as(img1)
            self.s3im_kernel = s3im_kernel
            self.channel = channel
        return self._ssim(img1, img2, s3im_kernel, self.s3im_kernel_size, channel, self.size_average, s3im_stride=self.s3im_stride)

    def forward(self, src_vec, tar_vec):
        loss = 0.0
        index_list = []
        for i in range(self.s3im_repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        tar_patch = tar_all.permute(1, 0).reshape(1, 3, self.s3im_patch_height, -1)
        src_patch = src_all.permute(1, 0).reshape(1, 3, self.s3im_patch_height, -1)
        loss = 1 - self.ssim_loss(src_patch, tar_patch)
        return loss


def get_homography(intersection_points: 'torch.Tensor', normal: 'torch.Tensor', cameras: 'Cameras', valid_angle_thres: 'float'=0.3):
    """get homography

    Args:
        intersection_points (torch.Tensor): _description_
        normal (torch.Tensor): _description_
        cameras (Cameras): _description_
    """
    device = intersection_points.device
    c2w = cameras.camera_to_worlds
    K = cameras.get_intrinsics_matrices()
    K_inv = torch.linalg.inv(K)
    c2w[:, :3, 1:3] *= -1
    w2c_r = c2w[:, :3, :3].transpose(1, 2)
    w2c_t = -w2c_r @ c2w[:, :3, 3:]
    w2c = torch.cat([w2c_r, w2c_t], dim=-1)
    R_rel = w2c[:, :3, :3] @ c2w[:1, :3, :3]
    t_rel = w2c[:, :3, :3] @ c2w[:1, :3, 3:] + w2c[:1, :3, 3:]
    p_ref = w2c[0, :3, :3] @ intersection_points.transpose(1, 0) + w2c[0, :3, 3:]
    n_ref = w2c[0, :3, :3] @ normal.transpose(1, 0)
    d = torch.sum(n_ref * p_ref, dim=0, keepdims=True)
    H = R_rel[:, None, :, :] + t_rel[:, None, :, :] @ n_ref.transpose(1, 0)[None, :, None, :] / d[..., None, None]
    H = K[:, None] @ H @ K_inv[None, :1]
    dir_src = torch.nn.functional.normalize(c2w[:, None, :, 3] - intersection_points[None], dim=-1)
    valid = (dir_src * normal[None]).sum(dim=-1) > valid_angle_thres
    p_src = w2c[:, :3, :3] @ intersection_points.transpose(1, 0) + w2c[:, :3, 3:]
    valid_2 = p_src[:, 2, :] > 0.01
    return H, valid & valid_2


def get_intersection_points(ray_samples: 'RaySamples', sdf: 'torch.Tensor', normal: 'torch.Tensor', in_image_mask: 'torch.Tensor'):
    """compute intersection points

    Args:
        ray_samples (RaySamples): _description_
        sdf (torch.Tensor): _description_
        normal (torch.Tensor): _description_
        in_image_mask (torch.Tensor): we only use the rays in the range of [half_patch:h-half_path, half_patch:w-half_path]
    Returns:
        _type_: _description_
    """
    n_rays, n_samples = ray_samples.shape
    starts = ray_samples.frustums.starts
    sign_matrix = torch.cat([torch.sign(sdf[:, :-1, 0] * sdf[:, 1:, 0]), torch.ones(n_rays, 1)], dim=-1)
    cost_matrix = sign_matrix * torch.arange(n_samples, 0, -1).float()
    values, indices = torch.min(cost_matrix, -1)
    mask_sign_change = values < 0
    mask_pos_to_neg = sdf[torch.arange(n_rays), indices, 0] > 0
    mask = mask_sign_change & mask_pos_to_neg & in_image_mask
    d_low = starts[torch.arange(n_rays), indices, 0][mask]
    v_low = sdf[torch.arange(n_rays), indices, 0][mask]
    n_low = normal[torch.arange(n_rays), indices, :][mask]
    indices = torch.clamp(indices + 1, max=n_samples - 1)
    d_high = starts[torch.arange(n_rays), indices, 0][mask]
    v_high = sdf[torch.arange(n_rays), indices, 0][mask]
    n_high = normal[torch.arange(n_rays), indices, :][mask]
    z = (v_low * d_high - v_high * d_low) / (v_low - v_high)
    origins = ray_samples.frustums.origins[torch.arange(n_rays), indices, :][mask]
    directions = ray_samples.frustums.directions[torch.arange(n_rays), indices, :][mask]
    intersection_points = origins + directions * z[..., None]
    points_normal = (v_low[..., None] * n_high - v_high[..., None] * n_low) / (v_low[..., None] - v_high[..., None])
    points_normal = torch.nn.functional.normalize(points_normal, dim=-1, p=2)
    valid = (points_normal * directions).sum(dim=-1).abs() > 0.1
    intersection_points = intersection_points[valid]
    points_normal = points_normal[valid]
    new_mask = mask.clone()
    new_mask[mask] &= valid
    return intersection_points, points_normal, new_mask


class PatchWarping(nn.Module):
    """Standard patch warping."""

    def __init__(self, patch_size: 'int'=31, pixel_offset: 'float'=0.5, valid_angle_thres: 'float'=0.3):
        super().__init__()
        self.patch_size = patch_size
        half_size = patch_size // 2
        self.valid_angle_thres = valid_angle_thres
        patch_coords = torch.meshgrid(torch.arange(-half_size, half_size + 1), torch.arange(-half_size, half_size + 1), indexing='xy')
        patch_coords = torch.stack(patch_coords, dim=-1) + pixel_offset
        self.patch_coords = torch.cat([patch_coords, torch.zeros_like(patch_coords[..., :1])], dim=-1)

    def forward(self, ray_samples: 'RaySamples', sdf: 'torch.Tensor', normal: 'torch.Tensor', cameras: 'Cameras', images: 'torch.Tensor', pix_indices: 'torch.Tensor'):
        device = sdf.device
        cameras = cameras
        in_image_mask = (pix_indices[:, 0] > self.patch_size // 2) & (pix_indices[:, 1] > self.patch_size // 2) & (pix_indices[:, 0] < cameras.image_height[0] - self.patch_size // 2 - 1) & (pix_indices[:, 1] < cameras.image_width[0] - self.patch_size // 2 - 1)
        intersection_points, normal, mask = get_intersection_points(ray_samples, sdf, normal, in_image_mask)
        H, H_valid_mask = get_homography(intersection_points, normal, cameras, self.valid_angle_thres)
        pix_indices = torch.flip(pix_indices, dims=[-1])[mask].float()
        pix_indices = torch.cat([pix_indices, torch.ones(pix_indices.shape[0], 1)], dim=-1)
        pix_indices = pix_indices[:, None, None, :] + self.patch_coords[None]
        pix_indices = pix_indices.permute(0, 3, 1, 2).reshape(1, -1, 3, self.patch_size ** 2)
        warped_indices = H @ pix_indices
        positive_depth_mask = warped_indices[:, :, 2, :] >= 0.2
        warped_indices[:, :, 2, :] *= positive_depth_mask
        warped_indices = warped_indices[:, :, :2, :] / (warped_indices[:, :, 2:, :] + 1e-06)
        pix_coords = warped_indices.permute(0, 1, 3, 2).contiguous()
        pix_coords[..., 0:1] /= cameras.image_width[:, None, None] - 1
        pix_coords[..., 1:2] /= cameras.image_height[:, None, None] - 1
        pix_coords = (pix_coords - 0.5) * 2
        valid = (pix_coords[..., 0] > -1.0) & (pix_coords[..., 0] < 1.0) & (pix_coords[..., 1] > -1.0) & (pix_coords[..., 1] < 1.0)
        valid = valid & H_valid_mask[..., None] & positive_depth_mask
        rgb = torch.nn.functional.grid_sample(images.permute(0, 3, 1, 2), pix_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        rgb = rgb.permute(0, 2, 3, 1)
        if False:
            vis_patch_num = 60
            image = rgb[:, :vis_patch_num, :, :].reshape(-1, vis_patch_num, self.patch_size, self.patch_size, 3).permute(1, 2, 0, 3, 4).reshape(vis_patch_num * self.patch_size, -1, 3)
            cv2.imwrite('vis.png', (image.detach().cpu().numpy() * 255).astype(np.uint8)[..., ::-1])
        return rgb, valid


class Model(nn.Module):
    """Model class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        config: configuration for instantiating model
        scene_box: dataset scene box
    """
    config: 'ModelConfig'

    def __init__(self, config: 'ModelConfig', scene_box: 'SceneBox', num_train_data: 'int', world_size: 'int'=1, local_rank: 'int'=0, **kwargs) ->None:
        super().__init__()
        self.config = config
        self.scene_box = scene_box
        self.num_train_data = num_train_data
        self.kwargs = kwargs
        self.collider = None
        self.world_size = world_size
        self.local_rank = local_rank
        self.populate_modules()
        self.callbacks = None
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def get_training_callbacks(self, training_callback_attributes: 'TrainingCallbackAttributes') ->List[TrainingCallback]:
        """Returns a list of callbacks that run functions at the specified training iterations."""
        return []

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        if self.config.enable_collider:
            self.collider = NearFarCollider(near_plane=self.config.collider_params['near_plane'], far_plane=self.config.collider_params['far_plane'])

    @abstractmethod
    def get_param_groups(self) ->Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

    @abstractmethod
    def get_outputs(self, ray_bundle: 'RayBundle') ->Dict[str, torch.Tensor]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

    def forward(self, ray_bundle: 'RayBundle') ->Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)
        return self.get_outputs(ray_bundle)

    def get_metrics_dict(self, outputs, batch) ->Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        return {}

    @abstractmethod
    def get_loss_dict(self, outputs, batch, metrics_dict=None) ->Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: 'RayBundle') ->Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                continue
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)
        return outputs

    @abstractmethod
    def get_image_metrics_and_images(self, outputs: 'Dict[str, torch.Tensor]', batch: 'Dict[str, torch.Tensor]') ->Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.
        TODO: This shouldn't return a loss

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """

    def load_model(self, loaded_state: 'Dict[str, Any]') ->None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: dictionary of pre-trained model states
        """
        state = {key.replace('module.', ''): value for key, value in loaded_state['model'].items()}
        self.load_state_dict(state)


L1Loss = nn.L1Loss


MSELoss = nn.MSELoss


BLACK = torch.tensor([0.0, 0.0, 0.0])


BLUE = torch.tensor([0.0, 0.0, 1.0])


GREEN = torch.tensor([0.0, 1.0, 0.0])


RED = torch.tensor([1.0, 0.0, 0.0])


WHITE = torch.tensor([1.0, 1.0, 1.0])


COLORS_DICT = {'white': WHITE, 'black': BLACK, 'red': RED, 'green': GREEN, 'blue': BLUE}


def monosdf_normal_loss(normal_pred: 'torch.Tensor', normal_gt: 'torch.Tensor'):
    """normal consistency loss as monosdf

    Args:
        normal_pred (torch.Tensor): volume rendered normal
        normal_gt (torch.Tensor): monocular normal
    """
    normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
    normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
    cos = (1.0 - torch.sum(normal_pred * normal_gt, dim=-1)).mean()
    return l1 + cos


class TrainingCallbackLocation(Enum):
    """Enum for specifying where the training callback should be run."""
    BEFORE_TRAIN_ITERATION = auto()
    AFTER_TRAIN_ITERATION = auto()


class NGPModel(Model):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """
    config: 'InstantNGPModelConfig'
    field: 'TCNNInstantNGPField'

    def __init__(self, config: 'InstantNGPModelConfig', **kwargs) ->None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.field = TCNNInstantNGPField(aabb=self.scene_box.aabb, contraction_type=self.config.contraction_type, use_appearance_embedding=self.config.use_appearance_embedding, num_images=self.num_train_data)
        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)
        self.occupancy_grid = nerfacc.OccupancyGrid(roi_aabb=self.scene_aabb, resolution=self.config.grid_resolution, contraction_type=self.config.contraction_type)
        vol_sampler_aabb = self.scene_box.aabb if self.config.contraction_type == ContractionType.AABB else None
        self.sampler = VolumetricSampler(scene_aabb=vol_sampler_aabb, occupancy_grid=self.occupancy_grid, density_fn=self.field.density_fn)
        background_color = 'random'
        if self.config.background_color in ['white', 'black']:
            background_color = colors.COLORS_DICT[self.config.background_color]
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method='expected')
        self.rgb_loss = MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_training_callbacks(self, training_callback_attributes: 'TrainingCallbackAttributes') ->List[TrainingCallback]:

        def update_occupancy_grid(step: 'int'):
            self.occupancy_grid.every_n_step(step=step, occ_eval_fn=lambda x: self.field.get_opacity(x, self.config.render_step_size))
        return [TrainingCallback(where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], update_every_num_iters=1, func=update_occupancy_grid)]

    def get_param_groups(self) ->Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError('populate_fields() must be called before get_param_groups')
        param_groups['fields'] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: 'RayBundle'):
        assert self.field is not None
        num_rays = len(ray_bundle)
        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(ray_bundle=ray_bundle, near_plane=self.config.near_plane, far_plane=self.config.far_plane, render_step_size=self.config.render_step_size, cone_angle=self.config.cone_angle, alpha_thre=self.config.alpha_thre)
        field_outputs = self.field(ray_samples)
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(packed_info=packed_info, sigmas=field_outputs[FieldHeadNames.DENSITY], t_starts=ray_samples.frustums.starts, t_ends=ray_samples.frustums.ends)
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays)
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        alive_ray_mask = accumulation.squeeze(-1) > 0
        outputs = {'rgb': rgb, 'accumulation': accumulation, 'depth': depth, 'alive_ray_mask': alive_ray_mask, 'num_samples_per_ray': packed_info[:, 1]}
        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch['image']
        metrics_dict = {}
        metrics_dict['psnr'] = self.psnr(outputs['rgb'], image)
        metrics_dict['num_samples_per_batch'] = outputs['num_samples_per_ray'].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch['image']
        mask = outputs['alive_ray_mask']
        rgb_loss = self.rgb_loss(image[mask], outputs['rgb'][mask])
        loss_dict = {'rgb_loss': rgb_loss}
        return loss_dict

    def get_image_metrics_and_images(self, outputs: 'Dict[str, torch.Tensor]', batch: 'Dict[str, torch.Tensor]') ->Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch['image']
        rgb = outputs['rgb']
        acc = colormaps.apply_colormap(outputs['accumulation'])
        depth = colormaps.apply_depth_colormap(outputs['depth'], accumulation=outputs['accumulation'])
        alive_ray_mask = colormaps.apply_colormap(outputs['alive_ray_mask'])
        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        combined_alive_ray_mask = torch.cat([alive_ray_mask], dim=1)
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)
        metrics_dict = {'psnr': float(psnr.item()), 'ssim': float(ssim), 'lpips': float(lpips)}
        images_dict = {'img': combined_rgb, 'accumulation': combined_acc, 'depth': combined_depth, 'alive_ray_mask': combined_alive_ray_mask}
        return metrics_dict, images_dict


class MipNerfModel(Model):
    """mip-NeRF model

    Args:
        config: MipNerf configuration to instantiate model
    """

    def __init__(self, config: 'ModelConfig', **kwargs) ->None:
        self.field = None
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()
        position_encoding = NeRFEncoding(in_dim=3, num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0, include_input=True)
        direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True)
        self.field = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding, use_integrated_encoding=True)
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, include_original=False)
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.rgb_loss = MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) ->Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError('populate_fields() must be called before get_param_groups')
        param_groups['fields'] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: 'RayBundle'):
        if self.field is None:
            raise ValueError('populate_fields() must be called before get_outputs')
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        field_outputs_coarse = self.field.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(rgb=field_outputs_coarse[FieldHeadNames.RGB], weights=weights_coarse)
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        field_outputs_fine = self.field.forward(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(rgb=field_outputs_fine[FieldHeadNames.RGB], weights=weights_fine)
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
        outputs = {'rgb_coarse': rgb_coarse, 'rgb_fine': rgb_fine, 'accumulation_coarse': accumulation_coarse, 'accumulation_fine': accumulation_fine, 'depth_coarse': depth_coarse, 'depth_fine': depth_fine}
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch['image']
        rgb_loss_coarse = self.rgb_loss(image, outputs['rgb_coarse'])
        rgb_loss_fine = self.rgb_loss(image, outputs['rgb_fine'])
        loss_dict = {'rgb_loss_coarse': rgb_loss_coarse, 'rgb_loss_fine': rgb_loss_fine}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(self, outputs: 'Dict[str, torch.Tensor]', batch: 'Dict[str, torch.Tensor]') ->Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch['image']
        rgb_coarse = outputs['rgb_coarse']
        rgb_fine = outputs['rgb_fine']
        acc_coarse = colormaps.apply_colormap(outputs['accumulation_coarse'])
        acc_fine = colormaps.apply_colormap(outputs['accumulation_fine'])
        depth_coarse = colormaps.apply_depth_colormap(outputs['depth_coarse'], accumulation=outputs['accumulation_coarse'], near_plane=self.config.collider_params['near_plane'], far_plane=self.config.collider_params['far_plane'])
        depth_fine = colormaps.apply_depth_colormap(outputs['depth_fine'], accumulation=outputs['accumulation_fine'], near_plane=self.config.collider_params['near_plane'], far_plane=self.config.collider_params['far_plane'])
        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
        rgb_coarse = torch.clip(rgb_coarse, min=-1, max=1)
        rgb_fine = torch.clip(rgb_fine, min=-1, max=1)
        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)
        metrics_dict = {'psnr': float(fine_psnr.item()), 'coarse_psnr': float(coarse_psnr.item()), 'fine_psnr': float(fine_psnr.item()), 'fine_ssim': float(fine_ssim.item()), 'fine_lpips': float(fine_lpips.item())}
        images_dict = {'img': combined_rgb, 'accumulation': combined_acc, 'depth': combined_depth}
        return metrics_dict, images_dict


def lossfun_distortion(t, w):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L142
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L266
    """
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)
    loss_intra = torch.sum(w ** 2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3
    return loss_inter + loss_intra


def ray_samples_to_sdist(ray_samples):
    """Convert ray samples to s space"""
    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends
    sdist = torch.cat([starts[..., 0], ends[..., -1:, 0]], dim=-1)
    return sdist


def distortion_loss(weights_list, ray_samples_list):
    """From mipnerf360"""
    c = ray_samples_to_sdist(ray_samples_list[-1])
    w = weights_list[-1][..., 0]
    loss = torch.mean(lossfun_distortion(c, w))
    return loss


EPS = 1e-07


def lossfun_outer(t: "TensorType[..., 'num_samples+1']", w: "TensorType[..., 'num_samples']", t_env: "TensorType[..., 'num_samples+1']", w_env: "TensorType[..., 'num_samples']"):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L136
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L80

    Args:
        t: interval edges
        w: weights
        t_env: interval edges of the upper bound enveloping historgram
        w_env: weights that should upper bound the inner (t,w) histogram
    """
    w_outer = outer(t[..., :-1], t[..., 1:], t_env[..., :-1], t_env[..., 1:], w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + EPS)


def interlevel_loss(weights_list, ray_samples_list):
    """Calculates the proposal loss in the MipNeRF-360 paper.

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/model.py#L515
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/train_utils.py#L133
    """
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach()
    w = weights_list[-1][..., 0].detach()
    loss_interlevel = 0.0
    for ray_samples, weights in zip(ray_samples_list[:-1], weights_list[:-1]):
        sdist = ray_samples_to_sdist(ray_samples)
        cp = sdist
        wp = weights[..., 0]
        loss_interlevel += torch.mean(lossfun_outer(c, w, cp, wp))
    return loss_interlevel


def orientation_loss(weights: "TensorType['bs':..., 'num_samples', 1]", normals: "TensorType['bs':..., 'num_samples', 3]", viewdirs: "TensorType['bs':..., 3]"):
    """Orientation loss proposed in Ref-NeRF.
    Loss that encourages that all visible normals are facing towards the camera.
    """
    w = weights
    n = normals
    v = viewdirs
    n_dot_v = (n * v[..., None, :]).sum(axis=-1)
    return (w[..., 0] * torch.fmin(torch.zeros_like(n_dot_v), n_dot_v) ** 2).sum(dim=-1)


def pred_normal_loss(weights: "TensorType['bs':..., 'num_samples', 1]", normals: "TensorType['bs':..., 'num_samples', 3]", pred_normals: "TensorType['bs':..., 'num_samples', 3]"):
    """Loss between normals calculated from density and normals from prediction network."""
    return (weights[..., 0] * (1.0 - torch.sum(normals * pred_normals, dim=-1))).sum(dim=-1)


class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """
    config: 'NerfactoModelConfig'

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        scene_contraction = SceneContraction(order=float('inf'))
        self.field = TCNNNerfactoField(self.scene_box.aabb, num_levels=self.config.num_levels, max_res=self.config.max_res, log2_hashmap_size=self.config.log2_hashmap_size, spatial_distortion=scene_contraction, num_images=self.num_train_data, use_pred_normals=self.config.predict_normals, use_average_appearance_embedding=self.config.use_average_appearance_embedding)
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, 'Only one proposal network is allowed.'
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args)
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])
        update_schedule = lambda step: np.clip(np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]), 1, self.config.proposal_update_every)
        self.proposal_sampler = ProposalNetworkSampler(num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray, num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray, num_proposal_network_iterations=self.config.num_proposal_iterations, single_jitter=self.config.use_single_jitter, update_sched=update_schedule)
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        background_color = get_color(self.config.background_color) if self.config.background_color in set(['white', 'black']) else self.config.background_color
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()
        self.rgb_loss = MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) ->Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups['proposal_networks'] = list(self.proposal_networks.parameters())
        param_groups['fields'] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(self, training_callback_attributes: 'TrainingCallbackAttributes') ->List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: b * x / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)
            callbacks.append(TrainingCallback(where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], update_every_num_iters=1, func=set_anneal))
            callbacks.append(TrainingCallback(where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION], update_every_num_iters=1, func=self.proposal_sampler.step_cb))
        return callbacks

    def get_outputs(self, ray_bundle: 'RayBundle'):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        outputs = {'rgb': rgb, 'accumulation': accumulation, 'depth': depth}
        if self.config.predict_normals:
            outputs['normals'] = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            outputs['pred_normals'] = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
        if True or self.training:
            outputs['weights_list'] = weights_list
            outputs['ray_samples_list'] = ray_samples_list
        if self.training and self.config.predict_normals:
            outputs['rendered_orientation_loss'] = orientation_loss(weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions)
            outputs['rendered_pred_normal_loss'] = pred_normal_loss(weights.detach(), field_outputs[FieldHeadNames.NORMALS].detach(), field_outputs[FieldHeadNames.PRED_NORMALS])
        for i in range(self.config.num_proposal_iterations):
            outputs[f'prop_depth_{i}'] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch['image']
        metrics_dict['psnr'] = self.psnr(outputs['rgb'], image)
        if self.training:
            metrics_dict['distortion'] = distortion_loss(outputs['weights_list'], outputs['ray_samples_list'])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch['image']
        loss_dict['rgb_loss'] = self.rgb_loss(image, outputs['rgb'])
        if self.training:
            loss_dict['interlevel_loss'] = self.config.interlevel_loss_mult * interlevel_loss(outputs['weights_list'], outputs['ray_samples_list'])
            assert metrics_dict is not None and 'distortion' in metrics_dict
            loss_dict['distortion_loss'] = self.config.distortion_loss_mult * metrics_dict['distortion']
            if self.config.predict_normals:
                loss_dict['orientation_loss'] = self.config.orientation_loss_mult * torch.mean(outputs['rendered_orientation_loss'])
                loss_dict['pred_normal_loss'] = self.config.pred_normal_loss_mult * torch.mean(outputs['rendered_pred_normal_loss'])
        return loss_dict

    def get_image_metrics_and_images(self, outputs: 'Dict[str, torch.Tensor]', batch: 'Dict[str, torch.Tensor]') ->Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch['image']
        rgb = outputs['rgb']
        acc = colormaps.apply_colormap(outputs['accumulation'])
        depth = colormaps.apply_depth_colormap(outputs['depth'], accumulation=outputs['accumulation'])
        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)
        metrics_dict = {'psnr': float(psnr.item()), 'ssim': float(ssim)}
        metrics_dict['lpips'] = float(lpips)
        images_dict = {'img': combined_rgb, 'accumulation': combined_acc, 'depth': combined_depth}
        if 'normals' in outputs:
            images_dict['normals'] = (outputs['normals'] + 1.0) / 2.0
        if 'pred_normals' in outputs:
            images_dict['pred_normals'] = (outputs['pred_normals'] + 1.0) / 2.0
        for i in range(self.config.num_proposal_iterations):
            key = f'prop_depth_{i}'
            prop_depth_i = colormaps.apply_depth_colormap(outputs[key], accumulation=outputs['accumulation'])
            images_dict[key] = prop_depth_i
        return metrics_dict, images_dict


class SemanticNerfWModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """
    config: 'SemanticNerfWModelConfig'

    def __init__(self, config: 'SemanticNerfWModelConfig', metadata: 'Dict', **kwargs) ->None:
        assert 'semantics' in metadata.keys() and isinstance(metadata['semantics'], Semantics)
        self.semantics = metadata['semantics']
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        scene_contraction = SceneContraction(order=float('inf'))
        if self.config.use_transient_embedding:
            raise ValueError('Transient embedding is not fully working for semantic nerf-w.')
        self.field = TCNNNerfactoField(self.scene_box.aabb, num_levels=self.config.num_levels, max_res=self.config.max_res, log2_hashmap_size=self.config.log2_hashmap_size, spatial_distortion=scene_contraction, num_images=self.num_train_data, use_average_appearance_embedding=self.config.use_average_appearance_embedding, use_transient_embedding=self.config.use_transient_embedding, use_semantics=True, num_semantic_classes=len(self.semantics.classes))
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction)
            self.proposal_networks.append(network)
            self.density_fns = [network.density_fn for _ in range(self.config.num_proposal_iterations)]
        else:
            for _ in range(self.config.num_proposal_iterations):
                network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction)
                self.proposal_networks.append(network)
            self.density_fns = [network.density_fn for network in self.proposal_networks]
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        self.proposal_sampler = ProposalNetworkSampler(num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray, num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray, num_proposal_network_iterations=self.config.num_proposal_iterations, single_jitter=self.config.use_single_jitter)
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_uncertainty = UncertaintyRenderer()
        self.renderer_semantics = SemanticRenderer()
        self.rgb_loss = MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) ->Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups['proposal_networks'] = list(self.proposal_networks.parameters())
        param_groups['fields'] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(self, training_callback_attributes: 'TrainingCallbackAttributes') ->List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: b * x / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)
            callbacks.append(TrainingCallback(where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], update_every_num_iters=1, func=set_anneal))
        return callbacks

    def get_outputs(self, ray_bundle: 'RayBundle'):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples)
        if self.training and self.config.use_transient_embedding:
            density = field_outputs[FieldHeadNames.DENSITY] + field_outputs[FieldHeadNames.TRANSIENT_DENSITY]
            weights = ray_samples.get_weights(density)
            weights_static = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            rgb_static_component = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            rgb_transient_component = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.TRANSIENT_RGB], weights=weights)
            rgb = rgb_static_component + rgb_transient_component
        else:
            weights_static = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            weights = weights_static
            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        weights_list.append(weights_static)
        ray_samples_list.append(ray_samples)
        depth = self.renderer_depth(weights=weights_static, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights_static)
        outputs = {'rgb': rgb, 'accumulation': accumulation, 'depth': depth}
        outputs['weights_list'] = weights_list
        outputs['ray_samples_list'] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f'prop_depth_{i}'] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        if self.training and self.config.use_transient_embedding:
            weights_transient = ray_samples.get_weights(field_outputs[FieldHeadNames.TRANSIENT_DENSITY])
            uncertainty = self.renderer_uncertainty(field_outputs[FieldHeadNames.UNCERTAINTY], weights_transient)
            outputs['uncertainty'] = uncertainty + 0.03
            outputs['density_transient'] = field_outputs[FieldHeadNames.TRANSIENT_DENSITY]
        outputs['semantics'] = self.renderer_semantics(field_outputs[FieldHeadNames.SEMANTICS], weights=weights_static.detach())
        semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs['semantics'], dim=-1), dim=-1)
        outputs['semantics_colormap'] = self.semantics.colors[semantic_labels]
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch['image']
        metrics_dict['psnr'] = self.psnr(outputs['rgb'], image)
        metrics_dict['distortion'] = distortion_loss(outputs['weights_list'], outputs['ray_samples_list'])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch['image']
        loss_dict['interlevel_loss'] = self.config.interlevel_loss_mult * interlevel_loss(outputs['weights_list'], outputs['ray_samples_list'])
        assert metrics_dict is not None and 'distortion' in metrics_dict
        loss_dict['distortion_loss'] = self.config.distortion_loss_mult * metrics_dict['distortion']
        if self.training and self.config.use_transient_embedding:
            betas = outputs['uncertainty']
            loss_dict['uncertainty_loss'] = 3 + torch.log(betas).mean()
            loss_dict['density_loss'] = 0.01 * outputs['density_transient'].mean()
            loss_dict['rgb_loss'] = (((image - outputs['rgb']) ** 2).sum(-1) / betas[..., 0] ** 2).mean()
        else:
            loss_dict['rgb_loss'] = self.rgb_loss(image, outputs['rgb'])
        loss_dict['semantics_loss'] = self.cross_entropy_loss(outputs['semantics'], batch['semantics'][..., 0].long())
        return loss_dict

    def get_image_metrics_and_images(self, outputs: 'Dict[str, torch.Tensor]', batch: 'Dict[str, torch.Tensor]') ->Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch['image']
        rgb = outputs['rgb']
        rgb = torch.clamp(rgb, min=0, max=1)
        acc = colormaps.apply_colormap(outputs['accumulation'])
        depth = colormaps.apply_depth_colormap(outputs['depth'], accumulation=outputs['accumulation'])
        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)
        metrics_dict = {'psnr': float(psnr.item()), 'ssim': float(ssim)}
        metrics_dict['lpips'] = float(lpips)
        images_dict = {'img': combined_rgb, 'accumulation': combined_acc, 'depth': combined_depth}
        for i in range(self.config.num_proposal_iterations):
            key = f'prop_depth_{i}'
            prop_depth_i = colormaps.apply_depth_colormap(outputs[key], accumulation=outputs['accumulation'])
            images_dict[key] = prop_depth_i
        semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs['semantics'], dim=-1), dim=-1)
        images_dict['semantics_colormap'] = self.semantics.colors[semantic_labels]
        images_dict['mask'] = batch['mask'].repeat(1, 1, 3)
        return metrics_dict, images_dict


class TensoRFModel(Model):
    """TensoRF Model

    Args:
        config: TensoRF configuration to instantiate model
    """

    def __init__(self, config: 'TensoRFModelConfig', **kwargs) ->None:
        self.init_resolution = config.init_resolution
        self.upsampling_iters = config.upsampling_iters
        self.num_den_components = config.num_den_components
        self.num_color_components = config.num_color_components
        self.appearance_dim = config.appearance_dim
        self.upsampling_steps = np.round(np.exp(np.linspace(np.log(config.init_resolution), np.log(config.final_resolution), len(config.upsampling_iters) + 1))).astype('int').tolist()[1:]
        super().__init__(config=config, **kwargs)

    def get_training_callbacks(self, training_callback_attributes: 'TrainingCallbackAttributes') ->List[TrainingCallback]:

        def reinitialize_optimizer(self, training_callback_attributes: 'TrainingCallbackAttributes', step: 'int'):
            resolution = self.upsampling_steps.pop(0)
            self.field.density_encoding.upsample_grid(resolution)
            self.field.color_encoding.upsample_grid(resolution)
            optimizers_config = training_callback_attributes.optimizers.config
            enc = training_callback_attributes.pipeline.get_param_groups()['encodings']
            lr_init = optimizers_config['encodings']['optimizer'].lr
            training_callback_attributes.optimizers.optimizers['encodings'] = optimizers_config['encodings']['optimizer'].setup(params=enc)
            if optimizers_config['encodings']['scheduler']:
                training_callback_attributes.optimizers.schedulers['encodings'] = optimizers_config['encodings']['scheduler'].setup(optimizer=training_callback_attributes.optimizers.optimizers['encodings'], lr_init=lr_init)
        callbacks = [TrainingCallback(where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION], iters=self.upsampling_iters, func=reinitialize_optimizer, args=[self, training_callback_attributes])]
        return callbacks

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()
        density_encoding = TensorVMEncoding(resolution=self.init_resolution, num_components=self.num_den_components)
        color_encoding = TensorVMEncoding(resolution=self.init_resolution, num_components=self.num_color_components)
        feature_encoding = NeRFEncoding(in_dim=self.appearance_dim, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)
        direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)
        self.field = TensoRFField(self.scene_box.aabb, feature_encoding=feature_encoding, direction_encoding=direction_encoding, density_encoding=density_encoding, color_encoding=color_encoding, appearance_dim=self.appearance_dim, head_mlp_num_layers=2, head_mlp_layer_width=128, use_sh=False)
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_samples, single_jitter=True)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_samples // 2, single_jitter=True)
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.rgb_loss = MSELoss()
        self.s3im_loss = S3IM(s3im_kernel_size=self.config.s3im_kernel_size, s3im_stride=self.config.s3im_stride, s3im_repeat_time=self.config.s3im_repeat_time, s3im_patch_height=self.config.s3im_patch_height)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

    def get_param_groups(self) ->Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups['fields'] = list(self.field.mlp_head.parameters()) + list(self.field.B.parameters()) + list(self.field.field_output_rgb.parameters())
        param_groups['encodings'] = list(self.field.color_encoding.parameters()) + list(self.field.density_encoding.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: 'RayBundle'):
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        dens = self.field.get_density(ray_samples_uniform)
        weights = ray_samples_uniform.get_weights(dens)
        coarse_accumulation = self.renderer_accumulation(weights)
        acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights)
        field_outputs_fine = self.field.forward(ray_samples_pdf, mask=acc_mask, bg_color=colors.WHITE)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        accumulation = self.renderer_accumulation(weights_fine)
        depth = self.renderer_depth(weights_fine, ray_samples_pdf)
        rgb = self.renderer_rgb(rgb=field_outputs_fine[FieldHeadNames.RGB], weights=weights_fine)
        rgb = torch.where(accumulation < 0, colors.WHITE, rgb)
        accumulation = torch.clamp(accumulation, min=0)
        outputs = {'rgb': rgb, 'accumulation': accumulation, 'depth': depth}
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) ->Dict[str, torch.Tensor]:
        device = outputs['rgb'].device
        image = batch['image']
        rgb_loss = self.rgb_loss(image, outputs['rgb'])
        loss_dict = {'rgb_loss': rgb_loss}
        if self.config.s3im_loss_mult > 0:
            loss_dict['s3im_loss'] = self.s3im_loss(image, outputs['rgb']) * self.config.s3im_loss_mult
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(self, outputs: 'Dict[str, torch.Tensor]', batch: 'Dict[str, torch.Tensor]') ->Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch['image']
        rgb = outputs['rgb']
        acc = colormaps.apply_colormap(outputs['accumulation'])
        depth = colormaps.apply_depth_colormap(outputs['depth'], accumulation=outputs['accumulation'], near_plane=self.config.collider_params['near_plane'], far_plane=self.config.collider_params['far_plane'])
        combined_rgb = torch.cat([image, rgb], dim=1)
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)
        metrics_dict = {'psnr': float(psnr.item()), 'ssim': float(ssim.item()), 'lpips': float(lpips.item())}
        images_dict = {'img': combined_rgb, 'accumulation': acc, 'depth': depth}
        return metrics_dict, images_dict


class NeRFModel(Model):
    """Vanilla NeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    def __init__(self, config: 'VanillaModelConfig', **kwargs) ->None:
        self.field_coarse = None
        self.field_fine = None
        self.temporal_distortion = None
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()
        position_encoding = NeRFEncoding(in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True)
        direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True)
        self.field_coarse = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.field_fine = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.rgb_loss = MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        if getattr(self.config, 'enable_temporal_distortion', False):
            params = self.config.temporal_distortion_params
            kind = params.pop('kind')
            self.temporal_distortion = kind.to_temporal_distortion(params)

    def get_param_groups(self) ->Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError('populate_fields() must be called before get_param_groups')
        param_groups['fields'] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        if self.temporal_distortion is not None:
            param_groups['temporal_distortion'] = list(self.temporal_distortion.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: 'RayBundle'):
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError('populate_fields() must be called before get_outputs')
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        if self.temporal_distortion is not None:
            offsets = self.temporal_distortion(ray_samples_uniform.frustums.get_positions(), ray_samples_uniform.times)
            ray_samples_uniform.frustums.set_offsets(offsets)
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(rgb=field_outputs_coarse[FieldHeadNames.RGB], weights=weights_coarse)
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        if self.temporal_distortion is not None:
            offsets = self.temporal_distortion(ray_samples_pdf.frustums.get_positions(), ray_samples_pdf.times)
            ray_samples_pdf.frustums.set_offsets(offsets)
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(rgb=field_outputs_fine[FieldHeadNames.RGB], weights=weights_fine)
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
        outputs = {'rgb_coarse': rgb_coarse, 'rgb_fine': rgb_fine, 'accumulation_coarse': accumulation_coarse, 'accumulation_fine': accumulation_fine, 'depth_coarse': depth_coarse, 'depth_fine': depth_fine}
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) ->Dict[str, torch.Tensor]:
        device = outputs['rgb_coarse'].device
        image = batch['image']
        rgb_loss_coarse = self.rgb_loss(image, outputs['rgb_coarse'])
        rgb_loss_fine = self.rgb_loss(image, outputs['rgb_fine'])
        loss_dict = {'rgb_loss_coarse': rgb_loss_coarse, 'rgb_loss_fine': rgb_loss_fine}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(self, outputs: 'Dict[str, torch.Tensor]', batch: 'Dict[str, torch.Tensor]') ->Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch['image']
        rgb_coarse = outputs['rgb_coarse']
        rgb_fine = outputs['rgb_fine']
        acc_coarse = colormaps.apply_colormap(outputs['accumulation_coarse'])
        acc_fine = colormaps.apply_colormap(outputs['accumulation_fine'])
        depth_coarse = colormaps.apply_depth_colormap(outputs['depth_coarse'], accumulation=outputs['accumulation_coarse'], near_plane=self.config.collider_params['near_plane'], far_plane=self.config.collider_params['far_plane'])
        depth_fine = colormaps.apply_depth_colormap(outputs['depth_fine'], accumulation=outputs['accumulation_fine'], near_plane=self.config.collider_params['near_plane'], far_plane=self.config.collider_params['far_plane'])
        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)
        metrics_dict = {'psnr': float(fine_psnr.item()), 'coarse_psnr': float(coarse_psnr), 'fine_psnr': float(fine_psnr), 'fine_ssim': float(fine_ssim), 'fine_lpips': float(fine_lpips)}
        images_dict = {'img': combined_rgb, 'accumulation': combined_acc, 'depth': combined_depth}
        return metrics_dict, images_dict


def module_wrapper(ddp_or_model: 'Union[DDP, Model]') ->Model:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return ddp_or_model


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (GradientLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MiDaSMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (NCC,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SSIM,
     lambda: ([], {'patch_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ScaleAndShiftInvariantLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 16, 4, 4]), torch.rand([4, 16, 4, 4]), torch.rand([4, 16, 4, 4])], {})),
    (SingleVarianceNetwork,
     lambda: ([], {'init_val': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

