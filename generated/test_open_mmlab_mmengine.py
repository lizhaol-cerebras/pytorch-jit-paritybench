
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


import torch.nn.functional as F


import torchvision


import torchvision.transforms as transforms


from torch.optim import SGD


import torch


import copy


from functools import partial


from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


from torch.optim import AdamW


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import numpy as np


from torchvision.datasets import VisionDataset


from torchvision.models.segmentation import deeplabv3_resnet50


import time


from abc import ABCMeta


from abc import abstractmethod


from collections import OrderedDict


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Sequence


from typing import Tuple


from typing import Union


import torch.nn as nn


from torch.optim import Optimizer


import inspect


from typing import Any


import logging


from torch.nn.parallel import DistributedDataParallel


from torch.distributed.fsdp import FullStateDictConfig


from torch.distributed.fsdp import FullyShardedDataParallel


from torch.distributed.fsdp import LocalStateDictConfig


from torch.distributed.fsdp import StateDictType


from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig


from torch.distributed.fsdp.fully_sharded_data_parallel import LocalOptimStateDictConfig


from torch.distributed.fsdp.fully_sharded_data_parallel import OptimStateDictConfig


from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictConfig


from torch.optim.lr_scheduler import LRScheduler


from torch._subclasses.fake_tensor import _is_tensor_constructor


from torch.utils._python_dispatch import TorchDispatchMode


import typing


from collections import defaultdict


from typing import Counter


from typing import DefaultDict


from torch import Tensor


import warnings


from collections import Counter


from copy import copy


from numbers import Number


from typing import Iterable


from typing import Iterator


from typing import Set


from typing import TypeVar


from torch.jit import TracerWarning


from torch.jit import _get_trace_graph


from torch import nn


import types


import uuid


from collections import abc


import functools


from collections.abc import Mapping


import math


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


import itertools


from typing import Sized


from torch.utils.data import Sampler


import random


from typing import Mapping


from torch.utils.data._utils.collate import default_collate as torch_default_collate


from typing import Generator


from torch import distributed as torch_dist


from torch._utils import _flatten_dense_tensors


from torch._utils import _take_tensors


from torch._utils import _unflatten_dense_tensors


from torch.distributed import ProcessGroup


from itertools import zip_longest


from itertools import chain


import torch.multiprocessing as mp


from collections.abc import Iterable


import re


from logging import Logger


from logging import LogRecord


from logging import handlers


from typing import TYPE_CHECKING


from copy import deepcopy


from logging import FileHandler


from torch.nn.parallel import DataParallel


import torch.distributed as dist


from torch.distributed.fsdp.api import FullStateDictConfig


from torch.distributed.fsdp.api import LocalOptimStateDictConfig


from torch.distributed.fsdp.api import LocalStateDictConfig


from torch.distributed.fsdp.api import OptimStateDictConfig


from torch.distributed.fsdp.api import ShardedOptimStateDictConfig


from torch.distributed.fsdp.api import ShardedStateDictConfig


from torch.distributed.fsdp.api import ShardingStrategy


from torch.distributed.fsdp.api import StateDictConfig


from torch.distributed.fsdp.api import StateDictSettings


from torch.distributed.fsdp.api import StateDictType


from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch


from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload


from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel


from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision


from torch.nn.parallel.distributed import DistributedDataParallel


from torch.nn import GroupNorm


from torch.nn import LayerNorm


from torch.distributed.rpc import is_available


from functools import wraps


from torch.utils.checkpoint import checkpoint


from collections import namedtuple


from typing import Type


from collections.abc import Sized


from enum import Enum


from typing import NamedTuple


from torch.multiprocessing import active_children


from torch.testing import assert_allclose as _assert_allclose


from uuid import uuid4


from torch.distributed import destroy_process_group


import collections.abc


from inspect import getfullargspec


from inspect import ismodule


from itertools import repeat


from collections.abc import MutableMapping


from torch.optim import *


from numpy import prod


from torch.autograd.function import Function


from torch.nn import functional as F


from itertools import product


import torch.distributed as torch_dist


from torch.nn.init import constant_


from torch.distributed import init_process_group


from torch.cuda.amp import GradScaler


from torch.optim import Adam


import torch.optim as optim


from torch.multiprocessing.spawn import start_processes


class BaseAveragedModel(nn.Module):
    """A base class for averaging model weights.

    Weight averaging, such as SWA and EMA, is a widely used technique for
    training neural networks. This class implements the averaging process
    for a model. All subclasses must implement the `avg_func` method.
    This class creates a copy of the provided module :attr:`model`
    on the :attr:`device` and allows computing running averages of the
    parameters of the :attr:`model`.

    The code is referenced from: https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py.

    Different from the `AveragedModel` in PyTorch, we use in-place operation
    to improve the parameter updating speed, which is about 5 times faster
    than the non-in-place version.

    In mmengine, we provide two ways to use the model averaging:

    1. Use the model averaging module in hook:
       We provide an :class:`mmengine.hooks.EMAHook` to apply the model
       averaging during training. Add ``custom_hooks=[dict(type='EMAHook')]``
       to the config or the runner.

    2. Use the model averaging module directly in the algorithm. Take the ema
       teacher in semi-supervise as an example:

       >>> from mmengine.model import ExponentialMovingAverage
       >>> student = ResNet(depth=50)
       >>> # use ema model as teacher
       >>> ema_teacher = ExponentialMovingAverage(student)

    Args:
        model (nn.Module): The model to be averaged.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(self, model: 'nn.Module', interval: 'int'=1, device: 'Optional[torch.device]'=None, update_buffers: 'bool'=False) ->None:
        super().__init__()
        self.module = deepcopy(model).requires_grad_(False)
        self.interval = interval
        if device is not None:
            self.module = self.module
        self.register_buffer('steps', torch.tensor(0, dtype=torch.long, device=device))
        self.update_buffers = update_buffers
        if update_buffers:
            self.avg_parameters = self.module.state_dict()
        else:
            self.avg_parameters = dict(self.module.named_parameters())

    @abstractmethod
    def avg_func(self, averaged_param: 'Tensor', source_param: 'Tensor', steps: 'int') ->None:
        """Use in-place operation to compute the average of the parameters. All
        subclasses must implement this method.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """

    def forward(self, *args, **kwargs):
        """Forward method of the averaged model."""
        return self.module(*args, **kwargs)

    def update_parameters(self, model: 'nn.Module') ->None:
        """Update the parameters of the model. This method will execute the
        ``avg_func`` to compute the new parameters and update the model's
        parameters.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        src_parameters = model.state_dict() if self.update_buffers else dict(model.named_parameters())
        if self.steps == 0:
            for k, p_avg in self.avg_parameters.items():
                p_avg.data.copy_(src_parameters[k].data)
        elif self.steps % self.interval == 0:
            for k, p_avg in self.avg_parameters.items():
                if p_avg.dtype.is_floating_point:
                    device = p_avg.device
                    self.avg_func(p_avg.data, src_parameters[k].data, self.steps)
        if not self.update_buffers:
            for b_avg, b_src in zip(self.module.buffers(), model.buffers()):
                b_avg.data.copy_(b_src.data)
        self.steps += 1


class StochasticWeightAverage(BaseAveragedModel):
    """Implements the stochastic weight averaging (SWA) of the model.

    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization, UAI 2018.
    <https://arxiv.org/abs/1803.05407>`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson.
    """

    def avg_func(self, averaged_param: 'Tensor', source_param: 'Tensor', steps: 'int') ->None:
        """Compute the average of the parameters using stochastic weight
        average.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        averaged_param.add_(source_param - averaged_param, alpha=1 / float(steps // self.interval + 1))


class FilterDuplicateWarning(logging.Filter):
    """Filter the repeated warning message.

    Args:
        name (str): name of the filter.
    """

    def __init__(self, name: 'str'='mmengine'):
        super().__init__(name)
        self.seen: 'set' = set()

    def filter(self, record: 'LogRecord') ->bool:
        """Filter the repeated warning message.

        Args:
            record (LogRecord): The log record.

        Returns:
            bool: Whether to output the log record.
        """
        if record.levelno != logging.WARNING:
            return True
        if record.msg not in self.seen:
            self.seen.add(record.msg)
            return True
        return False


class MMFormatter(logging.Formatter):
    """Colorful format for MMLogger. If the log level is error, the logger will
    additionally output the location of the code.

    Args:
        color (bool): Whether to use colorful format. filehandler is not
            allowed to use color format, otherwise it will be garbled.
        blink (bool): Whether to blink the ``INFO`` and ``DEBUG`` logging
            level.
        **kwargs: Keyword arguments passed to
            :meth:`logging.Formatter.__init__`.
    """
    _color_mapping: 'dict' = dict(ERROR='red', WARNING='yellow', INFO='white', DEBUG='green')

    def __init__(self, color: 'bool'=True, blink: 'bool'=False, **kwargs):
        super().__init__(**kwargs)
        assert not (not color and blink), 'blink should only be available when color is True'
        error_prefix = self._get_prefix('ERROR', color, blink=True)
        warn_prefix = self._get_prefix('WARNING', color, blink=True)
        info_prefix = self._get_prefix('INFO', color, blink)
        debug_prefix = self._get_prefix('DEBUG', color, blink)
        self.err_format = f'%(asctime)s - %(name)s - {error_prefix} - %(pathname)s - %(funcName)s - %(lineno)d - %(message)s'
        self.warn_format = f'%(asctime)s - %(name)s - {warn_prefix} - %(message)s'
        self.info_format = f'%(asctime)s - %(name)s - {info_prefix} - %(message)s'
        self.debug_format = f'%(asctime)s - %(name)s - {debug_prefix} - %(message)s'

    def _get_prefix(self, level: 'str', color: 'bool', blink=False) ->str:
        """Get the prefix of the target log level.

        Args:
            level (str): log level.
            color (bool): Whether to get colorful prefix.
            blink (bool): Whether the prefix will blink.

        Returns:
            str: The plain or colorful prefix.
        """
        if color:
            attrs = ['underline']
            if blink:
                attrs.append('blink')
            prefix = colored(level, self._color_mapping[level], attrs=attrs)
        else:
            prefix = level
        return prefix

    def format(self, record: 'LogRecord') ->str:
        """Override the `logging.Formatter.format`` method `. Output the
        message according to the specified log level.

        Args:
            record (LogRecord): A LogRecord instance represents an event being
                logged.

        Returns:
            str: Formatted result.
        """
        if record.levelno == logging.ERROR:
            self._style._fmt = self.err_format
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warn_format
        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_format
        elif record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_format
        result = logging.Formatter.format(self, record)
        return result


class ManagerMeta(type):
    """The metaclass for global accessible class.

    The subclasses inheriting from ``ManagerMeta`` will manage their
    own ``_instance_dict`` and root instances. The constructors of subclasses
    must contain the ``name`` argument.

    Examples:
        >>> class SubClass1(metaclass=ManagerMeta):
        >>>     def __init__(self, *args, **kwargs):
        >>>         pass
        AssertionError: <class '__main__.SubClass1'>.__init__ must have the
        name argument.
        >>> class SubClass2(metaclass=ManagerMeta):
        >>>     def __init__(self, name):
        >>>         pass
        >>> # valid format.
    """

    def __init__(cls, *args):
        cls._instance_dict = OrderedDict()
        params = inspect.getfullargspec(cls)
        params_names = params[0] if params[0] else []
        assert 'name' in params_names, f'{cls} must have the `name` argument'
        super().__init__(*args)


T = TypeVar('T', bound='JitModelAnalysis')


def _accquire_lock() ->None:
    """Acquire the module-level lock for serializing access to shared data.

    This should be released with _release_lock().
    """
    if _lock:
        _lock.acquire()


def _release_lock() ->None:
    """Release the module-level lock acquired by calling _accquire_lock()."""
    if _lock:
        _lock.release()


class ManagerMixin(metaclass=ManagerMeta):
    """``ManagerMixin`` is the base class for classes that have global access
    requirements.

    The subclasses inheriting from ``ManagerMixin`` can get their
    global instances.

    Examples:
        >>> class GlobalAccessible(ManagerMixin):
        >>>     def __init__(self, name=''):
        >>>         super().__init__(name)
        >>>
        >>> GlobalAccessible.get_instance('name')
        >>> instance_1 = GlobalAccessible.get_instance('name')
        >>> instance_2 = GlobalAccessible.get_instance('name')
        >>> assert id(instance_1) == id(instance_2)

    Args:
        name (str): Name of the instance. Defaults to ''.
    """

    def __init__(self, name: 'str'='', **kwargs):
        assert isinstance(name, str) and name, 'name argument must be an non-empty string.'
        self._instance_name = name

    @classmethod
    def get_instance(cls: 'Type[T]', name: 'str', **kwargs) ->T:
        """Get subclass instance by name if the name exists.

        If corresponding name instance has not been created, ``get_instance``
        will create an instance, otherwise ``get_instance`` will return the
        corresponding instance.

        Examples
            >>> instance1 = GlobalAccessible.get_instance('name1')
            >>> # Create name1 instance.
            >>> instance.instance_name
            name1
            >>> instance2 = GlobalAccessible.get_instance('name1')
            >>> # Get name1 instance.
            >>> assert id(instance1) == id(instance2)

        Args:
            name (str): Name of instance. Defaults to ''.

        Returns:
            object: Corresponding name instance, the latest instance, or root
            instance.
        """
        _accquire_lock()
        assert isinstance(name, str), f'type of name should be str, but got {type(cls)}'
        instance_dict = cls._instance_dict
        if name not in instance_dict:
            instance = cls(name=name, **kwargs)
            instance_dict[name] = instance
        elif kwargs:
            warnings.warn(f'{cls} instance named of {name} has been created, the method `get_instance` should not accept any other arguments')
        _release_lock()
        return instance_dict[name]

    @classmethod
    def get_current_instance(cls):
        """Get latest created instance.

        Before calling ``get_current_instance``, The subclass must have called
        ``get_instance(xxx)`` at least once.

        Examples
            >>> instance = GlobalAccessible.get_current_instance()
            AssertionError: At least one of name and current needs to be set
            >>> instance = GlobalAccessible.get_instance('name1')
            >>> instance.instance_name
            name1
            >>> instance = GlobalAccessible.get_current_instance()
            >>> instance.instance_name
            name1

        Returns:
            object: Latest created instance.
        """
        _accquire_lock()
        if not cls._instance_dict:
            raise RuntimeError(f'Before calling {cls.__name__}.get_current_instance(), you should call get_instance(name=xxx) at least once.')
        name = next(iter(reversed(cls._instance_dict)))
        _release_lock()
        return cls._instance_dict[name]

    @classmethod
    def check_instance_created(cls, name: 'str') ->bool:
        """Check whether the name corresponding instance exists.

        Args:
            name (str): Name of instance.

        Returns:
            bool: Whether the name corresponding instance exists.
        """
        return name in cls._instance_dict

    @property
    def instance_name(self) ->str:
        """Get the name of instance.

        Returns:
            str: Name of instance.
        """
        return self._instance_name


def _get_device_id():
    """Get device id of current machine."""
    try:
        import torch
    except ImportError:
        return 0
    else:
        MUSA_AVAILABLE = False
        try:
            MUSA_AVAILABLE = True
        except ImportError:
            pass
        if MUSA_AVAILABLE:
            local_rank = int(os.getenv('LOCAL_RANK', '0'))
            musa_visible_devices = os.getenv('MUSA_VISIBLE_DEVICES', None)
            if musa_visible_devices is None:
                num_device = torch_musa.device_count()
                musa_visible_devices = list(range(num_device))
            else:
                musa_visible_devices = musa_visible_devices.split(',')
            return int(musa_visible_devices[local_rank])
        else:
            local_rank = int(os.getenv('LOCAL_RANK', '0'))
            if not torch.cuda.is_available():
                return local_rank
            cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', None)
            if cuda_visible_devices is None:
                num_device = torch.cuda.device_count()
                cuda_visible_devices = list(range(num_device))
            else:
                cuda_visible_devices = cuda_visible_devices.split(',')
            try:
                return int(cuda_visible_devices[local_rank])
            except ValueError:
                return cuda_visible_devices[local_rank]


def _get_host_info() ->str:
    """Get hostname and username.

    Return empty string if exception raised, e.g. ``getpass.getuser()`` will
    lead to error in docker container
    """
    host = ''
    try:
        host = f'{getuser()}@{gethostname()}'
    except Exception as e:
        warnings.warn(f'Host or user not found: {str(e)}')
    return host


def _get_logging_file_handlers() ->Dict:
    """Get additional file_handlers in ``logging.handlers``.

    Returns:
        Dict: A map of file_handlers.
    """
    file_handlers_map = {}
    for module_name in dir(handlers):
        if module_name.startswith('__'):
            continue
        _fh = getattr(handlers, module_name)
        if inspect.isclass(_fh) and issubclass(_fh, logging.FileHandler):
            file_handlers_map[module_name] = _fh
    return file_handlers_map


def get_default_group() ->Optional[ProcessGroup]:
    """Return default process group."""
    return torch_dist.distributed_c10d._get_default_group()


def is_distributed() ->bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()


def get_rank(group: 'Optional[ProcessGroup]'=None) ->int:
    """Return the rank of the given process group.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Note:
        Calling ``get_rank`` in non-distributed environment will return 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the rank of the process group if in distributed
        environment, otherwise 0.
    """
    if is_distributed():
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0


def get_world_size(group: 'Optional[ProcessGroup]'=None) ->int:
    """Return the number of the given process group.

    Note:
        Calling ``get_world_size`` in non-distributed environment will return
        1.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the number of processes of the given process group if in
        distributed environment, otherwise 1.
    """
    if is_distributed():
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1


class MMLogger(Logger, ManagerMixin):
    """Formatted logger used to record messages.

    ``MMLogger`` can create formatted logger to log message with different
    log levels and get instance in the same way as ``ManagerMixin``.
    ``MMLogger`` has the following features:

    - Distributed log storage, ``MMLogger`` can choose whether to save log of
      different ranks according to `log_file`.
    - Message with different log levels will have different colors and format
      when displayed on terminal.

    Note:
        - The `name` of logger and the ``instance_name`` of ``MMLogger`` could
          be different. We can only get ``MMLogger`` instance by
          ``MMLogger.get_instance`` but not ``logging.getLogger``. This feature
          ensures ``MMLogger`` will not be incluenced by third-party logging
          config.
        - Different from ``logging.Logger``, ``MMLogger`` will not log warning
          or error message without ``Handler``.

    Examples:
        >>> logger = MMLogger.get_instance(name='MMLogger',
        >>>                                logger_name='Logger')
        >>> # Although logger has name attribute just like `logging.Logger`
        >>> # We cannot get logger instance by `logging.getLogger`.
        >>> assert logger.name == 'Logger'
        >>> assert logger.instance_name = 'MMLogger'
        >>> assert id(logger) != id(logging.getLogger('Logger'))
        >>> # Get logger that do not store logs.
        >>> logger1 = MMLogger.get_instance('logger1')
        >>> # Get logger only save rank0 logs.
        >>> logger2 = MMLogger.get_instance('logger2', log_file='out.log')
        >>> # Get logger only save multiple ranks logs.
        >>> logger3 = MMLogger.get_instance('logger3', log_file='out.log',
        >>>                                 distributed=True)

    Args:
        name (str): Global instance name.
        logger_name (str): ``name`` attribute of ``Logging.Logger`` instance.
            If `logger_name` is not defined, defaults to 'mmengine'.
        log_file (str, optional): The log filename. If specified, a
            ``FileHandler`` will be added to the logger. Defaults to None.
        log_level (str): The log level of the handler. Defaults to
            'INFO'. If log level is 'DEBUG', distributed logs will be saved
            during distributed training.
        file_mode (str): The file mode used to open log file. Defaults to 'w'.
        distributed (bool): Whether to save distributed logs, Defaults to
            false.
        file_handler_cfg (dict, optional): Configuration of file handler.
            Defaults to None. If ``file_handler_cfg`` is not specified,
            ``logging.FileHandler`` will be used by default. If it is
            specified, the ``type`` key should be set. It can be
            ``RotatingFileHandler``, ``TimedRotatingFileHandler``,
            ``WatchedFileHandler`` or other file handlers, and the remaining
            fields will be used to build the handler.

            Examples:
                >>> file_handler_cfg = dict(
                >>>    type='TimedRotatingFileHandler',
                >>>    when='MIDNIGHT',
                >>>    interval=1,
                >>>    backupCount=365)

            `New in version 0.9.0.`
    """

    def __init__(self, name: 'str', logger_name='mmengine', log_file: 'Optional[str]'=None, log_level: 'Union[int, str]'='INFO', file_mode: 'str'='w', distributed=False, file_handler_cfg: 'Optional[dict]'=None):
        Logger.__init__(self, logger_name)
        ManagerMixin.__init__(self, name)
        if isinstance(log_level, str):
            log_level = logging._nameToLevel[log_level]
        global_rank = _get_rank()
        device_id = _get_device_id()
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(MMFormatter(color=True, datefmt='%m/%d %H:%M:%S'))
        if global_rank == 0:
            stream_handler.setLevel(log_level)
        else:
            stream_handler.setLevel(logging.ERROR)
        stream_handler.addFilter(FilterDuplicateWarning(logger_name))
        self.handlers.append(stream_handler)
        if log_file is not None:
            world_size = _get_world_size()
            is_distributed = (log_level <= logging.DEBUG or distributed) and world_size > 1
            if is_distributed:
                filename, suffix = osp.splitext(osp.basename(log_file))
                hostname = _get_host_info()
                if hostname:
                    filename = f'{filename}_{hostname}_device{device_id}_rank{global_rank}{suffix}'
                else:
                    filename = f'{filename}_device{device_id}_rank{global_rank}{suffix}'
                log_file = osp.join(osp.dirname(log_file), filename)
            if global_rank == 0 or is_distributed:
                if file_handler_cfg is not None:
                    assert 'type' in file_handler_cfg
                    file_handler_type = file_handler_cfg.pop('type')
                    file_handlers_map = _get_logging_file_handlers()
                    if file_handler_type in file_handlers_map:
                        file_handler_cls = file_handlers_map[file_handler_type]
                        file_handler_cfg.setdefault('filename', log_file)
                        file_handler = file_handler_cls(**file_handler_cfg)
                    else:
                        raise ValueError(f'`logging.handlers` does not contain {file_handler_type}')
                else:
                    file_handler = logging.FileHandler(log_file, file_mode)
                file_handler.setFormatter(MMFormatter(color=False, datefmt='%Y/%m/%d %H:%M:%S'))
                file_handler.setLevel(log_level)
                file_handler.addFilter(FilterDuplicateWarning(logger_name))
                self.handlers.append(file_handler)
        self._log_file = log_file

    @property
    def log_file(self):
        return self._log_file

    @classmethod
    def get_current_instance(cls) ->'MMLogger':
        """Get latest created ``MMLogger`` instance.

        :obj:`MMLogger` can call :meth:`get_current_instance` before any
        instance has been created, and return a logger with the instance name
        "mmengine".

        Returns:
            MMLogger: Configured logger instance.
        """
        if not cls._instance_dict:
            cls.get_instance('mmengine')
        return super().get_current_instance()

    def callHandlers(self, record: 'LogRecord') ->None:
        """Pass a record to all relevant handlers.

        Override ``callHandlers`` method in ``logging.Logger`` to avoid
        multiple warning messages in DDP mode. Loop through all handlers of
        the logger instance and its parents in the logger hierarchy. If no
        handler was found, the record will not be output.

        Args:
            record (LogRecord): A ``LogRecord`` instance contains logged
                message.
        """
        for handler in self.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    def setLevel(self, level):
        """Set the logging level of this logger.

        If ``logging.Logger.selLevel`` is called, all ``logging.Logger``
        instances managed by ``logging.Manager`` will clear the cache. Since
        ``MMLogger`` is not managed by ``logging.Manager`` anymore,
        ``MMLogger`` should override this method to clear caches of all
        ``MMLogger`` instance which is managed by :obj:`ManagerMixin`.

        level must be an int or a str.
        """
        self.level = logging._checkLevel(level)
        _accquire_lock()
        for logger in MMLogger._instance_dict.values():
            logger._cache.clear()
        _release_lock()


def print_log(msg, logger: 'Optional[Union[Logger, str]]'=None, level=logging.INFO) ->None:
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (Logger or str, optional): If the type of logger is
        ``logging.Logger``, we directly use logger to log messages.
            Some special loggers are:

            - "silent": No message will be printed.
            - "current": Use latest created logger to log message.
            - other str: Instance name of logger. The corresponding logger
              will log message if it has been created, otherwise ``print_log``
              will raise a `ValueError`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object, "current", or a created logger instance name.
    """
    if logger is None:
        None
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif logger == 'current':
        logger_instance = MMLogger.get_current_instance()
        logger_instance.log(level, msg)
    elif isinstance(logger, str):
        if MMLogger.check_instance_created(logger):
            logger_instance = MMLogger.get_instance(logger)
            logger_instance.log(level, msg)
        else:
            raise ValueError(f'MMLogger: {logger} has not been created!')
    else:
        raise TypeError(f'`logger` should be either a logging.Logger object, str, "silent", "current" or None, but got {type(logger)}')


class ExponentialMovingAverage(BaseAveragedModel):
    """Implements the exponential moving average (EMA) of the model.

    All parameters are updated by the formula as below:

        .. math::

            Xema_{t+1} = (1 - momentum) * Xema_{t} +  momentum * X_t

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically,
        :math:`Xema_{t+1}` is the moving average and :math:`X_t` is the
        new observed value. The value of momentum is usually a small number,
        allowing observed values to slowly update the ema parameters.

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
            Ema's parameter are updated with the formula
            :math:`averaged\\_param = (1-momentum) * averaged\\_param +
            momentum * source\\_param`.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(self, model: 'nn.Module', momentum: 'float'=0.0002, interval: 'int'=1, device: 'Optional[torch.device]'=None, update_buffers: 'bool'=False) ->None:
        super().__init__(model, interval, device, update_buffers)
        assert 0.0 < momentum < 1.0, f'momentum must be in range (0.0, 1.0)but got {momentum}'
        if momentum > 0.5:
            print_log(f'The value of momentum in EMA is usually a small number,which is different from the conventional notion of momentum but got {momentum}. Please make sure the value is correct.', logger='current', level=logging.WARNING)
        self.momentum = momentum

    def avg_func(self, averaged_param: 'Tensor', source_param: 'Tensor', steps: 'int') ->None:
        """Compute the moving average of the parameters using exponential
        moving average.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        averaged_param.lerp_(source_param, self.momentum)


class MomentumAnnealingEMA(ExponentialMovingAverage):
    """Exponential moving average (EMA) with momentum annealing strategy.

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
            Ema's parameter are updated with the formula
            :math:`averaged\\_param = (1-momentum) * averaged\\_param +
            momentum * source\\_param`.
        gamma (int): Use a larger momentum early in training and gradually
            annealing to a smaller value to update the ema model smoothly. The
            momentum is calculated as max(momentum, gamma / (gamma + steps))
            Defaults to 100.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(self, model: 'nn.Module', momentum: 'float'=0.0002, gamma: 'int'=100, interval: 'int'=1, device: 'Optional[torch.device]'=None, update_buffers: 'bool'=False) ->None:
        super().__init__(model=model, momentum=momentum, interval=interval, device=device, update_buffers=update_buffers)
        assert gamma > 0, f'gamma must be greater than 0, but got {gamma}'
        self.gamma = gamma

    def avg_func(self, averaged_param: 'Tensor', source_param: 'Tensor', steps: 'int') ->None:
        """Compute the moving average of the parameters using the linear
        momentum strategy.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        momentum = max(self.momentum, self.gamma / (self.gamma + self.steps.item()))
        averaged_param.lerp_(source_param, momentum)


class BaseDataElement:
    """A base data interface that supports Tensor-like and dict-like
    operations.

    A typical data elements refer to predicted results or ground truth labels
    on a task, such as predicted bboxes, instance masks, semantic
    segmentation masks, etc. Because groundtruth labels and predicted results
    often have similar properties (for example, the predicted bboxes and the
    groundtruth bboxes), MMEngine uses the same abstract data interface to
    encapsulate predicted results and groundtruth labels, and it is recommended
    to use different name conventions to distinguish them, such as using
    ``gt_instances`` and ``pred_instances`` to distinguish between labels and
    predicted results. Additionally, we distinguish data elements at instance
    level, pixel level, and label level. Each of these types has its own
    characteristics. Therefore, MMEngine defines the base class
    ``BaseDataElement``, and implement ``InstanceData``, ``PixelData``, and
    ``LabelData`` inheriting from ``BaseDataElement`` to represent different
    types of ground truth labels or predictions.

    Another common data element is sample data. A sample data consists of input
    data (such as an image) and its annotations and predictions. In general,
    an image can have multiple types of annotations and/or predictions at the
    same time (for example, both pixel-level semantic segmentation annotations
    and instance-level detection bboxes annotations). All labels and
    predictions of a training sample are often passed between Dataset, Model,
    Visualizer, and Evaluator components. In order to simplify the interface
    between components, we can treat them as a large data element and
    encapsulate them. Such data elements are generally called XXDataSample in
    the OpenMMLab. Therefore, Similar to `nn.Module`, the `BaseDataElement`
    allows `BaseDataElement` as its attribute. Such a class generally
    encapsulates all the data of a sample in the algorithm library, and its
    attributes generally are various types of data elements. For example,
    MMDetection is assigned by the BaseDataElement to encapsulate all the data
    elements of the sample labeling and prediction of a sample in the
    algorithm library.

    The attributes in ``BaseDataElement`` are divided into two parts,
    the ``metainfo`` and the ``data`` respectively.

        - ``metainfo``: Usually contains the
          information about the image such as filename,
          image_shape, pad_shape, etc. The attributes can be accessed or
          modified by dict-like or object-like operations, such as
          ``.`` (for data access and modification), ``in``, ``del``,
          ``pop(str)``, ``get(str)``, ``metainfo_keys()``,
          ``metainfo_values()``, ``metainfo_items()``, ``set_metainfo()`` (for
          set or change key-value pairs in metainfo).

        - ``data``: Annotations or model predictions are
          stored. The attributes can be accessed or modified by
          dict-like or object-like operations, such as
          ``.``, ``in``, ``del``, ``pop(str)``, ``get(str)``, ``keys()``,
          ``values()``, ``items()``. Users can also apply tensor-like
          methods to all :obj:`torch.Tensor` in the ``data_fields``,
          such as ``.cuda()``, ``.cpu()``, ``.numpy()``, ``.to()``,
          ``to_tensor()``, ``.detach()``.

    Args:
        metainfo (dict, optional): A dict contains the meta information
            of single image, such as ``dict(img_shape=(512, 512, 3),
            scale_factor=(1, 1, 1, 1))``. Defaults to None.
        kwargs (dict, optional): A dict contains annotations of single image or
            model predictions. Defaults to None.

    Examples:
        >>> import torch
        >>> from mmengine.structures import BaseDataElement
        >>> gt_instances = BaseDataElement()
        >>> bboxes = torch.rand((5, 4))
        >>> scores = torch.rand((5,))
        >>> img_id = 0
        >>> img_shape = (800, 1333)
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=img_id, img_shape=img_shape),
        ...     bboxes=bboxes, scores=scores)
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=img_id, img_shape=(640, 640)))

        >>> # new
        >>> gt_instances1 = gt_instances.new(
        ...     metainfo=dict(img_id=1, img_shape=(640, 640)),
        ...                   bboxes=torch.rand((5, 4)),
        ...                   scores=torch.rand((5,)))
        >>> gt_instances2 = gt_instances1.new()

        >>> # add and process property
        >>> gt_instances = BaseDataElement()
        >>> gt_instances.set_metainfo(dict(img_id=9, img_shape=(100, 100)))
        >>> assert 'img_shape' in gt_instances.metainfo_keys()
        >>> assert 'img_shape' in gt_instances
        >>> assert 'img_shape' not in gt_instances.keys()
        >>> assert 'img_shape' in gt_instances.all_keys()
        >>> print(gt_instances.img_shape)
        (100, 100)
        >>> gt_instances.scores = torch.rand((5,))
        >>> assert 'scores' in gt_instances.keys()
        >>> assert 'scores' in gt_instances
        >>> assert 'scores' in gt_instances.all_keys()
        >>> assert 'scores' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.scores)
        tensor([0.5230, 0.7885, 0.2426, 0.3911, 0.4876])
        >>> gt_instances.bboxes = torch.rand((5, 4))
        >>> assert 'bboxes' in gt_instances.keys()
        >>> assert 'bboxes' in gt_instances
        >>> assert 'bboxes' in gt_instances.all_keys()
        >>> assert 'bboxes' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.bboxes)
        tensor([[0.0900, 0.0424, 0.1755, 0.4469],
                [0.8648, 0.0592, 0.3484, 0.0913],
                [0.5808, 0.1909, 0.6165, 0.7088],
                [0.5490, 0.4209, 0.9416, 0.2374],
                [0.3652, 0.1218, 0.8805, 0.7523]])

        >>> # delete and change property
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=0, img_shape=(640, 640)),
        ...     bboxes=torch.rand((6, 4)), scores=torch.rand((6,)))
        >>> gt_instances.set_metainfo(dict(img_shape=(1280, 1280)))
        >>> gt_instances.img_shape  # (1280, 1280)
        >>> gt_instances.bboxes = gt_instances.bboxes * 2
        >>> gt_instances.get('img_shape', None)  # (1280, 1280)
        >>> gt_instances.get('bboxes', None)  # 6x4 tensor
        >>> del gt_instances.img_shape
        >>> del gt_instances.bboxes
        >>> assert 'img_shape' not in gt_instances
        >>> assert 'bboxes' not in gt_instances
        >>> gt_instances.pop('img_shape', None)  # None
        >>> gt_instances.pop('bboxes', None)  # None

        >>> # Tensor-like
        >>> cuda_instances = gt_instances.cuda()
        >>> cuda_instances = gt_instances.to('cuda:0')
        >>> cpu_instances = cuda_instances.cpu()
        >>> cpu_instances = cuda_instances.to('cpu')
        >>> fp16_instances = cuda_instances.to(
        ...     device=None, dtype=torch.float16, non_blocking=False,
        ...     copy=False, memory_format=torch.preserve_format)
        >>> cpu_instances = cuda_instances.detach()
        >>> np_instances = cpu_instances.numpy()

        >>> # print
        >>> metainfo = dict(img_shape=(800, 1196, 3))
        >>> gt_instances = BaseDataElement(
        ...     metainfo=metainfo, det_labels=torch.LongTensor([0, 1, 2, 3]))
        >>> sample = BaseDataElement(metainfo=metainfo,
        ...                          gt_instances=gt_instances)
        >>> print(sample)
        <BaseDataElement(
            META INFORMATION
            img_shape: (800, 1196, 3)
            DATA FIELDS
            gt_instances: <BaseDataElement(
                    META INFORMATION
                    img_shape: (800, 1196, 3)
                    DATA FIELDS
                    det_labels: tensor([0, 1, 2, 3])
                ) at 0x7f0ec5eadc70>
        ) at 0x7f0fea49e130>

        >>> # inheritance
        >>> class DetDataSample(BaseDataElement):
        ...     @property
        ...     def proposals(self):
        ...         return self._proposals
        ...     @proposals.setter
        ...     def proposals(self, value):
        ...         self.set_field(value, '_proposals', dtype=BaseDataElement)
        ...     @proposals.deleter
        ...     def proposals(self):
        ...         del self._proposals
        ...     @property
        ...     def gt_instances(self):
        ...         return self._gt_instances
        ...     @gt_instances.setter
        ...     def gt_instances(self, value):
        ...         self.set_field(value, '_gt_instances',
        ...                        dtype=BaseDataElement)
        ...     @gt_instances.deleter
        ...     def gt_instances(self):
        ...         del self._gt_instances
        ...     @property
        ...     def pred_instances(self):
        ...         return self._pred_instances
        ...     @pred_instances.setter
        ...     def pred_instances(self, value):
        ...         self.set_field(value, '_pred_instances',
        ...                        dtype=BaseDataElement)
        ...     @pred_instances.deleter
        ...     def pred_instances(self):
        ...         del self._pred_instances
        >>> det_sample = DetDataSample()
        >>> proposals = BaseDataElement(bboxes=torch.rand((5, 4)))
        >>> det_sample.proposals = proposals
        >>> assert 'proposals' in det_sample
        >>> assert det_sample.proposals == proposals
        >>> del det_sample.proposals
        >>> assert 'proposals' not in det_sample
        >>> with self.assertRaises(AssertionError):
        ...     det_sample.proposals = torch.rand((5, 4))
    """

    def __init__(self, *, metainfo: Optional[dict]=None, **kwargs) ->None:
        self._metainfo_fields: 'set' = set()
        self._data_fields: 'set' = set()
        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if kwargs:
            self.set_data(kwargs)

    def set_metainfo(self, metainfo: 'dict') ->None:
        """Set or change key-value pairs in ``metainfo_field`` by parameter
        ``metainfo``.

        Args:
            metainfo (dict): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
        """
        assert isinstance(metainfo, dict), f'metainfo should be a ``dict`` but got {type(metainfo)}'
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            self.set_field(name=k, value=v, field_type='metainfo', dtype=None)

    def set_data(self, data: 'dict') ->None:
        """Set or change key-value pairs in ``data_field`` by parameter
        ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions.
        """
        assert isinstance(data, dict), f'data should be a `dict` but got {data}'
        for k, v in data.items():
            setattr(self, k, v)

    def update(self, instance: "'BaseDataElement'") ->None:
        """The update() method updates the BaseDataElement with the elements
        from another BaseDataElement object.

        Args:
            instance (BaseDataElement): Another BaseDataElement object for
                update the current object.
        """
        assert isinstance(instance, BaseDataElement), f'instance should be a `BaseDataElement` but got {type(instance)}'
        self.set_metainfo(dict(instance.metainfo_items()))
        self.set_data(dict(instance.items()))

    def new(self, *, metainfo: Optional[dict]=None, **kwargs) ->'BaseDataElement':
        """Return a new data element with same type. If ``metainfo`` and
        ``data`` are None, the new data element will have same metainfo and
        data. If metainfo or data is not None, the new result will overwrite it
        with the input value.

        Args:
            metainfo (dict, optional): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
                Defaults to None.
            kwargs (dict): A dict contains annotations of image or
                model predictions.

        Returns:
            BaseDataElement: A new data element with same type.
        """
        new_data = self.__class__()
        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))
        if kwargs:
            new_data.set_data(kwargs)
        else:
            new_data.set_data(dict(self.items()))
        return new_data

    def clone(self):
        """Deep copy the current data element.

        Returns:
            BaseDataElement: The copy of current data element.
        """
        clone_data = self.__class__()
        clone_data.set_metainfo(dict(self.metainfo_items()))
        clone_data.set_data(dict(self.items()))
        return clone_data

    def keys(self) ->list:
        """
        Returns:
            list: Contains all keys in data_fields.
        """
        private_keys = {('_' + key) for key in self._data_fields if isinstance(getattr(type(self), key, None), property)}
        return list(self._data_fields - private_keys)

    def metainfo_keys(self) ->list:
        """
        Returns:
            list: Contains all keys in metainfo_fields.
        """
        return list(self._metainfo_fields)

    def values(self) ->list:
        """
        Returns:
            list: Contains all values in data.
        """
        return [getattr(self, k) for k in self.keys()]

    def metainfo_values(self) ->list:
        """
        Returns:
            list: Contains all values in metainfo.
        """
        return [getattr(self, k) for k in self.metainfo_keys()]

    def all_keys(self) ->list:
        """
        Returns:
            list: Contains all keys in metainfo and data.
        """
        return self.metainfo_keys() + self.keys()

    def all_values(self) ->list:
        """
        Returns:
            list: Contains all values in metainfo and data.
        """
        return self.metainfo_values() + self.values()

    def all_items(self) ->Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo`` and ``data``.
        """
        for k in self.all_keys():
            yield k, getattr(self, k)

    def items(self) ->Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``data``.
        """
        for k in self.keys():
            yield k, getattr(self, k)

    def metainfo_items(self) ->Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo``.
        """
        for k in self.metainfo_keys():
            yield k, getattr(self, k)

    @property
    def metainfo(self) ->dict:
        """dict: A dict contains metainfo of current data element."""
        return dict(self.metainfo_items())

    def __setattr__(self, name: 'str', value: 'Any'):
        """Setattr is only used to set data."""
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a private attribute, which is immutable.')
        else:
            self.set_field(name=name, value=value, field_type='data', dtype=None)

    def __delattr__(self, item: 'str'):
        """Delete the item in dataelement.

        Args:
            item (str): The key to delete.
        """
        if item in ('_metainfo_fields', '_data_fields'):
            raise AttributeError(f'{item} has been used as a private attribute, which is immutable.')
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        elif item in self._data_fields:
            self._data_fields.remove(item)
    __delitem__ = __delattr__

    def get(self, key, default=None) ->Any:
        """Get property in data and metainfo as the same as python."""
        return getattr(self, key, default)

    def pop(self, *args) ->Any:
        """Pop property in data and metainfo as the same as python."""
        assert len(args) < 3, '``pop`` get more than 2 arguments'
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(args[0])
            return self.__dict__.pop(*args)
        elif name in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)
        elif len(args) == 2:
            return args[1]
        else:
            raise KeyError(f'{args[0]} is not contained in metainfo or data')

    def __contains__(self, item: 'str') ->bool:
        """Whether the item is in dataelement.

        Args:
            item (str): The key to inquire.
        """
        return item in self._data_fields or item in self._metainfo_fields

    def set_field(self, value: 'Any', name: 'str', dtype: 'Optional[Union[Type, Tuple[Type, ...]]]'=None, field_type: 'str'='data') ->None:
        """Special method for set union field, used as property.setter
        functions."""
        assert field_type in ['metainfo', 'data']
        if dtype is not None:
            assert isinstance(value, dtype), f'{value} should be a {dtype} but got {type(value)}'
        if field_type == 'metainfo':
            if name in self._data_fields:
                raise AttributeError(f'Cannot set {name} to be a field of metainfo because {name} is already a data field')
            self._metainfo_fields.add(name)
        else:
            if name in self._metainfo_fields:
                raise AttributeError(f'Cannot set {name} to be a field of data because {name} is already a metainfo field')
            self._data_fields.add(name)
        super().__setattr__(name, value)

    def to(self, *args, **kwargs) ->'BaseDataElement':
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, 'to'):
                v = v
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def cpu(self) ->'BaseDataElement':
        """Convert all tensors to CPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cpu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def cuda(self) ->'BaseDataElement':
        """Convert all tensors to GPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def musa(self) ->'BaseDataElement':
        """Convert all tensors to musa in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.musa()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def npu(self) ->'BaseDataElement':
        """Convert all tensors to NPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.npu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def mlu(self) ->'BaseDataElement':
        """Convert all tensors to MLU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.mlu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def detach(self) ->'BaseDataElement':
        """Detach all tensors in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def numpy(self) ->'BaseDataElement':
        """Convert all tensors to np.ndarray in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach().cpu().numpy()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def to_tensor(self) ->'BaseDataElement':
        """Convert all np.ndarray to tensor in data."""
        new_data = self.new()
        for k, v in self.items():
            data = {}
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                data[k] = v
            elif isinstance(v, BaseDataElement):
                v = v.to_tensor()
                data[k] = v
            new_data.set_data(data)
        return new_data

    def to_dict(self) ->dict:
        """Convert BaseDataElement to dict."""
        return {k: (v.to_dict() if isinstance(v, BaseDataElement) else v) for k, v in self.all_items()}

    def __repr__(self) ->str:
        """Represent the object."""

        def _addindent(s_: 'str', num_spaces: 'int') ->str:
            """This func is modified from `pytorch` https://github.com/pytorch/
            pytorch/blob/b17b2b1cc7b017c3daaeff8cc7ec0f514d42ec37/torch/nn/modu
            les/module.py#L29.

            Args:
                s_ (str): The string to add spaces.
                num_spaces (int): The num of space to add.

            Returns:
                str: The string after add indent.
            """
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ' + line) for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def dump(obj: 'Any') ->str:
            """Represent the object.

            Args:
                obj (Any): The obj to represent.

            Returns:
                str: The represented str.
            """
            _repr = ''
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _repr += f'\n{k}: {_addindent(dump(v), 4)}'
            elif isinstance(obj, BaseDataElement):
                _repr += '\n\n    META INFORMATION'
                metainfo_items = dict(obj.metainfo_items())
                _repr += _addindent(dump(metainfo_items), 4)
                _repr += '\n\n    DATA FIELDS'
                items = dict(obj.items())
                _repr += _addindent(dump(items), 4)
                classname = obj.__class__.__name__
                _repr = f'<{classname}({_repr}\n) at {hex(id(obj))}>'
            else:
                _repr += repr(obj)
            return _repr
        return dump(self)


CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str, None]


class BaseDataPreprocessor(nn.Module):
    """Base data pre-processor used for copying data to the target device.

    Subclasses inherit from ``BaseDataPreprocessor`` could override the
    forward method to implement custom data pre-processing, such as
    batch-resize, MixUp, or CutMix.

    Args:
        non_blocking (bool): Whether block current process
            when transferring data to device.
            New in version 0.3.0.

    Note:
        Data dictionary returned by dataloader must be a dict and at least
        contain the ``inputs`` key.
    """

    def __init__(self, non_blocking: 'Optional[bool]'=False):
        super().__init__()
        self._non_blocking = non_blocking
        self._device = torch.device('cpu')

    def cast_data(self, data: 'CastData') ->CastData:
        """Copying data to the target device.

        Args:
            data (dict): Data returned by ``DataLoader``.

        Returns:
            CollatedResult: Inputs and data sample at target device.
        """
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, tuple) and hasattr(data, '_fields'):
            return type(data)(*(self.cast_data(sample) for sample in data))
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample) for sample in data)
        elif isinstance(data, (torch.Tensor, BaseDataElement)):
            return data
        else:
            return data

    def forward(self, data: 'dict', training: 'bool'=False) ->Union[dict, list]:
        """Preprocesses the data into the model input format.

        After the data pre-processing of :meth:`cast_data`, ``forward``
        will stack the input tensor list to a batch tensor at the first
        dimension.

        Args:
            data (dict): Data returned by dataloader
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        return self.cast_data(data)

    @property
    def device(self):
        return self._device

    def to(self, *args, **kwargs) ->nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        if args and isinstance(args[0], str) and 'npu' in args[0]:
            args = tuple([list(args)[0].replace('npu', torch.npu.native_device)])
        if kwargs and 'npu' in str(kwargs.get('device', '')):
            kwargs['device'] = kwargs['device'].replace('npu', torch.npu.native_device)
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
        return super()

    def cuda(self, *args, **kwargs) ->nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.cuda.current_device())
        return super()

    def musa(self, *args, **kwargs) ->nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.musa.current_device())
        return super().musa()

    def npu(self, *args, **kwargs) ->nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.npu.current_device())
        return super().npu()

    def mlu(self, *args, **kwargs) ->nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.mlu.current_device())
        return super().mlu()

    def cpu(self, *args, **kwargs) ->nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device('cpu')
        return super().cpu()


def is_seq_of(seq: 'Any', expected_type: 'Union[Type, tuple]', seq_type: 'Optional[Type]'=None) ->bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def stack_batch(tensor_list: 'List[torch.Tensor]', pad_size_divisor: 'int'=1, pad_value: 'Union[int, float]'=0) ->torch.Tensor:
    """Stack multiple tensors to form a batch and pad the tensor to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need to be divisible by 32. Defaults to 1
        pad_value (int, float): The padding value. Defaults to 0.

    Returns:
       Tensor: The n dim tensor.
    """
    assert isinstance(tensor_list, list), f'Expected input type to be list, but got {type(tensor_list)}'
    assert tensor_list, '`tensor_list` could not be an empty list'
    assert len({tensor.ndim for tensor in tensor_list}) == 1, f'Expected the dimensions of all tensors must be the same, but got {[tensor.ndim for tensor in tensor_list]}'
    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: 'torch.Tensor' = torch.Tensor([tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)


class ImgDataPreprocessor(BaseDataPreprocessor):
    """Image pre-processor for normalization and bgr to rgb conversion.

    Accepts the data sampled by the dataloader, and preprocesses it into the
    format of the model input. ``ImgDataPreprocessor`` provides the
    basic data pre-processing as follows

    - Collates and moves data to the target device.
    - Converts inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalizes image with defined std and mean.
    - Pads inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.

    For ``ImgDataPreprocessor``, the dimension of the single inputs must be
    (3, H, W).

    Note:
        ``ImgDataPreprocessor`` and its subclass is built in the
        constructor of :class:`BaseDataset`.

    Args:
        mean (Sequence[float or int], optional): The pixel mean of image
            channels. If ``bgr_to_rgb=True`` it means the mean value of R,
            G, B channels. If the length of `mean` is 1, it means all
            channels have the same mean value, or the input is a gray image.
            If it is not specified, images will not be normalized. Defaults
            None.
        std (Sequence[float or int], optional): The pixel standard deviation of
            image channels. If ``bgr_to_rgb=True`` it means the standard
            deviation of R, G, B channels. If the length of `std` is 1,
            it means all channels have the same standard deviation, or the
            input is a gray image.  If it is not specified, images will
            not be normalized. Defaults None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        non_blocking (bool): Whether block current process
            when transferring data to device.
            New in version v0.3.0.

    Note:
        if images do not need to be normalized, `std` and `mean` should be
        both set to None, otherwise both of them should be set to a tuple of
        corresponding values.
    """

    def __init__(self, mean: 'Optional[Sequence[Union[float, int]]]'=None, std: 'Optional[Sequence[Union[float, int]]]'=None, pad_size_divisor: 'int'=1, pad_value: 'Union[float, int]'=0, bgr_to_rgb: 'bool'=False, rgb_to_bgr: 'bool'=False, non_blocking: 'Optional[bool]'=False):
        super().__init__(non_blocking)
        assert not (bgr_to_rgb and rgb_to_bgr), '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time'
        assert (mean is None) == (std is None), 'mean and std should be both None or tuple'
        if mean is not None:
            assert len(mean) == 3 or len(mean) == 1, f'`mean` should have 1 or 3 values, to be compatible with RGB or gray image, but got {len(mean)} values'
            assert len(std) == 3 or len(std) == 1, f'`std` should have 1 or 3 values, to be compatible with RGB or gray image, but got {len(std)} values'
            self._enable_normalize = True
            self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std', torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

    def forward(self, data: 'dict', training: 'bool'=False) ->Union[dict, list]:
        """Performs normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataset. If the collate
                function of DataLoader is :obj:`pseudo_collate`, data will be a
                list of dict. If collate function is :obj:`default_collate`,
                data will be a tuple with batch input tensor and list of data
                samples.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        _batch_inputs = data['inputs']
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                _batch_input = _batch_input.float()
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input.dim() == 3 and _batch_input.shape[0] == 3, f'If the mean has 3 values, the input tensor should in shape of (3, H, W), but got the tensor with shape {_batch_input.shape}'
                    _batch_input = (_batch_input - self.mean) / self.std
                batch_inputs.append(_batch_input)
            batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor, self.pad_value)
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, f'The input of `ImgDataPreprocessor` should be a NCHW tensor or a list of tensor, but got a tensor with shape: {_batch_inputs.shape}'
            if self._channel_conversion:
                _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
            _batch_inputs = _batch_inputs.float()
            if self._enable_normalize:
                _batch_inputs = (_batch_inputs - self.mean) / self.std
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(_batch_inputs, (0, pad_w, 0, pad_h), 'constant', self.pad_value)
        else:
            raise TypeError(f'Output of `cast_data` should be a dict of list/tuple with inputs and data_samples, but got {type(data)}: {data}')
        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)
        return data


class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""
    _schemes: 'Dict[str, Callable]' = {}

    @classmethod
    def _register_scheme(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if prefix not in cls._schemes or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(f'{prefix} is already registered as a loader backend, add "force=True" if you want to override it')
        cls._schemes = OrderedDict(sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True))

    @classmethod
    def register_scheme(cls, prefixes, loader=None, force=False):
        """Register a loader to CheckpointLoader.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or list[str] or tuple[str]):
            The prefix of the registered loader.
            loader (function, optional): The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """
        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls
        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path):
        """Finds a loader that supports the given path. Falls back to the local
        loader if no other loader is found.

        Args:
            path (str): checkpoint path

        Returns:
            callable: checkpoint loader
        """
        for p in cls._schemes:
            if re.match(p, path) is not None:
                return cls._schemes[p]

    @classmethod
    def load_checkpoint(cls, filename, map_location=None, logger='current'):
        """Load checkpoint through URL scheme path.

        Args:
            filename (str): checkpoint file name with given prefix
            map_location (str, optional): Same as :func:`torch.load`.
                Defaults to None
            logger (str): The logger for message. Defaults to 'current'.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """
        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__
        print_log(f'Loads checkpoint by {class_name[10:]} backend from path: {filename}', logger=logger)
        return checkpoint_loader(filename, map_location)


def _load_checkpoint(filename, map_location=None, logger=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str, optional): Same as :func:`torch.load`.
           Defaults to None.
        logger (:mod:`logging.Logger`, optional): The logger for error message.
           Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint. It can be either an
        OrderedDict storing model weights or a dict containing other
        information, which depends on the checkpoint.
    """
    return CheckpointLoader.load_checkpoint(filename, map_location, logger)


def _load_checkpoint_with_prefix(prefix, filename, map_location=None):
    """Load partial pretrained model with specific prefix.

    Args:
        prefix (str): The prefix of sub-module.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`.
            Defaults to None.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)
    state_dict = {k[prefix_len:]: v for k, v in state_dict.items() if k.startswith(prefix)}
    assert state_dict, f'{prefix} is not in the pretrained model'
    return state_dict

