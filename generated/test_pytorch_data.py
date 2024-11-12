
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


import itertools


import random


import torch


import torch.distributed as dist


import torchvision


from torchvision.transforms import transforms


import time


import warnings


import torch.utils.data


from torch import nn


from collections import defaultdict


from collections import deque


from typing import Any


from typing import Callable


from typing import Dict


from typing import Iterable


from typing import Iterator


from typing import List


from typing import Optional


from typing import Tuple


from typing import TypeVar


from typing import Union


import numpy as np


from torch.utils.data import get_worker_info


from torch.utils.data.datapipes.dataframe.dataframes import CaptureLikeMock


from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper


import re


from torch.utils.data.datapipes.utils.decoder import imagehandler


from torch.utils.data.datapipes.utils.decoder import mathandler


import torchvision.datasets as datasets


import torchvision.datasets.folder


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


from torch.testing._internal.common_utils import TestCase


from torch.utils.data import DistributedSampler


from torch.utils.data import RandomSampler


from torch.testing._internal.common_utils import IS_WINDOWS


from torch.testing._internal.common_utils import TEST_CUDA


import functools


import math


import torch.utils.data.datapipes as dp


from torch import multiprocessing as mp


from torch._utils import ExceptionWrapper


from torch.testing._internal.common_device_type import instantiate_device_type_tests


from torch.testing._internal.common_utils import IS_CI


from torch.testing._internal.common_utils import IS_JETSON


from torch.testing._internal.common_utils import IS_MACOS


from torch.testing._internal.common_utils import IS_SANDCASTLE


from torch.testing._internal.common_utils import load_tests


from torch.testing._internal.common_utils import NO_MULTIPROCESSING_SPAWN


from torch.testing._internal.common_utils import parametrize


from torch.testing._internal.common_utils import run_tests


from torch.testing._internal.common_utils import skipIfNoDill


from torch.testing._internal.common_utils import skipIfRocm


from torch.testing._internal.common_utils import slowTest


from torch.testing._internal.common_utils import TEST_NUMPY


from torch.testing._internal.common_utils import TEST_WITH_ASAN


from torch.testing._internal.common_utils import TEST_WITH_TSAN


from torch.utils.data import _utils


from torch.utils.data import ChainDataset


from torch.utils.data import ConcatDataset


from torch.utils.data import Dataset


from torch.utils.data import IterableDataset


from torch.utils.data import IterDataPipe


from torch.utils.data import StackDataset


from torch.utils.data import Subset


from torch.utils.data import TensorDataset


from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL


from torch.utils.data.datapipes.iter import IterableWrapper


from torch.utils.data.dataset import random_split


from copy import deepcopy


from typing import Set


import torch.utils.data.datapipes.gen_pyi as core_gen_pyi


from torch.utils.data.datapipes.gen_pyi import gen_from_template


from torch.utils.data.datapipes.gen_pyi import get_method_definitions


import queue


import torch.multiprocessing as mp


from typing import Mapping


from torch.utils.data import Sampler


from typing import Literal


from typing import Protocol


import torch.multiprocessing


from torch.utils.data._utils.pin_memory import pin_memory


from typing import Sized


import torch.utils.data.sampler


from torch.utils.data.dataloader import _InfiniteConstantSampler


import collections


import logging


import torch.multiprocessing as multiprocessing


import torch.utils.data._utils.worker


import torch.utils.data.graph_settings


from torch.utils.data import MapDataPipe


from torch.utils.data import SequentialSampler


from torch.utils.data.dataloader import _BaseDataLoaderIter


from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper


from torch.utils.data.datapipes.datapipe import _MapDataPipeSerializationWrapper


from torch.utils.data.dataloader import _collate_fn_t


from torch.utils.data.dataloader import _DatasetKind


from torch.utils.data.dataloader import _sharding_worker_init_fn


from torch.utils.data.dataloader import _worker_init_fn_t


from torch.utils.data.dataloader import default_collate


from torch.utils.data.dataloader import default_convert


from torch.utils.data.dataloader import get_worker_info


from torch.utils.data._utils import HAS_NUMPY


from torch.utils.data._utils import signal_handling


from torch.utils.data._utils.worker import _generate_state


from torch.utils.data._utils.worker import _IterableDatasetStopIteration


from torch.utils.data._utils.worker import _ResumeIteration


from torch.utils.data._utils.worker import ManagerWatchdog


from torch.utils.data._utils.worker import WorkerInfo

