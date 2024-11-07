import sys
_module = sys.modules[__name__]
del sys
benchmark_array_xd = _module
benchmark_getitem_100B = _module
benchmark_indices_mapping = _module
benchmark_iterating = _module
benchmark_map_filter = _module
format = _module
utils = _module
_config = _module
setup = _module
datasets = _module
arrow_dataset = _module
arrow_reader = _module
arrow_writer = _module
builder = _module
combine = _module
commands = _module
convert = _module
convert_to_parquet = _module
datasets_cli = _module
delete_from_hub = _module
env = _module
test = _module
config = _module
data_files = _module
dataset_dict = _module
distributed = _module
download = _module
download_config = _module
download_manager = _module
streaming_download_manager = _module
exceptions = _module
features = _module
audio = _module
features = _module
image = _module
translation = _module
video = _module
filesystems = _module
compression = _module
fingerprint = _module
formatting = _module
formatting = _module
jax_formatter = _module
np_formatter = _module
polars_formatter = _module
tf_formatter = _module
torch_formatter = _module
hub = _module
info = _module
inspect = _module
io = _module
abc = _module
csv = _module
generator = _module
json = _module
parquet = _module
spark = _module
sql = _module
text = _module
iterable_dataset = _module
keyhash = _module
load = _module
naming = _module
packaged_modules = _module
arrow = _module
audiofolder = _module
cache = _module
folder_based_builder = _module
imagefolder = _module
pandas = _module
videofolder = _module
webdataset = _module
_tenbin = _module
webdataset = _module
xml = _module
parallel = _module
search = _module
splits = _module
streaming = _module
table = _module
_dataset_viewer = _module
_dill = _module
_filelock = _module
deprecation_utils = _module
doc_utils = _module
experimental = _module
extract = _module
file_utils = _module
filelock = _module
info_utils = _module
logging = _module
metadata = _module
patching = _module
py_utils = _module
resources = _module
sharding = _module
stratify = _module
tf_utils = _module
tqdm = _module
track = _module
typing = _module
version = _module
new_dataset_script = _module
tests = _module
_test_patching = _module
conftest = _module
test_test = _module
run_torch_distributed = _module
test_array_xd = _module
test_audio = _module
test_features = _module
test_image = _module
test_video = _module
fixtures = _module
files = _module
fsspec = _module
test_csv = _module
test_json = _module
test_parquet = _module
test_sql = _module
test_text = _module
test_arrow = _module
test_audiofolder = _module
test_cache = _module
test_folder_based_builder = _module
test_imagefolder = _module
test_pandas = _module
test_spark = _module
test_webdataset = _module
test_arrow_dataset = _module
test_arrow_reader = _module
test_arrow_writer = _module
test_builder = _module
test_data_files = _module
test_dataset_dict = _module
test_dataset_list = _module
test_distributed = _module
test_download_manager = _module
test_exceptions = _module
test_experimental = _module
test_extract = _module
test_file_utils = _module
test_filelock = _module
test_filesystem = _module
test_fingerprint = _module
test_formatting = _module
test_hub = _module
test_info = _module
test_info_utils = _module
test_inspect = _module
test_iterable_dataset = _module
test_load = _module
test_metadata_util = _module
test_offline_util = _module
test_parallel = _module
test_patching = _module
test_py_utils = _module
test_search = _module
test_sharding_utils = _module
test_splits = _module
test_streaming_download_manager = _module
test_table = _module
test_tqdm = _module
test_upstream_hub = _module
test_version = _module
release = _module

from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import copy


import itertools


import math


import re


import time


import warnings


from collections import Counter


from collections.abc import Mapping


from copy import deepcopy


from functools import partial


from functools import wraps


from math import ceil


from math import floor


from random import sample


from typing import TYPE_CHECKING


from typing import Any


from typing import BinaryIO


from typing import Callable


from typing import Dict


from typing import Iterable


from typing import Iterator


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


from typing import overload


from typing import Sequence as Sequence_


import numpy as np


import pandas as pd


import logging


from typing import Sequence


from collections.abc import Iterable


from collections.abc import Sequence as SequenceABC


from functools import reduce


from typing import ClassVar


from pandas.api.extensions import ExtensionArray as PandasExtensionArray


from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype


from typing import Type


from collections.abc import MutableMapping


from typing import Generic


from typing import TypeVar


from itertools import cycle


from itertools import islice


from types import CodeType


from types import FunctionType


from itertools import chain


from typing import Generator


import functools


import queue


import types


from queue import Empty


from typing import Set


import torch.utils.data


import numpy.testing as npt

