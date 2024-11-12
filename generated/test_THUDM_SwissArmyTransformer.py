
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


import torch.nn as nn


import torchvision


import torchvision.transforms as transforms


import torch.nn.functional as F


import math


import copy


import warnings


import re


from typing import Optional


from typing import Tuple


from typing import Union


from typing import List


from typing import Callable


from typing import Dict


from typing import Any


import random


from functools import partial


from torch.nn import CrossEntropyLoss


from copy import deepcopy


from torchvision import transforms


from torchvision.transforms.functional import InterpolationMode


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from torchvision.transforms import RandomCrop


import torchvision.transforms as T


import time


from torchvision.transforms import functional as F


from collections import defaultdict


from collections import deque


import torch.distributed as dist


import torch.utils.data


import torchvision.transforms.functional as F


import matplotlib.pyplot as plt


from torch.utils.data import DataLoader


from matplotlib.patches import Polygon


import torch.utils.checkpoint as checkpoint


from torch import nn


from itertools import repeat


from scipy.optimize import linear_sum_assignment


from torchvision.transforms import ToPILImage


from torchvision.ops.boxes import box_area


import numpy


from torch import Tensor


import pandas as pd


import logging


from torch.utils import data


from torch.utils.data import ChainDataset


from torch.utils.data import IterableDataset


from torch.utils.data import Dataset


from torchvision.utils import save_image


import inspect


from re import L


from torch.nn import functional as F


import torch.nn.init as init


from collections.abc import Iterable


from math import pi


from torch.nn.parameter import Parameter


from torch.optim import SGD


from torch.autograd import Function


from abc import ABC


from abc import abstractmethod


from torch.nn import Linear


from torchvision.utils import make_grid


from collections import namedtuple


import itertools


from torch.optim.lr_scheduler import _LRScheduler


from collections import OrderedDict


from typing import Mapping


from typing import NamedTuple


from typing import Set


from torch.utils.data import TensorDataset


MODEL_URLS = {'bert-base-uncased': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Fbert-base-uncased.zip&dl=1', 'bert-large-uncased': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Fbert-large-uncased.zip&dl=1', 'roberta-base': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Froberta-base.zip&dl=1', 'roberta-large': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Froberta-large.zip&dl=1', 'vit-base-patch16-224-in21k': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Fvit-base-patch16-224-in21k.zip&dl=1', 'deit-tiny': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Fdeit-tiny.zip&dl=1', 'deit-small': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Fdeit-small.zip&dl=1', 'deit-base': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Fdeit-base.zip&dl=1', 'cait-s24-224': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Fcait-s24-224.zip&dl=1', 'gpt2': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Fgpt2.zip&dl=1', 'chatglm-6b': 'r2://chatglm-6b.zip', 'chatglm2-6b': 'r2://chatglm2-6b.zip', 'chatglm3-6b': 'r2://chatglm3-6b.zip', 'chatglm3-6b-base': 'r2://chatglm3-6b-base.zip', 'chatglm3-6b-32k': 'r2://chatglm3-6b-32k.zip', 'glm4v-9b-chat': 'r2://glm4v-9b-chat.zip', 'eva02_L_pt_m38m_p14': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Feva02_L_pt_m38m_p14.zip&dl=1', 'llama-7b': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Fllama-7b.zip&dl=1', 'llama-13b': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Fllama-13b.zip&dl=1', 'llama-30b': 'r2://llama-30b', 'llama-65b': 'r2://llama-65b', 'clip': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Fclip.zip&dl=1', 'clip-vit-base-patch16': 'https://lfs.aminer.cn/misc/clip/clip-vit-base-patch16.zip', 'clip-vit-large-patch14': 'https://lfs.aminer.cn/misc/clip/clip-vit-large-patch14.zip', 'eva-clip-4b-14-x-drop-last-layer': 'r2://eva-clip-4b-14-x-drop-last-layer.zip', 'yolos-tiny': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Fyolos-tiny.zip&dl=1', 'mae-vit-base': 'https://cloud.tsinghua.edu.cn/d/dd80f9d39d454bc29ce4/files/?p=%2Fmae-vit-base.zip&dl=1', 'cogview-base': 'https://cloud.tsinghua.edu.cn/f/df21f6d4109b4285bfd9/?dl=1', 'glm-large-zh': 'https://lfs.aminer.cn/misc/cogview/glm/glm-large-zh.zip', 'glm-large-en-blank': 'https://lfs.aminer.cn/misc/cogview/glm/glm-large-en-blank.zip', 'glm-10b-en': 'https://lfs.aminer.cn/misc/cogview/glm/glm-10b-en.zip', 'glm-10b-zh': 'https://lfs.aminer.cn/misc/cogview/glm/glm-10b-zh.zip', 'gpt-neo-1.3b': 'https://cloud.tsinghua.edu.cn/f/22e87976b5b745ad90af/?dl=1', 'coglm': 'https://lfs.aminer.cn/misc/cogview/cogview2/coglm.zip', 'cogview2-dsr': 'https://lfs.aminer.cn/misc/cogview/cogview2/cogview2-dsr.zip', 'cogview2-itersr': 'https://lfs.aminer.cn/misc/cogview/cogview2/cogview2-itersr.zip', 'cogvideo-stage1': 'https://lfs.aminer.cn/misc/cogvideo/cogvideo-stage1.zip', 'cogvideo-stage2': 'https://lfs.aminer.cn/misc/cogvideo/cogvideo-stage2.zip', 'dpr-ctx_encoder-single-nq-base': 'https://cloud.tsinghua.edu.cn/f/e5475f1211a948708baa/?dl=1', 'dpr-question_encoder-single-nq-base': 'https://cloud.tsinghua.edu.cn/f/5c4aae7d11fc4c45a5bd/?dl=1', 'dpr-reader-single-nq-base': 'https://cloud.tsinghua.edu.cn/f/e169889ab40d4615a34d/?dl=1'}


class ProgressPercentage(object):
    """ Progress Class
    Class for calculating and displaying download progress
    """

    def __init__(self, client, bucket, filename):
        """ Initialize
        initialize with: file name, file size and lock.
        Set seen_so_far to 0. Set progress bar length
        """
        self._filename = filename
        self._size = client.head_object(Bucket=bucket, Key=filename)['ContentLength']
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self.prog_bar_len = 80

    def __call__(self, bytes_amount):
        """ Call
        When called, increments seen_so_far by bytes_amount,
        calculates percentage of seen_so_far/total file size 
        and prints progress bar.
        """
        with self._lock:
            self._seen_so_far += bytes_amount
            ratio = round(float(self._seen_so_far) / float(self._size) * (self.prog_bar_len - 6), 1)
            current_length = int(round(ratio))
            percentage = round(100 * ratio / (self.prog_bar_len - 6), 1)
            bars = '+' * current_length
            output = bars + ' ' * (self.prog_bar_len - current_length - len(str(percentage)) - 1) + str(percentage) + '% ' + self.convert_bytes(self._seen_so_far) + ' / ' + self.convert_bytes(self._size) + ' ' * 5
            if self._seen_so_far != self._size:
                sys.stdout.write(output + '\r')
            else:
                sys.stdout.write(output + '\n')
            sys.stdout.flush()

    def convert_bytes(self, num):
        """ Convert Bytes
        Converts bytes to scaled format (e.g KB, MB, etc.)
        """
        step_unit = 1000.0
        for x in ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB']:
            if num < step_unit:
                return '%3.1f %s' % (num, x)
            num /= step_unit


SAT_ACCESS_KEY = 'eb4d69e273848089c7f9b9599cdcd983'


SAT_ACCOUNT = 'c8a00746a80e06c4632028e37de24d6e'


SAT_BUCKET = 'sat'


SAT_SECRET_KEY = '367e9b21fef313f187026320016962b47b74ca4ada7d64d551c43c51e195d7a5'


def download_s3(local_dir, remote_uri):
    """Download remote_dir into (under) local_dir
    """
    s3_resource = boto3.resource('s3', endpoint_url=f'https://{SAT_ACCOUNT}.r2.cloudflarestorage.com', aws_access_key_id=f'{SAT_ACCESS_KEY}', aws_secret_access_key=f'{SAT_SECRET_KEY}')
    client = boto3.client('s3', endpoint_url=f'https://{SAT_ACCOUNT}.r2.cloudflarestorage.com', aws_access_key_id=f'{SAT_ACCESS_KEY}', aws_secret_access_key=f'{SAT_SECRET_KEY}', verify=False)
    bucket = s3_resource.Bucket(SAT_BUCKET)
    transfer_config = boto3.s3.transfer.TransferConfig(use_threads=True, multipart_threshold=8 * 1024 * 1024, max_concurrency=64, multipart_chunksize=8 * 1024 * 1024)
    if '.' in os.path.basename(remote_uri):
        bucket.download_file(remote_uri, os.path.join(local_dir, os.path.basename(remote_uri)), Callback=ProgressPercentage(client, SAT_BUCKET, remote_uri), Config=transfer_config)
        return
    remote_dir = remote_uri
    key_prefix = remote_dir.split('/')[:-1]
    for obj in bucket.objects.filter(Prefix=remote_dir):
        key_suffix = obj.key[len(key_prefix):]
        target_dir = os.path.join(local_dir, os.path.dirname(key_suffix))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if os.path.exists(os.path.join(local_dir, key_suffix)) and os.path.getsize(os.path.join(local_dir, key_suffix)) == obj.size:
            continue
        bucket.download_file(obj.key, os.path.join(local_dir, key_suffix), Callback=ProgressPercentage(client, SAT_BUCKET, obj.key), Config=transfer_config)


def download_with_progress_bar(save_path, url, chunk_size=2048):
    resume_header = None
    file_size_downloaded = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        file_size_downloaded = os.path.getsize(save_path)
        resume_header = {'Range': f'bytes={file_size_downloaded}-'}
    response = requests.get(url, stream=True, headers=resume_header)
    total_size = int(response.headers.get('content-length', 0)) + file_size_downloaded
    if total_size == file_size_downloaded:
        return
    with open(save_path, 'ab') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=save_path, initial=file_size_downloaded) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))


def auto_create(name, *, path=None, url=None):
    if path is None:
        path = os.getenv('SAT_HOME', '~/.sat_models')
    path = os.path.expanduser(path)
    model_path = os.path.join(path, name)
    if url == 'local':
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    lock = FileLock(model_path + '.lock', mode=511)
    with lock:
        if url is None:
            url = MODEL_URLS[name]
        if os.path.isdir(model_path) and not url.startswith('r2://'):
            pass
        elif os.path.isdir(model_path) and url.startswith('r2://') and url.endswith('.zip'):
            pass
        else:
            None
            try:
                if url.startswith('r2://'):
                    download_s3(path, url[5:])
                else:
                    file_path = os.path.join(path, name + '.zip')
                    download_with_progress_bar(file_path, url)
            except Exception as e:
                None
                raise e
        if not os.path.isdir(model_path):
            file_path = os.path.join(path, name + '.zip')
            None
            f = zipfile.ZipFile(file_path, 'r')
            f.extractall(path=path)
            assert os.path.isdir(model_path), f'Unzip failed, or the first-level folder in zip is not {name}.'
    return model_path


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _NODE_GROUP
    _NODE_GROUP = None


def check_if_zero3(args):
    return hasattr(args, 'deepspeed_config') and args.deepspeed_config is not None and args.deepspeed_config.get('zero_optimization', {}).get('stage', 0) >= 3


def configure_logging():
    logger = logging.getLogger('sat')
    logger.setLevel(os.environ.get('SAT_LOGLEVEL', 'INFO'))
    if os.environ.get('LOGLEVEL', None) is not None:
        logger.setLevel(os.environ.get('LOGLEVEL', 'INFO'))
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    sh = logging.StreamHandler()
    logger.setLevel(os.environ.get('SAT_LOGLEVEL', 'INFO'))
    if os.environ.get('LOGLEVEL', None) is not None:
        logger.setLevel(os.environ.get('LOGLEVEL', 'INFO'))
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def print_all(msg, level=logging.INFO, flush=True):
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if torch.distributed.is_initialized():
        msg = f'[RANK {torch.distributed.get_rank()}] {msg}'
    logger.log(level=level, msg=msg)
    if flush:
        logger.handlers[0].flush()


def print_rank0(msg, level=logging.INFO, flush=True):
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if torch.distributed.is_initialized():
        msg = f'[RANK {torch.distributed.get_rank()}] {msg}'
        if torch.distributed.get_rank() == 0:
            logger.log(level=level, msg=msg)
            if flush:
                logger.handlers[0].flush()
    else:
        logger.log(level=level, msg=msg)


def get_model(args, model_cls, **kwargs):
    """Build the model."""
    import torch
    print_rank0(f'building {model_cls.__name__} model ...')
    if 'params_dtype' not in kwargs:
        if hasattr(args, 'fp16') and args.fp16:
            params_dtype = torch.half
        elif hasattr(args, 'bf16') and args.bf16:
            params_dtype = torch.bfloat16
        else:
            params_dtype = torch.float32
    else:
        params_dtype = kwargs.pop('params_dtype')
    if check_if_zero3(args):
        with deepspeed.zero.Init():
            model = model_cls(args, params_dtype=params_dtype, **kwargs)
    else:
        model = model_cls(args, params_dtype=params_dtype, **kwargs)
    if mpu.get_data_parallel_rank() == 0:
        print_all(' > number of parameters on model parallel rank {}: {}'.format(mpu.get_model_parallel_rank(), sum([p.nelement() for p in model.parameters()])), flush=True)
    if hasattr(args, 'fp16') and args.fp16:
        model.half()
    elif hasattr(args, 'bf16') and args.bf16:
        model.bfloat16()
    try:
        if not hasattr(args, 'device'):
            args.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        model = model
    except Exception as e:
        print_all(e)
    return model


_MODEL_PARALLEL_GROUP = None


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    return torch.distributed.get_rank(group=get_model_parallel_group())


_NODE_GROUP = None


def get_node_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _NODE_GROUP is not None, 'node group is not initialized, please pass LOCAL_WORLD_SIZE environment variable.'
    return _NODE_GROUP


def get_node_rank():
    """Return my rank for the node group."""
    return torch.distributed.get_rank(group=get_node_group())


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)


def initialize_model_parallel(model_parallel_size_):
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel groups as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    if torch.distributed.get_rank() == 0:
        print_rank0('> initializing model parallel with size {}'.format(model_parallel_size_))
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    model_parallel_size = min(model_parallel_size_, world_size)
    ensure_divisibility(world_size, model_parallel_size)
    rank = torch.distributed.get_rank()
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
    for i in range(model_parallel_size):
        ranks = range(i, world_size, model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if i == rank % model_parallel_size:
            _DATA_PARALLEL_GROUP = group
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for i in range(world_size // model_parallel_size):
        ranks = range(i * model_parallel_size, (i + 1) * model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if i == rank // model_parallel_size:
            _MODEL_PARALLEL_GROUP = group
    guess_local_world_size = world_size if world_size < 8 else 8
    local_world_size = os.environ.get('LOCAL_WORLD_SIZE', None)
    if local_world_size is None:
        local_world_size = guess_local_world_size
        print_rank0(f"You didn't pass in LOCAL_WORLD_SIZE environment variable. We use the guessed LOCAL_WORLD_SIZE={guess_local_world_size}. If this is wrong, please pass the LOCAL_WORLD_SIZE manually.")
    local_world_size = int(local_world_size)
    global _NODE_GROUP
    assert _NODE_GROUP is None, 'node group is already initialized'
    for i in range(world_size // local_world_size):
        ranks = range(i * local_world_size, (i + 1) * local_world_size)
        group = torch.distributed.new_group(ranks)
        if i == rank // local_world_size:
            _NODE_GROUP = group


def get_checkpoint_tracker_filename(checkpoints_path, old_checkpoint=False):
    return os.path.join(checkpoints_path, 'latest')


def get_checkpoint_iteration(load_path):
    tracker_filename = get_checkpoint_tracker_filename(load_path)
    if not os.path.isfile(tracker_filename):
        print_rank0('could not find the metadata file {} '.format(tracker_filename))
        raise ValueError('could not find the metadata file {}, please check --load'.format(tracker_filename))
        return 0, False, False
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank0('ERROR: Invalid metadata file {}. Exiting'.format(tracker_filename))
                exit()
    assert iteration > 0 or release, 'error parsing metadata file {}'.format(tracker_filename)
    return iteration, release, True


def get_checkpoint_name(checkpoints_path, iteration, release=False, zero=False):
    if release:
        d = 'release'
    else:
        d = '{:d}'.format(iteration)
    if zero:
        dp_rank = mpu.get_data_parallel_rank()
        d += '_zero_dp_rank_{}'.format(dp_rank)
    return os.path.join(checkpoints_path, d, 'mp_rank_{:02d}_model_states.pt'.format(mpu.get_model_parallel_rank()))


def load_checkpoint(model, args, load_path=None, prefix='', specific_iteration=None):
    """Load a model checkpoint."""
    if load_path is None:
        load_path = args.load
    if not hasattr(args, 'mode'):
        from copy import deepcopy
        args = deepcopy(args)
        args.mode = 'inference'
    iteration, release, success = get_checkpoint_iteration(load_path)
    if specific_iteration is not None:
        assert type(specific_iteration) == int and specific_iteration > 0
        print_rank0('Overriding checkpoint iteration to {}'.format(specific_iteration))
        iteration = specific_iteration
    if not success:
        return 0
    checkpoint_name = get_checkpoint_name(load_path, iteration, release)
    if mpu.get_data_parallel_rank() == 0:
        print_all('global rank {} is loading checkpoint {}'.format(torch.distributed.get_rank(), checkpoint_name))
    sd = torch.load(checkpoint_name, map_location='cpu')
    new_sd = {'module': {}}
    for k in sd:
        if k != 'module':
            new_sd[k] = sd[k]
    for k in sd['module']:
        if k.startswith(prefix):
            new_sd['module'][k[len(prefix):]] = sd['module'][k]
    sd = new_sd
    if hasattr(model, 'module'):
        module = model.module
    else:
        module = model
    missing_keys, unexpected_keys = module.load_state_dict(sd['module'], strict=False)
    if len(unexpected_keys) > 0:
        print_rank0(f'Will continue but found unexpected_keys! Check whether you are loading correct checkpoints: {unexpected_keys}.')
    if len(missing_keys) > 0:
        if args.mode == 'inference':
            if 'force_inference' in args and args.force_inference:
                print_rank0(f'Warning: Missing keys for inference: {missing_keys}.')
            else:
                raise ValueError(f'Missing keys for inference: {missing_keys}.\nIf you still want to inference anyway, pass --force_inference to args.')
        else:
            if not args.force_train:
                assert all(name.find('mixins') >= 0 or name.find('cross_attention') >= 0 for name in missing_keys), missing_keys
                assert args.mode == 'finetune'
            mixin_names = []
            for key_name in missing_keys:
                if key_name.find('mixins') < 0:
                    continue
                parts = key_name.split('.')
                mixin_name = parts[parts.index('mixins') + 1]
                if mixin_name not in mixin_names:
                    mixin_names.append(mixin_name)
            module.reinit(mixin_names)
    if args.mode == 'finetune':
        iteration = 0
    elif args.mode == 'pretrain' and not args.no_load_rng:
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.set_rng_state(sd['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        except KeyError:
            print_rank0('Unable to load optimizer from checkpoint {}, exiting. Specify --no-load-rng or --finetune to prevent attempting to load the random state.'.format(checkpoint_name))
            exit()
    elif args.mode == 'inference':
        module.eval()
    if mpu.get_data_parallel_rank() == 0:
        print_all('> successfully loaded {}'.format(checkpoint_name))
    del sd
    return iteration


class Registry:

    def __init__(self, name):
        self.name = name
        self.member = {}

    def register(self, cls):
        if type(cls) is str:

            def func(f):
                self.member[cls] = f
                return f
            return func
        self.member[cls.__name__] = cls
        return cls

    def unregister(self, name):
        self.member.pop(name)

    def get(self, name):
        if name not in self.member:
            raise ValueError(f'model_class {name} not found.')
        return self.member[name]

    def __repr__(self):
        return 'Registry: ' + self.name + ' ' + str(self.member)


model_registry = Registry('sat_models')


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def _initialize_affine_weight(weight, output_size, input_size, per_partition_size, partition_dim, init_method, stride=1, return_master_weight=False, module=None, name=None, self=None):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        init_method(weight, module=module, name=name)
        if return_master_weight:
            return weight
        return None
    master_weight = torch.empty(output_size, input_size, dtype=weight.dtype, requires_grad=False, device=weight.device)
    init_method(master_weight, module=module, name=name)
    weight_list = self.partition(full_weight=master_weight)
    rank = get_model_parallel_rank()
    with torch.no_grad():
        weight.copy_(weight_list[rank])
    del weight_list
    if return_master_weight:
        return master_weight
    return None


def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    group = get_model_parallel_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return input_
    torch.distributed.all_reduce(input_, group=group)
    return input_


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


def copy_to_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""
    group = get_model_parallel_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return input_
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    return output


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
            or a list of strides (ratios) for each partition.
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    last_dim = tensor.dim() - 1
    if isinstance(num_partitions, int):
        last_dim_size = divide(tensor.size()[last_dim], num_partitions)
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    elif isinstance(num_partitions, (list, tuple)):
        factor = tensor.size()[last_dim] // sum(num_partitions)
        tensor_list = torch.split(tensor, [(factor * x) for x in num_partitions], dim=last_dim)
    else:
        raise ValueError('num_partitions must be either int or list/tuple.')
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = get_model_parallel_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return input_
    world_size = torch.distributed.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)
    rank = torch.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()
    return output


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


def gather_from_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor, **kwargs):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)
    return init_


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers. only used in initialization.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True, init_method=unscaled_init_method(0.02), stride=1, keep_master_weight_for_test=False, params_dtype=torch.float, module=None, name=None, skip_init=False, device=torch.device('cpu')):
        super(ColumnParallelLinear, self).__init__()
        self.stride = stride
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.weight = Parameter(torch.empty(self.output_size_per_partition, self.input_size, dtype=params_dtype, device=device))
        self.weight.model_parallel = True
        self.weight.tensor_model_parallel = True
        if bias:
            self.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=params_dtype, device=device))
            self.bias.model_parallel = True
            self.bias.tensor_model_parallel = True
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        if not skip_init:
            self.master_weight = _initialize_affine_weight(self.weight, self.output_size, self.input_size, self.output_size_per_partition, 0, init_method, stride=self.stride, return_master_weight=keep_master_weight_for_test, module=module, name=name, self=self)

    def forward(self, input_):
        input_parallel = copy_to_model_parallel_region(input_)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output

    def repartition(self):
        assert self.output_size_per_partition == self.output_size
        self.output_size_per_partition = divide(self.output_size, get_model_parallel_world_size())
        mp_rank = get_model_parallel_rank()
        mp_size = get_model_parallel_world_size()
        self.original_weight = self.weight
        strides = [1] * self.stride if isinstance(self.stride, int) else self.stride
        assert self.weight.shape[0] % sum(strides) == 0, 'cannot divide weight evenly'
        factor = self.weight.shape[0] // sum(strides)
        strided_weights, _acm = [], 0
        for i in range(len(strides)):
            strided_weights.append(self.weight[_acm:_acm + factor * strides[i], :].detach())
            _acm += factor * strides[i]
        new_weight = torch.cat([strided_weight[strided_weight.shape[0] // mp_size * mp_rank:strided_weight.shape[0] // mp_size * (mp_rank + 1)] for strided_weight in strided_weights], dim=0).contiguous().view(self.output_size_per_partition, self.input_size)
        self.weight = torch.nn.Parameter(new_weight)
        del self.original_weight
        if self.bias is not None and self.bias.numel() != 0:
            self.original_bias = self.bias
            strided_biases, _acm = [], 0
            for i in range(len(strides)):
                strided_biases.append(self.bias[_acm:_acm + factor * strides[i]].detach())
                _acm += factor * strides[i]
            new_bias = torch.cat([strided_bias[strided_bias.shape[0] // mp_size * mp_rank:strided_bias.shape[0] // mp_size * (mp_rank + 1)] for strided_bias in strided_biases], dim=0).contiguous().view(self.output_size_per_partition)
            self.bias = torch.nn.Parameter(new_bias)
            del self.original_bias

    def partition(self, new_model_parallel_size=None, full_weight=None):
        assert self.output_size_per_partition == self.output_size or full_weight is not None
        flag = 1
        if full_weight is None:
            full_weight = self.weight
            flag = 2
        if new_model_parallel_size is None:
            new_model_parallel_size = get_model_parallel_world_size()
        output_size_per_partition = divide(self.output_size, new_model_parallel_size)
        new_weights = []
        new_biases = []
        mp_size = new_model_parallel_size
        strides = [1] * self.stride if isinstance(self.stride, int) else self.stride
        assert full_weight.shape[0] % sum(strides) == 0, 'cannot divide weight evenly'
        factor = full_weight.shape[0] // sum(strides)
        strided_weights, _acm = [], 0
        for i in range(len(strides)):
            strided_weights.append(full_weight[_acm:_acm + factor * strides[i], :].detach())
            _acm += factor * strides[i]
        if flag == 2 and self.bias is not None and self.bias.numel() != 0:
            strided_biases, _acm = [], 0
            for i in range(len(strides)):
                strided_biases.append(self.bias[_acm:_acm + factor * strides[i]].detach())
                _acm += factor * strides[i]
        for rank in range(new_model_parallel_size):
            mp_rank = rank
            new_weight = torch.cat([strided_weight[strided_weight.shape[0] // mp_size * mp_rank:strided_weight.shape[0] // mp_size * (mp_rank + 1)] for strided_weight in strided_weights], dim=0).contiguous().view(output_size_per_partition, self.input_size)
            new_weights.append(torch.clone(new_weight).detach())
            if flag == 2 and self.bias is not None and self.bias.numel() != 0:
                new_bias = torch.cat([strided_bias[strided_bias.shape[0] // mp_size * mp_rank:strided_bias.shape[0] // mp_size * (mp_rank + 1)] for strided_bias in strided_biases], dim=0).contiguous().view(output_size_per_partition)
                new_biases.append(torch.clone(new_bias).detach())
        if flag == 1:
            return new_weights
        else:
            return new_weights, new_biases

    def merge(self, new_weights, new_biases):
        strides = [1] * self.stride if isinstance(self.stride, int) else self.stride
        assert self.weight.shape[0] % sum(strides) == 0, 'cannot divide weight evenly'
        all_weights = []
        _acm = 0
        for stride in strides:
            for weight in new_weights:
                factor = weight.shape[0] // sum(strides)
                all_weights.append(weight[_acm:_acm + factor * stride])
            _acm += factor * stride
        self.weight.data.copy_(torch.cat(all_weights))
        if self.bias is not None and self.bias.numel() != 0:
            all_biases = []
            _acm = 0
            for stride in strides:
                for bias in new_biases:
                    factor = bias.shape[0] // sum(strides)
                    all_biases.append(bias[_acm:_acm + factor * stride])
                _acm += factor * stride
            self.bias.data.copy_(torch.cat(all_biases))


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def reduce_from_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


def scatter_to_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self, input_size, output_size, bias=True, input_is_parallel=False, init_method=unscaled_init_method(0.02), stride=1, keep_master_weight_for_test=False, params_dtype=torch.float, module=None, name=None, skip_init=False, device=torch.device('cpu'), final_bias=True):
        super(RowParallelLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.final_bias = final_bias
        self.weight = Parameter(torch.empty(self.output_size, self.input_size_per_partition, dtype=params_dtype, device=device))
        self.weight.model_parallel = True
        self.weight.tensor_model_parallel = True
        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype, device=device))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        if not skip_init:
            self.master_weight = _initialize_affine_weight(self.weight, self.output_size, self.input_size, self.input_size_per_partition, 1, init_method, stride=stride, return_master_weight=keep_master_weight_for_test, module=module, name=name, self=self)

    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        if self.final_bias or self.bias is None:
            output_parallel = F.linear(input_parallel, self.weight)
        else:
            output_parallel = F.linear(input_parallel, self.weight, self.bias / get_model_parallel_world_size())
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.final_bias and self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output

    def repartition(self):
        assert self.input_size_per_partition == self.input_size
        self.input_size_per_partition = divide(self.input_size, get_model_parallel_world_size())
        mp_rank = get_model_parallel_rank()
        self.original_weight = self.weight
        self.weight = torch.nn.Parameter(torch.clone(self.weight[:, mp_rank * self.input_size_per_partition:(mp_rank + 1) * self.input_size_per_partition]).detach())
        del self.original_weight

    def partition(self, new_model_parallel_size=None, full_weight=None):
        assert self.input_size_per_partition == self.input_size or full_weight is not None
        flag = 1
        if full_weight is None:
            full_weight = self.weight
            flag = 2
        if new_model_parallel_size is None:
            new_model_parallel_size = get_model_parallel_world_size()
        input_size_per_partition = divide(self.input_size, new_model_parallel_size)
        new_weights = []
        new_biases = []
        for rank in range(new_model_parallel_size):
            mp_rank = rank
            weight = torch.clone(full_weight[:, mp_rank * input_size_per_partition:(mp_rank + 1) * input_size_per_partition]).detach()
            new_weights.append(weight)
            if flag == 2 and self.bias is not None and self.bias.numel() != 0:
                new_biases.append(torch.clone(self.bias.data).detach())
        if flag == 1:
            return new_weights
        else:
            return new_weights, new_biases

    def merge(self, new_weights, new_biases):
        self.weight.data.copy_(torch.cat(new_weights, 1))
        if self.bias is not None and self.bias.numel() != 0:
            self.bias.data.copy_(new_biases[0])


class VocabUtility:
    """Split the vocabulary into `world_size` chunks amd return the
        first and last index of the vocabulary belonging to the `rank`
        partition: Note that indecies in [fist, last)"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size)


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim, params_dtype=torch.float, init_method=unscaled_init_method(0.02), skip_init=False, device=torch.device('cpu')):
        super(VocabParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(self.num_embeddings, get_model_parallel_rank(), get_model_parallel_world_size())
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        self.weight = Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim, dtype=params_dtype, device=device))
        self.weight.model_parallel = True
        self.weight.tensor_model_parallel = True
        if not skip_init:
            _initialize_affine_weight(self.weight, self.num_embeddings, self.embedding_dim, self.num_embeddings_per_partition, 0, init_method, self=self)

    def forward(self, input_):
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        output_parallel = F.embedding(masked_input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        output_parallel[input_mask, :] = 0.0
        output = reduce_from_model_parallel_region(output_parallel)
        return output

    def repartition(self):
        assert self.num_embeddings_per_partition == self.num_embeddings
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(self.num_embeddings, get_model_parallel_rank(), get_model_parallel_world_size())
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        self.original_weight = self.weight
        self.weight = torch.nn.Parameter(torch.clone(self.weight[self.vocab_start_index:self.vocab_end_index]).detach())
        del self.original_weight

    def partition(self, new_model_parallel_size=None, full_weight=None):
        assert self.num_embeddings_per_partition == self.num_embeddings or full_weight is not None
        flag = 1
        if full_weight is None:
            full_weight = self.weight
            flag = 2
        if new_model_parallel_size is None:
            new_model_parallel_size = get_model_parallel_world_size()
        new_weights = []
        for rank in range(new_model_parallel_size):
            vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(self.num_embeddings, rank, new_model_parallel_size)
            weight = torch.clone(full_weight[vocab_start_index:vocab_end_index]).detach()
            new_weights.append(weight)
        if flag == 1:
            return new_weights
        else:
            return new_weights, []

    def merge(self, new_weights, new_biases):
        self.weight.data.copy_(torch.cat(new_weights))


def mp_merge_model_rank0(model, model_full):
    assert get_model_parallel_world_size() == torch.distributed.get_world_size(), 'Merging model is only supported for model_parallel_size == world_size!'

    def iter_merge(new_model, module):
        for (new_name, sub_new_model), (name, sub_module) in zip(new_model.named_children(), module.named_children()):
            if isinstance(sub_module, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)):
                new_weights, new_biases = sub_module.partition()
                new_weights = [x for x in new_weights]
                torch.distributed.gather(sub_new_model.weight.data, gather_list=new_weights, dst=0)
                if new_biases:
                    new_biases = [x for x in new_biases]
                    torch.distributed.gather(sub_new_model.bias.data, gather_list=new_biases, dst=0)
                sub_module.merge([torch.clone(x.cpu()).detach() for x in new_weights], [torch.clone(x.cpu()).detach() for x in new_biases])
                del new_weights
                if new_biases:
                    del new_biases
            else:
                for (nn, np), (n, p) in zip(sub_new_model.named_parameters(recurse=False), sub_module.named_parameters(recurse=False)):
                    p.data.copy_(torch.clone(np.data.cpu()).detach())
            iter_merge(sub_new_model, sub_module)
    iter_merge(model, model_full)


def mp_merge_model_send(model):
    assert get_model_parallel_world_size() == torch.distributed.get_world_size(), 'Merging model is only supported for model_parallel_size == world_size!'

    def iter_merge(module):
        for name, sub_module in module.named_children():
            if isinstance(sub_module, VocabParallelEmbedding):
                torch.distributed.gather(sub_module.weight.data, dst=0)
            elif isinstance(sub_module, (ColumnParallelLinear, RowParallelLinear)):
                torch.distributed.gather(sub_module.weight.data, dst=0)
                if sub_module.bias is not None and sub_module.bias.numel() != 0:
                    torch.distributed.gather(sub_module.bias.data, dst=0)
            iter_merge(sub_module)
    iter_merge(model)


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_model_parallel_world_size()
    return global_rank // local_world_size * local_world_size


def get_node_world_size():
    """Return world size for the node group."""
    return torch.distributed.get_world_size(group=get_node_group())


def get_node_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the node group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_node_world_size()
    return global_rank // local_world_size * local_world_size


def mp_split_model_rank0(model, model_full, use_node_group=True):
    """
    This function loads partitions from rank 0.
    It takes less memory when world size is large.
    """
    group = get_node_group() if use_node_group else get_model_parallel_group()
    src = get_node_src_rank() if use_node_group else get_model_parallel_src_rank()
    local_world_size = get_node_world_size() if use_node_group else get_model_parallel_world_size()

    def iter_repartition(new_model, module):
        for (new_name, sub_new_model), (name, sub_module) in zip(new_model.named_children(), module.named_children()):
            if isinstance(sub_module, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)):
                new_weights, new_biases = sub_module.partition()
                for i in range(local_world_size):
                    if i == 0:
                        sub_new_model.weight.data.copy_(new_weights[src % len(new_weights)])
                    else:
                        torch.distributed.send(new_weights[(src + i) % len(new_weights)], src + i)
                if new_biases:
                    for i in range(local_world_size):
                        if i == 0:
                            sub_new_model.bias.data.copy_(new_biases[src % len(new_weights)])
                        else:
                            torch.distributed.send(new_biases[(src + i) % len(new_biases)], src + i)
            else:
                for (nn, np), (n, p) in zip(sub_new_model.named_parameters(recurse=False), sub_module.named_parameters(recurse=False)):
                    np.data.copy_(torch.clone(p.data).detach())
                    torch.distributed.broadcast(np.data, src, group=group)
            iter_repartition(sub_new_model, sub_module)
    iter_repartition(model, model_full)


def mp_split_model_receive(model, use_node_group=True):
    group = get_node_group() if use_node_group else get_model_parallel_group()
    src = get_node_src_rank() if use_node_group else get_model_parallel_src_rank()

    def iter_repartition(module):
        for name, sub_module in module.named_children():
            if isinstance(sub_module, VocabParallelEmbedding):
                torch.distributed.recv(sub_module.weight.data, src)
            elif isinstance(sub_module, (ColumnParallelLinear, RowParallelLinear)):
                torch.distributed.recv(sub_module.weight.data, src)
                if sub_module.bias is not None and sub_module.bias.numel() != 0:
                    torch.distributed.recv(sub_module.bias.data, src)
            else:
                for n, p in sub_module.named_parameters(recurse=False):
                    torch.distributed.broadcast(p.data, src, group=group)
            iter_repartition(sub_module)
    iter_repartition(model)


def overwrite_args_by_dict(args, overwrite_args={}):
    if 'decoder_freq' in overwrite_args:
        decoder_freq = overwrite_args['decoder_freq']
        del overwrite_args['decoder_freq']
    else:
        decoder_freq = None
    for k in overwrite_args:
        setattr(args, k, overwrite_args[k])
    if decoder_freq is not None:
        args.is_decoder = []
        for i in range(args.num_layers):
            if i % decoder_freq == 0:
                args.is_decoder.append(True)
            else:
                args.is_decoder.append(False)
    return args


_GLOBAL_RANDOM_SEED = None


def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None:
        global _GLOBAL_RANDOM_SEED
        _GLOBAL_RANDOM_SEED = seed
        assert seed > 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            if deepspeed.checkpointing.is_configured():
                mpu.model_parallel_cuda_manual_seed(seed)
        except ImportError:
            pass


def reset_random_seed(scale=1):
    assert _GLOBAL_RANDOM_SEED is not None, 'You have not set random seed. No need to reset it.'
    set_random_seed(_GLOBAL_RANDOM_SEED * scale)


def update_args_with_file(args, path):
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    folder = os.path.dirname(path)
    for k in config:
        if k.endswith('_path'):
            config[k] = os.path.join(folder, config[k])
            print_rank0(f'> parsing relative path {k} in model_config as {config[k]}.')
    args = vars(args)
    for k in list(args.keys()):
        if k in config:
            del args[k]
    args = argparse.Namespace(**config, **args)
    return args


class AutoModel:

    @classmethod
    def from_pretrained_base(cls, name, args=None, *, home_path=None, url=None, prefix='', build_only=False, overwrite_args={}, **kwargs):
        """Automatically find the class and instantiate it. Auto-download.
            Args:
                name: The identifier of the pretrained model.
                args: NameSpace. will add the loaded args into it.
                path: the parent folder of existing `name` model. Default: SAT_HOME.
                url: manually specified url for the `name` model.
        """
        if os.path.exists(name) and os.path.isdir(name):
            model_path = name
        else:
            model_path = auto_create(name, path=home_path, url=url)
        if args is None:
            args = argparse.Namespace()
            null_args = True
        else:
            null_args = False
        args = update_args_with_file(args, path=os.path.join(model_path, 'model_config.json'))
        args = overwrite_args_by_dict(args, overwrite_args=overwrite_args)
        if not hasattr(args, 'model_class'):
            raise ValueError('model_config.json must have key "model_class" for AutoModel.from_pretrained.')
        model_cls = model_registry.get(args.model_class)
        if null_args:
            model_default_args = model_cls.get_args()
            for k, v in model_default_args.__dict__.items():
                if not hasattr(args, k):
                    setattr(args, k, v)
        model = get_model(args, model_cls, **kwargs)
        if not build_only:
            load_checkpoint(model, args, load_path=model_path, prefix=prefix)
        return model, args

    @classmethod
    def from_pretrained(cls, name, args=None, *, home_path=None, url=None, prefix='', build_only=False, use_node_group=True, overwrite_args={}, **kwargs):
        if build_only or 'model_parallel_size' not in overwrite_args:
            return cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=build_only, overwrite_args=overwrite_args, **kwargs)
        else:
            new_model_parallel_size = overwrite_args['model_parallel_size']
            if new_model_parallel_size != 1 or new_model_parallel_size == 1 and args.model_parallel_size == 1:
                model, model_args = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=True, overwrite_args=overwrite_args, **kwargs)
                local_rank = get_node_rank() if use_node_group else get_model_parallel_rank()
                world_size = torch.distributed.get_world_size()
                assert world_size % new_model_parallel_size == 0, 'world size should be a multiplier of new model_parallel_size.'
                destroy_model_parallel()
                initialize_model_parallel(1)
                if local_rank == 0:
                    args.skip_init = True
                    args.use_gpu_initialization = False
                    args.device = 'cpu'
                    overwrite_args.pop('model_parallel_size')
                    model_full, args_ = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=False, overwrite_args=overwrite_args, **kwargs)
                    if args_.model_parallel_size != 1:
                        raise Exception("We do not support overwriting model_parallel_size when original model_parallel_size != 1. Try merging the model using `from_pretrained(xxx,overwrite_args={'model_parallel_size':1})` first if you still want to change model_parallel_size!")
                if hasattr(args, 'mode') and args.mode == 'inference':
                    torch.distributed.barrier()
                destroy_model_parallel()
                initialize_model_parallel(new_model_parallel_size)
                if local_rank == 0:
                    mp_split_model_rank0(model, model_full, use_node_group=use_node_group)
                    del model_full
                else:
                    mp_split_model_receive(model, use_node_group=use_node_group)
                reset_random_seed(6)
            else:
                overwrite_args.pop('model_parallel_size')
                model, model_args = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=False, overwrite_args=overwrite_args, **kwargs)
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                assert world_size == model_args.model_parallel_size, 'world size should be equal to model_parallel_size.'
                destroy_model_parallel()
                initialize_model_parallel(1)
                if rank == 0:
                    args.use_gpu_initialization = False
                    args.device = 'cpu'
                    overwrite_args['model_parallel_size'] = 1
                    model_full, args_ = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=True, overwrite_args=overwrite_args, **kwargs)
                torch.distributed.barrier()
                destroy_model_parallel()
                initialize_model_parallel(model_args.model_parallel_size)
                if rank == 0:
                    mp_merge_model_rank0(model, model_full)
                    model, model_args = model_full, args_
                else:
                    mp_merge_model_send(model)
                    model_args.model_parallel_size = 1
                destroy_model_parallel()
                initialize_model_parallel(1)
            return model, model_args


class CLIP_finetune(torch.nn.Module):

    def __init__(self, encoder, hidden_size, num_classes):
        super().__init__()
        self.final = torch.nn.Linear(hidden_size, num_classes)
        self.encoder = encoder

    def forward(self, tokens, position_ids, attention_mask, **kwargs):
        x, *mem = self.encoder(tokens, position_ids, attention_mask, **kwargs)
        x = x / x.norm(dim=-1, keepdim=True)
        x = self.final(x)
        return x

    def disable_untrainable_params(self):
        self.encoder.transformer.position_embeddings.requires_grad_(False)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CLIP-ft', 'CLIP-ft')
        group.add_argument('--num-finetune-classes', type=int, default=None)
        return parser


class Eva2Encoder(nn.Module):

    def __init__(self, image_size=224, ckpt_path=''):
        super(Eva2Encoder, self).__init__()
        self.config = get_model_config('EVA02-CLIP-bigE-14')
        self.config['vision_cfg']['image_size'] = image_size
        model = CustomCLIP(**self.config)
        load_checkpoint(model, ckpt_path)
        self.model = model.visual

    def forward(self, **kwargs):
        encode = self.model(kwargs['image'], return_all_features=True)[:, 1:, :]
        return encode


class MAE_finetune(torch.nn.Module):

    def __init__(self, encoder, hidden_size, num_classes):
        super().__init__()
        self.final = torch.nn.Linear(hidden_size, num_classes)
        self.encoder = encoder

    def forward(self, tokens, position_ids, attention_mask, **kwargs):
        x, *mem = self.encoder(tokens, position_ids, attention_mask, **kwargs)
        x = self.final(x[:, 0])
        return x

    def disable_untrainable_params(self):
        self.encoder.transformer.position_embeddings.requires_grad_(False)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('MAE-finetune', 'MAE finetuning Configurations')
        group.add_argument('--num-finetune-classes', type=int, default=None)
        return parser


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attention:
            return x, attn
        else:
            return x


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            y, attn = self.attn(self.norm1(x), return_attention=return_attention)
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, pre_len, post_len):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = np.concatenate([np.zeros([pre_len, embed_dim]), pos_embed, np.zeros([post_len, embed_dim])], axis=0)
    return pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        grid_size = int(self.patch_embed.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (grid_size, grid_size), 1, 0)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_grid_size = int(self.patch_embed.num_patches ** 0.5)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (decoder_grid_size, decoder_grid_size), 1, 0)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-06) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=nn.LayerNorm, is_distill=False):
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.depth = depth
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if is_distill:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.det_token_num = 0
        self.use_checkpoint = False
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self.has_mid_pe = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def finetune_det(self, img_size=[800, 1344], det_token_num=100, mid_pe_size=None, use_checkpoint=False):
        import math
        g = math.pow(self.pos_embed.size(1) - 1, 0.5)
        if int(g) - g != 0:
            self.pos_embed = torch.nn.Parameter(self.pos_embed[:, 1:, :])
        self.det_token_num = det_token_num
        self.det_token = nn.Parameter(torch.zeros(1, det_token_num, self.embed_dim))
        self.det_token = trunc_normal_(self.det_token, std=0.02)
        cls_pos_embed = self.pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = torch.zeros(1, det_token_num, self.embed_dim)
        det_pos_embed = trunc_normal_(det_pos_embed, std=0.02)
        patch_pos_embed = self.pos_embed[:, 1:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed = torch.nn.Parameter(torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1))
        self.img_size = img_size
        if mid_pe_size == None:
            self.has_mid_pe = False
            None
        else:
            None
            self.mid_pos_embed = nn.Parameter(torch.zeros(self.depth - 1, 1, 1 + mid_pe_size[0] * mid_pe_size[1] // self.patch_size ** 2 + 100, self.embed_dim))
            trunc_normal_(self.mid_pos_embed, std=0.02)
            self.has_mid_pe = True
            self.mid_pe_size = mid_pe_size
        self.use_checkpoint = use_checkpoint

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'det_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, -self.det_token_num:, :]
        patch_pos_embed = pos_embed[:, 1:-self.det_token_num, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        return scale_pos_embed

    def InterpolateMidPosEmbed(self, pos_embed, img_size=(800, 1344)):
        cls_pos_embed = pos_embed[:, :, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, :, -self.det_token_num:, :]
        patch_pos_embed = pos_embed[:, :, 1:-self.det_token_num, :]
        patch_pos_embed = patch_pos_embed.transpose(2, 3)
        D, B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.mid_pe_size[0] // self.patch_size, self.mid_pe_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(D * B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2).contiguous().view(D, B, new_P_H * new_P_W, E)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=2)
        return scale_pos_embed

    def forward_features(self, x):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        x = self.patch_embed(x)
        if self.pos_embed.shape[1] - 1 - self.det_token_num != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H, W))
        else:
            temp_pos_embed = self.pos_embed
        if self.has_mid_pe:
            if self.mid_pos_embed.shape[2] - 1 - self.det_token_num != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(self.mid_pos_embed, img_size=(H, W))
            else:
                temp_mid_pos_embed = self.mid_pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)
        for i in range(len(self.blocks)):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.blocks[i], x)
            else:
                x = self.blocks[i](x)
            if self.has_mid_pe:
                if i < self.depth - 1:
                    x = x + temp_mid_pos_embed[i]
        x = self.norm(x)
        return x[:, -self.det_token_num:, :]

    def forward_return_all_selfattention(self, x):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        x = self.patch_embed(x)
        if self.pos_embed.shape[1] - 1 - self.det_token_num != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H, W))
        else:
            temp_pos_embed = self.pos_embed
        if self.has_mid_pe:
            if self.mid_pos_embed.shape[2] - 1 - self.det_token_num != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(self.mid_pos_embed, img_size=(H, W))
            else:
                temp_mid_pos_embed = self.mid_pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)
        output = []
        for i in range(len(self.blocks)):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.blocks[i], x)
            else:
                x, attn = self.blocks[i](x, return_attention=True)
            if i == len(self.blocks) - 1:
                output.append(attn)
            if self.has_mid_pe:
                if i < self.depth - 1:
                    x = x + temp_mid_pos_embed[i]
        x = self.norm(x)
        return output

    def forward(self, x, return_attention=False):
        if return_attention == True:
            return self.forward_return_all_selfattention(x)
        else:
            x = self.forward_features(x)
            return x


def standard_attention(query_layer, key_layer, value_layer, attention_mask, attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
    if scaling_attention_score:
        query_layer = query_layer / math.sqrt(query_layer.shape[-1])
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    if log_attention_weights is not None:
        attention_scores += log_attention_weights
    if not (attention_mask.shape[-2] == 1 and (attention_mask > 0).all()):
        attention_scores = torch.mul(attention_scores, attention_mask) - 10000.0 * (1.0 - attention_mask)
    attention_probs = F.softmax(attention_scores, dim=-1)
    if attention_dropout is not None:
        if mpu.get_cuda_rng_tracker is not None:
            with mpu.get_cuda_rng_tracker().fork():
                attention_probs = attention_dropout(attention_probs)
        else:
            attention_probs = attention_dropout(attention_probs)
    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer


def attention_fn_default(query_layer, key_layer, value_layer, attention_mask, attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
    batch_size, num_query_heads = query_layer.shape[:2]
    num_kv_heads = key_layer.shape[1]
    key_layer = key_layer.unsqueeze(2).expand(-1, -1, num_query_heads // num_kv_heads, -1, -1).contiguous().view(batch_size, num_query_heads, *key_layer.shape[2:])
    value_layer = value_layer.unsqueeze(2).expand(-1, -1, num_query_heads // num_kv_heads, -1, -1).contiguous().view(batch_size, num_query_heads, *value_layer.shape[2:])
    is_low_triangle = (attention_mask == torch.ones_like(attention_mask, dtype=torch.float).tril()).all()
    is_full = attention_mask is None or (attention_mask > 0).all()
    if int(torch.__version__.split('.')[0]) >= 2 and scaling_attention_score and (is_full or is_low_triangle):
        dropout_p = 0.0 if attention_dropout is None or not attention_dropout.training else attention_dropout.p
        if dropout_p > 0 and mpu.get_cuda_rng_tracker is not None:
            context = mpu.get_cuda_rng_tracker().fork()
        else:
            context = contextlib.nullcontext()
        with context:
            attn_output = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask=None, dropout_p=dropout_p, is_causal=not is_full)
        return attn_output
    else:
        return standard_attention(query_layer, key_layer, value_layer, attention_mask, attention_dropout=attention_dropout, log_attention_weights=log_attention_weights, scaling_attention_score=scaling_attention_score, **kwargs)


def attention_forward_default(self, hidden_states, mask, **kw_args):
    self = self.transformer.layers[kw_args['layer_id']].attention
    attention_fn = attention_fn_default
    if 'attention_fn' in self.hooks:
        attention_fn = self.hooks['attention_fn']
    mixed_raw_layer = self.query_key_value(hidden_states)
    mixed_query_layer, mixed_key_layer, mixed_value_layer = split_tensor_along_last_dim(mixed_raw_layer, self.stride)
    dropout_fn = self.attention_dropout if self.training else None
    query_layer = self._transpose_for_scores(mixed_query_layer)
    key_layer = self._transpose_for_scores(mixed_key_layer)
    value_layer = self._transpose_for_scores(mixed_value_layer)
    if self.transformer.is_rotary_emb:
        query_layer, key_layer = self.transformer.position_embeddings(query_layer, key_layer, kw_args['position_ids'], max_seqlen=kw_args['position_ids'].max() + 1, layer_id=kw_args['layer_id'])
    context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)
    output = self.dense(context_layer)
    if self.training:
        output = self.output_dropout(output)
    return output


def cross_attention_forward_default(self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args):
    self = self.transformer.layers[kw_args['layer_id']].cross_attention
    attention_fn = attention_fn_default
    if 'attention_fn' in self.hooks:
        attention_fn = self.hooks['attention_fn']
    mixed_query_layer = self.query(hidden_states)
    query_layer = self._transpose_for_scores(mixed_query_layer)
    dropout_fn = self.attention_dropout if self.training else None
    if isinstance(encoder_outputs, torch.Tensor):
        mixed_x_layer = self.key_value(encoder_outputs)
        mixed_key_layer, mixed_value_layer = split_tensor_along_last_dim(mixed_x_layer, 2)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)
        mem_cross = key_layer, value_layer
    else:
        key_layer, value_layer = encoder_outputs[kw_args['layer_id']]
        mem_cross = key_layer, value_layer
    context_layer = attention_fn(query_layer, key_layer, value_layer, cross_attention_mask, dropout_fn, cross_attention=True, mem_cross=mem_cross, **kw_args)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)
    output = self.dense(context_layer)
    if self.training:
        output = self.output_dropout(output)
    return output


def final_forward_default(self, logits, **kw_args):
    logits_parallel = F.linear(logits, self.transformer.word_embeddings.weight)
    if not kw_args['parallel_output']:
        logits_parallel = gather_from_model_parallel_region(logits_parallel)
    return logits_parallel


def layer_forward_default(self, hidden_states, mask, *args, **kw_args):
    """
        hidden_states: [batch, seq_len, hidden_size]
        mask: [(1, 1), seq_len, seq_len]
    """
    self = self.transformer.layers[kw_args['layer_id']]
    attention_input = self.input_layernorm(hidden_states)
    attention_output = self.attention(attention_input, mask, **kw_args)
    if self.layernorm_order == 'sandwich':
        attention_output = self.third_layernorm(attention_output)
    if self.training and self.drop_path > 0.0:
        random_tensor = (1 - self.drop_path + torch.rand((attention_output.shape[0],), dtype=attention_output.dtype, device=attention_output.device)).floor_() / (1 - self.drop_path)
        attention_output = random_tensor.view(-1, 1, 1) * attention_output
    if self.layernorm_order == 'post':
        hidden_states = attention_input + attention_output
        mlp_input = self.post_attention_layernorm(hidden_states)
    else:
        hidden_states = hidden_states + attention_output
    if self.is_decoder:
        encoder_outputs = kw_args['encoder_outputs']
        if encoder_outputs is not None:
            assert 'cross_attention_mask' in kw_args
            if self.layernorm_order == 'post':
                attention_output = self.cross_attention(mlp_input, **kw_args)
                hidden_states = mlp_input + attention_output
                mlp_input = self.post_cross_attention_layernorm(hidden_states)
            else:
                cross_input = self.post_cross_attention_layernorm(hidden_states)
                attention_output = self.cross_attention(cross_input, **kw_args)
                hidden_states = hidden_states + attention_output
    if self.layernorm_order != 'post':
        mlp_input = self.post_attention_layernorm(hidden_states)
    mlp_output = self.mlp(mlp_input, **kw_args)
    if self.layernorm_order == 'sandwich':
        mlp_output = self.fourth_layernorm(mlp_output)
    if self.training and self.drop_path > 0.0:
        random_tensor = (1 - self.drop_path + torch.rand((mlp_output.shape[0],), dtype=mlp_output.dtype, device=mlp_output.device)).floor_() / (1 - self.drop_path)
        mlp_output = random_tensor.view(-1, 1, 1) * mlp_output
    if self.layernorm_order == 'post':
        output = mlp_input + mlp_output
    else:
        output = hidden_states + mlp_output
    return output


def routing_forward_default(self, hidden_states, **kw_args):
    num_experts = self.transformer.num_experts
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    router_logits = torch.randn((batch_size * sequence_length, num_experts), device=hidden_states.device, dtype=hidden_states.dtype)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights
    return routing_weights, selected_experts


def mlp_forward_default(self, hidden_states, expert_id=-1, **kw_args):
    if self.transformer.num_experts == 1 or expert_id > -1:
        self = self.transformer.layers[kw_args['layer_id']].mlp
        suffix = f'_{expert_id}' if expert_id > 0 else ''
        if self.is_gated_mlp:
            intermediate_parallel = getattr(self, 'dense_h_to_4h' + suffix)(hidden_states)
            gated_intermediate_parallel = getattr(self, 'dense_h_to_4h_gate' + suffix)(hidden_states)
            intermediate_parallel = self.activation_func(gated_intermediate_parallel) * intermediate_parallel
            output = getattr(self, 'dense_4h_to_h' + suffix)(intermediate_parallel)
        else:
            intermediate_parallel = getattr(self, 'dense_h_to_4h' + suffix)(hidden_states)
            intermediate_parallel = self.activation_func(intermediate_parallel)
            output = getattr(self, 'dense_4h_to_h' + suffix)(intermediate_parallel)
        return output
    else:
        mlp_forward = self.hooks.get('mlp_forward', partial(mlp_forward_default, self))
        routing_forward = self.hooks.get('routing_forward', partial(routing_forward_default, self))
        self = self.transformer.layers[kw_args['layer_id']].mlp
        fwd_weight, fwd_idx = routing_forward(hidden_states, **kw_args)
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        expert_mask = torch.nn.functional.one_hot(fwd_idx, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = hidden_states[top_x_list]
            current_hidden_states = mlp_forward(current_state, expert_id=expert_idx, **kw_args) * fwd_weight[top_x_list, idx_list, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        output = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return output


def position_embedding_forward_default(self, position_ids, output_cross_layer, **kw_args):
    if not self.transformer.is_rotary_emb:
        return self.transformer.position_embeddings(position_ids)
    return None


def word_embedding_forward_default(self, input_ids, output_cross_layer, **kw_args):
    return self.transformer.word_embeddings(input_ids)


HOOKS_DEFAULT = {'attention_fn': attention_fn_default, 'attention_forward': attention_forward_default, 'cross_attention_forward': cross_attention_forward_default, 'routing_forward': routing_forward_default, 'mlp_forward': mlp_forward_default, 'word_embedding_forward': word_embedding_forward_default, 'position_embedding_forward': position_embedding_forward_default, 'final_forward': final_forward_default, 'layer_forward': layer_forward_default}


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def gelu(x):
    return gelu_impl(x)


class MLP(torch.nn.Module):

    def __init__(self, hidden_size, output_dropout_prob, init_method, inner_hidden_size=None, output_layer_init_method=None, layer_id=None, row_parallel_linear_final_bias=True, hooks={}, bias=True, activation_func=gelu, transformer_pointer=None, is_gated_mlp=False, num_experts=1, params_dtype=torch.float, skip_init=False, device=torch.device('cpu')):
        super(MLP, self).__init__()
        self.layer_id = layer_id
        self.activation_func = activation_func
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = ColumnParallelLinear(self.hidden_size, self.inner_hidden_size, gather_output=False, init_method=init_method, bias=bias, params_dtype=params_dtype, module=self, name='dense_h_to_4h', skip_init=skip_init, device=device)
        self.dense_4h_to_h = RowParallelLinear(self.inner_hidden_size, self.hidden_size, input_is_parallel=True, init_method=output_layer_init_method, bias=bias, params_dtype=params_dtype, module=self, name='dense_4h_to_h', skip_init=skip_init, device=device, final_bias=row_parallel_linear_final_bias)
        self.is_gated_mlp = is_gated_mlp
        if is_gated_mlp:
            self.dense_h_to_4h_gate = ColumnParallelLinear(self.hidden_size, self.inner_hidden_size, gather_output=False, init_method=init_method, bias=False, params_dtype=params_dtype, module=self, name='dense_h_to_4h_gate', skip_init=skip_init, device=device)
        self.num_experts = num_experts
        for i in range(1, num_experts):
            self.register_module(f'dense_h_to_4h_{i}', ColumnParallelLinear(self.hidden_size, self.inner_hidden_size, gather_output=False, init_method=init_method, bias=bias, params_dtype=params_dtype, module=self, name=f'dense_h_to_4h_{i}', skip_init=skip_init, device=device))
            self.register_module(f'dense_4h_to_h_{i}', RowParallelLinear(self.inner_hidden_size, self.hidden_size, input_is_parallel=True, init_method=output_layer_init_method, bias=bias, params_dtype=params_dtype, module=self, name=f'dense_4h_to_h_{i}', skip_init=skip_init, device=device, final_bias=row_parallel_linear_final_bias))
            if is_gated_mlp:
                self.register_module(f'dense_h_to_4h_gate_{i}', ColumnParallelLinear(self.hidden_size, self.inner_hidden_size, gather_output=False, init_method=init_method, bias=False, params_dtype=params_dtype, module=self, name=f'dense_h_to_4h_gate_{i}', skip_init=skip_init, device=device))
        self.dropout = torch.nn.Dropout(output_dropout_prob)
        object.__setattr__(self, 'transformer', transformer_pointer)
        assert transformer_pointer is not None

    def forward(self, hidden_states, **kw_args):
        if 'mlp_forward' in self.hooks:
            output = self.hooks['mlp_forward'](hidden_states, **kw_args)
        else:
            output = HOOKS_DEFAULT['mlp_forward'](self, hidden_states, **kw_args)
        if self.training:
            output = self.dropout(output)
        return output


def base(pretrained=None, **kwargs):
    model = VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06), is_distill=True, **kwargs)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    return model, 768


class NestedTensor(object):

    def __init__(self, tensors, mask: 'Optional[Tensor]'):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: 'List[Tensor]') ->NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32))
        max_size.append(max_size_i)
    max_size = tuple(max_size)
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)
        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), 'constant', 1)
        padded_masks.append(padded_mask)
    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)
    return NestedTensor(tensor, mask=mask)


def nested_tensor_from_tensor_list(tensor_list: 'List[Tensor]'):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            return _onnx_nested_tensor_from_tensor_list(tensor_list)
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def small(pretrained=None, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06), **kwargs)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    return model, 384


def small_dWr(pretrained=None, **kwargs):
    model = VisionTransformer(img_size=240, patch_size=16, embed_dim=330, depth=14, num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06), **kwargs)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    return model, 330


def tiny(pretrained=None, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06))
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
    return model, 192


class Detector(nn.Module):

    def __init__(self, num_classes, pre_trained=None, det_token_num=100, backbone_name='tiny', init_pe_size=[800, 1344], mid_pe_size=None, use_checkpoint=False):
        super().__init__()
        if backbone_name == 'tiny':
            self.backbone, hidden_dim = tiny(pretrained=pre_trained)
        elif backbone_name == 'small':
            self.backbone, hidden_dim = small(pretrained=pre_trained)
        elif backbone_name == 'base':
            self.backbone, hidden_dim = base(pretrained=pre_trained)
        elif backbone_name == 'small_dWr':
            self.backbone, hidden_dim = small_dWr(pretrained=pre_trained)
        else:
            raise ValueError(f'backbone {backbone_name} not supported')
        self.backbone.finetune_det(det_token_num=det_token_num, img_size=init_pe_size, mid_pe_size=mid_pe_size, use_checkpoint=use_checkpoint)
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, samples: 'NestedTensor'):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        x = self.backbone(samples.tensors)
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return out

    def forward_return_attention(self, samples: 'NestedTensor'):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        attention = self.backbone(samples.tensors, return_attention=True)
        return attention


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__.split('.')[1]) < 7.0:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)
        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert 'pred_masks' in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t['masks'] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks
        target_masks = target_masks[tgt_idx]
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {'loss_mask': sigmoid_focal_loss(src_masks, target_masks, num_boxes), 'loss_dice': dice_loss(src_masks, target_masks, num_boxes)}
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {'labels': self.loss_labels, 'cardinality': self.loss_cardinality, 'boxes': self.loss_boxes, 'masks': self.loss_masks}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


def drop_block_2d(x, drop_prob: 'float'=0.1, block_size: 'int'=7, gamma_scale: 'float'=1.0, with_noise: 'bool'=False, inplace: 'bool'=False, batchwise: 'bool'=False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / ((W - block_size + 1) * (H - block_size + 1))
    w_i, h_i = torch.meshgrid(torch.arange(W), torch.arange(H))
    valid_block = (w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2) & ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W))
    if batchwise:
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = 2 - gamma - valid_block + uniform_noise >= 1
    block_mask = -F.max_pool2d(-block_mask, kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)
    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-07)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(x: 'torch.Tensor', drop_prob: 'float'=0.1, block_size: 'int'=7, gamma_scale: 'float'=1.0, with_noise: 'bool'=False, inplace: 'bool'=False, batchwise: 'bool'=False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / ((W - block_size + 1) * (H - block_size + 1))
    if batchwise:
        block_mask = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device) < gamma
    else:
        block_mask = torch.rand_like(x) < gamma
    block_mask = F.max_pool2d(block_mask, kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)
    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(1.0 - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1.0 - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-07)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """

    def __init__(self, drop_prob=0.1, block_size=7, gamma_scale=1.0, with_noise=False, inplace=False, batchwise=False, fast=True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)
        else:
            return drop_block_2d(x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: 'float'=1, cost_bbox: 'float'=1, cost_giou: 'float'=1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v['boxes']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


ARGS_DEFAULT = {'embedding_dropout_prob': ('hidden_dropout', 0), 'attention_dropout_prob': ('attention_dropout', 0), 'output_dropout_prob': ('hidden_dropout', 0), 'inner_hidden_size': ('inner_hidden_size', None), 'hidden_size_per_attention_head': ('hidden_size_per_attention_head', None), 'cross_hidden_size_per_attention_head': ('cross_hidden_size_per_attention_head', None), 'checkpoint_activations': ('checkpoint_activations', False), 'checkpoint_num_layers': ('checkpoint_num_layers', 1), 'checkpoint_skip_layers': ('checkpoint_skip_layers', 0), 'is_decoder': ('is_decoder', False), 'cross_attn_hidden_size': ('cross_attn_hidden_size', None), 'use_final_layernorm': ('use_final_layernorm', True), 'layernorm_epsilon': ('layernorm_epsilon', 1e-05), 'use_bias': ('use_bias', True), 'use_qkv_bias': ('use_qkv_bias', False), 'num_multi_query_heads': ('num_multi_query_heads', 0), 'cross_num_multi_query_heads': ('cross_num_multi_query_heads', 0), 'drop_path': ('drop_path', 0.0), 'row_parallel_linear_final_bias': ('row_parallel_linear_final_bias', True), 'is_gated_mlp': ('is_gated_mlp', False), 'is_rotary_emb': ('is_rotary_emb', False), 'parallel_output': ('parallel_output', False), 'num_experts': ('num_experts', 1)}


class CrossAttention(torch.nn.Module):
    """Parallel cross-attention layer for Transformer"""

    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, init_method, layer_id, hidden_size_per_attention_head=None, output_layer_init_method=None, bias=True, cross_num_multi_query_heads=0, row_parallel_linear_final_bias=True, hooks={}, cross_attn_hidden_size=None, transformer_pointer=None, params_dtype=torch.float, skip_init=False, device=torch.device('cpu')):
        super().__init__()
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        self.layer_id = layer_id
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        world_size = get_model_parallel_world_size()
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition
        self.cross_num_multi_query_heads = cross_num_multi_query_heads
        if cross_num_multi_query_heads == 0:
            kv_size = 2 * self.inner_hidden_size
        else:
            kv_size = self.hidden_size_per_attention_head * self.cross_num_multi_query_heads * 2
        self.query = ColumnParallelLinear(hidden_size, self.inner_hidden_size, gather_output=False, init_method=init_method, bias=bias, params_dtype=params_dtype, module=self, name='query', skip_init=skip_init, device=device)
        if cross_attn_hidden_size is None:
            cross_attn_hidden_size = hidden_size
        self.cross_attn_hidden_size = cross_attn_hidden_size
        self.key_value = ColumnParallelLinear(cross_attn_hidden_size, kv_size, stride=2, gather_output=False, init_method=init_method, bias=bias, params_dtype=params_dtype, module=self, name='key_value', skip_init=skip_init, device=device)
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)
        self.dense = RowParallelLinear(self.inner_hidden_size, hidden_size, input_is_parallel=True, init_method=output_layer_init_method, bias=bias, params_dtype=params_dtype, module=self, name='dense', skip_init=skip_init, device=device, final_bias=row_parallel_linear_final_bias)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)
        object.__setattr__(self, 'transformer', transformer_pointer)
        assert transformer_pointer is not None

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + (-1, self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args):
        if 'cross_attention_forward' in self.hooks:
            return self.hooks['cross_attention_forward'](hidden_states, cross_attention_mask, encoder_outputs, **kw_args)
        else:
            return HOOKS_DEFAULT['cross_attention_forward'](self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args)

    def repartition(self):
        world_size = get_model_parallel_world_size()
        self.num_attention_heads_per_partition = divide(self.num_attention_heads, world_size)
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition


class SelfAttention(torch.nn.Module):

    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, init_method, layer_id, hidden_size_per_attention_head=None, output_layer_init_method=None, bias=True, qkv_bias=False, num_multi_query_heads=0, row_parallel_linear_final_bias=True, hooks={}, transformer_pointer=None, params_dtype=torch.float, skip_init=False, device=torch.device('cpu')):
        super(SelfAttention, self).__init__()
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        self.layer_id = layer_id
        world_size = get_model_parallel_world_size()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_multi_query_heads = num_multi_query_heads
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.num_multi_query_heads_per_partition = divide(num_multi_query_heads, world_size)
        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition
        if num_multi_query_heads == 0:
            qkv_size = 3 * self.inner_hidden_size
            self.stride = 3
        else:
            qkv_size = self.inner_hidden_size + self.hidden_size_per_attention_head * self.num_multi_query_heads * 2
            self.stride = [self.num_attention_heads_per_partition, self.num_multi_query_heads_per_partition, self.num_multi_query_heads_per_partition]
        self.query_key_value = ColumnParallelLinear(hidden_size, qkv_size, stride=self.stride, gather_output=False, init_method=init_method, bias=bias or qkv_bias, params_dtype=params_dtype, module=self, name='query_key_value', skip_init=skip_init, device=device)
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)
        self.dense = RowParallelLinear(self.inner_hidden_size, hidden_size, input_is_parallel=True, init_method=output_layer_init_method, bias=bias, params_dtype=params_dtype, module=self, name='dense', skip_init=skip_init, device=device, final_bias=row_parallel_linear_final_bias)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)
        object.__setattr__(self, 'transformer', transformer_pointer)
        assert transformer_pointer is not None

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + (-1, self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, mask, *args, **kw_args):
        if 'attention_forward' in self.hooks:
            return self.hooks['attention_forward'](hidden_states, mask, **kw_args)
        else:
            return HOOKS_DEFAULT['attention_forward'](self, hidden_states, mask, **kw_args)

    def repartition(self):
        world_size = get_model_parallel_world_size()
        self.num_attention_heads_per_partition = divide(self.num_attention_heads, world_size)
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition


def apply_rotary(x: 'torch.Tensor', cos: 'torch.Tensor', sin: 'torch.Tensor', position_ids: 'torch.Tensor', seqlen_offsets: 'Union[int, torch.Tensor]'=0, cu_seqlens: 'Optional[torch.Tensor]'=None, max_seqlen: 'Optional[int]'=None, interleaved=False, inplace=False, conjugate=False) ->torch.Tensor:
    """
    Arguments:
        x: (batch, nheads, seqlen, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, nheads, seqlen, headdim)
    """
    batch, nheads, seqlen, headdim = x.shape
    seqlen_ro, rotary_dim = cos.shape
    batch_p, seqlen_p = position_ids.shape
    assert batch_p == batch and seqlen_p == seqlen
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, 'rotary_dim must be <= headdim'
    assert headdim <= 256, 'Only support headdim <= 256'
    assert seqlen_ro >= max_seqlen, 'seqlen_ro must be >= max_seqlen'
    assert cos.dtype == sin.dtype, f'cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}'
    assert x.dtype == cos.dtype, f'Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}'
    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + max_seqlen <= seqlen_ro
    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])
    BLOCK_K = 32 if rotary_dim <= 32 else 64 if rotary_dim <= 64 else 128 if rotary_dim <= 128 else 256
    grid = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), batch, nheads)
    BLOCK_M = 4 if interleaved else 8 if rotary_dim <= 64 else 4
    with torch.device(x.device.index):
        rotary_kernel[grid](output, x, cos, sin, position_ids, cu_seqlens, seqlen_offsets, seqlen, nheads, rotary_dim, seqlen_ro, seqlen // 128, output.stride(0), output.stride(-3), output.stride(-2), output.stride(-1), x.stride(0), x.stride(-3), x.stride(-2), x.stride(-1), position_ids.stride(0), BLOCK_K, isinstance(seqlen_offsets, torch.Tensor), False, interleaved, conjugate, BLOCK_M)
    return output


class ApplyRotaryEmb(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, cos, sin, position_id, interleaved=False, inplace=False, seqlen_offsets: 'Union[int, torch.Tensor]'=0, cu_seqlens: 'Optional[torch.Tensor]'=None, max_seqlen: 'Optional[int]'=None):
        out = apply_rotary(x, cos, sin, position_id, seqlen_offsets=seqlen_offsets, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, interleaved=interleaved, inplace=inplace)
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, position_id, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, position_id, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, position_id, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, position_id, cu_seqlens = ctx.saved_tensors
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = apply_rotary(do, cos, sin, position_id, seqlen_offsets=seqlen_offsets, cu_seqlens=cu_seqlens, max_seqlen=ctx.max_seqlen, interleaved=ctx.interleaved, inplace=ctx.inplace, conjugate=True)
        return dx, None, None, None, None, None, None, None, None


def apply_rotary_emb(x, cos, sin, position_id, interleaved=False, inplace=False, seqlen_offsets: 'Union[int, torch.Tensor]'=0, cu_seqlens: 'Optional[torch.Tensor]'=None, max_seqlen: 'Optional[int]'=None):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(x, cos, sin, position_id, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen)


apply_rotary_emb_func = apply_rotary_emb


class FastRotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(self, dim: 'int', base=10000, interleaved=False, scale_base=None, pos_idx_in_fp32=True, device=None, shift=0):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.shift = shift
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer('inv_freq', inv_freq)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim) if scale_base is not None else None
        self.register_buffer('scale', scale, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / self.base ** (torch.arange(self.shift, self.shift + self.dim, 2, device=device).float() / self.dim)

    def _update_cos_sin_cache(self, seqlen, position_id, device=None, dtype=None):
        if seqlen > self._seq_len_cached:
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs)
                self._sin_cached = torch.sin(freqs)
            else:
                power = (torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2) / self.scale_base
                scale = self.scale ** rearrange(power, 's -> s 1')
                self._cos_cached = torch.cos(freqs) * scale
                self._sin_cached = torch.sin(freqs) * scale
                self._cos_k_cached = torch.cos(freqs) / scale
                self._sin_k_cached = torch.sin(freqs) / scale

    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', position_id: 'torch.Tensor', max_seqlen, layer_id: 'int'=0) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, nheads, seqlen, headdim) 
        k: (batch, nheads, seqlen, headdim)
        position_id: (batch, seqlen)
        max_seqlen: max number of position_ids
        layer_id: deprecated
        Apply rotary embedding *inplace* to q k.
        """
        if position_id.shape[0] != q.shape[0]:
            position_id = position_id.expand(q.shape[0], -1)
        max_position_id = max_seqlen
        self._update_cos_sin_cache(max_seqlen, position_id, device=q.device, dtype=q.dtype)
        q = apply_rotary_emb_func(q, self._cos_cached, self._sin_cached, position_id, interleaved=self.interleaved, inplace=True, max_seqlen=max_position_id)
        k = apply_rotary_emb_func(k, self._cos_cached, self._sin_cached, position_id, interleaved=self.interleaved, inplace=True, max_seqlen=max_position_id)
        return q, k


def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor, **kwargs):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_


class MetaModel(type):

    def __new__(cls, clsname, bases, attrs):
        newclass = super().__new__(cls, clsname, bases, attrs)
        model_registry.register(newclass)
        return newclass

    def __setattr__(self, __name, __value):
        if __name == '__name__':
            model_registry.unregister(getattr(self, __name))
        tmp = super().__setattr__(__name, __value)
        if __name == '__name__':
            model_registry.register(self)
        return tmp


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        port = s.getsockname()[1]
    return port


def initialize_distributed(args):
    """Initialize torch.distributed."""
    if torch.distributed.is_initialized():
        if mpu.model_parallel_is_initialized():
            if args.model_parallel_size != mpu.get_model_parallel_world_size():
                raise ValueError('model_parallel_size is inconsistent with prior configuration.We currently do not support changing model_parallel_size.')
            return False
        else:
            if args.model_parallel_size > 1:
                warnings.warn('model_parallel_size > 1 but torch.distributed is not initialized via SAT.Please carefully make sure the correctness on your own.')
            mpu.initialize_model_parallel(args.model_parallel_size)
        return True
    if args.device == 'cpu':
        pass
    else:
        torch.cuda.set_device(args.device)
    init_method = 'tcp://'
    args.master_ip = os.getenv('MASTER_ADDR', 'localhost')
    if args.world_size == 1:
        default_master_port = str(get_free_port())
    else:
        default_master_port = '6000'
    args.master_port = os.getenv('MASTER_PORT', default_master_port)
    init_method += args.master_ip + ':' + args.master_port
    torch.distributed.init_process_group(backend=args.distributed_backend, world_size=args.world_size, rank=args.rank, init_method=init_method)
    mpu.initialize_model_parallel(args.model_parallel_size)
    if args.deepspeed:
        deepspeed.init_distributed(dist_backend=args.distributed_backend, world_size=args.world_size, rank=args.rank, init_method=init_method)
        deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    else:
        try:
            _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 1)
        except Exception as e:
            print_rank0(str(e), level='DEBUG')
    return True


def _simple_init(model_parallel_size=1, seed=0):
    """Necessary initialization for torch.distributed for model-only mode"""
    args = argparse.Namespace(distributed_backend='nccl' if torch.distributed.is_nccl_available() and torch.cuda.is_available() else 'gloo', model_parallel_size=model_parallel_size)
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv('WORLD_SIZE', '1'))
    args.local_rank = int(os.getenv('LOCAL_RANK', '0'))
    args.device = args.local_rank
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.deepspeed = False
    set_random_seed(seed)
    if initialize_distributed(args):
        print_rank0('You are using model-only mode.\nFor torch.distributed users or loading model parallel models, set environment variables RANK, WORLD_SIZE and LOCAL_RANK.')
        return True
    return False


def add_model_config_args(parser):
    """Model arguments"""
    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--num-layers', type=int, default=6, help='num of layers')
    group.add_argument('--hidden-size', type=int, default=1024, help='transformer hidden size')
    group.add_argument('--num-attention-heads', type=int, default=16, help='num of transformer attention heads')
    group.add_argument('--vocab-size', type=int, default=100, help='vocab size for tokenization. the size of word_embeddings.')
    group.add_argument('--max-sequence-length', type=int, default=512, help='maximum number of position embeddings to use')
    group.add_argument('--layernorm-order', type=str, default='pre', help='choose from "pre", "post", "sandwich".', choices=['post', 'pre', 'sandwich'])
    group.add_argument('--inner-hidden-size', type=int, default=None, help='inner hidden size for transformer FFN, None means 4*hidden-size')
    group.add_argument('--hidden-size-per-attention-head', type=int, default=None)
    group.add_argument('--model-parallel-size', type=int, default=1, help='size of the model parallel. only use if you are an expert.')
    group.add_argument('--skip-init', action='store_true', help='skip model initialization')
    group.add_argument('--use-gpu-initialization', action='store_true', help='initialize model on gpu')
    group.add_argument('--num-multi-query-heads', type=int, default=0, help='use multi-query attention, num of kv groups. 0 means multi-head attention.')
    group.add_argument('--is-gated-mlp', action='store_true', help='use gated MLP (GLU), common in LLAMA etc.')
    group.add_argument('--is-rotary-emb', action='store_true', help='use rotary embedding, common in LLAMA etc.')
    group.add_argument('--layernorm-epsilon', type=float, default=1e-05, help='layer norm epsilon')
    group.add_argument('--hidden-dropout', type=float, default=0.1, help='dropout probability for hidden state transformer')
    group.add_argument('--attention-dropout', type=float, default=0.1, help='dropout probability for attention weights')
    group.add_argument('--drop-path', type=float, default=0.0, help='drop path probability')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128, help='Pad the vocab size to be divisible by this value.This is added for computational efficieny reasons.')
    group.add_argument('--parallel-output', action='store_true', help='whether to gather model parallel outputs at final output. Need to be True if using mpu.vocab_parallel_cross_entropy.')
    return parser


def print_parser(parser, help_width=32):
    argument_list = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        if '--help' in action.option_strings:
            continue
        arg_name = ', '.join([opt.lstrip('-') for opt in action.option_strings])
        arg_help = action.help or ''
        arg_type = action.type.__name__ if action.type else 'str'
        arg_default = str(action.default) if action.default is not None else 'None'
        argument_list.append((arg_name, arg_help, arg_type, arg_default))
    max_name_len = max([len(arg[0]) for arg in argument_list])
    None
    None
    None
    wrapper = textwrap.TextWrapper(width=help_width)
    for arg_name, arg_help, arg_type, arg_default in argument_list:
        name_str = arg_name.ljust(max_name_len)
        type_str = arg_type.ljust(8)
        wrapped_help = wrapper.wrap(arg_help)
        if not wrapped_help:
            wrapped_help = ['']
        for i, line in enumerate(wrapped_help):
            if i == 0:
                None
            else:
                None
        None


class BaseModel(torch.nn.Module, metaclass=MetaModel):

    def __init__(self, args, transformer=None, params_dtype=torch.float, **kwargs):
        super(BaseModel, self).__init__()
        self.mixins = torch.nn.ModuleDict()
        self.collect_hooks_()
        if transformer is not None:
            self.transformer = transformer
        else:
            success = _simple_init(model_parallel_size=args.model_parallel_size, seed=args.seed if hasattr(args, 'seed') else 1234)
            args_dict = {k: (getattr(args, v[0]) if hasattr(args, v[0]) else v[1]) for k, v in ARGS_DEFAULT.items()}
            self.transformer = BaseTransformer(num_layers=args.num_layers, vocab_size=args.vocab_size, hidden_size=args.hidden_size, num_attention_heads=args.num_attention_heads, max_sequence_length=args.max_sequence_length, layernorm_order=args.layernorm_order, **args_dict, hooks=self.hooks, params_dtype=params_dtype, skip_init=args.skip_init, device=torch.cuda.current_device() if hasattr(args, 'use_gpu_initialization') and args.use_gpu_initialization else torch.device('cpu'), **kwargs)

    def reinit(self, mixin_names=None):
        for k, m in self.mixins.items():
            if mixin_names is None or k in mixin_names:
                m.reinit(self)

    def add_mixin(self, name, new_mixin, reinit=False):
        assert name not in self.mixins
        assert isinstance(new_mixin, BaseMixin)
        self.mixins[name] = new_mixin
        object.__setattr__(new_mixin, 'transformer', self.transformer)
        self.collect_hooks_()
        if reinit:
            new_mixin.reinit(self)

    def del_mixin(self, name):
        assert name in self.mixins
        del self.mixins[name]
        self.collect_hooks_()

    def get_mixin(self, name):
        return self.mixins[name]

    def forward(self, *args, **kwargs):
        self.transformer.hooks.clear()
        self.transformer.hooks.update(self.hooks)
        return self.transformer(*args, **kwargs)

    def collect_hooks_(self):
        names = list(HOOKS_DEFAULT.keys())
        hooks = {}
        hook_origins = {}
        for name in names:
            if hasattr(self, name):
                hooks[name] = getattr(self, name)
                hook_origins[name] = 'model'
            for mixin_name, m in self.mixins.items():
                if hasattr(m, name):
                    if hasattr(getattr(m, name), 'non_conflict'):
                        signature = inspect.signature(getattr(m, name))
                        if 'old_impl' not in signature.parameters:
                            raise ValueError(f'Hook {name} at {mixin_name} must accept old_impl as an argument.')
                        if name in hooks:
                            old_impl = hooks[name]
                        elif name == 'attention_fn':
                            old_impl = HOOKS_DEFAULT[name]
                        else:
                            old_impl = partial(HOOKS_DEFAULT[name], self)
                        old_origin = hook_origins.get(name, 'default')
                        hooks[name] = partial(getattr(m, name), old_impl=old_impl)
                        hook_origins[name] = mixin_name + ' -> ' + old_origin
                    elif name in hooks and not hasattr(hooks[name], 'replacable'):
                        raise ValueError(f'Hook {name} conflicts at {mixin_name} and {hook_origins[name]}.')
                    else:
                        if name in hooks and hasattr(hooks[name], 'replacable'):
                            warnings.warn(f'Hook {name} at {mixin_name} replaces {hook_origins[name]}.')
                        hooks[name] = getattr(m, name)
                        hook_origins[name] = mixin_name
        self.hooks = hooks
        self.hook_origins = hook_origins
        return hooks

    def disable_untrainable_params(self):
        pass

    @classmethod
    def add_model_specific_args(cls, parser):
        return parser

    @classmethod
    def from_pretrained_base(cls, name, args=None, *, home_path=None, url=None, prefix='', build_only=False, overwrite_args={}, **kwargs):
        """Load a pretrained checkpoint of the current model.
            Args:
                name: The identifier of the pretrained model.
                args: NameSpace. will add the loaded args into it. None will create a new model-only one with defaults.
                path: the parent folder of existing `name` model. Default: SAT_HOME.
                url: the url of the model. Default: SAT_URL.
                prefix: the prefix of the checkpoint. Default: ''.
            Returns:
                model: the loaded model.
                args: the loaded args.
        """
        if os.path.exists(name) and os.path.isdir(name):
            model_path = name
        else:
            model_path = auto_create(name, path=home_path, url=url)
        if args is None:
            args = cls.get_args()
        args = update_args_with_file(args, path=os.path.join(model_path, 'model_config.json'))
        args = overwrite_args_by_dict(args, overwrite_args=overwrite_args)
        specific_iteration = kwargs.pop('specific_iteration', None)
        model = get_model(args, cls, **kwargs)
        if not build_only:
            load_checkpoint(model, args, load_path=model_path, prefix=prefix, specific_iteration=specific_iteration)
        return model, args

    @classmethod
    def from_pretrained(cls, name, args=None, *, home_path=None, url=None, prefix='', build_only=False, use_node_group=True, overwrite_args={}, **kwargs):
        if build_only or 'model_parallel_size' not in overwrite_args:
            return cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=build_only, overwrite_args=overwrite_args, **kwargs)
        else:
            new_model_parallel_size = overwrite_args['model_parallel_size']
            if new_model_parallel_size != 1 or new_model_parallel_size == 1 and args.model_parallel_size == 1:
                model, model_args = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=True, overwrite_args=overwrite_args, **kwargs)
                local_rank = get_node_rank() if use_node_group else get_model_parallel_rank()
                world_size = torch.distributed.get_world_size()
                assert world_size % new_model_parallel_size == 0, 'world size should be a multiplier of new model_parallel_size.'
                destroy_model_parallel()
                initialize_model_parallel(1)
                if local_rank == 0:
                    args.skip_init = True
                    args.use_gpu_initialization = False
                    args.device = 'cpu'
                    overwrite_args.pop('model_parallel_size')
                    model_full, args_ = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=False, overwrite_args=overwrite_args, **kwargs)
                    if args_.model_parallel_size != 1:
                        raise Exception("We do not support overwriting model_parallel_size when original model_parallel_size != 1. Try merging the model using `from_pretrained(xxx,overwrite_args={'model_parallel_size':1})` first if you still want to change model_parallel_size!")
                if hasattr(args, 'mode') and args.mode == 'inference':
                    torch.distributed.barrier()
                destroy_model_parallel()
                initialize_model_parallel(new_model_parallel_size)
                if local_rank == 0:
                    mp_split_model_rank0(model, model_full, use_node_group=use_node_group)
                    del model_full
                else:
                    mp_split_model_receive(model, use_node_group=use_node_group)
                reset_random_seed(6)
            else:
                overwrite_args.pop('model_parallel_size')
                model, model_args = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=False, overwrite_args=overwrite_args, **kwargs)
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                assert world_size == model_args.model_parallel_size, 'world size should be equal to model_parallel_size.'
                destroy_model_parallel()
                initialize_model_parallel(1)
                if rank == 0:
                    args.use_gpu_initialization = False
                    args.device = 'cpu'
                    overwrite_args['model_parallel_size'] = 1
                    model_full, args_ = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=True, overwrite_args=overwrite_args, **kwargs)
                torch.distributed.barrier()
                destroy_model_parallel()
                initialize_model_parallel(model_args.model_parallel_size)
                if rank == 0:
                    mp_merge_model_rank0(model, model_full)
                    model, model_args = model_full, args_
                else:
                    mp_merge_model_send(model)
                    model_args.model_parallel_size = 1
                destroy_model_parallel()
                initialize_model_parallel(1)
            return model, model_args

    @classmethod
    def list_avail_args(cls, print=True):
        """List all available args of the current model."""
        parser = argparse.ArgumentParser()
        add_model_config_args(parser)
        if hasattr(cls, 'add_model_specific_args'):
            cls.add_model_specific_args(parser)
        if print:
            print_parser(parser)
        return parser

    @classmethod
    def get_args(cls, **kwargs):
        """Get the parsed args of the current model.
            Args:
                **kwargs: will override the default args.
            Returns:
                args: the parsed args.
        """
        parser = cls.list_avail_args(print=False)
        args = parser.parse_args([])
        for k, v in kwargs.items():
            if hasattr(args, k) or k in ['fp16']:
                setattr(args, k, v)
            else:
                print_rank0(f'warning: Unknown arg {k} for class {cls.__name__}.', level='DEBUG')
                setattr(args, k, v)
        return args


def non_conflict(func):
    """mark a hook function as non-conflict,
    so that it can be compatible with any already defined hooks.
    e.g. PrefixTuningMixin.attention_fn
    """
    func.non_conflict = True
    return func


class CachedAutoregressiveModel(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer=transformer, **kwargs)
        self.add_mixin('auto-regressive', CachedAutoregressiveMixin())


class EncoderDecoderModel(torch.nn.Module):

    def __init__(self, args, encoder=None, decoder=None, tie_word_embeddings=True, **kwargs):
        super(EncoderDecoderModel, self).__init__()
        if encoder is not None:
            assert isinstance(encoder, BaseModel)
            self.encoder = encoder
        else:
            self.encoder = BaseModel(args, **kwargs)
        self.encoder.add_mixin('final', EncoderFinalMixin())
        if decoder is not None:
            assert isinstance(decoder, BaseModel)
            self.decoder = decoder
        else:
            dec_args = argparse.Namespace(**vars(args))
            dec_args.enc_hidden_size = dec_args.hidden_size
            override_attrs = ['num_layers', 'hidden_size', 'num_attention_heads', 'layernorm_ordermax_sequence_length', 'inner_hidden_size', 'hidden_size_per_attention_head']
            for name in override_attrs:
                dec_attr = getattr(dec_args, 'dec_' + name, None)
                if dec_attr is not None:
                    setattr(dec_args, name, dec_attr)
            self.decoder = BaseModel(args, is_decoder=True, **kwargs)
        self.tie_word_embeddings = tie_word_embeddings
        if tie_word_embeddings:
            self.decoder.transformer.word_embeddings = self.encoder.transformer.word_embeddings

    def reinit(self, mixin_names):
        self.encoder.reinit(mixin_names)
        self.decoder.reinit(mixin_names)

    def disable_untrainable_params(self):
        self.encoder.disable_untrainable_params()
        self.decoder.disable_untrainable_params()

    def encode(self, input_ids, position_ids, attention_mask=None, **kw_args):
        encoder_outputs, *_dumps = self.encoder(input_ids, position_ids, attention_mask, **kw_args)
        return encoder_outputs

    def decode(self, input_ids, position_ids, attention_mask, encoder_outputs, cross_attention_mask=None, **kw_args):
        if attention_mask is None:
            batch_size, seq_length = input_ids.size()[:2]
            seq_ids = torch.arange(seq_length, device=input_ids.device)
            attention_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            attention_mask = attention_mask
            attention_mask = attention_mask[:, None, :, :]
        return self.decoder(input_ids, position_ids, attention_mask, encoder_outputs=encoder_outputs, cross_attention_mask=cross_attention_mask, **kw_args)

    def forward(self, enc_input_ids, enc_position_ids, dec_input_ids, dec_position_ids, *, enc_attention_mask=None, dec_attention_mask=None, cross_attention_mask=None, **kw_args):
        batch_size, seq_length = enc_input_ids.size()[:2]
        if enc_attention_mask is None:
            enc_attention_mask = torch.ones(1, 1, 1, seq_length, dtype=self.encoder.transformer.word_embeddings.weight.dtype, device=enc_input_ids.device)
        if cross_attention_mask is None:
            cross_attention_mask = enc_attention_mask
        encoder_outputs = self.encode(enc_input_ids, enc_position_ids, enc_attention_mask, **kw_args)
        decoder_outputs, *mems = self.decode(dec_input_ids, dec_position_ids, dec_attention_mask, encoder_outputs=encoder_outputs, cross_attention_mask=cross_attention_mask, **kw_args)
        return encoder_outputs, decoder_outputs, *mems

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('EncoderDecoderModel', 'T5 or Bart')
        group.add_argument('--dec-num-layers', type=int, default=None)
        group.add_argument('--dec-hidden-size', type=int, default=None)
        group.add_argument('--dec-num-attention-heads', type=int, default=None)
        group.add_argument('--dec-max-sequence-length', type=int, default=None)
        group.add_argument('--dec-inner-hidden-size', type=int, default=None)
        group.add_argument('--dec-hidden-size-per-attention-head', type=int, default=None)
        group.add_argument('--dec-layernorm-order', type=str, default=None)
        return parser

    @classmethod
    def from_pretrained(cls, args, name, *, home_path=None, url=None):
        model_path = auto_create(name, path=home_path, url=url)
        args = update_args_with_file(args, path=os.path.join(model_path, 'model_config.json'))
        model = get_model(args, cls)
        load_checkpoint(model, args, load_path=model_path)
        return model, args


class HackLinear(nn.Linear):

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'weight' in state_dict:
            self.weight.data.copy_(state_dict[prefix + 'weight'])
        if prefix + 'bias' in state_dict:
            self.bias.data.copy_(state_dict[prefix + 'bias'])


class HackParameterList(nn.ParameterList):

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for i in range(len(self)):
            if prefix + str(i) in state_dict:
                self[i].data.copy_(state_dict[prefix + str(i)])


class HackColumnParallelLinear(ColumnParallelLinear):

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'weight' in state_dict:
            self.weight.data.copy_(state_dict[prefix + 'weight'])
        if prefix + 'bias' in state_dict:
            self.bias.data.copy_(state_dict[prefix + 'bias'])


class HackRowParallelLinear(RowParallelLinear):

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'weight' in state_dict:
            self.weight.data.copy_(state_dict[prefix + 'weight'])
        if prefix + 'bias' in state_dict:
            self.bias.data.copy_(state_dict[prefix + 'bias'])


map_cls = {nn.Linear: (HackLinear, {}), ColumnParallelLinear: (HackColumnParallelLinear, {'gather_output': False}), RowParallelLinear: (HackRowParallelLinear, {'input_is_parallel': True})}


class LoraLinear(nn.Module):

    def __init__(self, original_cls, partition, in_dim, out_dim, r, lora_alpha=1.0, lora_dropout=0.0, qlora=False, original_obj=None):
        super().__init__()
        assert original_obj is not None, 'original linear object must be given!'
        if lora_dropout and lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        bias = original_obj.bias is not None
        dtype = original_obj.weight.dtype
        if qlora:
            try:
                self.original = HackLinearNF4(in_dim, out_dim, bias=bias)
            except:
                raise Exception('Build 4bit layer failed. You need to install the latest bitsandbytes. Try `pip install bitsandbytes`. If you still meet error after installation, try running `from bitsandbytes.nn import LinearNF4` with python and fix the error.')
        else:
            base_cls, kwargs = map_cls[original_cls]
            if original_cls is ColumnParallelLinear:
                kwargs['stride'] = partition
                kwargs['skip_init'] = True
                kwargs['params_dtype'] = dtype
            elif original_cls is RowParallelLinear:
                kwargs['final_bias'] = original_obj.final_bias
                kwargs['skip_init'] = True
                kwargs['params_dtype'] = dtype
            else:
                kwargs['dtype'] = dtype
            self.original = base_cls(in_dim, out_dim, **kwargs, bias=bias)
        self.original.weight.data.copy_(original_obj.weight.data.detach().clone())
        if bias:
            self.original.bias.data.copy_(original_obj.bias.data.detach().clone())
        if type(partition) is int:
            self.matrix_A = HackParameterList([nn.Parameter(torch.empty((r, original_obj.weight.shape[1]), dtype=dtype)) for _ in range(partition)])
            self.matrix_B = HackParameterList([nn.Parameter(torch.empty((original_obj.weight.shape[0] // partition, r), dtype=dtype)) for _ in range(partition)])
            for i in range(partition):
                nn.init.kaiming_uniform_(self.matrix_A[i], a=math.sqrt(5))
                nn.init.zeros_(self.matrix_B[i])
                self.matrix_B[i].model_parallel = True
                self.matrix_B[i].tensor_model_parallel = True
        else:
            new_sizes = [(original_obj.weight.shape[0] // sum(partition) * i) for i in partition]
            self.matrix_A = HackParameterList([nn.Parameter(torch.empty((r, original_obj.weight.shape[1]), dtype=dtype)) for _ in partition])
            self.matrix_B = HackParameterList([nn.Parameter(torch.empty((sz, r), dtype=dtype)) for sz in new_sizes])
            for i in range(len(partition)):
                nn.init.kaiming_uniform_(self.matrix_A[i], a=math.sqrt(5))
                nn.init.zeros_(self.matrix_B[i])
                self.matrix_B[i].model_parallel = True
                self.matrix_B[i].tensor_model_parallel = True
        self.partition = partition

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'weight' in state_dict:
            self.original._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        else:
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        mixed_raw_layer = self.original(x)
        x = self.lora_dropout(x)
        lora_outputs = []
        for mA, mB in zip(self.matrix_A, self.matrix_B):
            lora_outputs.append(copy_to_model_parallel_region(x @ mA.T) @ mB.T * self.scaling)
        mixed_raw_layer = mixed_raw_layer + torch.cat(lora_outputs, -1)
        return mixed_raw_layer


def merge_linear_lora(lin):
    if lin.original.weight.data.dtype is not torch.uint8:
        weight = lin.original.weight
        out_dim, in_dim = weight.shape
        new_lin = nn.Linear(in_dim, out_dim, dtype=weight.data.dtype, bias=lin.original.bias is not None)
    else:
        weight = F.dequantize_fp4(lin.original.weight.data, lin.original.weight.quant_state)
        out_dim, in_dim = weight.shape
        new_lin = HackLinearNF4(in_dim, out_dim, bias=lin.original.bias is not None)
    if lin.original.bias is not None:
        new_lin.bias.data = lin.original.bias.data
    new_qkv = []
    for mA, mB in zip(lin.matrix_A, lin.matrix_B):
        new_qkv.append(mB.data.float() @ mA.data.float() * lin.scaling)
    new_qkv = torch.cat(new_qkv, -2)
    guess_type = lin.original.bias.data.dtype if lin.original.bias is not None else lin.original.weight.data.dtype
    if guess_type is torch.uint8:
        guess_type = torch.float32
    new_lin.weight.data = weight + new_qkv
    return new_lin if torch.cuda.is_available() else new_lin


def replace_linear_with_lora(lin, partition, r, *args, **kw_args):
    if kw_args.get('in_size', None) is not None:
        in_size = kw_args.pop('in_size')
        out_size = kw_args.pop('out_size')
        if out_size is None:
            out_size = in_size * partition
        out_dim, in_dim = out_size, in_size
    else:
        out_dim, in_dim = lin.weight.shape
    original_cls = type(lin)
    new_layer = LoraLinear(original_cls, partition, in_dim, out_dim, r, *args, **kw_args, original_obj=lin)
    device = lin.weight.device
    del lin
    return new_layer


class lm_head(torch.nn.Module):

    def __init__(self, vocab_size, hidden_size, layernorm_epsilon=1e-05):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.dense(x)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class BertModel(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        super(BertModel, self).__init__(args, transformer=transformer, activation_func=gelu, **kwargs)
        self.add_mixin('bert-final', BertFinalMixin(args.vocab_size, args.hidden_size))
        self.add_mixin('bert-type', BertTypeMixin(args.num_types, args.hidden_size))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('BERT', 'BERT Configurations')
        group.add_argument('--num-types', type=int, default=2, help='Number of token types')
        return super().add_model_specific_args(parser)


class CaiTDecoder(BaseModel):

    def __init__(self, args, transformer=None, layernorm_epsilon=1e-06):
        super().__init__(args, is_decoder=True, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        self.add_mixin('cls', ClsMixin(args.hidden_size, args.num_classes))
        self.add_mixin('dec_forward', DecForward(args.hidden_size, args.num_layers, init_values=args.init_scale))

    @classmethod
    def add_model_specific_args(cls, parser):
        return super().add_model_specific_args(parser)


class ViTProperty:
    """
    Store some hyper-parameters such as image size and patch size.
    seq_len = pre_len + image_len + post_len
    """

    def __init__(self, image_size, patch_size, pre_len, post_len, **kwargs):
        assert isinstance(image_size, Iterable) and len(image_size) == 2
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.pre_len = pre_len
        self.post_len = post_len
        self.seq_len = self.pre_len + self.num_patches + self.post_len


class ViTModel(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        property = ViTProperty(args.image_size, args.patch_size, args.pre_len, args.post_len)
        args.max_sequence_length = property.pre_len + property.num_patches + property.post_len
        if 'activation_func' not in kwargs:
            kwargs['activation_func'] = gelu
        super().__init__(args, transformer=transformer, **kwargs)
        self.transformer.property = property
        self.add_mixin('patch_embedding', ImagePatchEmbeddingMixin(args.in_channels, args.hidden_size, property))
        self.add_mixin('pos_embedding', InterpolatedPositionEmbeddingMixin())
        self.add_mixin('cls', ClsMixin(args.hidden_size, args.num_classes))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ViT', 'ViT Configurations')
        group.add_argument('--image-size', nargs='+', type=int, default=[224, 224])
        group.add_argument('--pre-len', type=int, default=1)
        group.add_argument('--post-len', type=int, default=0)
        group.add_argument('--in-channels', type=int, default=3)
        group.add_argument('--num-classes', type=int, default=21843)
        group.add_argument('--patch-size', type=int, default=16)
        return parser


class CaiTEncoder(ViTModel):

    def __init__(self, args, transformer=None, layernorm_epsilon=1e-06, use_final_layernorm=False):
        super().__init__(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon, use_final_layernorm=use_final_layernorm)
        self.del_mixin('cls')
        self.add_mixin('attn', AttnMixin(args.num_attention_heads, args.num_layers))
        self.add_mixin('enc_forward', EncForward(args.hidden_size, args.num_layers, init_values=args.init_scale))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CaiT-enc', 'CaiT encoder Configurations')
        group.add_argument('--init-scale', type=float, default=0.0001)
        return super().add_model_specific_args(parser)


class CaiT(EncoderDecoderModel):

    def __init__(self, args, transformer=None, layernorm_epsilon=1e-06):
        encoder = CaiTEncoder(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        dec_args = argparse.Namespace(**vars(args))
        override_attrs = ['num_layers', 'hidden_size', 'num_attention_heads', 'layernorm_order', 'max_sequence_length', 'inner_hidden_size', 'hidden_size_per_attention_head']
        for name in override_attrs:
            dec_attr = getattr(dec_args, 'dec_' + name, None)
            if dec_attr is not None:
                setattr(dec_args, name, dec_attr)
        decoder = CaiTDecoder(dec_args, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        super().__init__(args, encoder=encoder, decoder=decoder)

    def forward(self, input_ids, enc_position_ids, dec_position_ids, *, enc_attention_mask=None, dec_attention_mask=None, cross_attention_mask=None, **kw_args):
        if enc_attention_mask is None:
            enc_attention_mask = torch.ones(1, 1, dtype=self.encoder.transformer.word_embeddings.weight.dtype, device=input_ids.device)
        if cross_attention_mask is None:
            cross_attention_mask = enc_attention_mask
        encoder_outputs = self.encode(input_ids, enc_position_ids, enc_attention_mask, **kw_args)
        decoder_outputs, *mems = self.decode(input_ids, dec_position_ids, dec_attention_mask, encoder_outputs=encoder_outputs, cross_attention_mask=cross_attention_mask, **kw_args)
        return encoder_outputs, decoder_outputs, *mems


class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class ChatGLM2Model(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        super(ChatGLM2Model, self).__init__(args, transformer=transformer, activation_func=F.silu, layernorm=RMSNorm, **kwargs)
        del self.transformer.position_embeddings
        self.add_mixin('chatglm-final', ChatGLMFinalMixin(args.vocab_size, args.hidden_size))
        self.add_mixin('attn', ChatGLM2AttnMixin(args.hidden_size, args.num_attention_heads, args.max_sequence_length))
        if not (hasattr(args, 'is_gated_mlp') and args.is_gated_mlp):
            self.add_mixin('mlp', SwiGLUMixin(args.num_layers, args.hidden_size, args.inner_hidden_size, bias=args.use_bias))

    def position_embedding_forward(self, position_ids, output_cross_layer, **kw_args):
        return None

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, dtype=next(self.parameters()).dtype, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length, dtype=next(self.parameters()).dtype, device=input_ids.device), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def get_position_ids(self, input_ids):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids

    def forward(self, input_ids, position_ids=None, attention_mask=None, past_key_values=None, **kwargs):
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids)
        if attention_mask is not None and attention_mask.ndim == 4:
            pass
        elif past_key_values is not None and input_ids.size(0) == 1:
            attention_mask = torch.tensor([[1]], dtype=torch.long, device=input_ids.device)
        else:
            attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        if attention_mask is not None and attention_mask.dtype is torch.bool:
            attention_mask = ~attention_mask
        attention_mask = attention_mask
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[..., -1:]
            if input_ids.size(0) != 1:
                attention_mask = attention_mask[:, :, -1:]
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, **kwargs)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ChatGLM2', 'ChatGLM2 Configurations')
        return super().add_model_specific_args(parser)


class ChatGLM3Model(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        super(ChatGLM3Model, self).__init__(args, transformer=transformer, activation_func=F.silu, layernorm=RMSNorm, **kwargs)
        del self.transformer.position_embeddings
        self.add_mixin('chatglm-final', ChatGLMFinalMixin(args.vocab_size, args.hidden_size))
        self.add_mixin('attn', ChatGLM3AttnMixin(args.hidden_size, args.num_attention_heads, args.max_sequence_length, args.base_scale))
        self.add_mixin('mlp', SwiGLUMixin(args.num_layers, args.hidden_size, args.inner_hidden_size, bias=args.use_bias))

    def position_embedding_forward(self, position_ids, output_cross_layer, **kw_args):
        return None

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, dtype=next(self.parameters()).dtype, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length, dtype=next(self.parameters()).dtype, device=input_ids.device), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def get_position_ids(self, input_ids):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids

    def forward(self, input_ids, position_ids=None, attention_mask=None, past_key_values=None, **kwargs):
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids)
        if attention_mask is not None and attention_mask.ndim == 4:
            pass
        elif past_key_values is not None and input_ids.size(0) == 1:
            attention_mask = torch.tensor([[1]], dtype=torch.long, device=input_ids.device)
        else:
            attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        if attention_mask is not None and attention_mask.dtype is torch.bool:
            attention_mask = ~attention_mask
        attention_mask = attention_mask
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[..., -1:]
            if input_ids.size(0) != 1:
                attention_mask = attention_mask[:, :, -1:]
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, **kwargs)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ChatGLM3', 'ChatGLM3 Configurations')
        group.add_argument('--base-scale', type=float, default=1.0)
        return super().add_model_specific_args(parser)


class ChatGLM4Model(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        super(ChatGLM4Model, self).__init__(args, transformer=transformer, activation_func=F.silu, layernorm=RMSNorm, **kwargs)
        del self.transformer.position_embeddings
        self.add_mixin('chatglm-final', ChatGLMFinalMixin(args.vocab_size, args.hidden_size))
        self.add_mixin('attn', ChatGLM4AttnMixin(args.hidden_size, args.num_attention_heads, args.max_sequence_length))
        if not (hasattr(args, 'is_gated_mlp') and args.is_gated_mlp):
            self.add_mixin('mlp', SwiGLUMixin(args.num_layers, args.hidden_size, args.inner_hidden_size, bias=args.use_bias))

    def position_embedding_forward(self, position_ids, output_cross_layer, **kw_args):
        return None

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, dtype=next(self.parameters()).dtype, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length, dtype=next(self.parameters()).dtype, device=input_ids.device), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def get_position_ids(self, input_ids):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids

    def forward(self, input_ids, position_ids=None, attention_mask=None, past_key_values=None, **kwargs):
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids)
        if attention_mask is not None and attention_mask.ndim == 4:
            pass
        elif past_key_values is not None and input_ids.size(0) == 1:
            attention_mask = torch.tensor([[1]], dtype=torch.long, device=input_ids.device)
        else:
            attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        if attention_mask is not None and attention_mask.dtype is torch.bool:
            attention_mask = ~attention_mask
        attention_mask = attention_mask
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[..., -1:]
            if input_ids.size(0) != 1:
                attention_mask = attention_mask[:, :, -1:]
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, **kwargs)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ChatGLM4', 'ChatGLM4 Configurations')
        return super().add_model_specific_args(parser)


class RotaryEmbedding(nn.Module):

    def __init__(self, dim, original_impl=False, device=None, dtype=None, base_scale=1.0):
        super().__init__()
        self.base_scale = base_scale
        inv_freq = 1.0 / (10000 * base_scale) ** (torch.arange(0, dim, 2, device=device) / dim)
        self.register_buffer('inv_freq', inv_freq)
        self.dim = dim
        self.original_impl = original_impl
        self.cache = None

    def forward_impl(self, seq_len: 'int', n_elem: 'int', dtype: 'torch.dtype', device: 'torch.device', base: 'int'=10000):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        if n_elem == self.dim:
            theta = self.inv_freq
        else:
            theta = 1.0 / (base * self.base_scale) ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem)
        seq_idx = torch.arange(seq_len, dtype=torch.float32, device=device)
        idx_theta = torch.outer(seq_idx, theta)
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        self.cache = cache
        return cache

    def forward(self, max_seq_len, offset=0):
        if self.cache is not None and max_seq_len <= self.cache.shape[0]:
            return self.cache[:max_seq_len]
        return self.forward_impl(max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device)


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x.ndim - 1)


@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
    return q, k


class ChatGLMModel(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        super(ChatGLMModel, self).__init__(args, transformer=transformer, activation_func=gelu, **kwargs)
        del self.transformer.position_embeddings
        self.add_mixin('chatglm-final', ChatGLMFinalMixin(args.vocab_size, args.hidden_size))
        self.add_mixin('chatglm-attn', ChatGLMAttnMixin(args.hidden_size, args.num_attention_heads))
        self.add_mixin('chatglm-layer', ChatGLMLayerMixin(args.num_layers))
        self.bos_token_id = args.bos_token_id
        self.mask_token_id = args.mask_token_id
        self.gmask_token_id = args.gmask_token_id
        self.pad_token_id = args.pad_token_id

    def position_embedding_forward(self, position_ids, output_cross_layer, **kw_args):
        return None

    def forward(self, input_ids, position_ids=None, attention_mask=None, past_key_values=None, **kwargs):
        if attention_mask is None and position_ids is None:
            attention_mask, position_ids = self.get_inputs(input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, **kwargs)
        if attention_mask is not None and attention_mask.dtype is torch.bool:
            attention_mask = (~attention_mask).long()
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[..., -1:]
            if input_ids.size(0) != 1:
                attention_mask = attention_mask[:, :, -1:]
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, **kwargs)

    def get_inputs(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        if attention_mask is None:
            if past_key_values is not None and input_ids.size(0) == 1:
                attention_mask = torch.tensor([[1]], dtype=torch.long, device=input_ids.device)
            else:
                attention_mask = self.get_masks(input_ids=input_ids, device=input_ids.device, **kwargs)
        if position_ids is None:
            MASK, gMASK = self.mask_token_id, self.gmask_token_id
            mask_token = gMASK if gMASK in input_ids else MASK
            use_gmask = True if gMASK in input_ids else False
            mask_positions = [seq.tolist().index(mask_token) for seq in input_ids]
            position_ids = self.get_position_ids(input_ids=input_ids, mask_positions=mask_positions, device=input_ids.device, gmask=use_gmask, **kwargs)
        return attention_mask, position_ids

    def get_pad_length(self, seq):
        l = 0
        while l < len(seq) and seq[l] == self.pad_token_id:
            l += 1
        return l

    def get_masks(self, input_ids, device, **kwargs):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.bos_token_id) for seq in input_ids]
        attention_mask = torch.ones((batch_size, seq_length, seq_length), dtype=next(self.parameters()).dtype, device=device)
        attention_mask.tril_()
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        pad_lengths = [self.get_pad_length(seq.tolist()) for seq in input_ids]
        for i, pad_length in enumerate(pad_lengths):
            attention_mask[i, :, :pad_length] = 0
            attention_mask[i, :pad_length, :] = 0
        attention_mask.unsqueeze_(1)
        return attention_mask

    def get_position_ids(self, input_ids, mask_positions, device, gmask=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        pad_lengths = [self.get_pad_length(seq.tolist()) for seq in input_ids]
        context_lengths = [seq.tolist().index(self.bos_token_id) for seq in input_ids]
        position_ids = [torch.arange(seq_length - pad_length, dtype=torch.long, device=device) for pad_length in pad_lengths]
        for i, (context_length, pad_length) in enumerate(zip(context_lengths, pad_lengths)):
            position_ids[i][context_length - pad_length:] = mask_positions[i] - pad_length
        block_position_ids = [torch.cat((torch.zeros(context_length, dtype=torch.long, device=device), torch.arange(seq_length - context_length, dtype=torch.long, device=device) + 1)) for context_length in context_lengths]
        block_position_ids = torch.stack(block_position_ids, dim=0)
        position_ids = [torch.cat((torch.zeros(pad_length, dtype=torch.long, device=device), range_pos)) for pad_length, range_pos in zip(pad_lengths, position_ids)]
        position_ids = torch.stack(position_ids, dim=0)
        position_ids = torch.stack((position_ids, block_position_ids), dim=1)
        return position_ids

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ChatGLM', 'ChatGLM Configurations')
        group.add_argument('--bos-token-id', type=int)
        group.add_argument('--mask-token-id', type=int)
        group.add_argument('--gmask-token-id', type=int)
        group.add_argument('--pad-token-id', type=int)
        return super().add_model_specific_args(parser)


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input):
        return input * torch.sigmoid(1.702 * input)


class EVA2CLIPModel(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        property = ViTProperty(args.image_size, args.patch_size, args.pre_len, args.post_len)
        args.max_sequence_length = property.pre_len + property.num_patches + property.post_len
        if 'activation_func' not in kwargs:
            kwargs['activation_func'] = gelu
        super().__init__(args, transformer=transformer, **kwargs)
        self.transformer.property = property
        self.add_mixin('patch_embedding', ImagePatchEmbeddingMixin(args.in_channels, args.hidden_size, property))
        self.add_mixin('pos_embedding', InterpolatedPositionEmbeddingMixin())
        self.add_mixin('final', IdentityMixin())
        self.add_mixin('newpost', NewLayerForward())
        if 'USE_XFORMERS' in os.environ and os.environ['USE_XFORMERS']:
            self.add_mixin('xattn', XAttn(args.hidden_size // args.num_attention_heads))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('EVA2CLIP', 'EVA2CLIP Configurations')
        group.add_argument('--image-size', nargs='+', type=int, default=[224, 224])
        group.add_argument('--pre-len', type=int, default=1)
        group.add_argument('--post-len', type=int, default=0)
        group.add_argument('--in-channels', type=int, default=3)
        group.add_argument('--patch-size', type=int, default=16)
        return parser


class GLU(nn.Module):

    def __init__(self, args, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, args.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(args.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(args.hidden_size, args.inner_hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.inner_hidden_size, bias=False)
        self.dense_4h_to_h = nn.Linear(args.inner_hidden_size, args.hidden_size, bias=False)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x


def extract_model_specific_args_from_model(args, model):
    parser = argparse.ArgumentParser()
    if hasattr(model, 'module'):
        model = model.module
    if isinstance(model, torch.nn.Module):
        for md in model.modules():
            if hasattr(md, 'add_model_specific_args'):
                try:
                    md.add_model_specific_args(parser)
                except argparse.ArgumentError as e:
                    None
    ret = {}
    for k in vars(parser.parse_args([])).keys():
        if hasattr(args, k):
            ret[k] = getattr(args, k)
    return ret


def extract_model_specific_args_to_dump(args, model):
    module = model.module if hasattr(model, 'module') else model
    to_dump = {'model_class': type(module).__name__}
    if hasattr(args, 'tokenizer_type') and args.tokenizer_type != 'fake':
        to_dump['tokenizer_type'] = args.tokenizer_type
    arch_args_list = ['num_layers', 'hidden_size', 'num_attention_heads', 'vocab_size', 'layernorm_order', 'model_parallel_size', 'max_sequence_length']
    for name in arch_args_list:
        if hasattr(args, name) and getattr(args, name) is not None:
            to_dump[name] = getattr(args, name)
    optional_arch_args_list = [('is_decoder', False), ('cross_attn_hidden_size', None), ('use_bias', True), ('use_qkv_bias', False), ('inner_hidden_size', None), ('hidden_size_per_attention_head', None), ('cross_hidden_size_per_attention_head', None), ('use_final_layernorm', True), ('layernorm_epsilon', 1e-05), ('num_multi_query_heads', 0), ('cross_num_multi_query_heads', 0), ('row_parallel_linear_final_bias', True), ('is_gated_mlp', False), ('is_rotary_emb', False), ('parallel_output', False), ('num_experts', 1)]
    if hasattr(module, 'transformer'):
        for name, default in optional_arch_args_list:
            if module.transformer.__dict__[name] != default:
                to_dump[name] = module.transformer.__dict__[name]
    model_specific_args = extract_model_specific_args_from_model(args, module)
    to_dump.update(model_specific_args)
    return to_dump


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def override_dist_dtype_device_args(args, b={}):
    if args.mode == 'inference':
        minimal_args = argparse.Namespace(world_size=args.world_size, rank=args.rank, local_rank=args.local_rank, skip_init=args.skip_init, use_gpu_initialization=args.use_gpu_initialization, deepspeed=args.deepspeed, bf16=args.bf16, fp16=args.fp16, mode=args.mode, device=args.device)
    else:
        minimal_args = argparse.Namespace(world_size=args.world_size, rank=args.rank, local_rank=args.local_rank, skip_init=args.skip_init, use_gpu_initialization=args.use_gpu_initialization, deepspeed=args.deepspeed, bf16=args.bf16, fp16=args.fp16, mode=args.mode, checkpoint_activations=args.checkpoint_activations, checkpoint_num_layers=args.checkpoint_num_layers, device=args.device, hidden_dropout=0.0, attention_dropout=0.0)
    if hasattr(args, 'model_parallel_size'):
        b['model_parallel_size'] = args.model_parallel_size
    return argparse.Namespace(**deepcopy(b), **vars(minimal_args))


class TextEncoder(BaseModel):

    def __init__(self, args, layernorm_epsilon=1e-05, activation_func=QuickGELUActivation()):
        super().__init__(args, layernorm_epsilon=layernorm_epsilon, activation_func=activation_func)
        self.add_mixin('text_enc', TextMixin(args.hidden_size, args.projection_dim))


class ImageEncoder(ViTModel):

    def __init__(self, args, layernorm_epsilon=1e-05, activation_func=QuickGELUActivation()):
        super().__init__(args, layernorm_epsilon=layernorm_epsilon, activation_func=activation_func)
        self.del_mixin('cls')
        self.add_mixin('image_enc', ImageMixin(args.hidden_size, args.projection_dim, layernorm_epsilon))
        self.del_mixin('patch_embedding')
        self.add_mixin('patch_embedding', PatchMixin(args.in_channels, args.hidden_size, self.property))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CLIP-image', 'CLIP image encoder Configurations')
        group.add_argument('--projection-dim', type=int)
        return super().add_model_specific_args(parser)


class CLIP(nn.Module):

    def __init__(self, args, layernorm_epsilon=1e-05):
        super().__init__()
        self.image_encoder = ImageEncoder(args, layernorm_epsilon=layernorm_epsilon)
        text_args = argparse.Namespace(**vars(args))
        override_attrs = ['vocab_size', 'num_layers', 'hidden_size', 'num_attention_heads', 'layernorm_order', 'max_sequence_length', 'inner_hidden_size', 'hidden_size_per_attention_head']
        for name in override_attrs:
            text_attr = getattr(text_args, 'text_' + name, None)
            if text_attr is not None:
                setattr(text_args, name, text_attr)
        self.text_encoder = TextEncoder(text_args, layernorm_epsilon=layernorm_epsilon)
        self.logit_scale = nn.Parameter(torch.ones([]) * args.logit_scale_init_value)

    def encode_image(self, input_ids, position_ids, attention_mask=None, **kw_args):
        return self.image_encoder(input_ids, position_ids, attention_mask, **kw_args)

    def encode_text(self, input_ids, position_ids, attention_mask, **kw_args):
        return self.text_encoder(input_ids, position_ids, attention_mask, **kw_args)

    def reinit(self, mixin_names):
        self.image_encoder.reinit(mixin_names)
        self.text_encoder.reinit(mixin_names)

    def forward(self, image_input_ids, image_position_ids, text_input_ids, text_position_ids, *, image_attention_mask=None, text_attention_mask=None, **kw_args):
        image_embeds, *image_mems = self.encode_image(image_input_ids, image_position_ids, attention_mask=image_attention_mask, **kw_args)
        text_embeds, *text_mems = self.encode_text(text_input_ids, text_position_ids, attention_mask=text_attention_mask, **kw_args)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T
        return image_embeds, text_embeds, logits_per_text, logits_per_image

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('SiameseModel', 'CLIP')
        group.add_argument('--text-layernorm-order', type=str, default=None)
        group.add_argument('--text-num-layers', type=int, default=None)
        group.add_argument('--text-hidden-size', type=int, default=None)
        group.add_argument('--text-num-attention-heads', type=int, default=None)
        group.add_argument('--text-max-sequence-length', type=int, default=None)
        group.add_argument('--text-inner-hidden-size', type=int, default=None)
        group.add_argument('--text-hidden-size-per-attention-head', type=int, default=None)
        group.add_argument('--logit-scale-init-value', type=float, default=None)
        return parser

    @classmethod
    def from_pretrained(cls, args, name, *, path=None, url=None):
        model_path = auto_create(name, path=path, url=url)
        args = update_args_with_file(args, path=os.path.join(model_path, 'model_config.json'))
        model = get_model(args, cls)
        load_checkpoint(model, args, load_path=model_path)
        return model, args


class similarFunction(Function):

    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW, casual_mask=False):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = kH, kW
        ctx.casual_mask = casual_mask
        output = similar_forward(x_ori, x_loc, kH, kW, casual_mask)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        casual_mask = ctx.casual_mask
        grad_outputs = grad_outputs.contiguous()
        grad_ori = similar_backward(x_ori, x_loc, grad_outputs, kH, kW, True, casual_mask)
        grad_loc = similar_backward(x_ori, x_loc, grad_outputs, kH, kW, False, casual_mask)
        return grad_ori, grad_loc, None, None, None


f_similar = similarFunction.apply


class weightingFunction(Function):

    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW, casual_mask=False):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = kH, kW
        ctx.casual_mask = casual_mask
        output = weighting_forward(x_ori, x_weight, kH, kW, casual_mask)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        casual_mask = ctx.casual_mask
        grad_outputs = grad_outputs.contiguous()
        grad_ori = weighting_backward_ori(x_ori, x_weight, grad_outputs, kH, kW, casual_mask)
        grad_weight = weighting_backward_weight(x_ori, x_weight, grad_outputs, kH, kW, casual_mask)
        return grad_ori, grad_weight, None, None, None


f_weighting = weightingFunction.apply


def sqrt(x):
    return int(math.sqrt(x) + 0.0001)


def sparse_attention_2d_light(q0, k0, v0, q1, k1, v1, attention_mask, n_head, text_len, kernel_size=9, kernel_size2=7, attention_dropout=None, log_attention_weights=None, **kwargs):
    """
    q0, k0, v0: [batch_size, 1088, hidden_size]
    q1, k1, v1: [batch_size, 4096, h2]
    n_head: int
    attention_mask: [batch_size, 1088, 1088]
    """
    b, s0, h0 = q0.shape
    b, s1, h1 = q1.shape
    h, l0, l1 = h0 // n_head, sqrt(s0 - text_len), sqrt(s1)
    q0 = q0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    v0 = v0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    k0T = k0.reshape(b, s0, n_head, h).permute(0, 2, 3, 1)
    attention_scores = torch.matmul(q0 / math.sqrt(q0.shape[-1]), k0T)
    if log_attention_weights is not None:
        attention_scores += log_attention_weights
    attention_scores = torch.mul(attention_scores, attention_mask) - 10000.0 * (1.0 - attention_mask)
    attention_probs0 = F.softmax(attention_scores, dim=-1)
    q1 = (q1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1) / math.sqrt(h1 // n_head)).contiguous().view(b * n_head, h1 // n_head, l1, l1)
    k1 = k1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1).contiguous().view(b * n_head, h1 // n_head, l1, l1)
    v1 = v1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1).contiguous().view(b * n_head, h1 // n_head, l1, l1)
    scores_1_to_1 = f_similar(q1, k1, kernel_size * 2 - 1, kernel_size, True)
    k0T = k0T[..., -l0 ** 2:].reshape(b * n_head, h, l0, l0).contiguous()
    scores_1_to_0 = f_similar(q1, k0T, kernel_size2, kernel_size2, False)
    scores_1 = torch.cat((scores_1_to_0.view(b * n_head, -1, scores_1_to_0.shape[3]), scores_1_to_1.view(b * n_head, -1, scores_1_to_1.shape[3])), dim=-1)
    attention_probs1 = F.softmax(scores_1, dim=-1)
    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs0 = attention_dropout(attention_probs0)
            attention_probs1 = attention_dropout(attention_probs1)
    context0 = torch.matmul(attention_probs0, v0)
    probs_1_to_1 = attention_probs1[:, :, -scores_1_to_1.shape[3]:].view_as(scores_1_to_1)
    context1_to_1 = f_weighting(v1, probs_1_to_1.contiguous(), kernel_size * 2 - 1, kernel_size, True)
    context1 = context1_to_1.view(b, n_head * h, l1 ** 2)
    probs_1_to_0 = attention_probs1[:, :, :scores_1_to_0.shape[3]].view_as(scores_1_to_0)
    v0_part = v0[:, :, -l0 ** 2:].transpose(-1, -2).contiguous().view(b * n_head, h, l0, l0)
    context1_to_0 = f_weighting(v0_part, probs_1_to_0.contiguous(), kernel_size2, kernel_size2, False)
    context1_to_0 = context1_to_0.view(b, n_head * h, l1 ** 2)
    context1 = context1 + context1_to_0
    return context0.transpose(1, 2).reshape(b, s0, h0), context1.transpose(-1, -2)


class Cuda2dModel(BaseModel):

    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        additional_seqlen = args.new_sequence_length - args.max_sequence_length
        self.add_mixin('extra_position_embedding', PositionEmbeddingMixin(additional_seqlen, args.hidden_size))
        self.add_mixin('attention_plus', AttentionMixin(num_layers=args.num_layers, hidden_size=args.hidden_size))
        self.layout = args.layout
        self.kernel_size = args.kernel_size
        self.kernel_size2 = args.kernel_size2
        self.log_attention_weights = None

    def position_embedding_forward(self, position_ids, **kw_args):
        position = position_ids[..., :self.layout[1]]
        position_plus = position_ids[..., self.layout[1]:]
        position_embeddings = torch.cat((self.transformer.position_embeddings(position), self.get_mixin('extra_position_embedding').position_embeddings(position_plus)), dim=-2)
        return position_embeddings

    def attention_forward(self, hidden_states, mask, layer_id=None, log_attention_weights=None, **kw_args):
        attn_module = self.transformer.layers[layer_id].attention
        query_key_value_plus = self.get_mixin('attention_plus').query_key_value[layer_id]
        dense_plus = self.get_mixin('attention_plus').dense[layer_id]
        hidden_states_plus = hidden_states[:, self.layout[1]:]
        hidden_states = hidden_states[:, :self.layout[1]]
        mixed_raw_layer = attn_module.query_key_value(hidden_states)
        q0, k0, v0 = split_tensor_along_last_dim(mixed_raw_layer, 3)
        mixed_raw_layer = query_key_value_plus(hidden_states_plus)
        q1, k1, v1 = split_tensor_along_last_dim(mixed_raw_layer, 3)
        dropout_fn = attn_module.attention_dropout if self.training else None
        context_layer0, context_layer1 = sparse_attention_2d_light(q0, k0, v0, q1, k1, v1, mask, n_head=attn_module.num_attention_heads_per_partition, text_len=self.layout[0], kernel_size=self.kernel_size, kernel_size2=self.kernel_size2, attention_dropout=dropout_fn, log_attention_weights=log_attention_weights)
        output_0 = attn_module.dense(context_layer0)
        output_1 = dense_plus(context_layer1)
        output = torch.cat((output_0, output_1), dim=1)
        return output

    def disable_untrainable_params(self):
        self.transformer.requires_grad_(False)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('Cuda2dModel', 'cuda2d model configurations')
        group.add_argument('--kernel-size', type=int, default=9)
        group.add_argument('--kernel-size2', type=int, default=7)
        group.add_argument('--layout', type=str, default='64,1088,5184')
        group.add_argument('--new-sequence-length', type=int, default=5185)
        return parser


class DistillModel(nn.Module):

    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def forward(self, teacher_kwargs, student_kwargs):
        teacher_logits, *mem_t = self.teacher(**teacher_kwargs)
        student_logits, *mem_s = self.student(**student_kwargs)
        return teacher_logits, student_logits

    def disable_untrainable_params(self):
        for n, p in self.teacher.named_parameters():
            p.requires_grad_(False)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('BERT-distill', 'BERT distill Configurations')
        group.add_argument('--teacher', type=str)
        group.add_argument('--tc-type', type=str)
        group.add_argument('--st-type', type=str)
        return parser

    @classmethod
    def from_pretrained(cls, args, teacher_cls, student_name, student_cls):
        student, args = student_cls.from_pretrained(student_name, args, prefix='student.')
        if isinstance(teacher_cls, type):
            teacher, t_args = teacher_cls.from_pretrained(args.teacher, args)
        else:
            teacher = teacher_cls
        model = DistillModel(teacher, student)
        return model, args


class DPRQuestionEncoder(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        super(DPRQuestionEncoder, self).__init__(args, transformer=transformer, **kwargs)
        self.add_mixin('dpr-type', DPRTypeMixin(args.num_types, args.hidden_size))
        self.add_mixin('dpr-final', DPREncoderFinalMixin())

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('DPRQuestionEncoder', 'DPRQuestionEncoder Configurations')
        group.add_argument('--num-types', type=int)
        return parser


class DPRContextEncoder(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        super(DPRContextEncoder, self).__init__(args, transformer=transformer, **kwargs)
        self.add_mixin('dpr-type', DPRTypeMixin(args.num_types, args.hidden_size))
        self.add_mixin('dpr-final', DPREncoderFinalMixin())

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('DPRContextEncoder', 'DPRContextEncoder Configurations')
        group.add_argument('--num-types', type=int)
        return parser


class DPRReader(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        super(DPRReader, self).__init__(args, transformer=transformer, **kwargs)
        self.add_mixin('dpr-type', DPRTypeMixin(args.num_types, args.hidden_size))
        self.add_mixin('dpr-final', DPRReaderFinalMixin(args.hidden_size, args.projection_dim))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('DPRReader', 'DPRReader Configurations')
        group.add_argument('--num-types', type=int)
        group.add_argument('--projection-dim', type=int)
        return parser


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = dim + shape_len if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


class VisionRotaryEmbeddingFast(nn.Module):

    def __init__(self, dim, pt_seq_len=16, ft_seq_len=None, custom_freqs=None, freqs_for='lang', theta=10000, max_freq=10, num_freqs=1):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)
        None

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class EVA2Model(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        self.property = ViTProperty(args.image_size, args.patch_size, args.pre_len, args.post_len)
        args.max_sequence_length = self.property.seq_len
        super().__init__(args, transformer=transformer, **kwargs)
        self.add_mixin('patch_embedding', MaskedPatchEmbedMixin(args.in_channels, args.hidden_size, self.property))
        self.add_mixin('eva2-final', EVA2FinalMixin(args.predict_feature_dim, args.hidden_size))
        self.add_mixin('eva2-mlp', SwiGLUMixin(args.num_layers, args.hidden_size, args.inner_hidden_size, eps=args.layernorm_epsilon))
        self.add_mixin('eva2-attn', EVA2AttnMixin(args.hidden_size, args.num_attention_heads, self.property))

    def position_embedding_forward(self, position_ids, output_cross_layer, **kw_args):
        return self.transformer.position_embeddings.weight.unsqueeze(0)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('EVA2', 'EVA2 Configurations')
        group.add_argument('--image-size', nargs='+', type=int, default=[224, 224])
        group.add_argument('--pre-len', type=int, default=1)
        group.add_argument('--post-len', type=int, default=0)
        group.add_argument('--in-channels', type=int, default=3)
        group.add_argument('--patch-size', type=int, default=14)
        group.add_argument('--predict-feature-dim', type=int, default=768)
        return parser


class GEGLU(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.activation_fn = F.gelu

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=x.ndim - 1)
        return x1 * self.activation_fn(x2)


class GLM130B(BaseModel):

    def __init__(self, args, transformer=None):
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        super().__init__(args, params_dtype=torch.half if args.fp16 else torch.float, transformer=transformer)
        self.add_mixin('glu-deepnorm', DeepNormWithGLUMixin(args.num_layers, args.hidden_size, args.inner_hidden_size))
        self.add_mixin('fp32-softmax', SelfAttentionWithFP32SoftmaxMixin(args.hidden_size, args.num_attention_heads, args.model_parallel_size))
        self.add_mixin('final-forward', FinalForwardMixin())
        self.add_mixin('non-position-embedding', NonePositionEmbedding())
        del self.transformer.position_embeddings
        self.add_mixin('word-embedding', WordEmbedding())
        self.add_mixin('rotary-embedding', RotaryEmbeddingMixin(args.fp16, args.hidden_size, args.num_attention_heads, args.model_parallel_size, args.position_encoding_2d))
        if not args.no_glu:
            self.get_mixin('glu-deepnorm').reinit()

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument('--position-encoding-2d', action='store_true', help='Use 2D rotary embedding.')
        parser.add_argument('--no-glu', action='store_true', help='Disable GLU.')


class GLM4VModel(ChatGLM4Model):

    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer=transformer, **kwargs)
        self.image_length = args.image_length
        self.add_mixin('eva', ImageMixin(args))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('GLM4V', 'GLM4V Configurations')
        group.add_argument('--image_length', type=int, default=256)
        group.add_argument('--eva_args', type=json.loads, default={})
        group.add_argument('--proj_hidden_size', type=int, default=None)
        return super().add_model_specific_args(parser)

    def forward(self, input_ids, **kwargs):
        if input_ids.shape[1] > 1:
            return super().forward(input_ids=input_ids, **kwargs)
        if 'vision_expert_mask' in kwargs:
            kwargs.pop('vision_expert_mask')
        if 'image_embed_mask' in kwargs:
            kwargs.pop('image_embed_mask')
        return super().forward(input_ids=input_ids, **kwargs)


class GLMModel(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer=transformer, **kwargs)
        self.add_mixin('block_position_embedding', BlockPositionEmbeddingMixin(args.max_sequence_length, args.hidden_size))

    @classmethod
    def add_model_specific_args(cls, parser):
        """Arguments for GLM"""
        group = parser.add_argument_group('GLM', 'GLM Configurations')
        group.add_argument('--block-lm', action='store_true', help='whether use the BlockLM pre-training')
        group.add_argument('--masked-lm', action='store_true', help='whether to use the mlm objective')
        group.add_argument('--bert-prob', type=float, default=0.5)
        group.add_argument('--gpt-infill-prob', type=float, default=0.5)
        group.add_argument('--gpt-min-ratio', type=float, default=0.5)
        group.add_argument('--gap-sentence-prob', type=float, default=0.0)
        group.add_argument('--gap-sentence-ratio', type=float, default=0.15)
        group.add_argument('--avg-block-length', type=int, default=3)
        group.add_argument('--short-seq-prob', type=float, default=0.0)
        group.add_argument('--single-span-prob', type=float, default=0.0)
        group.add_argument('--task-mask', action='store_true', help='Use different mask for generation and blank filling')
        group.add_argument('--no-shuffle-block', action='store_true', help='not shuffle the blocks when filling the blank')
        group.add_argument('--no-block-position', action='store_true', help='Use (rough) absolute positions instead of block positions')
        group.add_argument('--sentinel-token', action='store_true', help='Use sentinel (mask) tokens to replace 2d position encoding')
        group.add_argument('--block-mask-prob', type=float, default=0.0)
        group.add_argument('--context-mask-ratio', type=float, default=0.0)
        group.add_argument('--random-position', action='store_true', help='Use random start position to cover all the position embeddings')
        group.add_argument('--cloze-eval', action='store_true', help='Evaluation dataset with cloze task')
        group.add_argument('--old-checkpoint', action='store_true', help='Loading the checkpoint from old libraray')
        group.add_argument('--tokenizer-model-type', type=str, default=None, help="Model type to use for sentencepiece tokenization                            (one of ['bpe', 'char', 'unigram', 'word']) or                            bert vocab to use for BertWordPieceTokenizer (one of                            ['bert-large-uncased', 'bert-large-cased', etc.])")
        return parser


class GPT2Model(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        super(GPT2Model, self).__init__(args, transformer=transformer, activation_func=gelu, **kwargs)
        self.add_mixin('gpt2-final', GPT2FinalMixin(args.vocab_size, args.hidden_size))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('GPT2', 'GPT2 Configurations')
        return parser


class GPTNeoModel(BaseModel):

    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer, **kwargs)
        self.add_mixin('gpt-type', GPTNeoTypeMixin())
        self.add_mixin('gpt-attn', GPTNeoAttentionMixin(args.attention_types, args.window_size, args.max_sequence_length))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('GPTNeo', 'GPTNeo Configurations')
        group.add_argument('--attention-types', type=str)
        group.add_argument('--window-size', type=str)
        return parser


class LLaMAModel(BaseModel):

    def __init__(self, args, transformer=None, layernorm=RMSNorm, activation_func=nn.functional.silu, **kwargs):
        super().__init__(args, transformer=transformer, layernorm=layernorm, activation_func=activation_func, init_method_std=0.01, **kwargs)
        if 'inner_hidden_size' not in args:
            args.inner_hidden_size = None
        if not (hasattr(args, 'is_rotary_emb') and args.is_rotary_emb):
            del self.transformer.position_embeddings
            self.add_mixin('rotary', RotaryMixin(args.hidden_size, args.num_attention_heads))
        self.add_mixin('lm', LMMixin(args.vocab_size, args.hidden_size))
        if not (hasattr(args, 'is_gated_mlp') and args.is_gated_mlp):
            self.add_mixin('mlp', LLaMAMlpMixin(args.num_layers, args.hidden_size, args.inner_hidden_size))

    def position_embedding_forward(self, *args, **kwargs):
        return None

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('LLaMA', 'LLaMA Configurations')
        group.add_argument('--bos-token-id', type=int, default=0)
        group.add_argument('--eos-token-id', type=int, default=1)
        group.add_argument('--pad-token-id', type=int, default=-1)
        return parser


class MAEDecoder(BaseModel):

    def __init__(self, args, transformer=None, layernorm_epsilon=1e-06):
        super().__init__(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        self.add_mixin('mask_forward', MaskMixin(args))

    @classmethod
    def add_model_specific_args(cls, parser):
        return super().add_model_specific_args(parser)


class MAEEncoder(ViTModel):

    def __init__(self, args, transformer=None, layernorm_epsilon=1e-06):
        super().__init__(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        self.del_mixin('cls')
        self.del_mixin('pos_embedding')
        self.add_mixin('pos_embedding', PosMixin(args.hidden_size, self.old_property, self.property))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('MAE-enc', 'MAE encoder Configurations')
        return super().add_model_specific_args(parser)


class MAE(EncoderDecoderModel):

    def __init__(self, args, transformer=None, layernorm_epsilon=1e-06):
        encoder = MAEEncoder(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        dec_args = argparse.Namespace(**vars(args))
        override_attrs = ['num_layers', 'hidden_size', 'num_attention_heads', 'max_sequence_length', 'inner_hidden_size', 'hidden_size_per_attention_head']
        for name in override_attrs:
            dec_attr = getattr(dec_args, 'dec_' + name, None)
            if dec_attr is not None:
                setattr(dec_args, name, dec_attr)
        setattr(dec_args, 'enc_hidden_size', args.hidden_size)
        decoder = MAEDecoder(dec_args, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        super().__init__(args, encoder=encoder, decoder=decoder, tie_word_embeddings=False)

    def encode(self, input_ids, position_ids, attention_mask=None, **kw_args):
        return self.encoder(input_ids, position_ids, attention_mask, **kw_args)

    def decode(self, input_ids, position_ids, attention_mask, encoder_outputs, ids_restore, **kw_args):
        return self.decoder(input_ids, position_ids, attention_mask, encoder_outputs=encoder_outputs, ids_restore=ids_restore, **kw_args)

    def forward(self, input_ids, enc_position_ids, dec_position_ids, *, enc_attention_mask=None, dec_attention_mask=None, **kw_args):
        if enc_attention_mask is None:
            enc_attention_mask = torch.ones(1, 1, dtype=self.encoder.transformer.word_embeddings.weight.dtype, device=input_ids.device)
        encoder_outputs, *encoder_mems = self.encode(input_ids, enc_position_ids, enc_attention_mask, **kw_args)
        decoder_outputs, *decoder_mems = self.decode(input_ids, dec_position_ids, dec_attention_mask, encoder_outputs=encoder_outputs, ids_restore=encoder_mems[0]['ids_restore'], **kw_args)
        return encoder_outputs, decoder_outputs, encoder_mems, decoder_mems

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.encoder.property.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs


class MixtralModel(BaseModel):

    def __init__(self, args, transformer=None, layernorm=RMSNorm, activation_func=nn.functional.silu, **kwargs):
        super().__init__(args, transformer=transformer, layernorm=layernorm, activation_func=activation_func, init_method_std=0.01, **kwargs)
        del self.transformer.position_embeddings
        if 'inner_hidden_size' not in args:
            args.inner_hidden_size = None
        self.add_mixin('rotary', RotaryMixin(args.hidden_size, args.num_attention_heads))
        self.add_mixin('lm', LMMixin(args.vocab_size, args.hidden_size))
        self.add_mixin('mlp', MixtralMlpMixin(args.num_layers, args.hidden_size, args.num_experts, args.num_experts_per_tok))

    def position_embedding_forward(self, *args, **kwargs):
        return None

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('Mixtral-8x7b', 'Mixtral-8x7b Configurations')
        group.add_argument('--bos-token-id', type=int, default=1)
        group.add_argument('--eos-token-id', type=int, default=2)
        group.add_argument('--num-experts-per-tok', type=int, default=2)
        return parser


class T5LayerNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states
        elif self.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states
        return self.weight * hidden_states


def t5_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5Model(EncoderDecoderModel):

    def __init__(self, args, **kwargs):
        self.init_method_std = args.init_method_std
        super().__init__(args, tie_word_embeddings=True, **kwargs, use_bias=False, layernorm=T5LayerNorm, activation_func=torch.nn.functional.relu, init_method=self._init_weights)
        self.encoder.add_mixin('t5-attention', T5AttentionMixin(args.relative_attention_num_buckets, args.num_attention_heads))
        self.encoder.add_mixin('t5-position', T5PositionEmbeddingMixin())
        del self.encoder.transformer.position_embeddings
        num_attention_heads = args.dec_num_attention_heads if args.dec_num_attention_heads is not None else args.num_attention_heads
        self.decoder.add_mixin('t5-attention', T5AttentionMixin(args.relative_attention_num_buckets, num_attention_heads, is_decoder=True))
        self.decoder.add_mixin('t5-position', T5PositionEmbeddingMixin())
        self.decoder.add_mixin('t5-final', T5DecoderFinalMixin(args.vocab_size, args.hidden_size, tie_word_embeddings=not args.no_share_embeddings))
        del self.decoder.transformer.position_embeddings
        if args.gated_gelu_mlp:
            self.encoder.add_mixin('gated-mlp', T5GatedGeluMLPMixin(args.num_layers, args.hidden_size, init_method_std=self.init_method_std, inner_hidden_size=args.inner_hidden_size, bias=False))
            self.decoder.add_mixin('gated-mlp', T5GatedGeluMLPMixin(args.num_layers, args.hidden_size, init_method_std=self.init_method_std, inner_hidden_size=args.inner_hidden_size, bias=False))

    def _init_weights(self, weight, module, name):
        init_method_std = self.init_method_std
        if isinstance(module, MLP):
            if name == 'dense_h_to_4h':
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * module.hidden_size ** -0.5)
            elif name == 'dense_4h_to_h':
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * module.inner_hidden_size ** -0.5)
            else:
                raise NotImplementedError(name)
        elif isinstance(module, SelfAttention):
            if name == 'query_key_value':
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * module.hidden_size ** -0.5)
                torch.nn.init.normal_(weight[:module.inner_hidden_size], mean=0, std=init_method_std * (module.hidden_size * module.hidden_size_per_attention_head) ** -0.5)
            elif name == 'dense':
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * module.inner_hidden_size ** -0.5)
            else:
                raise NotImplementedError(name)
        elif isinstance(module, CrossAttention):
            if name == 'query':
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (module.hidden_size * module.hidden_size_per_attention_head) ** -0.5)
            elif name == 'key_value':
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * module.hidden_size ** -0.5)
            elif name == 'dense':
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * module.inner_hidden_size ** -0.5)
            else:
                raise NotImplementedError(name)
        else:
            raise NotImplementedError(module)

    @classmethod
    def add_model_specific_args(cls, parser):
        super().add_model_specific_args(parser)
        parser.add_argument('--relative-attention-num-buckets', type=int, default=None)
        parser.add_argument('--init-method-std', type=float, default=0.02)
        parser.add_argument('--gated-gelu-mlp', action='store_true')
        parser.add_argument('--no-share-embeddings', action='store_true')

    def encode(self, input_ids, attention_mask=None, **kw_args):
        return super().encode(input_ids, None, attention_mask, **kw_args)

    def decode(self, input_ids, attention_mask=None, encoder_outputs=None, cross_attention_mask=None, **kw_args):
        return super().decode(input_ids, None, attention_mask, encoder_outputs=encoder_outputs, cross_attention_mask=cross_attention_mask, **kw_args)

    def forward(self, enc_input_ids, dec_input_ids, *, enc_attention_mask=None, dec_attention_mask=None, cross_attention_mask=None, **kw_args):
        batch_size, seq_length = enc_input_ids.size()[:2]
        if enc_attention_mask is None:
            enc_attention_mask = torch.ones(1, 1, 1, seq_length, dtype=self.encoder.transformer.word_embeddings.weight.dtype, device=enc_input_ids.device)
        if cross_attention_mask is None:
            cross_attention_mask = enc_attention_mask
        encoder_outputs = self.encode(enc_input_ids, enc_attention_mask, **kw_args)
        decoder_outputs, *mems = self.decode(dec_input_ids, dec_attention_mask, encoder_outputs=encoder_outputs, cross_attention_mask=cross_attention_mask, **kw_args)
        return encoder_outputs, decoder_outputs, *mems


class YOLOS(ViTModel):

    def __init__(self, args, transformer=None, layernorm_epsilon=1e-06, **kwargs):
        super().__init__(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon, **kwargs)
        self.del_mixin('patch_embedding')
        self.add_mixin('patch_embedding', NewTokenMixin(args.vocab_size + args.num_det_tokens, args.in_channels, args.hidden_size, self.property))
        self.del_mixin('cls')
        self.add_mixin('det_head', DetHeadMixin(args))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('YOLOS', 'YOLOS Configurations')
        group.add_argument('--num-det-tokens', type=int)
        group.add_argument('--num-det-classes', type=int)
        return super().add_model_specific_args(parser)


class VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim, pt_seq_len, ft_seq_len=None, custom_freqs=None, freqs_for='lang', theta=10000, max_freq=10, num_freqs=1):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)
        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)
        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)
        self.register_buffer('freqs_cos', freqs.cos())
        self.register_buffer('freqs_sin', freqs.sin())
        None

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = t * self.freqs_cos + rotate_half(t) * self.freqs_sin
        return torch.cat((t_left, t, t_right), dim=-1)


def extract_weight_to_half(weight: 'torch.Tensor', scale_list: 'torch.Tensor', source_bit_width: 'int'):
    if source_bit_width == 8:
        func = kernels.int8WeightExtractionHalf
    elif source_bit_width == 4:
        func = kernels.int4WeightExtractionHalf
    else:
        assert False, 'Unsupported bit-width'
    with torch.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        out = torch.empty(n, m * (8 // source_bit_width), dtype=torch.half, device='cuda')
        stream = torch.cuda.current_stream()
        gridDim = n, 1, 1
        blockDim = min(round_up(m, 32), 1024), 1, 1
        func(gridDim, blockDim, 0, stream, [ctypes.c_void_p(weight.data_ptr()), ctypes.c_void_p(scale_list.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_int32(n), ctypes.c_int32(m)])
        return out


class W8A16Linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp: 'torch.Tensor', quant_w: 'torch.Tensor', scale_w: 'torch.Tensor', weight_bit_width):
        ctx.inp_shape = inp.size()
        ctx.weight_bit_width = weight_bit_width
        out_features = quant_w.size(0)
        inp = inp.contiguous().view(-1, inp.size(-1))
        weight = extract_weight_to_half(quant_w, scale_w, weight_bit_width)
        ctx.weight_shape = weight.size()
        output = inp.mm(weight.t())
        ctx.save_for_backward(inp, quant_w, scale_w)
        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))

    @staticmethod
    def backward(ctx, grad_output: 'torch.Tensor'):
        inp, quant_w, scale_w = ctx.saved_tensors
        weight = extract_weight_to_half(quant_w, scale_w, ctx.weight_bit_width)
        grad_output = grad_output.contiguous().view(-1, weight.size(0))
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(inp)
        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None, None


def compress_int4_weight(weight: 'torch.Tensor'):
    with torch.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        assert m % 2 == 0
        m = m // 2
        out = torch.empty(n, m, dtype=torch.int8, device='cuda')
        stream = torch.cuda.current_stream()
        gridDim = n, 1, 1
        blockDim = min(round_up(m, 32), 1024), 1, 1
        kernels.int4WeightCompression(gridDim, blockDim, 0, stream, [ctypes.c_void_p(weight.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_int32(n), ctypes.c_int32(m)])
        return out


class QuantizedLinear(Linear):

    def __init__(self, weight_bit_width: 'int', weight_tensor=None, bias_tensor=None, empty_init=False, *args, **kwargs):
        super(QuantizedLinear, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width
        shape = self.weight.shape
        del self.weight
        if weight_tensor is None or empty_init:
            self.weight = torch.empty(shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=kwargs['device'])
            self.weight_scale = torch.empty(shape[0], dtype=kwargs['dtype'], device=kwargs['device'])
        else:
            self.weight_scale = (weight_tensor.abs().max(dim=-1).values / (2 ** (weight_bit_width - 1) - 1)).half()
            self.weight = torch.round(weight_tensor / self.weight_scale[:, None])
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)
        self.weight = Parameter(self.weight, requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale, requires_grad=False)
        if bias_tensor is not None:
            self.bias = Parameter(bias_tensor, requires_grad=False)
        else:
            self.bias = None

    def forward(self, input):
        output = W8A16Linear.apply(input, self.weight, self.weight_scale, self.weight_bit_width)
        if self.bias is not None:
            output = output + self.bias
        return output


class QuantizedColumnParallelLinear(ColumnParallelLinear):

    def __init__(self, weight_bit_width: 'int', weight=None, *args, **kwargs):
        bias_val = kwargs.pop('bias_val')
        super(QuantizedColumnParallelLinear, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width
        shape = self.weight.shape
        del self.weight
        if weight is None:
            self.weight = torch.empty(shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=kwargs['device'])
            self.weight_scale = torch.empty(shape[0], dtype=kwargs['params_dtype'], device=kwargs['device'])
        else:
            self.weight_scale = (weight.abs().max(dim=-1).values / (2 ** (weight_bit_width - 1) - 1)).half()
            self.weight = torch.round(weight / self.weight_scale[:, None])
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)
        self.weight = Parameter(self.weight, requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale, requires_grad=False)
        if kwargs['bias']:
            self.bias = bias_val

    def forward(self, input_):
        input_parallel = copy_to_model_parallel_region(input_)
        output_parallel = W8A16Linear.apply(input_parallel, self.weight, self.weight_scale, self.weight_bit_width)
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class QuantizedRowParallelLinear(RowParallelLinear):

    def __init__(self, weight_bit_width: 'int', weight=None, *args, **kwargs):
        bias_val = kwargs.pop('bias_val')
        super(QuantizedRowParallelLinear, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width
        shape = self.weight.shape
        del self.weight
        if weight is None:
            self.weight = torch.empty(shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=kwargs['device'])
            self.weight_scale = torch.empty(shape[0], dtype=kwargs['params_dtype'], device=kwargs['device'])
        else:
            self.weight_scale = (weight.abs().max(dim=-1).values / (2 ** (weight_bit_width - 1) - 1)).half()
            self.weight = torch.round(weight / self.weight_scale[:, None])
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)
        self.weight = Parameter(self.weight, requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale, requires_grad=False)
        if kwargs['bias']:
            self.bias = bias_val

    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        output_parallel = W8A16Linear.apply(input_parallel, self.weight, self.weight_scale, self.weight_bit_width)
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output


class Upsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = 0, 1, 0, 1
            x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def nonlinearity(x):
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):

    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * int(c) ** -0.5
        w_ = torch.nn.functional.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Model(nn.Module):

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, use_timestep=True):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([torch.nn.Linear(self.ch, self.temb_ch), torch.nn.Linear(self.temb_ch, self.temb_ch)])
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t=None):
        if self.use_timestep:
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class ResBlock(nn.Module):

    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(in_channel, channel, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel, in_channel, 1))

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class Encoder(nn.Module):

    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride, embed_dim, n_embed, simple):
        super().__init__()
        if stride == 6:
            if simple:
                blocks = [nn.Conv2d(in_channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, 4, stride=2, padding=1)]
            else:
                blocks = [nn.Conv2d(in_channel, channel // 4, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel // 4, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1)]
        elif stride == 4:
            blocks = [nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, 3, padding=1)]
        elif stride == 2:
            blocks = [nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel // 2, channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel, embed_dim, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input).permute(0, 2, 3, 1)


class Decoder(nn.Module):

    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, simple):
        super().__init__()
        blocks = [nn.ConvTranspose2d(in_channel, channel, 4, stride=2, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4 and simple:
            blocks.extend([nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel, out_channel, 1)])
        elif stride == 4:
            blocks.extend([nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.ConvTranspose2d(channel, channel // 2, 1), nn.ReLU(inplace=True), nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1)])
        elif stride == 2:
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VUNet(nn.Module):

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, c_channels, resolution, z_channels, use_timestep=False, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([torch.nn.Linear(self.ch, self.temb_ch), torch.nn.Linear(self.temb_ch, self.temb_ch)])
        self.conv_in = torch.nn.Conv2d(c_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.z_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=1, stride=1, padding=0)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=2 * block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, z):
        if self.use_timestep:
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        z = self.z_in(z)
        h = torch.cat((h, z), dim=1)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class SimpleDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1), ResnetBlock(in_channels=in_channels, out_channels=2 * in_channels, temb_channels=0, dropout=0.0), ResnetBlock(in_channels=2 * in_channels, out_channels=4 * in_channels, temb_channels=0, dropout=0.0), ResnetBlock(in_channels=4 * in_channels, out_channels=2 * in_channels, temb_channels=0, dropout=0.0), nn.Conv2d(2 * in_channels, in_channels, 1), Upsample(in_channels, with_conv=True)])
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1, 2, 3]:
                x = layer(x, None)
            else:
                x = layer(x)
        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution, ch_mult=(2, 2), dropout=0.0):
        super().__init__()
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    """
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if not torch.jit.is_scripting():
        if type(logits) is not Tensor and has_torch_function((logits,)):
            return handle_torch_function(gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)
    if eps != 1e-10:
        warnings.warn('`eps` parameter is deprecated and has no effect.')
    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret, index
    else:
        ret = y_soft
        index = y_soft.max(dim, keepdim=True)[1]
        return ret, index


class Quantize(nn.Module):

    def __init__(self, dim, n_embed, decay=0.99, eps=1e-05):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        embed = torch.randn(dim, n_embed)
        torch.nn.init.xavier_uniform_(embed, gain=torch.nn.init.calculate_gain('tanh'))
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward_(self, input, continuous_relax=False, temperature=1.0, hard=False):
        flatten = input.reshape(-1, self.dim)
        dist = flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True)
        if not continuous_relax:
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)
        elif not hard:
            embed_soft, embed_ind = gumbel_softmax(-dist, tau=temperature, hard=False)
            embed_ind = embed_ind.view(*input.shape[:-1])
            embed_soft = embed_soft.view(*input.shape[:-1], self.n_embed)
            quantize = embed_soft @ self.embed.transpose(0, 1)
        else:
            embed_onehot, embed_ind = gumbel_softmax(-dist, tau=temperature, hard=True)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)
        if self.training and (continuous_relax and hard or not continuous_relax):
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        if not continuous_relax:
            diff = (quantize.detach() - input).pow(2).mean()
            quantize = input + (quantize - input).detach()
        else:
            qy = (-dist).softmax(-1)
            diff = torch.sum(qy * torch.log(qy * self.n_embed + 1e-20), dim=-1).mean()
            quantize = quantize
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class VQVAE(nn.Module):

    def __init__(self, in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=1024, stride=4, simple=True, decay=0.99, dif=False, ddconfig=None):
        super().__init__()
        if channel == 2048:
            n_res_block = 0
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride, embed_dim, n_embed, simple)
        self.quantize_t = Quantize(embed_dim, n_embed)
        if dif:
            self.dec = DifDecoder(**ddconfig)
        else:
            self.dec = Decoder(in_channel=embed_dim, out_channel=in_channel, channel=channel, n_res_block=n_res_block, n_res_channel=n_res_channel, stride=stride - 2, simple=simple)

    def forward(self, input, continuous_relax=False, temperature=1.0, hard=False, KL=False):
        quant_t, diff, _ = self.encode(input, continuous_relax, temperature, hard, KL)
        dec = self.dec(quant_t)
        return dec, diff

    def encode(self, input, continuous_relax=False, temperature=1.0, hard=False, KL=False):
        logits = self.enc_b(input)
        quant_t, diff_t, id_t = self.quantize_t.forward_(logits, continuous_relax, temperature, hard)
        quant_t = quant_t.permute(0, 3, 1, 2)
        if not continuous_relax or KL:
            diff_t = diff_t.unsqueeze(0)
        else:
            diff_t = torch.zeros_like(diff_t).unsqueeze(0)
        return quant_t, diff_t, id_t

    def decode(self, code):
        return self.dec(code)

    def decode_code(self, code_t):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        dec = self.dec(quant_t)
        return dec


class SatRotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half, learnable=False, device=torch.device('cpu')):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, device=device).float() / dim)
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        pass

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            cos_cached = cos_cached
            sin_cached = sin_cached
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Block,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Decoder,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'channel': 4, 'n_res_block': 4, 'n_res_channel': 4, 'stride': 1, 'simple': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Downsample,
     lambda: ([], {'in_channels': 4, 'with_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropBlock2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GLU,
     lambda: ([], {'args': SimpleNamespace(hidden_size=4, inner_hidden_size=4), 'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HackLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QuickGELUActivation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlock,
     lambda: ([], {'in_channel': 4, 'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SatRotaryEmbedding,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (T5LayerNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Upsample,
     lambda: ([], {'in_channels': 4, 'with_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VQVAE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

