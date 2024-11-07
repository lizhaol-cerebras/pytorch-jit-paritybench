import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
stable_baselines3 = _module
a2c = _module
a2c = _module
policies = _module
common = _module
atari_wrappers = _module
base_class = _module
buffers = _module
callbacks = _module
distributions = _module
env_checker = _module
env_util = _module
envs = _module
bit_flipping_env = _module
identity_env = _module
multi_input_envs = _module
evaluation = _module
logger = _module
monitor = _module
noise = _module
off_policy_algorithm = _module
on_policy_algorithm = _module
policies = _module
preprocessing = _module
results_plotter = _module
running_mean_std = _module
save_util = _module
sb2_compat = _module
rmsprop_tf_like = _module
torch_layers = _module
type_aliases = _module
utils = _module
vec_env = _module
base_vec_env = _module
dummy_vec_env = _module
patch_gym = _module
stacked_observations = _module
subproc_vec_env = _module
util = _module
vec_check_nan = _module
vec_extract_dict_obs = _module
vec_frame_stack = _module
vec_monitor = _module
vec_normalize = _module
vec_transpose = _module
vec_video_recorder = _module
ddpg = _module
ddpg = _module
dqn = _module
dqn = _module
policies = _module
her = _module
goal_selection_strategy = _module
her_replay_buffer = _module
ppo = _module
ppo = _module
sac = _module
policies = _module
sac = _module
td3 = _module
policies = _module
td3 = _module
tests = _module
test_buffers = _module
test_callbacks = _module
test_cnn = _module
test_custom_policy = _module
test_deterministic = _module
test_dict_env = _module
test_distributions = _module
test_env_checker = _module
test_envs = _module
test_gae = _module
test_her = _module
test_identity = _module
test_logger = _module
test_monitor = _module
test_predict = _module
test_preprocessing = _module
test_run = _module
test_save_load = _module
test_sde = _module
test_spaces = _module
test_tensorboard = _module
test_train_eval_mode = _module
test_utils = _module
test_vec_check_nan = _module
test_vec_envs = _module
test_vec_extract_dict_obs = _module
test_vec_monitor = _module
test_vec_normalize = _module
test_vec_stacked_obs = _module

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


from typing import Dict


from typing import Any


from typing import ClassVar


from typing import Optional


from typing import Type


from typing import TypeVar


from typing import Union


import torch as th


from torch.nn import functional as F


import time


import warnings


from abc import ABC


from abc import abstractmethod


from collections import deque


from typing import Iterable


from typing import List


from typing import Tuple


import numpy as np


from typing import Generator


from torch import nn


from torch.distributions import Bernoulli


from torch.distributions import Categorical


from torch.distributions import Normal


from collections import defaultdict


from typing import Mapping


from typing import Sequence


from typing import TextIO


import matplotlib.figure


import pandas


from copy import deepcopy


import collections


import copy


from functools import partial


import functools


from typing import Callable


import torch


from torch.optim import Optimizer


from enum import Enum


from typing import TYPE_CHECKING


from typing import NamedTuple


from typing import Protocol


from typing import SupportsFloat


import random


import re


from itertools import zip_longest


import torch.nn as nn


from matplotlib import pyplot as plt


from pandas.errors import EmptyDataError


from collections import OrderedDict


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: 'gym.Space', features_dim: 'int'=0) ->None:
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) ->int:
        """The number of features that the extractor outputs."""
        return self._features_dim


def is_image_space_channels_first(observation_space: 'spaces.Box') ->bool:
    """
    Check if an image observation space (see ``is_image_space``)
    is channels-first (CxHxW, True) or channels-last (HxWxC, False).

    Use a heuristic that channel dimension is the smallest of the three.
    If second dimension is smallest, raise an exception (no support).

    :param observation_space:
    :return: True if observation space is channels-first image, False if channels-last.
    """
    smallest_dimension = np.argmin(observation_space.shape).item()
    if smallest_dimension == 1:
        warnings.warn('Treating image space as channels-last, while second dimension was smallest of the three.')
    return smallest_dimension == 0


def is_image_space(observation_space: 'spaces.Space', check_channels: 'bool'=False, normalized_image: 'bool'=False) ->bool:
    """
    Check if a observation space has the shape, limits and dtype
    of a valid image.
    The check is conservative, so that it returns False if there is a doubt.

    Valid images: RGB, RGBD, GrayScale with values in [0, 255]

    :param observation_space:
    :param check_channels: Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    :return:
    """
    check_dtype = check_bounds = not normalized_image
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        if check_dtype and observation_space.dtype != np.uint8:
            return False
        incorrect_bounds = np.any(observation_space.low != 0) or np.any(observation_space.high != 255)
        if check_bounds and incorrect_bounds:
            return False
        if not check_channels:
            return True
        if is_image_space_channels_first(observation_space):
            n_channels = observation_space.shape[0]
        else:
            n_channels = observation_space.shape[-1]
        return n_channels in [1, 3, 4]
    return False


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(self, observation_space: 'gym.Space', features_dim: 'int'=512, normalized_image: 'bool'=False) ->None:
        assert isinstance(observation_space, spaces.Box), ('NatureCNN must be used with a gym.spaces.Box ', f'observation space, not {observation_space}')
        super().__init__(observation_space, features_dim)
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), f'You should use NatureCNN only with images not with {observation_space}\n(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\nIf you are using a custom environment,\nplease check it using our env checker:\nhttps://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\nIf you are using `VecNormalize` or already normalized channel-first images you should pass `normalize_images=False`: \nhttps://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html'
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(), nn.Flatten())
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: 'th.Tensor') ->th.Tensor:
        return self.linear(self.cnn(observations))


def get_flattened_obs_dim(observation_space: 'spaces.Space') ->int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    Used by the ``FlattenExtractor`` to compute the input shape.

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        return spaces.utils.flatdim(observation_space)


class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(self, observation_space: 'spaces.Dict', cnn_output_dim: 'int'=256, normalized_image: 'bool'=False) ->None:
        super().__init__(observation_space, features_dim=1)
        extractors: 'Dict[str, nn.Module]' = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: 'TensorDict') ->th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space: The observation space of the environment
    """

    def __init__(self, observation_space: 'gym.Space') ->None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: 'th.Tensor') ->th.Tensor:
        return self.flatten(observations)


TensorDict = Dict[str, th.Tensor]


PyTorchObs = Union[th.Tensor, TensorDict]


SelfBaseModel = TypeVar('SelfBaseModel', bound='BaseModel')


def get_device(device: 'Union[th.device, str]'='auto') ->th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    if device == 'auto':
        device = 'cuda'
    device = th.device(device)
    if device.type == th.device('cuda').type and not th.cuda.is_available():
        return th.device('cpu')
    return device


VecEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]


VecEnvStepReturn = Tuple[VecEnvObs, np.ndarray, np.ndarray, List[Dict]]


VecEnvIndices = Union[None, int, Iterable[int]]

