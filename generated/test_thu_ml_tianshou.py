import sys
_module = sys.modules[__name__]
del sys
autogen_rst = _module
create_toc = _module
nbstripout = _module
examples = _module
atari = _module
atari_c51 = _module
atari_dqn = _module
atari_dqn_hl = _module
atari_fqf = _module
atari_iqn = _module
atari_iqn_hl = _module
atari_network = _module
atari_ppo = _module
atari_ppo_hl = _module
atari_qrdqn = _module
atari_rainbow = _module
atari_sac = _module
atari_sac_hl = _module
atari_wrapper = _module
acrobot_dualdqn = _module
bipedal_bdq = _module
bipedal_hardcore_sac = _module
lunarlander_dqn = _module
mcc_sac = _module
discrete_dqn = _module
discrete_dqn_hl = _module
irl_gail = _module
analysis = _module
fetch_her_ddpg = _module
gen_json = _module
mujoco_a2c = _module
mujoco_a2c_hl = _module
mujoco_ddpg = _module
mujoco_ddpg_hl = _module
mujoco_env = _module
mujoco_npg = _module
mujoco_npg_hl = _module
mujoco_ppo = _module
mujoco_ppo_hl = _module
mujoco_redq = _module
mujoco_redq_hl = _module
mujoco_reinforce = _module
mujoco_reinforce_hl = _module
mujoco_sac = _module
mujoco_sac_hl = _module
mujoco_td3 = _module
mujoco_td3_hl = _module
mujoco_trpo = _module
mujoco_trpo_hl = _module
plotter = _module
tools = _module
atari_bcq = _module
atari_cql = _module
atari_crr = _module
atari_il = _module
convert_rl_unplugged_atari = _module
d4rl_bcq = _module
d4rl_cql = _module
d4rl_il = _module
d4rl_td3_bc = _module
utils = _module
env = _module
spectator = _module
replay = _module
vizdoom_c51 = _module
vizdoom_ppo = _module
test = _module
base = _module
test_action_space_sampling = _module
test_batch = _module
test_buffer = _module
test_collector = _module
test_env = _module
test_env_finite = _module
test_logger = _module
test_policy = _module
test_returns = _module
test_stats = _module
test_utils = _module
continuous = _module
test_ddpg = _module
test_npg = _module
test_ppo = _module
test_redq = _module
test_sac_with_il = _module
test_td3 = _module
test_trpo = _module
discrete = _module
test_a2c_with_il = _module
test_bdq = _module
test_c51 = _module
test_dqn = _module
test_drqn = _module
test_fqf = _module
test_iqn = _module
test_pg = _module
test_ppo = _module
test_qrdqn = _module
test_rainbow = _module
test_sac = _module
highlevel = _module
env_factory = _module
test_experiment_builder = _module
modelbased = _module
test_dqn_icm = _module
test_ppo_icm = _module
test_psrl = _module
offline = _module
gather_cartpole_data = _module
gather_pendulum_data = _module
test_bcq = _module
test_cql = _module
test_discrete_bcq = _module
test_discrete_cql = _module
test_discrete_crr = _module
test_gail = _module
test_td3_bc = _module
pistonball = _module
pistonball_continuous = _module
test_pistonball = _module
test_pistonball_continuous = _module
test_tic_tac_toe = _module
tic_tac_toe = _module
tianshou = _module
data = _module
batch = _module
buffer = _module
cached = _module
her = _module
manager = _module
prio = _module
vecbuf = _module
collector = _module
stats = _module
types = _module
converter = _module
segtree = _module
gym_wrappers = _module
pettingzoo_env = _module
venv_wrappers = _module
venvs = _module
worker = _module
dummy = _module
ray = _module
subproc = _module
evaluation = _module
rliable_evaluation_hl = _module
exploration = _module
random = _module
agent = _module
config = _module
experiment = _module
module = _module
core = _module
intermediate = _module
module_opt = _module
special = _module
optim = _module
params = _module
alpha = _module
dist_fn = _module
env_param = _module
lr_scheduler = _module
noise = _module
policy_wrapper = _module
trainer = _module
world = _module
policy = _module
base = _module
imitation = _module
base = _module
bcq = _module
cql = _module
discrete_bcq = _module
discrete_cql = _module
discrete_crr = _module
gail = _module
td3_bc = _module
icm = _module
psrl = _module
modelfree = _module
a2c = _module
bdq = _module
c51 = _module
ddpg = _module
discrete_sac = _module
dqn = _module
fqf = _module
iqn = _module
npg = _module
pg = _module
ppo = _module
qrdqn = _module
rainbow = _module
redq = _module
sac = _module
td3 = _module
trpo = _module
multiagent = _module
conversion = _module
logger = _module
tensorboard = _module
wandb = _module
logging = _module
lr_scheduler = _module
net = _module
common = _module
continuous = _module
discrete = _module
optim = _module
print = _module
progress_bar = _module
space_info = _module
statistics = _module
torch_utils = _module
warning = _module

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


import numpy as np


import torch


from collections.abc import Callable


from collections.abc import Sequence


from typing import Any


from torch import nn


from torch.distributions import Categorical


from torch.optim.lr_scheduler import LambdaLR


from torch.utils.tensorboard import SummaryWriter


from typing import SupportsFloat


from typing import cast


from torch.distributions import Distribution


from torch.distributions import Independent


from torch.distributions import Normal


from typing import Literal


import copy


from itertools import starmap


from torch.distributions.categorical import Categorical


import numpy.typing as npt


from collections import Counter


from collections.abc import Iterator


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import DistributedSampler


import torch.distributions as dist


import torch.nn as nn


import warnings


from copy import deepcopy


from functools import partial


from collections.abc import Collection


from collections.abc import Iterable


from collections.abc import KeysView


from numbers import Number


from typing import Protocol


from typing import TypeVar


from typing import Union


from typing import overload


from typing import runtime_checkable


import pandas as pd


import logging


import time


from abc import ABC


from abc import abstractmethod


from copy import copy


from typing import Generic


from typing import Optional


from typing import TypedDict


from typing import no_type_check


from typing import TYPE_CHECKING


from torch.optim import Adam


from torch.optim import RMSprop


from torch.optim.lr_scheduler import LRScheduler


from numpy.typing import ArrayLike


import torch.nn.functional as F


from torch.nn.utils import clip_grad_norm_


import math


from torch.distributions import kl_divergence


from matplotlib.figure import Figure


class ScaledObsInputModule(torch.nn.Module):

    def __init__(self, module: 'NetBase', denom: 'float'=255.0) ->None:
        super().__init__()
        self.module = module
        self.denom = denom
        self.output_dim = module.output_dim

    def forward(self, obs: 'np.ndarray | torch.Tensor', state: 'Any | None'=None, info: 'dict[str, Any] | None'=None) ->tuple[torch.Tensor, Any]:
        if info is None:
            info = {}
        return self.module.forward(obs / self.denom, state, info)


class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, c: 'int', h: 'int', w: 'int', device: 'str | int | torch.device'='cpu') ->None:
        super().__init__()
        self.device = device
        self.c = c
        self.h = h
        self.w = w
        self.net = nn.Sequential(nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True), nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True), nn.Flatten())
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])

    def forward(self, x: 'np.ndarray | torch.Tensor', state: 'Any | None'=None, info: 'dict[str, Any] | None'=None) ->tuple[torch.Tensor, Any]:
        """Mapping: x -> Q(x, \\*)."""
        if info is None:
            info = {}
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return self.net(x.reshape(-1, self.c, self.w, self.h)), state


class ProtocolCalledException(Exception):
    """The methods of a Protocol should never be called.

    Currently, no static type checker actually verifies that a class that inherits
    from a Protocol does in fact provide the correct interface. Thus, it may happen
    that a method of the protocol is called accidentally (this is an
    implementation error). The normal error for that is a somewhat cryptic
    AttributeError, wherefore we instead raise this custom exception in the
    BatchProtocol.

    Finally and importantly: using this in BatchProtocol makes mypy verify the fields
    in the various sub-protocols and thus renders is MUCH more useful!
    """


TBatch = TypeVar('TBatch', bound='BatchProtocol')


def _assert_type_keys(keys: 'Iterable[str]') ->None:
    assert all(isinstance(key, str) for key in keys), f'keys should all be string, but got {keys}'

