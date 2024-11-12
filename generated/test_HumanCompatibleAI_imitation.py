
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


from typing import TYPE_CHECKING


from typing import Optional


import torch as th


import abc


import logging


from typing import Callable


from typing import Iterable


from typing import Iterator


from typing import Mapping


from typing import Type


from typing import overload


import numpy as np


import torch.utils.tensorboard as thboard


from torch.nn import functional as F


from typing import Any


from typing import Generic


from typing import TypeVar


from typing import Union


from typing import cast


import torch.utils.data as th_data


import itertools


from typing import Dict


from typing import Tuple


import uuid


from typing import List


from typing import Sequence


from torch.utils import data as th_data


import collections


from typing import NoReturn


import scipy.special


import math


import re


from collections import defaultdict


from typing import NamedTuple


from scipy import special


from torch import nn


from torch.utils import data as data_th


import numbers


from typing import TypedDict


from typing import Protocol


from torch import optim


import functools


import torch.random


import torch


from collections import Counter


from typing import Generator


import pandas as pd


def dataclass_quick_asdict(obj) ->Dict[str, Any]:
    """Extract dataclass to items using `dataclasses.fields` + dict comprehension.

    This is a quick alternative to `dataclasses.asdict`, which expensively and
    undocumentedly deep-copies every numpy array value.
    See https://stackoverflow.com/a/52229565/1091722.

    This is also used to preserve DictObj objects, as `dataclasses.asdict`
    unwraps them recursively.

    Args:
        obj: A dataclass instance.

    Returns:
        A dictionary mapping from `obj` field names to values.
    """
    d = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return d


def _rews_validation(rews: 'np.ndarray', acts: 'np.ndarray'):
    if rews.shape != (len(acts),):
        raise ValueError(f'rewards must be 1D array, one entry for each action: {rews.shape} != ({len(acts)},)')
    if not np.issubdtype(rews.dtype, np.floating):
        raise ValueError(f'rewards dtype {rews.dtype} not a float')


def _trajectory_pair_includes_reward(fragment_pair: 'TrajectoryPair') ->bool:
    """Return true if and only if both fragments in the pair include rewards."""
    frag1, frag2 = fragment_pair
    return isinstance(frag1, TrajectoryWithRew) and isinstance(frag2, TrajectoryWithRew)


PolicyCallable = Callable[[Union[np.ndarray, Dict[str, np.ndarray]], Optional[Tuple[np.ndarray, ...]], Optional[np.ndarray]], Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]]


def policy_to_callable(policy: 'AnyPolicy', venv: 'VecEnv', deterministic_policy: 'bool'=False) ->PolicyCallable:
    """Converts any policy-like object into a function from observations to actions."""
    get_actions: 'PolicyCallable'
    if policy is None:

        def get_actions(observations: 'Union[np.ndarray, Dict[str, np.ndarray]]', states: 'Optional[Tuple[np.ndarray, ...]]', episode_starts: 'Optional[np.ndarray]') ->Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
            acts = [venv.action_space.sample() for _ in range(len(observations))]
            return np.stack(acts, axis=0), None
    elif isinstance(policy, (BaseAlgorithm, BasePolicy)):

        def get_actions(observations: 'Union[np.ndarray, Dict[str, np.ndarray]]', states: 'Optional[Tuple[np.ndarray, ...]]', episode_starts: 'Optional[np.ndarray]') ->Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
            assert isinstance(policy, (BaseAlgorithm, BasePolicy))
            acts, states = policy.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic_policy)
            return acts, states
    elif callable(policy):
        if deterministic_policy:
            raise ValueError('Cannot set deterministic_policy=True when policy is a callable, since deterministic_policy argument is ignored.')
        get_actions = policy
    else:
        raise TypeError(f'Policy must be None, a stable-baselines policy or algorithm, or a Callable, got {type(policy)} instead')
    if isinstance(policy, BaseAlgorithm):
        try:
            check_for_correct_spaces(venv, policy.observation_space, policy.action_space)
        except ValueError as e:
            venv_obs_shape = venv.observation_space.shape
            assert policy.observation_space is not None
            policy_obs_shape = policy.observation_space.shape
            assert venv_obs_shape is not None
            assert policy_obs_shape is not None
            if len(venv_obs_shape) != 3 or len(policy_obs_shape) != 3:
                raise e
            venv_obs_rearranged = venv_obs_shape[2], venv_obs_shape[0], venv_obs_shape[1]
            if venv_obs_rearranged != policy_obs_shape:
                raise e
            raise ValueError(f'Policy and environment observation shape mismatch. This is likely caused by https://github.com/HumanCompatibleAI/imitation/issues/599. If encountering this from rollout.rollout, try calling:\nrollout.rollout(expert, expert.get_env(), ...) instead of\nrollout.rollout(expert, env, ...)\n\nPolicy observation shape: {policy_obs_shape} \nEnvironment observation shape: {venv_obs_shape}')
    return get_actions


def rollout_stats(trajectories: 'Sequence[types.TrajectoryWithRew]') ->Mapping[str, float]:
    """Calculates various stats for a sequence of trajectories.

    Args:
        trajectories: Sequence of trajectories.

    Returns:
        Dictionary containing `n_traj` collected (int), along with episode return
        statistics (keys: `{monitor_,}return_{min,mean,std,max}`, float values)
        and trajectory length statistics (keys: `len_{min,mean,std,max}`, float
        values).

        `return_*` values are calculated from environment rewards.
        `monitor_*` values are calculated from Monitor-captured rewards, and
        are only included if the `trajectories` contain Monitor infos.
    """
    assert len(trajectories) > 0
    out_stats: 'Dict[str, float]' = {'n_traj': len(trajectories)}
    traj_descriptors = {'return': np.asarray([sum(t.rews) for t in trajectories]), 'len': np.asarray([len(t.rews) for t in trajectories])}
    monitor_ep_returns = []
    for t in trajectories:
        if t.infos is not None:
            ep_return = t.infos[-1].get('episode', {}).get('r')
            if ep_return is not None:
                monitor_ep_returns.append(ep_return)
    if monitor_ep_returns:
        traj_descriptors['monitor_return'] = np.asarray(monitor_ep_returns)
        out_stats['monitor_return_len'] = len(traj_descriptors['monitor_return'])
    stat_names = ['min', 'mean', 'std', 'max']
    for desc_name, desc_vals in traj_descriptors.items():
        for stat_name in stat_names:
            stat_value: 'np.generic' = getattr(np, stat_name)(desc_vals)
            out_stats[f'{desc_name}_{stat_name}'] = stat_value.item()
    for v in out_stats.values():
        assert isinstance(v, (int, float))
    return out_stats


class PreferenceModel(nn.Module):
    """Class to convert two fragments' rewards into preference probability."""

    def __init__(self, model: 'reward_nets.RewardNet', noise_prob: 'float'=0.0, discount_factor: 'float'=1.0, threshold: 'float'=50) ->None:
        """Create Preference Prediction Model.

        Args:
            model: base model to compute reward.
            noise_prob: assumed probability with which the preference
                is uniformly random (used for the model of preference generation
                that is used for the loss).
            discount_factor: the model of preference generation uses a softmax
                of returns as the probability that a fragment is preferred.
                This is the discount factor used to calculate those returns.
                Default is 1, i.e. undiscounted sums of rewards (which is what
                the DRLHP paper uses).
            threshold: the preference model used to compute the loss contains
                a softmax of returns. To avoid overflows, we clip differences
                in returns that are above this threshold. This threshold
                is therefore in logspace. The default value of 50 means
                that probabilities below 2e-22 are rounded up to 2e-22.

        Raises:
            ValueError: if `RewardEnsemble` is wrapped around a class
                other than `AddSTDRewardWrapper`.
        """
        super().__init__()
        self.model = model
        self.noise_prob = noise_prob
        self.discount_factor = discount_factor
        self.threshold = threshold
        base_model = get_base_model(model)
        self.ensemble_model = None
        if isinstance(base_model, reward_nets.RewardEnsemble):
            is_base = model is base_model
            is_std_wrapper = isinstance(model, reward_nets.AddSTDRewardWrapper) and model.base is base_model
            if not (is_base or is_std_wrapper):
                raise ValueError(f'RewardEnsemble can only be wrapped by AddSTDRewardWrapper but found {type(model).__name__}.')
            self.ensemble_model = base_model
            self.member_pref_models = []
            for member in self.ensemble_model.members:
                member_pref_model = PreferenceModel(cast(reward_nets.RewardNet, member), self.noise_prob, self.discount_factor, self.threshold)
                self.member_pref_models.append(member_pref_model)

    def forward(self, fragment_pairs: 'Sequence[TrajectoryPair]') ->Tuple[th.Tensor, Optional[th.Tensor]]:
        """Computes the preference probability of the first fragment for all pairs.

        Note: This function passes the gradient through for non-ensemble models.
              For an ensemble model, this function should not be used for loss
              calculation. It can be used in case where passing the gradient is not
              required such as during active selection or inference time.
              Therefore, the EnsembleTrainer passes each member network through this
              function instead of passing the EnsembleNetwork object with the use of
              `ensemble_member_index`.

        Args:
            fragment_pairs: batch of pair of fragments.

        Returns:
            A tuple with the first element as the preference probabilities for the
            first fragment for all fragment pairs given by the network(s).
            If the ground truth rewards are available, it also returns gt preference
            probabilities in the second element of the tuple (else None).
            Reward probability shape - (num_fragment_pairs, ) for non-ensemble reward
            network and (num_fragment_pairs, num_networks) for an ensemble of networks.

        """
        probs = th.empty(len(fragment_pairs), dtype=th.float32)
        gt_reward_available = _trajectory_pair_includes_reward(fragment_pairs[0])
        if gt_reward_available:
            gt_probs = th.empty(len(fragment_pairs), dtype=th.float32)
        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])
            rews1 = self.rewards(trans1)
            rews2 = self.rewards(trans2)
            probs[i] = self.probability(rews1, rews2)
            if gt_reward_available:
                frag1 = cast(TrajectoryWithRew, frag1)
                frag2 = cast(TrajectoryWithRew, frag2)
                gt_rews_1 = th.from_numpy(frag1.rews)
                gt_rews_2 = th.from_numpy(frag2.rews)
                gt_probs[i] = self.probability(gt_rews_1, gt_rews_2)
        return probs, gt_probs if gt_reward_available else None

    def rewards(self, transitions: 'Transitions') ->th.Tensor:
        """Computes the reward for all transitions.

        Args:
            transitions: batch of obs-act-obs-done for a fragment of a trajectory.

        Returns:
            The reward given by the network(s) for all the transitions.
            Shape - (num_transitions, ) for Single reward network and
            (num_transitions, num_networks) for ensemble of networks.
        """
        state = types.assert_not_dictobs(transitions.obs)
        action = transitions.acts
        next_state = types.assert_not_dictobs(transitions.next_obs)
        done = transitions.dones
        if self.ensemble_model is not None:
            rews_np = self.ensemble_model.predict_processed_all(state, action, next_state, done)
            assert rews_np.shape == (len(state), self.ensemble_model.num_members)
            rews = util.safe_to_tensor(rews_np)
        else:
            preprocessed = self.model.preprocess(state, action, next_state, done)
            rews = self.model(*preprocessed)
            assert rews.shape == (len(state),)
        return rews

    def probability(self, rews1: 'th.Tensor', rews2: 'th.Tensor') ->th.Tensor:
        """Computes the Boltzmann rational probability the first trajectory is best.

        Args:
            rews1: array/matrix of rewards for the first trajectory fragment.
                matrix for ensemble models and array for non-ensemble models.
            rews2: array/matrix of rewards for the second trajectory fragment.
                matrix for ensemble models and array for non-ensemble models.

        Returns:
            The softmax of the difference between the (discounted) return of the
            first and second trajectory.
            Shape - (num_ensemble_members, ) for ensemble model and
            () for non-ensemble model which is a torch scalar.
        """
        expected_dims = 2 if self.ensemble_model is not None else 1
        assert rews1.ndim == rews2.ndim == expected_dims
        if self.discount_factor == 1:
            returns_diff = (rews2 - rews1).sum(axis=0)
        else:
            device = rews1.device
            assert device == rews2.device
            discounts = self.discount_factor ** th.arange(len(rews1), device=device)
            if self.ensemble_model is not None:
                discounts = discounts.reshape(-1, 1)
            returns_diff = (discounts * (rews2 - rews1)).sum(axis=0)
        returns_diff = th.clip(returns_diff, -self.threshold, self.threshold)
        model_probability = 1 / (1 + returns_diff.exp())
        probability = self.noise_prob * 0.5 + (1 - self.noise_prob) * model_probability
        if self.ensemble_model is not None:
            assert probability.shape == (self.model.num_members,)
        else:
            assert probability.shape == ()
        return probability


class LossAndMetrics(NamedTuple):
    """Loss and auxiliary metrics for reward network training."""
    loss: 'th.Tensor'
    metrics: 'Mapping[str, th.Tensor]'


class RewardLoss(nn.Module, abc.ABC):
    """A loss function over preferences."""

    @abc.abstractmethod
    def forward(self, fragment_pairs: 'Sequence[TrajectoryPair]', preferences: 'np.ndarray', preference_model: 'PreferenceModel') ->LossAndMetrics:
        """Computes the loss.

        Args:
            fragment_pairs: Batch consisting of pairs of trajectory fragments.
            preferences: The probability that the first fragment is preferred
                over the second. Typically 0, 1 or 0.5 (tie).
            preference_model: model to predict the preferred fragment from a pair.

        Returns: # noqa: DAR202
            loss: the loss
            metrics: a dictionary of metrics that can be logged
        """


class CrossEntropyRewardLoss(RewardLoss):
    """Compute the cross entropy reward loss."""

    def __init__(self) ->None:
        """Create cross entropy reward loss."""
        super().__init__()

    def forward(self, fragment_pairs: 'Sequence[TrajectoryPair]', preferences: 'np.ndarray', preference_model: 'PreferenceModel') ->LossAndMetrics:
        """Computes the loss.

        Args:
            fragment_pairs: Batch consisting of pairs of trajectory fragments.
            preferences: The probability that the first fragment is preferred
                over the second. Typically 0, 1 or 0.5 (tie).
            preference_model: model to predict the preferred fragment from a pair.

        Returns:
            The cross-entropy loss between the probability predicted by the
                reward model and the target probabilities in `preferences`. Metrics
                are accuracy, and gt_reward_loss, if the ground truth reward is
                available.
        """
        probs, gt_probs = preference_model(fragment_pairs)
        predictions = probs > 0.5
        preferences_th = th.as_tensor(preferences, dtype=th.float32)
        ground_truth = preferences_th > 0.5
        metrics = {}
        metrics['accuracy'] = (predictions == ground_truth).float().mean()
        if gt_probs is not None:
            metrics['gt_reward_loss'] = th.nn.functional.binary_cross_entropy(gt_probs, preferences_th)
        metrics = {key: value.detach().cpu() for key, value in metrics.items()}
        return LossAndMetrics(loss=th.nn.functional.binary_cross_entropy(probs, preferences_th), metrics=metrics)


class RewardNet(nn.Module, abc.ABC):
    """Minimal abstract reward network.

    Only requires the implementation of a forward pass (calculating rewards given
    a batch of states, actions, next states and dones).
    """

    def __init__(self, observation_space: 'gym.Space', action_space: 'gym.Space', normalize_images: 'bool'=True):
        """Initialize the RewardNet.

        Args:
            observation_space: the observation space of the environment
            action_space: the action space of the environment
            normalize_images: whether to automatically normalize
                image observations to [0, 1] (from 0 to 255). Defaults to True.
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.normalize_images = normalize_images

    @abc.abstractmethod
    def forward(self, state: 'th.Tensor', action: 'th.Tensor', next_state: 'th.Tensor', done: 'th.Tensor') ->th.Tensor:
        """Compute rewards for a batch of transitions and keep gradients."""

    def preprocess(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray') ->Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Preprocess a batch of input transitions and convert it to PyTorch tensors.

        The output of this function is suitable for its forward pass,
        so a typical usage would be ``model(*model.preprocess(transitions))``.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            action: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.

        Returns:
            Preprocessed transitions: a Tuple of tensors containing
            observations, actions, next observations and dones.
        """
        state_th = util.safe_to_tensor(state)
        action_th = util.safe_to_tensor(action)
        next_state_th = util.safe_to_tensor(next_state)
        done_th = util.safe_to_tensor(done)
        del state, action, next_state, done
        state_th = cast(th.Tensor, preprocessing.preprocess_obs(state_th, self.observation_space, self.normalize_images))
        action_th = cast(th.Tensor, preprocessing.preprocess_obs(action_th, self.action_space, self.normalize_images))
        next_state_th = cast(th.Tensor, preprocessing.preprocess_obs(next_state_th, self.observation_space, self.normalize_images))
        done_th = done_th
        n_gen = len(state_th)
        assert state_th.shape == next_state_th.shape
        assert len(action_th) == n_gen
        return state_th, action_th, next_state_th, done_th

    def predict_th(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray') ->th.Tensor:
        """Compute th.Tensor rewards for a batch of transitions without gradients.

        Preprocesses the inputs, output th.Tensor reward arrays.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.

        Returns:
            Computed th.Tensor rewards of shape `(batch_size,`).
        """
        with networks.evaluating(self):
            state_th, action_th, next_state_th, done_th = self.preprocess(state, action, next_state, done)
            with th.no_grad():
                rew_th = self(state_th, action_th, next_state_th, done_th)
            assert rew_th.shape == state.shape[:1]
            return rew_th

    def predict(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray') ->np.ndarray:
        """Compute rewards for a batch of transitions without gradients.

        Converting th.Tensor rewards from `predict_th` to NumPy arrays.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.

        Returns:
            Computed rewards of shape `(batch_size,)`.
        """
        rew_th = self.predict_th(state, action, next_state, done)
        return rew_th.detach().cpu().numpy().flatten()

    def predict_processed(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray', **kwargs) ->np.ndarray:
        """Compute the processed rewards for a batch of transitions without gradients.

        Defaults to calling `predict`. Subclasses can override this to normalize or
        otherwise modify the rewards in ways that may help RL training or other
        applications of the reward function.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            kwargs: additional kwargs may be passed to change the functionality of
                subclasses.

        Returns:
            Computed processed rewards of shape `(batch_size,`).
        """
        del kwargs
        return self.predict(state, action, next_state, done)

    @property
    def device(self) ->th.device:
        """Heuristic to determine which device this module is on."""
        try:
            first_param = next(self.parameters())
            return first_param.device
        except StopIteration:
            return th.device('cpu')

    @property
    def dtype(self) ->th.dtype:
        """Heuristic to determine dtype of module."""
        try:
            first_param = next(self.parameters())
            return first_param.dtype
        except StopIteration:
            return th.get_default_dtype()


class RewardNetWrapper(RewardNet):
    """Abstract class representing a wrapper modifying a ``RewardNet``'s functionality.

    In general ``RewardNetWrapper``s should either subclass ``ForwardWrapper``
    or ``PredictProcessedWrapper``.
    """

    def __init__(self, base: 'RewardNet'):
        """Initialize a RewardNet wrapper.

        Args:
            base: the base RewardNet to wrap.
        """
        super().__init__(base.observation_space, base.action_space, base.normalize_images)
        self._base = base

    @property
    def base(self) ->RewardNet:
        return self._base

    @property
    def device(self) ->th.device:
        __doc__ = super().device.__doc__
        return self.base.device

    @property
    def dtype(self) ->th.dtype:
        __doc__ = super().dtype.__doc__
        return self.base.dtype

    def preprocess(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray') ->Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        __doc__ = super().preprocess.__doc__
        return self.base.preprocess(state, action, next_state, done)


class PredictProcessedWrapper(RewardNetWrapper):
    """An abstract RewardNetWrapper that changes the behavior of predict_processed.

    Subclasses should override `predict_processed`. Implementations
    should pass along `kwargs` to the `base` reward net's `predict_processed` method.

    Note: The wrapper will default to forwarding calls to `device`, `forward`,
        `preprocess` and `predict` to the base reward net unless
        explicitly overridden in a subclass.
    """

    def forward(self, state: 'th.Tensor', action: 'th.Tensor', next_state: 'th.Tensor', done: 'th.Tensor') ->th.Tensor:
        """Compute rewards for a batch of transitions and keep gradients."""
        return self.base.forward(state, action, next_state, done)

    @abc.abstractmethod
    def predict_processed(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray', **kwargs) ->np.ndarray:
        """Predict processed must be overridden in subclasses."""

    def predict(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray') ->np.ndarray:
        __doc__ = super().predict.__doc__
        return self.base.predict(state, action, next_state, done)

    def predict_th(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray') ->th.Tensor:
        __doc__ = super().predict_th.__doc__
        return self.base.predict_th(state, action, next_state, done)


class ForwardWrapper(RewardNetWrapper):
    """An abstract RewardNetWrapper that changes the behavior of forward.

    Note that all forward wrappers must be placed before all
    predict processed wrappers.
    """

    def __init__(self, base: 'RewardNet'):
        """Create a forward wrapper.

        Args:
            base: The base reward network

        Raises:
            ValueError: if the base network is a `PredictProcessedWrapper`.
        """
        super().__init__(base)
        if isinstance(base, PredictProcessedWrapper):
            raise ValueError('ForwardWrapper cannot be applied on top of PredictProcessedWrapper!')


class RewardNetWithVariance(RewardNet):
    """A reward net that keeps track of its epistemic uncertainty through variance."""

    @abc.abstractmethod
    def predict_reward_moments(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray', **kwargs) ->Tuple[np.ndarray, np.ndarray]:
        """Compute the mean and variance of the reward distribution.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            **kwargs: may modify the behavior of subclasses

        Returns:
            * Estimated reward mean of shape `(batch_size,)`.
            * Estimated reward variance of shape `(batch_size,)`. # noqa: DAR202
        """


class BasicRewardNet(RewardNet):
    """MLP that takes as input the state, action, next state and done flag.

    These inputs are flattened and then concatenated to one another. Each input
    can enabled or disabled by the `use_*` constructor keyword arguments.
    """

    def __init__(self, observation_space: 'gym.Space', action_space: 'gym.Space', use_state: 'bool'=True, use_action: 'bool'=True, use_next_state: 'bool'=False, use_done: 'bool'=False, **kwargs):
        """Builds reward MLP.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: should the current state be included as an input to the MLP?
            use_action: should the current action be included as an input to the MLP?
            use_next_state: should the next state be included as an input to the MLP?
            use_done: should the "done" flag be included as an input to the MLP?
            kwargs: passed straight through to `build_mlp`.
        """
        super().__init__(observation_space, action_space)
        combined_size = 0
        self.use_state = use_state
        if self.use_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)
        self.use_action = use_action
        if self.use_action:
            combined_size += preprocessing.get_flattened_obs_dim(action_space)
        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)
        self.use_done = use_done
        if self.use_done:
            combined_size += 1
        full_build_mlp_kwargs: 'Dict[str, Any]' = {'hid_sizes': (32, 32), **kwargs, 'in_size': combined_size, 'out_size': 1, 'squeeze_output': True}
        self.mlp = networks.build_mlp(**full_build_mlp_kwargs)

    def forward(self, state, action, next_state, done):
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(state, 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(next_state, 1))
        if self.use_done:
            inputs.append(th.reshape(done, [-1, 1]))
        inputs_concat = th.cat(inputs, dim=1)
        outputs = self.mlp(inputs_concat)
        assert outputs.shape == state.shape[:1]
        return outputs


def cnn_transpose(tens: 'th.Tensor') ->th.Tensor:
    """Transpose a (b,h,w,c)-formatted tensor to (b,c,h,w) format."""
    if len(tens.shape) == 4:
        return th.permute(tens, (0, 3, 1, 2))
    else:
        raise ValueError(f'Invalid input: len(tens.shape) = {len(tens.shape)} != 4.')


class CnnRewardNet(RewardNet):
    """CNN that takes as input the state, action, next state and done flag.

    Inputs are boosted to tensors with channel, height, and width dimensions, and then
    concatenated. Image inputs are assumed to be in (h,w,c) format, unless the argument
    hwc_format=False is passed in. Each input can be enabled or disabled by the `use_*`
    constructor keyword arguments, but either `use_state` or `use_next_state` must be
    True.
    """

    def __init__(self, observation_space: 'gym.Space', action_space: 'gym.Space', use_state: 'bool'=True, use_action: 'bool'=True, use_next_state: 'bool'=False, use_done: 'bool'=False, hwc_format: 'bool'=True, **kwargs):
        """Builds reward CNN.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: Should the current state be included as an input to the CNN?
            use_action: Should the current action be included as an input to the CNN?
            use_next_state: Should the next state be included as an input to the CNN?
            use_done: Should the "done" flag be included as an input to the CNN?
            hwc_format: Are image inputs in (h,w,c) format (True), or (c,h,w) (False)?
                If hwc_format is False, image inputs are not transposed.
            kwargs: Passed straight through to `build_cnn`.

        Raises:
            ValueError: if observation or action space is not easily massaged into a
                CNN input.
        """
        super().__init__(observation_space, action_space)
        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        self.use_done = use_done
        self.hwc_format = hwc_format
        if not (self.use_state or self.use_next_state):
            raise ValueError('CnnRewardNet must take current or next state as input.')
        if not preprocessing.is_image_space(observation_space):
            raise ValueError('CnnRewardNet requires observations to be images.')
        assert isinstance(observation_space, spaces.Box)
        if self.use_action and not isinstance(action_space, spaces.Discrete):
            raise ValueError('CnnRewardNet can only use Discrete action spaces.')
        input_size = 0
        output_size = 1
        if self.use_state:
            input_size += self.get_num_channels_obs(observation_space)
        if self.use_action:
            assert isinstance(action_space, spaces.Discrete)
            output_size = int(action_space.n)
        if self.use_next_state:
            input_size += self.get_num_channels_obs(observation_space)
        if self.use_done:
            output_size *= 2
        full_build_cnn_kwargs: 'Dict[str, Any]' = {'hid_channels': (32, 32), **kwargs, 'in_channels': input_size, 'out_size': output_size, 'squeeze_output': output_size == 1}
        self.cnn = networks.build_cnn(**full_build_cnn_kwargs)

    def get_num_channels_obs(self, space: 'spaces.Box') ->int:
        """Gets number of channels for the observation."""
        return space.shape[-1] if self.hwc_format else space.shape[0]

    def forward(self, state: 'th.Tensor', action: 'th.Tensor', next_state: 'th.Tensor', done: 'th.Tensor') ->th.Tensor:
        """Computes rewardNet value on input state, action, next_state, and done flag.

        Takes inputs that will be used, transposes image states to (c,h,w) format if
        needed, reshapes inputs to have compatible dimensions, concatenates them, and
        inputs them into the CNN.

        Args:
            state: current state.
            action: current action.
            next_state: next state.
            done: flag for whether the episode is over.

        Returns:
            th.Tensor: reward of the transition.
        """
        inputs = []
        if self.use_state:
            state_ = cnn_transpose(state) if self.hwc_format else state
            inputs.append(state_)
        if self.use_next_state:
            next_state_ = cnn_transpose(next_state) if self.hwc_format else next_state
            inputs.append(next_state_)
        inputs_concat = th.cat(inputs, dim=1)
        outputs = self.cnn(inputs_concat)
        if self.use_action and not self.use_done:
            rewards = th.sum(outputs * action, dim=1)
        elif self.use_action and self.use_done:
            action_done_false = action * (1 - done[:, None])
            action_done_true = action * done[:, None]
            full_acts = th.cat((action_done_false, action_done_true), dim=1)
            rewards = th.sum(outputs * full_acts, dim=1)
        elif not self.use_action and self.use_done:
            dones_binary = done.long()
            dones_one_hot = nn.functional.one_hot(dones_binary, num_classes=2)
            rewards = th.sum(outputs * dones_one_hot, dim=1)
        else:
            rewards = outputs
        return rewards


class NormalizedRewardNet(PredictProcessedWrapper):
    """A reward net that normalizes the output of its base network."""

    def __init__(self, base: 'RewardNet', normalize_output_layer: 'Type[networks.BaseNorm]'):
        """Initialize the NormalizedRewardNet.

        Args:
            base: a base RewardNet
            normalize_output_layer: The class to use to normalize rewards. This
                can be any nn.Module that preserves the shape; e.g. `nn.Identity`,
                `nn.LayerNorm`, or `networks.RunningNorm`.
        """
        super().__init__(base=base)
        self.normalize_output_layer = normalize_output_layer(1)

    def predict_processed(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray', update_stats: 'bool'=True, **kwargs) ->np.ndarray:
        """Compute normalized rewards for a batch of transitions without gradients.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            update_stats: Whether to update the running stats of the normalization
                layer.
            **kwargs: kwargs passed to base predict_processed call.

        Returns:
            Computed normalized rewards of shape `(batch_size,`).
        """
        with networks.evaluating(self):
            rew_th = th.tensor(self.base.predict_processed(state, action, next_state, done, **kwargs), device=self.device)
            rew = self.normalize_output_layer(rew_th).detach().cpu().numpy().flatten()
        if update_stats:
            with th.no_grad():
                self.normalize_output_layer.update_stats(rew_th)
        assert rew.shape == state.shape[:1]
        return rew


class ShapedRewardNet(ForwardWrapper):
    """A RewardNet consisting of a base network and a potential shaping."""

    def __init__(self, base: 'RewardNet', potential: 'Callable[[th.Tensor], th.Tensor]', discount_factor: 'float'):
        """Setup a ShapedRewardNet instance.

        Args:
            base: the base reward net to which the potential shaping
                will be added. Shaping must be applied directly to the raw reward net.
                See error below.
            potential: A callable which takes
                a batch of states (as a PyTorch tensor) and returns a batch of
                potentials for these states. If this is a PyTorch Module, it becomes
                a submodule of the ShapedRewardNet instance.
            discount_factor: discount factor to use for the potential shaping.
        """
        super().__init__(base=base)
        self.potential = potential
        self.discount_factor = discount_factor

    def forward(self, state: 'th.Tensor', action: 'th.Tensor', next_state: 'th.Tensor', done: 'th.Tensor'):
        base_reward_net_output = self.base(state, action, next_state, done)
        new_shaping_output = self.potential(next_state).flatten()
        old_shaping_output = self.potential(state).flatten()
        new_shaping = (1 - done.float()) * new_shaping_output
        final_rew = base_reward_net_output + self.discount_factor * new_shaping - old_shaping_output
        assert final_rew.shape == state.shape[:1]
        return final_rew


class BasicPotentialMLP(nn.Module):
    """Simple implementation of a potential using an MLP."""

    def __init__(self, observation_space: 'gym.Space', hid_sizes: 'Iterable[int]', **kwargs):
        """Initialize the potential.

        Args:
            observation_space: observation space of the environment.
            hid_sizes: widths of the hidden layers of the MLP.
            kwargs: passed straight through to `build_mlp`.
        """
        super().__init__()
        potential_in_size = preprocessing.get_flattened_obs_dim(observation_space)
        self._potential_net = networks.build_mlp(in_size=potential_in_size, hid_sizes=hid_sizes, squeeze_output=True, flatten_input=True, **kwargs)

    def forward(self, state: 'th.Tensor') ->th.Tensor:
        return self._potential_net(state)


class BasicShapedRewardNet(ShapedRewardNet):
    """Shaped reward net based on MLPs.

    This is just a very simple convenience class for instantiating a BasicRewardNet
    and a BasicPotentialMLP and wrapping them inside a ShapedRewardNet.
    Mainly exists for backwards compatibility after
    https://github.com/HumanCompatibleAI/imitation/pull/311
    to keep the scripts working.

    TODO(ejnnr): if we ever modify AIRL so that it takes in a RewardNet instance
        directly (instead of a class and kwargs) and instead instantiate the
        RewardNet inside the scripts, then it probably makes sense to get rid
        of this class.

    """

    def __init__(self, observation_space: 'gym.Space', action_space: 'gym.Space', *, reward_hid_sizes: Sequence[int]=(32,), potential_hid_sizes: Sequence[int]=(32, 32), use_state: bool=True, use_action: bool=True, use_next_state: bool=False, use_done: bool=False, discount_factor: float=0.99, **kwargs):
        """Builds a simple shaped reward network.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            reward_hid_sizes: sequence of widths for the hidden layers
                of the base reward MLP.
            potential_hid_sizes: sequence of widths for the hidden layers
                of the potential MLP.
            use_state: should the current state be included as an input
                to the reward MLP?
            use_action: should the current action be included as an input
                to the reward MLP?
            use_next_state: should the next state be included as an input
                to the reward MLP?
            use_done: should the "done" flag be included as an input to the reward MLP?
            discount_factor: discount factor for the potential shaping.
            kwargs: passed straight through to `BasicRewardNet` and `BasicPotentialMLP`.
        """
        base_reward_net = BasicRewardNet(observation_space=observation_space, action_space=action_space, use_state=use_state, use_action=use_action, use_next_state=use_next_state, use_done=use_done, hid_sizes=reward_hid_sizes, **kwargs)
        potential_net = BasicPotentialMLP(observation_space=observation_space, hid_sizes=potential_hid_sizes, **kwargs)
        super().__init__(base_reward_net, potential_net, discount_factor=discount_factor)


class BasicPotentialCNN(nn.Module):
    """Simple implementation of a potential using a CNN."""

    def __init__(self, observation_space: 'gym.Space', hid_sizes: 'Iterable[int]', hwc_format: 'bool'=True, **kwargs):
        """Initialize the potential.

        Args:
            observation_space: observation space of the environment.
            hid_sizes: number of channels in hidden layers of the CNN.
            hwc_format: format of the observation. True if channel dimension is last,
                False if channel dimension is first.
            kwargs: passed straight through to `build_cnn`.

        Raises:
            ValueError: if observations are not images.
        """
        super().__init__()
        self.hwc_format = hwc_format
        if not preprocessing.is_image_space(observation_space):
            raise ValueError('CNN potential must be given image inputs.')
        assert isinstance(observation_space, spaces.Box)
        obs_shape = observation_space.shape
        in_channels = obs_shape[-1] if self.hwc_format else obs_shape[0]
        self._potential_net = networks.build_cnn(in_channels=in_channels, hid_channels=hid_sizes, squeeze_output=True, **kwargs)

    def forward(self, state: 'th.Tensor') ->th.Tensor:
        state_ = cnn_transpose(state) if self.hwc_format else state
        return self._potential_net(state_)


class RewardEnsemble(RewardNetWithVariance):
    """A mean ensemble of reward networks.

    A reward ensemble is made up of individual reward networks. To maintain consistency
    the "output" of a reward network will be defined as the results of its
    `predict_processed`. Thus for example the mean of the ensemble is the mean of the
    results of its members predict processed classes.
    """
    members: 'nn.ModuleList'

    def __init__(self, observation_space: 'gym.Space', action_space: 'gym.Space', members: 'Iterable[RewardNet]'):
        """Initialize the RewardEnsemble.

        Args:
            observation_space: the observation space of the environment
            action_space: the action space of the environment
            members: the member networks that will make up the ensemble.

        Raises:
            ValueError: if num_members is less than 1
        """
        super().__init__(observation_space, action_space)
        members = list(members)
        if len(members) < 2:
            raise ValueError('Must be at least 2 member in the ensemble.')
        self.members = nn.ModuleList(members)

    @property
    def num_members(self):
        """The number of members in the ensemble."""
        return len(self.members)

    def predict_processed_all(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray', **kwargs) ->np.ndarray:
        """Get the results of predict processed on all of the members.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            kwargs: passed along to ensemble members.

        Returns:
            The result of predict processed for each member in the ensemble of
                shape `(batch_size, num_members)`.
        """
        batch_size = state.shape[0]
        rewards_list = [member.predict_processed(state, action, next_state, done, **kwargs) for member in self.members]
        rewards: 'np.ndarray' = np.stack(rewards_list, axis=-1)
        assert rewards.shape == (batch_size, self.num_members)
        return rewards

    @th.no_grad()
    def predict_reward_moments(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray', **kwargs) ->Tuple[np.ndarray, np.ndarray]:
        """Compute the standard deviation of the reward distribution for a batch.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            **kwargs: passed along to predict processed.

        Returns:
            * Reward mean of shape `(batch_size,)`.
            * Reward variance of shape `(batch_size,)`.
        """
        batch_size = state.shape[0]
        all_rewards = self.predict_processed_all(state, action, next_state, done, **kwargs)
        mean_reward = all_rewards.mean(-1)
        var_reward = all_rewards.var(-1, ddof=1)
        assert mean_reward.shape == var_reward.shape == (batch_size,)
        return mean_reward, var_reward

    def forward(self, *args) ->th.Tensor:
        """The forward method of the ensemble should in general not be used directly."""
        raise NotImplementedError

    def predict_processed(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray', **kwargs) ->np.ndarray:
        """Return the mean of the ensemble members."""
        return self.predict(state, action, next_state, done, **kwargs)

    def predict(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray', **kwargs):
        """Return the mean of the ensemble members."""
        mean, _ = self.predict_reward_moments(state, action, next_state, done, **kwargs)
        return mean


class AddSTDRewardWrapper(PredictProcessedWrapper):
    """Adds a multiple of the estimated standard deviation to mean reward."""
    base: 'RewardNetWithVariance'

    def __init__(self, base: 'RewardNetWithVariance', default_alpha: 'float'=0.0):
        """Create a reward network that adds a multiple of the standard deviation.

        Args:
            base: A reward network that keeps track of its epistemic variance.
                This is used to compute the standard deviation.
            default_alpha: multiple of standard deviation to add to the reward mean.
                Defaults to 0.0.

        Raises:
            TypeError: if base is not an instance of RewardNetWithVariance
        """
        super().__init__(base)
        if not isinstance(base, RewardNetWithVariance):
            raise TypeError('Cannot add standard deviation to reward net that is not an instance of RewardNetWithVariance!')
        self.default_alpha = default_alpha

    def predict_processed(self, state: 'np.ndarray', action: 'np.ndarray', next_state: 'np.ndarray', done: 'np.ndarray', alpha: 'Optional[float]'=None, **kwargs) ->np.ndarray:
        """Compute a lower/upper confidence bound on the reward without gradients.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            alpha: multiple of standard deviation to add to the reward mean. Defaults
                to the value provided at initialization.
            **kwargs: are not used

        Returns:
            Estimated lower confidence bounds on rewards of shape `(batch_size,`).
        """
        del kwargs
        if alpha is None:
            alpha = self.default_alpha
        reward_mean, reward_var = self.base.predict_reward_moments(state, action, next_state, done)
        return reward_mean + alpha * np.sqrt(reward_var)


class SqueezeLayer(nn.Module):
    """Torch module that squeezes a B*1 tensor down into a size-B vector."""

    def forward(self, x):
        assert x.ndim == 2 and x.shape[1] == 1
        new_value = x.squeeze(1)
        assert new_value.ndim == 1
        return new_value


class BaseNorm(nn.Module, abc.ABC):
    """Base class for layers that try to normalize the input to mean 0 and variance 1.

    Similar to BatchNorm, LayerNorm, etc. but whereas they only use statistics from
    the current batch at train time, we use statistics from all batches.
    """
    running_mean: 'th.Tensor'
    running_var: 'th.Tensor'
    count: 'th.Tensor'

    def __init__(self, num_features: 'int', eps: 'float'=1e-05):
        """Builds RunningNorm.

        Args:
            num_features: Number of features; the length of the non-batch dimension.
            eps: Small constant for numerical stability. Inputs are rescaled by
                `1 / sqrt(estimated_variance + eps)`.
        """
        super().__init__()
        self.eps = eps
        self.register_buffer('running_mean', th.empty(num_features))
        self.register_buffer('running_var', th.empty(num_features))
        self.register_buffer('count', th.empty((), dtype=th.int))
        BaseNorm.reset_running_stats(self)

    def reset_running_stats(self) ->None:
        """Resets running stats to defaults, yielding the identity transformation."""
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.count.zero_()

    def forward(self, x: 'th.Tensor') ->th.Tensor:
        """Updates statistics if in training mode. Returns normalized `x`."""
        if self.training:
            with th.no_grad():
                self.update_stats(x)
        return (x - self.running_mean) / th.sqrt(self.running_var + self.eps)

    @abc.abstractmethod
    def update_stats(self, batch: 'th.Tensor') ->None:
        """Update `self.running_mean`, `self.running_var` and `self.count`."""


class RunningNorm(BaseNorm):
    """Normalizes input to mean 0 and standard deviation 1 using a running average.

    Similar to BatchNorm, LayerNorm, etc. but whereas they only use statistics from
    the current batch at train time, we use statistics from all batches.

    This should replicate the common practice in RL of normalizing environment
    observations, such as using ``VecNormalize`` in Stable Baselines. Note that
    the behavior of this class is slightly different from `VecNormalize`, e.g.,
    it works with the current reward instead of return estimate, and subtracts the mean
    reward whereas ``VecNormalize`` only rescales it.
    """

    def update_stats(self, batch: 'th.Tensor') ->None:
        """Update `self.running_mean`, `self.running_var` and `self.count`.

        Uses Chan et al (1979), "Updating Formulae and a Pairwise Algorithm for
        Computing Sample Variances." to update the running moments in a numerically
        stable fashion.

        Args:
            batch: A batch of data to use to update the running mean and variance.
        """
        batch_mean = th.mean(batch, dim=0)
        batch_var = th.var(batch, dim=0, unbiased=False)
        batch_count = batch.shape[0]
        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count
        self.running_mean += delta * batch_count / tot_count
        self.running_var *= self.count
        self.running_var += batch_var * batch_count
        self.running_var += th.square(delta) * self.count * batch_count / tot_count
        self.running_var /= tot_count
        self.count += batch_count


class EMANorm(BaseNorm):
    """Similar to RunningNorm but uses an exponential weighting."""
    inv_learning_rate: 'th.Tensor'
    num_batches: 'th.IntTensor'

    def __init__(self, num_features: 'int', decay: 'float'=0.99, eps: 'float'=1e-05):
        """Builds EMARunningNorm.

        Args:
            num_features: Number of features; the length of the non-batch dim.
            decay: how quickly the weight on past samples decays over time.
            eps: small constant for numerical stability.

        Raises:
            ValueError: if decay is out of range.
        """
        super().__init__(num_features, eps=eps)
        if not 0 < decay < 1:
            raise ValueError('decay must be between 0 and 1')
        self.decay = decay
        self.register_buffer('inv_learning_rate', th.empty(()))
        self.register_buffer('num_batches', th.empty((), dtype=th.int))
        EMANorm.reset_running_stats(self)

    def reset_running_stats(self):
        """Reset the running stats of the normalization layer."""
        super().reset_running_stats()
        self.inv_learning_rate.zero_()
        self.num_batches.zero_()

    def update_stats(self, batch: 'th.Tensor') ->None:
        """Update `self.running_mean` and `self.running_var` in batch mode.

        Reference Algorithm 3 from:
        https://github.com/HumanCompatibleAI/imitation/files/9456540/Incremental_batch_EMA_and_EMV.pdf

        Args:
            batch: A batch of data to use to update the running mean and variance.
        """
        b_size = batch.shape[0]
        if len(batch.shape) == 1:
            batch = batch.reshape(b_size, 1)
        self.inv_learning_rate += self.decay ** self.num_batches
        learning_rate = 1 / self.inv_learning_rate
        delta_mean = batch.mean(0) - self.running_mean
        self.running_mean += learning_rate * delta_mean
        batch_var = batch.var(0, unbiased=False)
        delta_var = batch_var + (1 - learning_rate) * delta_mean ** 2 - self.running_var
        self.running_var += learning_rate * delta_var
        self.count += b_size
        self.num_batches += 1


class ZeroModule(nn.Module):
    """Module that always returns zeros of same shape as input."""

    def __init__(self, features_dim: 'int'):
        """Builds ZeroModule."""
        super().__init__()
        self.features_dim = features_dim

    def forward(self, x: 'th.Tensor') ->th.Tensor:
        """Returns zeros of same shape as `x`."""
        assert x.shape[1:] == (self.features_dim,)
        return th.zeros_like(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (EMANorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RunningNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ZeroModule,
     lambda: ([], {'features_dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

