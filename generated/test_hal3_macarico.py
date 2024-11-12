
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


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.parameter import Parameter


from torch.autograd import Variable as Var


from collections import Counter


import numpy as np


import random


import scipy.optimize


import torch.nn


import math


from collections import defaultdict


import time


from collections import deque


import itertools


from copy import deepcopy


import random as py_random


class TypeMemory(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.param = Parameter(torch.zeros(1))


def check_intentional_override(class_name, fn_name, override_bool_name, obj, *fn_args):
    if not getattr(obj, override_bool_name):
        try:
            getattr(obj, fn_name)(*fn_args)
        except NotImplementedError:
            None
        except:
            pass


class DynamicFeatures(nn.Module):
    """`DynamicFeatures` are any function that map an `Env` to a
    tensor. The dimension of the feature representation tensor should
    be (1, N, `dim`), where `N` is the length of the input, and
    `dim()` returns the dimensionality.

    The `forward` function computes the features."""
    OVERRIDE_FORWARD = False

    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim
        self._current_env = None
        self._features = None
        self._batched_features = None
        self._batched_lengths = None
        self._my_id = '%s #%d' % (type(self), id(self))
        self._recompute_always = True
        self._typememory = TypeMemory()
        check_intentional_override('DynamicFeatures', '_forward', 'OVERRIDE_FORWARD', self, None)

    def _forward(self, env):
        raise NotImplementedError('abstract')

    def forward(self, env):
        if self._batched_features is not None and hasattr(env, '_stored_batch_features') and self._my_id in env._stored_batch_features:
            i = env._stored_batch_features[self._my_id]
            assert 0 <= i and i < self._batched_features.shape[0]
            assert self._batched_lengths[i] <= self._batched_features.shape[1]
            l = self._batched_lengths[i]
            self._features = self._batched_features[i, :l, :].unsqueeze(0)
        if self._features is None or self._recompute_always:
            self._features = self._forward(env)
            assert self._features.dim() == 3
            assert self._features.shape[0] == 1
            assert self._features.shape[2] == self.dim
        assert self._features is not None
        return self._features

    def _forward_batch(self, envs):
        raise NotImplementedError('abstract')

    def forward_batch(self, envs):
        if self._batched_features is not None:
            return self._batched_features, self._batched_lengths
        try:
            res = self._forward_batch(envs)
            assert isinstance(res, tuple)
            self._batched_features, self._batched_lengths = res
            if self._batched_features.shape[0] != len(envs):
                ipdb.set_trace()
        except NotImplementedError:
            pass
        if self._batched_features is None:
            bf = [self._forward(env) for env in envs]
            for x in bf:
                assert x.dim() == 3
                assert x.shape[0] == 1
                assert x.shape[2] == self.dim
            max_len = max(x.shape[1] for x in bf)
            self._batched_features = Var(self._typememory.param.data.new(len(envs), max_len, self.dim).zero_())
            self._batched_lengths = []
            for i, x in enumerate(bf):
                self._batched_features[i, :x.shape[1], :] = x
                self._batched_lengths.append(x.shape[1])
        assert self._batched_features.shape[0] == len(envs)
        for i, env in enumerate(envs):
            if not hasattr(env, '_stored_batch_features'):
                env._stored_batch_features = dict()
            env._stored_batch_features[self._my_id] = i
        return self._batched_features, self._batched_lengths


class StaticFeatures(DynamicFeatures):

    def __init__(self, dim):
        DynamicFeatures.__init__(self, dim)
        self._recompute_always = False


class Actor(nn.Module):
    """An `Actor` is a module that computes features dynamically as a policy runs."""
    OVERRIDE_FORWARD = False

    def __init__(self, n_actions, dim, attention):
        nn.Module.__init__(self)
        self._current_env = None
        self._features = None
        self.n_actions = n_actions
        self.dim = dim
        self.attention = nn.ModuleList(attention)
        self._T = None
        self._last_t = 0
        for att in attention:
            if att.actor_dependent:
                att.set_actor(self)
        self._typememory = TypeMemory()
        check_intentional_override('Actor', '_forward', 'OVERRIDE_FORWARD', self, None)

    def reset(self):
        self._last_t = 0
        self._T = None
        self._features = None
        self._reset()

    def _reset(self):
        pass

    def _forward(self, state, x):
        raise NotImplementedError('abstract')

    def hidden(self):
        raise NotImplementedError('abstract')

    def forward(self, env):
        if self._features is None or self._T is None:
            self._T = env.horizon()
            self._features = [None] * self._T
            self._last_t = 0
        t = env.timestep()
        if t > self._last_t + 1:
            ipdb.set_trace()
        assert t <= self._last_t + 1, '%d <= %d+1' % (t, self._last_t)
        assert t >= self._last_t, '%d >= %d' % (t, self._last_t)
        self._last_t = t
        assert self._features is not None
        assert t >= 0, 'expect t>=0, bug?'
        assert t < self._T, '%d=t < T=%d' % (t, self._T)
        assert t < len(self._features)
        if self._features[t] is not None:
            return self._features[t]
        assert t == 0 or self._features[t - 1] is not None
        x = []
        for att in self.attention:
            x += att(env)
        ft = self._forward(env, x)
        assert ft.dim() == 2
        assert ft.shape[0] == 1
        assert ft.shape[1] == self.dim
        self._features[t] = ft
        return self._features[t]


class Policy(nn.Module):
    """A `Policy` is any function that contains a `forward` function that
    maps states to actions."""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, state):
        raise NotImplementedError('abstract')
    """
    cases where we need to reset:
    - 0. new minibatch. this means reset EVERYTHING.
    - 1. new example in a minibatch. this means reset dynamic and static _features, but not _batched_features
    - 2. replaying the current example. this means reset dynamic ONLY.
    flipped around:
    - Actors are reset in all cases
    - _features is reset in 0 and 1
    - _batched_features is reset in 0
    """

    def new_minibatch(self):
        self._reset_some(0, True)

    def new_example(self):
        self._reset_some(1, True)

    def new_run(self):
        self._reset_some(2, True)

    def _reset_some(self, reset_type, recurse):
        for module in self.modules():
            if isinstance(module, Actor):
                module.reset()
            if isinstance(module, DynamicFeatures):
                if reset_type == 0 or reset_type == 1:
                    module._features = None
                if reset_type == 0:
                    module._batched_features = None


class StochasticPolicy(Policy):

    def stochastic(self, state):
        raise NotImplementedError('abstract')

    def sample(self, state):
        return self.stochastic(state)[0]


class Example(object):

    def __init__(self, X=None, Y=None):
        self.X = X
        self.Y = Y
        self.Yhat = None

    def __str__(self):
        return '{ X: %s, Y: %s, Yhat: %s }' % (self.X, self.Y, self.Yhat)

    def __repr__(self):
        return str(self)

    def input_str(self):
        return self._simple_str(self.X)

    def output_str(self):
        return self._simple_str(self.Y)

    def prediction_str(self):
        return self._simple_str(self.Yhat)

    def _simple_str(self, A):
        if A is None:
            return '?'
        if isinstance(A, list):
            return ' '.join(map(str, A))
        return str(A)


class Env(object):
    """An implementation of an environment; aka a search task or MDP.

    Args:
        n_actions: the number of unique actions available to a policy
                   in this Env (actions are numbered [0, n_actions))

    Must provide a `_run_episode(policy)` function that performs a
    complete run through this environment, acting according to
    `policy`.

    May optionally provide a `_rewind` function that some learning
    algorithms (e.g., LOLS) requires.
    """
    OVERRIDE_RUN_EPISODE = False
    OVERRIDE_REWIND = False

    def __init__(self, n_actions, T, example=None):
        self.n_actions = n_actions
        self.T = T
        self.example = Example() if example is None else example
        self._trajectory = []

    def horizon(self):
        return self.T

    def timestep(self):
        return len(self._trajectory)

    def output(self):
        return self._trajectory

    def run_episode(self, policy):

        def _policy(state):
            assert self.timestep() < self.horizon()
            a = policy(state)
            self._trajectory.append(a)
            return a
        if hasattr(policy, 'new_example'):
            policy.new_example()
        self.rewind(policy)
        _policy.new_minibatch = policy.new_minibatch if hasattr(policy, 'new_minibatch') else None
        _policy.new_example = policy.new_example if hasattr(policy, 'new_example') else None
        _policy.new_run = policy.new_run if hasattr(policy, 'new_run') else None
        out = self._run_episode(_policy)
        self.example.Yhat = out if out is not None else self._trajectory
        return self.example.Yhat

    def input_x(self):
        return self.example.X

    def rewind(self, policy):
        self._trajectory = []
        if hasattr(policy, 'new_run'):
            policy.new_run()
        self._rewind()

    def _run_episode(self, policy):
        raise NotImplementedError('abstract')

    def _rewind(self):
        raise NotImplementedError('abstract')


class CostSensitivePolicy(Policy):
    OVERRIDE_UPDATE = False

    def __init__(self):
        nn.Module.__init__(self)
        check_intentional_override('CostSensitivePolicy', '_update', 'OVERRIDE_UPDATE', self, None, None)

    def predict_costs(self, state):
        raise NotImplementedError('abstract')

    def costs_to_action(self, state, pred_costs):
        if isinstance(pred_costs, Var):
            pred_costs = pred_costs.data
        if state.actions is None or len(state.actions) == 0 or len(state.actions) == pred_costs.shape[0]:
            return pred_costs.min(0)[1][0]
        i = None
        for a in state.actions:
            if i is None or pred_costs[a] < pred_costs[i]:
                i = a
        return i

    def update(self, state_or_pred_costs, truth, actions=None):
        if isinstance(state_or_pred_costs, Env):
            assert actions is None
            actions = state_or_pred_costs.actions
            state_or_pred_costs = self.predict_costs(state_or_pred_costs)
        return self._update(state_or_pred_costs, truth, actions)

    def _update(self, pred_costs, truth, actions=None):
        raise NotImplementedError('abstract')


class Learner(Policy):
    """A `Learner` behaves identically to a `Policy`, but does "stuff"
    internally to, eg., compute gradients through pytorch's `backward`
    procedure. Not all learning algorithms can be implemented this way
    (e.g., LOLS) but most can (DAgger, reinforce, etc.)."""

    def forward(self, state):
        raise NotImplementedError('abstract method not defined.')

    def get_objective(self, loss):
        raise NotImplementedError('abstract method not defined.')


class NoopLearner(Learner):

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, state):
        return self.policy(state)

    def get_objective(self, loss):
        return 0.0


class LearningAlg(nn.Module):

    def __call__(self, env):
        raise NotImplementedError('abstract method not defined.')


class Attention(nn.Module):
    """ It is usually the case that the `Features` one wants to compute
    are a function of only some part of the input at any given time
    step. For instance, in a sequence labeling task, one might only
    want to look at the `Features` of the word currently being
    labeled. Or in a machine translation task, one might want to have
    dynamic, differentiable softmax-style attention.

    For static `Attention`, the class must define its `arity`: the
    number of places that it looks (e.g., one in sequence labeling).
    """
    OVERRIDE_FORWARD = False
    arity = 0
    actor_dependent = False

    def __init__(self, features):
        nn.Module.__init__(self)
        self.features = features
        self.dim = (self.arity or 1) * self.features.dim
        check_intentional_override('Attention', '_forward', 'OVERRIDE_FORWARD', self, None)

    def forward(self, state):
        fts = self._forward(state)
        dim_sum = 0
        if self.arity is None:
            assert len(fts) == 1
        if self.arity is not None:
            assert len(fts) == self.arity
        for ft in fts:
            assert ft.dim() == 2
            assert ft.shape[0] == 1
            dim_sum += ft.shape[1]
        assert dim_sum == self.dim
        return fts

    def _forward(self, state):
        raise NotImplementedError('abstract')

    def set_actor(self, actor):
        raise NotImplementedError('abstract')

    def make_out_of_bounds(self):
        oob = Parameter(torch.Tensor(1, self.features.dim))
        oob.data.zero_()
        return oob


class Torch(nn.Module):

    def __init__(self, features, dim, layers):
        nn.Module.__init__(self)
        self.features = features
        self.dim = dim
        self.torch_layers = layers if isinstance(layers, nn.ModuleList) else nn.ModuleList(layers) if isinstance(layers, list) else nn.ModuleList([layers])

    def forward(self, x):
        x = self.features(x)
        for l in self.torch_layers:
            x = l(x)
        return x


def Varng(*args, **kwargs):
    return torch.autograd.Variable(*args, **kwargs, requires_grad=False)


qrnn_available = False


class Annealing:
    """Base case."""

    def __call__(self, T):
        raise NotImplementedError('abstract method not implemented.')


class NoAnnealing(Annealing):
    """Constant rate."""

    def __init__(self, value):
        self.value = value

    def __call__(self, T):
        return self.value


def break_ties_by_policy(reference, policy, state, force_advance_policy=True):
    costs = torch.zeros(state.n_actions)
    set_costs = False
    try:
        reference.set_min_costs_to_go(state, costs)
        set_costs = True
    except NotImplementedError:
        pass
    if not set_costs:
        ref = reference(state)
        if force_advance_policy:
            policy(state)
        return ref
    old_actions = state.actions
    min_cost = min(costs[a] for a in old_actions)
    state.actions = [a for a in old_actions if costs[a] <= min_cost]
    a = policy(state)
    assert a is not None, 'got action None in %s, costs=%s, old_actions=%s' % (state.actions, costs, old_actions)
    state.actions = old_actions
    return a


class stochastic(object):

    def __init__(self, inst):
        assert isinstance(inst, Annealing)
        self.inst = inst
        self.time = 1

    def step(self):
        self.time += 1

    def __call__(self):
        return random() <= self.inst(self.time)


def argmin(vec, allowed=None, dim=0):
    if isinstance(vec, Var):
        vec = vec.data
    if allowed is None or len(allowed) == 0 or len(allowed) == vec.shape[dim]:
        return vec.min(dim)[1].item()
    i = None
    for a in allowed:
        if i is None or dim == 0 and vec[a] < vec[i] or dim == 1 and vec[0, a] < vec[0, i] or dim == 2 and vec[0, 0, a] < vec[0, 0, i]:
            i = a
    return i


class TiedRandomness(object):

    def __init__(self, rand):
        self.tied = {}
        self.rand = rand

    def reset(self):
        self.tied = {}

    def __call__(self, t):
        if t not in self.tied:
            self.tied[t] = self.rand(t)
        return self.tied[t]


def one_step_deviation(T, rollin, rollout, dev_t, dev_a):

    def run(t):
        if t == dev_t:
            return EpisodeRunner.ACT, dev_a
        elif t < dev_t:
            return rollin(t)
        else:
            return rollout(t)
    return run


class EWMA(object):
    """Exponentially weighted moving average."""

    def __init__(self, rate, initial_value=0.0):
        self.rate = rate
        self.value = initial_value

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value += self.rate * (x - self.value)

    def __call__(self):
        return self.value


class LinearValueFn(nn.Module):

    def __init__(self, features, disconnect_values=True):
        nn.Module.__init__(self)
        self.features = features
        self.dim = features.dim
        self.disconnect_values = disconnect_values
        self.value_fn = nn.Linear(self.dim, 1)

    def forward(self, state):
        x = self.features(state)
        if self.disconnect_values:
            x = Varng(x.data)
        return self.value_fn(x)


def actions_to_probs(actions, n_actions):
    probs = torch.zeros(n_actions)
    bag_size = len(actions)
    prob = 1.0 / bag_size
    for action_set in actions:
        for action in action_set:
            probs[action] += prob / len(action_set)
    return probs


def min_set(costs, limit_actions=None):
    min_val = None
    min_set = []
    if limit_actions is None:
        for a, c in enumerate(costs):
            if min_val is None or c < min_val:
                min_val = c
                min_set = [a]
            elif c == min_val:
                min_set.append(a)
    else:
        for a in limit_actions:
            c = costs[a]
            if min_val is None or c < min_val:
                min_val = c
                min_set = [a]
            elif c == min_val:
                min_set.append(a)
    return min_set


class BootstrapCost:

    def __init__(self, costs, greedy_predict=True):
        self.costs = costs
        self.greedy_predict = greedy_predict

    def average_cost(self):
        return sum(self.costs) / len(self.costs)

    def data(self):
        if self.greedy_predict:
            return self.costs[0].data
        else:
            return self.average_cost().data

    def get_probs(self, limit_actions=None):
        assert len(self.costs) > 0
        n_actions = len(self.costs[0].data)
        actions = [min_set(c.data, limit_actions) for c in self.costs]
        return actions_to_probs(actions, n_actions)

    def __getitem__(self, idx):
        if self.greedy_predict:
            return self.costs[0][idx]
        else:
            return self.average_cost()[idx]

    def __neg__(self):
        if self.greedy_predict:
            return self.costs[0].__neg__()
        else:
            return self.average_cost().__neg__()

    def argmin(self):
        if self.greedy_predict:
            return self.costs[0].argmin()
        else:
            return self.average_cost().argmin()


def bootstrap_probabilities(n_actions, policy_bag, state, deviate_to):
    actions = [[policy(state, deviate_to)] for policy in policy_bag]
    probs = actions_to_probs(actions, n_actions)
    return probs


def build_policy_bag(features_bag, n_actions, loss_fn, n_layers, hidden_dim):
    return [LinearPolicy(features, n_actions, loss_fn=loss_fn, n_layers=n_layers, hidden_dim=hidden_dim) for features in features_bag]


def delegate_with_poisson(params, functions, greedy_update):
    total_loss = 0.0
    functions_params_pairs = zip(functions, params)
    for idx, (loss_fn, params) in enumerate(functions_params_pairs):
        loss_i = loss_fn(*params)
        if greedy_update and idx == 0:
            count_i = 1
        else:
            count_i = np.random.poisson(1)
        total_loss = total_loss + count_i * loss_i
    return total_loss


class CostEvalPolicy(Policy):

    def __init__(self, reference, policy):
        self.policy = policy
        self.reference = reference
        self.costs = None
        self.n_actions = policy.n_actions
        self.record = [None] * 1000
        self.record_i = 0

    def __call__(self, state):
        if self.costs is None:
            self.costs = torch.zeros(state.n_actions)
        self.costs *= 0
        self.reference.set_min_costs_to_go(state, self.costs)
        p_c = self.policy.predict_costs(state)
        self.record[self.record_i] = sum(abs(self.costs - p_c.data))
        self.record_i = (self.record_i + 1) % len(self.record)
        if np.random.random() < 0.0001 and self.record[-1] is not None:
            None
        return self.policy.greedy(state, pred_costs=p_c)

    def predict_costs(self, state):
        return self.policy.predict_costs(state)

    def forward_partial_complete(self, pred_costs, truth, actions):
        return self.policy.forward_partial_complete(pred_costs, truth, actions)

    def update(self, _):
        pass


def truth_to_vec(truth, tmp_vec):
    if isinstance(truth, torch.FloatTensor):
        return truth
    if isinstance(truth, int) or isinstance(truth, np.int32) or isinstance(truth, np.int64):
        tmp_vec.zero_()
        tmp_vec += 1
        tmp_vec[truth] = 0
        return tmp_vec
    if isinstance(truth, list) or isinstance(truth, set):
        tmp_vec.zero_()
        tmp_vec += 1
        for t in truth:
            tmp_vec[t] = 0
        return tmp_vec
    raise ValueError('invalid argument type for "truth", must be in, list or set; got "%s"' % type(truth))


class WAPPolicy(Policy):
    """Linear policy, with weighted all pairs

    Notes:

    This policy can be trained with
    - policy gradient via `policy.stochastic().reinforce(reward)`

    - Cost-sensitive one-against-all linear regression (CSOAA) via
      `policy.forward(state, truth)`

    """

    def __init__(self, features, n_actions):
        self.n_actions = n_actions
        dim = 1 if features is None else features.dim
        n_actions_choose_2 = n_actions * (n_actions - 1) // 2
        self._wap_w = dy_model.add_parameters((n_actions_choose_2, dim))
        self._wap_b = dy_model.add_parameters(n_actions_choose_2)
        self.features = features

    def __call__(self, state):
        return self.greedy(state)

    def sample(self, state):
        return self.stochastic(state)

    def stochastic(self, state, temperature=1):
        return self.stochastic_with_probability(state, temperature)[0]

    def stochastic_with_probability(self, state, temperature=1):
        assert False

    def predict_costs(self, state):
        """Predict costs using the csoaa model accounting for `state.actions`"""
        if self.features is None:
            feats = dy.parameter(self.dy_model.add_parameters(1))
            self.features = lambda _: feats
        else:
            feats = self.features(state)
        wap_w = dy.parameter(self._wap_w)
        wap_b = dy.parameter(self._wap_b)
        return dy.affine_transform([wap_b, wap_w, feats])

    def greedy(self, state, pred_costs=None):
        if pred_costs is None:
            pred_costs = self.predict_costs(state)
        if isinstance(pred_costs, dy.Expression):
            pred_costs = pred_costs.data
        costs = torch.zeros(self.n_actions)
        k = 0
        for i in xrange(self.n_actions):
            for j in xrange(i):
                costs[i] -= pred_costs[k]
                costs[j] += pred_costs[k]
                k += 1
        if len(state.actions) == self.n_actions:
            return costs.argmin()
        best = None
        for a in state.actions:
            if best is None or costs[a] < costs[best]:
                best = a
        return best

    def forward_partial_complete(self, pred_costs, truth, actions):
        if isinstance(truth, int):
            truth = [truth]
        if isinstance(truth, list):
            truth0 = truth
            truth = torch.ones(self.n_actions)
            for k in truth0:
                truth[k] = 0.0
        obj = 0.0
        k = 0
        if not isinstance(actions, set):
            actions = set(actions)
        for i in xrange(self.n_actions):
            for j in xrange(i):
                weight = abs(truth[i] - truth[j])
                label = -1 if truth[i] > truth[j] else +1
                if weight > 1e-06:
                    l = 1 - label * pred_costs[k]
                    l = 0.5 * (l + dy.abs(l))
                    obj += weight * l
                k += 1
        return obj

    def forward(self, state, truth):
        costs = self.predict_costs(state)
        return self.forward_partial_complete(costs, truth, state.actions)


def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):
    return sum(hand) + (10 if usable_ace(hand) else 0)


PASSABLE, SEED, POWER = 0, 1, 2


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (NoopLearner,
     lambda: ([], {'policy': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Torch,
     lambda: ([], {'features': torch.nn.ReLU(), 'dim': 4, 'layers': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

