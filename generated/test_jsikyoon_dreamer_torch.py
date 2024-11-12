
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


import collections


import functools


import warnings


import numpy as np


import torch


from torch import nn


from torch import distributions as torchd


import torch.nn.functional as F


import re


import time


import uuid


from torch.nn import functional as F


from torch.utils.data import Dataset


from torch.utils.tensorboard import SummaryWriter


def count_steps(folder):
    return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):

    def __init__(self, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._should_train = tools.Every(config.train_every)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = count_steps(config.traindir)
        config.actor_entropy = lambda x=config.actor_entropy: tools.schedule(x, self._step)
        config.actor_state_entropy = lambda x=config.actor_state_entropy: tools.schedule(x, self._step)
        config.imag_gradient_mix = lambda x=config.imag_gradient_mix: tools.schedule(x, self._step)
        self._dataset = dataset
        self._wm = models.WorldModel(self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm, config.behavior_stop_grad)
        reward = lambda f, s, a: self._wm.heads['reward'](f).mean
        self._expl_behavior = dict(greedy=lambda : self._task_behavior, random=lambda : expl.Random(config), plan2explore=lambda : expl.Plan2Explore(config, self._wm, reward))[config.expl_behavior]()

    def __call__(self, obs, reset, state=None, reward=None, training=True):
        step = self._step
        if self._should_reset(step):
            state = None
        if state is not None and reset.any():
            mask = 1 - reset
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for i in range(len(state[1])):
                state[1][i] *= mask[i]
        if training and self._should_train(step):
            steps = self._config.pretrain if self._should_pretrain() else self._config.train_steps
            for _ in range(steps):
                self._train(next(self._dataset))
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                openl = self._wm.video_pred(next(self._dataset))
                self._logger.video('train_openl', to_np(openl))
                self._logger.write(fps=True)
        policy_output, state = self._policy(obs, state, training)
        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            batch_size = len(obs['image'])
            latent = self._wm.dynamics.initial(len(obs['image']))
            action = torch.zeros((batch_size, self._config.num_actions))
        else:
            latent, action = state
        embed = self._wm.encoder(self._wm.preprocess(obs))
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, self._config.collect_dyn_sample)
        if self._config.eval_state_mean:
            latent['stoch'] = latent['mean']
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor_dist == 'onehot_gumble':
            action = torch.one_hot(torch.argmax(action, dim=-1), self._config.num_actions)
        action = self._exploration(action, training)
        policy_output = {'action': action, 'logprob': logprob}
        state = latent, action
        return policy_output, state

    def _exploration(self, action, training):
        amount = self._config.expl_amount if training else self._config.eval_noise
        if amount == 0:
            return action
        if 'onehot' in self._config.actor_dist:
            probs = amount / self._config.num_actions + (1 - amount) * action
            return tools.OneHotDist(probs=probs).sample()
        else:
            return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)
        raise NotImplementedError(self._config.action_noise)

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        if self._config.pred_discount:
            start = {k: v[:, :-1] for k, v in post.items()}
            context = {k: v[:, :-1] for k, v in context.items()}
        reward = lambda f, s, a: self._wm.heads['reward'](self._wm.dynamics.get_feat(s)).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != 'greedy':
            if self._config.pred_discount:
                data = {k: v[:, :-1] for k, v in data.items()}
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({('expl_' + key): value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


class Random(nn.Module):

    def __init__(self, config):
        self._config = config

    def actor(self, feat):
        shape = feat.shape[:-1] + [self._config.num_actions]
        if self._config.actor_dist == 'onehot':
            return tools.OneHotDist(torch.zeros(shape))
        else:
            ones = torch.ones(shape)
            return tools.ContDist(torchd.uniform.Uniform(-ones, ones))

    def train(self, start, context):
        return None, {}


class Plan2Explore(nn.Module):

    def __init__(self, config, world_model, reward=None):
        self._config = config
        self._reward = reward
        self._behavior = models.ImagBehavior(config, world_model)
        self.actor = self._behavior.actor
        stoch_size = config.dyn_stoch
        if config.dyn_discrete:
            stoch_size *= config.dyn_discrete
        size = {'embed': 32 * config.cnn_depth, 'stoch': stoch_size, 'deter': config.dyn_deter, 'feat': config.dyn_stoch + config.dyn_deter}[self._config.disag_target]
        kw = dict(inp_dim=config.dyn_stoch, shape=size, layers=config.disag_layers, units=config.disag_units, act=config.act)
        self._networks = [networks.DenseHead(**kw) for _ in range(config.disag_models)]
        self._opt = tools.optimizer(config.opt, self.parameters(), config.model_lr, config.opt_eps, config.weight_decay)

    def train(self, start, context, data):
        metrics = {}
        stoch = start['stoch']
        if self._config.dyn_discrete:
            stoch = tf.reshape(stoch, stoch.shape[:-2] + stoch.shape[-2] * stoch.shape[-1])
        target = {'embed': context['embed'], 'stoch': stoch, 'deter': start['deter'], 'feat': context['feat']}[self._config.disag_target]
        inputs = context['feat']
        if self._config.disag_action_cond:
            inputs = tf.concat([inputs, data['action']], -1)
        metrics.update(self._train_ensemble(inputs, target))
        metrics.update(self._behavior.train(start, self._intrinsic_reward)[-1])
        return None, metrics

    def _intrinsic_reward(self, feat, state, action):
        inputs = feat
        if self._config.disag_action_cond:
            inputs = tf.concat([inputs, action], -1)
        preds = [head(inputs, tf.float32).mean() for head in self._networks]
        disag = tf.reduce_mean(tf.math.reduce_std(preds, 0), -1)
        if self._config.disag_log:
            disag = tf.math.log(disag)
        reward = self._config.expl_intr_scale * disag
        if self._config.expl_extr_scale:
            reward += tf.cast(self._config.expl_extr_scale * self._reward(feat, state, action), tf.float32)
        return reward

    def _train_ensemble(self, inputs, targets):
        if self._config.disag_offset:
            targets = targets[:, self._config.disag_offset:]
            inputs = inputs[:, :-self._config.disag_offset]
        targets = tf.stop_gradient(targets)
        inputs = tf.stop_gradient(inputs)
        with tf.GradientTape() as tape:
            preds = [head(inputs) for head in self._networks]
            likes = [tf.reduce_mean(pred.log_prob(targets)) for pred in preds]
            loss = -tf.cast(tf.reduce_sum(likes), tf.float32)
        metrics = self._opt(tape, loss, self._networks)
        return metrics


class WorldModel(nn.Module):

    def __init__(self, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self.encoder = networks.ConvEncoder(config.grayscale, config.cnn_depth, config.act, config.encoder_kernels)
        if config.size[0] == 64 and config.size[1] == 64:
            embed_size = 2 ** (len(config.encoder_kernels) - 1) * config.cnn_depth
            embed_size *= 2 * 2
        else:
            raise NotImplemented(f'{config.size} is not applicable now')
        self.dynamics = networks.RSSM(config.dyn_stoch, config.dyn_deter, config.dyn_hidden, config.dyn_input_layers, config.dyn_output_layers, config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete, config.act, config.dyn_mean_act, config.dyn_std_act, config.dyn_temp_post, config.dyn_min_std, config.dyn_cell, config.num_actions, embed_size, config.device)
        self.heads = nn.ModuleDict()
        channels = 1 if config.grayscale else 3
        shape = (channels,) + config.size
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads['image'] = networks.ConvDecoder(feat_size, config.cnn_depth, config.act, shape, config.decoder_kernels, config.decoder_thin)
        self.heads['reward'] = networks.DenseHead(feat_size, [], config.reward_layers, config.units, config.act)
        if config.pred_discount:
            self.heads['discount'] = networks.DenseHead(feat_size, [], config.discount_layers, config.units, config.act, dist='binary')
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer('model', self.parameters(), config.model_lr, config.opt_eps, config.grad_clip, config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._scales = dict(reward=config.reward_scale, discount=config.discount_scale)

    def _train(self, data):
        data = self.preprocess(data)
        with tools.RequiresGrad(self):
            with torch.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(embed, data['action'])
                kl_balance = tools.schedule(self._config.kl_balance, self._step)
                kl_free = tools.schedule(self._config.kl_free, self._step)
                kl_scale = tools.schedule(self._config.kl_scale, self._step)
                kl_loss, kl_value = self.dynamics.kl_loss(post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)
                losses = {}
                likes = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    like = pred.log_prob(data[name])
                    likes[name] = like
                    losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
                model_loss = sum(losses.values()) + kl_loss
            metrics = self._model_opt(model_loss, self.parameters())
        metrics.update({f'{name}_loss': to_np(loss) for name, loss in losses.items()})
        metrics['kl_balance'] = kl_balance
        metrics['kl_free'] = kl_free
        metrics['kl_scale'] = kl_scale
        metrics['kl'] = to_np(torch.mean(kl_value))
        with torch.amp.autocast(self._use_amp):
            metrics['prior_ent'] = to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
            metrics['post_ent'] = to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
            context = dict(embed=embed, feat=self.dynamics.get_feat(post), kl=kl_value, postent=self.dynamics.get_dist(post).entropy())
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    def preprocess(self, obs):
        obs = obs.copy()
        obs['image'] = torch.Tensor(obs['image']) / 255.0 - 0.5
        if self._config.clip_rewards == 'tanh':
            obs['reward'] = torch.tanh(torch.Tensor(obs['reward'])).unsqueeze(-1)
        elif self._config.clip_rewards == 'identity':
            obs['reward'] = torch.Tensor(obs['reward']).unsqueeze(-1)
        else:
            raise NotImplemented(f'{self._config.clip_rewards} is not implemented')
        if 'discount' in obs:
            obs['discount'] *= self._config.discount
            obs['discount'] = torch.Tensor(obs['discount']).unsqueeze(-1)
        obs = {k: torch.Tensor(v) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        truth = data['image'][:6] + 0.5
        embed = self.encoder(data)
        states, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])
        recon = self.heads['image'](self.dynamics.get_feat(states)).mode()[:6]
        reward_post = self.heads['reward'](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine(data['action'][:6, 5:], init)
        openl = self.heads['image'](self.dynamics.get_feat(prior)).mode()
        reward_prior = self.heads['reward'](self.dynamics.get_feat(prior)).mode()
        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):

    def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._reward = reward
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.ActionHead(feat_size, config.num_actions, config.actor_layers, config.units, config.act, config.actor_dist, config.actor_init_std, config.actor_min_std, config.actor_dist, config.actor_temp, config.actor_outscale)
        self.value = networks.DenseHead(feat_size, [], config.value_layers, config.units, config.act, config.value_head)
        if config.slow_value_target or config.slow_actor_target:
            self._slow_value = networks.DenseHead(feat_size, [], config.value_layers, config.units, config.act)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer('actor', self.actor.parameters(), config.actor_lr, config.opt_eps, config.actor_grad_clip, **kw)
        self._value_opt = tools.Optimizer('value', self.value.parameters(), config.value_lr, config.opt_eps, config.value_grad_clip, **kw)

    def _train(self, start, objective=None, action=None, reward=None, imagine=None, tape=None, repeats=None):
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}
        with tools.RequiresGrad(self.actor):
            with torch.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(start, self.actor, self._config.imag_horizon, repeats)
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                target, weights = self._compute_target(imag_feat, imag_state, imag_action, reward, actor_ent, state_ent, self._config.slow_actor_target)
                actor_loss, mets = self._compute_actor_loss(imag_feat, imag_state, imag_action, target, actor_ent, state_ent, weights)
                metrics.update(mets)
                if self._config.slow_value_target != self._config.slow_actor_target:
                    target, weights = self._compute_target(imag_feat, imag_state, imag_action, reward, actor_ent, state_ent, self._config.slow_value_target)
                value_input = imag_feat
        with tools.RequiresGrad(self.value):
            with torch.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                value_loss = -value.log_prob(target.detach())
                if self._config.value_decay:
                    value_loss += self._config.value_decay * value.mode()
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
        metrics['reward_mean'] = to_np(torch.mean(reward))
        metrics['reward_std'] = to_np(torch.std(reward))
        metrics['actor_ent'] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon, repeats=None):
        dynamics = self._world_model.dynamics
        if repeats:
            raise NotImplemented('repeats is not implemented in this version')
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action
        feat = 0 * dynamics.get_feat(start)
        action = policy(feat).mode()
        succ, feats, actions = tools.static_scan(step, [torch.arange(horizon)], (start, feat, action))
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            raise NotImplemented('repeats is not implemented in this version')
        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent, slow):
        if 'discount' in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._world_model.heads['discount'](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        if self._config.future_entropy and self._config.actor_entropy() > 0:
            reward += self._config.actor_entropy() * actor_ent
        if self._config.future_entropy and self._config.actor_state_entropy() > 0:
            reward += self._config.actor_state_entropy() * state_ent
        if slow:
            value = self._slow_value(imag_feat).mode()
        else:
            value = self.value(imag_feat).mode()
        target = tools.lambda_return(reward[:-1], value[:-1], discount[:-1], bootstrap=value[-1], lambda_=self._config.discount_lambda, axis=0)
        weights = torch.cumprod(torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
        return target, weights

    def _compute_actor_loss(self, imag_feat, imag_state, imag_action, target, actor_ent, state_ent, weights):
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        policy = self.actor(inp)
        actor_ent = policy.entropy()
        target = torch.stack(target, dim=1)
        if self._config.imag_gradient == 'dynamics':
            actor_target = target
        elif self._config.imag_gradient == 'reinforce':
            actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (target - self.value(imag_feat[:-1]).mode()).detach()
        elif self._config.imag_gradient == 'both':
            actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (target - self.value(imag_feat[:-1]).mode()).detach()
            mix = self._config.imag_gradient_mix()
            actor_target = mix * target + (1 - mix) * actor_target
            metrics['imag_gradient_mix'] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        if not self._config.future_entropy and self._config.actor_entropy() > 0:
            actor_target += self._config.actor_entropy() * actor_ent[:-1][:, :, None]
        if not self._config.future_entropy and self._config.actor_state_entropy() > 0:
            actor_target += self._config.actor_state_entropy() * state_ent[:-1]
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.slow_value_target or self._config.slow_actor_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1


class GRUCell(nn.Module):

    def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + size, 3 * size, bias=norm is not None)
        if norm:
            self._norm = nn.LayerNorm(3 * size)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class RSSM(nn.Module):

    def __init__(self, stoch=30, deter=200, hidden=200, layers_input=1, layers_output=1, rec_depth=1, shared=False, discrete=False, act=nn.ELU, mean_act='none', std_act='softplus', temp_post=True, min_std=0.1, cell='gru', num_actions=None, embed=None, device=None):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._rec_depth = rec_depth
        self._shared = shared
        self._discrete = discrete
        self._act = act
        self._mean_act = mean_act
        self._std_act = std_act
        self._temp_post = temp_post
        self._embed = embed
        self._device = device
        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        if self._shared:
            inp_dim += self._embed
        for i in range(self._layers_input):
            inp_layers.append(nn.Linear(inp_dim, self._hidden))
            inp_layers.append(self._act())
            if i == 0:
                inp_dim = self._hidden
        self._inp_layers = nn.Sequential(*inp_layers)
        if cell == 'gru':
            self._cell = GRUCell(self._hidden, self._deter)
        elif cell == 'gru_layer_norm':
            self._cell = GRUCell(self._hidden, self._deter, norm=True)
        else:
            raise NotImplementedError(cell)
        img_out_layers = []
        inp_dim = self._deter
        for i in range(self._layers_output):
            img_out_layers.append(nn.Linear(inp_dim, self._hidden))
            img_out_layers.append(self._act())
            if i == 0:
                inp_dim = self._hidden
        self._img_out_layers = nn.Sequential(*img_out_layers)
        obs_out_layers = []
        if self._temp_post:
            inp_dim = self._deter + self._embed
        else:
            inp_dim = self._embed
        for i in range(self._layers_output):
            obs_out_layers.append(nn.Linear(inp_dim, self._hidden))
            obs_out_layers.append(self._act())
            if i == 0:
                inp_dim = self._hidden
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        if self._discrete:
            self._ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
        else:
            self._ims_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter)
        if self._discrete:
            state = dict(logit=torch.zeros([batch_size, self._stoch, self._discrete]), stoch=torch.zeros([batch_size, self._stoch, self._discrete]), deter=deter)
        else:
            state = dict(mean=torch.zeros([batch_size, self._stoch]), std=torch.zeros([batch_size, self._stoch]), stoch=torch.zeros([batch_size, self._stoch]), deter=deter)
        return state

    def observe(self, embed, action, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        embed, action = swap(embed), swap(action)
        post, prior = tools.static_scan(lambda prev_state, prev_act, embed: self.obs_step(prev_state[0], prev_act, embed), (action, embed), (state, state))
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state['stoch']
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state['deter']], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state['logit']
            dist = torchd.independent.Independent(tools.OneHotDist(logit), 1)
        else:
            mean, std = state['mean'], state['std']
            dist = tools.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), 1))
        return dist

    def obs_step(self, prev_state, prev_action, embed, sample=True):
        prior = self.img_step(prev_state, prev_action, None, sample)
        if self._shared:
            post = self.img_step(prev_state, prev_action, embed, sample)
        else:
            if self._temp_post:
                x = torch.cat([prior['deter'], embed], -1)
            else:
                x = embed
            x = self._obs_out_layers(x)
            stats = self._suff_stats_layer('obs', x)
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mode()
            post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, embed=None, sample=True):
        prev_stoch = prev_state['stoch']
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            prev_stoch = prev_stoch.reshape(shape)
        if self._shared:
            if embed is None:
                shape = list(prev_action.shape[:-1]) + [self._embed]
                embed = torch.zeros(shape)
            x = torch.cat([prev_stoch, prev_action, embed], -1)
        else:
            x = torch.cat([prev_stoch, prev_action], -1)
        x = self._inp_layers(x)
        for _ in range(self._rec_depth):
            deter = prev_state['deter']
            x, deter = self._cell(x, [deter])
            deter = deter[0]
        x = self._img_out_layers(x)
        stats = self._suff_stats_layer('ims', x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {'stoch': stoch, 'deter': deter, **stats}
        return prior

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == 'ims':
                x = self._ims_stat_layer(x)
            elif name == 'obs':
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {'logit': logit}
        else:
            if name == 'ims':
                x = self._ims_stat_layer(x)
            elif name == 'obs':
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {'none': lambda : mean, 'tanh5': lambda : 5.0 * torch.tanh(mean / 5.0)}[self._mean_act]()
            std = {'softplus': lambda : torch.softplus(std), 'abs': lambda : torch.abs(std + 1), 'sigmoid': lambda : torch.sigmoid(std), 'sigmoid2': lambda : 2 * torch.sigmoid(std / 2)}[self._std_act]()
            std = std + self._min_std
            return {'mean': mean, 'std': std}

    def kl_loss(self, post, prior, forward, balance, free, scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else 1 - balance
        if balance == 0.5:
            value = kld(dist(lhs) if self._discrete else dist(lhs)._dist, dist(rhs) if self._discrete else dist(rhs)._dist)
            loss = torch.mean(torch.maximum(value, free))
        else:
            value_lhs = value = kld(dist(lhs) if self._discrete else dist(lhs)._dist, dist(sg(rhs)) if self._discrete else dist(sg(rhs))._dist)
            value_rhs = kld(dist(sg(lhs)) if self._discrete else dist(sg(lhs))._dist, dist(rhs) if self._discrete else dist(rhs)._dist)
            loss_lhs = torch.maximum(torch.mean(value_lhs), torch.Tensor([free])[0])
            loss_rhs = torch.maximum(torch.mean(value_rhs), torch.Tensor([free])[0])
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        loss *= scale
        return loss, value


class ConvEncoder(nn.Module):

    def __init__(self, grayscale=False, depth=32, act=nn.ReLU, kernels=(4, 4, 4, 4)):
        super(ConvEncoder, self).__init__()
        self._act = act
        self._depth = depth
        self._kernels = kernels
        layers = []
        for i, kernel in enumerate(self._kernels):
            if i == 0:
                if grayscale:
                    inp_dim = 1
                else:
                    inp_dim = 3
            else:
                inp_dim = 2 ** (i - 1) * self._depth
            depth = 2 ** i * self._depth
            layers.append(nn.Conv2d(inp_dim, depth, kernel, 2))
            layers.append(act())
        self.layers = nn.Sequential(*layers)

    def __call__(self, obs):
        x = obs['image'].reshape((-1,) + tuple(obs['image'].shape[-3:]))
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        shape = list(obs['image'].shape[:-3]) + [x.shape[-1]]
        return x.reshape(shape)


class ConvDecoder(nn.Module):

    def __init__(self, inp_depth, depth=32, act=nn.ReLU, shape=(3, 64, 64), kernels=(5, 5, 6, 6), thin=True):
        super(ConvDecoder, self).__init__()
        self._inp_depth = inp_depth
        self._act = act
        self._depth = depth
        self._shape = shape
        self._kernels = kernels
        self._thin = thin
        if self._thin:
            self._linear_layer = nn.Linear(inp_depth, 32 * self._depth)
        else:
            self._linear_layer = nn.Linear(inp_depth, 128 * self._depth)
        inp_dim = 32 * self._depth
        cnnt_layers = []
        for i, kernel in enumerate(self._kernels):
            depth = 2 ** (len(self._kernels) - i - 2) * self._depth
            act = self._act
            if i == len(self._kernels) - 1:
                depth = self._shape[0]
                act = None
            if i != 0:
                inp_dim = 2 ** (len(self._kernels) - (i - 1) - 2) * self._depth
            cnnt_layers.append(nn.ConvTranspose2d(inp_dim, depth, kernel, 2))
            if act is not None:
                cnnt_layers.append(act())
        self._cnnt_layers = nn.Sequential(*cnnt_layers)

    def __call__(self, features, dtype=None):
        if self._thin:
            x = self._linear_layer(features)
            x = x.reshape([-1, 1, 1, 32 * self._depth])
            x = x.permute(0, 3, 1, 2)
        else:
            x = self._linear_layer(features)
            x = x.reshape([-1, 2, 2, 32 * self._depth])
            x = x.permute(0, 3, 1, 2)
        x = self._cnnt_layers(x)
        mean = x.reshape(features.shape[:-1] + self._shape)
        mean = mean.permute(0, 1, 3, 4, 2)
        return tools.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, 1), len(self._shape)))


class DenseHead(nn.Module):

    def __init__(self, inp_dim, shape, layers, units, act=nn.ELU, dist='normal', std=1.0):
        super(DenseHead, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if len(self._shape) == 0:
            self._shape = 1,
        self._layers = layers
        self._units = units
        self._act = act
        self._dist = dist
        self._std = std
        mean_layers = []
        for index in range(self._layers):
            mean_layers.append(nn.Linear(inp_dim, self._units))
            mean_layers.append(act())
            if index == 0:
                inp_dim = self._units
        mean_layers.append(nn.Linear(inp_dim, np.prod(self._shape)))
        self._mean_layers = nn.Sequential(*mean_layers)
        if self._std == 'learned':
            self._std_layer = nn.Linear(self._units, np.prod(self._shape))

    def __call__(self, features, dtype=None):
        x = features
        mean = self._mean_layers(x)
        if self._std == 'learned':
            std = self._std_layer(x)
            std = torch.softplus(std) + 0.01
        else:
            std = self._std
        if self._dist == 'normal':
            return tools.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), len(self._shape)))
        if self._dist == 'huber':
            return tools.ContDist(torchd.independent.Independent(tools.UnnormalizedHuber(mean, std, 1.0), len(self._shape)))
        if self._dist == 'binary':
            return tools.Bernoulli(torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=mean), len(self._shape)))
        raise NotImplementedError(self._dist)


class ActionHead(nn.Module):

    def __init__(self, inp_dim, size, layers, units, act=nn.ELU, dist='trunc_normal', init_std=0.0, min_std=0.1, action_disc=5, temp=0.1, outscale=0):
        super(ActionHead, self).__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        self._min_std = min_std
        self._init_std = init_std
        self._action_disc = action_disc
        self._temp = temp() if callable(temp) else temp
        self._outscale = outscale
        pre_layers = []
        for index in range(self._layers):
            pre_layers.append(nn.Linear(inp_dim, self._units))
            pre_layers.append(act())
            if index == 0:
                inp_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)
        if self._dist in ['tanh_normal', 'tanh_normal_5', 'normal', 'trunc_normal']:
            self._dist_layer = nn.Linear(self._units, 2 * self._size)
        elif self._dist in ['normal_1', 'onehot', 'onehot_gumbel']:
            self._dist_layer = nn.Linear(self._units, self._size)

    def __call__(self, features, dtype=None):
        x = features
        x = self._pre_layers(x)
        if self._dist == 'tanh_normal':
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(dist, tools.TanhBijector())
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == 'tanh_normal_5':
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(dist, tools.TanhBijector())
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == 'normal':
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == 'normal_1':
            x = self._dist_layer(x)
            dist = torchd.normal.Normal(mean, 1)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == 'trunc_normal':
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == 'onehot':
            x = self._dist_layer(x)
            dist = tools.OneHotDist(x)
        elif self._dist == 'onehot_gumble':
            x = self._dist_layer(x)
            temp = self._temp
            dist = tools.ContDist(torchd.gumbel.Gumbel(x, 1 / temp))
        else:
            raise NotImplementedError(self._dist)
        return dist

