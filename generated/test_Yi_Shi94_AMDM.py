
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


import numpy as np


import torch


import torch.optim as optim


import torch.utils.data as data


import math


import copy


from numpy import dot


from numpy.linalg import norm


import scipy.ndimage.filters as filters


from torch.utils import data


import random


from torch.utils.data._utils.collate import default_collate


from torch.utils.data import DataLoader


from typing import Tuple


from torch.utils.data import Dataset


import torch.nn.functional as F


import time


from torch.nn.utils import clip_grad_norm_


from collections import OrderedDict


import torch.nn as nn


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import matplotlib.pyplot as plt


import matplotlib.animation as animation


import matplotlib.cm as cm


import torch.nn.functional as f


from functools import partial


from copy import deepcopy


import abc


from abc import ABC


from abc import abstractmethod


from random import random


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import SubsetRandomSampler


import itertools


import matplotlib.cm as mpl_color


import functools


from typing import Optional


import warnings


import torch.onnx


from scipy.spatial.transform import Rotation as R


import matplotlib.colors as mcolors


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        return self.pe[pos]


def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MovementConvEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(nn.Conv1d(input_size, hidden_size, 4, 2, 1), nn.Dropout(0.2, inplace=True), nn.LeakyReLU(0.2, inplace=True), nn.Conv1d(hidden_size, output_size, 4, 2, 1), nn.Dropout(0.2, inplace=True), nn.LeakyReLU(0.2, inplace=True))
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class MovementConvDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvDecoder, self).__init__()
        self.main = nn.Sequential(nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class TextVAEDecoder(nn.Module):

    def __init__(self, text_size, input_size, output_size, hidden_size, n_layers):
        super(TextVAEDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2, inplace=True))
        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.positional_encoder = PositionalEncoding(hidden_size)
        self.output = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2, inplace=True), nn.Linear(hidden_size, output_size))
        self.output.apply(init_weight)
        self.emb.apply(init_weight)
        self.z2init.apply(init_weight)

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    def forward(self, inputs, last_pred, hidden, p):
        h_in = self.emb(inputs)
        pos_enc = self.positional_encoder(p).detach()
        h_in = h_in + pos_enc
        for i in range(self.n_layers):
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]
        pose_pred = self.output(h_in)
        return pose_pred, hidden


def reparameterize(mu, logvar):
    s_var = logvar.mul(0.5).exp_()
    eps = s_var.data.new(s_var.size()).normal_()
    return eps.mul(s_var).add_(mu)


class TextDecoder(nn.Module):

    def __init__(self, text_size, input_size, output_size, hidden_size, n_layers):
        super(TextDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2, inplace=True))
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.positional_encoder = PositionalEncoding(hidden_size)
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.emb.apply(init_weight)
        self.z2init.apply(init_weight)
        self.mu_net.apply(init_weight)
        self.logvar_net.apply(init_weight)

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    def forward(self, inputs, hidden, p):
        x_in = self.emb(inputs)
        pos_enc = self.positional_encoder(p).detach()
        x_in = x_in + pos_enc
        for i in range(self.n_layers):
            hidden[i] = self.gru[i](x_in, hidden[i])
            h_in = hidden[i]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = reparameterize(mu, logvar)
        return z, mu, logvar, hidden


class AttLayer(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim):
        super(AttLayer, self).__init__()
        self.W_q = nn.Linear(query_dim, value_dim)
        self.W_k = nn.Linear(key_dim, value_dim, bias=False)
        self.W_v = nn.Linear(key_dim, value_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dim = value_dim
        self.W_q.apply(init_weight)
        self.W_k.apply(init_weight)
        self.W_v.apply(init_weight)

    def forward(self, query, key_mat):
        """
        query (batch, query_dim)
        key (batch, seq_len, key_dim)
        """
        query_vec = self.W_q(query).unsqueeze(-1)
        val_set = self.W_v(key_mat)
        key_set = self.W_k(key_mat)
        weights = torch.matmul(key_set, query_vec) / np.sqrt(self.dim)
        co_weights = self.softmax(weights)
        values = val_set * co_weights
        pred = values.sum(dim=1)
        return pred, co_weights

    def short_cut(self, querys, keys):
        return self.W_q(querys), self.W_k(keys)


class TextEncoderBiGRU(nn.Module):

    def __init__(self, word_size, pos_size, hidden_size, device):
        super(TextEncoderBiGRU, self).__init__()
        self.device = device
        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]
        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)
        gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        gru_seq = pad_packed_sequence(gru_seq, batch_first=True)[0]
        forward_seq = gru_seq[..., :self.hidden_size]
        backward_seq = gru_seq[..., self.hidden_size:].clone()
        for i, length in enumerate(cap_lens):
            backward_seq[i:i + 1, :length] = torch.flip(backward_seq[i:i + 1, :length].clone(), dims=[1])
        gru_seq = torch.cat([forward_seq, backward_seq], dim=-1)
        return gru_seq, gru_last


class TextEncoderBiGRUCo(nn.Module):

    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device
        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2, inplace=True), nn.Linear(hidden_size, output_size))
        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]
        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)
        gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output_net(gru_last)


class MotionEncoderBiGRUCo(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device
        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2, inplace=True), nn.Linear(hidden_size, output_size))
        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)
        gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output_net(gru_last)


class MotionLenEstimatorBiGRU(nn.Module):

    def __init__(self, word_size, pos_size, hidden_size, output_size):
        super(MotionLenEstimatorBiGRU, self).__init__()
        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        nd = 512
        self.output = nn.Sequential(nn.Linear(hidden_size * 2, nd), nn.LayerNorm(nd), nn.LeakyReLU(0.2, inplace=True), nn.Linear(nd, nd // 2), nn.LayerNorm(nd // 2), nn.LeakyReLU(0.2, inplace=True), nn.Linear(nd // 2, nd // 4), nn.LayerNorm(nd // 4), nn.LeakyReLU(0.2, inplace=True), nn.Linear(nd // 4, output_size))
        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]
        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)
        gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output(gru_last)


class NoiseDecoder(nn.Module):

    def __init__(self, frame_size, hidden_size, time_emb_size, layer_num, norm_type, act_type):
        super().__init__()
        self.input_size = frame_size
        layers = []
        for _ in range(layer_num):
            if act_type == 'ReLU':
                non_linear = torch.nn.ReLU()
            elif act_type == 'SiLU':
                non_linear = Activation.SiLU()
            linear = nn.Linear(hidden_size + frame_size * 2 + time_emb_size, hidden_size)
            if norm_type == 'layer_norm':
                norm_layer = nn.LayerNorm(hidden_size)
            elif norm_type == 'group_norm':
                norm_layer = nn.GroupNorm(16, hidden_size)
            layers.append(norm_layer)
            layers.extend([non_linear, linear])
        self.net = nn.ModuleList(layers)
        self.fin = nn.Linear(frame_size * 2 + time_emb_size, hidden_size)
        self.fco = nn.Linear(hidden_size + frame_size * 2 + time_emb_size, frame_size)
        self.act = Activation.SiLU()

    def forward(self, xcur, xnext, latent):
        x0 = xnext
        y0 = xcur
        x = torch.cat([xcur, xnext, latent], dim=-1)
        x = self.fin(x)
        for i, layer in enumerate(self.net):
            if i % 3 == 2:
                x = torch.cat([x, x0, y0, latent], dim=-1)
                x = layer(x)
            else:
                x = layer(x)
        x = torch.cat([x, x0, y0, latent], dim=-1)
        x = self.fco(x)
        return x


class GaussianDiffusion(nn.Module):
    __doc__ = """Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.
    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    """

    def __init__(self, config):
        super().__init__()
        self.T = config['diffusion']['T']
        self.schedule_mode = config['diffusion']['noise_schedule_mode']
        self.estimate_mode = config['diffusion']['estimate_mode']
        self.norm_type = config['model_hyperparam']['norm_type']
        self.act_type = config['model_hyperparam']['act_type']
        self.time_emb_dim = config['model_hyperparam']['time_emb_size']
        self.hidden_dim = config['model_hyperparam']['hidden_size']
        self.layer_num = config['model_hyperparam']['layer_num']
        self.frame_dim = config['frame_dim']
        self.model = NoiseDecoder(self.frame_dim, self.hidden_dim, self.time_emb_dim, self.layer_num, self.norm_type, self.act_type)
        self.time_mlp = torch.nn.Sequential(Embedding.PositionalEmbedding(self.time_emb_dim, 1.0), torch.nn.Linear(self.time_emb_dim, self.time_emb_dim), Activation.SiLU(), torch.nn.Linear(self.time_emb_dim, self.time_emb_dim))
        betas = self._generate_diffusion_schedule()
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('reciprocal_sqrt_alphas', to_torch(np.sqrt(1.0 / alphas)))
        self.register_buffer('reciprocal_sqrt_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer('reciprocal_sqrt_alphas_cumprod_m1', to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        self.register_buffer('remove_noise_coeff', to_torch(betas / np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('sigma', to_torch(np.sqrt(betas)))

    def _generate_diffusion_schedule(self, s=0.008):

        def f(t, T):
            return np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
        if self.schedule_mode == 'cosine':
            alphas = []
            f0 = f(0, self.T)
            for t in range(self.T + 1):
                alphas.append(f(t, self.T) / f0)
            betas = []
            for t in range(1, self.T + 1):
                betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
            return np.array(betas)
        elif self.schedule_mode == 'uniform':
            beta_start = 0.0001
            beta_end = 0.02
            return np.linspace(beta_start, beta_end, self.T)
        elif self.schedule_mode == 'quadratic':
            beta_start = 0.0001
            beta_end = 0.02
            return np.linspace(beta_start ** 0.5, beta_end ** 0.5, self.T) ** 2
        elif self.schedule_mode == 'sigmoid':
            beta_start = 0.0001
            beta_end = 0.02
            betas = np.linspace(-6, 6, self.T)
            return 1 / (1 + np.exp(-betas)) * (beta_end - beta_start) + beta_start
        else:
            assert False, 'Unsupported diffusion schedule: {}'.format(self.schedule_mode)

    @torch.no_grad()
    def extract(self, a, ts, x_shape):
        b, *_ = ts.shape
        out = a.gather(-1, ts)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    @torch.no_grad()
    def add_noise(self, x, ts):
        return x + self.extract(self.sigma, ts, x.shape) * torch.randn_like(x)

    def add_noise_w(self, x, ts, noise):
        return x + self.extract(self.sigma, ts, x.shape) * noise

    @torch.no_grad()
    def compute_alpha(self, beta, ts):
        beta = torch.cat([torch.zeros(1), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, ts + 1).view(-1, 1)
        return a

    @torch.no_grad()
    def remove_noise(self, xt, pred, ts):
        output = (xt - self.extract(self.remove_noise_coeff, ts, pred.shape) * pred) * self.extract(self.reciprocal_sqrt_alphas, ts, pred.shape)
        return output

    def get_x0_from_xt(self, xt, ts, noise):
        output = (xt - self.extract(self.sqrt_one_minus_alphas_cumprod, ts, xt.shape) * noise) * self.extract(self.reciprocal_sqrt_alphas_cumprod, ts, xt.shape)
        return output

    def get_eps_from_x0(self, xt, ts, pred_x0):
        return (xt * self.extract(self.reciprocal_sqrt_alphas_cumprod, ts, xt.shape) - pred_x0) / self.extract(self.reciprocal_sqrt_alphas_cumprod_m1, ts, xt.shape)

    def perturb_x(self, x, ts, noise):
        return self.extract(self.sqrt_alphas_cumprod, ts, x.shape) * x + self.extract(self.sqrt_one_minus_alphas_cumprod, ts, x.shape) * noise

    @torch.no_grad()
    def sample_ddpm(self, last_x, extra_info, record_process=False):
        x = torch.randn(last_x.shape[0], last_x.shape[-1])
        if record_process:
            x0s = torch.zeros(last_x.shape[0], self.T, last_x.shape[-1], device=last_x.device)
        for t in range(self.T - 1, -1, -1):
            ts = torch.tensor([t], device=last_x.device).repeat(last_x.shape[0])
            te = self.time_mlp(ts)
            pred = self.model(last_x, x, te).detach()
            if self.estimate_mode == 'epsilon':
                x = self.remove_noise(x, pred, ts)
            elif self.estimate_mode == 'x0':
                x = pred
            if record_process:
                x0s[:, self.T - 1 - t, :] = x
            if t > 0:
                x = self.add_noise(x, ts)
        if record_process:
            return x0s
        return x

    def sample_rl_ddpm(self, last_x, action_dict, extra_info):
        steps = extra_info['action_step']
        train_rand_scale = extra_info['rand_scale']
        test_rand_scale = extra_info['test_rand_scale']
        clip_scale = extra_info['clip_scale']
        action_mode = extra_info['action_mode']
        is_train = extra_info['is_train']
        action_scale = extra_info['action_scale'] if is_train else extra_info['test_action_scale']
        action_dim_per_step = 8 if action_mode == 'loco' else self.frame_dim
        x = action_dict[..., :action_dim_per_step] / 3
        for t in range(self.T - 1, -1, -1):
            with torch.no_grad():
                ts = torch.tensor([t], device=last_x.device).repeat(last_x.shape[0])
                te = self.time_mlp(ts)
                pred = self.model(last_x, x, te).detach()
                if self.estimate_mode == 'epsilon':
                    x = self.remove_noise(x, pred, ts)
                elif self.estimate_mode == 'x0':
                    x = pred
            if t in steps:
                i = steps.index(t) + 1
                dx = action_dict[..., i * action_dim_per_step:(i + 1) * action_dim_per_step]
                rand_scale = train_rand_scale if is_train else test_rand_scale
                rand_scale *= torch.randn_like(dx)
                x += action_scale * (dx + rand_scale * self.extract(self.sigma, ts, x.shape)[0])
                x = torch.clamp(x, -clip_scale, clip_scale)
            if t > 0:
                x = self.add_noise(x, ts)
        return x

    @torch.no_grad()
    def sample_ddpm_interactive(self, last_x, edited_mask, edited_data, extra_info):
        repaint_step = extra_info['repaint_step']
        interact_stop_step = extra_info['interact_stop_step']
        edited_mask_inv = 1 - edited_mask
        x = torch.randn(last_x.shape[0], last_x.shape[-1])
        for t in range(self.T - 1, -1, -1):
            for t_rp in range(repaint_step):
                ts = torch.tensor([t], device=last_x.device).repeat(last_x.shape[0])
                te = self.time_mlp(ts)
                pred = self.model(last_x, x, te).detach()
                if self.estimate_mode == 'epsilon':
                    x = self.remove_noise(x, pred, ts)
                elif self.estimate_mode == 'x0':
                    x = pred
                cur_edited_mask_inv = edited_mask_inv.clone()
                if t > interact_stop_step:
                    x = edited_data * edited_mask + x * cur_edited_mask_inv
                if t > 0:
                    x = self.add_noise(x, ts)
        return x

    @torch.no_grad()
    def sample_ddim(self, bs, num_steps, device, eta=0.0):
        T_train = self.T - 1
        timesteps = torch.linspace(0, T_train, num_steps, dtype=torch.long, device=device)
        timesteps_next = [-1] + list(timesteps[:-1])
        x = torch.randn(bs, self.action_dim, device=device)
        for t in range(len(timesteps) - 1, -1, -1):
            ts = torch.tensor([timesteps[t]], device=device).repeat(bs)
            ts1 = torch.tensor([timesteps_next[t]], device=device).repeat(bs)
            alpha_bar = self.extract(self.alphas_cumprod, ts, x.shape)
            alpha_bar_prev = self.extract(self.alphas_cumprod, ts1, x.shape) if t > 0 else torch.ones_like(x)
            sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)).sqrt()
            te = self.time_mlp(ts)
            input = torch.cat([x, te], axis=-1)
            output = self.model(input)
            if self.estimate_mode == 'x0':
                pred_x0 = output
                pred_eps = self.get_eps_from_x0(x, ts, pred_x0)
            else:
                pred_eps = output
                pred_x0 = self.get_x0_from_xt(x, ts, pred_eps)
            mean_pred = pred_x0 * self.extract(self.sqrt_alphas_cumprod, ts, x.shape) + (1 - alpha_bar_prev - sigma ** 2).sqrt() * pred_eps
            nonzero_mask = (ts != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            x = mean_pred + nonzero_mask * sigma * torch.randn_like(x)
        return x

    def forward(self, cur_x, next_x, ts, extra_info):
        bs = cur_x.shape[0]
        device = cur_x.device
        if ts is None:
            ts = torch.randint(0, self.T, (bs,), device=device)
        time_emb = self.time_mlp(ts)
        noise = torch.randn_like(next_x)
        perturbed_x = self.perturb_x(next_x, ts.clone(), noise)
        latent = time_emb
        estimated = self.model(cur_x, perturbed_x, latent)
        return estimated, noise, perturbed_x, ts


class BaseModel(torch.nn.Module):

    def __init__(self, config, dataset, device):
        super().__init__()
        self.config = config
        self.device = device
        self.joint_parent = dataset.joint_parent
        self.joint_offset = dataset.joint_offset
        return

    @abc.abstractmethod
    def _build_model(self, config):
        return

    @abc.abstractmethod
    def eval_step(self, cur_x, extra_dict):
        return

    @abc.abstractmethod
    def compute_loss(self, cur_x, tar_x, extra_dict):
        return


class SiLU(torch.nn.Module):

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class PositionalEmbedding(nn.Module):
    __doc__ = """Computes a positional embedding of timesteps.
    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionEmbedding(nn.Module):

    def __init__(self, num_actions, latent_dim, guidance_scale=7.5, guidance_uncodp=0.1, force_mask=False):
        super().__init__()
        self.nclasses = num_actions
        self.guidance_scale = guidance_scale
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))
        self.guidance_uncodp = guidance_uncodp
        self.force_mask = force_mask
        self._reset_parameters()

    def forward(self, input):
        idx = input.argmax(dim=1).long()
        output = self.action_embedding[idx]
        return output.float()

    def mask_cond(self, output, force=False):
        bs, d = output.shape
        if self.force_mask or force:
            return torch.zeros_like(output)
        elif self.training and self.guidance_uncodp > 0.0:
            mask = torch.bernoulli(torch.ones(bs, device=output.device) * self.guidance_uncodp).view(bs, 1)
            return output * (1.0 - mask)
        else:
            return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class AddBias(nn.Module):

    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


FixedNormal = torch.distributions.Normal


class DiagGaussian_adaptive(nn.Module):

    def __init__(self, num_outputs):
        super(DiagGaussian_adaptive, self).__init__()
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, action_mean):
        zeros = torch.zeros(action_mean.size())
        if action_mean.is_cuda:
            zeros = zeros
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, 0.3 * torch.ones_like(action_mean))


class DiagGaussian_fixed(nn.Module):

    def __init__(self, std):
        super(DiagGaussian_fixed, self).__init__()
        self.std = std

    def forward(self, action_mean):
        return FixedNormal(action_mean, self.std * torch.ones_like(action_mean))


class ActorNet(nn.Module):

    def __init__(self, env):
        super().__init__()
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        h_size = 256
        self.actor = nn.Sequential(nn.Linear(self.observation_dim, h_size), nn.ReLU(), nn.Linear(h_size, h_size), nn.ReLU(), nn.Linear(h_size, h_size), nn.ReLU(), nn.Linear(h_size, self.action_dim), nn.Tanh())

    def forward(self, x):
        return self.actor(x)


class CriticNet(nn.Module):

    def __init__(self, env):
        super().__init__()
        self.observation_dim = env.observation_space.shape[0]
        h_size = 256
        self.critic = nn.Sequential(nn.Linear(self.observation_dim, h_size), nn.ReLU(), nn.Linear(h_size, h_size), nn.ReLU(), nn.Linear(h_size, h_size), nn.ReLU(), nn.Linear(h_size, 1))

    def forward(self, x):
        return self.critic(x)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zero_(m.weight)
        torch.nn.init.zero_(m.bias)


class PPOModel(nn.Module):
    NAME = 'PPO'

    def __init__(self, config, env, device):
        super().__init__()
        self.actor = ActorNet(env)
        self.critic = CriticNet(env)
        init_weights(self.actor)
        self.distr_type = config['distr_type']
        self.std_value = config['distr_std']
        if self.distr_type == 'fixed':
            self.dist = DiagGaussian_fixed(self.std_value)
        elif self.distr_type == 'adaptive':
            self.dist = DiagGaussian_adaptive(self.actor.action_dim)
        self.state_size = 1

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        action = self.actor(inputs)
        dist = self.dist(action)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
            action.clamp_(-1.0, 1.0)
        action_log_probs = dist.log_probs(action)
        value = self.critic(inputs)
        return value, action, action_log_probs

    def get_value(self, inputs):
        value = self.critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value = self.critic(inputs)
        mode = self.actor(inputs)
        dist = self.dist(mode)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy


class SMPLifyAnglePrior(nn.Module):

    def __init__(self, dtype=torch.float32, **kwargs):
        super(SMPLifyAnglePrior, self).__init__()
        angle_prior_idxs = np.array([55, 58, 12, 15], dtype=np.int64)
        angle_prior_idxs = torch.tensor(angle_prior_idxs, dtype=torch.long)
        self.register_buffer('angle_prior_idxs', angle_prior_idxs)
        angle_prior_signs = np.array([1, -1, -1, -1], dtype=np.float32 if dtype == torch.float32 else np.float64)
        angle_prior_signs = torch.tensor(angle_prior_signs, dtype=dtype)
        self.register_buffer('angle_prior_signs', angle_prior_signs)

    def forward(self, pose, with_global_pose=False):
        """ Returns the angle prior loss for the given pose

        Args:
            pose: (Bx[23 + 1] * 3) torch tensor with the axis-angle
            representation of the rotations of the joints of the SMPL model.
        Kwargs:
            with_global_pose: Whether the pose vector also contains the global
            orientation of the SMPL model. If not then the indices must be
            corrected.
        Returns:
            A sze (B) tensor containing the angle prior loss for each element
            in the batch.
        """
        angle_prior_idxs = self.angle_prior_idxs - (not with_global_pose) * 3
        return torch.exp(pose[:, angle_prior_idxs] * self.angle_prior_signs).pow(2)


DEFAULT_DTYPE = torch.float32


class L2Prior(nn.Module):

    def __init__(self, dtype=DEFAULT_DTYPE, reduction='sum', **kwargs):
        super(L2Prior, self).__init__()

    def forward(self, module_input, *args):
        return torch.sum(module_input.pow(2))


class MaxMixturePrior(nn.Module):

    def __init__(self, prior_folder='prior', num_gaussians=6, dtype=DEFAULT_DTYPE, epsilon=1e-16, use_merged=True, **kwargs):
        super(MaxMixturePrior, self).__init__()
        if dtype == DEFAULT_DTYPE:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            None
            sys.exit(-1)
        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)
        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            None
            sys.exit(-1)
        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')
        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            None
            sys.exit(-1)
        self.register_buffer('means', torch.tensor(means, dtype=dtype))
        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))
        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)
        self.register_buffer('precisions', torch.tensor(precisions, dtype=dtype))
        sqrdets = np.array([np.sqrt(np.linalg.det(c)) for c in gmm['covars']])
        const = (2 * np.pi) ** (69 / 2.0)
        nll_weights = np.asarray(gmm['weights'] / (const * (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)
        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)
        self.register_buffer('pi_term', torch.log(torch.tensor(2 * np.pi, dtype=dtype)))
        cov_dets = [np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon) for cov in covs]
        self.register_buffer('cov_dets', torch.tensor(cov_dets, dtype=dtype))
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        """ Returns the mean of the mixture """
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose, betas):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means
        prec_diff_prod = torch.einsum('mij,bmj->bmi', [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)
        curr_loglikelihood = 0.5 * diff_prec_quadratic - torch.log(self.nll_weights)
        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose, betas, *args, **kwargs):
        """ Create graph operation for negative log-likelihood calculation
        """
        likelihoods = []
        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean
            curr_loglikelihood = torch.einsum('bj,ji->bi', [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b', [curr_loglikelihood, diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term + self.random_var_dim * self.pi_term)
            likelihoods.append(curr_loglikelihood)
        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)
        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, pose, betas):
        if self.use_merged:
            return self.merged_log_likelihood(pose, betas)
        else:
            return self.log_likelihood(pose, betas)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ActionEmbedding,
     lambda: ([], {'num_actions': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AttLayer,
     lambda: ([], {'query_dim': 4, 'key_dim': 4, 'value_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ContrastiveLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DiagGaussian_adaptive,
     lambda: ([], {'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DiagGaussian_fixed,
     lambda: ([], {'std': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (L2Prior,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MotionEncoderBiGRUCo,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {})),
    (MotionLenEstimatorBiGRU,
     lambda: ([], {'word_size': 4, 'pos_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {})),
    (MovementConvDecoder,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MovementConvEncoder,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.ones([4], dtype=torch.int64)], {})),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TextEncoderBiGRU,
     lambda: ([], {'word_size': 4, 'pos_size': 4, 'hidden_size': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {})),
    (TextEncoderBiGRUCo,
     lambda: ([], {'word_size': 4, 'pos_size': 4, 'hidden_size': 4, 'output_size': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {})),
]

