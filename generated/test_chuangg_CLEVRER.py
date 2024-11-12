
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


import time


import random


import numpy as np


import scipy.spatial as spatial


from sklearn.cluster import MiniBatchKMeans


import matplotlib.pyplot as plt


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.autograd import Variable


import copy


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


from torch.nn import init


from torchvision import models


from torch.nn import functional as F


import torch.multiprocessing as mp


from torch.optim.lr_scheduler import ReduceLROnPlateau


import matplotlib as mpl


from matplotlib.collections import PatchCollection


from matplotlib.patches import Rectangle


from matplotlib.patches import Circle


class RelationEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RelationEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_size * 2, hidden_size * 3, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_size * 3, output_size, kernel_size=3, stride=1, padding=0)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """
        args:
            x: [n_relations, input_size]
        returns:
            [n_relations, output_size]
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x.view(x.size(0), -1)


class ParticleEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ParticleEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_size * 2, hidden_size * 3, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_size * 3, output_size, kernel_size=3, stride=1, padding=0)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """
        args:
            x: [n_particles, input_size]
        returns:
            [n_particles, output_size]
        """
        x_1 = self.relu(self.conv1(x))
        x_2 = self.relu(self.conv2(x_1))
        x_3 = self.relu(self.conv3(x_2))
        x_4 = self.relu(self.conv4(x_3))
        return x_1, x_2, x_3, x_4


class Propagator(nn.Module):

    def __init__(self, input_size, output_size, residual=False):
        super(Propagator, self).__init__()
        self.residual = residual
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        """
        Args:
            x: [n_relations/n_particles, input_size]
        Returns:
            [n_relations/n_particles, output_size]
        """
        if self.residual:
            x = self.relu(self.linear(x) + res)
        else:
            x = self.relu(self.linear(x))
        return x


class ParticlePredictor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()
        self.convt1 = nn.ConvTranspose2d(input_size * 2, hidden_size * 3, kernel_size=3, stride=1, padding=0)
        self.convt2 = nn.ConvTranspose2d(hidden_size * 3 * 2, hidden_size * 2, kernel_size=4, stride=2, padding=1)
        self.convt3 = nn.ConvTranspose2d(hidden_size * 2 * 2, hidden_size * 1, kernel_size=4, stride=2, padding=1)
        self.convt4 = nn.ConvTranspose2d(hidden_size * 1 * 2, output_size, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU()

    def forward(self, x, x_encode):
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        x = self.relu(self.convt1(torch.cat([x, x_encode[3]], 1)))
        x = self.relu(self.convt2(torch.cat([x, x_encode[2]], 1)))
        x = self.relu(self.convt3(torch.cat([x, x_encode[1]], 1)))
        x = self.convt4(torch.cat([x, x_encode[0]], 1))
        return x


class RelationPredictor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RelationPredictor, self).__init__()
        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        x = self.relu(self.linear_0(x))
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


class PropagationNetwork(nn.Module):

    def __init__(self, args, residual=False, use_gpu=False):
        super(PropagationNetwork, self).__init__()
        self.args = args
        input_dim = args.state_dim * (args.n_his + 1)
        relation_dim = args.relation_dim * (args.n_his + 1)
        output_dim = args.state_dim
        nf_particle = args.nf_particle
        nf_relation = args.nf_relation
        nf_effect = args.nf_effect
        self.nf_effect = args.nf_effect
        self.use_attr = args.use_attr
        self.use_gpu = use_gpu
        self.residual = residual
        if args.use_attr:
            self.particle_encoder = ParticleEncoder(input_dim + args.attr_dim, nf_particle, nf_effect)
        else:
            self.particle_encoder = ParticleEncoder(input_dim, nf_particle, nf_effect)
        if args.use_attr:
            self.relation_encoder = RelationEncoder(2 * input_dim + 2 * args.attr_dim + relation_dim, nf_relation, nf_effect)
        else:
            self.relation_encoder = RelationEncoder(2 * input_dim + relation_dim, nf_relation, nf_effect)
        self.relation_propagator = Propagator(3 * nf_effect, nf_effect)
        self.particle_propagator = Propagator(2 * nf_effect, nf_effect, self.residual)
        self.particle_predictor = ParticlePredictor(nf_effect, nf_particle, output_dim)
        self.relation_predictor = RelationPredictor(nf_effect, nf_effect, 1)

    def forward(self, attr, state, Rr, Rs, Ra, node_r_idx, node_s_idx, pstep, ret_feat=False):
        if self.use_gpu:
            particle_effect = Variable(torch.zeros((state.size(0), self.nf_effect)))
        else:
            particle_effect = Variable(torch.zeros((state.size(0), self.nf_effect)))
        Rrp = Rr.t()
        Rsp = Rs.t()
        n_relation_r, n_object_r = Rrp.size(0), Rrp.size(1)
        n_relation_s, n_object_s = Rsp.size(0), Rsp.size(1)
        n_attr, n_state, bbox_h, bbox_w = attr.size(1), state.size(1), state.size(2), state.size(3)
        attr_r = attr[node_r_idx]
        attr_s = attr[node_s_idx]
        attr_r_rel = torch.mm(Rrp, attr_r.view(n_object_r, -1)).view(n_relation_r, n_attr, bbox_h, bbox_w)
        attr_s_rel = torch.mm(Rsp, attr_s.view(n_object_s, -1)).view(n_relation_s, n_attr, bbox_h, bbox_w)
        state_r = state[node_r_idx]
        state_s = state[node_s_idx]
        state_r_rel = torch.mm(Rrp, state_r.view(n_object_r, -1)).view(n_relation_r, n_state, bbox_h, bbox_w)
        state_s_rel = torch.mm(Rsp, state_s.view(n_object_s, -1)).view(n_relation_s, n_state, bbox_h, bbox_w)
        if self.use_attr:
            particle_encode = self.particle_encoder(torch.cat([attr_r, state_r], 1))
        else:
            particle_encode = self.particle_encoder(state_r)
        if self.use_attr:
            relation_encode = self.relation_encoder(torch.cat([attr_r_rel, attr_s_rel, state_r_rel, state_s_rel, Ra], 1))
        else:
            relation_encode = self.relation_encoder(torch.cat([state_r_rel, state_s_rel, Ra], 1))
        for i in range(pstep):
            effect_p_r = particle_effect[node_r_idx]
            effect_p_s = particle_effect[node_s_idx]
            receiver_effect = Rrp.mm(effect_p_r)
            sender_effect = Rsp.mm(effect_p_s)
            effect_rel = self.relation_propagator(torch.cat([relation_encode, receiver_effect, sender_effect], 1))
            effect_p_r_agg = Rr.mm(effect_rel)
            effect_p = self.particle_propagator(torch.cat([particle_encode[-1].view(particle_encode[-1].size(0), -1), effect_p_r_agg], 1), res=effect_p_r)
            particle_effect[node_r_idx] = effect_p
        pred_obj = self.particle_predictor(particle_effect.view(particle_effect.size(0), particle_effect.size(1), 1, 1), particle_encode)
        pred_obj[:, 1] = torch.mean(pred_obj[:, 1].view(pred_obj.size(0), -1), 1).view(pred_obj.size(0), 1, 1)
        pred_obj[:, 2] = torch.mean(pred_obj[:, 2].view(pred_obj.size(0), -1), 1).view(pred_obj.size(0), 1, 1)
        pred_rel = self.relation_predictor(effect_rel)
        if ret_feat:
            return pred_obj, pred_rel, particle_effect
        else:
            return pred_obj, pred_rel


class ChamferLoss(torch.nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()

    def chamfer_distance(self, x, y):
        x = x.repeat(y.size(0), 1, 1)
        x = x.transpose(0, 1)
        y = y.repeat(x.size(0), 1, 1)
        dis = torch.norm(torch.add(x, -y), 2, dim=2)
        dis_xy = torch.mean(torch.min(dis, dim=1)[0])
        dis_yx = torch.mean(torch.min(dis, dim=0)[0])
        return dis_xy + dis_yx

    def __call__(self, pred, label):
        return self.chamfer_distance(pred, label)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ParticleEncoder,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (Propagator,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RelationEncoder,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (RelationPredictor,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

