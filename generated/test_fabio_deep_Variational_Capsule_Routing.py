
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


import torch.nn.functional as F


import torchvision


import scipy.io as sio


from torchvision import transforms


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import WeightedRandomSampler


import warnings


import logging


import random


from torch.utils.tensorboard import SummaryWriter


import time


import copy


import torch.optim as optim


class ConvCapsules2d(nn.Module):
    """Convolutional Capsule Layer"""

    def __init__(self, in_caps, out_caps, pose_dim, kernel_size, stride, padding=0, weight_init='xavier_uniform', share_W_ij=False, coor_add=False):
        super().__init__()
        self.B = in_caps
        self.C = out_caps
        self.P = pose_dim
        self.PP = np.max([2, self.P * self.P])
        self.K = kernel_size
        self.S = stride
        self.padding = padding
        self.share_W_ij = share_W_ij
        self.coor_add = coor_add
        self.W_ij = torch.empty(1, self.B, self.C, 1, self.P, self.P, 1, 1, self.K, self.K)
        if weight_init.split('_')[0] == 'xavier':
            fan_in = self.B * self.K * self.K * self.PP
            fan_out = self.C * self.K * self.K * self.PP
            std = np.sqrt(2.0 / (fan_in + fan_out))
            bound = np.sqrt(3.0) * std
            if weight_init.split('_')[1] == 'normal':
                self.W_ij = nn.Parameter(self.W_ij.normal_(0, std))
            elif weight_init.split('_')[1] == 'uniform':
                self.W_ij = nn.Parameter(self.W_ij.uniform_(-bound, bound))
            else:
                raise NotImplementedError('{} not implemented.'.format(weight_init))
        elif weight_init.split('_')[0] == 'kaiming':
            fan_in = self.B * self.K * self.K * self.PP
            std = np.sqrt(2.0) / np.sqrt(fan_in)
            bound = np.sqrt(3.0) * std
            if weight_init.split('_')[1] == 'normal':
                self.W_ij = nn.Parameter(self.W_ij.normal_(0, std))
            elif weight_init.split('_')[1] == 'uniform':
                self.W_ij = nn.Parameter(self.W_ij.uniform_(-bound, bound))
            else:
                raise NotImplementedError('{} not implemented.'.format(weight_init))
        elif weight_init == 'noisy_identity' and self.PP > 2:
            b = 0.01
            self.W_ij = nn.Parameter(torch.clamp(0.1 * torch.eye(self.P, self.P).repeat(1, self.B, self.C, 1, 1, 1, self.K, self.K, 1, 1) + torch.empty(1, self.B, self.C, 1, 1, 1, self.K, self.K, self.P, self.P).uniform_(0, b), max=1).permute(0, 1, 2, 3, -2, -1, 4, 5, 6, 7))
        else:
            raise NotImplementedError('{} not implemented.'.format(weight_init))
        if self.padding != 0:
            if isinstance(self.padding, int):
                self.padding = [self.padding] * 4

    def forward(self, activations, poses):
        if self.padding != 0:
            activations = F.pad(activations, self.padding)
            poses = F.pad(poses, self.padding + [0] * 4)
        if self.share_W_ij:
            self.K = poses.shape[-1]
        self.F = (poses.shape[-1] - self.K) // self.S + 1
        poses = poses.unfold(4, size=self.K, step=self.S).unfold(5, size=self.K, step=self.S)
        poses = poses.unsqueeze(2).unsqueeze(5)
        activations = activations.unfold(2, size=self.K, step=self.S).unfold(3, size=self.K, step=self.S)
        activations = activations.reshape(-1, self.B, 1, 1, 1, *activations.shape[2:4], self.K, self.K)
        V_ji = (poses * self.W_ij).sum(dim=4)
        V_ji = V_ji.reshape(-1, self.B, self.C, self.P * self.P, 1, *V_ji.shape[-4:-2], self.K, self.K)
        if self.coor_add:
            if V_ji.shape[-1] == 1:
                self.F = self.K
            coordinates = torch.arange(self.F, dtype=torch.float32).add(1.0) / (self.F * 10)
            i_vals = torch.zeros(self.P * self.P, self.F, 1)
            j_vals = torch.zeros(self.P * self.P, 1, self.F)
            i_vals[self.P - 1, :, 0] = coordinates
            j_vals[2 * self.P - 1, 0, :] = coordinates
            if V_ji.shape[-1] == 1:
                V_ji = V_ji + (i_vals + j_vals).reshape(1, 1, 1, self.P * self.P, 1, 1, 1, self.F, self.F)
                return activations, V_ji
            V_ji = V_ji + (i_vals + j_vals).reshape(1, 1, 1, self.P * self.P, 1, self.F, self.F, 1, 1)
        return activations, V_ji


class PrimaryCapsules2d(nn.Module):
    """Primary Capsule Layer"""

    def __init__(self, in_channels, out_caps, kernel_size, stride, padding=0, pose_dim=4, weight_init='xavier_uniform'):
        super().__init__()
        self.A = in_channels
        self.B = out_caps
        self.P = pose_dim
        self.K = kernel_size
        self.S = stride
        self.padding = padding
        w_kernel = torch.empty(self.B * self.P * self.P, self.A, self.K, self.K)
        a_kernel = torch.empty(self.B, self.A, self.K, self.K)
        if weight_init == 'kaiming_normal':
            nn.init.kaiming_normal_(w_kernel)
            nn.init.kaiming_normal_(a_kernel)
        elif weight_init == 'kaiming_uniform':
            nn.init.kaiming_uniform_(w_kernel)
            nn.init.kaiming_uniform_(a_kernel)
        elif weight_init == 'xavier_normal':
            nn.init.xavier_normal_(w_kernel)
            nn.init.xavier_normal_(a_kernel)
        elif weight_init == 'xavier_uniform':
            nn.init.xavier_uniform_(w_kernel)
            nn.init.xavier_uniform_(a_kernel)
        else:
            NotImplementedError('{} not implemented.'.format(weight_init))
        self.weight = nn.Parameter(torch.cat([w_kernel, a_kernel], dim=0))
        self.BN_a = nn.BatchNorm2d(self.B, affine=True)
        self.BN_p = nn.BatchNorm3d(self.B, affine=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, stride=self.S, padding=self.padding)
        poses, activations = torch.split(x, [self.B * self.P * self.P, self.B], dim=1)
        poses = self.BN_p(poses.reshape(-1, self.B, self.P * self.P, *x.shape[2:]))
        poses = poses.reshape(-1, self.B, self.P, self.P, *x.shape[2:])
        activations = torch.sigmoid(self.BN_a(activations))
        return activations, poses


class VariationalBayesRouting2d(nn.Module):
    """Variational Bayes Capsule Routing Layer"""

    def __init__(self, in_caps, out_caps, pose_dim, kernel_size, stride, alpha0, m0, kappa0, Psi0, nu0, cov='diag', iter=3, class_caps=False):
        super().__init__()
        self.B = in_caps
        self.C = out_caps
        self.P = pose_dim
        self.D = np.max([2, self.P * self.P])
        self.K = kernel_size
        self.S = stride
        self.cov = cov
        self.iter = iter
        self.class_caps = class_caps
        self.n_classes = out_caps if class_caps else None
        self.alpha0 = torch.tensor(alpha0).type(torch.FloatTensor)
        self.register_buffer('m0', m0.unsqueeze(0).repeat(self.C, 1).reshape(1, 1, self.C, self.D, 1, 1, 1, 1, 1))
        self.kappa0 = kappa0
        if self.cov == 'diag':
            self.register_buffer('Psi0', torch.diag(Psi0).unsqueeze(0).repeat(self.C, 1).reshape(1, 1, self.C, self.D, 1, 1, 1, 1, 1))
        elif self.cov == 'full':
            self.register_buffer('Psi0', Psi0.unsqueeze(0).repeat(self.C, 1, 1).reshape(1, 1, self.C, self.D, self.D, 1, 1, 1, 1))
        self.nu0 = nu0
        self.register_buffer('lndet_Psi0', 2 * torch.diagonal(torch.cholesky(Psi0)).log().sum())
        self.register_buffer('diga_arg', torch.arange(self.D).reshape(1, 1, 1, self.D, 1, 1, 1, 1, 1).type(torch.FloatTensor))
        self.register_buffer('Dlog2', self.D * torch.log(torch.tensor(2.0)).type(torch.FloatTensor))
        self.register_buffer('Dlog2pi', self.D * torch.log(torch.tensor(2.0 * np.pi)).type(torch.FloatTensor))
        self.register_buffer('filter', torch.eye(self.K * self.K).reshape(self.K * self.K, 1, self.K, self.K))
        self.beta_u = nn.Parameter(torch.zeros(1, 1, self.C, 1, 1, 1, 1, 1, 1))
        self.beta_a = nn.Parameter(torch.zeros(1, 1, self.C, 1, 1, 1, 1, 1, 1))
        self.BN_v = nn.BatchNorm3d(self.C, affine=False)
        self.BN_a = nn.BatchNorm2d(self.C, affine=False)

    def forward(self, a_i, V_ji):
        self.F_i = a_i.shape[-2:]
        self.F_o = a_i.shape[-4:-2]
        self.N = self.B * self.F_i[0] * self.F_i[1]
        R_ij = 1.0 / self.C * torch.ones(1, self.B, self.C, 1, 1, 1, 1, 1, 1, requires_grad=False)
        for i in range(self.iter):
            self.update_qparam(a_i, V_ji, R_ij)
            if i != self.iter - 1:
                R_ij = self.update_qlatent(a_i, V_ji)
        self.Elnlambda_j = self.reduce_poses(torch.digamma(0.5 * (self.nu_j - self.diga_arg))) + self.Dlog2 + self.lndet_Psi_j
        self.Elnpi_j = torch.digamma(self.alpha_j) - torch.digamma(self.alpha_j.sum(dim=2, keepdim=True))
        H_q_j = 0.5 * self.D * torch.log(torch.tensor(2 * np.pi * np.e)) - 0.5 * self.Elnlambda_j
        a_j = self.beta_a - (torch.exp(self.Elnpi_j) * H_q_j + self.beta_u)
        a_j = a_j.squeeze()
        self.m_j = self.m_j.squeeze()
        if self.class_caps:
            a_j = a_j[..., None, None]
            self.m_j = self.m_j[..., None, None]
        self.m_j = self.BN_v(self.m_j)
        self.m_j = self.m_j.reshape(-1, self.C, self.P, self.P, *self.F_o)
        a_j = torch.sigmoid(self.BN_a(a_j))
        return a_j.squeeze(), self.m_j.squeeze()

    def update_qparam(self, a_i, V_ji, R_ij):
        R_ij = R_ij * a_i
        self.R_j = self.reduce_icaps(R_ij)
        self.alpha_j = self.alpha0 + self.R_j
        self.kappa_j = self.kappa0 + self.R_j
        self.nu_j = self.nu0 + self.R_j
        mu_j = 1.0 / self.R_j * self.reduce_icaps(R_ij * V_ji)
        self.m_j = 1.0 / self.kappa_j * (self.R_j * mu_j)
        if self.cov == 'diag':
            sigma_j = self.reduce_icaps(R_ij * (V_ji - mu_j).pow(2))
            self.invPsi_j = self.Psi0 + sigma_j + self.R_j / self.kappa_j * mu_j.pow(2)
            self.lndet_Psi_j = -self.reduce_poses(torch.log(self.invPsi_j))
        elif self.cov == 'full':
            sigma_j = self.reduce_icaps(R_ij * (V_ji - mu_j) * (V_ji - mu_j).transpose(3, 4))
            self.invPsi_j = self.Psi0 + sigma_j + self.kappa0 * self.R_j / self.kappa_j * (mu_j - self.m0) * (mu_j - self.m0).transpose(3, 4)
            self.invPsi_j = self.invPsi_j.permute(0, 1, 2, 5, 6, 7, 8, 3, 4)
            self.lndet_Psi_j = -2 * torch.diagonal(torch.cholesky(self.invPsi_j), dim1=-2, dim2=-1).log().sum(-1, keepdim=True)[..., None]

    def update_qlatent(self, a_i, V_ji):
        self.Elnpi_j = torch.digamma(self.alpha_j) - torch.digamma(self.alpha_j.sum(dim=2, keepdim=True))
        self.Elnlambda_j = self.reduce_poses(torch.digamma(0.5 * (self.nu_j - self.diga_arg))) + self.Dlog2 + self.lndet_Psi_j
        if self.cov == 'diag':
            ElnQ = self.D / self.kappa_j + self.nu_j * self.reduce_poses(1.0 / self.invPsi_j * (V_ji - self.m_j).pow(2))
        elif self.cov == 'full':
            Vm_j = V_ji - self.m_j
            ElnQ = self.D / self.kappa_j + self.nu_j * self.reduce_poses(Vm_j.transpose(3, 4) * torch.inverse(self.invPsi_j).permute(0, 1, 2, 7, 8, 3, 4, 5, 6) * Vm_j)
        lnp_j = 0.5 * self.Elnlambda_j - 0.5 * self.Dlog2pi - 0.5 * ElnQ
        p_j = torch.exp(self.Elnpi_j + lnp_j)
        sum_p_j = F.conv_transpose2d(input=p_j.sum(dim=2, keepdim=True).reshape(-1, *self.F_o, self.K * self.K).permute(0, -1, 1, 2).contiguous(), weight=self.filter, stride=[self.S, self.S])
        sum_p_j = sum_p_j.unfold(2, size=self.K, step=self.S).unfold(3, size=self.K, step=self.S)
        sum_p_j = sum_p_j.reshape([-1, self.B, 1, 1, 1, *self.F_o, self.K, self.K])
        return 1.0 / torch.clamp(sum_p_j, min=1e-11) * p_j

    def reduce_icaps(self, x):
        return x.sum(dim=(1, -2, -1), keepdim=True)

    def reduce_poses(self, x):
        return x.sum(dim=(3, 4), keepdim=True)


class CapsuleNet(nn.Module):
    """ Example: Simple 3 layer CapsNet """

    def __init__(self, args):
        super(CapsuleNet, self).__init__()
        self.P = args.pose_dim
        self.PP = int(np.max([2, self.P * self.P]))
        self.A, self.B, self.C, self.D = args.arch[:-1]
        self.n_classes = args.n_classes = args.arch[-1]
        self.in_channels = args.n_channels
        self.Conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.A, kernel_size=5, stride=2, bias=False)
        nn.init.kaiming_uniform_(self.Conv_1.weight)
        self.BN_1 = nn.BatchNorm2d(self.A)
        self.PrimaryCaps = PrimaryCapsules2d(in_channels=self.A, out_caps=self.B, kernel_size=1, stride=1, pose_dim=self.P)
        self.ConvCaps_1 = ConvCapsules2d(in_caps=self.B, out_caps=self.C, kernel_size=3, stride=2, pose_dim=self.P)
        self.ConvRouting_1 = VariationalBayesRouting2d(in_caps=self.B, out_caps=self.C, kernel_size=3, stride=2, pose_dim=self.P, cov='diag', iter=args.routing_iter, alpha0=1.0, m0=torch.zeros(self.PP), kappa0=1.0, Psi0=torch.eye(self.PP), nu0=self.PP + 1)
        self.ConvCaps_2 = ConvCapsules2d(in_caps=self.C, out_caps=self.D, kernel_size=3, stride=1, pose_dim=self.P)
        self.ConvRouting_2 = VariationalBayesRouting2d(in_caps=self.C, out_caps=self.D, kernel_size=3, stride=1, pose_dim=self.P, cov='diag', iter=args.routing_iter, alpha0=1.0, m0=torch.zeros(self.PP), kappa0=1.0, Psi0=torch.eye(self.PP), nu0=self.PP + 1)
        self.ClassCaps = ConvCapsules2d(in_caps=self.D, out_caps=self.n_classes, kernel_size=1, stride=1, pose_dim=self.P, share_W_ij=True, coor_add=True)
        self.ClassRouting = VariationalBayesRouting2d(in_caps=self.D, out_caps=self.n_classes, kernel_size=4, stride=1, pose_dim=self.P, cov='diag', iter=args.routing_iter, alpha0=1.0, m0=torch.zeros(self.PP), kappa0=1.0, Psi0=torch.eye(self.PP), nu0=self.PP + 1, class_caps=True)

    def forward(self, x):
        x = F.relu(self.BN_1(self.Conv_1(x)))
        a, v = self.PrimaryCaps(x)
        a, v = self.ConvCaps_1(a, v)
        a, v = self.ConvRouting_1(a, v)
        a, v = self.ConvCaps_2(a, v)
        a, v = self.ConvRouting_2(a, v)
        a, v = self.ClassCaps(a, v)
        yhat, v = self.ClassRouting(a, v)
        return yhat


class tinyCapsuleNet(nn.Module):
    """ Example: Simple 1 layer CapsNet """

    def __init__(self, args):
        super(tinyCapsuleNet, self).__init__()
        self.P = args.pose_dim
        self.D = int(np.max([2, self.P * self.P]))
        self.A, self.B = args.arch[0], args.arch[2]
        self.n_classes = args.n_classes = args.arch[-1]
        self.in_channels = args.n_channels
        self.Conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.A, kernel_size=5, stride=2, bias=False)
        nn.init.kaiming_uniform_(self.Conv_1.weight)
        self.BN_1 = nn.BatchNorm2d(self.A)
        self.PrimaryCaps = PrimaryCapsules2d(in_channels=self.A, out_caps=self.B, kernel_size=3, stride=2, pose_dim=self.P)
        self.ClassCaps = ConvCapsules2d(in_caps=self.B, out_caps=self.n_classes, kernel_size=6, stride=1, pose_dim=self.P)
        self.ClassRouting = VariationalBayesRouting2d(in_caps=self.B, out_caps=self.n_classes, kernel_size=6, stride=1, pose_dim=self.P, cov='diag', iter=args.routing_iter, alpha0=1.0, m0=torch.zeros(self.D), kappa0=1.0, Psi0=torch.eye(self.D), nu0=self.D + 1, class_caps=True)

    def forward(self, x):
        x = F.relu(self.BN_1(self.Conv_1(x)))
        a, v = self.PrimaryCaps(x)
        a, v = self.ClassCaps(a, v)
        yhat, v = self.ClassRouting(a, v)
        return yhat


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (PrimaryCapsules2d,
     lambda: ([], {'in_channels': 4, 'out_caps': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

