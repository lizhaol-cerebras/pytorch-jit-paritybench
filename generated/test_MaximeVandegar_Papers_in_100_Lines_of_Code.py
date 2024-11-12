
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


import numpy as np


import matplotlib.pyplot as plt


from matplotlib import pyplot as plt


import torch.optim as optim


import scipy.io


from torch.nn import functional as F


from scipy.stats import norm


from torch import optim


import random


import torch.nn.functional as F


from torch import nn


import torch.utils.data


from sklearn.datasets import make_swiss_roll


import math


from torch.utils.data import DataLoader


import torchvision.transforms as transforms


from torch.utils.data import TensorDataset


from torch.autograd import Variable


import copy


from typing import Callable


from torch.distributions.multivariate_normal import MultivariateNormal


from torch.distributions.uniform import Uniform


from torch.autograd import grad


import torchvision


from torch.distributions.transformed_distribution import TransformedDistribution


from torch.distributions.transforms import SigmoidTransform


from torch.distributions.transforms import AffineTransform


from matplotlib.image import imread


from typing import Tuple


from sklearn.preprocessing import StandardScaler


from torch.optim import lr_scheduler


class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        self.network = nn.Sequential(nn.Linear(50, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=False), nn.ReLU(), nn.Linear(1024, 784), nn.Sigmoid())

    def forward(self, x):
        return self.network(x)


class E(nn.Module):

    def __init__(self):
        super(E, self).__init__()
        self.network = nn.Sequential(nn.Linear(784, 1024), nn.LeakyReLU(0.2), nn.Linear(1024, 1024), nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=False), nn.LeakyReLU(0.2), nn.Linear(1024, 50), nn.Sigmoid())

    def forward(self, x):
        return self.network(x)


class D(nn.Module):

    def __init__(self, in_channels, nd, kd):
        super(D, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels, nd, kd, stride=2, padding=1, bias=True), nn.BatchNorm2d(nd), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(nd, nd, kd, stride=1, padding=1, bias=True), nn.BatchNorm2d(nd), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.model(x)


class GeneratorZ(nn.Module):

    def __init__(self):
        super(GeneratorZ, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0), bias=False), nn.BatchNorm2d(32, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0), bias=False), nn.BatchNorm2d(64, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False), nn.BatchNorm2d(128, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0), bias=False), nn.BatchNorm2d(256, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False), nn.BatchNorm2d(512, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True), nn.BatchNorm2d(512, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False))

    def forward(self, x):
        z = self.network(x)
        mu, sigma = z[:, :256, :, :], z[:, 256:, :, :]
        return mu, sigma

    def sample(self, x):
        mu, log_sigma = self.forward(x)
        sigma = torch.exp(log_sigma)
        return torch.randn(sigma.shape, device=x.device) * sigma + mu


class GeneratorX(nn.Module):

    def __init__(self):
        super(GeneratorX, self).__init__()
        self.network = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, stride=1, padding=0, bias=False), nn.BatchNorm2d(256, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.ConvTranspose2d(256, 128, 4, stride=2, padding=0, bias=False), nn.BatchNorm2d(128, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.ConvTranspose2d(128, 64, 4, stride=1, padding=0, bias=False), nn.BatchNorm2d(64, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.ConvTranspose2d(64, 32, 4, stride=2, padding=0, bias=False), nn.BatchNorm2d(32, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.ConvTranspose2d(32, 32, 5, stride=1, padding=0, bias=False), nn.BatchNorm2d(32, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(32, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False))

    def forward(self, x):
        return self.network(x)


class S(nn.Module):

    def __init__(self, in_channels, ns, ks):
        super(S, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels, ns, ks, 1, padding=0, bias=True), nn.BatchNorm2d(ns), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.model(x)


class U(nn.Module):

    def __init__(self, in_channels, nu, ku):
        super(U, self).__init__()
        self.model = nn.Sequential(nn.BatchNorm2d(in_channels), nn.Conv2d(in_channels, nu, ku, 1, padding=1, bias=True), nn.BatchNorm2d(nu), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(nu, nu, 1, 1, padding=0, bias=True), nn.BatchNorm2d(nu), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode='bilinear'))

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.d1 = D(3, 8, 3)
        self.d2 = D(8, 16, 3)
        self.d3 = D(16, 32, 3)
        self.d4 = D(32, 64, 3)
        self.d5 = D(64, 128, 3)
        self.u1 = U(16, 8, 3)
        self.u2 = U(32, 16, 3)
        self.u3 = U(64, 32, 3)
        self.u4 = U(128 + 4, 64, 3)
        self.u5 = U(128 + 4, 128, 3)
        self.s4 = S(32, 4, 1)
        self.s5 = S(64, 4, 1)
        self.conv_out = nn.Conv2d(8, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        h = self.d1(x)
        h = self.d2(h)
        h = self.d3(h)
        skip3 = self.s4(h)
        h = self.d4(h)
        skip4 = self.s5(h)
        h = self.d5(h)
        h = self.u5(torch.cat((skip4[:, :, 4:-4, 6:-6], h), dim=1))
        h = self.u4(torch.cat((skip3[:, :, 8:-8, 12:-12], h), dim=1))
        h = self.u3(h)
        h = self.u2(h)
        h = self.u1(h)
        return torch.sigmoid(self.conv_out(h))


class Generator(nn.Module):

    def __init__(self, noise_dim=100, out_channel=3):
        super(Generator, self).__init__()
        self.network = nn.Sequential(nn.ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False), nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.ReLU(inplace=True), nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.ReLU(inplace=True), nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.ReLU(inplace=True), nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.ReLU(inplace=True), nn.ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.Tanh())

    def forward(self, noise):
        return self.network(noise)


class DQN(nn.Module):

    def __init__(self, nb_actions):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(4, 16, 8, stride=4), nn.ReLU(), nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(), nn.Flatten(), nn.Linear(2592, 256), nn.ReLU(), nn.Linear(256, nb_actions))

    def forward(self, x):
        return self.network(x / 255.0)


class MLP(nn.Module):

    def __init__(self, input_dim=28 * 28, hidden_dim=28 * 28, output_dim=10, depth=8):
        super(MLP, self).__init__()
        model = []
        for _ in range(depth):
            model += [nn.Linear(input_dim, hidden_dim), nn.ReLU(), torch.nn.BatchNorm1d(hidden_dim)]
        model += [nn.Linear(hidden_dim, output_dim), nn.LogSoftmax(dim=-1)]
        self.network = nn.Sequential(*model)

    def forward(self, x):
        return self.network(x)


def get_ndc_rays(H, W, focal, rays_o, rays_d, near=1.0):
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    rays_o = torch.stack([-focal / W / 2.0 * rays_o[..., 0] / rays_o[..., 2], -focal / H / 2.0 * rays_o[..., 1] / rays_o[..., 2], 1.0 + 2.0 * near / rays_o[..., 2]], -1)
    rays_d = torch.stack([-focal / W / 2.0 * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2]), -focal / H / 2.0 * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2]), -2.0 * near / rays_o[..., 2]], -1)
    return rays_o, rays_d


def sample_batch(camera_extrinsics, camera_intrinsics, images, batch_size, H, W, img_index=0, sample_all=False):
    if sample_all:
        image_indices = (torch.zeros(W * H) + img_index).type(torch.long)
        u, v = np.meshgrid(np.linspace(0, W - 1, W, dtype=int), np.linspace(0, H - 1, H, dtype=int))
        u = torch.from_numpy(u.reshape(-1))
        v = torch.from_numpy(v.reshape(-1))
    else:
        image_indices = (torch.zeros(batch_size) + img_index).type(torch.long)
        u = torch.randint(W, (batch_size,), device=camera_intrinsics.device)
        v = torch.randint(H, (batch_size,), device=camera_intrinsics.device)
    focal = camera_intrinsics[0] ** 2 * W
    t = camera_extrinsics[img_index, :3]
    r = camera_extrinsics[img_index, -3:]
    phi_skew = torch.stack([torch.cat([torch.zeros(1, device=r.device), -r[2:3], r[1:2]]), torch.cat([r[2:3], torch.zeros(1, device=r.device), -r[0:1]]), torch.cat([-r[1:2], r[0:1], torch.zeros(1, device=r.device)])], dim=0)
    alpha = r.norm() + 1e-15
    R = torch.eye(3, device=r.device) + torch.sin(alpha) / alpha * phi_skew + (1 - torch.cos(alpha)) / alpha ** 2 * (phi_skew @ phi_skew)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)
    c2w = torch.cat([c2w, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=c2w.device)], dim=0)
    rays_d_cam = torch.cat([((u - 0.5 * W) / focal).unsqueeze(-1), (-(v - 0.5 * H) / focal).unsqueeze(-1), -torch.ones_like(u).unsqueeze(-1)], dim=-1)
    rays_d_world = torch.matmul(c2w[:3, :3].view(1, 3, 3), rays_d_cam.unsqueeze(2)).squeeze(2)
    rays_o_world = c2w[:3, 3].view(1, 3).expand_as(rays_d_world)
    rays_o_world, rays_d_world = get_ndc_rays(H, W, focal, rays_o=rays_o_world, rays_d=rays_d_world)
    return rays_o_world, F.normalize(rays_d_world, p=2, dim=1), (image_indices, v.cpu(), u.cpu())


class DiffusionModel:

    def __init__(self, T: 'int', model: 'nn.Module', device: 'str'):
        self.T = T
        self.function_approximator = model
        self.device = device
        self.beta = torch.linspace(0.0001, 0.02, T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def training(self, batch_size, optimizer):
        """
        Algorithm 1 in Denoising Diffusion Probabilistic Models
        """
        x0 = sample_batch(batch_size, self.device)
        t = torch.randint(1, self.T + 1, (batch_size,), device=self.device, dtype=torch.long)
        eps = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        eps_predicted = self.function_approximator(torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps, t - 1)
        loss = nn.functional.mse_loss(eps, eps_predicted)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sampling(self, n_samples=1, image_channels=1, img_size=(32, 32), use_tqdm=True):
        """
        Algorithm 2 in Denoising Diffusion Probabilistic Models
        """
        x = torch.randn((n_samples, image_channels, img_size[0], img_size[1]), device=self.device)
        progress_bar = tqdm if use_tqdm else lambda x: x
        for t in progress_bar(range(self.T, 0, -1)):
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            t = torch.ones(n_samples, dtype=torch.long, device=self.device) * t
            beta_t = self.beta[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_t = self.alpha[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mean = 1 / torch.sqrt(alpha_t) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * self.function_approximator(x, t - 1))
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * z
        return x


class Downsample(nn.Module):

    def __init__(self, C):
        """
        :param C (int): number of input and output channels
        """
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(C, C, 3, stride=2, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        assert x.shape == (B, C, H // 2, W // 2)
        return x


class Upsample(nn.Module):

    def __init__(self, C):
        """
        :param C (int): number of input and output channels
        """
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(C, C, 3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = nn.functional.interpolate(x, size=None, scale_factor=2, mode='nearest')
        x = self.conv(x)
        assert x.shape == (B, C, H * 2, W * 2)
        return x


class Nin(nn.Module):

    def __init__(self, in_dim, out_dim, scale=1e-10):
        super(Nin, self).__init__()
        n = (in_dim + out_dim) / 2
        limit = np.sqrt(3 * scale / n)
        self.W = torch.nn.Parameter(torch.zeros((in_dim, out_dim), dtype=torch.float32).uniform_(-limit, limit))
        self.b = torch.nn.Parameter(torch.zeros((1, out_dim, 1, 1), dtype=torch.float32))

    def forward(self, x):
        return torch.einsum('bchw, co->bohw', x, self.W) + self.b


class ResNetBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout_rate=0.1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.dense = nn.Linear(512, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        if not in_ch == out_ch:
            self.nin = Nin(in_ch, out_ch)
        self.dropout_rate = dropout_rate
        self.nonlinearity = torch.nn.SiLU()

    def forward(self, x, temb):
        """
        :param x: (B, C, H, W)
        :param temb: (B, dim)
        """
        h = self.nonlinearity(nn.functional.group_norm(x, num_groups=32))
        h = self.conv1(h)
        h += self.dense(self.nonlinearity(temb))[:, :, None, None]
        h = self.nonlinearity(nn.functional.group_norm(h, num_groups=32))
        h = nn.functional.dropout(h, p=self.dropout_rate)
        h = self.conv2(h)
        if not x.shape[1] == h.shape[1]:
            x = self.nin(x)
        assert x.shape == h.shape
        return x + h


class AttentionBlock(nn.Module):

    def __init__(self, ch):
        super(AttentionBlock, self).__init__()
        self.Q = Nin(ch, ch)
        self.K = Nin(ch, ch)
        self.V = Nin(ch, ch)
        self.ch = ch
        self.nin = Nin(ch, ch, scale=0.0)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.ch
        h = nn.functional.group_norm(x, num_groups=32)
        q = self.Q(h)
        k = self.K(h)
        v = self.V(h)
        w = torch.einsum('bchw,bcHW->bhwHW', q, k) * int(C) ** -0.5
        w = torch.reshape(w, [B, H, W, H * W])
        w = torch.nn.functional.softmax(w, dim=-1)
        w = torch.reshape(w, [B, H, W, H, W])
        h = torch.einsum('bhwHW,bcHW->bchw', w, v)
        h = self.nin(h)
        assert h.shape == x.shape
        return x + h


class DownConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownConvBlock, self).__init__()
        self.block = nn.Sequential(nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super(UpConvBlock, self).__init__()
        layers = [nn.ReLU(inplace=True), nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        if dropout:
            layers.append(nn.Dropout(p=0.5, inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.down1 = DownConvBlock(64, 128)
        self.down2 = DownConvBlock(128, 256)
        self.down3 = DownConvBlock(256, 512)
        self.down4 = DownConvBlock(512, 512)
        self.down5 = DownConvBlock(512, 512)
        self.down6 = DownConvBlock(512, 512)
        self.middle = nn.Sequential(nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.ReLU(inplace=True), nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.up1 = UpConvBlock(1024, 512, dropout=True)
        self.up2 = UpConvBlock(1024, 512, dropout=True)
        self.up3 = UpConvBlock(1024, 512, dropout=True)
        self.up4 = UpConvBlock(1024, 256)
        self.up5 = UpConvBlock(512, 128)
        self.up6 = UpConvBlock(256, 64)
        self.outermost = nn.Sequential(nn.ReLU(inplace=True), nn.ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.Tanh())

    def forward(self, x):
        x0 = self.conv_in(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x = self.middle(x6)
        x = self.up1(torch.cat((x, x6), dim=1))
        x = self.up2(torch.cat((x, x5), dim=1))
        x = self.up3(torch.cat((x, x4), dim=1))
        x = self.up4(torch.cat((x, x3), dim=1))
        x = self.up5(torch.cat((x, x2), dim=1))
        x = self.up6(torch.cat((x, x1), dim=1))
        return self.outermost(torch.cat((x, x0), dim=1))


class FastNerf(nn.Module):

    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim_pos=384, hidden_dim_dir=128, D=8):
        super(FastNerf, self).__init__()
        self.Fpos = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim_pos), nn.ReLU(), nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(), nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(), nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(), nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(), nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(), nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(), nn.Linear(hidden_dim_pos, 3 * D + 1))
        self.Fdir = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + 3, hidden_dim_dir), nn.ReLU(), nn.Linear(hidden_dim_dir, hidden_dim_dir), nn.ReLU(), nn.Linear(hidden_dim_dir, hidden_dim_dir), nn.ReLU(), nn.Linear(hidden_dim_dir, D))
        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.D = D

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        sigma_uvw = self.Fpos(self.positional_encoding(o, self.embedding_dim_pos))
        sigma = torch.nn.functional.softplus(sigma_uvw[:, 0][..., None])
        uvw = torch.sigmoid(sigma_uvw[:, 1:].reshape(-1, 3, self.D))
        beta = torch.softmax(self.Fdir(self.positional_encoding(d, self.embedding_dim_direction)), -1)
        color = (beta.unsqueeze(1) * uvw).sum(-1)
        return color, sigma


class Cache(nn.Module):

    def __init__(self, model, scale, device, Np, Nd):
        super(Cache, self).__init__()
        with torch.no_grad():
            x, y, z = torch.meshgrid([torch.linspace(-scale / 2, scale / 2, Np), torch.linspace(-scale / 2, scale / 2, Np), torch.linspace(-scale / 2, scale / 2, Np)])
            xyz = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1)
            sigma_uvw = model.Fpos(model.positional_encoding(xyz, model.embedding_dim_pos))
            self.sigma_uvw = sigma_uvw.reshape((Np, Np, Np, -1))
            xd, yd = torch.meshgrid([torch.linspace(-scale / 2, scale / 2, Nd), torch.linspace(-scale / 2, scale / 2, Nd)])
            xyz_d = torch.cat((xd.reshape(-1, 1), yd.reshape(-1, 1), torch.sqrt((1 - xd ** 2 - yd ** 2).clip(0, 1)).reshape(-1, 1)), dim=1)
            beta = model.Fdir(model.positional_encoding(xyz_d, model.embedding_dim_direction))
            self.beta = beta.reshape((Nd, Nd, -1))
        self.scale = scale
        self.Np = Np
        self.Nd = Nd
        self.D = model.D

    def forward(self, x, d):
        color = torch.zeros_like(x)
        sigma = torch.zeros((x.shape[0], 1), device=x.device)
        mask = (x[:, 0].abs() < self.scale / 2) & (x[:, 1].abs() < self.scale / 2) & (x[:, 2].abs() < self.scale / 2)
        idx = (x[mask] / (self.scale / self.Np) + self.Np / 2).long().clip(0, self.Np - 1)
        sigma_uvw = self.sigma_uvw[idx[:, 0], idx[:, 1], idx[:, 2]]
        idx = (d[mask] * self.Nd).long().clip(0, self.Nd - 1)
        beta = torch.softmax(self.beta[idx[:, 0], idx[:, 1]], -1)
        sigma[mask] = torch.nn.functional.softplus(sigma_uvw[:, 0][..., None])
        uvw = torch.sigmoid(sigma_uvw[:, 1:].reshape(-1, 3, self.D))
        color[mask] = (beta.unsqueeze(1) * uvw).sum(-1)
        return color, sigma


class ELU(nn.Module):

    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        cond = x > 0
        y = x.clone()
        y[~cond] = self.alpha * (torch.exp(x[~cond]) - 1.0)
        return y


def eval_spherical_function(k, d):
    x, y, z = d[..., 0:1], d[..., 1:2], d[..., 2:3]
    return 0.282095 * k[..., 0] + -0.488603 * y * k[..., 1] + 0.488603 * z * k[..., 2] - 0.488603 * x * k[..., 3] + (1.092548 * x * y * k[..., 4] - 1.092548 * y * z * k[..., 5] + 0.315392 * (2.0 * z * z - x * x - y * y) * k[..., 6] + -1.092548 * x * z * k[..., 7] + 0.546274 * (x * x - y * y) * k[..., 8])


class NerfModel(nn.Module):

    def __init__(self, N=256, scale=1.5):
        """
        :param N
        :param scale: The maximum absolute value among all coordinates for objects in the scene
        """
        super(NerfModel, self).__init__()
        self.voxel_grid = nn.Parameter(torch.ones((N, N, N, 27 + 1)) / 100)
        self.scale = scale
        self.N = N

    def forward(self, x, d):
        color = torch.zeros_like(x)
        sigma = torch.zeros(x.shape[0], device=x.device)
        mask = (x[:, 0].abs() < self.scale) & (x[:, 1].abs() < self.scale) & (x[:, 2].abs() < self.scale)
        idx = (x[mask] / (2 * self.scale / self.N) + self.N / 2).long().clip(0, self.N - 1)
        tmp = self.voxel_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        sigma[mask], k = torch.nn.functional.relu(tmp[:, 0]), tmp[:, 1:]
        color[mask] = eval_spherical_function(k.reshape(-1, 3, 9), d[mask])
        return color, sigma


class GELU(nn.Module):

    def __init__(self, alpha=1.0):
        super(GELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return x * 0.5 * (1 + torch.erf(x / np.sqrt(2.0)))


class PatchGAN(nn.Module):

    def __init__(self):
        super(PatchGAN, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(6, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)))

    def forward(self, x):
        return self.network(x)


class SineLayer(nn.Module):

    def __init__(self, w0):
        super(SineLayer, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):

    def __init__(self, w0=30, in_dim=2, hidden_dim=256, out_dim=1):
        super(Siren, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), SineLayer(w0), nn.Linear(hidden_dim, hidden_dim), SineLayer(w0), nn.Linear(hidden_dim, hidden_dim), SineLayer(w0), nn.Linear(hidden_dim, hidden_dim), SineLayer(w0), nn.Linear(hidden_dim, out_dim))
        with torch.no_grad():
            self.net[0].weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
            self.net[2].weight.uniform_(-np.sqrt(6.0 / hidden_dim) / w0, np.sqrt(6.0 / hidden_dim) / w0)
            self.net[4].weight.uniform_(-np.sqrt(6.0 / hidden_dim) / w0, np.sqrt(6.0 / hidden_dim) / w0)
            self.net[6].weight.uniform_(-np.sqrt(6.0 / hidden_dim) / w0, np.sqrt(6.0 / hidden_dim) / w0)
            self.net[8].weight.uniform_(-np.sqrt(6.0 / hidden_dim) / w0, np.sqrt(6.0 / hidden_dim) / w0)

    def forward(self, x):
        return self.net(x)


class GaussianNoiseLayer(nn.Module):

    def __init__(self, sigma):
        super(GaussianNoiseLayer, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.train:
            noise = torch.randn(x.shape, device=x.device) * self.sigma
            return x + noise
        else:
            return x


class DownResBlock(nn.Module):

    def __init__(self, input_dim, output_dim, filter_size):
        super(DownResBlock, self).__init__()
        self.shortcut = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, bias=True), nn.AvgPool2d(2))
        self.network = nn.Sequential(nn.InstanceNorm2d(input_dim), nn.ReLU(inplace=True), nn.Conv2d(input_dim, input_dim, kernel_size=filter_size, padding=filter_size // 2), nn.InstanceNorm2d(input_dim), nn.ReLU(inplace=True), nn.Conv2d(input_dim, output_dim, kernel_size=filter_size, padding=filter_size // 2), nn.AvgPool2d(2))

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        output = self.network(inputs)
        return shortcut + output


class UpResBlock(nn.Module):

    def __init__(self, input_dim, output_dim, filter_size):
        super(UpResBlock, self).__init__()
        self.shortcut = nn.Sequential(nn.Conv2d(input_dim, 4 * output_dim, kernel_size=1, stride=1, bias=True), nn.PixelShuffle(2))
        self.network = nn.Sequential(nn.BatchNorm2d(input_dim), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(input_dim, output_dim, kernel_size=filter_size, padding=filter_size // 2), nn.BatchNorm2d(output_dim), nn.ReLU(inplace=True), nn.Conv2d(output_dim, output_dim, kernel_size=filter_size, padding=filter_size // 2))

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        output = self.network(inputs)
        return shortcut + output


class NGP(torch.nn.Module):

    def __init__(self, T, Nl, L, device, aabb_scale, F=2):
        super(NGP, self).__init__()
        self.T = T
        self.Nl = Nl
        self.F = F
        self.L = L
        self.aabb_scale = aabb_scale
        self.lookup_tables = torch.nn.ParameterDict({str(i): torch.nn.Parameter((torch.rand((T, 2), device=device) * 2 - 1) * 0.0001) for i in range(len(Nl))})
        self.pi1, self.pi2, self.pi3 = 1, 2654435761, 805459861
        self.density_MLP = nn.Sequential(nn.Linear(self.F * len(Nl), 64), nn.ReLU(), nn.Linear(64, 16))
        self.color_MLP = nn.Sequential(nn.Linear(27 + 16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3), nn.Sigmoid())

    def positional_encoding(self, x):
        out = [x]
        for j in range(self.L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, x, d):
        x /= self.aabb_scale
        mask = (x[:, 0].abs() < 0.5) & (x[:, 1].abs() < 0.5) & (x[:, 2].abs() < 0.5)
        x += 0.5
        color = torch.zeros((x.shape[0], 3), device=x.device)
        log_sigma = torch.zeros(x.shape[0], device=x.device) - 100000
        features = torch.empty((x[mask].shape[0], self.F * len(self.Nl)), device=x.device)
        for i, N in enumerate(self.Nl):
            floor = torch.floor(x[mask] * N)
            ceil = torch.ceil(x[mask] * N)
            vertices = torch.zeros((x[mask].shape[0], 8, 3), dtype=torch.int64, device=x.device)
            vertices[:, 0] = floor
            vertices[:, 1] = torch.cat((ceil[:, 0, None], floor[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 2] = torch.cat((floor[:, 0, None], ceil[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 4] = torch.cat((floor[:, 0, None], floor[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 6] = torch.cat((floor[:, 0, None], ceil[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 5] = torch.cat((ceil[:, 0, None], floor[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 3] = torch.cat((ceil[:, 0, None], ceil[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 7] = ceil
            a = vertices[:, :, 0] * self.pi1
            b = vertices[:, :, 1] * self.pi2
            c = vertices[:, :, 2] * self.pi3
            h_x = torch.remainder(torch.bitwise_xor(torch.bitwise_xor(a, b), c), self.T)
            looked_up = self.lookup_tables[str(i)][h_x].transpose(-1, -2)
            volume = looked_up.reshape((looked_up.shape[0], 2, 2, 2, 2))
            features[:, i * 2:(i + 1) * 2] = torch.nn.functional.grid_sample(volume, (x[mask] * N - floor - 0.5).unsqueeze(1).unsqueeze(1).unsqueeze(1)).squeeze(-1).squeeze(-1).squeeze(-1)
        xi = self.positional_encoding(d[mask])
        h = self.density_MLP(features)
        log_sigma[mask] = h[:, 0]
        color[mask] = self.color_MLP(torch.cat((h, xi), dim=1))
        return color, torch.exp(log_sigma)


class KiloNerf(nn.Module):

    def __init__(self, N, embedding_dim_pos=10, embedding_dim_direction=4, scene_scale=3):
        super(KiloNerf, self).__init__()
        self.layer1_w = torch.nn.Parameter(torch.zeros((N, N, N, 63, 32)).uniform_(-np.sqrt(6.0 / 85), np.sqrt(6.0 / 85)))
        self.layer1_b = torch.nn.Parameter(torch.zeros((N, N, N, 1, 32)))
        self.layer2_w = torch.nn.Parameter(torch.zeros((N, N, N, 32, 33)).uniform_(-np.sqrt(6.0 / 64), np.sqrt(6.0 / 64)))
        self.layer2_b = torch.nn.Parameter(torch.zeros((N, N, N, 1, 33)))
        self.layer3_w = torch.nn.Parameter(torch.zeros((N, N, N, 32, 32)).uniform_(-np.sqrt(6.0 / 64), np.sqrt(6.0 / 64)))
        self.layer3_b = torch.nn.Parameter(torch.zeros((N, N, N, 1, 32)))
        self.layer4_w = torch.nn.Parameter(torch.zeros((N, N, N, 27 + 32, 32)).uniform_(-np.sqrt(6.0 / 64), np.sqrt(6.0 / 64)))
        self.layer4_b = torch.nn.Parameter(torch.zeros((N, N, N, 1, 32)))
        self.layer5_w = torch.nn.Parameter(torch.zeros((N, N, N, 32, 3)).uniform_(-np.sqrt(6.0 / 35), np.sqrt(6.0 / 35)))
        self.layer5_b = torch.nn.Parameter(torch.zeros((N, N, N, 1, 3)))
        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.N = N
        self.scale = scene_scale

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, x, d):
        color = torch.zeros_like(x)
        sigma = torch.zeros(x.shape[0], device=x.device)
        mask = (x[:, 0].abs() < self.scale / 2) & (x[:, 1].abs() < self.scale / 2) & (x[:, 2].abs() < self.scale / 2)
        idx = (x[mask] / (self.scale / self.N) + self.N / 2).long().clip(0, self.N - 1)
        emb_x = self.positional_encoding(x[mask], self.embedding_dim_pos)
        emb_d = self.positional_encoding(d[mask], self.embedding_dim_direction)
        h = torch.relu(emb_x.unsqueeze(1) @ self.layer1_w[idx[:, 0], idx[:, 1], idx[:, 2]] + self.layer1_b[idx[:, 0], idx[:, 1], idx[:, 2]])
        h = torch.relu(h @ self.layer2_w[idx[:, 0], idx[:, 1], idx[:, 2]] + self.layer2_b[idx[:, 0], idx[:, 1], idx[:, 2]])
        h, density = h[:, :, :-1], h[:, :, -1]
        h = h @ self.layer3_w[idx[:, 0], idx[:, 1], idx[:, 2]] + self.layer3_b[idx[:, 0], idx[:, 1], idx[:, 2]]
        h = torch.relu(torch.cat((h, emb_d.unsqueeze(1)), dim=-1) @ self.layer4_w[idx[:, 0], idx[:, 1], idx[:, 2]] + self.layer4_b[idx[:, 0], idx[:, 1], idx[:, 2]])
        c = torch.sigmoid(h @ self.layer5_w[idx[:, 0], idx[:, 1], idx[:, 2]] + self.layer5_b[idx[:, 0], idx[:, 1], idx[:, 2]])
        color[mask] = c.squeeze(1)
        sigma[mask] = density.squeeze(1)
        return color, sigma


class Maxout(nn.Module):

    def __init__(self, din, dout, k):
        super(Maxout, self).__init__()
        self.net = nn.Linear(din, k * dout)
        self.k = k
        self.dout = dout

    def forward(self, x):
        return torch.max(self.net(x).reshape(-1, self.k * self.dout).reshape(-1, self.dout, self.k), dim=-1).values


class GaborFilter(nn.Module):

    def __init__(self, in_dim, out_dim, alpha, beta=1.0):
        super(GaborFilter, self).__init__()
        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim,)))
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.linear.weight.data *= 128.0 * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        norm = (x ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * x @ self.mu.T
        return torch.exp(-self.gamma.unsqueeze(0) / 2.0 * norm) * torch.sin(self.linear(x))


class GaborNet(nn.Module):

    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1, k=4):
        super(GaborNet, self).__init__()
        self.k = k
        self.gabon_filters = nn.ModuleList([GaborFilter(in_dim, hidden_dim, alpha=6.0 / k) for _ in range(k)])
        self.linear = nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(k - 1)] + [torch.nn.Linear(hidden_dim, out_dim)])
        for lin in self.linear[:k - 1]:
            lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim))

    def forward(self, x):
        zi = self.gabon_filters[0](x)
        for i in range(self.k - 1):
            zi = self.linear[i](zi) * self.gabon_filters[i + 1](x)
        return self.linear[self.k - 1](zi)


class NICE(nn.Module):

    def __init__(self, data_dim=28 * 28, hidden_dim=1000):
        super().__init__()
        self.m = torch.nn.ModuleList([nn.Sequential(nn.Linear(data_dim // 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, data_dim // 2)) for i in range(4)])
        self.s = torch.nn.Parameter(torch.randn(data_dim))

    def forward(self, x):
        x = x.clone()
        for i in range(len(self.m)):
            x_i1 = x[:, ::2] if i % 2 == 0 else x[:, 1::2]
            x_i2 = x[:, 1::2] if i % 2 == 0 else x[:, ::2]
            h_i1 = x_i1
            h_i2 = x_i2 + self.m[i](x_i1)
            x = torch.empty(x.shape, device=x.device)
            x[:, ::2] = h_i1
            x[:, 1::2] = h_i2
        z = torch.exp(self.s) * x
        log_jacobian = torch.sum(self.s)
        return z, log_jacobian

    def invert(self, z):
        x = z.clone() / torch.exp(self.s)
        for i in range(len(self.m) - 1, -1, -1):
            h_i1 = x[:, ::2]
            h_i2 = x[:, 1::2]
            x_i1 = h_i1
            x_i2 = h_i2 - self.m[i](x_i1)
            x = torch.empty(x.shape, device=x.device)
            x[:, ::2] = x_i1 if i % 2 == 0 else x_i2
            x[:, 1::2] = x_i2 if i % 2 == 0 else x_i1
        return x


class NiN(nn.Module):

    def __init__(self):
        super(NiN, self).__init__()
        conv1 = nn.Conv2d(1, 96, 5, padding=2)
        nn.init.normal_(conv1.weight, mean=0.0, std=0.05)
        cccp1 = nn.Conv2d(96, 64, 1)
        nn.init.normal_(cccp1.weight, mean=0.0, std=0.05)
        cccp2 = nn.Conv2d(64, 48, 1)
        nn.init.normal_(cccp2.weight, mean=0.0, std=0.05)
        conv2 = nn.Conv2d(48, 128, 5, padding=2)
        nn.init.normal_(conv2.weight, mean=0.0, std=0.05)
        cccp3 = nn.Conv2d(128, 96, 1)
        nn.init.normal_(cccp3.weight, mean=0.0, std=0.05)
        cccp4 = nn.Conv2d(96, 48, 1)
        nn.init.normal_(cccp4.weight, mean=0.0, std=0.05)
        conv3 = nn.Conv2d(48, 128, 5, padding=2)
        nn.init.normal_(conv3.weight, mean=0.0, std=0.05)
        cccp5 = nn.Conv2d(128, 96, 1)
        nn.init.normal_(cccp5.weight, mean=0.0, std=0.05)
        cccp6 = nn.Conv2d(96, 10, 1)
        nn.init.normal_(cccp6.weight, mean=0.0, std=0.05)
        self.model = nn.Sequential(conv1, nn.ReLU(), cccp1, nn.ReLU(), cccp2, nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1), nn.Dropout(p=0.5), conv2, nn.ReLU(), cccp3, nn.ReLU(), cccp4, nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1), nn.Dropout(p=0.5), conv3, nn.ReLU(), cccp5, nn.ReLU(), cccp6, torch.nn.AvgPool2d(7, stride=1, padding=0))
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        log_prob = self.logsoftmax(self.model(x).squeeze(-1).squeeze(-1))
        return log_prob


class WeightDecay(nn.Module):

    def __init__(self, model, device):
        super(WeightDecay, self).__init__()
        self.positive_constraint = torch.nn.Softplus()
        idx = 0
        self.parameter_dict = {}
        for m in model.parameters():
            self.parameter_dict[str(idx)] = torch.nn.Parameter(torch.rand(m.shape, device=device))
            idx += 1
        self.params = torch.nn.ParameterDict(self.parameter_dict)

    def forward(self, model):
        regularization = 0.0
        for coefficients, weights in zip(self.parameters(), model.parameters()):
            regularization += (self.positive_constraint(coefficients) * weights ** 2).sum()
        return regularization


class ActorCritic(nn.Module):

    def __init__(self, nb_actions):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(4, 16, 8, stride=4), nn.Tanh(), nn.Conv2d(16, 32, 4, stride=2), nn.Tanh(), nn.Flatten(), nn.Linear(2592, 256), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(256, nb_actions))
        self.critic = nn.Sequential(nn.Linear(256, 1))

    def forward(self, x):
        h = self.head(x)
        return self.actor(h), self.critic(h)


class SELU(nn.Module):

    def __init__(self):
        super(SELU, self).__init__()
        self.alpha = 1.6732632423543772
        self.lambda_ = 1.0507009873554805

    def forward(self, x):
        return self.lambda_ * (torch.maximum(torch.tensor([0], device=x.device), x) + torch.minimum(torch.tensor([0], device=x.device), self.alpha * (torch.exp(x) - 1)))


class SNN(nn.Module):

    def __init__(self, input_dim=28 * 28, hidden_dim=28 * 28, output_dim=10, depth=8):
        super(SNN, self).__init__()
        model = []
        for _ in range(depth):
            model += [nn.Linear(input_dim, hidden_dim), SELU()]
        model += [nn.Linear(hidden_dim, output_dim), nn.LogSoftmax(dim=-1)]
        self.network = nn.Sequential(*model)

    def forward(self, x):
        return self.network(x)


class ResnetBlock(nn.Module):

    def __init__(self, ch):
        super(ResnetBlock, self).__init__()
        self.model = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, kernel_size=3, padding=0, bias=True), nn.InstanceNorm2d(ch, affine=False, track_running_stats=False), nn.ReLU(True), nn.Dropout(0.5), nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, kernel_size=3, padding=0, bias=True), nn.InstanceNorm2d(ch, affine=False, track_running_stats=False))

    def forward(self, x):
        return x + self.model(x)


class NLayerDiscriminator(nn.Module):

    def __init__(self):
        super(NLayerDiscriminator, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True), nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True), nn.InstanceNorm2d(128, affine=False, track_running_stats=False), nn.LeakyReLU(0.2, True), nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True), nn.InstanceNorm2d(256, affine=False, track_running_stats=False), nn.LeakyReLU(0.2, True), nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=True), nn.InstanceNorm2d(512, affine=False, track_running_stats=False), nn.LeakyReLU(0.2, True), nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, x):
        return self.model(x)


class PlanarFlow(nn.Module):

    def __init__(self, data_dim):
        super().__init__()
        self.u = nn.Parameter(torch.rand(data_dim))
        self.w = nn.Parameter(torch.rand(data_dim))
        self.b = nn.Parameter(torch.rand(1))
        self.h = nn.Tanh()
        self.h_prime = lambda z: 1 - self.h(z) ** 2

    def constrained_u(self):
        """
        Constrain the parameters u to ensure invertibility
        """
        wu = torch.matmul(self.w.T, self.u)
        m = lambda x: -1 + torch.log(1 + torch.exp(x))
        return self.u + (m(wu) - wu) * (self.w / (torch.norm(self.w) ** 2 + 1e-15))

    def forward(self, z):
        u = self.constrained_u()
        hidden_units = torch.matmul(self.w.T, z.T) + self.b
        x = z + u.unsqueeze(0) * self.h(hidden_units).unsqueeze(-1)
        psi = self.h_prime(hidden_units).unsqueeze(0) * self.w.unsqueeze(-1)
        log_det = torch.log((1 + torch.matmul(u.T, psi)).abs() + 1e-15)
        return x, log_det


class NormalizingFlow(nn.Module):

    def __init__(self, flow_length, data_dim):
        super().__init__()
        self.layers = nn.Sequential(*(PlanarFlow(data_dim) for _ in range(flow_length)))

    def forward(self, z):
        log_jacobians = 0
        for layer in self.layers:
            z, log_jacobian = layer(z)
            log_jacobians += log_jacobian
        return z, log_jacobians


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (D,
     lambda: ([], {'in_channels': 4, 'nd': 4, 'kd': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (DownConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Downsample,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GaborFilter,
     lambda: ([], {'in_dim': 4, 'out_dim': 4, 'alpha': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GaussianNoiseLayer,
     lambda: ([], {'sigma': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Generator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 100, 4, 4])], {})),
    (GeneratorX,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 4, 4])], {})),
    (GeneratorZ,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Maxout,
     lambda: ([], {'din': 4, 'dout': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NLayerDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (NiN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {})),
    (Nin,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NormalizingFlow,
     lambda: ([], {'flow_length': 4, 'data_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PatchGAN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 6, 64, 64])], {})),
    (PlanarFlow,
     lambda: ([], {'data_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResnetBlock,
     lambda: ([], {'ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (S,
     lambda: ([], {'in_channels': 4, 'ns': 4, 'ks': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SineLayer,
     lambda: ([], {'w0': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (U,
     lambda: ([], {'in_channels': 4, 'nu': 4, 'ku': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {})),
    (UpConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Upsample,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

