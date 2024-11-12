
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


import copy


import random


from functools import wraps


import torch


from torch import nn


import torch.nn.functional as F


import torch.distributed as dist


from torchvision import transforms as T


from torch.nn import Module


from torch.nn import SyncBatchNorm


from torch.optim import Optimizer


from torch.optim import Adam


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import models


from torchvision import transforms


class RandomApply(nn.Module):

    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def default(val, def_val):
    return def_val if val is None else val


def MaybeSyncBatchnorm(is_distributed=None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d


def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(nn.Linear(dim, hidden_size), MaybeSyncBatchnorm(sync_batchnorm)(hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, projection_size))


def SimSiamMLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(nn.Linear(dim, hidden_size, bias=False), MaybeSyncBatchnorm(sync_batchnorm)(hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size, bias=False), MaybeSyncBatchnorm(sync_batchnorm)(hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, projection_size, bias=False), MaybeSyncBatchnorm(sync_batchnorm)(projection_size, affine=False))


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):

    def inner_fn(fn):

        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance
            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


class NetWrapper(nn.Module):

    def __init__(self, net, projection_size, projection_hidden_size, layer=-2, use_simsiam_mlp=False, sync_batchnorm=None):
        super().__init__()
        self.net = net
        self.layer = layer
        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.use_simsiam_mlp = use_simsiam_mlp
        self.sync_batchnorm = sync_batchnorm
        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size, sync_batchnorm=self.sync_batchnorm)
        return projector

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)
        if not self.hook_registered:
            self._register_hook()
        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection=True):
        representation = self.get_representation(x)
        if not return_projection:
            return representation
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


class EMA:

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def get_module_device(module):
    return next(module.parameters()).device


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class BYOL(nn.Module):

    def __init__(self, net, image_size, hidden_layer=-2, projection_size=256, projection_hidden_size=4096, augment_fn=None, augment_fn2=None, moving_average_decay=0.99, use_momentum=True, sync_batchnorm=None):
        super().__init__()
        self.net = net
        DEFAULT_AUG = torch.nn.Sequential(RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3), T.RandomGrayscale(p=0.2), T.RandomHorizontalFlip(), RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2), T.RandomResizedCrop((image_size, image_size)), T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])))
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer, use_simsiam_mlp=not use_momentum, sync_batchnorm=sync_batchnorm)
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        device = get_module_device(net)
        self
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, x, return_embedding=False, return_projection=True):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'
        if return_embedding:
            return self.online_encoder(x, return_projection=return_projection)
        image_one, image_two = self.augment1(x), self.augment2(x)
        images = torch.cat((image_one, image_two), dim=0)
        online_projections, _ = self.online_encoder(images)
        online_predictions = self.online_predictor(online_projections)
        online_pred_one, online_pred_two = online_predictions.chunk(2, dim=0)
        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_projections, _ = target_encoder(images)
            target_projections = target_projections.detach()
            target_proj_one, target_proj_two = target_projections.chunk(2, dim=0)
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        loss = loss_one + loss_two
        return loss.mean()


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (RandomApply,
     lambda: ([], {'fn': torch.nn.ReLU(), 'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

