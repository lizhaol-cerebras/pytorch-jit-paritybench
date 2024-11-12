
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


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


from abc import ABCMeta


from abc import abstractmethod


import torch.utils.data as torch_data


import torch.nn


import numpy as np


import random


from torchvision import transforms as tfs


import logging


import torch.multiprocessing as mp


import torch.distributed as dist


from torch.nn.utils import clip_grad_norm_


from torch.nn.utils import clip_grad_value_


import torch.optim as optim


import torch.optim.lr_scheduler as lr_sched


import torchvision


import matplotlib.pyplot as plt


import collections


import torch.nn as nn


import re


import time


import math


class ModelTemplate(nn.Module):

    def __init__(self):
        super().__init__()

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])
        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                logger.info('Update weight %s: %s' % (key, str(val.shape)))
        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)
        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)
        self.load_state_dict(checkpoint['model_state'])
        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])
        if 'version' in checkpoint:
            None
        logger.info('==> Done')
        return it, epoch

    @staticmethod
    def init_weights(model):
        if type(model) == nn.Linear:
            torch.nn.init.xavier_normal_(model.weight)
        elif type(model) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.zeros_(model.bias)
        elif type(model) == nn.BatchNorm2d:
            torch.nn.init.constant_(model.weight, 1)
            torch.nn.init.constant_(model.bias, 0)


class UnsuperShortcut(nn.Module):

    def __init__(self, **kwargs):
        super(UnsuperShortcut, self).__init__(**kwargs)
        self.stage1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(inplace=True), nn.Conv2d(32, 32, 3, 1, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(inplace=True))
        self.pool = nn.MaxPool2d(2, 2)
        self.stage2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(inplace=True), nn.Conv2d(64, 64, 3, 1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(inplace=True))
        self.stage3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(inplace=True), nn.Conv2d(128, 128, 3, 1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Conv2d(128, 256, 3, 1, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(inplace=True), nn.Conv2d(256, 256, 3, 1, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        layer1 = self.stage1(x)
        layer2 = self.stage2(self.pool(layer1))
        layer3 = self.stage3(self.pool(layer2))
        h_new, w_new = layer3.shape[-2:]
        layer1_down = nn.functional.interpolate(layer1, size=[h_new, w_new])
        layer2_down = nn.functional.interpolate(layer2, size=[h_new, w_new])
        out = torch.cat([layer1_down, layer2_down, layer3], axis=1)
        return out


class UnsuperVgg(nn.Module):

    def __init__(self, **kwargs):
        super(UnsuperVgg, self).__init__(**kwargs)
        self.cnn = nn.Sequential(nn.Conv2d(3, 32, 3, 2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, 1, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 64, 3, 2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, 1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 128, 3, 1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, 2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 256, 3, 1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, 1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.cnn(x)
        return out


class UnsuperVggTiny(nn.Module):

    def __init__(self, **kwargs):
        super(UnsuperVggTiny, self).__init__(**kwargs)
        self.cnn = nn.Sequential(nn.Conv2d(3, 16, 3, 2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 32, 3, 1, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 64, 3, 2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, 1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 128, 3, 2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, 1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, 1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.cnn(x)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1), nn.BatchNorm2d(outchannel), nn.ReLU(inplace=True), nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(outchannel))
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1), nn.BatchNorm2d(outchannel))

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(nn.Conv2d(3, self.inchannel, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(self.inchannel), nn.ReLU(inplace=True))
        self.ResidualBlock = ResidualBlock
        self.layer1 = self.make_layer(self.ResidualBlock, self.inchannel, 2, stride=2)
        self.layer2 = self.make_layer(self.ResidualBlock, 32, 1, stride=1)
        self.layer3 = self.make_layer(self.ResidualBlock, 64, 2, stride=2)
        self.layer4 = self.make_layer(self.ResidualBlock, 128, 1, stride=1)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ResidualBlock,
     lambda: ([], {'inchannel': 4, 'outchannel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UnsuperShortcut,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (UnsuperVgg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (UnsuperVggTiny,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

