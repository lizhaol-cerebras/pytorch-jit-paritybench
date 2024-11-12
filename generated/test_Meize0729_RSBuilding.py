
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


from typing import Sequence


from torch.utils.data import BatchSampler


from torch.utils.data import Sampler


import numpy as np


import torch


import logging


from collections import OrderedDict


from typing import Dict


from typing import List


from typing import Optional


from collections import defaultdict


import copy


from typing import Any


from typing import Union


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.modules.padding import ReplicationPad2d


from torchvision.models import vgg16


import torchvision


from torch import Tensor


from torch import reshape


from torch import stack


from torch.nn import Conv2d


from torch.nn import InstanceNorm2d


from torch.nn import Module


from torch.nn import ModuleList


from torch.nn import PReLU


from torch.nn import Sequential


from torch.nn import Upsample


import warnings


from torch.nn import functional as F


from torch.nn.modules.batchnorm import _BatchNorm


from torch.utils import checkpoint as cp


from typing import Tuple


from numbers import Number


from abc import ABCMeta


from abc import abstractmethod


from torch import nn


import math


from typing import Type


from torch.nn.modules.loss import _Loss


import time


import random


class FC_EF(nn.Module):
    """FC_EF segmentation network."""

    def __init__(self, in_channels, base_channel=16):
        super(FC_EF, self).__init__()
        filters = [base_channel, base_channel * 2, base_channel * 4, base_channel * 8, base_channel * 16]
        self.conv11 = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(filters[0])
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(filters[0])
        self.do12 = nn.Dropout2d(p=0.2)
        self.conv21 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(filters[1])
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(filters[1])
        self.do22 = nn.Dropout2d(p=0.2)
        self.conv31 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(filters[2])
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(filters[2])
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(filters[2])
        self.do33 = nn.Dropout2d(p=0.2)
        self.conv41 = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(filters[3])
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(filters[3])
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(filters[3])
        self.do43 = nn.Dropout2d(p=0.2)
        self.upconv4 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv43d = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(filters[3])
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(filters[3])
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(filters[2])
        self.do41d = nn.Dropout2d(p=0.2)
        self.upconv3 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv33d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(filters[2])
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(filters[2])
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(filters[1])
        self.do31d = nn.Dropout2d(p=0.2)
        self.upconv2 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv22d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(filters[1])
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(filters[0])
        self.do21d = nn.Dropout2d(p=0.2)
        self.upconv1 = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv12d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(filters[0])
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=3, padding=1)

    def forward(self, x1, x2):
        """Forward method."""
        x = torch.cat((x1, x2), 1)
        x11 = self.do11(F.relu(self.bn11(self.conv11(x))))
        x12 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12, kernel_size=2, stride=2)
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22, kernel_size=2, stride=2)
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33, kernel_size=2, stride=2)
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43, kernel_size=2, stride=2)
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43.size(3) - x4d.size(3), 0, x43.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), x43), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33.size(3) - x3d.size(3), 0, x33.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), x33), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22.size(3) - x2d.size(3), 0, x22.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), x22), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12.size(3) - x1d.size(3), 0, x12.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), x12), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)
        return x11d,


class FC_Siam_diff(nn.Module):
    """FC_Siam_diff segmentation network."""

    def __init__(self, in_channels, base_channel=16):
        super(FC_Siam_diff, self).__init__()
        filters = [base_channel, base_channel * 2, base_channel * 4, base_channel * 8, base_channel * 16]
        self.conv11 = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(filters[0])
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(filters[0])
        self.do12 = nn.Dropout2d(p=0.2)
        self.conv21 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(filters[1])
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(filters[1])
        self.do22 = nn.Dropout2d(p=0.2)
        self.conv31 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(filters[2])
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(filters[2])
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(filters[2])
        self.do33 = nn.Dropout2d(p=0.2)
        self.conv41 = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(filters[3])
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(filters[3])
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(filters[3])
        self.do43 = nn.Dropout2d(p=0.2)
        self.upconv4 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv43d = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(filters[3])
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(filters[3])
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(filters[2])
        self.do41d = nn.Dropout2d(p=0.2)
        self.upconv3 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv33d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(filters[2])
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(filters[2])
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(filters[1])
        self.do31d = nn.Dropout2d(p=0.2)
        self.upconv2 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv22d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(filters[1])
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(filters[0])
        self.do21d = nn.Dropout2d(p=0.2)
        self.upconv1 = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv12d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(filters[0])
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=3, padding=1)

    def forward(self, x1, x2):
        """Forward method."""
        x11 = self.do11(F.relu(self.bn11(self.conv11(x1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)
        x11 = self.do11(F.relu(self.bn11(self.conv11(x2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), torch.abs(x43_1 - x43_2)), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), torch.abs(x33_1 - x33_2)), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), torch.abs(x22_1 - x22_2)), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), torch.abs(x12_1 - x12_2)), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)
        return x11d,


class FC_Siam_conc(nn.Module):
    """FC_Siam_conc segmentation network."""

    def __init__(self, in_channels, base_channel=16):
        super(FC_Siam_conc, self).__init__()
        filters = [base_channel, base_channel * 2, base_channel * 4, base_channel * 8, base_channel * 16]
        self.conv11 = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(filters[0])
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(filters[0])
        self.do12 = nn.Dropout2d(p=0.2)
        self.conv21 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(filters[1])
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(filters[1])
        self.do22 = nn.Dropout2d(p=0.2)
        self.conv31 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(filters[2])
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(filters[2])
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(filters[2])
        self.do33 = nn.Dropout2d(p=0.2)
        self.conv41 = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(filters[3])
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(filters[3])
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(filters[3])
        self.do43 = nn.Dropout2d(p=0.2)
        self.upconv4 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv43d = nn.ConvTranspose2d(filters[3] + filters[4], filters[3], kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(filters[3])
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(filters[3])
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(filters[2])
        self.do41d = nn.Dropout2d(p=0.2)
        self.upconv3 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv33d = nn.ConvTranspose2d(filters[2] + filters[3], filters[2], kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(filters[2])
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(filters[2])
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(filters[1])
        self.do31d = nn.Dropout2d(p=0.2)
        self.upconv2 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv22d = nn.ConvTranspose2d(filters[1] + filters[2], filters[1], kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(filters[1])
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(filters[0])
        self.do21d = nn.Dropout2d(p=0.2)
        self.upconv1 = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv12d = nn.ConvTranspose2d(filters[0] + filters[1], filters[0], kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(filters[0])
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=3, padding=1)

    def forward(self, x1, x2):
        """Forward method."""
        x11 = self.do11(F.relu(self.bn11(self.conv11(x1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)
        x11 = self.do11(F.relu(self.bn11(self.conv11(x2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), x43_1, x43_2), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), x33_1, x33_2), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), x22_1, x22_2), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), x12_1, x12_2), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)
        return x11d,


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class Conv_CAM_Layer(nn.Module):

    def __init__(self, in_ch, out_in, use_pam=False):
        super(Conv_CAM_Layer, self).__init__()
        self.attn = nn.Sequential(nn.Conv2d(in_ch, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(), CAM_Module(32), nn.Conv2d(32, out_in, kernel_size=3, padding=1), nn.BatchNorm2d(out_in), nn.PReLU())

    def forward(self, x):
        return self.attn(x)


class FEC(nn.Module):
    """feature extraction cell"""

    def __init__(self, in_ch, mid_ch, out_ch):
        super(FEC, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class RowAttention(nn.Module):

    def __init__(self, in_dim, q_k_dim, use_pam=False):
        """
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        """
        super(RowAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        """
        b, _, h, w = x.size()
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)
        Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
        K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
        V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
        row_attn = torch.bmm(Q, K)
        row_attn = self.softmax(row_attn)
        out = torch.bmm(V, row_attn.permute(0, 2, 1))
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)
        out = self.gamma * out + x
        return out


class ColAttention(nn.Module):

    def __init__(self, in_dim, q_k_dim, use_pam=False):
        """
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        """
        super(ColAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        """
        b, _, h, w = x.size()
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)
        Q = Q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
        K = K.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        V = V.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        col_attn = torch.bmm(Q, K)
        col_attn = self.softmax(col_attn)
        out = torch.bmm(V, col_attn.permute(0, 2, 1))
        out = out.view(b, w, -1, h).permute(0, 2, 3, 1)
        out = self.gamma * out + x
        return out


class HAN(nn.Module):
    """HANet"""

    def __init__(self, in_channels, base_channel=40):
        super(HAN, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = base_channel
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        self.conv0_0 = nn.Conv2d(in_channels, n1, kernel_size=5, padding=2, stride=1)
        self.conv0 = FEC(filters[0], filters[0], filters[0])
        self.conv2 = FEC(filters[0], filters[1], filters[1])
        self.conv4 = FEC(filters[1], filters[2], filters[2])
        self.conv5 = FEC(filters[2], filters[3], filters[3])
        self.conv6 = nn.Conv2d(sum(filters), filters[1], kernel_size=1, stride=1)
        self.conv6_1_1 = nn.Conv2d(filters[0] * 2, filters[0], padding=1, kernel_size=3, groups=filters[0] // 2, dilation=1)
        self.conv6_1_2 = nn.Conv2d(filters[0] * 2, filters[0], padding=2, kernel_size=3, groups=filters[0] // 2, dilation=2)
        self.conv6_1_3 = nn.Conv2d(filters[0] * 2, filters[0], padding=3, kernel_size=3, groups=filters[0] // 2, dilation=3)
        self.conv6_1_4 = nn.Conv2d(filters[0] * 2, filters[0], padding=4, kernel_size=3, groups=filters[0] // 2, dilation=4)
        self.conv1_1 = nn.Conv2d(filters[0] * 4, filters[0], kernel_size=1, stride=1)
        self.conv6_2_1 = nn.Conv2d(filters[1] * 2, filters[1], padding=1, kernel_size=3, groups=filters[1] // 2, dilation=1)
        self.conv6_2_2 = nn.Conv2d(filters[1] * 2, filters[1], padding=2, kernel_size=3, groups=filters[1] // 2, dilation=2)
        self.conv6_2_3 = nn.Conv2d(filters[1] * 2, filters[1], padding=3, kernel_size=3, groups=filters[1] // 2, dilation=3)
        self.conv6_2_4 = nn.Conv2d(filters[1] * 2, filters[1], padding=4, kernel_size=3, groups=filters[1] // 2, dilation=4)
        self.conv2_1 = nn.Conv2d(filters[1] * 4, filters[1], kernel_size=1, stride=1)
        self.conv6_3_1 = nn.Conv2d(filters[2] * 2, filters[2], padding=1, kernel_size=3, groups=filters[2] // 2, dilation=1)
        self.conv6_3_2 = nn.Conv2d(filters[2] * 2, filters[2], padding=2, kernel_size=3, groups=filters[2] // 2, dilation=2)
        self.conv6_3_3 = nn.Conv2d(filters[2] * 2, filters[2], padding=3, kernel_size=3, groups=filters[2] // 2, dilation=3)
        self.conv6_3_4 = nn.Conv2d(filters[2] * 2, filters[2], padding=4, kernel_size=3, groups=filters[2] // 2, dilation=4)
        self.conv3_1 = nn.Conv2d(filters[2] * 4, filters[2], kernel_size=1, stride=1)
        self.conv6_4_1 = nn.Conv2d(filters[3] * 2, filters[3], padding=1, kernel_size=3, groups=filters[3] // 2, dilation=1)
        self.conv6_4_2 = nn.Conv2d(filters[3] * 2, filters[3], padding=2, kernel_size=3, groups=filters[3] // 2, dilation=2)
        self.conv6_4_3 = nn.Conv2d(filters[3] * 2, filters[3], padding=3, kernel_size=3, groups=filters[3] // 2, dilation=3)
        self.conv6_4_4 = nn.Conv2d(filters[3] * 2, filters[3], padding=4, kernel_size=3, groups=filters[3] // 2, dilation=4)
        self.conv4_1 = nn.Conv2d(filters[3] * 4, filters[3], kernel_size=1, stride=1)
        self.cam_attention_1 = Conv_CAM_Layer(filters[0], filters[0], False)
        self.cam_attention_2 = Conv_CAM_Layer(filters[1], filters[1], False)
        self.cam_attention_3 = Conv_CAM_Layer(filters[2], filters[2], False)
        self.cam_attention_4 = Conv_CAM_Layer(filters[3], filters[3], False)
        self.row_attention_1 = RowAttention(filters[0], filters[0], False)
        self.row_attention_2 = RowAttention(filters[1], filters[1], False)
        self.row_attention_3 = RowAttention(filters[2], filters[2], False)
        self.row_attention_4 = RowAttention(filters[3], filters[3], False)
        self.col_attention_1 = ColAttention(filters[0], filters[0], False)
        self.col_attention_2 = ColAttention(filters[1], filters[1], False)
        self.col_attention_3 = ColAttention(filters[2], filters[2], False)
        self.col_attention_4 = ColAttention(filters[3], filters[3], False)
        self.c4_conv = nn.Conv2d(filters[3], filters[1], kernel_size=3, padding=1)
        self.c3_conv = nn.Conv2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.c2_conv = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.c1_conv = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.Up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.Up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x1 = self.conv0(self.conv0_0(x1))
        x3 = self.conv2(self.pool(x1))
        x4 = self.conv4(self.pool(x3))
        A_F4 = self.conv5(self.pool(x4))
        x2 = self.conv0(self.conv0_0(x2))
        x5 = self.conv2(self.pool(x2))
        x6 = self.conv4(self.pool(x5))
        A_F8 = self.conv5(self.pool(x6))
        c4_1 = self.conv4_1(torch.cat([self.conv6_4_1(torch.cat([A_F4, A_F8], 1)), self.conv6_4_2(torch.cat([A_F4, A_F8], 1)), self.conv6_4_3(torch.cat([A_F4, A_F8], 1)), self.conv6_4_4(torch.cat([A_F4, A_F8], 1))], 1))
        c4 = self.cam_attention_4(c4_1) + self.row_attention_4(self.col_attention_4(c4_1))
        c3_1 = self.conv3_1(torch.cat([self.conv6_3_1(torch.cat([x4, x6], 1)), self.conv6_3_2(torch.cat([x4, x6], 1)), self.conv6_3_3(torch.cat([x4, x6], 1)), self.conv6_3_4(torch.cat([x4, x6], 1))], 1))
        c3 = torch.cat([self.cam_attention_3(c3_1) + self.row_attention_3(self.col_attention_3(c3_1)), self.Up1(c4)], 1)
        c2_1 = self.conv2_1(torch.cat([self.conv6_2_1(torch.cat([x3, x5], 1)), self.conv6_2_2(torch.cat([x3, x5], 1)), self.conv6_2_3(torch.cat([x3, x5], 1)), self.conv6_2_4(torch.cat([x3, x5], 1))], 1))
        c2 = torch.cat([self.cam_attention_2(c2_1) + self.row_attention_2(self.col_attention_2(c2_1)), self.Up1(c3)], 1)
        c1_1 = self.conv1_1(torch.cat([self.conv6_1_1(torch.cat([x1, x2], 1)), self.conv6_1_2(torch.cat([x1, x2], 1)), self.conv6_1_3(torch.cat([x1, x2], 1)), self.conv6_1_4(torch.cat([x1, x2], 1))], 1))
        c1 = torch.cat([self.cam_attention_1(c1_1) + self.row_attention_1(self.col_attention_1(c1_1)), self.Up1(c2)], 1)
        out1 = self.conv6(c1)
        return out1,


def get_act_layer():
    return nn.ReLU


def make_act(*args, **kwargs):
    act_layer = get_act_layer()
    return act_layer(*args, **kwargs)


def get_norm_layer():
    return nn.BatchNorm2d


def make_norm(*args, **kwargs):
    norm_layer = get_norm_layer()
    return norm_layer(*args, **kwargs)


class BasicConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, pad_mode='Zero', bias='auto', norm=False, act=False, **kwargs):
        super().__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(getattr(nn, pad_mode.capitalize() + 'Pad2d')(kernel_size // 2))
        seq.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=0, bias=(False if norm else True) if bias == 'auto' else bias, **kwargs))
        if norm:
            if norm is True:
                norm = make_norm(out_ch)
            seq.append(norm)
        if act:
            if act is True:
                act = make_act()
            seq.append(act)
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Conv1x1(BasicConv):

    def __init__(self, in_ch, out_ch, pad_mode='Zero', bias='auto', norm=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 1, pad_mode=pad_mode, bias=bias, norm=norm, act=act, **kwargs)


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = BasicConv(2, 1, kernel_size, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return F.sigmoid(x)


class VGG16FeaturePicker(nn.Module):

    def __init__(self, indices=(3, 8, 15, 22, 29)):
        super().__init__()
        features = list(vgg16(pretrained=True).features)[:30]
        self.features = nn.ModuleList(features).eval()
        self.indices = set(indices)

    def forward(self, x):
        picked_feats = []
        for idx, model in enumerate(self.features):
            x = model(x)
            if idx in self.indices:
                picked_feats.append(x)
        return picked_feats


def conv2d_bn(in_ch, out_ch, with_dropout=True):
    lst = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), nn.PReLU(), make_norm(out_ch)]
    if with_dropout:
        lst.append(nn.Dropout(p=0.6))
    return nn.Sequential(*lst)


class IFN(nn.Module):

    def __init__(self, use_dropout=False):
        super().__init__()
        self.encoder1 = self.encoder2 = VGG16FeaturePicker()
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()
        self.ca1 = ChannelAttention(in_ch=1024)
        self.bn_ca1 = make_norm(1024)
        self.o1_conv1 = conv2d_bn(1024, 512, use_dropout)
        self.o1_conv2 = conv2d_bn(512, 512, use_dropout)
        self.bn_sa1 = make_norm(512)
        self.o1_conv3 = Conv1x1(512, 1)
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.ca2 = ChannelAttention(in_ch=1536)
        self.bn_ca2 = make_norm(1536)
        self.o2_conv1 = conv2d_bn(1536, 512, use_dropout)
        self.o2_conv2 = conv2d_bn(512, 256, use_dropout)
        self.o2_conv3 = conv2d_bn(256, 256, use_dropout)
        self.bn_sa2 = make_norm(256)
        self.o2_conv4 = Conv1x1(256, 1)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.ca3 = ChannelAttention(in_ch=768)
        self.o3_conv1 = conv2d_bn(768, 256, use_dropout)
        self.o3_conv2 = conv2d_bn(256, 128, use_dropout)
        self.o3_conv3 = conv2d_bn(128, 128, use_dropout)
        self.bn_sa3 = make_norm(128)
        self.o3_conv4 = Conv1x1(128, 1)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.ca4 = ChannelAttention(in_ch=384)
        self.o4_conv1 = conv2d_bn(384, 128, use_dropout)
        self.o4_conv2 = conv2d_bn(128, 64, use_dropout)
        self.o4_conv3 = conv2d_bn(64, 64, use_dropout)
        self.bn_sa4 = make_norm(64)
        self.o4_conv4 = Conv1x1(64, 1)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.ca5 = ChannelAttention(in_ch=192)
        self.o5_conv1 = conv2d_bn(192, 64, use_dropout)
        self.o5_conv2 = conv2d_bn(64, 32, use_dropout)
        self.o5_conv3 = conv2d_bn(32, 16, use_dropout)
        self.bn_sa5 = make_norm(16)
        self.o5_conv4 = Conv1x1(16, 1)

    def forward(self, t1, t2):
        with torch.no_grad():
            self.encoder1.eval(), self.encoder2.eval()
            t1_feats = self.encoder1(t1)
            t2_feats = self.encoder2(t2)
        t1_f_l3, t1_f_l8, t1_f_l15, t1_f_l22, t1_f_l29 = t1_feats
        t2_f_l3, t2_f_l8, t2_f_l15, t2_f_l22, t2_f_l29 = t2_feats
        x = torch.cat([t1_f_l29, t2_f_l29], dim=1)
        x = self.o1_conv1(x)
        x = self.o1_conv2(x)
        x = self.sa1(x) * x
        x = self.bn_sa1(x)
        out1 = self.o1_conv3(x)
        x = self.trans_conv1(x)
        x = torch.cat([x, t1_f_l22, t2_f_l22], dim=1)
        x = self.ca2(x) * x
        x = self.o2_conv1(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa2(x) * x
        x = self.bn_sa2(x)
        out2 = self.o2_conv4(x)
        x = self.trans_conv2(x)
        x = torch.cat([x, t1_f_l15, t2_f_l15], dim=1)
        x = self.ca3(x) * x
        x = self.o3_conv1(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa3(x) * x
        x = self.bn_sa3(x)
        out3 = self.o3_conv4(x)
        x = self.trans_conv3(x)
        x = torch.cat([x, t1_f_l8, t2_f_l8], dim=1)
        x = self.ca4(x) * x
        x = self.o4_conv1(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa4(x) * x
        x = self.bn_sa4(x)
        out4 = self.o4_conv4(x)
        x = self.trans_conv4(x)
        x = torch.cat([x, t1_f_l3, t2_f_l3], dim=1)
        x = self.ca5(x) * x
        x = self.o5_conv1(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa5(x) * x
        x = self.bn_sa5(x)
        out5 = self.o5_conv4(x)
        return out1, out2, out3, out4, out5


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class up(nn.Module):

    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class SNUNet_ECAM(nn.Module):

    def __init__(self, in_channels, base_channel=32):
        super(SNUNet_ECAM, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = base_channel
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0_0 = conv_block_nested(in_channels, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.Up4_0 = up(filters[4])
        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.Up3_1 = up(filters[3])
        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])
        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])
        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])
        self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xA, xB):
        """xA"""
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        """xB"""
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))
        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))
        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        return out,


class PixelwiseLinear(Module):

    def __init__(self, fin: 'List[int]', fout: 'List[int]', last_activation: 'Module'=None) ->None:
        assert len(fout) == len(fin)
        super().__init__()
        n = len(fin)
        self._linears = Sequential(*[Sequential(Conv2d(fin[i], fout[i], kernel_size=1, bias=True), PReLU() if i < n - 1 or last_activation is None else last_activation) for i in range(n)])

    def forward(self, x: 'Tensor') ->Tensor:
        return self._linears(x)


class MixingBlock(Module):

    def __init__(self, ch_in: 'int', ch_out: 'int'):
        super().__init__()
        self._convmix = Sequential(Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1), PReLU(), InstanceNorm2d(ch_out))

    def forward(self, x: 'Tensor', y: 'Tensor') ->Tensor:
        mixed = stack((x, y), dim=2)
        mixed = reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))
        return self._convmix(mixed)


class MixingMaskAttentionBlock(Module):
    """use the grouped convolution to make a sort of attention"""

    def __init__(self, ch_in: 'int', ch_out: 'int', fin: 'List[int]', fout: 'List[int]', generate_masked: 'bool'=False):
        super().__init__()
        self._mixing = MixingBlock(ch_in, ch_out)
        self._linear = PixelwiseLinear(fin, fout)
        self._final_normalization = InstanceNorm2d(ch_out) if generate_masked else None
        self._mixing_out = MixingBlock(ch_in, ch_out) if generate_masked else None

    def forward(self, x: 'Tensor', y: 'Tensor') ->Tensor:
        z_mix = self._mixing(x, y)
        z = self._linear(z_mix)
        z_mix_out = 0 if self._mixing_out is None else self._mixing_out(x, y)
        return z if self._final_normalization is None else self._final_normalization(z_mix_out * z)


class UpMask(Module):

    def __init__(self, scale_factor: 'float', nin: 'int', nout: 'int'):
        super().__init__()
        self._upsample = Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self._convolution = Sequential(Conv2d(nin, nin, 3, 1, groups=nin, padding=1), PReLU(), InstanceNorm2d(nin), Conv2d(nin, nout, kernel_size=1, stride=1), PReLU(), InstanceNorm2d(nout))

    def forward(self, x: 'Tensor', y: 'Optional[Tensor]'=None) ->Tensor:
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)


def _get_backbone(bkbn_name, pretrained, output_layer_bkbn, freeze_backbone) ->ModuleList:
    entire_model = getattr(torchvision.models, bkbn_name)(pretrained=pretrained).features
    derived_model = ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model


class TinyCD(Module):

    def __init__(self, in_channels, bkbn_name='efficientnet_b4', pretrained=True, output_layer_bkbn='3', freeze_backbone=False):
        super().__init__()
        self._backbone = _get_backbone(bkbn_name, pretrained, output_layer_bkbn, freeze_backbone)
        self._first_mix = MixingMaskAttentionBlock(6, 3, [3, 10, 5], [10, 5, 1])
        self._mixing_mask = ModuleList([MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]), MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]), MixingBlock(112, 56)])
        self._up = ModuleList([UpMask(2, 56, 64), UpMask(2, 64, 64), UpMask(2, 64, 32)])
        self._classify = PixelwiseLinear([32, 16], [16, 1], None)

    def forward(self, x1: 'Tensor', x2: 'Tensor') ->Tensor:
        features = self._encode(x1, x2)
        latents = self._decode(features)
        out = self._classify(latents)
        return out,

    def _encode(self, ref, test) ->List[Tensor]:
        features = [self._first_mix(ref, test)]
        for num, layer in enumerate(self._backbone):
            ref, test = layer(ref), layer(test)
            if num != 0:
                features.append(self._mixing_mask[num - 1](ref, test))
        return features

    def _decode(self, features) ->Tensor:
        upping = features[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping = self._up[i](upping, features[j])
        return upping


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(self, embedding_dim: 'int', num_heads: 'int', downsample_rate: 'int'=1) ->None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, 'num_heads must divide embedding_dim.'
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: 'Tensor', num_heads: 'int') ->Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: 'Tensor') ->Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: 'Tensor', k: 'Tensor', v: 'Tensor') ->Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class CrossAttention(nn.Module):

    def __init__(self, in_dims, embed_dims, num_heads, dropout_rate=0.0, apply_softmax=True):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = in_dims ** -0.5
        self.apply_softmax = apply_softmax
        self.to_q = nn.Linear(in_dims, embed_dims, bias=False)
        self.to_k = nn.Linear(in_dims, embed_dims, bias=False)
        self.to_v = nn.Linear(in_dims, embed_dims, bias=False)
        self.fc_out = nn.Sequential(nn.Linear(embed_dims, in_dims), nn.Dropout(dropout_rate))

    def forward(self, x, ref):
        b, n = x.shape[:2]
        h = self.num_heads
        q = self.to_q(x)
        k = self.to_k(ref)
        v = self.to_v(ref)
        q = q.reshape((b, n, h, -1)).permute((0, 2, 1, 3))
        k = k.reshape((b, ref.shape[1], h, -1)).permute((0, 2, 1, 3))
        v = v.reshape((b, ref.shape[1], h, -1)).permute((0, 2, 1, 3))
        mult = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if self.apply_softmax:
            mult = F.softmax(mult, dim=-1)
        out = torch.matmul(mult, v)
        out = out.permute((0, 2, 1, 3)).flatten(2)
        return self.fc_out(out)


class FeedForward(nn.Sequential):

    def __init__(self, dim, hidden_dim, dropout_rate=0.0):
        super().__init__(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, dim), nn.Dropout(dropout_rate))


class TransformerEncoder(nn.Module):

    def __init__(self, in_dims, embed_dims, num_heads, drop_rate, norm_cfg, apply_softmax=True):
        super(TransformerEncoder, self).__init__()
        self.attn = CrossAttention(in_dims, embed_dims, num_heads, dropout_rate=drop_rate, apply_softmax=apply_softmax)
        self.ff = FeedForward(in_dims, embed_dims, drop_rate)
        self.norm1 = build_norm_layer(norm_cfg, in_dims)[1]
        self.norm2 = build_norm_layer(norm_cfg, in_dims)[1]

    def forward(self, x):
        x_ = self.attn(self.norm1(x), self.norm1(x)) + x
        y = self.ff(self.norm2(x_)) + x_
        return y


class TransformerDecoder(nn.Module):

    def __init__(self, in_dims, embed_dims, num_heads, drop_rate, norm_cfg, apply_softmax=True):
        super(TransformerDecoder, self).__init__()
        self.attn = CrossAttention(in_dims, embed_dims, num_heads, dropout_rate=drop_rate, apply_softmax=apply_softmax)
        self.ff = FeedForward(in_dims, embed_dims, drop_rate)
        self.norm1 = build_norm_layer(norm_cfg, in_dims)[1]
        self.norm1_ = build_norm_layer(norm_cfg, in_dims)[1]
        self.norm2 = build_norm_layer(norm_cfg, in_dims)[1]

    def forward(self, x, ref):
        x_ = self.attn(self.norm1(x), self.norm1_(ref)) + x
        y = self.ff(self.norm2(x_)) + x_
        return y


class BAM(nn.Module):
    """ Basic self-attention module
    """

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in // 8
        self.activation = activation
        self.ds = ds
        self.pool = nn.AvgPool2d(self.ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        x = self.pool(input)
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        energy = self.key_channel ** -0.5 * energy
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = F.interpolate(out, [width * self.ds, height * self.ds])
        out = out + input
        return out


class _PAMBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input/Output:
        N * C  *  H  *  (2*W)
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to partition the input feature maps
        ds                : downsampling scale
    """

    def __init__(self, in_channels, key_channels, value_channels, scale=1, ds=1):
        super(_PAMBlock, self).__init__()
        self.scale = scale
        self.ds = ds
        self.pool = nn.AvgPool2d(self.ds)
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.f_key = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(self.key_channels))
        self.f_query = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(self.key_channels))
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        x = input
        if self.ds != 1:
            x = self.pool(input)
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3) // 2
        local_y = []
        local_x = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == self.scale - 1:
                    end_x = h
                if j == self.scale - 1:
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)
        value = torch.stack([value[:, :, :, :w], value[:, :, :, w:]], 4)
        query = torch.stack([query[:, :, :, :w], query[:, :, :, w:]], 4)
        key = torch.stack([key[:, :, :, :w], key[:, :, :, w:]], 4)
        local_block_cnt = 2 * self.scale * self.scale

        def func(value_local, query_local, key_local):
            batch_size_new = value_local.size(0)
            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size_new, self.value_channels, -1)
            query_local = query_local.contiguous().view(batch_size_new, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size_new, self.key_channels, -1)
            sim_map = torch.bmm(query_local, key_local)
            sim_map = self.key_channels ** -0.5 * sim_map
            sim_map = F.softmax(sim_map, dim=-1)
            context_local = torch.bmm(value_local, sim_map.permute(0, 2, 1))
            context_local = context_local.view(batch_size_new, self.value_channels, h_local, w_local, 2)
            return context_local
        v_list = [value[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in range(0, local_block_cnt, 2)]
        v_locals = torch.cat(v_list, dim=0)
        q_list = [query[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in range(0, local_block_cnt, 2)]
        q_locals = torch.cat(q_list, dim=0)
        k_list = [key[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in range(0, local_block_cnt, 2)]
        k_locals = torch.cat(k_list, dim=0)
        context_locals = func(v_locals, q_locals, k_locals)
        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                left = batch_size * (j + i * self.scale)
                right = batch_size * (j + i * self.scale) + batch_size
                tmp = context_locals[left:right]
                row_tmp.append(tmp)
            context_list.append(torch.cat(row_tmp, 3))
        context = torch.cat(context_list, 2)
        context = torch.cat([context[:, :, :, :, 0], context[:, :, :, :, 1]], 3)
        if self.ds != 1:
            context = F.interpolate(context, [h * self.ds, 2 * w * self.ds])
        return context


class PAMBlock(_PAMBlock):

    def __init__(self, in_channels, key_channels=None, value_channels=None, scale=1, ds=1):
        if key_channels == None:
            key_channels = in_channels // 8
        if value_channels == None:
            value_channels = in_channels
        super(PAMBlock, self).__init__(in_channels, key_channels, value_channels, scale, ds)


class PAM(nn.Module):
    """
        PAM module
    """

    def __init__(self, in_channels, out_channels, sizes=[1], ds=1):
        super(PAM, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.ds = ds
        self.value_channels = out_channels
        self.key_channels = out_channels // 8
        self.stages = nn.ModuleList([self._make_stage(in_channels, self.key_channels, self.value_channels, size, self.ds) for size in sizes])
        self.conv_bn = nn.Sequential(nn.Conv2d(in_channels * self.group, out_channels, kernel_size=1, padding=0, bias=False))

    def _make_stage(self, in_channels, key_channels, value_channels, size, ds):
        return PAMBlock(in_channels, key_channels, value_channels, size, ds)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = []
        for i in range(0, len(priors)):
            context += [priors[i]]
        output = self.conv_bn(torch.cat(context, 1))
        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class CDSA(nn.Module):
    """self attention module for change detection
    """

    def __init__(self, in_c, ds=1, mode='BAM'):
        super(CDSA, self).__init__()
        self.in_C = in_c
        self.ds = ds
        self.mode = mode
        if self.mode == 'BAM':
            self.Self_Att = BAM(self.in_C, ds=self.ds)
        elif self.mode == 'PAM':
            self.Self_Att = PAM(in_channels=self.in_C, out_channels=self.in_C, sizes=[1, 2, 4, 8], ds=self.ds)
        elif self.mode == 'None':
            self.Self_Att = nn.Identity()
        self.apply(weights_init)

    def forward(self, x1, x2):
        height = x1.shape[3]
        x = torch.cat((x1, x2), 3)
        x = self.Self_Att(x)
        return x[:, :, :, 0:height], x[:, :, :, height:]


class MLP(nn.Module):

    def __init__(self, input_dim: 'int', hidden_dim: 'int', output_dim: 'int', num_layers: 'int', drop: 'float'=0.0, sigmoid_output: 'bool'=False) ->None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            x = self.drop(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels: 'int', eps: 'float'=1e-06) ->None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLPBlock(nn.Module):

    def __init__(self, embedding_dim: 'int', mlp_dim: 'int', act: 'Type[nn.Module]'=nn.GELU) ->None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class TwoWayAttentionBlock(nn.Module):

    def __init__(self, embedding_dim: 'int', num_heads: 'int', mlp_dim: 'int'=2048, activation: 'Type[nn.Module]'=nn.ReLU, attention_downsample_rate: 'int'=2, skip_first_layer_pe: 'bool'=False) ->None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: 'Tensor', keys: 'Tensor', query_pe: 'Tensor', key_pe: 'Tensor') ->Tuple[Tensor, Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class TwoWayTransformer(nn.Module):

    def __init__(self, depth: 'int', embedding_dim: 'int', num_heads: 'int', mlp_dim: 'int', activation: 'Type[nn.Module]'=nn.ReLU, attention_downsample_rate: 'int'=2) ->None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_dim=mlp_dim, activation=activation, attention_downsample_rate=attention_downsample_rate, skip_first_layer_pe=i == 0))
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: 'Tensor', image_pe: 'Tensor', point_embedding: 'Tensor') ->Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class TwoWayTransformer_fpn(nn.Module):

    def __init__(self, depth: 'int', embedding_dim, num_heads: 'int', mlp_dim: 'int', activation: 'Type[nn.Module]'=nn.ReLU, attention_downsample_rate: 'int'=2) ->None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(embedding_dim=embedding_dim[i], num_heads=num_heads, mlp_dim=mlp_dim, activation=activation, attention_downsample_rate=attention_downsample_rate, skip_first_layer_pe=0))

    def forward(self, image_embedding, image_pe: 'Tensor', point_embedding: 'Tensor') ->Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        queries = point_embedding
        results = []
        for i, layer in enumerate(self.layers):
            queries, keys = layer(queries=queries, keys=image_embedding[i], query_pe=point_embedding, key_pe=image_pe[i])
            b, n, c = keys.shape
            results.append(keys)
        return queries, results


class TwoWayTransformer_fpn_swin(nn.Module):

    def __init__(self, depth: 'int', embedding_dim, num_heads: 'int', mlp_dim: 'int', activation: 'Type[nn.Module]'=nn.ReLU, attention_downsample_rate: 'int'=2) ->None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(embedding_dim=embedding_dim[i], num_heads=num_heads, mlp_dim=mlp_dim, activation=activation, attention_downsample_rate=attention_downsample_rate, skip_first_layer_pe=0))

    def forward(self, image_embedding, image_pe: 'Tensor', point_embedding: 'Tensor') ->Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        queries = point_embedding
        results = []
        for i, layer in enumerate(self.layers):
            queries, keys = layer(queries=queries, keys=image_embedding[i], query_pe=point_embedding, key_pe=image_pe[i])
            b, n, c = keys.shape
            results.append(keys.permute(0, 2, 1).reshape(b, c, int(math.sqrt(n)), int(math.sqrt(n))))
        return queries, results


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: 'int'=64, scale: 'Optional[float]'=None) ->None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer('positional_encoding_gaussian_matrix', scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: 'torch.Tensor') ->torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        coords = 2 * coords - 1
        coords = coords.float() @ self.positional_encoding_gaussian_matrix.float()
        coords = 2 * np.pi * coords
        coords = coords.half()
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: 'Tuple[int, int]') ->torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: 'Any' = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(self, coords_input: 'torch.Tensor', image_size: 'Tuple[int, int]') ->torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)


def bcl_loss(pred, target, margin=2.0, eps=0.0001, ignore_index=255, **kwargs):
    pred = pred.squeeze()
    target = target.squeeze()
    assert pred.size() == target.size() and target.numel() > 0
    mask = (target != ignore_index).float()
    target = target * mask
    utarget = 1 - target
    n_u = utarget.sum() + eps
    n_c = target.sum() + eps
    loss = torch.sum(utarget * torch.pow(pred, 2) * mask) / n_u + torch.sum(target * torch.pow(torch.clamp(margin - pred, min=0.0), 2)) / n_c
    return loss


class BCLLoss(nn.Module):
    """Batch-balanced Contrastive Loss"""

    def __init__(self, margin=2.0, loss_weight=1.0, ignore_index=255, loss_name='bcl_loss', **kwargs):
        super().__init__()
        self.margin = margin
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, pred, target, **kwargs):
        loss = self.loss_weight * bcl_loss(pred, target, self.margin, self.ignore_index)
        return loss

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


BINARY_MODE = 'binary'


MULTICLASS_MODE = 'multiclass'


MULTILABEL_MODE = 'multilabel'


def soft_dice_score(output: 'torch.Tensor', target: 'torch.Tensor', smooth: 'float'=0.0, eps: 'float'=1e-07, dims=None) ->torch.Tensor:
    """

    :param output:
    :param target:
    :param smooth:
    :param eps:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means any number
            of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


def to_tensor(x, dtype=None) ->torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray) and x.dtype.kind not in {'O', 'M', 'U', 'S'}:
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    raise ValueError('Unsupported input type' + str(type(x)))


class DiceLoss(_Loss):
    """
    Implementation of Dice loss for image segmentation task.
    It supports binary, multiclass and multilabel cases
    """

    def __init__(self, mode: 'str'='binary', classes: 'List[int]'=None, log_loss=False, from_logits=True, smooth: 'float'=0.0, ignore_index=None, eps=1e-07):
        """

        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, 'Masking classes is not supported with mode=binary'
            classes = to_tensor(classes, dtype=torch.long)
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, y_pred: 'Tensor', y_true: 'Tensor') ->Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)
        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = 0, 2
        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask
        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)
                y_true = F.one_hot(y_true * mask, num_classes)
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                y_true = F.one_hot(y_true, num_classes)
                y_true = y_true.permute(0, 2, 1)
        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask
        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores
        mask = y_true.sum(dims) > 0
        loss *= mask
        if self.classes is not None:
            loss = loss[self.classes]
        return loss.mean()


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, first: 'nn.Module', second: 'nn.Module', first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)


class SoftBCEWithLogitsLoss(nn.Module):
    """
    Drop-in replacement for nn.BCEWithLogitsLoss with few additions:
    - Support of ignore_index value
    - Support of label smoothing
    """
    __constants__ = ['weight', 'pos_weight', 'reduction', 'ignore_index', 'smooth_factor']

    def __init__(self, weight=None, ignore_index: 'Optional[int]'=255, reduction='mean', smooth_factor=None, pos_weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        if self.smooth_factor is not None:
            soft_targets = ((1 - target) * self.smooth_factor + target * (1 - self.smooth_factor)).type_as(input)
        else:
            soft_targets = target.type_as(input)
        loss = F.binary_cross_entropy_with_logits(input, soft_targets, self.weight, pos_weight=self.pos_weight, reduction='none')
        if self.ignore_index is not None:
            not_ignored_mask: 'Tensor' = target != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class EdgeLoss(nn.Module):

    def __init__(self, ignore_index=255, edge_factor=10.0):
        super(EdgeLoss, self).__init__()
        self.main_loss = JointLoss(SoftBCEWithLogitsLoss(smooth_factor=0.05, ignore_index=ignore_index), DiceLoss(mode='binary', smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.edge_factor = edge_factor

    def get_boundary(self, x):
        laplacian_kernel_target = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False)
        x = F.conv2d(x, laplacian_kernel_target, padding=1)
        x = x.clamp(min=0, max=1.0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0
        return x

    def compute_edge_loss(self, logits, targets):
        bs = logits.size()[0]
        boundary_targets = self.get_boundary(targets)
        boundary_targets = boundary_targets.view(bs, 1, -1)
        logits = logits.sigmoid()
        boundary_pre = self.get_boundary(logits)
        boundary_pre = boundary_pre / (boundary_pre + 0.01)
        boundary_pre = boundary_pre.view(bs, 1, -1)
        edge_loss = F.binary_cross_entropy_with_logits(boundary_pre, boundary_targets)
        return edge_loss

    def forward(self, logits, targets):
        loss = self.main_loss(logits, targets) + self.compute_edge_loss(logits, targets) * self.edge_factor
        return loss


class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = normalized_shape,

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Attention,
     lambda: ([], {'embedding_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (BCLLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BasicConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CAM_Module,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ColAttention,
     lambda: ([], {'in_dim': 4, 'q_k_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv1x1,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv_CAM_Layer,
     lambda: ([], {'in_ch': 4, 'out_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CrossAttention,
     lambda: ([], {'in_dims': 4, 'embed_dims': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FEC,
     lambda: ([], {'in_ch': 4, 'mid_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LN2d,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm2d,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLPBlock,
     lambda: ([], {'embedding_dim': 4, 'mlp_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PixelwiseLinear,
     lambda: ([], {'fin': [4, 4], 'fout': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RowAttention,
     lambda: ([], {'in_dim': 4, 'q_k_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SoftBCEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SpatialAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UpMask,
     lambda: ([], {'scale_factor': 1.0, 'nin': 4, 'nout': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VGG16FeaturePicker,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (_PAMBlock,
     lambda: ([], {'in_channels': 4, 'key_channels': 4, 'value_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (conv_block_nested,
     lambda: ([], {'in_ch': 4, 'mid_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (up,
     lambda: ([], {'in_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

