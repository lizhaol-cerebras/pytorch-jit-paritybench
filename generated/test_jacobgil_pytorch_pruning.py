
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


import torch.backends.cudnn as cudnn


import torch.nn as nn


import torch.nn.parallel


import torch.optim as optim


import torch.utils.data as data


import torchvision.datasets as datasets


import torchvision.models as models


import torchvision.transforms as transforms


from torch.autograd import Variable


from torchvision import models


import torchvision


import torch.nn.functional as F


import time


class ModifiedVGG16Model(torch.nn.Module):

    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()
        model = models.vgg16(pretrained=True)
        self.features = model.features
        for param in self.features.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(25088, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

