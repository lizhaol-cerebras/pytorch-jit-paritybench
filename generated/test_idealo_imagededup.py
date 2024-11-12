
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


from typing import Dict


from typing import List


from typing import Optional


from typing import Union


import warnings


import numpy as np


import torch


from typing import Callable


from typing import Tuple


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from typing import NamedTuple


import torch.nn as nn


from torchvision import models


from torchvision.transforms import transforms


from torchvision.models import vit_b_16


from torchvision.models import EfficientNet_B4_Weights


from torchvision.models.vision_transformer import ViT_B_16_Weights


class MobilenetV3(torch.nn.Module):
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    name = 'mobilenet_v3_small'

    def __init__(self) ->None:
        """
        Initialize a mobilenetv3 model, cuts it at the global average pooling layer and returns the output features.
        """
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1').eval()
        self.mobilenet_gap_op = torch.nn.Sequential(mobilenet.features, mobilenet.avgpool)

    def forward(self, x) ->torch.tensor:
        x = self.mobilenet_gap_op(x)
        x = x.squeeze(dim=3).squeeze(dim=2)
        return x


class ViT(torch.nn.Module):
    transform = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
    name = 'vit_b_16'

    def __init__(self) ->None:
        """
        Initialize a ViT model, takes mean of the final encoder layer outputs and returns those as features for a given image.
        """
        super().__init__()
        self.model = vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1').eval()
        self.hidden_dim = 768
        self.patch_size = 16
        self.image_size = 384
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

    def _process_input(self, x: 'torch.Tensor') ->torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f'Wrong image height! Expected {self.image_size} but got {h}!')
        torch._assert(w == self.image_size, f'Wrong image width! Expected {self.image_size} but got {w}!')
        n_h = h // p
        n_w = w // p
        x = self.model.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x) ->torch.tensor:
        x = self._process_input(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.model.encoder(x)
        x = x.mean(dim=1)
        return x


class EfficientNet(torch.nn.Module):
    transform = EfficientNet_B4_Weights.IMAGENET1K_V1.transforms()
    name = 'efficientnet_b4'

    def __init__(self) ->None:
        """
        Initializes an EfficientNet model, cuts it at the global average pooling layer and returns the output features.
        """
        super().__init__()
        self.effnet_b4 = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1').eval()

    def forward(self, x) ->torch.tensor:
        x = self.effnet_b4.features(x)
        x = self.effnet_b4.avgpool(x)
        return x.squeeze(dim=3).squeeze(dim=2)


class CallModel(torch.nn.Module):
    transform = transforms.Compose([transforms.ToTensor()])
    name = 'test_custom_model'

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x


class ForwardModel(torch.nn.Module):
    transform = transforms.Compose([transforms.ToTensor()])
    name = 'test_custom_model'

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CallModel,
     lambda: ([], {}),
     lambda: ([], {'x': 4})),
    (EfficientNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ForwardModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MobilenetV3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

