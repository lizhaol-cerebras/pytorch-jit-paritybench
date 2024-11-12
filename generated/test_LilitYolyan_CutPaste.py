
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


from typing import Any


from typing import Dict


from typing import Tuple


from typing import Union


from sklearn.neighbors import KernelDensity


from sklearn.metrics import roc_curve


from sklearn.metrics import auc


from sklearn.preprocessing import normalize


from sklearn.manifold import TSNE


import matplotlib.pyplot as plt


from sklearn.utils import shuffle


from torch.utils.data import DataLoader


import numpy


import torch


import numpy as np


from torch.utils.data import Dataset


from torchvision import datasets


import torchvision.transforms as transforms


import math


from scipy import signal


from torchvision import models


import torch.nn as nn


from torchvision.models.feature_extraction import create_feature_extractor


from torch.nn import functional as F


from torchvision import transforms


import torch.utils.data


from torch import optim


from torch.distributions import MultivariateNormal


from torch.distributions import Normal


from torch.distributions.distribution import Distribution


from typing import Iterable


from typing import List


from typing import Optional


import torch.nn.functional as F


from torchvision.models.feature_extraction import get_graph_node_names


class _CutPasteNetBase(nn.Module):

    def __init__(self, encoder='resnet18', pretrained=True, dims=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_class=3):
        super().__init__()
        self.encoder = getattr(models, encoder)(pretrained=pretrained)
        last_layer = list(self.encoder.named_modules())[-1][0].split('.')[0]
        setattr(self.encoder, last_layer, nn.Identity())
        proj_layers = []
        for d in dims[:-1]:
            proj_layers.append(nn.Linear(d, d, bias=False)),
            proj_layers.append(nn.BatchNorm1d(d)),
            proj_layers.append(nn.ReLU(inplace=True))
        embeds = nn.Linear(dims[-2], dims[-1], bias=num_class > 0)
        proj_layers.append(embeds)
        self.head = nn.Sequential(*proj_layers)
        self.out = nn.Linear(dims[-1], num_class)

    def forward(self, x):
        features = self.encoder(x)
        embeds = self.head(features)
        logits = self.out(embeds)
        return logits

    def freeze(self, layer_name):
        check = False
        for name, param in self.encoder.named_parameters():
            if name == layer_name:
                check = True
            if not check and param.requires_grad != False:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def create_graph_model(self):
        return create_feature_extractor(model=self, return_nodes=['head', 'out'])


class CutPasteNet(_CutPasteNetBase):

    def __init__(self, encoder='resnet18', pretrained=True, dims=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_class=3):
        super().__init__(encoder, pretrained, dims, num_class)
        return

    def forward(self, x):
        features = self.encoder(x)
        embeds = self.head(features)
        logits = self.out(embeds)
        return logits, embeds


class GradCam(torch.nn.Module):

    def __init__(self, model: 'torch.nn.Module', name_layer: 'str') ->None:
        """GradCam

        Args:
            model (torch.nn.Module): input model.
            name_layer (str): node name of layer interested in
        """
        super().__init__()
        self.model = model
        self.model.eval()
        names_mid = name_layer.split('.')
        layer = model
        for name in names_mid:
            layer = layer.__getattr__(name)
        self.layer = layer
        self.cam = captum.attr.LayerGradCam(self.model, self.layer)
        return

    def forward(self, x: 'torch.Tensor', indices: 'Optional[Iterable[int]]'=None, with_upsample: 'bool'=False):
        """[summary]

        Args:
            x (torch.Tensor): input images, [B, C, H, W]
            indices (Optional[Iterable[int]], optional): indices of labels. Defaults to None.
            with_upsample (bool, optional): whether upsample featuremaps to image field. Defaults to False.

        Returns:
            featuremaps (torch.Tensor): output featuremaps, 
                [B, 1, H, W] if with_upsample == True
                [B, 1, _, _] if with_upsample == False
        """
        if indices is None:
            indices = self.auto_select_indices(self.model.forward(x))
        else:
            pass
        x = x.requires_grad_(True)
        featuremaps = self.cam.attribute(x, indices, relu_attributions=True)
        featuremaps = self.upsample(featuremaps, size_dst=x.shape[-2:]) if with_upsample else featuremaps
        return featuremaps

    def upsample(self, x: 'torch.Tensor', size_dst: 'Iterable[int]', method='bilinear'):
        x = F.interpolate(input=x, size=size_dst, mode=method, align_corners=True)
        return x

    @staticmethod
    def auto_select_indices(logits: 'torch.Tensor', with_softmax: 'bool'=True) ->torch.Tensor:
        """Auto selct indices of categroies with max probability.

        Args:
            logits (torch.Tensor): [B, C, ...]
            with_softmax (bool, optional): use softmax or not. Defaults to True.

        Returns:
            indices (torch.Tensor): [B, ]
        """
        props = F.softmax(logits, dim=1) if with_softmax else logits
        indices = torch.argmax(props, dim=1, keepdim=False)
        return indices

    @staticmethod
    def featuremaps_to_heatmaps(x: 'torch.Tensor') ->np.ndarray:
        """Convert featuremaps to heatmaps in BGR.

        Args:
            x (torch.Tensor): featuremaps of grad cam, [B, 1, H, W]

        Returns:
            heatmaps (np.ndarray): heatmaps, [B, H, W, C] in BGR
        """
        B, _, H, W = x.shape
        featuremaps = x.squeeze(1).detach().cpu().numpy()
        heatmaps = np.zeros((B, H, W, 3), dtype=np.uint8)
        for i_map, fmap in enumerate(featuremaps):
            hmap = cv2.normalize(fmap, None, 0, 1, cv2.NORM_MINMAX)
            hmap = cv2.convertScaleAbs(hmap, None, 255, None)
            hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
            heatmaps[i_map, :, :] = hmap
        return heatmaps

    @staticmethod
    def help(model: 'torch.nn.Module', mode: 'str'='eval') ->List[str]:
        """Show valid node names of model.

        Args:
            model (torch.nn.Module): [description]
            mode (str): "eval" or "train", Default is "eval"

        Returns:
            names (List[str]): 
                valid train node names, if mode == "train"
                valid eval node names, if mode == "eval"
        """
        names_train, names_eval = get_graph_node_names(model)
        if mode == 'eval':
            return names_eval
        elif mode == 'train':
            return names_train

    def visualize(self, x: 'torch.Tensor', indices: 'Optional[int]'=None) ->np.ndarray:
        """Visualize heatmaps on raw images.  
            
        Args:
            x (torch.Tensor): input images, [B, C, H, W] in RGB
            indices (Optional[Iterable[int]], optional): indices of labels. Defaults to None.
                if indices is None, it will be auto selected.

        Returns:
            images_show (np.ndarray): input images, [B, H, W, 3] in BGR
        """
        B, _, H, W = x.shape
        images_show = np.zeros((B, H, W, 3), dtype=np.uint8)
        images_raw = x.permute((0, 2, 3, 1))[..., [2, 1, 0]].detach().cpu().numpy()
        images_raw = (images_raw * 255).astype(np.uint8)
        heatmaps = GradCam.featuremaps_to_heatmaps(self.forward(x, indices=indices, with_upsample=True))
        images_show = cv2.addWeighted(images_raw, 0.7, heatmaps, 0.3, 0)
        return images_show


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CutPasteNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (_CutPasteNetBase,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

