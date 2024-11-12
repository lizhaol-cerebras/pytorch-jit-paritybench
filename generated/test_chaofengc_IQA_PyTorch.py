
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


import pandas as pd


from itertools import chain


import torch


from time import time


from typing import Dict


from typing import Optional


import inspect


import copy


import torch.nn as nn


import torch.nn.functional as F


from torchvision.ops.deform_conv import DeformConv2d


from collections import OrderedDict


import collections.abc


from itertools import repeat


import numpy as np


from torch import nn as nn


from torch.nn import functional as F


from torch.nn import init as init


from torch.nn.modules.batchnorm import _BatchNorm


import torchvision.transforms as T


from torch import nn


import torchvision.models


from typing import Tuple


import warnings


import scipy


import math


import torchvision as tv


from typing import Union


from typing import List


from torch.hub import get_dir


import torchvision.transforms.functional as F


import torchvision


from torchvision import models


from torchvision import transforms


from scipy import linalg


import functools


from scipy.stats import entropy


from torch.nn.modules.utils import _ntuple


from itertools import product


from collections import namedtuple


from numpy.fft import fftshift


import torch.utils.checkpoint as checkpoint


import torchvision.transforms.functional as TF


import scipy.io


import torchvision.transforms.functional as tf


from torch import Tensor


from torchvision.ops import RoIPool


from functools import partial


import torchvision.models as models


from torch import einsum


from torch.nn.functional import avg_pool2d


from torch.nn.functional import interpolate


from torch.nn.functional import pad


from typing import cast


import random


import torch.utils.data


from copy import deepcopy


from torch.utils import data as data


import torchvision.transforms as tf


from torch.utils.data.sampler import Sampler


import queue as Queue


from torch.utils.data import DataLoader


from collections.abc import Sequence


from torch import autograd as autograd


import typing


from scipy.special import factorial


import time


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


from collections import Counter


from torch.optim.lr_scheduler import _LRScheduler


import logging


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.hub import download_url_to_file


from torchvision.utils import make_grid


class DeformFusion(nn.Module):

    def __init__(self, patch_size=8, in_channels=768 * 5, cnn_channels=256 * 3, out_channels=256 * 3):
        super().__init__()
        self.d_hidn = 512
        if patch_size == 8:
            stride = 1
        else:
            stride = 2
        self.conv_offset = nn.Conv2d(in_channels, 2 * 3 * 3, 3, 1, 1)
        self.deform = DeformConv2d(cnn_channels, out_channels, 3, 1, 1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=out_channels, out_channels=self.d_hidn, kernel_size=3, padding=1, stride=2), nn.ReLU(), nn.Conv2d(in_channels=self.d_hidn, out_channels=out_channels, kernel_size=3, padding=1, stride=stride))

    def forward(self, cnn_feat, vit_feat):
        vit_feat = F.interpolate(vit_feat, size=cnn_feat.shape[-2:], mode='nearest')
        offset = self.conv_offset(vit_feat)
        deform_feat = self.deform(cnn_feat, offset)
        deform_feat = self.conv1(deform_feat)
        return deform_feat


class Pixel_Prediction(nn.Module):

    def __init__(self, inchannels=768 * 5 + 256 * 3, outchannels=256, d_hidn=1024):
        super().__init__()
        self.d_hidn = d_hidn
        self.down_channel = nn.Conv2d(inchannels, outchannels, kernel_size=1)
        self.feat_smoothing = nn.Sequential(nn.Conv2d(in_channels=256 * 3, out_channels=self.d_hidn, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(in_channels=self.d_hidn, out_channels=512, kernel_size=3, padding=1))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1), nn.ReLU())
        self.conv_attent = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1), nn.Sigmoid())
        self.conv = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1))

    def forward(self, f_dis, f_ref, cnn_dis, cnn_ref):
        f_dis = torch.cat((f_dis, cnn_dis), 1)
        f_ref = torch.cat((f_ref, cnn_ref), 1)
        f_dis = self.down_channel(f_dis)
        f_ref = self.down_channel(f_ref)
        f_cat = torch.cat((f_dis - f_ref, f_dis, f_ref), 1)
        feat_fused = self.feat_smoothing(f_cat)
        feat = self.conv1(feat_fused)
        f = self.conv(feat)
        w = self.conv_attent(feat)
        pred = (f * w).sum(dim=-1).sum(dim=-1) / w.sum(dim=-1).sum(dim=-1)
        return pred


def symm_pad(im: 'torch.Tensor', padding: 'Tuple[int, int, int, int]'):
    """Symmetric padding same as tensorflow.
    Ref: https://discuss.pytorch.org/t/symmetric-padding/19866/3
    """
    h, w = im.shape[-2:]
    left, right, top, bottom = padding
    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)

    def reflect(x, minx, maxx):
        """ Reflects an array around two points making a triangular waveform that ramps up
        and down,  allowing for pad lengths greater than the input length """
        rng = maxx - minx
        double_rng = 2 * rng
        mod = np.fmod(x - minx, double_rng)
        normed_mod = np.where(mod < 0, mod + double_rng, mod)
        out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return np.array(out, dtype=x.dtype)
    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]


to_2tuple = _ntuple(2)


def exact_padding_2d(x, kernel, stride=1, dilation=1, mode='same'):
    assert len(x.shape) == 4, f'Only support 4D tensor input, but got {x.shape}'
    kernel = to_2tuple(kernel)
    stride = to_2tuple(stride)
    dilation = to_2tuple(dilation)
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride[0])
    w2 = math.ceil(w / stride[1])
    pad_row = (h2 - 1) * stride[0] + (kernel[0] - 1) * dilation[0] + 1 - h
    pad_col = (w2 - 1) * stride[1] + (kernel[1] - 1) * dilation[1] + 1 - w
    pad_l, pad_r, pad_t, pad_b = pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2
    mode = mode if mode != 'same' else 'constant'
    if mode != 'symmetric':
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    elif mode == 'symmetric':
        x = symm_pad(x, (pad_l, pad_r, pad_t, pad_b))
    return x


class StdConv(nn.Conv2d):
    """
    Reference: https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def forward(self, x):
        x = exact_padding_2d(x, self.kernel_size, self.stride, mode='same')
        weight = self.weight
        weight = weight - weight.mean((1, 2, 3), keepdim=True)
        weight = weight / (weight.std((1, 2, 3), keepdim=True) + 1e-05)
        return F.conv2d(x, weight, self.bias, self.stride)


class Bottleneck(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):
        super().__init__()
        width = inplanes
        self.conv1 = StdConv(inplanes, width, 1, 1, bias=False)
        self.gn1 = nn.GroupNorm(32, width, eps=0.0001)
        self.conv2 = StdConv(width, width, 3, 1, bias=False)
        self.gn2 = nn.GroupNorm(32, width, eps=0.0001)
        self.conv3 = StdConv(width, outplanes, 1, 1, bias=False)
        self.gn3 = nn.GroupNorm(32, outplanes, eps=0.0001)
        self.relu = nn.ReLU(True)
        self.needs_projection = inplanes != outplanes or stride != 1
        if self.needs_projection:
            self.conv_proj = StdConv(inplanes, outplanes, 1, stride, bias=False)
            self.gn_proj = nn.GroupNorm(32, outplanes, eps=0.0001)

    def forward(self, x):
        identity = x
        if self.needs_projection:
            identity = self.gn_proj(self.conv_proj(identity))
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.gn3(self.conv3(x))
        out = self.relu(x + identity)
        return out


class SaveOutput:

    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def get_url_from_name(name: 'str', store_base: 'str'='hugging_face', base_url: 'str'=None) ->str:
    """
    Get the URL for a given file name from a specified storage base.

    Args:
        name (str): The name of the file.
        store_base (str, optional): The storage base to use. Options are "hugging_face" or "github". Default is "hugging_face".
        base_url (str, optional): Base URL to use if provided.

    Returns:
        str: The URL of the file.
    """
    if base_url is not None:
        url = f'{base_url}/{name}'
    elif store_base == 'hugging_face':
        url = hf_hub_url(repo_id='chaofengc/IQA-PyTorch-Weights', filename=name)
    elif store_base == 'github':
        url = f'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/{name}'
    return url


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    model_dir = model_dir or DEFAULT_CACHE_DIR
    os.makedirs(model_dir, exist_ok=True)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        None
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def clean_state_dict(state_dict):
    """
    Clean checkpoint by removing .module prefix from state dict if it exists from parallel training.

    Args:
        state_dict (dict): State dictionary from a model checkpoint.

    Returns:
        dict: Cleaned state dictionary.
    """
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def load_pretrained_network(net: 'torch.nn.Module', model_path: 'str', strict: 'bool'=True, weight_keys: 'str'=None) ->None:
    """
    Load a pretrained network from a given model path.

    Args:
        net (torch.nn.Module): The network to load the weights into.
        model_path (str): Path to the model weights file. Can be a URL or a local file path.
        strict (bool, optional): Whether to strictly enforce that the keys in state_dict match the keys returned by net's state_dict(). Default is True.
        weight_keys (str, optional): Specific key to extract from the state_dict. Default is None.

    Returns:
        None
    """
    if model_path.startswith('https://') or model_path.startswith('http://'):
        model_path = load_file_from_url(model_path)
    None
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    if weight_keys is not None:
        state_dict = state_dict[weight_keys]
    state_dict = clean_state_dict(state_dict)
    net.load_state_dict(state_dict, strict=strict)


def uniform_crop(input_list, crop_size, crop_num):
    """
    Crop the input_list of tensors into multiple crops with uniform steps according to input size and crop_num.

    Args:
        input_list (list or torch.Tensor): List of input tensors or a single input tensor.
        crop_size (int or tuple): Size of the crops. If int, the same size will be used for height and width.
                                  If tuple, should be (height, width).
        crop_num (int): Number of crops to generate.

    Returns:
        torch.Tensor or list of torch.Tensor: Cropped tensors. If input_list is a list, the output will be a list
                                              of cropped tensors. If input_list is a single tensor, the output will be a single tensor.
    """
    if not isinstance(input_list, collections.abc.Sequence):
        input_list = [input_list]
    b, c, h, w = input_list[0].shape
    ch, cw = to_2tuple(crop_size)
    if min(h, w) <= crop_size:
        scale_factor = (crop_size + 1) / min(h, w)
        input_list = [F.interpolate(x, scale_factor=scale_factor, mode='bilinear') for x in input_list]
        b, c, h, w = input_list[0].shape
    step_h = (h - ch) // int(np.sqrt(crop_num))
    step_w = (w - cw) // int(np.sqrt(crop_num))
    crops_list = []
    for inp in input_list:
        tmp_list = []
        for i in range(int(np.ceil(np.sqrt(crop_num)))):
            for j in range(int(np.ceil(np.sqrt(crop_num)))):
                sh = i * step_h
                sw = j * step_w
                tmp_list.append(inp[..., sh:sh + ch, sw:sw + cw])
        crops_list.append(torch.stack(tmp_list[:crop_num], dim=1).reshape(b * crop_num, c, ch, cw))
    if len(crops_list) == 1:
        crops_list = crops_list[0]
    return crops_list


class AHIQ(nn.Module):
    """
    AHIQ model implementation.

    This class implements the Attention-based Hybrid Image Quality (AHIQ) assessment network, which combines 
    ResNet50 and Vision Transformer (ViT) backbones with deformable convolution layers for enhanced image quality assessment.

    Args:
        - num_crop (int, optional): Number of crops to use for testing. Default is 20.
        - crop_size (int, optional): Size of the crops. Default is 224.
        - default_mean (list, optional): List of mean values for normalization. Default is [0.485, 0.456, 0.406].
        - default_std (list, optional): List of standard deviation values for normalization. Default is [0.229, 0.224, 0.225].
        - pretrained (bool, optional): Whether to use a pretrained model. Default is True.
        - pretrained_model_path (str, optional): Path to a pretrained model. Default is None.

    Attributes:
        - resnet50 (nn.Module): ResNet50 backbone.
        - vit (nn.Module): Vision Transformer backbone.
        - deform_net (nn.Module): Deformable fusion network.
        - regressor (nn.Module): Pixel prediction network.
        - default_mean (torch.Tensor): Mean values for normalization.
        - default_std (torch.Tensor): Standard deviation values for normalization.
        - eps (float): Small value to avoid division by zero.
        - crops (int): Number of crops to use for testing.
        - crop_size (int): Size of the crops.
    """

    def __init__(self, num_crop=20, crop_size=224, default_mean=[0.485, 0.456, 0.406], default_std=[0.229, 0.224, 0.225], pretrained=True, pretrained_model_path=None):
        super().__init__()
        self.resnet50 = timm.create_model('resnet50', pretrained=True)
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.fix_network(self.resnet50)
        self.fix_network(self.vit)
        self.deform_net = DeformFusion()
        self.regressor = Pixel_Prediction()
        self.init_saveoutput()
        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, weight_keys='params')
        elif pretrained:
            weight_path = load_file_from_url(default_model_urls['pipal'])
            checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
            self.regressor.load_state_dict(checkpoint['regressor_model_state_dict'])
            self.deform_net.load_state_dict(checkpoint['deform_net_model_state_dict'])
        self.eps = 1e-12
        self.crops = num_crop
        self.crop_size = crop_size

    def init_saveoutput(self):
        """
        Initializes the SaveOutput hook to get intermediate features.
        """
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.resnet50.modules():
            if isinstance(layer, Bottleneck):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

    def fix_network(self, model):
        """
        Fixes the network by setting all parameters to not require gradients.

        Args:
            model (nn.Module): The model to fix.
        """
        for p in model.parameters():
            p.requires_grad = False

    def preprocess(self, x):
        """
        Preprocesses the input tensor by normalizing it.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        x = (x - self.default_mean) / self.default_std
        return x

    def get_vit_feature(self, x):
        """
        Gets the intermediate features from the Vision Transformer backbone.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The intermediate features.
        """
        self.vit(x)
        feat = torch.cat((self.save_output.outputs[x.device][0][:, 1:, :], self.save_output.outputs[x.device][1][:, 1:, :], self.save_output.outputs[x.device][2][:, 1:, :], self.save_output.outputs[x.device][3][:, 1:, :], self.save_output.outputs[x.device][4][:, 1:, :]), dim=2)
        self.save_output.clear(x.device)
        return feat

    def get_resnet_feature(self, x):
        """
        Gets the intermediate features from the ResNet50 backbone.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The intermediate features.
        """
        self.resnet50(x)
        feat = torch.cat((self.save_output.outputs[x.device][0], self.save_output.outputs[x.device][1], self.save_output.outputs[x.device][2]), dim=1)
        self.save_output.clear(x.device)
        return feat

    def regress_score(self, dis, ref):
        """
        Computes the quality score for a distorted and reference image pair.

        Args:
            - dis (torch.Tensor): The distorted image.
            - ref (torch.Tensor): The reference image.

        Returns:
            torch.Tensor: The quality score.
        """
        self.resnet50.eval()
        self.vit.eval()
        dis = self.preprocess(dis)
        ref = self.preprocess(ref)
        vit_dis = self.get_vit_feature(dis)
        vit_ref = self.get_vit_feature(ref)
        B, N, C = vit_ref.shape
        H, W = 28, 28
        vit_ref = vit_ref.transpose(1, 2).view(B, C, H, W)
        vit_dis = vit_dis.transpose(1, 2).view(B, C, H, W)
        cnn_dis = self.get_resnet_feature(dis)
        cnn_ref = self.get_resnet_feature(ref)
        cnn_dis = self.deform_net(cnn_dis, vit_ref)
        cnn_ref = self.deform_net(cnn_ref, vit_ref)
        score = self.regressor(vit_dis, vit_ref, cnn_dis, cnn_ref)
        return score

    def forward(self, x, y):
        """
        Computes the quality score for a batch of distorted and reference image pairs.

        Args:
            - x (torch.Tensor): The batch of distorted images.
            - y (torch.Tensor): The batch of reference images.

        Returns:
            torch.Tensor: The quality scores.
        """
        bsz = x.shape[0]
        if self.crops > 1 and not self.training:
            x, y = uniform_crop([x, y], self.crop_size, self.crops)
            score = self.regress_score(x, y)
            score = score.reshape(bsz, self.crops, 1)
            score = score.mean(dim=1)
        else:
            score = self.regress_score(x, y)
        return score


DATASET_INFO = {'live': {'score_range': (1, 100), 'mos_type': 'dmos'}, 'csiq': {'score_range': (0, 1), 'mos_type': 'dmos'}, 'tid': {'score_range': (0, 9), 'mos_type': 'mos'}, 'kadid': {'score_range': (1, 5), 'mos_type': 'mos'}, 'koniq': {'score_range': (1, 100), 'mos_type': 'mos'}, 'clive': {'score_range': (1, 100), 'mos_type': 'mos'}, 'flive': {'score_range': (1, 100), 'mos_type': 'mos'}, 'spaq': {'score_range': (1, 100), 'mos_type': 'mos'}, 'ava': {'score_range': (1, 10), 'mos_type': 'mos'}}


IMAGENET_DEFAULT_MEAN = 0.485, 0.456, 0.406


IMAGENET_DEFAULT_STD = 0.229, 0.224, 0.225


class ARNIQA(nn.Module):
    """
    ARNIQA model implementation.

    This class implements the ARNIQA model for image quality assessment, which combines a ResNet50 encoder
    with a regressor network for predicting image quality scores.

    Args:
        regressor_dataset (str, optional): The dataset to use for the regressor. Default is "koniq".

    Attributes:
        regressor_dataset (str): The dataset to use for the regressor.
        encoder (nn.Module): The ResNet50 encoder.
        feat_dim (int): The feature dimension of the encoder.
        regressor (nn.Module): The regressor network.
        default_mean (torch.Tensor): The mean values for normalization.
        default_std (torch.Tensor): The standard deviation values for normalization.
    """

    def __init__(self, regressor_dataset: 'str'='koniq'):
        super().__init__()
        self.regressor_dataset = regressor_dataset
        self.encoder = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.feat_dim = self.encoder.fc.in_features
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        encoder_state_dict = torch.hub.load_state_dict_from_url(default_model_urls['ARNIQA'], progress=True, map_location='cpu')
        cleaned_encoder_state_dict = OrderedDict()
        for key, value in encoder_state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]
                cleaned_encoder_state_dict[new_key] = value
        self.encoder.load_state_dict(cleaned_encoder_state_dict)
        self.encoder.eval()
        self.regressor: 'nn.Module' = torch.hub.load_state_dict_from_url(default_model_urls[self.regressor_dataset], progress=True, map_location='cpu')
        self.regressor.eval()
        self.default_mean = torch.Tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)

    def forward(self, x: 'torch.Tensor') ->float:
        """
        Forward pass of the ARNIQA model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            float: The predicted quality score.
        """
        x, x_ds = self._preprocess(x)
        f = F.normalize(self.encoder(x), dim=1)
        f_ds = F.normalize(self.encoder(x_ds), dim=1)
        f_combined = torch.hstack((f, f_ds)).view(-1, self.feat_dim * 2)
        score = self.regressor(f_combined)
        score = self._scale_score(score)
        return score

    def _preprocess(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Downsample the input image with a factor of 2 and normalize the original and downsampled images.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The normalized original and downsampled tensors.
        """
        x_ds = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = (x - self.default_mean) / self.default_std
        x_ds = (x_ds - self.default_mean) / self.default_std
        return x, x_ds

    def _scale_score(self, score: 'torch.Tensor') ->torch.Tensor:
        """
        Scale the score in the range [0, 1], where higher is better.

        Args:
            score (torch.Tensor): The predicted score.

        Returns:
            torch.Tensor: The scaled score.
        """
        new_range = 0.0, 1.0
        original_range = DATASET_INFO[self.regressor_dataset]['score_range'][0], DATASET_INFO[self.regressor_dataset]['score_range'][1]
        original_width = original_range[1] - original_range[0]
        new_width = new_range[1] - new_range[0]
        scaling_factor = new_width / original_width
        scaled_score = new_range[0] + (score - original_range[0]) * scaling_factor
        if DATASET_INFO[self.regressor_dataset]['mos_type'] == 'dmos':
            scaled_score = new_range[1] - scaled_score
        return scaled_score


def estimate_aggd_param(block: 'torch.Tensor', return_sigma=False) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.
    Args:
        block (Tensor): Image block with shape (b, 1, h, w).
    Returns:
        Tensor: alpha, beta_l and beta_r for the AGGD distribution
        (Estimating the parames in Equation 7 in the paper).
    """
    gam = torch.arange(0.2, 10 + 0.001, 0.001)
    r_gam = (2 * torch.lgamma(2.0 / gam) - (torch.lgamma(1.0 / gam) + torch.lgamma(3.0 / gam))).exp()
    r_gam = r_gam.repeat(block.shape[0], 1)
    mask_left = block < 0
    mask_right = block > 0
    count_left = mask_left.sum(dim=(-1, -2), dtype=torch.float32)
    count_right = mask_right.sum(dim=(-1, -2), dtype=torch.float32)
    left_std = torch.sqrt((block * mask_left).pow(2).sum(dim=(-1, -2)) / count_left)
    right_std = torch.sqrt((block * mask_right).pow(2).sum(dim=(-1, -2)) / count_right)
    gammahat = left_std / right_std
    rhat = block.abs().mean(dim=(-1, -2)).pow(2) / block.pow(2).mean(dim=(-1, -2))
    rhatnorm = rhat * (gammahat.pow(3) + 1) * (gammahat + 1) / (gammahat.pow(2) + 1).pow(2)
    array_position = (r_gam - rhatnorm).abs().argmin(dim=-1)
    alpha = gam[array_position]
    beta_l = left_std.squeeze(-1) * (torch.lgamma(1 / alpha) - torch.lgamma(3 / alpha)).exp().sqrt()
    beta_r = right_std.squeeze(-1) * (torch.lgamma(1 / alpha) - torch.lgamma(3 / alpha)).exp().sqrt()
    if return_sigma:
        return alpha, left_std.squeeze(-1), right_std.squeeze(-1)
    else:
        return alpha, beta_l, beta_r


def compute_nss_features(luma_nrmlzd: 'torch.Tensor') ->torch.Tensor:
    alpha, betal, betar = estimate_aggd_param(luma_nrmlzd, return_sigma=False)
    features = [alpha, (betal + betar) / 2]
    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for shift in shifts:
        shifted_luma_nrmlzd = torch.roll(luma_nrmlzd, shifts=shift, dims=(-2, -1))
        alpha, betal, betar = estimate_aggd_param(luma_nrmlzd * shifted_luma_nrmlzd, return_sigma=False)
        distmean = (betar - betal) * torch.exp(torch.lgamma(2 / alpha) - torch.lgamma(1 / alpha))
        features.extend((alpha, distmean, betal, betar))
    return torch.stack(features, dim=-1)


_D = typing.Optional[torch.dtype]


def cast_input(x: 'torch.Tensor') ->typing.Tuple[torch.Tensor, _D]:
    if x.dtype != torch.float32 or x.dtype != torch.float64:
        dtype = x.dtype
        x = x.float()
    else:
        dtype = None
    return x, dtype


def cast_output(x: 'torch.Tensor', dtype: '_D') ->torch.Tensor:
    if dtype is not None:
        if not dtype.is_floating_point:
            x = x - x.detach() + x.round()
        if dtype is torch.uint8:
            x = x.clamp(0, 255)
        x = x
    return x


def reflect_padding(x: 'torch.Tensor', dim: 'int', pad_pre: 'int', pad_post: 'int') ->torch.Tensor:
    """
    Apply reflect padding to the given Tensor.
    Note that it is slightly different from the PyTorch functional.pad,
    where boundary elements are used only once.
    Instead, we follow the MATLAB implementation
    which uses boundary elements twice.
    For example,
    [a, b, c, d] would become [b, a, b, c, d, c] with the PyTorch implementation,
    while our implementation yields [a, a, b, c, d, d].
    """
    b, c, h, w = x.size()
    if dim == 2 or dim == -2:
        padding_buffer = x.new_zeros(b, c, h + pad_pre + pad_post, w)
        padding_buffer[..., pad_pre:h + pad_pre, :].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1, :].copy_(x[..., p, :])
        for p in range(pad_post):
            padding_buffer[..., h + pad_pre + p, :].copy_(x[..., -(p + 1), :])
    else:
        padding_buffer = x.new_zeros(b, c, h, w + pad_pre + pad_post)
        padding_buffer[..., pad_pre:w + pad_pre].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1].copy_(x[..., p])
        for p in range(pad_post):
            padding_buffer[..., w + pad_pre + p].copy_(x[..., -(p + 1)])
    return padding_buffer


def padding(x: 'torch.Tensor', dim: 'int', pad_pre: 'int', pad_post: 'int', padding_type: 'typing.Optional[str]'='reflect') ->torch.Tensor:
    if padding_type is None:
        return x
    elif padding_type == 'reflect':
        x_pad = reflect_padding(x, dim, pad_pre, pad_post)
    else:
        raise ValueError('{} padding is not supported!'.format(padding_type))
    return x_pad


def downsampling_2d(x: 'torch.Tensor', k: 'torch.Tensor', scale: 'int', padding_type: 'str'='reflect') ->torch.Tensor:
    c = x.size(1)
    k_h = k.size(-2)
    k_w = k.size(-1)
    k = k
    k = k.view(1, 1, k_h, k_w)
    k = k.repeat(c, c, 1, 1)
    e = torch.eye(c, dtype=k.dtype, device=k.device, requires_grad=False)
    e = e.view(c, c, 1, 1)
    k = k * e
    pad_h = (k_h - scale) // 2
    pad_w = (k_w - scale) // 2
    x = padding(x, -2, pad_h, pad_h, padding_type=padding_type)
    x = padding(x, -1, pad_w, pad_w, padding_type=padding_type)
    y = F.conv2d(x, k, padding=0, stride=scale)
    return y


_I = typing.Optional[int]


def reshape_input(x: 'torch.Tensor') ->typing.Tuple[torch.Tensor, _I, _I, int, int]:
    if x.dim() == 4:
        b, c, h, w = x.size()
    elif x.dim() == 3:
        c, h, w = x.size()
        b = None
    elif x.dim() == 2:
        h, w = x.size()
        b = c = None
    else:
        raise ValueError('{}-dim Tensor is not supported!'.format(x.dim()))
    x = x.view(-1, 1, h, w)
    return x, b, c, h, w


def reshape_output(x: 'torch.Tensor', b: '_I', c: '_I') ->torch.Tensor:
    rh = x.size(-2)
    rw = x.size(-1)
    if b is not None:
        x = x.view(b, c, rh, rw)
    elif c is not None:
        x = x.view(c, rh, rw)
    else:
        x = x.view(rh, rw)
    return x


def get_padding(base: 'torch.Tensor', kernel_size: 'int', x_size: 'int') ->typing.Tuple[int, int, torch.Tensor]:
    base = base.long()
    r_min = base.min()
    r_max = base.max() + kernel_size - 1
    if r_min <= 0:
        pad_pre = -r_min
        pad_pre = pad_pre.item()
        base += pad_pre
    else:
        pad_pre = 0
    if r_max >= x_size:
        pad_post = r_max - x_size + 1
        pad_post = pad_post.item()
    else:
        pad_post = 0
    return pad_pre, pad_post, base


def cubic_contribution(x: 'torch.Tensor', a: 'float'=-0.5) ->torch.Tensor:
    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax * ax2
    range_01 = ax.le(1)
    range_12 = torch.logical_and(ax.gt(1), ax.le(2))
    cont_01 = (a + 2) * ax3 - (a + 3) * ax2 + 1
    cont_01 = cont_01 * range_01
    cont_12 = a * ax3 - 5 * a * ax2 + 8 * a * ax - 4 * a
    cont_12 = cont_12 * range_12
    cont = cont_01 + cont_12
    return cont


def gaussian_contribution(x: 'torch.Tensor', sigma: 'float'=2.0) ->torch.Tensor:
    range_3sigma = x.abs() <= 3 * sigma + 1
    cont = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    cont = cont * range_3sigma
    return cont


def get_weight(dist: 'torch.Tensor', kernel_size: 'int', kernel: 'str'='cubic', sigma: 'float'=2.0, antialiasing_factor: 'float'=1) ->torch.Tensor:
    buffer_pos = dist.new_zeros(kernel_size, len(dist))
    for idx, buffer_sub in enumerate(buffer_pos):
        buffer_sub.copy_(dist - idx)
    buffer_pos *= antialiasing_factor
    if kernel == 'cubic':
        weight = cubic_contribution(buffer_pos)
    elif kernel == 'gaussian':
        weight = gaussian_contribution(buffer_pos, sigma=sigma)
    else:
        raise ValueError('{} kernel is not supported!'.format(kernel))
    weight /= weight.sum(dim=0, keepdim=True)
    return weight


def reshape_tensor(x: 'torch.Tensor', dim: 'int', kernel_size: 'int') ->torch.Tensor:
    if dim == 2 or dim == -2:
        k = kernel_size, 1
        h_out = x.size(-2) - kernel_size + 1
        w_out = x.size(-1)
    else:
        k = 1, kernel_size
        h_out = x.size(-2)
        w_out = x.size(-1) - kernel_size + 1
    unfold = F.unfold(x, k)
    unfold = unfold.view(unfold.size(0), -1, h_out, w_out)
    return unfold


def resize_1d(x: 'torch.Tensor', dim: 'int', size: 'int', scale: 'float', kernel: 'str'='cubic', sigma: 'float'=2.0, padding_type: 'str'='reflect', antialiasing: 'bool'=True) ->torch.Tensor:
    """
    Args:
        x (torch.Tensor): A torch.Tensor of dimension (B x C, 1, H, W).
        dim (int):
        scale (float):
        size (int):
    Return:
    """
    if scale == 1:
        return x
    if kernel == 'cubic':
        kernel_size = 4
    else:
        kernel_size = math.floor(6 * sigma)
    if antialiasing and scale < 1:
        antialiasing_factor = scale
        kernel_size = math.ceil(kernel_size / antialiasing_factor)
    else:
        antialiasing_factor = 1
    kernel_size += 2
    with torch.no_grad():
        pos = torch.linspace(0, size - 1, steps=size, dtype=x.dtype, device=x.device)
        pos = (pos + 0.5) / scale - 0.5
        base = pos.floor() - kernel_size // 2 + 1
        dist = pos - base
        weight = get_weight(dist, kernel_size, kernel=kernel, sigma=sigma, antialiasing_factor=antialiasing_factor)
        pad_pre, pad_post, base = get_padding(base, kernel_size, x.size(dim))
    x_pad = padding(x, dim, pad_pre, pad_post, padding_type=padding_type)
    unfold = reshape_tensor(x_pad, dim, kernel_size)
    if dim == 2 or dim == -2:
        sample = unfold[..., base, :]
        weight = weight.view(1, kernel_size, sample.size(2), 1)
    else:
        sample = unfold[..., base]
        weight = weight.view(1, kernel_size, 1, sample.size(3))
    x = sample * weight
    x = x.sum(dim=1, keepdim=True)
    return x


def imresize(x: 'torch.Tensor', scale: 'typing.Optional[float]'=None, sizes: 'typing.Optional[typing.Tuple[int, int]]'=None, kernel: 'typing.Union[str, torch.Tensor]'='cubic', sigma: 'float'=2, rotation_degree: 'float'=0, padding_type: 'str'='reflect', antialiasing: 'bool'=True) ->torch.Tensor:
    """
    Args:
        x (torch.Tensor):
        scale (float):
        sizes (tuple(int, int)):
        kernel (str, default='cubic'):
        sigma (float, default=2):
        rotation_degree (float, default=0):
        padding_type (str, default='reflect'):
        antialiasing (bool, default=True):
    Return:
        torch.Tensor:
    """
    if scale is None and sizes is None:
        raise ValueError('One of scale or sizes must be specified!')
    if scale is not None and sizes is not None:
        raise ValueError('Please specify scale or sizes to avoid conflict!')
    x, b, c, h, w = reshape_input(x)
    if sizes is None and scale is not None:
        """
        # Check if we can apply the convolution algorithm
        scale_inv = 1 / scale
        if isinstance(kernel, str) and scale_inv.is_integer():
            kernel = discrete_kernel(kernel, scale, antialiasing=antialiasing)
        elif isinstance(kernel, torch.Tensor) and not scale_inv.is_integer():
            raise ValueError(
                'An integer downsampling factor '
                'should be used with a predefined kernel!'
            )
        """
        sizes = math.ceil(h * scale), math.ceil(w * scale)
        scales = scale, scale
    if scale is None and sizes is not None:
        scales = sizes[0] / h, sizes[1] / w
    x, dtype = cast_input(x)
    if isinstance(kernel, str) and sizes is not None:
        x = resize_1d(x, -2, size=sizes[0], scale=scales[0], kernel=kernel, sigma=sigma, padding_type=padding_type, antialiasing=antialiasing)
        x = resize_1d(x, -1, size=sizes[1], scale=scales[1], kernel=kernel, sigma=sigma, padding_type=padding_type, antialiasing=antialiasing)
    elif isinstance(kernel, torch.Tensor) and scale is not None:
        x = downsampling_2d(x, kernel, scale=int(1 / scale))
    x = reshape_output(x, b, c)
    x = cast_output(x, dtype)
    return x


def estimate_ggd_param(x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
    """Estimate general gaussian distribution.

    Args:
        x (Tensor): shape (b, 1, h, w)
    """
    gamma = torch.arange(0.2, 10 + 0.001, 0.001)
    r_table = (torch.lgamma(1.0 / gamma) + torch.lgamma(3.0 / gamma) - 2 * torch.lgamma(2.0 / gamma)).exp()
    r_table = r_table.repeat(x.size(0), 1)
    sigma_sq = x.pow(2).mean(dim=(-1, -2))
    sigma = sigma_sq.sqrt().squeeze(dim=-1)
    assert not torch.isclose(sigma, torch.zeros_like(sigma)).all(), 'Expected image with non zero variance of pixel values'
    E = x.abs().mean(dim=(-1, -2))
    rho = sigma_sq / E ** 2
    indexes = (rho - r_table).abs().argmin(dim=-1)
    solution = gamma[indexes]
    return solution, sigma


def fspecial(size=None, sigma=None, channels=1, filter_type='gaussian'):
    """ Function same as 'fspecial' in MATLAB, only support gaussian now.
    Args:
        size (int or tuple): size of window
        sigma (float): sigma of gaussian
        channels (int): channels of output
    """
    if filter_type == 'gaussian':
        shape = to_2tuple(size)
        m, n = [((ss - 1.0) / 2.0) for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        h = torch.from_numpy(h).float().repeat(channels, 1, 1, 1)
        return h
    else:
        raise NotImplementedError(f'Only support gaussian filter now, got {filter_type}')


class ExactPadding2d(nn.Module):
    """This function calculate exact padding values for 4D tensor inputs,
    and support the same padding mode as tensorflow.

    Args:
        kernel (int or tuple): kernel size.
        stride (int or tuple): stride size.
        dilation (int or tuple): dilation size, default with 1.
        mode (srt): padding mode can be ('same', 'symmetric', 'replicate', 'circular')

    """

    def __init__(self, kernel, stride=1, dilation=1, mode='same'):
        super().__init__()
        self.kernel = to_2tuple(kernel)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)
        self.mode = mode

    def forward(self, x):
        if self.mode is None:
            return x
        else:
            return exact_padding_2d(x, self.kernel, self.stride, self.dilation, self.mode)


def imfilter(input, weight, bias=None, stride=1, padding='same', dilation=1, groups=1):
    """imfilter same as matlab.
    Args:
        input (tensor): (b, c, h, w) tensor to be filtered
        weight (tensor): (out_ch, in_ch, kh, kw) filter kernel
        padding (str): padding mode
        dilation (int): dilation of conv
        groups (int): groups of conv
    """
    kernel_size = weight.shape[2:]
    pad_func = ExactPadding2d(kernel_size, stride, dilation, mode=padding)
    return F.conv2d(pad_func(input), weight, bias, stride, dilation=dilation, groups=groups)


def safe_sqrt(x: 'torch.Tensor') ->torch.Tensor:
    """Safe sqrt with EPS to ensure numeric stability.

    Args:
        x (torch.Tensor): should be non-negative
    """
    EPS = torch.finfo(x.dtype).eps
    return torch.sqrt(x + EPS)


def normalize_img_with_gauss(img: 'torch.Tensor', kernel_size: 'int'=7, sigma: 'float'=7.0 / 6, C: 'int'=1, padding: 'str'='same'):
    kernel = fspecial(kernel_size, sigma, 1)
    mu = imfilter(img, kernel, padding=padding)
    std = imfilter(img ** 2, kernel, padding=padding)
    sigma = safe_sqrt((std - mu ** 2).abs())
    img_normalized = (img - mu) / (sigma + C)
    return img_normalized


def natural_scene_statistics(luma: 'torch.Tensor', kernel_size: 'int'=7, sigma: 'float'=7.0 / 6) ->torch.Tensor:
    """
    Compute natural scene statistics (NSS) features for a given luminance image.

    Args:
        luma (torch.Tensor): Luminance image tensor.
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        torch.Tensor: NSS features.
    """
    luma_nrmlzd = normalize_img_with_gauss(luma, kernel_size, sigma, padding='same')
    alpha, sigma = estimate_ggd_param(luma_nrmlzd)
    features = [alpha, sigma.pow(2)]
    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for shift in shifts:
        shifted_luma_nrmlzd = torch.roll(luma_nrmlzd, shifts=shift, dims=(-2, -1))
        alpha, sigma_l, sigma_r = estimate_aggd_param(luma_nrmlzd * shifted_luma_nrmlzd, return_sigma=True)
        eta = (sigma_r - sigma_l) * torch.exp(torch.lgamma(2.0 / alpha) - (torch.lgamma(1.0 / alpha) + torch.lgamma(3.0 / alpha)) / 2)
        features.extend((alpha, eta, sigma_l.pow(2), sigma_r.pow(2)))
    return torch.stack(features, dim=-1)


def rbf_kernel(features: 'torch.Tensor', sv: 'torch.Tensor', gamma: 'float'=0.05) ->torch.Tensor:
    """
    Compute the Radial Basis Function (RBF) kernel between features and support vectors.

    Args:
        features (torch.Tensor): Input features.
        sv (torch.Tensor): Support vectors.
        gamma (float): Gamma parameter for the RBF kernel.

    Returns:
        torch.Tensor: RBF kernel values.
    """
    dist = (features.unsqueeze(dim=-1) - sv.unsqueeze(dim=0)).pow(2).sum(dim=1)
    return torch.exp(-dist * gamma)


def scale_features(features: 'torch.Tensor') ->torch.Tensor:
    """
    Scale features to the range [-1, 1] based on predefined feature ranges.

    Args:
        features (torch.Tensor): Input features.

    Returns:
        torch.Tensor: Scaled features.
    """
    lower_bound = -1
    upper_bound = 1
    feature_ranges = torch.tensor([[0.338, 10], [0.017204, 0.806612], [0.236, 1.642], [-0.123884, 0.20293], [0.000155, 0.712298], [0.001122, 0.470257], [0.244, 1.641], [-0.123586, 0.179083], [0.000152, 0.710456], [0.000975, 0.470984], [0.249, 1.555], [-0.135687, 0.100858], [0.000174, 0.684173], [0.000913, 0.534174], [0.258, 1.561], [-0.143408, 0.100486], [0.000179, 0.685696], [0.000888, 0.536508], [0.471, 3.264], [0.012809, 0.703171], [0.218, 1.046], [-0.094876, 0.187459], [1.5e-05, 0.442057], [0.001272, 0.40803], [0.222, 1.042], [-0.115772, 0.162604], [1.6e-05, 0.444362], [0.001374, 0.40243], [0.227, 0.996], [-0.117188, 0.098323], [3e-05, 0.531903], [0.001122, 0.369589], [0.228, 0.99], [-0.12243, 0.098658], [2.8e-05, 0.530092], [0.001118, 0.370399]])
    scaled_features = lower_bound + (upper_bound - lower_bound) * (features - feature_ranges[..., 0]) / (feature_ranges[..., 1] - feature_ranges[..., 0])
    return scaled_features


def rgb2lhm(x: 'torch.Tensor') ->torch.Tensor:
    """Convert a batch of RGB images to a batch of LHM images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). LHM colour space.

    Reference:
        https://arxiv.org/pdf/1608.07433.pdf
    """
    lhm_weights = torch.tensor([[0.2989, 0.587, 0.114], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]]).t()
    x_lhm = torch.matmul(x.permute(0, 2, 3, 1), lhm_weights).permute(0, 3, 1, 2)
    return x_lhm


def rgb2ycbcr(x: 'torch.Tensor') ->torch.Tensor:
    """Convert a batch of RGB images to a batch of YCbCr images

    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB color space, range [0, 1].

    Returns:
        Batch of images with shape (N, 3, H, W). YCbCr color space.
    """
    weights_rgb_to_ycbcr = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]])
    bias_rgb_to_ycbcr = torch.tensor([16, 128, 128]).view(1, 3, 1, 1)
    x_ycbcr = torch.matmul(x.permute(0, 2, 3, 1), weights_rgb_to_ycbcr).permute(0, 3, 1, 2) + bias_rgb_to_ycbcr
    x_ycbcr = x_ycbcr / 255.0
    return x_ycbcr


def rgb2yiq(x: 'torch.Tensor') ->torch.Tensor:
    """Convert a batch of RGB images to a batch of YIQ images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). YIQ colour space.
    """
    yiq_weights = torch.tensor([[0.299, 0.587, 0.114], [0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]]).t()
    x_yiq = torch.matmul(x.permute(0, 2, 3, 1), yiq_weights).permute(0, 3, 1, 2)
    return x_yiq


def to_y_channel(img: 'torch.Tensor', out_data_range: 'float'=1.0, color_space: 'str'='yiq') ->torch.Tensor:
    """Change to Y channel
    Args:
        image tensor: tensor with shape (N, 3, H, W) in range [0, 1].
    Returns:
        image tensor: Y channel of the input tensor
    """
    assert img.ndim == 4 and img.shape[1] == 3, 'input image tensor should be RGB image batches with shape (N, 3, H, W)'
    color_space = color_space.lower()
    if color_space == 'yiq':
        img = rgb2yiq(img)
    elif color_space == 'ycbcr':
        img = rgb2ycbcr(img)
    elif color_space == 'lhm':
        img = rgb2lhm(img)
    out_img = img[:, [0], :, :] * out_data_range
    if out_data_range >= 255:
        out_img = out_img - out_img.detach() + out_img.round()
    return out_img


def brisque(x: 'torch.Tensor', kernel_size: 'int'=7, kernel_sigma: 'float'=7 / 6, test_y_channel: 'bool'=True, sv_coef: 'torch.Tensor'=None, sv: 'torch.Tensor'=None, gamma: 'float'=0.05, rho: 'float'=-153.591, scale: 'float'=1, version: 'str'='original') ->torch.Tensor:
    """Interface of BRISQUE index.

    Args:
        x (torch.Tensor): An input tensor. Shape :math:`(N, C, H, W)`.
        kernel_size (int): The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma (float): Sigma of normal distribution.
        test_y_channel (bool): Whether to use the y-channel of YCBCR.
        sv_coef (torch.Tensor): Support vector coefficients.
        sv (torch.Tensor): Support vectors.
        gamma (float): Gamma parameter for the RBF kernel.
        rho (float): Bias term in the decision function.
        scale (float): Scaling factor for the features.
        version (str): Version of the BRISQUE implementation ('original' or 'matlab').

    Returns:
        torch.Tensor: Value of BRISQUE index.

    References:
        Mittal, Anish, Anush Krishna Moorthy, and Alan Conrad Bovik.
        "No-reference image quality assessment in the spatial domain."
        IEEE Transactions on image processing 21, no. 12 (2012): 4695-4708.
    """
    if test_y_channel and x.size(1) == 3:
        x = to_y_channel(x, 255.0)
    else:
        x = x * 255
    features = []
    num_of_scales = 2
    for _ in range(num_of_scales):
        if version == 'matlab':
            xnorm = normalize_img_with_gauss(x, kernel_size, kernel_sigma, padding='replicate')
            features.append(compute_nss_features(xnorm))
        elif version == 'original':
            features.append(natural_scene_statistics(x, kernel_size, kernel_sigma))
        x = imresize(x, scale=0.5, antialiasing=True)
    features = torch.cat(features, dim=-1)
    sv_coef = sv_coef
    sv = sv
    if version == 'original':
        scaled_features = scale_features(features)
    elif version == 'matlab':
        scaled_features = features / scale
    sv.t_()
    kernel_features = rbf_kernel(features=scaled_features, sv=sv, gamma=gamma)
    score = kernel_features @ sv_coef - rho
    return score


class BRISQUE(torch.nn.Module):
    """Creates a criterion that measures the BRISQUE score.

    Args:
        kernel_size (int): By default, the mean and covariance of a pixel is obtained
                           by convolution with given filter_size. Must be an odd value.
        kernel_sigma (float): Standard deviation for Gaussian kernel.
        test_y_channel (bool): Whether to use the y-channel of YCBCR.
        version (str): Version of the BRISQUE implementation ('original' or 'matlab').
        pretrained_model_path (str, optional): The model path.

    Attributes:
        kernel_size (int): The side-length of the sliding window used in comparison.
        kernel_sigma (float): Sigma of normal distribution.
        test_y_channel (bool): Whether to use the y-channel of YCBCR.
        sv_coef (torch.Tensor): Support vector coefficients.
        sv (torch.Tensor): Support vectors.
        gamma (float): Gamma parameter for the RBF kernel.
        rho (float): Bias term in the decision function.
        scale (float): Scaling factor for the features.
        version (str): Version of the BRISQUE implementation ('original' or 'matlab').
    """

    def __init__(self, kernel_size: 'int'=7, kernel_sigma: 'float'=7 / 6, test_y_channel: 'bool'=True, version: 'str'='original', pretrained_model_path: 'str'=None) ->None:
        super().__init__()
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
        assert test_y_channel, f'Only [test_y_channel=True] is supported for current BRISQUE model, which is taken directly from official codes: https://github.com/utlive/BRISQUE.'
        self.kernel_sigma = kernel_sigma
        self.test_y_channel = test_y_channel
        if pretrained_model_path is not None:
            self.sv_coef, self.sv = torch.load(pretrained_model_path, weights_only=False)
        elif version == 'original':
            pretrained_model_path = load_file_from_url(default_model_urls['url'])
            self.sv_coef, self.sv = torch.load(pretrained_model_path, weights_only=False)
            self.gamma = 0.05
            self.rho = -153.591
            self.scale = 1
        elif version == 'matlab':
            pretrained_model_path = load_file_from_url(default_model_urls['brisque_matlab'])
            self.gamma = 1
            self.rho = -43.4582
            self.scale = 0.321
            params = scipy.io.loadmat(pretrained_model_path)
            sv = params['sv']
            sv_coef = np.ravel(params['sv_coef'])
            sv = torch.from_numpy(sv)
            self.sv_coef = torch.from_numpy(sv_coef)
            self.sv = sv / self.scale
        self.version = version

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Computation of BRISQUE score as a loss function.

        Args:
            x (torch.Tensor): An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            torch.Tensor: Value of BRISQUE metric.
        """
        return brisque(x, kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma, test_y_channel=self.test_y_channel, sv_coef=self.sv_coef, sv=self.sv, gamma=self.gamma, rho=self.rho, scale=self.scale, version=self.version)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.k = 3
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.head = 8
        self.qse_1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.qse_2 = self._make_layer(block, 64, layers[0])
        self.csp = self._make_layer(block, 128, layers[1], stride=2, dilate=False)
        self.inplanes = 64
        self.dte_1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.dte_2 = self._make_layer(block, 64, layers[0])
        self.aux_csp = self._make_layer(block, 128, layers[1], stride=2, dilate=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_ = nn.Sequential(nn.Linear(512 * 1 * 1, 2048), nn.ReLU(True), nn.Dropout(), nn.Linear(2048, 2048), nn.ReLU(True), nn.Dropout(), nn.Linear(2048, 1))
        self.fc1_ = nn.Sequential(nn.Linear(512 * 1 * 1, 2048), nn.ReLU(True), nn.Dropout(), nn.Linear(2048, 2048), nn.ReLU(True), nn.Dropout(), nn.Linear(2048, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        """
        Forward pass for the ResNet model.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W).
            y (torch.Tensor): Reference tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor after processing through the network.
        """
        rest1 = x
        dist1 = y
        rest1 = self.qse_2(self.maxpool(self.qse_1(rest1)))
        dist1 = self.dte_2(self.maxpool(self.dte_1(dist1)))
        x = rest1 - dist1
        x = self.csp(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        dr = torch.sigmoid(self.fc_(x))
        return dr


model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    """
    Helper function to create a ResNet model.

    Args:
        arch (str): Architecture name.
        block (nn.Module): Block type (BasicBlock or Bottleneck).
        layers (list): List of layer configurations.
        pretrained (bool): Whether to load pretrained weights.
        progress (bool): Whether to display progress bar.
        **kwargs: Additional arguments.

    Returns:
        ResNet: Instantiated ResNet model.
    """
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        keys = state_dict.keys()
        for key in list(keys):
            if 'conv1' in key:
                state_dict[key.replace('conv1', 'qse_1')] = state_dict[key]
                state_dict[key.replace('conv1', 'dte_1')] = state_dict[key]
            if 'layer1' in key:
                state_dict[key.replace('layer1', 'qse_2')] = state_dict[key]
                state_dict[key.replace('layer1', 'dte_2')] = state_dict[key]
            if 'layer2' in key:
                state_dict[key.replace('layer2', 'csp')] = state_dict[key]
                state_dict[key.replace('layer2', 'aux_csp')] = state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


class CKDN(nn.Module):
    """
    CKDN metric.

    Args:
        pretrained_model_path (str): The model path.
        use_default_preprocess (bool): Whether to use default preprocess, default: True.
        default_mean (tuple): The mean value.
        default_std (tuple): The std value.

    Reference:
        Zheng, Heliang, Huan Yang, Jianlong Fu, Zheng-Jun Zha, and Jiebo Luo.
        "Learning conditional knowledge distillation for degraded-reference image
        quality assessment." In Proceedings of the IEEE/CVF International Conference
        on Computer Vision (ICCV), pp. 10242-10251. 2021.
    """

    def __init__(self, pretrained=True, pretrained_model_path=None, use_default_preprocess=True, default_mean=(0.485, 0.456, 0.406), default_std=(0.229, 0.224, 0.225), **kwargs):
        super().__init__()
        self.net = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], True, True, **kwargs)
        self.use_default_preprocess = use_default_preprocess
        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path)
        elif pretrained:
            load_pretrained_network(self, default_model_urls['url'])

    def _default_preprocess(self, x, y):
        """
        Default preprocessing of CKDN.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W) in RGB format; value range, 0 ~ 1.
            y (torch.Tensor): Reference tensor with shape (N, C, H, W) in RGB format; value range, 0 ~ 1.

        Returns:
            tuple: Preprocessed tensors (x, y).
        """
        scaled_size = int(math.floor(288 / 0.875))
        x = tv.transforms.functional.resize(x, scaled_size, tv.transforms.InterpolationMode.BICUBIC)
        y = tv.transforms.functional.resize(y, scaled_size, tv.transforms.InterpolationMode.NEAREST)
        x = tv.transforms.functional.center_crop(x, 288)
        y = tv.transforms.functional.center_crop(y, 288)
        x = (x - self.default_mean) / self.default_std
        y = (y - self.default_mean) / self.default_std
        return x, y

    def forward(self, x, y):
        """
        Compute IQA using CKDN model.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W). RGB channel order for colour images.
            y (torch.Tensor): Reference tensor with shape (N, C, H, W). RGB channel order for colour images.

        Returns:
            torch.Tensor: Value of CKDN model.
        """
        if self.use_default_preprocess:
            x, y = self._default_preprocess(x, y)
        return self.net(x, y)


class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: 'int', embed_dim: 'int', num_heads: 'int', output_dim: 'int'=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.spacial_dim = spacial_dim
        self.embed_dim = embed_dim

    def forward(self, x, return_token=False, pos_embedding=False):
        n, c, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        if pos_embedding:
            positional_embedding_resize = F.interpolate(self.positional_embedding.unsqueeze(0).unsqueeze(0), size=(x.size(0), x.size(2)), mode='bicubic').squeeze(0).squeeze(0)
            x = x + positional_embedding_resize[:, None, :]
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        if return_token:
            return x[0], x[1:]
        else:
            return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        self.feature_dim_list = [width, width * 4, width * 8, width * 16, width * 32]
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward_features(self, x, return_token=False, pos_embedding=False):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        feat_list = [x]
        x = self.layer1(x)
        feat_list += [x]
        x = self.layer2(x)
        feat_list += [x]
        x = self.layer3(x)
        feat_list += [x]
        x = self.layer4(x)
        feat_list += [x]
        return feat_list

    def forward(self, x, return_token=False, pos_embedding=False):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if return_token:
            x, tokens = self.attnpool(x, return_token, pos_embedding)
            return x, tokens
        else:
            x = self.attnpool(x, return_token, pos_embedding)
            return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: 'torch.Tensor'):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: 'torch.Tensor'):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: 'int', n_head: 'int', attn_mask: 'torch.Tensor'=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: 'torch.Tensor'):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: 'torch.Tensor'):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class Transformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, return_intermediate_dec=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed):
        bs, c, h, w = src.shape
        src2 = src
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed2 = pos_embed
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


class VisionTransformer(nn.Module):

    def __init__(self, input_resolution: 'int', patch_size: 'int', width: 'int', layers: 'int', heads: 'int', output_dim: 'int'):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: 'torch.Tensor', return_token=False, pos_embedding=False):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        if pos_embedding:
            positional_embedding_resize = F.interpolate(self.positional_embedding.unsqueeze(0).unsqueeze(0), size=(x.size(1), x.size(2)), mode='bicubic').squeeze(0).squeeze(0)
            x = x + positional_embedding_resize
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        token = self.ln_post(x[:, 1:, :])
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        if return_token:
            return x, token
        else:
            return x


class CLIP(nn.Module):

    def __init__(self, embed_dim: 'int', image_resolution: 'int', vision_layers: 'Union[Tuple[int, int, int, int], int]', vision_width: 'int', vision_patch_size: 'int', context_length: 'int', vocab_size: 'int', transformer_width: 'int', transformer_heads: 'int', transformer_layers: 'int'):
        super().__init__()
        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers, output_dim=embed_dim, heads=vision_heads, input_resolution=image_resolution, width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim)
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith('bn3.weight'):
                        nn.init.zeros_(param)
        proj_std = self.transformer.width ** -0.5 * (2 * self.transformer.layers) ** -0.5
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, pos_embedding):
        return self.visual(image.type(self.dtype), pos_embedding=pos_embedding)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text, pos_embedding=False, text_features=None):
        image_features = self.encode_image(image, pos_embedding)
        if text_features is None:
            text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


class PromptLearner(nn.Module):
    """
    PromptLearner class for learning prompts for CLIP-IQA.

    Disclaimer:
        This implementation follows exactly the official codes in: https://github.com/IceClear/CLIP-IQA. 
        We have no idea why some tricks are implemented like this, which include:
            1. Using n_ctx prefix characters "X"
            2. Appending extra "." at the end
            3. Insert the original text embedding at the middle
    """

    def __init__(self, clip_model, n_ctx=16) ->None:
        """
        Initialize the PromptLearner.

        Args:
            clip_model (nn.Module): The CLIP model.
            n_ctx (int): Number of context tokens. Default is 16.
        """
        super().__init__()
        prompt_prefix = ' '.join(['X'] * n_ctx) + ' '
        init_prompts = [prompt_prefix + 'Good photo..', prompt_prefix + 'Bad photo..']
        with torch.no_grad():
            txt_token = clip.tokenize(init_prompts)
            self.tokenized_prompts = txt_token
            init_embedding = clip_model.token_embedding(txt_token)
        init_ctx = init_embedding[:, 1:1 + n_ctx]
        self.ctx = nn.Parameter(init_ctx)
        self.n_ctx = n_ctx
        self.n_cls = len(init_prompts)
        self.name_lens = [3, 3]
        self.register_buffer('token_prefix', init_embedding[:, :1, :])
        self.register_buffer('token_suffix', init_embedding[:, 1 + n_ctx:, :])

    def get_prompts_with_middle_class(self):
        """
        Get prompts with the original text embedding inserted in the middle.

        Returns:
            torch.Tensor: The generated prompts.
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        half_n_ctx = self.n_ctx // 2
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i + 1, :, :]
            class_i = self.token_suffix[i:i + 1, :name_len, :]
            suffix_i = self.token_suffix[i:i + 1, name_len:, :]
            ctx_i_half1 = ctx[i:i + 1, :half_n_ctx, :]
            ctx_i_half2 = ctx[i:i + 1, half_n_ctx:, :]
            prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)
        return prompts

    def forward(self, clip_model):
        """
        Forward pass for the PromptLearner.

        Args:
            clip_model (nn.Module): The CLIP model.

        Returns:
            torch.Tensor: The output features.
        """
        prompts = self.get_prompts_with_middle_class()
        x = prompts + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = clip_model.ln_final(x).type(clip_model.dtype)
        x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ clip_model.text_projection
        return x


OPENAI_CLIP_MEAN = 0.48145466, 0.4578275, 0.40821073


OPENAI_CLIP_STD = 0.26862954, 0.26130258, 0.27577711


_MODELS = {'RN50': 'https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt', 'RN101': 'https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt', 'RN50x4': 'https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt', 'RN50x16': 'https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt', 'RN50x64': 'https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt', 'ViT-B/32': 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt', 'ViT-B/16': 'https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt', 'ViT-L/14': 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt', 'ViT-L/14@336px': 'https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt'}


def _download(url: 'str', root: 'str'):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split('/')[-2]
    download_target = os.path.join(root, filename)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f'{download_target} exists and is not a regular file')
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, 'rb').read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f'{download_target} exists, but the SHA256 checksum does not match; re-downloading the file')
    with urllib.request.urlopen(url) as source, open(download_target, 'wb') as output:
        with tqdm(total=int(source.info().get('Content-Length')), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    if hashlib.sha256(open(download_target, 'rb').read()).hexdigest() != expected_sha256:
        raise RuntimeError('Model has been downloaded but the SHA256 checksum does not not match')
    return download_target


def available_models() ->List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def convert_weights(model: 'nn.Module'):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f'{s}_proj_weight' for s in ['in', 'q', 'k', 'v']], 'in_proj_bias', 'bias_k', 'bias_v']:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ['text_projection', 'proj']:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: 'dict'):
    vit = 'visual.proj' in state_dict
    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith('visual.') and k.endswith('.attn.in_proj_weight')])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grid_size = round((state_dict['visual.positional_embedding'].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: 'list' = [len(set(k.split('.')[2] for k in state_dict if k.startswith(f'visual.layer{b}'))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict['visual.layer1.0.conv1.weight'].shape[0]
        output_width = round((state_dict['visual.attnpool.positional_embedding'].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict['visual.attnpool.positional_embedding'].shape[0]
        image_resolution = output_width * 32
    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split('.')[2] for k in state_dict if k.startswith(f'transformer.resblocks')))
    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
    for key in ['input_resolution', 'context_length', 'vocab_size']:
        if key in state_dict:
            del state_dict[key]
    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


def load(name: 'str', device: 'Union[str, torch.device]'='cuda' if torch.cuda.is_available() else 'cpu', jit: 'bool'=False, download_root: 'str'=None):
    """Load a CLIP model
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).
    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.join(get_dir(), 'clip'))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f'Model {name} not found; available models = {available_models()}')
    with open(model_path, 'rb') as opened_file:
        try:
            model = torch.jit.load(opened_file, map_location=device if jit else 'cpu').eval()
            state_dict = None
        except RuntimeError:
            if jit:
                warnings.warn(f'File {model_path} is not a JIT archive. Loading as a state dict instead')
                jit = False
            state_dict = torch.load(opened_file, map_location='cpu')
    if not jit:
        model = build_model(state_dict or model.state_dict())
        if str(device) == 'cpu':
            model.float()
        return model
    device_holder = torch.jit.trace(lambda : torch.ones([]), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes('prim::Constant') if 'Device' in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, 'graph') else []
        except RuntimeError:
            graphs = []
        if hasattr(module, 'forward1'):
            graphs.append(module.forward1.graph)
        for graph in graphs:
            for node in graph.findAllNodes('prim::Constant'):
                if 'value' in node.attributeNames() and str(node['value']).startswith('cuda'):
                    node.copyAttributes(device_node)
    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)
    if str(device) == 'cpu':
        float_holder = torch.jit.trace(lambda : torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode('aten::to').inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, 'graph') else []
            except RuntimeError:
                graphs = []
            if hasattr(module, 'forward1'):
                graphs.append(module.forward1.graph)
            for graph in graphs:
                for node in graph.findAllNodes('aten::to'):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        if inputs[i].node()['value'] == 5:
                            inputs[i].node().copyAttributes(float_node)
        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()
    return model


class CLIPIQA(nn.Module):
    """
    CLIPIQA metric class.

    Args:
        model_type (str): The type of the model. Default is 'clipiqa'.
        backbone (str): The backbone model. Default is 'RN50'.
        pretrained (bool): Whether to load pretrained weights. Default is True.
        pos_embedding (bool): Whether to use positional embedding. Default is False.
    """

    def __init__(self, model_type='clipiqa', backbone='RN50', pretrained=True, pos_embedding=False) ->None:
        super().__init__()
        self.clip_model = [load(backbone, 'cpu')]
        self.prompt_pairs = clip.tokenize(['Good image', 'bad image', 'Sharp image', 'blurry image', 'sharp edges', 'blurry edges', 'High resolution image', 'low resolution image', 'Noise-free image', 'noisy image'])
        self.model_type = model_type
        self.pos_embedding = pos_embedding
        if 'clipiqa+' in model_type:
            self.prompt_learner = PromptLearner(self.clip_model[0])
        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)
        for p in self.clip_model[0].parameters():
            p.requires_grad = False
        if pretrained and 'clipiqa+' in model_type:
            if model_type == 'clipiqa+' and backbone == 'RN50':
                self.prompt_learner.ctx.data = torch.load(load_file_from_url(default_model_urls['clipiqa+']), weights_only=False)
            elif model_type in default_model_urls.keys():
                load_pretrained_network(self, default_model_urls[model_type], True, 'params')
            else:
                raise ValueError(f'No pretrained model for {model_type}')

    def forward(self, x):
        """
        Forward pass for the CLIPIQA model.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: The output probabilities.
        """
        x = (x - self.default_mean) / self.default_std
        clip_model = self.clip_model[0]
        if self.model_type == 'clipiqa':
            prompts = self.prompt_pairs
            logits_per_image, logits_per_text = clip_model(x, prompts, pos_embedding=self.pos_embedding)
        elif 'clipiqa+' in self.model_type:
            learned_prompt_feature = self.prompt_learner(clip_model)
            logits_per_image, logits_per_text = clip_model(x, None, text_features=learned_prompt_feature, pos_embedding=self.pos_embedding)
        probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 2).softmax(dim=-1)
        return probs[..., 0].mean(dim=1, keepdim=True)


def clip_preprocess_tensor(x: 'torch.Tensor', model):
    """
    Clip preprocess function with tensor input.

    NOTE: Results are slightly different with original preprocess function with PIL image input, because of differences in resize function.

    Args:
        x (torch.Tensor): Input tensor.
        model: Model with visual input resolution.

    Returns:
        torch.Tensor: Preprocessed tensor.
    """
    x = T.functional.resize(x, model.visual.input_resolution, interpolation=T.InterpolationMode.BICUBIC, antialias=True)
    x = T.functional.center_crop(x, model.visual.input_resolution)
    x = T.functional.normalize(x, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
    return x


class CLIPScore(nn.Module):
    """
    A PyTorch module for computing image-text similarity scores using the CLIP model.

    Args:
        backbone (str): The name of the CLIP model backbone to use. Default is 'ViT-B/32'.
        w (float): The weight to apply to the similarity score. Default is 2.5.
        prefix (str): The prefix to add to each caption when computing text features. Default is 'A photo depicts'.

    Attributes:
        clip_model (CLIP): The CLIP model used for computing image and text features.
        prefix (str): The prefix to add to each caption when computing text features.
        w (float): The weight to apply to the similarity score.

    Methods:
        forward(img, caption_list): Computes the similarity score between the input image and a list of captions.
    """

    def __init__(self, backbone='ViT-B/32', w=2.5, prefix='A photo depicts') ->None:
        super().__init__()
        self.clip_model, _ = clip.load(backbone)
        self.prefix = prefix
        self.w = w

    def forward(self, img, caption_list=None):
        """
        Computes the similarity score between the input image and a list of captions.

        Args:
            img (torch.Tensor): Input image tensor.
            caption_list (list of str): List of captions to compare with the image.

        Returns:
            torch.Tensor: The computed similarity scores.
        """
        assert caption_list is not None, 'caption_list is None'
        text = clip.tokenize([(self.prefix + ' ' + caption) for caption in caption_list], truncate=True)
        img_features = self.clip_model.encode_image(clip_preprocess_tensor(img, self.clip_model))
        text_features = self.clip_model.encode_text(text)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = self.w * torch.relu((img_features * text_features).sum(dim=-1))
        return score


class CNNIQA(nn.Module):
    """CNNIQA model.

    Args:
        ker_size (int): Kernel size.
        n_kers (int): Number of kernels.
        n1_nodes (int): Number of n1 nodes.
        n2_nodes (int): Number of n2 nodes.
        pretrained (str): Pretrained model name.
        pretrained_model_path (str): Pretrained model path.
    """

    def __init__(self, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800, pretrained='koniq10k', pretrained_model_path=None):
        super(CNNIQA, self).__init__()
        self.conv1 = nn.Conv2d(3, n_kers, ker_size)
        self.fc1 = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2 = nn.Linear(n1_nodes, n2_nodes)
        self.fc3 = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()
        if pretrained_model_path is None and pretrained is not None:
            pretrained_model_path = default_model_urls[pretrained]
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, 'params')

    def forward(self, x):
        """Compute IQA using CNNIQA model.

        Args:
            x (torch.Tensor): An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            torch.Tensor: Value of CNNIQA model.
        """
        h = self.conv1(x)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h = torch.cat((h1, h2), 1)
        h = h.squeeze(3).squeeze(2)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        q = self.fc3(h)
        return q


class Compare2Score(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained('q-future/Compare2Score', trust_remote_code=True, torch_dtype=torch.float16)

    def preprocess(self, x):
        assert x.shape[0] == 1, 'Currently, only support batch size 1.'
        images = F.to_pil_image(x[0])
        return images

    def forward(self, x):
        """
            x: str, path to image
        """
        image_tensor = self.preprocess(x)
        score = self.model.score(image_tensor)
        return score


class SCNN(nn.Module):
    """Network branch for synthetic distortions.

    Args:
        use_bn (bool): Whether to use batch normalization.

    Modified from https://github.com/zwx8981/DBCNN-PyTorch/blob/master/SCNN.py
    """

    def __init__(self, use_bn=True):
        super(SCNN, self).__init__()
        self.num_class = 39
        self.use_bn = use_bn
        self.features = nn.Sequential(*self._make_layers(3, 48, 3, 1, 1), *self._make_layers(48, 48, 3, 2, 1), *self._make_layers(48, 64, 3, 1, 1), *self._make_layers(64, 64, 3, 2, 1), *self._make_layers(64, 64, 3, 1, 1), *self._make_layers(64, 64, 3, 2, 1), *self._make_layers(64, 128, 3, 1, 1), *self._make_layers(128, 128, 3, 1, 1), *self._make_layers(128, 128, 3, 2, 1))
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(*self._make_layers(128, 256, 1, 1, 0), *self._make_layers(256, 256, 1, 1, 0))
        self.classifier = nn.Linear(256, self.num_class)

    def _make_layers(self, in_ch, out_ch, ksz, stride, pad):
        """Helper function to create layers for the network."""
        if self.use_bn:
            layers = [nn.Conv2d(in_ch, out_ch, ksz, stride, pad), nn.BatchNorm2d(out_ch), nn.ReLU(True)]
        else:
            layers = [nn.Conv2d(in_ch, out_ch, ksz, stride, pad), nn.ReLU(True)]
        return layers

    def forward(self, X):
        """
        Forward pass for the SCNN.

        Args:
            X (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor after processing through the network.
        """
        X = self.features(X)
        X = self.pooling(X)
        X = self.projection(X)
        X = X.view(X.shape[0], -1)
        X = self.classifier(X)
        return X


class DBCNN(nn.Module):
    """Full DBCNN network.

    Args:
        fc (bool): Whether to initialize the fc layers.
        use_bn (bool): Whether to use batch normalization.
        pretrained_scnn_path (str): Pretrained SCNN path.
        pretrained (bool): Whether to load pretrained weights.
        pretrained_model_path (str): Pretrained model path.
        default_mean (list): Default mean value.
        default_std (list): Default std value.
    """

    def __init__(self, fc=True, use_bn=True, pretrained_scnn_path=None, pretrained=True, pretrained_model_path=None, default_mean=[0.485, 0.456, 0.406], default_std=[0.229, 0.224, 0.225]):
        super(DBCNN, self).__init__()
        self.features1 = torchvision.models.vgg16(weights='IMAGENET1K_V1').features
        self.features1 = nn.Sequential(*list(self.features1.children())[:-1])
        scnn = SCNN(use_bn=use_bn)
        load_pretrained_network(scnn, default_model_urls['scnn'])
        self.features2 = scnn.features
        self.fc = torch.nn.Linear(512 * 128, 1)
        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)
        if fc:
            for param in self.features1.parameters():
                param.requires_grad = False
            for param in scnn.parameters():
                param.requires_grad = False
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
        if pretrained_model_path is None and pretrained:
            url_key = 'koniq' if isinstance(pretrained, bool) else pretrained
            pretrained_model_path = default_model_urls[url_key]
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, 'params')

    def preprocess(self, x):
        """
        Preprocess the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: Preprocessed tensor.
        """
        x = (x - self.default_mean) / self.default_std
        return x

    def forward(self, X):
        """
        Compute IQA using DBCNN model.

        Args:
            X (torch.Tensor): An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            torch.Tensor: Value of DBCNN model.
        """
        X = self.preprocess(X)
        X1 = self.features1(X)
        X2 = self.features2(X)
        N, _, H, W = X1.shape
        N, _, H2, W2 = X2.shape
        if H != H2 or W != W2:
            X2 = F.interpolate(X2, (H, W), mode='bilinear', align_corners=True)
        X1 = X1.view(N, 512, H * W)
        X2 = X2.view(N, 128, H * W)
        X = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (H * W)
        X = X.view(N, 512 * 128)
        X = torch.sqrt(X + 1e-08)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        return X


names = {'vgg19': ['image', 'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']}


class MultiVGGFeaturesExtractor(nn.Module):

    def __init__(self, target_features=('conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4'), use_input_norm=True, requires_grad=False):
        super(MultiVGGFeaturesExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        self.target_features = target_features
        model = torchvision.models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        names_key = 'vgg19'
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.target_indexes = [(names[names_key].index(k) - 1) for k in self.target_features]
        self.features = nn.Sequential(*list(model.features.children())[:max(self.target_indexes) + 1])
        if not requires_grad:
            for k, v in self.features.named_parameters():
                v.requires_grad = False
            self.features.eval()

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        y = OrderedDict()
        if 'image' in self.target_features:
            y.update({'image': x})
        for key, layer in self.features._modules.items():
            x = layer(x)
            if int(key) in self.target_indexes:
                y.update({self.target_features[self.target_indexes.index(int(key))]: x})
        return y

    def _normalize_tensor(sefl, in_feat, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
        return in_feat / (norm_factor + eps)


class DeepDC(nn.Module):

    def __init__(self, features_to_compute=('conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4')):
        super(DeepDC, self).__init__()
        self.MSE = torch.nn.MSELoss()
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, x, y):
        """Compute IQA using DeepDC model.

        Args:
            - x: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.
            - y: An reference tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of DeepDC model.

        """
        targets, inputs = x, y
        inputs_fea = self.features_extractor(inputs)
        with torch.no_grad():
            targets_fea = self.features_extractor(targets)
        dc_scores = []
        for _, key in enumerate(inputs_fea.keys()):
            inputs_dcdm = self._DCDM(inputs_fea[key])
            targets_dcdm = self._DCDM(targets_fea[key])
            dc_scores.append(self.Distance_Correlation(inputs_dcdm, targets_dcdm))
        dc_scores = torch.stack(dc_scores, dim=1)
        score = 1 - dc_scores.mean(dim=1, keepdim=True)
        return score

    def _DCDM(self, x):
        if len(x.shape) == 4:
            batchSize, dim, h, w = x.data.shape
            M = h * w
        elif len(x.shape) == 3:
            batchSize, M, dim = x.data.shape
        x = x.reshape(batchSize, dim, M)
        t = torch.log(1.0 / (torch.tensor(dim) * torch.tensor(dim)))
        I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(x.dtype)
        I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype)
        x_pow2 = x.bmm(x.transpose(1, 2))
        dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
        dcov = torch.clamp(dcov, min=0.0)
        dcov = torch.exp(t) * dcov
        dcov = torch.sqrt(dcov + 1e-05)
        dcdm = dcov - 1.0 / dim * dcov.bmm(I_M) - 1.0 / dim * I_M.bmm(dcov) + 1.0 / (dim * dim) * I_M.bmm(dcov).bmm(I_M)
        return dcdm

    def Distance_Correlation(self, matrix_A, matrix_B):
        Gamma_XY = torch.sum(matrix_A * matrix_B, dim=[1, 2])
        Gamma_XX = torch.sum(matrix_A * matrix_A, dim=[1, 2])
        Gamma_YY = torch.sum(matrix_B * matrix_B, dim=[1, 2])
        c = 1e-06
        correlation_r = (Gamma_XY + c) / (torch.sqrt(Gamma_XX * Gamma_YY) + c)
        return correlation_r


class L2pooling(nn.Module):

    def __init__(self, filter_size=5, stride=1, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


class DISTS(torch.nn.Module):
    """DISTS model.
    Args:
        pretrained_model_path (String): Pretrained model path.

    """

    def __init__(self, pretrained=True, pretrained_model_path=None, **kwargs):
        """Refer to official code https://github.com/dingkeyan93/DISTS
        """
        super(DISTS, self).__init__()
        vgg_pretrained_features = models.vgg16(weights='IMAGENET1K_V1').features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_parameter('alpha', nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_parameter('beta', nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, False)
        elif pretrained:
            load_pretrained_network(self, default_model_urls['url'], False)

    def forward_once(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y):
        """Compute IQA using DISTS model.

        Args:
            - x: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.
            - y: An reference tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of DISTS model.

        """
        feats0 = self.forward_once(x)
        feats1 = self.forward_once(y)
        dist1 = 0
        dist2 = 0
        c1 = 1e-06
        c2 = 1e-06
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)
            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)
        score = 1 - (dist1 + dist2)
        return score.squeeze(-1).squeeze(-1)


class Entropy(nn.Module):
    """
    Args:
        x (torch.Tensor): image tensor with shape (B, _, H, W), range [0, 1]
    Return:
        score (torch.Tensor): (B, 1)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        score = entropy(x, **self.kwargs)
        return score


def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`
    Skips default weight initialization if supported by torchvision version.
    See https://github.com/mseitzer/pytorch-fid/issues/28.
    """
    try:
        version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    except ValueError:
        version = 0,
    if version >= (0, 6):
        kwargs['init_weights'] = False
    return torchvision.models.inception_v3(*args, **kwargs)


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


def fid_inception_v3():
    """Build pretrained Inception model for FID computation
    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.
    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    load_pretrained_network(inception, FID_WEIGHTS_URL)
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=(DEFAULT_BLOCK_INDEX,), resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True):
        """Build pretrained InceptionV3
        Parameters
        ----------
        - output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        - resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        - normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        - requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        - use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        if isinstance(output_blocks, (list, tuple)):
            self.output_blocks = sorted(output_blocks)
            self.last_needed_block = max(output_blocks)
        elif isinstance(output_blocks, str) and 'logits' in output_blocks:
            self.output_blocks = output_blocks
            self.last_needed_block = 3
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(pretrained=True)
        self.fc = inception.fc
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp, resize_input=False, normalize_input=False):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if normalize_input:
            x = 2 * x - 1
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if not isinstance(self.output_blocks, str) and idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        if self.output_blocks == 'logits_unbiased':
            outp.append(x.flatten(1).mm(self.fc.weight.T))
        elif self.output_blocks == 'logits':
            outp.append(self.fc(x))
        return outp


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-06):
    """
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Danica J. Sutherland.
    Params:
        mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        mu2   : The sample mean over activations, precalculated on an
                representative data set.
        sigma1: The covariance matrix over activations for generated samples.
        sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = 'fid calculation produces singular product; adding %s to diagonal of cov estimates' % eps
        None
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=0.001):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def interpolate_bilinear_2d_like_tensorflow1x(input, size=None, scale_factor=None, align_corners=None, method='slow'):
    """Down/up samples the input to either the given :attr:`size` or the given :attr:`scale_factor`

    Epsilon-exact bilinear interpolation as it is implemented in TensorFlow 1.x:
    https://github.com/tensorflow/tensorflow/blob/f66daa493e7383052b2b44def2933f61faf196e0/tensorflow/core/kernels/image_resizer_state.h#L41
    https://github.com/tensorflow/tensorflow/blob/6795a8c3a3678fb805b6a8ba806af77ddfe61628/tensorflow/core/kernels/resize_bilinear_op.cc#L85
    as per proposal:
    https://github.com/pytorch/pytorch/issues/10604#issuecomment-465783319

    Related materials:
    https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
    https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/
    https://machinethink.net/blog/coreml-upsampling/

    Currently only 2D spatial sampling is supported, i.e. expected inputs are 4-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x height x width`.

    Args:
        input (Tensor): the input tensor
        size (Tuple[int, int]): output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
        align_corners (bool, optional): Same meaning as in TensorFlow 1.x.
        method (str, optional):
            'slow' (1e-4 L_inf error on GPU, bit-exact on CPU, with checkerboard 32x32->299x299), or
            'fast' (1e-3 L_inf error on GPU and CPU, with checkerboard 32x32->299x299)
    """
    if method not in ('slow', 'fast'):
        raise ValueError('how_exact can only be one of "slow", "fast"')
    if input.dim() != 4:
        raise ValueError('input must be a 4-D tensor')
    if not torch.is_floating_point(input):
        raise ValueError('input must be of floating point dtype')
    if size is not None and (type(size) not in (tuple, list) or len(size) != 2):
        raise ValueError('size must be a list or a tuple of two elements')
    if align_corners is None:
        raise ValueError('align_corners is not specified (use this function for a complete determinism)')

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if size is not None and scale_factor is not None:
            raise ValueError('only one of size or scale_factor should be defined')
        if scale_factor is not None and isinstance(scale_factor, tuple) and len(scale_factor) != dim:
            raise ValueError('scale_factor shape must match input shape. Input is {}D, scale_factor size is {}'.format(dim, len(scale_factor)))
    is_tracing = torch._C._get_tracing_state()

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            if is_tracing:
                return [torch.tensor(i) for i in size]
            else:
                return size
        scale_factors = _ntuple(dim)(scale_factor)
        if is_tracing:
            return [torch.floor((input.size(i + 2).float() * torch.tensor(scale_factors[i], dtype=torch.float32)).float()) for i in range(dim)]
        else:
            return [int(math.floor(float(input.size(i + 2)) * scale_factors[i])) for i in range(dim)]

    def tf_calculate_resize_scale(in_size, out_size):
        if align_corners:
            if is_tracing:
                return (in_size - 1) / (out_size.float() - 1).clamp(min=1)
            else:
                return (in_size - 1) / max(1, out_size - 1)
        elif is_tracing:
            return in_size / out_size.float()
        else:
            return in_size / out_size
    out_size = _output_size(2)
    scale_x = tf_calculate_resize_scale(input.shape[3], out_size[1])
    scale_y = tf_calculate_resize_scale(input.shape[2], out_size[0])

    def resample_using_grid_sample():
        grid_x = torch.arange(0, out_size[1], 1, dtype=input.dtype, device=input.device)
        grid_x = grid_x * (2 * scale_x / (input.shape[3] - 1)) - 1
        grid_y = torch.arange(0, out_size[0], 1, dtype=input.dtype, device=input.device)
        grid_y = grid_y * (2 * scale_y / (input.shape[2] - 1)) - 1
        grid_x = grid_x.view(1, out_size[1]).repeat(out_size[0], 1)
        grid_y = grid_y.view(out_size[0], 1).repeat(1, out_size[1])
        grid_xy = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=2).unsqueeze(0)
        grid_xy = grid_xy.repeat(input.shape[0], 1, 1, 1)
        out = F.grid_sample(input, grid_xy, mode='bilinear', padding_mode='border', align_corners=True)
        return out

    def resample_manually():
        grid_x = torch.arange(0, out_size[1], 1, dtype=input.dtype, device=input.device)
        grid_x = grid_x * torch.tensor(scale_x, dtype=torch.float32)
        grid_x_lo = grid_x.long()
        grid_x_hi = (grid_x_lo + 1).clamp_max(input.shape[3] - 1)
        grid_dx = grid_x - grid_x_lo.float()
        grid_y = torch.arange(0, out_size[0], 1, dtype=input.dtype, device=input.device)
        grid_y = grid_y * torch.tensor(scale_y, dtype=torch.float32)
        grid_y_lo = grid_y.long()
        grid_y_hi = (grid_y_lo + 1).clamp_max(input.shape[2] - 1)
        grid_dy = grid_y - grid_y_lo.float()
        in_00 = input[:, :, grid_y_lo, :][:, :, :, grid_x_lo]
        in_01 = input[:, :, grid_y_lo, :][:, :, :, grid_x_hi]
        in_10 = input[:, :, grid_y_hi, :][:, :, :, grid_x_lo]
        in_11 = input[:, :, grid_y_hi, :][:, :, :, grid_x_hi]
        in_0 = in_00 + (in_01 - in_00) * grid_dx.view(1, 1, 1, out_size[1])
        in_1 = in_10 + (in_11 - in_10) * grid_dx.view(1, 1, 1, out_size[1])
        out = in_0 + (in_1 - in_0) * grid_dy.view(1, 1, out_size[0], 1)
        return out
    if method == 'slow':
        out = resample_manually()
    else:
        out = resample_using_grid_sample()
    return out


class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores
    files: list of all files in the folder
    mode:
        - clean: use PIL resize before calculate features
        - legacy_pytorch: do not resize here, but before pytorch model
    """

    def __init__(self, files, mode, size=(299, 299)):
        self.files = files
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = str(self.files[i])
        img_pil = Image.open(path).convert('RGB')
        if self.mode == 'clean':

            def resize_single_channel(x_np):
                img = Image.fromarray(x_np.astype(np.float32), mode='F')
                img = img.resize(self.size, resample=Image.BICUBIC)
                return np.asarray(img).clip(0, 255).reshape(*self.size, 1)
            img_np = np.array(img_pil)
            img_np = [resize_single_channel(img_np[:, :, idx]) for idx in range(3)]
            img_np = np.concatenate(img_np, axis=2).astype(np.float32)
            img_np = (img_np - 128) / 128
            img_t = torch.tensor(img_np).permute(2, 0, 1)
        elif self.mode == 'legacy_tensorflow':
            img_np = np.array(img_pil).clip(0, 255)
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()
            img_t = interpolate_bilinear_2d_like_tensorflow1x(img_t.unsqueeze(0), size=self.size, align_corners=False)
            img_t = (img_t.squeeze(0) - 128) / 128
        else:
            img_np = np.array(img_pil).clip(0, 255)
            img_t = self.transforms(img_np)
            img_t = nn.functional.interpolate(img_t.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
            img_t = img_t.squeeze(0)
        return img_t


def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in Image.registered_extensions())


def get_file_paths(dir, max_dataset_size=float('inf'), followlinks=True):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir, followlinks=followlinks)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def get_folder_features(fdir, model=None, num_workers=12, batch_size=32, device=torch.device('cuda'), mode='clean', description='', verbose=True):
    """
    Compute the inception features for a folder of image files
    """
    files = get_file_paths(fdir)
    if verbose:
        None
    dataset = ResizeDataset(files, mode=mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader
    if mode == 'clean' or mode == 'legacy_tensorflow':
        normalize_input = False
    else:
        normalize_input = True
    l_feats = []
    with torch.no_grad():
        for batch in pbar:
            feat = model(batch, False, normalize_input)
            feat = feat[0].squeeze(-1).squeeze(-1).detach().cpu().numpy()
            l_feats.append(feat)
    np_feats = np.concatenate(l_feats)
    return np_feats


def get_reference_statistics(name, res, mode='clean', split='test', metric='FID'):
    """
        Load precomputed reference statistics for commonly used datasets
    """
    base_url = 'https://www.cs.cmu.edu/~clean-fid/stats'
    if split == 'custom':
        res = 'na'
    if metric == 'FID':
        rel_path = f'{name}_{mode}_{split}_{res}.npz'.lower()
        url = f'{base_url}/{rel_path}'
        if rel_path in default_model_urls.keys():
            fpath = load_file_from_url(default_model_urls[rel_path])
        else:
            fpath = load_file_from_url(url)
        stats = np.load(fpath)
        mu, sigma = stats['mu'], stats['sigma']
        return mu, sigma
    elif metric == 'KID':
        rel_path = f'{name}_{mode}_{split}_{res}_kid.npz'.lower()
        url = f'{base_url}/{rel_path}'
        if rel_path in default_model_urls.keys():
            fpath = load_file_from_url(default_model_urls[rel_path])
        else:
            fpath = load_file_from_url(url)
        stats = np.load(fpath)
        return stats['feats']


class FID(nn.Module):
    """Implements the Fréchet Inception Distance (FID) and Clean-FID metrics.

    The FID measures the distance between the feature representations of two sets of images,
    one generated by a model and the other from a reference dataset. The Clean-FID is a variant
    that uses a pre-trained Inception-v3 network to extract features from the images.

    Args:
        dims (int): The number of dimensions of the Inception-v3 feature representation to use.
            Must be one of 64, 192, 768, or 2048. Default: 2048.

    Attributes:
        model (nn.Module): The Inception-v3 network used to extract features.
    """

    def __init__(self, dims=2048) ->None:
        super().__init__()
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3(output_blocks=[block_idx])
        self.model.eval()

    def forward(self, fdir1=None, fdir2=None, mode='clean', dataset_name=None, dataset_res=1024, dataset_split='train', num_workers=4, batch_size=8, device=torch.device('cuda'), verbose=True, **kwargs):
        """Computes the FID or Clean-FID score between two sets of images or a set of images and a reference dataset.

        Args:
            fdir1 (str): The path to the first folder containing the images to compare.
            fdir2 (str): The path to the second folder containing the images to compare.
            mode (str): The calculation mode to use. Must be one of 'clean', 'legacy_pytorch', or 'legacy_tensorflow'.
                Default: 'clean'.
            dataset_name (str): The name of the reference dataset to use. Required if `fdir2` is not specified.
            dataset_res (int): The resolution of the reference dataset. Default: 1024.
            dataset_split (str): The split of the reference dataset to use. Default: 'train'.
            num_workers (int): The number of worker processes to use for data loading. Default: 12.
            batch_size (int): The batch size to use for data loading. Default: 32.
            device (torch.device): The device to use for computation. Default: 'cuda'.
            verbose (bool): Whether to print progress messages. Default: True.

        Returns:
            float: The FID or Clean-FID score between the two sets of images or the set of images and the reference dataset.
        """
        assert mode in ['clean', 'legacy_pytorch', 'legacy_tensorflow'], 'Invalid calculation mode, should be in [clean, legacy_pytorch, legacy_tensorflow]'
        if fdir1 is not None and fdir2 is not None:
            if verbose:
                None
            fbname1 = os.path.basename(fdir1)
            np_feats1 = get_folder_features(fdir1, self.model, num_workers=num_workers, batch_size=batch_size, device=device, mode=mode, description=f'FID {fbname1}: ', verbose=verbose)
            fbname2 = os.path.basename(fdir2)
            np_feats2 = get_folder_features(fdir2, self.model, num_workers=num_workers, batch_size=batch_size, device=device, mode=mode, description=f'FID {fbname2}: ', verbose=verbose)
            mu1, sig1 = np.mean(np_feats1, axis=0), np.cov(np_feats1, rowvar=False)
            mu2, sig2 = np.mean(np_feats2, axis=0), np.cov(np_feats2, rowvar=False)
            return frechet_distance(mu1, sig1, mu2, sig2)
        elif fdir1 is not None and fdir2 is None:
            assert dataset_name is not None, 'When fdir2 is not provided, the reference dataset_name should be specified to calculate fid score.'
            if verbose:
                None
            fbname1 = os.path.basename(fdir1)
            np_feats1 = get_folder_features(fdir1, self.model, num_workers=num_workers, batch_size=batch_size, device=device, mode=mode, description=f'FID {fbname1}: ', verbose=verbose)
            ref_mu, ref_sigma = get_reference_statistics(dataset_name, dataset_res, mode=mode, split=dataset_split)
            mu1, sig1 = np.mean(np_feats1, axis=0), np.cov(np_feats1, rowvar=False)
            score = frechet_distance(mu1, sig1, ref_mu, ref_sigma)
            return score
        else:
            raise ValueError('invalid combination of arguments entered')


def get_meshgrid(size: 'Tuple[int, int]') ->torch.Tensor:
    """Return coordinate grid matrices centered at zero point.
    Args:
        size: Shape of meshgrid to create
    """
    if size[0] % 2:
        x = torch.arange(-(size[0] - 1) / 2, size[0] / 2) / (size[0] - 1)
    else:
        x = torch.arange(-size[0] / 2, size[0] / 2) / size[0]
    if size[1] % 2:
        y = torch.arange(-(size[1] - 1) / 2, size[1] / 2) / (size[1] - 1)
    else:
        y = torch.arange(-size[1] / 2, size[1] / 2) / size[1]
    return torch.meshgrid(x, y, indexing='ij')


def ifftshift(x: 'torch.Tensor') ->torch.Tensor:
    """Similar to np.fft.ifftshift but applies to PyTorch Tensors"""
    shift = [(-(ax // 2)) for ax in x.size()]
    return torch.roll(x, shift, tuple(range(len(shift))))


def _lowpassfilter(size: 'Tuple[int, int]', cutoff: 'float', n: 'int') ->torch.Tensor:
    """
    Constructs a low-pass Butterworth filter.
    Args:
        size: Tuple with height and width of filter to construct
        cutoff: Cutoff frequency of the filter in (0, 0.5()
        n: Filter order. Higher `n` means sharper transition.
            Note that `n` is doubled so that it is always an even integer.

    Returns:
        f = 1 / (1 + w/cutoff) ^ 2n

    """
    assert 0 < cutoff <= 0.5, 'Cutoff frequency must be between 0 and 0.5'
    assert n > 1 and int(n) == n, 'n must be an integer >= 1'
    grid_x, grid_y = get_meshgrid(size)
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    return ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * n)))


def _construct_filters(x: 'torch.Tensor', scales: 'int'=4, orientations: 'int'=4, min_length: 'int'=6, mult: 'int'=2, sigma_f: 'float'=0.55, delta_theta: 'float'=1.2, k: 'float'=2.0, use_lowpass_filter=True):
    """Creates a stack of filters used for computation of phase congruensy maps

    Args:
        - x: Tensor. Shape :math:`(N, 1, H, W)`.
        - scales: Number of wavelets
        - orientations: Number of filter orientations
        - min_length: Wavelength of smallest scale filter
        - mult: Scaling factor between successive filters
        - sigma_f: Ratio of the standard deviation of the Gaussian
        describing the log Gabor filter's transfer function
        in the frequency domain to the filter center frequency.
        - delta_theta: Ratio of angular interval between filter orientations
        and the standard deviation of the angular Gaussian function
        used to construct filters in the freq. plane.
        - k: No of standard deviations of the noise energy beyond the mean
            at which we set the noise threshold point, below which phase
            congruency values get penalized.
        """
    N, _, H, W = x.shape
    theta_sigma = math.pi / (orientations * delta_theta)
    grid_x, grid_y = get_meshgrid((H, W))
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    theta = torch.atan2(-grid_y, grid_x)
    radius = ifftshift(radius)
    theta = ifftshift(theta)
    radius[0, 0] = 1
    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)
    lp = _lowpassfilter(size=(H, W), cutoff=0.45, n=15)
    log_gabor = []
    for s in range(scales):
        wavelength = min_length * mult ** s
        omega_0 = 1.0 / wavelength
        gabor_filter = torch.exp(-torch.log(radius / omega_0) ** 2 / (2 * math.log(sigma_f) ** 2))
        if use_lowpass_filter:
            gabor_filter = gabor_filter * lp
        gabor_filter[0, 0] = 0
        log_gabor.append(gabor_filter)
    spread = []
    for o in range(orientations):
        angl = o * math.pi / orientations
        ds = sintheta * math.cos(angl) - costheta * math.sin(angl)
        dc = costheta * math.cos(angl) + sintheta * math.sin(angl)
        dtheta = torch.abs(torch.atan2(ds, dc))
        spread.append(torch.exp(-dtheta ** 2 / (2 * theta_sigma ** 2)))
    spread = torch.stack(spread)
    log_gabor = torch.stack(log_gabor)
    filters = (spread.repeat_interleave(scales, dim=0) * log_gabor.repeat(orientations, 1, 1)).unsqueeze(0)
    return filters


def _phase_congruency(x: 'torch.Tensor', scales: 'int'=4, orientations: 'int'=4, min_length: 'int'=6, mult: 'int'=2, sigma_f: 'float'=0.55, delta_theta: 'float'=1.2, k: 'float'=2.0) ->torch.Tensor:
    """Compute Phase Congruence for a batch of greyscale images
    Args:
        x: Tensor. Shape :math:`(N, 1, H, W)`.
        scales: Number of wavelet scales
        orientations: Number of filter orientations
        min_length: Wavelength of smallest scale filter
        mult: Scaling factor between successive filters
        sigma_f: Ratio of the standard deviation of the Gaussian
            describing the log Gabor filter's transfer function
            in the frequency domain to the filter center frequency.
        delta_theta: Ratio of angular interval between filter orientations
            and the standard deviation of the angular Gaussian function
            used to construct filters in the freq. plane.
        k: No of standard deviations of the noise energy beyond the mean
            at which we set the noise threshold point, below which phase
            congruency values get penalized.
    Returns:
        Phase Congruency map with shape :math:`(N, H, W)`
    """
    EPS = torch.finfo(x.dtype).eps
    N, _, H, W = x.shape
    filters = _construct_filters(x, scales, orientations, min_length, mult, sigma_f, delta_theta, k)
    imagefft = torch.fft.fft2(x)
    filters_ifft = torch.fft.ifft2(filters)
    filters_ifft = filters_ifft.real * math.sqrt(H * W)
    even_odd = torch.view_as_real(torch.fft.ifft2(imagefft * filters)).view(N, orientations, scales, H, W, 2)
    an = torch.sqrt(torch.sum(even_odd ** 2, dim=-1))
    em_n = (filters.view(1, orientations, scales, H, W)[:, :, :1, ...] ** 2).sum(dim=[-2, -1], keepdims=True)
    sum_e = even_odd[..., 0].sum(dim=2, keepdims=True)
    sum_o = even_odd[..., 1].sum(dim=2, keepdims=True)
    x_energy = torch.sqrt(sum_e ** 2 + sum_o ** 2) + EPS
    mean_e = sum_e / x_energy
    mean_o = sum_o / x_energy
    even = even_odd[..., 0]
    odd = even_odd[..., 1]
    energy = (even * mean_e + odd * mean_o - torch.abs(even * mean_o - odd * mean_e)).sum(dim=2, keepdim=True)
    abs_eo = torch.sqrt(torch.sum(even_odd[:, :, :1, ...] ** 2, dim=-1)).reshape(N, orientations, 1, 1, H * W)
    median_e2n = torch.median(abs_eo ** 2, dim=-1, keepdim=True).values
    mean_e2n = -median_e2n / math.log(0.5)
    noise_power = mean_e2n / em_n
    filters_ifft = filters_ifft.view(1, orientations, scales, H, W)
    sum_an2 = torch.sum(filters_ifft ** 2, dim=-3, keepdim=True)
    sum_ai_aj = torch.zeros(N, orientations, 1, H, W)
    for s in range(scales - 1):
        sum_ai_aj = sum_ai_aj + (filters_ifft[:, :, s:s + 1] * filters_ifft[:, :, s + 1:]).sum(dim=-3, keepdim=True)
    sum_an2 = torch.sum(sum_an2, dim=[-1, -2], keepdim=True)
    sum_ai_aj = torch.sum(sum_ai_aj, dim=[-1, -2], keepdim=True)
    noise_energy2 = 2 * noise_power * sum_an2 + 4 * noise_power * sum_ai_aj
    tau = torch.sqrt(noise_energy2 / 2)
    noise_energy = tau * math.sqrt(math.pi / 2)
    moise_energy_sigma = torch.sqrt((2 - math.pi / 2) * tau ** 2)
    T = noise_energy + k * moise_energy_sigma
    T = T / 1.7
    energy = torch.max(energy - T, torch.zeros_like(T))
    eps = torch.finfo(energy.dtype).eps
    energy_all = energy.sum(dim=[1, 2]) + eps
    an_all = an.sum(dim=[1, 2]) + eps
    result_pc = energy_all / an_all
    return result_pc.unsqueeze(1)


def gradient_map(x: 'torch.Tensor', kernels: 'torch.Tensor') ->torch.Tensor:
    """Compute gradient map for a given tensor and stack of kernels.
    Args:
        x: Tensor with shape (N, C, H, W).
        kernels: Stack of tensors for gradient computation with shape (k_N, k_H, k_W)
    Returns:
        Gradients of x per-channel with shape (N, C, H, W)
    """
    padding = kernels.size(-1) // 2
    grads = torch.nn.functional.conv2d(x, kernels, padding=padding)
    return safe_sqrt(torch.sum(grads ** 2, dim=-3, keepdim=True))


EPS = torch.finfo(torch.float32).eps


def similarity_map(map_x: 'torch.Tensor', map_y: 'torch.Tensor', constant: 'float', alpha: 'float'=0.0) ->torch.Tensor:
    """Compute similarity_map between two tensors using Dice-like equation.
    Args:
        map_x: Tensor with map to be compared
        map_y: Tensor with map to be compared
        constant: Used for numerical stability
        alpha: Masking coefficient. Subtracts - `alpha` * map_x * map_y from denominator and nominator
    """
    return (2.0 * map_x * map_y - alpha * map_x * map_y + constant) / (map_x ** 2 + map_y ** 2 - alpha * map_x * map_y + constant + EPS)


def fsim(x: 'torch.Tensor', y: 'torch.Tensor', chromatic: 'bool'=True, scales: 'int'=4, orientations: 'int'=4, min_length: 'int'=6, mult: 'int'=2, sigma_f: 'float'=0.55, delta_theta: 'float'=1.2, k: 'float'=2.0) ->torch.Tensor:
    """Compute Feature Similarity Index Measure for a batch of images.
    Args:
        - x: An input tensor. Shape :math:`(N, C, H, W)`.
        - y: A target tensor. Shape :math:`(N, C, H, W)`.
        - chromatic: Flag to compute FSIMc, which also takes into account chromatic components
        - scales: Number of wavelets used for computation of phase congruensy maps
        - orientations: Number of filter orientations used for computation of phase congruensy maps
        - min_length: Wavelength of smallest scale filter
        - mult: Scaling factor between successive filters
        - sigma_f: Ratio of the standard deviation of the Gaussian describing the log Gabor filter's
        transfer function in the frequency domain to the filter center frequency.
        - delta_theta: Ratio of angular interval between filter orientations and the standard deviation
        of the angular Gaussian function used to construct filters in the frequency plane.
        - k: No of standard deviations of the noise energy beyond the mean at which we set the noise
            threshold  point, below which phase congruency values get penalized.

    Returns:
        - Index of similarity between two images. Usually in [0, 1] interval.
        Can be bigger than 1 for predicted :math:`x` images with higher contrast than the original ones.
    References:
        L. Zhang, L. Zhang, X. Mou and D. Zhang, "FSIM: A Feature Similarity Index for Image Quality Assessment,"
        IEEE Transactions on Image Processing, vol. 20, no. 8, pp. 2378-2386, Aug. 2011, doi: 10.1109/TIP.2011.2109730.
        https://ieeexplore.ieee.org/document/5705575

    """
    x = x / float(1.0) * 255
    y = y / float(1.0) * 255
    kernel_size = max(1, round(min(x.shape[-2:]) / 256))
    x = torch.nn.functional.avg_pool2d(x, kernel_size)
    y = torch.nn.functional.avg_pool2d(y, kernel_size)
    num_channels = x.size(1)
    if num_channels == 3:
        x_yiq = rgb2yiq(x)
        y_yiq = rgb2yiq(y)
        x_lum = x_yiq[:, :1]
        y_lum = y_yiq[:, :1]
        x_i = x_yiq[:, 1:2]
        y_i = y_yiq[:, 1:2]
        x_q = x_yiq[:, 2:]
        y_q = y_yiq[:, 2:]
    else:
        x_lum = x
        y_lum = y
    pc_x = _phase_congruency(x_lum, scales=scales, orientations=orientations, min_length=min_length, mult=mult, sigma_f=sigma_f, delta_theta=delta_theta, k=k)
    pc_y = _phase_congruency(y_lum, scales=scales, orientations=orientations, min_length=min_length, mult=mult, sigma_f=sigma_f, delta_theta=delta_theta, k=k)
    scharr_filter = torch.tensor([[[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]]) / 16
    kernels = torch.stack([scharr_filter, scharr_filter.transpose(-1, -2)])
    grad_map_x = gradient_map(x_lum, kernels)
    grad_map_y = gradient_map(y_lum, kernels)
    T1, T2, T3, T4, lmbda = 0.85, 160, 200, 200, 0.03
    PC = similarity_map(pc_x, pc_y, T1)
    GM = similarity_map(grad_map_x, grad_map_y, T2)
    pc_max = torch.where(pc_x > pc_y, pc_x, pc_y)
    score = GM * PC * pc_max
    if chromatic:
        assert num_channels == 3, 'Chromatic component can be computed only for RGB images!'
        S_I = similarity_map(x_i, y_i, T3)
        S_Q = similarity_map(x_q, y_q, T4)
        score = score * torch.abs(S_I * S_Q) ** lmbda
    result = score.sum(dim=[1, 2, 3]) / pc_max.sum(dim=[1, 2, 3])
    return result


class FSIM(nn.Module):
    """Args:
        - chromatic: Flag to compute FSIMc, which also takes into account chromatic components
        - scales: Number of wavelets used for computation of phase congruensy maps
        - orientations: Number of filter orientations used for computation of phase congruensy maps
        - min_length: Wavelength of smallest scale filter
        - mult: Scaling factor between successive filters
        - sigma_f: Ratio of the standard deviation of the Gaussian describing the log Gabor filter's
        transfer function in the frequency domain to the filter center frequency.
        - delta_theta: Ratio of angular interval between filter orientations and the standard deviation
        of the angular Gaussian function used to construct filters in the frequency plane.
        - k: No of standard deviations of the noise energy beyond the mean at which we set the noise
            threshold  point, below which phase congruency values get penalized.
    References:
        L. Zhang, L. Zhang, X. Mou and D. Zhang, "FSIM: A Feature Similarity Index for Image Quality Assessment,"
        IEEE Transactions on Image Processing, vol. 20, no. 8, pp. 2378-2386, Aug. 2011, doi: 10.1109/TIP.2011.2109730.
        https://ieeexplore.ieee.org/document/5705575
    """

    def __init__(self, chromatic: 'bool'=True, scales: 'int'=4, orientations: 'int'=4, min_length: 'int'=6, mult: 'int'=2, sigma_f: 'float'=0.55, delta_theta: 'float'=1.2, k: 'float'=2.0) ->None:
        super().__init__()
        self.fsim = functools.partial(fsim, chromatic=chromatic, scales=scales, orientations=orientations, min_length=min_length, mult=mult, sigma_f=sigma_f, delta_theta=delta_theta, k=k)

    def forward(self, X: 'torch.Tensor', Y: 'torch.Tensor') ->torch.Tensor:
        """Computation of FSIM as a loss function.
        Args:
            - x: An input tensor. Shape :math:`(N, C, H, W)`.
            - y: A target tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            - Value of FSIM loss to be minimized in [0, 1] range.
        """
        assert X.shape == Y.shape, f'Input and reference images should have the same shape, but got {X.shape} and {Y.shape}'
        score = self.fsim(X, Y)
        return score


def gmsd(x: 'torch.Tensor', y: 'torch.Tensor', T: 'int'=170, channels: 'int'=3, test_y_channel: 'bool'=True) ->torch.Tensor:
    """GMSD metric.
    Args:
        - x: A distortion tensor. Shape :math:`(N, C, H, W)`.
        - y: A reference tensor. Shape :math:`(N, C, H, W)`.
        - T: A positive constant that supplies numerical stability.
        - channels: Number of channels.
        - test_y_channel: bool, whether to use y channel on ycbcr.
    """
    if test_y_channel:
        x = to_y_channel(x, 255)
        y = to_y_channel(y, 255)
        channels = 1
    else:
        x = x * 255.0
        y = y * 255.0
    dx = (torch.Tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.0).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    dy = (torch.Tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 3.0).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    aveKernel = torch.ones(channels, 1, 2, 2) / 4.0
    Y1 = F.conv2d(x, aveKernel, stride=2, padding=0, groups=channels)
    Y2 = F.conv2d(y, aveKernel, stride=2, padding=0, groups=channels)
    IxY1 = F.conv2d(Y1, dx, stride=1, padding=1, groups=channels)
    IyY1 = F.conv2d(Y1, dy, stride=1, padding=1, groups=channels)
    gradientMap1 = torch.sqrt(IxY1 ** 2 + IyY1 ** 2 + 1e-12)
    IxY2 = F.conv2d(Y2, dx, stride=1, padding=1, groups=channels)
    IyY2 = F.conv2d(Y2, dy, stride=1, padding=1, groups=channels)
    gradientMap2 = torch.sqrt(IxY2 ** 2 + IyY2 ** 2 + 1e-12)
    quality_map = (2 * gradientMap1 * gradientMap2 + T) / (gradientMap1 ** 2 + gradientMap2 ** 2 + T)
    score = torch.std(quality_map.view(quality_map.shape[0], -1), dim=1)
    return score


class GMSD(nn.Module):
    """Gradient Magnitude Similarity Deviation Metric.
    Args:
        - channels: Number of channels.
        - test_y_channel: bool, whether to use y channel on ycbcr.
    Reference:
        Xue, Wufeng, Lei Zhang, Xuanqin Mou, and Alan C. Bovik.
        "Gradient magnitude similarity deviation: A highly efficient
        perceptual image quality index." IEEE Transactions on Image
        Processing 23, no. 2 (2013): 684-695.
    """

    def __init__(self, channels: 'int'=3, test_y_channel: 'bool'=True) ->None:
        super(GMSD, self).__init__()
        self.channels = channels
        self.test_y_channel = test_y_channel

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        """Args:
            x: A distortion tensor. Shape :math:`(N, C, H, W)`.
            y: A reference tensor. Shape :math:`(N, C, H, W)`.
            Order of input is important.
        """
        assert x.shape == y.shape, f'Input and reference images should have the same shape, but got {x.shape} and {y.shape}'
        score = gmsd(x, y, channels=self.channels, test_y_channel=self.test_y_channel)
        return score


class HyperNet(nn.Module):
    """HyperNet Model.
    Args:
        - base_model_name (String): pretrained model to extract features,
        can be any models supported by timm. Default: resnet50.
        - pretrained_model_path (String): Pretrained model path.
        - default_mean (list): Default mean value.
        - default_std (list): Default std value.

    Reference:
        Su, Shaolin, Qingsen Yan, Yu Zhu, Cheng Zhang, Xin Ge,
        Jinqiu Sun, and Yanning Zhang. "Blindly assess image
        quality in the wild guided by a self-adaptive hyper network."
        In Proceedings of the IEEE/CVF Conference on Computer Vision
        and Pattern Recognition (CVPR), pp. 3667-3676. 2020.

    """

    def __init__(self, base_model_name='resnet50', num_crop=25, pretrained=True, pretrained_model_path=None, default_mean=[0.485, 0.456, 0.406], default_std=[0.229, 0.224, 0.225]):
        super(HyperNet, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=True, features_only=True)
        lda_out_channels = 16
        hyper_in_channels = 112
        target_in_size = 224
        hyper_fc_channels = [112, 56, 28, 14, 1]
        feature_size = 7
        self.hyper_fc_channels = hyper_fc_channels
        self.num_crop = num_crop
        self.lda_modules = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False), nn.AvgPool2d(7, stride=7), nn.Flatten(), nn.Linear(16 * 64, lda_out_channels)), nn.Sequential(nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False), nn.AvgPool2d(7, stride=7), nn.Flatten(), nn.Linear(32 * 16, lda_out_channels)), nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0, bias=False), nn.AvgPool2d(7, stride=7), nn.Flatten(), nn.Linear(64 * 4, lda_out_channels)), nn.Sequential(nn.AvgPool2d(7, stride=7), nn.Flatten(), nn.Linear(2048, target_in_size - lda_out_channels * 3))])
        self.fc_w_modules = nn.ModuleList([])
        for i in range(4):
            if i == 0:
                out_ch = int(target_in_size * hyper_fc_channels[i] / feature_size ** 2)
            else:
                out_ch = int(hyper_fc_channels[i - 1] * hyper_fc_channels[i] / feature_size ** 2)
            self.fc_w_modules.append(nn.Conv2d(hyper_in_channels, out_ch, 3, padding=(1, 1)))
        self.fc_w_modules.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hyper_in_channels, hyper_fc_channels[3])))
        self.fc_b_modules = nn.ModuleList([])
        for i in range(5):
            self.fc_b_modules.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hyper_in_channels, hyper_fc_channels[i])))
        self.conv1 = nn.Sequential(nn.Conv2d(2048, 1024, 1, padding=(0, 0)), nn.ReLU(inplace=True), nn.Conv2d(1024, 512, 1, padding=(0, 0)), nn.ReLU(inplace=True), nn.Conv2d(512, hyper_in_channels, 1, padding=(0, 0)), nn.ReLU(inplace=True))
        self.global_pool = nn.Sequential()
        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)
        if pretrained and pretrained_model_path is None:
            load_pretrained_network(self, default_model_urls['resnet50-koniq'], True, weight_keys='params')
        elif pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, weight_keys='params')

    def preprocess(self, x):
        if x.shape[2:] != torch.Size([224, 224]):
            x = nn.functional.interpolate(x, (224, 224), mode='bicubic')
        x = (x - self.default_mean) / self.default_std
        return x

    def forward_patch(self, x):
        assert x.shape[2:] == torch.Size([224, 224]), f'Input patch size must be (224, 224), but got {x.shape[2:]}'
        x = self.preprocess(x)
        base_feats = self.base_model(x)[1:]
        lda_feat_list = []
        for bf, ldam in zip(base_feats, self.lda_modules):
            lda_feat_list.append(ldam(bf))
        lda_feat = torch.cat(lda_feat_list, dim=1)
        target_fc_w = []
        target_fc_b = []
        hyper_in_feat = self.conv1(base_feats[-1])
        batch_size = hyper_in_feat.shape[0]
        for i in range(len(self.fc_w_modules)):
            tmp_fc_w = self.fc_w_modules[i](hyper_in_feat).reshape(batch_size, self.hyper_fc_channels[i], -1)
            target_fc_w.append(tmp_fc_w)
            target_fc_b.append(self.fc_b_modules[i](hyper_in_feat))
        x = lda_feat.unsqueeze(1)
        for i in range(len(target_fc_w)):
            if i != 4:
                x = torch.sigmoid(torch.bmm(x, target_fc_w[i].transpose(1, 2)) + target_fc_b[i].unsqueeze(1))
            else:
                x = torch.bmm(x, target_fc_w[i].transpose(1, 2)) + target_fc_b[i].unsqueeze(1)
        return x.squeeze(-1)

    def forward(self, x):
        """HYPERNET model.
        Args:
            x: A distortion tensor. Shape :math:`(N, C, H, W)`.
        """
        if self.training:
            return self.forward_patch(x)
        else:
            b, c, h, w = x.shape
            crops = uniform_crop([x], 224, self.num_crop)
            results = self.forward_patch(crops)
            results = results.reshape(b, self.num_crop, -1).mean(dim=1)
        return results.unsqueeze(-1)


class InceptionScore(nn.Module):
    """Implements the Inception Score (IS) metric.

    Args:
        dims (int): The number of dimensions of the Inception-v3 feature representation to use.
            Must be one of 64, 192, 768, or 2048. Default: 2048.

    Attributes:
        model (nn.Module): The Inception-v3 network used to extract features.
    """

    def __init__(self) ->None:
        super().__init__()
        self.model = InceptionV3(output_blocks='logits_unbiased')
        self.model.eval()

    def forward(self, img_dir, mode='legacy_tensorflow', splits=10, num_workers=12, batch_size=32, device=torch.device('cuda'), verbose=True, **kwargs):
        if verbose:
            None
        np_feats = get_folder_features(img_dir, self.model, num_workers=num_workers, batch_size=batch_size, device=device, mode=mode, description=f'Inception Score {img_dir}: ', verbose=verbose)
        features = torch.from_numpy(np_feats)
        features = features[torch.randperm(features.shape[0])]
        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)
        prob = prob.chunk(splits, dim=0)
        log_prob = log_prob.chunk(splits, dim=0)
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
        kl_ = [(p * (log_p - m_p.log())) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
        scores = [k.sum(dim=1).mean().exp().item() for k in kl_]
        return {'inception_score_mean': np.mean(scores), 'inception_score_std': np.std(scores)}


class IQARegression(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_enc = nn.Conv2d(in_channels=320 * 6, out_channels=config.d_hidn, kernel_size=1)
        self.conv_dec = nn.Conv2d(in_channels=320 * 6, out_channels=config.d_hidn, kernel_size=1)
        self.transformer = Transformer(self.config)
        self.projection = nn.Sequential(nn.Linear(self.config.d_hidn, self.config.d_MLP_head, bias=False), nn.ReLU(), nn.Linear(self.config.d_MLP_head, self.config.n_output, bias=False))

    def forward(self, enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed):
        enc_inputs_embed = self.conv_enc(enc_inputs_embed)
        dec_inputs_embed = self.conv_dec(dec_inputs_embed)
        b, c, h, w = enc_inputs_embed.size()
        enc_inputs_embed = torch.reshape(enc_inputs_embed, (b, c, h * w))
        enc_inputs_embed = enc_inputs_embed.permute(0, 2, 1)
        dec_inputs_embed = torch.reshape(dec_inputs_embed, (b, c, h * w))
        dec_inputs_embed = dec_inputs_embed.permute(0, 2, 1)
        dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.transformer(enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed)
        dec_outputs = dec_outputs[:, 0, :]
        pred = self.projection(dec_outputs)
        return pred


class MultiHeadAttention(nn.Module):

    def __init__(self, dim, num_heads=6, bias=False, attn_drop=0.0, out_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.query = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1) * self.scale
        if mask is not None:
            mask_h = mask.reshape(B, 1, N, 1)
            mask_w = mask.reshape(B, 1, 1, N)
            mask2d = mask_h * mask_w
            attn = attn.masked_fill(mask2d == 0, -1000.0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out(x)
        x = self.out_drop(x)
        return x


class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(in_channels=self.config.d_hidn, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_hidn, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        output = self.conv1(inputs.transpose(1, 2))
        output = self.active(output)
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        return output


class EncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)

    def forward(self, inputs, attn_mask):
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        att_outputs = self.layer_norm1(inputs + att_outputs)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        return ffn_outputs, attn_prob


def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_embedding = nn.Parameter(torch.randn(1, self.config.n_enc_seq + 1, self.config.d_hidn))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.d_hidn))
        self.dropout = nn.Dropout(self.config.emb_dropout)
        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, inputs, inputs_embed):
        b, n, _ = inputs_embed.shape
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=torch.int64).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, inputs_embed), dim=1)
        x += self.pos_embedding
        outputs = self.dropout(x)
        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)
        attn_probs = []
        for layer in self.layers:
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)
        return outputs, attn_probs


class ScaledDotProductAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / self.config.d_head ** 0.5

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores = scores.mul_(self.scale)
        scores.masked_fill_(attn_mask, -1000000000.0)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, V)
        return context, attn_prob


class DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.dec_enc_attn = MultiHeadAttention(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm3 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, dec_enc_attn_mask):
        self_att_outputs, self_attn_prob = self.self_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        self_att_outputs = self.layer_norm1(dec_inputs + self_att_outputs)
        dec_enc_att_outputs, dec_enc_attn_prob = self.dec_enc_attn(self_att_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_enc_att_outputs = self.layer_norm2(self_att_outputs + dec_enc_att_outputs)
        ffn_outputs = self.pos_ffn(dec_enc_att_outputs)
        ffn_outputs = self.layer_norm3(dec_enc_att_outputs + ffn_outputs)
        return ffn_outputs, self_attn_prob, dec_enc_attn_prob


def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1)
    return subsequent_mask


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_embedding = nn.Parameter(torch.randn(1, self.config.n_enc_seq + 1, self.config.d_hidn))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.d_hidn))
        self.dropout = nn.Dropout(self.config.emb_dropout)
        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, dec_inputs, dec_inputs_embed, enc_inputs, enc_outputs):
        b, n, _ = dec_inputs_embed.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, dec_inputs_embed), dim=1)
        x += self.pos_embedding[:, :n + 1]
        dec_outputs = self.dropout(x)
        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs)
        dec_self_attn_mask = torch.gt(dec_attn_pad_mask + dec_attn_decoder_mask, 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.config.i_pad)
        self_attn_probs, dec_enc_attn_probs = [], []
        for layer in self.layers:
            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            self_attn_probs.append(self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)
        return dec_outputs, self_attn_probs, dec_enc_attn_probs


class MLP(nn.Module):

    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(nn.Linear(self.input_size, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2), nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))

    def forward(self, x):
        return self.layers(x)


class LAIONAes(nn.Module):
    """
    LAIONAes is a class that implements a neural network architecture for image quality assessment.

    The architecture is based on the ViT-L/14 model from the OpenAI CLIP library, and uses an MLP to predict image quality scores.

    Args:
        None

    Returns:
        A tensor representing the predicted image quality scores.
    """

    def __init__(self, pretrained=True, pretrained_model_path=None) ->None:
        super().__init__()
        clip_model, _ = clip.load('ViT-L/14')
        self.mlp = MLP(clip_model.visual.output_dim)
        self.clip_model = [clip_model]
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, weight_keys='params')
        elif pretrained:
            load_pretrained_network(self.mlp, default_model_urls['url'])

    def forward(self, x):
        clip_model = self.clip_model[0]
        if not self.training:
            img = clip_preprocess_tensor(x, clip_model)
        else:
            img = T.functional.normalize(x, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
        img_emb = clip_model.encode_image(img)
        img_emb = nn.functional.normalize(img_emb.float(), p=2, dim=-1)
        score = self.mlp(img_emb)
        return score


dists_map = ['jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color', 'contrast', 'overexposure', 'underexposure', 'spatial', 'quantization', 'other']


qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']


scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']


class LIQE(nn.Module):

    def __init__(self, model_type='liqe', backbone='ViT-B/32', step=32, num_patch=15, pretrained=True, pretrained_model_path=None, mtl=False) ->None:
        super().__init__()
        assert backbone == 'ViT-B/32', 'Only support ViT-B/32 now'
        self.backbone = backbone
        self.clip_model = load(self.backbone, 'cpu')
        self.model_type = model_type
        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)
        self.clip_model.logit_scale.requires_grad = False
        self.step = step
        self.num_patch = num_patch
        if pretrained_model_path is None and pretrained:
            url_key = 'koniq' if isinstance(pretrained, bool) else pretrained
            pretrained_model_path = default_model_urls[url_key]
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, 'params')
        if pretrained == 'mix':
            self.mtl = True
            text_feat_cache_path = os.path.join(DEFAULT_CACHE_DIR, 'liqe_text_feat_mix.pt')
        else:
            self.mtl = mtl
            text_feat_cache_path = os.path.join(DEFAULT_CACHE_DIR, 'liqe_text_feat.pt')
        if os.path.exists(text_feat_cache_path):
            self.text_features = torch.load(text_feat_cache_path, map_location='cpu', weights_only=False)
        else:
            None
            if self.mtl:
                self.joint_texts = torch.cat([clip.tokenize(f'a photo of a {c} with {d} artifacts, which is of {q} quality') for q, c, d in product(qualitys, scenes, dists_map)])
            else:
                self.joint_texts = torch.cat([clip.tokenize(f'a photo with {c} quality') for c in qualitys])
            self.text_features = self.get_text_features(self.joint_texts)
            torch.save(self.text_features, text_feat_cache_path)

    def get_text_features(self, x):
        text_features = self.clip_model.encode_text(self.joint_texts)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def forward(self, x):
        bs = x.size(0)
        h = x.size(2)
        w = x.size(3)
        assert (h >= 224) & (w >= 224), 'Short side is less than 224, try upsampling the original image'
        x = (x - self.default_mean) / self.default_std
        x = x.unfold(2, 224, self.step).unfold(3, 224, self.step).permute(0, 2, 3, 1, 4, 5).reshape(bs, -1, 3, 224, 224)
        if x.size(1) < self.num_patch:
            num_patch = x.size(1)
        else:
            num_patch = self.num_patch
        if self.training:
            sel = torch.randint(low=0, high=x.size(0), size=(num_patch,))
        else:
            sel_step = max(1, x.size(1) // num_patch)
            sel = torch.zeros(num_patch)
            for i in range(num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        x = x[:, sel, ...]
        x = x.reshape(bs, num_patch, x.shape[2], x.shape[3], x.shape[4])
        text_features = self.text_features
        x = x.view(bs * x.size(1), x.size(2), x.size(3), x.size(4))
        image_features = self.clip_model.encode_image(x, pos_embedding=True)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_image = logits_per_image.view(bs, num_patch, -1)
        logits_per_image = logits_per_image.mean(1)
        logits_per_image = F.softmax(logits_per_image, dim=1)
        if self.mtl:
            logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map))
            logits_quality = logits_per_image.sum(3).sum(2)
        else:
            logits_per_image = logits_per_image.view(-1, len(qualitys))
            logits_quality = logits_per_image
        quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + 4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
        return quality.unsqueeze(1)


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        None
    return PadLayer


class Downsample(nn.Module):

    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0, pad_size='', pad_more=False):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        if pad_size == '2k' or pad_more == True:
            self.pad_sizes = [int(1.0 * (filt_size - 1)), int(np.ceil(1.0 * (filt_size - 1))), int(1.0 * (filt_size - 1)), int(np.ceil(1.0 * (filt_size - 1)))]
        elif pad_size == 'none':
            self.pad_sizes = [0, 0, 0, 0]
        else:
            self.pad_sizes = [int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2)), int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2))]
        self.pad_sizes = [(pad_size + pad_off) for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        if self.filt_size == 1:
            a = np.array([1.0])
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class alexnet(nn.Module):

    def __init__(self, requires_grad=False, variant='shift_tolerant', filter_size=3):
        super(alexnet, self).__init__()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        if variant == 'vanilla':
            features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))
            for x in range(2):
                self.slice1.add_module(str(x), features[x])
            for x in range(2, 5):
                self.slice2.add_module(str(x), features[x])
            for x in range(5, 8):
                self.slice3.add_module(str(x), features[x])
            for x in range(8, 10):
                self.slice4.add_module(str(x), features[x])
            for x in range(10, 12):
                self.slice5.add_module(str(x), features[x])
        elif variant == 'antialiased':
            features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2), nn.ReLU(inplace=True), Downsample(filt_size=filter_size, stride=2, channels=64), nn.MaxPool2d(kernel_size=3, stride=1), Downsample(filt_size=filter_size, stride=2, channels=64), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=1), Downsample(filt_size=filter_size, stride=2, channels=192), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=1), Downsample(filt_size=filter_size, stride=2, channels=256))
            for x in range(2):
                self.slice1.add_module(str(x), features[x])
            for x in range(2, 7):
                self.slice2.add_module(str(x), features[x])
            for x in range(7, 11):
                self.slice3.add_module(str(x), features[x])
            for x in range(11, 13):
                self.slice4.add_module(str(x), features[x])
            for x in range(13, 15):
                self.slice5.add_module(str(x), features[x])
        elif variant == 'shift_tolerant':
            features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=2), Downsample(filt_size=filter_size, stride=2, channels=64, pad_more=True), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=1), Downsample(filt_size=filter_size, stride=2, channels=64, pad_more=True), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=1), Downsample(filt_size=filter_size, stride=2, channels=192, pad_more=True), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=1), Downsample(filt_size=filter_size, stride=2, channels=256, pad_more=True))
            for x in range(3):
                self.slice1.add_module(str(x), features[x])
            for x in range(3, 7):
                self.slice2.add_module(str(x), features[x])
            for x in range(7, 11):
                self.slice3.add_module(str(x), features[x])
            for x in range(11, 13):
                self.slice4.add_module(str(x), features[x])
            for x in range(13, 15):
                self.slice5.add_module(str(x), features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple('AlexnetOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)
        return out


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class squeezenet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = models.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple('SqueezeOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7'])
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)
        return out


def upsample(in_tens, out_HW=(64, 64)):
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels != m.out_channels or m.out_channels != m.groups or m.bias is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    None
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


cfg = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}


def make_layers(cfg: 'List[Union[str, int]]') ->nn.Sequential:
    layers: 'List[nn.Module]' = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(pretrained=False, filter_size=1, pad_more=False, fconv=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], filter_size=filter_size, pad_more=pad_more, fconv=fconv), **kwargs)
    return model


class LPIPS(nn.Module):
    """ LPIPS model.
    Args:
        lpips (Boolean) : Whether to use linear layers on top of base/trunk network.
        pretrained (Boolean): Whether means linear layers are calibrated with human
            perceptual judgments.
        pnet_rand (Boolean): Whether to randomly initialized trunk.
        net (String): ['alex','vgg','squeeze'] are the base/trunk networks available.
        version (String): choose the version ['v0.1'] is the default and latest;
            ['v0.0'] contained a normalization bug.
        pretrained_model_path (String): Petrained model path.

        The following parameters should only be changed if training the network:

        eval_mode (Boolean): choose the mode; True is for test mode (default).
        pnet_tune (Boolean): Whether to tune the base/trunk network.
        use_dropout (Boolean): Whether to use dropout when training linear layers.


        """

    def __init__(self, pretrained=True, net='alex', version='0.1', lpips=True, spatial=False, pnet_rand=False, pnet_tune=False, use_dropout=True, pretrained_model_path=None, eval_mode=True, semantic_weight_layer=-1, **kwargs):
        super(LPIPS, self).__init__()
        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.version = version
        self.scaling_layer = ScalingLayer()
        self.semantic_weight_layer = semantic_weight_layer
        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == 'squeeze':
            net_type = squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)
        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == 'squeeze':
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]
            self.lins = nn.ModuleList(self.lins)
            if pretrained_model_path is not None:
                load_pretrained_network(self, pretrained_model_path, False)
            elif pretrained:
                load_pretrained_network(self, default_model_urls[f'{version}_{net}'], False)
        if eval_mode:
            self.eval()

    def forward(self, in1, in0, retPerLayer=False, normalize=True):
        """Computation IQA using LPIPS.
        Args:
            in1: An input tensor. Shape :math:`(N, C, H, W)`.
            in0: A reference tensor. Shape :math:`(N, C, H, W)`.
            retPerLayer (Boolean): return result contains result of
                each layer or not. Default: False.
            normalize (Boolean): Whether to normalize image data range
                in [0,1] to [-1,1]. Default: True.

        Returns:
            Quality score.

        """
        if normalize:
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            elif self.semantic_weight_layer >= 0:
                res = []
                semantic_feat = outs0[self.semantic_weight_layer]
                for kk in range(self.L):
                    diff_score = self.lins[kk](diffs[kk])
                    semantic_weight = torch.nn.functional.interpolate(semantic_feat, size=diff_score.shape[2:], mode='bilinear', align_corners=False)
                    avg_score = torch.sum(diff_score * semantic_weight, dim=[1, 2, 3], keepdim=True) / torch.sum(semantic_weight, dim=[1, 2, 3], keepdim=True)
                    res.append(avg_score)
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        elif self.spatial:
            res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
        else:
            res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]
        val = 0
        for i in range(self.L):
            val += res[i]
        if retPerLayer:
            return val, res
        else:
            return val.squeeze(-1).squeeze(-1)


class resnet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if num == 18:
            self.net = models.resnet18(pretrained=pretrained)
        elif num == 34:
            self.net = models.resnet34(pretrained=pretrained)
        elif num == 50:
            self.net = models.resnet50(pretrained=pretrained)
        elif num == 101:
            self.net = models.resnet101(pretrained=pretrained)
        elif num == 152:
            self.net = models.resnet152(pretrained=pretrained)
        self.N_slices = 5
        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h
        outputs = namedtuple('Outputs', ['relu1', 'conv2', 'conv3', 'conv4', 'conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)
        return out


MAX = nn.MaxPool2d((2, 2), stride=1, padding=1)


def extract_patches_2d(img: 'torch.Tensor', patch_shape: 'list'=[64, 64], step: 'list'=[27, 27], batch_first: 'bool'=True, keep_last_patch: 'bool'=False) ->torch.Tensor:
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if img.size(2) < patch_H:
        num_padded_H_Top = (patch_H - img.size(2)) // 2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0, 0, num_padded_H_Top, num_padded_H_Bottom), 0)
        img = padding_H(img)
    if img.size(3) < patch_W:
        num_padded_W_Left = (patch_W - img.size(3)) // 2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left, num_padded_W_Right, 0, 0), 0)
        img = padding_W(img)
    step_int = [0, 0]
    step_int[0] = int(patch_H * step[0]) if isinstance(step[0], float) else step[0]
    step_int[1] = int(patch_W * step[1]) if isinstance(step[1], float) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if (img.size(2) - patch_H) % step_int[0] != 0 and keep_last_patch:
        patches_fold_H = torch.cat((patches_fold_H, img[:, :, -patch_H:].permute(0, 1, 3, 2).unsqueeze(2)), dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])
    if (img.size(3) - patch_W) % step_int[1] != 0 and keep_last_patch:
        patches_fold_HW = torch.cat((patches_fold_HW, patches_fold_H[:, :, :, -patch_W:, :].permute(0, 1, 2, 4, 3).unsqueeze(3)), dim=3)
    patches = patches_fold_HW.permute(2, 3, 0, 1, 4, 5)
    patches = patches.reshape(-1, img.size(0), img.size(1), patch_H, patch_W)
    if batch_first:
        patches = patches.permute(1, 0, 2, 3, 4)
    return patches


def get_moments(d, sk=False):
    mean = torch.mean(d, dim=[3, 4], keepdim=True)
    diffs = d - mean
    var = torch.mean(torch.pow(diffs, 2.0), dim=[3, 4], keepdim=True)
    std = torch.pow(var + 1e-12, 0.5)
    if sk:
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0), dim=[3, 4], keepdim=True)
        kurtoses = torch.mean(torch.pow(zscores, 4.0), dim=[3, 4], keepdim=True) - 3.0
        return mean, std, skews, kurtoses
    else:
        return mean, std


def ical_std(x, p=16, s=4):
    B, C, H, W = x.shape
    x1 = extract_patches_2d(x, patch_shape=[p, p], step=[s, s])
    mean, std = get_moments(x1)
    mean = mean.reshape(B, C, (H - (p - s)) // s, (W - (p - s)) // s)
    std = std.reshape(B, C, (H - (p - s)) // s, (W - (p - s)) // s)
    return mean, std


def make_csf(rows, cols, nfreq):
    xvals = np.arange(-(cols - 1) / 2.0, (cols + 1) / 2.0)
    yvals = np.arange(-(rows - 1) / 2.0, (rows + 1) / 2.0)
    xplane, yplane = np.meshgrid(xvals, yvals)
    plane = (xplane + 1.0j * yplane) / cols * 2 * nfreq
    radfreq = np.abs(plane)
    w = 0.7
    s = (1 - w) / 2 * np.cos(4 * np.angle(plane)) + (1 + w) / 2
    radfreq = radfreq / s
    csf = 2.6 * (0.0192 + 0.114 * radfreq) * np.exp(-(0.114 * radfreq) ** 1.1)
    csf[radfreq < 7.8909] = 0.9809
    return np.transpose(csf)


def hi_index(ref_img, dst_img):
    k = 0.02874
    G = 0.5
    C_slope = 1
    Ci_thrsh = -5
    Cd_thrsh = -5
    ref = k * (ref_img + 1e-12) ** (2.2 / 3)
    dst = k * (torch.abs(dst_img) + 1e-12) ** (2.2 / 3)
    B, C, H, W = ref.shape
    csf = make_csf(H, W, 32)
    csf = torch.from_numpy(csf.reshape(1, 1, H, W, 1)).float().repeat(1, C, 1, 1, 2)
    x = torch.fft.fft2(ref)
    x1 = math_util.batch_fftshift2d(x)
    x2 = math_util.batch_ifftshift2d(x1 * csf)
    ref = torch.fft.ifft2(x2).real
    x = torch.fft.fft2(dst)
    x1 = math_util.batch_fftshift2d(x)
    x2 = math_util.batch_ifftshift2d(x1 * csf)
    dst = torch.fft.ifft2(x2).real
    m1_1, std_1 = ical_std(ref)
    B, C, H1, W1 = m1_1.shape
    std_1 = (-MAX(-std_1) / 2)[:, :, :H1, :W1]
    _, std_2 = ical_std(dst - ref)
    BSIZE = 16
    eps = 1e-12
    Ci_ref = torch.log(torch.abs((std_1 + eps) / (m1_1 + eps)))
    Ci_dst = torch.log(torch.abs((std_2 + eps) / (m1_1 + eps)))
    Ci_dst = Ci_dst.masked_fill(m1_1 < G, -1000)
    idx1 = (Ci_ref > Ci_thrsh) & (Ci_dst > C_slope * (Ci_ref - Ci_thrsh) + Cd_thrsh)
    idx2 = (Ci_ref <= Ci_thrsh) & (Ci_dst > Cd_thrsh)
    msk = Ci_ref.clone()
    msk = msk.masked_fill(~idx1, 0)
    msk = msk.masked_fill(~idx2, 0)
    msk[idx1] = Ci_dst[idx1] - (C_slope * (Ci_ref[idx1] - Ci_thrsh) + Cd_thrsh)
    msk[idx2] = Ci_dst[idx2] - Cd_thrsh
    win = torch.ones((1, 1, BSIZE, BSIZE)).repeat(C, 1, 1, 1) / BSIZE ** 2
    xx = (ref_img - dst_img) ** 2
    lmse = F.conv2d(xx, win, stride=4, padding=0, groups=C)
    mp = msk * lmse
    B, C, H, W = mp.shape
    return torch.norm(mp.reshape(B, C, -1), dim=2) / math.sqrt(H * W) * 200


def gaborconvolve(im):
    nscale = 5
    norient = 4
    minWaveLength = 3
    mult = 3
    sigmaOnf = 0.55
    wavelength = [minWaveLength, minWaveLength * mult, minWaveLength * mult ** 2, minWaveLength * mult ** 3, minWaveLength * mult ** 4]
    dThetaOnSigma = 1.5
    B, C, rows, cols = im.shape
    imagefft = torch.fft.fft2(im)
    x = np.ones((rows, 1)) * np.arange(-cols / 2.0, cols / 2.0) / (cols / 2.0)
    y = np.dot(np.expand_dims(np.arange(-rows / 2.0, rows / 2.0), 1), np.ones((1, cols)) / (rows / 2.0))
    radius = np.sqrt(x ** 2 + y ** 2)
    radius[int(np.round(rows / 2 + 1)), int(np.round(cols / 2 + 1))] = 1
    radius = np.log(radius + 1e-12)
    theta = np.arctan2(-y, x)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    thetaSigma = math.pi / norient / dThetaOnSigma
    logGabors = []
    for s in range(nscale):
        fo = 1.0 / wavelength[s]
        rfo = fo / 0.5
        tmp = -(2 * np.log(sigmaOnf) ** 2)
        tmp2 = np.log(rfo)
        logGabors.append(np.exp((radius - tmp2) ** 2 / tmp))
        logGabors[s][int(np.round(rows / 2)), int(np.round(cols / 2))] = 0
    E0 = [[], [], [], []]
    for o in range(norient):
        angl = o * math.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread = np.exp(-dtheta ** 2 / (2 * thetaSigma ** 2))
        for s in range(nscale):
            filter = fftshift(logGabors[s] * spread)
            filter = torch.from_numpy(filter).reshape(1, 1, rows, cols)
            e0 = torch.fft.ifft2(imagefft * filter)
            E0[o].append(torch.stack((e0.real, e0.imag), -1))
    return E0


def ical_stat(x, p=16, s=4):
    B, C, H, W = x.shape
    x1 = extract_patches_2d(x, patch_shape=[p, p], step=[s, s])
    _, std, skews, kurt = get_moments(x1, sk=True)
    STD = std.reshape(B, C, (H - (p - s)) // s, (W - (p - s)) // s)
    SKEWS = skews.reshape(B, C, (H - (p - s)) // s, (W - (p - s)) // s)
    KURT = kurt.reshape(B, C, (H - (p - s)) // s, (W - (p - s)) // s)
    return STD, SKEWS, KURT


def lo_index(ref, dst):
    gabRef = gaborconvolve(ref)
    gabDst = gaborconvolve(dst)
    s = [0.5 / 13.25, 0.75 / 13.25, 1 / 13.25, 5 / 13.25, 6 / 13.25]
    mp = 0
    for gb_i in range(4):
        for gb_j in range(5):
            stdref, skwref, krtref = ical_stat(math_util.abs(gabRef[gb_i][gb_j]))
            stddst, skwdst, krtdst = ical_stat(math_util.abs(gabDst[gb_i][gb_j]))
            mp = mp + s[gb_i] * (torch.abs(stdref - stddst) + 2 * torch.abs(skwref - skwdst) + torch.abs(krtref - krtdst))
    B, C, rows, cols = mp.shape
    return torch.norm(mp.reshape(B, C, -1), dim=2) / np.sqrt(rows * cols)


class MAD(torch.nn.Module):
    """Args:
        - channel: Number of input channel.
        - test_y_channel: bool, whether to use y channel on ycbcr which mimics official matlab code.
    References:
        Larson, Eric Cooper, and Damon Michael Chandler. "Most apparent distortion: full-reference
        image quality assessment and the role of strategy." Journal of electronic imaging 19, no. 1
        (2010): 011006.
    """

    def __init__(self, channels=3, test_y_channel=False):
        super(MAD, self).__init__()
        self.channels = channels
        self.test_y_channel = test_y_channel

    def mad(self, ref, dst):
        """Compute MAD for a batch of images.
        Args:
            ref: An reference tensor. Shape :math:`(N, C, H, W)`.
            dst: A distortion tensor. Shape :math:`(N, C, H, W)`.
        """
        if self.test_y_channel and ref.shape[1] == 3:
            ref = to_y_channel(ref, 255.0)
            dst = to_y_channel(dst, 255.0)
            self.channels = 1
        else:
            ref = ref * 255
            dst = dst * 255
        HI = hi_index(ref, dst)
        LO = lo_index(ref, dst)
        thresh1 = 2.55
        thresh2 = 3.35
        b1 = math.exp(-thresh1 / thresh2)
        b2 = 1 / (math.log(10) * thresh2)
        sig = 1 / (1 + b1 * HI ** b2)
        MAD = LO ** (1 - sig) * HI ** sig
        return MAD.mean(1)

    def forward(self, X, Y):
        """Computation of CW-SSIM metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
            Y: A target tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of MAD metric in [0, 1] range.
        """
        assert X.shape == Y.shape, f'Input and reference images should have the same shape, but got {X.shape} and {Y.shape}'
        score = self.mad(Y, X)
        return score


class TABlock(nn.Module):

    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


IMAGENET_INCEPTION_MEAN = 0.5, 0.5, 0.5


IMAGENET_INCEPTION_STD = 0.5, 0.5, 0.5


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """ Feed-forward network (FFN, a.k.a. MLP) class. """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def get_relative_position_index(win_h, win_w):
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))
        self.register_buffer('relative_position_index', get_relative_position_index(win_h, win_w))
        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self) ->torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask: 'Optional[torch.Tensor]'=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size: 'int'):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: 'int', H: 'int', W: 'int'):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads=4, head_dim=None, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, num_heads=num_heads, head_dim=head_dim, window_size=to_2tuple(self.window_size), qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            cnt = 0
            for h in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)):
                for w in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        _assert(L == H * W, 'input feature has wrong size')
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        head_dim (int): Channels per head (dim // num_heads if not set)
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads=4, head_dim=None, window_size=7, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.grad_checkpointing = False
        self.blocks = nn.Sequential(*[SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, head_dim=head_dim, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _, _, H, W = x.shape
        out_H, out_W = H // self.patch_size[0], W // self.patch_size[1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        out = self.norm(x)
        return out, (out_H, out_W)


class PatchMerging(nn.Module):
    """ Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        _assert(L == H * W, 'input feature has wrong size')
        _assert(H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.')
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


to_ntuple = _ntuple


class SwinTransformer(nn.Module):
    """ Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        head_dim (int, tuple(int)):
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, global_pool='avg', embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), head_dim=None, window_size=7, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, weight_init='', **kwargs):
        super().__init__()
        assert global_pool in ('', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) if ape else None
        self.pos_drop = nn.Dropout(p=drop_rate)
        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        embed_out_dim = embed_dim[1:] + [None]
        head_dim = to_ntuple(self.num_layers)(head_dim)
        window_size = to_ntuple(self.num_layers)(window_size)
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        layers = []
        for i in range(self.num_layers):
            layers += [BasicLayer(dim=embed_dim[i], out_dim=embed_out_dim[i], input_resolution=(self.patch_grid[0] // 2 ** i, self.patch_grid[1] // 2 ** i), depth=depths[i], num_heads=num_heads[i], head_dim=head_dim[i], window_size=window_size[i], mlp_ratio=mlp_ratio[i], qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], norm_layer=norm_layer, downsample=PatchMerging if i < self.num_layers - 1 else None)]
        self.layers = nn.Sequential(*layers)
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'absolute_pos_embed'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^absolute_pos_embed|patch_embed', blocks='^layers\\.(\\d+)' if coarse else [('^layers\\.(\\d+).downsample', (0,)), ('^layers\\.(\\d+)\\.\\w+\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: 'bool'=False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def random_crop(input_list, crop_size, crop_num):
    """
    Randomly crops the input tensor(s) to the specified size and number of crops.

    Args:
        input_list (list or torch.Tensor): List of input tensors or a single input tensor.
        crop_size (int or tuple): Size of the crop. If an int is provided, a square crop of that size is used.
                                  If a tuple is provided, a crop of that size is used.
        crop_num (int): Number of crops to generate.

    Returns:
        torch.Tensor or list of torch.Tensor: If a single input tensor is provided, a tensor of cropped images is returned.
                                              If a list of input tensors is provided, a list of tensors of cropped images is returned.
    """
    if not isinstance(input_list, collections.abc.Sequence):
        input_list = [input_list]
    b, c, h, w = input_list[0].shape
    ch, cw = to_2tuple(crop_size)
    if min(h, w) <= crop_size:
        scale_factor = (crop_size + 1) / min(h, w)
        input_list = [F.interpolate(x, scale_factor=scale_factor, mode='bilinear') for x in input_list]
        b, c, h, w = input_list[0].shape
    crops_list = [[] for _ in range(len(input_list))]
    for _ in range(crop_num):
        sh = np.random.randint(0, h - ch + 1)
        sw = np.random.randint(0, w - cw + 1)
        for j in range(len(input_list)):
            crops_list[j].append(input_list[j][..., sh:sh + ch, sw:sw + cw])
    for i in range(len(crops_list)):
        crops_list[i] = torch.stack(crops_list[i], dim=1).reshape(b * crop_num, c, ch, cw)
    if len(crops_list) == 1:
        crops_list = crops_list[0]
    return crops_list


class MANIQA(nn.Module):
    """
    Implementation of the MANIQA model for image quality assessment.

    Args:
        - embed_dim (int): Embedding dimension for the model. Default is 768.
        - num_outputs (int): Number of output scores. Default is 1.
        - patch_size (int): Size of patches for the model. Default is 8.
        - drop (float): Dropout rate for the model. Default is 0.1.
        - depths (list): List of depths for the Swin Transformer blocks. Default is [2, 2].
        - window_size (int): Window size for the Swin Transformer blocks. Default is 4.
        - dim_mlp (int): Dimension of the MLP for the Swin Transformer blocks. Default is 768.
        - num_heads (list): List of number of heads for the Swin Transformer blocks. Default is [4, 4].
        - img_size (int): Size of the input image. Default is 224.
        - num_tab (int): Number of TA blocks for the model. Default is 2.
        - scale (float): Scale for the Swin Transformer blocks. Default is 0.13.
        - test_sample (int): Number of test samples for the model. Default is 20.
        - pretrained (bool): Whether to use a pretrained model. Default is True.
        - pretrained_model_path (str): Path to the pretrained model. Default is None.
        - train_dataset (str): Name of the training dataset. Default is 'pipal'.
        - default_mean (torch.Tensor): Default mean for the model. Default is None.
        - default_std (torch.Tensor): Default standard deviation for the model. Default is None.

    Returns:
        torch.Tensor: Predicted quality score for the input image.
    """

    def __init__(self, embed_dim=768, num_outputs=1, patch_size=8, drop=0.1, depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4], img_size=224, num_tab=2, scale=0.13, test_sample=20, pretrained=True, pretrained_model_path=None, train_dataset='pipal', default_mean=None, default_std=None, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.test_sample = test_sample
        self.patches_resolution = img_size // patch_size, img_size // patch_size
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock1.append(tab)
        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(patches_resolution=self.patches_resolution, depths=depths, num_heads=num_heads, embed_dim=embed_dim, window_size=window_size, dim_mlp=dim_mlp, scale=scale)
        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(patches_resolution=self.patches_resolution, depths=depths, num_heads=num_heads, embed_dim=embed_dim // 2, window_size=window_size, dim_mlp=dim_mlp, scale=scale)
        self.fc_score = nn.Sequential(nn.Linear(embed_dim // 2, embed_dim // 2), nn.ReLU(), nn.Dropout(drop), nn.Linear(embed_dim // 2, num_outputs), nn.ReLU())
        self.fc_weight = nn.Sequential(nn.Linear(embed_dim // 2, embed_dim // 2), nn.ReLU(), nn.Dropout(drop), nn.Linear(embed_dim // 2, num_outputs), nn.Sigmoid())
        if default_mean is None and default_std is None:
            self.default_mean = torch.Tensor(IMAGENET_INCEPTION_MEAN).view(1, 3, 1, 1)
            self.default_std = torch.Tensor(IMAGENET_INCEPTION_STD).view(1, 3, 1, 1)
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, weight_keys='params')
        elif pretrained:
            load_pretrained_network(self, default_model_urls[train_dataset], True)

    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def forward(self, x):
        """
        Forward pass of the MANIQA model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Predicted quality score for the input image.
        """
        x = (x - self.default_mean) / self.default_std
        bsz = x.shape[0]
        if self.training:
            x = random_crop(x, crop_size=224, crop_num=1)
        else:
            x = uniform_crop(x, crop_size=224, crop_num=self.test_sample)
        _x = self.vit(x)
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = self.swintransformer1(x)
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        per_patch_score = self.fc_score(x)
        per_patch_score = per_patch_score.reshape(bsz, -1)
        per_patch_weight = self.fc_weight(x)
        per_patch_weight = per_patch_weight.reshape(bsz, -1)
        score = (per_patch_weight * per_patch_score).sum(dim=-1) / (per_patch_weight.sum(dim=-1) + 1e-08)
        return score.unsqueeze(1)


class SwinBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, dim_mlp=1024.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.dim_mlp = dim_mlp
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = self.dim_mlp
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def color_space_transform(input_color, fromSpace2toSpace):
    """
    Transforms inputs between different color spaces
    :param input_color: tensor of colors to transform (with NxCxHxW layout)
    :param fromSpace2toSpace: string describing transform
    :return: transformed tensor (with NxCxHxW layout)
    """
    dim = input_color.size()
    device = input_color.device
    reference_illuminant = torch.tensor([[[0.950428545]], [[1.0]], [[1.088900371]]])
    inv_reference_illuminant = torch.tensor([[[1.052156925]], [[1.0]], [[0.91835767]]])
    if fromSpace2toSpace == 'srgb2linrgb':
        limit = 0.04045
        transformed_color = torch.where(input_color > limit, torch.pow((torch.clamp(input_color, min=limit) + 0.055) / 1.055, 2.4), input_color / 12.92)
    elif fromSpace2toSpace == 'linrgb2srgb':
        limit = 0.0031308
        transformed_color = torch.where(input_color > limit, 1.055 * torch.pow(torch.clamp(input_color, min=limit), 1.0 / 2.4) - 0.055, 12.92 * input_color)
    elif fromSpace2toSpace in ['linrgb2xyz', 'xyz2linrgb']:
        if fromSpace2toSpace == 'linrgb2xyz':
            a11 = 10135552 / 24577794
            a12 = 8788810 / 24577794
            a13 = 4435075 / 24577794
            a21 = 2613072 / 12288897
            a22 = 8788810 / 12288897
            a23 = 887015 / 12288897
            a31 = 1425312 / 73733382
            a32 = 8788810 / 73733382
            a33 = 70074185 / 73733382
        else:
            a11 = 3.241003275
            a12 = -1.537398934
            a13 = -0.498615861
            a21 = -0.969224334
            a22 = 1.875930071
            a23 = 0.041554224
            a31 = 0.055639423
            a32 = -0.204011202
            a33 = 1.057148933
        A = torch.Tensor([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        input_color = input_color.view(dim[0], dim[1], dim[2] * dim[3])
        transformed_color = torch.matmul(A, input_color)
        transformed_color = transformed_color.view(dim[0], dim[1], dim[2], dim[3])
    elif fromSpace2toSpace == 'xyz2ycxcz':
        input_color = torch.mul(input_color, inv_reference_illuminant)
        y = 116 * input_color[:, 1:2, :, :] - 16
        cx = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
        cz = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])
        transformed_color = torch.cat((y, cx, cz), 1)
    elif fromSpace2toSpace == 'ycxcz2xyz':
        y = (input_color[:, 0:1, :, :] + 16) / 116
        cx = input_color[:, 1:2, :, :] / 500
        cz = input_color[:, 2:3, :, :] / 200
        x = y + cx
        z = y - cz
        transformed_color = torch.cat((x, y, z), 1)
        transformed_color = torch.mul(transformed_color, reference_illuminant)
    elif fromSpace2toSpace == 'xyz2lab':
        input_color = torch.mul(input_color, inv_reference_illuminant)
        delta = 6 / 29
        delta_square = delta * delta
        delta_cube = delta * delta_square
        factor = 1 / (3 * delta_square)
        clamped_term = torch.pow(torch.clamp(input_color, min=delta_cube), 1.0 / 3.0)
        div = factor * input_color + 4 / 29
        input_color = torch.where(input_color > delta_cube, clamped_term, div)
        L = 116 * input_color[:, 1:2, :, :] - 16
        a = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
        b = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])
        transformed_color = torch.cat((L, a, b), 1)
    elif fromSpace2toSpace == 'lab2xyz':
        y = (input_color[:, 0:1, :, :] + 16) / 116
        a = input_color[:, 1:2, :, :] / 500
        b = input_color[:, 2:3, :, :] / 200
        x = y + a
        z = y - b
        xyz = torch.cat((x, y, z), 1)
        delta = 6 / 29
        delta_square = delta * delta
        factor = 3 * delta_square
        xyz = torch.where(xyz > delta, torch.pow(xyz, 3), factor * (xyz - 4 / 29))
        transformed_color = torch.mul(xyz, reference_illuminant)
    elif fromSpace2toSpace == 'srgb2xyz':
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
    elif fromSpace2toSpace == 'srgb2ycxcz':
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
    elif fromSpace2toSpace == 'linrgb2ycxcz':
        transformed_color = color_space_transform(input_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
    elif fromSpace2toSpace == 'srgb2lab':
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    elif fromSpace2toSpace == 'linrgb2lab':
        transformed_color = color_space_transform(input_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    elif fromSpace2toSpace == 'ycxcz2linrgb':
        transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
    elif fromSpace2toSpace == 'lab2srgb':
        transformed_color = color_space_transform(input_color, 'lab2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2srgb')
    elif fromSpace2toSpace == 'ycxcz2lab':
        transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    else:
        raise ValueError('Error: The color transform %s is not defined!' % fromSpace2toSpace)
    return transformed_color


class MS_SWD_learned(nn.Module):

    def __init__(self, resize_input: 'bool'=True, pretrained: 'bool'=True, pretrained_model_path: 'str'=None, **kwargs):
        super(MS_SWD_learned, self).__init__()
        self.conv11x11 = nn.Conv2d(3, 128, kernel_size=11, stride=1, padding=5, padding_mode='reflect', dilation=1, bias=False)
        self.conv_m1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.relu = nn.LeakyReLU()
        self.resize_input = resize_input
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, weight_keys='params')
        elif pretrained:
            load_pretrained_network(self, get_url_from_name('msswd_weights.pth'), weight_keys='net_dict')

    def preprocess_img(self, x):
        if self.resize_input and min(x.shape[2:]) > 256:
            x = TF.resize(x, 256)
        return x

    def forward_once(self, x):
        x = color_space_transform(x, 'srgb2lab')
        x = self.conv11x11(x)
        x = self.relu(x)
        x = self.conv_m1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x

    def forward(self, x, y):
        x = self.preprocess_img(x)
        y = self.preprocess_img(y)
        output_x = self.forward_once(x)
        output_y = self.forward_once(y)
        output_x, _ = torch.sort(output_x, dim=2)
        output_y, _ = torch.sort(output_y, dim=2)
        swd = torch.abs(output_x - output_y)
        swd = torch.mean(swd, dim=[1, 2])
        return swd


class TransformerBlock(nn.Module):

    def __init__(self, dim, mlp_dim, num_heads, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-06)
        self.attention = MultiHeadAttention(dim, num_heads, bias=True, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, eps=1e-06)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, inputs_masks):
        y = self.norm1(x)
        y = self.attention(y, inputs_masks)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AddHashSpatialPositionEmbs(nn.Module):
    """Adds learnable hash-based spatial embeddings to the inputs."""

    def __init__(self, spatial_pos_grid_size, dim):
        super().__init__()
        self.position_emb = nn.parameter.Parameter(torch.randn(1, spatial_pos_grid_size * spatial_pos_grid_size, dim))
        nn.init.normal_(self.position_emb, std=0.02)

    def forward(self, inputs, inputs_positions):
        return inputs + self.position_emb.squeeze(0)[inputs_positions.long()]


class AddScaleEmbs(nn.Module):
    """Adds learnable scale embeddings to the inputs."""

    def __init__(self, num_scales, dim):
        super().__init__()
        self.scale_emb = nn.parameter.Parameter(torch.randn(num_scales, dim))
        nn.init.normal_(self.scale_emb, std=0.02)

    def forward(self, inputs, inputs_scale_positions):
        return inputs + self.scale_emb[inputs_scale_positions.long()]


def dist_to_mos(dist_score: 'torch.Tensor') ->torch.Tensor:
    """
    Convert distribution prediction to MOS score.
    For datasets with detailed score labels, such as AVA.

    Args:
        dist_score (torch.Tensor): (*, C), C is the class number.

    Returns:
        torch.Tensor: (*, 1) MOS score.
    """
    num_classes = dist_score.shape[-1]
    mos_score = dist_score * torch.arange(1, num_classes + 1)
    mos_score = mos_score.sum(dim=-1, keepdim=True)
    return mos_score


def _ceil_divide_int(x, y):
    """Returns ceil(x / y) as int"""
    return int(math.ceil(x / y))


def _pad_or_cut_to_max_seq_len(x, max_seq_len):
    """Pads (or cuts) patch tensor `max_seq_len`.
    Args:
        x: input tensor of shape (n_crops, c, num_patches).
        max_seq_len: max sequence length.
    Returns:
        The padded or cropped tensor of shape (n_crops, c, max_seq_len).
    """
    n_crops, c, num_patches = x.shape
    paddings = torch.zeros((n_crops, c, max_seq_len))
    x = torch.cat([x, paddings], dim=-1)
    x = x[:, :, :max_seq_len]
    return x


def extract_image_patches(x, kernel, stride=1, dilation=1):
    """
    Ref: https://stackoverflow.com/a/65886666
    """
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2))
    patches = F.unfold(x, kernel, dilation, stride=stride)
    return patches


def get_hashed_spatial_pos_emb_index(grid_size, count_h, count_w):
    """Get hased spatial pos embedding index for each patch.
    The size H x W is hashed to grid_size x grid_size.
    Args:
      grid_size: grid size G for the hashed-based spatial positional embedding.
      count_h: number of patches in each row for the image.
      count_w: number of patches in each column for the image.
    Returns:
      hashed position of shape (1, HxW). Each value corresponded to the hashed
      position index in [0, grid_size x grid_size).
    """
    pos_emb_grid = torch.arange(grid_size).float()
    pos_emb_hash_w = pos_emb_grid.reshape(1, 1, grid_size)
    pos_emb_hash_w = F.interpolate(pos_emb_hash_w, count_w, mode='nearest')
    pos_emb_hash_w = pos_emb_hash_w.repeat(1, count_h, 1)
    pos_emb_hash_h = pos_emb_grid.reshape(1, 1, grid_size)
    pos_emb_hash_h = F.interpolate(pos_emb_hash_h, count_h, mode='nearest')
    pos_emb_hash_h = pos_emb_hash_h.transpose(1, 2)
    pos_emb_hash_h = pos_emb_hash_h.repeat(1, 1, count_w)
    pos_emb_hash = pos_emb_hash_h * grid_size + pos_emb_hash_w
    pos_emb_hash = pos_emb_hash.reshape(1, -1)
    return pos_emb_hash


def _extract_patches_and_positions_from_image(image, patch_size, patch_stride, hse_grid_size, n_crops, h, w, c, scale_id, max_seq_len):
    """Extracts patches and positional embedding lookup indexes for a given image.
    Args:
      image: the input image of shape [n_crops, c, h, w]
      patch_size: the extracted patch size.
      patch_stride: stride for extracting patches.
      hse_grid_size: grid size for hash-based spatial positional embedding.
      n_crops: number of crops from the input image.
      h: height of the image.
      w: width of the image.
      c: number of channels for the image.
      scale_id: the scale id for the image in the multi-scale representation.
      max_seq_len: maximum sequence length for the number of patches. If
        max_seq_len = 0, no patch is returned. If max_seq_len < 0 then we return
        all the patches.
    Returns:
      A concatenating vector of (patches, HSE, SCE, input mask). The tensor shape
      is (n_crops, num_patches, patch_size * patch_size * c + 3).
    """
    n_crops, c, h, w = image.shape
    p = extract_image_patches(image, patch_size, patch_stride)
    assert p.shape[1] == c * patch_size ** 2
    count_h = _ceil_divide_int(h, patch_stride)
    count_w = _ceil_divide_int(w, patch_stride)
    spatial_p = get_hashed_spatial_pos_emb_index(hse_grid_size, count_h, count_w)
    spatial_p = spatial_p.unsqueeze(1).repeat(n_crops, 1, 1)
    scale_p = torch.ones_like(spatial_p) * scale_id
    mask_p = torch.ones_like(spatial_p)
    out = torch.cat([p, spatial_p, scale_p, mask_p], dim=1)
    if max_seq_len >= 0:
        out = _pad_or_cut_to_max_seq_len(out, max_seq_len)
    return out


def resize_preserve_aspect_ratio(image, h, w, longer_side_length):
    """Aspect-ratio-preserving resizing with tf.image.ResizeMethod.GAUSSIAN.
    Args:
      image: The image tensor (n_crops, c, h, w).
      h: Height of the input image.
      w: Width of the input image.
      longer_side_length: The length of the longer side after resizing.
    Returns:
      A tuple of [Image after resizing, Resized height, Resized width].
    """
    ratio = longer_side_length / max(h, w)
    rh = round(h * ratio)
    rw = round(w * ratio)
    resized = F.interpolate(image, (rh, rw), mode='bicubic', align_corners=False)
    return resized, rh, rw


def get_multiscale_patches(image, patch_size=32, patch_stride=32, hse_grid_size=10, longer_side_lengths=[224, 384], max_seq_len_from_original_res=None):
    """Extracts image patches from multi-scale representation.
    Args:
      image: input image tensor with shape [n_crops, 3, h, w]
      patch_size: patch size.
      patch_stride: patch stride.
      hse_grid_size: Hash-based positional embedding grid size.
      longer_side_lengths: List of longer-side lengths for each scale in the
        multi-scale representation.
      max_seq_len_from_original_res: Maximum number of patches extracted from
        original resolution. <0 means use all the patches from the original
        resolution. None means we don't use original resolution input.
    Returns:
      A concatenating vector of (patches, HSE, SCE, input mask). The tensor shape
      is (n_crops, num_patches, patch_size * patch_size * c + 3).
    """
    longer_side_lengths = sorted(longer_side_lengths)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    n_crops, c, h, w = image.shape
    outputs = []
    for scale_id, longer_size in enumerate(longer_side_lengths):
        resized_image, rh, rw = resize_preserve_aspect_ratio(image, h, w, longer_size)
        max_seq_len = int(np.ceil(longer_size / patch_stride) ** 2)
        out = _extract_patches_and_positions_from_image(resized_image, patch_size, patch_stride, hse_grid_size, n_crops, rh, rw, c, scale_id, max_seq_len)
        outputs.append(out)
    if max_seq_len_from_original_res is not None:
        out = _extract_patches_and_positions_from_image(image, patch_size, patch_stride, hse_grid_size, n_crops, h, w, c, len(longer_side_lengths), max_seq_len_from_original_res)
        outputs.append(out)
    outputs = torch.cat(outputs, dim=-1)
    return outputs.transpose(1, 2)


class MUSIQ(nn.Module):
    """
    MUSIQ model architecture.

    Args:
        - patch_size (int): Size of the patches to extract from the images.
        - num_class (int): Number of classes to predict.
        - hidden_size (int): Size of the hidden layer in the transformer encoder.
        - mlp_dim (int): Size of the feedforward layer in the transformer encoder.
        - attention_dropout_rate (float): Dropout rate for the attention layer in the transformer encoder.
        - dropout_rate (float): Dropout rate for the transformer encoder.
        - num_heads (int): Number of attention heads in the transformer encoder.
        - num_layers (int): Number of layers in the transformer encoder.
        - num_scales (int): Number of scales to use in the transformer encoder.
        - spatial_pos_grid_size (int): Size of the spatial position grid in the transformer encoder.
        - use_scale_emb (bool): Whether to use scale embeddings in the transformer encoder.
        - use_sinusoid_pos_emb (bool): Whether to use sinusoidal position embeddings in the transformer encoder.
        - pretrained (bool or str): Whether to use a pretrained model. If str, specifies the path to the pretrained model.
        - pretrained_model_path (str): Path to the pretrained model.
        - longer_side_lengths (list): List of longer side lengths to use for multiscale evaluation.
        - max_seq_len_from_original_res (int): Maximum sequence length to use for multiscale evaluation.

    Attributes:
        - conv_root (StdConv): Convolutional layer for the root of the network.
        - gn_root (nn.GroupNorm): Group normalization layer for the root of the network.
        - root_pool (nn.Sequential): Max pooling layer for the root of the network.
        - block1 (Bottleneck): First bottleneck block in the network.
        - embedding (nn.Linear): Linear layer for the transformer encoder input.
        - transformer_encoder (TransformerEncoder): Transformer encoder.
        - head (nn.Sequential or nn.Linear): Output layer of the network.

    Methods:
        forward(x, return_mos=True, return_dist=False): Forward pass of the network.

    """

    def __init__(self, patch_size=32, num_class=1, hidden_size=384, mlp_dim=1152, attention_dropout_rate=0.0, dropout_rate=0, num_heads=6, num_layers=14, num_scales=3, spatial_pos_grid_size=10, use_scale_emb=True, use_sinusoid_pos_emb=False, pretrained=True, pretrained_model_path=None, longer_side_lengths=[224, 384], max_seq_len_from_original_res=-1):
        super(MUSIQ, self).__init__()
        resnet_token_dim = 64
        self.patch_size = patch_size
        self.data_preprocess_opts = {'patch_size': patch_size, 'patch_stride': patch_size, 'hse_grid_size': spatial_pos_grid_size, 'longer_side_lengths': longer_side_lengths, 'max_seq_len_from_original_res': max_seq_len_from_original_res}
        if pretrained_model_path is None and pretrained:
            url_key = 'ava' if isinstance(pretrained, bool) else pretrained
            num_class = 10 if url_key == 'ava' else num_class
            pretrained_model_path = default_model_urls[url_key]
        self.conv_root = StdConv(3, resnet_token_dim, 7, 2, bias=False)
        self.gn_root = nn.GroupNorm(32, resnet_token_dim, eps=1e-06)
        self.root_pool = nn.Sequential(nn.ReLU(True), ExactPadding2d(3, 2, mode='same'), nn.MaxPool2d(3, 2))
        token_patch_size = patch_size // 4
        self.block1 = Bottleneck(resnet_token_dim, resnet_token_dim * 4)
        self.embedding = nn.Linear(resnet_token_dim * 4 * token_patch_size ** 2, hidden_size)
        self.transformer_encoder = TransformerEncoder(hidden_size, mlp_dim, attention_dropout_rate, dropout_rate, num_heads, num_layers, num_scales, spatial_pos_grid_size, use_scale_emb, use_sinusoid_pos_emb)
        if num_class > 1:
            self.head = nn.Sequential(nn.Linear(hidden_size, num_class), nn.Softmax(dim=-1))
        else:
            self.head = nn.Linear(hidden_size, num_class)
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True)

    def forward(self, x, return_mos=True, return_dist=False):
        """
        Forward pass of the MUSIQ network.

        Args:
            x (torch.Tensor): Input tensor.
            return_mos (bool): Whether to return the mean opinion score (MOS).
            return_dist (bool): Whether to return the predicted distribution.

        Returns:
            torch.Tensor or tuple: If only one of return_mos and return_dist is True, returns a tensor. If both are True, returns a tuple of tensors.

        """
        if not self.training:
            x = (x - 0.5) * 2
            x = get_multiscale_patches(x, **self.data_preprocess_opts)
        assert len(x.shape) in [3, 4]
        if len(x.shape) == 4:
            b, num_crops, seq_len, dim = x.shape
            x = x.reshape(b * num_crops, seq_len, dim)
        else:
            b, seq_len, dim = x.shape
            num_crops = 1
        inputs_spatial_positions = x[:, :, -3]
        inputs_scale_positions = x[:, :, -2]
        inputs_masks = x[:, :, -1].bool()
        x = x[:, :, :-3]
        x = x.reshape(-1, 3, self.patch_size, self.patch_size)
        x = self.conv_root(x)
        x = self.gn_root(x)
        x = self.root_pool(x)
        x = self.block1(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, seq_len, -1)
        x = self.embedding(x)
        x = self.transformer_encoder(x, inputs_spatial_positions, inputs_scale_positions, inputs_masks)
        q = self.head(x[:, 0])
        q = q.reshape(b, num_crops, -1)
        q = q.mean(dim=1)
        mos = dist_to_mos(q)
        return_list = []
        if return_mos:
            return_list.append(mos)
        if return_dist:
            return_list.append(q)
        if len(return_list) > 1:
            return return_list
        else:
            return return_list[0]


class NIMA(nn.Module):
    """Neural IMage Assessment model.

    Modification:
        - for simplicity, we use global average pool for all models
        - we remove the dropout, because parameters with avg pool is much less.

    Args:
        base_model_name: pretrained model to extract features, can be any models supported by timm.
                         Models used in the paper: vgg16, inception_resnet_v2, mobilenetv2_100

        default input shape:
            - vgg and mobilenet: (N, 3, 224, 224)
            - inception: (N, 3, 299, 299)
    """

    def __init__(self, base_model_name='vgg16', train_dataset='ava', num_classes=10, dropout_rate=0.0, pretrained=True, pretrained_model_path=None):
        super(NIMA, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=True, features_only=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        in_ch = self.base_model.feature_info.channels()[-1]
        self.num_classes = num_classes
        self.classifier = [nn.Flatten(), nn.Dropout(p=dropout_rate), nn.Linear(in_features=in_ch, out_features=num_classes)]
        if num_classes > 1:
            self.classifier.append(nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(*self.classifier)
        default_mean = self.base_model.pretrained_cfg['mean']
        default_std = self.base_model.pretrained_cfg['std']
        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)
        if pretrained and pretrained_model_path is None:
            url_key = f'{base_model_name}-{train_dataset}'
            load_pretrained_network(self, default_model_urls[url_key], True, weight_keys='params')
        elif pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, weight_keys='params')

    def preprocess(self, x):
        if not self.training:
            x = T.functional.resize(x, self.base_model.default_cfg['input_size'][-1])
            x = T.functional.center_crop(x, self.base_model.default_cfg['input_size'][-1])
        x = (x - self.default_mean) / self.default_std
        return x

    def forward(self, x, return_mos=True, return_dist=False):
        """Computation image quality using NIMA.
        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            return_mos: Whether to return mos_score.
            retuen_dist: Whether to return dist_score.

        """
        x = self.preprocess(x)
        x = self.base_model(x)[-1]
        x = self.global_pool(x)
        dist = self.classifier(x)
        mos = dist_to_mos(dist)
        return_list = []
        if return_mos:
            return_list.append(mos)
        if return_dist:
            return_list.append(dist)
        if len(return_list) > 1:
            return return_list
        else:
            return return_list[0]


def diff_round(x: 'torch.Tensor') ->torch.Tensor:
    """Differentiable round."""
    return x - x.detach() + x.round()


def blockproc(x, kernel, fun, border_size=None, pad_partial=False, pad_method='zero', **func_args):
    """blockproc function like matlab

    Difference:
        - Partial blocks is discarded (if exist) for fast GPU process.

    Args:
        x (tensor): shape (b, c, h, w)
        kernel (int or tuple): block size
        func (function): function to process each block
        border_size (int or tuple): border pixels to each block
        pad_partial: pad partial blocks to make them full-sized, default False
        pad_method: [zero, replicate, symmetric] how to pad partial block when pad_partial is set True

    Return:
        results (tensor): concatenated results of each block
    """
    assert len(x.shape) == 4, f'Shape of input has to be (b, c, h, w) but got {x.shape}'
    kernel = to_2tuple(kernel)
    if pad_partial:
        b, c, h, w = x.shape
        stride = kernel
        h2 = math.ceil(h / stride[0])
        w2 = math.ceil(w / stride[1])
        pad_row = (h2 - 1) * stride[0] + kernel[0] - h
        pad_col = (w2 - 1) * stride[1] + kernel[1] - w
        padding = 0, pad_col, 0, pad_row
        if pad_method == 'zero':
            x = F.pad(x, padding, mode='constant')
        elif pad_method == 'symmetric':
            x = symm_pad(x, padding)
        else:
            x = F.pad(x, padding, mode=pad_method)
    if border_size is not None:
        raise NotImplementedError('Blockproc with border is not implemented yet')
    else:
        b, c, h, w = x.shape
        block_size_h, block_size_w = kernel
        num_block_h = math.floor(h / block_size_h)
        num_block_w = math.floor(w / block_size_w)
        blocks = F.unfold(x, kernel, stride=kernel)
        blocks = blocks.reshape(b, c, *kernel, num_block_h, num_block_w)
        blocks = blocks.permute(5, 4, 0, 1, 2, 3).reshape(num_block_h * num_block_w * b, c, *kernel)
        results = fun(blocks, func_args)
        results = results.reshape(num_block_h * num_block_w, b, *results.shape[1:]).transpose(0, 1)
        return results


def fitweibull(x, iters=50, eps=0.01):
    """Simulate wblfit function in matlab.

    ref: https://github.com/mlosch/python-weibullfit/blob/master/weibull/backend_pytorch.py

    Fits a 2-parameter Weibull distribution to the given data using maximum-likelihood estimation.
    :param x (tensor): (B, N), batch of samples from an (unknown) distribution. Each value must satisfy x > 0.
    :param iters: Maximum number of iterations
    :param eps: Stopping criterion. Fit is stopped ff the change within two iterations is smaller than eps.
    :param use_cuda: Use gpu
    :return: Tuple (Shape, Scale) which can be (NaN, NaN) if a fit is impossible.
        Impossible fits may be due to 0-values in x.
    """
    ln_x = torch.log(x)
    k = 1.2 / torch.std(ln_x, dim=1, keepdim=True)
    k_t_1 = k
    for t in range(iters):
        x_k = x ** k.repeat(1, x.shape[1])
        x_k_ln_x = x_k * ln_x
        ff = torch.sum(x_k_ln_x, dim=-1, keepdim=True)
        fg = torch.sum(x_k, dim=-1, keepdim=True)
        f1 = torch.mean(ln_x, dim=-1, keepdim=True)
        f = ff / fg - f1 - 1.0 / k
        ff_prime = torch.sum(x_k_ln_x * ln_x, dim=-1, keepdim=True)
        fg_prime = ff
        f_prime = ff_prime / fg - ff / fg * fg_prime / fg + 1.0 / (k * k)
        k = k - f / f_prime
        error = torch.abs(k - k_t_1).max().item()
        if error < eps:
            break
        k_t_1 = k
    lam = torch.mean(x ** k.repeat(1, x.shape[1]), dim=-1, keepdim=True) ** (1.0 / k)
    return torch.cat((k, lam), dim=1)


def compute_feature(block: 'torch.Tensor', ilniqe: 'bool'=False) ->torch.Tensor:
    """Compute features.
    Args:
        block (Tensor): Image block in shape (b, c, h, w).
    Returns:
        list: Features with length of 18.
    """
    bsz = block.shape[0]
    aggd_block = block[:, [0]]
    alpha, beta_l, beta_r = estimate_aggd_param(aggd_block)
    feat = [alpha, (beta_l + beta_r) / 2]
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = torch.roll(aggd_block, shifts[i], dims=(2, 3))
        alpha, beta_l, beta_r = estimate_aggd_param(aggd_block * shifted_block)
        mean = (beta_r - beta_l) * (torch.lgamma(2 / alpha) - torch.lgamma(1 / alpha)).exp()
        feat.extend((alpha, mean, beta_l, beta_r))
    feat = [x.reshape(bsz, 1) for x in feat]
    if ilniqe:
        tmp_block = block[:, 1:4]
        channels = 4 - 1
        shape_scale = fitweibull(tmp_block.reshape(bsz * channels, -1))
        scale_shape = shape_scale[:, [1, 0]].reshape(bsz, -1)
        feat.append(scale_shape)
        mu = torch.mean(block[:, 4:7], dim=(2, 3))
        sigmaSquare = torch.var(block[:, 4:7], dim=(2, 3))
        mu_sigma = torch.stack((mu, sigmaSquare), dim=-1).reshape(bsz, -1)
        feat.append(mu_sigma)
        channels = 85 - 7
        tmp_block = block[:, 7:85].reshape(bsz * channels, 1, *block.shape[2:])
        alpha_data, beta_l_data, beta_r_data = estimate_aggd_param(tmp_block)
        alpha_data = alpha_data.reshape(bsz, channels)
        beta_l_data = beta_l_data.reshape(bsz, channels)
        beta_r_data = beta_r_data.reshape(bsz, channels)
        alpha_beta = torch.stack([alpha_data, (beta_l_data + beta_r_data) / 2], dim=-1).reshape(bsz, -1)
        feat.append(alpha_beta)
        tmp_block = block[:, 85:109]
        channels = 109 - 85
        shape_scale = fitweibull(tmp_block.reshape(bsz * channels, -1))
        scale_shape = shape_scale[:, [1, 0]].reshape(bsz, -1)
        feat.append(scale_shape)
    feat = torch.cat(feat, dim=-1)
    return feat


def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)
    Ref: https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    correction = int(not bias) if tensor.shape[-1] > 1 else 0
    return torch.cov(tensor, correction=correction)


def nancov(x):
    """Calculate nancov for batched tensor, rows that contains nan value 
    will be removed.

    Args:
        x (tensor): (B, row_num, feat_dim)  

    Return:
        cov (tensor): (B, feat_dim, feat_dim)
    """
    assert len(x.shape) == 3, f'Shape of input should be (batch_size, row_num, feat_dim), but got {x.shape}'
    b, rownum, feat_dim = x.shape
    nan_mask = torch.isnan(x).any(dim=2, keepdim=True)
    cov_x = []
    for i in range(b):
        x_no_nan = x[i].masked_select(~nan_mask[i]).reshape(-1, feat_dim)
        cov_x.append(cov(x_no_nan, rowvar=False))
    return torch.stack(cov_x)


def nanmean(v, *args, inplace=False, **kwargs):
    """nanmean same as matlab function: calculate mean values by removing all nan.
    """
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def niqe(img: 'torch.Tensor', mu_pris_param: 'torch.Tensor', cov_pris_param: 'torch.Tensor', block_size_h: 'int'=96, block_size_w: 'int'=96) ->torch.Tensor:
    """Calculate NIQE (Natural Image Quality Evaluator) metric.
    Args:
        img (Tensor): Input image.
        mu_pris_param (Tensor): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (Tensor): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        gaussian_window (Tensor): A 7x7 Gaussian window used for smoothing the image.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
    """
    assert img.ndim == 4, 'Input image must be a gray or Y (of YCbCr) image with shape (b, c, h, w).'
    b, c, h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[..., 0:num_block_h * block_size_h, 0:num_block_w * block_size_w]
    distparam = []
    for scale in (1, 2):
        img_normalized = normalize_img_with_gauss(img, padding='replicate')
        distparam.append(blockproc(img_normalized, [block_size_h // scale, block_size_w // scale], fun=compute_feature))
        if scale == 1:
            img = imresize(img / 255.0, scale=0.5, antialiasing=True)
            img = img * 255.0
    distparam = torch.cat(distparam, -1)
    mu_distparam = nanmean(distparam, dim=1)
    cov_distparam = nancov(distparam)
    invcov_param = torch.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    diff = (mu_pris_param - mu_distparam).unsqueeze(1)
    quality = torch.bmm(torch.bmm(diff, invcov_param), diff.transpose(1, 2)).squeeze()
    quality = torch.sqrt(quality)
    return quality


def calculate_niqe(img: 'torch.Tensor', crop_border: 'int'=0, test_y_channel: 'bool'=True, color_space: 'str'='yiq', mu_pris_param: 'torch.Tensor'=None, cov_pris_param: 'torch.Tensor'=None, **kwargs) ->torch.Tensor:
    """Calculate NIQE (Natural Image Quality Evaluator) metric.
    Args:
        img (Tensor): Input image whose quality needs to be computed.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        test_y_channel (Bool): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
        pretrained_model_path (str): The pretrained model path.
    Returns:
        Tensor: NIQE result.
    """
    if img.shape[1] == 3:
        img = to_y_channel(img, 255, color_space)
    elif img.shape[1] == 1:
        img = img * 255
    img = diff_round(img)
    img = img
    mu_pris_param = mu_pris_param.repeat(img.size(0), 1)
    cov_pris_param = cov_pris_param.repeat(img.size(0), 1, 1)
    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
    niqe_result = niqe(img, mu_pris_param, cov_pris_param)
    return niqe_result


class NIQE(torch.nn.Module):
    """Args:
        - channels (int): Number of processed channel.
        - test_y_channel (bool): whether to use y channel on ycbcr.
        - crop_border (int): Cropped pixels in each edge of an image. These
        pixels are not involved in the metric calculation.
        - pretrained_model_path (str): The pretrained model path.
    References:
        Mittal, Anish, Rajiv Soundararajan, and Alan C. Bovik.
        "Making a “completely blind” image quality analyzer."
        IEEE Signal Processing Letters (SPL) 20.3 (2012): 209-212.
    """

    def __init__(self, channels: 'int'=1, test_y_channel: 'bool'=True, color_space: 'str'='yiq', crop_border: 'int'=0, version: 'str'='original', pretrained_model_path: 'str'=None) ->None:
        super(NIQE, self).__init__()
        self.channels = channels
        self.test_y_channel = test_y_channel
        self.color_space = color_space
        self.crop_border = crop_border
        if pretrained_model_path is not None:
            pretrained_model_path = pretrained_model_path
        elif version == 'original':
            pretrained_model_path = load_file_from_url(default_model_urls['url'])
        elif version == 'matlab':
            pretrained_model_path = load_file_from_url(default_model_urls['niqe_matlab'])
        params = scipy.io.loadmat(pretrained_model_path)
        mu_pris_param = np.ravel(params['mu_prisparam'])
        cov_pris_param = params['cov_prisparam']
        self.mu_pris_param = torch.from_numpy(mu_pris_param)
        self.cov_pris_param = torch.from_numpy(cov_pris_param)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Computation of NIQE metric.
        Input:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
        Output:
            score (tensor): results of ilniqe metric, should be a positive real number. Shape :math:`(N, 1)`.
        """
        score = calculate_niqe(x, self.crop_border, self.test_y_channel, self.color_space, self.mu_pris_param, self.cov_pris_param)
        return score


def conv2d(input, weight, bias=None, stride=1, padding='same', dilation=1, groups=1):
    """Matlab like conv2, weights needs to be flipped.
    Args:
        input (tensor): (b, c, h, w)
        weight (tensor): (out_ch, in_ch, kh, kw), conv weight
        bias (bool or None): bias
        stride (int or tuple): conv stride
        padding (str): padding mode
        dilation (int): conv dilation
    """
    kernel_size = weight.shape[2:]
    pad_func = ExactPadding2d(kernel_size, stride, dilation, mode=padding)
    weight = torch.flip(weight, dims=(-1, -2))
    return F.conv2d(pad_func(input), weight, bias, stride, dilation=dilation, groups=groups)


def gauDerivative(sigma, in_ch=1, out_ch=1, device=None):
    halfLength = math.ceil(3 * sigma)
    x, y = np.meshgrid(np.linspace(-halfLength, halfLength, 2 * halfLength + 1), np.linspace(-halfLength, halfLength, 2 * halfLength + 1))
    gauDerX = x * np.exp(-(x ** 2 + y ** 2) / 2 / sigma / sigma)
    gauDerY = y * np.exp(-(x ** 2 + y ** 2) / 2 / sigma / sigma)
    dx = torch.from_numpy(gauDerX)
    dy = torch.from_numpy(gauDerY)
    dx = dx.repeat(out_ch, in_ch, 1, 1)
    dy = dy.repeat(out_ch, in_ch, 1, 1)
    return dx, dy


def ilniqe(img: 'torch.Tensor', mu_pris_param: 'torch.Tensor', cov_pris_param: 'torch.Tensor', principleVectors: 'torch.Tensor', meanOfSampleData: 'torch.Tensor', resize: 'bool'=True, block_size_h: 'int'=84, block_size_w: 'int'=84) ->torch.Tensor:
    """Calculate IL-NIQE (Integrated Local Natural Image Quality Evaluator) metric.
    Args:
        img (Tensor): Input image.
        mu_pris_param (Tensor): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (Tensor): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        principleVectors (Tensor): Features from official .mat file.
        meanOfSampleData (Tensor): Features from official .mat file.
        resize (Bloolean): resize image. Default: True.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 84 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 84 (the official recommended value).
    """
    assert img.ndim == 4, 'Input image must be a gray or Y (of YCbCr) image with shape (b, c, h, w).'
    sigmaForGauDerivative = 1.66
    KforLog = 1e-05
    normalizedWidth = 524
    minWaveLength = 2.4
    sigmaOnf = 0.55
    mult = 1.31
    dThetaOnSigma = 1.1
    scaleFactorForLoG = 0.87
    scaleFactorForGaussianDer = 0.28
    sigmaForDownsample = 0.9
    EPS = 1e-08
    scales = 3
    orientations = 4
    infConst = 10000
    if resize:
        img = imresize(img, sizes=(normalizedWidth, normalizedWidth))
        img = img.clamp(0.0, 255.0)
    b, c, h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[..., 0:num_block_h * block_size_h, 0:num_block_w * block_size_w]
    ospace_weight = torch.tensor([[0.3, 0.04, -0.35], [0.34, -0.6, 0.17], [0.06, 0.63, 0.27]])
    O_img = img.permute(0, 2, 3, 1) @ ospace_weight.T
    O_img = O_img.permute(0, 3, 1, 2)
    distparam = []
    for scale in (1, 2):
        struct_dis = normalize_img_with_gauss(O_img[:, [2]], kernel_size=5, sigma=5.0 / 6, padding='replicate')
        dx, dy = gauDerivative(sigmaForGauDerivative / scale ** scaleFactorForGaussianDer, device=img)
        Ix = conv2d(O_img, dx.repeat(3, 1, 1, 1), groups=3)
        Iy = conv2d(O_img, dy.repeat(3, 1, 1, 1), groups=3)
        GM = torch.sqrt(Ix ** 2 + Iy ** 2 + EPS)
        Ixy = torch.stack((Ix, Iy), dim=2).reshape(Ix.shape[0], Ix.shape[1] * 2, *Ix.shape[2:])
        logRGB = torch.log(img + KforLog)
        logRGBMS = logRGB - logRGB.mean(dim=(2, 3), keepdim=True)
        Intensity = logRGBMS.sum(dim=1, keepdim=True) / np.sqrt(3)
        BY = (logRGBMS[:, [0]] + logRGBMS[:, [1]] - 2 * logRGBMS[:, [2]]) / np.sqrt(6)
        RG = (logRGBMS[:, [0]] - logRGBMS[:, [1]]) / np.sqrt(2)
        compositeMat = torch.cat([struct_dis, GM, Intensity, BY, RG, Ixy], dim=1)
        O3 = O_img[:, [2]]
        LGFilters = _construct_filters(O3, scales=scales, orientations=orientations, min_length=minWaveLength / scale ** scaleFactorForLoG, sigma_f=sigmaOnf, mult=mult, delta_theta=dThetaOnSigma, use_lowpass_filter=False)
        b, _, h, w = LGFilters.shape
        LGFilters = LGFilters.reshape(b, orientations, scales, h, w).transpose(1, 2).reshape(b, -1, h, w)
        LGFilters = LGFilters.transpose(-1, -2)
        fftIm = torch.fft.fft2(O3)
        logResponse = []
        partialDer = []
        GM = []
        for index in range(LGFilters.shape[1]):
            filter = LGFilters[:, [index]]
            response = torch.fft.ifft2(filter * fftIm)
            realRes = torch.real(response)
            imagRes = torch.imag(response)
            partialXReal = conv2d(realRes, dx)
            partialYReal = conv2d(realRes, dy)
            realGM = torch.sqrt(partialXReal ** 2 + partialYReal ** 2 + EPS)
            partialXImag = conv2d(imagRes, dx)
            partialYImag = conv2d(imagRes, dy)
            imagGM = torch.sqrt(partialXImag ** 2 + partialYImag ** 2 + EPS)
            logResponse.append(realRes)
            logResponse.append(imagRes)
            partialDer.append(partialXReal)
            partialDer.append(partialYReal)
            partialDer.append(partialXImag)
            partialDer.append(partialYImag)
            GM.append(realGM)
            GM.append(imagGM)
        logResponse = torch.cat(logResponse, dim=1)
        partialDer = torch.cat(partialDer, dim=1)
        GM = torch.cat(GM, dim=1)
        compositeMat = torch.cat((compositeMat, logResponse, partialDer, GM), dim=1)
        distparam.append(blockproc(compositeMat, [block_size_h // scale, block_size_w // scale], fun=compute_feature, ilniqe=True))
        gauForDS = fspecial(math.ceil(6 * sigmaForDownsample), sigmaForDownsample)
        filterResult = imfilter(O_img, gauForDS.repeat(3, 1, 1, 1), padding='replicate', groups=3)
        O_img = filterResult[..., ::2, ::2]
        filterResult = imfilter(img, gauForDS.repeat(3, 1, 1, 1), padding='replicate', groups=3)
        img = filterResult[..., ::2, ::2]
    distparam = torch.cat(distparam, dim=-1)
    distparam[distparam > infConst] = infConst
    coefficientsViaPCA = torch.bmm(principleVectors.transpose(1, 2), (distparam - meanOfSampleData.unsqueeze(1)).transpose(1, 2))
    final_features = coefficientsViaPCA.transpose(1, 2)
    b, blk_num, feat_num = final_features.shape
    cov_distparam = nancov(final_features)
    mu_final_features = nanmean(final_features, dim=1, keepdim=True)
    final_features_withmu = torch.where(torch.isnan(final_features), mu_final_features, final_features)
    invcov_param = torch.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    diff = final_features_withmu - mu_pris_param.unsqueeze(1)
    quality = (torch.bmm(diff, invcov_param) * diff).sum(dim=-1)
    quality = torch.sqrt(quality).mean(dim=1)
    return quality


def calculate_ilniqe(img: 'torch.Tensor', crop_border: 'int'=0, mu_pris_param: 'torch.Tensor'=None, cov_pris_param: 'torch.Tensor'=None, principleVectors: 'torch.Tensor'=None, meanOfSampleData: 'torch.Tensor'=None, **kwargs) ->torch.Tensor:
    """Calculate IL-NIQE metric.
    Args:
        img (Tensor): Input image whose quality needs to be computed.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        pretrained_model_path (str): The pretrained model path.
    Returns:
        Tensor: IL-NIQE result.
    """
    img = img * 255.0
    img = diff_round(img)
    img = img
    mu_pris_param = mu_pris_param.repeat(img.size(0), 1)
    cov_pris_param = cov_pris_param.repeat(img.size(0), 1, 1)
    meanOfSampleData = meanOfSampleData.repeat(img.size(0), 1)
    principleVectors = principleVectors.repeat(img.size(0), 1, 1)
    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
    ilniqe_result = ilniqe(img, mu_pris_param, cov_pris_param, principleVectors, meanOfSampleData)
    return ilniqe_result


class ILNIQE(torch.nn.Module):
    """Args:
        - channels (int): Number of processed channel.
        - test_y_channel (bool): whether to use y channel on ycbcr.
        - crop_border (int): Cropped pixels in each edge of an image. These
        pixels are not involved in the metric calculation.
        - pretrained_model_path (str): The pretrained model path.
    References:
        Zhang, Lin, Lei Zhang, and Alan C. Bovik. "A feature-enriched
        completely blind image quality evaluator." IEEE Transactions
        on Image Processing 24.8 (2015): 2579-2591.
    """

    def __init__(self, channels: 'int'=3, crop_border: 'int'=0, pretrained_model_path: 'str'=None) ->None:
        super(ILNIQE, self).__init__()
        self.channels = channels
        self.crop_border = crop_border
        if pretrained_model_path is not None:
            self.pretrained_model_path = pretrained_model_path
        else:
            self.pretrained_model_path = load_file_from_url(default_model_urls['ilniqe'])
        params = scipy.io.loadmat(self.pretrained_model_path)
        mu_pris_param = np.ravel(params['templateModel'][0][0])
        cov_pris_param = params['templateModel'][0][1]
        meanOfSampleData = np.ravel(params['templateModel'][0][2])
        principleVectors = params['templateModel'][0][3]
        self.mu_pris_param = torch.from_numpy(mu_pris_param)
        self.cov_pris_param = torch.from_numpy(cov_pris_param)
        self.meanOfSampleData = torch.from_numpy(meanOfSampleData)
        self.principleVectors = torch.from_numpy(principleVectors)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Computation of NIQE metric.
        Input:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
        Output:
            score (tensor): results of ilniqe metric, should be a positive real number. Shape :math:`(N, 1)`.
        """
        assert x.shape[1] == 3, 'ILNIQE only support input image with 3 channels'
        score = calculate_ilniqe(x, self.crop_border, self.mu_pris_param, self.cov_pris_param, self.principleVectors, self.meanOfSampleData)
        return score


LAPLACIAN_FILTER = np.array([[0.0025, 0.0125, 0.02, 0.0125, 0.0025], [0.0125, 0.0625, 0.1, 0.0625, 0.0125], [0.02, 0.1, 0.16, 0.1, 0.02], [0.0125, 0.0625, 0.1, 0.0625, 0.0125], [0.0025, 0.0125, 0.02, 0.0125, 0.0025]], dtype=np.float32)


class NLPD(nn.Module):
    """Normalised lapalcian pyramid distance
    Args:
        - channels: Number of channel expected to calculate.
        - test_y_channel: Boolean, whether to use y channel on ycbcr which mimics official matlab code.

    References:
        Laparra, Valero, Johannes Ballé, Alexander Berardino, and Eero P. Simoncelli.
        "Perceptual image quality assessment using a normalized Laplacian pyramid."
        Electronic Imaging 2016, no. 16 (2016): 1-6.

    """

    def __init__(self, channels=1, test_y_channel=True, k=6, filt=None):
        super(NLPD, self).__init__()
        if filt is None:
            filt = np.reshape(np.tile(LAPLACIAN_FILTER, (channels, 1, 1)), (channels, 1, 5, 5))
        self.k = k
        self.channels = channels
        self.test_y_channel = test_y_channel
        self.filt = nn.Parameter(torch.Tensor(filt), requires_grad=False)
        self.dn_filts, self.sigmas = self.DN_filters()
        self.pad_zero_one = nn.ZeroPad2d(1)
        self.pad_zero_two = nn.ZeroPad2d(2)
        self.pad_sym = ExactPadding2d(5, mode='symmetric')
        self.rep_one = nn.ReplicationPad2d(1)
        self.ps = nn.PixelShuffle(2)

    def DN_filters(self):
        """Define parameters for the divisive normalization
        """
        sigmas = [0.0248, 0.0185, 0.0179, 0.0191, 0.022, 0.2782]
        dn_filts = []
        dn_filts.append(torch.Tensor(np.reshape([[0, 0.1011, 0], [0.1493, 0, 0.146], [0, 0.1015, 0.0]] * self.channels, (self.channels, 1, 3, 3)).astype(np.float32)))
        dn_filts.append(torch.Tensor(np.reshape([[0, 0.0757, 0], [0.1986, 0, 0.1846], [0, 0.0837, 0]] * self.channels, (self.channels, 1, 3, 3)).astype(np.float32)))
        dn_filts.append(torch.Tensor(np.reshape([[0, 0.0477, 0], [0.2138, 0, 0.2243], [0, 0.0467, 0]] * self.channels, (self.channels, 1, 3, 3)).astype(np.float32)))
        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0], [0.2503, 0, 0.2616], [0, 0, 0]] * self.channels, (self.channels, 1, 3, 3)).astype(np.float32)))
        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0], [0.2598, 0, 0.2552], [0, 0, 0]] * self.channels, (self.channels, 1, 3, 3)).astype(np.float32)))
        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0], [0.2215, 0, 0.0717], [0, 0, 0]] * self.channels, (self.channels, 1, 3, 3)).astype(np.float32)))
        dn_filts = nn.ParameterList([nn.Parameter(x, requires_grad=False) for x in dn_filts])
        sigmas = nn.ParameterList([nn.Parameter(torch.Tensor(np.array(x)), requires_grad=False) for x in sigmas])
        return dn_filts, sigmas

    def pyramid(self, im):
        """Compute Laplacian Pyramid
        Args:
            im: An input tensor. Shape :math:`(N, C, H, W)`.
        """
        out = []
        J = im
        pyr = []
        for i in range(0, self.k - 1):
            I = F.conv2d(self.pad_sym(J), self.filt, stride=2, padding=0, groups=self.channels)
            odd_h, odd_w = 2 * I.size(2) - J.size(2), 2 * I.size(3) - J.size(3)
            I_pad = self.rep_one(I)
            I_rep1, I_rep2, I_rep3 = torch.zeros_like(I_pad), torch.zeros_like(I_pad), torch.zeros_like(I_pad)
            R = torch.cat([I_pad * 4, I_rep1, I_rep2, I_rep3], dim=1)
            I_up = self.ps(R)
            I_up_conv = F.conv2d(self.pad_zero_two(I_up), self.filt, stride=1, padding=0, groups=self.channels)
            I_up_conv = I_up_conv[:, :, 2:I_up.shape[2] - 2 - odd_h, 2:I_up.shape[3] - 2 - odd_w]
            out = J - I_up_conv
            out_conv = F.conv2d(self.pad_zero_one(torch.abs(out)), tf.rotate(self.dn_filts[i], 180), stride=1, groups=self.channels)
            out_norm = out / (self.sigmas[i] + out_conv)
            pyr.append(out_norm)
            J = I
        out_conv = F.conv2d(self.pad_zero_one(torch.abs(J)), tf.rotate(self.dn_filts[-1], 180), stride=1, groups=self.channels)
        out_norm = J / (self.sigmas[-1] + out_conv)
        pyr.append(out_norm)
        return pyr

    def nlpd(self, x1, x2):
        """Compute Normalised lapalcian pyramid distance for a batch of images.
        Args:
            x1: An input tensor. Shape :math:`(N, C, H, W)`.
            x2: A target tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Index of similarity between two images. Usually in [0, 1] interval.
        """
        assert self.test_y_channel and self.channels == 1 or not self.test_y_channel and self.channels == 3, 'Number of channel and convert to YCBCR should be match'
        if self.test_y_channel and self.channels == 1:
            x1 = to_y_channel(x1)
            x2 = to_y_channel(x2)
        y1 = self.pyramid(x1)
        y2 = self.pyramid(x2)
        total = []
        for z1, z2 in zip(y1, y2):
            diff = (z1 - z2) ** 2
            sqrt = torch.sqrt(torch.mean(diff, (1, 2, 3)))
            total.append(sqrt)
        score = torch.stack(total, dim=1).mean(1)
        return score

    def forward(self, X, Y):
        """Computation of NLPD metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
            Y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of NLPD metric in [0, 1] range.

        """
        assert X.shape == Y.shape, f'Input {X.shape} and reference images should have the same shape'
        score = self.nlpd(X, Y)
        return score


def get_var_gen_gauss(x, eps=1e-07):
    """Get mean and variance of input local patch.
    """
    std = x.abs().std(dim=-1, unbiased=True)
    mean = x.abs().mean(dim=-1)
    rho = std / (mean + eps)
    return rho


def coeff_var_dct(dct_img_block: 'torch.Tensor'):
    """Gaussian var, mean features
    """
    b, _, _, h, w = dct_img_block.shape
    dct_flatten = dct_img_block.reshape(b, -1, h * w)[:, :, 1:]
    rho = get_var_gen_gauss(dct_flatten)
    rho = torch.sort(rho, dim=-1)[0]
    return rho


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    Args:
        x: the input signal
        norm: the normalization, None or 'ortho'
    Return:
        the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=-1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=-1))
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V


def dct2d(x, norm='ortho'):
    """
    2-dimensional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def extract_2d_patches(x, kernel, stride=1, dilation=1, padding='same'):
    """
    Extracts 2D patches from a 4D tensor.

    Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        - kernel (int): Size of the kernel to be used for patch extraction.
        - stride (int): Stride of the kernel. Default is 1.
        - dilation (int): Dilation rate of the kernel. Default is 1.
        - padding (str): Type of padding to be applied. Can be "same" or "none". Default is "same".

    Returns:
        torch.Tensor: Extracted patches tensor of shape (batch_size, num_patches, channels, kernel, kernel).
    """
    b, c, h, w = x.shape
    if padding != 'none':
        x = exact_padding_2d(x, kernel, stride, dilation, mode=padding)
    patches = F.unfold(x, kernel, dilation, stride=stride)
    b, _, pnum = patches.shape
    patches = patches.transpose(1, 2).reshape(b, pnum, c, kernel, kernel)
    return patches


def gamma_gen_gauss(x: 'Tensor', block_seg=10000.0):
    """General gaussian distribution estimation.

    Args:
        block_seg: maximum number of blocks in parallel to avoid OOM
    """
    pshape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    eps = 1e-07
    gamma = torch.arange(0.03, 10 + 0.001, 0.001)
    r_table = (torch.lgamma(1.0 / gamma) + torch.lgamma(3.0 / gamma) - 2 * torch.lgamma(2.0 / gamma)).exp()
    r_table = r_table.unsqueeze(0)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=True)
    mean_abs = (x - mean).abs().mean(dim=-1, keepdim=True) ** 2
    rho = var / (mean_abs + eps)
    if rho.shape[0] > block_seg:
        rho_seg = rho.chunk(int(rho.shape[0] // block_seg))
        indexes = []
        for r in rho_seg:
            tmp_idx = (r - r_table).abs().argmin(dim=-1)
            indexes.append(tmp_idx)
        indexes = torch.cat(indexes)
    else:
        indexes = (rho - r_table).abs().argmin(dim=-1)
    solution = gamma[indexes].reshape(*pshape)
    return solution


def gamma_dct(dct_img_block: 'torch.Tensor'):
    """Generalized gaussian distribution features
    """
    b, _, _, h, w = dct_img_block.shape
    dct_flatten = dct_img_block.reshape(b, -1, h * w)[:, :, 1:]
    g = gamma_gen_gauss(dct_flatten)
    g = torch.sort(g, dim=-1)[0]
    return g


def oriented_dct_rho(dct_img_block: 'torch.Tensor'):
    """Oriented frequency features
    """
    eps = 1e-08
    feat1 = torch.cat([dct_img_block[..., 0, 1:], dct_img_block[..., 1, 2:], dct_img_block[..., 2, 4:], dct_img_block[..., 3, 5:]], dim=-1).squeeze(-2)
    g1 = get_var_gen_gauss(feat1, eps)
    feat2 = torch.cat([dct_img_block[..., 1, [1]], dct_img_block[..., 2, 2:4], dct_img_block[..., 3, 2:5], dct_img_block[..., 4, 3:], dct_img_block[..., 5, 4:], dct_img_block[..., 6, 4:]], dim=-1).squeeze(-2)
    g2 = get_var_gen_gauss(feat2, eps)
    feat3 = torch.cat([dct_img_block[..., 1:, 0], dct_img_block[..., 2:, 1], dct_img_block[..., 4:, 2], dct_img_block[..., 5:, 3]], dim=-1).squeeze(-2)
    g3 = get_var_gen_gauss(feat3, eps)
    rho = torch.stack([g1, g2, g3], dim=-1).var(dim=-1)
    rho = torch.sort(rho, dim=-1)[0]
    return rho


def block_dct(img: 'Tensor'):
    """Get local frequency features
    """
    img_blocks = extract_2d_patches(img, 3 + 2 * 2, 3)
    dct_img_blocks = dct2d(img_blocks)
    features = []
    gamma_L1 = gamma_dct(dct_img_blocks)
    p10_gamma_L1 = gamma_L1[:, :math.ceil(0.1 * gamma_L1.shape[-1]) + 1].mean(dim=-1)
    p100_gamma_L1 = gamma_L1.mean(dim=-1)
    features += [p10_gamma_L1, p100_gamma_L1]
    coeff_var_L1 = coeff_var_dct(dct_img_blocks)
    p10_last_cv_L1 = coeff_var_L1[:, math.floor(0.9 * coeff_var_L1.shape[-1]):].mean(dim=-1)
    p100_cv_L1 = coeff_var_L1.mean(dim=-1)
    features += [p10_last_cv_L1, p100_cv_L1]
    ori_dct_feat = oriented_dct_rho(dct_img_blocks)
    p10_last_orientation_L1 = ori_dct_feat[:, math.floor(0.9 * ori_dct_feat.shape[-1]):].mean(dim=-1)
    p100_orientation_L1 = ori_dct_feat.mean(dim=-1)
    features += [p10_last_orientation_L1, p100_orientation_L1]
    dct_feat = torch.stack(features, dim=1)
    return dct_feat


def get_gauss_pyramid(x: 'Tensor', scale: 'int'=2):
    """Get gaussian pyramid images with gaussian kernel.
    """
    pyr = [x]
    kernel = fspecial(3, 0.5, x.shape[1])
    pad_func = ExactPadding2d(3, stride=1, mode='same')
    for i in range(scale):
        x = F.conv2d(pad_func(x), kernel, groups=x.shape[1])
        x = x[:, :, 1::2, 1::2]
        pyr.append(x)
    return pyr


class SCFpyr_PyTorch(object):
    """
    This is a modified version of buildSFpyr, that constructs a
    complex-valued steerable pyramid  using Hilbert-transform pairs
    of filters. Note that the imaginary parts will *not* be steerable.
    Pytorch version >= 1.8.0

    """

    def __init__(self, height=5, nbands=4, scale_factor=2, device=None):
        self.height = height
        self.nbands = nbands
        self.scale_factor = scale_factor
        self.device = torch.device('cpu') if device is None else device
        self.lutsize = 1024
        self.Xcosn = np.pi * np.array(range(-(2 * self.lutsize + 1), self.lutsize + 2)) / self.lutsize
        self.alpha = (self.Xcosn + np.pi) % (2 * np.pi) - np.pi
        self.complex_fact_construct = np.power(complex(0, -1), self.nbands - 1)
        self.complex_fact_reconstruct = np.power(complex(0, 1), self.nbands - 1)

    def build(self, im_batch):
        """ Decomposes a batch of images into a complex steerable pyramid.
        The pyramid typically has ~4 levels and 4-8 orientations.

        Args:
            im_batch (torch.Tensor): Batch of images of shape [N,C,H,W]

        Returns:
            pyramid: list containing torch.Tensor objects storing the pyramid
        """
        assert im_batch.device == self.device, 'Devices invalid (pyr = {}, batch = {})'.format(self.device, im_batch.device)
        assert im_batch.dim() == 4, 'Image batch must be of shape [N,C,H,W]'
        assert im_batch.shape[1] == 1, 'Second dimension must be 1 encoding grayscale image'
        im_batch = im_batch.squeeze(1)
        height, width = im_batch.shape[1], im_batch.shape[2]
        if self.height > int(np.floor(np.log2(min(width, height))) - 2):
            raise RuntimeError('Cannot build {} levels, image too small.'.format(self.height))
        log_rad, angle = math_util.prepare_grid(height, width)
        Xrcos, Yrcos = math_util.rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(1 - Yrcos ** 2)
        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)
        lo0mask = torch.from_numpy(lo0mask).float()[None, :, :, None]
        hi0mask = torch.from_numpy(hi0mask).float()[None, :, :, None]
        batch_dft = torch.fft.fft2(im_batch)
        batch_dft = math_util.batch_fftshift2d(batch_dft)
        lo0dft = batch_dft * lo0mask
        coeff = self._build_levels(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height)
        hi0dft = batch_dft * hi0mask
        hi0 = math_util.batch_ifftshift2d(hi0dft)
        hi0 = torch.fft.ifft2(hi0)
        hi0_real = hi0.real
        coeff.insert(0, hi0_real)
        return coeff

    def _build_levels(self, lodft, log_rad, angle, Xrcos, Yrcos, height):
        if height <= 0:
            lo0 = math_util.batch_ifftshift2d(lodft)
            lo0 = torch.fft.ifft2(lo0)
            lo0_real = lo0.real
            coeff = [lo0_real]
        else:
            Xrcos = Xrcos - np.log2(self.scale_factor)
            himask = pointOp(log_rad, Yrcos, Xrcos)
            himask = torch.from_numpy(himask[None, :, :, None]).float()
            order = self.nbands - 1
            const = np.power(2, 2 * order) * np.square(factorial(order)) / (self.nbands * factorial(2 * order))
            Ycosn = 2 * np.sqrt(const) * np.power(np.cos(self.Xcosn), order) * (np.abs(self.alpha) < np.pi / 2)
            orientations = []
            for b in range(self.nbands):
                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi * b / self.nbands)
                anglemask = anglemask[None, :, :, None]
                anglemask = torch.from_numpy(anglemask).float()
                banddft = lodft * anglemask * himask
                banddft = torch.unbind(banddft, -1)
                banddft_real = self.complex_fact_construct.real * banddft[0] - self.complex_fact_construct.imag * banddft[1]
                banddft_imag = self.complex_fact_construct.real * banddft[1] + self.complex_fact_construct.imag * banddft[0]
                banddft = torch.stack((banddft_real, banddft_imag), -1)
                band = math_util.batch_ifftshift2d(banddft)
                band = torch.fft.ifft2(band)
                orientations.append(torch.stack((band.real, band.imag), -1))
            dims = np.array(lodft.shape[1:3])
            low_ind_start = (np.ceil((dims + 0.5) / 2) - np.ceil((np.ceil((dims - 0.5) / 2) + 0.5) / 2)).astype(int)
            low_ind_end = (low_ind_start + np.ceil((dims - 0.5) / 2)).astype(int)
            log_rad = log_rad[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]
            angle = angle[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]
            lodft = lodft[:, low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1], :]
            YIrcos = np.abs(np.sqrt(1 - Yrcos ** 2))
            lomask = pointOp(log_rad, YIrcos, Xrcos)
            lomask = torch.from_numpy(lomask[None, :, :, None]).float()
            lomask = lomask
            lodft = lomask * lodft
            coeff = self._build_levels(lodft, log_rad, angle, Xrcos, Yrcos, height - 1)
            coeff.insert(0, orientations)
        return coeff


def norm_sender_normalized(pyr, num_scale=2, num_bands=6, blksz=3, eps=1e-12):
    """Normalize pyramid with local spatial neighbor and band neighbor
    """
    border = blksz // 2
    guardband = 16
    subbands = []
    for si in range(num_scale):
        for bi in range(num_bands):
            idx = si * num_bands + bi
            current_band = pyr[idx]
            N = blksz ** 2
            tmp = F.unfold(current_band.unsqueeze(1), 3, stride=1)
            tmp = tmp.transpose(1, 2)
            b, hw = tmp.shape[:2]
            parent_idx = idx + num_bands
            if parent_idx < len(pyr):
                tmp_parent = pyr[parent_idx]
                tmp_parent = imresize(tmp_parent, sizes=current_band.shape[-2:])
                tmp_parent = tmp_parent[:, border:-border, border:-border].reshape(b, hw, 1)
                tmp = torch.cat((tmp, tmp_parent), dim=-1)
                N += 1
            for ni in range(num_bands):
                if ni != bi:
                    ni_idx = si * num_bands + ni
                    tmp_nei = pyr[ni_idx]
                    tmp_nei = tmp_nei[:, border:-border, border:-border].reshape(b, hw, 1)
                    tmp = torch.cat((tmp, tmp_nei), dim=-1)
            C_x = tmp.transpose(1, 2) @ tmp / tmp.shape[1]
            L, Q = torch.linalg.eigh(C_x)
            L_pos = L * (L > 0)
            L_pos_sum = L_pos.sum(dim=1, keepdim=True)
            L = L_pos * L.sum(dim=1, keepdim=True) / (L_pos_sum + (L_pos_sum == 0))
            C_x = Q @ torch.diag_embed(L) @ Q.transpose(1, 2)
            o_c = current_band[:, border:-border, border:-border]
            b, h, w = o_c.shape
            o_c = o_c.reshape(b, hw)
            o_c = o_c - o_c.mean(dim=1, keepdim=True)
            tmp_y = torch.linalg.solve(C_x.transpose(1, 2), tmp.transpose(1, 2)).transpose(1, 2) * tmp / N
            tmp_y = tmp_y
            z = tmp_y.sum(dim=2).sqrt()
            mask = z != 0
            g_c = o_c * mask / (z * mask + eps)
            g_c = g_c.reshape(b, h, w)
            gb = int(guardband / 2 ** si)
            g_c = g_c[:, gb:-gb, gb:-gb]
            g_c = g_c - g_c.mean(dim=(1, 2), keepdim=True)
            subbands.append(g_c)
    return subbands


def global_gsm(img: 'Tensor'):
    """Global feature from gassian scale mixture model
    """
    batch_size = img.shape[0]
    num_bands = 6
    pyr = SCFpyr_PyTorch(height=2, nbands=num_bands, device=img.device).build(img)
    lp_bands = [x[..., 0] for x in pyr[1]] + [x[..., 0] for x in pyr[2]]
    subbands = norm_sender_normalized(lp_bands)
    feat = []
    for sb in subbands:
        feat.append(gamma_gen_gauss(sb.reshape(batch_size, -1)))
    for i in range(num_bands):
        sb1 = subbands[i].reshape(batch_size, -1)
        sb2 = subbands[i + num_bands].reshape(batch_size, -1)
        gs = gamma_gen_gauss(torch.cat((sb1, sb2), dim=1))
        feat.append(gs)
    hp_band = pyr[0]
    for sb in lp_bands:
        curr_band = imresize(sb, sizes=hp_band.shape[1:]).unsqueeze(1)
        _, tmpscore = ssim_func(curr_band, hp_band.unsqueeze(1), get_cs=True, data_range=255)
        feat.append(tmpscore)
    for i in range(num_bands):
        for j in range(i + 1, num_bands):
            _, tmpscore = ssim_func(subbands[i].unsqueeze(1), subbands[j].unsqueeze(1), get_cs=True, data_range=255)
            feat.append(tmpscore)
    feat = torch.stack(feat, dim=1)
    return feat


def im2col(x, kernel, mode='sliding'):
    """simple im2col as matlab

    Args:
        x (Tensor): shape (b, c, h, w)
        kernel (int): kernel size
        mode (string): 
            - sliding (default): rearranges sliding image neighborhoods of kernel size into columns with no zero-padding
            - distinct: rearranges discrete image blocks of kernel size into columns, zero pad right and bottom if necessary
    Return:
        flatten patch (Tensor): (b, h * w / kernel **2, kernel * kernel)
    """
    b, c, h, w = x.shape
    kernel = to_2tuple(kernel)
    if mode == 'sliding':
        stride = 1
    elif mode == 'distinct':
        stride = kernel
        h2 = math.ceil(h / stride[0])
        w2 = math.ceil(w / stride[1])
        pad_row = (h2 - 1) * stride[0] + kernel[0] - h
        pad_col = (w2 - 1) * stride[1] + kernel[1] - w
        x = F.pad(x, (0, pad_col, 0, pad_row))
    else:
        raise NotImplementedError(f'Type {mode} is not implemented yet.')
    patches = F.unfold(x, kernel, dilation=1, stride=stride)
    b, _, pnum = patches.shape
    patches = patches.transpose(1, 2).reshape(b, pnum, -1)
    return patches


def tree_regression(feat, ldau, rdau, threshold_value, pred_value, best_attri):
    """Simple decision tree regression.
    """
    prev_k = k = 0
    for i in range(ldau.shape[0]):
        best_col = best_attri[k] - 1
        threshold = threshold_value[k]
        key_value = feat[best_col]
        prev_k = k
        k = ldau[k] - 1 if key_value <= threshold else rdau[k] - 1
        if k == -1:
            break
    y_pred = pred_value[prev_k]
    return y_pred


def random_forest_regression(feat, ldau, rdau, threshold_value, pred_value, best_attri):
    """Simple random forest regression.

    Note: currently, this is non-differentiable and only support CPU.
    """
    feat = feat.cpu().data.numpy()
    b, dim = feat.shape
    node_num, tree_num = ldau.shape
    pred = []
    for i in range(b):
        tmp_feat = feat[i]
        tmp_pred = []
        for i in range(tree_num):
            tmp_result = tree_regression(tmp_feat, ldau[:, i], rdau[:, i], threshold_value[:, i], pred_value[:, i], best_attri[:, i])
            tmp_pred.append(tmp_result)
        pred.append(tmp_pred)
    pred = torch.tensor(pred)
    return pred.mean(dim=1, keepdim=True)


def nrqm(img: 'Tensor', linear_param, rf_param) ->Tensor:
    """Calculate NRQM
    Args:
        img (Tensor): Input image.
        linear_param (np.array): (4, 1) linear regression params
        rf_param: params of 3 random forest for 3 kinds of features
    """
    assert img.ndim == 4, 'Input image must be a gray or Y (of YCbCr) image with shape (b, c, h, w).'
    b, c, h, w = img.shape
    img = img.double()
    img_pyr = get_gauss_pyramid(img / 255.0)
    f1 = []
    for im in img_pyr:
        f1.append(block_dct(im))
    f1 = torch.cat(f1, dim=1)
    f2 = global_gsm(img)
    f3 = []
    for im in img_pyr:
        col = im2col(im, 5, 'distinct')
        _, s, _ = torch.linalg.svd(col, full_matrices=False)
        f3.append(s)
    f3 = torch.cat(f3, dim=1)
    preds = torch.ones(b, 1)
    for feat, rf in zip([f1, f2, f3], rf_param):
        tmp_pred = random_forest_regression(feat, *rf)
        preds = torch.cat((preds, tmp_pred), dim=1)
    quality = preds @ torch.tensor(linear_param)
    return quality.squeeze()


def calculate_nrqm(img: 'torch.Tensor', crop_border: 'int'=0, test_y_channel: 'bool'=True, color_space: 'str'='yiq', linear_param: 'torch.Tensor'=None, rf_params_list: 'list'=None, **kwargs) ->torch.Tensor:
    """Calculate NRQM
    Args:
        img (Tensor): Input image whose quality needs to be computed.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        test_y_channel (Bool): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
        pretrained_model_path (String): The pretrained model path.
    Returns:
        Tensor: NIQE result.
    """
    if test_y_channel and img.shape[1] == 3:
        img = to_y_channel(img, 255, color_space)
    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
    nrqm_result = nrqm(img, linear_param, rf_params_list)
    return nrqm_result


class NRQM(torch.nn.Module):
    """ NRQM metric

    Ma, Chao, Chih-Yuan Yang, Xiaokang Yang, and Ming-Hsuan Yang.
    "Learning a no-reference quality metric for single-image super-resolution."
    Computer Vision and Image Understanding 158 (2017): 1-16.

    Args:
        - channels (int): Number of processed channel.
        - test_y_channel (Boolean): whether to use y channel on ycbcr.
        - crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        - pretrained_model_path (String): The pretrained model path.
    """

    def __init__(self, test_y_channel: 'bool'=True, color_space: 'str'='yiq', crop_border: 'int'=0, pretrained_model_path: 'str'=None) ->None:
        super(NRQM, self).__init__()
        self.test_y_channel = test_y_channel
        self.crop_border = crop_border
        self.color_space = color_space
        if pretrained_model_path is not None:
            pretrained_model_path = pretrained_model_path
        else:
            pretrained_model_path = load_file_from_url(default_model_urls['url'])
        params = scipy.io.loadmat(pretrained_model_path)['model']
        linear_param = params['linear'][0, 0]
        rf_params_list = []
        for i in range(3):
            tmp_list = []
            tmp_param = params['rf'][0, 0][0, i][0, 0]
            tmp_list.append(tmp_param[0])
            tmp_list.append(tmp_param[1])
            tmp_list.append(tmp_param[4])
            tmp_list.append(tmp_param[5])
            tmp_list.append(tmp_param[6])
            rf_params_list.append(tmp_list)
        self.linear_param = linear_param
        self.rf_params_list = rf_params_list

    def forward(self, X: 'torch.Tensor') ->torch.Tensor:
        """Computation of NRQM metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of nrqm metric.
        """
        score = calculate_nrqm(X, self.crop_border, self.test_y_channel, self.color_space, self.linear_param, self.rf_params_list)
        return score


class PI(torch.nn.Module):
    """ Perceptual Index (PI), introduced by

    Blau, Yochai, Roey Mechrez, Radu Timofte, Tomer Michaeli, and Lihi Zelnik-Manor.
    "The 2018 pirm challenge on perceptual image super-resolution."
    In Proceedings of the European Conference on Computer Vision (ECCV) Workshops, pp. 0-0. 2018.
    Ref url: https://github.com/roimehrez/PIRM2018

    It is a combination of NIQE and NRQM: 1/2 * ((10 - NRQM) + NIQE)

    Args:
        - color_space (str): color space of y channel, default ycbcr.
        - crop_border (int): Cropped pixels in each edge of an image, default 4.
    """

    def __init__(self, crop_border=4, color_space='ycbcr'):
        super(PI, self).__init__()
        self.nrqm = NRQM(crop_border=crop_border, color_space=color_space)
        self.niqe = NIQE(crop_border=crop_border, color_space=color_space)

    def forward(self, X: 'Tensor') ->Tensor:
        """Computation of PI metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of PI metric.
        """
        nrqm_score = self.nrqm(X)
        niqe_score = self.niqe(X)
        score = 1 / 2 * (10 - nrqm_score + niqe_score)
        return score


class AdaptiveConcatPool2d(nn.Module):

    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class PAQ2PIQ(nn.Module):

    def __init__(self, backbone='resnet18', pretrained=True, pretrained_model_path=None):
        super(PAQ2PIQ, self).__init__()
        if backbone == 'resnet18':
            model = tv.models.resnet18(weights='IMAGENET1K_V1')
            cut = -2
            spatial_scale = 1 / 32
        self.blk_size = 20, 20
        self.model_type = self.__class__.__name__
        self.body = nn.Sequential(*list(model.children())[:cut])
        self.head = nn.Sequential(AdaptiveConcatPool2d(), nn.Flatten(), nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.Dropout(p=0.25, inplace=False), nn.Linear(in_features=1024, out_features=512, bias=True), nn.ReLU(inplace=True), nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.Dropout(p=0.5, inplace=False), nn.Linear(in_features=512, out_features=1, bias=True))
        self.roi_pool = RoIPool((2, 2), spatial_scale)
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path)
        elif pretrained:
            load_pretrained_network(self, default_model_urls['url'])

    def forward(self, x):
        im_data = x
        batch_size = im_data.shape[0]
        feats = self.body(im_data)
        global_rois = torch.tensor([0, 0, x.shape[-1], x.shape[-2]]).reshape(1, 4)
        feats = self.roi_pool(feats, [global_rois] * batch_size)
        preds = self.head(feats)
        return preds.view(batch_size, -1)


class CompactLinear(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.randn(1))
        self.bias = nn.parameter.Parameter(torch.randn(1))

    def forward(self, x):
        return x * self.weight + self.bias


class PieAPP(nn.Module):
    """
    PieAPP model implementation.
    
    Args:
        - patch_size (int): Size of the patches to extract from the images.
        - stride (int): Stride to use when extracting patches.
        - pretrained (bool): Whether to use a pretrained model or not.
        - pretrained_model_path (str): Path to the pretrained model.
    
    Methods:
        - flatten(matrix): Takes NxCxHxW input and outputs NxHWC.
        compute_features(input): Computes the features of the input image.
        - preprocess(x): Preprocesses the input image.
        forward(dist, ref): Computes the PieAPP score between the distorted and reference images.
    """

    def __init__(self, patch_size=64, stride=27, pretrained=True, pretrained_model_path=None):
        super(PieAPP, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool6 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool8 = nn.MaxPool2d(2, 2)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool10 = nn.MaxPool2d(2, 2)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1_score = nn.Linear(120832, 512)
        self.fc2_score = nn.Linear(512, 1)
        self.fc1_weight = nn.Linear(2048, 512)
        self.fc2_weight = nn.Linear(512, 1)
        self.ref_score_subtract = CompactLinear()
        self.patch_size = patch_size
        self.stride = stride
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path)
        elif pretrained:
            load_pretrained_network(self, default_model_urls['url'])
        self.pretrained = pretrained

    def flatten(self, matrix):
        return torch.flatten(matrix, 1)

    def compute_features(self, input):
        x3 = F.relu(self.conv3(self.pool2(F.relu(self.conv2(F.relu(self.conv1(input)))))))
        x5 = F.relu(self.conv5(self.pool4(F.relu(self.conv4(x3)))))
        x7 = F.relu(self.conv7(self.pool6(F.relu(self.conv6(x5)))))
        x9 = F.relu(self.conv9(self.pool8(F.relu(self.conv8(x7)))))
        x11 = self.flatten(F.relu(self.conv11(self.pool10(F.relu(self.conv10(x9))))))
        feature_ms = torch.cat((self.flatten(x3), self.flatten(x5), self.flatten(x7), self.flatten(x9), x11), 1)
        return feature_ms, x11

    def preprocess(self, x):
        """Default BGR in [0, 255] in original codes
        """
        x = x[:, [2, 1, 0]] * 255.0
        return x

    def forward(self, dist, ref):
        assert dist.shape == ref.shape, f'Input and reference images should have the same shape, but got {dist.shape}'
        f""" and {ref.shape}"""
        dist = self.preprocess(dist)
        ref = self.preprocess(ref)
        if not self.training:
            image_A_patches = extract_2d_patches(dist, self.patch_size, self.stride, padding='none')
            image_ref_patches = extract_2d_patches(ref, self.patch_size, self.stride, padding='none')
        else:
            image_A_patches, image_ref_patches = dist, ref
            image_A_patches = image_A_patches.unsqueeze(1)
            image_ref_patches = image_ref_patches.unsqueeze(1)
        bsz, num_patches, c, psz, psz = image_A_patches.shape
        image_A_patches = image_A_patches.reshape(bsz * num_patches, c, psz, psz)
        image_ref_patches = image_ref_patches.reshape(bsz * num_patches, c, psz, psz)
        A_multi_scale, A_coarse = self.compute_features(image_A_patches)
        ref_multi_scale, ref_coarse = self.compute_features(image_ref_patches)
        diff_ms = ref_multi_scale - A_multi_scale
        diff_coarse = ref_coarse - A_coarse
        per_patch_score = self.ref_score_subtract(0.01 * self.fc2_score(F.relu(self.fc1_score(diff_ms))))
        per_patch_score = per_patch_score.view((-1, num_patches))
        per_patch_weight = self.fc2_weight(F.relu(self.fc1_weight(diff_coarse))) + 1e-06
        per_patch_weight = per_patch_weight.view((-1, num_patches))
        score = (per_patch_weight * per_patch_score).sum(dim=-1) / per_patch_weight.sum(dim=-1)
        return score.reshape(bsz, 1)


def cal_center_sur_dev(block, block_size):
    """Function to compute center surround Deviation of a block.
    """
    center1 = (block_size + 1) // 2
    center2 = center1 + 1
    center = torch.stack((block[..., center1 - 1], block[..., center2 - 1]), dim=3)
    block = torch.cat((block[..., :center1 - 1], block[..., center1:]), dim=-1)
    block = torch.cat((block[..., :center2 - 1], block[..., center2:]), dim=-1)
    center_std = torch.std(center, dim=[2, 3], unbiased=True)
    surround_std = torch.std(block, dim=[2, 3], unbiased=True)
    cen_sur_dev = center_std / surround_std
    cen_sur_dev = torch.nan_to_num(cen_sur_dev)
    return cen_sur_dev


def noise_criterion(block, block_size, block_var):
    """Function to analyze block for Gaussian noise distortions.
    """
    block_sigma = torch.sqrt(block_var)
    cen_sur_dev = cal_center_sur_dev(block, block_size)
    block_beta = torch.abs(block_sigma - cen_sur_dev) / torch.max(block_sigma, cen_sur_dev)
    return block_sigma, block_beta


def notice_dist_criterion(blocks, window_size, block_impaired_threshold, N):
    """
    Analyze blocks for noticeable artifacts and Gaussian noise distortions.

    Args:
        blocks (torch.Tensor): Tensor of shape (b, num_blocks, block_size, block_size).
        window_size (int): Size of the window for segment analysis.
        block_impaired_threshold (float): Threshold for considering a block as impaired.
        N (int): Size of the blocks (same as block_size).

    Returns:
        torch.Tensor: Tensor indicating impaired blocks.
    """
    top_edge = blocks[:, :, 0, :]
    seg_top_edge = top_edge.unfold(-1, window_size, 1)
    right_side_edge = blocks[:, :, :, N - 1]
    seg_right_side_edge = right_side_edge.unfold(-1, window_size, 1)
    down_side_edge = blocks[:, :, N - 1, :]
    seg_down_side_edge = down_side_edge.unfold(-1, window_size, 1)
    left_side_edge = blocks[:, :, :, 0]
    seg_left_side_edge = left_side_edge.unfold(-1, window_size, 1)
    seg_top_edge_std_dev = torch.std(seg_top_edge, dim=-1, unbiased=True)
    seg_right_side_edge_std_dev = torch.std(seg_right_side_edge, dim=-1, unbiased=True)
    seg_down_side_edge_std_dev = torch.std(seg_down_side_edge, dim=-1, unbiased=True)
    seg_left_side_edge_std_dev = torch.std(seg_left_side_edge, dim=-1, unbiased=True)
    block_impaired = (seg_top_edge_std_dev < block_impaired_threshold).sum(dim=2) + (seg_right_side_edge_std_dev < block_impaired_threshold).sum(dim=2) + (seg_down_side_edge_std_dev < block_impaired_threshold).sum(dim=2) + (seg_left_side_edge_std_dev < block_impaired_threshold).sum(dim=2) > 0
    return block_impaired


def piqe(img: 'torch.Tensor', block_size: 'int'=16, activity_threshold: 'float'=0.1, block_impaired_threshold: 'float'=0.1, window_size: 'int'=6) ->torch.Tensor:
    """
        Calculates the Perceptual Image Quality Estimator (PIQE) score for an input image.
        Args:
            - img (torch.Tensor): The input image tensor.
            - block_size (int, optional): The size of the blocks used for processing. Defaults to 16.
            - activity_threshold (float, optional): The threshold for considering a block as active. Defaults to 0.1.
            - block_impaired_threshold (float, optional): The threshold for considering a block as impaired. Defaults to 0.1.
            - window_size (int, optional): The size of the window used for block analysis. Defaults to 6.
        Returns:
            - torch.Tensor: The PIQE score for the input image.
    """
    if img.shape[1] == 3:
        img = to_y_channel(img, out_data_range=1, color_space='yiq')
    img = torch.round(255 * (img / torch.max(img.flatten(1), dim=-1)[0].reshape(img.shape[0], 1, 1, 1)))
    bsz, _, height, width = img.shape
    col_pad, row_pad = width % block_size, height % block_size
    img = symm_pad(img, (0, col_pad, 0, row_pad))
    new_height, new_width = img.shape[2], img.shape[3]
    img_normalized = normalize_img_with_gauss(img, padding='replicate')
    blocks = img_normalized.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    blocks = blocks.contiguous().view(bsz, -1, block_size, block_size)
    block_var = torch.var(blocks, dim=[2, 3], unbiased=True)
    active_blocks = block_var > activity_threshold
    block_sigma, block_beta = noise_criterion(blocks, block_size - 1, block_var)
    noise_mask = block_sigma > 2 * block_beta
    block_impaired = notice_dist_criterion(blocks, window_size, block_impaired_threshold, block_size)
    WHSA = active_blocks.float()
    WNDC = block_impaired.float()
    WNC = noise_mask.float()
    dist_block_scores = WHSA * WNDC * (1 - block_var) + WHSA * WNC * block_var
    NHSA = active_blocks.sum(dim=1)
    dist_block_scores = dist_block_scores.sum(dim=1)
    C = 1
    score = (dist_block_scores + C) / (C + NHSA) * 100
    noticeable_artifacts_mask = block_impaired.view(bsz, 1, new_height // block_size, new_width // block_size)
    noticeable_artifacts_mask = F.interpolate(noticeable_artifacts_mask.float(), scale_factor=block_size, mode='nearest')[..., :height, :width]
    noise_mask = noise_mask.view(bsz, 1, new_height // block_size, new_width // block_size)
    noise_mask = F.interpolate(noise_mask.float(), scale_factor=block_size, mode='nearest')[..., :height, :width]
    activity_mask = active_blocks.view(bsz, 1, new_height // block_size, new_width // block_size)
    activity_mask = F.interpolate(activity_mask.float(), scale_factor=block_size, mode='nearest')[..., :height, :width]
    return score, noticeable_artifacts_mask, noise_mask, activity_mask


class PIQE(torch.nn.Module):
    """
    PIQE module.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: PIQE score.
    """

    def get_masks(self):
        assert self.results is not None, 'Please calculate the piqe score first.'
        return {'noticeable_artifacts_mask': self.results[1], 'noise_mask': self.results[2], 'activity_mask': self.results[3]}

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        self.results = piqe(x)
        return self.results[0]


def psnr(x, y, test_y_channel=False, data_range=1.0, eps=1e-08, color_space='yiq'):
    """Compute Peak Signal-to-Noise Ratio for a batch of images.
    Supports both greyscale and color images with RGB channel order.
    Args:
        - x: An input tensor. Shape :math:`(N, C, H, W)`.
        - y: A target tensor. Shape :math:`(N, C, H, W)`.
        - test_y_channel (Boolean): Convert RGB image to YCbCr format and computes PSNR
        only on luminance channel if `True`. Compute on all 3 channels otherwise.
        - data_range: Maximum value range of images (default 1.0).

    Returns:
        PSNR Index of similarity between two images.
    """
    if x.shape[1] == 3 and test_y_channel:
        x = to_y_channel(x, data_range, color_space)
        y = to_y_channel(y, data_range, color_space)
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    score = 10 * torch.log10(data_range ** 2 / (mse + eps))
    return score


class PSNR(nn.Module):
    """
    Args:
        - X, Y (torch.Tensor): distorted image and reference image tensor with shape (B, 3, H, W)
        - test_y_channel (Boolean): Convert RGB image to YCbCr format and computes PSNR
            only on luminance channel if `True`. Compute on all 3 channels otherwise.
        - kwargs: other parameters, including
            - data_range: maximum numeric value
            - eps: small constant for numeric stability
    Return:
        score (torch.Tensor): (B, 1)
    """

    def __init__(self, test_y_channel=False, crop_border=0, **kwargs):
        super().__init__()
        self.test_y_channel = test_y_channel
        self.kwargs = kwargs
        self.crop_border = crop_border

    def forward(self, X, Y):
        assert X.shape == Y.shape, f'Input and reference images should have the same shape, but got {X.shape} and {Y.shape}'
        if self.crop_border != 0:
            crop_border = self.crop_border
            X = X[..., crop_border:-crop_border, crop_border:-crop_border]
            Y = Y[..., crop_border:-crop_border, crop_border:-crop_border]
        score = psnr(X, Y, self.test_y_channel, **self.kwargs)
        return score


def expand2square(pil_img):
    background_color = tuple(int(x * 255) for x in OPENAI_CLIP_MEAN)
    width, height = pil_img.size
    maxwh = max(width, height)
    result = Image.new(pil_img.mode, (maxwh, maxwh), background_color)
    result.paste(pil_img, ((maxwh - width) // 2, (maxwh - height) // 2))
    return result


class QAlign(nn.Module):

    def __init__(self, dtype='fp16') ->None:
        super().__init__()
        assert dtype in ['fp16', '4bit', '8bit'], f"Invalid dtype {dtype}. Choose from 'nf4', 'int8', or 'fp16'."
        self.model = AutoModelForCausalLM.from_pretrained('q-future/one-align', trust_remote_code=True, load_in_4bit=True if dtype == '4bit' else False, load_in_8bit=True if dtype == '8bit' else False, torch_dtype=torch.float16 if dtype == 'fp16' else None)
        self.image_processor = CLIPImageProcessor.from_pretrained('q-future/one-align')

    def preprocess(self, x):
        assert x.shape[0] == 1, 'Currently, only support batch size 1.'
        images = F.to_pil_image(x[0])
        images = expand2square(images)
        image_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half()
        return image_tensor

    def forward(self, x, task_='quality', input_='image'):
        """
            task_: str, optional [quality, aesthetic]
        """
        if input_ == 'image':
            image_tensor = self.preprocess(x)
            score = self.model.score(images=None, image_tensor=image_tensor, task_=task_, input_=input_)
        else:
            raise NotImplementedError(f'Input type {input_} is not supported yet.')
        return score


def preprocess_rgb(x, test_y_channel, data_range: 'float'=1, color_space='yiq'):
    """
    Preprocesses an RGB image tensor.

    Args:
        - x (torch.Tensor): The input RGB image tensor.
        - test_y_channel (bool): Whether to test the Y channel.
        - data_range (float): The data range of the input tensor. Default is 1.
        - color_space (str): The color space of the input tensor. Default is "yiq".

    Returns:
        torch.Tensor: The preprocessed RGB image tensor.
    """
    if test_y_channel and x.shape[1] == 3:
        x = to_y_channel(x, data_range, color_space)
    else:
        x = x * data_range
    if data_range == 255:
        x = x - x.detach() + x.round()
    return x


def filter2(input, weight, shape='same'):
    if shape == 'same':
        return imfilter(input, weight, groups=input.shape[1])
    elif shape == 'valid':
        return F.conv2d(input, weight, stride=1, padding=0, groups=input.shape[1])
    else:
        raise NotImplementedError(f'Shape type {shape} is not implemented.')


def ssim(X, Y, win=None, get_ssim_map=False, get_cs=False, get_weight=False, downsample=False, data_range=1.0):
    if win is None:
        win = fspecial(11, 1.5, X.shape[1])
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    f = max(1, round(min(X.size()[-2:]) / 256))
    if f > 1 and downsample:
        X = F.avg_pool2d(X, kernel_size=f)
        Y = F.avg_pool2d(Y, kernel_size=f)
    mu1 = filter2(X, win, 'valid')
    mu2 = filter2(Y, win, 'valid')
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(X * X, win, 'valid') - mu1_sq
    sigma2_sq = filter2(Y * Y, win, 'valid') - mu2_sq
    sigma12 = filter2(X * Y, win, 'valid') - mu1_mu2
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    cs_map = F.relu(cs_map)
    ssim_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1) * cs_map
    ssim_val = ssim_map.mean([1, 2, 3])
    if get_weight:
        weights = torch.log((1 + sigma1_sq / C2) * (1 + sigma2_sq / C2))
        return ssim_map, weights
    if get_ssim_map:
        return ssim_map
    if get_cs:
        return ssim_val, cs_map.mean([1, 2, 3])
    return ssim_val


class SSIM(torch.nn.Module):
    """Args:
        - channel: number of channel.
        - downsample: boolean, whether to downsample same as official matlab code.
        - test_y_channel: boolean, whether to use y channel on ycbcr same as official matlab code.
    """

    def __init__(self, channels=3, downsample=False, test_y_channel=True, color_space='yiq', crop_border=0.0):
        super(SSIM, self).__init__()
        self.downsample = downsample
        self.test_y_channel = test_y_channel
        self.color_space = color_space
        self.crop_border = crop_border
        self.data_range = 255

    def forward(self, X, Y):
        assert X.shape == Y.shape, f'Input {X.shape} and reference images should have the same shape'
        if self.crop_border != 0:
            crop_border = self.crop_border
            X = X[..., crop_border:-crop_border, crop_border:-crop_border]
            Y = Y[..., crop_border:-crop_border, crop_border:-crop_border]
        X = preprocess_rgb(X, self.test_y_channel, self.data_range, self.color_space)
        Y = preprocess_rgb(Y, self.test_y_channel, self.data_range, self.color_space)
        score = ssim(X, Y, data_range=self.data_range, downsample=self.downsample)
        return score


def ms_ssim(X, Y, win=None, data_range=1.0, downsample=False, test_y_channel=True, is_prod=True, color_space='yiq'):
    """Compute Multiscale structural similarity for a batch of images.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        win: Window setting.
        downsample: Boolean, whether to downsample which mimics official SSIM matlab code.
        test_y_channel: Boolean, whether to use y channel on ycbcr.
        is_prod: Boolean, calculate product or sum between mcs and weight.
    Returns:
        Index of similarity between two images. Usually in [0, 1] interval.
    """
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = ssim(X, Y, win=win, get_cs=True, downsample=downsample, data_range=data_range)
        mcs.append(cs)
        padding = X.shape[2] % 2, X.shape[3] % 2
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)
    mcs = torch.stack(mcs, dim=0)
    if is_prod:
        msssim_val = torch.prod(mcs[:-1] ** weights[:-1].unsqueeze(1), dim=0) * ssim_val ** weights[-1]
    else:
        weights = weights / torch.sum(weights)
        msssim_val = torch.sum(mcs[:-1] * weights[:-1].unsqueeze(1), dim=0) + ssim_val * weights[-1]
    return msssim_val


class MS_SSIM(torch.nn.Module):
    """Multiscale structure similarity

    References:
        Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale structural similarity for image
        quality assessment." In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers,
        2003, vol. 2, pp. 1398-1402. Ieee, 2003.

    Args:
        channel: Number of channel.
        downsample: Boolean, whether to downsample which mimics official SSIM matlab code.
        test_y_channel: Boolean, whether to use y channel on ycbcr which mimics official matlab code.
    """

    def __init__(self, channels=3, downsample=False, test_y_channel=True, is_prod=True, color_space='yiq'):
        super(MS_SSIM, self).__init__()
        self.downsample = downsample
        self.test_y_channel = test_y_channel
        self.color_space = color_space
        self.is_prod = is_prod
        self.data_range = 255

    def forward(self, X, Y):
        """Computation of MS-SSIM metric.
        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of MS-SSIM metric in [0, 1] range.
        """
        assert X.shape == Y.shape, 'Input and reference images should have the same shape, but got'
        f"""{X.shape} and {Y.shape}"""
        X = preprocess_rgb(X, self.test_y_channel, self.data_range, self.color_space)
        Y = preprocess_rgb(Y, self.test_y_channel, self.data_range, self.color_space)
        score = ms_ssim(X, Y, data_range=self.data_range, downsample=self.downsample, is_prod=self.is_prod)
        return score


class CW_SSIM(torch.nn.Module):
    """Complex-Wavelet Structural SIMilarity (CW-SSIM) index.

    References:
        M. P. Sampat, Z. Wang, S. Gupta, A. C. Bovik, M. K. Markey.
        "Complex Wavelet Structural Similarity: A New Image Similarity Index",
        IEEE Transactions on Image Processing, 18(11), 2385-401, 2009.

    Args:
        - channel: Number of channel.
        - test_y_channel: Boolean, whether to use y channel on ycbcr.
        - level: The number of levels to used in the complex steerable pyramid decomposition
        - ori: The number of orientations to be used in the complex steerable pyramid decomposition
        - guardb: How much is discarded from the four image boundaries.
        - K: the constant in the CWSSIM index formula (see the above reference) default value: K=0
    """

    def __init__(self, channels=1, level=4, ori=8, guardb=0, K=0, test_y_channel=True, color_space='yiq'):
        super(CW_SSIM, self).__init__()
        self.channels = channels
        self.level = level
        self.ori = ori
        self.guardb = guardb
        self.K = K
        self.test_y_channel = test_y_channel
        self.color_space = color_space
        self.register_buffer('win7', torch.ones(channels, 1, 7, 7) / (7 * 7))

    def conj(self, x, y):
        a = x[..., 0]
        b = x[..., 1]
        c = y[..., 0]
        d = -y[..., 1]
        return torch.stack((a * c - b * d, b * c + a * d), dim=1)

    def conv2d_complex(self, x, win, groups=1):
        real = F.conv2d(x[:, 0, ...].unsqueeze(1), win, groups=groups)
        imaginary = F.conv2d(x[:, 1, ...].unsqueeze(1), win, groups=groups)
        return torch.stack((real, imaginary), dim=-1)

    def cw_ssim(self, x, y, test_y_channel):
        """Compute CW-SSIM for a batch of images.
        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.
            test_y_channel: Boolean, whether to use y channel on ycbcr.
        Returns:
            Index of similarity between two images. Usually in [0, 1] interval.
        """
        if test_y_channel and x.shape[1] == 3:
            x = to_y_channel(x, 255, self.color_space)
            y = to_y_channel(y, 255, self.color_space)
        pyr = SCFpyr_PyTorch(height=self.level, nbands=self.ori, scale_factor=2, device=x.device)
        cw_x = pyr.build(x)
        cw_y = pyr.build(y)
        bandind = self.level
        band_cssim = []
        s = np.array(cw_x[bandind][0].size()[1:3])
        w = fspecial(s - 7 + 1, s[0] / 4, 1)
        gb = int(self.guardb / 2 ** (self.level - 1))
        self.win7 = self.win7
        for i in range(self.ori):
            band1 = cw_x[bandind][i]
            band2 = cw_y[bandind][i]
            band1 = band1[:, gb:s[0] - gb, gb:s[1] - gb, :]
            band2 = band2[:, gb:s[0] - gb, gb:s[1] - gb, :]
            corr = self.conj(band1, band2)
            corr_band = self.conv2d_complex(corr, self.win7, groups=self.channels)
            varr = (math_util.abs(band1) ** 2 + math_util.abs(band2) ** 2).unsqueeze(1)
            varr_band = F.conv2d(varr, self.win7, stride=1, padding=0, groups=self.channels)
            cssim_map = (2 * math_util.abs(corr_band) + self.K) / (varr_band + self.K)
            band_cssim.append((cssim_map * w.repeat(cssim_map.shape[0], 1, 1, 1)).sum([2, 3]).mean(1))
        return torch.stack(band_cssim, dim=1).mean(1)

    def forward(self, X, Y):
        """Computation of CW-SSIM metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
            Y: A target tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of CW-SSIM metric in [0, 1] range.
        """
        assert X.shape == Y.shape, f'Input {X.shape} and reference images should have the same shape'
        score = self.cw_ssim(X, Y, self.test_y_channel)
        return score


class vggnet(nn.Module):

    def __init__(self, requires_grad=False, variant='shift_tolerant', filter_size=3):
        super(vggnet, self).__init__()
        filter_size = 3
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        if variant == 'vanilla':
            vgg_features = tv.vgg16(pretrained=False).features
            for x in range(4):
                self.slice1.add_module(str(x), vgg_features[x])
            for x in range(4, 9):
                self.slice2.add_module(str(x), vgg_features[x])
            for x in range(9, 16):
                self.slice3.add_module(str(x), vgg_features[x])
            for x in range(16, 23):
                self.slice4.add_module(str(x), vgg_features[x])
            for x in range(23, 30):
                self.slice5.add_module(str(x), vgg_features[x])
        elif variant == 'shift_tolerant':
            vgg_features = vgg16(filter_size=filter_size, pad_more=True).features
            for x in range(4):
                self.slice1.add_module(str(x), vgg_features[x])
            for x in range(4, 10):
                self.slice2.add_module(str(x), vgg_features[x])
            for x in range(10, 18):
                self.slice3.add_module(str(x), vgg_features[x])
            for x in range(18, 26):
                self.slice4.add_module(str(x), vgg_features[x])
            for x in range(26, 34):
                self.slice5.add_module(str(x), vgg_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class STLPIPS(nn.Module):
    """ST-LPIPS model.
    Args:
        lpips (Boolean) : Whether to use linear layers on top of base/trunk network.
        pretrained (Boolean): Whether means linear layers are calibrated with human
            perceptual judgments.
        net (String): ['alex','vgg','squeeze'] are the base/trunk networks available.
        pretrained_model_path (String): Petrained model path.

        The following parameters should only be changed if training the network:

        eval_mode (Boolean): choose the mode; True is for test mode (default).
        pnet_tune (Boolean): Whether to tune the base/trunk network.
        use_dropout (Boolean): Whether to use dropout when training linear layers.
    """

    def __init__(self, pretrained=True, net='alex', variant='shift_tolerant', lpips=True, spatial=False, pnet_tune=False, use_dropout=True, pretrained_model_path=None, eval_mode=True, blur_filter_size=3):
        super(STLPIPS, self).__init__()
        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.spatial = spatial
        self.lpips = lpips
        self.scaling_layer = ScalingLayer()
        if self.pnet_type in ['vgg']:
            net_type = vggnet
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = alexnet
            self.chns = [64, 192, 384, 256, 256]
        self.net = net_type(requires_grad=self.pnet_tune, variant=variant, filter_size=blur_filter_size)
        self.L = len(self.chns)
        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == 'squeeze':
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]
            self.lins = nn.ModuleList(self.lins)
            if pretrained_model_path is not None:
                load_pretrained_network(self, pretrained_model_path, False)
            elif pretrained:
                load_pretrained_network(self, default_model_urls[f'{net}_{variant}'], False)
        if eval_mode:
            self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=True):
        """Computation IQA using LPIPS.
        Args:
            in1: An input tensor. Shape :math:`(N, C, H, W)`.
            in0: A reference tensor. Shape :math:`(N, C, H, W)`.
            retPerLayer (Boolean): return result contains result of
                each layer or not. Default: False.
            normalize (Boolean): Whether to normalize image data range
                in [0,1] to [-1,1]. Default: True.

        Returns:
            Quality score.

        """
        if normalize:
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        elif self.spatial:
            res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
        else:
            res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]
        val = res[0]
        for l in range(1, self.L):
            val += res[l]
        if retPerLayer:
            return val, res
        else:
            return val


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt, memory):
        memory = self.norm2(memory)
        tgt2 = self.norm1(tgt)
        tgt2, self.attn_map = self.multihead_attn(query=tgt2, key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory)
        return output


class GatedConv(nn.Module):

    def __init__(self, weightdim, ksz=3):
        super().__init__()
        self.splitconv = nn.Conv2d(weightdim, weightdim * 2, 1, 1, 0)
        self.act = nn.GELU()
        self.weight_blk = nn.Sequential(nn.Conv2d(weightdim, 64, 1, stride=1), nn.GELU(), nn.Conv2d(64, 64, ksz, stride=1, padding=1), nn.GELU(), nn.Conv2d(64, 1, ksz, stride=1, padding=1), nn.Sigmoid())

    def forward(self, x):
        x1, x2 = self.splitconv(x).chunk(2, dim=1)
        weight = self.weight_blk(x2)
        x1 = self.act(x1)
        return x1 * weight


def _cfg(url='', **kwargs):
    return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None, 'crop_pct': 0.9, 'interpolation': 'bicubic', 'fixed_input_size': True, 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'first_conv': 'patch_embed.proj', 'classifier': 'head', **kwargs}


default_cfgs = {'swin_base_patch4_window12_384': _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth', input_size=(3, 384, 384), crop_pct=1.0), 'swin_base_patch4_window7_224': _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth'), 'swin_large_patch4_window12_384': _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth', input_size=(3, 384, 384), crop_pct=1.0), 'swin_large_patch4_window7_224': _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth'), 'swin_small_patch4_window7_224': _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'), 'swin_tiny_patch4_window7_224': _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'), 'swin_base_patch4_window12_384_in22k': _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth', input_size=(3, 384, 384), crop_pct=1.0, num_classes=21841), 'swin_base_patch4_window7_224_in22k': _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth', num_classes=21841), 'swin_large_patch4_window12_384_in22k': _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth', input_size=(3, 384, 384), crop_pct=1.0, num_classes=21841), 'swin_large_patch4_window7_224_in22k': _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth', num_classes=21841), 'swin_s3_tiny_224': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/s3_t-1d53f6a8.pth'), 'swin_s3_small_224': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/s3_s-3bb4c69d.pth'), 'swin_s3_base_224': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/s3_b-a1e95db4.pth')}


def create_swin(name, **kwargs):
    return eval(name)(pretrained_cfg=default_cfgs[name], **kwargs)


class CFANet(nn.Module):

    def __init__(self, semantic_model_name='resnet50', model_name='cfanet_nr_koniq_res50', backbone_pretrain=True, in_size=None, use_ref=True, num_class=1, num_crop=1, crop_size=256, inter_dim=256, num_heads=4, num_attn_layers=1, dprate=0.1, activation='gelu', pretrained=True, pretrained_model_path=None, out_act=False, block_pool='weighted_avg', test_img_size=None, align_crop_face=True, default_mean=IMAGENET_DEFAULT_MEAN, default_std=IMAGENET_DEFAULT_STD):
        super().__init__()
        self.in_size = in_size
        self.model_name = model_name
        self.semantic_model_name = semantic_model_name
        self.semantic_level = -1
        self.crop_size = crop_size
        self.use_ref = use_ref
        self.num_class = num_class
        self.block_pool = block_pool
        self.test_img_size = test_img_size
        self.align_crop_face = align_crop_face
        if 'swin' in semantic_model_name:
            self.semantic_model = create_swin(semantic_model_name, pretrained=True, drop_path_rate=0.0)
            feature_dim = self.semantic_model.num_features
            feature_dim_list = [int(self.semantic_model.embed_dim * 2 ** i) for i in range(self.semantic_model.num_layers)]
            feature_dim_list = feature_dim_list[1:] + [feature_dim]
            all_feature_dim = sum(feature_dim_list)
        elif 'clip' in semantic_model_name:
            semantic_model_name = semantic_model_name.replace('clip_', '')
            self.semantic_model = [load(semantic_model_name, 'cpu')]
            feature_dim_list = self.semantic_model[0].visual.feature_dim_list
            default_mean, default_std = OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
        else:
            self.semantic_model = timm.create_model(semantic_model_name, pretrained=backbone_pretrain, features_only=True)
            feature_dim_list = self.semantic_model.feature_info.channels()
            feature_dim = feature_dim_list[self.semantic_level]
            all_feature_dim = sum(feature_dim_list)
            self.fix_bn(self.semantic_model)
        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)
        self.fusion_mul = 3 if use_ref else 1
        ca_layers = sa_layers = num_attn_layers
        self.act_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()
        dim_feedforward = min(4 * inter_dim, 2048)
        tmp_layer = TransformerEncoderLayer(inter_dim, nhead=num_heads, dim_feedforward=dim_feedforward, normalize_before=True, dropout=dprate, activation=activation)
        self.sa_attn_blks = nn.ModuleList()
        self.dim_reduce = nn.ModuleList()
        self.weight_pool = nn.ModuleList()
        for idx, dim in enumerate(feature_dim_list):
            dim = dim * 3 if use_ref else dim
            if use_ref:
                self.weight_pool.append(nn.Sequential(nn.Conv2d(dim // 3, 64, 1, stride=1), self.act_layer, nn.Conv2d(64, 64, 3, stride=1, padding=1), self.act_layer, nn.Conv2d(64, 1, 3, stride=1, padding=1), nn.Sigmoid()))
            else:
                self.weight_pool.append(GatedConv(dim))
            self.dim_reduce.append(nn.Sequential(nn.Conv2d(dim, inter_dim, 1, 1), self.act_layer))
            self.sa_attn_blks.append(TransformerEncoder(tmp_layer, sa_layers))
        self.attn_blks = nn.ModuleList()
        tmp_layer = TransformerDecoderLayer(inter_dim, nhead=num_heads, dim_feedforward=dim_feedforward, normalize_before=True, dropout=dprate, activation=activation)
        for i in range(len(feature_dim_list) - 1):
            self.attn_blks.append(TransformerDecoder(tmp_layer, ca_layers))
        self.attn_pool = TransformerEncoderLayer(inter_dim, nhead=num_heads, dim_feedforward=dim_feedforward, normalize_before=True, dropout=dprate, activation=activation)
        linear_dim = inter_dim
        self.score_linear = [nn.LayerNorm(linear_dim), nn.Linear(linear_dim, linear_dim), self.act_layer, nn.LayerNorm(linear_dim), nn.Linear(linear_dim, linear_dim), self.act_layer, nn.Linear(linear_dim, self.num_class)]
        if out_act and self.num_class == 1:
            self.score_linear.append(nn.Softplus())
        if self.num_class > 1:
            self.score_linear.append(nn.Softmax(dim=-1))
        self.score_linear = nn.Sequential(*self.score_linear)
        self.h_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 32, 1))
        self.w_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 1, 32))
        nn.init.trunc_normal_(self.h_emb.data, std=0.02)
        nn.init.trunc_normal_(self.w_emb.data, std=0.02)
        self._init_linear(self.dim_reduce)
        self._init_linear(self.sa_attn_blks)
        self._init_linear(self.attn_blks)
        self._init_linear(self.attn_pool)
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, False, weight_keys='params')
        elif pretrained:
            load_pretrained_network(self, default_model_urls[model_name], True, weight_keys='params')
        self.eps = 1e-08
        self.crops = num_crop
        if 'gfiqa' in model_name:
            self.face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', use_parse=True, model_rootpath=DEFAULT_CACHE_DIR)

    def _init_linear(self, m):
        for module in m.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                nn.init.constant_(module.bias.data, 0)

    def preprocess(self, x):
        x = (x - self.default_mean) / self.default_std
        return x

    def fix_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

    def get_swin_feature(self, model, x):
        b, c, h, w = x.shape
        x = model.patch_embed(x)
        if model.absolute_pos_embed is not None:
            x = x + model.absolute_pos_embed
        x = model.pos_drop(x)
        feat_list = []
        for ly in model.layers:
            x = ly(x)
            feat_list.append(x)
        h, w = h // 8, w // 8
        for idx, f in enumerate(feat_list):
            feat_list[idx] = f.transpose(1, 2).reshape(b, f.shape[-1], h, w)
            if idx < len(feat_list) - 2:
                h, w = h // 2, w // 2
        return feat_list

    def dist_func(self, x, y, eps=1e-12):
        return torch.sqrt((x - y) ** 2 + eps)

    def forward_cross_attention(self, x, y=None):
        if not self.training:
            if 'swin' in self.semantic_model_name:
                x = TF.resize(x, [384, 384], antialias=True)
            elif self.test_img_size is not None:
                x = TF.resize(x, self.test_img_size, antialias=True)
        x = self.preprocess(x)
        if self.use_ref:
            y = self.preprocess(y)
        if 'swin' in self.semantic_model_name:
            dist_feat_list = self.get_swin_feature(self.semantic_model, x)
            if self.use_ref:
                ref_feat_list = self.get_swin_feature(self.semantic_model, y)
            self.semantic_model.eval()
        elif 'clip' in self.semantic_model_name:
            visual_model = self.semantic_model[0].visual
            dist_feat_list = visual_model.forward_features(x)
            if self.use_ref:
                ref_feat_list = visual_model.forward_features(y)
        else:
            dist_feat_list = self.semantic_model(x)
            if self.use_ref:
                ref_feat_list = self.semantic_model(y)
            self.fix_bn(self.semantic_model)
            self.semantic_model.eval()
        start_level = 0
        end_level = len(dist_feat_list)
        b, c, th, tw = dist_feat_list[end_level - 1].shape
        pos_emb = torch.cat((self.h_emb.repeat(1, 1, 1, self.w_emb.shape[3]), self.w_emb.repeat(1, 1, self.h_emb.shape[2], 1)), dim=1)
        token_feat_list = []
        for i in reversed(range(start_level, end_level)):
            tmp_dist_feat = dist_feat_list[i]
            if self.use_ref:
                tmp_ref_feat = ref_feat_list[i]
                diff = self.dist_func(tmp_dist_feat, tmp_ref_feat)
                tmp_feat = torch.cat([tmp_dist_feat, tmp_ref_feat, diff], dim=1)
                weight = self.weight_pool[i](diff)
                tmp_feat = tmp_feat * weight
            else:
                tmp_feat = self.weight_pool[i](tmp_dist_feat)
            if tmp_feat.shape[2] > th and tmp_feat.shape[3] > tw:
                tmp_feat = F.adaptive_avg_pool2d(tmp_feat, (th, tw))
            tmp_pos_emb = F.interpolate(pos_emb, size=tmp_feat.shape[2:], mode='bicubic', align_corners=False)
            tmp_pos_emb = tmp_pos_emb.flatten(2).permute(2, 0, 1)
            tmp_feat = self.dim_reduce[i](tmp_feat)
            tmp_feat = tmp_feat.flatten(2).permute(2, 0, 1)
            tmp_feat = tmp_feat + tmp_pos_emb
            tmp_feat = self.sa_attn_blks[i](tmp_feat)
            token_feat_list.append(tmp_feat)
        query = token_feat_list[0]
        query_list = [query]
        for i in range(len(token_feat_list) - 1):
            key_value = token_feat_list[i + 1]
            query = self.attn_blks[i](query, key_value)
            query_list.append(query)
        final_feat = self.attn_pool(query)
        out_score = self.score_linear(final_feat.mean(dim=0))
        return out_score

    def preprocess_face(self, x):
        warnings.warn(f'The faces will be aligned, cropped and resized to 512x512 with facexlib. Currently, this metric does not support batch size > 1 and gradient backpropagation.', UserWarning)
        device = x.device
        assert x.shape[0] == 1, f'Only support batch size 1, but got {x.shape[0]}'
        self.face_helper.clean_all()
        self.face_helper.input_img = x[0].permute(1, 2, 0).cpu().numpy() * 255
        self.face_helper.input_img = self.face_helper.input_img[..., ::-1]
        if self.face_helper.get_face_landmarks_5(only_center_face=True, eye_dist_threshold=5) > 0:
            self.face_helper.align_warp_face()
            x = self.face_helper.cropped_faces[0]
            x = torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            return x
        else:
            assert False, f'No face detected in the input image.'

    def forward(self, x, y=None, return_mos=True, return_dist=False):
        if self.use_ref:
            assert y is not None, f'Please input y when use reference is True.'
        else:
            y = None
        if 'gfiqa' in self.model_name:
            if self.align_crop_face:
                x = self.preprocess_face(x)
        if self.crops > 1 and not self.training:
            bsz = x.shape[0]
            if y is not None:
                x, y = uniform_crop([x, y], self.crop_size, self.crops)
            else:
                x = uniform_crop([x], self.crop_size, self.crops)[0]
            score = self.forward_cross_attention(x, y)
            score = score.reshape(bsz, self.crops, self.num_class)
            score = score.mean(dim=1)
        else:
            score = self.forward_cross_attention(x, y)
        mos = dist_to_mos(score)
        return_list = []
        if return_mos:
            return_list.append(mos)
        if return_dist:
            return_list.append(score)
        if len(return_list) > 1:
            return return_list
        else:
            return return_list[0]


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_val):
        x = tensor_val
        mask = torch.gt(torch.zeros(x.shape), 0)[:, 0, :, :]
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class TReS(nn.Module):

    def __init__(self, network='resnet50', train_dataset='koniq', nheadt=16, num_encoder_layerst=2, dim_feedforwardt=64, test_sample=50, default_mean=[0.485, 0.456, 0.406], default_std=[0.229, 0.224, 0.225], pretrained=True, pretrained_model_path=None):
        super().__init__()
        self.test_sample = test_sample
        self.L2pooling_l1 = L2pooling(channels=256)
        self.L2pooling_l2 = L2pooling(channels=512)
        self.L2pooling_l3 = L2pooling(channels=1024)
        self.L2pooling_l4 = L2pooling(channels=2048)
        if network == 'resnet50':
            dim_modelt = 3840
            self.model = models.resnet50(weights='IMAGENET1K_V1')
        elif network == 'resnet34':
            self.model = models.resnet34(weights='IMAGENET1K_V1')
            dim_modelt = 960
            self.L2pooling_l1 = L2pooling(channels=64)
            self.L2pooling_l2 = L2pooling(channels=128)
            self.L2pooling_l3 = L2pooling(channels=256)
            self.L2pooling_l4 = L2pooling(channels=512)
        elif network == 'resnet18':
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            dim_modelt = 960
            self.L2pooling_l1 = L2pooling(channels=64)
            self.L2pooling_l2 = L2pooling(channels=128)
            self.L2pooling_l3 = L2pooling(channels=256)
            self.L2pooling_l4 = L2pooling(channels=512)
        self.dim_modelt = dim_modelt
        nheadt = nheadt
        num_encoder_layerst = num_encoder_layerst
        dim_feedforwardt = dim_feedforwardt
        ddropout = 0.5
        normalize = True
        self.transformer = Transformer(d_model=dim_modelt, nhead=nheadt, num_encoder_layers=num_encoder_layerst, dim_feedforward=dim_feedforwardt, normalize_before=normalize, dropout=ddropout)
        self.position_embedding = PositionEmbeddingSine(dim_modelt // 2, normalize=True)
        self.fc2 = nn.Linear(dim_modelt, self.model.fc.in_features)
        self.fc = nn.Linear(self.model.fc.in_features * 2, 1)
        self.ReLU = nn.ReLU()
        self.avg7 = nn.AvgPool2d((7, 7))
        self.avg8 = nn.AvgPool2d((8, 8))
        self.avg4 = nn.AvgPool2d((4, 4))
        self.avg2 = nn.AvgPool2d((2, 2))
        self.drop2d = nn.Dropout(p=0.1)
        self.consistency = nn.L1Loss()
        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True)
        elif pretrained:
            load_pretrained_network(self, default_model_urls[train_dataset], True)

    def forward_backbone(self, model, x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        l1 = x
        x = model.layer2(x)
        l2 = x
        x = model.layer3(x)
        l3 = x
        x = model.layer4(x)
        l4 = x
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.fc(x)
        return x, l1, l2, l3, l4

    def forward(self, x):
        x = (x - self.default_mean) / self.default_std
        bsz = x.shape[0]
        if self.training:
            x = random_crop(x, 224, 1)
            num_patches = 1
        else:
            x = uniform_crop(x, 224, self.test_sample)
            num_patches = self.test_sample
        self.pos_enc_1 = self.position_embedding(torch.ones(1, self.dim_modelt, 7, 7))
        self.pos_enc = self.pos_enc_1.repeat(x.shape[0], 1, 1, 1).contiguous()
        out, layer1, layer2, layer3, layer4 = self.forward_backbone(self.model, x)
        layer1_t = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(layer1, dim=1, p=2))))
        layer2_t = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(layer2, dim=1, p=2))))
        layer3_t = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(layer3, dim=1, p=2))))
        layer4_t = self.drop2d(self.L2pooling_l4(F.normalize(layer4, dim=1, p=2)))
        layers = torch.cat((layer1_t, layer2_t, layer3_t, layer4_t), dim=1)
        out_t_c = self.transformer(layers, self.pos_enc)
        out_t_o = torch.flatten(self.avg7(out_t_c), start_dim=1)
        out_t_o = self.fc2(out_t_o)
        layer4_o = self.avg7(layer4)
        layer4_o = torch.flatten(layer4_o, start_dim=1)
        predictionQA = self.fc(torch.flatten(torch.cat((out_t_o, layer4_o), dim=1), start_dim=1))
        fout, flayer1, flayer2, flayer3, flayer4 = self.forward_backbone(self.model, torch.flip(x, [3]))
        flayer1_t = self.avg8(self.L2pooling_l1(F.normalize(flayer1, dim=1, p=2)))
        flayer2_t = self.avg4(self.L2pooling_l2(F.normalize(flayer2, dim=1, p=2)))
        flayer3_t = self.avg2(self.L2pooling_l3(F.normalize(flayer3, dim=1, p=2)))
        flayer4_t = self.L2pooling_l4(F.normalize(flayer4, dim=1, p=2))
        flayers = torch.cat((flayer1_t, flayer2_t, flayer3_t, flayer4_t), dim=1)
        fout_t_c = self.transformer(flayers, self.pos_enc)
        fout_t_o = torch.flatten(self.avg7(fout_t_c), start_dim=1)
        fout_t_o = self.fc2(fout_t_o)
        flayer4_o = self.avg7(flayer4)
        flayer4_o = torch.flatten(flayer4_o, start_dim=1)
        fpredictionQA = self.fc(torch.flatten(torch.cat((fout_t_o, flayer4_o), dim=1), start_dim=1))
        consistloss1 = self.consistency(out_t_c, fout_t_c.detach())
        consistloss2 = self.consistency(layer4, flayer4.detach())
        consistloss = 1 * (consistloss1 + consistloss2)
        predictionQA = predictionQA.reshape(bsz, num_patches, 1)
        predictionQA = predictionQA.mean(dim=1)
        if self.training:
            return predictionQA, consistloss
        else:
            return predictionQA


class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class BCNN(nn.Module):

    def __init__(self, thresh=1e-08, is_vec=True, input_dim=512):
        super(BCNN, self).__init__()
        self.thresh = thresh
        self.is_vec = is_vec
        self.output_dim = input_dim * input_dim

    def _bilinearpool(self, x):
        batchSize, dim, h, w = x.data.shape
        x = x.reshape(batchSize, dim, h * w)
        x = 1.0 / (h * w) * x.bmm(x.transpose(1, 2))
        return x

    def _signed_sqrt(self, x):
        x = torch.mul(x.sign(), torch.sqrt(x.abs() + self.thresh))
        return x

    def _l2norm(self, x):
        x = nn.functional.normalize(x)
        return x

    def forward(self, x):
        x = self._bilinearpool(x)
        x = self._signed_sqrt(x)
        if self.is_vec:
            x = x.view(x.size(0), -1)
        x = self._l2norm(x)
        return x


class UNIQUE(nn.Module):
    """Full UNIQUE network.
    Args:
        - default_mean (list): Default mean value.
        - default_std (list): Default std value.

        """

    def __init__(self):
        super(UNIQUE, self).__init__()
        self.backbone = torchvision.models.resnet34(pretrained=True)
        outdim = 2
        self.representation = BCNN()
        self.fc = nn.Linear(512 * 512, outdim)
        self.preprocess = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        pretrained_model_path = default_model_urls['mix']
        load_pretrained_network(self, pretrained_model_path, True)

    def forward(self, x):
        """Compute IQA using UNIQUE model.

        Args:
            X: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of UNIQUE model.

        """
        x = self.preprocess(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.representation(x)
        x = self.fc(x)
        mean = x[:, 0]
        return mean


class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """

    def __init__(self, Ch, h, window):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                    1. An integer of window size, which assigns all attention heads with the same window size in ConvRelPosEnc.
                    2. A dict mapping window size to #attention head splits (e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                       It will apply different window size to the attention head splits.
        """
        super().__init__()
        if isinstance(window, int):
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()
        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(cur_head_split * Ch, cur_head_split * Ch, kernel_size=(cur_window, cur_window), padding=(padding_size, padding_size), dilation=(dilation, dilation), groups=cur_head_split * Ch)
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [(x * Ch) for x in self.head_splits]

    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size
        assert N == 1 + H * W or N == 2 + H * W
        diff = N - H * W
        q_img = q[:, :, diff:, :]
        v_img = v[:, :, diff:, :]
        v_img = rearrange(v_img, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)
        EV_hat_img = q_img * conv_v_img
        zero = torch.zeros((B, h, diff, Ch), dtype=q.dtype, layout=q.layout, device=q.device)
        EV_hat = torch.cat((zero, EV_hat_img), dim=2)
        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, shared_crpe=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)
        factor_att = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)
        crpe = self.crpe(q, v, size=size)
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ConvPosEnc(nn.Module):
    """ Convolutional Position Encoding. 
        Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == 1 + H * W or N == 2 + H * W
        diff = N - H * W
        other_token, img_tokens = x[:, :diff], x[:, diff:]
        feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((other_token, x), dim=1)
        return x


class SerialBlock(nn.Module):
    """ Serial block class.
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module. """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, shared_cpe=None, shared_crpe=None):
        super().__init__()
        self.cpe = shared_cpe
        self.norm1 = norm_layer(dim)
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size):
        x = self.cpe(x, size)
        cur = self.norm1(x)
        cur = self.factoratt_crpe(cur, size)
        x = x + self.drop_path(cur)
        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)
        return x


class ParallelBlock(nn.Module):
    """ Parallel block class. """

    def __init__(self, dims, num_heads, mlp_ratios=[], qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, shared_cpes=None, shared_crpes=None, connect_type='neighbor'):
        super().__init__()
        self.connect_type = connect_type
        if self.connect_type == 'dynamic':
            self.alpha1 = nn.Parameter(torch.zeros(1) + 0.05)
            self.alpha2 = nn.Parameter(torch.zeros(1) + 0.05)
            self.alpha3 = nn.Parameter(torch.zeros(1) + 0.05)
            self.alpha4 = nn.Parameter(torch.zeros(1) + 0.05)
            self.alpha5 = nn.Parameter(torch.zeros(1) + 0.05)
            self.alpha6 = nn.Parameter(torch.zeros(1) + 0.05)
        self.cpes = shared_cpes
        self.norm12 = norm_layer(dims[1])
        self.norm13 = norm_layer(dims[2])
        self.norm14 = norm_layer(dims[3])
        self.factoratt_crpe2 = FactorAtt_ConvRelPosEnc(dims[1], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpes[1])
        self.factoratt_crpe3 = FactorAtt_ConvRelPosEnc(dims[2], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpes[2])
        self.factoratt_crpe4 = FactorAtt_ConvRelPosEnc(dims[3], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpes[3])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm22 = norm_layer(dims[1])
        self.norm23 = norm_layer(dims[2])
        self.norm24 = norm_layer(dims[3])
        assert dims[1] == dims[2] == dims[3]
        assert mlp_ratios[1] == mlp_ratios[2] == mlp_ratios[3]
        mlp_hidden_dim = int(dims[1] * mlp_ratios[1])
        self.mlp2 = self.mlp3 = self.mlp4 = Mlp(in_features=dims[1], hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def upsample(self, x, output_size, size):
        """ Feature map up-sampling. """
        return self.interpolate(x, output_size=output_size, size=size)

    def downsample(self, x, output_size, size):
        """ Feature map down-sampling. """
        return self.interpolate(x, output_size=output_size, size=size)

    def interpolate(self, x, output_size, size):
        """ Feature map interpolation. """
        B, N, C = x.shape
        H, W = size
        assert N == 1 + H * W or 2 + H * W
        diff = N - H * W
        other_token = x[:, :diff, :]
        img_tokens = x[:, diff:, :]
        img_tokens = img_tokens.transpose(1, 2).reshape(B, C, H, W)
        img_tokens = F.interpolate(img_tokens, size=output_size, mode='bilinear')
        img_tokens = img_tokens.reshape(B, C, -1).transpose(1, 2)
        out = torch.cat((other_token, img_tokens), dim=1)
        return out

    def forward(self, x1, x2, x3, x4, sizes):
        _, (H2, W2), (H3, W3), (H4, W4) = sizes
        x2 = self.cpes[1](x2, size=(H2, W2))
        x3 = self.cpes[2](x3, size=(H3, W3))
        x4 = self.cpes[3](x4, size=(H4, W4))
        cur2 = self.norm12(x2)
        cur3 = self.norm13(x3)
        cur4 = self.norm14(x4)
        cur2 = self.factoratt_crpe2(cur2, size=(H2, W2))
        cur3 = self.factoratt_crpe3(cur3, size=(H3, W3))
        cur4 = self.factoratt_crpe4(cur4, size=(H4, W4))
        upsample3_2 = self.upsample(cur3, output_size=(H2, W2), size=(H3, W3))
        upsample4_3 = self.upsample(cur4, output_size=(H3, W3), size=(H4, W4))
        upsample4_2 = self.upsample(cur4, output_size=(H2, W2), size=(H4, W4))
        downsample2_3 = self.downsample(cur2, output_size=(H3, W3), size=(H2, W2))
        downsample3_4 = self.downsample(cur3, output_size=(H4, W4), size=(H3, W3))
        downsample2_4 = self.downsample(cur2, output_size=(H4, W4), size=(H2, W2))
        if self.connect_type == 'neighbor':
            cur2 = cur2 + upsample3_2
            cur3 = cur3 + upsample4_3 + downsample2_3
            cur4 = cur4 + downsample3_4
        elif self.connect_type == 'dense':
            cur2 = cur2 + upsample3_2 + upsample4_2
            cur3 = cur3 + upsample4_3 + downsample2_3
            cur4 = cur4 + downsample3_4 + downsample2_4
        elif self.connect_type == 'direct':
            cur2 = cur2
            cur3 = cur3
            cur4 = cur4
        elif self.connect_type == 'dynamic':
            cur2 = cur2 + self.alpha1 * upsample3_2 + self.alpha2 * upsample4_2
            cur3 = cur3 + self.alpha3 * upsample4_3 + self.alpha4 * downsample2_3
            cur4 = cur4 + self.alpha5 * downsample3_4 + self.alpha6 * downsample2_4
        del upsample3_2, upsample4_3, upsample4_2, downsample2_3, downsample2_4, downsample3_4
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)
        del cur2, cur3, cur4
        cur2 = self.norm22(x2)
        cur3 = self.norm23(x3)
        cur4 = self.norm24(x4)
        cur2 = self.mlp2(cur2)
        cur3 = self.mlp3(cur3)
        cur4 = self.mlp4(cur4)
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)
        return x1, x2, x3, x4


@torch.no_grad()
def build_historgram(img):
    b, _, _, _ = img.shape
    r_his = torch.histc(img[0][0], 64, min=0.0, max=1.0)
    g_his = torch.histc(img[0][1], 64, min=0.0, max=1.0)
    b_his = torch.histc(img[0][2], 64, min=0.0, max=1.0)
    historgram = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)
    for i in range(1, b):
        r_his = torch.histc(img[i][0], 64, min=0.0, max=1.0)
        g_his = torch.histc(img[i][1], 64, min=0.0, max=1.0)
        b_his = torch.histc(img[i][2], 64, min=0.0, max=1.0)
        historgram_temp = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)
        historgram = torch.cat((historgram, historgram_temp), dim=0)
    return historgram


def padding_img(img):
    b, c, h, w = img.shape
    h_out = math.ceil(h / 32) * 32
    w_out = math.ceil(w / 32) * 32
    left_pad = (w_out - w) // 2
    right_pad = w_out - w - left_pad
    top_pad = (h_out - h) // 2
    bottom_pad = h_out - h - top_pad
    img = nn.ZeroPad2d((left_pad, right_pad, top_pad, bottom_pad))(img)
    return img


def preprocessing(d_img_org):
    d_img_org = padding_img(d_img_org)
    x_his = build_historgram(d_img_org)
    return d_img_org, x_his


class URanker(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, num_classes=1, embed_dims=[152, 320, 320, 320], serial_depths=[2, 2, 2, 2], parallel_depth=6, num_heads=8, mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-06), return_interm_layers=False, out_features=None, crpe_window={(3): 2, (5): 3, (7): 3}, add_historgram=True, his_channel=192, connect_type='dynamic', pretrained=True, pretrained_model_path=None, **kwargs):
        super().__init__()
        self.return_interm_layers = return_interm_layers
        self.out_features = out_features
        self.num_classes = num_classes
        self.add_historgram = add_historgram
        self.connect_type = connect_type
        if self.add_historgram:
            self.historgram_embed1 = nn.Linear(his_channel, embed_dims[0])
            self.historgram_embed2 = nn.Linear(his_channel, embed_dims[1])
            self.historgram_embed3 = nn.Linear(his_channel, embed_dims[2])
            self.historgram_embed4 = nn.Linear(his_channel, embed_dims[3])
        self.patch_embed1 = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))
        self.cpe1 = ConvPosEnc(dim=embed_dims[0], k=3)
        self.cpe2 = ConvPosEnc(dim=embed_dims[1], k=3)
        self.cpe3 = ConvPosEnc(dim=embed_dims[2], k=3)
        self.cpe4 = ConvPosEnc(dim=embed_dims[3], k=3)
        self.crpe1 = ConvRelPosEnc(Ch=embed_dims[0] // num_heads, h=num_heads, window=crpe_window)
        self.crpe2 = ConvRelPosEnc(Ch=embed_dims[1] // num_heads, h=num_heads, window=crpe_window)
        self.crpe3 = ConvRelPosEnc(Ch=embed_dims[2] // num_heads, h=num_heads, window=crpe_window)
        self.crpe4 = ConvRelPosEnc(Ch=embed_dims[3] // num_heads, h=num_heads, window=crpe_window)
        dpr = drop_path_rate
        self.serial_blocks1 = nn.ModuleList([SerialBlock(dim=embed_dims[0], num_heads=num_heads, mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe1, shared_crpe=self.crpe1) for _ in range(serial_depths[0])])
        self.serial_blocks2 = nn.ModuleList([SerialBlock(dim=embed_dims[1], num_heads=num_heads, mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe2, shared_crpe=self.crpe2) for _ in range(serial_depths[1])])
        self.serial_blocks3 = nn.ModuleList([SerialBlock(dim=embed_dims[2], num_heads=num_heads, mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe3, shared_crpe=self.crpe3) for _ in range(serial_depths[2])])
        self.serial_blocks4 = nn.ModuleList([SerialBlock(dim=embed_dims[3], num_heads=num_heads, mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe4, shared_crpe=self.crpe4) for _ in range(serial_depths[3])])
        self.parallel_depth = parallel_depth
        if self.parallel_depth > 0:
            self.parallel_blocks = nn.ModuleList([ParallelBlock(dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpes=[self.cpe1, self.cpe2, self.cpe3, self.cpe4], shared_crpes=[self.crpe1, self.crpe2, self.crpe3, self.crpe4], connect_type=self.connect_type) for _ in range(parallel_depth)])
        if not self.return_interm_layers:
            self.norm1 = norm_layer(embed_dims[0])
            self.norm2 = norm_layer(embed_dims[1])
            self.norm3 = norm_layer(embed_dims[2])
            self.norm4 = norm_layer(embed_dims[3])
            if self.parallel_depth > 0:
                assert embed_dims[1] == embed_dims[2] == embed_dims[3]
                self.head2 = nn.Linear(embed_dims[3], num_classes)
                self.head3 = nn.Linear(embed_dims[3], num_classes)
                self.head4 = nn.Linear(embed_dims[3], num_classes)
            else:
                self.head2 = nn.Linear(embed_dims[3], num_classes)
                self.head3 = nn.Linear(embed_dims[3], num_classes)
                self.head4 = nn.Linear(embed_dims[3], num_classes)
        trunc_normal_(self.cls_token1, std=0.02)
        trunc_normal_(self.cls_token2, std=0.02)
        trunc_normal_(self.cls_token3, std=0.02)
        trunc_normal_(self.cls_token4, std=0.02)
        self.apply(self._init_weights)
        if pretrained and pretrained_model_path is None:
            load_pretrained_network(self, default_model_urls['uranker'])
        elif pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, weight_keys='params')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token1', 'cls_token2', 'cls_token3', 'cls_token4'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def insert_cls(self, x, cls_token):
        """ Insert CLS token. """
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

    def insert_his(self, x, his_token):
        x = torch.cat((his_token, x), dim=1)
        return x

    def remove_token(self, x):
        """ Remove CLS token. """
        if self.add_historgram:
            return x[:, 2:, :]
        else:
            return x[:, 1:, :]

    def forward_features(self, x0, x_his):
        B = x0.shape[0]
        x1, (H1, W1) = self.patch_embed1(x0)
        if self.add_historgram:
            x1 = self.insert_his(x1, self.historgram_embed1(x_his))
        x1 = self.insert_cls(x1, self.cls_token1)
        for blk in self.serial_blocks1:
            x1 = blk(x1, size=(H1, W1))
        x1_nocls = self.remove_token(x1)
        x1_nocls = x1_nocls.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x2, (H2, W2) = self.patch_embed2(x1_nocls)
        if self.add_historgram:
            x2 = self.insert_his(x2, self.historgram_embed2(x_his))
        x2 = self.insert_cls(x2, self.cls_token2)
        for blk in self.serial_blocks2:
            x2 = blk(x2, size=(H2, W2))
        x2_nocls = self.remove_token(x2)
        x2_nocls = x2_nocls.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        x3, (H3, W3) = self.patch_embed3(x2_nocls)
        if self.add_historgram:
            x3 = self.insert_his(x3, self.historgram_embed3(x_his))
        x3 = self.insert_cls(x3, self.cls_token3)
        for blk in self.serial_blocks3:
            x3 = blk(x3, size=(H3, W3))
        x3_nocls = self.remove_token(x3)
        x3_nocls = x3_nocls.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        x4, (H4, W4) = self.patch_embed4(x3_nocls)
        if self.add_historgram:
            x4 = self.insert_his(x4, self.historgram_embed4(x_his))
        x4 = self.insert_cls(x4, self.cls_token4)
        for blk in self.serial_blocks4:
            x4 = blk(x4, size=(H4, W4))
        x4_nocls = self.remove_token(x4)
        x4_nocls = x4_nocls.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        if self.parallel_depth == 0:
            if self.return_interm_layers:
                feat_out = {}
                if 'x1_nocls' in self.out_features:
                    feat_out['x1_nocls'] = x1_nocls
                if 'x2_nocls' in self.out_features:
                    feat_out['x2_nocls'] = x2_nocls
                if 'x3_nocls' in self.out_features:
                    feat_out['x3_nocls'] = x3_nocls
                if 'x4_nocls' in self.out_features:
                    feat_out['x4_nocls'] = x4_nocls
                return feat_out
            else:
                x4 = self.norm4(x4)
                x4_cls = x4[:, 0]
                return x4_cls
        for blk in self.parallel_blocks:
            x1, x2, x3, x4 = blk(x1, x2, x3, x4, sizes=[(H1, W1), (H2, W2), (H3, W3), (H4, W4)])
        if self.return_interm_layers:
            feat_out = {}
            if 'x1_nocls' in self.out_features:
                x1_nocls = self.remove_token(x1)
                x1_nocls = x1_nocls.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x1_nocls'] = x1_nocls
            if 'x2_nocls' in self.out_features:
                x2_nocls = self.remove_token(x2)
                x2_nocls = x2_nocls.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x2_nocls'] = x2_nocls
            if 'x3_nocls' in self.out_features:
                x3_nocls = self.remove_token(x3)
                x3_nocls = x3_nocls.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x3_nocls'] = x3_nocls
            if 'x4_nocls' in self.out_features:
                x4_nocls = self.remove_token(x4)
                x4_nocls = x4_nocls.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x4_nocls'] = x4_nocls
            return feat_out
        else:
            x2 = self.norm2(x2)
            x3 = self.norm3(x3)
            x4 = self.norm4(x4)
            x2_cls = x2[:, :1]
            x3_cls = x3[:, :1]
            x4_cls = x4[:, :1]
            return x2_cls, x3_cls, x4_cls

    def forward(self, x):
        x, x_his = preprocessing(x)
        if self.return_interm_layers:
            return self.forward_features(x, x_his)
        else:
            x2, x3, x4 = self.forward_features(x, x_his)
            pred2 = self.head2(x2)
            pred3 = self.head3(x3)
            pred4 = self.head4(x4)
            x = (pred2 + pred3 + pred4) / 3.0
            return x.squeeze(1)


def corrDn(image, filt, step=1, channels=1):
    """Compute correlation of image with FILT, followed by downsampling.
    Args:
        image: A tensor. Shape :math:`(N, C, H, W)`.
        filt: A filter.
        step: Downsampling factors.
        channels: Number of channels.
    """
    filt_ = torch.from_numpy(filt).float().unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    p = (filt_.shape[2] - 1) // 2
    image = F.pad(image, (p, p, p, p), 'reflect')
    img = F.conv2d(image, filt_, stride=step, padding=0, groups=channels)
    return img


def sp5_filters():
    """Define spatial filters.
    """
    filters = {}
    filters['harmonics'] = np.array([1, 3, 5])
    filters['mtx'] = np.array([[0.3333, 0.2887, 0.1667, 0.0, -0.1667, -0.2887], [0.0, 0.1667, 0.2887, 0.3333, 0.2887, 0.1667], [0.3333, -0.0, -0.3333, -0.0, 0.3333, -0.0], [0.0, 0.3333, 0.0, -0.3333, 0.0, 0.3333], [0.3333, -0.2887, 0.1667, -0.0, -0.1667, 0.2887], [-0.0, 0.1667, -0.2887, 0.3333, -0.2887, 0.1667]])
    filters['hi0filt'] = np.array([[-0.00033429, -0.00113093, -0.00171484, -0.00133542, -0.00080639, -0.00133542, -0.00171484, -0.00113093, -0.00033429], [-0.00113093, -0.00350017, -0.00243812, 0.00631653, 0.01261227, 0.00631653, -0.00243812, -0.00350017, -0.00113093], [-0.00171484, -0.00243812, -0.00290081, -0.00673482, -0.00981051, -0.00673482, -0.00290081, -0.00243812, -0.00171484], [-0.00133542, 0.00631653, -0.00673482, -0.07027679, -0.11435863, -0.07027679, -0.00673482, 0.00631653, -0.00133542], [-0.00080639, 0.01261227, -0.00981051, -0.11435863, 0.813802, -0.11435863, -0.00981051, 0.01261227, -0.00080639], [-0.00133542, 0.00631653, -0.00673482, -0.07027679, -0.11435863, -0.07027679, -0.00673482, 0.00631653, -0.00133542], [-0.00171484, -0.00243812, -0.00290081, -0.00673482, -0.00981051, -0.00673482, -0.00290081, -0.00243812, -0.00171484], [-0.00113093, -0.00350017, -0.00243812, 0.00631653, 0.01261227, 0.00631653, -0.00243812, -0.00350017, -0.00113093], [-0.00033429, -0.00113093, -0.00171484, -0.00133542, -0.00080639, -0.00133542, -0.00171484, -0.00113093, -0.00033429]])
    filters['lo0filt'] = np.array([[0.00341614, -0.01551246, -0.03848215, -0.01551246, 0.00341614], [-0.01551246, 0.05586982, 0.1592557, 0.05586982, -0.01551246], [-0.03848215, 0.1592557, 0.40304148, 0.1592557, -0.03848215], [-0.01551246, 0.05586982, 0.1592557, 0.05586982, -0.01551246], [0.00341614, -0.01551246, -0.03848215, -0.01551246, 0.00341614]])
    filters['lofilt'] = 2 * np.array([[0.00085404, -0.00244917, -0.00387812, -0.00944432, -0.00962054, -0.00944432, -0.00387812, -0.00244917, 0.00085404], [-0.00244917, -0.00523281, -0.00661117, 0.004106, 0.01002988, 0.004106, -0.00661117, -0.00523281, -0.00244917], [-0.00387812, -0.00661117, 0.01396746, 0.03277038, 0.03981393, 0.03277038, 0.01396746, -0.00661117, -0.00387812], [-0.00944432, 0.004106, 0.03277038, 0.06426333, 0.08169618, 0.06426333, 0.03277038, 0.004106, -0.00944432], [-0.00962054, 0.01002988, 0.03981393, 0.08169618, 0.1009654, 0.08169618, 0.03981393, 0.01002988, -0.00962054], [-0.00944432, 0.004106, 0.03277038, 0.06426333, 0.08169618, 0.06426333, 0.03277038, 0.004106, -0.00944432], [-0.00387812, -0.00661117, 0.01396746, 0.03277038, 0.03981393, 0.03277038, 0.01396746, -0.00661117, -0.00387812], [-0.00244917, -0.00523281, -0.00661117, 0.004106, 0.01002988, 0.004106, -0.00661117, -0.00523281, -0.00244917], [0.00085404, -0.00244917, -0.00387812, -0.00944432, -0.00962054, -0.00944432, -0.00387812, -0.00244917, 0.00085404]])
    filters['bfilts'] = np.array([[0.00277643, 0.00496194, 0.01026699, 0.01455399, 0.01026699, 0.00496194, 0.00277643, -0.00986904, -0.00893064, 0.01189859, 0.02755155, 0.01189859, -0.00893064, -0.00986904, -0.01021852, -0.03075356, -0.08226445, -0.11732297, -0.08226445, -0.03075356, -0.01021852, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01021852, 0.03075356, 0.08226445, 0.11732297, 0.08226445, 0.03075356, 0.01021852, 0.00986904, 0.00893064, -0.01189859, -0.02755155, -0.01189859, 0.00893064, 0.00986904, -0.00277643, -0.00496194, -0.01026699, -0.01455399, -0.01026699, -0.00496194, -0.00277643], [-0.00343249, -0.00640815, -0.00073141, 0.01124321, 0.00182078, 0.00285723, 0.01166982, -0.00358461, -0.01977507, -0.04084211, -0.00228219, 0.03930573, 0.01161195, 0.00128, 0.01047717, 0.01486305, -0.04819057, -0.1222723, -0.05394139, 0.00853965, -0.00459034, 0.00790407, 0.04435647, 0.09454202, -0.0, -0.09454202, -0.04435647, -0.00790407, 0.00459034, -0.00853965, 0.05394139, 0.1222723, 0.04819057, -0.01486305, -0.01047717, -0.00128, -0.01161195, -0.03930573, 0.00228219, 0.04084211, 0.01977507, 0.00358461, -0.01166982, -0.00285723, -0.00182078, -0.01124321, 0.00073141, 0.00640815, 0.00343249], [0.00343249, 0.00358461, -0.01047717, -0.00790407, -0.00459034, 0.00128, 0.01166982, 0.00640815, 0.01977507, -0.01486305, -0.04435647, 0.00853965, 0.01161195, 0.00285723, 0.00073141, 0.04084211, 0.04819057, -0.09454202, -0.05394139, 0.03930573, 0.00182078, -0.01124321, 0.00228219, 0.1222723, -0.0, -0.1222723, -0.00228219, 0.01124321, -0.00182078, -0.03930573, 0.05394139, 0.09454202, -0.04819057, -0.04084211, -0.00073141, -0.00285723, -0.01161195, -0.00853965, 0.04435647, 0.01486305, -0.01977507, -0.00640815, -0.01166982, -0.00128, 0.00459034, 0.00790407, 0.01047717, -0.00358461, -0.00343249], [-0.00277643, 0.00986904, 0.01021852, -0.0, -0.01021852, -0.00986904, 0.00277643, -0.00496194, 0.00893064, 0.03075356, -0.0, -0.03075356, -0.00893064, 0.00496194, -0.01026699, -0.01189859, 0.08226445, -0.0, -0.08226445, 0.01189859, 0.01026699, -0.01455399, -0.02755155, 0.11732297, -0.0, -0.11732297, 0.02755155, 0.01455399, -0.01026699, -0.01189859, 0.08226445, -0.0, -0.08226445, 0.01189859, 0.01026699, -0.00496194, 0.00893064, 0.03075356, -0.0, -0.03075356, -0.00893064, 0.00496194, -0.00277643, 0.00986904, 0.01021852, -0.0, -0.01021852, -0.00986904, 0.00277643], [-0.01166982, -0.00128, 0.00459034, 0.00790407, 0.01047717, -0.00358461, -0.00343249, -0.00285723, -0.01161195, -0.00853965, 0.04435647, 0.01486305, -0.01977507, -0.00640815, -0.00182078, -0.03930573, 0.05394139, 0.09454202, -0.04819057, -0.04084211, -0.00073141, -0.01124321, 0.00228219, 0.1222723, -0.0, -0.1222723, -0.00228219, 0.01124321, 0.00073141, 0.04084211, 0.04819057, -0.09454202, -0.05394139, 0.03930573, 0.00182078, 0.00640815, 0.01977507, -0.01486305, -0.04435647, 0.00853965, 0.01161195, 0.00285723, 0.00343249, 0.00358461, -0.01047717, -0.00790407, -0.00459034, 0.00128, 0.01166982], [-0.01166982, -0.00285723, -0.00182078, -0.01124321, 0.00073141, 0.00640815, 0.00343249, -0.00128, -0.01161195, -0.03930573, 0.00228219, 0.04084211, 0.01977507, 0.00358461, 0.00459034, -0.00853965, 0.05394139, 0.1222723, 0.04819057, -0.01486305, -0.01047717, 0.00790407, 0.04435647, 0.09454202, -0.0, -0.09454202, -0.04435647, -0.00790407, 0.01047717, 0.01486305, -0.04819057, -0.1222723, -0.05394139, 0.00853965, -0.00459034, -0.00358461, -0.01977507, -0.04084211, -0.00228219, 0.03930573, 0.01161195, 0.00128, -0.00343249, -0.00640815, -0.00073141, 0.01124321, 0.00182078, 0.00285723, 0.01166982]]).T
    return filters


def SteerablePyramidSpace(image, height=4, order=5, channels=1):
    """Construct a steerable pyramid on image.
    Args:
        image: A tensor. Shape :math:`(N, C, H, W)`.
        height (int): Number of pyramid levels to build.
        order (int): Number of orientations.
        channels (int): Number of channels.
    """
    num_orientations = order + 1
    filters = sp5_filters()
    hi0 = corrDn(image, filters['hi0filt'], step=1, channels=channels)
    pyr_coeffs = []
    pyr_coeffs.append(hi0)
    lo = corrDn(image, filters['lo0filt'], step=1, channels=channels)
    for _ in range(height):
        bfiltsz = int(np.floor(np.sqrt(filters['bfilts'].shape[0])))
        for b in range(num_orientations):
            filt = filters['bfilts'][:, b].reshape(bfiltsz, bfiltsz).T
            band = corrDn(lo, filt, step=1, channels=channels)
            pyr_coeffs.append(band)
        lo = corrDn(lo, filters['lofilt'], step=2, channels=channels)
    pyr_coeffs.append(lo)
    return pyr_coeffs


class VIF(torch.nn.Module):
    """Image Information and Visual Quality metric
    Args:
        channels (int): Number of channels.
        level (int): Number of levels to build.
        ori (int): Number of orientations.
    Reference:
        Sheikh, Hamid R., and Alan C. Bovik. "Image information and visual quality."
        IEEE Transactions on image processing 15, no. 2 (2006): 430-444.
    """

    def __init__(self, channels=1, level=4, ori=6):
        super(VIF, self).__init__()
        self.ori = ori - 1
        self.level = level
        self.channels = channels
        self.M = 3
        self.subbands = [4, 7, 10, 13, 16, 19, 22, 25]
        self.sigma_nsq = 0.4
        self.tol = 1e-12

    def corrDn_win(self, image, filt, step=1, channels=1, start=[0, 0], end=[0, 0]):
        """Compute correlation of image with FILT using window, followed by downsampling.
        Args:
            image: A tensor. Shape :math:`(N, C, H, W)`.
            filt: A filter.
            step (int): Downsampling factors.
            channels (int): Number of channels.
            start (list): The window over which the convolution occurs.
            end (list): The window over which the convolution occurs.
        """
        filt_ = torch.from_numpy(filt).float().unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        p = (filt_.shape[2] - 1) // 2
        image = F.pad(image, (p, p, p, p), 'reflect')
        img = F.conv2d(image, filt_, stride=1, padding=0, groups=channels)
        img = img[:, :, start[0]:end[0]:step, start[1]:end[1]:step]
        return img

    def vifsub_est_M(self, org, dist):
        """Calculate the parameters of the distortion channel.
        Args:
            org: A reference tensor. Shape :math:`(N, C, H, W)`.
            dist: A distortion tensor. Shape :math:`(N, C, H, W)`.
        """
        g_all = []
        vv_all = []
        for i in range(len(self.subbands)):
            sub = self.subbands[i] - 1
            y = org[sub]
            yn = dist[sub]
            lev = np.ceil((sub - 1) / 6)
            winsize = int(2 ** lev + 1)
            win = np.ones((winsize, winsize))
            newsizeX = int(np.floor(y.shape[2] / self.M) * self.M)
            newsizeY = int(np.floor(y.shape[3] / self.M) * self.M)
            y = y[:, :, :newsizeX, :newsizeY]
            yn = yn[:, :, :newsizeX, :newsizeY]
            winstart = [int(1 * np.floor(self.M / 2)), int(1 * np.floor(self.M / 2))]
            winend = [int(y.shape[2] - np.ceil(self.M / 2)) + 1, int(y.shape[3] - np.ceil(self.M / 2)) + 1]
            mean_x = self.corrDn_win(y, win / winsize ** 2, step=self.M, channels=self.channels, start=winstart, end=winend)
            mean_y = self.corrDn_win(yn, win / winsize ** 2, step=self.M, channels=self.channels, start=winstart, end=winend)
            cov_xy = self.corrDn_win(y * yn, win, step=self.M, channels=self.channels, start=winstart, end=winend) - winsize ** 2 * mean_x * mean_y
            ss_x = self.corrDn_win(y ** 2, win, step=self.M, channels=self.channels, start=winstart, end=winend) - winsize ** 2 * mean_x ** 2
            ss_y = self.corrDn_win(yn ** 2, win, step=self.M, channels=self.channels, start=winstart, end=winend) - winsize ** 2 * mean_y ** 2
            ss_x = F.relu(ss_x)
            ss_y = F.relu(ss_y)
            g = cov_xy / (ss_x + self.tol)
            vv = (ss_y - g * cov_xy) / winsize ** 2
            g = g.masked_fill(ss_x < self.tol, 0)
            vv[ss_x < self.tol] = ss_y[ss_x < self.tol]
            ss_x = ss_x.masked_fill(ss_x < self.tol, 0)
            g = g.masked_fill(ss_y < self.tol, 0)
            vv = vv.masked_fill(ss_y < self.tol, 0)
            vv[g < 0] = ss_y[g < 0]
            g = F.relu(g)
            vv = vv.masked_fill(vv < self.tol, self.tol)
            g_all.append(g)
            vv_all.append(vv)
        return g_all, vv_all

    def refparams_vecgsm(self, org):
        """Calculate the parameters of the reference image.
        Args:
            org: A reference tensor. Shape :math:`(N, C, H, W)`.
        """
        ssarr, l_arr, cu_arr = [], [], []
        for i in range(len(self.subbands)):
            sub = self.subbands[i] - 1
            y = org[sub]
            M = self.M
            newsizeX = int(np.floor(y.shape[2] / M) * M)
            newsizeY = int(np.floor(y.shape[3] / M) * M)
            y = y[:, :, :newsizeX, :newsizeY]
            B, C, H, W = y.shape
            temp = []
            for j in range(M):
                for k in range(M):
                    temp.append(y[:, :, k:H - (M - k) + 1, j:W - (M - j) + 1].reshape(B, C, -1))
            temp = torch.stack(temp, dim=3)
            mcu = torch.mean(temp, dim=2).unsqueeze(2).repeat(1, 1, temp.shape[2], 1)
            cu = torch.matmul((temp - mcu).permute(0, 1, 3, 2), temp - mcu) / temp.shape[2]
            temp = []
            for j in range(M):
                for k in range(M):
                    temp.append(y[:, :, k:H + 1:M, j:W + 1:M].reshape(B, C, -1))
            temp = torch.stack(temp, dim=2)
            ss = torch.matmul(torch.pinverse(cu), temp)
            ss = torch.sum(ss * temp, dim=2) / (M * M)
            ss = ss.reshape(B, C, H // M, W // M)
            v, _ = torch.linalg.eigh(cu, UPLO='U')
            l_arr.append(v)
            ssarr.append(ss)
            cu_arr.append(cu)
        return ssarr, l_arr, cu_arr

    def vif(self, x, y):
        """VIF metric. Order of input is important.
        Args:
            x: A distortion tensor. Shape :math:`(N, C, H, W)`.
            y: A reference tensor. Shape :math:`(N, C, H, W)`.
        """
        x = to_y_channel(x, 255)
        y = to_y_channel(y, 255)
        sp_x = SteerablePyramidSpace(x, height=self.level, order=self.ori, channels=self.channels)[::-1]
        sp_y = SteerablePyramidSpace(y, height=self.level, order=self.ori, channels=self.channels)[::-1]
        g_all, vv_all = self.vifsub_est_M(sp_y, sp_x)
        ss_arr, l_arr, cu_arr = self.refparams_vecgsm(sp_y)
        num, den = [], []
        for i in range(len(self.subbands)):
            sub = self.subbands[i]
            g = g_all[i]
            vv = vv_all[i]
            ss = ss_arr[i]
            lamda = l_arr[i]
            neigvals = lamda.shape[2]
            lev = np.ceil((sub - 1) / 6)
            winsize = 2 ** lev + 1
            offset = (winsize - 1) / 2
            offset = int(np.ceil(offset / self.M))
            _, _, H, W = g.shape
            g = g[:, :, offset:H - offset, offset:W - offset]
            vv = vv[:, :, offset:H - offset, offset:W - offset]
            ss = ss[:, :, offset:H - offset, offset:W - offset]
            temp1 = 0
            temp2 = 0
            for j in range(neigvals):
                cc = lamda[:, :, j].unsqueeze(2).unsqueeze(3)
                temp1 = temp1 + torch.sum(torch.log2(1 + g * g * ss * cc / (vv + self.sigma_nsq)), dim=[2, 3])
                temp2 = temp2 + torch.sum(torch.log2(1 + ss * cc / self.sigma_nsq), dim=[2, 3])
            num.append(temp1.mean(1))
            den.append(temp2.mean(1))
        return torch.stack(num, dim=1).sum(1) / (torch.stack(den, dim=1).sum(1) + 1e-12)

    def forward(self, X, Y):
        """Args:
            x: A distortion tensor. Shape :math:`(N, C, H, W)`.
            y: A reference tensor. Shape :math:`(N, C, H, W)`.
            Order of input is important.
        """
        assert X.shape == Y.shape, 'Input and reference images should have the same shape, but got'
        f"""{X.shape} and {Y.shape}"""
        score = self.vif(X, Y)
        return score


def rgb2lmn(x: 'torch.Tensor') ->torch.Tensor:
    """Convert a batch of RGB images to a batch of LMN images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). LMN colour space.
    """
    weights_rgb_to_lmn = torch.tensor([[0.06, 0.63, 0.27], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]]).t()
    x_lmn = torch.matmul(x.permute(0, 2, 3, 1), weights_rgb_to_lmn).permute(0, 3, 1, 2)
    return x_lmn


def scharr_filter() ->torch.Tensor:
    """Utility function that returns a normalized 3x3 Scharr kernel in X direction
    Returns:
        kernel: Tensor with shape (1, 3, 3)
    """
    return torch.tensor([[[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]]) / 16


def _log_gabor(size: 'Tuple[int, int]', omega_0: 'float', sigma_f: 'float') ->torch.Tensor:
    """Creates log Gabor filter
    Args:
        size: size of the requires log Gabor filter
        omega_0: center frequency of the filter
        sigma_f: bandwidth of the filter

    Returns:
        log Gabor filter
    """
    xx, yy = get_meshgrid(size)
    radius = (xx ** 2 + yy ** 2).sqrt()
    mask = radius <= 0.5
    r = radius * mask
    r = ifftshift(r)
    r[0, 0] = 1
    lg = torch.exp(-(r / omega_0).log().pow(2) / (2 * sigma_f ** 2))
    lg[0, 0] = 0
    return lg


def rgb2xyz(x: 'torch.Tensor') ->torch.Tensor:
    """Convert a batch of RGB images to a batch of XYZ images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). XYZ colour space.
    """
    mask_below = x <= 0.04045
    mask_above = x > 0.04045
    tmp = x / 12.92 * mask_below + torch.pow((x + 0.055) / 1.055, 2.4) * mask_above
    weights_rgb_to_xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.072175], [0.0193339, 0.119192, 0.9503041]])
    x_xyz = torch.matmul(tmp.permute(0, 2, 3, 1), weights_rgb_to_xyz.t()).permute(0, 3, 1, 2)
    return x_xyz


def safe_frac_pow(x: 'torch.Tensor', p) ->torch.Tensor:
    EPS = torch.finfo(x.dtype).eps
    return torch.sign(x) * torch.abs(x + EPS).pow(p)


def xyz2lab(x: 'torch.Tensor', illuminant: 'str'='D50', observer: 'str'='2') ->torch.Tensor:
    """Convert a batch of XYZ images to a batch of LAB images

    Args:
        x: Batch of images with shape (N, 3, H, W). XYZ colour space.
        illuminant: {“A”, “D50”, “D55”, “D65”, “D75”, “E”}, optional. The name of the illuminant.
        observer: {“2”, “10”}, optional. The aperture angle of the observer.

    Returns:
        Batch of images with shape (N, 3, H, W). LAB colour space.
    """
    epsilon = 0.008856
    kappa = 903.3
    illuminants: 'Dict[str, Dict]' = {'A': {'2': (1.098466069456375, 1, 0.3558228003436005), '10': (1.111420406956693, 1, 0.3519978321919493)}, 'D50': {'2': (0.9642119944211994, 1, 0.8251882845188288), '10': (0.9672062750333777, 1, 0.8142801513128616)}, 'D55': {'2': (0.956797052643698, 1, 0.9214805860173273), '10': (0.9579665682254781, 1, 0.9092525159847462)}, 'D65': {'2': (0.95047, 1.0, 1.08883), '10': (0.94809667673716, 1, 1.0730513595166162)}, 'D75': {'2': (0.9497220898840717, 1, 1.226393520724154), '10': (0.9441713925645873, 1, 1.2064272211720228)}, 'E': {'2': (1.0, 1.0, 1.0), '10': (1.0, 1.0, 1.0)}}
    illuminants_to_use = torch.tensor(illuminants[illuminant][observer]).view(1, 3, 1, 1)
    tmp = x / illuminants_to_use
    mask_below = tmp <= epsilon
    mask_above = tmp > epsilon
    tmp = safe_frac_pow(tmp, 1.0 / 3.0) * mask_above + (kappa * tmp + 16.0) / 116.0 * mask_below
    weights_xyz_to_lab = torch.tensor([[0, 116.0, 0], [500.0, -500.0, 0], [0, 200.0, -200.0]])
    bias_xyz_to_lab = torch.tensor([-16.0, 0.0, 0.0]).view(1, 3, 1, 1)
    x_lab = torch.matmul(tmp.permute(0, 2, 3, 1), weights_xyz_to_lab.t()).permute(0, 3, 1, 2) + bias_xyz_to_lab
    return x_lab


def rgb2lab(x: 'torch.Tensor', data_range: 'Union[int, float]'=255) ->torch.Tensor:
    """Convert a batch of RGB images to a batch of LAB images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.
        data_range: dynamic range of the input image.

    Returns:
        Batch of images with shape (N, 3, H, W). LAB colour space.
    """
    return xyz2lab(rgb2xyz(x / float(data_range)))


def sdsp(x: 'torch.Tensor', data_range: 'Union[int, float]'=255, omega_0: 'float'=0.021, sigma_f: 'float'=1.34, sigma_d: 'float'=145.0, sigma_c: 'float'=0.001) ->torch.Tensor:
    """SDSP algorithm for salient region detection from a given image.
    Supports only colour images with RGB channel order.
    Args:
        x: Tensor. Shape :math:`(N, 3, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        omega_0: coefficient for log Gabor filter
        sigma_f: coefficient for log Gabor filter
        sigma_d: coefficient for the central areas, which have a bias towards attention
        sigma_c: coefficient for the warm colors, which have a bias towards attention

    Returns:
        torch.Tensor: Visual saliency map
    """
    x = x / data_range * 255
    size = x.size()
    size_to_use = 256, 256
    x = interpolate(input=x, size=size_to_use, mode='bilinear', align_corners=False)
    x_lab = rgb2lab(x, data_range=255)
    lg = _log_gabor(size_to_use, omega_0, sigma_f).view(1, 1, *size_to_use)
    x_fft = torch.fft.fft2(x_lab)
    x_ifft_real = torch.fft.ifft2(x_fft * lg).real
    s_f = safe_sqrt(x_ifft_real.pow(2).sum(dim=1, keepdim=True))
    coordinates = torch.stack(get_meshgrid(size_to_use), dim=0)
    coordinates = coordinates * size_to_use[0] + 1
    s_d = torch.exp(-torch.sum(coordinates ** 2, dim=0) / sigma_d ** 2).view(1, 1, *size_to_use)
    eps = torch.finfo(x_lab.dtype).eps
    min_x = x_lab.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
    max_x = x_lab.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
    normalized = (x_lab - min_x) / (max_x - min_x + eps)
    norm = normalized[:, 1:].pow(2).sum(dim=1, keepdim=True)
    s_c = 1 - torch.exp(-norm / sigma_c ** 2)
    vs_m = s_f * s_d * s_c
    vs_m = interpolate(vs_m, size[-2:], mode='bilinear', align_corners=True)
    min_vs_m = vs_m.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
    max_vs_m = vs_m.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
    return (vs_m - min_vs_m) / (max_vs_m - min_vs_m + eps)


def vsi(x: 'torch.Tensor', y: 'torch.Tensor', data_range: 'Union[int, float]'=1.0, c1: 'float'=1.27, c2: 'float'=386.0, c3: 'float'=130.0, alpha: 'float'=0.4, beta: 'float'=0.02, omega_0: 'float'=0.021, sigma_f: 'float'=1.34, sigma_d: 'float'=145.0, sigma_c: 'float'=0.001) ->torch.Tensor:
    """Compute Visual Saliency-induced Index for a batch of images.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        c1: coefficient to calculate saliency component of VSI.
        c2: coefficient to calculate gradient component of VSI.
        c3: coefficient to calculate color component of VSI.
        alpha: power for gradient component of VSI.
        beta: power for color component of VSI.
        omega_0: coefficient to get log Gabor filter at SDSP.
        sigma_f: coefficient to get log Gabor filter at SDSP.
        sigma_d: coefficient to get SDSP.
        sigma_c: coefficient to get SDSP.

    Returns:
        Index of similarity between two images. Usually in [0, 1] range.

    References:
        L. Zhang, Y. Shen and H. Li, "VSI: A Visual Saliency-Induced Index for Perceptual
        Image Quality Assessment," IEEE Transactions on Image Processing, vol. 23, no. 10,
        pp. 4270-4281, Oct. 2014, doi: 10.1109/TIP.2014.2346028
        https://ieeexplore.ieee.org/document/6873260

    Note:
        The original method supports only RGB image.
    """
    x, y = x.double(), y.double()
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        warnings.warn('The original VSI supports only RGB images. The input images were converted to RGB by copying the grey channel 3 times.')
    x = x * 255.0 / data_range
    y = y * 255.0 / data_range
    vs_x = sdsp(x, data_range=255, omega_0=omega_0, sigma_f=sigma_f, sigma_d=sigma_d, sigma_c=sigma_c)
    vs_y = sdsp(y, data_range=255, omega_0=omega_0, sigma_f=sigma_f, sigma_d=sigma_d, sigma_c=sigma_c)
    x_lmn = rgb2lmn(x)
    y_lmn = rgb2lmn(y)
    kernel_size = max(1, round(min(vs_x.size()[-2:]) / 256))
    padding = kernel_size // 2
    if padding:
        upper_pad = padding
        bottom_pad = (kernel_size - 1) // 2
        pad_to_use = [upper_pad, bottom_pad, upper_pad, bottom_pad]
        mode = 'replicate'
        vs_x = pad(vs_x, pad=pad_to_use, mode=mode)
        vs_y = pad(vs_y, pad=pad_to_use, mode=mode)
        x_lmn = pad(x_lmn, pad=pad_to_use, mode=mode)
        y_lmn = pad(y_lmn, pad=pad_to_use, mode=mode)
    vs_x = avg_pool2d(vs_x, kernel_size=kernel_size)
    vs_y = avg_pool2d(vs_y, kernel_size=kernel_size)
    x_lmn = avg_pool2d(x_lmn, kernel_size=kernel_size)
    y_lmn = avg_pool2d(y_lmn, kernel_size=kernel_size)
    kernels = torch.stack([scharr_filter(), scharr_filter().transpose(1, 2)])
    gm_x = gradient_map(x_lmn[:, :1], kernels)
    gm_y = gradient_map(y_lmn[:, :1], kernels)
    s_vs = similarity_map(vs_x, vs_y, c1)
    s_gm = similarity_map(gm_x, gm_y, c2)
    s_m = similarity_map(x_lmn[:, 1:2], y_lmn[:, 1:2], c3)
    s_n = similarity_map(x_lmn[:, 2:], y_lmn[:, 2:], c3)
    s_c = s_m * s_n
    s_c_complex = [s_c.abs(), torch.atan2(torch.zeros_like(s_c), s_c)]
    s_c_complex_pow = [s_c_complex[0] ** beta, s_c_complex[1] * beta]
    s_c_real_pow = s_c_complex_pow[0] * torch.cos(s_c_complex_pow[1])
    s = s_vs * s_gm.pow(alpha) * s_c_real_pow
    vs_max = torch.max(vs_x, vs_y)
    eps = torch.finfo(vs_max.dtype).eps
    output = s * vs_max
    output = ((output.sum(dim=(-1, -2)) + eps) / (vs_max.sum(dim=(-1, -2)) + eps)).squeeze(-1)
    return output


class VSI(nn.Module):
    """Creates a criterion that measures Visual Saliency-induced Index error between
    each element in the input and target.
    Args:
        data_range: Maximum value range of images (usually 1.0 or 255).
        c1: coefficient to calculate saliency component of VSI
        c2: coefficient to calculate gradient component of VSI
        c3: coefficient to calculate color component of VSI
        alpha: power for gradient component of VSI
        beta: power for color component of VSI
        omega_0: coefficient to get log Gabor filter at SDSP
        sigma_f: coefficient to get log Gabor filter at SDSP
        sigma_d: coefficient to get SDSP
        sigma_c: coefficient to get SDSP

    References:
        L. Zhang, Y. Shen and H. Li, "VSI: A Visual Saliency-Induced Index for Perceptual
        Image Quality Assessment," IEEE Transactions on Image Processing, vol. 23, no. 10,
        pp. 4270-4281, Oct. 2014, doi: 10.1109/TIP.2014.2346028
        https://ieeexplore.ieee.org/document/6873260
    """

    def __init__(self, c1: 'float'=1.27, c2: 'float'=386.0, c3: 'float'=130.0, alpha: 'float'=0.4, beta: 'float'=0.02, data_range: 'Union[int, float]'=1.0, omega_0: 'float'=0.021, sigma_f: 'float'=1.34, sigma_d: 'float'=145.0, sigma_c: 'float'=0.001) ->None:
        super().__init__()
        self.data_range = data_range
        self.vsi = functools.partial(vsi, c1=c1, c2=c2, c3=c3, alpha=alpha, beta=beta, omega_0=omega_0, sigma_f=sigma_f, sigma_d=sigma_d, sigma_c=sigma_c, data_range=data_range)

    def forward(self, x, y):
        """Computation of VSI as a loss function.
        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of VSI loss to be minimized in [0, 1] range.
        Note:
            Both inputs are supposed to have RGB channels order in accordance with the original approach.
            Nevertheless, the method supports greyscale images, which they are converted to RGB by copying the grey
            channel 3 times.
        """
        return self.vsi(x=x, y=y)


class WaDIQaM(nn.Module):
    """WaDIQaM model.
    Args:
        metric_type (String): Choose metric mode.
        weighted_average (Boolean): Average the weight.
        train_patch_num (int): Number of patch trained. Default: 32.
        pretrained_model_path (String): The pretrained model path.
        load_feature_weight_only (Boolean): Only load featureweight.
        eps (float): Constant value.

    """

    def __init__(self, metric_type='FR', model_name='wadiqam_fr_kadid', pretrained=True, weighted_average=True, train_patch_num=32, pretrained_model_path=None, load_feature_weight_only=False, eps=1e-08):
        super(WaDIQaM, self).__init__()
        backbone_cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']
        self.features = make_layers(backbone_cfg)
        self.train_patch_num = train_patch_num
        self.patch_size = 32
        self.metric_type = metric_type
        fc_in_channel = 512 * 3 if metric_type == 'FR' else 512
        self.eps = eps
        self.fc_q = nn.Sequential(nn.Linear(fc_in_channel, 512), nn.ReLU(True), nn.Dropout(), nn.Linear(512, 1))
        self.weighted_average = weighted_average
        if weighted_average:
            self.fc_w = nn.Sequential(nn.Linear(fc_in_channel, 512), nn.ReLU(True), nn.Dropout(), nn.Linear(512, 1), nn.ReLU(True))
        if pretrained_model_path is not None:
            self.load_pretrained_network(pretrained_model_path, load_feature_weight_only)
        elif pretrained:
            self.metric_type = model_name.split('_')[1].upper()
            load_pretrained_network(self, default_model_urls[model_name], True, weight_keys='params')

    def load_pretrained_network(self, model_path, load_feature_weight_only=False):
        None
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))['params']
        if load_feature_weight_only:
            None
            new_state_dict = {}
            for k in state_dict.keys():
                if 'features' in k:
                    new_state_dict[k] = state_dict[k]
            self.load_state_dict(new_state_dict, strict=False)
        else:
            self.load_state_dict(state_dict, strict=True)

    def _get_random_patches(self, x, y=None):
        """train with random crop patches"""
        self.patch_num = self.train_patch_num
        b, c, h, w = x.shape
        th = tw = self.patch_size
        cropped_x = []
        cropped_y = []
        for s in range(self.train_patch_num):
            i = torch.randint(0, h - th + 1, size=(1,)).item()
            j = torch.randint(0, w - tw + 1, size=(1,)).item()
            cropped_x.append(x[:, :, i:i + th, j:j + tw])
            if y is not None:
                cropped_y.append(y[:, :, i:i + th, j:j + tw])
        if y is not None:
            cropped_x = torch.stack(cropped_x, dim=1).reshape(-1, c, th, tw)
            cropped_y = torch.stack(cropped_y, dim=1).reshape(-1, c, th, tw)
            return cropped_x, cropped_y
        else:
            cropped_x = torch.stack(cropped_x, dim=1).reshape(-1, c, th, tw)
            return cropped_x

    def _get_nonoverlap_patches(self, x, y=None):
        """test with non overlap patches"""
        self.patch_num = 0
        b, c, h, w = x.shape
        th = tw = self.patch_size
        cropped_x = []
        cropped_y = []
        for i in range(0, h - th, th):
            for j in range(0, w - tw, tw):
                cropped_x.append(x[:, :, i:i + th, j:j + tw])
                if y is not None:
                    cropped_y.append(y[:, :, i:i + th, j:j + tw])
                self.patch_num += 1
        if y is not None:
            cropped_x = torch.stack(cropped_x, dim=1).reshape(-1, c, th, tw)
            cropped_y = torch.stack(cropped_y, dim=1).reshape(-1, c, th, tw)
            return cropped_x, cropped_y
        else:
            cropped_x = torch.stack(cropped_x, dim=1).reshape(-1, c, th, tw)
            return cropped_x

    def get_patches(self, x, y=None):
        if self.training:
            return self._get_random_patches(x, y)
        else:
            return self._get_nonoverlap_patches(x, y)

    def extract_features(self, patches):
        h = self.features(patches)
        h = h.reshape(-1, self.patch_num, 512)
        return h

    def forward(self, x, y=None):
        """WaDIQaM model.
        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A reference tensor. Shape :math:`(N, C, H, W)`.
        """
        if self.metric_type == 'FR':
            assert y is not None, 'Full reference metric requires reference input'
            x_patches, y_patches = self.get_patches(x, y)
            feat_img = self.extract_features(x_patches)
            feat_ref = self.extract_features(y_patches)
            feat_q = torch.cat((feat_ref, feat_img, feat_img - feat_ref), dim=-1)
        else:
            x_patches = self.get_patches(x)
            feat_q = self.extract_features(x_patches)
        q_score = self.fc_q(feat_q)
        weight = self.fc_w(feat_q) + self.eps
        if self.weighted_average:
            q_final = torch.sum(q_score * weight, dim=1) / torch.sum(weight, dim=1)
        else:
            q_final = q_score.mean(dim=1)
        return q_final.reshape(-1, 1)


def _is_pair(x):
    if isinstance(x, (tuple, list)) and len(x) >= 2:
        return True


class PairedRandomRot90(torch.nn.Module):
    """Pair version of random hflip"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, imgs):
        if _is_pair(imgs):
            if torch.rand(1) < self.p:
                for i in range(len(imgs)):
                    imgs[i] = F.rotate(imgs[i], 90)
            return imgs
        elif isinstance(imgs, Image.Image):
            if torch.rand(1) < self.p:
                imgs = F.rotate(imgs, 90)
            return imgs


class PairedRandomARPResize(torch.nn.Module):
    """Pair version of resize"""

    def __init__(self, size_range, interpolation=tf.InterpolationMode.BILINEAR, antialias=None):
        super().__init__()
        self.interpolation = interpolation
        self.antialias = antialias
        self.size_range = size_range
        if not (isinstance(size_range, Sequence) and len(size_range) == 2):
            raise TypeError(f'size_range should be sequence with 2 int. Got {size_range} with {type(size_range)}')

    def forward(self, imgs):
        min_size, max_size = sorted(self.size_range)
        target_size = random.randint(min_size, max_size)
        if _is_pair(imgs):
            for i in range(len(imgs)):
                imgs[i] = F.resize(imgs[i], target_size, self.interpolation)
            return imgs
        elif isinstance(imgs, Image.Image):
            return F.resize(imgs, target_size, self.interpolation)


class PairedRandomSquareResize(torch.nn.Module):
    """Pair version of resize"""

    def __init__(self, size_range, interpolation=tf.InterpolationMode.BILINEAR, antialias=None):
        super().__init__()
        self.interpolation = interpolation
        self.antialias = antialias
        self.size_range = size_range
        if not (isinstance(size_range, Sequence) and len(size_range) == 2):
            raise TypeError(f'size_range should be sequence with 2 int. Got {size_range} with {type(size_range)}')

    def forward(self, imgs):
        min_size, max_size = sorted(self.size_range)
        target_size = random.randint(min_size, max_size)
        target_size = target_size, target_size
        if _is_pair(imgs):
            for i in range(len(imgs)):
                imgs[i] = F.resize(imgs[i], target_size, self.interpolation)
            return imgs
        elif isinstance(imgs, Image.Image):
            return F.resize(imgs, target_size, self.interpolation)


class PairedAdaptivePadding(torch.nn.Module):
    """Pair version of resize"""

    def __init__(self, target_size, fill=0, padding_mode='constant'):
        super().__init__()
        self.target_size = to_2tuple(target_size)
        self.fill = fill
        self.padding_mode = padding_mode

    def get_padding(self, x):
        w, h = x.size
        th, tw = self.target_size
        assert th >= h and tw >= w, f'Target size {self.target_size} should be larger than image size ({h}, {w})'
        pad_row = th - h
        pad_col = tw - w
        pad_l, pad_r, pad_t, pad_b = pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2
        return pad_l, pad_t, pad_r, pad_b

    def forward(self, imgs):
        if _is_pair(imgs):
            for i in range(len(imgs)):
                padding = self.get_padding(imgs[i])
                imgs[i] = F.pad(imgs[i], padding, self.fill, self.padding_mode)
            return imgs
        elif isinstance(imgs, Image.Image):
            padding = self.get_padding(imgs)
            imgs = F.pad(imgs, padding, self.fill, self.padding_mode)
            return imgs


_reduction_modes = ['none', 'mean', 'sum']


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss
    return wrapper


@weighted_loss
def emd_loss(pred, target, r=2):
    """
    Args:
        pred (Tensor): of shape (N, C). Predicted tensor.
        target (Tensor): of shape (N, C). Ground truth tensor.
        r (float): norm level, default l2 norm.
    """
    loss = torch.abs(torch.cumsum(pred, dim=-1) - torch.cumsum(target, dim=-1)) ** r
    loss = loss.mean(dim=-1) ** (1.0 / r)
    return loss


class EMDLoss(nn.Module):
    """EMD (earth mover distance) loss.

    """

    def __init__(self, loss_weight=1.0, r=2, reduction='mean'):
        super(EMDLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.r = r
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * emd_loss(pred, target, r=self.r, weight=weight, reduction=self.reduction)


def plcc_loss(pred, target):
    """
    Args:
        pred (Tensor): of shape (N, 1). Predicted tensor.
        target (Tensor): of shape (N, 1). Ground truth tensor.
    """
    batch_size = pred.shape[0]
    if batch_size > 1:
        vx = pred - pred.mean()
        vy = target - target.mean()
        loss = F.normalize(vx, p=2, dim=0) * F.normalize(vy, p=2, dim=0)
        loss = (1 - loss.sum()) / 2
    else:
        loss = F.l1_loss(pred, target)
    return loss.mean()


class PLCCLoss(nn.Module):
    """PLCC loss, induced from Pearson’s Linear Correlation Coefficient.

    """

    def __init__(self, loss_weight=1.0):
        super(PLCCLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return self.loss_weight * plcc_loss(pred, target)


class RankLoss(nn.Module):
    """Monotonicity regularization loss, will be zero when rankings of pred and target are the same.

    Reference:
        - https://github.com/lidq92/LinearityIQA/blob/master/IQAloss.py

    """

    def __init__(self, detach=False, loss_weight=1.0):
        super(RankLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        if pred.size(0) > 1:
            ranking_loss = F.relu((pred - pred.t()) * torch.sign(target.t() - target))
            scale = 1 + torch.max(ranking_loss.detach())
            loss = ranking_loss.mean() / scale
        else:
            loss = F.l1_loss(pred, target.detach())
        return self.loss_weight * loss


def norm_loss_with_normalization(pred, target, p, q):
    """
    Args:
        pred (Tensor): of shape (N, 1). Predicted tensor.
        target (Tensor): of shape (N, 1). Ground truth tensor.
    """
    batch_size = pred.shape[0]
    if batch_size > 1:
        vx = pred - pred.mean()
        vy = target - target.mean()
        scale = np.power(2, p) * np.power(batch_size, max(0, 1 - p / q))
        norm_pred = F.normalize(vx, p=q, dim=0)
        norm_target = F.normalize(vy, p=q, dim=0)
        loss = torch.norm(norm_pred - norm_target, p=p) / scale
    else:
        loss = F.l1_loss(pred, target)
    return loss.mean()


class NiNLoss(nn.Module):
    """NiN (Norm in Norm) loss

    Reference:

        - Dingquan Li, Tingting Jiang, and Ming Jiang. Norm-in-Norm Loss with Faster Convergence and Better
          Performance for Image Quality Assessment. ACMM2020.
        - https://arxiv.org/abs/2008.03889
        - https://github.com/lidq92/LinearityIQA

    This loss can be simply described as: l1_norm(normalize(pred - pred_mean), normalize(target - target_mean))

    """

    def __init__(self, loss_weight=1.0, p=1, q=2):
        super(NiNLoss, self).__init__()
        self.loss_weight = loss_weight
        self.p = p
        self.q = q

    def forward(self, pred, target):
        return self.loss_weight * norm_loss_with_normalization(pred, target, self.p, self.q)


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@weighted_loss
def cross_entropy(pred, target):
    return F.cross_entropy(pred, target, reduction='none')


class CrossEntropyLoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * cross_entropy(pred, target, weight, reduction=self.reduction)


@weighted_loss
def nll_loss(pred, target):
    return F.nll_loss(pred, target, reduction='none')


class NLLLoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * nll_loss(pred, target, weight, reduction=self.reduction)


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]
        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)
        loss = x_diff + y_diff
        return loss


DEFAULT_CONFIGS = OrderedDict({'ahiq': {'metric_opts': {'type': 'AHIQ'}, 'metric_mode': 'FR', 'score_range': '~0, ~1'}, 'ckdn': {'metric_opts': {'type': 'CKDN'}, 'metric_mode': 'FR', 'score_range': '0, 1'}, 'lpips': {'metric_opts': {'type': 'LPIPS', 'net': 'alex', 'version': '0.1'}, 'metric_mode': 'FR', 'lower_better': True, 'score_range': '0, 1'}, 'lpips-vgg': {'metric_opts': {'type': 'LPIPS', 'net': 'vgg', 'version': '0.1'}, 'metric_mode': 'FR', 'lower_better': True, 'score_range': '0, 1'}, 'lpips+': {'metric_opts': {'type': 'LPIPS', 'net': 'alex', 'version': '0.1', 'semantic_weight_layer': 2}, 'metric_mode': 'FR', 'lower_better': True, 'score_range': '0, 1'}, 'lpips-vgg+': {'metric_opts': {'type': 'LPIPS', 'net': 'vgg', 'version': '0.1', 'semantic_weight_layer': 2}, 'metric_mode': 'FR', 'lower_better': True, 'score_range': '0, 1'}, 'stlpips': {'metric_opts': {'type': 'STLPIPS', 'net': 'alex', 'variant': 'shift_tolerant'}, 'metric_mode': 'FR', 'lower_better': True, 'score_range': '0, 1'}, 'stlpips-vgg': {'metric_opts': {'type': 'STLPIPS', 'net': 'vgg', 'variant': 'shift_tolerant'}, 'metric_mode': 'FR', 'lower_better': True, 'score_range': '0, 1'}, 'dists': {'metric_opts': {'type': 'DISTS'}, 'metric_mode': 'FR', 'lower_better': True, 'score_range': '0, 1'}, 'deepdc': {'metric_opts': {'type': 'DeepDC'}, 'metric_mode': 'FR', 'lower_better': True, 'score_range': '0, 1'}, 'ssim': {'metric_opts': {'type': 'SSIM', 'downsample': False, 'test_y_channel': True}, 'metric_mode': 'FR', 'score_range': '0, 1'}, 'ssimc': {'metric_opts': {'type': 'SSIM', 'downsample': False, 'test_y_channel': False}, 'metric_mode': 'FR', 'score_range': '0, 1'}, 'psnr': {'metric_opts': {'type': 'PSNR', 'test_y_channel': False}, 'metric_mode': 'FR', 'score_range': '~0, ~40'}, 'psnry': {'metric_opts': {'type': 'PSNR', 'test_y_channel': True}, 'metric_mode': 'FR', 'score_range': '~0, ~60'}, 'fsim': {'metric_opts': {'type': 'FSIM', 'chromatic': True}, 'metric_mode': 'FR', 'score_range': '0, 1'}, 'ms_ssim': {'metric_opts': {'type': 'MS_SSIM', 'downsample': False, 'test_y_channel': True, 'is_prod': True}, 'metric_mode': 'FR', 'score_range': '0, 1'}, 'vif': {'metric_opts': {'type': 'VIF'}, 'metric_mode': 'FR', 'score_range': '0, ~1'}, 'gmsd': {'metric_opts': {'type': 'GMSD', 'test_y_channel': True}, 'metric_mode': 'FR', 'lower_better': True, 'score_range': '0, ~1'}, 'nlpd': {'metric_opts': {'type': 'NLPD', 'channels': 1, 'test_y_channel': True}, 'metric_mode': 'FR', 'lower_better': True, 'score_range': '0, 1'}, 'vsi': {'metric_opts': {'type': 'VSI'}, 'metric_mode': 'FR', 'score_range': '0, ~1'}, 'cw_ssim': {'metric_opts': {'type': 'CW_SSIM', 'channels': 1, 'level': 4, 'ori': 8, 'test_y_channel': True}, 'metric_mode': 'FR', 'score_range': '0, 1'}, 'mad': {'metric_opts': {'type': 'MAD'}, 'metric_mode': 'FR', 'lower_better': True, 'score_range': '0, ~'}, 'piqe': {'metric_opts': {'type': 'PIQE'}, 'metric_mode': 'NR', 'lower_better': True, 'score_range': '0, 100'}, 'niqe': {'metric_opts': {'type': 'NIQE', 'test_y_channel': True}, 'metric_mode': 'NR', 'lower_better': True, 'score_range': '~0, ~100'}, 'niqe_matlab': {'metric_opts': {'type': 'NIQE', 'test_y_channel': True, 'version': 'matlab'}, 'metric_mode': 'NR', 'lower_better': True, 'score_range': '~0, ~100'}, 'ilniqe': {'metric_opts': {'type': 'ILNIQE'}, 'metric_mode': 'NR', 'lower_better': True, 'score_range': '~0, ~100'}, 'brisque': {'metric_opts': {'type': 'BRISQUE', 'test_y_channel': True}, 'metric_mode': 'NR', 'lower_better': True, 'score_range': '~0, ~150'}, 'brisque_matlab': {'metric_opts': {'type': 'BRISQUE', 'test_y_channel': True, 'version': 'matlab'}, 'metric_mode': 'NR', 'lower_better': True, 'score_range': '~0, ~100'}, 'nrqm': {'metric_opts': {'type': 'NRQM'}, 'metric_mode': 'NR', 'score_range': '~0, ~10'}, 'pi': {'metric_opts': {'type': 'PI'}, 'metric_mode': 'NR', 'lower_better': True, 'score_range': '~0, ~'}, 'cnniqa': {'metric_opts': {'type': 'CNNIQA', 'pretrained': 'koniq10k'}, 'metric_mode': 'NR', 'score_range': '~0, ~1'}, 'musiq': {'metric_opts': {'type': 'MUSIQ', 'pretrained': 'koniq10k'}, 'metric_mode': 'NR', 'score_range': '~0, ~100'}, 'musiq-ava': {'metric_opts': {'type': 'MUSIQ', 'pretrained': 'ava'}, 'metric_mode': 'NR', 'score_range': '1, 10'}, 'musiq-paq2piq': {'metric_opts': {'type': 'MUSIQ', 'pretrained': 'paq2piq'}, 'metric_mode': 'NR', 'score_range': '~0, ~100'}, 'musiq-spaq': {'metric_opts': {'type': 'MUSIQ', 'pretrained': 'spaq'}, 'metric_mode': 'NR', 'score_range': '~0, ~100'}, 'nima': {'metric_opts': {'type': 'NIMA', 'num_classes': 10, 'base_model_name': 'inception_resnet_v2'}, 'metric_mode': 'NR', 'score_range': '0, 10'}, 'nima-koniq': {'metric_opts': {'type': 'NIMA', 'train_dataset': 'koniq', 'num_classes': 1, 'base_model_name': 'inception_resnet_v2'}, 'metric_mode': 'NR', 'score_range': '~0, ~1'}, 'nima-spaq': {'metric_opts': {'type': 'NIMA', 'train_dataset': 'spaq', 'num_classes': 1, 'base_model_name': 'inception_resnet_v2'}, 'metric_mode': 'NR', 'score_range': '~0, ~1'}, 'nima-vgg16-ava': {'metric_opts': {'type': 'NIMA', 'num_classes': 10, 'base_model_name': 'vgg16'}, 'metric_mode': 'NR', 'score_range': '0, 10'}, 'pieapp': {'metric_opts': {'type': 'PieAPP'}, 'metric_mode': 'FR', 'lower_better': True, 'score_range': '~0, ~5'}, 'paq2piq': {'metric_opts': {'type': 'PAQ2PIQ'}, 'metric_mode': 'NR', 'score_range': '~0, ~100'}, 'dbcnn': {'metric_opts': {'type': 'DBCNN', 'pretrained': 'koniq'}, 'metric_mode': 'NR', 'score_range': '~0, ~1'}, 'fid': {'metric_opts': {'type': 'FID'}, 'metric_mode': 'NR', 'lower_better': True, 'score_range': '0, ~'}, 'maniqa': {'metric_opts': {'type': 'MANIQA', 'train_dataset': 'koniq', 'scale': 0.8}, 'metric_mode': 'NR', 'score_range': '~0, ~1'}, 'maniqa-pipal': {'metric_opts': {'type': 'MANIQA', 'train_dataset': 'pipal'}, 'metric_mode': 'NR', 'score_range': '~0, ~1'}, 'maniqa-kadid': {'metric_opts': {'type': 'MANIQA', 'train_dataset': 'kadid', 'scale': 0.8}, 'metric_mode': 'NR', 'score_range': '~0, ~1'}, 'clipiqa': {'metric_opts': {'type': 'CLIPIQA'}, 'metric_mode': 'NR', 'score_range': '0, 1'}, 'clipiqa+': {'metric_opts': {'type': 'CLIPIQA', 'model_type': 'clipiqa+'}, 'metric_mode': 'NR', 'score_range': '0, 1'}, 'clipiqa+_vitL14_512': {'metric_opts': {'type': 'CLIPIQA', 'model_type': 'clipiqa+_vitL14_512', 'backbone': 'ViT-L/14', 'pos_embedding': True}, 'metric_mode': 'NR', 'score_range': '0, 1'}, 'clipiqa+_rn50_512': {'metric_opts': {'type': 'CLIPIQA', 'model_type': 'clipiqa+_rn50_512', 'backbone': 'RN50', 'pos_embedding': True}, 'metric_mode': 'NR', 'score_range': '0, 1'}, 'tres': {'metric_opts': {'type': 'TReS', 'train_dataset': 'koniq'}, 'metric_mode': 'NR', 'score_range': '~0, ~100'}, 'tres-flive': {'metric_opts': {'type': 'TReS', 'train_dataset': 'flive'}, 'metric_mode': 'NR', 'score_range': '~0, ~100'}, 'hyperiqa': {'metric_opts': {'type': 'HyperNet'}, 'metric_mode': 'NR', 'score_range': '~0, ~1'}, 'uranker': {'metric_opts': {'type': 'URanker'}, 'metric_mode': 'NR', 'score_range': '~-1, ~2'}, 'clipscore': {'metric_opts': {'type': 'CLIPScore'}, 'metric_mode': 'NR', 'score_range': '0, 2.5'}, 'entropy': {'metric_opts': {'type': 'Entropy'}, 'metric_mode': 'NR', 'score_range': '0, 8'}, 'topiq_nr': {'metric_opts': {'type': 'CFANet', 'semantic_model_name': 'resnet50', 'model_name': 'cfanet_nr_koniq_res50', 'use_ref': False}, 'metric_mode': 'NR', 'score_range': '~0, ~1'}, 'topiq_nr-flive': {'metric_opts': {'type': 'CFANet', 'semantic_model_name': 'resnet50', 'model_name': 'cfanet_nr_flive_res50', 'use_ref': False}, 'metric_mode': 'NR', 'score_range': '~0, ~1'}, 'topiq_nr-spaq': {'metric_opts': {'type': 'CFANet', 'semantic_model_name': 'resnet50', 'model_name': 'cfanet_nr_spaq_res50', 'use_ref': False}, 'metric_mode': 'NR', 'score_range': '~0, ~1'}, 'topiq_nr-face': {'metric_opts': {'type': 'CFANet', 'semantic_model_name': 'resnet50', 'model_name': 'topiq_nr_cgfiqa_res50', 'use_ref': False, 'test_img_size': 512}, 'metric_mode': 'NR', 'score_range': '~0, ~1'}, 'topiq_nr_swin-face': {'metric_opts': {'type': 'CFANet', 'semantic_model_name': 'swin_base_patch4_window12_384', 'model_name': 'topiq_nr_cgfiqa_swin', 'use_ref': False, 'test_img_size': 384}, 'metric_mode': 'NR', 'score_range': '~0, ~1'}, 'topiq_nr-face-v1': {'metric_opts': {'type': 'CFANet', 'semantic_model_name': 'resnet50', 'model_name': 'topiq_nr_gfiqa_res50', 'use_ref': False, 'test_img_size': 512}, 'metric_mode': 'NR', 'score_range': '~0, ~1'}, 'topiq_fr': {'metric_opts': {'type': 'CFANet', 'semantic_model_name': 'resnet50', 'model_name': 'cfanet_fr_kadid_res50', 'use_ref': True}, 'metric_mode': 'FR', 'score_range': '~0, ~1'}, 'topiq_fr-pipal': {'metric_opts': {'type': 'CFANet', 'semantic_model_name': 'resnet50', 'model_name': 'cfanet_fr_pipal_res50', 'use_ref': True}, 'metric_mode': 'FR', 'score_range': '~0, ~1'}, 'topiq_iaa': {'metric_opts': {'type': 'CFANet', 'semantic_model_name': 'swin_base_patch4_window12_384', 'model_name': 'cfanet_iaa_ava_swin', 'use_ref': False, 'inter_dim': 512, 'num_heads': 8, 'num_class': 10}, 'metric_mode': 'NR', 'score_range': '1, 10'}, 'topiq_iaa_res50': {'metric_opts': {'type': 'CFANet', 'semantic_model_name': 'resnet50', 'model_name': 'cfanet_iaa_ava_res50', 'use_ref': False, 'inter_dim': 512, 'num_heads': 8, 'num_class': 10, 'test_img_size': 384}, 'metric_mode': 'NR', 'score_range': '1, 10'}, 'laion_aes': {'metric_opts': {'type': 'LAIONAes'}, 'metric_mode': 'NR', 'score_range': '~1, ~10'}, 'liqe': {'metric_opts': {'type': 'LIQE', 'pretrained': 'koniq'}, 'metric_mode': 'NR', 'score_range': '1, 5'}, 'liqe_mix': {'metric_opts': {'type': 'LIQE', 'pretrained': 'mix'}, 'metric_mode': 'NR', 'score_range': '1, 5'}, 'wadiqam_fr': {'metric_opts': {'type': 'WaDIQaM', 'metric_type': 'FR', 'model_name': 'wadiqam_fr_kadid'}, 'metric_mode': 'FR', 'score_range': '~-1, ~0.1'}, 'wadiqam_nr': {'metric_opts': {'type': 'WaDIQaM', 'metric_type': 'NR', 'model_name': 'wadiqam_nr_koniq'}, 'metric_mode': 'NR', 'score_range': '~-1, ~0.1'}, 'qalign': {'metric_opts': {'type': 'QAlign'}, 'metric_mode': 'NR', 'score_range': '1, 5'}, 'qalign_8bit': {'metric_opts': {'type': 'QAlign', 'dtype': '8bit'}, 'metric_mode': 'NR', 'score_range': '1, 5'}, 'qalign_4bit': {'metric_opts': {'type': 'QAlign', 'dtype': '4bit'}, 'metric_mode': 'NR', 'score_range': '1, 5'}, 'compare2score': {'metric_opts': {'type': 'Compare2Score'}, 'metric_mode': 'NR', 'score_range': '0, 100'}, 'unique': {'metric_opts': {'type': 'UNIQUE'}, 'metric_mode': 'NR', 'score_range': '~-3, ~3'}, 'inception_score': {'metric_opts': {'type': 'InceptionScore'}, 'metric_mode': 'NR', 'lower_better': False, 'score_range': '0, ~'}, 'arniqa': {'metric_opts': {'type': 'ARNIQA', 'regressor_dataset': 'koniq'}, 'metric_mode': 'NR', 'score_range': '0, 1'}, 'arniqa-live': {'metric_opts': {'type': 'ARNIQA', 'regressor_dataset': 'live'}, 'metric_mode': 'NR', 'score_range': '0, 1'}, 'arniqa-csiq': {'metric_opts': {'type': 'ARNIQA', 'regressor_dataset': 'csiq'}, 'metric_mode': 'NR', 'score_range': '0, 1'}, 'arniqa-tid': {'metric_opts': {'type': 'ARNIQA', 'regressor_dataset': 'tid'}, 'metric_mode': 'NR', 'score_range': '0, 1'}, 'arniqa-kadid': {'metric_opts': {'type': 'ARNIQA', 'regressor_dataset': 'kadid'}, 'metric_mode': 'NR', 'score_range': '0, 1'}, 'arniqa-clive': {'metric_opts': {'type': 'ARNIQA', 'regressor_dataset': 'clive'}, 'metric_mode': 'NR', 'score_range': '0, 1'}, 'arniqa-flive': {'metric_opts': {'type': 'ARNIQA', 'regressor_dataset': 'flive'}, 'metric_mode': 'NR', 'score_range': '0, 1'}, 'arniqa-spaq': {'metric_opts': {'type': 'ARNIQA', 'regressor_dataset': 'spaq'}, 'metric_mode': 'NR', 'score_range': '0, 1'}, 'msswd': {'metric_opts': {'type': 'MS_SWD_learned'}, 'metric_mode': 'FR', 'score_range': '0, ~10', 'lower_better': True}})


class Registry:
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        assert name not in self._obj_map, f"An object named '{name}' was already registered in '{self._name}' registry!"
        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:

            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class
            return deco
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


ARCH_REGISTRY = Registry('arch')


class ClassMapper:
    """
    ClassMapper is responsible for mapping class names to their corresponding file paths.
    It uses a cache file to store the mapping and refreshes it if necessary.

    Args:
        cache_file (str): JSON file to store the mapping. Default is 'class_mapping.json'.
    """

    def __init__(self, cache_file: 'str'='class_mapping.json'):
        self.arch_folder = Path(osp.dirname(osp.abspath(__file__)))
        self.cache_file = self.arch_folder / cache_file
        self._mapping: 'Optional[Dict]' = None
        self._load_cache()

    def _load_cache(self) ->Dict:
        """Load mapping from cache file."""
        if not osp.exists(self.cache_file):
            None
            self.refresh()
        with open(self.cache_file, 'r') as f:
            self._mapping = json.load(f)

    def _save_cache(self, mapping: 'Dict') ->None:
        """Save mapping to cache file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(mapping, f, indent=4)
        except Exception as e:
            None

    def _find_classes_in_file(self, file_path: 'Path') ->Dict[str, str]:
        """
        Find classes in a Python file that match our criteria.
        Returns a dict of {class_name: file_path}.

        Args:
            file_path (Path): Path to the Python file.

        Returns:
            Dict[str, str]: Mapping of class names to file paths.
        """
        classes = {}
        try:
            module = importlib.import_module(f'pyiqa.archs.{file_path.stem}')
            classes_in_module = inspect.getmembers(module, inspect.isclass)
            for class_name, class_type in classes_in_module:
                classes[class_name] = file_path.stem
        except Exception as e:
            None
        return classes

    def _scan_architecture_files(self) ->Dict:
        """Scan architecture files and create mapping."""
        mapping = {}
        for file_path in self.arch_folder.glob('*_arch.py'):
            file_classes = self._find_classes_in_file(file_path)
            mapping.update(file_classes)
        return mapping

    def get_mapping(self) ->Dict:
        """
        Get the class to filename mapping.

        Returns:
            Dict: Mapping of class names to relative file paths.
        """
        self._mapping = self._scan_architecture_files()
        self._save_cache(self._mapping)
        return self._mapping

    def get_file_for_class(self, class_name: 'str') ->Optional[str]:
        """
        Get the file path for a specific class.

        Args:
            class_name (str): Name of the class to find.

        Returns:
            Optional[str]: Relative path to the file containing the class, or None if not found.
        """
        return self._mapping.get(class_name)

    def refresh(self) ->Dict:
        """
        Force refresh the mapping.

        Returns:
            Dict: Updated mapping dictionary.
        """
        return self.get_mapping()


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


initialized_logger = {}


def get_root_logger(logger_name='pyiqa', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    if logger_name in initialized_logger:
        return logger
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger


def build_network(opt):
    """
    Build a network based on the provided options.

    Args:
        opt (dict): Dictionary containing network options. Must include the 'type' key.

    Returns:
        nn.Module: The constructed network.

    Example:
        >>> net = build_network(opt)
        >>> print(net)
    """
    opt = copy.deepcopy(opt)
    network_type = opt.pop('type')
    logger = get_root_logger()
    if network_type not in ARCH_REGISTRY:
        file_name = class_mapper.get_file_for_class(network_type)
        if file_name is None:
            logger.info(f'Class [{network_type}] not found in cache. Refreshing class mapper file cache.')
            class_mapper.refresh()
            file_name = class_mapper.get_file_for_class(network_type)
        importlib.import_module(f'pyiqa.archs.{file_name}')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net


def imread2tensor(img_source, rgb=False):
    """Read image to tensor.

    Args:
        img_source (str, bytes, or PIL.Image): image filepath string, image contents as a bytearray or a PIL Image instance
        rgb: convert input to RGB if true
    """
    if type(img_source) == bytes:
        img = Image.open(io.BytesIO(img_source))
    elif type(img_source) == str:
        assert is_image_file(img_source), f'{img_source} is not a valid image file.'
        img = Image.open(img_source)
    elif isinstance(img_source, Image.Image):
        img = img_source
    else:
        raise Exception('Unsupported source type')
    if rgb:
        img = img.convert('RGB')
    img_tensor = TF.to_tensor(img)
    return img_tensor


class InferenceModel(torch.nn.Module):
    """Common interface for quality inference of images with default setting of each metric."""

    def __init__(self, metric_name, as_loss=False, loss_weight=None, loss_reduction='mean', device=None, seed=123, check_input_range=True, **kwargs):
        super(InferenceModel, self).__init__()
        self.metric_name = metric_name
        self.lower_better = DEFAULT_CONFIGS[metric_name].get('lower_better', False)
        self.metric_mode = DEFAULT_CONFIGS[metric_name].get('metric_mode', None)
        self.score_range = DEFAULT_CONFIGS[metric_name].get('score_range', None)
        if self.metric_mode is None:
            self.metric_mode = kwargs.pop('metric_mode')
        elif 'metric_mode' in kwargs:
            kwargs.pop('metric_mode')
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.as_loss = as_loss
        self.loss_weight = loss_weight
        self.loss_reduction = loss_reduction
        if metric_name == 'compare2score':
            self.as_loss = True
            self.loss_reduction = 'none'
        self.check_input_range = check_input_range if not as_loss else False
        net_opts = OrderedDict()
        if metric_name in DEFAULT_CONFIGS.keys():
            default_opt = DEFAULT_CONFIGS[metric_name]['metric_opts']
            net_opts.update(default_opt)
        net_opts.update(kwargs)
        self.net = build_network(net_opts)
        self.net = self.net
        self.net.eval()
        self.seed = seed
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def load_weights(self, weights_path, weight_keys='params'):
        load_pretrained_network(self.net, weights_path, weight_keys=weight_keys)

    def is_valid_input(self, x):
        if x is not None:
            assert isinstance(x, torch.Tensor), 'Input must be a torch.Tensor'
            assert x.dim() == 4, 'Input must be 4D tensor (B, C, H, W)'
            assert x.shape[1] in [1, 3], 'Input must be RGB or gray image'
            if self.check_input_range:
                assert x.min() >= 0 and x.max() <= 1, f'Input must be normalized to [0, 1], but got min={x.min():.4f}, max={x.max():.4f}'

    def forward(self, target, ref=None, **kwargs):
        device = self.dummy_param.device
        with torch.set_grad_enabled(self.as_loss):
            if self.metric_name == 'fid':
                output = self.net(target, ref, device=device, **kwargs)
            elif self.metric_name == 'inception_score':
                output = self.net(target, device=device, **kwargs)
            else:
                if not torch.is_tensor(target):
                    target = imread2tensor(target, rgb=True)
                    target = target.unsqueeze(0)
                    if self.metric_mode == 'FR':
                        assert ref is not None, 'Please specify reference image for Full Reference metric'
                        ref = imread2tensor(ref, rgb=True)
                        ref = ref.unsqueeze(0)
                        self.is_valid_input(ref)
                self.is_valid_input(target)
                if self.metric_mode == 'FR':
                    assert ref is not None, 'Please specify reference image for Full Reference metric'
                    output = self.net(target, ref, **kwargs)
                elif self.metric_mode == 'NR':
                    output = self.net(target, **kwargs)
        if self.as_loss:
            if isinstance(output, tuple):
                output = output[0]
            return weight_reduce_loss(output, self.loss_weight, self.loss_reduction)
        else:
            return output


class TmpHead(torch.nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.imagenet_head = torch.nn.Linear(384, 1000)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AdaptiveConcatPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AddHashSpatialPositionEmbs,
     lambda: ([], {'spatial_pos_grid_size': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (AddScaleEmbs,
     lambda: ([], {'num_scales': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BCNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CompactLinear,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DeepDC,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 512, 512]), torch.rand([4, 3, 64, 64])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Entropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GatedConv,
     lambda: ([], {'weightdim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiVGGFeaturesExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (NetLinLayer,
     lambda: ([], {'chn_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Normalize,
     lambda: ([], {'mean': 4, 'std': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PSNR,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (PositionEmbeddingSine,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualAttentionBlock,
     lambda: ([], {'d_model': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (SCNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ScalingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {})),
    (TABlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (TransformerBlock,
     lambda: ([], {'dim': 4, 'mlp_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 1, 4, 1])], {})),
    (resnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (squeezenet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

