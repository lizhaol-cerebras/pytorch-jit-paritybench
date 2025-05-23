
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


import torch.nn.functional as F


from functools import partial


from typing import Optional


from typing import List


from typing import Union


from typing import Tuple


import torch.nn as nn


from typing import Mapping


class Merger:

    def __init__(self, type: 'str'='mean', n: 'int'=1):
        if type not in ['mean', 'gmean', 'sum', 'max', 'min', 'tsharpen']:
            raise ValueError('Not correct merge type `{}`.'.format(type))
        self.output = None
        self.type = type
        self.n = n

    def append(self, x):
        if self.type == 'tsharpen':
            x = x ** 0.5
        if self.output is None:
            self.output = x
        elif self.type in ['mean', 'sum', 'tsharpen']:
            self.output = self.output + x
        elif self.type == 'gmean':
            self.output = self.output * x
        elif self.type == 'max':
            self.output = F.max(self.output, x)
        elif self.type == 'min':
            self.output = F.min(self.output, x)

    @property
    def result(self):
        if self.type in ['sum', 'max', 'min']:
            result = self.output
        elif self.type in ['mean', 'tsharpen']:
            result = self.output / self.n
        elif self.type in ['gmean']:
            result = self.output ** (1 / self.n)
        else:
            raise ValueError('Not correct merge type `{}`.'.format(self.type))
        return result


class SegmentationTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (segmentation model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): segmentation model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_mask_key (str): if model output is `dict`, specify which key belong to `mask`
    """

    def __init__(self, model: 'nn.Module', transforms: 'Compose', merge_mode: 'str'='mean', output_mask_key: 'Optional[str]'=None):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_mask_key

    def forward(self, image: 'torch.Tensor', *args) ->Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        merger = Merger(type=self.merge_mode, n=len(self.transforms))
        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_mask(augmented_output)
            merger.append(deaugmented_output)
        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}
        return result


class ClassificationTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (classification model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): classification model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_label_key (str): if model output is `dict`, specify which key belong to `label`
    """

    def __init__(self, model: 'nn.Module', transforms: 'Compose', merge_mode: 'str'='mean', output_label_key: 'Optional[str]'=None):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_label_key

    def forward(self, image: 'torch.Tensor', *args) ->Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        merger = Merger(type=self.merge_mode, n=len(self.transforms))
        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_label(augmented_output)
            merger.append(deaugmented_output)
        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}
        return result


class KeypointsTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (keypoints model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): keypoints model with single input and single output
         in format [x1,y1, x2, y2, ..., xn, yn]
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_keypoints_key (str): if model output is `dict`, specify which key belong to `label`
        scaled (bool): True if model return x, y scaled values in [0, 1], else False

    """

    def __init__(self, model: 'nn.Module', transforms: 'Compose', merge_mode: 'str'='mean', output_keypoints_key: 'Optional[str]'=None, scaled: 'bool'=False):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_keypoints_key
        self.scaled = scaled

    def forward(self, image: 'torch.Tensor', *args) ->Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        merger = Merger(type=self.merge_mode, n=len(self.transforms))
        size = image.size()
        batch_size, image_height, image_width = size[0], size[2], size[3]
        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            augmented_output = augmented_output.reshape(batch_size, -1, 2)
            if not self.scaled:
                augmented_output[..., 0] /= image_width
                augmented_output[..., 1] /= image_height
            deaugmented_output = transformer.deaugment_keypoints(augmented_output)
            merger.append(deaugmented_output)
        result = merger.result
        if not self.scaled:
            result[..., 0] *= image_width
            result[..., 1] *= image_height
        result = result.reshape(batch_size, -1)
        if self.output_key is not None:
            result = {self.output_key: result}
        return result

