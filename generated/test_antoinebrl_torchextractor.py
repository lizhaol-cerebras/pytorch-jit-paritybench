
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


from collections import OrderedDict


import torch


from torch import nn


from functools import partial


from typing import Any


from typing import Callable


from typing import Dict


from typing import Iterable as IterableType


from typing import List


from typing import Tuple


import logging


from collections.abc import Iterable


class MyTinyVGG(nn.Module):

    def __init__(self):
        super(MyTinyVGG, self).__init__()
        in_channels = 3
        nb_channels = 12
        nb_classes = 17
        self.block1 = self._make_layer(in_channels, nb_channels)
        in_channels, nb_channels = nb_channels, 2 * nb_channels
        self.block2 = self._make_layer(in_channels, nb_channels)
        in_channels, nb_channels = nb_channels, 2 * nb_channels
        self.block3 = self._make_layer(in_channels, nb_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(nb_channels, nb_channels), nn.ReLU(True), nn.Dropout(), nn.Linear(nb_channels, nb_channels), nn.ReLU(True), nn.Dropout(), nn.Linear(nb_channels, nb_classes))

    def _make_layer(self, in_channels, nb_channels):
        layer1 = nn.Sequential(nn.Conv2d(in_channels, nb_channels, kernel_size=3, padding=1), nn.BatchNorm2d(nb_channels), nn.ReLU(inplace=True))
        layer2 = nn.Sequential(nn.Conv2d(nb_channels, nb_channels, kernel_size=3, padding=1), nn.BatchNorm2d(nb_channels), nn.ReLU(inplace=True))
        return nn.Sequential(OrderedDict([('layer1', layer1), ('layer2', layer2), ('pool', nn.MaxPool2d(kernel_size=2, stride=2))]))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.squeeze(3).squeeze(2)
        x = self.classifier(x)
        return x


def attach_name_to_modules(model: 'nn.Module') ->nn.Module:
    """
    Assign a unique name to each module based on the nested structure of the model.

    Parameters
    ----------
    model: nn.Module
        PyTorch model to decorate with fully qualifying names for each module.

    Returns
    -------
    model: nn.Module.
        The provided model as input.

    """
    for name, module in model.named_modules():
        module._extractor_fullname = name
    return model


def hook_capture_module_output(module: 'nn.Module', input: 'Any', output: 'Any', module_name: 'str', feature_maps: 'Dict[str, Any]'):
    """
    Hook function to capture the output of the module.

    Parameters
    ----------
    module: nn.Module
        The module doing the computations.
    input:
        Whatever is provided to the module.
    output:
        Whatever is computed by the module.
    module_name: str
        Fully qualifying name of the module
    feature_maps: dictionary - keys: fully qualifying module names
        Placeholder to store the output of the modules so it can be used later on
    """
    feature_maps[module_name] = output


def hook_wrapper(module: 'nn.Module', input: 'Any', output: 'Any', capture_fn: 'Callable', feature_maps: 'Dict[str, Any]'):
    """
    Hook wrapper to expose module name to hook
    """
    capture_fn(module, input, output, module._extractor_fullname, feature_maps)


def register_hook(module_filter_fn: 'Callable', hook: 'Callable', hook_handles: 'List') ->Callable:
    """
    Attach a hook to some relevant modules.

    Parameters
    ----------
    module_filter_fn: callable
        A filtering function called for each module. When evaluated to `True` a hook is registered.
    hook: callable
        The hook to register. See documentation about PyTorch hooks.
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
    hook_handles: list
        A placeholders containing all newly registered hooks

    Returns
    -------
        callable function to apply on each module

    """

    def init_hook(module: 'nn.Module'):
        if module_filter_fn(module, module._extractor_fullname):
            handle = module.register_forward_hook(hook)
            hook_handles.append(handle)
    return init_hook


class Extractor(nn.Module):

    def __init__(self, model: 'nn.Module', module_names: 'IterableType[str]'=None, module_filter_fn: 'Callable'=None, capture_fn: 'Callable'=None):
        """
        Capture the intermediate feature maps of of model.

        Parameters
        ----------
        model: nn.Module,
            The model to extract features from.

        module_names: list of str, default None
            The fully qualified names of the modules producing the relevant feature maps.

        module_filter_fn: callable, default None
            A filtering function. Takes a module and module name as input and returns True for modules
            producing the relevant features. Either `module_names` or `module_filter_fn` should be
            provided but not both at the same time.

            Example::

                def module_filter_fn(module, name):
                    return isinstance(module, torch.nn.Conv2d)

        capture_fn: callable, default None
            Operation to carry at each forward pass. The function should comply to the following interface.

            Example::

                def capture_fn(
                        module: nn.Module,
                        input: Any,
                        output: Any,
                        module_name:str,
                        feature_maps: Dict[str, Any]
                    ):
                    feature_maps[module_name] = output
        """
        assert module_names is not None or module_filter_fn is not None, 'Module names or a filtering function must be provided'
        assert not (module_names is not None and module_filter_fn is not None), 'You should either specify the fully qualifying names of the modules or a filtering function but not both at the same time'
        super(Extractor, self).__init__()
        self.model = attach_name_to_modules(model)
        self.feature_maps = {}
        self.hook_handles = []
        module_filter_fn = module_filter_fn or (lambda module, name: name in module_names)
        capture_fn = capture_fn or hook_capture_module_output
        hook_fn = partial(hook_wrapper, capture_fn=capture_fn, feature_maps=self.feature_maps)
        self.model.apply(register_hook(module_filter_fn, hook_fn, self.hook_handles))

    def collect(self) ->Dict[str, nn.Module]:
        """
        Returns the structure holding the most recent feature maps.

        Notes
        _____
            The return structure is mutated at each forward pass of the model.
            It is the caller responsibility to duplicate the structure content if needed.
        """
        return self.feature_maps

    def clear_placeholder(self):
        """
        Resets the structure holding captured feature maps.
        """
        self.feature_maps.clear()

    def forward(self, *args, **kwargs) ->Tuple[Any, Dict[str, nn.Module]]:
        """
        Performs model computations and collects feature maps

        Returns
        -------
            Model output and intermediate feature maps
        """
        output = self.model(*args, **kwargs)
        return output, self.feature_maps

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (MyTinyVGG,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

