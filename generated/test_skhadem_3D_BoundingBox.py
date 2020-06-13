import sys
_module = sys.modules[__name__]
del sys
Run = _module
Run_no_yolo = _module
Train = _module
File = _module
Math = _module
Plotting = _module
library = _module
ClassAverages = _module
Dataset = _module
Model = _module
torch_lib = _module
yolo = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import numpy as np


import torch


import torch.nn as nn


from torch.autograd import Variable


from torch.utils import data


import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, features=None, bins=2, w=0.4):
        super(Model, self).__init__()
        self.bins = bins
        self.w = w
        self.features = features
        self.orientation = nn.Sequential(nn.Linear(512 * 7 * 7, 256), nn.
            ReLU(True), nn.Dropout(), nn.Linear(256, 256), nn.ReLU(True),
            nn.Dropout(), nn.Linear(256, bins * 2))
        self.confidence = nn.Sequential(nn.Linear(512 * 7 * 7, 256), nn.
            ReLU(True), nn.Dropout(), nn.Linear(256, 256), nn.ReLU(True),
            nn.Dropout(), nn.Linear(256, bins))
        self.dimension = nn.Sequential(nn.Linear(512 * 7 * 7, 512), nn.ReLU
            (True), nn.Dropout(), nn.Linear(512, 512), nn.ReLU(True), nn.
            Dropout(), nn.Linear(512, 3))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 7 * 7)
        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        confidence = self.confidence(x)
        dimension = self.dimension(x)
        return orientation, confidence, dimension


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_skhadem_3D_BoundingBox(_paritybench_base):
    pass