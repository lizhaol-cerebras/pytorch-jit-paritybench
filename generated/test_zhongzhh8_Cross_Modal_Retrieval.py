
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


import numpy as np


import torchvision.transforms as transforms


import torch.utils.data as torchdata


import re


import torch.optim as optim


from torch.autograd import Variable


import torchvision.models as models


import torch.nn as nn


import torch.nn.functional as F


import time


import itertools


class ImageNet(nn.Module):

    def __init__(self, hash_length):
        super(ImageNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, hash_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        resnet_feature = self.resnet(x)
        image_feature = self.tanh(resnet_feature)
        return image_feature


class TextNet(nn.Module):

    def __init__(self, code_length):
        super(TextNet, self).__init__()
        modelConfig = BertConfig.from_pretrained('/home/disk1/zhaoyuying/models/modeling_bert/bert-base-uncased-config.json')
        self.textExtractor = BertModel.from_pretrained('/home/disk1/zhaoyuying/models/modeling_bert/bert-base-uncased-pytorch_model.bin', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size
        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments, attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        hash_features = self.fc(text_embeddings)
        hash_features = self.tanh(hash_features)
        return hash_features


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ImageNet,
     lambda: ([], {'hash_length': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

