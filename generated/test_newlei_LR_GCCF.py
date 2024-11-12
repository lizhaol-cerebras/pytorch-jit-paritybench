
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


import pandas as pd


import scipy.sparse as sp


import torch.utils.data as data


from torch.autograd import Variable


import torch


import math


import random


import time


import torch.nn as nn


import torchvision.transforms as transforms


from torchvision.utils import save_image


from torch.utils.data import DataLoader


from torchvision import datasets


import torch.nn.functional as F


import torch.autograd as autograd


from collections import defaultdict


class BPR(nn.Module):

    def __init__(self, user_num, item_num, factor_num, user_item_matrix, item_user_matrix, d_i_train, d_j_train):
        super(BPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]
        self.d_i_train = torch.FloatTensor(d_i_train)
        self.d_j_train = torch.FloatTensor(d_j_train)
        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight
        gcn1_users_embedding = torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(self.d_i_train)
        gcn1_items_embedding = torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(self.d_j_train)
        gcn2_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(self.d_i_train)
        gcn2_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(self.d_j_train)
        gcn3_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) + gcn2_users_embedding.mul(self.d_i_train)
        gcn3_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) + gcn2_items_embedding.mul(self.d_j_train)
        gcn_users_embedding = torch.cat((users_embedding, gcn1_users_embedding, gcn2_users_embedding, gcn3_users_embedding), -1)
        gcn_items_embedding = torch.cat((items_embedding, gcn1_items_embedding, gcn2_items_embedding, gcn3_items_embedding), -1)
        user = F.embedding(user, gcn_users_embedding)
        item_i = F.embedding(item_i, gcn_items_embedding)
        item_j = F.embedding(item_j, gcn_items_embedding)
        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)
        l2_regulization = 0.01 * (user ** 2 + item_i ** 2 + item_j ** 2).sum(dim=-1)
        loss2 = -(prediction_i - prediction_j).sigmoid().log().mean()
        loss = -(prediction_i - prediction_j).sigmoid().log().mean() + l2_regulization.mean()
        return prediction_i, prediction_j, loss, loss2

