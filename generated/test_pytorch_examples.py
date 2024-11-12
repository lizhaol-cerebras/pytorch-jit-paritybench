
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


import matplotlib.pyplot as plt


import torch


from torchvision import models


import random


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


import torchvision.datasets as dset


import torchvision.transforms as transforms


import torchvision.utils as vutils


import torch.nn.functional as F


import functools


from torch.optim.lr_scheduler import StepLR


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data.distributed import DistributedSampler


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.distributed.fsdp import CPUOffload


from torch.distributed.fsdp import MixedPrecision


from torch.distributed.fsdp import BackwardPrefetch


from torch.distributed.fsdp import ShardingStrategy


from torch.distributed.fsdp import FullStateDictConfig


from torch.distributed.fsdp import StateDictType


from functools import partial


from torch.utils.data import DataLoader


from typing import Type


import time


from typing import ClassVar


from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


from torch.distributed.fsdp import LocalStateDictConfig


from torch.distributed.checkpoint import FileSystemReader


from torch.distributed.checkpoint import FileSystemWriter


from torch.distributed.checkpoint import save_state_dict


from torch.distributed.checkpoint import load_state_dict


from torch.distributed.checkpoint.default_planner import DefaultSavePlanner


from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner


import torch.distributed.checkpoint as dist_cp


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing


from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP


from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload


from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch


from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision


from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


from torch.distributed.fsdp.wrap import enable_wrap


from torch.distributed.fsdp.wrap import wrap


import logging


import re


from itertools import chain


from string import punctuation


import pandas as pd


import numpy as np


from torch.utils.data import Dataset


import torch.cuda.nccl as nccl


from torch.distributed import init_process_group


from torch.distributed import destroy_process_group


from torch.utils.data import random_split


import math


from torch.nn import functional as F


from collections import OrderedDict


from typing import Optional


from typing import Any


from typing import Dict


import torch.distributed.rpc as rpc


from torch import optim


import torchvision


from torch.distributed.rpc import RRef


from torch.distributed.rpc import rpc_sync


from torch.distributed.rpc import rpc_async


from torch.distributed.rpc import remote


from torch.distributions import Categorical


import torch.distributed.autograd as dist_autograd


from torch.distributed.nn import RemoteModule


from torch.distributed.optim import DistributedOptimizer


from torch.distributed.rpc import TensorPipeRpcBackendOptions


from torchvision import datasets


from torchvision import transforms


from functools import wraps


from torchvision.models.resnet import Bottleneck


from itertools import count


from torch.distributed.device_mesh import init_device_mesh


from torch.distributed._tensor import Shard


from torch.distributed._tensor import Replicate


from torch.distributed.tensor.parallel import parallelize_module


from torch.distributed.tensor.parallel import ColwiseParallel


from torch.distributed.tensor.parallel import RowwiseParallel


from torch.distributed.tensor.parallel import PrepareModuleInput


from torch.distributed.tensor.parallel import SequenceParallel


from typing import Tuple


from torch import nn


from torch.distributed._tensor.device_mesh import init_device_mesh


from torch.optim import Adam


import torch.onnx


from collections import namedtuple


from torch.fx import symbolic_trace


from torch.fx import Tracer


from torch.fx import Graph


from torch.fx import GraphModule


from torch.fx import Node


from typing import Callable


from typing import Union


from torch.fx import Proxy


from torch.fx.node import map_arg


import torch.fx as fx


import torch.fx


import torchvision.models as models


from torch.fx import replace_pattern


from enum import Enum


from enum import auto


import warnings


import torch.optim


import torch.utils.data.distributed


import torchvision.datasets as datasets


from torch.utils.data import Subset


from time import time


from torch.nn.utils.rnn import pad_sequence


import torch.optim as O


from torchvision.datasets import MNIST


from torchvision.transforms import Compose


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from torchvision.transforms import Lambda


from torch.utils.data.sampler import Sampler


from collections import deque


import copy


from torchvision import transforms as T


import torch.utils.data as data


from math import log10


import torch.nn.init as init


import matplotlib


from torchvision.utils import save_image


class Generator(nn.Module):

    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True), nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))


class ToyMpModel(nn.Module):

    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = x
        x = self.relu(self.net1(x))
        x = x
        return self.net2(x)


class MultiheadAttentionLayer(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config, device='cpu', dtype=torch.float32):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.attn = torch.nn.MultiheadAttention(embed_dim=config.n_embd, num_heads=config.n_head, dropout=config.attn_pdrop, batch_first=True, device=device, dtype=dtype)

    def forward(self, x):
        _, seq_size, _ = x.size()
        y = self.attn(x, x, x, attn_mask=self.mask[0, 0, :seq_size, :seq_size])[0]
        y = self.resid_drop(self.c_proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadAttentionLayer(config)
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(), nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.resid_pdrop))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class EmbeddingStem(nn.Module):

    def __init__(self, config: 'GPTConfig', device='cpu', dtype=torch.float32):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, device=device, dtype=dtype)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd, device=device, dtype=dtype))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.block_size = config.block_size

    def reset_parameters(self):
        self.tok_emb.reset_parameters()

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, f'Cannot forward sequence of length {t}, block size is only {self.block_size}'
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        return self.drop(token_embeddings + position_embeddings)


class GPT(nn.Module):
    """ GPT Language Model """

    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        self.block_size = config.block_size
        config = self._set_model_config(config)
        self.emb_stem = EmbeddingStem(config)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                p.data.normal_(mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        n_params = sum(p.numel() for p in self.blocks.parameters())
        None

    def _set_model_config(self, config):
        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        if type_given and not params_given:
            config.__dict__.update({'openai-gpt': dict(n_layer=12, n_head=12, n_embd=768), 'gpt2': dict(n_layer=12, n_head=12, n_embd=768), 'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), 'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), 'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), 'gopher-44m': dict(n_layer=8, n_head=16, n_embd=512), 'gpt-mini': dict(n_layer=6, n_head=6, n_embd=192), 'gpt-micro': dict(n_layer=4, n_head=4, n_embd=128), 'gpt-nano': dict(n_layer=3, n_head=3, n_embd=48)}[config.model_type])
        return config

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        x = self.emb_stem(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class HybridModel(torch.nn.Module):
    """
    The model consists of a sparse part and a dense part.
    1) The dense part is an nn.Linear module that is replicated across all trainers using DistributedDataParallel.
    2) The sparse part is a Remote Module that holds an nn.EmbeddingBag on the parameter server.
    This remote model can get a Remote Reference to the embedding table on the parameter server.
    """

    def __init__(self, remote_emb_module, device):
        super(HybridModel, self).__init__()
        self.remote_emb_module = remote_emb_module
        self.fc = DDP(torch.nn.Linear(16, 8), device_ids=[device])
        self.device = device

    def forward(self, indices, offsets):
        emb_lookup = self.remote_emb_module.forward(indices, offsets)
        return self.fc(emb_lookup)


class Net(nn.Module):

    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


class ParameterServer(nn.Module):

    def __init__(self, num_gpus=0):
        super().__init__()
        model = Net(num_gpus=num_gpus)
        self.model = model
        self.input_device = torch.device('cuda:0' if torch.cuda.is_available() and num_gpus > 0 else 'cpu')

    def forward(self, inp):
        inp = inp
        out = self.model(inp)
        out = out
        return out

    def get_dist_gradients(self, cid):
        grads = dist_autograd.get_gradients(cid)
        cpu_grads = {}
        for k, v in grads.items():
            k_cpu, v_cpu = k, v
            cpu_grads[k_cpu] = v_cpu
        return cpu_grads

    def get_param_rrefs(self):
        param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
        return param_rrefs


def get_parameter_server(num_gpus=0):
    global param_server
    with global_lock:
        if not param_server:
            param_server = ParameterServer(num_gpus=num_gpus)
        return param_server


def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


class TrainerNet(nn.Module):

    def __init__(self, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        self.param_server_rref = rpc.remote('parameter_server', get_parameter_server, args=(num_gpus,))

    def get_global_param_rrefs(self):
        remote_params = remote_method(ParameterServer.get_param_rrefs, self.param_server_rref)
        return remote_params

    def forward(self, x):
        model_output = remote_method(ParameterServer.forward, self.param_server_rref, x)
        return model_output


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNetBase(nn.Module):

    def __init__(self, block, inplanes, num_classes=1000, groups=1, width_per_group=64, norm_layer=None):
        super(ResNetBase, self).__init__()
        self._lock = threading.Lock()
        self._block = block
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

    def _make_layer(self, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * self._block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * self._block.expansion, stride), norm_layer(planes * self._block.expansion))
        layers = []
        layers.append(self._block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * self._block.expansion
        for _ in range(1, blocks):
            layers.append(self._block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def parameter_rrefs(self):
        """
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


num_classes = 1000


class ResNetShard1(ResNetBase):
    """
    The first part of ResNet.
    """

    def __init__(self, device, *args, **kwargs):
        super(ResNetShard1, self).__init__(Bottleneck, 64, *args, num_classes=num_classes, **kwargs)
        self.device = device
        self.seq = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False), self._norm_layer(self.inplanes), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), self._make_layer(64, 3), self._make_layer(128, 4, stride=2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_rref):
        x = x_rref.to_here()
        with self._lock:
            out = self.seq(x)
        return out.cpu()


class ResNetShard2(ResNetBase):
    """
    The second part of ResNet.
    """

    def __init__(self, device, *args, **kwargs):
        super(ResNetShard2, self).__init__(Bottleneck, 512, *args, num_classes=num_classes, **kwargs)
        self.device = device
        self.seq = nn.Sequential(self._make_layer(256, 6, stride=2), self._make_layer(512, 3, stride=2), nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * self._block.expansion, num_classes)

    def forward(self, x_rref):
        x = x_rref.to_here()
        with self._lock:
            out = self.fc(torch.flatten(self.seq(x), 1))
        return out.cpu()


class DistResNet50(nn.Module):
    """
    Assemble two parts as an nn.Module and define pipelining logic
    """

    def __init__(self, split_size, workers, *args, **kwargs):
        super(DistResNet50, self).__init__()
        self.split_size = split_size
        self.p1_rref = rpc.remote(workers[0], ResNetShard1, args=('cuda:0',) + args, kwargs=kwargs)
        self.p2_rref = rpc.remote(workers[1], ResNetShard2, args=('cuda:1',) + args, kwargs=kwargs)

    def forward(self, xs):
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params


class EmbeddingTable(nn.Module):
    """
    Encoding layers of the RNNModel
    """

    def __init__(self, ntoken, ninp, dropout):
        super(EmbeddingTable, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if torch.cuda.is_available():
            self.encoder = self.encoder
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)

    def forward(self, input):
        if torch.cuda.is_available():
            input = input
        return self.drop(self.encoder(input)).cpu()


class Decoder(nn.Module):
    """
    Decoding layers of the RNNModel
    """

    def __init__(self, ntoken, nhid, dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)

    def forward(self, output):
        return self.decoder(self.drop(output))


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError as e:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""") from e
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return weight.new_zeros(self.nlayers, bsz, self.nhid), weight.new_zeros(self.nlayers, bsz, self.nhid)
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: 'int', eps: 'float'=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: 'torch.Tensor'):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: 'torch.Tensor'):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


def reshape_for_broadcast(freqs_cis: 'torch.Tensor', x: 'torch.Tensor'):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [(d if i == 1 or i == ndim - 1 else 1) for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: 'torch.Tensor', xk: 'torch.Tensor', freqs_cis: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: 'torch.Tensor', n_rep: 'int') ->torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_local_kv_heads (int): Number of local key and value heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: 'ModelArgs'):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads
        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)

    def init_weights(self, init_std: 'float'):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(self, x: 'torch.Tensor', freqs_cis: 'torch.Tensor'):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)
        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(1, 2).contiguous()
        output = output.view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(self, dim: 'int', hidden_dim: 'int', multiple_of: 'int', ffn_dim_multiplier: 'Optional[float]'):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: 'float'):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: 'int', model_args: 'ModelArgs'):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(dim=model_args.dim, hidden_dim=4 * model_args.dim, multiple_of=model_args.multiple_of, ffn_dim_multiplier=model_args.ffn_dim_multiplier)
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers
        self.attention_norm = RMSNorm(dim=model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = RMSNorm(dim=model_args.dim, eps=model_args.norm_eps)
        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

    def forward(self, x: 'torch.Tensor', freqs_cis: 'torch.Tensor'):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


def precompute_freqs_cis(dim: 'int', end: 'int', theta: 'float'=10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class Transformer(nn.Module):
    """
    Transformer Module

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        model_args (ModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: 'ModelArgs'):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.model_dim = model_args.dim
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer('freqs_cis', precompute_freqs_cis(model_args.dim // model_args.n_heads, model_args.max_seq_len * 2))
        self.layers = torch.nn.ModuleList()
        for layer_id in range(model_args.n_layers):
            self.layers.append(TransformerBlock(layer_id, model_args))
        self.norm = RMSNorm(dim=model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    def init_weights(self):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        with torch.device(self.freqs_cis.device):
            self.freqs_cis = precompute_freqs_cis(self.model_args.dim // self.model_args.n_heads, self.model_args.max_seq_len * 2)
        nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers:
            layer.init_weights()
        self.norm.reset_parameters()
        final_out_std = self.model_args.dim ** -0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(self.output.weight, mean=0.0, std=final_out_std, a=-cutoff_factor * final_out_std, b=cutoff_factor * final_out_std)

    def forward(self, tokens: 'torch.Tensor'):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis
        freqs_cis = self.freqs_cis[0:seqlen]
        for layer in self.layers:
            h = layer(h, freqs_cis)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    @classmethod
    def from_model_args(cls, model_args: 'ModelArgs') ->'Transformer':
        """
        Initialize a Transformer model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)


class ConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class TransformerNet(torch.nn.Module):

    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class Vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
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
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class M1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class M2(torch.nn.Module):

    def forward(self, a, b):
        return a + b


class M(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        y = torch.cat([x, y])
        return y


class MyElementwiseModule(torch.nn.Module):

    def forward(self, x, y):
        return x * y + y


class Foo(torch.nn.Module):

    def forward(self, x):
        with torch.profiler.record_function('foo'):
            return torch.relu(x)


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.

        This operation can be mathematically described as:

            e_ij = a(W h_i, W h_j)
            α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k(exp(e_ik))     
            h_i' = σ(Σ_j(α_ij W h_j))
            
            where h_i and h_j are the feature vectors of nodes i and j respectively, W is a learnable weight matrix,
            a is an attention mechanism that computes the attention coefficients e_ij, and σ is an activation function.

    """

    def __init__(self, in_features: 'int', out_features: 'int', n_heads: 'int', concat: 'bool'=False, dropout: 'float'=0.4, leaky_relu_slope: 'float'=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        if concat:
            self.out_features = out_features
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
        self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads)))
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1)))
        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope)
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)

    def _get_attention_scores(self, h_transformed: 'torch.Tensor'):
        """calculates the attention scores e_ij for all pairs of nodes (i, j) in the graph
        in vectorized parallel form. for each pair of source and target nodes (i, j),
        the attention score e_ij is computed as follows:

            e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) 

            where || denotes the concatenation operation, and a and W are the learnable parameters.

        Args:
            h_transformed (torch.Tensor): Transformed feature matrix with shape (n_nodes, n_heads, n_hidden),
                where n_nodes is the number of nodes and out_features is the number of output features per node.

        Returns:
            torch.Tensor: Attention score matrix with shape (n_heads, n_nodes, n_nodes), where n_nodes is the number of nodes.
        """
        source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])
        target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])
        e = source_scores + target_scores.mT
        return self.leakyrelu(e)

    def forward(self, h: 'torch.Tensor', adj_mat: 'torch.Tensor'):
        """
        Performs a graph attention layer operation.

        Args:
            h (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """
        n_nodes = h.shape[0]
        h_transformed = torch.mm(h, self.W)
        h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)
        h_transformed = h_transformed.view(n_nodes, self.n_heads, self.n_hidden).permute(1, 0, 2)
        e = self._get_attention_scores(h_transformed)
        connectivity_mask = -9e+16 * torch.ones_like(e)
        e = torch.where(adj_mat > 0, e, connectivity_mask)
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h_transformed)
        if self.concat:
            h_prime = h_prime.permute(1, 0, 2).contiguous().view(n_nodes, self.out_features)
        else:
            h_prime = h_prime.mean(dim=0)
        return h_prime


class GAT(nn.Module):
    """
    Graph Attention Network (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.
    Consists of a 2-layer stack of Graph Attention Layers (GATs). The fist GAT Layer is followed by an ELU activation.
    And the second (final) layer is a GAT layer with a single attention head and softmax activation function. 
    """

    def __init__(self, in_features, n_hidden, n_heads, num_classes, concat=False, dropout=0.4, leaky_relu_slope=0.2):
        """ Initializes the GAT model. 

        Args:
            in_features (int): number of input features per node.
            n_hidden (int): output size of the first Graph Attention Layer.
            n_heads (int): number of attention heads in the first Graph Attention Layer.
            num_classes (int): number of classes to predict for each node.
            concat (bool, optional): Wether to concatinate attention heads or take an average over them for the
                output of the first Graph Attention Layer. Defaults to False.
            dropout (float, optional): dropout rate. Defaults to 0.4.
            leaky_relu_slope (float, optional): alpha (slope) of the leaky relu activation. Defaults to 0.2.
        """
        super(GAT, self).__init__()
        self.gat1 = GraphAttentionLayer(in_features=in_features, out_features=n_hidden, n_heads=n_heads, concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope)
        self.gat2 = GraphAttentionLayer(in_features=n_hidden, out_features=num_classes, n_heads=1, concat=False, dropout=dropout, leaky_relu_slope=leaky_relu_slope)

    def forward(self, input_tensor: 'torch.Tensor', adj_mat: 'torch.Tensor'):
        """
        Performs a forward pass through the network.

        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        x = self.gat1(input_tensor, adj_mat)
        x = F.elu(x)
        x = self.gat2(x, adj_mat)
        return F.log_softmax(x, dim=1)


class GraphConv(nn.Module):
    """
        Graph Convolutional Layer described in "Semi-Supervised Classification with Graph Convolutional Networks".

        Given an input feature representation for each node in a graph, the Graph Convolutional Layer aims to aggregate
        information from the node's neighborhood to update its own representation. This is achieved by applying a graph
        convolutional operation that combines the features of a node with the features of its neighboring nodes.

        Mathematically, the Graph Convolutional Layer can be described as follows:

            H' = f(D^(-1/2) * A * D^(-1/2) * H * W)

        where:
            H: Input feature matrix with shape (N, F_in), where N is the number of nodes and F_in is the number of 
                input features per node.
            A: Adjacency matrix of the graph with shape (N, N), representing the relationships between nodes.
            W: Learnable weight matrix with shape (F_in, F_out), where F_out is the number of output features per node.
            D: The degree matrix.
    """

    def __init__(self, input_dim, output_dim, use_bias=False):
        super(GraphConv, self).__init__()
        self.kernel = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_normal_(self.kernel)
        self.bias = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)

    def forward(self, input_tensor, adj_mat):
        """
        Performs a graph convolution operation.

        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Normalized adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """
        support = torch.mm(input_tensor, self.kernel)
        output = torch.spmm(adj_mat, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) as described in the paper `"Semi-Supervised Classification with Graph 
    Convolutional Networks" <https://arxiv.org/pdf/1609.02907.pdf>`.

    The Graph Convolutional Network is a deep learning architecture designed for semi-supervised node 
    classification tasks on graph-structured data. It leverages the graph structure to learn node representations 
    by propagating information through the graph using graph convolutional layers.

    The original implementation consists of two stacked graph convolutional layers. The ReLU activation function is 
    applied to the hidden representations, and the Softmax activation function is applied to the output representations.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, use_bias=True, dropout_p=0.1):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(input_dim, hidden_dim, use_bias=use_bias)
        self.gc2 = GraphConv(hidden_dim, output_dim, use_bias=use_bias)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_tensor, adj_mat):
        """
        Performs forward pass of the Graph Convolutional Network (GCN).

        Args:
            input_tensor (torch.Tensor): Input node feature matrix with shape (N, input_dim), where N is the number of nodes
                and input_dim is the number of input features per node.
            adj_mat (torch.Tensor): Normalized adjacency matrix of the graph with shape (N, N), representing the relationships between
                nodes.

        Returns:
            torch.Tensor: Output tensor with shape (N, output_dim), representing the predicted class probabilities for each node.
        """
        x = self.gc1(input_tensor, adj_mat)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, adj_mat)
        return F.log_softmax(x, dim=1)


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \\text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Translator(nn.Module):

    def __init__(self, num_encoder_layers, num_decoder_layers, embed_size, num_heads, src_vocab_size, tgt_vocab_size, dim_feedforward, dropout):
        super(Translator, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.pos_enc = PositionalEncoding(embed_size, dropout)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.ff = nn.Linear(embed_size, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.pos_enc(self.src_embedding(src))
        tgt_emb = self.pos_enc(self.tgt_embedding(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.ff(outs)

    def encode(self, src, src_mask):
        embed = self.src_embedding(src)
        pos_enc = self.pos_enc(embed)
        return self.transformer.encoder(pos_enc, src_mask)

    def decode(self, tgt, memory, tgt_mask):
        embed = self.tgt_embedding(tgt)
        pos_enc = self.pos_enc(embed)
        return self.transformer.decoder(pos_enc, memory, tgt_mask)


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.d_embed
        dropout = 0 if config.n_layers == 1 else config.dp_ratio
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden, num_layers=config.n_layers, dropout=dropout, bidirectional=config.birnn)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 = inputs.new_zeros(state_shape)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.config.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class SNLIClassifier(nn.Module):

    def __init__(self, config):
        super(SNLIClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = Linear(config.d_embed, config.d_proj)
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()
        seq_in_size = 2 * config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2
        lin_config = [seq_in_size] * 2
        self.out = nn.Sequential(Linear(*lin_config), self.relu, self.dropout, Linear(*lin_config), self.relu, self.dropout, Linear(*lin_config), self.relu, self.dropout, Linear(seq_in_size, config.d_out))

    def forward(self, batch):
        prem_embed = self.embed(batch.premise)
        hypo_embed = self.embed(batch.hypothesis)
        if self.config.fix_emb:
            prem_embed = prem_embed.detach()
            hypo_embed = hypo_embed.detach()
        if self.config.projection:
            prem_embed = self.relu(self.projection(prem_embed))
            hypo_embed = self.relu(self.projection(hypo_embed))
        premise = self.encoder(prem_embed)
        hypothesis = self.encoder(hypo_embed)
        scores = self.out(torch.cat([premise, hypothesis], 1))
        return scores


class Layer(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=args.lr)
        self.threshold = args.threshold
        self.num_epochs = args.epochs

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 0.0001)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in range(self.num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            loss = torch.log1p(torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if i % args.log_interval == 0:
                None
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Sequential(nn.Linear(self.fc_in_features * 2, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))
        self.sigmoid = nn.Sigmoid()
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output


class Sequence(nn.Module):

    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src))
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Block,
     lambda: ([], {'config': SimpleNamespace(n_embd=4, n_head=4, resid_pdrop=0.5, block_size=4, attn_pdrop=4)}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Decoder,
     lambda: ([], {'ntoken': 4, 'nhid': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4, 'multiple_of': 4, 'ffn_dim_multiplier': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Foo,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GAT,
     lambda: ([], {'in_features': 4, 'n_hidden': 4, 'n_heads': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (GCN,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (GraphAttentionLayer,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'n_heads': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (GraphConv,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (M,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (M1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (M2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MultiheadAttentionLayer,
     lambda: ([], {'config': SimpleNamespace(n_embd=4, n_head=4, resid_pdrop=0.5, block_size=4, attn_pdrop=4)}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MyElementwiseModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Policy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SiameseNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64]), torch.rand([4, 1, 64, 64])], {})),
    (TransformerNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (UpsampleConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VAE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 784])], {})),
    (Vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

