
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


import torch.nn as nn


from math import comb


from torch.nn import functional


import random


from functools import partial


import numpy as np


import pandas as pd


import torch.utils.data


import collections.abc


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import TypeVar


from typing import Union


import torch.nn.utils.rnn as rnn_utils


import torchvision.transforms as transforms


from collections import Counter


from typing import Callable


from abc import ABC


from collections import namedtuple


import torchvision


from abc import abstractmethod


import math


import torch.nn.functional as F


from sklearn.metrics import mean_squared_error


import torch.optim as optim


from torch.optim.lr_scheduler import ReduceLROnPlateau


import warnings


from functools import wraps


from functools import reduce


import matplotlib.pyplot as plt


import matplotlib.ticker as ticker


from typing import Iterable


class BaseInput(nn.Module, ABC):
    """
    General Input class.
    """

    def __init__(self):
        """
        Initializer of the inputs
        """
        super().__init__()
        self.schema = None

    def __len__(self) ->int:
        """
        Return outputs size
        
        Returns:
            int: size of embedding tensor, or number of inputs' fields
        """
        return self.length

    def set_schema(self, inputs: 'Union[str, List[str]]', **kwargs):
        """
        Initialize input layer's schema
        
        Args:
            inputs (Union[str, List[str]]): string or list of strings of inputs' field names
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        schema = namedtuple('Schema', ['inputs'])
        self.schema = schema(inputs=inputs)


class ConcatInput(BaseInput):
    """
    Base Input class for concatenation of list of Base Input class in row-wise.
    The shape of output is :math:`(B, 1, E_{1} + ... + E_{k})`, where :math:`E_{i}` 
    is embedding size of :math:`i-th` field. 
    """

    def __init__(self, inputs: 'List[BaseInput]'):
        """
        Initialize ConcatInput
        
        Args:
            inputs (List[_Inputs]): List of input's layers (trs.inputs.base._Inputs),
                i.e. class of trs.inputs.base

                e.g.
                .. code-block:: python
                    
                    import torecsys as trs

                    # initialize embedding layers used in ConcatInput
                    single_index_emb_0 = trs.inputs.base.SingleIndexEmbedding(2, 8)
                    single_index_emb_1 = trs.inputs.base.SingleIndexEmbedding(2, 8)

                    # set schema, including field names etc.
                    single_index_emb_0.set_schema(['userId'])
                    single_index_emb_1.set_schema(['movieId'])

                    # create ConcatInput embedding layer
                    inputs = [single_index_emb_0, single_index_emb_1]
                    concat_emb = trs.inputs.base.ConcatInput(inputs=inputs)
        """
        super().__init__()
        self.inputs = inputs
        self.length = sum([len(inp) for inp in self.inputs])
        inputs = []
        for idx, inp in enumerate(self.inputs):
            self.add_module(f'input_{idx}', inp)
            schema = inp.schema
            for arguments in schema:
                if isinstance(arguments, list):
                    inputs.extend(arguments)
                elif isinstance(arguments, str):
                    inputs.append(arguments)
        self.set_schema(inputs=list(set(inputs)))

    def __getitem__(self, idx: 'Union[int, slice, str]') ->Union[nn.Module, List[nn.Module]]:
        """
        Get Embedding Layer by index from inputs
        
        Args:
            idx (Union[int, slice, str]): index to get embedding layer from the schema
        
        Returns:
            Union[nn.Module, List[nn.Module]]: embedding layer(s) of the given index
        """
        if isinstance(idx, int):
            emb_layers = self.inputs[idx]
        elif isinstance(idx, slice):
            start = idx.start if idx.start else 0
            stop = idx.stop if idx.stop else len(self.inputs)
            step = idx.step if idx.step else 1
            emb_layers = []
            for i in range(start, stop, step):
                emb_layers.append(self.inputs[i])
        elif isinstance(idx, str):
            emb_layers = []
            for inp in self.inputs:
                if idx in inp.schema.inputs:
                    emb_layers.append(inp)
        else:
            raise ValueError('__getitem__ only accept int, slice, and str.')
        return emb_layers

    def forward(self, inputs: 'Dict[str, torch.Tensor]') ->torch.Tensor:
        """
        Forward calculation of ConcatInput
        
        Args:
            inputs (Dict[str, T]): dictionary of inputs, where key is name of input fields,
                and value is tensor pass to Input class
        
        Returns:
            T, shape = (B, 1, E_{sum}), data_type = torch.float: output of ConcatInput, where
                the values are concatenated in the third dimension
        """
        outputs = []
        for inp in self.inputs:
            inp_val = []
            for k in inp.schema.inputs:
                v = inputs[k]
                v = v.unsqueeze(-1) if v.dim() == 1 else v
                inp_val.append(v)
            inp_val = torch.cat(inp_val, dim=1)
            inp_args = [inp_val]
            if inp.__class__.__name__ == 'SequenceIndexEmbedding':
                inp_args.append(inputs[inp.schema.lengths])
            output = inp(*inp_args)
            if output.dim() < 3:
                output = output.unflatten('E', (('N', 1), ('E', output.size('E'))))
            outputs.append(output)
        outputs = torch.cat(outputs, dim='E')
        return outputs


class ImageInput(BaseInput):
    """
    Base Input class for image, which embed image by a stack of convolution neural network (CNN)
    and fully-connect layer.
    """
    ImageInput = TypeVar('ImageInput')

    def __init__(self, embed_size: 'int', in_channels: 'int', layers_size: 'List[int]', kernels_size: 'List[int]', strides: 'List[int]', paddings: 'List[int]', pooling: 'Optional[str]'='avg_pooling', use_batchnorm: 'Optional[bool]'=True, dropout_p: 'Optional[float]'=0.0, activation: 'Optional[nn.Module]'=nn.ReLU()):
        """
        Initialize ImageInput.
        
        Args:
            embed_size (int): Size of embedding tensor
            in_channels (int): Number of channel of inputs
            layers_size (List[int]): Layers size of CNN
            kernels_size (List[int]): Kernels size of CNN
            strides (List[int]): Strides of CNN
            paddings (List[int]): Paddings of CNN
            pooling (str, optional): Method of pooling layer
                Defaults to avg_pooling
            use_batchnorm (bool, optional): Whether batch normalization is applied or not after Conv2d
                Defaults to True
            dropout_p (float, optional): Probability of Dropout2d
                Defaults to 0.0
            activation (torch.nn.modules.activation, optional): Activation function of Conv2d
                Defaults to nn.ReLU()
        
        Raises:
            ValueError: when pooling is not in ["max_pooling", "avg_pooling"]
        """
        super().__init__()
        self.length = embed_size
        self.model = nn.Sequential()
        layers_size = [in_channels] + layers_size
        iterations = enumerate(zip(layers_size[:-1], layers_size[1:], kernels_size, strides, paddings))
        for i, (in_c, out_c, k, s, p) in iterations:
            conv2d_i = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
            self.model.add_module(f'conv2d_{i}', conv2d_i)
            if use_batchnorm:
                self.model.add_module(f'batchnorm2d_{i}', nn.BatchNorm2d(out_c))
            self.model.add_module(f'dropout2d_{i}', nn.Dropout2d(p=dropout_p))
            self.model.add_module(f'activation_{i}', activation)
        if pooling == 'max_pooling':
            pooling_layer = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        elif pooling == 'avg_pooling':
            pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        else:
            raise ValueError('pooling must be in ["max_pooling", "avg_pooling"].')
        self.model.add_module('pooling', pooling_layer)
        self.fc = nn.Linear(layers_size[-1], embed_size)

    def forward(self, inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of ImageInput
        
        Args:
            inputs (torch.tensor), shape = (B, C, H_{i}, W_{i}), data_type = torch.float: tensor of images
        
        Returns:
            torch.tensor, shape = (B, 1, E): output of ImageInput
        """
        outputs = self.model(inputs.rename(None))
        outputs.names = 'B', 'C', 'H', 'W'
        outputs = self.fc(outputs.rename(None).squeeze())
        outputs = outputs.unsqueeze(1)
        outputs.names = 'B', 'N', 'E'
        return outputs


def dummy_attention(key: 'torch.Tensor', query: 'torch.Tensor', value: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a dummy tensor which have the same inputs and outputs to nn.MultiHeadAttention().__call__()
    
    Args:
        key (T): inputs to be passed as output
        query (T): dummy inputs
        value (T): dummy inputs
    
    Returns:
        Tuple[T, T]: values = (key, dummy outputs = torch.Tensor([]))
    """
    return key, torch.Tensor([])


class ListIndicesEmbedding(BaseInput):
    """
    Base Input class for embedding of list of indices without order, which embed the list
    by multi head attention and aggregate before return
    """

    def __init__(self, embed_size: 'Optional[int]'=None, field_size: 'Optional[int]'=None, padding_idx: 'Optional[int]'=0, nn_embedding: 'Optional[nn.Parameter]'=None, use_attn: 'Optional[bool]'=False, output_method: 'Optional[str]'='avg_pooling', **kwargs):
        """
        Initialize ListIndicesEmbedding.
        
        Args:
            embed_size (int, optional): size of embedding tensor. Defaults to None
            field_size (int, optional): size of inputs field. Defaults to None
            padding_idx (int, optional): padding index. Defaults to 0
            nn_embedding (nn.Parameter, optional): pretrained embedding values. Defaults to None
            use_attn (bool, optional): whether multi head attention is used or not. Defaults to False
            output_method (str, optional): method of aggregation.
                allow: ["avg_pooling", "max_pooling", "mean", "none", "sum"].
                Defaults to "avg_pooling"

        Kwargs:
            num_heads (int): number of heads for multi head attention. Required when use_attn is True.
                Default to 1
            dropout (float, optional): probability of Dropout in multi head attention. Default to 0.0
            bias (bool, optional): Whether bias is added to multi head attention or not. Default to True
            add_bias_kv (bool, optional): Whether bias is added to the key and value sequences at dim = 1
                in multi head attention or not. Default to False
            add_zero_attn (bool, optional): Whether a new batch of zeros is added to the key and value sequences
                at dim = 1 in multi head attention or not. Default to False
        
        Attributes:
            length (int): size of embedding tensor
            embed_size (int): size of embedding tensor
            field_size (int): size of inputs' field
            padding_idx (int): padding index of the embedder
            embedding (torch.nn.Module): embedding layer
            use_attn (bool): flag to show attention is used or not
            attn_args (dict): dictionary of arguments used in multi head attention.
            attention (Union[torch.nn.Module, callable]): multi head attention layer or dummy_attention
            aggregation (Union[torch.nn.Module, callable]): pooling layer or aggregation function
            output_method (string): type of output_method
            
        Raises:
            ValueError: when output_method is not in ["avg_pooling", "max_pooling", "mean", "sum"]
        """
        super().__init__()
        if nn_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(nn_embedding)
        elif field_size is not None and embed_size is not None:
            self.embedding = nn.Embedding(field_size, embed_size, padding_idx=padding_idx)
        else:
            raise ValueError('missing required arguments')
        self.field_size = self.embedding.num_embeddings
        self.embed_size = self.embedding.embedding_dim
        self.padding_idx = self.embedding.padding_idx
        self.length = self.embed_size
        self.use_attn = use_attn
        if self.use_attn:
            self.attn_args = {'embed_dim': self.embed_size, 'num_heads': kwargs.get('num_heads', 1), 'dropout': kwargs.get('dropout', 0.0), 'bias': kwargs.get('bias', True), 'add_bias_kv': kwargs.get('add_bias_kv', False), 'add_zero_attn': kwargs.get('add_zero_attn', False)}
            self.attention = nn.MultiheadAttention(**self.attn_args)
        else:
            self.attention = dummy_attention
        if output_method == 'avg_pooling':
            self.aggregation = nn.AdaptiveAvgPool1d(1)
        elif output_method == 'max_pooling':
            self.aggregation = nn.AdaptiveMaxPool1d(1)
        elif output_method == 'mean':
            self.aggregation = partial(torch.mean, dim='N', keepdim=True)
        elif output_method == 'none':
            self.aggregation = torch.Tensor
        elif output_method == 'sum':
            self.aggregation = partial(torch.sum, dim='N', keepdim=True)
        else:
            raise ValueError('output_method only allows ["avg_pooling", "max_pooling", "mean", "none", "sum"].')
        self.output_method = output_method

    def forward(self, inputs: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward calculation of ListIndicesEmbedding.

        Args:
            inputs (T), shape = (B, L), data_type = torch.long: list of tensor of indices in inputs fields.
        
        Returns:
            Tuple[T, T], shape = ((B, 1 or L, E), (B, L, L) or (None)), 
                data_type = (torch.float, torch.float): outputs of ListIndicesEmbedding and Attention weights.

        TODO: it will raise error now if inputs contains any empty lists. Planning to add padding idx to prevent the
            error.
        """
        outputs = self.embedding(inputs.rename(None))
        outputs.names = 'B', 'L', 'E'
        outputs = outputs.align_to('L', 'B', 'E')
        outputs = outputs.rename(None)
        outputs, _ = self.attention(outputs, outputs, outputs)
        outputs.names = 'L', 'B', 'E'
        outputs = outputs.align_to('B', 'L', 'E')
        if self.output_method == 'avg_pooling' or self.output_method == 'max_pooling':
            outputs = outputs.align_to('B', 'E', 'L')
            outputs = self.aggregation(outputs.rename(None))
            outputs.names = 'B', 'E', 'N'
            outputs = outputs.align_to('B', 'N', 'E')
        else:
            outputs = self.aggregation(outputs.rename(None))
            outputs.names = 'B', 'N', 'E'
        return outputs

    def show_attention(self, inputs: 'torch.Tensor', save_dir: 'Optional[str]'=None):
        """
        Show heat map of self-attention in multi head attention.
        
        Args:
            inputs (T), shape = (1, L), data_type = torch.long: a single sample of list of tensor of indices
                in inputs fields
            save_dir (str, optional): directory to save heat map. Defaults to None
        
        Raises:
            ValueError: when batch size is not equal to 1
            ValueError: when self.attn is not True
        """
        if self.use_attn:
            if inputs.size('B') != 1:
                raise ValueError('batch size must be equal to 1')
            with torch.no_grad():
                outputs = self.embedding(inputs.rename(None))
                outputs.names = 'B', 'L', 'E'
                outputs = outputs.align_to('L', 'B', 'E')
                outputs = outputs.rename(None)
                _, attn_weights = self.attention(outputs, outputs, outputs)
            attn_weights = np.squeeze(attn_weights.numpy(), axis=0)
            axis = [str(x) for x in inputs.rename(None).squeeze().tolist()]
            show_attention(attn_weights, x_axis=axis, y_axis=axis, save_dir=save_dir)
        else:
            raise ValueError('show_attention cannot be called if use_attn is False.')


class MultiIndicesEmbedding(BaseInput):
    """
    Base Input class for embedding indices in multi fields of inputs, which is more efficient than
    embedding with a number of SingleIndexEmbedding.
    """
    MultiIndicesEmbedding = TypeVar('MultiIndicesEmbedding')

    def __init__(self, embed_size: 'Optional[int]'=None, field_sizes: 'Optional[List[int]]'=None, nn_embedding: 'Optional[nn.Parameter]'=None, device: 'str'='cpu', flatten: 'Optional[bool]'=False, **kwargs):
        """
        Initialize MultiIndicesEmbedding.
        
        Args:
            embed_size (int): size of embedding tensor. Defaults to None
            field_sizes (List[int]): list of inputs fields' sizes. Defaults to None
            nn_embedding (nn.Parameter, optional): pretrained embedding values. Defaults to None
            device (str): device of torch. Defaults to cpu
            flatten (bool, optional): whether outputs is reshaped to (B, 1, N * E) or not before return.
                Defaults to False
        
        Attributes:
            length (int): size of embedding tensor multiply by number of fields if flatten is True,
                else Size of embedding tensor
            embedding (torch.nn.Module): embedding layer
            flatten (bool): flag to show outputs will be flattened or not
            offsets (T): tensor of offsets to adjust values of inputs to fit the indices of embedding tensors
        """
        super().__init__()
        if nn_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(nn_embedding)
        elif sum(field_sizes) is not None and embed_size is not None:
            self.embedding = nn.Embedding(sum(field_sizes), embed_size, **kwargs)
        else:
            raise ValueError('missing required arguments')
        self.embedding = self.embedding
        self.offsets = torch.Tensor((0, *np.cumsum(field_sizes)[:-1])).long()
        self.offsets.names = 'N',
        self.offsets = self.offsets.unflatten('N', (('B', 1), ('N', self.offsets.size('N'))))
        self.offsets = self.offsets
        self.flatten = flatten
        self.field_size = self.embedding.num_embeddings
        self.embed_size = self.embedding.embedding_dim
        self.padding_idx = self.embedding.padding_idx
        self.length = self.embed_size * len(field_sizes) if self.flatten else self.embed_size

    def cuda(self, device=None) ->MultiIndicesEmbedding:
        """
        Set MultiIndicesEmbedding to GPU
        
        Returns:
            MultiIndicesEmbedding: self
        """
        super()
        self.offsets = self.offsets
        return self

    def cpu(self) ->MultiIndicesEmbedding:
        """
        Set MultiIndicesEmbedding to CPU
        
        Returns:
            MultiIndicesEmbedding: self
        """
        super().cpu()
        self.offsets = self.offsets.cpu()
        return self

    def forward(self, inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of MultiIndicesEmbedding
        
        Args:
            inputs (T), shape = (B, N), data_type = torch.long: tensor of indices in inputs fields
        
        Returns:
            T, shape = (B, 1, N * E) | (B, N, E), data_type = torch.float:
                outputs of MultiIndicesEmbedding
        """
        self.offsets = self.offsets if inputs.is_cuda else self.offsets.cpu()
        inputs = inputs + self.offsets
        outputs = self.embedding(inputs.rename(None))
        outputs.names = 'B', 'N', 'E'
        if self.flatten:
            outputs = outputs.flatten(('N', 'E'), 'E').rename(None).unsqueeze(1)
        outputs.names = 'B', 'N', 'E'
        return outputs


class MultiIndicesFieldAwareEmbedding(BaseInput):
    """
    Base Input class for Field-aware embedding of multiple indices, which is used in Field-aware
    Factorization (FFM) or the variants. The shape of output is :math:`(B, N * N, E)`, where the embedding 
    tensor :math:`E_{feat_{i, k}, field_{j}}` are looked up the k-th row from the j-th tensor of i-th feature.

    :Reference:

    #. `Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction
        <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_
    
    """
    MultiIndicesFieldAwareEmbedding = TypeVar('MultiIndicesFieldAwareEmbedding')

    def __init__(self, embed_size: 'int', field_sizes: 'List[int]', device: 'str'='cpu', flatten: 'Optional[bool]'=False):
        """
        Initialize MultiIndicesFieldAwareEmbedding.
        
        Args:
            embed_size (int): size of embedding tensor
            field_sizes (List[int]): list of inputs fields' sizes
            device (str): device of torch. Default to cpu.
            flatten (bool, optional): whether outputs is reshaped to (B, 1, N * N * E) or not before return.
                Defaults to False

        Attributes:
            length (int): size of embedding tensor multiply by number of fields if flatten is True,
                else Size of embedding tensor
            embeddings (torch.nn.Module): embedding layers
            flatten (bool): flag to show outputs will be flattened or not
            offsets (T): tensor of offsets to adjust values of inputs to fit the indices of embedding tensors
        """
        super().__init__()
        self.num_fields = len(field_sizes)
        self.embeddings = nn.ModuleList([nn.Embedding(sum(field_sizes), embed_size) for _ in range(self.num_fields)])
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)
        self.embeddings = self.embeddings
        self.offsets = torch.Tensor((0, *np.cumsum(field_sizes)[:-1])).long()
        self.offsets.names = 'N',
        self.offsets = self.offsets.unflatten('N', (('B', 1), ('N', self.offsets.size('N'))))
        self.offsets
        self.flatten = flatten
        self.length = embed_size

    def cuda(self, device=None) ->MultiIndicesFieldAwareEmbedding:
        """
        Set MultiIndicesEmbedding to GPU

        Returns:
            MultiIndicesEmbedding: self
        """
        super()
        self.offsets = self.offsets
        return self

    def cpu(self) ->MultiIndicesFieldAwareEmbedding:
        """
        Set MultiIndicesEmbedding to CPU

        Returns:
            MultiIndicesEmbedding: self
        """
        super().cpu()
        self.offsets = self.offsets.cpu()
        return self

    def forward(self, inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of MultiIndicesFieldAwareEmbedding.
        
        Args:
            inputs (T), shape = (B, N), data_type = torch.long: Tensor of indices in inputs fields.
        
        Returns:
            T, shape = (B, 1, N * N * E) | (B, N * N, E), data_type = torch.float: Embedded Input:
                :math:`\\bm{E} = \\bm{\\Big[} e_{\\text{index}_{i}, \\text{feat}_{j}} \\footnotesize{\\text{, for} \\ i =
                \\text{i-th field} \\ \\text{and} \\ j = \\text{j-th field}} \\bm{\\Big]}`.
        """
        self.offsets = self.offsets if inputs.is_cuda else self.offsets.cpu()
        inputs = inputs + self.offsets
        inputs = inputs.rename(None)
        outputs = torch.cat([self.embeddings[i](inputs) for i in range(self.num_fields)], dim=1)
        if self.flatten:
            outputs = outputs.flatten(('N', 'E'), 'E').rename(None).unsqueeze(1)
        outputs.names = 'B', 'N', 'E'
        return outputs


class PretrainedImageInput(BaseInput):
    """
    Base Input class for image, which embed by famous pretrained model in Computer Vision.
    """

    def __init__(self, embed_size: 'int', model_name: 'str', pretrained: 'Optional[bool]'=True, no_grad: 'Optional[bool]'=False, verbose: 'Optional[bool]'=False, device: 'str'='cpu'):
        """
        Initialize PretrainedImageInput
        
        Args:
            embed_size (int): size of embedding tensor
            model_name (str): model name. Refer to: `torchvision.models
                <https://pytorch.org/vision/stable/models.html>`_
            pretrained (bool, optional): whether use pretrained model or not
                Defaults to True
            verbose (bool, optional): whether display progress bar of the download or not
                Defaults to False
            no_grad (bool, optional): whether parameters in pretrained model
                (excluding fc, i.e. output nn.Linear layer) is set to no gradients or not
                Defaults to False
            device (str): device of torch. Default to cpu.

        :Reference:

        #. `Docs of torchvision.models <https://pytorch.org/docs/stable/torchvision/models.html>`_

        """
        super().__init__()
        self.length = embed_size
        self.model = getattr(torchvision.models, model_name)(pretrained=pretrained, progress=verbose)
        if model_name in ['vgg16', 'vgg19']:
            classifier = self.model.classifier
            last_in_size = classifier[-1].in_features
            classifier[-1] = nn.Linear(last_in_size, embed_size)
        else:
            last_in_size = self.model.fc.in_features
            self.model.fc = nn.Linear(last_in_size, embed_size)
        if no_grad:
            for name, param in self.model.named_parameters():
                if not name.startswith('fc'):
                    param.requires_grad = False
        self.model = self.model

    def forward(self, inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of PretrainedImageInput
        
        Args:
            inputs (T), shape = (B, C, H_{i}, W_{i}), data_type = torch.float: tensor of images.
        
        Returns:
            T, shape = (B, 1, E): output of PretrainedImageInput.
        """
        self.model = self.model if inputs.is_cuda else self.model.cpu()
        outputs = self.model(inputs.rename(None))
        if outputs.dim() == 2:
            outputs = outputs.unsqueeze(dim=1)
        outputs.names = 'B', 'N', 'E'
        return outputs


class SequenceIndicesEmbedding(BaseInput):
    """
    Base Input class for embedding of sequence of indices with order, which embed the sequence by
    Recurrent Neural Network (RNN) and aggregate before return.
    """

    def __init__(self, embed_size: 'int', field_size: 'int', padding_idx: 'Optional[int]'=0, rnn_method: 'Optional[str]'='lstm', output_method: 'Optional[str]'='avg_pooling', nn_embedding: 'Optional[nn.Parameter]'=None, **kwargs):
        """
        Initialize SequenceIndicesEmbedding.
        
        Args:
            embed_size (int): size of embedding tensor
            field_size (int): size of inputs field
            padding_idx (int, optional): padding index. Defaults to 0
            rnn_method (str, optional): method of RNN. allow: ["gru", "lstm", "rnn"].
                Defaults to "lstm".
            output_method (str, optional): method of aggregation.
                allow: ["avg_pooling", "max_pooling", "mean", "none", "sum"].
                Defaults to "avg_pooling".
            nn_embedding (nn.Parameter, optional): pretrained embedding values. Defaults to None.
            
        Kwargs:
            num_layers (int): number of layers of RNN. Default to 1.
            bias (bool): whether bias is added to RNN or not. Default to True.
            dropout (float): probability of Dropout in RNN. Default to 0.0.
            bidirectional (bool): whether bidirectional is used in RNN or not. Default to False.
        
        Attributes:
            length (int): size of embedding tensor.
            embedding (torch.nn.Module): embedding layer.
            rnn_layers (torch.nn.Module): rnn layers.
            aggregation (Union[torch.nn.Module, callable]): pooling layer or aggregation function.
            output_method (string): type of output_method.
        
        Raises:
            ValueError: when rnn_method is not in ["gru", "lstm", "rnn"].
            ValueError: when output_method is not in ["avg_pooling", "max_pooling", "mean", "sum"].
        """
        super().__init__()
        if nn_embedding is not None:
            self.length = nn_embedding.size('E')
            self.embedding = nn.Embedding.from_pretrained(nn_embedding)
        else:
            self.length = embed_size
            self.embedding = nn.Embedding(field_size, embed_size, padding_idx=padding_idx, **kwargs)
        bidirectional = kwargs.get('bidirectional', False)
        hidden_size = embed_size // 2 if bidirectional else embed_size
        rnn_args = {'input_size': embed_size, 'hidden_size': hidden_size, 'num_layers': kwargs.get('num_layers', 1), 'bias': kwargs.get('bias', True), 'batch_first': True, 'dropout': kwargs.get('dropout', 0.0), 'bidirectional': bidirectional}
        if rnn_method == 'rnn':
            self.rnn_layers = nn.RNN(**rnn_args)
        elif rnn_method == 'lstm':
            self.rnn_layers = nn.LSTM(**rnn_args)
        elif rnn_method == 'gru':
            self.rnn_layers = nn.GRU(**rnn_args)
        else:
            raise ValueError('rnn_method only allows ["rnn", "lstm", "gru"].')
        self.output_method = output_method
        if output_method == 'avg_pooling':
            self.aggregation = nn.AdaptiveAvgPool1d(1)
        elif output_method == 'max_pooling':
            self.aggregation = nn.AdaptiveMaxPool1d(1)
        elif output_method == 'mean':
            self.aggregation = partial(torch.mean, dim='N', keepdim=True)
        elif output_method == 'none':
            self.aggregation = torch.Tensor
        elif output_method == 'sum':
            self.aggregation = partial(torch.sum, dim='N', keepdim=True)
        else:
            raise ValueError('output_method only allows ["avg_pooling", "max_pooling", "mean", "none", "sum"].')

    def set_schema(self, inputs: 'str', **kwargs):
        """
        Initialize input layer's schema of SequenceIndicesEmbedding
        
        Args:
            inputs (str): String of input's field name
        
        Kwargs:
            lengths (str): String of length's field name
        """
        lengths = kwargs.get('lengths', None)
        if lengths is None:
            raise ValueError('')
        schema = namedtuple('Schema', ['inputs', 'lengths'])
        self.schema = schema(inputs=[inputs], lengths=lengths)

    def forward(self, inputs: 'torch.Tensor', lengths: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of SequenceIndicesEmbedding

        Args:
            inputs (T), shape = (B, L), data_type = torch.long: sequence of tensor of indices in inputs fields
            lengths (T), shape = (B), data_type = torch.long: length of sequence of tensor of indices
        
        Returns:
            T, shape = (B, 1 or L, E): outputs of SequenceIndicesEmbedding
        """
        lengths, perm_idx = lengths.rename(None).sort(0, descending=True)
        inputs = inputs.rename(None)[perm_idx]
        _, desort_idx = perm_idx.sort()
        embedded = self.embedding(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu().numpy(), batch_first=True)
        rnn_outputs, state = self.rnn_layers(packed)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        outputs = unpacked[desort_idx]
        if self.output_method in ['avg_pooling' or 'max_pooling']:
            outputs.names = 'B', 'L', 'E'
            outputs = outputs.align_to('B', 'E', 'L')
            outputs = self.aggregation(outputs.rename(None))
            outputs.names = 'B', 'E', 'N'
            outputs = outputs.align_to('B', 'N', 'E')
        else:
            outputs = self.aggregation(outputs.rename(None))
            outputs.names = 'B', 'N', 'E'
        return outputs


class SingleIndexEmbedding(BaseInput):
    """
    Base Input class for embedding a single index of an input field
    """

    def __init__(self, embed_size: 'int', field_size: 'int', padding_idx: 'Optional[int]'=None, nn_embedding: 'Optional[nn.Parameter]'=None, **kwargs):
        """
        Initialize SingleIndexEmbedding
        
        Args:
            embed_size (int): size of embedding tensor
            field_size (int): size of inputs field
            padding_idx (int, optional): padding index. Defaults to None
            nn_embedding (nn.Parameter, optional): pretrained embedding values. Defaults to None
        
        Arguments:
            length (int): size of embedding tensor
            embedding (torch.nn.Module): embedding layer
            schema (namedtuple): list of string of field names to be embedded
        """
        super().__init__()
        if nn_embedding is not None:
            embed_size = nn_embedding.size('E')
            self.embedding = nn.Embedding.from_pretrained(nn_embedding)
        else:
            embed_size = embed_size
            self.embedding = nn.Embedding(field_size, embed_size, padding_idx=padding_idx, **kwargs)
        self.length = embed_size

    def forward(self, inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of SingleIndexEmbedding
        
        Args:
            inputs (T), shape = (B, 1), data_type = torch.long: tensor of indices in inputs fields
        
        Returns:
            T, (B, 1, E): outputs of SingleIndexEmbedding
        """
        inputs = inputs.long()
        embedded_tensor = self.embedding(inputs.rename(None))
        embedded_tensor.names = 'B', 'N', 'E'
        return embedded_tensor


class StackedInput(BaseInput):
    """
    Base Input class for stacking of list of Base Input class in column-wise. The shape of output is
        :math:`(B, N_{1} + ... + N_{k}, E)`,
        where :math:`N_{i}` is number of fields of inputs class i.
    """

    def __init__(self, inputs: 'List[BaseInput]'):
        """
        Initialize StackedInputs
        
        Args:
            inputs (List[BaseInput]): list of input's layers (trs.inputs.Inputs).

                e.g.
                .. code-block:: python
                    
                    import torecsys as trs

                    # initialize embedding layers used in StackedInputs
                    single_index_emb_0 = trs.inputs.base.SingleIndexEmbedding(2, 8)
                    single_index_emb_1 = trs.inputs.base.SingleIndexEmbedding(2, 8)

                    # set schema, including field names etc.
                    single_index_emb_0.set_schema(['userId'])
                    single_index_emb_1.set_schema(['movieId'])

                    # create StackedInputs embedding layer
                    inputs = [single_index_emb_0, single_index_emb_1]
                    stacked_emb = trs.inputs.base.StackedInputs(inputs=inputs)
        
        Raise:
            ValueError: when lengths of inputs are not equal.
        """
        super().__init__()
        self.length = len(inputs[0])
        if not all(len(inp) == self.length for inp in inputs):
            raise ValueError('Lengths of inputs, i.e. number of fields or embedding size, must be equal.')
        self.inputs = inputs
        inputs = []
        for idx, inp in enumerate(self.inputs):
            self.add_module(f'Input_{idx}', inp)
            schema = inp.schema
            for arguments in schema:
                if isinstance(arguments, list):
                    inputs.extend(arguments)
                elif isinstance(arguments, str):
                    inputs.append(arguments)
        self.set_schema(inputs=list(set(inputs)))

    def __getitem__(self, idx: 'Union[int, slice, str]') ->Union[nn.Module, List[nn.Module]]:
        """
        Get Embedding Layer by index of the schema
        
        Args:
            idx (Union[int, slice, str]): index to get embedding layer from the schema
        
        Returns:
            Union[nn.Module, List[nn.Module]]: embedding layer(s) of the given index
        """
        if isinstance(idx, int):
            emb_layers = self.inputs[idx]
        elif isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self.schema)
            step = idx.step if idx.step is not None else 1
            emb_layers = []
            for i in range(start, stop, step):
                emb_layers.append(self.inputs[i])
        elif isinstance(idx, str):
            emb_layers = []
            for inp in self.inputs:
                if idx in inp.schema.inputs:
                    emb_layers.append(inp)
        else:
            raise ValueError('__getitem__ only accept int, slice, and str.')
        return emb_layers

    def forward(self, inputs: 'Dict[str, torch.Tensor]') ->torch.Tensor:
        """
        Forward calculation of StackedInput
        
        Args:
            inputs (Dict[str, T]): dictionary of inputs, where key is name of input fields,
                and value is tensor pass to Input class. Remark: key should exist in schema.
            
        Returns:
            T, shape = (B, N_{sum}, E), data_type = torch.float: output of StackedInput,
                where the values are stacked in the second dimension.
        """
        outputs = []
        for inp in self.inputs:
            if inp.__class__.__name__ == 'ConcatInput':
                inp_dict = {i: inputs[i] for i in inp.schema.inputs}
                inp_args = [inp_dict]
            else:
                inp_val = []
                for k in inp.schema.inputs:
                    v = inputs[k]
                    v = v.unsqueeze(-1) if v.dim() == 1 else v
                    inp_val.append(v)
                inp_val = torch.cat(inp_val, dim=1)
                inp_args = [inp_val]
                if inp.__class__.__name__ == 'SequenceIndexEmbedding':
                    inp_args.append(inputs[inp.schema.lengths])
            output = inp(*inp_args)
            if output.dim() < 3:
                output = output.unflatten('E', (('N', 1), ('E', output.size('E'))))
            outputs.append(output)
        outputs = torch.cat(outputs, dim='N')
        return outputs


class ValueInput(BaseInput):
    """
    Base Input class for value to be passed directly.
    """

    def __init__(self, num_fields: 'int', transforms: 'Optional[Callable]'=None):
        """
        Initialize ValueInput
        
        Args:
            num_fields (int): Number of inputs' fields.
        """
        super().__init__()
        self.num_fields = num_fields
        self.transforms = transforms
        self.length = 1

    def forward(self, inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of ValueInput.
        
        Args:
            inputs (T), shape = (B, N): Tensor of values in input fields.
        
        Returns:
            T, shape = (B, 1, N): Outputs of ValueInput
        """
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(dim=-1)
        if self.transforms:
            inputs = self.transforms(inputs)
        inputs.names = 'B', 'N', 'E'
        return inputs


class Inputs(BaseInput):
    """
    Inputs class for wrapping a number of Base Input class into a dictionary,
    the output is a dictionary, where keys are string and values are torch.tensor.
    """
    Inputs = TypeVar('Inputs')

    def __init__(self, schema: 'Union[Dict[str, nn.Module], None]'):
        """
        Initialize Inputs
        
        Args:
            schema (Dict[str, nn.Module]): schema of Input. Dictionary,
                where keys are names of inputs' fields, and values are tensor of fields

                e.g.
                .. code-block:: python
                    
                    import torecsys as trs

                    # initialize embedding layers used in Inputs
                    single_index_emb_0 = trs.inputs.base.SingleIndexEmbedding(2, 8)
                    single_index_emb_1 = trs.inputs.base.SingleIndexEmbedding(2, 8)

                    # set schema, including field names etc.
                    single_index_emb_0.set_schema(["userId"])
                    single_index_emb_1.set_schema(["movieId"])

                    # create Inputs
                    schema = {
                        'user'  : single_index_emb_0,
                        'movie' : single_index_emb_1
                    }
                    inputs = trs.inputs.Inputs(schema=schema)
        
        Attributes:
            schema (Dict[str, nn.Module]): schema of Inputs
        """
        super().__init__()
        self.schema = schema if schema is not None else {}
        for k, emb_fn in self.schema.items():
            self.add_module(k, emb_fn)
        self.length = None

    def forward(self, inputs: 'Dict[str, torch.Tensor]') ->Dict[str, torch.Tensor]:
        """
        Forward calculation of Input
        
        Args:
            inputs (Dict[str, T]): dictionary of inputs, where keys are string and values are torch.tensor.
            
        Returns:
            Dict[str, T], data_type = torch.float: Output of Input, which is a dictionary
                where keys are string and values are torch.tensor.
        """
        outputs = {}
        for k, emb_fn in self.schema.items():
            if emb_fn.__class__.__name__ in ['ConcatInput', 'StackedInput']:
                inp_val = {i: inputs[i] for i in emb_fn.schema.inputs}
                inp_args = [inp_val]
            else:
                inp_val = []
                for emb_k in emb_fn.schema.inputs:
                    v = inputs[emb_k]
                    v = v.unsqueeze(-1) if v.dim() == 1 else v
                    inp_val.append(v)
                inp_val = torch.cat(inp_val, dim=1)
                inp_args = [inp_val]
                if emb_fn.__class__.__name__ == 'SequenceIndexEmbedding':
                    inp_args.append(inputs[emb_fn.schema.lengths])
            outputs[k] = emb_fn(*inp_args)
        return outputs

    def add_inputs(self, name: 'Optional[str]'=None, model: 'Optional[nn.Module]'=None, schema: 'Optional[Dict[str, nn.Module]]'=None) ->Inputs:
        """
        Add new input field to the schema
        
        Args:
            name (str, optional): Name of the input field
            model (nn.Module, optional): torch.nn.Module of the inputs for the input field
            schema (Dict[str, nn.Module], optional): Schema for the inputs of the input field
        
        Raises:
            TypeError: given non-allowed type of schema
            TypeError: given non-allowed type of name
            TypeError: given non-allowed type of model
            AssertionError: given the name of input field is declared in the schema
        
        Returns:
            torecsys.inputs.Inputs: self
        """
        if schema is not None:
            if not isinstance(schema, dict):
                raise TypeError(f'type of schema is not allowed, given {type(schema).__name__}')
            for name, model in schema.items():
                self.add_inputs(name=name, model=model)
        else:
            if not isinstance(name, str):
                raise TypeError(f'type of name is not allowed, given {type(name).__name__}')
            if name in self.schema:
                raise AssertionError(f'Given {name} is defined in the schema.')
            if not isinstance(model, nn.Module):
                raise TypeError(f'type of model is not not allowed, given {type(model).__name__}')
            self.schema.update([(name, model)])
            self.add_module(name, model)
        return self


class BaseLayer(nn.Module, ABC):
    """
    Base Layer for the torecsys module
    """

    def __init__(self, **kwargs):
        """
        Initializer for BaseLayer

        Args:
            **kwargs: kwargs
        """
        super(BaseLayer, self).__init__()

    @property
    @abstractmethod
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        """
        Get inputs size of the layer

        Returns:
            Dict[str, Tuple[str, ...]]: dictionary of inputs_size
        """
        raise NotImplemented('not implemented')

    @property
    @abstractmethod
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        """
        Get outputs size of the layer

        Returns:
            Dict[str, Tuple[str, ...]]: dictionary of outputs_size
        """
        raise NotImplemented('not implemented')


class AttentionalFactorizationMachineLayer(BaseLayer):
    """
    Layer class of Attentional Factorization Machine (AFM).
    
    Attentional Factorization Machine is to calculate interaction between each pair of features 
    by using element-wise product (i.e. Pairwise Interaction Layer), compressing interaction 
    tensors to a single representation. The output shape is (B, 1, E).

    Attributes:
        row_idx: ...
        col_idx: ...
        attention: ...
        dropout: ...

    References:

    - `Jun Xiao et al, 2017. Attentional Factorization Machines: Learning the Weight of Feature Interactions via
    Attention Networks∗ <https://arxiv.org/abs/1708.04617>`_.

    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        """
        Dict[str, Tuple[str, ...]]: inputs_size of the layer
        """
        return {'inputs': ('B', 'N', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        """
        Dict[str, Tuple[str, ...]]: outputs_size of the layer
        """
        return {'outputs': ('B', 'E'), 'attn_scores': ('B', 'NC2', '1')}

    def __init__(self, embed_size: 'int', num_fields: 'int', attn_size: 'int', dropout_p: 'float'=0.1):
        """
        Initialize AttentionalFactorizationMachineLayer
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            attn_size (int): size of attention layer
            dropout_p (float, optional): probability of Dropout in AFM
                Defaults to 0.1.
        """
        super().__init__()
        self.row_idx: 'list' = []
        self.col_idx: 'list' = []
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.row_idx.append(i)
                self.col_idx.append(j)
        self.row_idx = torch.LongTensor(self.row_idx)
        self.col_idx = torch.LongTensor(self.col_idx)
        self.attention: 'nn.Sequential' = nn.Sequential()
        self.attention.add_module('Linear', nn.Linear(embed_size, attn_size))
        self.attention.add_module('Activation', nn.ReLU())
        self.attention.add_module('OutProj', nn.Linear(attn_size, 1))
        self.attention.add_module('Softmax', nn.Softmax(dim=1))
        self.attention.add_module('Dropout', nn.Dropout(dropout_p))
        self.dropout: 'nn.Module' = nn.Dropout(dropout_p)

    def forward(self, emb_inputs: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward calculation of AttentionalFactorizationMachineLayer

        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns: Tuple[T], shape = ((B, E) (B, NC2, 1)), data_type = torch.float: output of
            AttentionalFactorizationMachineLayer and Attention weights
        """
        emb_inputs = emb_inputs.rename(None)
        products = torch.einsum('ijk,ijk->ijk', emb_inputs[:, self.row_idx], emb_inputs[:, self.col_idx])
        attn_scores = self.attention(products.rename(None))
        outputs = torch.einsum('ijk,ijh->ijk', products, attn_scores)
        outputs.names = 'B', 'N', 'E'
        outputs = outputs.sum(dim='N')
        outputs = self.dropout(outputs)
        return outputs, attn_scores


class BiasEncodingLayer(BaseLayer):
    """
    Layer class of Bias Encoding

    Bias Encoding was used in Deep Session Interest Network :title:`Yufei Feng et al, 2019`[1],
    which is to add three types of session-positional bias to session embedding tensors, 
    including: bias of session, bias of position in the session and bias of index in the 
    session.

    :Reference:

    #. `Yufei Feng, 2019. Deep Session Interest Network for Click-Through Rate Prediction
    <https://arxiv.org/abs/1905.06482>`_.
    
    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'session_embedding': ('B', 'L', 'E'), 'session_index': ('B',)}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'L', 'E')}

    def __init__(self, embed_size: 'int', max_num_session: 'int', max_length: 'int'):
        """
        Initialize BiasEncodingLayer
        
        Args:
            embed_size (int): size of embedding tensor
            max_num_session (int): maximum number of session in sequences.
            max_length (int): maximum number of position in sessions.
        """
        super().__init__()
        self.session_bias = nn.Parameter(torch.Tensor(max_num_session, 1, 1))
        self.position_bias = nn.Parameter(torch.Tensor(1, max_length, 1))
        self.item_bias = nn.Parameter(torch.Tensor(1, 1, embed_size))
        nn.init.normal_(self.session_bias)
        nn.init.normal_(self.position_bias)
        nn.init.normal_(self.item_bias)

    def forward(self, session_embed_inputs: 'Tuple[torch.Tensor, torch.Tensor]') ->torch.Tensor:
        """
        Forward calculation of BiasEncodingLayer
        
        Args: session_embed_inputs ((T, T)), shape = ((B, L, E), (B, )), data_type = (torch.float, torch.long):
            embedded feature tensors of session and position of session in sequence.
        
        Returns:
            T, shape = (B, L, E), data_type = torch.float: output of BiasEncodingLayer
        """
        session_index = session_embed_inputs[1]
        batch_size = session_index.size('B')
        session_index = session_index.rename(None)
        session_index = session_index.view(batch_size, 1, 1)
        session_bias = self.session_bias.gather(dim=0, index=session_index)
        output = session_embed_inputs[0] + session_bias + self.position_bias + self.item_bias
        output.names = 'B', 'L', 'E'
        return output


class BilinearNetworkLayer(BaseLayer):
    """
    Layer class of Bilinear.
    
    Bilinear is to calculate interaction in element-wise by nn.Bilinear, which the calculation
    is: for i-th layer, :math:`x_{i} = (x_{0} * A_{i} * x_{i - 1}) + b_{i} + x_{0}`, where 
    :math:`A_{i}` is the weight of model of shape :math:`(O_{i}, I_{i1}, I_{i2})`.

    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'N', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'N', 'E')}

    def __init__(self, inputs_size: 'int', num_layers: 'int'):
        """
        Initialize BilinearNetworkLayer

        Args:
            inputs_size (int): input size of Bilinear, i.e. size of embedding tensor
            num_layers (int): number of layers of Bilinear Network
        """
        super().__init__()
        self.model = nn.ModuleList()
        for _ in range(num_layers):
            self.model.append(nn.Bilinear(inputs_size, inputs_size, inputs_size))

    def forward(self, emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of BilinearNetworkLayer
        
        Args:
            emb_inputs (T), shape = shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, N, E), data_type = torch.float: output of BilinearNetworkLayer
        """
        outputs = emb_inputs.detach().requires_grad_()
        for layer in self.model:
            outputs = layer(emb_inputs.rename(None), outputs.rename(None))
            outputs = outputs + emb_inputs
        if outputs.dim() == 2:
            outputs.names = 'B', 'N'
        elif outputs.dim() == 3:
            outputs.names = 'B', 'N', 'O'
        return outputs


class FieldAllTypeBilinear(BaseLayer):
    """
    Applies a bilinear transformation to the incoming data: :math:`y = x_1 \\cdot W \\odot x_2 + b`
    
    Args:
        in1_features: size of each first input sample
        in2_features: size of each second input sample
        bias: If set to False, the layer will not learn an additive bias. Default: ``True``

    Shape:
        - Input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\\text{in1\\_features}` and
            :math:`*` means any number of additional dimensions. All but the last dimension
            of the inputs should be the same
        - Input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\\text{in2\\_features}`.
        - Output: :math:`(N, *, H_{out})` where :math:`H_{out}=\\text{out\\_features}`
            and all but the last dimension are the same shape as the input
    
    Examples::

        >>> m = FieldAllTypeBilinear(20, 20)
        >>> input1 = torch.randn(128, 10, 20)
        >>> input2 = torch.randn(128, 10, 3)
        >>> output = m(input1, input2)
        >>> print(output.size())
            torch.Size([128, 10, 3])
    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs1': ('B', 'NC2', 'E'), 'inputs2': ('B', 'NC2', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'NC2', 'E')}
    __constants__ = ['in1_features', 'in2_features', 'bias']

    def __init__(self, in1_features, in2_features, bias=True):
        super(FieldAllTypeBilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.weight = nn.Parameter(torch.Tensor(in1_features, in2_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in2_features))
        else:
            self.register_parameter('bias', nn.Parameter(torch.tensor([0])))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.shape[0])
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1, input2):
        output = torch.mul(torch.matmul(input1, self.weight), input2)
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self):
        return f'in1_features={self.in1_features}, in2_features={self.in2_features}, bias={self.bias is not None}'


class FieldEachTypeBilinear(BaseLayer):
    """
    Applies a bilinear transformation to the incoming data: :math:`y = x_1 \\cdot W \\odot x_2 + b`
    
    Args:
        in_features: size of first dimension in first input sample and second input sample
        in1_features: size of each first input sample
        in2_features: size of each second input sample
        bias: If set to False, the layer will not learn an additive bias. Default: ``True``

    Shape:
        - Input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\\text{in1\\_features}` and
              :math:`*` means any number of additional dimensions. All but the last dimension
              of the inputs should be the same
        - Input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\\text{in2\\_features}`
        - Output: :math:`(N, *, H_{out})` where :math:`H_{out}=\\text{out\\_features}`
          and all but the last dimension are the same shape as the input
    
    Examples::

        >>> m = FieldAllTypeBilinear(20, 20)
        >>> input1 = torch.randn(128, 10, 20)
        >>> input2 = torch.randn(128, 10, 3)
        >>> output = m(input1, input2)
        >>> print(output.size())
            torch.Size([128, 10, 3])
    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs1': ('B', 'NC2', 'E'), 'inputs2': ('B', 'NC2', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'NC2', 'E')}
    __constants__ = ['in_features', 'in1_features', 'in2_features', 'bias']

    def __init__(self, in_features, in1_features, in2_features, bias=True):
        super(FieldEachTypeBilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.weight = nn.Parameter(torch.Tensor(in_features, in1_features, in2_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_features, in2_features))
        else:
            self.register_parameter('bias', nn.Parameter(torch.tensor([0])))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.shape[0])
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1, input2):
        output = torch.matmul(input1.unsqueeze(-2), self.weight).squeeze(-2)
        output = torch.mul(output, input2)
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self):
        return f'in1_features={self.in1_features}, in2_features={self.in2_features}, bias={self.bias is not None}'


def combination(n: 'int', r: 'int') ->int:
    """
    Calculate combination
    
    Args:
        n (int): integer of number of elements
        r (int): integer of size of combinations
    
    Returns:
        int: An integer of number of combinations
    """
    r = min(r, n - r)
    num = reduce(op.mul, range(n, n - r, -1), 1)
    den = reduce(op.mul, range(1, r + 1), 1)
    return int(num / den)


class BilinearInteractionLayer(BaseLayer):
    """
    Layer class of Bilinear-Interaction.

    Bilinear-Interaction layer is used in FiBiNet proposed by `Tongwen Huang et al.`[1] to combine inner-product and
    Hadamard product to learn features' interactions with extra parameters W.

    :Reference:

    #. `Tongwen Huang et al, 2019. FibiNET: Combining Feature Importance and Bilinear feature Interaction for
    Click-Through Rate Prediction <https://arxiv.org/abs/1905.09433>`_.
     
    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'N', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return self.bilinear.outputs_size

    def __init__(self, embed_size: 'int', num_fields: 'int', bilinear_type: 'str'='all', bias: 'bool'=True):
        """
        Initialize BilinearInteractionLayer
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            bilinear_type (str, optional): type of bilinear to calculate interactions. Defaults to "all"
            bias (bool, optional): flag to control using bias. Defaults to True
        
        Raises:
            NotImplementedError: /
            ValueError: when bilinear_type is not in ["all", "each", "interaction"]
        """
        super().__init__()
        self.row_idx = []
        self.col_idx = []
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.row_idx.append(i)
                self.col_idx.append(j)
        self.row_idx = torch.LongTensor(self.row_idx)
        self.col_idx = torch.LongTensor(self.col_idx)
        num_interaction = combination(num_fields, 2)
        self.bilinear_type = bilinear_type
        if bilinear_type == 'all':
            self.bilinear = FieldAllTypeBilinear(embed_size, embed_size, bias=bias)
        elif bilinear_type == 'each':
            self.bilinear = FieldEachTypeBilinear(num_interaction, embed_size, embed_size, bias=bias)
        elif bilinear_type == 'interaction':
            raise NotImplementedError()
        else:
            raise ValueError('bilinear_type only allows: ["all", "each", "interaction"].')

    def extra_repr(self) ->str:
        """
        Return information in print-statement of layer
        
        Returns:
            str: Information of print-statement of layer
        """
        return f'bilinear_type={self.bilinear_type}'

    def forward(self, emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of BilinearInteractionLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: Embedded features tensors
        
        Returns:
            T, shape = (B, NC2, E), data_type = torch.float: Output of BilinearInteractionLayer
        """
        p = emb_inputs.rename(None)[:, self.row_idx]
        q = emb_inputs.rename(None)[:, self.col_idx]
        output = self.bilinear(p, q)
        output.names = 'B', 'N', 'O'
        return output


class ComposeExcitationNetworkLayer(BaseLayer):
    """
    Layer class of Compose Excitation Network (CEN) / Squeeze-and-Excitation Network (SENET).
    
    Compose Excitation Network was used in FAT-Deep :title:`Junlin Zhang et al., 2019`[1] and
    Squeeze-and-Excitation Network was used in FibiNET :title:`Tongwen Huang et al, 2019`[2]
    
    #. compose field-aware embedded tensors by a 1D convolution with a :math:`1 * 1` kernel
    feature-wisely from a :math:`k * n` tensor of field i into a :math:`k * 1` tensor. 
    
    #. concatenate the tensors and feed them to dense network to calculate attention 
    weights.
    
    #. inputs' tensor are multiplied by attention weights, and return outputs' tensor with
    shape = (B, N * N, E).

    :Reference:

    #. `Junlin Zhang et al, 2019. FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine
    <https://arxiv.org/abs/1905.06336>`_.

    #. `Tongwen Huang et al, 2019. FibiNET: Combining Feature Importance and Bilinear feature Interaction for
    Click-Through Rate Prediction <https://arxiv.org/abs/1905.09433>`_.

    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'N^2', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'N^2', 'E')}

    def __init__(self, num_fields: 'int', reduction: 'int', squared: 'Optional[bool]'=True, activation: 'Optional[nn.Module]'=nn.ReLU()):
        """
        Initialize ComposeExcitationNetworkLayer
        
        Args:
            num_fields (int): number of inputs' fields
            reduction (int): size of reduction in dense layer
            activation (torch.nn.Module, optional): activation function in dense layers. Defaults to nn.ReLU()
        """
        super().__init__()
        inputs_num_fields = num_fields ** 2 if squared else num_fields
        reduced_num_fields = inputs_num_fields // reduction
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential()
        self.fc.add_module('ReductionLinear', nn.Linear(inputs_num_fields, reduced_num_fields))
        self.fc.add_module('ReductionActivation', activation)
        self.fc.add_module('AdditionLinear', nn.Linear(reduced_num_fields, inputs_num_fields))
        self.fc.add_module('AdditionActivation', activation)

    def forward(self, emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of ComposeExcitationNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N^2, E), data_type = torch.float: field aware embedded features tensors
        
        Returns:
            T, shape = (B, N^2, E), data_type = torch.float: output of ComposeExcitationNetworkLayer
        """
        pooled_inputs = self.pooling(emb_inputs.rename(None))
        pooled_inputs.names = 'B', 'N', 'E'
        pooled_inputs = pooled_inputs.flatten(('N', 'E'), 'N')
        attn_w = self.fc(pooled_inputs.rename(None))
        attn_w.names = 'B', 'N'
        attn_w = attn_w.unflatten('N', (('N', attn_w.size('N')), ('E', 1)))
        outputs = torch.einsum('ijk,ijh->ijk', emb_inputs.rename(None), attn_w.rename(None))
        outputs.names = 'B', 'N', 'E'
        return outputs


class CompressInteractionNetworkLayer(BaseLayer):
    """
    Layer class of Compress Interaction Network (CIN).
    
    Compress Interaction Network was used in xDeepFM by Jianxun Lian et al., 2018.
    
    It compresses cross-features tensors calculated by element-wise cross features interactions
    with outer product by 1D convolution with a :math:`1 * 1` kernel.

    :Reference:

    #. `Jianxun Lian et al, 2018. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommendation
    Systems <https://arxiv.org/abs/1803.05170.pdf>`_.

    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'N', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'O')}

    def __init__(self, embed_size: 'int', num_fields: 'int', output_size: 'int', layer_sizes: 'List[int]', is_direct: 'bool'=False, use_bias: 'bool'=True, use_batchnorm: 'bool'=True, activation: 'Optional[nn.Module]'=nn.ReLU()):
        """
        Initialize CompressInteractionNetworkLayer
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            output_size (int): output size of compress interaction network
            layer_sizes (List[int]): layer sizes of compress interaction network
            is_direct (bool, optional): whether outputs are passed to next step directly or not. Defaults to False
            use_bias (bool, optional): whether bias added to Conv1d or not. Defaults to True
            use_batchnorm (bool, optional): whether batch normalization is applied or not after Conv1d.
                Defaults to True
            activation (nn.Module, optional): activation function of Conv1d. Defaults to nn.ReLU()
        """
        super().__init__()
        self.embed_size = embed_size
        self.is_direct = is_direct
        self.layer_sizes = [num_fields] + layer_sizes
        self.model = nn.ModuleList()
        for i, (s_i, s_j) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            in_c = self.layer_sizes[0] * s_i
            out_c = s_j if is_direct or i == len(self.layer_sizes) - 1 else s_j * 2
            cin = nn.Sequential()
            cin.add_module('Conv1d', nn.Conv1d(in_c, out_c, kernel_size=1, bias=use_bias))
            if use_batchnorm:
                cin.add_module('Batchnorm', nn.BatchNorm1d(out_c))
            if activation is not None:
                cin.add_module('Activation', activation)
            self.model.append(cin)
        model_output_size = int(sum(layer_sizes))
        self.fc = nn.Linear(model_output_size, output_size)

    def forward(self, emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of CompressInteractionNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, O), data_type = torch.float: output of CompressInteractionNetworkLayer
        """
        direct_list = []
        hidden_list = []
        emb_inputs.names = 'B', 'N', 'E'
        x0 = emb_inputs.align_to('B', 'E', 'N')
        hidden_list.append(x0)
        x0 = x0.unflatten('N', (('Nx', x0.size('N')), ('H', 1)))
        for i, layer_size in enumerate(self.layer_sizes[:-1]):
            xi = hidden_list[-1]
            xi = xi.unflatten('N', (('H', 1), ('Ny', xi.size('N'))))
            out_prod = torch.einsum('ijkn,ijnh->ijkh', x0.rename(None), xi.rename(None))
            out_prod.names = 'B', 'E', 'Nx', 'Ny'
            out_prod = out_prod.flatten(('Nx', 'Ny'), 'N')
            out_prod = out_prod.align_to('B', 'N', 'E')
            outputs = self.model[i](out_prod.rename(None))
            outputs.names = 'B', 'N', 'E'
            if self.is_direct:
                direct = outputs
                hidden = outputs.align_to('B', 'E', 'N')
            elif i != len(self.layer_sizes) - 1:
                direct, hidden = torch.chunk(outputs, 2, dim=1)
                hidden = hidden.align_to('B', 'E', 'N')
            else:
                direct = outputs
                hidden = 0
            direct_list.append(direct)
            hidden_list.append(hidden)
        outputs = torch.cat(direct_list, dim='N')
        outputs = self.fc(outputs.sum('E'))
        outputs.names = 'B', 'O'
        return outputs


class CrossNetworkLayer(BaseLayer):
    """
    Layer class of Cross Network.
    
    Cross Network was used in Deep & Cross Network, to calculate cross features interaction between element,
    by the following equation: for i-th layer, :math:`x_{i} = x_{0} * (w_{i} * x_{i-1} + b_{i}) + x_{0}`.

    :Reference:

    #. `Ruoxi Wang et al, 2017. Deep & Cross Network for Ad Click Predictions <https://arxiv.org/abs/1708.05123>`
    
    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'N', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'N', 'E')}

    def __init__(self, inputs_size: 'int', num_layers: 'int'):
        """
        Initialize CrossNetworkLayer
        
        Args:
            inputs_size (int): inputs size of Cross Network, i.e. size of embedding tensor
            num_layers (int): number of layers of Cross Network
        """
        super().__init__()
        self.embed_size = inputs_size
        self.model = nn.ModuleList()
        for _ in range(num_layers):
            self.model.append(nn.Linear(inputs_size, inputs_size))

    def forward(self, emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of CrossNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, N, E), data_type = torch.float: output of CrossNetworkLayer
        """
        outputs = emb_inputs.detach().requires_grad_()
        emb_inputs.names = None
        outputs.names = None
        for layer in self.model:
            outputs = layer(outputs)
            outputs = torch.einsum('ijk,ijk->ijk', emb_inputs, outputs)
            outputs = outputs + emb_inputs
        if outputs.dim() == 2:
            outputs.names = 'B', 'O'
        elif outputs.dim() == 3:
            outputs.names = 'B', 'N', 'O'
        return outputs


def squash(inputs: 'torch.Tensor', dim: 'Optional[int]'=-1) ->torch.Tensor:
    """
    Apply `squash` non-linearity to inputs
    
    Args:
        inputs (T): input tensor which is to be applied squashing
        dim (int, optional): dimension to be applied squashing. Defaults to -1
    
    Returns:
        T: squashed tensor
    """
    squared_norm = torch.sum(torch.pow(inputs, 2), dim=dim, keepdim=True)
    return squared_norm / (1 + squared_norm) * (inputs / (torch.sqrt(squared_norm) + 1e-08))


class DynamicRoutingLayer(BaseLayer):
    """
    Layer class of Behaviour to Interest Dynamic Routing (B2I Dynamic Routing).

    Behaviour to Interest Dynamic Routing is purposed by :title:`Chao Li et al., 2019`[1] to
    transform users' behaviour to users' interest in a variant of Capsule Neural Network,
    which is a new architecture purposed by :title:`Sara Sabour et al, 2017`[2] for image 
    recognition to solve the issues due to back propagation by taking vectors' inputs and
    generate vectors' outputs and keep more information of inference by length anf angle. 

    Behaviour to Interest Dynamic Routing make two changes comparing with the origin: 

    #. Instead of using K projection matrices (i.e. project by different matrix for each 
    activity capsules), all activity capsules share single projection matrices to solve an 
    issues due to the length difference between difference user-item interactions.

    #. To solve an issue due to the above change, use Gaussian Initializer for the
    projection matrix instead of initializing by zero to prevent the same outputs for each 
    activity capsule.

    #. Instead of calculating a fixed number of capsule j (marked by K), the number of 
    activity capsules is calculated dynamically by the following formula:
    :math:`K'_{u} = max(1, min(K, log_{2}(\\left | I_{u} \\right |)))`.

    :Reference:

    #. `Chao Li et al, 2019. Multi-Interest Network with Dynamic Routing for Recommendation at
    Tmall<https://arxiv.org/abs/1904.08030>`_.

    #. `Sara Sabour, 2017 et al. Dynamic Routing Between Capsules <https://arxiv.org/abs/1710.09829>`_.

    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'N', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'Number of Caps', 'Routed Size')}

    def __init__(self, embed_size: 'int', routed_size: 'int', max_num_caps: 'int', num_iter: 'int'):
        """
        Initialize DynamicRoutingLayer
        
        Args:
            embed_size (int): size of embedding tensor
            routed_size (int): size of routed tensor, i.e. output size
            max_num_caps (int): maximum number of capsules
            num_iter (int): number of iterations to update coupling coefficients
        """
        super().__init__()
        self.max_num_caps = max_num_caps
        self.num_caps = None
        self.num_iter = num_iter
        self.S = nn.Parameter(torch.randn(embed_size, routed_size))
        self.S.names = 'E', 'COut'

    def _dynamic_interest_number(self, i: 'int') ->int:
        """
        Calculate number of interest capsules adaptively
        
        Args:
            i (int): number of items in items set interacted with a given user
        
        Returns:
            int: number of interest capsules
        """
        return int(max(1, min(self.max_num_caps, np.log2(i))))

    def forward(self, emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of DynamicRoutingLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, N = N_cap, O = ECap), data_type = torch.float: output of DynamicRoutingLayer
        """
        emb_inputs.names = 'B', 'N', 'E'
        self.num_caps = self._dynamic_interest_number(emb_inputs.size('N'))
        batch_size = emb_inputs.size('B')
        priors = torch.matmul(emb_inputs, self.S)
        priors = priors.unflatten('B', (('B', batch_size), ('C', 1)))
        priors = priors.rename(None).repeat(1, self.num_caps, 1, 1)
        priors.names = 'B', 'K', 'N', 'ECap'
        priors_temp = priors.detach()
        coup_coefficient = torch.randn_like(priors_temp.rename(None), device=priors.device)
        coup_coefficient.names = priors_temp.names
        for _ in range(self.num_iter - 1):
            weights = torch.softmax(coup_coefficient, dim='K')
            z = (weights * priors_temp).sum(dim='N')
            v = squash(z)
            v_temp = v.unflatten('ECap', (('ECap', v.size('ECap')), ('N', 1)))
            similarity = torch.matmul(priors_temp.rename(None), v_temp.rename(None))
            similarity.names = coup_coefficient.names
            coup_coefficient = coup_coefficient + similarity
        weights = torch.softmax(coup_coefficient, dim='K')
        z = (weights * priors).sum(dim='N')
        output = squash(z)
        output.names = 'B', 'N', 'O'
        return output


class FactorizationMachineLayer(BaseLayer):
    """
    Layer class of Factorization Machine (FM).
    
    Factorization Machine is purposed by Steffen Rendle, 2010 to calculate low dimension cross 
    features interactions of sparse field by using a general form of matrix factorization.

    :Reference:

    #. `Steffen Rendle, 2010. Factorization Machine <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_.
    
    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'N', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'E')}

    def __init__(self, dropout_p: 'Optional[float]'=0.0):
        """
        Initialize FactorizationMachineLayer
        
        Args:
            dropout_p (float, optional): probability of Dropout in FM. Defaults to 0.0
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of FactorizationMachineLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, O), data_type = torch.float: output of FactorizationMachineLayer
        """
        emb_inputs.names = 'B', 'N', 'E'
        squared_sum_emb = emb_inputs.sum(dim='N') ** 2
        sum_squared_emb = (emb_inputs ** 2).sum(dim='N')
        outputs = 0.5 * (squared_sum_emb - sum_squared_emb)
        outputs = self.dropout(outputs)
        outputs.names = 'B', 'O'
        return outputs


class FieldAwareFactorizationMachineLayer(BaseLayer):
    """
    Layer class of Field-aware Factorization Machine (FFM).
    
    Field-aware Factorization Machine is purposed by Yuchin Juan et al., 2016, to calculate element-wise cross feature
    interaction per field of sparse fields by using dot product between field-wise feature tensors.

    :Reference:

    #. `Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction
        <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_.
    
    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'N^2', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'NC2', 'E')}

    def __init__(self, num_fields: 'int', dropout_p: 'float'=0.0):
        """
        Initialize FieldAwareFactorizationMachineLayer

        Args:
            num_fields (int): number of inputs' fields
            dropout_p (float, optional): probability of Dropout in FFM. Defaults to 0.0
        """
        super().__init__()
        self.num_fields = num_fields
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, field_emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of FieldAwareFactorizationMachineLayer

        Args:
            field_emb_inputs (T), shape = (B, N * N, E), data_type = torch.float: field aware embedded features tensors
        
        Returns:
            T, shape = (B, NC2, E), data_type = torch.float: output of FieldAwareFactorizationMachineLayer
        """
        field_emb_inputs.names = 'B', 'N', 'E'
        outputs = []
        field_emb_inputs = field_emb_inputs.unflatten('N', (('Nx', self.num_fields), ('Ny', self.num_fields)))
        field_emb_inputs.names = None
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                fij = field_emb_inputs[:, i, j]
                fji = field_emb_inputs[:, j, i]
                output = torch.einsum('ij,ij->ij', fij, fji)
                output.names = 'B', 'E'
                output = output.unflatten('B', (('B', output.size('B')), ('N', 1)))
                outputs.append(output)
        outputs = torch.cat(outputs, dim='N')
        outputs = self.dropout(outputs)
        return outputs


class InnerProductNetworkLayer(BaseLayer):
    """
    Layer class of Inner Product Network.
    
    Inner Product Network is an option in Product based Neural Network by Yanru Qu et at, 2016, by calculating inner
    product between embedded tensors element-wisely to get cross features interactions
    
    :Reference:

    #. `Yanru Qu et at, 2016. Product-based Neural Networks for User Response Prediction
    <https://arxiv.org/abs/1611.00144>`_.
    
    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'N', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'NC2')}

    def __init__(self, num_fields: 'int'):
        """
        Initialize InnerProductNetworkLayer
        
        Args:
            num_fields (int): number of inputs' fields
        """
        super().__init__()
        row_idx = []
        col_idx = []
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row_idx.append(i)
                col_idx.append(j)
        self.row_idx = torch.LongTensor(row_idx)
        self.col_idx = torch.LongTensor(col_idx)

    def forward(self, emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of InnerProductNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors.
        
        Returns:
            T, shape = (B, NC2), data_type = torch.float: output of InnerProductNetworkLayer
        """
        emb_inputs = emb_inputs.rename(None)
        inner = emb_inputs[:, self.row_idx] * emb_inputs[:, self.col_idx]
        inner.names = 'B', 'N', 'E'
        outputs = torch.sum(inner, dim='E')
        outputs.names = 'B', 'O'
        return outputs


class MixtureOfExpertsLayer(BaseLayer):
    """
    Layer class of Mixture-of-Experts (MoE), which is to combine outputs of several models, each of which called
    `expert` and specialized in a different part of input space. To combine them, a gate, which is a stack of linear
    and softmax, will be trained to weight experts' outputs before return.

    :Reference:

    #. `Robert A. Jacobs et al., 1991. Adaptive Mixtures of Local Experts
    <https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf>_.

    #. `David Eigen et al, 2013. Learning Factored Representations in a Deep Mixture of Experts
    <https://arxiv.org/abs/1312.4314>`_.

    #. `Jiaqi Ma et al, 2018. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts
    <https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi
    -gate-mixture->_.

    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'N', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', '1', 'Number of Experts * Expert Output Size')}

    def __init__(self, inputs_size: 'int', output_size: 'int', num_experts: 'int', expert_func: 'type', num_gates: 'int'=1, **kwargs):
        """
        Initialize MixtureOfExpertsLayer
        
        Args:
            inputs_size (int): input size of MOE, i.e. number of fields * size of embedding tensor
            output_size (int): output size of MOE, i.e. number of experts * output size of expert
            num_experts (int): number of expert
            expert_func (type): module of expert, e.g. trs.layers.DNNLayer
            num_gates (int): number of gates. Defaults to 1
        
        Arguments:
            expert_*: Arguments of expert model, e.g. expert_inputs_size
        
        Example:
            to initialize a mixture-of-experts layer, you can follow the below example:

            .. code-block:: python

                import torecsys as trs

                embed_size = 128
                num_fields = 4
                num_experts = 4
                expert_output_size = 16

                layer = trs.layers.MOELayer(
                    inputs_size = embed_size * num_fields,
                    outputs_size = expert_output_size * num_experts,
                    num_experts = num_experts,
                    expert_func = trs.layers.DNNLayer,
                    expert_inputs_size = embed_size * num_fields,
                    expert_output_size = expert_output_size,
                    expert_layer_sizes = [128, 64, 64]
                )
        """
        super().__init__()
        expert_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith('expert_'):
                expert_kwargs[k[7:]] = v
        self.experts = nn.ModuleDict()
        for i in range(num_experts):
            self.experts[f'Expert_{i}'] = expert_func(**expert_kwargs)
        self.gates = nn.ModuleDict()
        for i in range(num_gates):
            gate = nn.Sequential()
            gate.add_module('Linear', nn.Linear(inputs_size, output_size))
            gate.add_module('Softmax', nn.Softmax())
            self.gates[f'Gate_{i}'] = gate

    def forward(self, emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of MixtureOfExpertsLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, O), data_type = torch.float: output of Mixture-of-Experts.
        """
        emb_inputs.names = 'B', 'N', 'E'
        emb_inputs = emb_inputs.flatten(('N', 'E'), 'E')
        emb_inputs.names = None
        experts_output = []
        for expert_name, expert_module in self.experts.items():
            expert_output = expert_module(emb_inputs)
            expert_output.names = 'B', 'O'
            experts_output.append(expert_output)
        experts_output = torch.cat(experts_output, dim='O')
        experts_output.names = None
        gated_weights = []
        for gate_name, gate_module in self.gates.items():
            gated_weight = gate_module(emb_inputs)
            gated_weight.names = 'B', 'O'
            gated_weight = gated_weight.unflatten('O', (('N', 1), ('O', gated_weight.size('O'))))
            gated_weights.append(gated_weight)
        gated_weights = torch.cat(gated_weights, dim='N')
        gated_weights = gated_weights.rename(None)
        gated_experts = torch.einsum('ik,ijk->ijk', experts_output, gated_weights)
        gated_experts.names = 'B', 'N', 'O'
        return gated_experts


class MultilayerPerceptionLayer(BaseLayer):
    """
    Layer class of Multilayer Perception (MLP), which is also called fully connected layer, dense layer,
    deep neural network, etc., to calculate high order non-linear relations of features with a stack of linear,
    dropout and activation.
    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'N', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'N', 'O')}

    def __init__(self, inputs_size: 'int', output_size: 'int', layer_sizes: 'List[int]', dropout_p: 'Optional[List[float]]'=None, activation: 'Optional[nn.Module]'=nn.ReLU()):
        """
        Initialize MultilayerPerceptionLayer
        
        Args:
            inputs_size (int): input size of MLP, i.e. size of embedding tensor
            output_size (int): output size of MLP
            layer_sizes (List[int]): layer sizes of MLP
            dropout_p (List[float], optional): probability of Dropout in MLP. Defaults to None
            activation (torch.nn.Module, optional): activation function in MLP. Defaults to nn.ReLU()
        """
        super().__init__()
        if dropout_p is not None and len(dropout_p) != len(layer_sizes):
            raise ValueError('length of dropout_p must be equal to length of layer_sizes.')
        self.embed_size = inputs_size
        layer_sizes = [inputs_size] + layer_sizes
        self.model = nn.Sequential()
        for i, (in_f, out_f) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.model.add_module(f'Linear_{i}', nn.Linear(in_f, out_f))
            if activation is not None:
                self.model.add_module(f'Activation_{i}', activation)
            if dropout_p is not None:
                self.model.add_module(f'Dropout_{i}', nn.Dropout(dropout_p[i]))
        self.model.add_module('LinearOutput', nn.Linear(layer_sizes[-1], output_size))

    def forward(self, emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of MultilayerPerceptionLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, N, O), data_type = torch.float: output of MLP
        """
        outputs = self.model(emb_inputs.rename(None))
        if outputs.dim() == 2:
            outputs.names = 'B', 'O'
        elif outputs.dim() == 3:
            outputs.names = 'B', 'N', 'O'
        return outputs


class OuterProductNetworkLayer(BaseLayer):
    """
    Layer class of Outer Product Network.
    
    Outer Product Network is an option in Product based Neural Network by Yanru Qu et at, 2016, by calculating
    outer product between embedded tensors element-wisely and compressing by a kernel to get cross features
    interactions.
    
    :Reference:

    #. `Yanru Qu et at, 2016. Product-based Neural Networks for User Response Prediction
    <https://arxiv.org/abs/1611.00144>`_.
    
    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'N', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'NC2')}

    def __init__(self, embed_size: 'int', num_fields: 'int', kernel_type: 'Optional[str]'='mat'):
        """
        Initialize OuterProductNetworkLayer
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            kernel_type (str, optional): type of kernel to compress outer-product. Defaults to 'mat'
        """
        super().__init__()
        self.row_idx = []
        self.col_idx = []
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.row_idx.append(i)
                self.col_idx.append(j)
        self.row_idx = torch.LongTensor(self.row_idx)
        self.col_idx = torch.LongTensor(self.col_idx)
        if kernel_type == 'mat':
            kernel_size = embed_size, num_fields * (num_fields - 1) // 2, embed_size
        elif kernel_type == 'vec':
            kernel_size = 1, num_fields * (num_fields - 1) // 2, embed_size
        elif kernel_type == 'num':
            kernel_size = 1, num_fields * (num_fields - 1) // 2, 1
        else:
            raise ValueError('kernel_type only allows: ["mat", "num", "vec"].')
        self.kernel_type = kernel_type
        self.kernel = nn.Parameter(torch.zeros(kernel_size))
        nn.init.xavier_normal_(self.kernel.data)

    def extra_repr(self) ->str:
        """
        Return information in print-statement of layer
        
        Returns:
            str: information of print-statement of layer
        """
        return f'kernel_type={self.kernel_type}'

    def forward(self, emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of OuterProductNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, NC2), data_type = torch.float: output of OuterProductNetworkLayer
        """
        p = emb_inputs.rename(None)[:, self.row_idx]
        q = emb_inputs.rename(None)[:, self.col_idx]
        p.names = 'B', 'N', 'E'
        q.names = 'B', 'N', 'E'
        if self.kernel_type == 'mat':
            kp = p.unflatten('N', (('H', 1), ('N', p.size('N')))) * self.kernel
            kp = kp.sum(dim='E').align_to('B', 'N', 'H').rename(H='E')
            outputs = (kp * q).sum(dim='E')
        else:
            outputs = (p * q * self.kernel).sum(dim='E')
        outputs.names = 'B', 'O'
        return outputs


class PositionEmbeddingLayer(BaseLayer):
    """
    Layer class of Position Embedding

    Position Embedding was used in Personalized Re-ranking Model :title:`Changhua Pei et al, 2019`[1], which is to
    add a trainable tensors per position to the session-based embedding features tensor.

    :Reference:

    `Changhua Pei et al., 2019. Personalized Re-ranking for Recommendation <https://arxiv.org/abs/1904.06813>`_.

    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'L', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'L', 'E')}

    def __init__(self, max_num_position: 'int'):
        """
        Initialize PositionEmbedding
        
        Args:
            max_num_position (int): maximum number of position in a sequence
        """
        super().__init__()
        self.bias = nn.Parameter(torch.Tensor(1, max_num_position, 1))
        nn.init.normal_(self.bias)

    def forward(self, session_embed_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of PositionEmbedding
        
        Args:
            session_embed_inputs (T), shape = (B, L, E), data_type = torch.float: embedded feature tensors of session
        
        Returns:
            T, shape = (B, L, E), data_type = torch.float: output of PositionEmbedding
        """
        return session_embed_inputs + self.bias


class PositionBiasAwareLearningFrameworkLayer(BaseLayer):
    """
    TODO: missing documentation of this layer.
    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'feature_tensors': ('B', 'E'), 'session_position': ('B',)}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'E')}

    def __init__(self, input_size: 'int', max_num_position: 'int'):
        """
        TODO: missing documentation of this layer.

        Args:
            input_size (int):
            max_num_position (int):
        """
        super().__init__()
        self.position_bias = nn.Embedding(max_num_position, input_size)

    def forward(self, position_embed_tensor: 'Tuple[torch.Tensor, torch.Tensor]') ->torch.Tensor:
        """
        Forward calculation of PositionBiasAwareLearningFrameworkLayer
        
        Args: position_embed_tensor ((T, T)), shape = ((B, E), (B, )), data_type = (torch.float, torch.long):
            embedded feature tensors of session and position of session in sequence.
        
        Returns:
            T, shape = (B, E), data_type = torch.float: output of PositionBiasAwareLearningFrameworkLayer
        """
        pos = position_embed_tensor[1].rename(None)
        position_embed_bias = self.position_bias(pos)
        return position_embed_tensor[0] + position_embed_bias


class WideLayer(BaseLayer):
    """
    Layer class of wide
    
    Wide is a stack of linear and dropout, used in calculation of linear relation frequently

    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'N', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'N', 'O')}

    def __init__(self, inputs_size: 'int', output_size: 'int', dropout_p: 'Optional[float]'=None):
        """
        Initialize WideLayer
        
        Args:
            inputs_size (int): size of inputs, i.e. size of embedding tensor
            output_size (int): output size of wide layer
            dropout_p (float, optional): probability of Dropout in wide layer. Defaults to None
        """
        super().__init__()
        self.embed_size = inputs_size
        self.model = nn.Sequential()
        self.model.add_module('Linear', nn.Linear(inputs_size, output_size))
        if dropout_p is not None:
            self.model.add_module('Dropout', nn.Dropout(dropout_p))

    def forward(self, emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of WideLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, N, E), data_type = torch.float: output of wide layer
        """
        outputs = self.model(emb_inputs.rename(None))
        if outputs.dim() == 2:
            outputs.names = 'B', 'O'
        elif outputs.dim() == 3:
            outputs.names = 'B', 'N', 'O'
        return outputs


class GeneralizedMatrixFactorizationLayer(BaseLayer):
    """
    Layer class of Matrix Factorization (MF).
    
    Matrix Factorization is to calculate matrix factorization in a general linear format, which is used in
    Neural Collaborative Filtering to calculate dot product between user tensors and items tensors.
    
    Reference:

    #. `Xiangnan He et al, 2017. Neural Collaborative Filtering <https://arxiv.org/abs/1708.05031>`_.
    
    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', '2', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', '1')}

    def __init__(self):
        """
        Initialize GeneralizedMatrixFactorizationLayer
        """
        super().__init__()

    @staticmethod
    def forward(emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of GeneralizedMatrixFactorizationLayer
        
        Args:
            emb_inputs (T), shape = (B, 2, E), data_type = torch.float: embedded features tensors
                of users and items.
        
        Returns:
            T, shape = (B, 1), data_type = torch.float: output of GeneralizedMatrixFactorizationLayer
        """
        emb_inputs.names = 'B', 'N', 'E'
        outputs = (emb_inputs[:, 0, :] * emb_inputs[:, 1, :]).sum(dim='E', keepdim=True)
        outputs.names = 'B', 'O'
        return outputs


class StarSpaceLayer(BaseLayer):
    """
    Layer class of Starspace.
    
    StarSpace by Ledell Wu et al., 2017 is proposed by Facebook in 2017.

    It was implemented in C++ originally for a general purpose to embed different kinds of 
    relations between different pairs, like (word, tag), (user-group) etc. Starspace is 
    calculated in the following way: 

    #. calculate similarity between context and positive samples or negative samples 
    
    #. calculate margin ranking loss between similarity of positive samples and those of negative 
    samples 
    
    :Reference:

    #. `Ledell Wu et al, 2017 StarSpace: Embed All The Things! <https://arxiv.org/abs/1709.03856>`_.

    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', '2', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'E')}

    def __init__(self, similarity: 'Callable[[torch.Tensor, torch.Tensor], torch.Tensor]'):
        """
        Initialize StarSpaceLayer
        
        Args:
            similarity (Callable[[T, T], T]): function of similarity between two tensors.
                e.g. torch.nn.functional.cosine_similarity
        """
        super().__init__()
        self.similarity = similarity

    def extra_repr(self) ->str:
        """
        Return information in print-statement of layer.
        
        Returns:
            str: information of print-statement of layer.
        """
        return f"similarity={self.similarity.__qualname__.split('.')[-1].lower()}"

    def forward(self, samples_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of StarSpaceLayer
        
        Args:
            samples_inputs (T), shape = (B, N = 2, E), data_type = torch.float: embedded features tensors of context
                and target
        
        Returns:
            T, shape = (B, E), data_type = torch.float: output of StarSpaceLayer
        """
        samples_inputs.names = 'B', 'N', 'E'
        context = samples_inputs[:, 0, :]
        context = context.unflatten('E', (('N', 1), ('E', context.size('E'))))
        target = samples_inputs[:, 1, :]
        target = target.unflatten('E', [('N', 1), ('E', context.size('E'))])
        context = context.rename(None)
        target = target.rename(None)
        outputs = self.similarity(context, target)
        outputs.names = 'B', 'O'
        return outputs


def regularize(parameters: 'List[Tuple[str, nn.Parameter]]', weight_decay: 'float'=0.01, norm: 'int'=2) ->torch.Tensor:
    """
    Calculate p-th order regularization
    
    Args:
        parameters (List[Tuple[str, nn.Parameter]]): parameters to calculate the regularized loss
        weight_decay (float, optional): multiplier of regularized loss. Defaults to 0.01
        norm (int, optional): order of norm to calculate regularized loss. Defaults to 2
    
    Returns:
        T, data_type=torch.float: regularized loss
    """
    loss = 0.0
    for name, param in parameters:
        if 'weight' in name:
            loss += torch.norm(param, p=norm)
    return torch.Tensor([loss * weight_decay]).data[0]


class Regularizer(nn.Module):
    """
    Module of Regularizer
    """

    def __init__(self, weight_decay: 'float'=0.01, norm: 'int'=2):
        """
        Initialize the regularizer model
        
        Args:
            weight_decay (float, optional): multiplier of regularized loss. Defaults to 0.01
            norm (int, optional): order of norm to calculate regularized loss. Defaults to 2
        """
        super().__init__()
        self.weight_decay = weight_decay
        self.norm = norm

    def extra_repr(self) ->str:
        """
        Return information in print-statement of layer
        
        Returns:
            str: Information of print-statement of layer
        """
        return f'weight_decay={self.weight_decay}, norm={self.norm}'

    def forward(self, parameters: 'List[Tuple[str, nn.Parameter]]') ->torch.Tensor:
        """
        Forward calculation of Regularizer
        
        Args:
            parameters (List[Tuple[str, nn.Parameter]]): list of tuple of names and parameters to calculate
                the regularized loss
        
        Returns:
            T, shape = (1, ), data_type = torch.float: regularized loss
        """
        return regularize(parameters, self.weight_decay, self.norm)


class Loss(nn.Module, ABC):
    """
    General Loss class
    """

    def __init__(self):
        """
        Initialize Loss
        """
        super().__init__()


class BaseMiner(nn.Module, ABC):
    """

    """

    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super().__init__()

    @abstractmethod
    def forward(self, anchor: 'torch.Tensor', target: 'torch.Tensor') ->Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raise NotImplemented


class UniformBatchMiner(BaseMiner):
    """

    """

    def __init__(self, sample_size: 'int'):
        super().__init__()
        self.sample_size = sample_size

    def forward(self, anchor: 'Dict[str, torch.Tensor]', target: 'Dict[str, torch.Tensor]') ->Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        any_key = list(target.keys())[0]
        any_value = target[any_key]
        batch_size = any_value.size(0)
        rand_idx = torch.randint(0, batch_size, (self.sample_size * batch_size,))
        neg_samples = {k: v[rand_idx].unsqueeze(1) for k, v in target.items()}
        pos_samples = {k: v.unsqueeze(1) for k, v in target.items()}
        pos = {}
        neg = {}
        for k, v in anchor.items():
            pos_v = v.unsqueeze(1)
            neg_v = torch.repeat_interleave(pos_v, self.sample_size, dim=0)
            pos[k] = torch.cat([pos_v, pos_samples[k]], dim=1)
            neg[k] = torch.cat([neg_v, neg_samples[k]], dim=1)
        return pos, neg


class BaseModel(nn.Module, ABC):

    def __init__(self):
        super().__init__()


class EmbBaseModel(BaseModel, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, *args, **kwargs) ->torch.Tensor:
        raise NotImplemented


class MatrixFactorizationModel(EmbBaseModel):
    """
    Model class of Matrix Factorization (MF).
    
    Matrix Factorization is to embed relations between paris of data, like user and item.

    """

    def __init__(self):
        """
        Initialize MatrixFactorizationModel
        """
        super().__init__()
        self.mf = GeneralizedMatrixFactorizationLayer()

    def forward(self, emb_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of MatrixFactorizationModel
        
        Args:
            emb_inputs (T), shape = (B, 2, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, 1), data_type = torch.float: output of MatrixFactorizationModel
        """
        outputs = self.mf(emb_inputs)
        outputs = outputs.rename(None)
        return outputs

    def predict(self, *args, **kwargs) ->torch.Tensor:
        raise NotImplemented


def inner_product_similarity(a: 'torch.Tensor', b: 'torch.Tensor', dim: 'int'=1) ->torch.Tensor:
    """
    Calculate inner product of two vectors
    
    Args:
        a (T, shape = (B, N_{a}, E)), data_type=torch.float: the first batch of vector to be multiplied
        b (T, shape = (B, N_{b}, E)), data_type=torch.float: the second batch of vector to be multiplied
        dim (int): dimension to sum the tensor
    
    Returns:
        T, data_type=torch.float: inner product tensor
    """
    return (a * b).sum(dim=dim)


class StarSpaceModel(EmbBaseModel):
    """
    Model class of StatSpaceModel.

    Starspace is a model proposed by Facebook in 2017, which to embed relationship between different kinds of pairs,
    like (word, tag), (user-group) etc, by calculating similarity between context and positive samples or negative
    samples, and margin ranking loss between similarities.

    :Reference:

    #. `Ledell Wu et al, 2017 StarSpace: Embed All The Things! <https://arxiv.org/abs/1709.03856>`_.
    
    """

    def __init__(self, embed_size: 'int', num_neg: 'int', similarity: 'Any'=partial(inner_product_similarity, dim=2)):
        """
        Initialize StarSpaceModel
        
        Args:
            embed_size (int): size of embedding tensor
            num_neg (int): number of negative samples
            similarity (Any, optional): function of similarity between two tensors.
                e.g. torch.nn.functional.cosine_similarity. Defaults to partial(inner_product_similarity, dim=2)
        """
        super().__init__()
        self.embed_size = embed_size
        self.num_neg = num_neg
        self.starspace = StarSpaceLayer(similarity)

    def forward(self, context_inputs: 'torch.Tensor', target_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of StarSpaceModel
        
        Args:
            context_inputs (T), shape = (B * (1 + N Neg), 1, E), data_type = torch.float:
                embedded features tensors of context
            target_inputs (T), shape = (B * (1 + N Neg), 1, E), data_type = torch.float:
                embedded features tensors of target
        
        Returns:
            T, shape = (B, 1 / N Neg): Output of StarSpaceModel.
        """
        context_inputs.names = 'B', 'N', 'E'
        agg_batch_size = context_inputs.size('B')
        batch_size = int(agg_batch_size // (1 + self.num_neg))
        context_inputs = context_inputs.rename(None).view(batch_size, self.num_neg + 1, self.embed_size)
        context_inputs.names = 'B', 'N', 'E'
        target_inputs = target_inputs.rename(None).view(batch_size, self.num_neg + 1, self.embed_size)
        target_inputs.names = 'B', 'N', 'E'
        context_inputs_pos = context_inputs[:, 0, :].unflatten('E', (('N', 1), ('E', context_inputs.size('E'))))
        context_inputs_neg = context_inputs[:, 1:, :].rename(None).contiguous()
        context_inputs_neg = context_inputs_neg.view(batch_size * self.num_neg, 1, self.embed_size)
        context_inputs_neg.names = context_inputs_pos.names
        target_inputs_pos = target_inputs[:, 0, :].unflatten('E', (('N', 1), ('E', context_inputs.size('E'))))
        target_inputs_neg = target_inputs[:, 1:, :].rename(None).contiguous()
        target_inputs_neg = target_inputs_neg.view(batch_size * self.num_neg, 1, self.embed_size)
        target_inputs_neg.names = target_inputs_pos.names
        positive_tensor = torch.cat([context_inputs_pos, target_inputs_pos], dim='N')
        pos_sim = self.starspace(positive_tensor)
        negative_tensor = torch.cat([context_inputs_neg, target_inputs_neg], dim='N')
        neg_sim = self.starspace(negative_tensor)
        pos_sim = pos_sim.rename(None).view(batch_size, 1)
        pos_sim.names = 'B', 'O'
        neg_sim = neg_sim.rename(None).view(batch_size, self.num_neg)
        neg_sim.names = 'B', 'O'
        outputs = torch.cat([pos_sim, neg_sim], dim='O')
        outputs = outputs.rename(None).view(batch_size * (1 + self.num_neg), 1)
        outputs.names = 'B', 'O'
        outputs = outputs.rename(None)
        return outputs

    def predict(self, *args, **kwargs) ->torch.Tensor:
        raise NotImplemented


class Sequential(nn.Module):
    """
    Sequential container, where the model of embeddings and model will be stacked in the order they are passed to
    the constructor
    """

    def __init__(self, inputs: 'BaseInput', model: 'nn.Module'):
        """
        Initialize Sequential container
        
        Args:
            inputs (BaseInput): inputs where the return is a dictionary of inputs' tensors which are passed to
                the model directly
            model (nn.Module): model class to be trained and used in prediction
        """
        super().__init__()
        self._inputs = inputs
        self._model = model

    def forward(self, inputs: 'Dict[str, torch.Tensor]') ->torch.Tensor:
        """
        Forward calculation of Sequential
        
        Args:
            inputs (Dict[str, T]): dictionary of inputs,
                where key is string of input fields' name, and value is torch.tensor pass to Input class
        
        Returns:
            torch.Tensor: output of model
        """
        inputs = self._inputs(inputs)
        outputs = self._model(**inputs)
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentionalFactorizationMachineLayer,
     lambda: ([], {'embed_size': 4, 'num_fields': 4, 'attn_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BilinearInteractionLayer,
     lambda: ([], {'embed_size': 4, 'num_fields': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BilinearNetworkLayer,
     lambda: ([], {'inputs_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CrossNetworkLayer,
     lambda: ([], {'inputs_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (DynamicRoutingLayer,
     lambda: ([], {'embed_size': 4, 'routed_size': 4, 'max_num_caps': 4, 'num_iter': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (FactorizationMachineLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (FieldAllTypeBilinear,
     lambda: ([], {'in1_features': 4, 'in2_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FieldEachTypeBilinear,
     lambda: ([], {'in_features': 4, 'in1_features': 4, 'in2_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (GeneralizedMatrixFactorizationLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (InnerProductNetworkLayer,
     lambda: ([], {'num_fields': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MatrixFactorizationModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (OuterProductNetworkLayer,
     lambda: ([], {'embed_size': 4, 'num_fields': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PositionEmbeddingLayer,
     lambda: ([], {'max_num_position': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SingleIndexEmbedding,
     lambda: ([], {'embed_size': 4, 'field_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ValueInput,
     lambda: ([], {'num_fields': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (WideLayer,
     lambda: ([], {'inputs_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

