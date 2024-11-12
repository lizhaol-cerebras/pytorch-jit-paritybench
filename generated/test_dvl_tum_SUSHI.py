
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


import math


import time


import numpy as np


from torch.nn.utils.rnn import pad_sequence


import torch.nn.functional as F


from numpy import pad


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from collections import OrderedDict


from torch.nn import functional as F


from typing import OrderedDict


import pandas as pd


from torch import nn


from copy import deepcopy


import warnings


import torch.utils.model_zoo as model_zoo


import torch.optim as optim


from scipy.sparse import csr_matrix


from scipy.sparse.csgraph import connected_components


import matplotlib.pyplot as plt


from torch.utils.tensorboard.writer import SummaryWriter


import random


class HICLNet(nn.Module):
    """
    Hierarchical network that contains all layers
    """

    def __init__(self, submodel_type, submodel_params, hicl_depth, use_motion, use_reid_edge, use_pos_edge, share_weights, edge_level_embed, node_level_embed):
        """
        :param model_type: Network to use at each layer
        :param model_params: Parameters of the model for each layer
        :param depth: Number of layers in the hierarchical model
        """
        super(HICLNet, self).__init__()
        for per_layer_params in (use_motion, use_reid_edge, use_pos_edge):
            assert hicl_depth == len(per_layer_params), f'{hicl_depth}, {per_layer_params}'
        assert share_weights in ('none', 'all_but_first', 'all')
        _SHARE_WEIGHTS_IDXS = {'none': range(hicl_depth), 'all_but_first': [0] + (hicl_depth - 1) * [1], 'all': hicl_depth * [0]}
        layer_idxs = _SHARE_WEIGHTS_IDXS[share_weights]
        layers = [submodel_type(submodel_params, motion=motion, pos_feats=pos_feats, reid=reid) for motion, pos_feats, reid in zip(use_motion, use_pos_edge, use_reid_edge)]
        self.layers = nn.ModuleList([layers[idx] for idx in layer_idxs])
        if edge_level_embed:
            edge_dim = submodel_params['encoder_feats_dict']['edge_out_dim']
            self.edge_level_embed = nn.Embedding(hicl_depth, edge_dim)
        else:
            self.edge_level_embed = None
        if node_level_embed:
            node_dim = submodel_params['encoder_feats_dict']['node_out_dim']
            self.node_level_embed = nn.Embedding(hicl_depth, node_dim)
        else:
            self.node_level_embed = None

    def forward(self, data, ix_layer):
        """
        Forward pass with the self.layers[ix_layer]
        """
        edge_level_embed = node_level_embed = None
        if self.edge_level_embed is not None:
            edge_level_embed = self.edge_level_embed.weight[ix_layer]
        if self.node_level_embed is not None:
            node_level_embed = self.node_level_embed.weight[ix_layer]
        return self.layers[ix_layer](data, node_level_embed=node_level_embed, edge_level_embed=edge_level_embed)


class MLP(nn.Module):

    def __init__(self, input_dim, fc_dims, dropout_p=0.4, use_batchnorm=False):
        super(MLP, self).__init__()
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))
            if dim != 1:
                layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc_layers(input)


class MetaLayer(torch.nn.Module):
    """
    Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
    (https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html)
    """

    def __init__(self, edge_model=None, node_model=None):
        """
        Args:
            edge_model: Callable Edge Update Model
            node_model: Callable Node Update Model
        """
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)

        Returns: Updated Node and Edge Feature matrices

        """
        row, col = edge_index
        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr)
        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr)
        return x, edge_attr

    def __repr__(self):
        return '{}(edge_model={}, node_model={})'.format(self.__class__.__name__, self.edge_model, self.node_model)


class EdgeModel(nn.Module):
    """
    Class used to peform the edge update during Neural message passing
    """

    def __init__(self, edge_mlp):
        super(EdgeModel, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, source, target, edge_attr):
        out = torch.cat([source, target, edge_attr], dim=1)
        return self.edge_mlp(out)


class TimeAwareNodeModel(nn.Module):
    """
    Class used to peform the node update during Neural mwssage passing
    """

    def __init__(self, flow_in_mlp, flow_out_mlp, node_mlp, node_agg_fn):
        super(TimeAwareNodeModel, self).__init__()
        self.flow_in_mlp = flow_in_mlp
        self.flow_out_mlp = flow_out_mlp
        self.node_mlp = node_mlp
        self.node_agg_fn = node_agg_fn

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        flow_out_mask = row < col
        flow_out_row, flow_out_col = row[flow_out_mask], col[flow_out_mask]
        flow_out_input = torch.cat([x[flow_out_col], edge_attr[flow_out_mask]], dim=1)
        flow_out = self.flow_out_mlp(flow_out_input)
        flow_out = self.node_agg_fn(flow_out, flow_out_row, x.size(0))
        flow_in_mask = row > col
        flow_in_row, flow_in_col = row[flow_in_mask], col[flow_in_mask]
        flow_in_input = torch.cat([x[flow_in_col], edge_attr[flow_in_mask]], dim=1)
        flow_in = self.flow_in_mlp(flow_in_input)
        flow_in = self.node_agg_fn(flow_in, flow_in_row, x.size(0))
        flow = torch.cat((flow_in, flow_out), dim=1)
        return self.node_mlp(flow)


class MLPGraphIndependent(nn.Module):
    """
    Class used to to encode (resp. classify) features before (resp. after) neural message passing.
    It consists of two MLPs, one for nodes and one for edges, and they are applied independently to node and edge
    features, respectively.

    This class is based on: https://github.com/deepmind/graph_nets tensorflow implementation.
    """

    def __init__(self, edge_in_dim=None, node_in_dim=None, edge_out_dim=None, node_out_dim=None, node_fc_dims=None, edge_fc_dims=None, dropout_p=None, use_batchnorm=None):
        super(MLPGraphIndependent, self).__init__()
        if node_in_dim is not None:
            self.node_mlp = MLP(input_dim=node_in_dim, fc_dims=list(node_fc_dims) + [node_out_dim], dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.node_mlp = None
        if edge_in_dim is not None:
            self.edge_mlp = MLP(input_dim=edge_in_dim, fc_dims=list(edge_fc_dims) + [edge_out_dim], dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.edge_mlp = None

    def forward(self, edge_feats=None, nodes_feats=None):
        if self.node_mlp is not None:
            out_node_feats = self.node_mlp(nodes_feats)
        else:
            out_node_feats = nodes_feats
        if self.edge_mlp is not None:
            out_edge_feats = self.edge_mlp(edge_feats)
        else:
            out_edge_feats = edge_feats
        return out_edge_feats, out_node_feats


class HiclFeatsEncoder(nn.Module):

    def __init__(self, node_dim, detach_hicl_grad, merge_method='cat', skip_conn=False, ignore_mpn_out=False, use_layerwise=False):
        super().__init__()
        assert merge_method in ('cat', 'sum')
        self.merge_method = merge_method
        self.ignore_mpn_out = ignore_mpn_out
        self.skip_conn = skip_conn
        self.detach_hicl_grad = detach_hicl_grad
        self.encoder_hicl_feats_post_mpn = nn.Sequential(*[nn.Linear(node_dim, node_dim), nn.ReLU(), nn.Linear(node_dim, node_dim), nn.ReLU()])
        self.encoder_hicl_feats = nn.Sequential(*[nn.Linear(node_dim, node_dim), nn.ReLU(), nn.Linear(node_dim, node_dim), nn.ReLU()])
        self.merge_skip_conn = nn.Sequential(*[nn.Linear(2 * node_dim, 2 * node_dim), nn.ReLU(), nn.Linear(2 * node_dim, node_dim), nn.ReLU()])
        if self.merge_method == 'cat':
            self.merge_hicl_feats = nn.Sequential(*[nn.Linear(2 * node_dim, 2 * node_dim), nn.ReLU(), nn.Linear(2 * node_dim, node_dim), nn.ReLU()])
        else:
            self.merge_hicl_feats = None
        self.use_layerwise = use_layerwise
        if self.use_layerwise:
            self.layerwise_merge = nn.Linear(2 * node_dim, node_dim)

    def pool_node_feats(self, node_feats, labels):
        return scatter_mean(node_feats, torch.as_tensor(labels, device=node_feats.device).long(), dim=0)

    def forward(self, latent_node_feats, hicl_feats):
        hicl_feats = self.encoder_hicl_feats(hicl_feats)
        if self.use_layerwise:
            hicl_feats = torch.cat((hicl_feats, latent_node_feats), dim=1)
            hicl_feats = self.layerwise_merge(hicl_feats)
        return hicl_feats

    def post_mpn_encode_node_feats(self, latent_node_feats, initial_hicl_feats, initial_node_feats):
        if initial_hicl_feats is None:
            initial_hicl_feats = initial_node_feats
        if self.ignore_mpn_out:
            return initial_hicl_feats
        if self.detach_hicl_grad:
            latent_node_feats = latent_node_feats.detach()
        latent_node_feats = self.encoder_hicl_feats_post_mpn(latent_node_feats)
        if self.skip_conn and initial_hicl_feats is not None:
            latent_node_feats = torch.cat((latent_node_feats, initial_hicl_feats), dim=1)
            latent_node_feats = self.merge_skip_conn(latent_node_feats)
        return latent_node_feats


class MOTMPNet(nn.Module):
    """
    Main Model Class. Contains all the components of the model. It consists of of several networks:
    - 2 encoder MLPs (1 for nodes, 1 for edges) that provide the initial node and edge embeddings, respectively,
    - 4 update MLPs (3 for nodes, 1 per edges used in the 'core' Message Passing Network
    - 1 edge classifier MLP that performs binary classification over the Message Passing Network's output.

    This class was initially based on: https://github.com/deepmind/graph_nets tensorflow implementation.
    """

    def __init__(self, model_params, bb_encoder=None, motion=None, pos_feats=None, reid=None):
        """
        Defines all components of the model
        Args:
            bb_encoder: (might be 'None') CNN used to encode bounding box apperance information.
            model_params: dictionary contaning all model hyperparameters
        """
        super(MOTMPNet, self).__init__()
        self.node_cnn = bb_encoder
        self.model_params = model_params
        encoder_feats_dict = deepcopy(model_params['encoder_feats_dict'])
        if motion:
            encoder_feats_dict['edge_in_dim'] += 1
        if not reid:
            encoder_feats_dict['edge_in_dim'] -= 1
        if not pos_feats:
            encoder_feats_dict['edge_in_dim'] -= 4
        classifier_feats_dict = model_params['classifier_feats_dict']
        self.encoder = MLPGraphIndependent(**encoder_feats_dict)
        self.classifier = MLPGraphIndependent(**classifier_feats_dict)
        self.MPNet = self._build_core_MPNet(model_params=model_params, encoder_feats_dict=encoder_feats_dict)
        self.num_enc_steps = model_params['num_enc_steps']
        self.num_class_steps = model_params['num_class_steps']
        if model_params['do_hicl_feats']:
            self.hicl_feats_encoder = HiclFeatsEncoder(node_dim=encoder_feats_dict['node_out_dim'], **model_params['hicl_feats_encoder'])
        else:
            self.hicl_feats_encoder = None

    def _build_core_MPNet(self, model_params, encoder_feats_dict):
        """
        Builds the core part of the Message Passing Network: Node Update and Edge Update models.
        Args:
            model_params: dictionary contaning all model hyperparameters
            encoder_feats_dict: dictionary containing the hyperparameters for the initial node/edge encoder
        """
        node_agg_fn = model_params['node_agg_fn']
        assert node_agg_fn.lower() in ('mean', 'max', 'sum'), "node_agg_fn can only be 'max', 'mean' or 'sum'."
        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size: scatter_mean(out, row, dim=0, dim_size=x_size)
        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size: scatter_max(out, row, dim=0, dim_size=x_size)[0]
        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size: scatter_add(out, row, dim=0, dim_size=x_size)
        self.reattach_initial_nodes = model_params['reattach_initial_nodes']
        self.reattach_initial_edges = model_params['reattach_initial_edges']
        edge_factor = 2 if self.reattach_initial_edges else 1
        node_factor = 2 if self.reattach_initial_nodes else 1
        edge_model_in_dim = node_factor * 2 * encoder_feats_dict['node_out_dim'] + edge_factor * encoder_feats_dict['edge_out_dim']
        node_model_in_dim = node_factor * encoder_feats_dict['node_out_dim'] + encoder_feats_dict['edge_out_dim']
        edge_model_feats_dict = model_params['edge_model_feats_dict']
        node_model_feats_dict = model_params['node_model_feats_dict']
        edge_mlp = MLP(input_dim=edge_model_in_dim, fc_dims=edge_model_feats_dict['fc_dims'], dropout_p=edge_model_feats_dict['dropout_p'], use_batchnorm=edge_model_feats_dict['use_batchnorm'])
        flow_in_mlp = MLP(input_dim=node_model_in_dim, fc_dims=node_model_feats_dict['fc_dims'], dropout_p=node_model_feats_dict['dropout_p'], use_batchnorm=node_model_feats_dict['use_batchnorm'])
        flow_out_mlp = MLP(input_dim=node_model_in_dim, fc_dims=node_model_feats_dict['fc_dims'], dropout_p=node_model_feats_dict['dropout_p'], use_batchnorm=node_model_feats_dict['use_batchnorm'])
        node_mlp = nn.Sequential(*[nn.Linear(2 * node_model_feats_dict['fc_dims'][-1], encoder_feats_dict['node_out_dim']), nn.ReLU(inplace=True)])
        return MetaLayer(edge_model=EdgeModel(edge_mlp=edge_mlp), node_model=TimeAwareNodeModel(flow_in_mlp=flow_in_mlp, flow_out_mlp=flow_out_mlp, node_mlp=node_mlp, node_agg_fn=node_agg_fn))

    def forward(self, data, edge_level_embed=None, node_level_embed=None):
        """
        Provides a fractional solution to the data association problem.
        First, node and edge features are independently encoded by the encoder network. Then, they are iteratively
        'combined' for a fixed number of steps via the Message Passing Network (self.MPNet). Finally, they are
        classified independently by the classifiernetwork.
        Args:
            data: object containing attribues
              - x: node features matrix
              - edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
                graph adjacency (i.e. edges) (i.e. sparse adjacency)
              - edge_attr: edge features matrix (sorted by edge apperance in edge_index)

        Returns:
            classified_edges: list of unnormalized node probabilites after each MP step
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_is_img = len(x.shape) == 4
        if self.node_cnn is not None and x_is_img:
            x = self.node_cnn(x)
            emb_dists = nn.functional.pairwise_distance(x[edge_index[0]], x[edge_index[1]]).view(-1, 1)
            edge_attr = torch.cat((edge_attr, emb_dists), dim=1)
        latent_edge_feats, latent_node_feats = self.encoder(edge_attr, x)
        if node_level_embed is not None:
            n_nodes = latent_node_feats.shape[0]
            node_embed = node_level_embed.unsqueeze(0).expand(n_nodes, -1)
            latent_node_feats = node_embed
            None
        if edge_level_embed is not None:
            n_edges = latent_edge_feats.shape[0]
            edge_embed = edge_level_embed.unsqueeze(0).expand(n_edges, -1)
            latent_edge_feats = latent_edge_feats + edge_embed
        if hasattr(data, 'hicl_feats') and data.hicl_feats is not None and self.hicl_feats_encoder is not None:
            hicl_feats = data.hicl_feats
            latent_node_feats = self.hicl_feats_encoder(latent_node_feats, hicl_feats)
        else:
            hicl_feats = None
        initial_edge_feats = latent_edge_feats
        initial_node_feats = latent_node_feats
        first_class_step = self.num_enc_steps - self.num_class_steps + 1
        outputs_dict = {'classified_edges': []}
        for step in range(1, self.num_enc_steps + 1):
            if self.reattach_initial_edges:
                latent_edge_feats = torch.cat((initial_edge_feats, latent_edge_feats), dim=1)
            if self.reattach_initial_nodes:
                latent_node_feats = torch.cat((initial_node_feats, latent_node_feats), dim=1)
            latent_node_feats, latent_edge_feats = self.MPNet(latent_node_feats, edge_index, latent_edge_feats)
            if step >= first_class_step:
                dec_edge_feats, _ = self.classifier(latent_edge_feats)
                outputs_dict['classified_edges'].append(dec_edge_feats)
        if self.num_enc_steps == 0:
            dec_edge_feats, _ = self.classifier(latent_edge_feats)
            outputs_dict['classified_edges'].append(dec_edge_feats)
        if self.hicl_feats_encoder is not None:
            outputs_dict['node_feats'] = self.hicl_feats_encoder.post_mpn_encode_node_feats(latent_node_feats, hicl_feats, initial_node_feats)
        return outputs_dict


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """Residual network.

    Reference:
        - He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
        - Xie et al. Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017.

    Public keys:
        - ``resnet18``: ResNet18.
        - ``resnet34``: ResNet34.
        - ``resnet50``: ResNet50.
        - ``resnet101``: ResNet101.
        - ``resnet152``: ResNet152.
        - ``resnext50_32x4d``: ResNeXt50.
        - ``resnext101_32x8d``: ResNeXt101.
        - ``resnet50_fc512``: ResNet50 + FC.
    """

    def __init__(self, num_classes, loss, block, layers, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, last_stride=2, fc_dims=None, dropout_p=None, **kwargs):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.loss = loss
        self.feature_dim = 512 * block.expansion
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride, dilate=replace_stride_with_dilation[2])
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self._construct_fc_layer(fc_dims, 512 * block.expansion, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()
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

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            return v, self.fc(v)
        return v, v


class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EdgeModel,
     lambda: ([], {'edge_mlp': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (HiclFeatsEncoder,
     lambda: ([], {'node_dim': 4, 'detach_hicl_grad': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_dim': 4, 'fc_dims': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLPGraphIndependent,
     lambda: ([], {}),
     lambda: ([], {})),
    (MetaLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), (torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])), torch.rand([4, 4])], {})),
]

