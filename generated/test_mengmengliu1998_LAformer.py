
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


import abc


import math


import numpy as np


from typing import Union


import torch.utils.data as torch_data


import random


import torch


import pandas as pd


import logging


from functools import partial


from torch.utils.data import SequentialSampler


from typing import Dict


from typing import List


from torch import nn


from typing import Tuple


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


from typing import NamedTuple


from typing import Any


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data.distributed import DistributedSampler


import inspect


import time


from collections import defaultdict


from random import randint


import matplotlib as mpl


import matplotlib.pyplot as plt


from matplotlib.path import Path


from matplotlib.pyplot import MultipleLocator


class LayerNorm(nn.Module):
    """
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-05):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP(nn.Module):

    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.nn.functional.relu(hidden_states)
        return hidden_states


class DecoderResCat(nn.Module):

    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class LaplaceNLLLoss(nn.Module):

    def __init__(self, eps: 'float'=1e-06, reduction: 'str'='mean') ->None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))


class SoftTargetCrossEntropyLoss(nn.Module):

    def __init__(self, reduction: 'str'='mean') ->None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        elif self.reduction == 'none':
            return cross_entropy
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))


def get_angle_diff(gt_traj, pred_traj, past_traj):
    """
    :param gt_traj: [B, T, 2]
    :param pred_traj: [B, T, 2]
    :param past_traj: [F, B, T, 2]
    :return: angle_diff: [B, T]
    """
    top = 5
    gt_traj_angle = gt_traj[:, :, :] - past_traj[0, :, -1, :].unsqueeze(1)
    pred_traj_angle = pred_traj[:, :, :] - past_traj[0, :, -1, :].unsqueeze(1)
    angle_label = torch.atan2(gt_traj_angle[:, :, 1], gt_traj_angle[:, :, 0])
    angle_pred = torch.atan2(pred_traj_angle[:, :, 1], pred_traj_angle[:, :, 0])
    angle_diff = angle_label - angle_pred
    angle_loss = -1 * torch.cos(angle_diff).mean(dim=-1)
    return angle_loss


def init_weights(m: 'nn.Module') ->None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


class trajectory_refinement(nn.Module):

    def __init__(self, args):
        super(trajectory_refinement, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_modes = args.mode_num
        self.MLP = MLP(2, self.hidden_size)
        self.MLP_2 = MLP(self.hidden_size, self.hidden_size)
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)
        self.loc_delta = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size), nn.LayerNorm(self.hidden_size), nn.ReLU(inplace=True), nn.Linear(self.hidden_size, self.hidden_size), nn.LayerNorm(self.hidden_size), nn.ReLU(inplace=True), nn.Linear(self.hidden_size, 2))

    def forward(self, stage_one_out, full_traj, global_embed, local_embed):
        """two stage motion refinement module like faster rcnn and cascade rcnn
            parameters:stage_one_out: out feature embedding from GRU [F x N, H, D]
                        full_traj: past trajectory [F, N, H+T, 2]
                        local_embed: [1, F x N, D]
                        global_embed: [H, F x N, D]
        """
        sequence_length = full_traj.shape[2]
        full_traj_embed = self.MLP(full_traj)
        full_traj_embed = self.MLP_2(full_traj_embed) + full_traj_embed
        full_traj_embed = full_traj_embed.view(-1, full_traj.shape[2], self.hidden_size)
        full_traj_embed = full_traj_embed.permute(1, 0, 2)
        stage_two_out, _ = self.gru(full_traj_embed)
        stage_two_out = stage_two_out.permute(1, 0, 2)
        stage_two_out_pred = stage_two_out[:, int(sequence_length - self.args.future_frame_num):, :]
        pi = None
        global_embed = global_embed.permute(1, 0, 2)
        local_embed = local_embed.squeeze(0)
        local_embed = local_embed.expand(self.args.future_frame_num, *local_embed.shape)
        local_embed = local_embed.permute(1, 0, 2)
        loc_delta = self.loc_delta(torch.cat((stage_one_out, stage_two_out_pred, global_embed, local_embed), dim=-1))
        loc_delta = loc_delta.view(self.num_modes, -1, self.args.future_frame_num, 2)
        return loc_delta, pi


class GRUDecoder(nn.Module):

    def __init__(self, args, vectornet) ->None:
        super(GRUDecoder, self).__init__()
        min_scale: 'float' = 0.001
        self.input_size = args.hidden_size
        self.hidden_size = args.hidden_size
        self.future_steps = args.future_frame_num
        self.num_modes = args.mode_num
        self.min_scale = min_scale
        self.args = args
        self.dense = args.future_frame_num
        self.z_size = args.z_size
        self.smothl1 = torch.nn.SmoothL1Loss(reduction='none')
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)
        self.loc = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.LayerNorm(self.hidden_size), nn.ReLU(inplace=True), nn.Linear(self.hidden_size, 2))
        self.scale = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.LayerNorm(self.hidden_size), nn.ReLU(inplace=True), nn.Linear(self.hidden_size, 2))
        self.pi = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size), nn.LayerNorm(self.hidden_size), nn.ReLU(inplace=True), nn.Linear(self.hidden_size, self.hidden_size), nn.LayerNorm(self.hidden_size), nn.ReLU(inplace=True), nn.Linear(self.hidden_size, 1))
        self.aggregate_global_z = nn.Sequential(nn.Linear(self.hidden_size + 2, self.hidden_size), nn.LayerNorm(self.hidden_size), nn.ReLU(inplace=True))
        self.reg_loss = LaplaceNLLLoss(reduction='none')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='none')
        if 'step_lane_score' in args.other_params:
            self.multihead_proj_global = nn.Sequential(nn.Linear(self.hidden_size * 2, self.num_modes * self.hidden_size), nn.LayerNorm(self.num_modes * self.hidden_size), nn.ReLU(inplace=True))
            decoder_layer_dense_label = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=32, dim_feedforward=self.hidden_size)
            self.dense_label_cross_attention = nn.TransformerDecoder(decoder_layer_dense_label, num_layers=1)
            self.dense_lane_decoder = DecoderResCat(self.hidden_size, self.hidden_size * 3, out_features=self.dense)
            self.proj_topk = MLP(self.hidden_size + 1, self.hidden_size)
            decoder_layer_aggregation = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=32, dim_feedforward=self.hidden_size)
            self.aggregation_cross_att = nn.TransformerDecoder(decoder_layer_aggregation, num_layers=1)
        else:
            self.multihead_proj_global = nn.Sequential(nn.Linear(self.hidden_size, self.num_modes * self.hidden_size), nn.LayerNorm(self.num_modes * self.hidden_size), nn.ReLU(inplace=True))
        self.apply(init_weights)
        if 'stage_two' in args.other_params:
            if args.do_train:
                model_recover = torch.load(args.other_params['stage-two-train_recover'])
                vectornet.decoder = self
                utils.load_model(vectornet, model_recover)
            self.trajectory_refinement = trajectory_refinement(args)

    def dense_lane_aware(self, i, mapping, lane_states_batch, lane_states_length, element_hidden_states, element_hidden_states_lengths, global_hidden_states, device, loss):
        """dense lane aware
        Args:
            mapping (list): data mapping
            lane_states_batch (tensor): [max_len, N]
            lane_states_length (tensor): [N]
            element_hidden_states (tensor): [N]
            global_hidden_states (tensor): [N]
            device (device): device"""

        def dense_lane_scores():
            lane_states_batch_attention = lane_states_batch + self.dense_label_cross_attention(lane_states_batch, element_hidden_states.unsqueeze(0), tgt_key_padding_mask=src_attention_mask_lane)
            dense_lane_scores = self.dense_lane_decoder(torch.cat([global_hidden_states.unsqueeze(0).expand(lane_states_batch.shape), lane_states_batch, lane_states_batch_attention], dim=-1))
            dense_lane_scores = F.log_softmax(dense_lane_scores, dim=0)
            return dense_lane_scores
        max_vector_num = lane_states_batch.shape[1]
        batch_size = len(mapping)
        src_attention_mask_lane = torch.zeros([batch_size, lane_states_batch.shape[1]], device=device)
        for i in range(batch_size):
            assert lane_states_length[i] > 0
            src_attention_mask_lane[i, :lane_states_length[i]] = 1
        src_attention_mask_lane = src_attention_mask_lane == 0
        lane_states_batch = lane_states_batch.permute(1, 0, 2)
        dense_lane_pred = dense_lane_scores()
        dense_lane_pred = dense_lane_pred.permute(1, 0, 2)
        lane_states_batch = lane_states_batch.permute(1, 0, 2)
        dense = self.dense
        dense_lane_pred = dense_lane_pred.permute(0, 2, 1)
        dense_lane_pred = dense_lane_pred.contiguous().view(-1, max_vector_num)
        if self.args.do_train:
            dense_lane_targets = torch.zeros([batch_size, dense], device=device, dtype=torch.long)
            for i in range(batch_size):
                dense_lane_targets[i, :] = torch.tensor(np.array(mapping[i]['dense_lane_labels']), dtype=torch.long, device=device)
            loss_weight = self.args.lane_loss_weight
            dense_lane_targets = dense_lane_targets.view(-1)
            loss += loss_weight * F.nll_loss(dense_lane_pred, dense_lane_targets, reduction='none').view(batch_size, dense).sum(dim=1)
        mink = self.args.topk
        dense_lane_topk = torch.zeros((dense_lane_pred.shape[0], mink, self.hidden_size), device=device)
        dense_lane_topk_scores = torch.zeros((dense_lane_pred.shape[0], mink), device=device)
        for i in range(dense_lane_topk_scores.shape[0]):
            idxs_lane = i // dense
            k = min(mink, lane_states_length[idxs_lane])
            _, idxs_topk = torch.topk(dense_lane_pred[i], k)
            dense_lane_topk[i][:k] = lane_states_batch[idxs_lane, idxs_topk]
            dense_lane_topk_scores[i][:k] = dense_lane_pred[i][idxs_topk]
        dense_lane_topk = torch.cat([dense_lane_topk, dense_lane_topk_scores.unsqueeze(-1)], dim=-1)
        dense_lane_topk = dense_lane_topk.view(batch_size, dense * mink, self.hidden_size + 1)
        return dense_lane_topk

    def forward(self, mapping: 'List[Dict]', batch_size, lane_states_batch, lane_states_length, inputs: 'Tensor', inputs_lengths: 'List[int]', hidden_states: 'Tensor', device) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        :param global_embed: hidden states of agents after encoding by global graph (shape [batch_size, hidden_size])
        :param local_embed: hidden states of agents before encoding by global graph (shape [batch_size, hidden_size])
        :param lane_states_batch: hidden states of lanes (shape [batch_size, max_num_lanes, hidden_size])
        """
        labels = utils.get_from_mapping(mapping, 'labels')
        labels_is_valid = utils.get_from_mapping(mapping, 'labels_is_valid')
        loss = torch.zeros(batch_size, device=device)
        DE = np.zeros([batch_size, self.future_steps])
        local_embed = inputs[:, 0, :]
        global_embed = hidden_states[:, 0, :]
        if 'step_lane_score' in self.args.other_params:
            dense_lane_topk = self.dense_lane_aware(0, mapping, lane_states_batch, lane_states_length, local_embed, inputs_lengths, global_embed, device, loss)
            dense_lane_topk = dense_lane_topk.permute(1, 0, 2)
            dense_lane_topk = self.proj_topk(dense_lane_topk)
            global_embed_att = global_embed + self.aggregation_cross_att(global_embed.unsqueeze(0), dense_lane_topk).squeeze(0)
            global_embed = torch.cat([global_embed, global_embed_att], dim=-1)
        local_embed = local_embed.repeat(self.num_modes, 1, 1)
        global_embed = self.multihead_proj_global(global_embed).view(-1, self.num_modes, self.hidden_size)
        batch_size = global_embed.shape[0]
        global_embed = global_embed.transpose(0, 1)
        pi = self.pi(torch.cat((local_embed, global_embed), dim=-1)).squeeze(-1).t()
        global_embed = global_embed.reshape(-1, self.input_size)
        z_size = self.z_size
        z = torch.randn(self.num_modes * batch_size, z_size, device=device)
        global_embed = torch.cat([global_embed, z], dim=-1)
        global_embed = self.aggregate_global_z(global_embed)
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)
        local_embed = local_embed.reshape(-1, self.input_size).unsqueeze(0)
        out, _ = self.gru(global_embed, local_embed)
        out = out.transpose(0, 1)
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)
        scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale
        scale = scale.view(self.num_modes, -1, self.future_steps, 2)
        if 'stage_two' in self.args.other_params:
            past_traj = utils.get_from_mapping(mapping, 'past_traj')
            past_traj = torch.tensor(np.array(past_traj), dtype=torch.float32, device=device)
            past_traj = past_traj[:, :, :2]
            past_traj = past_traj.expand(self.num_modes, *past_traj.shape)
            full_traj = torch.cat((past_traj, loc), dim=2)
            loc_delta, _ = self.trajectory_refinement(out, full_traj, global_embed, local_embed)
        if 'stage_two' in self.args.other_params:
            return self.laplace_decoder_loss((loc, loc_delta, past_traj), scale, pi, labels_is_valid, loss, DE, device, labels, mapping)
        else:
            return self.laplace_decoder_loss(loc, scale, pi, labels_is_valid, loss, DE, device, labels, mapping)

    def laplace_decoder_loss(self, loc, scale, pi, labels_is_valid, loss, DE, device, labels, mapping=None):
        if 'stage_two' in self.args.other_params:
            original_loc, loc_delta, past_traj = loc
            loc = original_loc + loc_delta
        y_hat = torch.cat((loc, scale), dim=-1)
        batch_size = y_hat.shape[1]
        labels = torch.tensor(np.array(labels), device=device)
        l2_norm = torch.norm(y_hat[:, :, :, :2] - labels, p=2, dim=-1).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(y_hat.shape[1])]
        if 'stage_two' in self.args.other_params and self.args.do_train:
            loc_delta_best = loc_delta[best_mode, torch.arange(y_hat.shape[1])]
            delta_label = labels - original_loc[best_mode, torch.arange(y_hat.shape[1])]
            reg_delta_loss = torch.norm(loc_delta_best - delta_label, p=2, dim=-1)
            reg_loss = self.reg_loss(y_hat_best, labels).sum(dim=-1) + 5 * reg_delta_loss
            loss += get_angle_diff(labels, y_hat_best[:, :, :2], past_traj) * 2
            soft_target = F.softmax(-l2_norm / self.future_steps, dim=0).t().detach()
            cls_loss = self.cls_loss(pi, soft_target)
        else:
            reg_loss = self.reg_loss(y_hat_best, labels).sum(dim=-1)
            soft_target = F.softmax(-l2_norm / self.future_steps, dim=0).t().detach()
            cls_loss = self.cls_loss(pi, soft_target)
        if self.args.do_train:
            for i in range(batch_size):
                if self.args.do_train:
                    assert labels_is_valid[i][-1]
                loss_ = reg_loss[i]
                loss_ = loss_ * torch.tensor(labels_is_valid[i], device=device, dtype=torch.float).view(self.future_steps, 1)
                if labels_is_valid[i].sum() > utils.eps:
                    loss[i] += loss_.sum() / labels_is_valid[i].sum()
                loss[i] += cls_loss[i]
        if self.args.do_eval:
            outputs = loc.permute(1, 0, 2, 3).detach()
            pred_probs = F.softmax(pi, dim=-1).cpu().detach().numpy()
            for i in range(batch_size):
                if self.args.visualize:
                    labels = utils.get_from_mapping(mapping, 'labels')
                    labels = np.array(labels)
                    utils.visualize_gifs(mapping[i], self.args.future_frame_num, labels[i], outputs[i].cpu().numpy())
                outputs[i] = utils.to_origin_coordinate(outputs[i], i)
                if 'vis_nuscenes' in self.args.other_params:
                    vis_nuscenes.generate_nuscenes_gif(mapping[i], self.args.future_frame_num, outputs[i].cpu().numpy())
            outputs = outputs.cpu().numpy()
            return outputs, pred_probs, None
        return loss.mean(), DE, None


class GlobalGraph(nn.Module):
    """
    Global graph

    It's actually a self-attention.
    """

    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1):
        super(GlobalGraph, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_qkv = 1
        self.query = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.key = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.value = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)

    def get_extended_attention_mask(self, attention_mask):
        """
        1 in attention_mask stands for doing attention, 0 for not doing attention.

        After this function, 1 turns to 0, 0 turns to -10000.0

        Because the -10000.0 will be fed into softmax and -10000.0 can be thought as 0 in softmax.
        """
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*sz)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, mapping=None, return_scores=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = nn.functional.linear(hidden_states, self.key.weight)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        if attention_mask is not None:
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if utils.args.visualize and mapping is not None:
            for i, each in enumerate(attention_probs.tolist()):
                mapping[i]['attention_scores'] = np.array(each[0])
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            assert attention_probs.shape[1] == 1
            attention_probs = torch.squeeze(attention_probs, dim=1)
            assert len(attention_probs.shape) == 3
            return context_layer, attention_probs
        return context_layer


class GlobalGraphRes(nn.Module):

    def __init__(self, hidden_size):
        super(GlobalGraphRes, self).__init__()
        self.global_graph = GlobalGraph(hidden_size, hidden_size // 2)
        self.global_graph2 = GlobalGraph(hidden_size, hidden_size // 2)

    def forward(self, hidden_states, attention_mask=None, mapping=None):
        hidden_states = torch.cat([self.global_graph(hidden_states, attention_mask, mapping), self.global_graph2(hidden_states, attention_mask, mapping)], dim=-1)
        return hidden_states


class CrossAttention(GlobalGraph):

    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1, key_hidden_size=None, query_hidden_size=None):
        super(CrossAttention, self).__init__(hidden_size, attention_head_size, num_attention_heads)
        if query_hidden_size is not None:
            self.query = nn.Linear(query_hidden_size, self.all_head_size * self.num_qkv)
        if key_hidden_size is not None:
            self.key = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)
            self.value = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)

    def forward(self, hidden_states_query, hidden_states_key=None, attention_mask=None, mapping=None, return_scores=False):
        mixed_query_layer = self.query(hidden_states_query)
        mixed_key_layer = self.key(hidden_states_key)
        mixed_value_layer = self.value(hidden_states_key)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        if attention_mask is not None:
            assert hidden_states_query.shape[1] == attention_mask.shape[1] and hidden_states_key.shape[1] == attention_mask.shape[2]
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            return context_layer, torch.squeeze(attention_probs, dim=1)
        return context_layer


class PointLevelSubGraph(nn.Module):

    def __init__(self, hidden_size, depth=None):
        super(PointLevelSubGraph, self).__init__()
        if depth is None:
            depth = args.sub_graph_depth
        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])
        self.layer_0 = MLP(args.vector_size, hidden_size)
        if 'point_level-4-3' in args.other_params:
            self.layer_0_again = MLP(hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True)

    def forward(self, hidden_states, lengths):
        hidden_states = self.layer_0(hidden_states)
        if 'point_level-4-3' in args.other_params:
            hidden_states = hidden_states + self.layer_0_again(hidden_states)
        output, hn = self.GRU(hidden_states)
        return hn[-1], None


class PointLevelSubGraph_lane(nn.Module):

    def __init__(self, hidden_size, depth=None):
        super(PointLevelSubGraph_lane, self).__init__()
        if depth is None:
            depth = 3
        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])
        self.layer_0 = MLP(args.vector_size, hidden_size)
        if 'point_level-4-3' in args.other_params:
            self.layer_0_again = MLP(hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True)

    def forward(self, hidden_states, lengths):
        hidden_states = self.layer_0(hidden_states)
        if 'point_level-4-3' in args.other_params:
            hidden_states = self.layer_0_again(hidden_states)
        output, hn = self.GRU(hidden_states)
        return hn[-1], None


class VectorNet(nn.Module):
    """
    VectorNet

    It has two main components, sub graph and global graph.

    Sub graph encodes a polyline as a single vector.
    """

    def __init__(self, args_: 'config.Args'):
        super(VectorNet, self).__init__()
        global args
        args = args_
        hidden_size = args.hidden_size
        self.point_level_sub_graph = PointLevelSubGraph(hidden_size)
        self.point_level_sub_graph_lane = PointLevelSubGraph_lane(hidden_size)
        self.point_level_cross_attention = CrossAttention(hidden_size)
        if 'nuscenes' in args.other_params:
            num_layers = 1
        else:
            num_layers = 3
        decoder_layer_A2L = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size)
        self.laneGCN_A2L = nn.TransformerDecoder(decoder_layer_A2L, num_layers=num_layers)
        decoder_layer_L2A = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size)
        self.laneGCN_L2A = nn.TransformerDecoder(decoder_layer_L2A, num_layers=num_layers)

    def forward(self, mapping: 'List[Dict]', matrix: 'List[np.ndarray]', polyline_spans: 'List[List[slice]]', device, batch_size) ->Tuple[List[Tensor], List[Tensor]]:
        """
        :param matrix: each value in list is vectors of all element (shape [-1, 128])
        :param polyline_spans: vectors of i_th element is matrix[polyline_spans[i]]
        :return: hidden states of all elements and hidden states of lanes
        """
        all_agent_lists, all_lane_lists, batch_split_agent, batch_split_lane = [], [], [], []
        start_lane, end_lane = 0, 0
        start_agent, end_agent = 0, 0
        for i in range(batch_size):
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
            for j, polyline_span in enumerate(polyline_spans[i]):
                tensor = torch.tensor(matrix[i][polyline_span], device=device)
                if j < map_start_polyline_idx:
                    all_agent_lists.append(tensor)
                else:
                    all_lane_lists.append(tensor)
                if j == map_start_polyline_idx or map_start_polyline_idx == len(polyline_spans[i]):
                    end_agent += map_start_polyline_idx
                    batch_split_agent.append([start_agent, end_agent])
                    start_agent = end_agent
            end_lane += len(polyline_spans[i]) - map_start_polyline_idx
            batch_split_lane.append([start_lane, end_lane])
            start_lane = end_lane
        device = all_agent_lists[0].device
        all_agent_lists, lengths = utils.merge_tensors(all_agent_lists, device, args.vector_size)
        all_lane_lists, lengths_lane = utils.merge_tensors_lane(all_lane_lists, device, args.vector_size)
        all_agent_states_unspilited, _ = self.point_level_sub_graph(all_agent_lists, lengths)
        all_lane_states_unspilited, _ = self.point_level_sub_graph_lane(all_lane_lists, lengths_lane)
        agent_states_batch, lane_states_batch = [], []
        for i in range(batch_size):
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
            agents = all_agent_states_unspilited[batch_split_agent[i][0]:batch_split_agent[i][1]]
            lanes = all_lane_states_unspilited[batch_split_lane[i][0]:batch_split_lane[i][1]]
            agent_states_batch.append(agents)
            lane_states_batch.append(lanes)
        agent_states_batch, lengths = utils.merge_tensors(agent_states_batch, device, args.hidden_size)
        lane_states_batch, lengths_lane = utils.merge_tensors_lane(lane_states_batch, device, args.hidden_size)
        src_attention_mask_lane = torch.zeros([batch_size, lane_states_batch.shape[1]], device=device)
        src_attention_mask_agent = torch.zeros([batch_size, agent_states_batch.shape[1]], device=device)
        for i in range(batch_size):
            assert lengths[i] > 0
            assert lengths_lane[i] > 0
            src_attention_mask_lane[i, :lengths_lane[i]] = 1
            src_attention_mask_agent[i, :lengths[i]] = 1
        src_attention_mask_lane = src_attention_mask_lane == 0
        src_attention_mask_agent = src_attention_mask_agent == 0
        lane_states_batch = lane_states_batch.permute(1, 0, 2)
        agent_states_batch = agent_states_batch.permute(1, 0, 2)
        lane_states_batch = lane_states_batch + self.laneGCN_A2L(lane_states_batch, agent_states_batch, memory_key_padding_mask=src_attention_mask_agent, tgt_key_padding_mask=src_attention_mask_lane)
        agent_states_batch = agent_states_batch + self.laneGCN_L2A(agent_states_batch, lane_states_batch, memory_key_padding_mask=src_attention_mask_lane, tgt_key_padding_mask=src_attention_mask_agent)
        agent_states_batch = agent_states_batch.permute(1, 0, 2)
        lane_states_batch = lane_states_batch.permute(1, 0, 2)
        element_states_batch = []
        for i in range(batch_size):
            element_states_batch.append(torch.cat([agent_states_batch[i], lane_states_batch[i]], dim=0))
        return element_states_batch, lane_states_batch


class ModelMain(nn.Module):

    def __init__(self, args_: 'config.Args'):
        super(ModelMain, self).__init__()
        global args
        args = args_
        hidden_size = args.hidden_size
        self.encoder = VectorNet(args)
        self.global_graph = GlobalGraphRes(hidden_size)
        self.decoder = GRUDecoder(args, self)

    def forward(self, mapping: 'List[Dict]', device):
        vector_matrix = utils.get_from_mapping(mapping, 'matrix')
        polyline_spans = utils.get_from_mapping(mapping, 'polyline_spans')
        batch_size = len(vector_matrix)
        utils.batch_origin_init(mapping)
        all_element_states_batch, lane_states_batch = self.encoder.forward(mapping, vector_matrix, polyline_spans, device, batch_size)
        inputs, inputs_lengths = utils.merge_tensors(all_element_states_batch, device=device)
        lane_states_batch, lane_states_length = utils.merge_tensors(lane_states_batch, device=device)
        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)
        global_hidden_states = self.global_graph(inputs, attention_mask, mapping)
        return self.decoder(mapping, batch_size, lane_states_batch, lane_states_length, inputs, inputs_lengths, global_hidden_states, device)

    def load_state_dict(self, state_dict, strict: 'bool'=True):
        state_dict_rename_key = {}
        for key in state_dict.keys():
            if key.startswith('point_level_') or key.startswith('laneGCN_'):
                state_dict_rename_key['encoder.' + key] = state_dict[key]
            else:
                state_dict_rename_key[key] = state_dict[key]
        super(ModelMain, self).load_state_dict(state_dict_rename_key, strict)


class GaussianNLLLoss(nn.Module):
    """https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
    """

    def __init__(self, eps: 'float'=1e-06, reduction: 'str'='mean') ->None:
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = 0.5 * (torch.log(scale ** 2) + torch.abs(target - loc) ** 2 / scale ** 2)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (DecoderResCat,
     lambda: ([], {'hidden_size': 4, 'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GaussianNLLLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 2])], {})),
    (LaplaceNLLLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 2])], {})),
    (LayerNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SoftTargetCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

