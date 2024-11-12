
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


import time


import torch


import random


import numpy as np


from torch.utils.data import DataLoader


from collections import defaultdict


import torch.nn as nn


import logging


import torchvision


import matplotlib.pyplot as plt


import torchvision.transforms as transforms


import scipy.io as sio


from scipy.io import loadmat


from torch.utils.data import Dataset


from torchvision.transforms.functional import to_tensor


from torch.utils.data import ConcatDataset


import torch.nn.functional as F


from torch.nn.utils import spectral_norm


from torch.autograd import Variable


import math


import torchvision.models.resnet as resnet


from collections import OrderedDict


from matplotlib import pyplot as plt


from torch.nn import functional as F


from functools import reduce


from typing import List


from typing import Union


def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    """
        Inputs:
            disc_value: N x 25
    """
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb


def batch_encoder_disc_l2_loss(disc_value):
    """
        Inputs:
            disc_value: N x 25
    """
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50

    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    batch_size = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    axisang_norm = torch.norm(axisang + 1e-08, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


class TCMRLoss(nn.Module):

    def __init__(self, e_loss_weight=60.0, e_3d_loss_weight=30.0, e_pose_loss_weight=1.0, e_shape_loss_weight=0.001, d_motion_loss_weight=1.0, device='cuda'):
        super(TCMRLoss, self).__init__()
        self.e_loss_weight = e_loss_weight
        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight
        self.d_motion_loss_weight = d_motion_loss_weight
        self.device = device
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_regr = nn.MSELoss()
        self.criterion_attention = nn.CrossEntropyLoss()
        self.enc_loss = batch_encoder_disc_l2_loss
        self.dec_loss = batch_adv_disc_l2_loss

    def forward(self, generator_outputs, generator_outputs_Dm, data_2d, data_3d, scores, data_body_mosh=None, data_motion_mosh=None, body_discriminator=None, motion_discriminator=None):
        reduce = lambda x: x.contiguous().view((x.shape[0] * x.shape[1],) + x.shape[2:])
        flatten = lambda x: x.reshape(-1)
        accumulate_thetas = lambda x: torch.cat([output['theta_forDM'] for output in x], 0)
        if data_2d:
            sample_2d_count = data_2d['kp_2d'].shape[0]
            real_2d = torch.cat((data_2d['kp_2d'], data_3d['kp_2d']), 0)
        else:
            sample_2d_count = 0
            real_2d = data_3d['kp_2d']
        real_2d = reduce(real_2d)
        real_3d = reduce(data_3d['kp_3d'])
        data_3d_theta = reduce(data_3d['theta'])
        w_3d = data_3d['w_3d'].type(torch.bool)
        w_smpl = data_3d['w_smpl'].type(torch.bool)
        total_predict_thetas = accumulate_thetas(generator_outputs_Dm)
        preds = generator_outputs[-1]
        pred_j3d = preds['kp_3d'][sample_2d_count:]
        pred_theta = preds['theta'][sample_2d_count:]
        theta_size = pred_theta.shape[:2]
        pred_theta = reduce(pred_theta)
        pred_j2d = reduce(preds['kp_2d'])
        pred_j3d = reduce(pred_j3d)
        w_3d = flatten(w_3d)
        w_smpl = flatten(w_smpl)
        pred_theta = pred_theta[w_smpl]
        pred_j3d = pred_j3d[w_3d]
        data_3d_theta = data_3d_theta[w_smpl]
        real_3d = real_3d[w_3d]
        loss_kp_2d = self.keypoint_loss(pred_j2d, real_2d, openpose_weight=1.0, gt_weight=1.0) * self.e_loss_weight
        loss_kp_3d = self.keypoint_3d_loss(pred_j3d, real_3d)
        loss_kp_3d = loss_kp_3d * self.e_3d_loss_weight
        real_shape, pred_shape = data_3d_theta[:, 75:], pred_theta[:, 75:]
        real_pose, pred_pose = data_3d_theta[:, 3:75], pred_theta[:, 3:75]
        loss_dict = {'loss_kp_2d': loss_kp_2d, 'loss_kp_3d': loss_kp_3d}
        if pred_theta.shape[0] > 0:
            loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape, real_pose, real_shape)
            loss_shape = loss_shape * self.e_shape_loss_weight
            loss_pose = loss_pose * self.e_pose_loss_weight
            loss_dict['loss_shape'] = loss_shape
            loss_dict['loss_pose'] = loss_pose
        gen_loss = torch.stack(list(loss_dict.values())).sum()
        end_idx = 75
        start_idx = 6
        pred_motion = total_predict_thetas
        e_motion_disc_loss = self.enc_loss(motion_discriminator(pred_motion[:, :, start_idx:end_idx]))
        e_motion_disc_loss = e_motion_disc_loss * self.d_motion_loss_weight
        fake_motion = pred_motion.detach()
        real_motion = data_motion_mosh['theta']
        fake_disc_value = motion_discriminator(fake_motion[:, :, start_idx:end_idx])
        real_disc_value = motion_discriminator(real_motion[:, :, start_idx:end_idx])
        d_motion_disc_real, d_motion_disc_fake, d_motion_disc_loss = self.dec_loss(real_disc_value, fake_disc_value)
        d_motion_disc_real = d_motion_disc_real * self.d_motion_loss_weight
        d_motion_disc_fake = d_motion_disc_fake * self.d_motion_loss_weight
        d_motion_disc_loss = d_motion_disc_loss * self.d_motion_loss_weight
        loss_dict['e_m_disc_loss'] = e_motion_disc_loss
        loss_dict['d_m_disc_real'] = d_motion_disc_real
        loss_dict['d_m_disc_fake'] = d_motion_disc_fake
        loss_dict['d_m_disc_loss'] = d_motion_disc_loss
        gen_loss = gen_loss + e_motion_disc_loss
        motion_dis_loss = d_motion_disc_loss
        return gen_loss, motion_dis_loss, loss_dict

    def attention_loss(self, pred_scores):
        gt_labels = torch.ones(len(pred_scores)).long()
        return -0.5 * self.criterion_attention(pred_scores, gt_labels)

    def accel_3d_loss(self, pred_accel, gt_accel):
        pred_accel = pred_accel[:, 25:39]
        gt_accel = gt_accel[:, 25:39]
        if len(gt_accel) > 0:
            return self.criterion_accel(pred_accel, gt_accel).mean()
        else:
            return torch.FloatTensor(1).fill_(0.0)

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:39, :]
        gt_keypoints_3d = gt_keypoints_3d[:, 25:39, :]
        pred_keypoints_3d = pred_keypoints_3d
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d).mean()
        else:
            return torch.FloatTensor(1).fill_(0.0)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.0)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.0)
        return loss_regr_pose, loss_regr_betas


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        m.bias.data.fill_(0.01)


class SelfAttention(nn.Module):

    def __init__(self, attention_size, batch_first=False, layers=1, dropout=0.0, non_linearity='tanh'):
        super(SelfAttention, self).__init__()
        self.batch_first = batch_first
        if non_linearity == 'relu':
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()
        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))
        self.attention = nn.Sequential(*modules)
        self.attention.apply(init_weights)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        scores = self.attention(inputs).squeeze().reshape(inputs.shape[0], inputs.shape[1])
        scores = self.softmax(scores)
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()
        return representations, scores


class MotionDiscriminator(nn.Module):

    def __init__(self, rnn_size, input_size, num_layers, output_size=2, feature_pool='concat', use_spectral_norm=False, attention_size=1024, attention_layers=1, attention_dropout=0.5):
        super(MotionDiscriminator, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout
        self.gru = nn.GRU(self.input_size, self.rnn_size, num_layers=num_layers)
        linear_size = self.rnn_size if not feature_pool == 'concat' else self.rnn_size * 2
        if feature_pool == 'attention':
            self.attention = SelfAttention(attention_size=self.attention_size, layers=self.attention_layers, dropout=self.attention_dropout)
        if use_spectral_norm:
            self.fc = spectral_norm(nn.Linear(linear_size, output_size))
        else:
            self.fc = nn.Linear(linear_size, output_size)

    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, input_size]
        """
        batchsize, seqlen, input_size = sequence.shape
        sequence = torch.transpose(sequence, 0, 1)
        outputs, state = self.gru(sequence)
        if self.feature_pool == 'concat':
            outputs = F.relu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            output = self.fc(torch.cat([avg_pool, max_pool], dim=1))
        elif self.feature_pool == 'attention':
            outputs = outputs.permute(1, 0, 2)
            y, attentions = self.attention(outputs)
            output = self.fc(y)
        else:
            output = self.fc(outputs[-1])
        return output


class TemporalAttention(nn.Module):

    def __init__(self, attention_size, seq_len, non_linearity='tanh'):
        super(TemporalAttention, self).__init__()
        if non_linearity == 'relu':
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()
        self.fc = nn.Linear(attention_size, 256)
        self.relu = nn.ReLU()
        self.attention = nn.Sequential(nn.Linear(256 * seq_len, 256), activation, nn.Linear(256, 256), activation, nn.Linear(256, seq_len), activation)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch = x.shape[0]
        x = self.fc(x)
        x = x.view(batch, -1)
        scores = self.attention(x)
        scores = self.softmax(scores)
        return scores


class TemporalEncoder(nn.Module):

    def __init__(self, channel):
        super(TemporalEncoder, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv1d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta = nn.Conv1d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv1d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.conv_mask = nn.Conv1d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_mask_forR = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.attention = TemporalAttention(attention_size=2048, seq_len=3, non_linearity='tanh')

    def forward(self, x, is_train=False):
        x = x.permute(0, 2, 1)
        b, c, thw = x.size()
        x_ = x.permute(0, 2, 1)
        xx = torch.matmul(x_, x)
        xx = self.softmax(xx)
        x_phi = self.conv_phi(x).view(b, self.inter_channel, -1)
        x_theta = self.conv_theta(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        R = torch.cat((xx, mul_theta_phi), dim=0).view(xx.size(0), -1, xx.size(1), xx.size(2))
        Y = self.conv_mask_forR(R).reshape(b, thw, thw)
        Y = self.softmax(Y)
        mul_theta_phi_g = torch.matmul(Y, x_g)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, thw)
        mask = self.conv_mask(mul_theta_phi_g)
        out_ = mask + x
        y_cur_2 = out_[:, :, 8]
        y_cur_1 = out_[:, :, 7]
        y_cur_3 = out_[:, :, 9]
        y_bef_2 = out_[:, :, 5]
        y_bef_1 = out_[:, :, 4]
        y_bef_3 = out_[:, :, 6]
        y_aft_2 = out_[:, :, 11]
        y_aft_1 = out_[:, :, 10]
        y_aft_3 = out_[:, :, 12]
        y_cur_ = torch.cat((y_cur_1[:, None, :], y_cur_2[:, None, :], y_cur_3[:, None, :]), dim=1)
        y_bef_ = torch.cat((y_bef_1[:, None, :], y_bef_2[:, None, :], y_bef_3[:, None, :]), dim=1)
        y_aft_ = torch.cat((y_aft_1[:, None, :], y_aft_2[:, None, :], y_aft_3[:, None, :]), dim=1)
        scores = self.attention(y_cur_)
        y_cur = torch.mul(y_cur_, scores[:, :, None])
        y_cur = torch.sum(y_cur, dim=1)
        scores = self.attention(y_bef_)
        y_bef = torch.mul(y_bef_, scores[:, :, None])
        y_bef = torch.sum(y_bef, dim=1)
        scores = self.attention(y_aft_)
        y_aft = torch.mul(y_aft_, scores[:, :, None])
        y_aft = torch.sum(y_aft, dim=1)
        y = torch.cat((y_bef[:, None, :], y_cur[:, None, :], y_aft[:, None, :]), dim=1)
        scores = self.attention(y)
        out = torch.mul(y, scores[:, :, None])
        out = torch.sum(out, dim=1)
        if not is_train:
            return out, scores, out_
        else:
            y = torch.cat((out[:, None, :], out[:, None, :], out[:, None, :]), dim=1)
            return y, scores, out_


BASE_DATA_DIR = 'data/base_data'


H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]


H36M_TO_J14 = H36M_TO_J17[:14]


JOINT_MAP = {'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17, 'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16, 'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0, 'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8, 'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7, 'OP REye': 25, 'OP LEye': 26, 'OP REar': 27, 'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30, 'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34, 'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45, 'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7, 'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17, 'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20, 'Neck (LSP)': 47, 'Top of Head (LSP)': 48, 'Pelvis (MPII)': 49, 'Thorax (MPII)': 50, 'Spine (H36M)': 51, 'Jaw (H36M)': 52, 'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26, 'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27}


JOINT_NAMES = ['OP Nose', 'OP Neck', 'OP RShoulder', 'OP RElbow', 'OP RWrist', 'OP LShoulder', 'OP LElbow', 'OP LWrist', 'OP MidHip', 'OP RHip', 'OP RKnee', 'OP RAnkle', 'OP LHip', 'OP LKnee', 'OP LAnkle', 'OP REye', 'OP LEye', 'OP REar', 'OP LEar', 'OP LBigToe', 'OP LSmallToe', 'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel', 'Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip', 'Left Knee', 'Left Ankle', 'Right Wrist', 'Right Elbow', 'Right Shoulder', 'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Neck (LSP)', 'Top of Head (LSP)', 'Pelvis (MPII)', 'Thorax (MPII)', 'Spine (H36M)', 'Jaw (H36M)', 'Head (H36M)', 'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear']


SMPL_MODEL_DIR = BASE_DATA_DIR


def perspective_projection(points, rotation, translation, focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)
    projected_points = points / points[:, :, -1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)
    return projected_points[:, :, :-1]


def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1], pred_camera[:, 2], 2 * 5000.0 / (224.0 * pred_camera[:, 0] + 1e-09)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints, rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1), translation=pred_cam_t, focal_length=5000.0, camera_center=camera_center)
    pred_keypoints_2d = pred_keypoints_2d / (224.0 / 2.0)
    return pred_keypoints_2d


def rot6d_to_rotmat(x):
    x = x.view(-1, 3, 2)
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-06)
    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-06)
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)
    return rot_mats


def quaternion_to_angle_axis(quaternion: 'torch.Tensor') ->torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(quaternion)))
    if not quaternion.shape[-1] == 4:
        raise ValueError('Input must be a tensor of shape Nx4 or 4. Got {}'.format(quaternion.shape))
    q1: 'torch.Tensor' = quaternion[..., 1]
    q2: 'torch.Tensor' = quaternion[..., 2]
    q3: 'torch.Tensor' = quaternion[..., 3]
    sin_squared_theta: 'torch.Tensor' = q1 * q1 + q2 * q2 + q3 * q3
    sin_theta: 'torch.Tensor' = torch.sqrt(sin_squared_theta)
    cos_theta: 'torch.Tensor' = quaternion[..., 0]
    two_theta: 'torch.Tensor' = 2.0 * torch.where(cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta))
    k_pos: 'torch.Tensor' = two_theta / sin_theta
    k_neg: 'torch.Tensor' = 2.0 * torch.ones_like(sin_theta)
    k: 'torch.Tensor' = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)
    angle_axis: 'torch.Tensor' = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-06):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(rotation_matrix)))
    if len(rotation_matrix.shape) > 3:
        raise ValueError('Input size must be a three dimensional tensor. Got {}'.format(rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError('Input size must be a N x 3 x 4  tensor. Got {}'.format(rotation_matrix.shape))
    rmat_t = torch.transpose(rotation_matrix, 1, 2)
    mask_d2 = rmat_t[:, 2, 2] < eps
    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]
    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()
    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0], t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()
    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2], rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()
    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1], rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()
    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)
    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3)
    q *= 0.5
    return q


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32, device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


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


class Bottleneck(nn.Module):
    """
    Redefinition of Bottleneck residual block
    Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
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
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class HMR(nn.Module):
    """
    SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params):
        self.inplanes = 64
        super(HMR, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=64, create_transl=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def feature_extractor(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)
        return xf

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, return_features=False):
        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        pred_output = self.smpl(betas=pred_shape, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        output = [{'theta': torch.cat([pred_cam, pose, pred_shape], dim=1), 'verts': pred_vertices, 'kp_2d': pred_keypoints_2d, 'kp_3d': pred_joints}]
        if return_features:
            return xf, output
        else:
            return output


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MotionDiscriminator,
     lambda: ([], {'rnn_size': 4, 'input_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SelfAttention,
     lambda: ([], {'attention_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (TemporalAttention,
     lambda: ([], {'attention_size': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

