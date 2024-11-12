
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


import torch


import torch.nn as nn


from torch import nn


from torch.nn import functional as F


import torch.nn.functional as F


from torch import nn as nn


from functools import partial


import math


from collections import namedtuple


import scipy.sparse


import matplotlib.pyplot as plt


from copy import copy


import logging


from collections import Counter


import scipy.sparse as sp


import time


import random


def transform_mat(R, t):
    """ Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    """
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    transforms_mat = transform_mat(rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]
    posed_joints = transforms[:, :, :3, 3]
    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    return posed_joints, rel_transforms


def batch_rodrigues(rot_vecs, epsilon=1e-08, dtype=torch.float32):
    """ Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    """
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    angle = torch.norm(rot_vecs + 1e-08, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def blend_shapes(betas, shape_disps):
    """ Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def vertices2joints(J_regressor, vertices):
    """ Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    """
    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, joints=None, pose2rot=True, v_shaped=None, dtype=torch.float32):
    """ Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device
    if v_shaped is None:
        v_shaped = v_template + blend_shapes(betas, shapedirs)
    if joints is not None:
        J = joints
    else:
        J = vertices2joints(J_regressor, v_shaped)
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)
        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1), posedirs).view(batch_size, -1, 3)
    v_posed = pose_offsets + v_shaped
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    return verts, J_transformed


class BodyModel(nn.Module):

    def __init__(self, bm_fname, num_betas=10, num_dmpls=None, dmpl_fname=None, num_expressions=80, use_posedirs=True, dtype=torch.float32, persistant_buffer=False):
        super(BodyModel, self).__init__()
        """
        :param bm_fname: path to a SMPL model as pkl file
        :param num_betas: number of shape parameters to include.
        :param device: default on gpu
        :param dtype: float precision of the computations
        :return: verts, trans, pose, betas 
        """
        self.dtype = dtype
        if '.npz' in bm_fname:
            smpl_dict = np.load(bm_fname, encoding='latin1')
        else:
            raise ValueError('bm_fname should be either a .pkl nor .npz file')
        self.num_betas = num_betas
        self.num_dmpls = num_dmpls
        self.num_expressions = num_expressions
        njoints = smpl_dict['posedirs'].shape[2] // 3
        self.model_type = {(69): 'smpl', (153): 'smplh', (162): 'smplx', (45): 'mano', (105): 'animal_horse', (102): 'animal_dog'}[njoints]
        assert self.model_type in ['smpl', 'smplh', 'smplx', 'mano', 'mano', 'animal_horse', 'animal_dog'], ValueError('model_type should be in smpl/smplh/smplx/mano.')
        self.use_dmpl = False
        if num_dmpls is not None:
            if dmpl_fname is not None:
                self.use_dmpl = True
            else:
                raise ValueError('dmpl_fname should be provided when using dmpls!')
        if self.use_dmpl and self.model_type in ['smplx', 'mano', 'animal_horse', 'animal_dog']:
            raise NotImplementedError('DMPLs only work with SMPL/SMPLH models for now.')
        self.comp_register('init_v_template', torch.tensor(smpl_dict['v_template'][None], dtype=dtype), persistent=persistant_buffer)
        self.comp_register('f', torch.tensor(smpl_dict['f'].astype(np.int32), dtype=torch.int32), persistent=persistant_buffer)
        num_total_betas = smpl_dict['shapedirs'].shape[-1]
        if num_betas < 1:
            num_betas = num_total_betas
        shapedirs = smpl_dict['shapedirs'][:, :, :num_betas]
        self.comp_register('shapedirs', torch.tensor(shapedirs, dtype=dtype), persistent=persistant_buffer)
        if self.model_type == 'smplx':
            if smpl_dict['shapedirs'].shape[-1] > 300:
                begin_shape_id = 300
            else:
                begin_shape_id = 10
                num_expressions = smpl_dict['shapedirs'].shape[-1] - 10
            exprdirs = smpl_dict['shapedirs'][:, :, begin_shape_id:begin_shape_id + num_expressions]
            self.comp_register('exprdirs', torch.tensor(exprdirs, dtype=dtype), persistent=persistant_buffer)
            expression = torch.tensor(np.zeros((1, num_expressions)), dtype=dtype)
            self.comp_register('init_expression', expression, persistent=persistant_buffer)
        if self.use_dmpl:
            dmpldirs = np.load(dmpl_fname)['eigvec']
            dmpldirs = dmpldirs[:, :, :num_dmpls]
            self.comp_register('dmpldirs', torch.tensor(dmpldirs, dtype=dtype), persistent=persistant_buffer)
        self.comp_register('J_regressor', torch.tensor(smpl_dict['J_regressor'], dtype=dtype), persistent=persistant_buffer)
        if use_posedirs:
            posedirs = smpl_dict['posedirs']
            posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
            self.comp_register('posedirs', torch.tensor(posedirs, dtype=dtype), persistent=persistant_buffer)
        else:
            self.posedirs = None
        kintree_table = smpl_dict['kintree_table'].astype(np.int32)
        self.comp_register('kintree_table', torch.tensor(kintree_table, dtype=torch.int32), persistent=persistant_buffer)
        weights = smpl_dict['weights']
        self.comp_register('weights', torch.tensor(weights, dtype=dtype), persistent=persistant_buffer)
        self.comp_register('init_trans', torch.zeros((1, 3), dtype=dtype), persistent=persistant_buffer)
        self.comp_register('init_root_orient', torch.zeros((1, 3), dtype=dtype), persistent=persistant_buffer)
        if self.model_type in ['smpl', 'smplh', 'smplx']:
            self.comp_register('init_pose_body', torch.zeros((1, 63), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type == 'animal_horse':
            self.comp_register('init_pose_body', torch.zeros((1, 105), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type == 'animal_dog':
            self.comp_register('init_pose_body', torch.zeros((1, 102), dtype=dtype), persistent=persistant_buffer)
        if self.model_type in ['smpl']:
            self.comp_register('init_pose_hand', torch.zeros((1, 1 * 3 * 2), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type in ['smplh', 'smplx']:
            self.comp_register('init_pose_hand', torch.zeros((1, 15 * 3 * 2), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type in ['mano']:
            self.comp_register('init_pose_hand', torch.zeros((1, 15 * 3), dtype=dtype), persistent=persistant_buffer)
        if self.model_type == 'smplx':
            self.comp_register('init_pose_jaw', torch.zeros((1, 1 * 3), dtype=dtype), persistent=persistant_buffer)
            self.comp_register('init_pose_eye', torch.zeros((1, 2 * 3), dtype=dtype), persistent=persistant_buffer)
        self.comp_register('init_betas', torch.zeros((1, num_betas), dtype=dtype), persistent=persistant_buffer)
        if self.use_dmpl:
            self.comp_register('init_dmpls', torch.zeros((1, num_dmpls), dtype=dtype), persistent=persistant_buffer)

    def comp_register(self, name, value, persistent=False):
        if sys.version_info[0] > 2:
            self.register_buffer(name, value, persistent)
        else:
            self.register_buffer(name, value)

    def r(self):
        return c2c(self.forward().v)

    def forward(self, root_orient=None, pose_body=None, pose_hand=None, pose_jaw=None, pose_eye=None, betas=None, trans=None, dmpls=None, expression=None, v_template=None, joints=None, v_shaped=None, return_dict=False, **kwargs):
        """

        :param root_orient: Nx3
        :param pose_body:
        :param pose_hand:
        :param pose_jaw:
        :param pose_eye:
        :param kwargs:
        :return:
        """
        batch_size = 1
        for arg in [root_orient, pose_body, pose_hand, pose_jaw, pose_eye, betas, trans, dmpls, expression, v_template, joints]:
            if arg is not None:
                batch_size = arg.shape[0]
                break
        assert self.model_type in ['smpl', 'smplh', 'smplx', 'mano', 'animal_horse', 'animal_dog'], ValueError('model_type should be in smpl/smplh/smplx/mano')
        if root_orient is None:
            root_orient = self.init_root_orient.expand(batch_size, -1)
        if self.model_type in ['smplh', 'smpl']:
            if pose_body is None:
                pose_body = self.init_pose_body.expand(batch_size, -1)
            if pose_hand is None:
                pose_hand = self.init_pose_hand.expand(batch_size, -1)
        elif self.model_type == 'smplx':
            if pose_body is None:
                pose_body = self.init_pose_body.expand(batch_size, -1)
            if pose_hand is None:
                pose_hand = self.init_pose_hand.expand(batch_size, -1)
            if pose_jaw is None:
                pose_jaw = self.init_pose_jaw.expand(batch_size, -1)
            if pose_eye is None:
                pose_eye = self.init_pose_eye.expand(batch_size, -1)
        elif self.model_type in ['mano']:
            if pose_hand is None:
                pose_hand = self.init_pose_hand.expand(batch_size, -1)
        elif self.model_type in ['animal_horse', 'animal_dog']:
            if pose_body is None:
                pose_body = self.init_pose_body.expand(batch_size, -1)
        if pose_hand is None and self.model_type not in ['animal_horse', 'animal_dog']:
            pose_hand = self.init_pose_hand.expand(batch_size, -1)
        if trans is None:
            trans = self.init_trans.expand(batch_size, -1)
        if v_template is None:
            v_template = self.init_v_template.expand(batch_size, -1, -1)
        if betas is None:
            betas = self.init_betas.expand(batch_size, -1)
        if self.model_type in ['smplh', 'smpl']:
            full_pose = torch.cat([root_orient, pose_body, pose_hand], dim=-1)
        elif self.model_type == 'smplx':
            full_pose = torch.cat([root_orient, pose_body, pose_jaw, pose_eye, pose_hand], dim=-1)
        elif self.model_type in ['mano']:
            full_pose = torch.cat([root_orient, pose_hand], dim=-1)
        elif self.model_type in ['animal_horse', 'animal_dog']:
            full_pose = torch.cat([root_orient, pose_body], dim=-1)
        if self.use_dmpl:
            if dmpls is None:
                dmpls = self.init_dmpls.expand(batch_size, -1)
            shape_components = torch.cat([betas, dmpls], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.dmpldirs], dim=-1)
        elif self.model_type == 'smplx':
            if expression is None:
                expression = self.init_expression.expand(batch_size, -1)
            shape_components = torch.cat([betas, expression], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)
        else:
            shape_components = betas
            shapedirs = self.shapedirs
        verts, Jtr = lbs(betas=shape_components, pose=full_pose, v_template=v_template, shapedirs=shapedirs, posedirs=self.posedirs, J_regressor=self.J_regressor, parents=self.kintree_table[0].long(), lbs_weights=self.weights, joints=joints, v_shaped=v_shaped, dtype=self.dtype)
        Jtr = Jtr + trans.unsqueeze(dim=1)
        verts = verts + trans.unsqueeze(dim=1)
        res = {}
        res['v'] = verts
        res['f'] = self.f
        res['Jtr'] = Jtr
        res['full_pose'] = full_pose
        if not return_dict:


            class result_meta(object):
                pass
            res_class = result_meta()
            for k, v in res.items():
                res_class.__setattr__(k, v)
            res = res_class
        return res


class ResBlock(nn.Module):

    def __init__(self, Fin, Fout, n_neurons=256):
        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout
        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)
        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)
        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)
        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))
        Xout = self.fc1(x)
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)
        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout
        if final_nl:
            return self.ll(Xout)
        return Xout


def CRot2rotmat(pose):
    reshaped_input = pose.view(-1, 3, 2)
    b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
    dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
    b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)


def quaternion_to_angle_axis(quaternion: 'torch.Tensor') ->torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

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
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
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
    """Convert 3x4 rotation matrix to 4d quaternion vector

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
        >>> output = rotation_matrix_to_quaternion(input)  # Nx4
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
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = rotation_matrix_to_angle_axis(input)  # Nx3
    """
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def rotmat2aa(rotmat):
    """
    :param rotmat: Nx1xnum_jointsx9
    :return: Nx1xnum_jointsx3
    """
    batch_size = rotmat.shape[0]
    homogen_matrot = F.pad(rotmat.view(-1, 3, 3), [0, 1])
    pose = rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
    return pose


def parms_decode(pose, trans):
    bs = trans.shape[0]
    pose_full = CRot2rotmat(pose)
    pose = pose_full.view([bs, 1, -1, 9])
    pose = rotmat2aa(pose).view(bs, -1)
    global_orient = pose[:, :3]
    hand_pose = pose[:, 3:]
    pose_full = pose_full.view([bs, -1, 3, 3])
    hand_parms = {'global_orient': global_orient, 'hand_pose': hand_pose, 'transl': trans, 'fullpose': pose_full}
    return hand_parms


class CoarseNet(nn.Module):

    def __init__(self, n_neurons=512, latentD=16, in_bps=4096, in_pose=12, **kwargs):
        super(CoarseNet, self).__init__()
        self.latentD = latentD
        self.enc_bn0 = nn.BatchNorm1d(in_bps)
        self.enc_bn1 = nn.BatchNorm1d(in_bps + in_pose)
        self.enc_rb1 = ResBlock(in_bps + in_pose, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons + in_bps + in_pose, n_neurons)
        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)
        self.do = nn.Dropout(p=0.1, inplace=False)
        self.dec_bn1 = nn.BatchNorm1d(in_bps)
        self.dec_rb1 = ResBlock(latentD + in_bps, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + latentD + in_bps, n_neurons)
        self.dec_pose = nn.Linear(n_neurons, 16 * 6)
        self.dec_trans = nn.Linear(n_neurons, 3)

    def encode(self, bps_object, trans_rhand, global_orient_rhand_rotmat):
        bs = bps_object.shape[0]
        X = torch.cat([bps_object, global_orient_rhand_rotmat.view(bs, -1), trans_rhand], dim=1)
        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0, True)
        X = self.enc_rb2(torch.cat([X0, X], dim=1), True)
        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))

    def decode(self, Zin, bps_object):
        bs = Zin.shape[0]
        o_bps = self.dec_bn1(bps_object)
        X0 = torch.cat([Zin, o_bps], dim=1)
        X = self.dec_rb1(X0, True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), True)
        pose = self.dec_pose(X)
        trans = self.dec_trans(X)
        results = parms_decode(pose, trans)
        results['z'] = Zin
        return results

    def forward(self, bps_object, trans_rhand, global_orient_rhand_rotmat, **kwargs):
        """

        :param bps_object: bps_delta of object: Nxn_bpsx3
        :param delta_hand_mano: bps_delta of subject, e.g. hand: Nxn_bpsx3
        :param output_type: bps_delta of something, e.g. hand: Nxn_bpsx3
        :return:
        """
        z = self.encode(bps_object, trans_rhand, global_orient_rhand_rotmat)
        z_s = z.rsample()
        hand_parms = self.decode(z_s, bps_object)
        results = {'mean': z.mean, 'std': z.scale}
        results.update(hand_parms)
        return results

    def sample_poses(self, bps_object, seed=None):
        bs = bps_object.shape[0]
        np.random.seed(seed)
        dtype = bps_object.dtype
        device = bps_object.device
        self.eval()
        with torch.no_grad():
            Zgen = np.random.normal(0.0, 1.0, size=(bs, self.latentD))
            Zgen = torch.tensor(Zgen, dtype=dtype)
        return self.decode(Zgen, bps_object)


def point2point_signed(x, y, x_normals=None, y_normals=None):
    """
    signed distance between two pointclouds
    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
    Returns:
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - x2y_signed: Torch.Tensor
            the sign distance from x to y
    """
    N, P1, D = x.shape
    P2 = y.shape[1]
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError('y does not have the correct shape.')
    ch_dist = chd.ChamferDistance()
    x_near, y_near, xidx_near, yidx_near = ch_dist(x, y)
    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D)
    x_near = y.gather(1, xidx_near_expanded)
    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D)
    y_near = x.gather(1, yidx_near_expanded)
    x2y = x - x_near
    y2x = y - y_near
    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out
    else:
        y2x_signed = y2x.norm(dim=2)
    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)
    return y2x_signed, x2y_signed


class RefineNet(nn.Module):

    def __init__(self, in_size=778 + 16 * 6 + 3, h_size=512, n_iters=3):
        super(RefineNet, self).__init__()
        self.n_iters = n_iters
        self.bn1 = nn.BatchNorm1d(778)
        self.rb1 = ResBlock(in_size, h_size)
        self.rb2 = ResBlock(in_size + h_size, h_size)
        self.rb3 = ResBlock(in_size + h_size, h_size)
        self.out_p = nn.Linear(h_size, 16 * 6)
        self.out_t = nn.Linear(h_size, 3)
        self.dout = nn.Dropout(0.3)
        self.actvf = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, h2o_dist, fpose_rhand_rotmat_f, trans_rhand_f, global_orient_rhand_rotmat_f, verts_object, **kwargs):
        bs = h2o_dist.shape[0]
        init_pose = fpose_rhand_rotmat_f[..., :2].reshape(bs, -1)
        init_rpose = global_orient_rhand_rotmat_f[..., :2].reshape(bs, -1)
        init_pose = torch.cat([init_rpose, init_pose], dim=1)
        init_trans = trans_rhand_f
        for i in range(self.n_iters):
            if i != 0:
                hand_parms = parms_decode(init_pose, init_trans)
                verts_rhand = self.rhm_train(**hand_parms).vertices
                _, h2o_dist = point2point_signed(verts_rhand, verts_object)
            h2o_dist = self.bn1(h2o_dist)
            X0 = torch.cat([h2o_dist, init_pose, init_trans], dim=1)
            X = self.rb1(X0)
            X = self.dout(X)
            X = self.rb2(torch.cat([X, X0], dim=1))
            X = self.dout(X)
            X = self.rb3(torch.cat([X, X0], dim=1))
            X = self.dout(X)
            pose = self.out_p(X)
            trans = self.out_t(X)
            init_trans = init_trans + trans
            init_pose = init_pose + pose
        hand_parms = parms_decode(init_pose, init_trans)
        return hand_parms


class View(nn.Module):

    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args
        self._name = 'reshape'

    def forward(self, x):
        return x.view(self.shape)


class BatchFlatten(nn.Module):

    def __init__(self):
        super(BatchFlatten, self).__init__()
        self._name = 'batch_flatten'

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Regress(nn.Module):
    """
    Description:
    - Data
        Input: body_pose
        Output: body pitch, roll and z-transl
    - Model
        2-layer MLP with ReLU
    """

    def __init__(self, cfg):
        super(Regress, self).__init__()
        self.cfg = cfg
        self.out_size = 3
        self.fc1 = nn.Linear(63, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 3)

    def forward(self, X, **kwargs):
        """
        :param X            (torch.Tensor) -- (b, 63)
        :return out_dict    (dict) -- of tensors of size (b,) each.
        """
        x = self.fc1(X)
        x = self.relu(x)
        out = self.fc2(x)
        out_dict = {self.cfg.height: out[:, 0], 'pitch': out[:, 1], 'roll': out[:, 2]}
        return out_dict


class ContinousRotReprDecoder(nn.Module):

    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)


class NormalDistDecoder(nn.Module):

    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()
        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        return torch.distributions.normal.Normal(self.mu(Xout), F.softplus(self.logvar(Xout)))


def aa2matrot(pose):
    """
    :param Nx3
    :return: pose_matrot: Nx3x3
    """
    bs = pose.size(0)
    num_joints = pose.size(1) // 3
    pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose)[:, :3, :3].contiguous()
    return pose_body_matrot


class geodesic_loss_R(nn.Module):

    def __init__(self, reduction='batchmean'):
        super(geodesic_loss_R, self).__init__()
        self.reduction = reduction
        self.eps = 1e-06

    def bgdR(self, m1, m2):
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))
        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.min(cos, m1.new(np.ones(batch)))
        cos = torch.max(cos, m1.new(np.ones(batch)) * -1)
        return torch.acos(cos)

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred, ytrue)
        if self.reduction == 'mean':
            return torch.mean(theta)
        if self.reduction == 'batchmean':
            breakpoint()
            return torch.mean(torch.sum(theta, dim=theta.shape[1:]))
        else:
            return theta


def hook(name, grad):
    index_tuple = (grad == math.inf).nonzero(as_tuple=True)
    if len(index_tuple[0]) > 0:
        None
    grad[index_tuple] = 0
    index_tuple = (grad == -math.inf).nonzero(as_tuple=True)
    if len(index_tuple[0]) > 0:
        None
    grad[index_tuple] = 0
    return grad


def matrot2aa(pose_matrot):
    """
    :param pose_matrot: Nx3x3
    :return: Nx3
    """
    homogen_matrot = F.pad(pose_matrot, [0, 1])
    pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot)
    return pose


class VPoser(nn.Module):

    def __init__(self, model_ps):
        super(VPoser, self).__init__()
        self.vp_ps = model_ps
        self.bm_train = BodyModel(f'{model_ps.smplx_dir}/smplx/SMPLX_NEUTRAL.npz')
        num_neurons, self.latentD = model_ps.num_neurons, model_ps.latentD
        self.num_joints = 21
        n_features = self.num_joints * 3
        self.encoder_net = nn.Sequential(BatchFlatten(), nn.BatchNorm1d(n_features), nn.Linear(n_features, num_neurons), nn.LeakyReLU(), nn.BatchNorm1d(num_neurons), nn.Dropout(0.1), nn.Linear(num_neurons, num_neurons), nn.Linear(num_neurons, num_neurons), NormalDistDecoder(num_neurons, self.latentD))
        self.decoder_net = nn.Sequential(nn.Linear(self.latentD, num_neurons), nn.LeakyReLU(), nn.Dropout(0.1), nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, self.num_joints * 6), ContinousRotReprDecoder())

    def encode(self, pose_body):
        """
        :param Pin: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        """
        return self.encoder_net(pose_body)

    def decode(self, Zin):
        bs = Zin.shape[0]
        prec = self.decoder_net(Zin)
        if prec.requires_grad:
            prec.register_hook(partial(hook, 'prec'))
        prec1 = matrot2aa(prec).reshape(bs, -1, 3)
        if prec1.requires_grad:
            prec1.register_hook(partial(hook, 'prec1'))
        return {'pose_body': prec1.reshape(bs, -1, 3), 'pose_body_matrot': prec.reshape(bs, -1, 9)}

    def forward(self, pose_body):
        """
        :param Pin: aa: Nx1xnum_jointsx3 / matrot: Nx1xnum_jointsx9
        :param input_type: matrot / aa for matrix rotations or axis angles
        :param output_type: matrot / aa
        :return:
        """
        q_z = self.encode(pose_body)
        q_z_sample = q_z.rsample()
        decode_results = self.decode(q_z_sample)
        decode_results.update({'poZ_body_mean': q_z.mean, 'poZ_body_std': q_z.scale, 'q_z': q_z})
        return decode_results

    def sample_poses(self, num_poses):
        some_weight = [a for a in self.parameters()][0]
        dtype = some_weight.dtype
        device = some_weight.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(np.random.normal(0.0, 1.0, size=(num_poses, self.latentD)), dtype=dtype, device=device)
        out = self.decode(Zgen)
        out['latent'] = Zgen
        return out

    def loss_function(self, dorig, drec, current_epoch, mode):
        l1_loss = torch.nn.L1Loss(reduction='mean')
        geodesic_loss = geodesic_loss_R(reduction='mean')
        bs, latentD = drec['poZ_body_mean'].shape
        device = drec['poZ_body_mean'].device
        loss_kl_wt = self.vp_ps.loss_kl_wt
        loss_rec_wt = self.vp_ps.loss_rec_wt
        loss_matrot_wt = self.vp_ps.loss_matrot_wt
        loss_jtr_wt = self.vp_ps.loss_jtr_wt
        q_z = drec['q_z']
        self.bm_train
        with torch.no_grad():
            bm_orig = self.bm_train(pose_body=dorig)
        bm_rec = self.bm_train(pose_body=drec['pose_body'].contiguous().view(bs, -1))
        v2v = l1_loss(bm_rec.v, bm_orig.v)
        p_z = torch.distributions.normal.Normal(loc=torch.zeros((bs, latentD), device=device, requires_grad=False), scale=torch.ones((bs, latentD), device=device, requires_grad=False))
        weighted_loss_dict = {'loss_kl': loss_kl_wt * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1])), 'loss_mesh_rec': loss_rec_wt * v2v}
        if current_epoch < self.vp_ps.keep_extra_loss_terms_until_epoch:
            weighted_loss_dict['matrot'] = loss_matrot_wt * geodesic_loss(drec['pose_body_matrot'].view(-1, 3, 3), aa2matrot(dorig.view(-1, 3)))
            weighted_loss_dict['jtr'] = loss_jtr_wt * l1_loss(bm_rec.Jtr, bm_orig.Jtr)
        weighted_loss_dict['loss_total'] = torch.stack(list(weighted_loss_dict.values())).sum()
        with torch.no_grad():
            unweighted_loss_dict = {'v2v': torch.sqrt(torch.pow(bm_rec.v - bm_orig.v, 2).sum(-1)).mean()}
            unweighted_loss_dict['loss_total'] = torch.cat(list({k: v.view(-1) for k, v in unweighted_loss_dict.items()}.values()), dim=-1).sum().view(1)
        if mode == 'train':
            if current_epoch < self.vp_ps.keep_extra_loss_terms_until_epoch:
                return weighted_loss_dict['loss_total'], {'loss/total': weighted_loss_dict['loss_total'], 'loss/KLD': weighted_loss_dict['loss_kl'], 'loss/Mesh_rec': weighted_loss_dict['loss_mesh_rec'], 'loss/MatRot': weighted_loss_dict['matrot'], 'loss/Jtr': weighted_loss_dict['jtr']}
            return weighted_loss_dict['loss_total'], {'loss/total': weighted_loss_dict['loss_total'], 'loss/KLD': weighted_loss_dict['loss_kl'], 'loss/Mesh_rec': weighted_loss_dict['loss_mesh_rec']}
        return unweighted_loss_dict['loss_total'], {'loss/total': unweighted_loss_dict['loss_total'], 'loss/Mesh_rec': unweighted_loss_dict['v2v']}


model_output = namedtuple('output', ['vertices', 'global_orient', 'transl'])


class ObjectModel(nn.Module):

    def __init__(self, v_template, batch_size=1, dtype=torch.float32):
        """ 3D rigid object model

                Parameters
                ----------
                v_template: np.array Vx3, dtype = np.float32
                    The vertices of the object
                batch_size: int, N, optional
                    The batch size used for creating the model variables

                dtype: torch.dtype
                    The data type for the created variables
            """
        super(ObjectModel, self).__init__()
        self.dtype = dtype
        v_template = np.repeat(v_template[np.newaxis], batch_size, axis=0)
        self.register_buffer('v_template', v_template)
        transl = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('transl', nn.Parameter(transl, requires_grad=True))
        global_orient = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('global_orient', nn.Parameter(global_orient, requires_grad=True))
        self.batch_size = batch_size

    def forward(self, global_orient=None, transl=None, v_template=None, **kwargs):
        """ Forward pass for the object model

        Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)

            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            v_template: torch.tensor, optional, shape BxVx3
                The new object vertices to overwrite the default vertices

        Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        """
        if global_orient is None:
            global_orient = self.global_orient
        if transl is None:
            transl = self.transl
        if v_template is None:
            v_template = self.v_template
        rot_mats = batch_rodrigues(global_orient.view(-1, 3)).view([self.batch_size, 3, 3])
        transl = transl.view(self.batch_size, 1, 3)
        vertices = torch.matmul(v_template, rot_mats) + transl
        output = model_output(vertices=vertices, global_orient=global_orient, transl=transl)
        return output


pi = torch.Tensor([3.141592653589793])


def rad2deg(tensor):
    """Function that converts angles from radians to degrees.

    See :class:`~torchgeometry.RadToDeg` for details.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Example:
        >>> input = tgm.pi * torch.rand(1, 3, 3)
        >>> output = tgm.rad2deg(input)
    """
    if not torch.is_tensor(tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(tensor)))
    return 180.0 * tensor / pi.type(tensor.dtype)


class RadToDeg(nn.Module):
    """Creates an object that converts angles from radians to degrees.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Examples::

        >>> input = tgm.pi * torch.rand(1, 3, 3)
        >>> output = tgm.RadToDeg()(input)
    """

    def __init__(self):
        super(RadToDeg, self).__init__()

    def forward(self, input):
        return rad2deg(input)


def deg2rad(tensor):
    """Function that converts angles from degrees to radians.

    See :class:`~torchgeometry.DegToRad` for details.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Examples::

        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = tgm.deg2rad(input)
    """
    if not torch.is_tensor(tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(tensor)))
    return tensor * pi.type(tensor.dtype) / 180.0


class DegToRad(nn.Module):
    """Function that converts angles from degrees to radians.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Examples::

        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = tgm.DegToRad()(input)
    """

    def __init__(self):
        super(DegToRad, self).__init__()

    def forward(self, input):
        return deg2rad(input)


def convert_points_from_homogeneous(points):
    """Function that converts points from homogeneous to Euclidean space.

    See :class:`~torchgeometry.ConvertPointsFromHomogeneous` for details.

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = tgm.convert_points_from_homogeneous(input)  # BxNx2
    """
    if not torch.is_tensor(points):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(points)))
    if len(points.shape) < 2:
        raise ValueError('Input must be at least a 2D tensor. Got {}'.format(points.shape))
    return points[..., :-1] / points[..., -1:]


class ConvertPointsFromHomogeneous(nn.Module):
    """Creates a transformation that converts points from homogeneous to
    Euclidean space.

    Args:
        points (Tensor): tensor of N-dimensional points.

    Returns:
        Tensor: tensor of N-1-dimensional points.

    Shape:
        - Input: :math:`(B, D, N)` or :math:`(D, N)`
        - Output: :math:`(B, D, N + 1)` or :math:`(D, N + 1)`

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> transform = tgm.ConvertPointsFromHomogeneous()
        >>> output = transform(input)  # BxNx2
    """

    def __init__(self):
        super(ConvertPointsFromHomogeneous, self).__init__()

    def forward(self, input):
        return convert_points_from_homogeneous(input)


def convert_points_to_homogeneous(points):
    """Function that converts points from Euclidean to homogeneous space.

    See :class:`~torchgeometry.ConvertPointsToHomogeneous` for details.

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = tgm.convert_points_to_homogeneous(input)  # BxNx4
    """
    if not torch.is_tensor(points):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(points)))
    if len(points.shape) < 2:
        raise ValueError('Input must be at least a 2D tensor. Got {}'.format(points.shape))
    return nn.functional.pad(points, (0, 1), 'constant', 1.0)


class ConvertPointsToHomogeneous(nn.Module):
    """Creates a transformation to convert points from Euclidean to
    homogeneous space.

    Args:
        points (Tensor): tensor of N-dimensional points.

    Returns:
        Tensor: tensor of N+1-dimensional points.

    Shape:
        - Input: :math:`(B, D, N)` or :math:`(D, N)`
        - Output: :math:`(B, D, N + 1)` or :math:`(D, N + 1)`

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> transform = tgm.ConvertPointsToHomogeneous()
        >>> output = transform(input)  # BxNx4
    """

    def __init__(self):
        super(ConvertPointsToHomogeneous, self).__init__()

    def forward(self, input):
        return convert_points_to_homogeneous(input)


def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = angle_axis_to_rotation_matrix(input)  # Nx4x4
    """

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-06):
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)
    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)
    eps = 1e-06
    mask = (theta2 > eps).view(-1, 1, 1)
    mask_pos = mask.type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    rotation_matrix[..., :3, :3] = mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix


def aa2rotmat(axis_angle):
    """
    :param Nx1xnum_jointsx3
    :return: pose_matrot: Nx1xnum_jointsx9
    """
    batch_size = axis_angle.shape[0]
    pose_body_matrot = angle_axis_to_rotation_matrix(axis_angle.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
    return pose_body_matrot


def euler_torch(rots, order='xyz', units='deg'):
    """
    TODO: Confirm that copying does not affect gradient.

    :param rots     (torch.Tensor) -- (b, 3)
    :param order    (str)
    :param units    (str)

    :return r       (torch.Tensor) -- (b, 3, 3)
    """
    bs = rots.shape[0]
    single_val = False if len(rots.shape) > 1 else True
    rots = rots.reshape(-1, 3)
    if units == 'deg':
        rots = torch.deg2rad(rots)
    r = torch.eye(3)[None].repeat(bs, 1, 1)
    for axis in range(3):
        theta, axis = rots[:, axis], order[axis]
        c = torch.cos(theta)
        s = torch.sin(theta)
        aux_r = torch.eye(3)[None].repeat(bs, 1, 1)
        if axis == 'x':
            aux_r[:, 1, 1] = aux_r[:, 2, 2] = c
            aux_r[:, 1, 2] = s
            aux_r[:, 2, 1] = -s
        if axis == 'y':
            aux_r[:, 0, 0] = aux_r[:, 2, 2] = c
            aux_r[:, 0, 2] = s
            aux_r[:, 2, 0] = -s
        if axis == 'z':
            aux_r[:, 0, 0] = aux_r[:, 1, 1] = c
            aux_r[:, 0, 1] = -s
            aux_r[:, 1, 0] = s
        r = torch.matmul(aux_r, r)
    if single_val:
        return r[0]
    else:
        return r


def load_obj_verts(rand_rotmat, object_mesh, n_sample_verts=10000):
    """
    Load object vertices corresponding to BPS representation which should be in the same distribution as used for GrabNet data.
    NOTE: the returned vertices are not transformed, but are simply meant to be used as input for RefineNet.

    :param cfg             (OmegaConf dict) with at least keys [obj_meshes_dir]
    :param obj             (str) e.g., 'wineglass'
    :param rand_rotmat     (torch.Tensor) -- (bs, 3, 3)
    :param scale           (float)
    :param n_sample_verts  (int)

    :return verts_sampled  (torch.Tensor) -- (bs, n_sample_verts, 3) - e.g., (250, 10000, 3)
    """
    obj_mesh_v = object_mesh[0]
    obj_mesh_f = object_mesh[1]
    max_length = np.linalg.norm(obj_mesh_v, axis=1).max()
    if max_length > 0.3:
        re_scale = max_length / 0.08
        None
        obj_mesh_v = obj_mesh_v / re_scale
    object_fullpts = obj_mesh_v
    maximum = object_fullpts.max(0, keepdims=True)
    minimum = object_fullpts.min(0, keepdims=True)
    offset = (maximum + minimum) / 2
    verts_obj = object_fullpts - offset
    bs = rand_rotmat.shape[0]
    obj_mesh_verts = torch.Tensor(verts_obj)[None].repeat(bs, 1, 1)
    obj_mesh_verts_rotated = torch.bmm(obj_mesh_verts, torch.transpose(rand_rotmat, 1, 2))
    if verts_obj.shape[-2] < n_sample_verts:
        verts_sample_id = np.arange(verts_obj.shape[-2])
        repeated_verts = np.random.choice(verts_obj.shape[-2], n_sample_verts - verts_obj.shape[-2], replace=False)
        verts_sample_id = np.concatenate((verts_sample_id, repeated_verts))
    else:
        verts_sample_id = np.random.choice(verts_obj.shape[-2], n_sample_verts, replace=False)
    verts_sampled = obj_mesh_verts_rotated[..., verts_sample_id, :]
    obj_mesh_v = obj_mesh_verts_rotated
    obj_mesh_new = [obj_mesh_v, obj_mesh_f]
    return verts_sampled, obj_mesh_new


def local2global(verts, rot_mat, trans):
    """
    Convert local mesh vertices to global using parameters (rot_mat, trans).

    :param verts    (torch.Tensor) -- size (b, N, 3)
    :param rot_mat  (torch.Tensor) -- size (b, 3, 3)
    :param trans    (torch.Tensor) -- size (b, 3)

    :return verts   (torch.Tensor) -- size (b, N, 3)
    """
    return torch.transpose(torch.bmm(rot_mat, torch.transpose(verts, 1, 2)), 1, 2) + trans[:, None, :]


def recompose_angle(yaw, pitch, roll, style='rotmat'):
    """
    Given yaw, pitch and roll, get final rotation matrix as a function of them.
    Return it in required 'style', i.e., rotmat / aa

    :param yaw    (torch.Tensor) -- shape (b,)
    :param pitch  (torch.Tensor) -- shape (b,)
    :param roll   (torch.Tensor) -- shape (b,)

    :return angle (torch.Tensor) -- shape (b,3) or (b,3,3)
    """
    bs = yaw.shape[0]
    yaw_rotmat = torch.vstack((torch.cos(yaw), -torch.sin(yaw), torch.zeros(bs), torch.sin(yaw), torch.cos(yaw), torch.zeros(bs), torch.zeros(bs), torch.zeros(bs), torch.ones(bs))).transpose(0, 1).reshape(-1, 3, 3)
    pitch_rotmat = torch.vstack((torch.cos(pitch), torch.zeros(bs), torch.sin(pitch), torch.zeros(bs), torch.ones(bs), torch.zeros(bs), -torch.sin(pitch), torch.zeros(bs), torch.cos(pitch))).transpose(0, 1).reshape(-1, 3, 3)
    roll_rotmat = torch.vstack((torch.ones(bs), torch.zeros(bs), torch.zeros(bs), torch.zeros(bs), torch.cos(roll), -torch.sin(roll), torch.zeros(bs), torch.sin(roll), torch.cos(roll))).transpose(0, 1).reshape(-1, 3, 3)
    angle = torch.bmm(torch.bmm(yaw_rotmat, pitch_rotmat), roll_rotmat)
    if style == 'aa':
        angle = rotmat2aa(angle.reshape(bs, 1, 1, 9)).reshape(bs, 3)
    return angle


class Losses(object):

    def __init__(self, cfg, gan_body, gan_rh):
        self.cfg = cfg
        self.device = 'cuda:' + str(cfg.cuda_id) if cfg.cuda_id != -1 else 'cpu'
        self.gan_body = gan_body
        self.gan_rh_coarse, self.gan_rh_refine = gan_rh
        self.sbj_m = smplx.create(model_path=cfg.smplx_dir, model_type='smplx', gender=cfg.gender, use_pca=False, flat_hand_mean=True, batch_size=cfg.batch_size).eval()
        self.rh_m = mano.load(model_path=cfg.mano_dir, is_right=True, model_type='mano', gender=cfg.gender, num_pca_comps=45, flat_hand_mean=True, batch_size=cfg.batch_size).eval()
        self.sbj_verts_region_map = np.load(self.cfg.sbj_verts_region_map_pth, allow_pickle=True)
        self.adj_matrix_original = np.load(self.cfg.adj_matrix_orig)
        if cfg.subsample_sbj:
            self.sbj_verts_id = np.load(self.cfg.sbj_verts_simplified)
            self.sbj_faces_simplified = np.load(self.cfg.sbj_faces_simplified)
            self.sbj_verts_region_map = self.sbj_verts_region_map[self.sbj_verts_id]
            self.adj_matrix_simplified = np.load(self.cfg.adj_matrix_simplified)
        with open(cfg.mano2smplx_verts_ids, 'rb') as f:
            self.correspondences = np.load(f)
        self.interesting = dict(np.load(cfg.interesting_pth))

    def get_rh_match_loss(self, z, transl, global_orient, w, angle, extras):
        """
        Get full-body of human based on pose and parameters (transl, global_orient), along with right-hand pose obtained from rh-grasping model.
        Compute loss between predicted joint/vertex locations versus expected for right hand.

        :param z                (torch.Tensor)
        :param transl           (torch.Tensor)
        :param global_orient    (torch.Tensor)
        :param w                (torch.Tensor)
        :param angle            (torch.Tensor)
        :param extras           (dict) - keys ['obj_bps, 'obj_bps_verts', 'obj_transl'] (stuff that is not optimized over and is necessary for loss computation)

        :return rh_match_loss   (torch.Tensor) - (bs,)
        :return bm_output       (SMPLX body-model output)
        :return rv              (torch.Tensor) - (bs, 778, 3)
        :return rf              (torch.Tensor) - (1538, 3)
        """
        bs = z.shape[0]
        sbj_pose = self.gan_body.decode(z)['pose_body'].reshape(bs, -1)
        angle_rotmat = euler_torch(360 * angle)
        obj_bps_verts, _ = load_obj_verts(angle_rotmat, extras['object_mesh'])
        obj_bps_verts = obj_bps_verts
        bps = bps_torch(custom_basis=torch.from_numpy(np.load(self.cfg.bps_pth)['basis']))
        bps_object = bps.encode(obj_bps_verts, feature_type='dists')['dists']
        drec_cnet = self.gan_rh_coarse.decode(w, bps_object)
        verts_rh_gen_cnet = self.rh_m(**drec_cnet).vertices
        _, h2o = point2point_signed(verts_rh_gen_cnet, obj_bps_verts)
        drec_cnet['trans_rhand_f'] = drec_cnet['transl']
        drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)
        drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)
        drec_cnet['verts_object'] = obj_bps_verts
        drec_cnet['h2o_dist'] = h2o.abs()
        drec_rnet = self.gan_rh_refine(**drec_cnet)
        rh_out = self.rh_m(**drec_rnet)
        joints_constraint = rh_out.joints
        vertices_constraint = rh_out.vertices
        obj_global_orient_rotmat = aa2rotmat(extras['obj_global_orient'].reshape(1, 1, 1, 3)).reshape(1, 3, 3).repeat(bs, 1, 1)
        R = torch.bmm(obj_global_orient_rotmat, torch.inverse(angle_rotmat))
        t = extras['obj_transl'].repeat(bs, 1)
        joints_constraint = local2global(joints_constraint, R, t)
        vertices_constraint = local2global(vertices_constraint, R, t)
        bodydict_output = {'body_pose': sbj_pose, 'transl': transl, 'global_orient': global_orient, 'right_hand_pose': drec_rnet['hand_pose']}
        bm_output = self.sbj_m(**bodydict_output)
        bm_output.vertices[..., 2] -= bm_output.vertices[..., 2].min(-1, keepdims=True)[0]
        if self.cfg.rh_match_type == 'joints':
            joints_output_wrist = bm_output.joints[:, 21:22, :]
            joint_output_fingers = bm_output.joints[:, 40:55, :]
            joints_output = torch.cat((joints_output_wrist, joint_output_fingers), 1)
            match_loss = F.mse_loss(joints_output, joints_constraint, reduction='none')
            joints_wts = torch.Tensor([[10, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]]).repeat(bs, 1).reshape(bs, 16, 1)
            match_loss *= joints_wts
            match_loss = match_loss.reshape(bs, -1).mean(1)
        else:
            bm_output_vertices = bm_output.vertices[:, self.correspondences]
            match_loss = F.mse_loss(bm_output_vertices, vertices_constraint, reduction='none').reshape(bs, -1).mean(1)
            if self.cfg.alpha_interesting_verts > 0:
                iv = self.interesting['interesting_vertices']
                bm_output_vertices_interest = bm_output_vertices[:, iv]
                vertices_constraint_interest = vertices_constraint[:, iv]
                match_loss_interest = F.mse_loss(bm_output_vertices_interest, vertices_constraint_interest, reduction='none').reshape(bs, -1).mean(1)
                match_loss += self.cfg.alpha_interesting_verts * match_loss_interest
        return match_loss, bm_output, vertices_constraint, torch.from_numpy(self.rh_m.faces.astype(float))

    def intersection(self, sbj_verts, obj_verts, sbj_faces, obj_faces, full_body=True, adjacency_matrix=None):
        """
        Compute intersection penalty between body and object (or obstacle) given vertices and normals of both.

        :param sbj_verts                (torch.Tensor) on device - (bs, N_sbj, 3)
        :param obj_verts                (torch.Tensor) on device - (1, N_obj, 3)
        :param sbj_faces                (torch.Tensor) on device - (F_sbj, 3)
        :param obj_faces                (torch.Tensor) on device - (F_obj, 3)
        :param full_body                (bool) -- for full-body if True; else for rhand
        :param adjacency_matrix         (optional)

        :return penet_loss_batched_in   (torch.Tensor) - (bs,) - loss values for each batch element - penetration
        :return penet_loss_batched_out  (torch.Tensor) - (bs,) - loss values for each batch element - outside
        """
        device = sbj_verts.device
        bs = sbj_verts.shape[0]
        obj_verts = obj_verts.repeat(bs, 1, 1)
        num_obj_verts, num_sbj_verts = obj_verts.shape[1], sbj_verts.shape[1]
        penet_loss_batched_in, penet_loss_batched_out = torch.zeros(bs), torch.zeros(bs)
        thresh = self.cfg.intersection_thresh
        if self.cfg.obstacle_obj2sbj:
            sign = kaolin.ops.mesh.check_sign(sbj_verts, sbj_faces, obj_verts)
            ones = torch.ones_like(sign.long())
            sign = torch.where(sign, -ones, ones)
            face_vertices = kaolin.ops.mesh.index_vertices_by_faces(sbj_verts, sbj_faces)
            dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(obj_verts.contiguous(), face_vertices)
            obj2sbj = dist * sign
            zeros_o2s, ones_o2s = torch.zeros_like(obj2sbj), torch.ones_like(obj2sbj)
            loss_o2s_in = torch.sum(abs(torch.where(obj2sbj < thresh, obj2sbj - thresh, zeros_o2s)), 1) / num_obj_verts
            loss_o2s_out = torch.sum(torch.log(torch.where(obj2sbj > thresh, obj2sbj + ones_o2s, ones_o2s)), 1) / num_obj_verts
            penet_loss_batched_in += loss_o2s_in
            penet_loss_batched_out += loss_o2s_out
        if self.cfg.obstacle_sbj2obj:
            face_vertices = kaolin.ops.mesh.index_vertices_by_faces(obj_verts, obj_faces)
            indices_good_faces = face_vertices[0].det().abs() > 0.001
            obj_faces = obj_faces[indices_good_faces]
            face_vertices = face_vertices[0][indices_good_faces][None].repeat(bs, 1, 1, 1)
            sign = kaolin.ops.mesh.check_sign(obj_verts, obj_faces, sbj_verts)
            ones = torch.ones_like(sign.long())
            sign = torch.where(sign, -ones, ones)
            dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(sbj_verts.contiguous(), face_vertices)
            sbj2obj = dist * sign
            zeros_s2o, ones_s2o = torch.zeros_like(sbj2obj), torch.ones_like(sbj2obj)
            loss_s2o_out = torch.sum(torch.log(torch.where(sbj2obj > thresh, sbj2obj + ones_s2o, ones_s2o)), 1) / num_sbj_verts
            loss_s2o_in = torch.sum(abs(torch.where(sbj2obj < thresh, sbj2obj - thresh, zeros_s2o)), 1) / num_sbj_verts
            if full_body and self.cfg.obstacle_sbj2obj_extra == 'connected_components' and loss_s2o_in.mean() > 0:
                edges = np.stack(np.where(adjacency_matrix))
                num_nodes = adjacency_matrix.shape[0]
                v_to_edges = torch.zeros((num_nodes, edges.shape[1]))
                v_to_edges[edges[0], range(edges.shape[1])] = 1
                v_to_edges[edges[1], range(edges.shape[1])] = 1
                indices_inter = sbj2obj < thresh
                v_to_edges = v_to_edges[None].expand(bs, -1, -1).clone()
                v_to_edges[torch.where(indices_inter)] = 0
                edges_indices = v_to_edges.sum(1) == 2
                num_inter_v = indices_inter.sum(-1)
                for i in range(bs):
                    if loss_s2o_in[i] > 0:
                        edges_i = edges[:, edges_indices[i]]
                        adj = pytorch_geometric.to_scipy_sparse_matrix(edges_i, num_nodes=num_nodes)
                        n_components, labels = sp.csgraph.connected_components(adj)
                        n_components -= num_inter_v[i]
                        if n_components > 1:
                            indices_out = torch.ones([num_sbj_verts])
                            indices_out[indices_inter[i]] = 0
                            labels_ = labels[indices_out.bool()]
                            most_common_label = Counter(labels_).most_common()[0][0]
                            penalized_joints = (labels != most_common_label) * indices_out.bool().numpy()
                            loss_s2o_in[i] += sbj2obj[i][penalized_joints].sum() / num_sbj_verts
            penet_loss_batched_in += loss_s2o_in
            penet_loss_batched_out += loss_s2o_out
        return penet_loss_batched_in, penet_loss_batched_out

    def get_rh_obstacle_penet_loss(self, rv, rf, extras):
        """
        Compute penetration loss between right-hand grasp and all provided obstacle vertices.

        :param rv                           (torch.Tensor) - (bs, 778, 3)
        :param rf                           (torch.Tensor) - (bs, 1538, 3)
        :param extras                       (dict)         - keys ['ov', 'obj_normals', 'o_verts_wts'] (stuff that is not optimized over and that can be copied over from GT and is necessary for loss computation)

        :return obstacle_loss_batched_in   (torch.tensor) - (bs,)
        :return obstacle_loss_batched_out  (torch.tensor) - (bs,)
        """
        bs = self.cfg.batch_size
        obstacle_loss_batched_in, obstacle_loss_batched_out = torch.zeros(bs), torch.zeros(bs)
        for obstacle in extras['obstacles_info']:
            olb_in, olb_out = self.intersection(rv, obstacle['o_verts'][None], rf, obstacle['o_faces'], False)
            obstacle_loss_batched_in += olb_in
            obstacle_loss_batched_out += olb_out
        if len(extras['obstacles_info']):
            obstacle_loss_batched_in /= len(extras['obstacles_info'])
            obstacle_loss_batched_out /= len(extras['obstacles_info'])
        return obstacle_loss_batched_in, obstacle_loss_batched_out

    def get_obstacle_penet_loss(self, bm_output, extras):
        """
        Compute penetration loss between human and all provided obstacle vertices.

        :param bm_output                    (SMPLX body-model output)
        :param extras                       (dict)         - keys ['o_verts', 'o_faces'] (stuff that is not optimized over and that can be copied over from GT and is necessary for loss computation)

        :return obstacle_loss_batched_in   (torch.tensor) - (bs,)
        :return obstacle_loss_batched_out  (torch.tensor) - (bs,)
        """
        bs = self.cfg.batch_size
        bv = bm_output.vertices.reshape(bs, -1, 3)
        bf = torch.LongTensor(self.sbj_m.faces.astype('float32'))
        if self.cfg.subsample_sbj:
            bf = self.sbj_faces_simplified
            bv = bv[:, self.sbj_verts_id, :]
            adjacency_matrix = self.adj_matrix_simplified
        else:
            adjacency_matrix = self.adj_matrix_original
        obstacle_loss_batched_in, obstacle_loss_batched_out = torch.zeros(bs), torch.zeros(bs)
        for obstacle in extras['obstacles_info']:
            olb_in, olb_out = self.intersection(bv, obstacle['o_verts'][None], bf, obstacle['o_faces'], True, adjacency_matrix)
            obstacle_loss_batched_in += olb_in
            obstacle_loss_batched_out += olb_out
        if len(extras['obstacles_info']):
            obstacle_loss_batched_in /= len(extras['obstacles_info'])
            obstacle_loss_batched_out /= len(extras['obstacles_info'])
        return obstacle_loss_batched_in, obstacle_loss_batched_out

    def get_lowermost_loss(self, bm_output):
        """
        Compute absolute distance between lowermost point of body. This assumes that floor is at zero height.
        Useful to correct for when ground-loss is not perfect.

        :param bm_output       (SMPLX body-model output)

        :return lowermost_loss (torch.tensor) - (bs,)
        """
        bs = self.cfg.batch_size
        body_vertices = bm_output.vertices.reshape(bs, -1, 3)
        lowermost_loss = abs(torch.min(body_vertices[:, :, 2], 1).values)
        return lowermost_loss

    def get_gaze_loss(self, bm_output, extras):
        """
        Compute gaze angle between head vector and back-of-the-head-to-object vectors.

        :param bm_output       (SMPLX body-model output)
        :param extras          (dict)         - keys containing 'obj_transl'

        :return gaze_loss (torch.tensor) - (bs,)
        """
        bs = self.cfg.batch_size
        body_vertices = bm_output.vertices.reshape(bs, -1, 3)
        head_front, head_back = body_vertices[:, 8970], body_vertices[:, 8973]
        obj_transl = extras['obj_transl'].repeat(bs, 1)
        vec_head, vec_obj = head_front - head_back, obj_transl - head_back
        dot = torch.bmm(vec_head.view(bs, 1, -1), vec_obj.view(bs, -1, 1))[:, 0, 0]
        norm_head = torch.norm(vec_head, dim=1) + 0.0001
        norm_obj = torch.norm(vec_obj, dim=1) + 0.0001
        gaze_loss = torch.arccos(dot / (norm_head * norm_obj))
        return gaze_loss

    def get_wrist_loss(self, bm_output, rh_vertices):
        """
        Compute wrist angle between MANO grasp hand and SMLPX full-body hand.

        :param bm_output       (SMPLX body-model output)
        :param rh_vertices     (MANO hand-model output vertices)
        :param correspondences (SMLPX to MANO correspondences)

        :return gaze_loss (torch.tensor) - (bs,)
        """
        bs = self.cfg.batch_size
        body_vertices = bm_output.vertices.reshape(bs, -1, 3)
        rh_left, rh_right = body_vertices[:, self.correspondences][:, 90], body_vertices[:, self.correspondences][:, 51]
        body_left, body_right = rh_vertices[:, 90], rh_vertices[:, 51]
        vec_rh, vec_bm = rh_left - rh_right, body_left - body_right
        dot = torch.bmm(vec_rh.view(bs, 1, -1), vec_bm.view(bs, -1, 1))[:, 0, 0]
        norm_rh = torch.norm(vec_rh, dim=1) + 0.0001
        norm_bm = torch.norm(vec_bm, dim=1) + 0.0001
        wrist_loss = torch.arccos(dot / (norm_rh * norm_bm))
        return wrist_loss

    def gan_loss(self, z, transl, global_orient, w, a, extras={}, alpha_lowermost=0.0, alpha_rh_match=1.0, alpha_obstacle_in=0.0, alpha_obstacle_out=0.0, alpha_gaze=0.0, alpha_rh_obstacle_in=0.0, alpha_rh_obstacle_out=0.0, alpha_wrist=0.0):
        """
        GAN loss for Test-Time Optimization:
            (1) MSE loss for full-body joints/vertices after passing body parameters of VPoser through in SMPLX.
            (2) Lowermost loss which makes lowest body vertex along vertical axis zero
            (3) Human-object penetration loss.
            (4) Human-obstacle penetration loss.

        :param z             (torch.Tensor) - (1, 32)
        :param transl        (list of 3 torch.Tensors) - each of shape (1,)
        :param global_orient (list of 3 torch.Tensors) - each of shape (1,)
        :param w             (torch.Tensor) - (1, 16)
        :param a             (torch.Tensor) - (1, 3, 3)
        :param extras        (dict)         - keys ['ov', 'obj_normals'] (stuff that is not optimized over and that can be copied over from GT and is necessary for loss computation)

        :return loss_dict    (dict of torch.Tensor) - each single loss value
        """
        transl_x, transl_y, transl_z = transl
        transl = torch.stack([transl_x, transl_y, transl_z], 1)
        global_orient1, global_orient2, global_orient3 = global_orient
        if self.cfg.orient_optim_type == 'aa':
            global_orient = torch.stack([global_orient1, global_orient2, global_orient3], 1)
        else:
            global_orient = recompose_angle(global_orient1, global_orient2, global_orient3, 'aa')
        rh_match_loss, bm_output, rv, rf = self.get_rh_match_loss(z, transl, global_orient, w, a, extras)
        rh_obstacle_loss_in = rh_obstacle_loss_out = torch.zeros_like(rh_match_loss)
        if alpha_rh_obstacle_in + alpha_rh_obstacle_out > 0:
            rh_obstacle_loss_in, rh_obstacle_loss_out = self.get_rh_obstacle_penet_loss(rv, rf, extras)
        obstacle_loss_in = obstacle_loss_out = torch.zeros_like(rh_match_loss)
        if alpha_obstacle_in + alpha_obstacle_out > 0:
            obstacle_loss_in, obstacle_loss_out = self.get_obstacle_penet_loss(bm_output, extras)
        lowermost_loss = torch.zeros_like(rh_match_loss)
        if alpha_lowermost > 0:
            lowermost_loss = self.get_lowermost_loss(bm_output)
        gaze_loss = torch.zeros_like(rh_match_loss)
        if alpha_gaze > 0:
            gaze_loss = self.get_gaze_loss(bm_output, extras)
        wrist_loss = torch.zeros_like(rh_match_loss)
        if alpha_wrist > 0:
            wrist_loss = self.get_wrist_loss(bm_output, rv)
        total_loss = lowermost_loss * alpha_lowermost + rh_match_loss * alpha_rh_match + obstacle_loss_in * alpha_obstacle_in + obstacle_loss_out * alpha_obstacle_out + rh_obstacle_loss_in * alpha_rh_obstacle_in + rh_obstacle_loss_out * alpha_rh_obstacle_out + gaze_loss * alpha_gaze + wrist_loss * alpha_wrist
        loss_dict = {'total': total_loss, 'lowermost_loss': lowermost_loss, 'rh_match_loss': rh_match_loss, 'obstacle_loss_in': obstacle_loss_in, 'obstacle_loss_out': obstacle_loss_out, 'gaze_loss': gaze_loss, 'wrist_loss': wrist_loss, 'rh_obstacle_loss_in': rh_obstacle_loss_in, 'rh_obstacle_loss_out': rh_obstacle_loss_out}
        return loss_dict


class Registry:
    """Class for registry object which acts as central source of truth
    """
    mapping = {'class': {}, 'state': {}}

    @classmethod
    def register_class(cls, name):

        def wrap(func):
            cls.mapping['class'][name] = func
            return func
        return wrap

    @classmethod
    def register(cls, name, obj):
        """Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from mmf.common.registry import registry

            registry.register("config", {})
        """
        path = name.split('.')
        current = cls.mapping['state']
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[path[-1]] = obj

    @classmethod
    def get_class(cls, name):
        return cls.mapping['class'].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        """Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for MMF's
                               internal operations. Default: False
        Usage::

            from mmf.common.registry import registry

            config = registry.get("config")
        """
        original_name = name
        name = name.split('.')
        value = cls.mapping['state']
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break
        if 'writer' in cls.mapping['state'] and value == default and no_warning is False:
            cls.mapping['state']['writer'].warning('Key {} is not present in registry, returning default value of {}'.format(original_name, default))
        return value

    @classmethod
    def unregister(cls, name):
        """Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from mmf.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping['state'].pop(name, None)


registry = Registry()


class FLEX(nn.Module):

    def __init__(self, cfg, z_init, transl_init, global_orient_init, w_init, a_init, gan_body, gan_rh, task='', extras={}, requires_grad=True):
        super(FLEX, self).__init__()
        self.cfg = cfg
        self.bs = z_init.shape[0]
        self.device = 'cuda:' + str(cfg.cuda_id) if cfg.cuda_id != -1 else 'cpu'
        self.losses = Losses(cfg, gan_body, gan_rh)
        self.gan_body = gan_body
        self.gan_rh_coarse, self.gan_rh_refine = gan_rh
        self.task = task
        self.extras = extras
        z_param_init = z_init.detach().clone()
        self.z = nn.Parameter(z_param_init, requires_grad=requires_grad)
        transl_param_init = transl_init.detach().clone()
        self.transl_x = nn.Parameter(transl_param_init[:, 0], requires_grad=requires_grad)
        self.transl_y = nn.Parameter(transl_param_init[:, 1], requires_grad=requires_grad)
        self.transl_z = nn.Parameter(transl_param_init[:, 2], requires_grad=requires_grad)
        global_orient_param_init = global_orient_init.detach().clone()
        self.global_orient1 = nn.Parameter(global_orient_param_init[:, 0], requires_grad=requires_grad)
        self.global_orient2 = nn.Parameter(global_orient_param_init[:, 1], requires_grad=requires_grad)
        self.global_orient3 = nn.Parameter(global_orient_param_init[:, 2], requires_grad=requires_grad)
        w_param_init = w_init.detach().clone()
        self.w = nn.Parameter(w_param_init, requires_grad=requires_grad)
        a_param_init = a_init.detach().clone()
        self.angle = nn.Parameter(a_param_init, requires_grad=requires_grad)
        for _, param in self.gan_body.named_parameters():
            param.requires_grad = False
        for _, param in self.gan_rh_coarse.named_parameters():
            param.requires_grad = False
        for _, param in self.gan_rh_refine.named_parameters():
            param.requires_grad = False
        self.pose_ground_prior = registry.get_class('Regress')(cfg)
        self.load_model()
        for _, param in self.pose_ground_prior.named_parameters():
            param.requires_grad = False

    def load_model(self):
        """
        Load best saved ckpt into self.pose_ground_prior.
        """
        state = torch.load(self.cfg.best_pgprior, map_location=self.device)
        self.pose_ground_prior.load_state_dict(state['state_dict'], strict=True)
        self.pose_ground_prior.eval()
        vars_pnet = [var[1] for var in self.pose_ground_prior.named_parameters()]
        pnet_n_params = sum(p.numel() for p in vars_pnet if p.requires_grad)
        None
        None

    def pose_ground_pred(self):
        """
        Perform forward pass on pre-trained model for pose-ground relation.
        :return    (dict) of ['transl_z', 'pitch', 'yaw'] -- all torch.Tensors of size (1,)
        """
        with torch.no_grad():
            pose_output = self.gan_body.decode(self.z)['pose_body'].reshape(self.bs, -1)
            pred = self.pose_ground_prior(pose_output)
            return pred

    def get_dout(self, best_transl, best_orient, best_z, best_w, best_angle, extras):
        """
        Use optimization parameters (t, g, z, w) to visualize corresponding mesh results.

        :param best_transl  (torch.Tensor) -- (bs, 3)
        :param best_orient  (torch.Tensor) -- (bs, 3)
        :param best_z       (torch.Tensor) -- (bs, 32)
        :param best_w       (torch.Tensor) -- (bs, 16)
        :param best_angle   (torch.Tensor) -- (bs, 3)

        :return dout        (dict) of keys ['pose_body', 'pose_body_matrot', 'z', 'rh_verts', 'transl', 'global_orient'] -- torch.Tensors of size [(bs, 63); (bs, 21, 9); (bs, 32); (bs, 778, 3); (bs, 3); (bs, 3)]
        """
        dout = self.gan_body.decode(best_z)
        dout['pose_body'] = dout['pose_body'].reshape(self.bs, -1)
        dout['z'] = best_z
        rh_m = self.losses.rh_m
        sbj_m = self.losses.sbj_m
        angle_rotmat = euler_torch(360 * best_angle)
        obj_bps_verts, _ = load_obj_verts(angle_rotmat, extras['object_mesh'])
        obj_bps_verts = obj_bps_verts
        bps = bps_torch(custom_basis=torch.from_numpy(np.load(self.cfg.bps_pth)['basis']))
        bps_object = bps.encode(obj_bps_verts, feature_type='dists')['dists']
        drec_cnet = self.gan_rh_coarse.decode(best_w, bps_object)
        verts_rh_gen_cnet = rh_m(**drec_cnet).vertices
        _, h2o = point2point_signed(verts_rh_gen_cnet, obj_bps_verts)
        drec_cnet['trans_rhand_f'] = drec_cnet['transl']
        drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)
        drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)
        drec_cnet['verts_object'] = obj_bps_verts
        drec_cnet['h2o_dist'] = h2o.abs()
        drec_rnet = self.gan_rh_refine(**drec_cnet)
        rh_out = rh_m(**drec_rnet)
        vertices_constraint = rh_out.vertices
        obj_global_orient_rotmat = aa2rotmat(self.extras['obj_global_orient'].reshape(1, 1, 1, 3)).reshape(1, 3, 3).repeat(self.bs, 1, 1)
        R = torch.bmm(obj_global_orient_rotmat, torch.inverse(angle_rotmat))
        t = self.extras['obj_transl'].repeat(self.bs, 1)
        rh_verts = local2global(vertices_constraint, R, t)
        dout['rh_verts'] = rh_verts
        dout['transl'] = best_transl
        dout['global_orient'] = best_orient
        bodydict_output = {'body_pose': dout['pose_body'], 'transl': best_transl, 'global_orient': best_orient, 'right_hand_pose': drec_rnet['hand_pose']}
        bm_output = sbj_m(**bodydict_output)
        bm_output.vertices[..., 2] -= bm_output.vertices[..., 2].min(-1, keepdims=True)[0]
        dout['human_vertices'] = bm_output.vertices
        return dout

    def forward(self, mode=None, alpha_lowermost=0.0, alpha_rh_match=1.0, alpha_obstacle_in=0.0, alpha_obstacle_out=0.0, alpha_gaze=0.0, alpha_rh_obstacle_in=0.0, alpha_rh_obstacle_out=0.0, alpha_wrist=0.0):
        """
        :param mode    (str)                -- str indicating which step of n-part optimization is happening - e.g., 'tgz,w'
        :return loss   (torch.Tensor item)  -- for whichever mode is selected
        """
        if self.cfg.pgprior:
            self.z.requires_grad = True if 'z' in mode else False
            self.w.requires_grad = True if 'w' in mode else False
            self.angle.requires_grad = True if 'a' in mode else False
            self.transl_z.requires_grad, self.global_orient2.requires_grad, self.global_orient3.requires_grad = False, False, False
            if 'tg' in mode or 'gt' in mode:
                self.transl_x.requires_grad, self.transl_y.requires_grad, self.global_orient1.requires_grad = True, True, True
                pred_params = self.pose_ground_pred()
                self.transl_z.data, self.global_orient2.data, self.global_orient3.data = pred_params['transl_z'], pred_params['pitch'], pred_params['roll']
            else:
                self.transl_x.requires_grad, self.transl_y.requires_grad, self.global_orient1.requires_grad = False, False, False
        else:
            pass
        loss = self.losses.gan_loss(transl=[self.transl_x, self.transl_y, self.transl_z], global_orient=[self.global_orient1, self.global_orient2, self.global_orient3], z=self.z, w=self.w, a=self.angle, extras=self.extras, alpha_lowermost=alpha_lowermost, alpha_rh_match=alpha_rh_match, alpha_obstacle_in=alpha_obstacle_in, alpha_obstacle_out=alpha_obstacle_out, alpha_gaze=alpha_gaze, alpha_rh_obstacle_in=alpha_rh_obstacle_in, alpha_rh_obstacle_out=alpha_rh_obstacle_out, alpha_wrist=alpha_wrist)
        return loss


def backward_hook(self, grad_inputs, grad_outputs, scale=1):
    new_grad_inputs = tuple([(g * scale if g is not None else g) for g in grad_inputs])
    return new_grad_inputs


class ScaleGradient(nn.Module):

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.register_full_backward_hook(partial(backward_hook, scale=scale))

    def forward(self, x):
        return x


def scale_grad(x, scale):
    return ScaleGradient(scale)(x)


class Latent(FLEX):
    """
    Latent Optimization for smoothing & having a single control over all latents (Reference: https://arxiv.org/abs/2206.09027).
    """

    def __init__(self, *args, **kwargs):
        super(Latent, self).__init__(*args, **kwargs, requires_grad=False)
        self.z_size = self.z.shape[1]
        self.w_size = self.w.shape[1]
        self.a_size = self.angle.shape[1]
        self.g_size = 3
        self.t_size = 3
        batch_size = self.z.shape[0]
        self.latent_z = nn.Parameter(torch.randn((batch_size, 512), device=self.device), requires_grad=True)
        total_size = self.z_size + self.w_size + self.a_size + self.t_size + self.g_size + 1
        if not self.cfg.pgprior:
            total_size += 3
        mlp = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, total_size))
        self.mlp = mlp

    def forward(self, mode=None, alpha_lowermost=0.0, alpha_rh_match=1.0, alpha_obstacle_in=0.0, alpha_obstacle_out=0.0, alpha_gaze=0.0, alpha_rh_obstacle_in=0.0, alpha_rh_obstacle_out=0.0, alpha_wrist=0.0, prediction_scale=None, gradient_scale=None):
        """
        :return loss   (torch.Tensor item)  -- for whichever mode is selected
        """
        prediction_scale = {k: float(v) for k, v in prediction_scale.items()}
        gradient_scale = {k: float(v) for k, v in gradient_scale.items()}
        prediction = self.mlp(self.latent_z)
        d = 0
        if 'z' in mode:
            z = prediction[:, d:d + self.z.shape[1]] * prediction_scale['z']
            d += self.z_size
        else:
            z = self.z
        if 't' in mode:
            transl_x = prediction[:, d] * prediction_scale['transl']
            d += 1
            transl_y = prediction[:, d] * prediction_scale['transl']
            d += 1
            if not self.cfg.pgprior:
                transl_z = prediction[:, d] * prediction_scale['transl']
                d += 1
        else:
            transl_x = self.transl_x
            transl_y = self.transl_y
            transl_z = self.transl_z
        if 'g' in mode:
            global_orient1 = prediction[:, d] * prediction_scale['orient']
            d += 1
            if not self.cfg.pgprior:
                global_orient2 = prediction[:, d] * prediction_scale['orient']
                d += 1
                global_orient3 = prediction[:, d] * prediction_scale['orient']
                d += 1
        else:
            global_orient1 = self.global_orient1
            global_orient2 = self.global_orient2
            global_orient3 = self.global_orient3
        if 'w' in mode:
            w = prediction[:, d:d + self.w.shape[1]]
            w = w / torch.norm(w, dim=1, keepdim=True) * prediction_scale['w']
            d += self.w_size
        else:
            w = self.w
        if 'a' in mode:
            angle = prediction[:, d:d + self.angle.shape[1]] * prediction_scale['angle']
            d += self.a_size
        else:
            angle = self.angle
        if 'z' in mode:
            z = scale_grad(z, gradient_scale['z'])
        if 'tg' in mode or 'gt' in mode:
            transl_x = scale_grad(transl_x, gradient_scale['transl'])
            transl_y = scale_grad(transl_y, gradient_scale['transl'])
            global_orient1 = scale_grad(global_orient1, gradient_scale['orient'])
            pose_output = self.gan_body.decode(z)['pose_body'].reshape(self.bs, -1)
            if self.cfg.pgprior:
                pred_params = self.pose_ground_prior(pose_output)
                transl_z, global_orient2, global_orient3 = pred_params['transl_z'], pred_params['pitch'], pred_params['roll']
        if 'w' in mode:
            w = scale_grad(w, gradient_scale['w'])
        if 'a' in mode:
            angle = scale_grad(angle, gradient_scale['angle'])
        loss = self.losses.gan_loss(transl=[transl_x, transl_y, transl_z], global_orient=[global_orient1, global_orient2, global_orient3], z=z, w=w, a=angle, extras=self.extras, alpha_lowermost=alpha_lowermost, alpha_rh_match=alpha_rh_match, alpha_obstacle_in=alpha_obstacle_in, alpha_obstacle_out=alpha_obstacle_out, alpha_gaze=alpha_gaze, alpha_rh_obstacle_in=alpha_rh_obstacle_in, alpha_rh_obstacle_out=alpha_rh_obstacle_out, alpha_wrist=alpha_wrist)
        self.z.data = z
        self.transl_x.data = transl_x
        self.transl_y.data = transl_y
        self.transl_z.data = transl_z
        self.global_orient1.data = global_orient1
        self.global_orient2.data = global_orient2
        self.global_orient3.data = global_orient3
        self.w.data = w
        self.angle.data = angle
        return loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BatchFlatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ContinousRotReprDecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 2])], {})),
    (ConvertPointsFromHomogeneous,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvertPointsToHomogeneous,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DegToRad,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NormalDistDecoder,
     lambda: ([], {'num_feat_in': 4, 'latentD': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RadToDeg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlock,
     lambda: ([], {'Fin': 4, 'Fout': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ScaleGradient,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

