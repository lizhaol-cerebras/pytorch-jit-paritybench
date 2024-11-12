
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


import torchvision


import random


import re


import torch.nn as nn


import torch.nn.functional as F


import numpy.random as npr


import itertools


import time


from torch import nn


from torch.autograd import Variable


from torch.autograd import Function


from enum import Enum


import numpy.testing as npt


import torch.optim as optim


import scipy


import scipy.optimize as opt


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch import Tensor


from scipy.spatial import Delaunay


from scipy.spatial.qhull import QhullError


class ReidEncoder(nn.Module):

    def __init__(self):
        super(ReidEncoder, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self._init_param()

    def _init_param(self):
        nn.init.eye_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.eye_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    original_dist = np.concatenate([np.concatenate([q_q_dist, q_g_dist], axis=1), np.concatenate([q_g_dist.T, g_g_dist], axis=1)], axis=0)
    original_dist = 2.0 - 2 * original_dist
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1.0 * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argpartition(original_dist, range(1, k1 + 1))
    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]
    for i in range(all_num):
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2.0 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1.0 * weight / np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2.0 - temp_min)
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


class GraphNet(nn.Module):

    def __init__(self):
        super(GraphNet, self).__init__()
        self.reid_enc = ReidEncoder()
        self.cross_graph = nn.Linear(512, 512)

    def kronecker(self, A, B):
        AB = torch.einsum('ab,cd->acbd', A, B)
        AB = AB.view(A.size(0) * B.size(0), A.size(1) * B.size(1))
        return AB

    def forward(self, U_src, U_tgt, kf_gate, reid_thr, iou, start_src, end_src, start_tgt, end_tgt, seq_name, inverse_flag):
        if seq_name in cfg.DATA.STATIC:
            Mp0 = torch.matmul(U_src.transpose(1, 2), U_tgt) + iou.unsqueeze(0)
        elif seq_name in cfg.DATA.MOVING:
            Mp0 = torch.matmul(U_src.transpose(1, 2), U_tgt)
        emb1, emb2 = U_src.transpose(1, 2), U_tgt.transpose(1, 2)
        m_emb1 = torch.bmm(Mp0, emb2)
        m_emb2 = torch.bmm(Mp0.transpose(1, 2), emb1)
        lambda_1 = torch.norm(emb1, p=2, dim=2, keepdim=True).repeat(1, 1, 512) / torch.norm(m_emb1, p=2, dim=2, keepdim=True).repeat(1, 1, 512)
        lambda_2 = torch.norm(emb2, p=2, dim=2, keepdim=True).repeat(1, 1, 512) / torch.norm(m_emb2, p=2, dim=2, keepdim=True).repeat(1, 1, 512)
        emb1_new = F.relu(self.cross_graph(emb1 + lambda_1 * m_emb1))
        emb2_new = F.relu(self.cross_graph(emb2 + lambda_2 * m_emb2))
        emb1_new = F.normalize(emb1_new.squeeze(0), p=2, dim=1).unsqueeze(0)
        emb2_new = F.normalize(emb2_new.squeeze(0), p=2, dim=1).unsqueeze(0)
        Mp_before = torch.matmul(emb1_new, emb2_new.transpose(1, 2)).squeeze(0)
        Mp = torch.Tensor(1.0 - re_ranking(Mp_before, emb1_new.squeeze(0) @ emb1_new.squeeze(0).t(), emb2_new.squeeze(0) @ emb2_new.squeeze(0).t(), k1=1, k2=1))
        Mpp = Mp.transpose(0, 1).reshape(Mp.shape[0] * Mp.shape[1]).unsqueeze(0).t()
        if Mp.shape[0] == 1 and Mp.shape[1] == 1:
            thr_flag = torch.Tensor(Mp.shape[0], Mp.shape[1]).zero_()
            for i in range(Mp.shape[0]):
                for j in range(Mp.shape[1]):
                    if kf_gate[i][j] == -1 or iou[i][j] == 0 or Mp_before[i][j] < reid_thr:
                        thr_flag[i][j] = 1
            return np.array([0, 0]), thr_flag
        kro_one_src = torch.ones(emb1_new.shape[1], emb1_new.shape[1])
        kro_one_tgt = torch.ones(emb2_new.shape[1], emb2_new.shape[1])
        mee1 = self.kronecker(kro_one_tgt, start_src).long()
        mee2 = self.kronecker(kro_one_tgt, end_src).long()
        mee3 = self.kronecker(start_tgt, kro_one_src).long()
        mee4 = self.kronecker(end_tgt, kro_one_src).long()
        src = torch.cat([emb1_new.squeeze(0).unsqueeze(1).repeat(1, emb1_new.shape[1], 1), emb1_new.repeat(emb1_new.shape[1], 1, 1)], dim=2)
        tgt = torch.cat([emb2_new.squeeze(0).unsqueeze(1).repeat(1, emb2_new.shape[1], 1), emb2_new.repeat(emb2_new.shape[1], 1, 1)], dim=2)
        src_tgt = (src.reshape(-1, 1024) @ tgt.reshape(-1, 1024).t()).reshape(emb1_new.shape[1], emb1_new.shape[1], emb2_new.shape[1], emb2_new.shape[1])
        mask = ((mee1 - mee2).bool() & (mee3 - mee4).bool()).float()
        M = src_tgt[mee1, mee2, mee3, mee4] / 2
        M = mask * M
        M = M.unsqueeze(0)
        k = (Mp.shape[0] - 1) * (Mp.shape[1] - 1)
        M[0] = k * torch.eye(M.shape[1], M.shape[2]) - M[0]
        if Mp.shape[0] == 1 or Mp.shape[1] == 1:
            M[0] = torch.zeros_like(M[0])
            None
        else:
            M[0] = torch.cholesky(M[0])
        if Mp.shape[0] > Mp.shape[1]:
            n, m, p = M.shape[1], Mp.shape[1], Mp.shape[0]
            a = np.zeros((p, n))
            b = np.zeros((m, n))
            for i in range(p):
                for j in range(m):
                    a[i][j * p + i] = 1
            for i in range(m):
                b[i][i * p:(i + 1) * p] = 1
            x = cp.Variable(n)
            obj = cp.Minimize(0.5 * cp.sum_squares(M.squeeze(0).numpy() @ x) - Mpp.numpy().T @ x)
            cons = [a @ x <= 1, b @ x == 1, x >= 0]
            prob = cp.Problem(obj, cons)
            prob.solve(solver=cp.SCS, gpu=True, use_indirect=True)
            s = torch.tensor(x.value)
            s = s.reshape(Mp.shape[1], Mp.shape[0]).t().unsqueeze(0)
            s = torch.relu(s) - torch.relu(s - 1)
        elif Mp.shape[0] == Mp.shape[1]:
            n, m, p = M.shape[1], Mp.shape[0], Mp.shape[1]
            x = cp.Variable(n)
            a = np.zeros((m + p, n))
            for i in range(p):
                for j in range(m):
                    a[i][j * p + i] = 1
            for i in range(m):
                a[i + p][i * p:(i + 1) * p] = 1
            obj = cp.Minimize(0.5 * cp.sum_squares(M.squeeze(0).numpy() @ x) - Mpp.numpy().T @ x)
            cons = [a @ x == 1, x >= 0]
            prob = cp.Problem(obj, cons)
            prob.solve(solver=cp.SCS, gpu=True, use_indirect=True)
            s = torch.tensor(x.value)
            s = s.reshape(Mp.shape[1], Mp.shape[0]).t().unsqueeze(0)
            s = torch.relu(s) - torch.relu(s - 1)
        thr_flag = torch.Tensor(Mp.shape[0], Mp.shape[1]).zero_()
        for i in range(Mp.shape[0]):
            for j in range(Mp.shape[1]):
                if kf_gate[i][j] == -1 or iou[i][j] == 0 or Mp_before[i][j] < reid_thr:
                    thr_flag[i][j] = 1
        if s.shape[1] >= s.shape[2]:
            s = s.squeeze(0).t()
            s = np.array(s)
            n = min(s.shape)
            Y = s.copy()
            Z = np.zeros(Y.shape)
            replace = np.min(Y) - 1
            for i in range(n):
                z = np.unravel_index(np.argmax(Y), Y.shape)
                Z[z] = 1
                Y[z[0], :] = replace
                Y[:, z[1]] = replace
            match_tra = np.argmax(Z, 1)
            match_tra = torch.tensor(match_tra)
            if inverse_flag == False:
                output = np.array(torch.cat([match_tra.unsqueeze(0), torch.arange(len(match_tra)).unsqueeze(0)], 0)).T
            if inverse_flag == True:
                thr_flag = thr_flag.t()
                output = np.array(torch.cat([torch.arange(len(match_tra)).unsqueeze(0), match_tra.unsqueeze(0)], 0)).T
        return output, thr_flag


class QPSolvers(Enum):
    PDIPM_BATCHED = 1
    CVXPY = 2


def bger(x, y):
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


def expandParam(X, nBatch, nDim):
    if X.ndimension() in (0, nDim) or X.nelement() == 0:
        return X, False
    elif X.ndimension() == nDim - 1:
        return X.unsqueeze(0).expand(*([nBatch] + list(X.size()))), True
    else:
        raise RuntimeError('Unexpected number of dimensions.')


def extract_nBatch(Q, p, G, h, A, b):
    dims = [3, 2, 3, 2, 3, 2]
    params = [Q, p, G, h, A, b]
    for param, dim in zip(params, dims):
        if param.ndimension() == dim:
            return param.size(0)
    return 1


def QPFunction(eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20, solver=QPSolvers.CVXPY, check_Q_spd=True):


    class QPFunctionFn(Function):

        @staticmethod
        def forward(ctx, Q_, p_, G_, h_, A_, b_):
            """Solve a batch of QPs.

            This function solves a batch of QPs, each optimizing over
            `nz` variables and having `nineq` inequality constraints
            and `neq` equality constraints.
            The optimization problem for each instance in the batch
            (dropping indexing from the notation) is of the form

                \\hat z =   argmin_z 1/2 z^T Q z + p^T z
                        subject to Gz <= h
                                    Az  = b

            where Q \\in S^{nz,nz},
                S^{nz,nz} is the set of all positive semi-definite matrices,
                p \\in R^{nz}
                G \\in R^{nineq,nz}
                h \\in R^{nineq}
                A \\in R^{neq,nz}
                b \\in R^{neq}

            These parameters should all be passed to this function as
            Variable- or Parameter-wrapped Tensors.
            (See torch.autograd.Variable and torch.nn.parameter.Parameter)

            If you want to solve a batch of QPs where `nz`, `nineq` and `neq`
            are the same, but some of the contents differ across the
            minibatch, you can pass in tensors in the standard way
            where the first dimension indicates the batch example.
            This can be done with some or all of the coefficients.

            You do not need to add an extra dimension to coefficients
            that will not change across all of the minibatch examples.
            This function is able to infer such cases.

            If you don't want to use any equality or inequality constraints,
            you can set the appropriate values to:

                e = Variable(torch.Tensor())

            Parameters:
            Q:  A (nBatch, nz, nz) or (nz, nz) Tensor.
            p:  A (nBatch, nz) or (nz) Tensor.
            G:  A (nBatch, nineq, nz) or (nineq, nz) Tensor.
            h:  A (nBatch, nineq) or (nineq) Tensor.
            A:  A (nBatch, neq, nz) or (neq, nz) Tensor.
            b:  A (nBatch, neq) or (neq) Tensor.

            Returns: \\hat z: a (nBatch, nz) Tensor.
            """
            nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
            Q, _ = expandParam(Q_, nBatch, 3)
            p, _ = expandParam(p_, nBatch, 2)
            G, _ = expandParam(G_, nBatch, 3)
            h, _ = expandParam(h_, nBatch, 2)
            A, _ = expandParam(A_, nBatch, 3)
            b, _ = expandParam(b_, nBatch, 2)
            if check_Q_spd:
                for i in range(nBatch):
                    e, _ = torch.eig(Q[i])
                    if not torch.all(e[:, 0] > 0):
                        raise RuntimeError('Q is not SPD.')
            _, nineq, nz = G.size()
            neq = A.size(1) if A.nelement() > 0 else 0
            assert neq > 0 or nineq > 0
            ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz
            if solver == QPSolvers.PDIPM_BATCHED:
                ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)
                zhats, ctx.nus, ctx.lams, ctx.slacks = pdipm_b.forward(Q, p, G, h, A, b, ctx.Q_LU, ctx.S_LU, ctx.R, eps, verbose, notImprovedLim, maxIter)
            elif solver == QPSolvers.CVXPY:
                vals = torch.Tensor(nBatch).type_as(Q)
                zhats = torch.Tensor(nBatch, ctx.nz).type_as(Q)
                lams = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
                nus = torch.Tensor(nBatch, ctx.neq).type_as(Q) if ctx.neq > 0 else torch.Tensor()
                slacks = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
                for i in range(nBatch):
                    Ai, bi = (A[i], b[i]) if neq > 0 else (None, None)
                    vals[i], zhati, nui, lami, si = solvers.cvxpy.forward_single_np(*[(x.detach().cpu().numpy() if x is not None else None) for x in (Q[i], p[i], G[i], h[i], Ai, bi)])
                    zhats[i] = torch.Tensor(zhati)
                    lams[i] = torch.Tensor(lami)
                    slacks[i] = torch.Tensor(si)
                    if neq > 0:
                        nus[i] = torch.Tensor(nui)
                ctx.vals = vals
                ctx.lams = lams
                ctx.nus = nus
                ctx.slacks = slacks
            else:
                assert False
            ctx.save_for_backward(zhats, Q_, p_, G_, h_, A_, b_)
            return zhats

        @staticmethod
        def backward(ctx, dl_dzhat):
            zhats, Q, p, G, h, A, b = ctx.saved_tensors
            nBatch = extract_nBatch(Q, p, G, h, A, b)
            Q, Q_e = expandParam(Q, nBatch, 3)
            p, p_e = expandParam(p, nBatch, 2)
            G, G_e = expandParam(G, nBatch, 3)
            h, h_e = expandParam(h, nBatch, 2)
            A, A_e = expandParam(A, nBatch, 3)
            b, b_e = expandParam(b, nBatch, 2)
            neq, nineq = ctx.neq, ctx.nineq
            if solver == QPSolvers.CVXPY:
                ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)
            d = torch.clamp(ctx.lams, min=1e-08) / torch.clamp(ctx.slacks, min=1e-08)
            pdipm_b.factor_kkt(ctx.S_LU, ctx.R, d)
            dx, _, dlam, dnu = pdipm_b.solve_kkt(ctx.Q_LU, d, G, A, ctx.S_LU, dl_dzhat, torch.zeros(nBatch, nineq).type_as(G), torch.zeros(nBatch, nineq).type_as(G), torch.zeros(nBatch, neq).type_as(G) if neq > 0 else torch.Tensor())
            dps = dx
            dGs = bger(dlam, zhats) + bger(ctx.lams, dx)
            if G_e:
                dGs = dGs.mean(0)
            dhs = -dlam
            if h_e:
                dhs = dhs.mean(0)
            if neq > 0:
                dAs = bger(dnu, zhats) + bger(ctx.nus, dx)
                dbs = -dnu
                if A_e:
                    dAs = dAs.mean(0)
                if b_e:
                    dbs = dbs.mean(0)
            else:
                dAs, dbs = None, None
            dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
            if Q_e:
                dQs = dQs.mean(0)
            if p_e:
                dps = dps.mean(0)
            grads = dQs, dps, dGs, dhs, dAs, dbs
            return grads
    return QPFunctionFn.apply


class Voting(nn.Module):
    """
    Voting Layer computes a new row-stotatic matrix with softmax. A large number (alpha) is multiplied to the input
    stochastic matrix to scale up the difference.
    Parameter: value multiplied before softmax alpha
               threshold that will ignore such points while calculating displacement in pixels pixel_thresh
    Input: permutation or doubly stochastic matrix s
           ///point set on source image P_src
           ///point set on target image P_tgt
           ground truth number of effective points in source image ns_gt
    Output: softmax matrix s
    """

    def __init__(self, alpha=200, pixel_thresh=None):
        super(Voting, self).__init__()
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)
        self.pixel_thresh = pixel_thresh

    def forward(self, s, nrow_gt, ncol_gt=None):
        ret_s = torch.zeros_like(s)
        for b, n in enumerate(nrow_gt):
            if ncol_gt is None:
                ret_s[b, 0:n, :] = self.softmax(self.alpha * s[b, 0:n, :])
            else:
                ret_s[b, 0:n, 0:ncol_gt[b]] = self.softmax(self.alpha * s[b, 0:n, 0:ncol_gt[b]])
        return ret_s


def gh(n):
    A = np.ones((n, n)) - np.eye(n)
    edge_num = int(np.sum(A, axis=(0, 1)))
    n_pad = n
    edge_pad = edge_num
    G = np.zeros((n_pad, edge_pad), dtype=np.float32)
    H = np.zeros((n_pad, edge_pad), dtype=np.float32)
    start = np.zeros((n_pad, n_pad), dtype=np.float32)
    end = np.zeros((n_pad, n_pad), dtype=np.float32)
    edge_idx = 0
    for i in range(n):
        for j in range(n):
            start[i, j] = i
            end[i, j] = j
            if A[i, j] == 1:
                G[i, edge_idx] = 1
                H[j, edge_idx] = 1
                edge_idx += 1
    return G, H, start, end


def kronecker(A, B):
    AB = torch.einsum('ab,cd->acbd', A, B)
    AB = AB.view(A.size(0) * B.size(0), A.size(1) * B.size(1))
    return AB


def reshape_edge_feature(F: 'Tensor', G: 'Tensor', H: 'Tensor', device=None):
    """
    Reshape edge feature matrix into X, where features are arranged in the order in G, H.
    :param F: raw edge feature matrix
    :param G: factorized adjacency matrix, where A = G * H^T
    :param H: factorized adjacancy matrix, where A = G * H^T
    :param device: device. If not specified, it will be the same as the input
    :return: X
    """
    if device is None:
        device = F.device
    batch_num = F.shape[0]
    feat_dim = F.shape[1]
    point_num, edge_num = G.shape[1:3]
    X = torch.zeros(batch_num, 2 * feat_dim, edge_num, dtype=torch.float32, device=device)
    X[:, 0:feat_dim, :] = torch.matmul(F, G)
    X[:, feat_dim:2 * feat_dim, :] = torch.matmul(F, H)
    return X


class GMNet(nn.Module):

    def __init__(self):
        super(GMNet, self).__init__()
        self.reid_enc = ReidEncoder()
        self.voting_layer = Voting(alpha=cfg.TRAIN.VOTING_ALPHA)
        self.cross_graph = nn.Linear(512, 512)
        nn.init.eye_(self.cross_graph.weight)
        nn.init.constant_(self.cross_graph.bias, 0)

    def forward(self, tra, det, iou):
        feat_tra_all = []
        for tra_i in range(len(tra.fnm)):
            feat_tra0 = self.reid_enc(tra.fnm[tra_i])
            tracklet_feat = feat_tra0.mean(0).unsqueeze(0)
            feat_tra_all.append(tracklet_feat)
        feat_tra = torch.cat(feat_tra_all)
        feat_det = self.reid_enc(det.x)
        data1 = feat_tra
        data2 = feat_det
        G1, H1, edge_num1 = gh(data1)
        G2, H2, edge_num2 = gh(data2)
        G_src = torch.tensor(G1).unsqueeze(0)
        G_tgt = torch.tensor(G2).unsqueeze(0)
        H_src = torch.tensor(H1).unsqueeze(0)
        H_tgt = torch.tensor(H2).unsqueeze(0)
        U_src = data1.t().unsqueeze(0).cpu()
        U_tgt = data2.t().unsqueeze(0).cpu()
        Mp0 = torch.matmul(U_src.transpose(1, 2), U_tgt)
        Mp0 = Mp0 + iou
        emb1, emb2 = U_src.transpose(1, 2), U_tgt.transpose(1, 2)
        m_emb1 = torch.bmm(Mp0, emb2)
        m_emb2 = torch.bmm(Mp0.transpose(1, 2), emb1)
        lambda_1 = (torch.norm(emb1, p=2, dim=2, keepdim=True).repeat(1, 1, 512) / torch.norm(m_emb1, p=2, dim=2, keepdim=True).repeat(1, 1, 512)).detach()
        lambda_2 = (torch.norm(emb2, p=2, dim=2, keepdim=True).repeat(1, 1, 512) / torch.norm(m_emb2, p=2, dim=2, keepdim=True).repeat(1, 1, 512)).detach()
        emb1_new = F.relu(self.cross_graph(emb1 + lambda_1 * m_emb1))
        emb2_new = F.relu(self.cross_graph(emb2 + lambda_2 * m_emb2))
        emb1_new = F.normalize(emb1_new.squeeze(0), p=2, dim=1).unsqueeze(0)
        emb2_new = F.normalize(emb2_new.squeeze(0), p=2, dim=1).unsqueeze(0)
        X2 = reshape_edge_feature(emb1_new.transpose(1, 2).cpu(), G_src, H_src)
        Y2 = reshape_edge_feature(emb2_new.transpose(1, 2).cpu(), G_tgt, H_tgt)
        Me = torch.matmul(X2.transpose(1, 2), Y2).squeeze(0) / 2
        Mp = torch.matmul(emb1_new, emb2_new.transpose(1, 2)).squeeze(0)
        a1 = Me.transpose(0, 1)
        a2 = a1.reshape(Me.shape[0] * Me.shape[1])
        K_G = kronecker(G_tgt.squeeze(0), G_src.squeeze(0)).detach()
        K1Me = a2 * K_G
        del K_G
        del a2
        del Me
        gc.collect()
        torch.cuda.empty_cache()
        K_H = kronecker(H_tgt.squeeze(0), H_src.squeeze(0)).detach()
        M = torch.mm(K1Me, K_H.t())
        del K1Me
        del K_H
        gc.collect()
        torch.cuda.empty_cache()
        Mpp = Mp.transpose(0, 1).reshape(Mp.shape[0] * Mp.shape[1])
        M = M.unsqueeze(0)
        k = (Mp.shape[0] - 1) * (Mp.shape[1] - 1)
        M[0] = k * torch.eye(M.shape[1], M.shape[2]) - M[0]
        M = M.squeeze(0)
        if Mp.shape[0] > Mp.shape[1]:
            n, m, p = M.shape[0], Mp.shape[1], Mp.shape[0]
            a = torch.zeros(p, n)
            b = torch.zeros(m, n)
            for i in range(p):
                for j in range(m):
                    a[i][j * p + i] = 1
            for i in range(m):
                b[i][i * p:(i + 1) * p] = 1
            qp = QPFunction(check_Q_spd=False)
            G = -torch.eye(n)
            h = torch.zeros(n)
            bb = torch.ones(m)
            bbb = torch.ones(p)
            hh = torch.cat((h, bbb))
            GG = torch.cat((G, a), 0)
            s = qp(M, -Mpp, GG, hh, b, bb)
            s = s.reshape(Mp.shape[1], Mp.shape[0]).unsqueeze(0)
            s = torch.relu(s) - torch.relu(s - 1)
            s = self.voting_layer(s, torch.tensor([Mp.shape[1]]), torch.tensor([Mp.shape[0]])).permute(0, 2, 1)
        elif Mp.shape[0] == Mp.shape[1]:
            n, m, p = M.shape[0], Mp.shape[0], Mp.shape[1]
            a = torch.zeros(m + p, n)
            for i in range(p):
                for j in range(m):
                    a[i][j * p + i] = 1
            for i in range(m):
                a[i + p][i * p:(i + 1) * p] = 1
            qp = QPFunction(check_Q_spd=False)
            G = -torch.eye(n)
            h = torch.zeros(n)
            b = torch.ones(m + p)
            s = qp(M, -Mpp, G, h, a, b)
            s = s.reshape(Mp.shape[1], Mp.shape[0]).t().unsqueeze(0)
            s = torch.relu(s) - torch.relu(s - 1)
            s = self.voting_layer(s, torch.tensor([Mp.shape[0]]), torch.tensor([Mp.shape[1]]))
        else:
            n, m, p = M.shape[0], Mp.shape[0], Mp.shape[1]
            a = torch.zeros(p, n)
            b = torch.zeros(m, n)
            for i in range(p):
                for j in range(m):
                    a[i][j * p + i] = 1
            for i in range(m):
                b[i][i * p:(i + 1) * p] = 1
            qp = QPFunction(check_Q_spd=False)
            G = -torch.eye(n)
            h = torch.zeros(n)
            bb = torch.ones(m)
            bbb = torch.ones(p)
            hh = torch.cat((h, bbb))
            GG = torch.cat((G, a), 0)
            s = qp(M, -Mpp, GG, hh, b, bb)
            s = s.reshape(Mp.shape[1], Mp.shape[0]).t().unsqueeze(0)
            s = torch.relu(s) - torch.relu(s - 1)
            s = self.voting_layer(s, torch.tensor([Mp.shape[0]]), torch.tensor([Mp.shape[1]]))
        return s


class Loss(nn.Module):

    def forward(self, x, gt):
        x = x.reshape(-1)
        gt = gt.reshape(-1)
        pa = len(gt) * len(gt) / (len(gt) - gt.sum()) / 2 / gt.sum()
        weight_0 = pa * gt.sum() / len(gt)
        weight_1 = pa * (len(gt) - gt.sum()) / len(gt)
        weight = torch.zeros(len(gt))
        for i in range(len(gt)):
            if gt[i] == 0:
                weight[i] = weight_0
            elif gt[i] == 1:
                weight[i] = weight_1
            else:
                raise RuntimeError('loss weight error')
        loss = nn.BCELoss(weight=weight)
        out = loss(x, gt)
        return out

