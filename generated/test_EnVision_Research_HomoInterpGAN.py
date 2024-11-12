
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


import scipy.misc


import numpy as np


import torch


import pandas as pd


import torchvision as tv


import random


import torch.utils.data as data


import torchvision.transforms as transforms


import math


from collections import OrderedDict


import torch.nn as nn


from torch.utils import model_zoo


from torch import nn


import torch.nn.functional as F


import torch.optim as optim


from torch.utils.data import DataLoader


import torch as th


from scipy import linalg


from torch.autograd import Variable


from torch.nn.functional import adaptive_avg_pool2d


from torchvision import models


from torch.autograd import Function


import scipy.linalg


import time


class opt(object):

    def __init__(self, **entries):
        self.__dict__.update(entries)

    def merge_dict(self, d):
        self.__dict__.update(d)
        return self

    def merge_opt(self, o):
        d = vars(o)
        self.__dict__.update(d)
        return self

    def __str__(self):
        args_dict = vars(self)
        print_str = ''
        for k, v in args_dict.items():
            print_str += '%s: %s \n' % (k, v)
        return print_str

    def load(self, path):
        with open(path, 'r') as f:
            opt_now = yaml.load(f)
            None
        for k, v in opt_now.items():
            setattr(self, k, v)


class BaseModel(object):

    def _default_opt(self):
        self.opt = opt.opt()
        self.opt.continue_train = False
        self.opt.lr = 0.001

    def name(self):
        return 'BaseModel'

    def set_input(self, input):
        self.input = input

    def forward(self, x):
        pass

    def add_summary(self, global_step):
        d = self.get_current_errors()
        for key, value in d.items():
            self.writer.add_scalar(key, value, global_step)

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def print_current_errors(self, epoch, i, record_file=None, print_msg=True):
        errors = self.get_current_errors()
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
        if print_msg:
            None
        if record_file is not None:
            with open(record_file + '/loss.txt', 'w') as f:
                None
        return message

    def save(self, label):
        pass

    def load(self, pretrain_path, label):
        pass

    def save_network(self, network, save_dir, save_name):
        save_path = os.path.join(save_dir, save_name)
        None
        torch.save(network.cpu().state_dict(), save_path)
        network

    def resume_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.save_dir, save_filename)
        None
        network.load_state_dict(torch.load(save_path))

    def load_network(self, pretrain_path, network, file_name):
        save_path = os.path.join(pretrain_path, file_name)
        None
        network.load_state_dict(torch.load(save_path))

    def print_network(self, net, filepath=None):
        if filepath is None:
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            None
            None
        else:
            num_params = 0
            with open(filepath + '/network.txt', 'w') as f:
                for param in net.parameters():
                    num_params += param.numel()
                None
                f.write('Total number of parameters: %d' % num_params)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        None

    def backward_recon(self, img, net, target_feature, layerList, lr=1, iter=500, tv=10):
        """
        backward to reconstruct the target feature
        :param img: the input image
        :param target_feature: the target feature, it is a list whose length is 3
        :return: output image
        """
        _img = util.toVariable(img, requires_grad=True)
        target_feature = [util.toVariable(t, requires_grad=False) for t in target_feature]
        optim = torch.optim.LBFGS([_img], lr=lr, max_iter=iter)
        optim.n_steps = 0
        MSELoss = torch.nn.MSELoss(size_average=False)
        tv_loss = TVLoss2()

        def step():
            if _img.grad is not None:
                _img.grad.data.fill_(0)
            feat = net.forward(_img)
            loss_all = []
            for layer in layerList:
                loss_all += [MSELoss(feat[layer], target_feature[layer])]
            losstv = tv_loss(_img)
            loss = sum(loss_all) + tv * losstv
            loss.backward()
            if optim.n_steps % 25 == 0:
                msg = 'lossall=%f, ' % loss.data[0]
                for idx, l in enumerate(loss_all):
                    msg += 'loss=%f, ' % l.data[0]
                msg += 'loss_tv=%f' % losstv.data[0]
                None
            optim.n_steps += 1
            return loss
        optim.step(step)
        img = _img
        return img

    def backward_recon_1feat(self, img, net, target_feature, lr=1, iter=500, tv=10):
        """
        backward to reconstruct the target feature
        :param img: the input image
        :param target_feature: the target feature
        :return: output image
        """
        _img = util.toVariable(img, requires_grad=True)
        target_feature = target_feature.detach()
        optim = torch.optim.LBFGS([_img], lr=lr, max_iter=iter)
        optim.n_steps = 0
        MSELoss = torch.nn.MSELoss(size_average=False)
        tv_loss = TVLoss2()

        def step():
            if _img.grad is not None:
                _img.grad.data.fill_(0)
            feat = net.forward(_img)
            loss_all = MSELoss(feat, target_feature)
            losstv = tv_loss(_img)
            loss = loss_all + tv * losstv
            loss.backward()
            if optim.n_steps % 25 == 0:
                msg = 'lossall=%f, ' % loss.data[0]
                for idx, l in enumerate(loss_all):
                    msg += 'loss=%f, ' % l.data[0]
                msg += 'loss_tv=%f' % losstv.data[0]
                None
            optim.n_steps += 1
            return loss
        optim.step(step)
        img = _img
        return img

    def backward_recon_adam(self, img, net, target_feature, layerList, lr=0.001, iter=500, tv=10):
        """
        backward to reconstruct the target feature
        :param img: the input image
        :param target_feature: the target feature, it is a list whose length is 3
        :return: output image
        """
        _img = util.toVariable(img, requires_grad=True)
        target_feature = [util.toVariable(t, requires_grad=False) for t in target_feature]
        optim = torch.optim.Adam([_img], lr=lr)
        optim.n_steps = 0
        MSELoss = torch.nn.MSELoss(size_average=False)
        tv_loss = TVLoss2()

        def step():
            if _img.grad is not None:
                _img.grad.data.fill_(0)
            feat = net.forward(_img)
            loss_all = []
            for layer in layerList:
                loss_all += [MSELoss(feat[layer], target_feature[layer])]
            losstv = tv_loss(_img)
            loss = sum(loss_all) + tv * losstv
            loss.backward()
            if optim.n_steps % 25 == 0:
                msg = 'lossall=%f, ' % loss.data[0]
                for idx, l in enumerate(loss_all):
                    msg += 'loss=%f, ' % l.data[0]
                msg += 'loss_tv=%f' % losstv.data[0]
                None
            optim.n_steps += 1
            return loss
        for i in range(iter):
            optim.step(step)
            img = _img
        return img


class VGG(nn.Module, BaseModel):

    def __init__(self, pretrained=True, local_model_path=None, nChannel=64):
        super(VGG, self).__init__()
        self.features_1 = nn.Sequential(OrderedDict([('conv1_1', nn.Conv2d(3, nChannel, kernel_size=3, padding=1)), ('relu1_1', nn.ReLU(inplace=True)), ('conv1_2', nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1)), ('relu1_2', nn.ReLU(inplace=True)), ('pool1', nn.MaxPool2d(2, 2)), ('conv2_1', nn.Conv2d(nChannel, nChannel * 2, kernel_size=3, padding=1)), ('relu2_1', nn.ReLU(inplace=True)), ('conv2_2', nn.Conv2d(nChannel * 2, nChannel * 2, kernel_size=3, padding=1)), ('relu2_2', nn.ReLU(inplace=True)), ('pool2', nn.MaxPool2d(2, 2)), ('conv3_1', nn.Conv2d(nChannel * 2, nChannel * 4, kernel_size=3, padding=1)), ('relu3_1', nn.ReLU(inplace=True))]))
        self.features_2 = nn.Sequential(OrderedDict([('conv3_2', nn.Conv2d(nChannel * 4, nChannel * 4, kernel_size=3, padding=1)), ('relu3_2', nn.ReLU(inplace=True)), ('conv3_3', nn.Conv2d(nChannel * 4, nChannel * 4, kernel_size=3, padding=1)), ('relu3_3', nn.ReLU(inplace=True)), ('conv3_4', nn.Conv2d(nChannel * 4, nChannel * 4, kernel_size=3, padding=1)), ('relu3_5', nn.ReLU(inplace=True)), ('pool3', nn.MaxPool2d(2, 2)), ('conv4_1', nn.Conv2d(nChannel * 4, nChannel * 8, kernel_size=3, padding=1)), ('relu4_1', nn.ReLU(inplace=True))]))
        self.features_3 = nn.Sequential(OrderedDict([('conv4_2', nn.Conv2d(nChannel * 8, nChannel * 8, kernel_size=3, padding=1)), ('relu4_2', nn.ReLU(inplace=True)), ('conv4_3', nn.Conv2d(nChannel * 8, nChannel * 8, kernel_size=3, padding=1)), ('relu4_3', nn.ReLU(inplace=True)), ('conv4_4', nn.Conv2d(nChannel * 8, nChannel * 8, kernel_size=3, padding=1)), ('relu4_4', nn.ReLU(inplace=True)), ('pool4', nn.MaxPool2d(2, 2)), ('conv5_1', nn.Conv2d(nChannel * 8, nChannel * 8, kernel_size=3, padding=1)), ('relu5_1', nn.ReLU(inplace=True))]))
        if pretrained:
            if local_model_path is None:
                None
                model_path = 'https://www.dropbox.com/s/4lbt58k10o84l5h/vgg19g-4aff041b.pth?dl=1'
                state_dict = torch.utils.model_zoo.load_url(model_path, 'checkpoints/vgg')
            else:
                None
                state_dict = torch.load(local_model_path)
            model_dict = self.state_dict()
            state_dict = {key: value for key, value in state_dict.items() if key in model_dict}
            self.load_state_dict(state_dict)

    def forward(self, x):
        features_1 = self.features_1(x)
        features_2 = self.features_2(features_1)
        features_3 = self.features_3(features_2)
        return features_1, features_2, features_3


class _PoolingBlock(nn.Sequential):

    def __init__(self, n_convs, n_input_filters, n_output_filters, drop_rate, norm='batch'):
        super(_PoolingBlock, self).__init__()
        if norm == 'batch':
            norm_fn = nn.BatchNorm2d
        elif norm == 'instance':
            norm_fn = nn.InstanceNorm2d
        else:
            raise NotImplemented
        for i in range(n_convs):
            self.add_module('conv.%d' % (i + 1), nn.Conv2d(n_input_filters if i == 0 else n_output_filters, n_output_filters, kernel_size=3, padding=1))
            self.add_module('norm.%d' % (i + 1), norm_fn(n_output_filters))
            self.add_module('relu.%d' % (i + 1), nn.ReLU(inplace=True))
            if drop_rate > 0:
                self.add_module('drop.%d' % (i + 1), nn.Dropout(p=drop_rate))


class _TransitionUp(nn.Sequential):

    def __init__(self, n_input_filters, n_output_filters, norm='batch'):
        super(_TransitionUp, self).__init__()
        if norm == 'batch':
            norm_fn = nn.BatchNorm2d
        elif norm == 'instance':
            norm_fn = nn.InstanceNorm2d
        else:
            raise NotImplemented
        self.add_module('unpool.conv', nn.ConvTranspose2d(n_input_filters, n_output_filters, kernel_size=4, stride=2, padding=1))
        self.add_module('unpool.norm', nn.BatchNorm2d(n_output_filters))


class _Upsample(nn.Sequential):

    def __init__(self, n_input_filters, n_output_filters, norm='batch'):
        super(_Upsample, self).__init__()
        if norm == 'batch':
            norm_fn = nn.BatchNorm2d
        elif norm == 'instance':
            norm_fn = nn.InstanceNorm2d
        else:
            raise NotImplemented
        self.add_module('interp.conv', nn.Conv2d(n_input_filters, n_output_filters, kernel_size=3, padding=1))
        self.add_module('interp.norm', nn.BatchNorm2d(n_output_filters))


class Vgg_recon_noskip(nn.Module):

    def __init__(self, drop_rate=0, norm='batch'):
        super(Vgg_recon_noskip, self).__init__()
        self.recon5 = _PoolingBlock(3, 512, 512, drop_rate=drop_rate, norm=norm)
        self.upool4 = _TransitionUp(512, 512, norm=norm)
        self.upsample4 = _Upsample(512, 512, norm=norm)
        self.recon4 = _PoolingBlock(3, 512, 512, drop_rate=drop_rate, norm=norm)
        self.upool3 = _TransitionUp(512, 256, norm=norm)
        self.upsample3 = _Upsample(512, 256, norm=norm)
        self.recon3 = _PoolingBlock(3, 256, 256, drop_rate=drop_rate, norm=norm)
        self.upool2 = _TransitionUp(256, 128, norm=norm)
        self.upsample2 = _Upsample(256, 128, norm=norm)
        self.recon2 = _PoolingBlock(2, 128, 128, drop_rate=drop_rate, norm=norm)
        self.upool1 = _TransitionUp(128, 64, norm=norm)
        self.upsample1 = _Upsample(128, 64, norm=norm)
        self.recon1 = _PoolingBlock(1, 64, 64, drop_rate=drop_rate, norm=norm)
        self.recon0 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, features_3):
        recon5 = self.recon5(features_3)
        recon5 = nn.functional.upsample(recon5, scale_factor=2, mode='bilinear')
        upool4 = self.upsample4(recon5)
        recon4 = self.recon4(upool4)
        recon4 = nn.functional.upsample(recon4, scale_factor=2, mode='bilinear')
        upool3 = self.upsample3(recon4)
        recon3 = self.recon3(upool3)
        recon3 = nn.functional.upsample(recon3, scale_factor=2, mode='bilinear')
        upool2 = self.upsample2(recon3)
        recon2 = self.recon2(upool2)
        recon2 = nn.functional.upsample(recon2, scale_factor=2, mode='bilinear')
        upool1 = self.upsample1(recon2)
        recon1 = self.recon1(upool1)
        recon0 = self.recon0(recon1)
        return recon0


class perceptural_loss(nn.Module):

    def __init__(self):
        super(perceptural_loss, self).__init__()
        self.vgg = base_network.VGG(pretrained=True).eval()
        self.vgg.features_1 = nn.DataParallel(self.vgg.features_1)
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        feat_x = self.vgg.features_1(x)
        feat_y = self.vgg.features_1(y)
        loss = self.mse(feat_x, feat_y.detach())
        return loss


class encoder(nn.Module):

    def __init__(self, pretrained=True):
        super(encoder, self).__init__()
        self.model = base_network.VGG(pretrained=pretrained)

    def forward(self, x):
        y = self.model(x)
        y = y[-1]
        return y


class _interp_branch(nn.Module):
    """
    one branch of the interpolator network
    """

    def __init__(self, in_channels, out_channels):
        super(_interp_branch, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        return self.model(x)


class interp_net(nn.Module):

    def __init__(self, n_branch, channels=512):
        """
        the multi-branch interpolator network
        :param channels: channels of the latent space.
        :param n_branch: number of branches. each branch deals with an attribtue
        """
        super(interp_net, self).__init__()
        self.n_branch = n_branch
        branch = []
        branch_fn = _interp_branch
        for i in range(n_branch):
            branch += [branch_fn(channels, channels)]
        self.branch = nn.ModuleList(branch)

    def forward(self, feat1, feat2, selective_vector, **kwargs):
        y = feat2 - feat1
        selective_tensor = selective_vector.unsqueeze(2).unsqueeze(3)
        selective_tensor = selective_tensor.expand((-1, -1, y.size(2), y.size(3)))
        z = []
        for i in range(self.n_branch):
            tmp = self.branch[i](y)
            tmp = tmp * selective_tensor[:, i:i + 1, :, :]
            z += [tmp]
        z = feat1 + sum(z)
        return z


class decoder(nn.Module):

    def __init__(self, upsample_mode='nearest', pretrained=True):
        super(decoder, self).__init__()
        self.model = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.Upsample(scale_factor=2, mode=upsample_mode), nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Upsample(scale_factor=2, mode=upsample_mode), nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Upsample(scale_factor=2, mode=upsample_mode), nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Upsample(scale_factor=2, mode=upsample_mode), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 3, kernel_size=3, padding=1)])
        if pretrained:
            model_path = 'https://www.dropbox.com/s/8lwmwfs42w5oioi/homo-decoder-8a84d0ce.pth?dl=1'
            state_dict = torch.utils.model_zoo.load_url(model_path, 'checkpoints/vgg')
            self.load_state_dict(state_dict)

    def forward(self, x):
        y = self.model(x)
        return y


class discrim(nn.Module):
    """
    attr is the input attribute list that is consistent with data/attreibuteDataset/Dataset_attr_merged_v2,
    e.g., Moustache@#No_Beard@Goatee,Smile,Young,Bangs
    """

    def __init__(self, attr):
        super(discrim, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(512, 256, 1), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.2, True), nn.Conv2d(256, 256, 4, padding=1, stride=2), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.2, True), nn.Conv2d(256, 256, 4, padding=1, stride=2), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.2, True), nn.Conv2d(256, 256, 2))
        self.ifReal = nn.Conv2d(256, 256, 1)
        attr_branches = []
        attr = attr.split(',')
        for i in range(len(attr)):
            attr_now = attr[i].split('@')
            branch_now = nn.Conv2d(256, len(attr_now), 1)
            attr_branches += [branch_now]
        attr_branches = nn.ModuleList(attr_branches)
        self.attr_branches = attr_branches
        self.model = self.model
        self.ifReal = self.ifReal

    def forward(self, x):
        y = self.model(x)
        ifReal = self.ifReal(y)
        attributes = []
        for branch_now in self.attr_branches:
            attribute_now = branch_now(y).squeeze(2).squeeze(2)
            attributes += [attribute_now]
        return ifReal, attributes


class decoder2(nn.Module):

    def __init__(self, res=False, pretrained=True):
        super(decoder2, self).__init__()
        self.res = res
        self.model = base_network.Vgg_recon_noskip()
        self.model = nn.DataParallel(self.model)
        if pretrained:
            None
            state_dict = torch.load('checkpoints/decoder_noskip.pth')
            self.load_state_dict(state_dict)

    def forward(self, x, image=None):
        y = self.model(x)
        if self.res:
            y += image
        return y


class model_deploy_container(nn.Module):

    def __init__(self, encoder, interp_net, decoder):
        super(model_deploy_container, self).__init__()
        self.encoder = encoder
        self.interp_net = interp_net
        self.decoder = decoder

    def forward(self, x1, x2, v):
        """
        :param x1: the testing image
        :param x2: the reference image
        :param v: the control vector
        :return: the output image
        """
        f1, f2 = self.encoder(x1), self.encoder(x2)
        fout = self.interp_net(f1, f2, v)
        xout = self.decoder(fout)
        return xout


class model_deploy(model_deploy_container):
    """ the model used for testing """

    def __init__(self, n_branch, model_path, label='latest', parallel=False):
        nn.Module.__init__(self)
        self.encoder = encoder()
        self.interp_net = interp_net(n_branch)
        self.decoder = decoder(pretrained=False)
        self.encoder = nn.DataParallel(self.encoder)
        self.interp_net = nn.DataParallel(self.interp_net)
        self.decoder = nn.DataParallel(self.decoder)
        self.encoder.load_state_dict(torch.load('{}/encoder-{}.pth'.format(model_path, label)))
        self.interp_net.load_state_dict(torch.load('{}/interp_net-{}.pth'.format(model_path, label)))
        self.decoder.load_state_dict(torch.load('{}/decoder-{}.pth'.format(model_path, label)))


class base_optimizer(nn.Module):

    def __init__(self):
        super(base_optimizer, self).__init__()

    def _check_and_load(self, net, path):
        if os.path.exists(path):
            net.load_state_dict(torch.load(path))
        else:
            None

    def get_current_errors(self):
        d = OrderedDict()
        return d

    def add_summary(self, global_step):
        d = self.get_current_errors()
        for keys, values in zip(d.keys(), d.values()):
            self.writer.add_scalar(keys, values, global_step=global_step)

    def add_summary_heavy(self, epoch):
        pass

    def print_current_errors(self, epoch, i, record_file=None, print_msg=True):
        errors = self.get_current_errors()
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
        if print_msg:
            None
        if record_file is not None:
            with open(record_file + '/loss.txt', 'w') as f:
                None
        return message

    def print_network(self, net, filepath=None):
        if filepath is None:
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            None
            None
        else:
            num_params = 0
            with open(filepath + '/network.txt', 'w') as f:
                for param in net.parameters():
                    num_params += param.numel()
                None
                f.write('Total number of parameters: %d' % num_params)


def classification_loss(logit, target):
    """Compute binary or softmax cross entropy loss."""
    loss = F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
    return loss


def classification_loss_list(logit, target):
    """
    compute the classification loss when logit and target are lists, e.g., [[1,2,3],[4],[5,6]]
    """
    loss = 0
    for logit_now, target_now in zip(logit, target):
        loss += classification_loss(logit_now, target_now)
    return loss


class logger(object):

    def __init__(self, valid=True, file=None, if_print=True):
        super(logger, self).__init__()
        self.valid = valid
        self.file = file
        self.if_print = if_print

    def __call__(self, *args):
        if self.valid:
            if self.file is not None:
                with open(self.file, 'a') as f:
                    None
            args = tuple([fg(2)] + list(args) + [attr('reset')])
            if self.if_print:
                None


log = logger(True)


class Inception3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(Inception3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        inception = models.inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.upsample(x, size=(299, 299), mode='bilinear')
        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


class layer_logger(torch.nn.Module):

    def __init__(self, prefix, log_type='size'):
        super(layer_logger, self).__init__()
        self.prefix = prefix
        self.log_type = log_type

    def forward(self, x):
        if self.log_type == 'size':
            log(self.prefix, x.size())
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Inception3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {})),
    (VGG,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (_interp_branch,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (discrim,
     lambda: ([], {'attr': '2,2'}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
    (interp_net,
     lambda: ([], {'n_branch': 4}),
     lambda: ([torch.rand([4, 512, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
]

