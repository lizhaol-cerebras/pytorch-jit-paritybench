
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


import torch.utils.data as data


import torchvision.transforms as transforms


import numpy as np


from torch.autograd import Variable


from torch import nn


from collections import OrderedDict


import itertools


from scipy.ndimage import zoom


import functools


import torch.nn as nn


import torch.nn.init as init


from collections import namedtuple


from torchvision import models as tv


from scipy.stats import entropy


import torch.nn.functional as F


import torchvision.transforms as Transforms


from torchvision.models.inception import inception_v3


import time


from torch.nn import init


from torch.nn.utils import spectral_norm


from torchvision import models


import collections


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


from torch.nn.parallel.data_parallel import DataParallel


from torchvision.utils import make_grid


from torchvision.utils import save_image


from torchvision.utils import make_grid as make_image_grid


from torch.utils.data import Subset


from torch.nn import functional as F


from torchvision.transforms import transforms


from torchvision import transforms


class PerceptualLoss(torch.nn.Module):

    def __init__(self, model='net-lin', net='alex', colorspace='rgb', spatial=False, use_gpu=True, gpu_ids=[0]):
        super(PerceptualLoss, self).__init__()
        None
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_gpu, colorspace=colorspace, spatial=self.spatial, gpu_ids=gpu_ids)
        None
        None

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]

        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1
        return self.model.forward(target, pred)


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_H=64):
    in_H = in_tens.shape[2]
    scale_factor = 1.0 * out_H / in_H
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)(in_tens)


class PNetLin(nn.Module):

    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, spatial=False, version='0.1', lpips=True):
        super(PNetLin, self).__init__()
        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.version = version
        self.scaling_layer = ScalingLayer()
        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = pn.vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = pn.alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == 'squeeze':
            net_type = pn.squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)
        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == 'squeeze':
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]

    def forward(self, in0, in1, retPerLayer=False):
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = util.normalize_tensor(outs0[kk]), util.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk].model(diffs[kk]), out_H=in0.shape[2]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk].model(diffs[kk]), keepdim=True) for kk in range(self.L)]
        elif self.spatial:
            res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_H=in0.shape[2]) for kk in range(self.L)]
        else:
            res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]
        val = res[0]
        for l in range(1, self.L):
            val += res[l]
        if retPerLayer:
            return val, res
        else:
            return val


class Dist2LogitLayer(nn.Module):
    """ takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) """

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()
        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True)]
        layers += [nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True)]
        layers += [nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True)]
        if use_sigmoid:
            layers += [nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):

    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge + 1.0) / 2.0
        self.logit = self.net.forward(d0, d1)
        return self.loss(self.logit, per)


class FakeNet(nn.Module):

    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace


class L2(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            N, C, X, Y = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y), dim=3).view(N)
            return value
        elif self.colorspace == 'Lab':
            value = util.l2(util.tensor2np(util.tensor2tensorlab(in0.data, to_norm=False)), util.tensor2np(util.tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
            ret_var = Variable(torch.Tensor((value,)))
            if self.use_gpu:
                ret_var = ret_var
            return ret_var


class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            value = util.dssim(1.0 * util.tensor2im(in0.data), 1.0 * util.tensor2im(in1.data), range=255.0).astype('float')
        elif self.colorspace == 'Lab':
            value = util.dssim(util.tensor2np(util.tensor2tensorlab(in0.data, to_norm=False)), util.tensor2np(util.tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
        ret_var = Variable(torch.Tensor((value,)))
        if self.use_gpu:
            ret_var = ret_var
        return ret_var


class squeezenet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple('SqueezeOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7'])
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)
        return out


class alexnet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple('AlexnetOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)
        return out


class vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
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
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class resnet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if num == 18:
            self.net = tv.resnet18(pretrained=pretrained)
        elif num == 34:
            self.net = tv.resnet34(pretrained=pretrained)
        elif num == 50:
            self.net = tv.resnet50(pretrained=pretrained)
        elif num == 101:
            self.net = tv.resnet101(pretrained=pretrained)
        elif num == 152:
            self.net = tv.resnet152(pretrained=pretrained)
        self.N_slices = 5
        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h
        outputs = namedtuple('Outputs', ['relu1', 'conv2', 'conv3', 'conv4', 'conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)
        return out


class BaseNetwork(nn.Module):

    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        None

    def init_weights(self, init_type='normal', gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__
            if 'BatchNorm2d' in classname:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif ('Conv' in classname or 'Linear' in classname) and hasattr(m, 'weight'):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':
                    m.reset_parameters()
                else:
                    raise NotImplementedError("initialization method '{}' is not implemented".format(init_type))
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
        self.apply(init_func)

    def forward(self, *inputs):
        pass


class MaskNorm(nn.Module):

    def __init__(self, norm_nc):
        super(MaskNorm, self).__init__()
        self.norm_layer = nn.InstanceNorm2d(norm_nc, affine=False)

    def normalize_region(self, region, mask):
        b, c, h, w = region.size()
        num_pixels = mask.sum((2, 3), keepdim=True)
        num_pixels[num_pixels == 0] = 1
        mu = region.sum((2, 3), keepdim=True) / num_pixels
        normalized_region = self.norm_layer(region + (1 - mask) * mu)
        return normalized_region * torch.sqrt(num_pixels / (h * w))

    def forward(self, x, mask):
        mask = mask.detach()
        normalized_foreground = self.normalize_region(x * mask, mask)
        normalized_background = self.normalize_region(x * (1 - mask), 1 - mask)
        return normalized_foreground + normalized_background


class SPADENorm(nn.Module):

    def __init__(self, opt, norm_type, norm_nc, label_nc):
        super(SPADENorm, self).__init__()
        self.param_opt = opt
        self.noise_scale = nn.Parameter(torch.zeros(norm_nc))
        assert norm_type.startswith('alias')
        param_free_norm_type = norm_type[len('alias'):]
        if param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'mask':
            self.param_free_norm = MaskNorm(norm_nc)
        else:
            raise ValueError("'{}' is not a recognized parameter-free normalization type in SPADENorm".format(param_free_norm_type))
        nhidden = 128
        ks = 3
        pw = ks // 2
        self.conv_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.conv_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, seg, misalign_mask=None):
        b, c, h, w = x.size()
        if self.param_opt.cuda:
            noise = (torch.randn(b, w, h, 1) * self.noise_scale).transpose(1, 3)
        else:
            noise = (torch.randn(b, w, h, 1) * self.noise_scale).transpose(1, 3)
        if misalign_mask is None:
            normalized = self.param_free_norm(x + noise)
        else:
            normalized = self.param_free_norm(x + noise, misalign_mask)
        actv = self.conv_shared(seg)
        gamma = self.conv_gamma(actv)
        beta = self.conv_beta(actv)
        output = normalized * (1 + gamma) + beta
        return output


class SPADEResBlock(nn.Module):

    def __init__(self, opt, input_nc, output_nc, use_mask_norm=True):
        super(SPADEResBlock, self).__init__()
        self.param_opt = opt
        self.learned_shortcut = input_nc != output_nc
        middle_nc = min(input_nc, output_nc)
        self.conv_0 = nn.Conv2d(input_nc, middle_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_nc, output_nc, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(input_nc, output_nc, kernel_size=1, bias=False)
        subnorm_type = opt.norm_G
        if subnorm_type.startswith('spectral'):
            subnorm_type = subnorm_type[len('spectral'):]
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        gen_semantic_nc = opt.gen_semantic_nc
        if use_mask_norm:
            subnorm_type = 'aliasmask'
            gen_semantic_nc = gen_semantic_nc + 1
        self.norm_0 = SPADENorm(opt, subnorm_type, input_nc, gen_semantic_nc)
        self.norm_1 = SPADENorm(opt, subnorm_type, middle_nc, gen_semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADENorm(opt, subnorm_type, input_nc, gen_semantic_nc)
        self.relu = nn.LeakyReLU(0.2)

    def shortcut(self, x, seg, misalign_mask):
        if self.learned_shortcut:
            return self.conv_s(self.norm_s(x, seg, misalign_mask))
        else:
            return x

    def forward(self, x, seg, misalign_mask=None):
        seg = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        if misalign_mask is not None:
            misalign_mask = F.interpolate(misalign_mask, size=x.size()[2:], mode='nearest')
        x_s = self.shortcut(x, seg, misalign_mask)
        dx = self.conv_0(self.relu(self.norm_0(x, seg, misalign_mask)))
        dx = self.conv_1(self.relu(self.norm_1(dx, seg, misalign_mask)))
        output = x_s + dx
        return output


class SPADEGenerator(BaseNetwork):

    def __init__(self, opt, input_nc):
        super(SPADEGenerator, self).__init__()
        self.num_upsampling_layers = opt.num_upsampling_layers
        self.param_opt = opt
        self.sh, self.sw = self.compute_latent_vector_size(opt)
        nf = opt.ngf
        self.conv_0 = nn.Conv2d(input_nc, nf * 16, kernel_size=3, padding=1)
        for i in range(1, 8):
            self.add_module('conv_{}'.format(i), nn.Conv2d(input_nc, 16, kernel_size=3, padding=1))
        self.head_0 = SPADEResBlock(opt, nf * 16, nf * 16, use_mask_norm=False)
        self.G_middle_0 = SPADEResBlock(opt, nf * 16 + 16, nf * 16, use_mask_norm=False)
        self.G_middle_1 = SPADEResBlock(opt, nf * 16 + 16, nf * 16, use_mask_norm=False)
        self.up_0 = SPADEResBlock(opt, nf * 16 + 16, nf * 8, use_mask_norm=False)
        self.up_1 = SPADEResBlock(opt, nf * 8 + 16, nf * 4, use_mask_norm=False)
        self.up_2 = SPADEResBlock(opt, nf * 4 + 16, nf * 2, use_mask_norm=False)
        self.up_3 = SPADEResBlock(opt, nf * 2 + 16, nf * 1, use_mask_norm=False)
        if self.num_upsampling_layers == 'most':
            self.up_4 = SPADEResBlock(opt, nf * 1 + 16, nf // 2, use_mask_norm=False)
            nf = nf // 2
        self.conv_img = nn.Conv2d(nf, 3, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def compute_latent_vector_size(self, opt):
        if self.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif self.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif self.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError("opt.num_upsampling_layers '{}' is not recognized".format(self.num_upsampling_layers))
        sh = opt.fine_height // 2 ** num_up_layers
        sw = opt.fine_width // 2 ** num_up_layers
        return sh, sw

    def forward(self, x, seg):
        samples = [F.interpolate(x, size=(self.sh * 2 ** i, self.sw * 2 ** i), mode='nearest') for i in range(8)]
        features = [self._modules['conv_{}'.format(i)](samples[i]) for i in range(8)]
        x = self.head_0(features[0], seg)
        x = self.up(x)
        x = self.G_middle_0(torch.cat((x, features[1]), 1), seg)
        if self.num_upsampling_layers in ['more', 'most']:
            x = self.up(x)
        x = self.G_middle_1(torch.cat((x, features[2]), 1), seg)
        x = self.up(x)
        x = self.up_0(torch.cat((x, features[3]), 1), seg)
        x = self.up(x)
        x = self.up_1(torch.cat((x, features[4]), 1), seg)
        x = self.up(x)
        x = self.up_2(torch.cat((x, features[5]), 1), seg)
        x = self.up(x)
        x = self.up_3(torch.cat((x, features[6]), 1), seg)
        if self.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(torch.cat((x, features[7]), 1), seg)
        x = self.conv_img(self.relu(x))
        return self.tanh(x)


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False, Ddropout=False, spectral=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.spectral_norm = spectral_norm if spectral else lambda x: x
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            if Ddropout:
                sequence += [[self.spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)), norm_layer(nf), nn.LeakyReLU(0.2, True), nn.Dropout(0.5)]]
            else:
                sequence += [[self.spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]
        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3, getIntermFeat=False, Ddownx2=False, Ddropout=False, spectral=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        self.Ddownx2 = Ddownx2
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, Ddropout, spectral=spectral)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        if self.Ddownx2:
            input_downsampled = self.downsample(input)
        else:
            input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        return result


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = self.real_label_var is None or self.real_label_var.numel() != input.numel()
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = self.fake_label_var is None or self.fake_label_var.numel() != input.numel()
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class ResBlock(nn.Module):

    def __init__(self, in_nc, out_nc, scale='down', norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        assert scale in ['up', 'down', 'same'], "ResBlock scale must be in 'up' 'down' 'same'"
        if scale == 'same':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=True)
        if scale == 'up':
            self.scale = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=True))
        if scale == 'down':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.block = nn.Sequential(nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(out_nc), nn.ReLU(inplace=True), nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(out_nc))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.scale(x)
        return self.relu(residual + self.block(residual))


class ConditionGenerator(nn.Module):

    def __init__(self, opt, input1_nc, input2_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(ConditionGenerator, self).__init__()
        self.warp_feature = opt.warp_feature
        self.out_layer_opt = opt.out_layer
        self.ClothEncoder = nn.Sequential(ResBlock(input1_nc, ngf, norm_layer=norm_layer, scale='down'), ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'), ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'), ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'), ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'))
        self.PoseEncoder = nn.Sequential(ResBlock(input2_nc, ngf, norm_layer=norm_layer, scale='down'), ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'), ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'), ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'), ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'))
        self.conv = ResBlock(ngf * 4, ngf * 8, norm_layer=norm_layer, scale='same')
        if opt.warp_feature == 'T1':
            self.SegDecoder = nn.Sequential(ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'), ResBlock(ngf * 4 * 2 + ngf * 4, ngf * 4, norm_layer=norm_layer, scale='up'), ResBlock(ngf * 4 * 2 + ngf * 4, ngf * 2, norm_layer=norm_layer, scale='up'), ResBlock(ngf * 2 * 2 + ngf * 4, ngf, norm_layer=norm_layer, scale='up'), ResBlock(ngf * 1 * 2 + ngf * 4, ngf, norm_layer=norm_layer, scale='up'))
        if opt.warp_feature == 'encoder':
            self.SegDecoder = nn.Sequential(ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'), ResBlock(ngf * 4 * 3, ngf * 4, norm_layer=norm_layer, scale='up'), ResBlock(ngf * 4 * 3, ngf * 2, norm_layer=norm_layer, scale='up'), ResBlock(ngf * 2 * 3, ngf, norm_layer=norm_layer, scale='up'), ResBlock(ngf * 1 * 3, ngf, norm_layer=norm_layer, scale='up'))
        if opt.out_layer == 'relu':
            self.out_layer = ResBlock(ngf + input1_nc + input2_nc, output_nc, norm_layer=norm_layer, scale='same')
        if opt.out_layer == 'conv':
            self.out_layer = nn.Sequential(ResBlock(ngf + input1_nc + input2_nc, ngf, norm_layer=norm_layer, scale='same'), nn.Conv2d(ngf, output_nc, kernel_size=1, bias=True))
        self.conv1 = nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True), nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True), nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True), nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True))
        self.conv2 = nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True), nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True), nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True), nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True))
        self.flow_conv = nn.ModuleList([nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True), nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True), nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True), nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True), nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True)])
        self.bottleneck = nn.Sequential(nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()), nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()), nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()), nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()))

    def normalize(self, x):
        return x

    def forward(self, opt, input1, input2, upsample='bilinear'):
        E1_list = []
        E2_list = []
        flow_list = []
        for i in range(5):
            if i == 0:
                E1_list.append(self.ClothEncoder[i](input1))
                E2_list.append(self.PoseEncoder[i](input2))
            else:
                E1_list.append(self.ClothEncoder[i](E1_list[i - 1]))
                E2_list.append(self.PoseEncoder[i](E2_list[i - 1]))
        for i in range(5):
            N, _, iH, iW = E1_list[4 - i].size()
            grid = make_grid(N, iH, iW, opt)
            if i == 0:
                T1 = E1_list[4 - i]
                T2 = E2_list[4 - i]
                E4 = torch.cat([T1, T2], 1)
                flow = self.flow_conv[i](self.normalize(E4)).permute(0, 2, 3, 1)
                flow_list.append(flow)
                x = self.conv(T2)
                x = self.SegDecoder[i](x)
            else:
                T1 = F.interpolate(T1, scale_factor=2, mode=upsample) + self.conv1[4 - i](E1_list[4 - i])
                T2 = F.interpolate(T2, scale_factor=2, mode=upsample) + self.conv2[4 - i](E2_list[4 - i])
                flow = F.interpolate(flow_list[i - 1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((iW / 2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH / 2 - 1.0) / 2.0)], 3)
                warped_T1 = F.grid_sample(T1, flow_norm + grid, padding_mode='border')
                flow = flow + self.flow_conv[i](self.normalize(torch.cat([warped_T1, self.bottleneck[i - 1](x)], 1))).permute(0, 2, 3, 1)
                flow_list.append(flow)
                if self.warp_feature == 'T1':
                    x = self.SegDecoder[i](torch.cat([x, E2_list[4 - i], warped_T1], 1))
                if self.warp_feature == 'encoder':
                    warped_E1 = F.grid_sample(E1_list[4 - i], flow_norm + grid, padding_mode='border')
                    x = self.SegDecoder[i](torch.cat([x, E2_list[4 - i], warped_E1], 1))
        N, _, iH, iW = input1.size()
        grid = make_grid(N, iH, iW, opt)
        flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)
        flow_norm = torch.cat([flow[:, :, :, 0:1] / ((iW / 2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH / 2 - 1.0) / 2.0)], 3)
        warped_input1 = F.grid_sample(input1, flow_norm + grid, padding_mode='border')
        x = self.out_layer(torch.cat([x, input2, warped_input1], 1))
        warped_c = warped_input1[:, :-1, :, :]
        warped_cm = warped_input1[:, -1:, :, :]
        return flow_list, x, warped_c, warped_cm


class Vgg19(nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):

    def __init__(self, opt, layids=None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        if opt.cuda:
            self.vgg
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, "Previous result has't been fetched."
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()
            res = self._result
            self._result = None
            return res


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True
        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())
        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'
        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)
        for i in range(self.nr_slaves):
            assert self._queue.get() is True
        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class _SynchronizedBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))
        if self.affine:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 + 2])))
        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    """Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    """Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    """Applies Batch Normalization over a 5d input that is seen as a mini-batch
    of 4d inputs

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm3d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
    or Spatio-temporal BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)


class BatchNorm2dReimpl(nn.Module):
    """
    A re-implementation of batch normalization, used for testing the numerical
    stability.

    Author: acgtyrant
    See also:
    https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input_):
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
        unbias_var = sumvar / (numel - 1)
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.detach()
        bias_var = sumvar / numel
        inv_std = 1 / (bias_var + self.eps).pow(0.5)
        output = (input_ - mean.unsqueeze(1)) * inv_std.unsqueeze(1) * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)
        return output.view(channels, batchsize, height, width).permute(1, 0, 2, 3).contiguous()


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BaseNetwork,
     lambda: ([], {}),
     lambda: ([], {})),
    (BatchNorm2dReimpl,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DataParallelWithCallback,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (Dist2LogitLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 1, 4, 4])], {})),
    (MaskNorm,
     lambda: ([], {'norm_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MultiscaleDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NLayerDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlock,
     lambda: ([], {'in_nc': 4, 'out_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScalingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {})),
    (SynchronizedBatchNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SynchronizedBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SynchronizedBatchNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VGGLoss,
     lambda: ([], {'opt': SimpleNamespace(cuda=False)}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {})),
    (Vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (_SynchronizedBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (alexnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (resnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (squeezenet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

