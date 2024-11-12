
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


from torch.utils import data


from torchvision import transforms


from torch import nn


from torch.nn import functional as F


from torch.nn import init


from torch.nn.utils.spectral_norm import spectral_norm


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


class SegGenerator(BaseNetwork):

    def __init__(self, opt, input_nc, output_nc=13, norm_layer=nn.InstanceNorm2d):
        super(SegGenerator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1), norm_layer(1024), nn.ReLU(), nn.Conv2d(1024, 1024, kernel_size=3, padding=1), norm_layer(1024), nn.ReLU())
        self.up6 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(1024, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU())
        self.up7 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(512, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU())
        self.up8 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(256, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU())
        self.up9 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(128, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU(), nn.Conv2d(64, output_nc, kernel_size=3, padding=1))
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.print_network()
        self.init_weights(opt.init_type, opt.init_variance)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.drop(self.conv4(self.pool(conv3)))
        conv5 = self.drop(self.conv5(self.pool(conv4)))
        conv6 = self.conv6(torch.cat((conv4, self.up6(conv5)), 1))
        conv7 = self.conv7(torch.cat((conv3, self.up7(conv6)), 1))
        conv8 = self.conv8(torch.cat((conv2, self.up8(conv7)), 1))
        conv9 = self.conv9(torch.cat((conv1, self.up9(conv8)), 1))
        return self.sigmoid(conv9)


class FeatureExtraction(BaseNetwork):

    def __init__(self, input_nc, ngf=64, num_layers=4, norm_layer=nn.BatchNorm2d):
        super(FeatureExtraction, self).__init__()
        nf = ngf
        layers = [nn.Conv2d(input_nc, nf, kernel_size=4, stride=2, padding=1), nn.ReLU(), norm_layer(nf)]
        for i in range(1, num_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1), nn.ReLU(), norm_layer(nf)]
        layers += [nn.Conv2d(nf, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(), norm_layer(512)]
        layers += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        self.model = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        return self.model(x)


class FeatureCorrelation(nn.Module):

    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, featureA, featureB):
        b, c, h, w = featureA.size()
        featureA = featureA.permute(0, 3, 2, 1).reshape(b, w * h, c)
        featureB = featureB.reshape(b, c, h * w)
        corr = torch.bmm(featureA, featureB).reshape(b, w * h, h, w)
        return corr


class FeatureRegression(nn.Module):

    def __init__(self, input_nc=512, output_size=6, norm_layer=nn.BatchNorm2d):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1), norm_layer(512), nn.ReLU(), nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1), norm_layer(256), nn.ReLU(), nn.Conv2d(256, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU(), nn.Conv2d(128, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU())
        self.linear = nn.Linear(64 * (input_nc // 16), output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x.reshape(x.size(0), -1))
        return self.tanh(x)


class TpsGridGen(nn.Module):

    def __init__(self, opt, dtype=torch.float):
        super(TpsGridGen, self).__init__()
        grid_X, grid_Y = np.meshgrid(np.linspace(-0.9, 0.9, opt.load_width), np.linspace(-0.9, 0.9, opt.load_height))
        grid_X = torch.tensor(grid_X, dtype=dtype).unsqueeze(0).unsqueeze(3)
        grid_Y = torch.tensor(grid_Y, dtype=dtype).unsqueeze(0).unsqueeze(3)
        self.N = opt.grid_size * opt.grid_size
        coords = np.linspace(-0.9, 0.9, opt.grid_size)
        P_Y, P_X = np.meshgrid(coords, coords)
        P_X = torch.tensor(P_X, dtype=dtype).reshape(self.N, 1)
        P_Y = torch.tensor(P_Y, dtype=dtype).reshape(self.N, 1)
        P_X_base = P_X.clone()
        P_Y_base = P_Y.clone()
        Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
        P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
        P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
        self.register_buffer('grid_X', grid_X, False)
        self.register_buffer('grid_Y', grid_Y, False)
        self.register_buffer('P_X_base', P_X_base, False)
        self.register_buffer('P_Y_base', P_Y_base, False)
        self.register_buffer('Li', Li, False)
        self.register_buffer('P_X', P_X, False)
        self.register_buffer('P_Y', P_Y, False)

    def compute_L_inverse(self, X, Y):
        N = X.size()[0]
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = torch.pow(Xmat - Xmat.transpose(0, 1), 2) + torch.pow(Ymat - Ymat.transpose(0, 1), 2)
        P_dist_squared[P_dist_squared == 0] = 1
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        O = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        batch_size = theta.size()[0]
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_Y)
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_Y)
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 1].size() + (1, self.N))
        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            delta_X = points_X_for_summation - P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(points_Y_for_summation)
        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        dist_squared[dist_squared == 0] = 1
        U = torch.mul(dist_squared, torch.log(dist_squared))
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])
        points_X_prime = A_X[:, :, :, :, 0] + torch.mul(A_X[:, :, :, :, 1], points_X_batch) + torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)
        points_Y_prime = A_Y[:, :, :, :, 0] + torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)
        return torch.cat((points_X_prime, points_Y_prime), 3)

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))
        return warped_grid


class GMM(nn.Module):

    def __init__(self, opt, inputA_nc, inputB_nc):
        super(GMM, self).__init__()
        self.extractionA = FeatureExtraction(inputA_nc, ngf=64, num_layers=4)
        self.extractionB = FeatureExtraction(inputB_nc, ngf=64, num_layers=4)
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=opt.load_width // 64 * (opt.load_height // 64), output_size=2 * opt.grid_size ** 2)
        self.gridGen = TpsGridGen(opt)

    def forward(self, inputA, inputB):
        featureA = F.normalize(self.extractionA(inputA), dim=1)
        featureB = F.normalize(self.extractionB(inputB), dim=1)
        corr = self.correlation(featureA, featureB)
        theta = self.regression(corr)
        warped_grid = self.gridGen(theta)
        return theta, warped_grid


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


class ALIASNorm(nn.Module):

    def __init__(self, norm_type, norm_nc, label_nc):
        super(ALIASNorm, self).__init__()
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
            raise ValueError("'{}' is not a recognized parameter-free normalization type in ALIASNorm".format(param_free_norm_type))
        nhidden = 128
        ks = 3
        pw = ks // 2
        self.conv_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.conv_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, seg, misalign_mask=None):
        b, c, h, w = x.size()
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


class ALIASResBlock(nn.Module):

    def __init__(self, opt, input_nc, output_nc, use_mask_norm=True):
        super(ALIASResBlock, self).__init__()
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
        semantic_nc = opt.semantic_nc
        if use_mask_norm:
            subnorm_type = 'aliasmask'
            semantic_nc = semantic_nc + 1
        self.norm_0 = ALIASNorm(subnorm_type, input_nc, semantic_nc)
        self.norm_1 = ALIASNorm(subnorm_type, middle_nc, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = ALIASNorm(subnorm_type, input_nc, semantic_nc)
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


class ALIASGenerator(BaseNetwork):

    def __init__(self, opt, input_nc):
        super(ALIASGenerator, self).__init__()
        self.num_upsampling_layers = opt.num_upsampling_layers
        self.sh, self.sw = self.compute_latent_vector_size(opt)
        nf = opt.ngf
        self.conv_0 = nn.Conv2d(input_nc, nf * 16, kernel_size=3, padding=1)
        for i in range(1, 8):
            self.add_module('conv_{}'.format(i), nn.Conv2d(input_nc, 16, kernel_size=3, padding=1))
        self.head_0 = ALIASResBlock(opt, nf * 16, nf * 16)
        self.G_middle_0 = ALIASResBlock(opt, nf * 16 + 16, nf * 16)
        self.G_middle_1 = ALIASResBlock(opt, nf * 16 + 16, nf * 16)
        self.up_0 = ALIASResBlock(opt, nf * 16 + 16, nf * 8)
        self.up_1 = ALIASResBlock(opt, nf * 8 + 16, nf * 4)
        self.up_2 = ALIASResBlock(opt, nf * 4 + 16, nf * 2, use_mask_norm=False)
        self.up_3 = ALIASResBlock(opt, nf * 2 + 16, nf * 1, use_mask_norm=False)
        if self.num_upsampling_layers == 'most':
            self.up_4 = ALIASResBlock(opt, nf * 1 + 16, nf // 2, use_mask_norm=False)
            nf = nf // 2
        self.conv_img = nn.Conv2d(nf, 3, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        self.print_network()
        self.init_weights(opt.init_type, opt.init_variance)

    def compute_latent_vector_size(self, opt):
        if self.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif self.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif self.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError("opt.num_upsampling_layers '{}' is not recognized".format(self.num_upsampling_layers))
        sh = opt.load_height // 2 ** num_up_layers
        sw = opt.load_width // 2 ** num_up_layers
        return sh, sw

    def forward(self, x, seg, seg_div, misalign_mask):
        samples = [F.interpolate(x, size=(self.sh * 2 ** i, self.sw * 2 ** i), mode='nearest') for i in range(8)]
        features = [self._modules['conv_{}'.format(i)](samples[i]) for i in range(8)]
        x = self.head_0(features[0], seg_div, misalign_mask)
        x = self.up(x)
        x = self.G_middle_0(torch.cat((x, features[1]), 1), seg_div, misalign_mask)
        if self.num_upsampling_layers in ['more', 'most']:
            x = self.up(x)
        x = self.G_middle_1(torch.cat((x, features[2]), 1), seg_div, misalign_mask)
        x = self.up(x)
        x = self.up_0(torch.cat((x, features[3]), 1), seg_div, misalign_mask)
        x = self.up(x)
        x = self.up_1(torch.cat((x, features[4]), 1), seg_div, misalign_mask)
        x = self.up(x)
        x = self.up_2(torch.cat((x, features[5]), 1), seg)
        x = self.up(x)
        x = self.up_3(torch.cat((x, features[6]), 1), seg)
        if self.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(torch.cat((x, features[7]), 1), seg)
        x = self.conv_img(self.relu(x))
        return self.tanh(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BaseNetwork,
     lambda: ([], {}),
     lambda: ([], {})),
    (FeatureCorrelation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FeatureExtraction,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (MaskNorm,
     lambda: ([], {'norm_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

