import sys
_module = sys.modules[__name__]
del sys
back2future = _module
correlation_package = _module
build = _module
functions = _module
correlation = _module
modules = _module
correlation = _module
demo = _module
flow_io = _module
test_back2future = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from torch.autograd import Function


from torch.nn.modules.module import Module


from torchvision.transforms import ToTensor


class CorrelationFunction(Function):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        super(CorrelationFunction, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        self.save_for_backward(input1, input2)
        assert input1.is_contiguous() == True
        assert input2.is_contiguous() == True
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()
            correlation.Correlation_forward_cuda(input1, input2, rbot1, rbot2, output, self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)
        return output

    def backward(self, grad_output):
        input1, input2 = self.saved_tensors
        assert grad_output.is_contiguous() == True
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            grad_input1 = input1.new()
            grad_input2 = input2.new()
            correlation.Correlation_backward_cuda(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2, self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)
        return grad_input1, grad_input2


class Correlation(Module):

    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        result = CorrelationFunction(self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)(input1, input2)
        return result


def conv_dec_block(nIn):
    return nn.Sequential(nn.Conv2d(nIn, 128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1))


def conv_feat_block(nIn, nOut):
    return nn.Sequential(nn.Conv2d(nIn, nOut, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(nOut, nOut, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2))


class Model(nn.Module):

    def __init__(self, pretrained=None):
        super(Model, self).__init__()
        idx = [list(range(n, -1, -9)) for n in range(80, 71, -1)]
        idx = list(np.array(idx).flatten())
        self.idx_fwd = Variable(torch.LongTensor(np.array(idx)), requires_grad=False)
        self.idx_bwd = Variable(torch.LongTensor(np.array(list(reversed(idx)))), requires_grad=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.softmax2d = nn.Softmax2d()
        self.conv1a = conv_feat_block(3, 16)
        self.conv1b = conv_feat_block(3, 16)
        self.conv1c = conv_feat_block(3, 16)
        self.conv2a = conv_feat_block(16, 32)
        self.conv2b = conv_feat_block(16, 32)
        self.conv2c = conv_feat_block(16, 32)
        self.conv3a = conv_feat_block(32, 64)
        self.conv3b = conv_feat_block(32, 64)
        self.conv3c = conv_feat_block(32, 64)
        self.conv4a = conv_feat_block(64, 96)
        self.conv4b = conv_feat_block(64, 96)
        self.conv4c = conv_feat_block(64, 96)
        self.conv5a = conv_feat_block(96, 128)
        self.conv5b = conv_feat_block(96, 128)
        self.conv5c = conv_feat_block(96, 128)
        self.conv6a = conv_feat_block(128, 192)
        self.conv6b = conv_feat_block(128, 192)
        self.conv6c = conv_feat_block(128, 192)
        self.corr = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)
        self.decoder_fwd6 = conv_dec_block(162)
        self.decoder_bwd6 = conv_dec_block(162)
        self.decoder_fwd5 = conv_dec_block(292)
        self.decoder_bwd5 = conv_dec_block(292)
        self.decoder_fwd4 = conv_dec_block(260)
        self.decoder_bwd4 = conv_dec_block(260)
        self.decoder_fwd3 = conv_dec_block(228)
        self.decoder_bwd3 = conv_dec_block(228)
        self.decoder_fwd2 = conv_dec_block(196)
        self.decoder_bwd2 = conv_dec_block(196)
        self.decoder_occ6 = conv_dec_block(354)
        self.decoder_occ5 = conv_dec_block(292)
        self.decoder_occ4 = conv_dec_block(260)
        self.decoder_occ3 = conv_dec_block(228)
        self.decoder_occ2 = conv_dec_block(196)
        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained))
            None

    def load(self, key, load_id):
        module = getattr(self, key)
        loadpath = os.path.join(LOAD_DIR, load_id)
        for i, m in enumerate(module):
            if type(m) == nn.Conv2d:
                weight_path = loadpath + '_' + str(i + 1) + 'weight.t7'
                bias_path = loadpath + '_' + str(i + 1) + 'bias.t7'
                m.weight.data.copy_(load_lua(weight_path))
                m.bias.data.copy_(load_lua(bias_path))

    def initialize(self):
        map = {'conv1a': '3', 'conv1b': '6', 'conv1c': '22', 'conv2a': '4', 'conv2b': '7', 'conv2c': '23', 'conv3a': '9', 'conv3b': '10', 'conv3c': '24', 'conv4a': '12', 'conv4b': '13', 'conv4c': '25', 'conv5a': '15', 'conv5b': '16', 'conv5c': '26', 'conv6a': '18', 'conv6b': '19', 'conv6c': '27', 'decoder_fwd6': '30', 'decoder_bwd6': '93', 'decoder_occ6': '183', 'decoder_fwd5': '45', 'decoder_bwd5': '96', 'decoder_occ5': '164', 'decoder_fwd4': '60', 'decoder_bwd4': '99', 'decoder_occ4': '145', 'decoder_fwd3': '75', 'decoder_bwd3': '102', 'decoder_occ3': '126', 'decoder_fwd2': '90', 'decoder_bwd2': '105', 'decoder_occ2': '109'}
        for key in map.keys():
            None
            self.load(key, map[key])

    def normalize(self, ims):
        imt = []
        for im in ims:
            im[:, 0, :, :] = im[:, 0, :, :] - 0.485
            im[:, 1, :, :] = im[:, 1, :, :] - 0.456
            im[:, 2, :, :] = im[:, 2, :, :] - 0.406
            im[:, 0, :, :] = im[:, 0, :, :] / 0.229
            im[:, 1, :, :] = im[:, 1, :, :] / 0.224
            im[:, 2, :, :] = im[:, 2, :, :] / 0.225
            imt.append(im)
        return imt

    def forward(self, im_tar, im_refs):
        """
            Arguments:
                im_tar : Centre Frame
                im_refs : List constaining [Past_Frame, Future_Frame]
        """
        im_norm = self.normalize([im_tar] + im_refs)
        feat1a = self.conv1a(im_norm[0])
        feat2a = self.conv2a(feat1a)
        feat3a = self.conv3a(feat2a)
        feat4a = self.conv4a(feat3a)
        feat5a = self.conv5a(feat4a)
        feat6a = self.conv6a(feat5a)
        feat1b = self.conv1b(im_norm[2])
        feat2b = self.conv2b(feat1b)
        feat3b = self.conv3b(feat2b)
        feat4b = self.conv4b(feat3b)
        feat5b = self.conv5b(feat4b)
        feat6b = self.conv6b(feat5b)
        feat1c = self.conv1c(im_norm[1])
        feat2c = self.conv2c(feat1c)
        feat3c = self.conv3c(feat2c)
        feat4c = self.conv4c(feat3c)
        feat5c = self.conv5c(feat4c)
        feat6c = self.conv6c(feat5c)
        corr6_fwd = self.corr(feat6a, feat6b)
        corr6_fwd = corr6_fwd.index_select(1, self.idx_fwd)
        corr6_bwd = self.corr(feat6a, feat6c)
        corr6_bwd = corr6_bwd.index_select(1, self.idx_bwd)
        corr6 = torch.cat((corr6_fwd, corr6_bwd), 1)
        flow6_fwd = self.decoder_fwd6(corr6)
        flow6_fwd_up = self.upsample(flow6_fwd)
        flow6_bwd = self.decoder_bwd6(corr6)
        flow6_bwd_up = self.upsample(flow6_bwd)
        feat5b_warped = self.warp(feat5b, 0.625 * flow6_fwd_up)
        feat5c_warped = self.warp(feat5c, -0.625 * flow6_bwd_up)
        occ6_feat = torch.cat((corr6, feat6a), 1)
        occ6 = self.softmax2d(self.decoder_occ6(occ6_feat))
        corr5_fwd = self.corr(feat5a, feat5b_warped)
        corr5_fwd = corr5_fwd.index_select(1, self.idx_fwd)
        corr5_bwd = self.corr(feat5a, feat5c_warped)
        corr5_bwd = corr5_bwd.index_select(1, self.idx_bwd)
        corr5 = torch.cat((corr5_fwd, corr5_bwd), 1)
        upfeat5_fwd = torch.cat((corr5, feat5a, flow6_fwd_up), 1)
        flow5_fwd = self.decoder_fwd5(upfeat5_fwd)
        flow5_fwd_up = self.upsample(flow5_fwd)
        upfeat5_bwd = torch.cat((corr5, feat5a, flow6_bwd_up), 1)
        flow5_bwd = self.decoder_bwd5(upfeat5_bwd)
        flow5_bwd_up = self.upsample(flow5_bwd)
        feat4b_warped = self.warp(feat4b, 1.25 * flow5_fwd_up)
        feat4c_warped = self.warp(feat4c, -1.25 * flow5_bwd_up)
        occ5 = self.softmax2d(self.decoder_occ5(upfeat5_fwd))
        corr4_fwd = self.corr(feat4a, feat4b_warped)
        corr4_fwd = corr4_fwd.index_select(1, self.idx_fwd)
        corr4_bwd = self.corr(feat4a, feat4c_warped)
        corr4_bwd = corr4_bwd.index_select(1, self.idx_bwd)
        corr4 = torch.cat((corr4_fwd, corr4_bwd), 1)
        upfeat4_fwd = torch.cat((corr4, feat4a, flow5_fwd_up), 1)
        flow4_fwd = self.decoder_fwd4(upfeat4_fwd)
        flow4_fwd_up = self.upsample(flow4_fwd)
        upfeat4_bwd = torch.cat((corr4, feat4a, flow5_bwd_up), 1)
        flow4_bwd = self.decoder_bwd4(upfeat4_bwd)
        flow4_bwd_up = self.upsample(flow4_bwd)
        feat3b_warped = self.warp(feat3b, 2.5 * flow4_fwd_up)
        feat3c_warped = self.warp(feat3c, -2.5 * flow4_bwd_up)
        occ4 = self.softmax2d(self.decoder_occ4(upfeat4_fwd))
        corr3_fwd = self.corr(feat3a, feat3b_warped)
        corr3_fwd = corr3_fwd.index_select(1, self.idx_fwd)
        corr3_bwd = self.corr(feat3a, feat3c_warped)
        corr3_bwd = corr3_bwd.index_select(1, self.idx_bwd)
        corr3 = torch.cat((corr3_fwd, corr3_bwd), 1)
        upfeat3_fwd = torch.cat((corr3, feat3a, flow4_fwd_up), 1)
        flow3_fwd = self.decoder_fwd3(upfeat3_fwd)
        flow3_fwd_up = self.upsample(flow3_fwd)
        upfeat3_bwd = torch.cat((corr3, feat3a, flow4_bwd_up), 1)
        flow3_bwd = self.decoder_bwd3(upfeat3_bwd)
        flow3_bwd_up = self.upsample(flow3_bwd)
        feat2b_warped = self.warp(feat2b, 5.0 * flow3_fwd_up)
        feat2c_warped = self.warp(feat2c, -5.0 * flow3_bwd_up)
        occ3 = self.softmax2d(self.decoder_occ3(upfeat3_fwd))
        corr2_fwd = self.corr(feat2a, feat2b_warped)
        corr2_fwd = corr2_fwd.index_select(1, self.idx_fwd)
        corr2_bwd = self.corr(feat2a, feat2c_warped)
        corr2_bwd = corr2_bwd.index_select(1, self.idx_bwd)
        corr2 = torch.cat((corr2_fwd, corr2_bwd), 1)
        upfeat2_fwd = torch.cat((corr2, feat2a, flow3_fwd_up), 1)
        flow2_fwd = self.decoder_fwd2(upfeat2_fwd)
        flow2_fwd_up = self.upsample(flow2_fwd)
        upfeat2_bwd = torch.cat((corr2, feat2a, flow3_bwd_up), 1)
        flow2_bwd = self.decoder_bwd2(upfeat2_bwd)
        flow2_bwd_up = self.upsample(flow2_bwd)
        occ2 = self.softmax2d(self.decoder_occ2(upfeat2_fwd))
        flow2_fwd_fullres = 20 * self.upsample(flow2_fwd_up)
        flow3_fwd_fullres = 10 * self.upsample(flow3_fwd_up)
        flow4_fwd_fullres = 5 * self.upsample(flow4_fwd_up)
        flow5_fwd_fullres = 2.5 * self.upsample(flow5_fwd_up)
        flow6_fwd_fullres = 1.25 * self.upsample(flow6_fwd_up)
        flow2_bwd_fullres = -20 * self.upsample(flow2_bwd_up)
        flow3_bwd_fullres = -10 * self.upsample(flow3_bwd_up)
        flow4_bwd_fullres = -5 * self.upsample(flow4_bwd_up)
        flow5_bwd_fullres = -2.5 * self.upsample(flow5_bwd_up)
        flow6_bwd_fullres = -1.25 * self.upsample(flow6_bwd_up)
        occ2_fullres = F.upsample(occ2, scale_factor=4)
        occ3_fullres = F.upsample(occ3, scale_factor=4)
        occ4_fullres = F.upsample(occ4, scale_factor=4)
        occ5_fullres = F.upsample(occ5, scale_factor=4)
        occ6_fullres = F.upsample(occ6, scale_factor=4)
        flow_fwd = [flow2_fwd_fullres, flow3_fwd_fullres, flow4_fwd_fullres, flow5_fwd_fullres, flow6_fwd_fullres]
        flow_bwd = [flow2_bwd_fullres, flow3_bwd_fullres, flow4_bwd_fullres, flow5_bwd_fullres, flow6_bwd_fullres]
        occ = [occ2_fullres, occ3_fullres, occ4_fullres, occ5_fullres, occ6_fullres]
        return flow_fwd, flow_bwd, occ

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        if x.is_cuda:
            grid = grid
        vgrid = Variable(grid) + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, padding_mode='border')
        mask = torch.autograd.Variable(torch.ones(x.size()), requires_grad=False)
        mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask.data < 0.9999] = 0
        mask[mask.data > 0] = 1
        return output

