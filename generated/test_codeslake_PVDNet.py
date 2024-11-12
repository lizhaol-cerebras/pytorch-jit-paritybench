
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


import numpy as np


import math


import torch.utils.data


from torch.utils.data.sampler import Sampler


import torch.distributed as dist


import random


import torch.utils.data as data


import torchvision.utils as vutils


import torch.nn.functional as F


from sklearn.metrics import mean_absolute_error


import collections


import torch.nn as nn


import re


from collections import Counter


from collections import defaultdict


from torch.optim.lr_scheduler import _LRScheduler


from torch.nn.parallel import DataParallel as DP


from torch.nn.parallel import DistributedDataParallel as DDP


import torch.nn.utils as torch_utils


import time


import torch.multiprocessing as mp


import numpy


def cupy_kernel(strFunction, objectVariables):
    strKernel = globals()[strFunction].replace('{{intStride}}', str(objectVariables['intStride']))
    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\\()([^\\)]*)(\\))', strKernel)
        if objectMatch is None:
            break
        intArg = int(objectMatch.group(2))
        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()
        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
    while True:
        objectMatch = re.search('(VALUE_)([0-4])(\\()([^\\)]+)(\\))', strKernel)
        if objectMatch is None:
            break
        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')
        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = [('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')') for intArg in range(intArgs)]
        strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    return strKernel


class _FunctionCorrelation(torch.autograd.Function):

    @staticmethod
    def forward(self, first, second, intStride):
        rbot0 = first.new_zeros([first.size(0), first.size(2) + 6 * intStride, first.size(3) + 6 * intStride, first.size(1)])
        rbot1 = first.new_zeros([first.size(0), first.size(2) + 6 * intStride, first.size(3) + 6 * intStride, first.size(1)])
        self.save_for_backward(first, second, rbot0, rbot1)
        self.intStride = intStride
        assert first.is_contiguous() == True
        assert second.is_contiguous() == True
        output = first.new_zeros([first.size(0), 49, int(math.ceil(first.size(2) / intStride)), int(math.ceil(first.size(3) / intStride))])
        if first.is_cuda == True:
            n = first.size(2) * first.size(3)
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {'intStride': self.intStride, 'input': first, 'output': rbot0}))(grid=tuple([int((n + 16 - 1) / 16), first.size(1), first.size(0)]), block=tuple([16, 1, 1]), args=[n, first.data_ptr(), rbot0.data_ptr()], stream=Stream)
            n = second.size(2) * second.size(3)
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {'intStride': self.intStride, 'input': second, 'output': rbot1}))(grid=tuple([int((n + 16 - 1) / 16), second.size(1), second.size(0)]), block=tuple([16, 1, 1]), args=[n, second.data_ptr(), rbot1.data_ptr()], stream=Stream)
            n = output.size(1) * output.size(2) * output.size(3)
            cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {'intStride': self.intStride, 'rbot0': rbot0, 'rbot1': rbot1, 'top': output}))(grid=tuple([output.size(3), output.size(2), output.size(0)]), block=tuple([32, 1, 1]), shared_mem=first.size(1) * 4, args=[n, rbot0.data_ptr(), rbot1.data_ptr(), output.data_ptr()], stream=Stream)
        elif first.is_cuda == False:
            raise NotImplementedError()
        return output

    @staticmethod
    def backward(self, gradOutput):
        first, second, rbot0, rbot1 = self.saved_tensors
        assert gradOutput.is_contiguous() == True
        gradFirst = first.new_zeros([first.size(0), first.size(1), first.size(2), first.size(3)]) if self.needs_input_grad[0] == True else None
        gradSecond = first.new_zeros([first.size(0), first.size(1), first.size(2), first.size(3)]) if self.needs_input_grad[1] == True else None
        if first.is_cuda == True:
            if gradFirst is not None:
                for intSample in range(first.size(0)):
                    n = first.size(1) * first.size(2) * first.size(3)
                    cupy_launch('kernel_Correlation_updateGradFirst', cupy_kernel('kernel_Correlation_updateGradFirst', {'intStride': self.intStride, 'rbot0': rbot0, 'rbot1': rbot1, 'gradOutput': gradOutput, 'gradFirst': gradFirst, 'gradSecond': None}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[n, intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(), gradFirst.data_ptr(), None], stream=Stream)
            if gradSecond is not None:
                for intSample in range(first.size(0)):
                    n = first.size(1) * first.size(2) * first.size(3)
                    cupy_launch('kernel_Correlation_updateGradSecond', cupy_kernel('kernel_Correlation_updateGradSecond', {'intStride': self.intStride, 'rbot0': rbot0, 'rbot1': rbot1, 'gradOutput': gradOutput, 'gradFirst': None, 'gradSecond': gradSecond}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[n, intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(), None, gradSecond.data_ptr()], stream=Stream)
        elif first.is_cuda == False:
            raise NotImplementedError()
        return gradFirst, gradSecond, None


def FunctionCorrelation(tensorFirst, tensorSecond, intStride):
    return _FunctionCorrelation.apply(tensorFirst, tensorSecond, intStride)


Backward_tensorGrid = {}


def warp(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1)
    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)
    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)


class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()


        class Features(torch.nn.Module):

            def __init__(self):
                super(Features, self).__init__()
                self.moduleOne = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.moduleTwo = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.moduleThr = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.moduleFou = torch.nn.Sequential(torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.moduleFiv = torch.nn.Sequential(torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.moduleSix = torch.nn.Sequential(torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

            def forward(self, tensorInput):
                tensorOne = self.moduleOne(tensorInput)
                tensorTwo = self.moduleTwo(tensorOne)
                tensorThr = self.moduleThr(tensorTwo)
                tensorFou = self.moduleFou(tensorThr)
                tensorFiv = self.moduleFiv(tensorFou)
                tensorSix = self.moduleSix(tensorFiv)
                return [tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix]


        class Matching(torch.nn.Module):

            def __init__(self, intLevel):
                super(Matching, self).__init__()
                self.dblBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
                if intLevel != 2:
                    self.moduleFeat = torch.nn.Sequential()
                elif intLevel == 2:
                    self.moduleFeat = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                if intLevel == 6:
                    self.moduleUpflow = None
                elif intLevel != 6:
                    self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1, bias=False, groups=2)
                if intLevel >= 4:
                    self.moduleUpcorr = None
                elif intLevel < 4:
                    self.moduleUpcorr = torch.nn.ConvTranspose2d(in_channels=49, out_channels=49, kernel_size=4, stride=2, padding=1, bias=False, groups=49)
                self.moduleMain = torch.nn.Sequential(torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):
                tensorFeaturesFirst = self.moduleFeat(tensorFeaturesFirst)
                tensorFeaturesSecond = self.moduleFeat(tensorFeaturesSecond)
                if tensorFlow is not None:
                    tensorFlow = self.moduleUpflow(tensorFlow)
                if tensorFlow is not None:
                    tensorFeaturesSecond = warp(tensorInput=tensorFeaturesSecond, tensorFlow=tensorFlow * self.dblBackward)
                if self.moduleUpcorr is None:
                    tensorCorrelation = torch.nn.functional.leaky_relu(input=FunctionCorrelation(tensorFirst=tensorFeaturesFirst, tensorSecond=tensorFeaturesSecond, intStride=1), negative_slope=0.1, inplace=False)
                elif self.moduleUpcorr is not None:
                    tensorCorrelation = self.moduleUpcorr(torch.nn.functional.leaky_relu(input=FunctionCorrelation(tensorFirst=tensorFeaturesFirst, tensorSecond=tensorFeaturesSecond, intStride=2), negative_slope=0.1, inplace=False))
                return (tensorFlow if tensorFlow is not None else 0.0) + self.moduleMain(tensorCorrelation)


        class Subpixel(torch.nn.Module):

            def __init__(self, intLevel):
                super(Subpixel, self).__init__()
                self.dblBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
                if intLevel != 2:
                    self.moduleFeat = torch.nn.Sequential()
                elif intLevel == 2:
                    self.moduleFeat = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.moduleMain = torch.nn.Sequential(torch.nn.Conv2d(in_channels=[0, 0, 130, 130, 194, 258, 386][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):
                tensorFeaturesFirst = self.moduleFeat(tensorFeaturesFirst)
                tensorFeaturesSecond = self.moduleFeat(tensorFeaturesSecond)
                if tensorFlow is not None:
                    tensorFeaturesSecond = warp(tensorInput=tensorFeaturesSecond, tensorFlow=tensorFlow * self.dblBackward)
                return (tensorFlow if tensorFlow is not None else 0.0) + self.moduleMain(torch.cat([tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow], 1))


        class Regularization(torch.nn.Module):

            def __init__(self, intLevel):
                super(Regularization, self).__init__()
                self.dblBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
                self.intUnfold = [0, 0, 7, 5, 5, 3, 3][intLevel]
                if intLevel >= 5:
                    self.moduleFeat = torch.nn.Sequential()
                elif intLevel < 5:
                    self.moduleFeat = torch.nn.Sequential(torch.nn.Conv2d(in_channels=[0, 0, 32, 64, 96, 128, 192][intLevel], out_channels=128, kernel_size=1, stride=1, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.moduleMain = torch.nn.Sequential(torch.nn.Conv2d(in_channels=[0, 0, 131, 131, 131, 131, 195][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                if intLevel >= 5:
                    self.moduleDist = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))
                elif intLevel < 5:
                    self.moduleDist = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], kernel_size=([0, 0, 7, 5, 5, 3, 3][intLevel], 1), stride=1, padding=([0, 0, 3, 2, 2, 1, 1][intLevel], 0)), torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], kernel_size=(1, [0, 0, 7, 5, 5, 3, 3][intLevel]), stride=1, padding=(0, [0, 0, 3, 2, 2, 1, 1][intLevel])))
                self.moduleScaleX = torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
                self.moduleScaleY = torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):
                tensorDifference = (tensorFirst - warp(tensorInput=tensorSecond, tensorFlow=tensorFlow * self.dblBackward)).pow(2.0).sum(1, True).sqrt().detach()
                tensorDist = self.moduleDist(self.moduleMain(torch.cat([tensorDifference, tensorFlow - tensorFlow.view(tensorFlow.size(0), 2, -1).mean(2, True).view(tensorFlow.size(0), 2, 1, 1), self.moduleFeat(tensorFeaturesFirst)], 1)))
                tensorDist = tensorDist.pow(2.0).neg()
                tensorDist = (tensorDist - tensorDist.max(1, True)[0]).exp()
                tensorDivisor = tensorDist.sum(1, True).reciprocal()
                tensorScaleX = self.moduleScaleX(tensorDist * torch.nn.functional.unfold(input=tensorFlow[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tensorDist)) * tensorDivisor
                tensorScaleY = self.moduleScaleY(tensorDist * torch.nn.functional.unfold(input=tensorFlow[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tensorDist)) * tensorDivisor
                return torch.cat([tensorScaleX, tensorScaleY], 1)
        self.moduleFeatures = Features()
        self.moduleMatching = torch.nn.ModuleList([Matching(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.moduleSubpixel = torch.nn.ModuleList([Subpixel(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.moduleRegularization = torch.nn.ModuleList([Regularization(intLevel) for intLevel in [2, 3, 4, 5, 6]])

    def forward(self, tensorFirst, tensorSecond):
        tensorFeaturesFirst = self.moduleFeatures(tensorFirst)
        tensorFeaturesSecond = self.moduleFeatures(tensorSecond)
        tensorFirst = [tensorFirst]
        tensorSecond = [tensorSecond]
        for intLevel in [1, 2, 3, 4, 5]:
            tensorFirst.append(torch.nn.functional.interpolate(input=tensorFirst[-1], size=(tensorFeaturesFirst[intLevel].size(2), tensorFeaturesFirst[intLevel].size(3)), mode='bilinear', align_corners=False))
            tensorSecond.append(torch.nn.functional.interpolate(input=tensorSecond[-1], size=(tensorFeaturesSecond[intLevel].size(2), tensorFeaturesSecond[intLevel].size(3)), mode='bilinear', align_corners=False))
        tensorFlow = None
        for intLevel in [-1, -2, -3, -4, -5]:
            tensorFlow = self.moduleMatching[intLevel](tensorFirst[intLevel], tensorSecond[intLevel], tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel], tensorFlow)
            tensorFlow = self.moduleSubpixel[intLevel](tensorFirst[intLevel], tensorSecond[intLevel], tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel], tensorFlow)
            tensorFlow = self.moduleRegularization[intLevel](tensorFirst[intLevel], tensorSecond[intLevel], tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel], tensorFlow)
        return tensorFlow * 20.0


class ModuleCorrelation(torch.nn.Module):

    def __init__(self):
        super(ModuleCorrelation, self).__init__()

    def forward(self, tensorFirst, tensorSecond, intStride):
        return _FunctionCorrelation.apply(tensorFirst, tensorSecond, intStride)


Grid = {}


def get_pixel_value(img, x, y, h, w):
    img = img.permute(0, 3, 1, 2)
    x_ = torch.unsqueeze(x, 3)
    y_ = torch.unsqueeze(y, 3)
    f = torch.cat([x_, y_], 3)
    f = torch.cat([2.0 * f[:, :, :, 0:1] / (w - 1.0) - 1, 2.0 * f[:, :, :, 1:2] / (h - 1.0) - 1], 3)
    return torch.nn.functional.grid_sample(input=img, grid=f, mode='nearest', padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1)


def get_pixel_volume(img, flow, pad, H, W, ksize=5):
    img = img.permute(0, 2, 3, 1)
    flow = flow.permute(0, 2, 3, 1)
    pad = pad.permute(0, 2, 3, 1)
    batch_size = img.size()[0]
    hksize = int(np.floor(ksize / 2))
    flow = flow.permute(0, 3, 1, 2)
    if str(flow.size()) not in Grid:
        x, y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H))
        x = torch.unsqueeze(torch.unsqueeze(x.permute(1, 0), 0), 0).type(torch.FloatTensor)
        y = torch.unsqueeze(torch.unsqueeze(y.permute(1, 0), 0), 0).type(torch.FloatTensor)
        grid = torch.cat([x, y], axis=1)
        Grid[str(flow.size())] = grid
        flows = grid + flow
    else:
        flows = Grid[str(flow.size())] + flow
    x = flows[:, 0, :, :]
    y = flows[:, 1, :, :]
    max_y = H - 1
    max_x = W - 1
    img_gray = torch.unsqueeze(0.2989 * img[:, :, :, 0] + 0.587 * img[:, :, :, 1] + 0.114 * img[:, :, :, 2], axis=3)
    pad_gray = torch.unsqueeze(0.2989 * pad[:, :, :, 0] + 0.587 * pad[:, :, :, 1] + 0.114 * pad[:, :, :, 2], axis=3)
    out = []
    for i in range(-hksize, hksize + 1):
        for j in range(-hksize, hksize + 1):
            x0_ = x.type(torch.IntTensor) + i
            y0_ = y.type(torch.IntTensor) + j
            x0 = torch.clamp(x0_, min=0, max=max_x)
            y0 = torch.clamp(y0_, min=0, max=max_y)
            Ia = get_pixel_value(img_gray, x0, y0, H, W)
            mask_x = torch.lt(x0_, 1.0).type(torch.FloatTensor) + torch.gt(x0_, max_x - 1).type(torch.FloatTensor)
            mask_y = torch.lt(y0_, 1.0).type(torch.FloatTensor) + torch.gt(y0_, max_y - 1).type(torch.FloatTensor)
            mask = torch.gt(mask_x + mask_y, 0).type(torch.FloatTensor)
            mask = F.pad(mask, (hksize, hksize, hksize, hksize, 0, 0))
            mask = mask[:, hksize - j:hksize - j + H, hksize - i:hksize - i + W]
            Ia = F.pad(Ia, (0, 0, hksize, hksize, hksize, hksize, 0, 0))
            Ia = Ia[:, hksize - j:hksize - j + H, hksize - i:hksize - i + W, :]
            Ia = torch.mul(Ia, 1 - torch.unsqueeze(mask, axis=3)) + torch.mul(pad_gray, torch.unsqueeze(mask, axis=3))
            out.append(Ia)
    out = torch.cat(out, 3)
    out = out.permute(0, 3, 1, 2)
    return out


def norm(inp):
    return (inp + 1.0) / 2.0


def upsample(inp, h=None, w=None, mode='bilinear'):
    return F.interpolate(input=inp, size=(int(h), int(w)), mode=mode, align_corners=False)


class DeblurNet(nn.Module):

    def __init__(self, config):
        super(DeblurNet, self).__init__()
        self.rank = torch.distributed.get_rank() if config.dist else -1
        self.config = config
        if self.rank <= 0:
            None
        lib = importlib.import_module('models.archs.{}'.format(config.network_BIMNet))
        self.BIMNet = lib.Network()
        lib = importlib.import_module('models.archs.{}'.format(config.network))
        self.PVDNet = lib.Network(config.PV_ksize ** 2)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight, gain=self.config.wi)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, 0)

    def init(self):
        if self.config.fix_BIMNet:
            None
            if self.rank <= 0:
                None
            for param in self.BIMNet.parameters():
                param.requires_grad_(False)
        else:
            self.BIMNet.apply(self.weights_init)
        self.PVDNet.apply(self.weights_init)

    def input_constructor(self, res):
        b, c, h, w = res[:]
        img = torch.FloatTensor(np.random.randn(b, c, h, w))
        return {'I_prev': img, 'I_curr': img, 'I_next': img, 'I_prev_deblurred': img}

    def forward(self, I_prev, I_curr, I_next, I_prev_deblurred, gt_prev=None, gt_curr=None, is_train=False):
        _, _, h, w = I_curr.size()
        refine_h = h - h % 32
        refine_w = w - w % 32
        I_curr_refined = I_curr[:, :, 0:refine_h, 0:refine_w]
        I_prev_refined = I_prev[:, :, 0:refine_h, 0:refine_w]
        outs = collections.OrderedDict()
        w_bb = upsample(self.BIMNet(norm(I_curr_refined), norm(I_prev_refined)), refine_h, refine_w)
        if refine_h != h or refine_w != w:
            w_bb = F.pad(w_bb, (0, w - refine_w, 0, h - refine_h, 0, 0, 0, 0))
        outs['PV'] = get_pixel_volume(I_prev_deblurred, w_bb, I_curr, h, w)
        outs['result'] = self.PVDNet(outs['PV'], I_prev, I_curr, I_next)
        if is_train and self.config.fix_BIMNet is False:
            gt_curr_refined = gt_curr[:, :, 0:refine_h, 0:refine_w]
            gt_prev_refined = gt_prev[:, :, 0:refine_h, 0:refine_w]
            w_bs = upsample(self.BIMNet(norm(gt_curr_refined), norm(I_prev_refined)), refine_h, refine_w)
            w_sb = upsample(self.BIMNet(norm(I_curr_refined), norm(gt_prev_refined)), refine_h, refine_w)
            w_ss = upsample(self.BIMNet(norm(gt_curr_refined), norm(gt_prev_refined)), refine_h, refine_w)
            if refine_h != h or refine_w != w:
                w_bs = F.pad(w_bs, (0, w - refine_w, 0, h - refine_h, 0, 0, 0, 0))
                w_sb = F.pad(w_sb, (0, w - refine_w, 0, h - refine_h, 0, 0, 0, 0))
                w_ss = F.pad(w_ss, (0, w - refine_w, 0, h - refine_h, 0, 0, 0, 0))
            outs['warped_bb'] = warp(norm(gt_prev), w_bb)
            outs['warped_bs'] = warp(norm(gt_prev), w_bs)
            outs['warped_sb'] = warp(norm(gt_prev), w_sb)
            outs['warped_ss'] = warp(norm(gt_prev), w_ss)
            with torch.no_grad():
                outs['warped_bb_mask'] = warp(torch.ones_like(gt_prev), w_bb)
                outs['warped_bs_mask'] = warp(torch.ones_like(gt_prev), w_bs)
                outs['warped_sb_mask'] = warp(torch.ones_like(gt_prev), w_sb)
                outs['warped_ss_mask'] = warp(torch.ones_like(gt_prev), w_ss)
        elif self.config.save_sample and gt_prev is not None:
            outs['warped_bb'] = warp(norm(gt_prev), w_bb)
        return outs

