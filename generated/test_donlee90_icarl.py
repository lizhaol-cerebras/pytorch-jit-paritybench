
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


from torchvision.datasets import CIFAR10


import numpy as np


import torch


import torch.nn as nn


import torchvision.datasets as dsets


import torchvision.models as models


import torchvision.transforms as transforms


from torch.autograd import Variable


import torch.optim as optim


import torch.nn.functional as F


import matplotlib.pyplot as plt


import matplotlib.gridspec as gridspec


batch_size = 100


learning_rate = 0.002


num_epochs = 50


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.layers = nn.Sequential(conv3x3(in_planes, planes, stride), nn.BatchNorm2d(planes), nn.ReLU(True), conv3x3(planes, planes), nn.BatchNorm2d(planes))
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


class ResNet(nn.Module):

    def __init__(self, block, nblocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.pre_layers = nn.Sequential(conv3x3(3, 64), nn.BatchNorm2d(64), nn.ReLU(True))
        self.layer1 = self._make_layer(block, 64, nblocks[0])
        self.layer2 = self._make_layer(block, 128, nblocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, nblocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, nblocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, nblocks, stride=1):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


class iCaRLNet(nn.Module):

    def __init__(self, feature_size, n_classes):
        super(iCaRLNet, self).__init__()
        self.feature_extractor = resnet18()
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, feature_size)
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(feature_size, n_classes, bias=False)
        self.n_classes = n_classes
        self.n_known = 0
        self.exemplar_sets = []
        self.cls_loss = nn.CrossEntropyLoss()
        self.dist_loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-05)
        self.compute_means = True
        self.exemplar_means = []

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x

    def increment_classes(self, n):
        """Add n classes in the final fc layer"""
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data
        self.fc = nn.Linear(in_features, out_features + n, bias=False)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

    def classify(self, x, transform):
        """Classify images by neares-means-of-exemplars

        Args:
            x: input image batch
        Returns:
            preds: Tensor of size (batch_size,)
        """
        batch_size = x.size(0)
        if self.compute_means:
            None
            exemplar_means = []
            for P_y in self.exemplar_sets:
                features = []
                for ex in P_y:
                    ex = Variable(transform(Image.fromarray(ex)), volatile=True)
                    feature = self.feature_extractor(ex.unsqueeze(0))
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm()
                    features.append(feature)
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm()
                exemplar_means.append(mu_y)
            self.exemplar_means = exemplar_means
            self.compute_means = False
            None
        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means)
        means = torch.stack([means] * batch_size)
        means = means.transpose(1, 2)
        feature = self.feature_extractor(x)
        for i in range(feature.size(0)):
            feature.data[i] = feature.data[i] / feature.data[i].norm()
        feature = feature.unsqueeze(2)
        feature = feature.expand_as(means)
        dists = (feature - means).pow(2).sum(1).squeeze()
        _, preds = dists.min(1)
        return preds

    def construct_exemplar_set(self, images, m, transform):
        """Construct an exemplar set for image set

        Args:
            images: np.array containing images of a class
        """
        features = []
        for img in images:
            x = Variable(transform(Image.fromarray(img)), volatile=True)
            feature = self.feature_extractor(x.unsqueeze(0)).data.cpu().numpy()
            feature = feature / np.linalg.norm(feature)
            features.append(feature[0])
        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean)
        exemplar_set = []
        exemplar_features = []
        for k in range(m):
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0 / (k + 1) * (phi + S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))
            exemplar_set.append(images[i])
            exemplar_features.append(features[i])
            """
            print "Selected example", i
            print "|exemplar_mean - class_mean|:",
            print np.linalg.norm((np.mean(exemplar_features, axis=0) - class_mean))
            #features = np.delete(features, i, axis=0)
            """
        self.exemplar_sets.append(np.array(exemplar_set))

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]

    def combine_dataset_with_exemplars(self, dataset):
        for y, P_y in enumerate(self.exemplar_sets):
            exemplar_images = P_y
            exemplar_labels = [y] * len(P_y)
            dataset.append(exemplar_images, exemplar_labels)

    def update_representation(self, dataset):
        self.compute_means = True
        classes = list(set(dataset.train_labels))
        new_classes = [cls for cls in classes if cls > self.n_classes - 1]
        self.increment_classes(len(new_classes))
        self
        None
        self.combine_dataset_with_exemplars(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        q = torch.zeros(len(dataset), self.n_classes)
        for indices, images, labels in loader:
            images = Variable(images)
            indices = indices
            g = F.sigmoid(self.forward(images))
            q[indices] = g.data
        q = Variable(q)
        optimizer = self.optimizer
        for epoch in range(num_epochs):
            for i, (indices, images, labels) in enumerate(loader):
                images = Variable(images)
                labels = Variable(labels)
                indices = indices
                optimizer.zero_grad()
                g = self.forward(images)
                loss = self.cls_loss(g, labels)
                if self.n_known > 0:
                    g = F.sigmoid(g)
                    q_i = q[indices]
                    dist_loss = sum(self.dist_loss(g[:, y], q_i[:, y]) for y in range(self.n_known))
                    loss += dist_loss
                loss.backward()
                optimizer.step()
                if (i + 1) % 10 == 0:
                    None


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False), nn.BatchNorm2d(planes), nn.ReLU(True), nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(planes), nn.ReLU(True), nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False), nn.BatchNorm2d(planes * 4))
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

