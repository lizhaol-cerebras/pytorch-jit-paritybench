
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


from typing import Callable


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import pandas as pd


from pandas import DataFrame


from torch import Tensor


import warnings


from typing import Set


from typing import Dict


import numpy as np


import torch


from numpy import ndarray


from abc import abstractmethod


from torch.utils.data import Dataset


from torchvision.datasets import ImageFolder


from torch import nn


from collections import OrderedDict


import math


from torchvision.models.resnet import conv3x3


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


from typing import Type


from torchvision.models.resnet import conv1x1


import random


from typing import Iterator


from torch.utils.data import Sampler


import torch.testing


from torchvision import transforms


from itertools import product


from torch.utils.data import DataLoader


import torchvision


from matplotlib import pyplot as plt


from torchvision.transforms import InterpolationMode


def compute_prototypes(support_features: 'Tensor', support_labels: 'Tensor') ->Tensor:
    """
    Compute class prototypes from support features and labels
    Args:
        support_features: for each instance in the support set, its feature vector
        support_labels: for each instance in the support set, its label

    Returns:
        for each label of the support set, the average feature vector of instances with this label
    """
    n_way = len(torch.unique(support_labels))
    return torch.cat([support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])


class FewShotClassifier(nn.Module):
    """
    Abstract class providing methods usable by all few-shot classification algorithms
    """

    def __init__(self, backbone: 'Optional[nn.Module]'=None, use_softmax: 'bool'=False, feature_centering: 'Optional[Tensor]'=None, feature_normalization: 'Optional[float]'=None):
        """
        Initialize the Few-Shot Classifier
        Args:
            backbone: the feature extractor used by the method. Must output a tensor of the
                appropriate shape (depending on the method).
                If None is passed, the backbone will be initialized as nn.Identity().
            use_softmax: whether to return predictions as soft probabilities
            feature_centering: a features vector on which to center all computed features.
                If None is passed, no centering is performed.
            feature_normalization: a value by which to normalize all computed features after centering.
                It is used as the p argument in torch.nn.functional.normalize().
                If None is passed, no normalization is performed.
        """
        super().__init__()
        self.backbone = backbone if backbone is not None else nn.Identity()
        self.use_softmax = use_softmax
        self.prototypes = torch.tensor(())
        self.support_features = torch.tensor(())
        self.support_labels = torch.tensor(())
        self.feature_centering = feature_centering if feature_centering is not None else torch.tensor(0)
        self.feature_normalization = feature_normalization

    @abstractmethod
    def forward(self, query_images: 'Tensor') ->Tensor:
        """
        Predict classification labels.
        Args:
            query_images: images of the query set of shape (n_query, **image_shape)
        Returns:
            a prediction of classification scores for query images of shape (n_query, n_classes)
        """
        raise NotImplementedError('All few-shot algorithms must implement a forward method.')

    def process_support_set(self, support_images: 'Tensor', support_labels: 'Tensor'):
        """
        Harness information from the support set, so that query labels can later be predicted using a forward call.
        The default behaviour shared by most few-shot classifiers is to compute prototypes and store the support set.
        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
        self.compute_prototypes_and_store_support_set(support_images, support_labels)

    @staticmethod
    def is_transductive() ->bool:
        raise NotImplementedError('All few-shot algorithms must implement a is_transductive method.')

    def compute_features(self, images: 'Tensor') ->Tensor:
        """
        Compute features from images and perform centering and normalization.
        Args:
            images: images of shape (n_images, **image_shape)
        Returns:
            features of shape (n_images, feature_dimension)
        """
        original_features = self.backbone(images)
        centered_features = original_features - self.feature_centering
        if self.feature_normalization is not None:
            return nn.functional.normalize(centered_features, p=self.feature_normalization, dim=1)
        return centered_features

    def softmax_if_specified(self, output: 'Tensor', temperature: 'float'=1.0) ->Tensor:
        """
        If the option is chosen when the classifier is initialized, we perform a softmax on the
        output in order to return soft probabilities.
        Args:
            output: output of the forward method of shape (n_query, n_classes)
            temperature: temperature of the softmax
        Returns:
            output as it was, or output as soft probabilities, of shape (n_query, n_classes)
        """
        return (temperature * output).softmax(-1) if self.use_softmax else output

    def l2_distance_to_prototypes(self, samples: 'Tensor') ->Tensor:
        """
        Compute prediction logits from their euclidean distance to support set prototypes.
        Args:
            samples: features of the items to classify of shape (n_samples, feature_dimension)
        Returns:
            prediction logits of shape (n_samples, n_classes)
        """
        return -torch.cdist(samples, self.prototypes)

    def cosine_distance_to_prototypes(self, samples) ->Tensor:
        """
        Compute prediction logits from their cosine distance to support set prototypes.
        Args:
            samples: features of the items to classify of shape (n_samples, feature_dimension)
        Returns:
            prediction logits of shape (n_samples, n_classes)
        """
        return nn.functional.normalize(samples, dim=1) @ nn.functional.normalize(self.prototypes, dim=1).T

    def compute_prototypes_and_store_support_set(self, support_images: 'Tensor', support_labels: 'Tensor'):
        """
        Extract support features, compute prototypes, and store support labels, features, and prototypes.
        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
        self.support_labels = support_labels
        self.support_features = self.compute_features(support_images)
        self._raise_error_if_features_are_multi_dimensional(self.support_features)
        self.prototypes = compute_prototypes(self.support_features, support_labels)

    @staticmethod
    def _raise_error_if_features_are_multi_dimensional(features: 'Tensor'):
        if len(features.shape) != 2:
            raise ValueError('Illegal backbone or feature shape. Expected output for an image is a 1-dim tensor.')


class Finetune(FewShotClassifier):
    """
    Wei-Yu Chen, Yen-Cheng Liu, Zsolt Kira, Yu-Chiang Frank Wang, Jia-Bin Huang
    A Closer Look at Few-shot Classification (ICLR 2019)
    https://arxiv.org/abs/1904.04232

    Fine-tune prototypes based on classification error on support images.
    Classify queries based on their cosine distances to updated prototypes.
    As is, it is incompatible with episodic training because we freeze the backbone to perform
    fine-tuning.

    This is an inductive method.
    """

    def __init__(self, *args, fine_tuning_steps: int=200, fine_tuning_lr: float=0.0001, temperature: float=1.0, **kwargs):
        """
        Args:
            fine_tuning_steps: number of fine-tuning steps
            fine_tuning_lr: learning rate for fine-tuning
            temperature: temperature applied to the logits before computing
                softmax or cross-entropy. Higher temperature means softer predictions.
        """
        super().__init__(*args, **kwargs)
        self.backbone.requires_grad_(False)
        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr
        self.temperature = temperature

    def forward(self, query_images: 'Tensor') ->Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Fine-tune prototypes based on support classification error.
        Then classify w.r.t. to cosine distance to prototypes.
        """
        query_features = self.compute_features(query_images)
        with torch.enable_grad():
            self.prototypes.requires_grad_()
            optimizer = torch.optim.Adam([self.prototypes], lr=self.fine_tuning_lr)
            for _ in range(self.fine_tuning_steps):
                support_logits = self.cosine_distance_to_prototypes(self.support_features)
                loss = nn.functional.cross_entropy(self.temperature * support_logits, self.support_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.softmax_if_specified(self.cosine_distance_to_prototypes(query_features), temperature=self.temperature).detach()

    @staticmethod
    def is_transductive() ->bool:
        return False


def default_matching_networks_query_encoder(feature_dimension: 'int') ->nn.Module:
    return nn.LSTMCell(feature_dimension * 2, feature_dimension)


def default_matching_networks_support_encoder(feature_dimension: 'int') ->nn.Module:
    return nn.LSTM(input_size=feature_dimension, hidden_size=feature_dimension, num_layers=1, batch_first=True, bidirectional=True)


class MatchingNetworks(FewShotClassifier):
    """
    Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra.
    "Matching networks for one shot learning." (2016)
    https://arxiv.org/pdf/1606.04080.pdf

    Matching networks extract feature vectors for both support and query images. Then they refine
    these feature by using the context of the whole support set, using LSTMs. Finally they compute
    query labels using their cosine similarity to support images.

    Be careful: while some methods use Cross Entropy Loss for episodic training, Matching Networks
    output log-probabilities, so you'll want to use Negative Log Likelihood Loss.
    """

    def __init__(self, *args, feature_dimension: int, support_encoder: Optional[nn.Module]=None, query_encoder: Optional[nn.Module]=None, **kwargs):
        """
        Build Matching Networks by calling the constructor of FewShotClassifier.
        Args:
            feature_dimension: dimension of the feature vectors extracted by the backbone.
            support_encoder: module encoding support features. If none is specific, we use
                the default encoder from the original paper.
            query_encoder: module encoding query features. If none is specific, we use
                the default encoder from the original paper.
        """
        super().__init__(*args, **kwargs)
        self.feature_dimension = feature_dimension
        self.support_features_encoder = support_encoder if support_encoder else default_matching_networks_support_encoder(self.feature_dimension)
        self.query_features_encoding_cell = query_encoder if query_encoder else default_matching_networks_query_encoder(self.feature_dimension)
        self.softmax = nn.Softmax(dim=1)
        self.contextualized_support_features = torch.tensor(())
        self.one_hot_support_labels = torch.tensor(())

    def process_support_set(self, support_images: 'Tensor', support_labels: 'Tensor'):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract features from the support set with full context embedding.
        Store contextualized feature vectors, as well as support labels in the one hot format.

        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
        support_features = self.compute_features(support_images)
        self._validate_features_shape(support_features)
        self.contextualized_support_features = self.encode_support_features(support_features)
        self.one_hot_support_labels = nn.functional.one_hot(support_labels).float()

    def forward(self, query_images: 'Tensor') ->Tensor:
        """
        Overrides method forward in FewShotClassifier.
        Predict query labels based on their cosine similarity to support set features.
        Classification scores are log-probabilities.

        Args:
            query_images: images of the query set of shape (n_query, **image_shape)
        Returns:
            a prediction of classification scores for query images of shape (n_query, n_classes)
        """
        query_features = self.compute_features(query_images)
        self._validate_features_shape(query_features)
        contextualized_query_features = self.encode_query_features(query_features)
        similarity_matrix = self.softmax(contextualized_query_features.mm(nn.functional.normalize(self.contextualized_support_features).T))
        log_probabilities = (similarity_matrix.mm(self.one_hot_support_labels) + 1e-06).log()
        return self.softmax_if_specified(log_probabilities)

    def encode_support_features(self, support_features: 'Tensor') ->Tensor:
        """
        Refine support set features by putting them in the context of the whole support set,
        using a bidirectional LSTM.
        Args:
            support_features: output of the backbone of shape (n_support, feature_dimension)

        Returns:
            contextualised support features, with the same shape as input features
        """
        hidden_state = self.support_features_encoder(support_features.unsqueeze(0))[0].squeeze(0)
        contextualized_support_features = support_features + hidden_state[:, :self.feature_dimension] + hidden_state[:, self.feature_dimension:]
        return contextualized_support_features

    def encode_query_features(self, query_features: 'Tensor') ->Tensor:
        """
        Refine query set features by putting them in the context of the whole support set,
        using attention over support set features.
        Args:
            query_features: output of the backbone of shape (n_query, feature_dimension)

        Returns:
            contextualized query features, with the same shape as input features
        """
        hidden_state = query_features
        cell_state = torch.zeros_like(query_features)
        for _ in range(len(self.contextualized_support_features)):
            attention = self.softmax(hidden_state.mm(self.contextualized_support_features.T))
            read_out = attention.mm(self.contextualized_support_features)
            lstm_input = torch.cat((query_features, read_out), 1)
            hidden_state, cell_state = self.query_features_encoding_cell(lstm_input, (hidden_state, cell_state))
            hidden_state = hidden_state + query_features
        return hidden_state

    def _validate_features_shape(self, features: 'Tensor'):
        self._raise_error_if_features_are_multi_dimensional(features)
        if features.shape[1] != self.feature_dimension:
            raise ValueError(f'Expected feature dimension is {self.feature_dimension}, but got {features.shape[1]}.')

    @staticmethod
    def is_transductive() ->bool:
        return False


class PrototypicalNetworks(FewShotClassifier):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their euclidean distance to the prototypes.
    """

    def forward(self, query_images: 'Tensor') ->Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Predict query labels based on their distance to class prototypes in the feature space.
        Classification scores are the negative of euclidean distances.
        """
        query_features = self.compute_features(query_images)
        self._raise_error_if_features_are_multi_dimensional(query_features)
        scores = self.l2_distance_to_prototypes(query_features)
        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() ->bool:
        return False


MAXIMUM_SINKHORN_ITERATIONS = 1000


def power_transform(features: 'Tensor', power_factor: 'float') ->Tensor:
    """
    Apply power transform to features.
    Args:
        features: input features of shape (n_features, feature_dimension)
        power_factor: power to apply to features

    Returns:
        Tensor: shape (n_features, feature_dimension), power transformed features.
    """
    return (features.relu() + 1e-06).pow(power_factor)


class PTMAP(FewShotClassifier):
    """
    Yuqing Hu, Vincent Gripon, Stéphane Pateux.
    "Leveraging the Feature Distribution in Transfer-based Few-Shot Learning" (2020)
    https://arxiv.org/abs/2006.03806

    Query soft assignments are computed as the optimal transport plan to class prototypes.
    At each iteration, prototypes are fine-tuned based on the soft assignments.
    This is a transductive method.
    """

    def __init__(self, *args, fine_tuning_steps: int=10, fine_tuning_lr: float=0.2, lambda_regularization: float=10.0, power_factor: float=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr
        self.lambda_regularization = lambda_regularization
        self.power_factor = power_factor

    def forward(self, query_images: 'Tensor') ->Tensor:
        """
        Predict query soft assignments following Algorithm 1 of the paper.
        """
        query_features = self.compute_features(query_images)
        support_assignments = nn.functional.one_hot(self.support_labels, len(self.prototypes))
        for _ in range(self.fine_tuning_steps):
            query_soft_assignments = self.compute_soft_assignments(query_features)
            all_features = torch.cat([self.support_features, query_features], 0)
            all_assignments = torch.cat([support_assignments, query_soft_assignments], dim=0)
            self.update_prototypes(all_features, all_assignments)
        return self.compute_soft_assignments(query_features)

    def compute_features(self, images: 'Tensor') ->Tensor:
        """
        Apply power transform on features following Equation (1) in the paper.
        Args:
            images: images of shape (n_images, **image_shape)
        Returns:
            features of shape (n_images, feature_dimension) with power-transform.
        """
        features = super().compute_features(images)
        return power_transform(features, self.power_factor)

    def compute_soft_assignments(self, query_features: 'Tensor') ->Tensor:
        """
        Compute soft assignments from queries to prototypes, following Equation (3) of the paper.
        Args:
            query_features: query features, of shape (n_queries, feature_dim)

        Returns:
            soft assignments from queries to prototypes, of shape (n_queries, n_classes)
        """
        distances_to_prototypes = torch.cdist(query_features, self.prototypes) ** 2
        soft_assignments = self.compute_optimal_transport(distances_to_prototypes, epsilon=1e-06)
        return soft_assignments

    def compute_optimal_transport(self, cost_matrix: 'Tensor', epsilon: 'float'=1e-06) ->Tensor:
        """
        Compute the optimal transport plan from queries to prototypes using Sinkhorn-Knopp algorithm.
        Args:
            cost_matrix: euclidean distances from queries to prototypes,
                of shape (n_queries, n_classes)
            epsilon: convergence parameter. Stop when the update is smaller than epsilon.
        Returns:
            transport plan from queries to prototypes of shape (n_queries, n_classes)
        """
        instance_multiplication_factor = cost_matrix.shape[0] // cost_matrix.shape[1]
        transport_plan = torch.exp(-self.lambda_regularization * cost_matrix)
        transport_plan /= transport_plan.sum(dim=(0, 1), keepdim=True)
        for _ in range(MAXIMUM_SINKHORN_ITERATIONS):
            per_class_sums = transport_plan.sum(1)
            transport_plan *= (1 / (per_class_sums + 1e-10)).unsqueeze(1)
            transport_plan *= (instance_multiplication_factor / (transport_plan.sum(0) + 1e-10)).unsqueeze(0)
            if torch.max(torch.abs(per_class_sums - transport_plan.sum(1))) < epsilon:
                break
        return transport_plan

    def update_prototypes(self, all_features, all_assignments) ->None:
        """
        Update prototypes by weigh-averaging the features with their soft assignments,
            following Equation (6) of the paper.
        Args:
            all_features: concatenation of support and query features,
                of shape (n_support + n_query, feature_dim)
            all_assignments: concatenation of support and query soft assignments,
                of shape (n_support + n_query, n_classes)-
        """
        new_prototypes = all_assignments.T @ all_features / all_assignments.sum(0).unsqueeze(1)
        delta = new_prototypes - self.prototypes
        self.prototypes += self.fine_tuning_lr * delta

    @staticmethod
    def is_transductive() ->bool:
        return True


def default_relation_module(feature_dimension: 'int', inner_channels: 'int'=8) ->nn.Module:
    """
    Build the relation module that takes as input the concatenation of two feature maps, from
    Sung et al. : "Learning to compare: Relation network for few-shot learning." (2018)
    In order to make the network robust to any change in the dimensions of the input images,
    we made some changes to the architecture defined in the original implementation
    from Sung et al.(typically the use of adaptive pooling).
    Args:
        feature_dimension: the dimension of the feature space i.e. size of a feature vector
        inner_channels: number of hidden channels between the linear layers of  the relation module
    Returns:
        the constructed relation module
    """
    return nn.Sequential(nn.Sequential(nn.Conv2d(feature_dimension * 2, feature_dimension, kernel_size=3, padding=1), nn.BatchNorm2d(feature_dimension, momentum=1, affine=True), nn.ReLU(), nn.AdaptiveMaxPool2d((5, 5))), nn.Sequential(nn.Conv2d(feature_dimension, feature_dimension, kernel_size=3, padding=0), nn.BatchNorm2d(feature_dimension, momentum=1, affine=True), nn.ReLU(), nn.AdaptiveMaxPool2d((1, 1))), nn.Flatten(), nn.Linear(feature_dimension, inner_channels), nn.ReLU(), nn.Linear(inner_channels, 1), nn.Sigmoid())


class RelationNetworks(FewShotClassifier):
    """
    Sung, Flood, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M. Hospedales.
    "Learning to compare: Relation network for few-shot learning." (2018)
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf

    In the Relation Networks algorithm, we first extract feature maps for both support and query
    images. Then we compute the mean of support features for each class (called prototypes).
    To predict the label of a query image, its feature map is concatenated with each class prototype
    and fed into a relation module, i.e. a CNN that outputs a relation score. Finally, the
    classification vector of the query is its relation score to each class prototype.

    Note that for most other few-shot algorithms we talk about feature vectors, because for each
    input image, the backbone outputs a 1-dim feature vector. Here we talk about feature maps,
    because for each input image, the backbone outputs a "feature map" of shape
    (n_channels, width, height). This raises different constraints on the architecture of the
    backbone: while other algorithms require a "flatten" operation in the backbone, here "flatten"
    operations are forbidden.

    Relation Networks use Mean Square Error. This is unusual because this is a classification
    problem. The authors justify this choice by the fact that the output of the model is a relation
    score, which makes it a regression problem. See the article for more details.
    """

    def __init__(self, *args, feature_dimension: int, relation_module: Optional[nn.Module]=None, **kwargs):
        """
        Build Relation Networks by calling the constructor of FewShotClassifier.
        Args:
            feature_dimension: first dimension of the feature maps extracted by the backbone.
            relation_module: module that will take the concatenation of a query features vector
                and a prototype to output a relation score. If none is specific, we use the default
                relation module from the original paper.
        """
        super().__init__(*args, **kwargs)
        self.feature_dimension = feature_dimension
        self.relation_module = relation_module if relation_module else default_relation_module(self.feature_dimension)

    def process_support_set(self, support_images: 'Tensor', support_labels: 'Tensor'):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract feature maps from the support set and store class prototypes.
        """
        support_features = self.compute_features(support_images)
        self._validate_features_shape(support_features)
        self.prototypes = compute_prototypes(support_features, support_labels)

    def forward(self, query_images: 'Tensor') ->Tensor:
        """
        Overrides method forward in FewShotClassifier.
        Predict the label of a query image by concatenating its feature map with each class
        prototype and feeding the result into a relation module, i.e. a CNN that outputs a relation
        score. Finally, the classification vector of the query is its relation score to each class
        prototype.
        """
        query_features = self.compute_features(query_images)
        self._validate_features_shape(query_features)
        query_prototype_feature_pairs = torch.cat((self.prototypes.unsqueeze(dim=0).expand(query_features.shape[0], -1, -1, -1, -1), query_features.unsqueeze(dim=1).expand(-1, self.prototypes.shape[0], -1, -1, -1)), dim=2).view(-1, 2 * self.feature_dimension, *query_features.shape[2:])
        relation_scores = self.relation_module(query_prototype_feature_pairs).view(-1, self.prototypes.shape[0])
        return self.softmax_if_specified(relation_scores)

    def _validate_features_shape(self, features):
        if len(features.shape) != 4:
            raise ValueError('Illegal backbone for Relation Networks. Expected output for an image is a 3-dim  tensor of shape (n_channels, width, height).')
        if features.shape[1] != self.feature_dimension:
            raise ValueError(f'Expected feature dimension is {self.feature_dimension}, but got {features.shape[1]}.')

    @staticmethod
    def is_transductive() ->bool:
        return False


class SimpleShot(FewShotClassifier):
    """
    Yan Wang, Wei-Lun Chao, Kilian Q. Weinberger, and Laurens van der Maaten.
    "SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning" (2019)
    https://arxiv.org/abs/1911.04623

    Almost exactly Prototypical Classification, but with cosine distance instead of euclidean distance.
    """

    def forward(self, query_images: 'Tensor') ->Tensor:
        """
        Predict classification labels.
        Args:
            query_images: images of the query set of shape (n_query, **image_shape)
        Returns:
            a prediction of classification scores for query images of shape (n_query, n_classes)
        """
        query_features = self.compute_features(query_images)
        self._raise_error_if_features_are_multi_dimensional(query_features)
        scores = self.cosine_distance_to_prototypes(query_features)
        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() ->bool:
        return False


class TIM(FewShotClassifier):
    """
    Malik Boudiaf, Ziko Imtiaz Masud, Jérôme Rony, José Dolz, Pablo Piantanida, Ismail Ben Ayed.
    "Transductive Information Maximization For Few-Shot Learning" (NeurIPS 2020)
    https://arxiv.org/abs/2008.11297

    Fine-tune prototypes based on
        1) classification error on support images
        2) mutual information between query features and their label predictions
    Classify w.r.t. to euclidean distance to updated prototypes.
    As is, it is incompatible with episodic training because we freeze the backbone to perform
    fine-tuning.

    TIM is a transductive method.
    """

    def __init__(self, *args, fine_tuning_steps: int=50, fine_tuning_lr: float=0.0001, cross_entropy_weight: float=1.0, marginal_entropy_weight: float=1.0, conditional_entropy_weight: float=0.1, temperature: float=10.0, **kwargs):
        """
        Args:
            fine_tuning_steps: number of fine-tuning steps
            fine_tuning_lr: learning rate for fine-tuning
            cross_entropy_weight: weight given to the cross-entropy term of the loss
            marginal_entropy_weight: weight given to the marginal entropy term of the loss
            conditional_entropy_weight: weight given to the conditional entropy term of the loss
            temperature: temperature applied to the logits before computing
                softmax or cross-entropy. Higher temperature means softer predictions.
        """
        super().__init__(*args, **kwargs)
        self.backbone.requires_grad_(False)
        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr
        self.cross_entropy_weight = cross_entropy_weight
        self.marginal_entropy_weight = marginal_entropy_weight
        self.conditional_entropy_weight = conditional_entropy_weight
        self.temperature = temperature

    def forward(self, query_images: 'Tensor') ->Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Fine-tune prototypes based on support classification error and mutual information between
        query features and their label predictions.
        Then classify w.r.t. to euclidean distance to prototypes.
        """
        query_features = self.compute_features(query_images)
        num_classes = self.support_labels.unique().size(0)
        support_labels_one_hot = nn.functional.one_hot(self.support_labels, num_classes)
        with torch.enable_grad():
            self.prototypes.requires_grad_()
            optimizer = torch.optim.Adam([self.prototypes], lr=self.fine_tuning_lr)
            for _ in range(self.fine_tuning_steps):
                support_logits = self.temperature * self.cosine_distance_to_prototypes(self.support_features)
                query_logits = self.temperature * self.cosine_distance_to_prototypes(query_features)
                support_cross_entropy = -(support_labels_one_hot * support_logits.log_softmax(1)).sum(1).mean(0)
                query_soft_probs = query_logits.softmax(1)
                query_conditional_entropy = -(query_soft_probs * torch.log(query_soft_probs + 1e-12)).sum(1).mean(0)
                marginal_prediction = query_soft_probs.mean(0)
                marginal_entropy = -(marginal_prediction * torch.log(marginal_prediction)).sum(0)
                loss = self.cross_entropy_weight * support_cross_entropy - (self.marginal_entropy_weight * marginal_entropy - self.conditional_entropy_weight * query_conditional_entropy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.softmax_if_specified(self.cosine_distance_to_prototypes(query_features), temperature=self.temperature).detach()

    @staticmethod
    def is_transductive() ->bool:
        return True


def entropy(logits: 'Tensor') ->Tensor:
    """
    Compute entropy of prediction.
    WARNING: takes logit as input, not probability.
    Args:
        logits: shape (n_images, n_way)
    Returns:
        Tensor: shape(), Mean entropy.
    """
    probabilities = logits.softmax(dim=1)
    return (-(probabilities * (probabilities + 1e-12).log()).sum(dim=1)).mean()


class TransductiveFinetuning(Finetune):
    """
    Guneet S. Dhillon, Pratik Chaudhari, Avinash Ravichandran, Stefano Soatto.
    "A Baseline for Few-Shot Image Classification" (ICLR 2020)
    https://arxiv.org/abs/1909.02729

    Fine-tune the parameters of the pre-trained model based on
        1) classification error on support images
        2) classification entropy for query images
    Classify queries based on their euclidean distance to prototypes.
    This is a transductive method.
    WARNING: this implementation only updates prototypes, not the whole set of model's
    parameters. Updating the model's parameters raises performance issues that we didn't
    have time to solve yet. We welcome contributions.
    As is, it is incompatible with episodic training because we freeze the backbone to perform
    fine-tuning.
    """

    def __init__(self, *args, fine_tuning_steps: int=25, fine_tuning_lr: float=5e-05, temperature: float=1.0, **kwargs):
        """
        TransductiveFinetuning is very similar to the inductive method Finetune.
        The difference only resides in the way we perform the fine-tuning step and in the
        distance we use. Therefore, we call the super constructor of Finetune
        (and same for preprocess_support_set()).
        Args:
            fine_tuning_steps: number of fine-tuning steps
            fine_tuning_lr: learning rate for fine-tuning
            temperature: temperature applied to the logits before computing
                softmax or cross-entropy. Higher temperature means softer predictions.
        """
        super().__init__(*args, fine_tuning_steps=fine_tuning_steps, fine_tuning_lr=fine_tuning_lr, temperature=temperature, **kwargs)

    def forward(self, query_images: 'Tensor') ->Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Fine-tune model's parameters based on support classification error and
        query classification entropy.
        """
        query_features = self.compute_features(query_images)
        with torch.enable_grad():
            self.prototypes.requires_grad_()
            optimizer = torch.optim.Adam([self.prototypes], lr=self.fine_tuning_lr)
            for _ in range(self.fine_tuning_steps):
                support_cross_entropy = nn.functional.cross_entropy(self.temperature * self.l2_distance_to_prototypes(self.support_features), self.support_labels)
                query_conditional_entropy = entropy(self.temperature * self.l2_distance_to_prototypes(query_features))
                loss = support_cross_entropy + query_conditional_entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.softmax_if_specified(self.l2_distance_to_prototypes(query_features), temperature=self.temperature).detach()

    @staticmethod
    def is_transductive() ->bool:
        return True


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = attention / self.temperature
        raw_attention = attention
        log_attention = nn.functional.log_softmax(attention, 2)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        output = torch.bmm(attention, v)
        return output, attention, log_attention, raw_attention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, flag_norm=True):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=math.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=math.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=math.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=math.sqrt(d_k))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.flag_norm = flag_norm

    def forward(self, q, k, v):
        """
        Go through the multi-head attention module.
        """
        sz_q, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_q, len_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head, self.d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.d_v)
        output, _, _, _ = self.attention(q, k, v)
        output = output.view(self.n_head, sz_q, len_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_q, len_q, -1)
        resout = self.fc(output)
        output = self.dropout(resout)
        if self.flag_norm:
            output = self.layer_norm(output + residual)
        return output, resout


class FEATBasicBlock(nn.Module):
    """
    BasicBlock for FEAT. Uses 3 convolutions instead of 2, a LeakyReLU instead of ReLU, and a MaxPool2d.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample

    def forward(self, x):
        """
        Pass input through the block, including an activation and maxpooling at the end.
        """
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
        out = self.maxpool(out)
        return out


class FEATResNet12(nn.Module):
    """
    ResNet12 for FEAT. See feat_resnet12 doc for more details.
    """

    def __init__(self, block=FEATBasicBlock):
        self.inplanes = 3
        super().__init__()
        channels = [64, 160, 320, 640]
        self.layer_dims = [(channels[i] * block.expansion) for i in range(4) for j in range(4)]
        self.layer1 = self._make_layer(block, 64, stride=2)
        self.layer2 = self._make_layer(block, 160, stride=2)
        self.layer3 = self._make_layer(block, 320, stride=2)
        self.layer4 = self._make_layer(block, 640, stride=2)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Iterate over the blocks and apply them sequentially.
        """
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x.mean((-2, -1))


class ResNet(nn.Module):

    def __init__(self, block: 'Type[Union[BasicBlock, Bottleneck]]', layers: 'List[int]', planes: 'Optional[List[int]]'=None, use_fc: 'bool'=False, num_classes: 'int'=1000, use_pooling: 'bool'=True, big_kernel: 'bool'=False, zero_init_residual: 'bool'=False):
        """
        Custom ResNet architecture, with some design differences compared to the built-in
        PyTorch ResNet.
        This implementation and its usage in predesigned_modules is derived from
        https://github.com/fiveai/on-episodes-fsl/blob/master/src/models/ResNet.py
        Args:
            block: which core block to use (BasicBlock, Bottleneck, or any child of one of these)
            layers: number of blocks in each of the 4 layers
            planes: number of planes in each of the 4 layers
            use_fc: whether to use one last linear layer on features
            num_classes: output dimension of the last linear layer (only used if use_fc is True)
            use_pooling: whether to average pool the features (must be True if use_fc is True)
            big_kernel: whether to use the shape of the built-in PyTorch ResNet designed for
                ImageNet. If False, make the first convolutional layer less destructive.
            zero_init_residual: zero-initialize the last BN in each residual branch, so that the
                residual branch starts with zeros, and each residual block behaves like an identity.
                This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        """
        super().__init__()
        if planes is None:
            planes = [64, 128, 256, 512]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=1, bias=False) if big_kernel else nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, planes[0], layers[0])
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2)
        self.use_pooling = use_pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.use_fc = use_fc
        self.fc = nn.Linear(self.inplanes, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def _make_layer(self, block: 'Type[Union[BasicBlock, Bottleneck]]', planes: 'int', blocks: 'int', stride: 'int'=1) ->nn.Module:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Forward pass through the ResNet.
        Args:
            x: input tensor of shape (batch_size, **image_shape)
        Returns:
            x: output tensor of shape (batch_size, num_classes) if self.use_fc is True,
                otherwise of shape (batch_size, **feature_shape)
        """
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x)))))))
        if self.use_pooling:
            x = torch.flatten(self.avgpool(x), 1)
            if self.use_fc:
                return self.fc(x)
        elif self.use_fc:
            raise ValueError("You can't use the fully connected layer without pooling features.")
        return x

    def set_use_fc(self, use_fc: 'bool'):
        """
        Change the use_fc property. Allow to decide when and where the model should use its last
        fully connected layer.
        Args:
            use_fc: whether to set self.use_fc to True or False
        """
        self.use_fc = use_fc


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (FEATBasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FEATResNet12,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (MultiHeadAttention,
     lambda: ([], {'n_head': 4, 'd_model': 4, 'd_k': 4, 'd_v': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
]

