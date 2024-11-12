
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


import copy


import numpy as np


import random


import torchvision.transforms as T


import torch.distributed as dist


import logging


import time


from scipy.optimize import linear_sum_assignment


import torch.nn as nn


import torch.nn.functional as F


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


class Prototypes(nn.Module):

    def __init__(self, output_dim, num_prototypes):
        super().__init__()
        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        x = F.normalize(x)
        return self.prototypes(x)


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        super().__init__()
        if num_hidden_layers > 0:
            layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
            for _ in range(num_hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = nn.Identity()

    def forward(self, x):
        return self.mlp(x)


class MultiHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_prototypes, num_heads, num_hidden_layers=1):
        super().__init__()
        self.num_heads = num_heads
        self.projectors = torch.nn.ModuleList([MLP(input_dim, hidden_dim, output_dim, num_hidden_layers) for _ in range(num_heads)])
        if num_hidden_layers > 0:
            self.prototypes = torch.nn.ModuleList([Prototypes(output_dim, num_prototypes) for _ in range(num_heads)])
        else:
            self.prototypes = torch.nn.ModuleList([Prototypes(input_dim, num_prototypes) for _ in range(num_heads)])
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        z = self.projectors[head_idx](feats)
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class SinkhornKnopp(torch.nn.Module):

    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, logits):
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        for it in range(self.num_iters):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()


class SinkhornKnoppLognormalPrior(torch.nn.Module):

    def __init__(self, temp, gauss_sd, lamb):
        super().__init__()
        self.temp = temp
        self.gauss_sd = gauss_sd
        self.lamb = lamb
        self.dist = None

    @torch.no_grad()
    def forward(self, logits):
        PS = torch.nn.functional.softmax(logits / self.temp, dim=1, dtype=torch.float64)
        N = PS.size(0)
        K = PS.size(1)
        _K_dist = torch.ones((K, 1), dtype=torch.float64)
        marginals_argsort = torch.argsort(PS.sum(0))
        if self.dist is None:
            _K_dist = torch.distributions.log_normal.LogNormal(torch.tensor([1.0]), torch.tensor([self.gauss_sd])).sample(sample_shape=(K, 1)).reshape(-1, 1) * N / K
            _K_dist = torch.clamp(_K_dist, min=1)
            self.dist = _K_dist
        else:
            _K_dist = self.dist
        _K_dist[marginals_argsort] = torch.sort(_K_dist)[0]
        beta = torch.ones((N, 1), dtype=torch.float64) / N
        PS.pow_(0.5 * self.lamb)
        r = 1.0 / _K_dist
        r /= r.sum()
        c = 1.0 / N
        err = 1000000.0
        _counter = 0
        ones = torch.ones(N, dtype=torch.float64)
        while err > 0.1 and _counter < 2000:
            alpha = r / torch.matmul(beta.t(), PS).t()
            beta_new = c / torch.matmul(PS, alpha)
            if _counter % 10 == 0:
                err = torch.sum(torch.abs(beta.squeeze() / beta_new.squeeze() - ones)).cpu().item()
            beta = beta_new
            _counter += 1
        torch.mul(PS, beta, out=PS)
        torch.mul(alpha.t(), PS, out=PS)
        PS = PS / torch.sum(PS, dim=1, keepdim=True)
        PS = PS
        return PS


class DiscoveryClassifier(nn.Module):

    def __init__(self, num_labeled, num_unlabeled, feat_dim, hidden_dim, proj_dim, num_views, memory_batches, items_per_batch, memory_patience, num_iters_sk, epsilon_sk, temperature, batch_size, sk_mode='classical', gauss_sd_sk_new=0.5, lamb_sk_new=20, num_hidden_layers=1):
        super().__init__()
        self.head_lab = nn.Linear(feat_dim, num_labeled)
        self.head_unlab = MultiHead(input_dim=feat_dim, hidden_dim=hidden_dim, output_dim=proj_dim, num_prototypes=num_unlabeled, num_heads=1, num_hidden_layers=num_hidden_layers)
        self.num_views = num_views
        self.feat_dim = feat_dim
        self.num_labeled = num_labeled
        self.num_unlabeled = num_unlabeled
        self.num_heads = 1
        self.num_hidden_layers = 1
        self.memory_batches = memory_batches
        self.items_per_batch = items_per_batch
        self.memory_batches = memory_batches
        self.memory_patience = memory_patience
        self.num_iters_sk = num_iters_sk
        self.epsilon_sk = epsilon_sk
        self.temperature = temperature
        self.batch_size = batch_size
        if sk_mode == 'classical':
            self.sk = SinkhornKnopp(num_iters=self.num_iters_sk, epsilon=self.epsilon_sk)
        else:
            self.sk = SinkhornKnoppLognormalPrior(temp=temperature, gauss_sd=gauss_sd_sk_new, lamb=lamb_sk_new)
        self.memory_size = self.memory_batches * self.items_per_batch * self.batch_size
        self.memory_last_idx = torch.zeros(self.num_views).long()
        self.register_buffer('memory_feat', torch.empty((self.num_views, self.memory_size, self.feat_dim)))

    def update_memory(self, view_num, features):
        _n = features.shape[0]
        features = features.detach()
        last_idx = self.memory_last_idx[view_num]
        if last_idx + _n <= self.memory_size:
            self.memory_feat[view_num][last_idx:last_idx + _n] = features
        else:
            _n1 = self.memory_size - last_idx
            _n2 = _n - _n1
            self.memory_feat[view_num][last_idx:] = features[:_n1]
            self.memory_feat[view_num][:_n2] = features[_n1:]
        self.memory_last_idx[view_num] = (self.memory_last_idx[view_num] + _n) % self.memory_size

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds, dim=-1)
        return -torch.mean(torch.sum(targets * preds, dim=-1))

    def get_swapped_prediction_loss(self, logits, targets):
        loss = 0
        for view in range(self.num_views):
            for other_view in np.delete(range(self.num_views), view):
                loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / (self.num_views * (self.num_views - 1))

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_unlab.normalize_prototypes()

    def forward_knowns_bg_head_single_view(self, x):
        self.normalize_prototypes()
        logits = self.head_lab(x)
        return logits

    def forward_heads_single_view(self, x):
        """Note: does not support multi-head scenario (uses only the first head)."""
        self.normalize_prototypes()
        logits_knowns = self.head_lab(x)
        logits_knowns = logits_knowns[None, :, :]
        logits_novels = self.head_unlab(x)[0] / self.temperature
        logits_full = torch.cat([logits_knowns, logits_novels], dim=-1)
        logits_full = logits_full[0]
        return logits_full

    def forward_heads(self, feats):
        logits_lab = self.head_lab(feats)
        logits_unlab, _ = self.head_unlab(feats)
        logits_unlab /= self.temperature
        out = {'logits_lab': logits_lab, 'logits_unlab': logits_unlab}
        return out

    def forward_classifier(self, feats):
        out = [self.forward_heads(f) for f in feats]
        out_dict = {'feats': torch.stack(feats)}
        for key in out[0].keys():
            out_dict[key] = torch.stack([o[key] for o in out])
        return out_dict

    def forward(self, views):
        outputs = self.forward_classifier(views)
        outputs['logits_lab'] = outputs['logits_lab'].unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        logits = torch.cat([outputs['logits_lab'], outputs['logits_unlab']], dim=-1)
        targets = torch.zeros_like(logits)
        if self.memory_patience > 0:
            self.memory_patience -= 1
            for v in range(self.num_views):
                self.update_memory(v, outputs['feats'][v])
            loss_cluster = torch.zeros(1)[0]
        else:
            _batch_size = logits.shape[2]
            for v in range(self.num_views):
                for h in range(self.num_heads):
                    logits_sk = logits[v, h]
                    mem_feat = self.memory_feat[v]
                    mem_logits_lab = self.head_lab(mem_feat)
                    mem_logits_unlab, _ = self.head_unlab.forward_head(h, mem_feat)
                    mem_logits_unlab /= self.temperature
                    mem_logits_full = torch.cat([mem_logits_lab, mem_logits_unlab], dim=1)
                    logits_sk = torch.cat([logits_sk, mem_logits_full], dim=0)
                    logits_sk *= self.temperature
                    targets_sk = self.sk(logits_sk).type_as(targets)
                    targets_sk = targets_sk[:_batch_size]
                    targets[v, h] = targets_sk
                self.update_memory(v, outputs['feats'][v])
            loss_cluster = self.get_swapped_prediction_loss(logits, targets)
        losses = {'loss': loss_cluster, 'loss_cluster': loss_cluster}
        return losses


class FeatureExtractionROIHeadsWrapper(nn.Module):

    def __init__(self, roi_heads):
        super().__init__()
        self.roi_heads = roi_heads

    def forward(self, images: 'ImageList', features: 'Dict[str, torch.Tensor]', proposals: 'List[Instances]', targets: 'Optional[List[Instances]]'=None):
        """Based on the parent's `forward()`. All arguments are kept for compatibility.

        Returns: a Tensor of shape (M, channel_dim * output_size * output_size), i.e. (M, 256 * 7 * 7),
        where M is the total number of proposals aggregated over all N batch images.
        See: https://github.com/facebookresearch/detectron2/blob/a24729abd7a08aa29b453e1754779004807fc8ee/detectron2/modeling/poolers.py#L195
        """
        if isinstance(self.roi_heads, CascadeROIHeads):
            features = [features[f] for f in self.roi_heads.box_in_features]
            prev_pred_boxes = None
            image_sizes = [x.image_size for x in proposals]
            for k in range(self.roi_heads.num_cascade_stages):
                if k > 0:
                    proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                if k < self.roi_heads.num_cascade_stages - 1:
                    predictions = self._run_stage(features, proposals, k)
                    prev_pred_boxes = self.roi_heads.box_predictor[k].predict_boxes(predictions, proposals)
                else:
                    box_features = self.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
                    box_features = self.roi_heads.box_head[k](box_features)
        elif isinstance(self.roi_heads, StandardROIHeads):
            features = [features[f] for f in self.roi_heads.box_in_features]
            box_features = self.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = self.roi_heads.box_head(box_features)
        else:
            raise ValueError('self.roi_heads must be an instance of StandardROIHeads or CascadeROIHeads')
        return box_features

    def forward_with_given_boxes(self, features: 'Dict[str, torch.Tensor]', instances: 'List[Instances]'):
        return self.roi_heads.forward_with_given_boxes(features, instances)

    def _forward_box(self, features: 'Dict[str, torch.Tensor]', proposals: 'List[Instances]'):
        return self.roi_heads._forward_box(features, proposals)

    def _forward_mask(self, features: 'Dict[str, torch.Tensor]', instances: 'List[Instances]'):
        return self.roi_heads._forward_box(features, instances)

    def _forward_keypoint(self, features: 'Dict[str, torch.Tensor]', instances: 'List[Instances]'):
        return self.roi_heads._forward_keypoint(features, instances)

    def _match_and_label_boxes(self, proposals, stage, targets):
        return self.roi_heads._match_and_label_boxes(proposals, stage, targets)

    def _run_stage(self, features, proposals, stage):
        return self.roi_heads._run_stage(features, proposals, stage)

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        return self.roi_heads._create_proposals_from_boxes(boxes, image_sizes)


class LVISPredictionsLoader(nn.Module):
    """
    A class that is used as a dummy model that loads pre-generated predictions from the discovery phase and then
    returns them per given image when `forward()` is called.
    """

    def __init__(self, file_path):
        """Loads pre-generated predictions grouped by "image_id".

        Args:
            file_path: path to the pre-generated predictions file. The file contains a dict with keys corresponding
            to "image_id"s and values being list[dict], each dict with the keys: "image_id", "category_id",
            "bbox", "score", "segmentation" (matching the output format of ``instances_to_coco_json()``).
        """
        super().__init__()
        with open(file_path, 'r') as f:
            self.image_predictions = json.loads(f.read())
        self.dummy_parameter = nn.Parameter(torch.zeros(1))

    def forward(self, batched_inputs: 'List[Dict[str, torch.Tensor]]'):
        """Returns pre-generated predictions for the given images.

        Args: same as ``GeneralizedRCNN.forward()``.
        Returns:
            list[dict]: each dict is the output for one input image. It matches the output format of
                        ``instances_to_coco_json()``.
        """
        predictions = []
        for inp in batched_inputs:
            im_id = inp['image_id']
            instances = self.image_predictions[str(im_id)]
            for inst in instances:
                inst['category_id'] -= 1
            predictions.append({'image_id': im_id, 'instances': instances})
        return predictions


class ForwardMode:
    SUPERVISED_TRAIN = 0
    SUPERVISED_INFERENCE = 1
    PROPOSALS_EXTRACTION = 2
    DISCOVERY_FEATURE_EXTRACTION = 3
    DISCOVERY_CLASSIFIER = 4
    DISCOVERY_GT_CLASS_PREDICTIONS_EXTRACTION = 5
    DISCOVERY_INFERENCE = 6


def init_discovery_feature_extractor_model(model_supervised):
    model_discovery_feature_extractor = FeatureExtractionRCNN(backbone=model_supervised.backbone, proposal_generator=None, roi_heads=FeatureExtractionROIHeadsWrapper(roi_heads=model_supervised.roi_heads), pixel_mean=model_supervised.pixel_mean, pixel_std=model_supervised.pixel_std, input_format=model_supervised.input_format)
    return model_discovery_feature_extractor


def init_discovery_gt_prediction_extractor_model(model_supervised):
    model_gt_extraction = GTPredictionExtractionRCNN(backbone=model_supervised.backbone, proposal_generator=None, roi_heads=model_supervised.roi_heads, pixel_mean=model_supervised.pixel_mean, pixel_std=model_supervised.pixel_std, input_format=model_supervised.input_format)
    return model_gt_extraction


def extract_class_agnostic_proposals(boxes, image_shapes, proposals_logits, min_box_size, nms_thresh, topk_per_image):
    results_per_image = []
    for boxes_per_image, image_shape, scores_per_image in zip(boxes, image_shapes, proposals_logits):
        valid_mask = torch.isfinite(boxes_per_image).all(dim=1)
        if not valid_mask.all():
            boxes_per_image = boxes_per_image[valid_mask]
            scores_per_image = scores_per_image[valid_mask]
        boxes_per_image = Boxes(boxes_per_image.reshape(-1, 4))
        boxes_per_image.clip(image_shape)
        keep = boxes_per_image.nonempty(threshold=min_box_size - 1e-08)
        boxes_per_image, scores_per_image = boxes_per_image[keep], scores_per_image[keep]
        boxes_per_image = boxes_per_image.tensor.view(-1, 4)
        keep = batched_nms(boxes_per_image, scores_per_image, torch.zeros(boxes_per_image.shape[0]), nms_thresh)
        keep = keep[:topk_per_image]
        boxes_per_image, scores_per_image = boxes_per_image[keep], scores_per_image[keep]
        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes_per_image)
        result.scores = scores_per_image
        result.pred_classes = -1 * result.scores.new_ones(result.scores.shape)
        results_per_image.append(result)
    return results_per_image


class ClassAgnosticFastRCNNDiscoveryOutputLayersWrapper(nn.Module):

    def __init__(self, box_predictor, min_box_size=0, test_nms_thresh=0.5, test_topk_per_image=50):
        super().__init__()
        self.box_predictor = box_predictor
        self.min_box_size = min_box_size
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

    def inference(self, predictions: 'Tuple[torch.Tensor, torch.Tensor]', proposals: 'List[Instances]'):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            - list[Instances]: list of localization-adjusted predictions with the following properties:
                * `pred_boxes`: Boxes object of shape (Ni, 4) with bbox coordinates in XYXY_ABS format;
                * `scores`: RPN objectness logits
                * `pred_classes`: 1-D Tensor of -1s (kept for compatibility)
            - None: placeholder for compatibility.
        """
        boxes = self.box_predictor.predict_boxes(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        proposals_logits = [proposals_per_image.objectness_logits for proposals_per_image in proposals]
        results_per_image = extract_class_agnostic_proposals(boxes, image_shapes, proposals_logits, self.min_box_size, self.test_nms_thresh, self.test_topk_per_image)
        return results_per_image, None

    def forward(self, x):
        return self.box_predictor.forward(x)

    def losses(self, predictions, proposals):
        return self.box_predictor.losses(predictions, proposals)

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        return self.box_predictor.box_reg_loss(proposal_boxes, gt_boxes, pred_deltas, gt_classes)

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        return self.box_predictor.predict_boxes_for_gt_classes(predictions, proposals)

    def predict_boxes(self, predictions, proposals):
        return self.box_predictor.predict_boxes(predictions, proposals)

    def predict_probs(self, predictions, proposals):
        return self.box_predictor.predict_probs(predictions, proposals)


def init_proposals_extraction_model(model_supervised, cfg):
    roi_heads = StandardROIHeads(num_classes=model_supervised.roi_heads.num_classes, batch_size_per_image=model_supervised.roi_heads.batch_size_per_image, positive_fraction=model_supervised.roi_heads.positive_fraction, proposal_matcher=model_supervised.roi_heads.proposal_matcher, box_in_features=model_supervised.roi_heads.box_in_features, box_pooler=model_supervised.roi_heads.box_pooler, box_head=model_supervised.roi_heads.box_head, box_predictor=ClassAgnosticFastRCNNDiscoveryOutputLayersWrapper(box_predictor=model_supervised.roi_heads.box_predictor, min_box_size=cfg.model_proposals_extraction_param.min_box_size, test_nms_thresh=cfg.model_proposals_extraction_param.test_nms_thresh, test_topk_per_image=cfg.model_proposals_extraction_param.test_topk_per_image), mask_in_features=model_supervised.roi_heads.mask_in_features, mask_pooler=model_supervised.roi_heads.mask_pooler, mask_head=model_supervised.roi_heads.mask_head)
    model_proposals_extraction = GeneralizedRCNN(backbone=model_supervised.backbone, proposal_generator=model_supervised.proposal_generator, roi_heads=roi_heads, pixel_mean=model_supervised.pixel_mean, pixel_std=model_supervised.pixel_std, input_format=model_supervised.input_format)
    return model_proposals_extraction


class DiscoveryRCNN(nn.Module):

    def __init__(self, *, supervised_rcnn, cfg):
        super().__init__()
        self.supervised_rcnn = supervised_rcnn
        self.remove_bkg_class_from_discovery_model()
        proposals_extractor_rcnn = init_proposals_extraction_model(supervised_rcnn, cfg)
        discovery_feature_extractor = init_discovery_feature_extractor_model(supervised_rcnn)
        discovery_gt_class_predictions_extractor = init_discovery_gt_prediction_extractor_model(supervised_rcnn)
        self.proposals_extractor_rcnn = proposals_extractor_rcnn
        self.discovery_feature_extractor = discovery_feature_extractor
        self.discovery_gt_class_predictions_extractor = discovery_gt_class_predictions_extractor
        self.eval_knowns_param = cfg.eval_knowns_param
        self.eval_all_param = cfg.eval_all_param
        self.num_known_classes = self.supervised_rcnn.roi_heads.num_classes
        self._default_forward_mode = None
        self.discovery_nms_thresh = cfg.model_proposals_extraction_param.test_nms_thresh

    def forward(self, x, mode=None):
        if mode is None:
            if self._default_forward_mode:
                mode = self._default_forward_mode
            else:
                raise ValueError('Forward mode must be specified or the default one must be set')
        if mode == ForwardMode.SUPERVISED_TRAIN:
            return self._forward_supervised_train(x)
        elif mode == ForwardMode.SUPERVISED_INFERENCE:
            return self._forward_supervised_inference(x)
        elif mode == ForwardMode.PROPOSALS_EXTRACTION:
            return self._forward_proposals_extractor(x)
        elif mode == ForwardMode.DISCOVERY_FEATURE_EXTRACTION:
            return self._forward_discovery_feature_extractor(x)
        elif mode == ForwardMode.DISCOVERY_CLASSIFIER:
            return self._forward_discovery_classifier(x)
        elif mode == ForwardMode.DISCOVERY_GT_CLASS_PREDICTIONS_EXTRACTION:
            return self._forward_discovery_gt_class_predictions_extractor(x)
        elif mode == ForwardMode.DISCOVERY_INFERENCE:
            return self._forward_discovery_inference(x)
        else:
            raise ValueError(f'Unknown forward mode: {mode}')

    def _forward_supervised_train(self, batched_inputs):
        self.supervised_rcnn.train()
        if not isinstance(self.supervised_rcnn.roi_heads.box_predictor, nn.ModuleList):
            self.supervised_rcnn.proposal_generator.nms_thresh = 0.7
        else:
            self.supervised_rcnn.proposal_generator.nms_thresh_train = 0.9
            self.supervised_rcnn.proposal_generator.nms_thresh_test = 0.9
        return self.supervised_rcnn(batched_inputs)

    def _forward_supervised_inference(self, batched_inputs):
        self.supervised_rcnn.eval()
        if not isinstance(self.supervised_rcnn.roi_heads.box_predictor, nn.ModuleList):
            self.supervised_rcnn.proposal_generator.nms_thresh = 0.7
        else:
            self.supervised_rcnn.proposal_generator.nms_thresh_train = 0.9
            self.supervised_rcnn.proposal_generator.nms_thresh_test = 0.9
        if isinstance(self.supervised_rcnn.roi_heads.box_predictor, nn.ModuleList):
            box_predictors = self.supervised_rcnn.roi_heads.box_predictor
        else:
            box_predictors = [self.supervised_rcnn.roi_heads.box_predictor]
        for box_predictor in box_predictors:
            box_predictor.allow_novel_classes_during_inference = False
            box_predictor.test_topk_per_image = self.eval_knowns_param['test_topk_per_image']
            box_predictor.test_score_thresh = self.eval_knowns_param['test_score_thresh']
        return self.supervised_rcnn(batched_inputs)

    def _forward_proposals_extractor(self, batched_inputs):
        self.proposals_extractor_rcnn.eval()
        if not isinstance(self.supervised_rcnn.roi_heads.box_predictor, nn.ModuleList):
            self.proposals_extractor_rcnn.proposal_generator.nms_thresh = self.discovery_nms_thresh
        else:
            self.supervised_rcnn.proposal_generator.nms_thresh_train = self.discovery_nms_thresh
            self.supervised_rcnn.proposal_generator.nms_thresh_test = self.discovery_nms_thresh
        results = self.proposals_extractor_rcnn(batched_inputs)
        return results

    def _forward_discovery_feature_extractor(self, batched_inputs):
        self.discovery_feature_extractor.eval()
        return self.discovery_feature_extractor(batched_inputs)

    def _forward_discovery_classifier(self, features):
        box_predictor = self.supervised_rcnn.roi_heads.box_predictor
        if isinstance(box_predictor, nn.ModuleList):
            box_predictor = box_predictor[0]
        box_predictor.discovery_model.train()
        return box_predictor.discovery_model(features)

    def _forward_discovery_gt_class_predictions_extractor(self, batched_inputs):
        self.discovery_gt_class_predictions_extractor.eval()
        if isinstance(self.discovery_gt_class_predictions_extractor.roi_heads.box_predictor, nn.ModuleList):
            box_predictors = self.discovery_gt_class_predictions_extractor.roi_heads.box_predictor
        else:
            box_predictors = [self.discovery_gt_class_predictions_extractor.roi_heads.box_predictor]
        for box_predictor in box_predictors:
            box_predictor.allow_novel_classes_during_inference = True
        results = self.discovery_gt_class_predictions_extractor(batched_inputs)
        return results

    def _forward_discovery_inference(self, batched_inputs):
        self.supervised_rcnn.eval()
        if not isinstance(self.supervised_rcnn.roi_heads.box_predictor, nn.ModuleList):
            self.supervised_rcnn.proposal_generator.nms_thresh = 0.9
        else:
            self.supervised_rcnn.proposal_generator.nms_thresh_train = 0.9
            self.supervised_rcnn.proposal_generator.nms_thresh_test = 0.9
        if isinstance(self.supervised_rcnn.roi_heads.box_predictor, nn.ModuleList):
            box_predictors = self.supervised_rcnn.roi_heads.box_predictor
        else:
            box_predictors = [self.supervised_rcnn.roi_heads.box_predictor]
        for box_predictor in box_predictors:
            box_predictor.allow_novel_classes_during_inference = True
            box_predictor.test_topk_per_image = self.eval_all_param['test_topk_per_image']
            box_predictor.test_score_thresh = self.eval_all_param['test_score_thresh']
        results = self.supervised_rcnn(batched_inputs)
        return results

    def is_discovery_network_memory_filled(self):
        box_predictor = self.supervised_rcnn.roi_heads.box_predictor
        if isinstance(box_predictor, nn.ModuleList):
            box_predictor = box_predictor[0]
        is_memory_filled = box_predictor.discovery_model.memory_patience == 0
        return is_memory_filled

    def remove_bkg_class_from_discovery_model(self):
        if isinstance(self.supervised_rcnn.roi_heads.box_predictor, nn.ModuleList):
            box_predictors = self.supervised_rcnn.roi_heads.box_predictor
        else:
            box_predictors = [self.supervised_rcnn.roi_heads.box_predictor]
        with torch.no_grad():
            for box_predictor in box_predictors:
                box_predictor.discovery_model.head_lab.weight = nn.Parameter(box_predictor.discovery_model.head_lab.weight[:-1])
                box_predictor.discovery_model.head_lab.bias = nn.Parameter(box_predictor.discovery_model.head_lab.bias[:-1])

    def set_default_forward_mode(self, mode):
        self._default_forward_mode = mode

    def remove_default_forward_mode(self):
        self._default_forward_mode = None


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MultiHead,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_prototypes': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Prototypes,
     lambda: ([], {'output_dim': 4, 'num_prototypes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SinkhornKnopp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {})),
    (SinkhornKnoppLognormalPrior,
     lambda: ([], {'temp': 4, 'gauss_sd': 4, 'lamb': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

