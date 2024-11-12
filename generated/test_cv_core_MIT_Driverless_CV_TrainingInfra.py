
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


import random


import time


import copy


import torch


import torch.nn as nn


from torch.utils.data import DataLoader


import torchvision


import warnings


import numpy as np


from torch.utils.cpp_extension import CUDAExtension


import math


import torch.utils.data


import torch.nn.functional as F


from torchvision import transforms


from torch import nn


from torch.autograd import Variable


from torch.utils.data import Dataset


import torch.nn.functional


from torch.backends import cudnn


from torch import optim


import pandas as pd


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes.
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-12)
    return iou


def build_targets(target, anchors, num_anchors, num_classes, grid_size_h, grid_size_w, ignore_thres):
    n_b = target.size(0)
    n_a = num_anchors
    n_c = num_classes
    n_g_h = grid_size_h
    n_g_w = grid_size_w
    mask = torch.zeros(n_b, n_a, n_g_h, n_g_w, dtype=torch.uint8, device=target.device)
    conf_mask = torch.ones(n_b, n_a, n_g_h, n_g_w, dtype=torch.uint8, device=target.device)
    tx = torch.zeros(n_b, n_a, n_g_h, n_g_w, dtype=torch.float, device=target.device)
    ty = torch.zeros(n_b, n_a, n_g_h, n_g_w, dtype=torch.float, device=target.device)
    tw = torch.zeros(n_b, n_a, n_g_h, n_g_w, dtype=torch.float, device=target.device)
    th = torch.zeros(n_b, n_a, n_g_h, n_g_w, dtype=torch.float, device=target.device)
    tconf = torch.zeros(n_b, n_a, n_g_h, n_g_w, dtype=torch.float, device=target.device)
    tcls = torch.zeros(n_b, n_a, n_g_h, n_g_w, n_c, dtype=torch.uint8, device=target.device)
    master_mask = torch.sum(target, dim=2) > 0
    gx = target[:, :, 1] * n_g_w
    gy = target[:, :, 2] * n_g_h
    gw = target[:, :, 3] * n_g_w
    gh = target[:, :, 4] * n_g_h
    gi = gx.long()
    gj = gy.long()
    gi[~master_mask] = gi[:, 0].unsqueeze(1).expand(*gi.shape)[~master_mask]
    gj[~master_mask] = gj[:, 0].unsqueeze(1).expand(*gj.shape)[~master_mask]
    gx[~master_mask] = gx[:, 0].unsqueeze(1).expand(*gx.shape)[~master_mask]
    gy[~master_mask] = gy[:, 0].unsqueeze(1).expand(*gy.shape)[~master_mask]
    gw[~master_mask] = gw[:, 0].unsqueeze(1).expand(*gw.shape)[~master_mask]
    gh[~master_mask] = gh[:, 0].unsqueeze(1).expand(*gh.shape)[~master_mask]
    a = torch.zeros((target.shape[0], target.shape[1], 2), dtype=torch.float, device=target.device)
    b = torch.unsqueeze(gw, -1)
    c = torch.unsqueeze(gh, -1)
    gt_box = torch.cat((a, b, c), dim=2)
    anchor_shapes = torch.cat((torch.zeros((anchors.shape[0], 2), device=target.device, dtype=torch.float), anchors), 1)
    gt_box_1 = torch.unsqueeze(gt_box, 2).expand(-1, -1, anchor_shapes.shape[0], -1)
    anchor_shapes_1 = anchor_shapes.view(1, 1, anchor_shapes.shape[0], anchor_shapes.shape[1]).expand(*gt_box_1.shape)
    anch_ious = bbox_iou(gt_box_1, anchor_shapes_1).permute(0, 2, 1)
    gj_mask = gj.unsqueeze(1).expand(-1, num_anchors, -1)[anch_ious > ignore_thres]
    gi_mask = gi.unsqueeze(1).expand(-1, num_anchors, -1)[anch_ious > ignore_thres]
    conf_mask[:, :, gj_mask, gi_mask] = 0
    best_n = torch.argmax(anch_ious, dim=1)
    img_dim = torch.arange(0, n_b, device=target.device).view(-1, 1).expand(*best_n.shape)
    mask[img_dim, best_n, gj, gi] = 1
    conf_mask[img_dim, best_n, gj, gi] = 1
    tx[img_dim, best_n, gj, gi] = gx - gi.float()
    ty[img_dim, best_n, gj, gi] = gy - gj.float()
    tw[img_dim, best_n, gj, gi] = torch.log(gw / anchors[best_n, 0] + 1e-16)
    th[img_dim, best_n, gj, gi] = torch.log(gh / anchors[best_n, 1] + 1e-16)
    target_label = target[:, :, 0].long()
    tcls[img_dim, best_n, gj, gi, target_label] = 1
    tconf[img_dim, best_n, gj, gi] = 1
    return mask, conf_mask, tx, ty, tw, th, tconf, tcls


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_height, img_width, build_targets_ignore_thresh, conv_activation, xy_loss, wh_loss, object_loss, no_object_loss):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_height = img_height
        self.image_width = img_width
        self.ignore_thres = build_targets_ignore_thresh
        self.xy_loss = xy_loss
        self.wh_loss = wh_loss
        self.no_object_loss = no_object_loss
        self.object_loss = object_loss
        self.conv_activation = conv_activation
        self.mse_loss = nn.MSELoss(size_average=True)
        self.bce_loss = nn.BCELoss(size_average=True)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, sample, targets=None):
        nA = self.num_anchors
        nB = sample.size(0)
        nGh = sample.size(2)
        nGw = sample.size(3)
        stride = self.image_height / nGh
        prediction = sample.view(nB, nA, self.bbox_attrs, nGh, nGw).permute(0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        grid_x = torch.arange(nGw, dtype=torch.float, device=x.device).repeat(nGh, 1).view([1, 1, nGh, nGw])
        grid_y = torch.arange(nGh, dtype=torch.float, device=x.device).repeat(nGw, 1).t().view([1, 1, nGh, nGw]).contiguous()
        scaled_anchors = torch.tensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors], dtype=torch.float, device=x.device)
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))
        pred_boxes = torch.zeros(prediction[..., :4].shape, dtype=torch.float, device=x.device)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        if targets is not None:
            self.mse_loss = self.mse_loss
            self.bce_loss = self.bce_loss
            self.ce_loss = self.ce_loss
            mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(target=targets, anchors=scaled_anchors, num_anchors=nA, num_classes=self.num_classes, grid_size_h=nGh, grid_size_w=nGw, ignore_thres=self.ignore_thres)
            tx.requires_grad_(False)
            ty.requires_grad_(False)
            tw.requires_grad_(False)
            th.requires_grad_(False)
            tconf.requires_grad_(False)
            tcls.requires_grad_(False)
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask
            loss_x = self.xy_loss * self.mse_loss(x[mask], tx[mask])
            loss_y = self.xy_loss * self.mse_loss(y[mask], ty[mask])
            loss_w = self.wh_loss * self.mse_loss(w[mask], tw[mask])
            loss_h = self.wh_loss * self.mse_loss(h[mask], th[mask])
            loss_cls_constant = 0
            loss_cls = loss_cls_constant * (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss_noobj = self.no_object_loss * self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false])
            loss_obj = self.object_loss * self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
            loss = loss_x + loss_y + loss_w + loss_h + loss_noobj + loss_obj + loss_cls
            return loss, torch.tensor((loss_x, loss_y, loss_w, loss_h, loss_obj, loss_noobj), device=targets.device)
        else:
            output = torch.cat((pred_boxes.view(nB, -1, 4) * stride, pred_conf.view(nB, -1, 1), pred_cls.view(nB, -1, self.num_classes)), -1)
            return output


vanilla_anchor_list = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]


def create_modules(module_defs, xy_loss, wh_loss, no_object_loss, object_loss, vanilla_anchor):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    img_width = int(hyperparams['width'])
    img_height = int(hyperparams['height'])
    onnx_height = int(hyperparams['onnx_height'])
    num_classes = int(hyperparams['classes'])
    leaky_slope = float(hyperparams['leaky_slope'])
    conv_activation = hyperparams['conv_activation']
    yolo_masks = [[int(y) for y in x.split(',')] for x in hyperparams['yolo_masks'].split('|')]
    csv_uri = hyperparams['train_uri']
    training_csv_tempfile = csv_uri
    with open(training_csv_tempfile) as f:
        csv_reader = csv.reader(f)
        row = next(csv_reader)
        row = str(row)[2:-2]
        anchor_list = [[float(y) for y in x.split(',')] for x in row.split("'")[0].split('|')]
    if vanilla_anchor:
        anchor_list = vanilla_anchor_list
    build_targets_ignore_thresh = float(hyperparams['build_targets_ignore_thresh'])
    module_list = nn.ModuleList()
    yolo_count = 0
    act_flag = 1
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def['type'] == 'convolutional':
            bn = 1
            if module_def['filters'] == 'preyolo':
                filters = (num_classes + 5) * len(yolo_masks[yolo_count])
                act_flag = 0
                bn = 0
            else:
                filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = int((kernel_size - 1) // 2)
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1], out_channels=filters, kernel_size=kernel_size, stride=int(module_def['stride']), padding=pad, bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if conv_activation == 'leaky' and act_flag == 1:
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(leaky_slope))
            if conv_activation == 'ReLU' and act_flag == 1:
                modules.add_module('ReLU_%d' % i, nn.ReLU())
            act_flag = 1
        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module('_debug_padding_%d' % i, padding)
            maxpool = nn.MaxPool2d(kernel_size=int(module_def['size']), stride=int(module_def['stride']), padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)
        elif module_def['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)
        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = 0
            for layer_i in layers:
                if layer_i > 0:
                    layer_i += 1
                filters += output_filters[layer_i]
            modules.add_module('route_%d' % i, EmptyLayer())
        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())
        elif module_def['type'] == 'yolo':
            anchors = [anchor_list[i] for i in yolo_masks[yolo_count]]
            yolo_layer = YOLOLayer(anchors, num_classes, img_height, img_width, build_targets_ignore_thresh, conv_activation, xy_loss, wh_loss, object_loss, no_object_loss)
            modules.add_module('yolo_%d' % i, yolo_layer)
            yolo_count += 1
        module_list.append(modules)
        output_filters.append(filters)
    return hyperparams, module_list


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    module_defs = []
    for line in lines:
        if line.startswith('['):
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, xy_loss, wh_loss, no_object_loss, object_loss, vanilla_anchor):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(module_defs=self.module_defs, xy_loss=xy_loss, wh_loss=wh_loss, no_object_loss=no_object_loss, object_loss=object_loss, vanilla_anchor=vanilla_anchor)
        self.img_width = int(self.hyperparams['width'])
        self.img_height = int(self.hyperparams['height'])
        self.onnx_height = int(self.hyperparams['onnx_height'])
        self.onnx_name = config_path.split('/')[-1].split('.')[0] + '_' + str(self.img_width) + str(self.onnx_height) + '.onnx'
        self.num_classes = int(self.hyperparams['classes'])
        if int(self.hyperparams['channels']) == 1:
            self.bw = True
        elif int(self.hyperparams['channels']) == 3:
            self.bw = False
        else:
            None
            self.bw = False
        current_month = datetime.now().strftime('%B').lower()
        current_year = str(datetime.now().year)
        self.validate_uri = self.hyperparams['validate_uri']
        self.train_uri = self.hyperparams['train_uri']
        self.num_train_images = int(self.hyperparams['num_train_images'])
        self.num_validate_images = int(self.hyperparams['num_validate_images'])
        self.conf_thresh = float(self.hyperparams['conf_thresh'])
        self.nms_thresh = float(self.hyperparams['nms_thresh'])
        self.iou_thresh = float(self.hyperparams['iou_thresh'])
        self.start_weights_dim = [int(x) for x in self.hyperparams['start_weights_dim'].split(',')]
        self.conv_activation = self.hyperparams['conv_activation']
        self.xy_loss = xy_loss
        self.wh_loss = wh_loss
        self.no_object_loss = no_object_loss
        self.object_loss = object_loss
        csv_uri = self.hyperparams['train_uri']
        training_csv_tempfile = csv_uri
        with open(training_csv_tempfile) as f:
            csv_reader = csv.reader(f)
            row = next(csv_reader)
            row = str(row)[2:-2]
            anchor_list = [[float(y) for y in x.split(',')] for x in row.split("'")[0].split('|')]
        if vanilla_anchor:
            anchor_list = vanilla_anchor_list
        self.anchors = anchor_list
        self.seen = 0
        self.header_info = torch.tensor([0, 0, 0, self.seen, 0])

    def get_start_weight_dim(self):
        return self.start_weights_dim

    def get_onnx_name(self):
        return self.onnx_name

    def get_bw(self):
        return self.bw

    def get_loss_constant(self):
        return [self.xy_loss, self.wh_loss, self.no_object_loss, self.object_loss]

    def get_conv_activation(self):
        return self.conv_activation

    def get_num_classes(self):
        return self.num_classes

    def get_anchors(self):
        return self.anchors

    def get_threshs(self):
        return self.conf_thresh, self.nms_thresh, self.iou_thresh

    def img_size(self):
        return self.img_width, self.img_height

    def get_links(self):
        return self.validate_uri, self.train_uri

    def num_images(self):
        return self.num_validate_images, self.num_train_images

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        if is_training:
            total_losses = torch.zeros(6, device=targets.device)
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                if is_training:
                    x, losses = module[0](x, targets)
                    total_losses += losses
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)
        return (sum(output), *total_losses) if is_training else torch.cat(output, 1)

    def load_weights(self, weights_path, start_weight_dim):
        fp = open(weights_path, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header_info = header
        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()
        ptr = 0
        yolo_count = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['filters'] != 'preyolo':
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                    num_w = conv_layer.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w
                elif module_def['filters'] == 'preyolo':
                    orig_dim = start_weight_dim[yolo_count]
                    yolo_count += 1
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += orig_dim
                    num_w = conv_layer.weight.numel()
                    dummyDims = [orig_dim] + list(conv_layer.weight.size()[1:])
                    dummy = torch.zeros(tuple(dummyDims))
                    conv_w = torch.from_numpy(weights[ptr:ptr + int(num_w * orig_dim / num_b)]).view_as(dummy)
                    conv_w = conv_w[0:num_b][:][:][:]
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += int(num_w * orig_dim / num_b)
                else:
                    None
                    raise Exception('The above layer has its BN or preyolo defined wrong')

    def save_weights(self, path, cutoff=-1):
        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['filters'] != 'preyolo':
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                conv_layer.weight.data.cpu().numpy().tofile(fp)
        fp.close()


class CrossRatioLoss(nn.Module):

    def __init__(self, loss_type, include_geo, geo_loss_gamma_horz, geo_loss_gamma_vert):
        super(CrossRatioLoss, self).__init__()
        self.loss_type = loss_type
        self.include_geo = include_geo
        self.geo_loss_gamma_vert = geo_loss_gamma_vert
        self.geo_loss_gamma_horz = geo_loss_gamma_horz
        None
        None

    def forward(self, heatmap, points, target_hm, target_points):
        if self.loss_type == 'l2_softargmax' or self.loss_type == 'l2_sm':
            mse_loss = (points - target_points) ** 2
            location_loss = mse_loss.sum(2).sum(1).mean()
        elif self.loss_type == 'l2_heatmap' or self.loss_type == 'l2_hm':
            mse_loss = (heatmap - target_hm) ** 2
            location_loss = mse_loss.sum(3).sum(2).sum(1).mean()
        elif self.loss_type == 'l1_softargmax' or self.loss_type == 'l1_sm':
            l1_loss = torch.abs(points - target_points)
            location_loss = l1_loss.sum(2).sum(1).mean()
        else:
            None
            sys.exit(1)
        if self.include_geo:
            v53 = F.normalize(points[:, 5] - points[:, 3], dim=1)
            v31 = F.normalize(points[:, 3] - points[:, 1], dim=1)
            vA = 1.0 - torch.tensordot(v31, v53, dims=([1], [1]))
            v10 = F.normalize(points[:, 1] - points[:, 0], dim=1)
            vB = 1.0 - torch.tensordot(v10, v31, dims=([1], [1]))
            v64 = F.normalize(points[:, 6] - points[:, 4], dim=1)
            v42 = F.normalize(points[:, 4] - points[:, 2], dim=1)
            vC = 1.0 - torch.tensordot(v64, v42, dims=([1], [1]))
            v20 = F.normalize(points[:, 2] - points[:, 0], dim=1)
            vD = 1.0 - torch.tensordot(v42, v20, dims=([1], [1]))
            h21 = F.normalize(points[:, 2] - points[:, 1], dim=1)
            h43 = F.normalize(points[:, 4] - points[:, 3], dim=1)
            hA = 1.0 - torch.tensordot(h43, h21, dims=([1], [1]))
            h65 = F.normalize(points[:, 6] - points[:, 5], dim=1)
            hB = 1.0 - torch.tensordot(h65, h43, dims=([1], [1]))
            geo_loss = self.geo_loss_gamma_horz * (hA + hB).mean() / 2 + self.geo_loss_gamma_vert * (vA + vB + vC + vD).mean() / 4
        else:
            geo_loss = torch.tensor(0)
        return location_loss, geo_loss, location_loss + geo_loss


class ResNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.shortcut_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        c1 = self.conv1(x)
        b1 = self.bn1(c1)
        act1 = self.relu1(b1)
        out = self.relu2(self.shortcut_bn(self.shortcut_conv(x)) + self.bn2(self.conv2(act1)))
        return out


class KeypointNet(nn.Module):

    def __init__(self, num_kpt=7, image_size=(80, 80), onnx_mode=False, init_weight=True):
        super(KeypointNet, self).__init__()
        net_size = 16
        self.conv = nn.Conv2d(in_channels=3, out_channels=net_size, kernel_size=7, stride=1, padding=3)
        self.bn = nn.BatchNorm2d(net_size)
        self.relu = nn.ReLU()
        self.res1 = ResNet(net_size, net_size)
        self.res2 = ResNet(net_size, net_size * 2)
        self.res3 = ResNet(net_size * 2, net_size * 4)
        self.res4 = ResNet(net_size * 4, net_size * 8)
        self.out = nn.Conv2d(in_channels=net_size * 8, out_channels=num_kpt, kernel_size=1, stride=1, padding=0)
        if init_weight:
            self._initialize_weights()
        self.image_size = image_size
        self.num_kpt = num_kpt
        self.onnx_mode = onnx_mode

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def flat_softmax(self, inp):
        flat = inp.view(-1, self.image_size[0] * self.image_size[1])
        flat = torch.nn.functional.softmax(flat, 1)
        return flat.view(-1, self.num_kpt, self.image_size[0], self.image_size[1])

    def soft_argmax(self, inp):
        values_y = torch.linspace(0, (self.image_size[0] - 1.0) / self.image_size[0], self.image_size[0], dtype=inp.dtype, device=inp.device)
        values_x = torch.linspace(0, (self.image_size[1] - 1.0) / self.image_size[1], self.image_size[1], dtype=inp.dtype, device=inp.device)
        exp_y = (inp.sum(3) * values_y).sum(-1)
        exp_x = (inp.sum(2) * values_x).sum(-1)
        return torch.stack([exp_x, exp_y], -1)

    def forward(self, x):
        act1 = self.relu(self.bn(self.conv(x)))
        act2 = self.res1(act1)
        act3 = self.res2(act2)
        act4 = self.res3(act3)
        act5 = self.res4(act4)
        hm = self.out(act5)
        if self.onnx_mode:
            return hm
        else:
            hm = self.flat_softmax(self.out(act5))
            out = self.soft_argmax(hm)
            return hm, out.view(-1, self.num_kpt, 2)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ResNet,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

