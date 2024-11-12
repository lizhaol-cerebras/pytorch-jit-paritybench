
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


import time


import logging


import copy


import random


import itertools


import torch


import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import ConcatDataset


import math


import matplotlib.pyplot as plt


import matplotlib as mpl


import scipy.io as scio


import torchvision


from collections import OrderedDict


from scipy.sparse import csr_matrix


import torch.nn.functional as F


from torch import nn


from torch.autograd import Variable


from torch.nn.parallel._functions import Scatter


from torch.nn.parallel._functions import Gather


from torch.nn.modules import Module


from torch.nn.parallel.scatter_gather import gather


from torch.nn.parallel.replicate import replicate


from torch.nn.parallel.parallel_apply import parallel_apply


import torchvision.models.resnet as resnet


import torchvision.transforms.functional as F


from collections import deque


from torchvision.ops import nms


import functools


from torch.nn import functional as F


from scipy.spatial.transform import Rotation as R


import matplotlib


import pandas


from torch.cuda.amp import autocast


from itertools import product


from time import time


import re


import numpy


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from sklearn.model_selection import PredefinedSplit


from torch.autograd import Function


from torch.nn.modules.utils import _pair


from torchvision.transforms import ColorJitter


from scipy import interpolate


import scipy.spatial.transform.rotation as R


from random import sample


from torch.distributions import VonMises


from torch.distributions.multivariate_normal import _batch_mahalanobis


from torch.distributions.multivariate_normal import _standard_normal


from torch.distributions.multivariate_normal import _batch_mv


from abc import ABCMeta


from abc import abstractmethod


from functools import partial


from numpy import result_type


import torch.cuda.comm as comm


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn.parallel.parallel_apply import get_a_var


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


from typing import Optional


from typing import Dict


from typing import Union


from collections import namedtuple


def _calc_radius_(bboxes_hw_norm, map_size=64):
    if len(bboxes_hw_norm) == 0:
        return []
    minimum_radius = map_size / 32.0
    scale_factor = map_size / 16.0
    scales = np.linalg.norm(np.stack(bboxes_hw_norm, 0) / 2, ord=2, axis=1)
    radius = (scales * scale_factor + minimum_radius).astype(np.uint8)
    return radius


def _calc_uv_radius_(scales, map_size=64):
    minimum_radius = map_size / 32.0
    scale_factor = map_size / 16.0
    radius = (scales * scale_factor + minimum_radius).astype(np.uint8)
    return radius


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description='ROMP: Monocular, One-stage, Regression of Multiple 3D People')
    parser.add_argument('--tab', type=str, default='ROMP_v1', help='additional tabs')
    parser.add_argument('--configs_yml', type=str, default='configs/v1.yml', help='settings')
    mode_group = parser.add_argument_group(title='mode options')
    mode_group.add_argument('--model_return_loss', type=bool, default=False, help='wether return loss value from the model for balanced GPU memory usage')
    mode_group.add_argument('--model_version', type=int, default=1, help='model version')
    mode_group.add_argument('--multi_person', type=bool, default=True, help='whether to make Multi-person Recovery')
    mode_group.add_argument('--new_training', type=bool, default=False, help='learning centermap only in first few iterations for stable training.')
    mode_group.add_argument('--perspective_proj', type=bool, default=False, help='whether to use perspective projection, else use orthentic projection.')
    mode_group.add_argument('--center3d_loss', type=bool, default=False, help='whether to use dynamic supervision.')
    mode_group.add_argument('--high_resolution_input', type=bool, default=False, help='whether to process the high-resolution input.')
    mode_group.add_argument('--rotation360_aug', type=bool, default=False, help='whether to augment the rotation in -180~180 degree.')
    mode_group.add_argument('--relative_depth_scale_aug', type=bool, default=False, help='whether to augment scale of image.')
    mode_group.add_argument('--relative_depth_scale_aug_ratio', type=float, default=0.25, help='the ratio of augmenting the scale of image for alleviating the depth uncertainty.')
    mode_group.add_argument('--high_resolution_folder', type=str, default='demo/mpii', help='path to high-resolution image.')
    mode_group.add_argument('--add_depth_encoding', type=bool, default=True, help='whether to add the depth encoding to the feature vector.')
    mode_group.add_argument('--old_trace_implement', type=bool, default=True, help='whether to add the depth encoding to the feature vector.')
    mode_group.add_argument('--video', type=bool, default=False, help='whether to use the video model.')
    mode_group.add_argument('--tmodel_type', type=str, default='conv3D', help='the architecture type of temporal model.')
    mode_group.add_argument('--clip_sampling_way', type=str, default='nooverlap', help='the way of sampling n frames from a video given an anchor frame id. 3 way: nooverlap, overlap')
    mode_group.add_argument('--clip_sampling_position', type=str, default='middle', help='the way of sampling n frames from a video given an anchor frame id. 3 way: start, middle, end')
    mode_group.add_argument('--clip_interval', type=int, default=1, help='The number of non-overlapping interval frames between clips while data loading, like conv slides. 1 for overlapping')
    mode_group.add_argument('--video_batch_size', type=int, default=32, help='The input frames of video inpute')
    mode_group.add_argument('--temp_clip_length', type=int, default=7, help='The input frames of video inpute')
    mode_group.add_argument('--random_temp_sample_internal', type=int, default=6, help='sampling the video clip with random interval between frames max=10, like sampling every 5 frames from the video to form a clip')
    mode_group.add_argument('--bev_distillation', type=bool, default=False, help='whether to use the BEV to distillation some prority of .')
    mode_group.add_argument('--eval_hard_seq', type=bool, default=False, help='whether to evaluate the checkpoint on hard sequence only for faster.')
    mode_group.add_argument('--image_datasets', type=str, default='agora', help='image datasets used for learning more attributes.')
    mode_group.add_argument('--learn_cam_with_fbboxes', type=bool, default=False, help='whether to learn camera parameter with full body bounding boxes.')
    mode_group.add_argument('--regressor_type', type=str, default='gru', help='transformer or mlpmixer')
    mode_group.add_argument('--with_gru', type=bool, default=True, help='transformer or mlpmixer')
    mode_group.add_argument('--dynamic_augment', type=bool, default=False, help='transformer or mlpmixer')
    mode_group.add_argument('--dynamic_augment_ratio', type=float, default=0.6, help='possibility of performing dynamic augments')
    mode_group.add_argument('--dynamic_changing_ratio', type=float, default=0.6, help='ratio of dynamic change / cropping width')
    mode_group.add_argument('--dynamic_aug_tracking_ratio', type=float, default=0.5, help='ratio of dynamic augments via tracking a single target')
    mode_group.add_argument('--learn_foot_contact', type=bool, default=True, help='transformer or mlpmixer')
    mode_group.add_argument('--learn_motion_offset3D', type=bool, default=True, help='transformer or mlpmixer')
    mode_group.add_argument('--learn_cam_init', type=bool, default=False, help='transformer or mlpmixer')
    mode_group.add_argument('--more_param_head_layer', type=bool, default=False, help='transformer or mlpmixer')
    mode_group.add_argument('--compute_verts_org', type=bool, default=False)
    mode_group.add_argument('--debug_tracking', type=bool, default=False)
    mode_group.add_argument('--tracking_target_max_num', type=int, default=100)
    mode_group.add_argument('--video_show_results', type=bool, default=True)
    mode_group.add_argument('--joint_num', type=int, default=44, help='44 for smpl, 73 for smplx')
    mode_group.add_argument('--render_option_path', type=str, default=os.path.join(source_dir, 'lib', 'visualization', 'vis_cfgs', 'render_options.json'), help='default rendering preference for Open3D')
    mode_group.add_argument('--using_motion_offsets_tracking', type=bool, default=True)
    mode_group.add_argument('--tracking_with_kalman_filter', type=bool, default=False)
    mode_group.add_argument('--use_optical_flow', type=bool, default=False)
    mode_group.add_argument('--raft_model_path', type=str, default=os.path.join(trained_model_dir, 'raft-things.pth'))
    mode_group.add_argument('--CGRU_temp_prop', type=bool, default=True)
    mode_group.add_argument('--learn_temp_cam_consist', type=bool, default=False)
    mode_group.add_argument('--learn_temp_globalrot_consist', type=bool, default=False)
    mode_group.add_argument('--learn_image', type=bool, default=True, help='whether to learn from image at the same time.')
    mode_group.add_argument('--image_repeat_time', type=int, default=2, help='whether to learn from image at the same time.')
    mode_group.add_argument('--drop_first_frame_loss', type=bool, default=True, help='drop the loss of the first frame to facilitate more stable loss learning.')
    mode_group.add_argument('--left_first_frame_num', type=int, default=2, help='drop the loss of the first frame to facilitate more stable loss learning.')
    mode_group.add_argument('--learn_dense_correspondence', type=bool, default=False, help='whether to learn the dense correspondence between image pixel and IUV map (from densepose).')
    mode_group.add_argument('--learnbev2adjustZ', type=bool, default=False, help='whether to fix the wrong cam_offset_bev adjustment from X to Z.')
    mode_group.add_argument('--image_loading_mode', type=str, default='image_relative', help='The Base Class (image, image_relative) used for loading image datasets.')
    mode_group.add_argument('--video_loading_mode', type=str, default='video_relative', help='The Base Class (image, image_relative, video_relative) used for loading video datasets.')
    mode_group.add_argument('--temp_upsampling_layer', type=str, default='trilinear', help='the way of upsampling in decoder of Trajectory3D model. 2 way: trilinear, deconv')
    mode_group.add_argument('--temp_transfomer_layer', type=int, default=3, help='the number of transfomer layers. 2, 3, 4, 5, 6')
    mode_group.add_argument('--calc_smpl_mesh', type=bool, default=True, help='whether to calculate smpl mesh during inference.')
    mode_group.add_argument('--calc_mesh_loss', type=bool, default=True, help='whether to calculate smpl mesh during inference.')
    mode_group.add_argument('--eval_video', type=bool, default=False, help='whether to evaluate on video benchmark.')
    mode_group.add_argument('--mp_tracker', type=str, default='byte', help='Which tracker is employed to retrieve the 3D trjectory of multiple detected person.')
    mode_group.add_argument('--inference_video', type=bool, default=False, help='run in inference mode.')
    mode_group.add_argument('--old_temp_model', type=bool, default=False, help='run in inference mode.')
    mode_group.add_argument('--evaluation_gpu', type=int, default=1, help='the gpu device used for evaluating the temporal model, better not using 0 to avoid out of memory during evaluation.')
    mode_group.add_argument('--deform_motion', type=bool, default=False, help='run in inference mode.')
    mode_group.add_argument('--temp_simulator', type=bool, default=False, help='use a simulator to simulate the temporal feature.')
    mode_group.add_argument('--tmodel_version', type=int, default=1, help='the version ID of temporal model.')
    mode_group.add_argument('--separate_smil_betas', type=bool, default=False, help='estimating individual beta for smil baby model.')
    mode_group.add_argument('--no_evaluation', type=bool, default=False, help='focus on training.')
    mode_group.add_argument('--temp_clip_length_eval', type=int, default=64, help='The temp_clip_length during evaluation')
    mode_group.add_argument('--learn_temporal_shape_consistency', type=bool, default=False, help='whether to learn the shape consistency in temporal dim.')
    mode_group.add_argument('--learn_deocclusion', type=bool, default=False, help='focus on training.')
    mode_group.add_argument('--BEV_matching_gts2preds', type=str, default='3D+2D_center', help='the way of properly matching the ground truths to the predictions.')
    mode_group.add_argument('--estimate_camera', type=bool, default=False, help='also estimate the extrinsics and FOV of camera.')
    mode_group.add_argument('--learn2Dprojection', type=bool, default=True, help='also estimate the extrinsics and FOV of camera.')
    mode_group.add_argument('--train_backbone', type=bool, default=True, help='also estimate the extrinsics and FOV of camera.')
    mode_group.add_argument('--temp_cam_regression', type=bool, default=True, help='regression of camera parameter in temporal mode.')
    mode_group.add_argument('--learn_cam_motion_composition_yz', type=bool, default=True, help='regression of camera parameter in temporal mode.')
    mode_group.add_argument('--learn_cam_motion_composition_xyz', type=bool, default=False, help='regression of camera parameter in temporal mode.')
    mode_group.add_argument('--learn_CamState', type=bool, default=False)
    mode_group.add_argument('--tracker_match_thresh', type=float, default=1.2)
    mode_group.add_argument('--tracker_det_thresh', type=float, default=0.18)
    mode_group.add_argument('--feature_update_thresh', type=float, default=0.3)
    V6_group = parser.add_argument_group(title='V6 options')
    V6_group.add_argument('--bv_with_fv_condition', type=bool, default=True)
    V6_group.add_argument('--add_offsetmap', type=bool, default=True)
    V6_group.add_argument('--fv_conditioned_way', type=str, default='attention')
    V6_group.add_argument('--num_depth_level', type=int, default=8, help='number of depth.')
    V6_group.add_argument('--scale_anchor', type=bool, default=True)
    V6_group.add_argument('--sampling_aggregation_way', type=str, default='floor')
    V6_group.add_argument('--acquire_pa_trans_scale', type=bool, default=False)
    V6_group.add_argument('--cam_dist_thresh', type=float, default=0.1)
    V6_group.add_argument('--focal_length', type=float, default=443.4, help='Default focal length, adopted from JTA dataset')
    V6_group.add_argument('--multi_depth', type=bool, default=False, help='whether to use the multi_depth mode')
    V6_group.add_argument('--depth_degree', default=1, type=int, help='whether to use the multi_depth mode')
    V6_group.add_argument('--FOV', type=int, default=60, help='Field of View')
    V6_group.add_argument('--matching_pckh_thresh', type=float, default=0.6, help='Threshold to determine the sucess matching based on pckh')
    V6_group.add_argument('--baby_threshold', type=float, default=0.8)
    train_group = parser.add_argument_group(title='training options')
    train_group.add_argument('--lr', help='lr', default=0.0003, type=float)
    train_group.add_argument('--adjust_lr_factor', type=float, default=0.1, help='factor for adjusting the lr')
    train_group.add_argument('--weight_decay', help='weight_decay', default=1e-06, type=float)
    train_group.add_argument('--epoch', type=int, default=80, help='training epochs')
    train_group.add_argument('--fine_tune', type=bool, default=True, help='whether to run online')
    train_group.add_argument('--gpu', default='0', help='gpus', type=str)
    train_group.add_argument('--batch_size', default=64, help='batch_size', type=int)
    train_group.add_argument('--input_size', default=512, type=int, help='size of input image')
    train_group.add_argument('--master_batch_size', default=-1, help='batch_size', type=int)
    train_group.add_argument('--nw', default=4, help='number of workers', type=int)
    train_group.add_argument('--optimizer_type', type=str, default='Adam', help='choice of optimizer')
    train_group.add_argument('--pretrain', type=str, default='simplebaseline', help='imagenet or spin or simplebaseline')
    train_group.add_argument('--fix_backbone_training_scratch', type=bool, default=False, help='whether to fix the backbone features if we train the model from scratch.')
    train_group.add_argument('--large_kernel_size', default=False, help='whether use large centermap kernel size', type=bool)
    model_group = parser.add_argument_group(title='model settings')
    model_group.add_argument('--backbone', type=str, default='hrnetv4', help='backbone model: resnet50 or hrnet')
    model_group.add_argument('--model_precision', type=str, default='fp16', help='the model precision: fp16/fp32')
    model_group.add_argument('--deconv_num', type=int, default=0)
    model_group.add_argument('--head_block_num', type=int, default=2, help='number of conv block in head')
    model_group.add_argument('--merge_smpl_camera_head', type=bool, default=False)
    model_group.add_argument('--use_coordmaps', type=bool, default=True, help='use the coordmaps')
    model_group.add_argument('--hrnet_pretrain', type=str, default=os.path.join(data_dir, 'trained_models/pretrain_hrnet.pkl'))
    model_group.add_argument('--resnet_pretrain', type=str, default=os.path.join(data_dir, 'trained_models/pretrain_resnet.pkl'))
    model_group.add_argument('--resnet_pretrain_sb', type=str, default=os.path.join(data_dir, 'trained_models/single_noeval_bs64_3dpw_106.2_75.0.pkl'))
    loss_group = parser.add_argument_group(title='loss options')
    loss_group.add_argument('--loss_thresh', default=1000, type=float, help='max loss value for a single loss')
    loss_group.add_argument('--max_supervise_num', default=-1, type=int, help='max person number supervised in each batch for stable GPU memory usage')
    loss_group.add_argument('--supervise_cam_params', type=bool, default=False)
    loss_group.add_argument('--match_preds_to_gts_for_supervision', type=bool, default=False, help='whether to match preds to gts for supervision')
    loss_group.add_argument('--matching_mode', type=str, default='all', help='all | random_one | ')
    loss_group.add_argument('--supervise_global_rot', type=bool, default=False, help='whether supervise the global rotation of the estimated SMPL model')
    loss_group.add_argument('--HMloss_type', type=str, default='MSE', help='supervision for 2D pose heatmap: MSE or focal loss')
    loss_group.add_argument('--learn_gmm_prior', type=bool, default=False)
    eval_group = parser.add_argument_group(title='evaluation options')
    eval_group.add_argument('--eval', type=bool, default=False, help='whether to run evaluation')
    eval_group.add_argument('--eval_datasets', type=str, default='pw3d', help='whether to run evaluation')
    eval_group.add_argument('--val_batch_size', default=64, help='valiation batch_size', type=int)
    eval_group.add_argument('--test_interval', default=2000, help='interval iteration between validation', type=int)
    eval_group.add_argument('--fast_eval_iter', type=int, default=-1, help='whether to run validation on a few iterations, like 200.')
    eval_group.add_argument('--top_n_error_vis', type=int, default=6, help='visulize the top n results during validation')
    eval_group.add_argument('--eval_2dpose', type=bool, default=False)
    eval_group.add_argument('--calc_pck', type=bool, default=False, help='whether calculate PCK during evaluation')
    eval_group.add_argument('--PCK_thresh', type=int, default=150, help='training epochs')
    eval_group.add_argument('--calc_PVE_error', type=bool, default=False)
    maps_group = parser.add_argument_group(title='Maps options')
    maps_group.add_argument('--centermap_size', type=int, default=64, help='the size of center map')
    maps_group.add_argument('--centermap_conf_thresh', type=float, default=0.25, help='the threshold of the centermap confidence for the valid subject')
    maps_group.add_argument('--collision_aware_centermap', type=bool, default=False, help='whether to use collision_aware_centermap')
    maps_group.add_argument('--collision_factor', type=float, default=0.2, help='whether to use collision_aware_centermap')
    maps_group.add_argument('--center_def_kp', type=bool, default=True, help='center definition: keypoints or bbox')
    distributed_train_group = parser.add_argument_group(title='options for distributed training')
    distributed_train_group.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    distributed_train_group.add_argument('--init_method', type=str, default='tcp://127.0.0.1:52468', help='URL:port of main server for distributed training')
    distributed_train_group.add_argument('--local_world_size', type=int, default=4, help='Number of processes participating in the job')
    distributed_train_group.add_argument('--distributed_training', type=bool, default=False, help='wether train model in distributed mode')
    reid_group = parser.add_argument_group(title='options for ReID')
    reid_group.add_argument('--with_reid', type=bool, default=False, help='whether estimate reid embedding')
    reid_group.add_argument('--reid_dim', type=int, default=64, help='channel number of reid embedding maps')
    relative_group = parser.add_argument_group(title='options for learning relativites')
    relative_group.add_argument('--learn_relative', type=bool, default=False)
    relative_group.add_argument('--learn_relative_age', type=bool, default=False)
    relative_group.add_argument('--learn_relative_depth', type=bool, default=False)
    relative_group.add_argument('--depth_loss_type', type=str, default='Log', help='Log | Piecewise | ')
    relative_group.add_argument('--learn_uncertainty', type=bool, default=False)
    log_group = parser.add_argument_group(title='log options')
    log_group.add_argument('--print_freq', type=int, default=50, help='training epochs')
    log_group.add_argument('--model_path', type=str, default='', help='trained model path')
    log_group.add_argument('--temp_model_path', type=str, default='', help='trained model path')
    log_group.add_argument('--log-path', type=str, default=os.path.join(root_dir, 'log/'), help='Path to save log file')
    hm_ae_group = parser.add_argument_group(title='learning 2D pose/associate embeddings options')
    hm_ae_group.add_argument('--learn_2dpose', type=bool, default=False)
    hm_ae_group.add_argument('--learn_AE', type=bool, default=False)
    hm_ae_group.add_argument('--learn_kp2doffset', type=bool, default=False)
    augmentation_group = parser.add_argument_group(title='augmentation options')
    augmentation_group.add_argument('--shuffle_crop_mode', type=bool, default=True, help='whether to shuffle the data loading mode between crop / uncrop for indoor 3D pose datasets only')
    augmentation_group.add_argument('--shuffle_crop_ratio_3d', default=0.9, type=float, help='the probability of changing the data loading mode from uncrop multi_person to crop single person')
    augmentation_group.add_argument('--shuffle_crop_ratio_2d', default=0.9, type=float, help='the probability of changing the data loading mode from uncrop multi_person to crop single person')
    augmentation_group.add_argument('--Synthetic_occlusion_ratio', default=0, type=float, help='whether to use use Synthetic occlusion')
    augmentation_group.add_argument('--color_jittering_ratio', default=0.2, type=float, help='whether to use use color jittering')
    augmentation_group.add_argument('--rotate_prob', default=0.2, type=float, help='whether to use rotation augmentation')
    dataset_group = parser.add_argument_group(title='datasets options')
    dataset_group.add_argument('--dataset_rootdir', type=str, default='/home/yusun/DataCenter/datasets', help='root dir of all datasets')
    dataset_group.add_argument('--datasets', type=str, default='h36m,mpii,coco,aich,up,ochuman,lsp,movi', help='which datasets are used')
    dataset_group.add_argument('--voc_dir', type=str, default=os.path.join(root_dir, 'datasets/VOC2012/'), help='VOC dataset path')
    dataset_group.add_argument('--max_person', default=64, type=int, help='max person number of each image')
    dataset_group.add_argument('--homogenize_pose_space', type=bool, default=False, help='whether to homogenize the pose space of 3D datasets')
    dataset_group.add_argument('--use_eft', type=bool, default=True, help='wether use eft annotations for training')
    smpl_group = parser.add_argument_group(title='SMPL options')
    smpl_group.add_argument('--smpl_mesh_root_align', type=bool, default=True)
    mode_group.add_argument('--Rot_type', type=str, default='6D', help='rotation representation type: angular, 6D')
    mode_group.add_argument('--rot_dim', type=int, default=6, help='rotation representation type: 3D angular, 6D')
    smpl_group.add_argument('--cam_dim', type=int, default=3, help='the dimention of camera param')
    smpl_group.add_argument('--beta_dim', type=int, default=10, help='the dimention of SMPL shape param, beta')
    smpl_group.add_argument('--smpl_joint_num', type=int, default=22, help='joint number of SMPL model we estimate')
    smpl_group.add_argument('--smpl_model_path', type=str, default=os.path.join(model_dir, 'parameters', 'SMPL_NEUTRAL.pth'), help='smpl model path')
    smpl_group.add_argument('--smpla_model_path', type=str, default=os.path.join(model_dir, 'parameters', 'SMPLA_NEUTRAL.pth'), help='smpl model path')
    smpl_group.add_argument('--smil_model_path', type=str, default=os.path.join(model_dir, 'parameters', 'SMIL_NEUTRAL.pth'), help='smpl model path')
    smpl_group.add_argument('--smpl_prior_path', type=str, default=os.path.join(model_dir, 'parameters', 'gmm_08.pkl'), help='smpl model path')
    smpl_group.add_argument('--smpl_J_reg_h37m_path', type=str, default=os.path.join(model_dir, 'parameters', 'J_regressor_h36m.npy'), help='SMPL regressor for 17 joints from H36M datasets')
    smpl_group.add_argument('--smpl_J_reg_extra_path', type=str, default=os.path.join(model_dir, 'parameters', 'J_regressor_extra.npy'), help='SMPL regressor for 9 extra joints from different datasets')
    smpl_group.add_argument('--smplx_model_folder', type=str, default=os.path.join(model_dir, 'parameters'), help='folder containing SMPLX folder')
    smpl_group.add_argument('--smplx_model_path', type=str, default=os.path.join(model_dir, 'parameters', 'SMPLX_NEUTRAL.pth'), help='folder containing SMPLX folder')
    smpl_group.add_argument('--smplxa_model_path', type=str, default=os.path.join(model_dir, 'parameters', 'SMPLXA_NEUTRAL.pth'), help='folder containing SMPLX folder')
    smpl_group.add_argument('--smpl_model_type', type=str, default='smpl', help='wether to use smpl, SMPL+A, SMPL-X')
    smpl_group.add_argument('--smpl_uvmap', type=str, default=os.path.join(model_dir, 'parameters', 'smpl_vt_ft.npz'), help='smpl UV Map coordinates for each vertice')
    smpl_group.add_argument('--wardrobe', type=str, default=os.path.join(model_dir, 'wardrobe'), help='path of smpl UV textures')
    smpl_group.add_argument('--cloth', type=str, default='f1', help='pick up cloth from the wardrobe or simplely use a single color')
    debug_group = parser.add_argument_group(title='Debug options')
    debug_group.add_argument('--track_memory_usage', type=bool, default=False)
    parsed_args = parser.parse_args(args=input_args)
    parsed_args.adjust_lr_epoch = []
    parsed_args.kernel_sizes = [5]
    with open(parsed_args.configs_yml) as file:
        configs_update = yaml.full_load(file)
    for key, value in configs_update['ARGS'].items():
        if sum([('--{}'.format(key) in input_arg) for input_arg in input_args]) == 0:
            if isinstance(value, str):
                exec("parsed_args.{} = '{}'".format(key, value))
            else:
                exec('parsed_args.{} = {}'.format(key, value))
    if 'loss_weight' in configs_update:
        for key, value in configs_update['loss_weight'].items():
            exec('parsed_args.{}_weight = {}'.format(key, value))
    if 'sample_prob' in configs_update:
        parsed_args.sample_prob_dict = configs_update['sample_prob']
    if 'image_sample_prob' in configs_update:
        parsed_args.image_sample_prob_dict = configs_update['image_sample_prob']
    if parsed_args.large_kernel_size:
        parsed_args.kernel_sizes = [11]
    if parsed_args.video:
        parse_args.tab = '{}_bs{}_tcl{}_{}'.format(parsed_args.tab, parsed_args.batch_size, parsed_args.temp_clip_length, parsed_args.datasets)
    else:
        parsed_args.tab = '{}_cm{}_{}'.format(parsed_args.backbone, parsed_args.centermap_size, parsed_args.tab, parsed_args.datasets)
    if parsed_args.distributed_training:
        parsed_args.local_rank = int(os.environ['LOCAL_RANK'])
    None
    return parsed_args


def args():
    return ConfigContext.parsed_args


def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


def gaussian2D(shape, sigma=1):
    m, n = [((ss - 1.0) / 2.0) for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian3D(d, h, w, center, s=2, device='cuda'):
    """
    :param d: hmap depth
    :param h: hmap height
    :param w: hmap width
    :param center: center of the Gaussian | ORDER: (x, y, z)
    :param s: sigma of the Gaussian
    :return: heatmap (shape torch.Size([d, h, w])) with a gaussian centered in `center`
    """
    x = torch.arange(0, w, 1).float()
    y = torch.arange(0, h, 1).float()
    y = y.unsqueeze(1)
    z = torch.arange(0, d, 1).float()
    z = z.unsqueeze(1).unsqueeze(1)
    x0 = center[0]
    y0 = center[1]
    z0 = center[2]
    return torch.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / s ** 2)


def get_3Dcoord_maps(size=128, z_base=None):
    range_arr = torch.arange(size, dtype=torch.float32)
    if z_base is None:
        Z_map = range_arr.reshape(1, size, 1, 1, 1).repeat(1, 1, size, size, 1) / size * 2 - 1
    else:
        Z_map = z_base.reshape(1, size, 1, 1, 1).repeat(1, 1, size, size, 1)
    Y_map = range_arr.reshape(1, 1, size, 1, 1).repeat(1, size, 1, size, 1) / size * 2 - 1
    X_map = range_arr.reshape(1, 1, 1, size, 1).repeat(1, size, size, 1, 1) / size * 2 - 1
    out = torch.cat([Z_map, Y_map, X_map], dim=-1)
    return out


class CenterMap(object):

    def __init__(self, style='heatmap_adaptive_scale'):
        self.style = style
        self.size = args().centermap_size
        self.max_person = args().max_person
        self.shrink_scale = float(args().input_size // self.size)
        self.dims = 1
        self.sigma = 1
        self.conf_thresh = args().centermap_conf_thresh
        None
        self.gk_group, self.pool_group = self.generate_kernels(args().kernel_sizes)
        if args().model_version > 4:
            self.prepare_parsing()

    def prepare_parsing(self):
        self.coordmap_3d = get_3Dcoord_maps(size=self.size)
        self.maxpool3d = torch.nn.MaxPool3d(5, 1, (5 - 1) // 2)

    def generate_kernels(self, kernel_size_list):
        gk_group, pool_group = {}, {}
        for kernel_size in set(kernel_size_list):
            x = np.arange(0, kernel_size, 1, float)
            y = x[:, np.newaxis]
            x0, y0 = (kernel_size - 1) // 2, (kernel_size - 1) // 2
            gaussian_distribution = -((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)
            gk_group[kernel_size] = np.exp(gaussian_distribution)
            pool_group[kernel_size] = torch.nn.MaxPool2d(kernel_size, 1, (kernel_size - 1) // 2)
        return gk_group, pool_group

    def process_gt_CAM(self, center_normed):
        center_list = []
        valid_mask = center_normed[:, :, 0] > -1
        valid_inds = torch.where(valid_mask)
        valid_batch_inds, valid_person_ids = valid_inds[0], valid_inds[1]
        center_gt = ((center_normed + 1) / 2 * self.size).long()
        center_gt_valid = center_gt[valid_mask]
        return valid_batch_inds, valid_person_ids, center_gt_valid

    def generate_centermap(self, center_locs, **kwargs):
        if self.style == 'heatmap':
            return self.generate_centermap_heatmap(center_locs, **kwargs)
        elif self.style == 'heatmap_adaptive_scale':
            return self.generate_centermap_heatmap_adaptive_scale(center_locs, **kwargs)
        else:
            raise NotImplementedError

    def parse_centermap(self, center_map):
        if self.style == 'heatmap':
            return self.parse_centermap_heatmap(center_map)
        elif self.style == 'heatmap_adaptive_scale' and center_map.shape[1] == 1:
            return self.parse_centermap_heatmap_adaptive_scale_batch(center_map)
        elif self.style == 'heatmap_adaptive_scale' and center_map.shape[1] == self.size:
            return self.parse_3dcentermap_heatmap_adaptive_scale_batch(center_map)
        else:
            raise NotImplementedError

    def generate_centermap_mask(self, center_locs):
        centermap = np.ones((self.dims, self.size, self.size))
        centermap[-1] = 0
        for center_loc in center_locs:
            map_coord = ((center_loc + 1) / 2 * self.size).astype(np.int32) - 1
            centermap[0, map_coord[0], map_coord[1]] = 0
            centermap[1, map_coord[0], map_coord[1]] = 1
        return centermap

    def generate_centermap_heatmap(self, center_locs, kernel_size=5, **kwargs):
        hms = np.zeros((self.dims, self.size, self.size), dtype=np.float32)
        offset = (kernel_size - 1) // 2
        for idx, pt in enumerate(center_locs):
            x, y = int((pt[0] + 1) / 2 * self.size), int((pt[1] + 1) / 2 * self.size)
            if x < 0 or y < 0 or x >= self.size or y >= self.size:
                continue
            ul = int(np.round(x - offset)), int(np.round(y - offset))
            br = int(np.round(x + offset + 1)), int(np.round(y + offset + 1))
            c, d = max(0, -ul[0]), min(br[0], self.size) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], self.size) - ul[1]
            cc, dd = max(0, ul[0]), min(br[0], self.size)
            aa, bb = max(0, ul[1]), min(br[1], self.size)
            hms[0, aa:bb, cc:dd] = np.maximum(hms[0, aa:bb, cc:dd], self.gk_group[kernel_size][a:b, c:d])
        return hms

    def generate_centermap_heatmap_adaptive_scale(self, center_locs, bboxes_hw_norm, occluded_by_who=None, **kwargs):
        """
           center_locs is in the order of (y,x), corresponding to (w,h), while in the loading data, we have rectified it to the correct (x, y) order
        """
        radius_list = _calc_radius_(bboxes_hw_norm, map_size=self.size)
        if args().collision_aware_centermap and occluded_by_who is not None:
            for cur_idx, occluded_idx in enumerate(occluded_by_who):
                if occluded_idx > -1:
                    dist_onmap = np.sqrt(((center_locs[occluded_idx] - center_locs[cur_idx]) ** 2).sum()) + 0.0001
                    least_dist = (radius_list[occluded_idx] + radius_list[cur_idx] + 1) / self.size * 2
                    if dist_onmap < least_dist:
                        offset = np.abs(((radius_list[occluded_idx] + radius_list[cur_idx] + 1) / self.size * 2 - dist_onmap) / dist_onmap) * (center_locs[occluded_idx] - center_locs[cur_idx] + 0.0001) * args().collision_factor
                        center_locs[cur_idx] -= offset / 2
                        center_locs[occluded_idx] += offset / 2
            center_locs = np.clip(center_locs, -1, 1)
            center_locs[center_locs == -1] = -0.96
            center_locs[center_locs == 1] = 0.96
        heatmap = self.generate_heatmap_adaptive_scale(center_locs, radius_list)
        heatmap = torch.from_numpy(heatmap)
        return heatmap

    def generate_heatmap_adaptive_scale(self, center_locs, radius_list, k=1):
        heatmap = np.zeros((1, self.size, self.size), dtype=np.float32)
        for center, radius in zip(center_locs, radius_list):
            diameter = 2 * radius + 1
            gaussian = gaussian2D((diameter, diameter), sigma=float(diameter) / 6)
            x, y = int((center[0] + 1) / 2 * self.size), int((center[1] + 1) / 2 * self.size)
            if x < 0 or y < 0 or x >= self.size or y >= self.size:
                continue
            height, width = heatmap.shape[1:]
            left, right = min(x, radius), min(width - x, radius + 1)
            top, bottom = min(y, radius), min(height - y, radius + 1)
            masked_heatmap = heatmap[0, y - top:y + bottom, x - left:x + right]
            masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
            if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
                np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
            heatmap[0, y, x] = 1
        return heatmap

    def generate_centermap_3dheatmap_adaptive_scale_batch(self, batch_center_locs, radius=3, depth_num=None, device='cuda:0'):
        if depth_num is None:
            depth_num = int(self.size // 2)
        heatmap = torch.zeros((len(batch_center_locs), depth_num, self.size, self.size), device=device)
        for bid, center_locs in enumerate(batch_center_locs):
            for cid, center in enumerate(center_locs):
                diameter = int(2 * radius + 1)
                gaussian_patch = gaussian3D(w=diameter, h=diameter, d=diameter, center=(diameter // 2, diameter // 2, diameter // 2), s=float(diameter) / 6, device=device)
                xa, ya, za = int(max(0, center[0] - diameter // 2)), int(max(0, center[1] - diameter // 2)), int(max(0, center[2] - diameter // 2))
                xb, yb, zb = int(min(center[0] + diameter // 2, self.size - 1)), int(min(center[1] + diameter // 2, self.size - 1)), int(min(center[2] + diameter // 2, depth_num - 1))
                gxa = xa - int(center[0] - diameter // 2)
                gya = ya - int(center[1] - diameter // 2)
                gza = za - int(center[2] - diameter // 2)
                gxb = xb + 1 - xa + gxa
                gyb = yb + 1 - ya + gya
                gzb = zb + 1 - za + gza
                heatmap[bid, za:zb + 1, ya:yb + 1, xa:xb + 1] = torch.max(torch.cat(tuple([heatmap[bid, za:zb + 1, ya:yb + 1, xa:xb + 1].unsqueeze(0), gaussian_patch[gza:gzb, gya:gyb, gxa:gxb].unsqueeze(0)])), 0)[0]
        return heatmap

    def generate_centermap_3dheatmap_adaptive_scale(self, center_locs, depth_num=None, device='cpu'):
        """
        center_locs: center locations (X,Y,Z) on 3D center map (BxDxHxW)
        """
        if depth_num is None:
            depth_num = int(self.size // 2)
        heatmap = torch.zeros((depth_num, self.size, self.size))
        if len(center_locs) == 0:
            return heatmap, False
        adaptive_depth_uncertainty = np.array(center_locs)[:, 2].astype(np.float16) / depth_num
        depth_uncertainty = (4 + adaptive_depth_uncertainty * 4).astype(np.int32) // 2 * 2 + 1
        adaptive_image_scale = (1 - adaptive_depth_uncertainty) / 2.0
        uv_radius = (_calc_uv_radius_(adaptive_image_scale, map_size=self.size) * 2 + 1).astype(np.int32)
        for cid, center in enumerate(center_locs):
            width, height = uv_radius[cid], uv_radius[cid]
            depth = depth_uncertainty[cid]
            diameter = np.linalg.norm([width / 2.0, height / 2.0, depth / 2.0], ord=2, axis=0) * 2
            gaussian_patch = gaussian3D(w=width, h=height, d=depth, center=(width // 2, height // 2, depth // 2), s=float(diameter) / 6, device=device)
            xa, ya, za = int(max(0, center[0] - width // 2)), int(max(0, center[1] - height // 2)), int(max(0, center[2] - depth // 2))
            xb, yb, zb = int(min(center[0] + width // 2, self.size - 1)), int(min(center[1] + height // 2, self.size - 1)), int(min(center[2] + depth // 2, depth_num - 1))
            gxa = xa - int(center[0] - width // 2)
            gya = ya - int(center[1] - height // 2)
            gza = za - int(center[2] - depth // 2)
            gxb = xb + 1 - xa + gxa
            gyb = yb + 1 - ya + gya
            gzb = zb + 1 - za + gza
            heatmap[za:zb + 1, ya:yb + 1, xa:xb + 1] = torch.max(torch.cat(tuple([heatmap[za:zb + 1, ya:yb + 1, xa:xb + 1].unsqueeze(0), gaussian_patch[gza:gzb, gya:gyb, gxa:gxb].unsqueeze(0)])), 0)[0]
        return heatmap, True

    def generate_centermap_3dheatmap_adaptive_scale_org(self, center_locs, radius=3, depth_num=None, device='cpu'):
        """
        center_locs: center locations (X,Y,Z) on 3D center map (BxDxHxW)
        """
        if depth_num is None:
            depth_num = int(self.size // 2)
        heatmap = torch.zeros((depth_num, self.size, self.size))
        if len(center_locs) == 0:
            return heatmap, False
        for cid, center in enumerate(center_locs):
            diameter = int(2 * radius + 1)
            gaussian_patch = gaussian3D(w=diameter, h=diameter, d=diameter, center=(diameter // 2, diameter // 2, diameter // 2), s=float(diameter) / 6, device=device)
            xa, ya, za = int(max(0, center[0] - diameter // 2)), int(max(0, center[1] - diameter // 2)), int(max(0, center[2] - diameter // 2))
            xb, yb, zb = int(min(center[0] + diameter // 2, self.size - 1)), int(min(center[1] + diameter // 2, self.size - 1)), int(min(center[2] + diameter // 2, depth_num - 1))
            gxa = xa - int(center[0] - diameter // 2)
            gya = ya - int(center[1] - diameter // 2)
            gza = za - int(center[2] - diameter // 2)
            gxb = xb + 1 - xa + gxa
            gyb = yb + 1 - ya + gya
            gzb = zb + 1 - za + gza
            heatmap[za:zb + 1, ya:yb + 1, xa:xb + 1] = torch.max(torch.cat(tuple([heatmap[za:zb + 1, ya:yb + 1, xa:xb + 1].unsqueeze(0), gaussian_patch[gza:gzb, gya:gyb, gxa:gxb].unsqueeze(0)])), 0)[0]
        return heatmap, True

    def multi_channel_nms(self, center_maps):
        center_map_pooled = []
        for depth_idx, center_map in enumerate(center_maps):
            center_map_pooled.append(nms(center_map[None], pool_func=self.pool_group[args().kernel_sizes[depth_idx]]))
        center_maps_max = torch.max(torch.cat(center_map_pooled, 0), 0).values
        center_map_nms = nms(center_maps_max[None], pool_func=self.pool_group[args().kernel_sizes[-1]])[0]
        return center_map_nms

    def parse_centermap_mask(self, center_map):
        center_map_bool = torch.argmax(center_map, 1).bool()
        center_idx = torch.stack(torch.where(center_map_bool)).transpose(1, 0)
        return center_idx

    def parse_centermap_heatmap(self, center_maps):
        if center_maps.shape[0] > 1:
            center_map_nms = self.multi_channel_nms(center_maps)
        else:
            center_map_nms = nms(center_maps, pool_func=self.pool_group[args().kernel_sizes[-1]])[0]
        h, w = center_map_nms.shape
        centermap = center_map_nms.view(-1)
        confidence, index = centermap.topk(self.max_person)
        x = index % w
        y = (index / w).long()
        idx_topk = torch.stack((y, x), dim=1)
        center_preds, conf_pred = idx_topk[confidence > self.conf_thresh], confidence[confidence > self.conf_thresh]
        return center_preds, conf_pred

    def parse_centermap_heatmap_adaptive_scale(self, center_maps):
        center_map_nms = nms(center_maps, pool_func=self.pool_group[args().kernel_sizes[-1]])[0]
        h, w = center_map_nms.shape
        centermap = center_map_nms.view(-1)
        confidence, index = centermap.topk(self.max_person)
        x = index % w
        y = (index / float(w)).long()
        idx_topk = torch.stack((y, x), dim=1)
        center_preds, conf_pred = idx_topk[confidence > self.conf_thresh], confidence[confidence > self.conf_thresh]
        return center_preds, conf_pred

    def parse_centermap_heatmap_adaptive_scale_batch(self, center_maps, top_n_people=None):
        center_map_nms = nms(center_maps, pool_func=self.pool_group[args().kernel_sizes[-1]])
        b, c, h, w = center_map_nms.shape
        K = self.max_person if top_n_people is None else top_n_people
        topk_scores, topk_inds = torch.topk(center_map_nms.reshape(b, c, -1), K)
        topk_inds = topk_inds % (h * w)
        topk_ys = torch.div(topk_inds.long(), w).float()
        topk_xs = (topk_inds % w).int().float()
        topk_score, index = torch.topk(topk_scores.reshape(b, -1), K)
        topk_clses = torch.div(index.long(), K)
        topk_inds = gather_feature(topk_inds.view(b, -1, 1), index).reshape(b, K)
        topk_ys = gather_feature(topk_ys.reshape(b, -1, 1), index).reshape(b, K)
        topk_xs = gather_feature(topk_xs.reshape(b, -1, 1), index).reshape(b, K)
        if top_n_people is not None:
            mask = topk_score > 0
            mask[:] = True
        else:
            mask = topk_score > self.conf_thresh
        batch_ids = torch.where(mask)[0]
        center_yxs = torch.stack([topk_ys[mask], topk_xs[mask]]).permute((1, 0))
        return batch_ids, topk_inds[mask], center_yxs, topk_score[mask]

    def parse_3dcentermap_heatmap_adaptive_scale_batch(self, center_maps, top_n_people=None):
        center_map_nms = nms(center_maps, pool_func=self.maxpool3d).squeeze(1)
        b, c, h, w = center_map_nms.shape
        K = self.max_person if top_n_people is None else top_n_people
        topk_scores, topk_inds = torch.topk(center_map_nms.reshape(b, c, -1), K)
        topk_inds = topk_inds % (h * w)
        topk_ys = torch.div(topk_inds.long(), w).float()
        topk_xs = (topk_inds % w).int().float()
        topk_score, index = torch.topk(topk_scores.reshape(b, -1), K)
        topk_inds = gather_feature(topk_inds.view(b, -1, 1), index).reshape(b, K)
        topk_zs = torch.div(index.long(), K)
        topk_ys = gather_feature(topk_ys.reshape(b, -1, 1), index).reshape(b, K)
        topk_xs = gather_feature(topk_xs.reshape(b, -1, 1), index).reshape(b, K)
        if top_n_people is not None:
            mask = topk_score > 0
            mask[:] = True
        else:
            mask = topk_score > self.conf_thresh
        batch_ids = torch.where(mask)[0]
        center_zyxs = torch.stack([topk_zs[mask], topk_ys[mask], topk_xs[mask]]).permute((1, 0)).long()
        return [batch_ids, center_zyxs, topk_score[mask]]

    def parse_local_centermap3D(self, center_maps, pred_batch_ids, center_yxs, only_max=False):
        if len(center_yxs) == 0:
            return [], [], []
        cys = center_yxs[:, 0]
        cxs = center_yxs[:, 1]
        center_map_nms = nms(center_maps, pool_func=self.maxpool3d).squeeze(1)
        b, c, h, w = center_map_nms.shape
        cys = torch.clip(cys, 0, h - 1)
        cxs = torch.clip(cxs, 0, w - 1)
        device = center_maps.device
        local_K = 16
        czyxs = []
        new_pred_batch_inds = []
        top_scores = []
        for batch_id, cy, cx in zip(pred_batch_ids, cys, cxs):
            local_vec = center_map_nms[batch_id, :, cy, cx]
            topk_scores, topk_zs = torch.topk(local_vec, local_K)
            if only_max:
                mask = torch.zeros(len(topk_scores)).bool()
                mask[0] = True
            else:
                mask = topk_scores > self.conf_thresh
                if mask.sum() == 0:
                    mask[0] = True
            for cz, score in zip(topk_zs[mask], topk_scores[mask]):
                czyxs.append(torch.Tensor([cz, cy, cx]))
                new_pred_batch_inds.append(batch_id)
                top_scores.append(score)
        czyxs = torch.stack(czyxs).long()
        new_pred_batch_inds = torch.Tensor(new_pred_batch_inds).long()
        top_scores = torch.Tensor(top_scores)
        return new_pred_batch_inds, czyxs, top_scores


DEFAULT_DTYPE = torch.float32


def _calc_matched_PCKh_(real, pred, kp2d_mask, error_thresh=0.143):
    PCKs = torch.ones(len(kp2d_mask)).float() * -1.0
    if kp2d_mask.sum() > 0:
        vis = (real > -1.0).sum(-1) == real.shape[-1]
        error = torch.norm(real - pred, p=2, dim=-1)
        for ind, (e, v) in enumerate(zip(error, vis)):
            if v.sum() < 2:
                continue
            real_valid = real[ind, v]
            person_scales = torch.sqrt((real_valid[:, 0].max(-1).values - real_valid[:, 0].min(-1).values) ** 2 + (real_valid[:, 1].max(-1).values - real_valid[:, 1].min(-1).values) ** 2)
            error_valid = e[v]
            correct_kp_mask = (error_valid / person_scales < error_thresh).float()
            PCKs[ind] = correct_kp_mask.sum() / len(correct_kp_mask)
    return PCKs


def angle_axis_to_quaternion(angle_axis: 'torch.Tensor') ->torch.Tensor:
    """Convert an angle axis to a quaternion.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        angle_axis (torch.Tensor): tensor with angle axis.

    Return:
        torch.Tensor: tensor with quaternion.

    Shape:
        - Input: :math:`(*, 3)` where `*` means, any number of dimensions
        - Output: :math:`(*, 4)` in (w,x,y,z) order.
    """
    if not torch.is_tensor(angle_axis):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(angle_axis)))
    if not angle_axis.shape[-1] == 3:
        raise ValueError('Input must be a tensor of shape Nx3 or 3. Got {}'.format(angle_axis.shape))
    a0: 'torch.Tensor' = angle_axis[..., 0:1]
    a1: 'torch.Tensor' = angle_axis[..., 1:2]
    a2: 'torch.Tensor' = angle_axis[..., 2:3]
    theta_squared: 'torch.Tensor' = a0 * a0 + a1 * a1 + a2 * a2
    theta: 'torch.Tensor' = torch.sqrt(theta_squared)
    half_theta: 'torch.Tensor' = theta * 0.5
    mask: 'torch.Tensor' = theta_squared > 0.0
    ones: 'torch.Tensor' = torch.ones_like(half_theta)
    k_neg: 'torch.Tensor' = 0.5 * ones
    k_pos: 'torch.Tensor' = torch.sin(half_theta) / theta
    k: 'torch.Tensor' = torch.where(mask, k_pos, k_neg)
    w: 'torch.Tensor' = torch.where(mask, torch.cos(half_theta), ones)
    quaternion: 'torch.Tensor' = torch.zeros_like(angle_axis)
    quaternion[..., 0:1] += a0 * k
    quaternion[..., 1:2] += a1 * k
    quaternion[..., 2:3] += a2 * k
    return torch.cat([w, quaternion], dim=-1)


def clip_frame_pairs_indes(N, device):
    first_index, second_index = torch.meshgrid(torch.arange(N), torch.arange(N))
    first_index = first_index.reshape(-1)
    second_index = second_index.reshape(-1)
    k = first_index != second_index
    first_index = first_index[k]
    second_index = second_index[k]
    return first_index, second_index


def q_conjugate(q):
    """
    Returns the complex conjugate of the input quaternion tensor of shape [*, 4].
    """
    assert q.shape[-1] == 4
    conj = torch.tensor([1, -1, -1, -1], device=q.device)
    return q * conj.expand_as(q)


def q_mul(q1, q2):
    """
    Multiply quaternion q1 with q2.
    Expects two equally-sized tensors of shape [*, 4], where * denotes any number of dimensions.
    Returns q1*q2 as a tensor of shape [*, 4].
    """
    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4
    original_shape = q1.shape
    terms = torch.bmm(q2.view(-1, 4, 1), q1.view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def q_normalize(q):
    """
    Normalize the coefficients of a given quaternion tensor of shape [*, 4].
    """
    assert q.shape[-1] == 4
    norm = torch.sqrt(torch.sum(torch.square(q), dim=-1))
    assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=q.device)))
    return torch.div(q, norm[:, None])


def quaternion_difference(q1, q2):
    q1_normed, q2_normed = q_normalize(q1), q_normalize(q2)
    q1_inv = q_conjugate(q1_normed)
    difference = q_mul(q1_inv, q2_normed)
    return difference


def quaternion_loss(predictions, targets):
    """
    Computes the quaternion loss between predicted and target quaternions.
    Args:
        predictions (torch.Tensor): predicted quaternions of shape (batch_size, 4)
        targets (torch.Tensor): target quaternions of shape (batch_size, 4)
    Returns:
        quaternion_loss (torch.Tensor): quaternion loss value
    """
    predictions_norm = predictions / torch.norm(predictions, dim=1, keepdim=True)
    targets_norm = targets / torch.norm(targets, dim=1, keepdim=True)
    dot_product = torch.sum(predictions_norm * targets_norm, dim=1)
    angle = 2 * torch.acos(torch.clamp(dot_product, min=-1, max=1))
    quaternion_loss = torch.mean(angle)
    return quaternion_loss


def _calc_world_gros_loss_(preds, gts, vmasks, sequence_inds):
    loss = []
    device = preds.device
    gts, vmasks = gts, vmasks
    for seq_inds in sequence_inds:
        pred, gt, vmask = preds[seq_inds], gts[seq_inds], vmasks[seq_inds]
        if vmask.sum() == 0:
            continue
        N = pred.shape[0]
        first_index, second_index = clip_frame_pairs_indes(N, device)
        quaternion_pred = angle_axis_to_quaternion(pred)
        quaternion_gt = angle_axis_to_quaternion(gt)
        delta_pred = quaternion_difference(quaternion_pred[first_index], quaternion_pred[second_index])
        delta_gt = quaternion_difference(quaternion_gt[first_index], quaternion_gt[second_index])
        error = quaternion_loss(delta_pred, delta_gt)
        loss.append(error)
    loss = torch.stack(loss) if len(loss) > 0 else torch.zeros(1, device=device)
    return loss


def _check_params_(params):
    assert params.shape[0] > 0, logging.error('meta_data[params] dim 0 is empty, params: {}'.format(params))
    assert params.shape[1] > 0, logging.error('meta_data[params] dim 1 is empty, params shape: {}, params: {}'.format(params.shape, params))


def batch_kp_2d_l2_loss(real, pred, images):
    """ 
    Directly supervise the 2D coordinates of global joints, like torso
    While supervise the relative 2D coordinates of part joints, like joints on face, feets
    """
    vis_mask = ((real > -1.99).sum(-1) == real.shape[-1]).float()
    for parent_joint, leaf_joints in constants.joint2D_tree.items():
        parent_id = constants.SMPL_ALL_44[parent_joint]
        leaf_ids = np.array([constants.SMPL_ALL_44[leaf_joint] for leaf_joint in leaf_joints])
        vis_mask[:, leaf_ids] = vis_mask[:, [parent_id]] * vis_mask[:, leaf_ids]
        real[:, leaf_ids] -= real[:, [parent_id]]
        pred[:, leaf_ids] -= pred[:, [parent_id]]
    bv_mask = torch.logical_and(vis_mask.sum(-1) > 0, (real - pred).sum(-1).sum(-1) != 0)
    vis_mask = vis_mask[bv_mask]
    loss = 0
    if vis_mask.sum() > 0:
        diff = torch.norm(real[bv_mask] - pred[bv_mask], p=2, dim=-1)
        loss = (diff * vis_mask).sum(-1) / (vis_mask.sum(-1) + 0.0001)
        if torch.isnan(loss).sum() > 0 or (loss > 1000).sum() > 0:
            return 0
    return loss


def batch_kp_2d_l2_loss_old(real, pred, top_limit=100):
    vis_mask = (real > -1.99).sum(-1) == real.shape[-1]
    loss = torch.norm(real[vis_mask] - pred[vis_mask], p=2, dim=-1)
    loss = loss[~torch.isnan(loss)]
    loss = loss[loss < top_limit]
    if len(loss) == 0:
        return 0
    return loss


def batch_l2_loss(real, predict):
    if len(real) == 0:
        return 0
    loss = torch.norm(real - predict, p=2, dim=1)
    loss = loss[~torch.isnan(loss)]
    if len(loss) == 0:
        return 0
    return loss


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    把四元组的系数转化成旋转矩阵。四元组表示三维旋转
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def batch_rodrigues(param):
    batch_size = param.shape[0]
    l1norm = torch.norm(param + 1e-08, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(param, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat2mat(quat)


def batch_smpl_pose_l2_error(real, predict):
    batch_size = real.shape[0]
    real = batch_rodrigues(real.reshape(-1, 3)).contiguous()
    predict = batch_rodrigues(predict.reshape(-1, 3)).contiguous()
    loss = torch.norm((real - predict).view(-1, 9), p=2, dim=-1)
    loss = loss.reshape(batch_size, -1).mean(-1)
    return loss


def get_valid_offset_mask(traj_gts, clip_frame_ids):
    dims = traj_gts.shape[-1]
    current_position = traj_gts[torch.arange(len(clip_frame_ids)), clip_frame_ids]
    previous_position = traj_gts[torch.arange(len(clip_frame_ids)), clip_frame_ids - 1]
    valid_offset_mask = (clip_frame_ids != 0) * ((current_position != -2.0).sum(-1) == dims) * ((previous_position != -2.0).sum(-1) == dims)
    offset_gts = current_position[valid_offset_mask] - previous_position[valid_offset_mask]
    return valid_offset_mask, offset_gts


def get_valid_offset_maskV2(traj_gts, clip_frame_ids):
    dims = traj_gts.shape[-1]
    current_position = traj_gts.clone()
    previous_position = traj_gts[torch.arange(len(traj_gts)) - 1].clone()
    valid_offset_mask = (clip_frame_ids.clone() != 0) * ((current_position != -2.0).sum(-1) == dims) * ((previous_position != -2.0).sum(-1) == dims)
    offset_gts = current_position[valid_offset_mask] - previous_position[valid_offset_mask]
    return valid_offset_mask, offset_gts


def calc_motion_offsets3D_loss(motion_offsets, clip_frame_ids, traj3D_gts):
    if len(traj3D_gts.shape) == 3:
        valid_3Doffset_masks, offset3D_gts = get_valid_offset_mask(traj3D_gts, clip_frame_ids)
    elif len(traj3D_gts.shape) == 2:
        valid_3Doffset_masks, offset3D_gts = get_valid_offset_maskV2(traj3D_gts, clip_frame_ids)
    if valid_3Doffset_masks.sum() > 0:
        valid_3Doffset_masks = valid_3Doffset_masks.detach()
        offset3D_loss = torch.norm(motion_offsets[valid_3Doffset_masks] - offset3D_gts, p=2, dim=-1)
    else:
        offset3D_loss = 0
    return offset3D_loss


def align_by_parts(joints, align_inds=None):
    if align_inds is None:
        return joints
    pelvis = joints[:, align_inds].mean(1)
    return joints - torch.unsqueeze(pelvis, dim=1)


def calc_mpjpe(real, pred, align_inds=None, sample_wise=True, trans=None, return_org=False):
    vis_mask = real[:, :, 0] != -2.0
    if align_inds is not None:
        pred_aligned = align_by_parts(pred, align_inds=align_inds)
        if trans is not None:
            pred_aligned += trans
        real_aligned = align_by_parts(real, align_inds=align_inds)
    else:
        pred_aligned, real_aligned = pred, real
    mpjpe_each = compute_mpjpe(pred_aligned, real_aligned, vis_mask, sample_wise=sample_wise)
    if return_org:
        return mpjpe_each, (real_aligned, pred_aligned, vis_mask)
    return mpjpe_each


def batch_compute_similarity_transform_torch(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    var1 = torch.sum(X1 ** 2, dim=1).sum(dim=1)
    K = X1.bmm(X2.permute(0, 2, 1))
    U, s, Vh = torch.linalg.svd(K)
    V = Vh.transpose(-2, -1)
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1
    t = mu2 - scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(mu1)
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t
    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)
    return S1_hat, (scale, R, t)


def calc_pampjpe(real, pred, sample_wise=True, return_transform_mat=False):
    real, pred = real.float(), pred.float()
    vis_mask = (real[:, :, 0] != -2.0).sum(0) == len(real)
    pred_tranformed, PA_transform = batch_compute_similarity_transform_torch(pred[:, vis_mask], real[:, vis_mask])
    pa_mpjpe_each = compute_mpjpe(pred_tranformed, real[:, vis_mask], sample_wise=sample_wise)
    if return_transform_mat:
        return pa_mpjpe_each, PA_transform
    else:
        return pa_mpjpe_each


def calc_pj2d_error(real, pred, joint_inds=None):
    if joint_inds is not None:
        real, pred = real[:, joint_inds], pred[:, joint_inds]
    vis_mask = (real > -1.99).sum(-1) == real.shape[-1]
    bv_mask = torch.logical_and(vis_mask.float().sum(-1) > 0, (real - pred).sum(-1).sum(-1) != 0)
    batch_errors = torch.zeros(len(pred))
    for bid in torch.where(bv_mask)[0]:
        vmask = vis_mask[bid]
        diff = torch.norm(real[bid][vmask] - pred[bid][vmask], p=2, dim=-1).mean()
        batch_errors[bid] = diff.item()
    return batch_errors


def calc_temporal_shape_consistency_loss(pred_betas, sequence_inds, weights=None):
    temp_shape_consist_loss = 0
    seq_losses = []
    for seq_inds in sequence_inds:
        seq_pred_betas = pred_betas[seq_inds]
        average_shape = seq_pred_betas.mean(0).unsqueeze(0).detach()
        diff = seq_pred_betas - average_shape
        if weights is not None:
            diff = diff * weights.unsqueeze(0)
        seq_losses.append(torch.norm(diff, p=2, dim=1))
    if len(seq_losses) > 0:
        temp_shape_consist_loss = torch.cat(seq_losses, 0)
    return temp_shape_consist_loss


def extract_sequence_inds(subject_ids, video_seq_ids, clip_frame_ids):
    sequence_inds = []
    for seq_id in torch.unique(video_seq_ids):
        seq_inds = torch.where(video_seq_ids == seq_id)[0]
        for subj_id in torch.unique(subject_ids[seq_inds]):
            if subj_id == -1:
                continue
            subj_mask = subject_ids[seq_inds] == subj_id
            sequence_inds.append(seq_inds[subj_mask][torch.argsort(clip_frame_ids[seq_inds][subj_mask])])
    return sequence_inds


def focal_loss(pred, gt, max_loss_limit=0.8):
    """ Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = torch.zeros(gt.size(0))
    pred_log = torch.clamp(pred.clone(), min=0.001, max=1 - 0.001)
    pos_loss = torch.log(pred_log) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred_log) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum(-1).sum(-1)
    pos_loss = pos_loss.sum(-1).sum(-1)
    neg_loss = neg_loss.sum(-1).sum(-1)
    mask = num_pos > 0
    loss[~mask] = loss[~mask] - neg_loss[~mask]
    loss[mask] = loss[mask] - (pos_loss[mask] + neg_loss[mask]) / (num_pos[mask] + 0.0001)
    while (loss > max_loss_limit).sum() > 0:
        exclude_mask = loss > max_loss_limit
        loss[exclude_mask] = loss[exclude_mask] / 4
    return loss.mean(-1)


def focal_loss_3D(pred, gt):
    """ Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x z x h x w)
      gt_regr (batch x z x h x w)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = torch.zeros(gt.size(0))
    pred[torch.isnan(pred)] = 0.001
    pred_log = torch.clamp(pred.clone(), min=0.001, max=1 - 0.001)
    pos_loss = torch.log(pred_log) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred_log) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum(-1).sum(-1).sum(-1)
    pos_loss = pos_loss.sum(-1).sum(-1).mean(-1)
    neg_loss = neg_loss.sum(-1).sum(-1).mean(-1)
    mask = num_pos > 0
    loss[~mask] = loss[~mask] - neg_loss[~mask]
    loss[mask] = loss[mask] - (pos_loss[mask] + neg_loss[mask]) / (num_pos[mask] + 0.0001)
    if torch.isnan(loss).sum() > 0:
        None
        loss[torch.isnan(loss)] = 0
    return loss.mean(-1)


def kid_offset_loss(kid_offset_preds, kid_offset_gts, matched_mask=None):
    device = kid_offset_preds.device
    kid_offset_gts = kid_offset_gts
    age_vmask = kid_offset_gts != -1
    if matched_mask is not None:
        age_vmask = age_vmask * matched_mask
    if age_vmask.sum() == 0:
        return 0
    return ((kid_offset_preds[age_vmask] - kid_offset_gts[age_vmask]) ** 2).mean()


def relative_age_loss(kid_offset_preds, age_gts, matched_mask=None):
    device = kid_offset_preds.device
    age_gts = age_gts
    age_vmask = age_gts != -1
    if matched_mask is not None:
        age_vmask = age_vmask * matched_mask
    if age_vmask.sum() == 0:
        return 0
    adult_loss = (kid_offset_preds * (age_gts == 0)) ** 2
    teen_thresh = constants.age_threshold['teen']
    teen_loss = ((kid_offset_preds - teen_thresh[1]) * (kid_offset_preds > teen_thresh[2]).float() * (age_gts == 1).float()) ** 2 + ((kid_offset_preds - teen_thresh[1]) * (kid_offset_preds <= teen_thresh[0]).float() * (age_gts == 1).float()) ** 2
    kid_thresh = constants.age_threshold['kid']
    kid_loss = ((kid_offset_preds - kid_thresh[1]) * (kid_offset_preds > kid_thresh[2]).float() * (age_gts == 2).float()) ** 2 + ((kid_offset_preds - kid_thresh[1]) * (kid_offset_preds <= kid_thresh[0]).float() * (age_gts == 2).float()) ** 2
    baby_thresh = constants.age_threshold['baby']
    baby_loss = ((kid_offset_preds - baby_thresh[1]) * (kid_offset_preds > baby_thresh[2]).float() * (age_gts == 3).float()) ** 2 + ((kid_offset_preds - baby_thresh[1]) * (kid_offset_preds <= baby_thresh[0]).float() * (age_gts == 3).float()) ** 2
    age_loss = adult_loss.mean() + teen_loss.mean() + kid_loss.mean() + baby_loss.mean()
    return age_loss


def relative_depth_loss(pred_depths, depth_ids, reorganize_idx, dist_thresh=0.3, uncertainty=None, matched_mask=None):
    depth_ordering_loss = []
    depth_ids = depth_ids
    depth_ids_vmask = depth_ids != -1
    pred_depths_valid = pred_depths[depth_ids_vmask]
    valid_inds = reorganize_idx[depth_ids_vmask]
    depth_ids = depth_ids[depth_ids_vmask]
    if uncertainty is not None:
        uncertainty_valid = uncertainty[depth_ids_vmask]
    for b_ind in torch.unique(valid_inds):
        sample_inds = valid_inds == b_ind
        if matched_mask is not None:
            sample_inds = sample_inds * matched_mask[depth_ids_vmask]
        did_num = sample_inds.sum()
        if did_num > 1:
            pred_depths_sample = pred_depths_valid[sample_inds]
            triu_mask = torch.triu(torch.ones(did_num, did_num), diagonal=1).bool()
            dist_mat = (pred_depths_sample.unsqueeze(0).repeat(did_num, 1) - pred_depths_sample.unsqueeze(1).repeat(1, did_num))[triu_mask]
            did_mat = (depth_ids[sample_inds].unsqueeze(0).repeat(did_num, 1) - depth_ids[sample_inds].unsqueeze(1).repeat(1, did_num))[triu_mask]
            sample_loss = []
            if args().depth_loss_type == 'Piecewise':
                eq_mask = did_mat == 0
                cd_mask = did_mat < 0
                cd_mask[did_mat < 0] = cd_mask[did_mat < 0] * (dist_mat[did_mat < 0] - did_mat[did_mat < 0] * dist_thresh) > 0
                fd_mask = did_mat > 0
                fd_mask[did_mat > 0] = fd_mask[did_mat > 0] * (dist_mat[did_mat > 0] - did_mat[did_mat > 0] * dist_thresh) < 0
                if eq_mask.sum() > 0:
                    sample_loss.append(dist_mat[eq_mask] ** 2)
                if cd_mask.sum() > 0:
                    cd_loss = torch.log(1 + torch.exp(dist_mat[cd_mask]))
                    sample_loss.append(cd_loss)
                if fd_mask.sum() > 0:
                    fd_loss = torch.log(1 + torch.exp(-dist_mat[fd_mask]))
                    sample_loss.append(fd_loss)
            elif args().depth_loss_type == 'Log':
                eq_loss = dist_mat[did_mat == 0] ** 2
                cd_loss = torch.log(1 + torch.exp(dist_mat[did_mat < 0]))
                fd_loss = torch.log(1 + torch.exp(-dist_mat[did_mat > 0]))
                sample_loss = [eq_loss, cd_loss, fd_loss]
            else:
                raise NotImplementedError
            if len(sample_loss) > 0:
                this_sample_loss = torch.cat(sample_loss).mean()
                depth_ordering_loss.append(this_sample_loss)
    if len(depth_ordering_loss) == 0:
        depth_ordering_loss = 0
    else:
        depth_ordering_loss = sum(depth_ordering_loss) / len(depth_ordering_loss)
    return depth_ordering_loss


def match_batch_subject_ids(reorganize_idx, subject_ids, torso_pj2d_errors, a_id, b_id, pj2d_thresh=0.1):
    matched_inds = [[], []]
    a_mask = reorganize_idx == a_id
    b_mask = reorganize_idx == b_id
    all_subject_ids = set(subject_ids[a_mask].cpu().numpy()).intersection(set(subject_ids[b_mask].cpu().numpy()))
    if len(all_subject_ids) == 0:
        return matched_inds
    for ind, sid in enumerate(all_subject_ids):
        a_ind = torch.where(torch.logical_and(subject_ids == sid, a_mask))[0][0]
        b_ind = torch.where(torch.logical_and(subject_ids == sid, b_mask))[0][0]
        a_error, b_error = torso_pj2d_errors[a_ind], torso_pj2d_errors[b_ind]
        if a_error < pj2d_thresh and b_error < pj2d_thresh:
            if a_error > b_error:
                matched_inds[0].append(b_ind)
                matched_inds[1].append(a_ind)
            else:
                matched_inds[0].append(a_ind)
                matched_inds[1].append(b_ind)
    matched_inds[0] = torch.Tensor(matched_inds[0]).long()
    matched_inds[1] = torch.Tensor(matched_inds[1]).long()
    return matched_inds


def relative_depth_scale_loss(pred_depths, img_scale, subject_ids, reorganize_idx, torso_pj2d_errors):
    batch_size = args().batch_size
    rds_ratio = args().relative_depth_scale_aug_ratio
    rds_size = int(batch_size * rds_ratio)
    reorganize_idx_cpu = reorganize_idx.cpu()
    subject_ids_cpu = subject_ids.cpu()
    a_ids = torch.arange(batch_size - 2 * rds_size, batch_size - rds_size)
    b_ids = torch.arange(batch_size - rds_size, batch_size)
    rds_loss = []
    for a_id, b_id in zip(a_ids, b_ids):
        if a_id not in reorganize_idx_cpu or b_id not in reorganize_idx_cpu:
            continue
        matched_inds = match_batch_subject_ids(reorganize_idx_cpu, subject_ids_cpu, torso_pj2d_errors, a_id, b_id)
        if len(matched_inds[0]) == 0 or len(matched_inds[1]) == 0:
            continue
        img_scale_anchor = img_scale[matched_inds[0]].reshape(-1)
        img_scale_learn = img_scale[matched_inds[1]].reshape(-1)
        pred_depths_anchor = pred_depths[matched_inds[0]].detach()
        pred_depths_learn = pred_depths[matched_inds[1]]
        scale_diff = torch.sqrt((torch.div(pred_depths_learn, pred_depths_anchor) - torch.div(img_scale_learn, img_scale_anchor)) ** 2 + 1e-06)
        rds_loss.append(scale_diff)
    if len(rds_loss) == 0:
        return torch.zeros(1, device=pred_depths.device)
    else:
        return torch.cat(rds_loss).mean()


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.gmm_prior = MaxMixturePrior(smpl_prior_path=args().smpl_prior_path, num_gaussians=8, dtype=torch.float32)
        if args().HMloss_type == 'focal':
            args().heatmap_weight /= 1000
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        self.align_inds_MPJPE = np.array([constants.SMPL_ALL_44['L_Hip'], constants.SMPL_ALL_44['R_Hip']])
        self.shape_pca_weight = torch.Tensor([1, 0.64, 0.32, 0.32, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16]).unsqueeze(0).float()
        if args().center3d_loss == 'dynamic':
            self.CM = CenterMap()

    def forward(self, outputs, **kwargs):
        meta_data = outputs['meta_data']
        detect_loss_dict = self._calc_detection_loss(outputs, meta_data)
        detection_flag = outputs['detection_flag'].sum()
        loss_dict = detect_loss_dict
        kp_error = None
        if (detection_flag or args().model_return_loss) and args().calc_mesh_loss:
            mPCKh = _calc_matched_PCKh_(outputs['meta_data']['full_kp2d'].float(), outputs['pj2d'].float(), outputs['meta_data']['valid_masks'][:, 0])
            matched_mask = mPCKh > args().matching_pckh_thresh
            kp_loss_dict, kp_error = self._calc_keypoints_loss(outputs, meta_data, matched_mask)
            loss_dict = dict(loss_dict, **kp_loss_dict)
            params_loss_dict = self._calc_param_loss(outputs, meta_data, matched_mask)
            loss_dict = dict(loss_dict, **params_loss_dict)
            if args().video:
                temp_loss_dict = self._calc_temp_loss(outputs, meta_data)
                loss_dict = dict(loss_dict, **temp_loss_dict)
            if args().estimate_camera:
                camera_loss_dict = self._calc_camera_loss(outputs, meta_data)
                loss_dict = dict(loss_dict, **camera_loss_dict)
        loss_names = list(loss_dict.keys())
        for name in loss_names:
            if isinstance(loss_dict[name], tuple):
                loss_dict[name] = loss_dict[name][0]
            elif isinstance(loss_dict[name], int):
                loss_dict[name] = torch.zeros(1, device=outputs[list(outputs.keys())[0]].device)
            loss_dict[name] = loss_dict[name].mean() * eval('args().{}_weight'.format(name))
        return {'loss_dict': loss_dict, 'kp_error': kp_error}

    def _calc_camera_loss(self, outputs, meta_data):
        camera_loss_dict = {'fovs': 0}
        camera_loss_dict['fovs'] = calc_fov_loss(outputs['fovs'], meta_data['fovs'].squeeze())
        return camera_loss_dict

    def _calc_temp_loss(self, outputs, meta_data):
        temp_loss_dict = {item: (0) for item in ['world_foot', 'world_grots', 'temp_shape_consist']}
        if args().learn_motion_offset3D:
            temp_loss_dict.update({item: (0) for item in ['motion_offsets3D']})
        sequence_mask = outputs['pred_seq_mask']
        if sequence_mask.sum() == 0:
            return temp_loss_dict
        pred_batch_ids = outputs['pred_batch_ids'][sequence_mask].detach().long() - meta_data['batch_ids'][0]
        subject_ids = meta_data['subject_ids'][sequence_mask]
        pred_cams = outputs['cam'][sequence_mask].float()
        clip_frame_ids = meta_data['seq_inds'][pred_batch_ids, 1]
        video_seq_ids = meta_data['seq_inds'][pred_batch_ids, 0]
        sequence_inds = extract_sequence_inds(subject_ids, video_seq_ids, clip_frame_ids)
        if args().dynamic_augment:
            world_cam_masks = meta_data['world_cam_mask'][sequence_mask]
            world_cams_gt = meta_data['world_cams'][sequence_mask]
            world_trans_gts = meta_data['world_root_trans'][sequence_mask]
            pred_world_cams = outputs['world_cams'][sequence_mask].float()
            world_trans_preds = outputs['world_trans'][sequence_mask].float()
            world_global_rots_gt = meta_data['world_global_rots'][sequence_mask]
            world_global_rots_pred = outputs['world_global_rots'][sequence_mask]
            grot_masks = meta_data['valid_masks'][sequence_mask][:, 3]
            valid_world_global_rots_mask = torch.logical_and(grot_masks, world_cam_masks)
            temp_loss_dict['world_grots'] = _calc_world_gros_loss_(world_global_rots_pred, world_global_rots_gt, valid_world_global_rots_mask, sequence_inds)
            temp_loss_dict['wrotsL2'] = 0
            if valid_world_global_rots_mask.sum() > 0:
                temp_loss_dict['wrotsL2'] = batch_smpl_pose_l2_error(world_global_rots_gt[valid_world_global_rots_mask].contiguous(), world_global_rots_pred[valid_world_global_rots_mask].contiguous()).mean()
            temp_loss_dict['world_pj2D'] = batch_kp_2d_l2_loss_old(meta_data['dynamic_kp2ds'], outputs['world_pj2d'])
        if args().learn_temporal_shape_consistency:
            pred_betas = outputs['smpl_betas'][sequence_mask]
            temp_loss_dict['temp_shape_consist'] = calc_temporal_shape_consistency_loss(pred_betas, sequence_inds)
        if args().learn_motion_offset3D:
            pred_motion_offsets = outputs['motion_offsets3D'][sequence_mask]
            traj3D_gts = meta_data['traj3D_gts'][sequence_mask]
            traj2D_gts = meta_data['traj2D_gts'][sequence_mask]
            temp_loss_dict['motion_offsets3D'] = calc_motion_offsets3D_loss(pred_motion_offsets, clip_frame_ids, traj3D_gts)
        return temp_loss_dict

    def _calc_detection_loss(self, outputs, meta_data):
        detect_loss_dict = {}
        all_person_mask = meta_data['all_person_detected_mask']
        if args().calc_mesh_loss and 'center_map' in outputs:
            if all_person_mask.sum() > 0:
                detect_loss_dict['CenterMap'] = focal_loss(outputs['center_map'][all_person_mask], meta_data['centermap'][all_person_mask])
            else:
                detect_loss_dict['CenterMap'] = 0
        reorganize_idx_on_each_gpu = outputs['reorganize_idx'] - outputs['meta_data']['batch_ids'][0]
        if 'center_map_3d' in outputs:
            valid_mask_c3d = meta_data['valid_centermap3d_mask'].squeeze()
            valid_mask_c3d = torch.logical_and(valid_mask_c3d, all_person_mask.squeeze())
            detect_loss_dict['CenterMap_3D'] = 0
            valid_mask_c3d = valid_mask_c3d.reshape(-1)
            if valid_mask_c3d.sum() > 0:
                detect_loss_dict['CenterMap_3D'] = focal_loss_3D(outputs['center_map_3d'][valid_mask_c3d], meta_data['centermap_3d'][valid_mask_c3d])
        return detect_loss_dict

    def _calc_keypoints_loss(self, outputs, meta_data, matched_mask):
        kp_loss_dict, error = {'P_KP2D': 0, 'MPJPE': 0, 'PAMPJPE': 0}, {'3d': {'error': [], 'idx': []}, '2d': {'error': [], 'idx': []}}
        real_2d = meta_data['full_kp2d']
        if 'pj2d' in outputs and args().learn2Dprojection:
            if args().model_version == 3:
                kp_loss_dict['joint_sampler'] = self.joint_sampler_loss(real_2d, outputs['joint_sampler_pred'])
            kp_loss_dict['P_KP2D'] = batch_kp_2d_l2_loss_old(real_2d, outputs['pj2d'])
        kp3d_mask = meta_data['valid_masks'][:, 1]
        if kp3d_mask.sum() > 1 and 'j3d' in outputs:
            kp3d_gt = meta_data['kp_3d'].contiguous()
            preds_kp3d = outputs['j3d'][:, :kp3d_gt.shape[1]].contiguous()
            if not args().model_return_loss and args().PAMPJPE_weight > 0:
                try:
                    pampjpe_each = calc_pampjpe(kp3d_gt[kp3d_mask].contiguous(), preds_kp3d[kp3d_mask].contiguous())
                    kp_loss_dict['PAMPJPE'] = pampjpe_each
                except Exception as exp_error:
                    None
            if args().MPJPE_weight > 0:
                fit_mask = kp3d_mask.bool()
                if fit_mask.sum() > 0:
                    mpjpe_each = calc_mpjpe(kp3d_gt[fit_mask].contiguous(), preds_kp3d[fit_mask].contiguous(), align_inds=self.align_inds_MPJPE)
                    kp_loss_dict['MPJPE'] = mpjpe_each
                    error['3d']['error'].append(mpjpe_each.detach() * 1000)
                    error['3d']['idx'].append(torch.where(fit_mask)[0])
        return kp_loss_dict, error

    def _calc_param_loss(self, outputs, meta_data, matched_mask):
        params_loss_dict = {'Pose': 0, 'Shape': 0}
        _check_params_(meta_data['params'])
        device = outputs['body_pose'].device
        grot_masks, smpl_pose_masks, smpl_shape_masks = meta_data['valid_masks'][:, 3], meta_data['valid_masks'][:, 4], meta_data['valid_masks'][:, 5]
        if grot_masks.sum() > 0:
            params_loss_dict['Pose'] += batch_smpl_pose_l2_error(meta_data['params'][grot_masks, :3].contiguous(), outputs['global_orient'][grot_masks].contiguous()).mean()
        if smpl_pose_masks.sum() > 0:
            params_loss_dict['Pose'] += batch_smpl_pose_l2_error(meta_data['params'][smpl_pose_masks, 3:22 * 3].contiguous(), outputs['body_pose'][smpl_pose_masks, :21 * 3].contiguous()).mean()
        if smpl_shape_masks.sum() > 0:
            smpl_shape_diff = meta_data['params'][smpl_shape_masks, -10:].contiguous() - outputs['smpl_betas'][smpl_shape_masks, :10].contiguous()
            params_loss_dict['Shape'] += torch.norm(smpl_shape_diff * self.shape_pca_weight, p=2, dim=-1).mean() / 20.0
        if args().separate_smil_betas:
            pass
        if (~smpl_shape_masks).sum() > 0:
            params_loss_dict['Shape'] += (outputs['smpl_betas'][~smpl_shape_masks, :10] ** 2).mean() / 80.0
        if args().supervise_cam_params:
            params_loss_dict.update({'Cam': 0})
            cam_mask = meta_data['cam_mask']
            if cam_mask.sum() > 0:
                params_loss_dict['Cam'] += batch_l2_loss(meta_data['cams'][cam_mask], outputs['cam'][cam_mask])
        if args().learn_relative:
            if args().learn_relative_age:
                params_loss_dict['R_Age'] = relative_age_loss(outputs['kid_offsets_pred'], meta_data['depth_info'][:, 0], matched_mask=matched_mask) + kid_offset_loss(outputs['kid_offsets_pred'], meta_data['kid_shape_offsets'], matched_mask=matched_mask) * 2
            if args().learn_relative_depth:
                params_loss_dict['R_Depth'] = relative_depth_loss(outputs['cam_trans'][:, 2], meta_data['depth_info'][:, 3], outputs['reorganize_idx'], matched_mask=matched_mask)
            if args().relative_depth_scale_aug and not args().model_return_loss:
                torso_pj2d_errors = calc_pj2d_error(meta_data['full_kp2d'].clone(), outputs['pj2d'].float().clone(), joint_inds=constants.torso_joint_inds)
                params_loss_dict['R_Depth_scale'] = relative_depth_scale_loss(outputs['cam_trans'][:, 2], meta_data['img_scale'], meta_data['subject_ids'], outputs['reorganize_idx'], torso_pj2d_errors)
        if args().learn_gmm_prior:
            gmm_prior_loss = self.gmm_prior(outputs['body_pose']).mean() / 100.0
            valuable_prior_loss_thresh = 8.0
            gmm_prior_loss[gmm_prior_loss < valuable_prior_loss_thresh] = 0
            params_loss_dict['Prior'] = gmm_prior_loss
        return params_loss_dict

    def joint_sampler_loss(self, real_2d, joint_sampler):
        batch_size = joint_sampler.shape[0]
        joint_sampler = joint_sampler.view(batch_size, -1, 2)
        joint_gt = real_2d[:, constants.joint_sampler_mapper]
        loss = batch_kp_2d_l2_loss(joint_gt, joint_sampler)
        return loss


def kabsch_umeyama(A, B):
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1) ** 2)
    H = (A - EA).T @ (B - EB) / n
    U, D, VT = torch.svd(H)
    c = VarA / torch.trace(torch.diag(D))
    return c


def _calc_world_trans_loss_(preds, gts, vmasks, sequence_inds):
    loss = []
    device = preds.device
    gts, vmasks = gts, vmasks
    for seq_inds in sequence_inds:
        pred, gt, vmask = preds[seq_inds], gts[seq_inds], vmasks[seq_inds]
        if vmask.sum() == 0:
            continue
        N = pred.shape[0]
        first_index, second_index = clip_frame_pairs_indes(N, device)
        scale2align = kabsch_umeyama(gt, pred).detach().clamp(max=10.0)
        pred_aligned = pred * scale2align
        delta_pred_aligned = pred_aligned[first_index] - pred_aligned[second_index]
        delta_gt = gt[first_index] - gt[second_index]
        error = torch.norm(delta_pred_aligned - delta_gt, dim=-1).mean()
        if torch.isnan(error):
            continue
        loss.append(error)
    loss = torch.stack(loss) if len(loss) > 0 else torch.zeros(1, device=device)
    return loss


class Learnable_Loss(nn.Module):
    """docstring for  Learnable_Loss"""

    def __init__(self, ID_num=0):
        super(Learnable_Loss, self).__init__()
        self.loss_class = {'det': ['CenterMap', 'CenterMap_3D'], 'loc': ['Cam', 'init_pj2d', 'cams_init', 'P_KP2D'], 'reg': ['MPJPE', 'PAMPJPE', 'P_KP2D', 'Pose', 'Shape', 'Prior', 'ortho']}
        if args().learn_relative:
            self.loss_class['rel'] = ['R_Age', 'R_Gender', 'R_Weight', 'R_Depth', 'R_Depth_scale']
        if args().video:
            self.loss_class['temp'] = ['temp_rot_consist', 'temp_cam_consist', 'temp_shape_consist']
        if args().dynamic_augment:
            self.loss_class['dynamic'] = ['world_cams_consist', 'world_cams', 'world_pj2D', 'world_foot', 'wrotsL2', 'world_cams_init_consist', 'world_cams_init', 'init_world_pj2d', 'world_grots', 'world_trans']
        if args().learn_motion_offset3D:
            self.loss_class['motion'] = ['motion_offsets3D', 'associate_offsets3D']
        self.all_loss_names = np.concatenate([loss_list for task_name, loss_list in self.loss_class.items()]).tolist()

    def forward(self, outputs, new_training=False):
        loss_dict = outputs['loss_dict']
        if args().model_return_loss and args().calc_mesh_loss and not new_training:
            if args().PAMPJPE_weight > 0 and outputs['detection_flag'].sum() > 0:
                try:
                    kp3d_mask = outputs['meta_data']['valid_masks'][:, 1]
                    kp3d_gt = outputs['meta_data']['kp_3d'][kp3d_mask].contiguous()
                    preds_kp3d = outputs['j3d'][kp3d_mask, :kp3d_gt.shape[1]].contiguous()
                    if len(preds_kp3d) > 0:
                        loss_dict['PAMPJPE'] = calc_pampjpe(kp3d_gt.contiguous().float(), preds_kp3d.contiguous().float()).mean() * args().PAMPJPE_weight
                except Exception as exp_error:
                    None
        if args().model_return_loss and args().dynamic_augment:
            meta_data = outputs['meta_data']
            sequence_mask = outputs['pred_seq_mask']
            pred_batch_ids = outputs['pred_batch_ids'][sequence_mask].detach().long() - meta_data['batch_ids'][0]
            subject_ids = meta_data['subject_ids'][sequence_mask]
            clip_frame_ids = meta_data['seq_inds'][pred_batch_ids, 1]
            video_seq_ids = meta_data['seq_inds'][pred_batch_ids, 0]
            sequence_inds = extract_sequence_inds(subject_ids, video_seq_ids, clip_frame_ids)
            world_cam_masks = meta_data['world_cam_mask'][sequence_mask]
            world_trans_gts = meta_data['world_root_trans'][sequence_mask]
            world_trans_preds = outputs['world_trans'][sequence_mask].float()
            loss_dict['world_trans'] = _calc_world_trans_loss_(world_trans_preds, world_trans_gts, world_cam_masks, sequence_inds)
            loss_dict['world_trans'] = loss_dict['world_trans'] * args().world_trans_weight
        loss_dict = {key: value.mean() for key, value in loss_dict.items() if not isinstance(value, int)}
        loss_list = []
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                if not torch.isnan(value):
                    if value.item() < args().loss_thresh:
                        loss_list.append(value)
                    else:
                        loss_list.append(value / (value.item() / args().loss_thresh))
        loss = sum(loss_list)
        loss_tasks = {}
        for loss_class in self.loss_class:
            loss_tasks[loss_class] = sum([loss_dict[item] for item in self.loss_class[loss_class] if item in loss_dict])
        left_loss = sum([loss_dict[loss_item] for loss_item in loss_dict if loss_item not in self.all_loss_names])
        if left_loss != 0:
            loss_tasks.update({'Others': left_loss})
        outputs['loss_dict'] = dict(loss_tasks, **loss_dict)
        return loss, outputs


def make_input(t, requires_grad=False, need_cuda=True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    if need_cuda:
        inp = inp
    return inp


class AELoss(nn.Module):

    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    def singleTagLoss(self, pred_tag, joints):
        """
        associative embedding loss for one image
        """
        tags = []
        pull = 0
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp)) ** 2)
        num_tags = len(tags)
        if num_tags == 0:
            return make_input(torch.zeros(1).float()), make_input(torch.zeros(1).float())
        elif num_tags == 1:
            return make_input(torch.zeros(1).float()), pull / num_tags
        tags = torch.stack(tags)
        size = num_tags, num_tags
        A = tags.expand(*size)
        B = A.permute(1, 0)
        diff = A - B
        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unkown ae loss type')
        return push / ((num_tags - 1) * num_tags) * 0.5, pull / num_tags

    def forward(self, tags, joints):
        """
        accumulate the tag loss for each image in the batch
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)


class JointsMSELoss(nn.Module):

    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        return loss / num_joints


class Heatmap_AE_loss(nn.Module):

    def __init__(self, num_joints, loss_type_HM='MSE', loss_type_AE='exp'):
        super().__init__()
        self.num_joints = num_joints
        self.heatmaps_loss = HeatmapLoss(loss_type_HM)
        self.heatmaps_loss_factor = 1.0
        self.ae_loss = AELoss(loss_type_AE)
        self.push_loss_factor = 1.0
        self.pull_loss_factor = 1.0

    def forward(self, outputs, heatmaps, joints):
        heatmaps_pred = outputs[:, :self.num_joints]
        tags_pred = outputs[:, self.num_joints:]
        heatmaps_loss = None
        push_loss = None
        pull_loss = None
        if self.heatmaps_loss is not None:
            heatmaps_loss = self.heatmaps_loss(heatmaps_pred, heatmaps)
            heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor
        if self.ae_loss is not None:
            batch_size = tags_pred.size()[0]
            tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)
            push_loss, pull_loss = self.ae_loss(tags_pred, joints)
            push_loss = push_loss * self.push_loss_factor
            pull_loss = pull_loss * self.pull_loss_factor
        return heatmaps_loss, push_loss, pull_loss


class Interperlation_penalty(nn.Module):

    def __init__(self, faces_tensor, df_cone_height=0.5, point2plane=False, penalize_outside=True, max_collisions=8, part_segm_fn=None):
        super(Interperlation_penalty, self).__init__()
        self.pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=df_cone_height, point2plane=point2plane, vectorized=True, penalize_outside=penalize_outside)
        self.coll_loss_weight = 1.0
        self.search_tree = BVH(max_collisions=max_collisions)
        self.body_model_faces = faces_tensor
        if part_segm_fn:
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file, encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            self.tri_filtering_module = FilterFaces(faces_segm=faces_segm, faces_parents=faces_parents)

    def forward(self, vertices):
        pen_loss = 0.0
        batch_size = vertices.shape[0]
        triangles = torch.index_select(vertices, 1, self.body_model_faces).view(batch_size, -1, 3, 3)
        with torch.no_grad():
            collision_idxs = self.search_tree(triangles)
        if self.tri_filtering_module is not None:
            collision_idxs = self.tri_filtering_module(collision_idxs)
        if collision_idxs.ge(0).sum().item() > 0:
            pen_loss = torch.sum(self.coll_loss_weight * self.pen_distance(triangles, collision_idxs))
        return pen_loss


class SMPLifyAnglePrior(nn.Module):

    def __init__(self, dtype=torch.float32, **kwargs):
        super(SMPLifyAnglePrior, self).__init__()
        angle_prior_idxs = np.array([55, 58, 12, 15], dtype=np.int64)
        angle_prior_idxs = torch.tensor(angle_prior_idxs, dtype=torch.long)
        self.register_buffer('angle_prior_idxs', angle_prior_idxs)
        angle_prior_signs = np.array([1, -1, -1, -1], dtype=np.float32 if dtype == torch.float32 else np.float64)
        angle_prior_signs = torch.tensor(angle_prior_signs, dtype=dtype)
        self.register_buffer('angle_prior_signs', angle_prior_signs)

    def forward(self, pose, with_global_pose=False):
        """ Returns the angle prior loss for the given pose
        Args:
            pose: (Bx[23 + 1] * 3) torch tensor with the axis-angle
            representation of the rotations of the joints of the SMPL model.
        Kwargs:
            with_global_pose: Whether the pose vector also contains the global
            orientation of the SMPL model. If not then the indices must be
            corrected.
        Returns:
            A sze (B) tensor containing the angle prior loss for each element
            in the batch.
        """
        angle_prior_idxs = self.angle_prior_idxs - (not with_global_pose) * 3
        return torch.exp(pose[:, angle_prior_idxs] * self.angle_prior_signs).pow(2)


class L2Prior(nn.Module):

    def __init__(self, dtype=DEFAULT_DTYPE, reduction='sum', **kwargs):
        super(L2Prior, self).__init__()

    def forward(self, module_input, *args):
        return torch.sum(module_input.pow(2))


class MultiLossFactory(nn.Module):

    def __init__(self, num_joints):
        super().__init__()
        self.num_joints = num_joints
        self.num_stages = 1
        self.heatmaps_loss = nn.ModuleList([(HeatmapLoss() if with_heatmaps_loss else None) for with_heatmaps_loss in [True]])
        self.heatmaps_loss_factor = [1.0]
        self.ae_loss = nn.ModuleList([(AELoss('exp') if with_ae_loss else None) for with_ae_loss in [True]])
        self.push_loss_factor = [0.001]
        self.pull_loss_factor = [0.001]

    def forward(self, outputs, heatmaps, masks, joints):
        self._forward_check(outputs, heatmaps, masks, joints)
        heatmaps_losses = []
        push_losses = []
        pull_losses = []
        for idx in range(len(outputs)):
            offset_feat = 0
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs[idx][:, :self.num_joints]
                offset_feat = self.num_joints
                heatmaps_loss = self.heatmaps_loss[idx](heatmaps_pred, heatmaps[idx], masks[idx])
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmaps_losses.append(heatmaps_loss)
            else:
                heatmaps_losses.append(None)
            if self.ae_loss[idx]:
                tags_pred = outputs[idx][:, offset_feat:]
                batch_size = tags_pred.size()[0]
                tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)
                push_loss, pull_loss = self.ae_loss[idx](tags_pred, joints[idx])
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]
                push_losses.append(push_loss)
                pull_losses.append(pull_loss)
            else:
                push_losses.append(None)
                pull_losses.append(None)
        return heatmaps_losses, push_losses, pull_losses

    def _forward_check(self, outputs, heatmaps, masks, joints):
        assert isinstance(outputs, list), 'outputs should be a list, got {} instead.'.format(type(outputs))
        assert isinstance(heatmaps, list), 'heatmaps should be a list, got {} instead.'.format(type(heatmaps))
        assert isinstance(masks, list), 'masks should be a list, got {} instead.'.format(type(masks))
        assert isinstance(joints, list), 'joints should be a list, got {} instead.'.format(type(joints))
        assert len(outputs) == self.num_stages, 'len(outputs) and num_stages should been same, got {} vs {}.'.format(len(outputs), self.num_stages)
        assert len(outputs) == len(heatmaps), 'outputs and heatmaps should have same length, got {} vs {}.'.format(len(outputs), len(heatmaps))
        assert len(outputs) == len(masks), 'outputs and masks should have same length, got {} vs {}.'.format(len(outputs), len(masks))
        assert len(outputs) == len(joints), 'outputs and joints should have same length, got {} vs {}.'.format(len(outputs), len(joints))
        assert len(outputs) == len(self.heatmaps_loss), 'outputs and heatmaps_loss should have same length, got {} vs {}.'.format(len(outputs), len(self.heatmaps_loss))
        assert len(outputs) == len(self.ae_loss), 'outputs and ae_loss should have same length, got {} vs {}.'.format(len(outputs), len(self.ae_loss))


def regress_joints_from_vertices(vertices, J_regressor):
    if J_regressor.is_sparse:
        J = torch.stack([torch.sparse.mm(J_regressor, vertices[i]) for i in range(len(vertices))])
    else:
        J = torch.einsum('bik,ji->bjk', [vertices, J_regressor])
    return J


class VertexJointSelector(nn.Module):
    """
    Different from SMPL which directly sellect the face/hand/foot joints as specific vertex points from mesh surface         via torch.index_select(vertices, 1, self.extra_joints_idxs)
    The right joints should be regressed in SMPL-X joints manner via joint regressor. 
    """

    def __init__(self, extra_joints_idxs, J_regressor_extra9, J_regressor_h36m17, dtype=torch.float32, sparse_joint_regressor=False):
        super(VertexJointSelector, self).__init__()
        if not sparse_joint_regressor:
            J_regressor_extra9 = J_regressor_extra9.to_dense()
            J_regressor_h36m17 = J_regressor_h36m17.to_dense()
        self.register_buffer('facial_foot_joints_idxs', extra_joints_idxs)
        self.register_buffer('J_regressor_extra9', J_regressor_extra9)
        self.register_buffer('J_regressor_h36m17', J_regressor_h36m17)

    def forward(self, vertices, joints):
        facial_foot_joints9 = torch.index_select(vertices, 1, self.facial_foot_joints_idxs)
        extra_joints9 = regress_joints_from_vertices(vertices, self.J_regressor_extra9)
        joints_h36m17 = regress_joints_from_vertices(vertices, self.J_regressor_h36m17)
        joints73_17 = torch.cat([joints, facial_foot_joints9, extra_joints9, joints_h36m17], dim=1)
        return joints73_17


def transform_mat(R, t):
    """ Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    """
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    transforms_mat = transform_mat(rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]
    posed_joints = transforms[:, :, :3, 3]
    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    return posed_joints, rel_transforms


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, dtype=torch.float32):
    """ Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    """
    batch_size = betas.shape[0]
    v_shaped = v_template + torch.einsum('bl,mkl->bmk', [betas, shapedirs])
    J = regress_joints_from_vertices(v_shaped, J_regressor)
    dtype = pose.dtype
    posedirs = posedirs.type(dtype)
    ident = torch.eye(3, dtype=dtype, device=J_regressor.device)
    rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3]).type(dtype)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).type(dtype)
    pose_offsets = torch.matmul(pose_feature, posedirs.type(dtype)).view(batch_size, -1, 3)
    v_posed = pose_offsets + v_shaped
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=J_regressor.device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    return verts, J_transformed


class SMPL(nn.Module):

    def __init__(self, model_path, model_type='smpl', dtype=torch.float32):
        super(SMPL, self).__init__()
        self.dtype = dtype
        model_info = torch.load(model_path)
        self.vertex_joint_selector = VertexJointSelector(model_info['extra_joints_index'], model_info['J_regressor_extra9'], model_info['J_regressor_h36m17'], dtype=self.dtype)
        self.register_buffer('faces_tensor', model_info['f'])
        self.register_buffer('v_template', model_info['v_template'])
        if model_type == 'smpl':
            self.register_buffer('shapedirs', model_info['shapedirs'])
        elif model_type == 'smpla':
            self.register_buffer('shapedirs', model_info['smpla_shapedirs'])
        self.register_buffer('J_regressor', model_info['J_regressor'])
        self.register_buffer('posedirs', model_info['posedirs'])
        self.register_buffer('parents', model_info['kintree_table'])
        self.register_buffer('lbs_weights', model_info['weights'])

    def forward(self, betas=None, poses=None, root_align=True, **kwargs):
        """ Forward pass for the SMPL model
            Parameters
            ----------
            betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            Return
            ----------
            outputs: dict, {'verts': vertices of body meshes, (B x 6890 x 3),
                            'joints54': 54 joints of body meshes, (B x 54 x 3), }
                            #'joints_h36m17': 17 joints of body meshes follow h36m skeleton format, (B x 17 x 3)}
        """
        if isinstance(betas, np.ndarray):
            betas = torch.from_numpy(betas).type(self.dtype)
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses).type(self.dtype)
        if poses.shape[-1] == 66:
            poses = torch.cat([poses, torch.zeros_like(poses[..., :6])], -1)
        default_device = self.shapedirs.device
        betas, poses = betas, poses
        vertices, joints = lbs(betas, poses, self.v_template, self.shapedirs, self.posedirs, self.J_regressor, self.parents, self.lbs_weights, dtype=self.dtype)
        joints44_17 = self.vertex_joint_selector(vertices, joints)
        if root_align:
            root_trans = joints44_17[:, [constants.SMPL_ALL_44['R_Hip'], constants.SMPL_ALL_44['L_Hip']]].mean(1).unsqueeze(1)
            joints44_17 = joints44_17 - root_trans
            vertices = vertices - root_trans
        return vertices, joints44_17


class SMPLA_parser(nn.Module):

    def __init__(self, smpla_path, smil_path, baby_thresh=0.8):
        super(SMPLA_parser, self).__init__()
        self.smil_model = SMPL(smil_path, model_type='smpl')
        self.smpla_model = SMPL(smpla_path, model_type='smpla')
        self.baby_thresh = baby_thresh

    def forward(self, betas=None, poses=None, root_align=True, separate_smil_betas=False):
        baby_mask = betas[:, 10] > self.baby_thresh
        if baby_mask.sum() > 0:
            adult_mask = ~baby_mask
            verts, joints = torch.zeros(len(poses), 6890, 3, device=poses.device, dtype=poses.dtype), torch.zeros(len(poses), args().joint_num + 17, 3, device=poses.device, dtype=poses.dtype)
            if separate_smil_betas:
                verts[baby_mask], joints[baby_mask] = self.smil_model(betas=betas[baby_mask, 11:], poses=poses[baby_mask], root_align=root_align)
            else:
                verts[baby_mask], joints[baby_mask] = self.smil_model(betas=betas[baby_mask, :10], poses=poses[baby_mask], root_align=root_align)
            if adult_mask.sum() > 0:
                verts[adult_mask], joints[adult_mask] = self.smpla_model(betas=betas[adult_mask, :11], poses=poses[adult_mask], root_align=root_align)
        else:
            verts, joints = self.smpla_model(betas=betas[:, :11], poses=poses, root_align=root_align)
        return verts, joints


class SMPLX(SMPL):
    """
    sparse_joint_regressor: 
        True: using sparse coo matrix for joint regressor, 
        when batch size = 1, faster (65%, 8.45e-3 v.s. 3e-3) on CPU, while slower (25%, 1.5e-3 v.s. 1.2e-3) on GPU. 
        when batch size >4, on GPU, they cost equal GPU memory, and direct dense matrix multiplation is always faster than the sparse one. 
        Maybe sparse matrix multiplation is not optmized as good as dense matrix multiplation.
             
    """

    def __init__(self, model_path, model_type='smplx', sparse_joint_regressor=True, pca_hand_pose_num=0, flat_hand_mean=True, expression_dim=10, dtype=torch.float32):
        super(SMPLX, self).__init__(model_path, model_type='smpl')
        model_info = torch.load(model_path)
        self.vertex_joint_selector = VertexJointSelector(model_info['extra_joints_index'], model_info['J_regressor_extra9'], model_info['J_regressor_h36m17'], dtype=self.dtype, sparse_joint_regressor=sparse_joint_regressor)
        if not sparse_joint_regressor:
            self.J_regressor = self.J_regressor.to_dense()
        self.expression_dim = expression_dim
        self.register_buffer('expr_dirs', model_info['expr_dirs'][..., :expression_dim])
        self.pca_hand_pose = pca_hand_pose_num > 0
        self.pca_hand_pose_num = pca_hand_pose_num
        if self.pca_hand_pose:
            self.hand_pose_dim = self.pca_hand_pose_num
            self.register_buffer('left_hand_components', model_info['hands_componentsl'][:pca_hand_pose_num])
            self.register_buffer('right_hand_components', model_info['hands_componentsr'][:pca_hand_pose_num])
        else:
            self.hand_pose_dim = 45
        self.flat_hand_mean = flat_hand_mean
        if not self.flat_hand_mean:
            self.register_buffer('left_hand_mean', model_info['hands_meanl'])
            self.register_buffer('right_hand_mean', model_info['hands_meanr'])

    def forward(self, betas=None, poses=None, head_poses=None, expression=None, left_hand_pose=None, right_hand_pose=None, root_align=True, **kwargs):
        """ Forward pass for the SMPL model
            Parameters
            ----------
            betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            head_poses: Bx(3*3), 3 joints including jaw_pose,leye_pose,reye_pose
            expression: Bxexpression_dim, usually first 10 parameters to control the facial expression
            left_hand_pose / right_hand_pose: 
                if self.pca_hand_pose is True, then use PCA hand pose space, (B,self.pca_hand_pose_num)
                    else (B,(15*3)), each finger has 3 joints to control the hand pose. 

            Return
            ----------
            outputs: dict, {'verts': vertices of body meshes, (B x 6890 x 3),
                            'joints54': 73 joints of body meshes, (B x 73 x 3), }
                            #'joints_h36m17': 17 joints of body meshes follow h36m skeleton format, (B x 17 x 3)}
        """
        if isinstance(betas, np.ndarray):
            betas = torch.from_numpy(betas).type(self.dtype)
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses).type(self.dtype)
        if left_hand_pose is None:
            left_hand_pose = torch.zeros(len(poses), self.hand_pose_dim, dtype=poses.dtype, device=poses.device)
        if right_hand_pose is None:
            right_hand_pose = torch.zeros(len(poses), self.hand_pose_dim, dtype=poses.dtype, device=poses.device)
        if head_poses is None:
            head_poses = torch.zeros(len(poses), 3 * 3, dtype=poses.dtype, device=poses.device)
        if expression is None:
            expression = torch.zeros(len(poses), self.expression_dim, dtype=poses.dtype, device=poses.device)
        if self.pca_hand_pose:
            left_hand_pose = torch.einsum('bi,ij->bj', [left_hand_pose, self.left_hand_components])
            right_hand_pose = torch.einsum('bi,ij->bj', [right_hand_pose, self.right_hand_components])
        if not self.flat_hand_mean:
            left_hand_pose = left_hand_pose + self.left_hand_mean
            right_hand_pose = right_hand_pose + self.right_hand_mean
        default_device = self.shapedirs.device
        betas, poses = betas, poses
        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)
        full_pose = torch.cat([poses, head_poses, left_hand_pose, right_hand_pose], dim=1)
        vertices, joints = lbs(shape_components, full_pose, self.v_template, shapedirs, self.posedirs, self.J_regressor, self.parents, self.lbs_weights, dtype=self.dtype)
        joints73_17 = self.vertex_joint_selector(vertices, joints)
        if root_align:
            root_trans = joints73_17[:, [constants.SMPL_ALL_44['R_Hip'], constants.SMPL_ALL_44['L_Hip']]].mean(1).unsqueeze(1)
            joints73_17 = joints73_17 - root_trans
            vertices = vertices - root_trans
        return vertices, joints73_17
        if return_shaped:
            v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)


def create_model(model_type, model_path=None, **kwargs):
    if model_type == 'smpl':
        model_path = args().smpl_model_path if model_path is None else model_path
        return SMPL(model_path, model_type='smpl', **kwargs)
    if model_type == 'smpla':
        return SMPLA_parser(args().smpla_model_path, args().smil_model_path, baby_thresh=args().baby_threshold, **kwargs)
    if model_type == 'smplx':
        model_path = os.path.join(args().smplx_model_folder, 'SMPLX_NEUTRAL.pth') if model_path is None else model_path
        return SMPLX(model_path, **kwargs)


def parse_age_cls_results(age_probs):
    age_preds = torch.ones_like(age_probs).long() * -1
    age_preds[(age_probs <= constants.age_threshold['adult'][2]) & (age_probs > constants.age_threshold['adult'][0])] = 0
    age_preds[(age_probs <= constants.age_threshold['teen'][2]) & (age_probs > constants.age_threshold['teen'][0])] = 1
    age_preds[(age_probs <= constants.age_threshold['kid'][2]) & (age_probs > constants.age_threshold['kid'][0])] = 2
    age_preds[(age_probs <= constants.age_threshold['baby'][2]) & (age_probs > constants.age_threshold['baby'][0])] = 3
    return age_preds


def rot6d_to_rotmat(x):
    x = x.view(-1, 3, 2)
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-06)
    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-06)
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)
    return rot_mats


def quaternion_to_angle_axis(quaternion: 'torch.Tensor') ->torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(quaternion)))
    if not quaternion.shape[-1] == 4:
        raise ValueError('Input must be a tensor of shape Nx4 or 4. Got {}'.format(quaternion.shape))
    q1: 'torch.Tensor' = quaternion[..., 1]
    q2: 'torch.Tensor' = quaternion[..., 2]
    q3: 'torch.Tensor' = quaternion[..., 3]
    sin_squared_theta: 'torch.Tensor' = q1 * q1 + q2 * q2 + q3 * q3
    sin_theta: 'torch.Tensor' = torch.sqrt(sin_squared_theta)
    cos_theta: 'torch.Tensor' = quaternion[..., 0]
    two_theta: 'torch.Tensor' = 2.0 * torch.where(cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta))
    k_pos: 'torch.Tensor' = two_theta / sin_theta
    k_neg: 'torch.Tensor' = 2.0 * torch.ones_like(sin_theta)
    k: 'torch.Tensor' = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)
    angle_axis: 'torch.Tensor' = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-06):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(rotation_matrix)))
    if len(rotation_matrix.shape) > 3:
        raise ValueError('Input size must be a three dimensional tensor. Got {}'.format(rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError('Input size must be a N x 3 x 4  tensor. Got {}'.format(rotation_matrix.shape))
    rmat_t = torch.transpose(rotation_matrix, 1, 2)
    mask_d2 = rmat_t[:, 2, 2] < eps
    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]
    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()
    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0], t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()
    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2], rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()
    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1], rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()
    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)
    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q = q * torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3) * 0.5
    return q


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32, device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def rot6D_to_angular(rot6D):
    batch_size = rot6D.shape[0]
    pred_rotmat = rot6d_to_rotmat(rot6D).view(batch_size, -1, 3, 3)
    pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(batch_size, -1)
    return pose


def batch_orth_proj(X, camera, mode='2d', keep_dim=False):
    camera = camera.view(-1, 1, 3)
    X_camed = X[:, :, :2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:, :, 2].unsqueeze(-1)], -1)
    return X_camed


def convert_cam_to_3d_trans(cams, weight=2.0):
    s, tx, ty = cams[:, 0], cams[:, 1], cams[:, 2]
    depth, dx, dy = 1.0 / s, tx / s, ty / s
    trans3d = torch.stack([dx, dy, depth], 1) * weight
    return trans3d


def convert_kp2d_from_input_to_orgimg(kp2ds, offsets):
    offsets = offsets.float()
    img_pad_size, crop_trbl, pad_trbl = offsets[:, :2], offsets[:, 2:6], offsets[:, 6:10]
    leftTop = torch.stack([crop_trbl[:, 3] - pad_trbl[:, 3], crop_trbl[:, 0] - pad_trbl[:, 0]], 1)
    kp2ds_on_orgimg = (kp2ds[:, :, :2] + 1) * img_pad_size.unsqueeze(1) / 2 + leftTop.unsqueeze(1)
    if kp2ds.shape[-1] == 3:
        kp2ds_on_orgimg = torch.cat([kp2ds_on_orgimg, (kp2ds[:, :, [2]] + 1) * img_pad_size.unsqueeze(1)[:, :, [0]] / 2], -1)
    return kp2ds_on_orgimg


def convert_kp2ds2org_images(projected_outputs, input2orgimg_offsets):
    projected_outputs['pj2d_org'] = convert_kp2d_from_input_to_orgimg(projected_outputs['pj2d'], input2orgimg_offsets)
    if 'verts_camed' in projected_outputs:
        projected_outputs['verts_camed_org'] = convert_kp2d_from_input_to_orgimg(projected_outputs['verts_camed'], input2orgimg_offsets)
    if 'pj2d_h36m17' in projected_outputs:
        projected_outputs['pj2d_org_h36m17'] = convert_kp2d_from_input_to_orgimg(projected_outputs['pj2d_h36m17'], input2orgimg_offsets)
    return projected_outputs


def convert_scale_to_depth(scale, fovs):
    return fovs / (scale + 1e-06)


class SMPLWrapper(nn.Module):

    def __init__(self):
        super(SMPLWrapper, self).__init__()
        logging.info('Building SMPL family for relative learning in temporal!')
        self.smpl_model = create_model(args().smpl_model_type)
        self.part_name = ['cam', 'global_orient', 'body_pose', 'smpl_betas']
        self.part_idx = [3, 6, (args().smpl_joint_num - 1) * 6, 11 if not args().separate_smil_betas else 21]
        self.params_num = np.array(self.part_idx).sum()

    def parse_params_pred(self, params_pred, without_cam=False):
        params_dict = self.pack_params_dict(params_pred, without_cam=without_cam)
        params_dict['smpl_betas'], cls_dict = self.process_betas(params_dict['smpl_betas'])
        vertices, joints44_17 = self.smpl_model(betas=params_dict['smpl_betas'], poses=params_dict['smpl_thetas'], separate_smil_betas=args().separate_smil_betas)
        return vertices, joints44_17, params_dict, cls_dict

    def forward(self, outputs, meta_data, calc_pj2d_org=True):
        vertices, joints44_17, params_dict, cls_dict = self.parse_params_pred(outputs['params_pred'])
        if 'world_global_rots' in outputs:
            world_vertices, world_joints44_17 = self.smpl_model(betas=params_dict['smpl_betas'].detach(), poses=torch.cat([outputs['world_global_rots'], params_dict['smpl_thetas'][:, 3:].detach()], 1), separate_smil_betas=args().separate_smil_betas)
            outputs.update({'world_verts': world_vertices, 'world_j3d': world_joints44_17[:, :args().joint_num], 'world_joints_h36m17': world_joints44_17[:, args().joint_num:]})
        outputs.update({'verts': vertices, 'j3d': joints44_17[:, :args().joint_num], 'joints_h36m17': joints44_17[:, args().joint_num:], **params_dict, **cls_dict})
        outputs.update(vertices_kp3d_projection(outputs['j3d'], outputs['joints_h36m17'], outputs['cam'], input2orgimg_offsets=meta_data['offsets'] if calc_pj2d_org else None, presp=args().perspective_proj, vertices=outputs['verts'] if args().compute_verts_org else None))
        if args().dynamic_augment:
            dyna_pouts = vertices_kp3d_projection(outputs['world_j3d'].detach(), outputs['joints_h36m17'].detach(), outputs['world_cams'], input2orgimg_offsets=meta_data['offsets'] if calc_pj2d_org else None, presp=args().perspective_proj, vertices=outputs['verts'] if args().compute_verts_org else None)
            outputs.update({'world_pj2d': dyna_pouts['pj2d'], 'world_trans': dyna_pouts['cam_trans'], 'world_joints_h36m17': dyna_pouts['pj2d_h36m17']})
            if args().compute_verts_org:
                outputs.update({'world_verts_camed_org': dyna_pouts['verts_camed_org']})
        return outputs

    def pack_params_dict(self, params_pred, without_cam=False):
        idx_list, params_dict = [0], {}
        for i, (idx, name) in enumerate(zip(self.part_idx, self.part_name)):
            if without_cam and i == 0:
                idx_list.append(0)
                continue
            idx_list.append(idx_list[i] + idx)
            params_dict[name] = params_pred[:, idx_list[i]:idx_list[i + 1]].contiguous()
        if params_dict['global_orient'].shape[-1] == 6:
            params_dict['body_pose'] = rot6D_to_angular(params_dict['body_pose'])
            params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient'])
        N = params_dict['body_pose'].shape[0]
        params_dict['body_pose'] = torch.cat([params_dict['body_pose'], torch.zeros(N, 6)], 1)
        params_dict['smpl_thetas'] = torch.cat([params_dict['global_orient'], params_dict['body_pose']], 1)
        return params_dict

    def process_betas(self, betas_pred):
        if not args().learn_relative_age:
            betas_pred[:, 10] = 0
        kid_offsets = betas_pred[:, 10]
        Age_preds = parse_age_cls_results(kid_offsets)
        betas_pred = betas_pred[:, :10]
        cls_dict = {'Age_preds': Age_preds, 'kid_offsets_pred': kid_offsets}
        return betas_pred, cls_dict


def _check_params_pred_(params_pred_shape, batch_length):
    assert len(params_pred_shape) == 2, logging.error('outputs[params_pred] dimension less than 2, is {}'.format(len(params_pred_shape)))
    assert params_pred_shape[0] == batch_length, logging.error('sampled length not equal.')


def _check_params_sampling_(param_maps_shape, dim_start, dim_end, batch_ids, sampler_flat_inds_i):
    assert len(param_maps_shape) == 3, logging.error('During parameter sampling, param_maps dimension is not equal 3, is {}'.format(len(param_maps_shape)))
    assert param_maps_shape[2] > dim_end >= dim_start, logging.error('During parameter sampling, param_maps dimension -1 is not larger than dim_end and dim_start, they are {},{},{}'.format(param_maps_shape[-1], dim_end, dim_start))
    assert (batch_ids >= param_maps_shape[0]).sum() == 0, logging.error('During parameter sampling, batch_ids {} out of boundary, param_maps_shape[0] is {}'.format(batch_ids, param_maps_shape[0]))
    assert (sampler_flat_inds_i >= param_maps_shape[1]).sum() == 0, logging.error('During parameter sampling, sampler_flat_inds_i {} out of boundary, param_maps_shape[1] is {}'.format(sampler_flat_inds_i, param_maps_shape[1]))


def match_trajectory_gts(traj3D_gts, traj2D_gts, traj_sids, subject_ids, batch_ids):
    subject_num = len(subject_ids)
    traj3D_gts_matched = torch.ones(subject_num, args().temp_clip_length, 3).float() * -2.0
    traj2D_gts_matched = torch.ones(subject_num, args().temp_clip_length, 2).float() * -2.0
    parallel_start_id = int(batch_ids[0].item())
    for ind, (sid, bid) in enumerate(zip(subject_ids, batch_ids)):
        if sid == -1:
            continue
        input_batch_ind = int(bid.item()) // args().temp_clip_length - parallel_start_id // args().temp_clip_length
        input_subject_ind = torch.where(traj_sids[input_batch_ind, :, 0, 3] == sid)[0]
        try:
            if len(input_subject_ind) > 0:
                traj2D_gts_matched[ind] = traj2D_gts[input_batch_ind, input_subject_ind]
                traj3D_gts_matched[ind] = traj3D_gts[input_batch_ind, input_subject_ind]
        except:
            None
    return traj3D_gts_matched, traj2D_gts_matched


def flatten_inds(coords):
    coords = torch.clamp(coords, 0, args().centermap_size - 1)
    return coords[:, 0].long() * args().centermap_size + coords[:, 1].long()


def match_with_2d_centers(center_gts_info, center_preds_info, device, is_training):
    vgt_batch_ids, vgt_person_ids, vgt_centers = center_gts_info
    vpred_batch_ids, flat_inds, cyxs, top_score = center_preds_info
    mc = {key: [] for key in ['batch_ids', 'flat_inds', 'person_ids', 'conf']}
    if args().match_preds_to_gts_for_supervision:
        for match_ind in torch.arange(len(vgt_batch_ids)):
            batch_id, person_id, center_gt = vgt_batch_ids[match_ind], vgt_person_ids[match_ind], vgt_centers[match_ind]
            pids = torch.where(vpred_batch_ids == batch_id)[0]
            if len(pids) == 0:
                continue
            closet_center_ind = pids[torch.argmin(torch.norm(cyxs[pids].float() - center_gt[None].float(), dim=-1))]
            center_matched = cyxs[closet_center_ind].long()
            cy, cx = torch.clamp(center_matched, 0, args().centermap_size - 1)
            flat_ind = cy * args().centermap_size + cx
            mc['batch_ids'].append(batch_id)
            mc['flat_inds'].append(flat_ind)
            mc['person_ids'].append(person_id)
            mc['conf'].append(top_score[closet_center_ind])
        keys_list = list(mc.keys())
        for key in keys_list:
            if key != 'conf':
                mc[key] = torch.Tensor(mc[key]).long()
            if args().max_supervise_num != -1 and is_training:
                mc[key] = mc[key][:args().max_supervise_num]
    if not args().match_preds_to_gts_for_supervision or len(mc['batch_ids']) == 0:
        mc['batch_ids'] = vgt_batch_ids.long()
        mc['flat_inds'] = flatten_inds(vgt_centers.long())
        mc['person_ids'] = vgt_person_ids.long()
        mc['conf'] = torch.zeros(len(vgt_person_ids))
    return mc


def convert_cam_params_to_centermap_coords(cam_params):
    center_coords = torch.ones_like(cam_params)
    center_coords[:, 1:] = cam_params[:, 1:].clone()
    cam3dmap_anchors = cam3dmap_anchor[None]
    if len(cam_params) != 0:
        center_coords[:, 0] = torch.argmin(torch.abs(cam_params[:, [0]].repeat(1, scale_num) - cam3dmap_anchors), dim=1).float() / args().centermap_size * 2.0 - 1.0
    return center_coords


def process_gt_center(center_normed):
    valid_mask = center_normed[:, :, 0] > -1
    valid_inds = torch.where(valid_mask)
    valid_batch_inds, valid_person_ids = valid_inds[0], valid_inds[1]
    center_normed[valid_inds] = torch.max(center_normed[valid_inds], torch.ones_like(center_normed[valid_inds]) * -1)
    center_gt = ((center_normed + 1) / 2 * (args().centermap_size - 1)).long()
    center_gt = torch.min(center_gt, torch.ones_like(center_gt) * (args().centermap_size - 1))
    center_gt_valid = center_gt[valid_mask]
    return valid_batch_inds, valid_person_ids, center_gt_valid


def match_with_3d_2d_centers(meta_data, outputs, cfg):
    with_2d_matching = cfg['with_2d_matching']
    is_training = cfg['is_training']
    cam_mask = meta_data['cam_mask']
    batch_size = len(cam_mask)
    pred_batch_ids = outputs['pred_batch_ids']
    pred_czyxs = outputs['pred_czyxs']
    top_score = outputs['top_score']
    device = outputs['center_map_3d'].device
    center_gts_info_3d = parse_gt_center3d(cam_mask, meta_data['cams'])
    person_centers = meta_data['person_centers'].clone()
    person_centers[cam_mask] = -2.0
    center_gts_info_2d = process_gt_center(person_centers)
    vgt_batch_ids, vgt_person_ids, vgt_centers = center_gts_info_2d
    vgt_batch_ids_3d, vgt_person_ids_3d, vgt_czyxs = center_gts_info_3d
    mc = {key: [] for key in ['batch_ids', 'matched_ids', 'person_ids', 'conf']}
    for match_ind in torch.arange(len(vgt_batch_ids_3d)):
        batch_id, person_id, center_gt = vgt_batch_ids_3d[match_ind], vgt_person_ids_3d[match_ind], vgt_czyxs[match_ind]
        pids = torch.where(pred_batch_ids == batch_id)[0]
        if len(pids) == 0:
            continue
        center_dist_3d = torch.norm(pred_czyxs[pids].float() - center_gt[None].float(), dim=-1)
        matched_pred_id = pids[torch.argmin(center_dist_3d)]
        mc['batch_ids'].append(batch_id)
        mc['matched_ids'].append(matched_pred_id)
        mc['person_ids'].append(person_id)
        mc['conf'].append(top_score[matched_pred_id])
    for match_ind in torch.arange(len(vgt_batch_ids)):
        batch_id, person_id, center_gt = vgt_batch_ids[match_ind], vgt_person_ids[match_ind], vgt_centers[match_ind]
        pids = torch.where(pred_batch_ids == batch_id)[0]
        if len(pids) == 0:
            continue
        matched_pred_id = pids[torch.argmin(torch.norm(pred_czyxs[pids, 1:].float() - center_gt[None].float(), dim=-1))]
        center_matched = pred_czyxs[matched_pred_id].long()
        mc['batch_ids'].append(batch_id)
        mc['matched_ids'].append(matched_pred_id)
        mc['person_ids'].append(person_id)
        mc['conf'].append(top_score[matched_pred_id])
    if args().eval_2dpose:
        for inds, (batch_id, person_id, center_gt) in enumerate(zip(vgt_batch_ids, vgt_person_ids, vgt_centers)):
            if batch_id in pred_batch_ids:
                center_pred = pred_czyxs[pred_batch_ids == batch_id]
                matched_id = torch.argmin(torch.norm(center_pred[:, 1:].float() - center_gt[None].float(), dim=-1))
                matched_pred_id = np.where((pred_batch_ids == batch_id).cpu())[0][matched_id]
                mc['matched_ids'].append(matched_pred_id)
                mc['batch_ids'].append(batch_id)
                mc['person_ids'].append(person_id)
    if len(mc['matched_ids']) == 0:
        mc.update({'batch_ids': [0], 'matched_ids': [0], 'person_ids': [0], 'conf': [0]})
    keys_list = list(mc.keys())
    for key in keys_list:
        if key == 'conf':
            mc[key] = torch.Tensor(mc[key])
        else:
            mc[key] = torch.Tensor(mc[key]).long()
        if args().max_supervise_num != -1 and is_training:
            mc[key] = mc[key][:args().max_supervise_num]
    return mc['batch_ids'], mc['person_ids'], mc['matched_ids'], mc['conf']


def get_bbx_overlap(p1, p2, imgpath, baseline=None):
    min_p1 = np.min(p1, axis=0)
    min_p2 = np.min(p2, axis=0)
    max_p1 = np.max(p1, axis=0)
    max_p2 = np.max(p2, axis=0)
    bb1 = {}
    bb2 = {}
    bb1['x1'] = min_p1[0]
    bb1['x2'] = max_p1[0]
    bb1['y1'] = min_p1[1]
    bb1['y2'] = max_p1[1]
    bb2['x1'] = min_p2[0]
    bb2['x2'] = max_p2[0]
    bb2['y1'] = min_p2[1]
    bb2['y2'] = max_p2[1]
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def greedy_matching_kp2ds(pred_kps, gtkp, valid_mask, iou_thresh=0.05, valid=None):
    """
    matches groundtruth keypoints to the detection by considering all possible matchings.
    :return: best possible matching, a list of tuples, where each tuple corresponds to one match of pred_person.to gt_person.
            the order within one tuple is as follows (idx_pred_kps, idx_gt_kps)
    """
    predList = np.arange(len(pred_kps))
    gtList = np.arange(len(gtkp))
    combs = list(product(predList, gtList))
    errors_per_pair = {}
    errors_per_pair_list = []
    for comb in combs:
        vmask = valid_mask[comb[1]]
        if vmask.sum() < 1:
            None
        errors_per_pair[str(comb)] = np.linalg.norm(pred_kps[comb[0]][vmask, :2] - gtkp[comb[1]][vmask, :2], 2)
        errors_per_pair_list.append(errors_per_pair[str(comb)])
    gtAssigned = np.zeros((len(gtkp),), dtype=bool)
    opAssigned = np.zeros((len(pred_kps),), dtype=bool)
    errors_per_pair_list = np.array(errors_per_pair_list)
    bestMatch = []
    excludedGtBecauseInvalid = []
    falsePositiveCounter = 0
    while np.sum(gtAssigned) < len(gtAssigned) and np.sum(opAssigned) + falsePositiveCounter < len(pred_kps):
        found = False
        falsePositive = False
        while not found:
            if sum(np.inf == errors_per_pair_list) == len(errors_per_pair_list):
                None
            minIdx = np.argmin(errors_per_pair_list)
            minComb = combs[minIdx]
            iou = get_bbx_overlap(pred_kps[minComb[0]], gtkp[minComb[1]])
            if not opAssigned[minComb[0]] and not gtAssigned[minComb[1]] and iou >= iou_thresh:
                found = True
                errors_per_pair_list[minIdx] = np.inf
            else:
                errors_per_pair_list[minIdx] = np.inf
                if iou < iou_thresh:
                    found = True
                    falsePositive = True
                    falsePositiveCounter += 1
        if not valid is None:
            if valid[minComb[1]]:
                if not falsePositive:
                    bestMatch.append(minComb)
                    opAssigned[minComb[0]] = True
                    gtAssigned[minComb[1]] = True
            else:
                gtAssigned[minComb[1]] = True
                excludedGtBecauseInvalid.append(minComb[1])
        elif not falsePositive:
            bestMatch.append(minComb)
            opAssigned[minComb[0]] = True
            gtAssigned[minComb[1]] = True
    bestMatch = np.array(bestMatch)
    opAssigned = []
    gtAssigned = []
    for pair in bestMatch:
        opAssigned.append(pair[0])
        gtAssigned.append(pair[1])
    opAssigned.sort()
    gtAssigned.sort()
    falsePositives = []
    misses = []
    opIds = np.arange(len(pred_kps))
    notAssignedIds = np.setdiff1d(opIds, opAssigned)
    for notAssignedId in notAssignedIds:
        falsePositives.append(notAssignedId)
    gtIds = np.arange(len(gtList))
    notAssignedIdsGt = np.setdiff1d(gtIds, gtAssigned)
    for notAssignedIdGt in notAssignedIdsGt:
        if not valid is None:
            if valid[notAssignedIdGt]:
                misses.append(notAssignedIdGt)
            else:
                excludedGtBecauseInvalid.append(notAssignedIdGt)
        else:
            misses.append(notAssignedIdGt)
    return bestMatch, falsePositives, misses


def match_with_kp2ds(meta_data, outputs, cfg):
    pred_kp2ds = outputs['pj2d']
    device = pred_kp2ds.device
    pred_batch_ids = outputs['pred_batch_ids'].long()
    body_center_confs = outputs['top_score']
    gt_kp2ds = meta_data['full_kp2d']
    valid_mask = meta_data['valid_masks'][:, :, 0]
    vgt_batch_ids, vgt_person_ids = torch.where(valid_mask)
    vgt_kp2ds = gt_kp2ds[vgt_batch_ids, vgt_person_ids]
    vgt_valid_mask = ((vgt_kp2ds != -2.0).sum(-1) == 2).sum(-1) > 0
    if vgt_valid_mask.sum() != len(vgt_valid_mask):
        vgt_kp2ds, vgt_batch_ids, vgt_person_ids = vgt_kp2ds[vgt_valid_mask], vgt_batch_ids[vgt_valid_mask], vgt_person_ids[vgt_valid_mask]
    mc = {key: [] for key in ['batch_ids', 'matched_ids', 'person_ids', 'conf']}
    matching_batch_ids = torch.unique(pred_batch_ids)
    for batch_id in matching_batch_ids:
        gt_ids = torch.where(vgt_batch_ids == batch_id)[0]
        if len(gt_ids) == 0:
            continue
        pred_ids = torch.where(pred_batch_ids == batch_id)[0]
        pred_kp2ds_matching = pred_kp2ds[pred_ids].detach().cpu().numpy()
        gt_kp2ds_matching = vgt_kp2ds[gt_ids].cpu().numpy()
        gt_valid_mask_matching = (gt_kp2ds_matching == -2.0).sum(-1) == 0
        bestMatch, falsePositives, misses = greedy_matching_kp2ds(pred_kp2ds_matching, gt_kp2ds_matching, gt_valid_mask_matching)
        for pid, gtid in bestMatch:
            matched_gt_id = gt_ids[gtid]
            gt_batch_id = vgt_batch_ids[matched_gt_id]
            gt_person_id = vgt_person_ids[matched_gt_id]
            pred_batch_id = pred_ids[pid]
            mc['batch_ids'].append(gt_batch_id)
            mc['person_ids'].append(gt_person_id)
            mc['matched_ids'].append(pred_batch_id)
            mc['conf'].append(body_center_confs[int(pred_batch_id)])
    if len(mc['matched_ids']) == 0:
        None
        mc.update({'batch_ids': [0], 'matched_ids': [0], 'person_ids': [0], 'conf': [0]})
    keys_list = list(mc.keys())
    for key in keys_list:
        if key == 'conf':
            mc[key] = torch.Tensor(mc[key])
        else:
            mc[key] = torch.Tensor(mc[key]).long()
        if args().max_supervise_num != -1 and cfg['is_training']:
            mc[key] = mc[key][:args().max_supervise_num]
    return mc['batch_ids'], mc['person_ids'], mc['matched_ids'], mc['conf']


matching_gts2preds = {'kp2ds': match_with_kp2ds, '3D+2D_center': match_with_3d_2d_centers, '2D_center': match_with_2d_centers}


def reorganize_gts(meta_data, key_list, batch_ids):
    for key in key_list:
        if key in meta_data:
            try:
                if isinstance(meta_data[key], torch.Tensor):
                    meta_data[key] = meta_data[key][batch_ids]
                elif isinstance(meta_data[key], list):
                    meta_data[key] = [meta_data[key][ind] for ind in batch_ids]
            except:
                None
    return meta_data


def reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids):
    exclude_keys += gt_keys
    outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
    info_vis = []
    for key, item in meta_data.items():
        if key not in exclude_keys:
            info_vis.append(key)
    meta_data = reorganize_gts(meta_data, info_vis, batch_ids)
    for gt_key in gt_keys:
        if gt_key in meta_data:
            try:
                meta_data[gt_key] = meta_data[gt_key][batch_ids, person_ids]
            except:
                None
    return outputs, meta_data


def reorganize_gts_cpu(meta_data, key_list, batch_ids):
    for key in key_list:
        if key in meta_data:
            if isinstance(meta_data[key], torch.Tensor):
                meta_data[key] = meta_data[key].cpu()[batch_ids.cpu()]
            elif isinstance(meta_data[key], list):
                meta_data[key] = [meta_data[key][ind] for ind in batch_ids]
    return meta_data


def suppressing_silimar_mesh_and_2D_center(params_preds, pred_batch_ids, pred_czyxs, top_score, rot_dim=6, center2D_thresh=5, pose_thresh=2.5):
    pose_params_preds = params_preds[:, args().cam_dim:args().cam_dim + 22 * rot_dim]
    N = len(pred_czyxs)
    center2D_similarity = torch.norm((pred_czyxs[:, 1:].unsqueeze(1).repeat(1, N, 1) - pred_czyxs[:, 1:].unsqueeze(0).repeat(N, 1, 1)).float(), p=2, dim=-1)
    same_batch_id_mask = pred_batch_ids.unsqueeze(1).repeat(1, N) == pred_batch_ids.unsqueeze(0).repeat(N, 1)
    center2D_similarity[~same_batch_id_mask] = center2D_thresh + 1
    similarity = center2D_similarity <= center2D_thresh
    center_similar_inds = torch.where(similarity.sum(-1) > 1)[0]
    for s_inds in center_similar_inds:
        if rot_dim == 6:
            pose_angulars = rot6D_to_angular(pose_params_preds[similarity[s_inds]])
            pose_angular_base = rot6D_to_angular(pose_params_preds[s_inds].unsqueeze(0)).repeat(len(pose_angulars), 1)
        elif rot_dim == 3:
            pose_angulars = pose_params_preds[similarity[s_inds]]
            pose_angular_base = pose_params_preds[s_inds].unsqueeze(0).repeat(len(pose_angulars))
        pose_similarity = batch_smpl_pose_l2_error(pose_angulars, pose_angular_base)
        sim_past = similarity[s_inds].clone()
        similarity[s_inds, sim_past] = pose_similarity < pose_thresh
    score_map = similarity * top_score.unsqueeze(0).repeat(N, 1)
    nms_inds = torch.argmax(score_map, 1) == torch.arange(N)
    return [item[nms_inds] for item in [pred_batch_ids, pred_czyxs, top_score]], nms_inds


class ResultParser(nn.Module):

    def __init__(self, with_smpl_parser=True):
        super(ResultParser, self).__init__()
        self.map_size = args().centermap_size
        self.with_smpl_parser = with_smpl_parser
        if args().calc_smpl_mesh and with_smpl_parser:
            self.params_map_parser = SMPLWrapper()
        self.centermap_parser = CenterMap()
        self.match_preds_to_gts_for_supervision = args().match_preds_to_gts_for_supervision

    def matching_forward(self, outputs, meta_data, cfg):
        if args().BEV_matching_gts2preds == 'kp2ds':
            outputs = self.params_map_parser(outputs, meta_data, calc_pj2d_org=False)
            if args().model_version in [6, 8, 9]:
                outputs, meta_data = self.match_params_new(outputs, meta_data, cfg)
            else:
                outputs, meta_data = self.match_params(outputs, meta_data, cfg)
        else:
            if args().model_version in [6, 8, 9]:
                outputs, meta_data = self.match_params_new(outputs, meta_data, cfg)
            else:
                outputs, meta_data = self.match_params(outputs, meta_data, cfg)
            if 'params_pred' in outputs and self.with_smpl_parser and args().calc_smpl_mesh:
                outputs = self.params_map_parser(outputs, meta_data)
        if 'detection_flag' not in outputs:
            outputs['detection_flag'] = self.determine_detection_flag(outputs, meta_data)
        return outputs, meta_data

    @torch.no_grad()
    def parsing_forward(self, outputs, meta_data, cfg):
        outputs, meta_data = self.parse_maps(outputs, meta_data, cfg)
        if 'params_pred' in outputs and self.with_smpl_parser:
            outputs = self.params_map_parser(outputs, meta_data)
        if 'detection_flag' not in outputs:
            outputs['detection_flag'] = self.determine_detection_flag(outputs, meta_data)
        return outputs, meta_data

    def determine_detection_flag(self, outputs, meta_data):
        detected_ids = torch.unique(outputs['reorganize_idx'])
        detection_flag = torch.Tensor([(batch_id in detected_ids) for batch_id in meta_data['batch_ids']])
        return detection_flag

    def match_params_new(self, outputs, meta_data, cfg):
        gt_keys = ['params', 'full_kp2d', 'kp_3d', 'valid_masks', 'subject_ids', 'verts', 'cam_mask', 'kid_shape_offsets', 'root_trans_cam', 'cams']
        if args().learn_relative:
            gt_keys += ['depth_info']
        if args().learn_cam_with_fbboxes:
            gt_keys += ['full_body_bboxes']
        exclude_keys = ['heatmap', 'centermap', 'AE_joints', 'person_centers', 'params_pred', 'all_person_detected_mask', 'person_scales', 'dynamic_supervise']
        if cfg['with_nms']:
            outputs = suppressing_duplicate_mesh(outputs)
        batch_ids, person_ids, matched_pred_ids, center_confs = matching_gts2preds[args().BEV_matching_gts2preds](meta_data, outputs, cfg)
        outputs['params_pred'] = outputs['params_pred'][matched_pred_ids]
        if args().video:
            outputs['motion_offsets'] = outputs['motion_offsets'][matched_pred_ids]
            exclude_keys += ['traj3D_gts', 'traj2D_gts', 'Tj_flag', 'traj_gt_ids']
        for center_key in ['pred_batch_ids', 'pred_czyxs', 'top_score']:
            outputs[center_key] = outputs[center_key][matched_pred_ids]
        outputs['pred_batch_ids'] += meta_data['batch_ids'][0]
        exclude_keys += ['centermap_3d', 'valid_centermap3d_mask']
        outputs, meta_data = reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids)
        outputs['center_preds'] = outputs['pred_czyxs'] * args().input_size / args().centermap_size
        outputs['center_confs'] = center_confs
        if args().BEV_matching_gts2preds == 'kp2ds':
            output_keys = ['verts', 'j3d', 'joints_h36m17', 'cam', 'global_orient', 'body_pose', 'smpl_betas', 'smpl_thetas', 'Age_preds', 'kid_offsets_pred', 'cam_trans', 'pj2d', 'pj2d_h36m17', 'verts_camed']
            for key in output_keys:
                outputs[key] = outputs[key][matched_pred_ids]
            outputs = convert_kp2ds2org_images(outputs, meta_data['offsets'])
        if 'traj3D_gts' in meta_data:
            meta_data['traj3D_gts'], meta_data['traj2D_gts'] = match_trajectory_gts(meta_data['traj3D_gts'], meta_data['traj2D_gts'], meta_data['traj_gt_ids'], meta_data['subject_ids'], meta_data['batch_ids'])
        return outputs, meta_data

    def match_params(self, outputs, meta_data, cfg):
        gt_keys = ['params', 'full_kp2d', 'kp_3d', 'subject_ids', 'valid_masks', 'verts', 'cam_mask', 'kid_shape_offsets', 'root_trans_cam', 'cams']
        if args().learn_relative:
            gt_keys += ['depth_info']
        exclude_keys = ['heatmap', 'centermap', 'AE_joints', 'person_centers', 'all_person_detected_mask']
        center_gts_info = process_gt_center(meta_data['person_centers'])
        center_preds_info = self.centermap_parser.parse_centermap(outputs['center_map'])
        mc_centers = matching_gts2preds['2D_center'](center_gts_info, center_preds_info, outputs['center_map'].device, cfg['is_training'])
        batch_ids, flat_inds, person_ids = mc_centers['batch_ids'], mc_centers['flat_inds'], mc_centers['person_ids']
        if 'params_maps' in outputs and 'params_pred' not in outputs:
            outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
        if 'uncertainty_map' in outputs:
            outputs['uncertainty_pred'] = torch.sqrt(self.parameter_sampling(outputs['uncertainty_map'], batch_ids, flat_inds, use_transform=True) ** 2) + 1
        if 'reid_map' in outputs:
            outputs['reid_embeds'] = self.parameter_sampling(outputs['reid_map'], batch_ids, flat_inds, use_transform=True)
        if 'joint_sampler_maps' in outputs:
            outputs['joint_sampler_pred'] = self.parameter_sampling(outputs['joint_sampler_maps'], batch_ids, flat_inds, use_transform=True)
            outputs['joint_sampler'] = self.parameter_sampling(outputs['joint_sampler_maps_filtered'], batch_ids, flat_inds, use_transform=True)
            if 'params_pred' in outputs:
                _check_params_pred_(outputs['params_pred'].shape, len(batch_ids))
                outputs['params_pred'] = self.adjust_to_joint_level_sampling(outputs['params_pred'], outputs['joint_sampler'], outputs['params_maps'], batch_ids)
        outputs, meta_data = reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids)
        outputs['center_preds'] = torch.stack([flat_inds % args().centermap_size, flat_inds // args().centermap_size], 1) * args().input_size / args().centermap_size
        return outputs, meta_data

    def adjust_to_joint_level_sampling(self, param_preds, joint_sampler, param_maps, batch_ids):
        sampler_flat_inds = self.process_joint_sampler(joint_sampler)
        batch, channel = param_maps.shape[:2]
        param_maps = param_maps.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
        for inds, joint_inds in enumerate(constants.joint_sampler_relationship):
            start_inds = joint_inds * args().rot_dim + args().cam_dim
            end_inds = start_inds + args().rot_dim
            _check_params_sampling_(param_maps.shape, start_inds, end_inds, batch_ids, sampler_flat_inds[inds])
            param_preds[..., start_inds:end_inds] = param_maps[..., start_inds:end_inds][batch_ids, sampler_flat_inds[inds]].contiguous()
        return param_preds

    def process_joint_sampler(self, joint_sampler, thresh=0.999):
        joint_sampler = torch.clamp(joint_sampler, -1 * thresh, thresh)
        joint_sampler = (joint_sampler + 1) * self.map_size // 2
        xs, ys = joint_sampler[:, ::2], joint_sampler[:, 1::2]
        sampler_flat_inds = (ys * self.map_size + xs).permute((1, 0)).long().contiguous()
        return sampler_flat_inds

    def parameter_sampling(self, maps, batch_ids, flat_inds, use_transform=True):
        device = maps.device
        if use_transform:
            batch, channel = maps.shape[:2]
            maps = maps.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
        results = maps[batch_ids, flat_inds].contiguous()
        return results

    @torch.no_grad()
    def parse_maps(self, outputs, meta_data, cfg):
        if 'pred_batch_ids' in outputs:
            if cfg['with_nms'] and args().model_version in [6, 9]:
                outputs = suppressing_duplicate_mesh(outputs)
            batch_ids = outputs['pred_batch_ids'].long()
            outputs['center_preds'] = outputs['pred_czyxs'] * args().input_size / args().centermap_size
            outputs['center_confs'] = outputs['top_score']
        else:
            batch_ids, flat_inds, cyxs, top_score = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['center_map'])
            if len(batch_ids) == 0:
                batch_ids, flat_inds, cyxs, top_score = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['center_map'], top_n_people=1)
                outputs['detection_flag'] = torch.Tensor([(False) for _ in range(len(batch_ids))])
        if 'params_pred' not in outputs and 'params_maps' in outputs:
            outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
        if 'center_preds' not in outputs:
            outputs['center_preds'] = torch.stack([flat_inds % args().centermap_size, flat_inds // args().centermap_size], 1) * args().input_size / args().centermap_size
            outputs['center_confs'] = self.parameter_sampling(outputs['center_map'], batch_ids, flat_inds, use_transform=True)
        if 'joint_sampler_maps_filtered' in outputs:
            outputs['joint_sampler'] = self.parameter_sampling(outputs['joint_sampler_maps_filtered'], batch_ids, flat_inds, use_transform=True)
            if 'params_pred' in outputs:
                _check_params_pred_(outputs['params_pred'].shape, len(batch_ids))
                self.adjust_to_joint_level_sampling(outputs['params_pred'], outputs['joint_sampler'], outputs['params_maps'], batch_ids)
        if 'reid_map' in outputs:
            outputs['reid_embeds'] = self.parameter_sampling(outputs['reid_map'], batch_ids, flat_inds, use_transform=True)
        if 'uncertainty_map' in outputs:
            outputs['uncertainty_pred'] = torch.sqrt(self.parameter_sampling(outputs['uncertainty_map'], batch_ids, flat_inds, use_transform=True) ** 2) + 1
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = ['image', 'offsets', 'imgpath', 'camMats']
        if len(args().gpu) == 1:
            meta_data = reorganize_gts_cpu(meta_data, info_vis, batch_ids)
        else:
            meta_data = reorganize_gts(meta_data, info_vis, batch_ids)
        if 'pred_batch_ids' in outputs:
            outputs['pred_batch_ids'] += meta_data['batch_ids'][0]
        return outputs, meta_data


class AddCoords(nn.Module):

    def __init__(self, radius_channel=False):
        super(AddCoords, self).__init__()
        self.radius_channel = radius_channel

    def forward(self, in_tensor):
        """
        in_tensor: (batch_size, channels, x_dim, y_dim)
        [0,0,0,0]   [0,1,2,3]
        [1,1,1,1]   [0,1,2,3]    << (i,j)th coordinates of pixels added as separate channels
        [2,2,2,2]   [0,1,2,3]
        taken from mkocabas.
        """
        batch_size_tensor = in_tensor.shape[0]
        xx_ones = torch.ones([1, in_tensor.shape[2]], dtype=torch.int32)
        xx_ones = xx_ones.unsqueeze(-1)
        xx_range = torch.arange(in_tensor.shape[2], dtype=torch.int32).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range)
        xx_channel = xx_channel.unsqueeze(-1)
        yy_ones = torch.ones([1, in_tensor.shape[3]], dtype=torch.int32)
        yy_ones = yy_ones.unsqueeze(1)
        yy_range = torch.arange(in_tensor.shape[3], dtype=torch.int32).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones)
        yy_channel = yy_channel.unsqueeze(-1)
        xx_channel = xx_channel.permute(0, 3, 1, 2)
        yy_channel = yy_channel.permute(0, 3, 1, 2)
        xx_channel = xx_channel.float() / (in_tensor.shape[2] - 1)
        yy_channel = yy_channel.float() / (in_tensor.shape[3] - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)
        out = torch.cat([in_tensor, xx_channel, yy_channel], dim=1)
        if self.radius_channel:
            radius_calc = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, radius_calc], dim=1)
        return out


class CoordConv(nn.Module):
    """ add any additional coordinate channels to the input tensor """

    def __init__(self, *args, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoord = AddCoords(radius_channel=False)
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.conv(out)
        return out


class CoordConvTranspose(nn.Module):
    """CoordConvTranspose layer for segmentation tasks."""

    def __init__(self, *args, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoord = AddCoords(radius_channel=False)
        self.convT = nn.ConvTranspose2d(*args, **kwargs)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.convT(out)
        return out


def scatter(inputs, target_gpus, dim=0, chunk_sizes=None):
    """
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
        assert not torch.is_tensor(obj), 'Tensors not supported in scatter.'
        if isinstance(obj, tuple):
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list):
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict):
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]
    return scatter_map(inputs)


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0, chunk_sizes=None):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim, chunk_sizes) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim, chunk_sizes) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class _DataParallel(Module):
    """Implements data parallelism at the module level.
    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.
    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is the
    same size (so that each GPU processes the same number of samples).
    See also: :ref:`cuda-nn-dataparallel-instead`
    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel EXCEPT Tensors. All variables will be scattered on dim
    specified (default 0). Primitive types will be broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass.
    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])
    Example::
        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
        super(_DataParallel, self).__init__()
        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.chunk_sizes = chunk_sizes
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids, self.chunk_sizes)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids, chunk_sizes):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim, chunk_sizes=self.chunk_sizes)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


class Base(nn.Module):

    def forward(self, meta_data, **cfg):
        if cfg['mode'] == 'matching_gts':
            return self.matching_forward(meta_data, **cfg)
        elif cfg['mode'] == 'parsing':
            return self.parsing_forward(meta_data, **cfg)
        elif cfg['mode'] == 'forward':
            return self.pure_forward(meta_data, **cfg)
        elif cfg['mode'] == 'extract_img_feature_maps':
            return self.extract_img_feature_maps(meta_data, **cfg)
        elif cfg['mode'] == 'extract_mesh_feature_maps':
            return self.extract_mesh_feature_maps(meta_data, **cfg)
        elif cfg['mode'] == 'mesh_regression':
            return self.regress_mesh_from_sampled_features(meta_data, **cfg)
        else:
            raise NotImplementedError('forward mode is not recognized! please set proper mode (parsing/matching_gts)')

    def matching_forward(self, meta_data, **cfg):
        if args().model_precision == 'fp16':
            with autocast():
                outputs = self.feed_forward(meta_data)
                outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)
        else:
            outputs = self.feed_forward(meta_data)
            outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)
        outputs['meta_data'] = meta_data
        if cfg['calc_loss']:
            outputs.update(self._calc_loss(outputs))
        return outputs

    @torch.no_grad()
    def parsing_forward(self, meta_data, **cfg):
        if args().model_precision == 'fp16':
            with autocast():
                outputs = self.feed_forward(meta_data)
                outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)
        else:
            outputs = self.feed_forward(meta_data)
            outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)
        outputs['meta_data'] = meta_data
        return outputs

    def feed_forward(self, meta_data):
        x = self.backbone(meta_data['image'].contiguous())
        outputs = self.head_forward(x)
        return outputs

    @torch.no_grad()
    def pure_forward(self, meta_data, **cfg):
        if args().model_precision == 'fp16':
            with autocast():
                outputs = self.feed_forward(meta_data)
        else:
            outputs = self.feed_forward(meta_data)
        return outputs

    def extract_feature_maps(self, image):
        x = self.backbone(image.contiguous())
        if args().learn_deocclusion:
            outputs = self.acquire_maps(x)
        else:
            outputs = {'image_feature_maps': x.float()}
        return outputs

    def extract_img_feature_maps(self, image_inputs, **cfg):
        if args().model_precision == 'fp16':
            with autocast():
                outputs = self.extract_feature_maps(image_inputs['image'])
        else:
            outputs = self.extract_feature_maps(image_inputs['image'])
        return outputs

    @torch.no_grad()
    def extract_mesh_feature_maps(self, image_inputs, **cfg):
        if args().model_precision == 'fp16':
            with autocast():
                mesh_feature_maps = self.param_head(self.backbone(image_inputs['image'].contiguous()))
        else:
            mesh_feature_maps = self.param_head(self.backbone(image_inputs['image'].contiguous()))
        return mesh_feature_maps

    def regress_mesh_from_sampled_features(self, packed_data, **cfg):
        features_sampled, cam_czyx, cam_preds, outputs = packed_data
        if args().model_precision == 'fp16':
            with autocast():
                outputs['params_pred'] = self.mesh_regression_from_features(features_sampled, cam_czyx, cam_preds)
                outputs = self._result_parser.params_map_parser(outputs, outputs['meta_data'])
        else:
            outputs['params_pred'] = self.mesh_regression_from_features(features_sampled, cam_czyx, cam_preds)
            outputs = self._result_parser.params_map_parser(outputs, outputs['meta_data'])
        if 'detection_flag' not in outputs:
            outputs['detection_flag'] = self.determine_detection_flag(outputs, outputs['meta_data'])
        return outputs

    def head_forward(self, x):
        return NotImplementedError

    def make_backbone(self):
        return NotImplementedError

    def backbone_forward(self, x):
        return NotImplementedError

    def _build_gpu_tracker(self):
        self.gpu_tracker = MemTracker()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)


BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class IBN_a(nn.Module):

    def __init__(self, planes, momentum=BN_MOMENTUM):
        super(IBN_a, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2, momentum=momentum)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class BasicBlock_IBN_a(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_IBN_a, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = IBN_a(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, BN=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
        return out


def conv3x3_1D(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock_1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_1D, self).__init__()
        self.conv1 = conv3x3_1D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_1D(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


def conv3x3_3D(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock_3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_3D, self).__init__()
        self.conv1 = conv3x3_3D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_3D(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False), nn.BatchNorm2d(num_inchannels[i]), nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3), nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


def get_3Dcoord_maps_halfz(size, z_base):
    range_arr = torch.arange(size, dtype=torch.float32)
    z_len = len(z_base)
    Z_map = z_base.reshape(1, z_len, 1, 1, 1).repeat(1, 1, size, size, 1)
    Y_map = range_arr.reshape(1, 1, size, 1, 1).repeat(1, z_len, 1, size, 1) / size * 2 - 1
    X_map = range_arr.reshape(1, 1, 1, size, 1).repeat(1, z_len, size, 1, 1) / size * 2 - 1
    out = torch.cat([Z_map, Y_map, X_map], dim=-1)
    return out


def get_coord_maps(size=128):
    xx_ones = torch.ones([1, size], dtype=torch.int32)
    xx_ones = xx_ones.unsqueeze(-1)
    xx_range = torch.arange(size, dtype=torch.int32).unsqueeze(0)
    xx_range = xx_range.unsqueeze(1)
    xx_channel = torch.matmul(xx_ones, xx_range)
    xx_channel = xx_channel.unsqueeze(-1)
    yy_ones = torch.ones([1, size], dtype=torch.int32)
    yy_ones = yy_ones.unsqueeze(1)
    yy_range = torch.arange(size, dtype=torch.int32).unsqueeze(0)
    yy_range = yy_range.unsqueeze(-1)
    yy_channel = torch.matmul(yy_range, yy_ones)
    yy_channel = yy_channel.unsqueeze(-1)
    xx_channel = xx_channel.permute(0, 3, 1, 2)
    yy_channel = yy_channel.permute(0, 3, 1, 2)
    xx_channel = xx_channel.float() / (size - 1)
    yy_channel = yy_channel.float() / (size - 1)
    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1
    out = torch.cat([xx_channel, yy_channel], dim=1)
    return out


def BHWC_to_BCHW(x):
    """
    :param x: torch tensor, B x H x W x C
    :return:  torch tensor, B x C x H x W
    """
    return x.unsqueeze(1).transpose(1, -1).squeeze(-1)


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


def copy_state_dict(cur_state_dict, pre_state_dict, prefix='module.', drop_prefix='', fix_loaded=False):
    success_layers, failed_layers = [], []

    def _get_params(key):
        key = key.replace(drop_prefix, '')
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None
    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                failed_layers.append(k)
                continue
            cur_state_dict[k].copy_(v)
            if prefix in k and prefix != '':
                k = k.split(prefix)[1]
            success_layers.append(k)
        except:
            None
            continue
    None
    if fix_loaded and len(failed_layers) > 0:
        logging.info('fixing the layers that were loaded successfully, while train the layers that failed,')
        fixed_layers = []
        for k in cur_state_dict.keys():
            try:
                if k in success_layers:
                    cur_state_dict[k].requires_grad = False
                    fixed_layers.append(k)
            except:
                logging.info('fixing the layer {} failed'.format(k))
    return success_layers


class HigherResolutionNet(nn.Module):

    def __init__(self, **kwargs):
        self.inplanes = 64
        super(HigherResolutionNet, self).__init__()
        self.make_baseline()
        self.backbone_channels = 32

    def load_pretrain_params(self):
        if os.path.exists(args().hrnet_pretrain):
            None
            success_layer = copy_state_dict(self.state_dict(), torch.load(args().hrnet_pretrain), prefix='', fix_loaded=True)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), nn.BatchNorm2d(num_channels_cur_layer[i]), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(outchannels), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1, BN=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, BN=BN))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, BN=BN))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def make_baseline(self):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4, BN=nn.BatchNorm2d)
        self.stage2_cfg = {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [32, 64], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4], 'NUM_CHANNELS': [32, 64, 128], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_CHANNELS': [32, 64, 128, 256], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=False)

    def forward(self, x):
        x = (BHWC_to_BCHW(x) / 255 * 2.0 - 1.0).contiguous()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        x = y_list[0]
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)


class ResNet_50(nn.Module):

    def __init__(self, **kwargs):
        self.inplanes = 64
        super(ResNet_50, self).__init__()
        self.make_resnet()
        self.backbone_channels = 64

    def load_pretrain_params(self):
        if os.path.exists(args().resnet_pretrain):
            success_layer = copy_state_dict(self.state_dict(), torch.load(args().resnet_pretrain), prefix='', fix_loaded=True)

    def image_preprocess(self, x):
        if args().pretrain == 'imagenet' or args().pretrain == 'spin':
            x = BHWC_to_BCHW(x) / 255.0
            x = torch.stack(list(map(lambda x: F.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=False), x)))
        else:
            x = (BHWC_to_BCHW(x) / 255.0 * 2.0 - 1.0).contiguous()
        return x

    def make_resnet(self):
        block, layers = Bottleneck, [3, 4, 6, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_resnet_layer(block, 64, layers[0])
        self.layer2 = self._make_resnet_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_resnet_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_resnet_layer(block, 512, layers[3], stride=2)
        self.deconv_layers = self._make_deconv_layer(3, (256, 128, 64), (4, 4, 4))

    def forward(self, x):
        x = self.image_preprocess(x)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv_layers(x)
        return x

    def _make_resnet_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):
            if i == 0:
                self.inplanes = 2048
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)


class ROMP(Base):

    def __init__(self, backbone=None, **kwargs):
        super(ROMP, self).__init__()
        None
        self.backbone = backbone
        self._result_parser = ResultParser()
        self._build_head()
        if args().model_return_loss:
            self._calc_loss = Loss()
        if not args().fine_tune and not args().eval:
            self.init_weights()
            self.backbone.load_pretrain_params()

    def head_forward(self, x):
        x = torch.cat((x, self.coordmaps.repeat(x.shape[0], 1, 1, 1)), 1)
        params_maps = self.final_layers[1](x)
        center_maps = self.final_layers[2](x)
        if args().merge_smpl_camera_head:
            cam_maps, params_maps = params_maps[:, :3], params_maps[:, 3:]
        else:
            cam_maps = self.final_layers[3](x)
        cam_maps[:, 0] = torch.pow(1.1, cam_maps[:, 0])
        params_maps = torch.cat([cam_maps, params_maps], 1)
        output = {'params_maps': params_maps.float(), 'center_map': center_maps.float()}
        return output

    def _build_head(self):
        self.outmap_size = args().centermap_size
        params_num, cam_dim = 145, 3
        self.head_cfg = {'NUM_HEADS': 1, 'NUM_CHANNELS': 64, 'NUM_BASIC_BLOCKS': args().head_block_num}
        self.output_cfg = {'NUM_PARAMS_MAP': params_num - cam_dim, 'NUM_CENTER_MAP': 1, 'NUM_CAM_MAP': cam_dim}
        self.final_layers = self._make_final_layers(self.backbone.backbone_channels)
        self.coordmaps = get_coord_maps(128)

    def _make_final_layers(self, input_channels):
        final_layers = []
        final_layers.append(None)
        input_channels += 2
        if args().merge_smpl_camera_head:
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP'] + self.output_cfg['NUM_CAM_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
        else:
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CAM_MAP']))
        return nn.ModuleList(final_layers)

    def _make_head_layers(self, input_channels, output_channels):
        head_layers = []
        num_channels = self.head_cfg['NUM_CHANNELS']
        kernel_sizes, strides, paddings = self._get_trans_cfg()
        for kernel_size, padding, stride in zip(kernel_sizes, paddings, strides):
            head_layers.append(nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
        for i in range(self.head_cfg['NUM_HEADS']):
            layers = []
            for _ in range(self.head_cfg['NUM_BASIC_BLOCKS']):
                layers.append(nn.Sequential(BasicBlock(num_channels, num_channels)))
            head_layers.append(nn.Sequential(*layers))
        head_layers.append(nn.Conv2d(in_channels=num_channels, out_channels=output_channels, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*head_layers)

    def _get_trans_cfg(self):
        if self.outmap_size == 32:
            kernel_sizes = [3, 3]
            paddings = [1, 1]
            strides = [2, 2]
        elif self.outmap_size == 64:
            kernel_sizes = [3]
            paddings = [1]
            strides = [2]
        elif self.outmap_size == 128:
            kernel_sizes = [3]
            paddings = [1]
            strides = [1]
        return kernel_sizes, strides, paddings


class SMPLR(nn.Module):

    def __init__(self, use_gender=False):
        super(SMPLR, self).__init__()
        model_path = os.path.join(config.model_dir, 'parameters', 'smpl')
        self.smpls = {}
        self.smpls['n'] = SMPL(args().smpl_model_path, model_type='smpl')
        if use_gender:
            self.smpls['f'] = SMPL(args().smpl_model_path.replace('NEUTRAL', 'FEMALE'))
            self.smpls['m'] = SMPL(args().smpl_model_path.replace('NEUTRAL', 'MALE'))

    def forward(self, pose, betas, gender='n', root_align=True):
        if isinstance(pose, np.ndarray):
            pose, betas = torch.from_numpy(pose).float(), torch.from_numpy(betas).float()
        if len(pose.shape) == 1:
            pose, betas = pose.unsqueeze(0), betas.unsqueeze(0)
        verts, joints44_17 = self.smpls[gender](poses=pose, betas=betas, root_align=root_align)
        return verts.numpy(), joints44_17[:, :args().joint_num].numpy()


class MeshRendererWithDepth(nn.Module):

    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) ->torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


mesh_color_table = {'pink': [0.7, 0.7, 0.9], 'neutral': [0.9, 0.9, 0.8], 'capsule': [0.7, 0.75, 0.5], 'yellow': [0.5, 0.7, 0.75]}


def set_mesh_color(verts_rgb, colors):
    if colors is None:
        colors = torch.Tensor(mesh_color_table['neutral'])
    if len(colors.shape) == 1:
        verts_rgb[:, :] = colors
    elif len(colors.shape) == 2:
        verts_rgb[:, :] = colors.unsqueeze(1)
    return verts_rgb


def get_remove_keys(dt, keys=[]):
    targets = []
    for key in keys:
        targets.append(dt[key])
    for key in keys:
        del dt[key]
    return targets


def img_preprocess(image, imgpath, input_size=512, ds='internet', single_img_input=False):
    image = image[:, :, ::-1]
    image_size = image.shape[:2][::-1]
    image_org = Image.fromarray(image)
    resized_image_size = (float(input_size) / max(image_size) * np.array(image_size) // 2 * 2).astype(np.int32)
    padding = tuple((input_size - resized_image_size) // 2)
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize([resized_image_size[1], resized_image_size[0]], interpolation=3), torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')])
    image = torch.from_numpy(np.array(transform(image_org))).float()
    padding_org = tuple((max(image_size) - np.array(image_size)) // 2)
    transform_org = torchvision.transforms.Compose([torchvision.transforms.Pad(padding_org, fill=0, padding_mode='constant'), torchvision.transforms.Resize((input_size * 2, input_size * 2), interpolation=3)])
    image_org = torch.from_numpy(np.array(transform_org(image_org)))
    padding_org = (np.array(list(padding_org)) * float(input_size * 2 / max(image_size))).astype(np.int32)
    if padding_org[0] > 0:
        image_org[:, :padding_org[0]] = 255
        image_org[:, -padding_org[0]:] = 255
    if padding_org[1] > 0:
        image_org[:padding_org[1]] = 255
        image_org[-padding_org[1]:] = 255
    offset = (max(image_size) - np.array(image_size)) / 2
    offsets = np.array([image_size[1], image_size[0], 0, resized_image_size[0] + padding[1], 0, resized_image_size[1] + padding[0], offset[1], resized_image_size[0], offset[0], resized_image_size[1], max(image_size)], dtype=np.int32)
    offsets = torch.from_numpy(offsets).float()
    name = os.path.basename(imgpath)
    if single_img_input:
        image = image.unsqueeze(0).contiguous()
        image_org = image_org.unsqueeze(0).contiguous()
        offsets = offsets.unsqueeze(0).contiguous()
        imgpath, name, ds = [imgpath], [name], [ds]
    input_data = {'image': image, 'image_org': image_org, 'imgpath': imgpath, 'offsets': offsets, 'name': name, 'data_set': ds}
    return input_data


def justify_detection_state(detection_flag, reorganize_idx):
    if detection_flag.sum() == 0:
        detection_flag = False
    else:
        reorganize_idx = reorganize_idx[detection_flag.bool()].long()
        detection_flag = True
    return detection_flag, reorganize_idx


def reorganize_items(items, reorganize_idx):
    items_new = [[] for _ in range(len(items))]
    for idx, item in enumerate(items):
        for ridx in reorganize_idx:
            items_new[idx].append(item[ridx])
    return items_new


class Predictor(Base):

    def __init__(self, **kwargs):
        super(Predictor, self).__init__(**kwargs)
        self._build_model_()
        self._prepare_modules_()
        self.demo_cfg = {'mode': 'parsing', 'calc_loss': False}
        if self.character == 'nvxia':
            assert os.path.exists(os.path.join('model_data', 'characters', 'nvxia')), 'Current released version does not support other characters, like Nvxia.'
            self.character_model = create_nvxia_model(self.nvxia_model_path)

    def net_forward(self, meta_data, cfg=None):
        ds_org, imgpath_org = get_remove_keys(meta_data, keys=['data_set', 'imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        if self.model_precision == 'fp16':
            with autocast():
                outputs = self.model(meta_data, **cfg)
        else:
            outputs = self.model(meta_data, **cfg)
        outputs['detection_flag'], outputs['reorganize_idx'] = justify_detection_state(outputs['detection_flag'], outputs['reorganize_idx'])
        meta_data.update({'imgpath': imgpath_org, 'data_set': ds_org})
        outputs['meta_data']['data_set'], outputs['meta_data']['imgpath'] = reorganize_items([ds_org, imgpath_org], outputs['reorganize_idx'].cpu().numpy())
        return outputs

    def _prepare_modules_(self):
        self.model.eval()
        self.demo_dir = os.path.join(config.project_dir, 'demo')

    def __initialize__(self):
        if self.save_mesh:
            self.smpl_faces = torch.load(args().smpl_model_path)['f'].numpy()
        None

    def single_image_forward(self, image):
        meta_data = img_preprocess(image, '0', input_size=args().input_size, single_img_input=True)
        if '-1' not in self.gpu:
            meta_data['image'] = meta_data['image']
        outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
        return outputs

    def reorganize_results(self, outputs, img_paths, reorganize_idx):
        results = {}
        cam_results = outputs['params']['cam'].detach().cpu().numpy().astype(np.float16)
        trans_results = outputs['cam_trans'].detach().cpu().numpy().astype(np.float16)
        smpl_pose_results = outputs['params']['poses'].detach().cpu().numpy().astype(np.float16)
        smpl_shape_results = outputs['params']['betas'].detach().cpu().numpy().astype(np.float16)
        joints_54 = outputs['j3d'].detach().cpu().numpy().astype(np.float16)
        kp3d_smpl24_results = outputs['joints_smpl24'].detach().cpu().numpy().astype(np.float16)
        kp3d_spin24_results = joints_54[:, constants.joint_mapping(constants.SMPL_ALL_54, constants.SPIN_24)]
        kp3d_op25_results = joints_54[:, constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)]
        verts_results = outputs['verts'].detach().cpu().numpy().astype(np.float16)
        pj2d_results = outputs['pj2d'].detach().cpu().numpy().astype(np.float16)
        pj2d_org_results = outputs['pj2d_org'].detach().cpu().numpy().astype(np.float16)
        center_confs = outputs['centers_conf'].detach().cpu().numpy().astype(np.float16)
        vids_org = np.unique(reorganize_idx)
        for idx, vid in enumerate(vids_org):
            verts_vids = np.where(reorganize_idx == vid)[0]
            img_path = img_paths[verts_vids[0]]
            results[img_path] = [{} for idx in range(len(verts_vids))]
            for subject_idx, batch_idx in enumerate(verts_vids):
                results[img_path][subject_idx]['cam'] = cam_results[batch_idx]
                results[img_path][subject_idx]['cam_trans'] = trans_results[batch_idx]
                results[img_path][subject_idx]['poses'] = smpl_pose_results[batch_idx]
                results[img_path][subject_idx]['betas'] = smpl_shape_results[batch_idx]
                results[img_path][subject_idx]['j3d_all54'] = joints_54[batch_idx]
                results[img_path][subject_idx]['j3d_smpl24'] = kp3d_smpl24_results[batch_idx]
                results[img_path][subject_idx]['j3d_spin24'] = kp3d_spin24_results[batch_idx]
                results[img_path][subject_idx]['j3d_op25'] = kp3d_op25_results[batch_idx]
                results[img_path][subject_idx]['verts'] = verts_results[batch_idx]
                results[img_path][subject_idx]['pj2d'] = pj2d_results[batch_idx]
                results[img_path][subject_idx]['pj2d_org'] = pj2d_org_results[batch_idx]
                results[img_path][subject_idx]['center_conf'] = center_confs[batch_idx]
        return results


class Time_counter:

    def __init__(self, thresh=0.1):
        self.thresh = thresh
        self.runtime = 0
        self.frame_num = 0

    def start(self):
        self.start_time = time.time()

    def count(self, frame_num=1):
        time_cost = time.time() - self.start_time
        if time_cost < self.thresh:
            self.runtime += time_cost
            self.frame_num += frame_num
        self.start()

    def fps(self):
        None
        None

    def reset(self):
        self.runtime = 0
        self.frame_num = 0


def convert_cam_to_stand_on_image_trans(cam, enlarge_scale=3):
    trans_3d = convert_cam_to_3d_trans(cam)
    stand_on_image_trans = np.zeros(3)
    stand_on_image_trans[0] = trans_3d[0] * 0.3
    stand_on_image_trans[1] = 0.6
    stand_on_image_trans[2] = trans_3d[1] * 0.4
    stand_on_image_trans *= enlarge_scale
    return stand_on_image_trans


def parse_nvxia_uvmap(uvs, face):
    verts_num = np.max(face) + 1
    uvs_verts = np.zeros((verts_num, 2))
    for uv, f in zip(uvs, face):
        uvs_verts[f] = uv[:, :2]
    return uvs_verts


class Vedo_visualizer(object):

    def __init__(self):
        if args().character == 'smpl':
            self.faces = torch.load(args().smpl_model_path)['f'].numpy()
        elif args().character == 'nvxia':
            params_dict = np.load(os.path.join(args().nvxia_model_path, 'nvxia.npz'), allow_pickle=True)
            self.faces = np.array([np.array(face) for face in params_dict['polygons']])
            self.texture_file = cv2.imread(os.path.join(args().nvxia_model_path, 'Kachujin_diffuse.png'))[:, :, ::-1]
            self.uvs = parse_nvxia_uvmap(params_dict['uvmap'], self.faces)
        self.scene_bg_color = [240, 255, 255]
        self.default_camera = {'pos': {'far': (0, 800, 1000), 'close': (0, 200, 800)}[args().soi_camera]}
        self.lights = [Light([0, 800, 1000], intensity=0.6, c='white'), Light([0, -800, 1000], intensity=0.6, c='white'), Light([0, 800, -1000], intensity=0.6, c='white'), Light([0, -800, -1000], intensity=0.6, c='white')]
        vedo.settings.screeshotLargeImage = True
        vedo.settings.screeshotScale = 2

    def plot_multi_meshes_batch(self, vertices, cam_params, meta_data, reorganize_idx, save_img=True, interactive_show=False, rotate_frames=[]):
        result_save_names = []
        for inds, img_id in enumerate(np.unique(reorganize_idx)):
            single_img_verts_inds = np.array(np.where(reorganize_idx == img_id)[0])
            plt = self.plot_multi_meshes(vertices[single_img_verts_inds].detach().cpu().numpy(), cam_params[single_img_verts_inds].detach().cpu().numpy(), meta_data['image'][single_img_verts_inds[0]].cpu().numpy().astype(np.uint8), interactive_show=interactive_show)
            if img_id in rotate_frames:
                result_imgs, rot_angles = self.render_rotating(plt)
                save_names = [os.path.join(args().output_dir, '3D_meshes-' + os.path.basename(meta_data['imgpath'][single_img_verts_inds[0]] + '_{:03d}.jpg'.format(ra))) for ra in rot_angles]
            else:
                result_imgs = self.render_one_time(plt, self.default_camera)
                save_names = [os.path.join(args().output_dir, '3D_meshes-' + os.path.basename(meta_data['imgpath'][single_img_verts_inds[0]] + '.jpg'))]
            plt.close()
            result_save_names += save_names
            if save_img:
                for save_name, result_img in zip(save_names, result_imgs):
                    cv2.imwrite(save_name, result_img[:, :, ::-1])
        return result_save_names

    def plot_multi_meshes(self, vertices, cam_params, img, mesh_colors=None, interactive_show=False, rotate_cam=False):
        plt = Plotter(bg=[240, 255, 255], axes=0, offscreen=not interactive_show)
        h, w = img.shape[:2]
        pic = Picture(img)
        pic.rotateX(-90).z(h // 2).x(-w // 2)
        verts_enlarge_scale = max(h, w) / 5
        cam_enlarge_scale = max(h, w) / 3
        plt += pic
        vertices_vis = []
        for inds, (vert, cam) in enumerate(zip(vertices, cam_params)):
            trans_3d = convert_cam_to_stand_on_image_trans(cam, cam_enlarge_scale)
            vert[:, 1:] *= -1
            vert = vert * verts_enlarge_scale
            vert += trans_3d[None]
            vertices_vis.append(vert)
        vertices_vis = np.stack(vertices_vis, 0)
        visulize_list = []
        for inds, vert in enumerate(vertices_vis):
            mesh = Mesh([vert, self.faces])
            if args().character == 'smpl':
                mesh = mesh.c([255, 255, 255]).smooth(niter=20)
                if mesh_colors is not None:
                    mesh.c(mesh_colors[inds].astype(np.uint8))
            elif args().character == 'nvxia':
                mesh.texture(self.texture_file, tcoords=self.uvs).smooth(niter=20)
            visulize_list.append(mesh)
        plt += visulize_list
        for light in self.lights:
            plt += light
        return plt

    def render_rotating(self, plt, internal=5):
        result_imgs = []
        pause_num = args().fps_save
        pause = np.zeros(pause_num).astype(np.int32)
        change_time = 90 // internal
        roates = np.ones(change_time) * internal
        go_up = np.sin(np.arange(change_time).astype(np.float32) / change_time) * 1
        go_down = np.sin(np.arange(change_time).astype(np.float32) / change_time - 1) * 1
        azimuth_angles = np.concatenate([pause, roates, roates, roates, roates])
        elevation_angles = np.concatenate([pause, go_up, go_down, go_up, go_down])
        plt.camera.Elevation(20)
        for rid, azimuth_angle in enumerate(azimuth_angles):
            plt.show(azimuth=azimuth_angle, elevation=elevation_angles[rid])
            result_img = plt.topicture(scale=2)
            rows, cols, _ = result_img._data.GetDimensions()
            vtkimage = result_img._data.GetPointData().GetScalars()
            image_result = vtk_to_numpy(vtkimage).reshape((rows, cols, 3))
            result_imgs.append(image_result[::-1])
        return result_imgs, np.arange(len(azimuth_angles))

    def render_one_time(self, plt, camera_pose):
        image_result = plt.show(camera=camera_pose)
        result_img = plt.topicture(scale=2)
        rows, cols, _ = result_img._data.GetDimensions()
        vtkimage = result_img._data.GetPointData().GetScalars()
        image_result = vtk_to_numpy(vtkimage).reshape((rows, cols, 3))
        image_result = image_result[::-1]
        return [image_result]


def collect_image_list(image_folder=None, collect_subdirs=False, img_exts=None):

    def collect_image_from_subfolders(image_folder, file_list, collect_subdirs, img_exts):
        for path in glob.glob(os.path.join(image_folder, '*')):
            if os.path.isdir(path) and collect_subdirs:
                collect_image_from_subfolders(path, file_list, collect_subdirs, img_exts)
            elif os.path.splitext(path)[1] in img_exts:
                file_list.append(path)
        return file_list
    file_list = collect_image_from_subfolders(image_folder, [], collect_subdirs, img_exts)
    return file_list


def save_obj(verts, faces, obj_mesh_name='mesh.obj'):
    with open(obj_mesh_name, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces:
            fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))


def save_meshes(reorganize_idx, outputs, output_dir, smpl_faces):
    vids_org = np.unique(reorganize_idx)
    for idx, vid in enumerate(vids_org):
        verts_vids = np.where(reorganize_idx == vid)[0]
        img_path = outputs['meta_data']['imgpath'][verts_vids[0]]
        obj_name = os.path.join(output_dir, '{}'.format(os.path.basename(img_path))).replace('.mp4', '').replace('.jpg', '').replace('.png', '') + '.obj'
        for subject_idx, batch_idx in enumerate(verts_vids):
            save_obj(outputs['verts'][batch_idx].detach().cpu().numpy().astype(np.float16), smpl_faces, obj_name.replace('.obj', '_{}.obj'.format(subject_idx)))


def save_result_dict_tonpz(results, test_save_dir):
    for img_path, result_dict in results.items():
        if platform.system() == 'Windows':
            path_list = img_path.split('\\')
        else:
            path_list = img_path.split('/')
        file_name = '_'.join(path_list)
        file_name = '_'.join(os.path.splitext(file_name)).replace('.', '') + '.npz'
        save_path = os.path.join(test_save_dir, file_name)
        np.savez(save_path, results=result_dict)


class Image_processor(Predictor):

    def __init__(self, **kwargs):
        super(Image_processor, self).__init__(**kwargs)
        self.__initialize__()

    @torch.no_grad()
    def run(self, image_folder, tracker=None):
        None
        os.makedirs(self.output_dir, exist_ok=True)
        self.visualizer.result_img_dir = self.output_dir
        counter = Time_counter(thresh=1)
        if self.show_mesh_stand_on_image:
            visualizer = Vedo_visualizer()
            stand_on_imgs_frames = []
        file_list = collect_image_list(image_folder=image_folder, collect_subdirs=self.collect_subdirs, img_exts=constants.img_exts)
        internet_loader = self._create_single_data_loader(dataset='internet', train_flag=False, file_list=file_list, shuffle=False)
        counter.start()
        results_all = {}
        for test_iter, meta_data in enumerate(internet_loader):
            outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
            reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
            counter.count(self.val_batch_size)
            results = self.reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx)
            if self.save_dict_results:
                save_result_dict_tonpz(results, self.output_dir)
            if self.save_visualization_on_img:
                show_items_list = ['org_img', 'mesh']
                if self.save_centermap:
                    show_items_list.append('centermap')
                results_dict, img_names = self.visualizer.visulize_result(outputs, outputs['meta_data'], show_items=show_items_list, vis_cfg={'settings': ['put_org']}, save2html=False)
                for img_name, mesh_rendering_orgimg in zip(img_names, results_dict['mesh_rendering_orgimgs']['figs']):
                    save_name = os.path.join(self.output_dir, os.path.basename(img_name))
                    cv2.imwrite(save_name, cv2.cvtColor(mesh_rendering_orgimg, cv2.COLOR_RGB2BGR))
            if self.show_mesh_stand_on_image:
                stand_on_imgs = visualizer.plot_multi_meshes_batch(outputs['verts'], outputs['params']['cam'], outputs['meta_data'], outputs['reorganize_idx'].cpu().numpy(), interactive_show=self.interactive_vis)
                stand_on_imgs_frames += stand_on_imgs
            if self.save_mesh:
                save_meshes(reorganize_idx, outputs, self.output_dir, self.smpl_faces)
            if test_iter % 8 == 0:
                None
            counter.start()
            results_all.update(results)
        return results_all


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self.ndim = ndim
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [(2 * self._std_weight_position * measurement[self.ndim - 1]) for _ in range(self.ndim)] + [(10 * self._std_weight_velocity * measurement[self.ndim - 1]) for _ in range(self.ndim)]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [(self._std_weight_position * mean[self.ndim - 1]) for _ in range(self.ndim)]
        std_vel = [(self._std_weight_velocity * mean[self.ndim - 1]) for _ in range(self.ndim)]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [(self._std_weight_position * mean[self.ndim - 1]) for _ in range(self.ndim)]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [(self._std_weight_position * mean[:, self.ndim - 1]) for _ in range(self.ndim)]
        std_vel = [(self._std_weight_velocity * mean[:, self.ndim - 1]) for _ in range(self.ndim)]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:self.ndim - 1], covariance[:self.ndim - 1, :self.ndim - 1]
            measurements = measurements[:, :self.ndim - 1]
        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0
    track_id = 0
    is_activated = False
    state = TrackState.New
    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0
    location = np.inf, np.inf

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, trans, score, czyx):
        self._trans = np.asarray(trans, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.czyx = czyx
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            """ don't know why doing this """
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self._trans)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.trans)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_trans = new_track.trans
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_trans)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.czyx = new_track.czyx
        self.score = new_track.score

    @property
    def trans(self):
        """Get current 3D body center position `(x,y,z)`.
        """
        if self.mean is None:
            return self._trans.copy()
        ret = self.mean[:4].copy()
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class STrack3D(BaseTrack):

    def __init__(self, trans, score, czyx):
        self._trans = np.asarray(trans, dtype=np.float32)
        self.is_activated = False
        self.score = score
        self.czyx = czyx
        self.tracklet_len = 0

    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self._trans = new_track.trans
        self.czyx = new_track.czyx
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    @property
    def trans(self):
        """Get current 3D body center position `(x,y,z)`.
        """
        return self._trans.copy()

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


def euclidean_distance(detection, tracked_object):
    dist = np.linalg.norm(detection.points - tracked_object.estimate)
    return dist


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def remove_duplicate_stracks(stracksa, stracksb, dist_thresh=0.15):
    pdist = matching.euclidean_distance(stracksa, stracksb, dim=2)
    pairs = np.where(pdist < dist_thresh)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


class Tracker(object):

    def __init__(self, det_thresh=0.05, first_frame_det_thresh=0.12, match_thresh=1.0, accept_new_dets=False, new_subject_det_thresh=0.8, axis_times=np.array([1.1, 0.9, 10]), track_buffer=60, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.match_thresh = match_thresh
        self.det_thresh = det_thresh
        self.first_frame_det_thresh = first_frame_det_thresh
        self.new_subject_det_thresh = 0.8
        self.accept_new_dets = accept_new_dets
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        if args().tracking_with_kalman_filter:
            self.kalman_filter = KalmanFilter()
        self.duplicat_dist_thresh = 0.66
        self.axis_times = axis_times

    def update(self, trans3D, scores, last_trans3D, czyxs, debug=True, never_forget=False, tracking_target_max_num=100, using_motion_offsets=True):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        if self.frame_id == 1:
            remain_inds = scores > self.first_frame_det_thresh
        else:
            remain_inds = scores > self.det_thresh
        dets = trans3D[remain_inds]
        scores_keep = scores[remain_inds]
        last_dets_keep = last_trans3D[remain_inds]
        czyxs_keep = czyxs[remain_inds]
        if len(dets) > 0:
            detections = [STrack3D(np.array([*trans[:2], 1 / (1 + trans[2])]), s, czyx) for trans, s, czyx in zip(dets, scores_keep, czyxs_keep)]
            detections_add_offsets = [STrack3D(np.array([*trans[:2], 1 / (1 + trans[2])]), s, czyx) for trans, s, czyx in zip(last_dets_keep, scores_keep, czyxs_keep)]
        else:
            detections = []
            detections_add_offsets = []
        """ Step 2: First association, with high score detection boxes"""
        strack_pool = self.tracked_stracks
        if debug:
            None
            if len(strack_pool) > 0:
                None
                None
            if len(detections_add_offsets) > 0:
                None
                None
        if using_motion_offsets:
            dists = euclidean_distance(strack_pool, detections_add_offsets, dim=3, aug=self.axis_times)
        else:
            dists = euclidean_distance(strack_pool, detections, dim=3, aug=self.axis_times)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_add_offsets[idet]
            if track.state == TrackState.Tracked:
                track.update(detections_add_offsets[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        for it in u_track:
            track = self.tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
            lost_stracks.append(track)
        """ Step 4: Init new stracks with the strack is empty, like the first frame"""
        if not self.accept_new_dets and len(self.tracked_stracks) < tracking_target_max_num:
            u_detection = np.array(u_detection)
            track_scores = np.array([detections[inew].score for inew in u_detection])
            scale_ok_mask = track_scores > self.first_frame_det_thresh
            u_detection = u_detection[scale_ok_mask]
            track_scales = np.array([detections[inew]._trans[2] for inew in u_detection])
            max_scale_subject_inds = u_detection[np.argsort(track_scales)[::-1][:tracking_target_max_num]]
            for inew in max_scale_subject_inds:
                track = detections[inew]
                track.activate(self.frame_id)
                activated_starcks.append(track)
        elif self.accept_new_dets:
            for inew in u_detection:
                track = detections[inew]
                if len(strack_pool) == 0 and track.score > self.first_frame_det_thresh:
                    track.activate(self.frame_id)
                    activated_starcks.append(track)
                elif track.score > self.new_subject_det_thresh:
                    track.activate(self.frame_id)
                    activated_starcks.append(track)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        """ Step 5: Update state"""
        for track in lost_stracks:
            if debug:
                None
            if self.frame_id - track.end_frame > self.max_time_lost and not never_forget:
                track.mark_removed()
                removed_stracks.append(track)
                self.tracked_stracks = sub_stracks(self.tracked_stracks, [track])
        output_results = np.array([np.array([*track.trans[:3], track.track_id, track.score, track.state == TrackState.Tracked, *track.czyx]) for track in self.tracked_stracks if track.is_activated])
        if debug:
            None
        return copy.deepcopy(output_results)

    def update_BT(self, trans3D, scores, last_trans3D, czyxs, debug=True, det_thresh=0.12, low_conf_det_thresh=0.05, match_thresh=1):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        remain_inds = scores > det_thresh
        dets = trans3D[remain_inds]
        scores_keep = scores[remain_inds]
        last_dets_keep = trans3D[remain_inds]
        czyxs_keep = czyxs[remain_inds]
        inds_second = np.logical_and(scores > low_conf_det_thresh, scores < det_thresh)
        dets_second = trans3D[inds_second]
        scores_second = scores[inds_second]
        last_dets_second = trans3D[inds_second]
        czyxs_second = czyxs[inds_second]
        if len(dets) > 0:
            detections = [STrack(np.array([*trans, 1 / (1 + trans[2])]), s, czyx) for trans, s, czyx in zip(dets, scores_keep, czyxs_keep)]
        else:
            detections = []
            detections_add_offsets = []
        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = matching.euclidean_distance(strack_pool, detections, dim=4)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=match_thresh)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        """ Step 3: Second association, with low score detection boxes"""
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [STrack(np.concatenate([trans, np.array([1 / (1 + trans[2])])], 0), s, czyx) for trans, s, czyx in zip(dets_second, scores_second, czyxs_second)]
        else:
            detections_second = []
            detections_add_offsets_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.euclidean_distance(r_tracked_stracks, detections_second, dim=4)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=match_thresh * 2)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = matching.euclidean_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=match_thresh * 3)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if debug:
                None
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks, dist_thresh=self.duplicat_dist_thresh)
        output_results = np.array([np.array([*track.trans[:3], track.track_id, track.score, track.state == TrackState.Tracked, *track.czyx]) for track in self.tracked_stracks if track.is_activated])
        if debug:
            None
        return copy.deepcopy(output_results)


class LowPassFilter:

    def __init__(self):
        self.prev_raw_value = None
        self.prev_filtered_value = None

    def process(self, value, alpha):
        if self.prev_raw_value is None:
            s = value
        else:
            s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
        self.prev_raw_value = value
        self.prev_filtered_value = s
        return s


class OneEuroFilter:

    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    def compute_alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def process(self, x, print_inter=False):
        prev_x = self.x_filter.prev_raw_value
        dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
        edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
        if isinstance(edx, float):
            cutoff = self.mincutoff + self.beta * np.abs(edx)
        elif isinstance(edx, np.ndarray):
            cutoff = self.mincutoff + self.beta * np.abs(edx)
        elif isinstance(edx, torch.Tensor):
            cutoff = self.mincutoff + self.beta * torch.abs(edx)
        if print_inter:
            None
        return self.x_filter.process(x, self.compute_alpha(cutoff))


def create_OneEuroFilter(smooth_coeff):
    return {'smpl_thetas': OneEuroFilter(smooth_coeff, 0.7), 'cam': OneEuroFilter(1.6, 0.7), 'smpl_betas': OneEuroFilter(0.6, 0.7), 'global_rot': OneEuroFilter(smooth_coeff, 0.7)}


def frames2video(images_path, video_name, images=None, fps=30):
    writer = imageio.get_writer(video_name, format='mp4', mode='I', fps=fps)
    if images is None:
        for path in images_path:
            image = imageio.imread(path)
            writer.append_data(image)
    else:
        for image in images:
            writer.append_data(image)
    writer.close()


def get_tracked_ids(detections, tracked_objects):
    tracked_ids_out = np.array([obj.id for obj in tracked_objects])
    tracked_points = np.array([obj.last_detection.points for obj in tracked_objects])
    org_points = np.array([obj.points for obj in detections])
    tracked_ids, tracked_bbox_ids = [], []
    for tid, tracked_point in enumerate(tracked_points):
        org_p_id = np.argmin(np.array([np.linalg.norm(tracked_point - org_point) for org_point in org_points]))
        tracked_bbox_ids.append(org_p_id)
        tracked_ids.append(tracked_ids_out[tid])
    return tracked_ids, tracked_bbox_ids


def transform_rot_representation(rot, input_type='mat', out_type='quat', input_is_degrees=True):
    """
    make transformation between different representation of 3D rotation
    input_type / out_type (np.array):
        'mat': rotation matrix (3*3)
        'quat': quaternion (4)
        'vec': rotation vector (3)
        'euler': Euler degrees (0-360 deg) in x,y,z (3)
    """
    if input_type == 'mat':
        r = R.from_matrix(rot)
    elif input_type == 'quat':
        r = R.from_quat(rot)
    elif input_type == 'vec':
        r = R.from_rotvec(rot)
    elif input_type == 'euler':
        r = R.from_euler('xyz', rot, degrees=input_is_degrees)
    if out_type == 'mat':
        out = r.as_matrix()
    elif out_type == 'quat':
        out = r.as_quat()
    elif out_type == 'vec':
        out = r.as_rotvec()
    elif out_type == 'euler':
        out = r.as_euler('xyz', degrees=False)
    return out


def temporal_optimize_result(result, filter_dict):
    result['cam'] = filter_dict['cam'].process(result['cam'])
    result['betas'] = filter_dict['betas'].process(result['betas'])
    pose_euler = np.array([transform_rot_representation(vec, input_type='vec', out_type='euler') for vec in result['poses'].reshape((-1, 3))])
    body_pose_euler = filter_dict['poses'].process(pose_euler[1:].reshape(-1))
    result['poses'][3:] = np.array([transform_rot_representation(bodypose, input_type='euler', out_type='vec') for bodypose in body_pose_euler.reshape(-1, 3)]).reshape(-1)
    return result


class OpenCVCapture:

    def __init__(self, video_file=None, show=False):
        if video_file is None:
            self.cap = cv2.VideoCapture(int(args().cam_id))
        else:
            self.cap = cv2.VideoCapture(video_file)
            self.length = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.whether_to_show = show

    def read(self, return_rgb=True):
        flag, frame = self.cap.read()
        if not flag:
            return None
        if self.whether_to_show:
            cv2.imshow('webcam', frame)
            cv2.waitKey(1)
        if return_rgb:
            frame = np.flip(frame, -1).copy()
        return frame


def video2frame(video_name, frame_save_dir=None):
    cap = OpenCVCapture(video_name)
    os.makedirs(frame_save_dir, exist_ok=True)
    frame_list = []
    for frame_id in range(int(cap.length)):
        frame = cap.read(return_rgb=False)
        save_path = os.path.join(frame_save_dir, '{:06d}.jpg'.format(frame_id))
        cv2.imwrite(save_path, frame)
        frame_list.append(save_path)
    return frame_list


def create_mesh(vertices, faces, colors=None, **kwargs):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        mesh.paint_uniform_color([1.0, 0.8, 0.8])
    mesh.compute_vertex_normals()
    return mesh


def create_mesh_with_uvmap(vertices, faces, texture_path=None, uvs=None, **kwargs):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if texture_path is not None and uvs is not None:
        if o3d_version == 9:
            mesh.texture = o3d.io.read_image(texture_path)
            mesh.triangle_uvs = uvs
        elif o3d_version >= 11:
            mesh.textures = [o3d.io.read_image(texture_path)]
            mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
            mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(faces), dtype=np.int32))
    mesh.compute_vertex_normals()
    return mesh


def get_uvs(uvmap_path):
    uv_map_vt_ft = np.load(uvmap_path, allow_pickle=True)
    vt, ft = uv_map_vt_ft['vt'], uv_map_vt_ft['ft']
    uvs = np.concatenate([vt[ft[:, ind]][:, None] for ind in range(3)], 1).reshape(-1, 2)
    uvs[:, 1] = 1 - uvs[:, 1]
    return uvs


class Open3d_visualizer(object):

    def __init__(self, multi_mode=False):
        self.view_mat = axangle2mat([1, 0, 0], np.pi)
        self.window_size = 1080
        smpl_param_dict = pickle.load(open(os.path.join(args().smpl_model_path, 'smpl', 'SMPL_NEUTRAL.pkl'), 'rb'), encoding='latin1')
        self.faces = smpl_param_dict['f']
        self.verts_mean = smpl_param_dict['v_template']
        self.viewer = o3d.visualization.Visualizer()
        self.viewer.create_window(width=self.window_size + 1, height=self.window_size + 1, window_name='ROMP - output')
        if multi_mode:
            self.current_mesh_num = 10
            self.zero_vertices = o3d.utility.Vector3dVector(np.zeros((6890, 3)))
            self.meshes = []
            for _ in range(self.current_mesh_num):
                new_mesh = self.create_single_mesh(self.verts_mean)
                self.meshes.append(new_mesh)
            self.set_meshes_zero(list(range(self.current_mesh_num)))
        else:
            self.mesh = self.create_single_mesh(self.verts_mean)
        view_control = self.viewer.get_view_control()
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        extrinsic = cam_params.extrinsic.copy()
        extrinsic[0:3, 3] = 0
        cam_params.extrinsic = extrinsic
        self.count = 0
        view_control.convert_from_pinhole_camera_parameters(cam_params)
        view_control.set_constant_z_far(1000)
        render_option = self.viewer.get_render_option()
        render_option.load_from_json('romp/lib/visualization/vis_cfgs/render_option.json')
        self.viewer.update_renderer()
        self.mesh_smoother = OneEuroFilter(4.0, 0.0)

    def set_meshes_zero(self, mesh_ids):
        for ind in mesh_ids:
            self.meshes[ind].vertices = self.zero_vertices

    def run(self, verts):
        verts = self.mesh_smoother.process(verts)
        verts = np.matmul(self.view_mat, verts.T).T
        self.mesh.vertices = o3d.utility.Vector3dVector(verts)
        self.mesh.compute_triangle_normals()
        self.mesh.compute_vertex_normals()
        self.viewer.update_geometry(self.mesh)
        self.viewer.poll_events()

    def run_multiperson(self, verts):
        None
        geometries = []
        for v_id, vert in enumerate(verts):
            self.meshes[v_id].vertices = o3d.utility.Vector3dVector(vert)
            self.meshes[v_id].compute_triangle_normals()
            self.viewer.update_geometry(self.meshes[v_id])
        self.viewer.poll_events()
        self.viewer.update_renderer()

    def create_single_mesh(self, vertices):
        if args().cloth in constants.wardrobe or args().cloth == 'random':
            uvs = get_uvs()
            if args().cloth == 'random':
                cloth_id = random.sample(list(constants.wardrobe.keys()), 1)[0]
                None
                texture_file = os.path.join(args().wardrobe, constants.wardrobe[cloth_id])
            else:
                texture_file = os.path.join(args().wardrobe, constants.wardrobe[args().cloth])
            mesh = create_mesh_with_uvmap(vertices, self.faces, texture_path=texture_file, uvs=uvs)
        elif args().cloth in constants.mesh_color_dict:
            mesh_color = np.array(constants.mesh_color_dict[args().cloth]) / 255.0
            mesh = create_mesh(vertices=vertices, faces=self.faces, colors=mesh_color)
        else:
            mesh = create_mesh(vertices=vertices, faces=self.faces)
        self.viewer.add_geometry(mesh)
        return mesh


def myarray2string(array, separator=', ', fmt='%.3f', indent=8):
    assert len(array.shape) == 2, 'Only support MxN matrix, {}'.format(array.shape)
    blank = ' ' * indent
    res = ['[']
    for i in range(array.shape[0]):
        res.append(blank + '  ' + '[{}]'.format(separator.join([(fmt % d) for d in array[i]])))
        if i != array.shape[0] - 1:
            res[-1] += ', '
    res.append(blank + ']')
    return '\r\n'.join(res)


def write_common_results(dumpname=None, results=[], keys=[], fmt='%2.3f'):
    format_out = {'float_kind': lambda x: fmt % x}
    out_text = []
    out_text.append('[\n')
    for idata, data in enumerate(results):
        out_text.append('    {\n')
        output = {}
        output['id'] = data['id']
        for key in keys:
            if key not in data.keys():
                continue
            output[key] = myarray2string(data[key], separator=', ', fmt=fmt)
        out_keys = list(output.keys())
        for key in out_keys:
            out_text.append('        "{}": {}'.format(key, output[key]))
            if key != out_keys[-1]:
                out_text.append(',\n')
            else:
                out_text.append('\n')
        out_text.append('    }')
        if idata != len(results) - 1:
            out_text.append(',\n')
        else:
            out_text.append('\n')
    out_text.append(']\n')
    if dumpname is not None:
        mkout(dumpname)
        with open(dumpname, 'w') as f:
            f.writelines(out_text)
    else:
        return ''.join(out_text)


def encode_detect(data):
    res = write_common_results(None, data, ['keypoints3d'])
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')


def encode_smpl(data):
    res = write_common_results(None, data, ['poses', 'betas', 'vertices', 'transl'])
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')


class BaseSocketClient:

    def __init__(self, host='127.0.0.1', port=9999) ->None:
        if host == 'auto':
            host = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        self.s = s

    def send(self, data):
        val = encode_detect(data)
        self.s.send(bytes('{}\n'.format(len(val)), 'ascii'))
        self.s.sendall(val)

    def send_smpl(self, data):
        val = encode_smpl(data)
        self.s.send(bytes('{}\n'.format(len(val)), 'ascii'))
        self.s.sendall(val)

    def close(self):
        self.s.close()


def log(x):
    time_now = datetime.now().strftime('%m-%d-%H:%M:%S.%f ')
    None


class Results_sender:

    def __init__(self):
        self.client = BaseSocketClient()
        self.queue = Queue()
        self.t = Thread(target=self.run)
        self.t.start()

    def run(self):
        while True:
            time.sleep(1)
            while not self.queue.empty():
                log('update')
                data = self.queue.get()
                self.client.send_smpl(data)

    def send_results(self, poses=None, betas=None, verts=None, kp3ds=None, trans=None, ids=[]):
        results = []
        None
        for ind, pid in enumerate(ids):
            result = {}
            result['id'] = pid
            if trans is not None:
                result['transl'] = trans[[ind]]
            if poses is not None:
                result['poses'] = poses[[ind]]
            if betas is not None:
                result['betas'] = betas[[ind]]
            results.append(result)
        self.queue.put(results)


class SocketClient_blender:

    def __init__(self, host='127.0.0.1', port=9999) ->None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        None
        self.sock, addr = s.accept()
        self.s = s

    def send(self, data_list):
        d = self.sock.recv(1024)
        if not d:
            self.close()
        data_send = json.dumps(data_list).encode('utf-8')
        self.sock.send(data_send)

    def close(self):
        self.s.close()


class Webcam_processor(Predictor):

    def __init__(self, **kwargs):
        super(Webcam_processor, self).__init__(**kwargs)
        if self.character == 'nvxia':
            assert os.path.exists(os.path.join('model_data', 'characters', 'nvxia')), 'Current released version does not support other characters, like Nvxia.'
            self.character_model = create_nvxia_model(self.nvxia_model_path)

    def webcam_run_local(self, video_file_path=None):
        """
        24.4 FPS of forward prop. on 1070Ti
        """
        capture = OpenCVCapture(video_file_path, show=False)
        None
        frame_id = 0
        if self.visulize_platform == 'integrated':
            visualizer = Open3d_visualizer(multi_mode=not args().show_largest_person_only)
        elif self.visulize_platform == 'blender':
            sender = SocketClient_blender()
        elif self.visulize_platform == 'vis_server':
            RS = Results_sender()
        if self.make_tracking:
            if args().tracker == 'norfair':
                if args().tracking_target == 'centers':
                    tracker = Tracker(distance_function=euclidean_distance, distance_threshold=80)
                elif args().tracking_target == 'keypoints':
                    tracker = Tracker(distance_function=keypoints_distance, distance_threshold=60)
            else:
                tracker = Tracker()
        if self.temporal_optimization:
            filter_dict = {}
            subjects_motion_sequences = {}
        for i in range(10):
            self.single_image_forward(np.zeros((512, 512, 3)).astype(np.uint8))
        counter = Time_counter(thresh=1)
        while True:
            start_time_perframe = time.time()
            frame = capture.read()
            if frame is None:
                continue
            frame_id += 1
            counter.start()
            with torch.no_grad():
                outputs = self.single_image_forward(frame)
            counter.count()
            if outputs is not None and outputs['detection_flag']:
                reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
                results = self.reorganize_results(outputs, [frame_id for _ in range(len(reorganize_idx))], reorganize_idx)
                if args().show_largest_person_only or self.visulize_platform == 'blender':
                    max_id = np.argmax(np.array([result['cam'][0] for result in results[frame_id]]))
                    results[frame_id] = [results[frame_id][max_id]]
                    tracked_ids = np.array([0])
                elif args().make_tracking:
                    if args().tracker == 'norfair':
                        if args().tracking_target == 'centers':
                            detections = [Detection(points=result['cam'][[2, 1]] * args().input_size) for result in results[frame_id]]
                        elif args().tracking_target == 'keypoints':
                            detections = [Detection(points=result['pj2d_org']) for result in results[frame_id]]
                        if frame_id == 1:
                            for _ in range(8):
                                tracked_objects = tracker.update(detections=detections)
                        tracked_objects = tracker.update(detections=detections)
                        if len(tracked_objects) == 0:
                            continue
                        tracked_ids = get_tracked_ids(detections, tracked_objects)
                    else:
                        tracked_ids = tracker.update(results[frame_id])
                    if len(tracked_ids) == 0 or len(tracked_ids) > len(results[frame_id]):
                        continue
                else:
                    tracked_ids = np.arange(len(results[frame_id]))
                cv2.imshow('Input', frame[:, :, ::-1])
                cv2.waitKey(1)
                if self.temporal_optimization:
                    for sid, tid in enumerate(tracked_ids):
                        if tid not in filter_dict:
                            filter_dict[tid] = create_OneEuroFilter(args().smooth_coeff)
                            subjects_motion_sequences[tid] = {}
                        results[frame_id][sid] = temporal_optimize_result(results[frame_id][sid], filter_dict[tid])
                        subjects_motion_sequences[tid][frame_id] = results[frame_id][sid]
                cams = np.array([result['cam'] for result in results[frame_id]])
                cams[:, 2] -= 0.26
                trans = np.array([convert_cam_to_3d_trans(cam) for cam in cams])
                poses = np.array([result['poses'] for result in results[frame_id]])
                betas = np.array([result['betas'] for result in results[frame_id]])
                kp3ds = np.array([result['j3d_smpl24'] for result in results[frame_id]])
                verts = np.array([result['verts'] for result in results[frame_id]])
                if self.visulize_platform == 'vis_server':
                    RS.send_results(poses=poses, betas=betas, trans=trans, ids=tracked_ids)
                elif self.visulize_platform == 'blender':
                    sender.send([0, poses[0].tolist(), trans[0].tolist(), frame_id])
                elif self.visulize_platform == 'integrated':
                    if self.character == 'nvxia':
                        verts = self.character_model(poses)['verts'].numpy()
                    if args().show_largest_person_only:
                        trans_largest = trans[0] if self.add_trans else None
                        visualizer.run(verts[0], trans=trans_largest)
                    else:
                        visualizer.run_multiperson(verts, trans=trans, tracked_ids=tracked_ids)

    def webcam_run_remote(self):
        None
        capture = Server_port_receiver()
        while True:
            frame = capture.receive()
            if isinstance(frame, list):
                continue
            with torch.no_grad():
                outputs = self.single_image_forward(frame)
            if outputs is not None:
                verts = outputs['verts'][0].cpu().numpy()
                verts = verts * 50 + np.array([0, 0, 100])
                capture.send(verts)
            else:
                capture.send(['failed'])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeter_Dict(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.dict_store = {}
        self.count = 0

    def update(self, val, n=1):
        for key, value in val.items():
            if key not in self.dict_store:
                self.dict_store[key] = []
            if torch.is_tensor(value):
                value = value.item()
            self.dict_store[key].append(value)
        self.count += n

    def sum(self):
        dict_sum = {}
        for k, v in self.dict_store.items():
            dict_sum[k] = round(float(sum(v)), 2)
        return dict_sum

    def avg(self):
        dict_sum = self.sum()
        dict_avg = {}
        for k, v in dict_sum.items():
            dict_avg[k] = round(v / self.count, 2)
        return dict_avg


class AlternateCorrBlock:

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]
        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()
            coords_i = (coords / 2 ** i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))
        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()
        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = layer1, layer2
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class BasicMotionEncoder(nn.Module):

    def __init__(self, corr_levels, corr_radius):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class FlowHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class BasicUpdateBlock(nn.Module):

    def __init__(self, corr_levels, corr_radius, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(corr_levels, corr_radius)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


class CorrBlock:

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class BottleneckBlock(nn.Module):

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()
        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class SmallEncoder(nn.Module):

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = layer1, layer2
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class ConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class SmallMotionEncoder(nn.Module):

    def __init__(self, corr_levels, corr_radius):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):

    def __init__(self, corr_levels, corr_radius, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(corr_levels, corr_radius)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + 64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        return net, None, delta_flow


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = 8 * flow.shape[2], 8 * flow.shape[3]
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class RAFT(nn.Module):

    def __init__(self, small=False):
        super(RAFT, self).__init__()
        if small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.corr_levels = 4
            self.corr_radius = 3
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.corr_levels = 4
            self.corr_radius = 4
        self.alternate_corr = False
        if small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=0)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=0)
            self.update_block = SmallUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=0)
            self.update_block = BasicUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        hdim = self.hidden_dim
        cdim = self.context_dim
        with autocast(enabled=True):
            fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)
        with autocast(enabled=True):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init
        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(enabled=True):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)
        if test_mode:
            return coords1 - coords0, flow_up
        return flow_predictions


class FlowExtract(nn.Module):

    def __init__(self, device='cuda'):
        super(FlowExtract, self).__init__()
        model = torch.nn.DataParallel(RAFT())
        model.load_state_dict(torch.load(args().raft_model_path))
        self.device = device
        self.model = model.module.eval()

    @torch.no_grad()
    def forward(self, images, seq_inds):
        input_images = images.permute(0, 3, 1, 2)
        clip_ids = seq_inds[:, 1]
        target_img_inds = torch.arange(len(input_images))
        source_img_inds = target_img_inds - 1
        source_img_inds[clip_ids == 0] = target_img_inds[clip_ids == 0]
        num = len(source_img_inds) // 2
        flows_low1, flows_high1 = self.model(input_images[source_img_inds[:num]].contiguous(), input_images[target_img_inds[:num]].contiguous(), iters=20, upsample=False, test_mode=True)
        flows_low2, flows_high2 = self.model(input_images[source_img_inds[num:]].contiguous(), input_images[target_img_inds[num:]].contiguous(), iters=20, upsample=False, test_mode=True)
        flows = F.interpolate(torch.cat([flows_high1, flows_high2], 0), size=(128, 128), mode='bilinear', align_corners=True) / 8
        return flows


def huber_d_kernel(s_sqrt, delta, eps: 'float'=1e-10):
    if s_sqrt.requires_grad or delta.requires_grad:
        rho_d_sqrt = (delta.clamp(min=eps).sqrt() * s_sqrt.clamp(min=eps).rsqrt()).clamp(max=1.0)
    else:
        rho_d_sqrt = (delta / s_sqrt.clamp_(min=eps)).clamp_(max=1.0).sqrt_()
    return rho_d_sqrt


def huber_kernel(s_sqrt, delta):
    half_rho = torch.where(s_sqrt <= delta, 0.5 * torch.square(s_sqrt), delta * s_sqrt - 0.5 * torch.square(delta))
    return half_rho


class HuberPnPCost(object):

    def __init__(self, delta=1.0, eps=1e-10):
        super(HuberPnPCost, self).__init__()
        self.eps = eps
        self.delta = delta

    def set_param(self, *args, **kwargs):
        pass

    def compute(self, x2d_proj, x2d, w2d, jac_cam=None, out_residual=False, out_cost=False, out_jacobian=False):
        """
        Args:
            x2d_proj: Shape (*, n, 2)
            x2d: Shape (*, n, 2)
            w2d: Shape (*, n, 2)
            jac_cam: Shape (*, n, 2, 4 or 6), Jacobian of x2d_proj w.r.t. pose
            out_residual (Tensor | bool): Shape (*, n*2) or equivalent shape
            out_cost (Tensor | bool): Shape (*, )
            out_jacobian (Tensor | bool): Shape (*, n*2, 4 or 6) or equivalent shape
        """
        bs = x2d_proj.shape[:-2]
        pn = x2d_proj.size(-2)
        delta = self.delta
        if not isinstance(delta, torch.Tensor):
            delta = x2d.new_tensor(delta)
        delta = delta[..., None]
        residual = (x2d_proj - x2d) * w2d
        s_sqrt = residual.norm(dim=-1)
        if out_cost is not False:
            half_rho = huber_kernel(s_sqrt, delta)
            if not isinstance(out_cost, torch.Tensor):
                out_cost = None
            cost = torch.sum(half_rho, dim=-1, out=out_cost)
        else:
            cost = None
        if out_residual is not False or out_jacobian is not False:
            rho_d_sqrt = huber_d_kernel(s_sqrt, delta, eps=self.eps)
            if out_residual is not False:
                if isinstance(out_residual, torch.Tensor):
                    out_residual = out_residual.view(*bs, pn, 2)
                else:
                    out_residual = None
                residual = torch.mul(residual, rho_d_sqrt[..., None], out=out_residual).view(*bs, pn * 2)
            if out_jacobian is not False:
                assert jac_cam is not None
                dof = jac_cam.size(-1)
                if isinstance(out_jacobian, torch.Tensor):
                    out_jacobian = out_jacobian.view(*bs, pn, 2, dof)
                else:
                    out_jacobian = None
                jacobian = torch.mul(jac_cam, (w2d * rho_d_sqrt[..., None])[..., None], out=out_jacobian).view(*bs, pn * 2, dof)
        if out_residual is False:
            residual = None
        if out_jacobian is False:
            jacobian = None
        return residual, cost, jacobian

    def reshape_(self, *batch_shape):
        if isinstance(self.delta, torch.Tensor):
            self.delta = self.delta.reshape(*batch_shape)
        return self

    def expand_(self, *batch_shape):
        if isinstance(self.delta, torch.Tensor):
            self.delta = self.delta.expand(*batch_shape)
        return self

    def repeat_(self, *batch_repeat):
        if isinstance(self.delta, torch.Tensor):
            self.delta = self.delta.repeat(*batch_repeat)
        return self

    def shallow_copy(self):
        return HuberPnPCost(delta=self.delta, eps=self.eps)


class AdaptiveHuberPnPCost(HuberPnPCost):

    def __init__(self, delta=None, relative_delta=0.5, eps=1e-10):
        super(HuberPnPCost, self).__init__()
        self.delta = delta
        self.relative_delta = relative_delta
        self.eps = eps

    def set_param(self, x2d, w2d):
        x2d_std = torch.var(x2d, dim=-2).sum(dim=-1).sqrt()
        self.delta = w2d.mean(dim=(-2, -1)) * x2d_std * self.relative_delta

    def shallow_copy(self):
        return AdaptiveHuberPnPCost(delta=self.delta, relative_delta=self.relative_delta, eps=self.eps)


def evaluate_pnp(x3d, x2d, w2d, pose, camera, cost_fun, out_jacobian=False, out_residual=False, out_cost=False, **kwargs):
    """
    Args:
        x3d (torch.Tensor): Shape (*, n, 3)
        x2d (torch.Tensor): Shape (*, n, 2)
        w2d (torch.Tensor): Shape (*, n, 2)
        pose (torch.Tensor): Shape (*, 4 or 7)
        camera: Camera object of batch size (*, )
        cost_fun: PnPCost object of batch size (*, )
        out_jacobian (torch.Tensor | bool): When a tensor is passed, treated as the output tensor;
            when True, returns the Jacobian; when False, skip the computation and returns None
        out_residual (torch.Tensor | bool): When a tensor is passed, treated as the output tensor;
            when True, returns the residual; when False, skip the computation and returns None
        out_cost (torch.Tensor | bool): When a tensor is passed, treated as the output tensor;
            when True, returns the cost; when False, skip the computation and returns None

    Returns:
        Tuple:
            residual (torch.Tensor | None): Shape (*, n*2)
            cost (torch.Tensor | None): Shape (*, )
            jacobian (torch.Tensor | None): Shape (*, n*2, 4 or 6)
    """
    x2d_proj, jac_cam = camera.project(x3d, pose, out_jac=out_jacobian.view(x2d.shape[:-1] + (2, out_jacobian.size(-1))) if isinstance(out_jacobian, torch.Tensor) else out_jacobian, **kwargs)
    residual, cost, jacobian = cost_fun.compute(x2d_proj, x2d, w2d, jac_cam=jac_cam, out_residual=out_residual, out_cost=out_cost, out_jacobian=out_jacobian)
    return residual, cost, jacobian


def skew(x):
    """
    Args:
        x (torch.Tensor): shape (*, 3)

    Returns:
        torch.Tensor: (*, 3, 3), skew symmetric matrices
    """
    mat = x.new_zeros(x.shape[:-1] + (3, 3))
    mat[..., [2, 0, 1], [1, 2, 0]] = x
    mat[..., [1, 2, 0], [2, 0, 1]] = -x
    return mat


def quaternion_to_rot_mat(quaternions):
    """
    Args:
        quaternions (torch.Tensor): (*, 4)

    Returns:
        torch.Tensor: (*, 3, 3)
    """
    if quaternions.requires_grad:
        w, i, j, k = torch.unbind(quaternions, -1)
        rot_mats = torch.stack((1 - 2 * (j * j + k * k), 2 * (i * j - k * w), 2 * (i * k + j * w), 2 * (i * j + k * w), 1 - 2 * (i * i + k * k), 2 * (j * k - i * w), 2 * (i * k - j * w), 2 * (j * k + i * w), 1 - 2 * (i * i + j * j)), dim=-1).reshape(quaternions.shape[:-1] + (3, 3))
    else:
        w, v = quaternions.split([1, 3], dim=-1)
        rot_mats = 2 * (w.unsqueeze(-1) * skew(v) + v.unsqueeze(-1) * v.unsqueeze(-2))
        diag = torch.diagonal(rot_mats, dim1=-2, dim2=-1)
        diag += w * w - (v.unsqueeze(-2) @ v.unsqueeze(-1)).squeeze(-1)
    return rot_mats


def yaw_to_rot_mat(yaw):
    """
    Args:
        yaw (torch.Tensor): (*)

    Returns:
        torch.Tensor: (*, 3, 3)
    """
    sin_yaw = torch.sin(yaw)
    cos_yaw = torch.cos(yaw)
    rot_mats = yaw.new_zeros(yaw.shape + (3, 3))
    rot_mats[..., 0, 0] = cos_yaw
    rot_mats[..., 2, 2] = cos_yaw
    rot_mats[..., 0, 2] = sin_yaw
    rot_mats[..., 2, 0] = -sin_yaw
    rot_mats[..., 1, 1] = 1
    return rot_mats


def pnp_denormalize(offset, pose_norm):
    pose = torch.empty_like(pose_norm)
    pose[..., 3:] = pose_norm[..., 3:]
    pose[..., :3] = pose_norm[..., :3] - ((yaw_to_rot_mat(pose_norm[..., 3]) if pose_norm.size(-1) == 4 else quaternion_to_rot_mat(pose_norm[..., 3:])) @ offset.unsqueeze(-1)).squeeze(-1)
    return pose


def pnp_normalize(x3d, pose=None, detach_transformation=True):
    """
    Args:
        x3d (torch.Tensor): Shape (*, n, 3)
        pose (torch.Tensor | None): Shape (*, 4)
        detach_transformation (bool)

    Returns:
        Tuple[torch.Tensor]:
            offset: Shape (*, 1, 3)
            x3d_norm: Shape (*, n, 3), normalized x3d
            pose_norm: Shape (*, ), transformed pose
    """
    offset = torch.mean(x3d.detach() if detach_transformation else x3d, dim=-2)
    x3d_norm = x3d - offset.unsqueeze(-2)
    if pose is not None:
        pose_norm = torch.empty_like(pose)
        pose_norm[..., 3:] = pose[..., 3:]
        pose_norm[..., :3] = pose[..., :3] + ((yaw_to_rot_mat(pose[..., 3]) if pose.size(-1) == 4 else quaternion_to_rot_mat(pose[..., 3:])) @ offset.unsqueeze(-1)).squeeze(-1)
    else:
        pose_norm = None
    return offset, x3d_norm, pose_norm


class EProPnPBase(torch.nn.Module, metaclass=ABCMeta):
    """
    End-to-End Probabilistic Perspective-n-Points.

    Args:
        mc_samples (int): Number of total Monte Carlo samples
        num_iter (int): Number of AMIS iterations
        normalize (bool)
        eps (float)
        solver (dict): PnP solver
    """

    def __init__(self, mc_samples=512, num_iter=4, normalize=False, eps=1e-05, solver=None):
        super(EProPnPBase, self).__init__()
        assert num_iter > 0
        assert mc_samples % num_iter == 0
        self.mc_samples = mc_samples
        self.num_iter = num_iter
        self.iter_samples = self.mc_samples // self.num_iter
        self.eps = eps
        self.normalize = normalize
        self.solver = solver

    @abstractmethod
    def allocate_buffer(self, *args, **kwargs):
        pass

    @abstractmethod
    def initial_fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def gen_new_distr(self, *args, **kwargs):
        pass

    @abstractmethod
    def gen_old_distr(self, *args, **kwargs):
        pass

    @abstractmethod
    def estimate_params(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        return self.solver(*args, **kwargs)

    def monte_carlo_forward(self, x3d, x2d, w2d, camera, cost_fun, pose_init=None, force_init_solve=True, **kwargs):
        """
        Monte Carlo PnP forward. Returns weighted pose samples drawn from the probability
        distribution of pose defined by the correspondences {x_{3D}, x_{2D}, w_{2D}}.

        Args:
            x3d (Tensor): Shape (num_obj, num_points, 3)
            x2d (Tensor): Shape (num_obj, num_points, 2)
            w2d (Tensor): Shape (num_obj, num_points, 2)
            camera: Camera object of batch size (num_obj, )
            cost_fun: PnPCost object of batch size (num_obj, )
            pose_init (Tensor | None): Shape (num_obj, 4 or 7), optional. The target pose
                (y_{gt}) can be passed for training with Monte Carlo pose loss
            force_init_solve (bool): Whether to force using the initialization solver when
                pose_init is not None

        Returns:
            Tuple:
                pose_opt (Tensor): Shape (num_obj, 4 or 7), PnP solution y*
                cost (Tensor | None): Shape (num_obj, ), is not None when with_cost=True
                pose_opt_plus (Tensor | None): Shape (num_obj, 4 or 7), y* + Δy, used in derivative
                    regularization loss, is not None when with_pose_opt_plus=True, can be backpropagated
                pose_samples (Tensor): Shape (mc_samples, num_obj, 4 or 7)
                pose_sample_logweights (Tensor): Shape (mc_samples, num_obj), can be backpropagated
                cost_init (Tensor | None): Shape (num_obj, ), is None when pose_init is None, can be
                    backpropagated
        """
        if self.normalize:
            transform, x3d, pose_init = pnp_normalize(x3d, pose_init, detach_transformation=True)
        assert x3d.dim() == x2d.dim() == w2d.dim() == 3
        num_obj = x3d.size(0)
        evaluate_fun = partial(evaluate_pnp, x3d=x3d, x2d=x2d, w2d=w2d, camera=camera, cost_fun=cost_fun, out_cost=True)
        cost_init = evaluate_fun(pose=pose_init)[1] if pose_init is not None else None
        pose_opt, pose_cov, cost, pose_opt_plus = self.solver(x3d, x2d, w2d, camera, cost_fun, pose_init=pose_init, cost_init=cost_init, with_pose_cov=True, force_init_solve=force_init_solve, normalize_override=False, **kwargs)
        if num_obj > 0:
            pose_samples = x3d.new_empty((self.num_iter, self.iter_samples) + pose_opt.size())
            logprobs = x3d.new_empty((self.num_iter, self.num_iter, self.iter_samples, num_obj))
            cost_pred = x3d.new_empty((self.num_iter, self.iter_samples, num_obj))
            distr_params = self.allocate_buffer(num_obj, dtype=x3d.dtype, device=x3d.device)
            with torch.no_grad():
                self.initial_fit(pose_opt, pose_cov, camera, *distr_params)
            for i in range(self.num_iter):
                new_trans_distr, new_rot_distr = self.gen_new_distr(i, *distr_params)
                pose_samples[i, :, :, :3] = new_trans_distr.sample((self.iter_samples,))
                pose_samples[i, :, :, 3:] = new_rot_distr.sample((self.iter_samples,))
                cost_pred[i] = evaluate_fun(pose=pose_samples[i])[1]
                logprobs[i, :i + 1] = new_trans_distr.log_prob(pose_samples[:i + 1, :, :, :3]) + new_rot_distr.log_prob(pose_samples[:i + 1, :, :, 3:]).flatten(2)
                if i > 0:
                    old_trans_distr, old_rot_distr = self.gen_old_distr(i, *distr_params)
                    logprobs[:i, i] = old_trans_distr.log_prob(pose_samples[i, :, :, :3]) + old_rot_distr.log_prob(pose_samples[i, :, :, 3:]).flatten(2)
                mix_logprobs = torch.logsumexp(logprobs[:i + 1, :i + 1], dim=0) - math.log(i + 1)
                pose_sample_logweights = -cost_pred[:i + 1] - mix_logprobs
                if i == self.num_iter - 1:
                    break
                with torch.no_grad():
                    self.estimate_params(i, pose_samples[:i + 1].reshape(((i + 1) * self.iter_samples,) + pose_opt.size()), pose_sample_logweights.reshape((i + 1) * self.iter_samples, num_obj), *distr_params)
            pose_samples = pose_samples.reshape((self.mc_samples,) + pose_opt.size())
            pose_sample_logweights = pose_sample_logweights.reshape(self.mc_samples, num_obj)
        else:
            pose_samples = x2d.new_zeros((self.mc_samples,) + pose_opt.size())
            pose_sample_logweights = x3d.reshape(self.mc_samples, 0) + x2d.reshape(self.mc_samples, 0) + w2d.reshape(self.mc_samples, 0)
        if self.normalize:
            pose_opt = pnp_denormalize(transform, pose_opt)
            pose_samples = pnp_denormalize(transform, pose_samples)
            if pose_opt_plus is not None:
                pose_opt_plus = pnp_denormalize(transform, pose_opt_plus)
        return pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_init


def cholesky_wrapper(mat, default_diag=None, force_cpu=True):
    device = mat.device
    if force_cpu:
        mat = mat.cpu()
    try:
        tril = torch.cholesky(mat, upper=False)
    except RuntimeError:
        n_dims = mat.size(-1)
        tril = []
        default_tril_single = torch.diag(mat.new_tensor(default_diag)) if default_diag is not None else torch.eye(n_dims, dtype=mat.dtype, device=mat.device)
        for cov in mat.reshape(-1, n_dims, n_dims):
            try:
                tril.append(torch.cholesky(cov, upper=False))
            except RuntimeError:
                tril.append(default_tril_single)
        tril = torch.stack(tril, dim=0).reshape(mat.shape)
    return tril


class EProPnP6DoF(EProPnPBase):
    """
    End-to-End Probabilistic Perspective-n-Points for 6DoF pose estimation.
    The pose is parameterized as [x, y, z, w, i, j, k], where [w, i, j, k]
    is the unit quaternion.
    Adopted proposal distributions:
        position: multivariate t-distribution, degrees of freedom = 3
        orientation: angular central Gaussian distribution
    """

    def __init__(self, *args, acg_mle_iter=3, acg_dispersion=0.001, **kwargs):
        super(EProPnP6DoF, self).__init__(*args, **kwargs)
        self.acg_mle_iter = acg_mle_iter
        self.acg_dispersion = acg_dispersion

    def allocate_buffer(self, num_obj, dtype=torch.float32, device=None):
        trans_mode = torch.empty((self.num_iter, num_obj, 3), dtype=dtype, device=device)
        trans_cov_tril = torch.empty((self.num_iter, num_obj, 3, 3), dtype=dtype, device=device)
        rot_cov_tril = torch.empty((self.num_iter, num_obj, 4, 4), dtype=dtype, device=device)
        return trans_mode, trans_cov_tril, rot_cov_tril

    def initial_fit(self, pose_opt, pose_cov, camera, trans_mode, trans_cov_tril, rot_cov_tril):
        trans_mode[0], rot_mode = pose_opt.split([3, 4], dim=-1)
        trans_cov_tril[0] = cholesky_wrapper(pose_cov[:, :3, :3])
        eye_4 = torch.eye(4, dtype=pose_opt.dtype, device=pose_opt.device)
        transform_mat = camera.get_quaternion_transfrom_mat(rot_mode)
        rot_cov = (transform_mat @ pose_cov[:, 3:, 3:].inverse() @ transform_mat.transpose(-1, -2) + eye_4).inverse()
        rot_cov.div_(rot_cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)[..., None, None])
        rot_cov_tril[0] = cholesky_wrapper(rot_cov + rot_cov.det()[:, None, None] ** 0.25 * (self.acg_dispersion * eye_4))

    @staticmethod
    def gen_new_distr(iter_id, trans_mode, trans_cov_tril, rot_cov_tril):
        new_trans_distr = MultivariateStudentT(3, trans_mode[iter_id], trans_cov_tril[iter_id])
        new_rot_distr = AngularCentralGaussian(rot_cov_tril[iter_id])
        return new_trans_distr, new_rot_distr

    @staticmethod
    def gen_old_distr(iter_id, trans_mode, trans_cov_tril, rot_cov_tril):
        mix_trans_distr = MultivariateStudentT(3, trans_mode[:iter_id, None], trans_cov_tril[:iter_id, None])
        mix_rot_distr = AngularCentralGaussian(rot_cov_tril[:iter_id, None])
        return mix_trans_distr, mix_rot_distr

    def estimate_params(self, iter_id, pose_samples, pose_sample_logweights, trans_mode, trans_cov_tril, rot_cov_tril):
        sample_weights_norm = torch.softmax(pose_sample_logweights, dim=0)
        trans_mode[iter_id + 1] = (sample_weights_norm[..., None] * pose_samples[..., :3]).sum(dim=0)
        trans_dev = pose_samples[..., :3] - trans_mode[iter_id + 1]
        trans_cov = (sample_weights_norm[..., None, None] * trans_dev.unsqueeze(-1) * trans_dev.unsqueeze(-2)).sum(dim=0)
        trans_cov_tril[iter_id + 1] = cholesky_wrapper(trans_cov)
        eye_4 = torch.eye(4, dtype=pose_samples.dtype, device=pose_samples.device)
        rot = pose_samples[..., 3:]
        r_r_t = rot[:, :, :, None] * rot[:, :, None, :]
        rot_cov = eye_4.expand(pose_samples.size(1), 4, 4).clone()
        for _ in range(self.acg_mle_iter):
            M = rot[:, :, None, :] @ rot_cov.inverse() @ rot[:, :, :, None]
            invM_weighted = sample_weights_norm[..., None, None] / M.clamp(min=self.eps)
            invM_weighted_norm = invM_weighted / invM_weighted.sum(dim=0)
            rot_cov = (invM_weighted_norm * r_r_t).sum(dim=0) + eye_4 * self.eps
        rot_cov_tril[iter_id + 1] = cholesky_wrapper(rot_cov + rot_cov.det()[:, None, None] ** 0.25 * (self.acg_dispersion * eye_4))


def solve_wrapper(b, A):
    if A.numel() > 0:
        return torch.linalg.solve(A, b)
    else:
        return b + A.reshape_as(b)


class LMSolver(nn.Module):
    """
    Levenberg-Marquardt solver, with fixed number of iterations.

    - For 4DoF case, the pose is parameterized as [x, y, z, yaw], where yaw is the
    rotation around the Y-axis in radians.
    - For 6DoF case, the pose is parameterized as [x, y, z, w, i, j, k], where
    [w, i, j, k] is the unit quaternion.
    """

    def __init__(self, dof=4, num_iter=10, min_lm_diagonal=1e-06, max_lm_diagonal=1e+32, min_relative_decrease=0.001, initial_trust_region_radius=30.0, max_trust_region_radius=1e+16, eps=1e-05, normalize=False, init_solver=None):
        super(LMSolver, self).__init__()
        self.dof = dof
        self.num_iter = num_iter
        self.min_lm_diagonal = min_lm_diagonal
        self.max_lm_diagonal = max_lm_diagonal
        self.min_relative_decrease = min_relative_decrease
        self.initial_trust_region_radius = initial_trust_region_radius
        self.max_trust_region_radius = max_trust_region_radius
        self.eps = eps
        self.normalize = normalize
        self.init_solver = init_solver

    def forward(self, x3d, x2d, w2d, camera, cost_fun, with_pose_opt_plus=False, pose_init=None, normalize_override=None, **kwargs):
        if isinstance(normalize_override, bool):
            normalize = normalize_override
        else:
            normalize = self.normalize
        if normalize:
            transform, x3d, pose_init = pnp_normalize(x3d, pose_init, detach_transformation=True)
        pose_opt, pose_cov, cost = self.solve(x3d, x2d, w2d, camera, cost_fun, pose_init=pose_init, **kwargs)
        if with_pose_opt_plus:
            step = self.gn_step(x3d, x2d, w2d, pose_opt, camera, cost_fun)
            pose_opt_plus = self.pose_add(pose_opt, step, camera)
        else:
            pose_opt_plus = None
        if normalize:
            pose_opt = pnp_denormalize(transform, pose_opt)
            if pose_cov is not None:
                raise NotImplementedError('Normalized covariance unsupported')
            if pose_opt_plus is not None:
                pose_opt_plus = pnp_denormalize(transform, pose_opt_plus)
        return pose_opt, pose_cov, cost, pose_opt_plus

    def solve(self, x3d, x2d, w2d, camera, cost_fun, pose_init=None, cost_init=None, with_pose_cov=False, with_cost=False, force_init_solve=False, fast_mode=False):
        """
        Args:
            x3d (Tensor): Shape (num_obj, num_pts, 3)
            x2d (Tensor): Shape (num_obj, num_pts, 2)
            w2d (Tensor): Shape (num_obj, num_pts, 2)
            camera: Camera object of batch size (num_obj, )
            cost_fun: PnPCost object of batch size (num_obj, )
            pose_init (None | Tensor): Shape (num_obj, 4 or 7) in [x, y, z, yaw], optional
            cost_init (None | Tensor): Shape (num_obj, ), PnP cost of pose_init, optional
            with_pose_cov (bool): Whether to compute the covariance of pose_opt
            with_cost (bool): Whether to compute the cost of pose_opt
            force_init_solve (bool): Whether to force using the initialization solver when
                pose_init is not None
            fast_mode (bool): Fall back to Gauss-Newton for fast inference

        Returns:
            tuple:
                pose_opt (Tensor): Shape (num_obj, 4 or 7)
                pose_cov (Tensor | None): Shape (num_obj, 4, 4) or (num_obj, 6, 6), covariance
                    of local pose parameterization
                cost (Tensor | None): Shape (num_obj, )
        """
        with torch.no_grad():
            num_obj, num_pts, _ = x2d.size()
            tensor_kwargs = dict(dtype=x2d.dtype, device=x2d.device)
            if num_obj > 0:
                evaluate_fun = partial(evaluate_pnp, x3d=x3d, x2d=x2d, w2d=w2d, camera=camera, cost_fun=cost_fun, clip_jac=not fast_mode)
                if pose_init is None or force_init_solve:
                    assert self.init_solver is not None
                    if pose_init is None:
                        pose_init_solve, _, _ = self.init_solver.solve(x3d, x2d, w2d, camera, cost_fun, fast_mode=fast_mode)
                        pose_opt = pose_init_solve
                    else:
                        if cost_init is None:
                            cost_init = evaluate_fun(pose=pose_init, out_cost=True)[1]
                        pose_init_solve, _, cost_init_solve = self.init_solver.solve(x3d, x2d, w2d, camera, cost_fun, with_cost=True, fast_mode=fast_mode)
                        use_init = cost_init < cost_init_solve
                        pose_init_solve[use_init] = pose_init[use_init]
                        pose_opt = pose_init_solve
                else:
                    pose_opt = pose_init.clone()
                jac = torch.empty((num_obj, num_pts * 2, self.dof), **tensor_kwargs)
                residual = torch.empty((num_obj, num_pts * 2), **tensor_kwargs)
                cost = torch.empty((num_obj,), **tensor_kwargs)
                if fast_mode:
                    for i in range(self.num_iter):
                        evaluate_fun(pose=pose_opt, out_jacobian=jac, out_residual=residual, out_cost=cost)
                        jac_t = jac.transpose(-1, -2)
                        jtj = jac_t @ jac
                        diagonal = torch.diagonal(jtj, dim1=-2, dim2=-1)
                        diagonal += self.eps
                        gradient = jac_t @ residual.unsqueeze(-1)
                        if self.dof == 4:
                            pose_opt -= solve_wrapper(gradient, jtj).squeeze(-1)
                        else:
                            step = -solve_wrapper(gradient, jtj).squeeze(-1)
                            pose_opt[..., :3] += step[..., :3]
                            pose_opt[..., 3:] = F.normalize(pose_opt[..., 3:] + (camera.get_quaternion_transfrom_mat(pose_opt[..., 3:]) @ step[..., 3:, None]).squeeze(-1), dim=-1)
                else:
                    evaluate_fun(pose=pose_opt, out_jacobian=jac, out_residual=residual, out_cost=cost)
                    jac_new = torch.empty_like(jac)
                    residual_new = torch.empty_like(residual)
                    cost_new = torch.empty_like(cost)
                    radius = x2d.new_full((num_obj,), self.initial_trust_region_radius)
                    decrease_factor = x2d.new_full((num_obj,), 2.0)
                    step_is_successful = x2d.new_zeros((num_obj,), dtype=torch.bool)
                    i = 0
                    while i < self.num_iter:
                        self._lm_iter(pose_opt, jac, residual, cost, jac_new, residual_new, cost_new, step_is_successful, radius, decrease_factor, evaluate_fun, camera)
                        i += 1
                    if with_pose_cov:
                        jac[step_is_successful] = jac_new[step_is_successful]
                        jtj = jac.transpose(-1, -2) @ jac
                        diagonal = torch.diagonal(jtj, dim1=-2, dim2=-1)
                        diagonal += self.eps
                    if with_cost:
                        cost[step_is_successful] = cost_new[step_is_successful]
                if with_pose_cov:
                    pose_cov = torch.inverse(jtj)
                else:
                    pose_cov = None
                if not with_cost:
                    cost = None
            else:
                pose_opt = torch.empty((0, 4 if self.dof == 4 else 7), **tensor_kwargs)
                pose_cov = torch.empty((0, self.dof, self.dof), **tensor_kwargs) if with_pose_cov else None
                cost = torch.empty((0,), **tensor_kwargs) if with_cost else None
            return pose_opt, pose_cov, cost

    def _lm_iter(self, pose_opt, jac, residual, cost, jac_new, residual_new, cost_new, step_is_successful, radius, decrease_factor, evaluate_fun, camera):
        jac[step_is_successful] = jac_new[step_is_successful]
        residual[step_is_successful] = residual_new[step_is_successful]
        cost[step_is_successful] = cost_new[step_is_successful]
        residual_ = residual.unsqueeze(-1)
        jac_t = jac.transpose(-1, -2)
        jtj = jac_t @ jac
        jtj_lm = jtj.clone()
        diagonal = torch.diagonal(jtj_lm, dim1=-2, dim2=-1)
        diagonal += diagonal.clamp(min=self.min_lm_diagonal, max=self.max_lm_diagonal) / radius[:, None] + self.eps
        gradient = jac_t @ residual_
        step_ = -solve_wrapper(gradient, jtj_lm)
        pose_new = self.pose_add(pose_opt, step_.squeeze(-1), camera)
        evaluate_fun(pose=pose_new, out_jacobian=jac_new, out_residual=residual_new, out_cost=cost_new)
        model_cost_change = -(step_.transpose(-1, -2) @ (jtj @ step_ / 2 + gradient)).flatten()
        relative_decrease = (cost - cost_new) / model_cost_change
        torch.bitwise_and(relative_decrease >= self.min_relative_decrease, model_cost_change > 0.0, out=step_is_successful)
        pose_opt[step_is_successful] = pose_new[step_is_successful]
        radius[step_is_successful] /= (1.0 - (2.0 * relative_decrease[step_is_successful] - 1.0) ** 3).clamp(min=1.0 / 3.0)
        radius.clamp_(max=self.max_trust_region_radius, min=self.eps)
        decrease_factor.masked_fill_(step_is_successful, 2.0)
        radius[~step_is_successful] /= decrease_factor[~step_is_successful]
        decrease_factor[~step_is_successful] *= 2.0
        return

    def gn_step(self, x3d, x2d, w2d, pose, camera, cost_fun):
        residual, _, jac = evaluate_pnp(x3d, x2d, w2d, pose, camera, cost_fun, out_jacobian=True, out_residual=True)
        jac_t = jac.transpose(-1, -2)
        jtj = jac_t @ jac
        jtj = jtj + torch.eye(self.dof, device=jtj.device, dtype=jtj.dtype) * self.eps
        gradient = jac_t @ residual.unsqueeze(-1)
        step = -solve_wrapper(gradient, jtj).squeeze(-1)
        return step

    def pose_add(self, pose_opt, step, camera):
        if self.dof == 4:
            pose_new = pose_opt + step
        else:
            pose_new = torch.cat((pose_opt[..., :3] + step[..., :3], F.normalize(pose_opt[..., 3:] + (camera.get_quaternion_transfrom_mat(pose_opt[..., 3:]) @ step[..., 3:, None]).squeeze(-1), dim=-1)), dim=-1)
        return pose_new


def project_a(x3d, pose, cam_mats, z_min: 'float'):
    if pose.size(-1) == 4:
        x3d_rot = x3d @ yaw_to_rot_mat(pose[..., -1]).transpose(-1, -2)
    else:
        x3d_rot = x3d @ quaternion_to_rot_mat(pose[..., 3:]).transpose(-1, -2)
    x2dh_proj = (x3d_rot + pose[..., None, :3]) @ cam_mats.transpose(-1, -2)
    z = x2dh_proj[..., 2:3].clamp(min=z_min)
    x2d_proj = x2dh_proj[..., :2] / z
    return x2d_proj, x3d_rot, z


def project_b(x3d, pose, cam_mats, z_min: 'float'):
    if pose.size(-1) == 4:
        x2dh_proj = x3d @ (cam_mats @ yaw_to_rot_mat(pose[..., -1])).transpose(-1, -2) + (cam_mats @ pose[..., :3, None]).squeeze(-1).unsqueeze(-2)
    else:
        x2dh_proj = x3d @ (cam_mats @ quaternion_to_rot_mat(pose[..., 3:])).transpose(-1, -2) + (cam_mats @ pose[..., :3, None]).squeeze(-1).unsqueeze(-2)
    z = x2dh_proj[..., 2:3].clamp(min=z_min)
    x2d_proj = x2dh_proj[..., :2] / z
    return x2d_proj, z


class PerspectiveCamera(object):

    def __init__(self, cam_mats=None, z_min=0.1, img_shape=None, allowed_border=200, lb=None, ub=None):
        """
        Args:
            cam_mats (Tensor): Shape (*, 3, 3)
            img_shape (Tensor | None): Shape (*, 2) in [h, w]
            lb (Tensor | None): Shape (*, 2), lower bound in [x, y]
            ub (Tensor | None): Shape (*, 2), upper bound in [x, y]
        """
        super(PerspectiveCamera, self).__init__()
        self.z_min = z_min
        self.allowed_border = allowed_border
        self.set_param(cam_mats, img_shape, lb, ub)

    def set_param(self, cam_mats, img_shape=None, lb=None, ub=None):
        self.cam_mats = cam_mats
        if img_shape is not None:
            self.lb = -0.5 - self.allowed_border
            self.ub = img_shape[..., [1, 0]] + (-0.5 + self.allowed_border)
        else:
            self.lb = lb
            self.ub = ub

    def project(self, x3d, pose, out_jac=False, clip_jac=True):
        """
        Args:
            x3d (Tensor): Shape (*, n, 3)
            pose (Tensor): Shape (*, 4 or 7)
            out_jac (bool | Tensor): Shape (*, n, 2, 4 or 6)

        Returns:
            Tuple[Tensor]:
                x2d_proj: Shape (*, n, 2)
                jac: Shape (*, n, 2, 4 or 6), Jacobian w.r.t. the local pose in tangent space
        """
        if out_jac is not False:
            x2d_proj, x3d_rot, zcam = project_a(x3d, pose, self.cam_mats, self.z_min)
        else:
            x2d_proj, zcam = project_b(x3d, pose, self.cam_mats, self.z_min)
        lb, ub = self.lb, self.ub
        if lb is not None and ub is not None:
            requires_grad = x2d_proj.requires_grad
            if isinstance(lb, torch.Tensor):
                lb = lb.unsqueeze(-2)
                x2d_proj = torch.max(lb, x2d_proj, out=x2d_proj if not requires_grad else None)
            else:
                x2d_proj.clamp_(min=lb)
            if isinstance(ub, torch.Tensor):
                ub = ub.unsqueeze(-2)
                x2d_proj = torch.min(x2d_proj, ub, out=x2d_proj if not requires_grad else None)
            else:
                x2d_proj.clamp_(max=ub)
        if out_jac is not False:
            if not isinstance(out_jac, torch.Tensor):
                out_jac = None
            jac = self.project_jacobian(x3d_rot, zcam, x2d_proj, out_jac=out_jac, dof=4 if pose.size(-1) == 4 else 6)
            if clip_jac:
                if lb is not None and ub is not None:
                    clip_mask = (zcam == self.z_min) | ((x2d_proj == lb) | (x2d_proj == ub))
                else:
                    clip_mask = zcam == self.z_min
                jac.masked_fill_(clip_mask[..., None], 0)
        else:
            jac = None
        return x2d_proj, jac

    def project_jacobian(self, x3d_rot, zcam, x2d_proj, out_jac, dof):
        if dof == 4:
            d_xzcam_d_yaw = torch.stack((x3d_rot[..., 2], -x3d_rot[..., 0]), dim=-1).unsqueeze(-1)
        elif dof == 6:
            d_x3dcam_d_rot = skew(x3d_rot * 2)
        else:
            raise ValueError('dof must be 4 or 6')
        if zcam.requires_grad or x2d_proj.requires_grad:
            assert out_jac is None, 'out_jac is not supported for backward'
            d_x2d_d_x3dcam = torch.cat((self.cam_mats[..., None, :2, :2] / zcam.unsqueeze(-1), (self.cam_mats[..., None, :2, 2:3] - x2d_proj.unsqueeze(-1)) / zcam.unsqueeze(-1)), dim=-1)
            jac = torch.cat((d_x2d_d_x3dcam, d_x2d_d_x3dcam[..., ::2] @ d_xzcam_d_yaw if dof == 4 else d_x2d_d_x3dcam @ d_x3dcam_d_rot), dim=-1)
        else:
            if out_jac is None:
                jac = torch.empty(x3d_rot.shape[:-1] + (2, dof), device=x3d_rot.device, dtype=x3d_rot.dtype)
            else:
                jac = out_jac
            jac[..., :2] = self.cam_mats[..., None, :2, :2] / zcam.unsqueeze(-1)
            jac[..., 2:3] = (self.cam_mats[..., None, :2, 2:3] - x2d_proj.unsqueeze(-1)) / zcam.unsqueeze(-1)
            jac[..., 3:] = jac[..., ::2] @ d_xzcam_d_yaw if dof == 4 else jac[..., :3] @ d_x3dcam_d_rot
        return jac

    @staticmethod
    def get_quaternion_transfrom_mat(quaternions):
        """
        Get the transformation matrix that maps the local rotation delta in 3D tangent
        space to the 4D space where the quaternion is embedded.

        Args:
            quaternions (torch.Tensor): (*, 4), the quaternion that determines the source
                tangent space

        Returns:
            torch.Tensor: (*, 4, 3)
        """
        w, i, j, k = torch.unbind(quaternions, -1)
        transfrom_mat = torch.stack((i, j, k, -w, -k, j, k, -w, -i, -j, i, -w), dim=-1)
        return transfrom_mat.reshape(quaternions.shape[:-1] + (4, 3))

    def reshape_(self, *batch_shape):
        self.cam_mats = self.cam_mats.reshape(*batch_shape, 3, 3)
        if isinstance(self.lb, torch.Tensor):
            self.lb = self.lb.reshape(*batch_shape, 2)
        if isinstance(self.ub, torch.Tensor):
            self.ub = self.ub.reshape(*batch_shape, 2)
        return self

    def expand_(self, *batch_shape):
        self.cam_mats = self.cam_mats.expand(*batch_shape, -1, -1)
        if isinstance(self.lb, torch.Tensor):
            self.lb = self.lb.expand(*batch_shape, -1)
        if isinstance(self.ub, torch.Tensor):
            self.ub = self.ub.expand(*batch_shape, -1)
        return self

    def repeat_(self, *batch_repeat):
        self.cam_mats = self.cam_mats.repeat(*batch_repeat, 1, 1)
        if isinstance(self.lb, torch.Tensor):
            self.lb = self.lb.repeat(*batch_repeat, 1)
        if isinstance(self.ub, torch.Tensor):
            self.ub = self.ub.repeat(*batch_repeat, 1)
        return self

    def shallow_copy(self):
        return PerspectiveCamera(cam_mats=self.cam_mats, z_min=self.z_min, allowed_border=self.allowed_border, lb=self.lb, ub=self.ub)


class RSLMSolver(LMSolver):
    """
    Random Sample Levenberg-Marquardt solver, a generalization of RANSAC.
    Used for initialization in ambiguous problems.
    """

    def __init__(self, num_points=16, num_proposals=64, num_iter=3, **kwargs):
        super(RSLMSolver, self).__init__(num_iter=num_iter, **kwargs)
        self.num_points = num_points
        self.num_proposals = num_proposals

    def center_based_init(self, x2d, x3d, camera, eps=1e-06):
        x2dc = solve_wrapper(F.pad(x2d, [0, 1], mode='constant', value=1.0).transpose(-1, -2), camera.cam_mats).transpose(-1, -2)
        x2dc = x2dc[..., :2] / x2dc[..., 2:].clamp(min=eps)
        x2dc_std, x2dc_mean = torch.std_mean(x2dc, dim=-2)
        x3d_std = torch.std(x3d, dim=-2)
        if self.dof == 4:
            t_vec = F.pad(x2dc_mean, [0, 1], mode='constant', value=1.0) * (x3d_std[..., 1] / x2dc_std[..., 1].clamp(min=eps)).unsqueeze(-1)
        else:
            t_vec = F.pad(x2dc_mean, [0, 1], mode='constant', value=1.0) * (math.sqrt(2 / 3) * x3d_std.norm(dim=-1) / x2dc_std.norm(dim=-1).clamp(min=eps)).unsqueeze(-1)
        return t_vec

    def solve(self, x3d, x2d, w2d, camera, cost_fun, **kwargs):
        with torch.no_grad():
            bs, pn, _ = x2d.size()
            if bs > 0:
                mean_weight = w2d.mean(dim=-1).reshape(1, bs, pn).expand(self.num_proposals, -1, -1)
                inds = torch.multinomial(mean_weight.reshape(-1, pn), self.num_points).reshape(self.num_proposals, bs, self.num_points)
                bs_inds = torch.arange(bs, device=inds.device)
                inds += (bs_inds * pn)[:, None]
                x2d_samples = x2d.reshape(-1, 2)[inds]
                x3d_samples = x3d.reshape(-1, 3)[inds]
                w2d_samples = w2d.reshape(-1, 2)[inds]
                pose_init = x2d.new_empty((self.num_proposals, bs, 4 if self.dof == 4 else 7))
                pose_init[..., :3] = self.center_based_init(x2d, x3d, camera)
                if self.dof == 4:
                    pose_init[..., 3] = torch.rand((self.num_proposals, bs), dtype=x2d.dtype, device=x2d.device) * (2 * math.pi)
                else:
                    pose_init[..., 3:] = torch.randn((self.num_proposals, bs, 4), dtype=x2d.dtype, device=x2d.device)
                    q_norm = pose_init[..., 3:].norm(dim=-1)
                    pose_init[..., 3:] /= q_norm.unsqueeze(-1)
                    pose_init.view(-1, 7)[(q_norm < self.eps).flatten(), 3:] = x2d.new_tensor([1, 0, 0, 0])
                camera_expand = camera.shallow_copy()
                camera_expand.repeat_(self.num_proposals)
                cost_fun_expand = cost_fun.shallow_copy()
                cost_fun_expand.repeat_(self.num_proposals)
                pose, _, _ = super(RSLMSolver, self).solve(x3d_samples.reshape(self.num_proposals * bs, self.num_points, 3), x2d_samples.reshape(self.num_proposals * bs, self.num_points, 2), w2d_samples.reshape(self.num_proposals * bs, self.num_points, 2), camera_expand, cost_fun_expand, pose_init=pose_init.reshape(self.num_proposals * bs, pose_init.size(-1)), **kwargs)
                pose = pose.reshape(self.num_proposals, bs, pose.size(-1))
                cost = evaluate_pnp(x3d, x2d, w2d, pose, camera, cost_fun, out_cost=True)[1]
                min_cost, min_cost_ind = cost.min(dim=0)
                pose = pose[min_cost_ind, torch.arange(bs, device=pose.device)]
            else:
                pose = x2d.new_empty((0, 4 if self.dof == 4 else 7))
                min_cost = x2d.new_empty((0,))
            return pose, None, min_cost


def prepare_camera_mats(fovs, length, device):
    cam_mats = torch.zeros(length, 3, 3)
    if fovs is not None:
        cam_mats[:, 0, 0] = cam_mats[:, 1, 1] = fovs
    else:
        cam_mats[:, 0, 0] = cam_mats[:, 1, 1] = 1.0 / np.tan(np.radians(args().FOV / 2))
    cam_mats[:, 2, 2] = 1
    return cam_mats


class EProPnP6DoFSolver(nn.Module):

    def __init__(self):
        super(EProPnP6DoFSolver, self).__init__()
        self.epropnp = EProPnP6DoF(mc_samples=512, num_iter=4, solver=LMSolver(dof=6, num_iter=10, init_solver=RSLMSolver(dof=6, num_points=8, num_proposals=128, num_iter=5)))
        self.camera = PerspectiveCamera()
        self.cost_fun = AdaptiveHuberPnPCost(relative_delta=0.5)
        self.log_weight_scale = nn.Parameter(torch.zeros(2))

    def epro_pnp_train(self, x3d, x2d, w2d, cam_mats, out_pose):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            self.camera.set_param(cam_mats)
            self.cost_fun.set_param(x2d, w2d)
            pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt = self.epropnp.monte_carlo_forward(x3d.detach(), x2d, w2d, self.camera, self.cost_fun, pose_init=out_pose, force_init_solve=True, with_pose_opt_plus=True)
        return pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt

    def epro_pnp_inference(self, x3d, x2d, w2d, cam_mats, fast_mode=False):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            self.camera.set_param(cam_mats)
            self.cost_fun.set_param(x2d, w2d)
            pose_opt, _, _, _ = self.epropnp(x3d.detach(), x2d, w2d, self.camera, self.cost_fun, fast_mode=fast_mode)
        return pose_opt

    def solve(self, x3ds, x2ds, w2ds, fovs, fast_mode=True):
        cam_mats = prepare_camera_mats(fovs, len(x3ds), x3ds.device)
        self.camera.set_param(cam_mats)
        self.cost_fun.set_param(x2ds.detach(), w2ds)
        pose_opt, _, _, _ = self.epropnp(x3ds, x2ds, w2ds, self.camera, self.cost_fun, fast_mode=fast_mode)
        return pose_opt


class HeatmapGenerator:

    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        gaussian_distribution = -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)
        self.g = np.exp(gaussian_distribution)

    def single_process(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res), dtype=np.float32)
        sigma = self.sigma
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                        continue
                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))
                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]
                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

    def batch_process(self, batch_joints):
        vis = ((batch_joints > -1.0).sum(-1) == batch_joints.shape[-1]).unsqueeze(-1).float()
        batch_joints = (torch.cat([batch_joints, vis], -1).unsqueeze(1) + 1) / 2 * self.output_res
        heatmaps = []
        for joints in batch_joints:
            heatmaps.append(torch.from_numpy(self.single_process(joints)))
        return torch.stack(heatmaps)


class JointsGenerator:

    def __init__(self, max_num_people, num_joints, output_res, tag_per_joint):
        self.max_num_people = max_num_people
        self.num_joints = num_joints
        self.output_res = output_res
        self.tag_per_joint = tag_per_joint

    def single_process(self, joints):
        visible_nodes = np.zeros((self.max_num_people, self.num_joints, 2))
        output_res = self.output_res
        for i in range(min(len(joints), self.max_num_people)):
            tot = 0
            for idx, pt in enumerate(joints[i]):
                x, y = int(pt[0]), int(pt[1])
                if pt[2] > 0 and x >= 0 and y >= 0 and x < self.output_res and y < self.output_res:
                    if self.tag_per_joint:
                        visible_nodes[i][tot] = idx * output_res ** 2 + y * output_res + x, 1
                    else:
                        visible_nodes[i][tot] = y * output_res + x, 1
                    tot += 1
        return visible_nodes

    def batch_process(self, batch_joints):
        vis = ((batch_joints > -1.0).sum(-1) == batch_joints.shape[-1]).unsqueeze(-1).float()
        batch_joints = (torch.cat([batch_joints, vis], -1).unsqueeze(1) + 1) / 2 * self.output_res
        joints_processed = []
        for joints in batch_joints:
            joints_processed.append(self.single_process(joints))
        return torch.from_numpy(np.array(joints_processed)).long()


def paste_over(im_src, im_dst, center):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.
    Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
    `im_src` becomes visible).
    Args:
        im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
            at each pixel. Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where the center of `im_src` should be placed.
    """
    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])
    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src
    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]
    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32) / 255
    im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = alpha * color_src + (1 - alpha) * region_dst


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def occlude_with_objects(im, occluders, occluder=None, center=None):
    """Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset."""
    result = im.copy()
    width_height = np.asarray([im.shape[1], im.shape[0]])
    im_scale_factor = min(width_height) / 256
    if occluder is None:
        occluder_name = random.choice(occluders)
        occluder = np.load(occluder_name, allow_pickle=True)
        random_scale_factor = np.random.uniform(0.3, 0.6)
        scale_factor = random_scale_factor * im_scale_factor
        occluder = resize_by_factor(occluder, scale_factor)
    if center is None:
        center = np.random.uniform(width_height / 4, width_height * 3 / 4)
    paste_over(im_src=occluder, im_dst=result, center=center)
    return result, occluder, center


class Synthetic_occlusion(object):

    def __init__(self, path):
        None
        if not os.path.exists(path):
            path = '/home/yusun/DataCenter2/datasets/VOC2012'
        occluders_dir = os.path.join(path, 'syn_occlusion_objects')
        self.occluders = glob.glob(os.path.join(occluders_dir, '*.npy'))
        None

    def __call__(self, img, occluder=None, center=None):
        occluded_img, occluder, center = occlude_with_objects(img, self.occluders, occluder=occluder, center=center)
        return occluded_img, occluder, center


def calc_aabb(ptSets):
    ptLeftTop = np.array([np.min(ptSets[:, 0]), np.min(ptSets[:, 1])])
    ptRightBottom = np.array([np.max(ptSets[:, 0]), np.max(ptSets[:, 1])])
    return np.array([ptLeftTop, ptRightBottom])


def _calc_bbox_normed(full_kps):
    bboxes = []
    for kps_i in full_kps:
        if (kps_i[:, 0] > -2).sum() > 0:
            bboxes.append(calc_aabb(kps_i[kps_i[:, 0] > -2]))
        else:
            bboxes.append(np.zeros((2, 2)))
    return bboxes


def _check_upper_bound_lower_bound_(kps, ub=1, lb=-1):
    for k in kps:
        if k >= ub or k <= lb:
            return False
    return True


def convert_bbox2scale(ltrb, input_size):
    h, w = input_size
    l, t, r, b = ltrb
    scale = max((r - l) / w, (b - t) / h)
    return scale


def convert_scale_to_depth_level(scale):
    cam3dmap_anchors = cam3dmap_anchor[None]
    return torch.argmin(torch.abs(scale[:, None].repeat(1, scale_num) - cam3dmap_anchors), dim=1)


def detect_occluded_person(person_centers, full_kp2ds, thresh=2 * 64 / 512.0):
    person_num = len(person_centers)
    occluded_by_who = np.ones(person_num) * -1
    if person_num > 1:
        for inds, (person_center, kp2d) in enumerate(zip(person_centers, full_kp2ds)):
            dist = np.sqrt(((person_centers - person_center) ** 2).sum(-1))
            if (dist > 0).sum() > 0:
                if (dist[dist > 0] < thresh).sum() > 0:
                    closet_idx = np.where(dist == np.min(dist[dist > 0]))[0][0]
                    if occluded_by_who[closet_idx] < 0:
                        occluded_by_who[inds] = closet_idx
    return occluded_by_who.astype(np.int32)


def get_bounding_bbox(full_kp2d):
    full_kp2d = full_kp2d[(full_kp2d != -2).sum(-1) >= 2]
    if len(full_kp2d) > 0:
        box = calc_aabb(full_kp2d)
        return box
    else:
        return np.array([[0, 0], [512, 512]])


def get_image_cut_box(leftTop, rightBottom, ExpandsRatio, Center=None, force_square=False):
    ExpandsRatio = [ExpandsRatio, ExpandsRatio, ExpandsRatio, ExpandsRatio]

    def _expand_crop_box(lt, rb, scale):
        center = (lt + rb) / 2.0
        xl, xr, yt, yb = lt[0] - center[0], rb[0] - center[0], lt[1] - center[1], rb[1] - center[1]
        xl, xr, yt, yb = xl * scale[0], xr * scale[1], yt * scale[2], yb * scale[3]
        lt, rb = np.array([center[0] + xl, center[1] + yt]), np.array([center[0] + xr, center[1] + yb])
        lb, rt = np.array([center[0] + xl, center[1] + yb]), np.array([center[0] + xr, center[1] + yt])
        center = (lt + rb) / 2
        return center, lt, rt, rb, lb
    if Center == None:
        Center = (leftTop + rightBottom) // 2
    Center, leftTop, rightTop, rightBottom, leftBottom = _expand_crop_box(leftTop, rightBottom, ExpandsRatio)
    offset = (rightBottom - leftTop) // 2
    cx = offset[0]
    cy = offset[1]
    if force_square:
        r = max(cx, cy)
        cx = r
        cy = r
    x = int(Center[0])
    y = int(Center[1])
    return [x - cx, y - cy], [x + cx, y + cy]


def flip_pose(pose):
    flipped_parts = constants.SMPL_POSE_FLIP_PERM
    pose = pose[flipped_parts]
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose


def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    R = np.array([[np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot)), 0], [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot)), 0], [0, 0, 1]])
    per_rdg, _ = cv2.Rodrigues(aa)
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = resrot.T[0]
    return aa


def pose_processing(pose, rot, flip, valid_grot=False, valid_pose=False):
    """Process SMPL theta parameters  and apply all augmentation transforms."""
    if valid_grot:
        pose[:3] = rot_aa(pose[:3], rot)
    if flip and valid_pose:
        pose = flip_pose(pose)
    return pose


def image_pad_white_bg(image, pad_trbl=None, pad_ratio=1.0, pad_cval=255):
    if pad_trbl is None:
        pad_trbl = compute_paddings_to_reach_aspect_ratio(image.shape, pad_ratio)
    pad_func = iaa.Sequential([iaa.Pad(px=pad_trbl, keep_size=False, pad_mode='constant', pad_cval=pad_cval)])
    image_aug = pad_func(image=image)
    return image_aug, np.array([*image_aug.shape[:2], *[0, 0, 0, 0], *pad_trbl])


def convert2keypointsonimage(kp2d, image_shape):
    kps = KeypointsOnImage([Keypoint(x=x, y=y) for x, y in kp2d], shape=image_shape)
    return kps


def img_kp_rotate(image, kp2ds=None, rotate=0):
    """
    Perform augmentation of image (and kp2ds) via rotation.
    Input args:
        image : np.array, size H x W x 3
        kp2ds : np.array, size N x K x 2/3, the K 2D joints of N people
        rotate : int, radians angle of rotation on image plane, such as 30 degree
    return:
        augmented image: np.array, size H x W x 3
        augmented kp2ds if given, in the same size as input kp2ds
    """
    aug_list = []
    if rotate != 0:
        aug_list += [iaa.Affine(rotate=rotate)]
        aug_seq = iaa.Sequential(aug_list)
        image_aug = np.array(aug_seq(image=image))
        if kp2ds is not None:
            kp2ds_aug = []
            invalid_mask = [(kp2d <= 0) for kp2d in kp2ds]
            for idx, kp2d in enumerate(kp2ds):
                kps = convert2keypointsonimage(kp2d[:, :2], image.shape)
                kps_aug = aug_seq(keypoints=kps)
                kp2d[:, :2] = kps_aug.to_xy_array()
                kp2d[invalid_mask[idx]] = -2.0
                kp2ds_aug.append(kp2d)
        else:
            kp2ds_aug = None
    if kp2ds is not None:
        return image_aug, kp2ds_aug
    else:
        return image_aug


def process_image(originImage, full_kp2ds=None, augments=None, is_pose2d=[True], random_crop=False, syn_occlusion=None):
    orgImage_white_bg, pad_trbl = image_pad_white_bg(originImage)
    if full_kp2ds is None and augments is None:
        return orgImage_white_bg, pad_trbl
    if syn_occlusion is not None:
        synthetic_occlusion, occluder, center = syn_occlusion
        if random.random() < 0.1:
            center = center + np.random.uniform([-16, -16], [16, 16])
        originImage, _, _ = synthetic_occlusion(originImage, occluder, center)
    crop_bbox = np.array([0, 0, originImage.shape[1], originImage.shape[0]])
    if augments is not None:
        rot, flip, crop_bbox, img_scale = augments
        if rot != 0:
            originImage, full_kp2ds = img_kp_rotate(originImage, full_kp2ds, rot)
        if flip:
            originImage = np.fliplr(originImage)
            full_kp2ds = [flip_kps(kps_i, width=originImage.shape[1], is_pose=is_2d_pose) for kps_i, is_2d_pose in zip(full_kp2ds, is_pose2d)]
    image_aug, kp2ds_aug, offsets = image_crop_pad(originImage, bbox=crop_bbox, kp2ds=full_kp2ds, pad_ratio=1.0)
    return image_aug, orgImage_white_bg, kp2ds_aug, offsets


def rot_imgplane(kp3d, angle):
    if angle == 0:
        return kp3d
    invalid_mask = kp3d[:, -1] == -2
    rot_mat = np.eye(3)
    rot_rad = angle * np.pi / 180
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    kp3d = np.einsum('ij,kj->ki', rot_mat, kp3d)
    kp3d[invalid_mask] = -2
    return kp3d


def angle_axis_to_rotation_matrix(angle_axis: 'torch.Tensor') ->torch.Tensor:
    """Convert 3d vector of axis-angle rotation to 3x3 rotation matrix
    Args:
        angle_axis (torch.Tensor): tensor of 3d vector of axis-angle rotations.
    Returns:
        torch.Tensor: tensor of 3x3 rotation matrices.
    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 3, 3)`
    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = angle_axis_to_rotation_matrix(input)  # Nx3x3
    """
    if not isinstance(angle_axis, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(angle_axis)))
    if not angle_axis.shape[-1] == 3:
        raise ValueError('Input size must be a (*, 3) tensor. Got {}'.format(angle_axis.shape))

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-06):
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)
    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)
    eps = 1e-06
    mask = (theta2 > eps).view(-1, 1, 1)
    mask_pos = mask.type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(3).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 3, 3).repeat(batch_size, 1, 1)
    rotation_matrix[..., :3, :3] = mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix


def camera_pitch_yaw_roll2rotation_matrix(pitch, yaw, roll=0):
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    Rx, _ = cv2.Rodrigues(x_axis * np.radians(pitch))
    Ry, _ = cv2.Rodrigues(y_axis * np.radians(yaw))
    Rz, _ = cv2.Rodrigues(z_axis * np.radians(roll))
    R = Rz @ Ry @ Rx
    return R


def convertRT2transform(R, T):
    transform4x4 = np.eye(4)
    transform4x4[:3, :3] = R
    transform4x4[:3, 3] = T
    return transform4x4


def inverse_transform(transform_mat):
    transform_inv = np.zeros_like(transform_mat)
    transform_inv[:3, :3] = np.transpose(transform_mat[:3, :3], (1, 0))
    transform_inv[:3, 3] = -np.matmul(transform_mat[:3, 3][None], transform_mat[:3, :3])
    transform_inv[3, 3] = 1.0
    return transform_inv


def transform_trans(transform_mat, trans):
    trans = np.concatenate((trans, np.ones_like(trans[[0]])), axis=-1)[None, :]
    trans_new = np.matmul(trans, np.transpose(transform_mat, (1, 0)))[0, :3]
    return trans_new


def convert_camera2world_RT(body_rots_cam, body_trans_cam, fov, pitch_yaw_roll):
    camera_R_mat = camera_pitch_yaw_roll2rotation_matrix(*pitch_yaw_roll)
    camera_T = np.zeros(3)
    world2camera = convertRT2transform(camera_R_mat, camera_T)
    camera2world = inverse_transform(world2camera)
    body_R_in_world = np.stack([np.matmul(camera2world[:3, :3], body_R_in_cam) for body_R_in_cam in body_rots_cam], 0)
    body_R_in_world = rotation_matrix_to_angle_axis(torch.from_numpy(body_R_in_world).float())
    body_T_in_world = np.stack([transform_trans(camera2world, body_T_in_cam) for body_T_in_cam in body_trans_cam], 0)
    return body_R_in_world, body_T_in_world, world2camera


def convert_camera2world_RT2(body_rots_cam, body_trans_cam, world2camera):
    camera2world = inverse_transform(world2camera)
    body_R_in_world = np.stack([np.matmul(camera2world[:3, :3], body_R_in_cam) for body_R_in_cam in body_rots_cam], 0)
    body_R_in_world = rotation_matrix_to_angle_axis(torch.from_numpy(body_R_in_world).float())
    body_T_in_world = np.stack([transform_trans(camera2world, body_T_in_cam) for body_T_in_cam in body_trans_cam], 0)
    return body_R_in_world, body_T_in_world


def normalize_trans_to_cam_params(trans):
    normed_cams = np.zeros_like(trans)
    normed_cams[..., 0] = 1 / (trans[..., 2] * tan_fov)
    normed_cams[..., 1] = trans[..., 1] / (trans[..., 2] * tan_fov)
    normed_cams[..., 2] = trans[..., 0] / (trans[..., 2] * tan_fov)
    _check_valid_cam(normed_cams)
    return normed_cams


def perspective_projection_withfovs(points, translation=None, rotation=None, keep_dim=False, fovs=None):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float()
    if isinstance(translation, np.ndarray):
        translation = torch.from_numpy(translation).float()
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = fovs
    K[:, 1, 1] = fovs
    K[:, 2, 2] = 1.0
    if rotation is not None:
        points = torch.einsum('bij,bkj->bki', rotation, points)
    if translation is not None:
        points = points + translation.unsqueeze(1)
    projected_points = points / (points[:, :, -1].unsqueeze(-1) + 0.0001)
    if torch.isnan(points).sum() > 0 or torch.isnan(projected_points).sum() > 0:
        None
    projected_points = torch.matmul(projected_points.contiguous(), K.contiguous())
    if not keep_dim:
        projected_points = projected_points[:, :, :-1].contiguous()
    return projected_points


def pack_data(vertex_save_dir, data_folder, split, annots_path):
    import pandas
    annots = {}
    smpl_subject_dict, subject_id = {}, 0
    os.makedirs(vertex_save_dir, exist_ok=True)
    all_annot_paths = glob.glob(os.path.join(data_folder, 'CAM2', '{}*_withj2.pkl'.format(split)))
    for af_ind, annot_file in enumerate(all_annot_paths):
        annot = pandas.read_pickle(annot_file)
        annot_dicts = annot.to_dict(orient='records')
        for annot_ind, annot_dict in enumerate(annot_dicts):
            None
            img_annot, img_verts, valid_num = [], [], 0
            pimg_annot = {'cam_locs': np.array([annot_dict['camX'], annot_dict['camY'], annot_dict['camZ'], annot_dict['camYaw']]), 'trans': np.array([annot_dict['X'], annot_dict['Y'], annot_dict['Z'], annot_dict['Yaw']]).transpose((1, 0)), 'props': np.array([annot_dict['gender'], annot_dict['kid'], annot_dict['occlusion'], annot_dict['age'], annot_dict['ethnicity']]), 'isValid': annot_dict['isValid'], 'gt_path_smpl': annot_dict['gt_path_smpl'], 'gt_path_smplx': annot_dict['gt_path_smplx']}
            for ind, smpl_annot_path in enumerate(annot_dict['gt_path_smpl']):
                if annot_dict['isValid'][ind]:
                    valid_num += 1
                subj_annot = {}
                smpl_annot = pandas.read_pickle(os.path.join(data_folder, smpl_annot_path.replace('.obj', '.pkl')))
                subj_annot['body_pose'] = smpl_annot['body_pose'].detach().cpu().numpy()
                subj_annot['betas'] = smpl_annot['betas'].detach().cpu().numpy()
                subj_annot['root_rot'] = smpl_annot['root_pose'].detach().cpu().numpy()
                subj_annot['props'] = [annot_dict['gender'][ind], 'kid' if annot_dict['kid'][ind] else 'adult', annot_dict['age'][ind], annot_dict['ethnicity'][ind]]
                if annot_dict['gt_path_smpl'][ind].replace('.obj', '') not in smpl_subject_dict:
                    smpl_subject_dict[annot_dict['gt_path_smpl'][ind].replace('.obj', '')] = subject_id
                    subject_id += 1
                subj_annot['ID'] = smpl_subject_dict[annot_dict['gt_path_smpl'][ind].replace('.obj', '')]
                subj_annot['occlusion'] = annot_dict['occlusion'][ind]
                subj_annot['isValid'] = annot_dict['isValid'][ind]
                subj_annot['kp2d'] = annot_dict['gt_joints_2d'][ind]
                subj_annot['kp3d'] = annot_dict['gt_joints_3d'][ind]
                subj_annot['cam_locs'] = pimg_annot['cam_locs']
                subj_annot['smpl_trans'] = pimg_annot['trans'][ind]
                subj_annot['camMats'] = annot_dict['camMats'][ind]
                subj_annot['root_rotMats'] = annot_dict['root_rotMats'][ind]
                img_annot.append(subj_annot)
            if valid_num != 0:
                annots[annot_dict['imgPath']] = img_annot
            vertex_save_name = os.path.join(vertex_save_dir, os.path.basename(annot_dict['imgPath']).replace('.png', '.npz'))
            np.savez(vertex_save_name, verts=img_verts)
        np.savez(self.annots_path.replace('.npz', '_{}.npz'.format(af_ind)), annots=annots)
    np.savez(annots_path, annots=annots)
    np.savez(os.path.join(self.data_folder, 'subject_IDs_dict_{}.npz'.format(self.split)), subject_ids=smpl_subject_dict)
    return annots


def line_intersect(sa, sb):
    al, ar, bl, br = sa[0], sa[1], sb[0], sb[1]
    assert al <= ar and bl <= br
    if al >= br or bl >= ar:
        return False
    return True


def rectangle_intersect(ra, rb):
    ax = [ra[0][0], ra[1][0]]
    ay = [ra[0][1], ra[1][1]]
    bx = [rb[0][0], rb[1][0]]
    by = [rb[0][1], rb[1][1]]
    return line_intersect(ax, bx) and line_intersect(ay, by)


def get_intersected_rectangle(lt0, rb0, lt1, rb1):
    if not rectangle_intersect([lt0, rb0], [lt1, rb1]):
        return None, None
    lt = lt0.copy()
    rb = rb0.copy()
    lt[0] = max(lt[0], lt1[0])
    lt[1] = max(lt[1], lt1[1])
    rb[0] = min(rb[0], rb1[0])
    rb[1] = min(rb[1], rb1[1])
    return lt, rb


def get_rectangle_area(lt, rb):
    return (rb[0] - lt[0]) * (rb[1] - lt[1])


def get_union_rectangle(lt0, rb0, lt1, rb1):
    lt = lt0.copy()
    rb = rb0.copy()
    lt[0] = min(lt[0], lt1[0])
    lt[1] = min(lt[1], lt1[1])
    rb[0] = max(rb[0], rb1[0])
    rb[1] = max(rb[1], rb1[1])
    return lt, rb


def get_rectangle_intersect_ratio(lt0, rb0, lt1, rb1):
    (lt0, rb0), (lt1, rb1) = get_intersected_rectangle(lt0, rb0, lt1, rb1), get_union_rectangle(lt0, rb0, lt1, rb1)
    if lt0 is None:
        return 0.0
    else:
        return 1.0 * get_rectangle_area(lt0, rb0) / get_rectangle_area(lt1, rb1)


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class COCO:

    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.anno_file = [annotation_file]
        if not annotation_file == None:
            None
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            None
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        None
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann
        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img
        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat
        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])
        None
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            None

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id'] in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        """
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]
        self.anno_file.append(resFile)
        None
        tic = time.time()
        if type(resFile) == str or type(resFile) == unicode:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == set(annsImgIds) & set(self.getImgIds()), 'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id + 1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1 - x0) * (y1 - y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]
        None
        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def download(self, tarDir=None, imgIds=[]):
        """
        Download COCO images from mscoco.org server.
        :param tarDir (str): COCO results directory name
               imgIds (list): images to be downloaded
        :return:
        """
        if tarDir is None:
            None
            return -1
        if len(imgIds) == 0:
            imgs = self.imgs.values()
        else:
            imgs = self.loadImgs(imgIds)
        N = len(imgs)
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        for i, img in enumerate(imgs):
            tic = time.time()
            fname = os.path.join(tarDir, img['file_name'])
            if not os.path.exists(fname):
                urlretrieve(img['coco_url'], fname)
            None

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        None
        assert type(data) == np.ndarray
        None
        assert data.shape[1] == 7
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                None
            ann += [{'image_id': int(data[i, 0]), 'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]], 'score': data[i, 5], 'category_id': int(data[i, 6])}]
        return ann

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        t = self.imgs[ann['image_id']]
        h, w = t['height'], t['width']
        segm = ann['segmentation']
        if type(segm) == list:
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann)
        m = maskUtils.decode(rle)
        return m


def load_func(fpath):
    None
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records


h36m_cameras_intrinsic_params = [{'id': '54138969', 'center': [512.54150390625, 515.4514770507812], 'focal_length': [1145.0494384765625, 1143.7811279296875], 'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043], 'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235], 'res_w': 1000, 'res_h': 1002, 'azimuth': 70}, {'id': '55011271', 'center': [508.8486328125, 508.0649108886719], 'focal_length': [1149.6756591796875, 1147.5916748046875], 'radial_distortion': [-0.1942136287689209, 0.2404085397720337, 0.006819975562393665], 'tangential_distortion': [-0.0016190266469493508, -0.0027408944442868233], 'res_w': 1000, 'res_h': 1000, 'azimuth': -70}, {'id': '58860488', 'center': [519.8158569335938, 501.40264892578125], 'focal_length': [1149.1407470703125, 1148.7989501953125], 'radial_distortion': [-0.2083381861448288, 0.25548800826072693, -0.0024604974314570427], 'tangential_distortion': [0.0014843869721516967, -0.0007599993259645998], 'res_w': 1000, 'res_h': 1000, 'azimuth': 110}, {'id': '60457274', 'center': [514.9682006835938, 501.88201904296875], 'focal_length': [1145.5113525390625, 1144.77392578125], 'radial_distortion': [-0.198384091258049, 0.21832367777824402, -0.008947807364165783], 'tangential_distortion': [-0.0005872055771760643, -0.0018133620033040643], 'res_w': 1000, 'res_h': 1002, 'azimuth': -110}]


def _check_visible(joints, w=2048, h=2048, get_mask=False):
    visibility = True
    x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
    y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
    ok_pts = np.logical_and(x_in, y_in)
    if np.sum(ok_pts) < 16:
        visibility = False
    if get_mask:
        return ok_pts
    return visibility


def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i * 7 + 5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i * 7 + 6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3, :3]
        T = RT[:3, 3] / 1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts


def calculate_score(output_path, annot_path, thr=250):
    with open(output_path, 'r') as f:
        output = json.load(f)

    def return_score(pred):
        return pred['score']
    output.sort(reverse=True, key=return_score)
    db = COCO(annot_path)
    gt_num = len([k for k, v in db.anns.items() if v['is_valid'] == 1])
    tp_acc = 0
    fp_acc = 0
    precision = []
    recall = []
    is_matched = {}
    for n in range(len(output)):
        image_id = output[n]['image_id']
        pred_root = output[n]['root_cam']
        score = output[n]['score']
        img = db.loadImgs(image_id)[0]
        ann_ids = db.getAnnIds(image_id)
        anns = db.loadAnns(ann_ids)
        valid_frame_num = len([item for item in anns if item['is_valid'] == 1])
        if valid_frame_num == 0:
            continue
        if str(image_id) not in is_matched:
            is_matched[str(image_id)] = [(0) for _ in range(len(anns))]
        min_dist = 9999
        save_ann_id = -1
        for ann_id, ann in enumerate(anns):
            if ann['is_valid'] == 0:
                continue
            gt_root = np.array(ann['keypoints_cam'])
            root_idx = 14
            gt_root = gt_root[root_idx]
            dist = math.sqrt(np.sum((pred_root - gt_root) ** 2))
            if min_dist > dist:
                min_dist = dist
                save_ann_id = ann_id
        is_tp = False
        if save_ann_id != -1 and min_dist < thr:
            if is_matched[str(image_id)][save_ann_id] == 0:
                is_tp = True
                is_matched[str(image_id)][save_ann_id] = 1
        if is_tp:
            tp_acc += 1
        else:
            fp_acc += 1
        precision.append(tp_acc / (tp_acc + fp_acc))
        recall.append(tp_acc / gt_num)
    AP = 0
    for n in range(len(precision) - 1):
        AP += precision[n + 1] * (recall[n + 1] - recall[n])
    None


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord


def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    kp_mask = np.copy(img)
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


PW3D_OCsubset = ['courtyard_backpack', 'courtyard_basketball', 'courtyard_bodyScannerMotions', 'courtyard_box', 'courtyard_golf', 'courtyard_jacket', 'courtyard_laceShoe', 'downtown_stairs', 'flat_guitar', 'flat_packBags', 'outdoors_climbing', 'outdoors_crosscountry', 'outdoors_fencing', 'outdoors_freestyle', 'outdoors_golf', 'outdoors_parcours', 'outdoors_slalom']


PW3D_PCsubset = {'courtyard_basketball_00': [200, 280], 'courtyard_captureSelfies_00': [500, 600], 'courtyard_dancing_00': [60, 370], 'courtyard_dancing_01': [60, 270], 'courtyard_hug_00': [100, 500], 'downtown_bus_00': [1620, 1900]}


Backbones = {'hrnet': HigherResolutionNet, 'resnet': ResNet_50}


def build_model(backbone, model_version, **kwargs):
    if backbone in Backbones:
        backbone = Backbones[backbone]()
    else:
        raise NotImplementedError('Backbone is not recognized')
    if model_version in Heads:
        ROMP = Heads[model_version]
    else:
        raise NotImplementedError('Head is not recognized')
    model = ROMP(backbone=backbone, **kwargs)
    return model


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, im2col_step=80):
        if input is not None and input.dim() != 4:
            raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(input, offset, weight)
        output = input.new_empty(DeformConvFunction._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride))
        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            deform_conv_cuda.deform_conv_forward_cuda(input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_cuda.deform_conv_backward_input_cuda(input, offset, grad_output, grad_input, grad_offset, weight, ctx.bufs_[0], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_cuda.deform_conv_backward_parameters_cuda(input, offset, grad_output, grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1, cur_im2col_step)
        return grad_input, grad_offset, grad_weight, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
        return output_size


deform_conv = DeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()
        assert not bias
        assert in_channels % groups == 0, 'in_channels {} cannot be divisible by groups {}'.format(in_channels, groups)
        assert out_channels % groups == 0, 'out_channels {} cannot be divisible by groups {}'.format(out_channels, groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class TempResultParser(nn.Module):

    def __init__(self, **kwargs):
        super(TempResultParser, self).__init__()
        self.map_size = args().centermap_size
        self.params_map_parser = SMPLWrapper()
        self.centermap_parser = CenterMap()

    def matching_forward(self, outputs, meta_data, cfg):
        outputs, meta_data = self.match_params_traj(outputs, meta_data, cfg)
        outputs = self.params_map_parser(outputs, meta_data)
        return outputs, meta_data

    @torch.no_grad()
    def parsing_forward(self, outputs, meta_data, cfg):
        outputs, meta_data = self.parse_maps(outputs, meta_data, cfg)
        outputs = self.params_map_parser(outputs, meta_data)
        return outputs, meta_data

    def match_params_traj(self, outputs, meta_data, cfg):
        org_traj_gt_ids = meta_data['traj_gt_ids'].long()
        pred_traj_gt_inds = outputs['traj_gt_inds']
        batch_ids = org_traj_gt_ids[pred_traj_gt_inds[:, 0], pred_traj_gt_inds[:, 1], pred_traj_gt_inds[:, 2], 0] * args().temp_clip_length + org_traj_gt_ids[pred_traj_gt_inds[:, 0], pred_traj_gt_inds[:, 1], pred_traj_gt_inds[:, 2], 1]
        person_ids = org_traj_gt_ids[pred_traj_gt_inds[:, 0], pred_traj_gt_inds[:, 1], pred_traj_gt_inds[:, 2], 2]
        batch_ids -= meta_data['batch_ids'][0]
        meta_data['traj3D_gts'] = meta_data['traj3D_gts'][pred_traj_gt_inds[:, 0], pred_traj_gt_inds[:, 1]]
        meta_data['traj2D_gts'] = meta_data['traj2D_gts'][pred_traj_gt_inds[:, 0], pred_traj_gt_inds[:, 1]]
        gt_keys = ['params', 'full_kp2d', 'kp_3d', 'valid_masks', 'subject_ids', 'verts', 'cam_mask', 'kid_shape_offsets', 'root_trans_cam', 'cams', 'world_global_rots']
        if args().learn_relative:
            gt_keys += ['depth_info']
        if args().learn_cam_with_fbboxes:
            gt_keys += ['full_body_bboxes']
        if args().dynamic_augment:
            gt_keys += ['dynamic_kp2ds', 'world_cams', 'world_cam_mask', 'world_root_trans']
        exclude_keys = ['heatmap', 'centermap', 'AE_joints', 'person_centers', 'fovs', 'seq_inds', 'params_pred', 'all_person_detected_mask', 'person_scales', 'dynamic_supervise']
        exclude_keys += ['traj3D_gts', 'traj2D_gts', 'Tj_flag', 'traj_gt_ids']
        exclude_keys += ['centermap_3d', 'valid_centermap3d_mask']
        outputs['pred_batch_ids'] += meta_data['batch_ids'][0]
        outputs, meta_data = reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids)
        outputs['center_preds'] = outputs['pred_czyxs'] * args().input_size / args().centermap_size
        outputs['center_confs'] = outputs['top_score']
        return outputs, meta_data

    def adjust_to_joint_level_sampling(self, param_preds, joint_sampler, param_maps, batch_ids, rot_dim=3):
        sampler_flat_inds = self.process_joint_sampler(joint_sampler)
        batch, channel = param_maps.shape[:2]
        param_maps = param_maps.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
        for inds, joint_inds in enumerate(constants.joint_sampler_relationship):
            start_inds = joint_inds * rot_dim + args().cam_dim
            end_inds = start_inds + rot_dim
            _check_params_sampling_(param_maps.shape, start_inds, end_inds, batch_ids, sampler_flat_inds[inds])
            param_preds[..., start_inds:end_inds] = param_maps[..., start_inds:end_inds][batch_ids, sampler_flat_inds[inds]].contiguous()
        return param_preds

    def process_joint_sampler(self, joint_sampler, thresh=0.999):
        joint_sampler = torch.clamp(joint_sampler, -1 * thresh, thresh)
        joint_sampler = (joint_sampler + 1) * self.map_size // 2
        xs, ys = joint_sampler[:, ::2], joint_sampler[:, 1::2]
        sampler_flat_inds = (ys * self.map_size + xs).permute((1, 0)).long().contiguous()
        return sampler_flat_inds

    def parameter_sampling(self, maps, batch_ids, flat_inds, use_transform=True):
        device = maps.device
        if use_transform:
            batch, channel = maps.shape[:2]
            maps = maps.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
        results = maps[batch_ids, flat_inds].contiguous()
        return results

    @torch.no_grad()
    def parse_maps(self, outputs, meta_data, cfg):
        if 'pred_batch_ids' in outputs:
            batch_ids = outputs['pred_batch_ids'].long()
            outputs['center_preds'] = outputs['pred_czyxs'] * args().input_size / args().centermap_size
            outputs['center_confs'] = outputs['top_score']
        else:
            batch_ids, flat_inds, cyxs, top_score = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['center_map'])
            if len(batch_ids) == 0:
                batch_ids, flat_inds, cyxs, top_score = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['center_map'], top_n_people=1)
                outputs['detection_flag'] = torch.Tensor([(False) for _ in range(len(batch_ids))])
        if 'params_pred' not in outputs and 'params_maps' in outputs:
            outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
        if 'center_preds' not in outputs:
            outputs['center_preds'] = torch.stack([flat_inds % args().centermap_size, flat_inds // args().centermap_size], 1) * args().input_size / args().centermap_size
            outputs['center_confs'] = self.parameter_sampling(outputs['center_map'], batch_ids, flat_inds, use_transform=True)
        if 'joint_sampler_maps_filtered' in outputs:
            outputs['joint_sampler'] = self.parameter_sampling(outputs['joint_sampler_maps_filtered'], batch_ids, flat_inds, use_transform=True)
            if 'params_pred' in outputs:
                _check_params_pred_(outputs['params_pred'].shape, len(batch_ids))
                self.adjust_to_joint_level_sampling(outputs['params_pred'], outputs['joint_sampler'], outputs['params_maps'], batch_ids)
        if 'reid_map' in outputs:
            outputs['reid_embeds'] = self.parameter_sampling(outputs['reid_map'], batch_ids, flat_inds, use_transform=True)
        if 'uncertainty_map' in outputs:
            outputs['uncertainty_pred'] = torch.sqrt(self.parameter_sampling(outputs['uncertainty_map'], batch_ids, flat_inds, use_transform=True) ** 2) + 1
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = ['image', 'offsets', 'imgpath', 'camMats']
        meta_data = reorganize_gts_cpu(meta_data, info_vis, batch_ids)
        if 'pred_batch_ids' in outputs:
            outputs['pred_batch_ids'] += meta_data['batch_ids'][0]
        return outputs, meta_data


class TemporalEncoder(nn.Module):

    def __init__(self, with_gru=False, input_size=128, out_size=[6], n_gru_layers=1, hidden_size=128):
        super(TemporalEncoder, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True))
        self.out_layers = nn.ModuleList([nn.Linear(hidden_size, size) for size in out_size])

    def forward(self, x, *args):
        n, t, f = x.shape
        y = self.regressor(x.reshape(-1, f))
        y = torch.cat([self.out_layers[ind](y) for ind in range(len(self.out_layers))], -1)
        return y.reshape(n, t, -1)

    def image_forward(self, x):
        y = self.regressor(x)
        y = torch.cat([self.out_layers[ind](y) for ind in range(len(self.out_layers))], -1)
        return y


def merge_item(source, target, key):
    if key not in target:
        target[key] = source[key].cpu()
    else:
        target[key] = torch.cat([target[key], source[key].cpu()], 0)


def merge_output(outs, meta_data, outputs):
    keys = ['meta_data', 'params_pred', 'reorganize_idx', 'j3d', 'verts', 'verts_camed_org', 'world_cams', 'world_trans', 'world_global_rots', 'world_verts', 'world_j3d', 'world_verts_camed_org', 'pj2d_org', 'pj2d', 'cam_trans', 'detection_flag', 'pj2d_org_h36m17', 'joints_h36m17', 'center_confs', 'track_ids', 'smpl_thetas', 'smpl_betas']
    for key in keys:
        if key == 'meta_data':
            for key1 in meta_data:
                merge_item(meta_data, outputs['meta_data'], key1)
        elif key in outs:
            merge_item(outs, outputs, key)


def convertback2batch(tensor):
    return tensor[::args().temp_clip_length].contiguous()


def reorganize_trajectory_info(meta_data):
    for item in ['traj3D_gts', 'traj2D_gts', 'Tj_flag', 'traj_gt_ids']:
        if item in meta_data:
            meta_data[item] = convertback2batch(meta_data[item])
    return meta_data


class VideoBase(Base):

    def forward(self, feat_inputs, meta_data=None, **cfg):
        if cfg['mode'] == 'matching_gts':
            meta_data = reorganize_trajectory_info(meta_data)
            return self.matching_forward(feat_inputs, meta_data, **cfg)
        elif cfg['mode'] == 'parsing':
            return self.parsing_forward(feat_inputs, meta_data, **cfg)
        elif cfg['mode'] == 'forward':
            return self.pure_forward(feat_inputs, meta_data, **cfg)
        else:
            raise NotImplementedError('forward mode is not recognized! please set proper mode (parsing/matching_gts)')

    def matching_forward(self, feat_inputs, meta_data, **cfg):
        if args().model_precision == 'fp16':
            with autocast():
                outputs = self.train_forward(feat_inputs, traj2D_gts=meta_data['traj2D_gts'])
                outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)
        else:
            outputs = self.train_forward(feat_inputs, traj2D_gts=meta_data['traj2D_gts'])
            outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)
        outputs['meta_data'] = meta_data
        if cfg['calc_loss']:
            outputs.update(self._calc_loss(outputs))
        return outputs

    @torch.no_grad()
    def parsing_forward(self, feat_inputs, meta_data, **cfg):
        if args().model_precision == 'fp16':
            with autocast():
                sequence_length = feat_inputs['image_feature_maps'].shape[0]
                outputs = {'meta_data': {}, 'params': {}}
                memory5D, hidden_state, tracker, init_world_cams, init_world_grots = None, None, None, None, None
                track_id_start = 0
                for iter_num in range(int(np.ceil(sequence_length / float(args().temp_clip_length_eval)))):
                    start, end = iter_num * args().temp_clip_length_eval, (iter_num + 1) * args().temp_clip_length_eval
                    seq_inds = feat_inputs['seq_inds'][start:end]
                    seq_inds[:, :3] = seq_inds[:, :3] - seq_inds[[0], :3]
                    split_outputs, hidden_state, memory5D, tracker, init_world_cams, init_world_grots = self.inference_forward({'image_feature_maps': feat_inputs['image_feature_maps'][start:end], 'seq_inds': seq_inds, 'optical_flows': feat_inputs['optical_flows'][start:end]}, hidden_state=hidden_state, memory5D=memory5D, temp_clip_length=args().temp_clip_length_eval, track_id_start=track_id_start, tracker=tracker, init_world_cams=init_world_cams, init_world_grots=init_world_grots, seq_cfgs=cfg['seq_cfgs'])
                    if split_outputs is None:
                        continue
                    split_meta_data = {k: v[start:end] for k, v in meta_data.items()}
                    split_outputs, split_meta_data = self._result_parser.parsing_forward(split_outputs, split_meta_data, cfg)
                    merge_output(split_outputs, split_meta_data, outputs)
                torch.cuda.empty_cache()
        else:
            outputs, hidden_state, tracker = self.inference_forward(feat_inputs)
            outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)
            outputs['meta_data'] = meta_data
        BaseTrack._count = 0
        return outputs

    @torch.no_grad()
    def pure_forward(self, feat_inputs, meta_data, **cfg):
        default_cfgs = {'tracker_det_thresh': args().tracker_det_thresh, 'tracker_match_thresh': args().tracker_match_thresh, 'first_frame_det_thresh': args().first_frame_det_thresh, 'accept_new_dets': args().accept_new_dets, 'new_subject_det_thresh': args().new_subject_det_thresh, 'time2forget': args().time2forget, 'large_object_thresh': args().large_object_thresh, 'suppress_duplicate_thresh': args().suppress_duplicate_thresh, 'motion_offset3D_norm_limit': args().motion_offset3D_norm_limit, 'feature_update_thresh': args().feature_update_thresh, 'feature_inherent': args().feature_inherent, 'occlusion_cam_inherent_or_interp': args().occlusion_cam_inherent_or_interp, 'tracking_target_max_num': args().tracking_target_max_num, 'axis_times': np.array([1.2, 2.5, 25]), 'smooth_pose_shape': args().smooth_pose_shape, 'pose_smooth_coef': args().pose_smooth_coef, 'smooth_pos_cam': False}
        if args().model_precision == 'fp16':
            with autocast():
                outputs, hidden_state, memory5D, tracker, init_world_cams = self.inference_forward({'image_feature_maps': feat_inputs['image_feature_maps'], 'seq_inds': feat_inputs['seq_inds'], 'optical_flows': feat_inputs['optical_flows']}, seq_cfgs=default_cfgs)
        else:
            outputs, hidden_state, tracker = self.feed_forward(feat_inputs)
        return outputs

    def head_forward(self, x):
        return NotImplementedError

    def make_backbone(self):
        return NotImplementedError

    def backbone_forward(self, x):
        return NotImplementedError

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)


def rotation_6d_to_matrix(d6: 'torch.Tensor') ->torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def angle_between(rot1: 'torch.Tensor', rot2: 'torch.Tensor'):
    """
    Calculate the angle in radians between two rotations. (torch, batch)
    :param rot1: Rotation tensor 1 that can reshape to [batch_size, rep_dim].
    :param rot2: Rotation tensor 2 that can reshape to [batch_size, rep_dim].
    :param rep: The rotation representation used in the input.
    :return: Tensor in shape [batch_size] for angles in radians.
    """
    rot_mat1 = rotation_6d_to_matrix(rot1[None])
    rot_mat2 = rotation_6d_to_matrix(rot2[None])
    offsets = rot_mat1.transpose(1, 2).bmm(rot_mat2)
    angles = rotation_matrix_to_angle_axis(offsets).norm(dim=1)
    return angles[0].item() * 180 / np.pi


def assert_detection_flag(center_maps, pred_batch_ids, pred_czyxs, top_scores, outmap_size):
    detection_flag = torch.Tensor([(False) for _ in range(len(center_maps))])
    if len(pred_czyxs) == 0:
        device = center_maps.device
        pred_batch_ids = torch.arange(1)
        pred_czyxs = torch.Tensor([[outmap_size // 4, outmap_size // 2, outmap_size // 2]]).long()
        top_scores = torch.ones(1)
    else:
        detection_flag[pred_batch_ids] = True
    return pred_batch_ids, pred_czyxs, top_scores, detection_flag


def convert_traj2D2center_yxs(traj2D_gts, outmap_size, seq_mask):
    """
    Flattening N valuable trajectories in 2D body centers, traj2D_gts,
        from shape (batch, 64, 8, 2) to (N, 8, 2).
    Inputs:
        traj2D_gts: torch.Tensor, shape [batch, person_num, clip, 2] the ground truth 2D body centers
    Return:
        batch_inds: torch.Tensor, shape [N]  
        traj2D_cyxs: torch.Tensor, shape [N, clip, 2]  
        top_scores: torch.Tensor, shape [N]  
        gt_inds: torch.Tensor, shape [N, 2]  the (batch, subject) Index in gts matrix.
    """
    batch_size, max_person_num, clip_length, dim = traj2D_gts.shape
    device = traj2D_gts.device
    batch_inds = []
    gt_inds = []
    traj2D_cyxs = []
    seq_masks = []
    for batch_id in range(batch_size):
        for person_id in range(max_person_num):
            valid_mask = traj2D_gts[batch_id, person_id][:, 0] != -2
            if valid_mask.sum() == 0:
                break
            if valid_mask.sum() != clip_length and seq_mask[batch_id * clip_length]:
                continue
            cyxs = traj2D_gts[batch_id, person_id].clone()
            cyxs[valid_mask] = (cyxs[valid_mask] + 1) / 2 * outmap_size
            if not seq_mask[batch_id * clip_length]:
                cyxs = cyxs[valid_mask]
                valid_clip_inds = torch.where(valid_mask)[0].cpu()
                batch_ind = torch.Tensor([(batch_id * clip_length + i) for i in valid_clip_inds]).long()
                valid_mask = valid_mask[valid_mask]
                seq_masks.append(torch.zeros(len(batch_ind)).bool())
            else:
                batch_ind = torch.Tensor([(batch_id * clip_length + i) for i in range(clip_length)]).long()
                seq_masks.append(torch.ones(len(batch_ind)).bool())
                valid_clip_inds = torch.arange(clip_length)
            batch_ind[~valid_mask] = -1
            traj2D_cyxs.append(cyxs)
            batch_inds.append(batch_ind)
            gt_inds.append(torch.stack([torch.ones(len(valid_clip_inds)) * batch_id, torch.ones(len(valid_clip_inds)) * person_id, valid_clip_inds], 1).long())
    traj2D_cyxs = torch.cat(traj2D_cyxs, 0).long()
    batch_inds = torch.cat(batch_inds, 0)
    gt_inds = torch.cat(gt_inds, 0)
    seq_masks = torch.cat(seq_masks, 0)
    top_scores = torch.ones(len(batch_inds))
    return batch_inds, traj2D_cyxs, top_scores, gt_inds, seq_masks


def get_3Dcoord_maps_zeroz(size, zsize=64):
    range_arr = torch.arange(size, dtype=torch.float32)
    Y_map = range_arr.reshape(1, 1, size, 1, 1).repeat(1, zsize, 1, size, 1) / size * 2 - 1
    X_map = range_arr.reshape(1, 1, 1, size, 1).repeat(1, zsize, size, 1, 1) / size * 2 - 1
    Z_map = torch.zeros_like(Y_map)
    out = torch.cat([Z_map, Y_map, X_map], dim=-1)
    return out


def infilling_cams_of_low_quality_dets(normed_cams, seq_trackIDs, memory5D, seq_inherent_flags, direct_inherent=False, smooth_cam=True, pose_smooth_coef=1.0):
    for ind, track_id in enumerate(seq_trackIDs):
        track_id = track_id.item()
        clip_cams = normed_cams[ind]
        infilling_clip_ids = torch.where(seq_inherent_flags[0][track_id])[0]
        good_clip_ids = torch.where(~seq_inherent_flags[0][track_id])[0]
        if smooth_cam:
            if 'cams' not in memory5D[0][track_id]:
                memory5D[0][track_id]['cams'] = OneEuroFilter(pose_smooth_coef, 0.7)
            if len(infilling_clip_ids) > 0:
                for clip_id in infilling_clip_ids:
                    fore_clips_ids = torch.where(~seq_inherent_flags[0][track_id][:clip_id])[0]
                    if len(fore_clips_ids) == 0:
                        if memory5D[0][track_id]['cams'].x_filter.prev_raw_value is not None:
                            normed_cams[ind, clip_id] = memory5D[0][track_id]['cams'].x_filter.prev_raw_value
                        continue
                    after_clips_ids = torch.where(~seq_inherent_flags[0][track_id][clip_id:])[0]
                    if len(after_clips_ids) == 0:
                        normed_cams[ind, clip_id] = clip_cams[good_clip_ids[-1]]
                        continue
                    valid_fore_ind = fore_clips_ids[-1]
                    valid_after_ind = after_clips_ids[0] + clip_id
                    normed_cams[ind, clip_id] = (valid_after_ind - clip_id) / (valid_after_ind - valid_fore_ind) * clip_cams[valid_fore_ind] + (clip_id - valid_fore_ind) / (valid_after_ind - valid_fore_ind) * clip_cams[valid_after_ind]
            for clip_id in range(len(clip_cams)):
                normed_cams[ind, clip_id] = memory5D[0][track_id]['cams'].process(clip_cams[clip_id])
        else:
            if 'cams' not in memory5D[0][track_id]:
                memory5D[0][track_id]['cams'] = clip_cams[good_clip_ids[0]] if len(good_clip_ids) > 0 else None
            if direct_inherent:
                for clip_id in range(normed_cams.shape[1]):
                    if seq_inherent_flags[0][track_id][clip_id] and memory5D[0][track_id]['cams'] is not None:
                        normed_cams[ind, clip_id] = memory5D[0][track_id]['cams']
                    elif not seq_inherent_flags[0][track_id][clip_id]:
                        memory5D[0][track_id]['cams'] = normed_cams[ind, clip_id]
            else:
                if len(infilling_clip_ids) > 0:
                    for clip_id in infilling_clip_ids:
                        fore_clips_ids = torch.where(~seq_inherent_flags[0][track_id][:clip_id])[0]
                        if len(fore_clips_ids) == 0:
                            if memory5D[0][track_id]['cams'] is not None:
                                normed_cams[ind, clip_id] = memory5D[0][track_id]['cams']
                            continue
                        after_clips_ids = torch.where(~seq_inherent_flags[0][track_id][clip_id:])[0]
                        if len(after_clips_ids) == 0:
                            normed_cams[ind, clip_id] = clip_cams[good_clip_ids[-1]]
                            continue
                        valid_fore_ind = fore_clips_ids[-1]
                        valid_after_ind = after_clips_ids[0] + clip_id
                        normed_cams[ind, clip_id] = (valid_after_ind - clip_id) / (valid_after_ind - valid_fore_ind) * clip_cams[valid_fore_ind] + (clip_id - valid_fore_ind) / (valid_after_ind - valid_fore_ind) * clip_cams[valid_after_ind]
                if len(good_clip_ids) > 0:
                    memory5D[0][track_id]['cams'] = clip_cams[good_clip_ids[-1]]
    return normed_cams, memory5D


def make_heatmaps_composite(heatmaps, image=None):
    if image is None:
        image = np.ones((512, 512, 3), dtype=np.uint8) * 255
    heatmaps = torch.nn.functional.interpolate(heatmaps[None].float(), size=image.shape[:2], mode='bilinear')[0]
    heatmaps = heatmaps.mul(255).clamp(0, 255).byte().detach().cpu().numpy()
    num_joints, height, width = heatmaps.shape
    image_grid = np.zeros((height, num_joints * width, 3), dtype=np.uint8)
    for j in range(num_joints):
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap * 0.7 + image * 0.3
        width_begin = width * j
        width_end = width * (j + 1)
        image_grid[:, width_begin:width_end, :] = image_fused
    return image_grid


def filter_out_small_objects(cams, thresh=0.1):
    large_object_mask = cams[:, 0] > thresh
    return large_object_mask


def parse_tracking_ids(seq_tracking_ids, pred_batch_ids, clip_length):
    all_track_results = torch.cat([v[0] for k, v in seq_tracking_ids.items()], 0)
    all_track_ids = torch.unique(all_track_results[:, 3].long())
    seq_tracking_results = {tid.item(): torch.zeros(clip_length, 3).long() for tid in all_track_ids}
    seq_tracking_batch_inds = {tid.item(): (torch.ones(clip_length).long() * -1) for tid in all_track_ids}
    seq_tracking_quality = {tid.item(): torch.zeros(clip_length, 2).float() for tid in all_track_ids}
    seq_masks = {tid.item(): (True) for tid in all_track_ids}
    valid_tracked_num = {tid.item(): (0) for tid in all_track_ids}
    frame_ids = sorted(seq_tracking_ids.keys())
    subject_traj_dict = {}
    for frame_id in frame_ids:
        trans_track_ids, batch_ids, org_trans, czyxs = seq_tracking_ids[frame_id]
        trans_tracked = trans_track_ids[:, :3]
        detected_track_ids = trans_track_ids[:, 3]
        tracked_det_conf = trans_track_ids[:, 4]
        tracked_flag = trans_track_ids[:, 5]
        tracked_czyx = trans_track_ids[:, 6:9]
        for track_id in seq_tracking_results:
            this_subject_mask = detected_track_ids == track_id
            if this_subject_mask.sum() == 0:
                continue
            seq_tracking_results[track_id][frame_id] = tracked_czyx[this_subject_mask]
            seq_tracking_quality[track_id][frame_id, 0] = tracked_det_conf[this_subject_mask]
            seq_tracking_quality[track_id][frame_id, 1] = tracked_flag[this_subject_mask]
            seq_tracking_batch_inds[track_id][frame_id] = batch_ids[0]
            valid_tracked_num[track_id] += 1
    for tid, valid_num in valid_tracked_num.items():
        if valid_num < min(len(frame_ids), 6):
            del seq_tracking_results[tid], seq_tracking_batch_inds[tid], seq_masks[tid], seq_tracking_quality[tid]
    return seq_tracking_results, seq_tracking_batch_inds, seq_masks, seq_tracking_quality


def suppress_duplicate_dets(cams, det_confs, thresh=0.1):
    sdd_mask = torch.ones(len(cams)).bool()
    if len(cams) > 1:
        del_index = []
        for ind in range(len(cams)):
            dists = torch.sort(torch.norm(cams - cams[[ind]], p=2, dim=-1), descending=False)
            dup_inds = dists.indices[dists.values < thresh][1:].tolist()
            if len(dup_inds) > 0:
                dup_inds = dup_inds + [ind]
                dup_inds_inds = torch.sort(det_confs[dup_inds], descending=True).indices[1:].tolist()
                det_ind_with_nonmaximum_conf = [dup_inds[i] for i in dup_inds_inds]
                sdd_mask[det_ind_with_nonmaximum_conf] = False
    return sdd_mask


def seach_clip_id(batch_id, seq_inds):
    batch_ids = seq_inds[:, 2]
    clip_id = seq_inds[torch.where(batch_ids == batch_id)[0], 1]
    return clip_id


def prepare_complete_trajectory_features(self, seq_tracking_results, mesh_feature_maps, seq_inds):
    seq_traj_features = {}
    seq_traj_czyxs = {}
    seq_traj_batch_inds = {}
    seq_traj_valid_masks = {}
    seq_masks = {}
    seq_track_ids = {}
    for seq_id, (center_traj3Ds, batch_inds, seq_flags) in seq_tracking_results.items():
        seq_traj_features[seq_id] = []
        seq_traj_czyxs[seq_id] = []
        seq_traj_batch_inds[seq_id] = []
        seq_traj_valid_masks[seq_id] = []
        seq_masks[seq_id] = []
        seq_track_ids[seq_id] = []
        for track_id in center_traj3Ds:
            subj_center3D_traj = center_traj3Ds[track_id]
            subj_batch_inds = batch_inds[track_id]
            seq_flag = seq_flags[track_id]
            valid_mask = torch.ones(len(subj_batch_inds)).bool()
            fore_subj_center3D_traj = subj_center3D_traj.clone().detach()
            fore_subj_center3D_traj_weight = torch.zeros(len(subj_center3D_traj)) + 0.5
            fore_subj_batch_inds = subj_batch_inds.clone().detach()
            after_subj_center3D_traj = subj_center3D_traj.clone().detach()
            after_subj_center3D_traj_weight = torch.zeros(len(subj_center3D_traj)) + 0.5
            after_subj_batch_inds = subj_batch_inds.clone().detach()
            for sbid in range(len(subj_batch_inds)):
                subj_batch_ind = subj_batch_inds[sbid]
                if sbid == 0 and subj_batch_ind == -1:
                    valid_ind = torch.where(subj_batch_inds != -1)[0]
                    if len(valid_ind) == 0:
                        None
                    valid_ind = 0 if len(valid_ind) == 0 else valid_ind[0]
                    fore_subj_center3D_traj[sbid] = fore_subj_center3D_traj[valid_ind]
                    after_subj_center3D_traj[sbid] = after_subj_center3D_traj[valid_ind]
                    fore_subj_batch_inds[sbid] = subj_batch_inds[valid_ind]
                    after_subj_batch_inds[sbid] = subj_batch_inds[valid_ind]
                    subj_batch_inds[0] = subj_batch_inds[valid_ind] - seach_clip_id(subj_batch_inds[valid_ind], seq_inds)
                    valid_mask[0] = False
                if subj_batch_ind == -1:
                    valid_fore_ind = torch.where(subj_batch_inds[:sbid] != -1)[0].max()
                    if len(torch.where(subj_batch_inds[sbid + 1:] != -1)[0]) > 0:
                        valid_after_ind = sbid + 1 + torch.where(subj_batch_inds[sbid + 1:] != -1)[0].min()
                        if valid_after_ind == valid_fore_ind:
                            None
                            None
                            None
                            None
                        fore_subj_center3D_traj_weight[sbid] = (valid_after_ind - sbid) / (valid_after_ind - valid_fore_ind)
                        after_subj_center3D_traj_weight[sbid] = (sbid - valid_fore_ind) / (valid_after_ind - valid_fore_ind)
                    else:
                        valid_after_ind = valid_fore_ind
                    fore_subj_center3D_traj[sbid] = fore_subj_center3D_traj[valid_fore_ind]
                    after_subj_center3D_traj[sbid] = after_subj_center3D_traj[valid_after_ind]
                    fore_subj_batch_inds[sbid] = subj_batch_inds[valid_fore_ind]
                    after_subj_batch_inds[sbid] = subj_batch_inds[valid_after_ind]
                    subj_batch_inds[sbid] = sbid + subj_batch_inds[valid_fore_ind] - seach_clip_id(subj_batch_inds[valid_fore_ind], seq_inds)
                    valid_mask[sbid] = False
            fore_subj_features = self.image_feature_sampling(mesh_feature_maps, fore_subj_center3D_traj, fore_subj_batch_inds)
            after_subj_features = self.image_feature_sampling(mesh_feature_maps, after_subj_center3D_traj, after_subj_batch_inds)
            subj_features = fore_subj_features * fore_subj_center3D_traj_weight.unsqueeze(1) + after_subj_center3D_traj_weight.unsqueeze(1) * after_subj_features
            subj_czyxs = fore_subj_center3D_traj * fore_subj_center3D_traj_weight.unsqueeze(1) + after_subj_center3D_traj * after_subj_center3D_traj_weight.unsqueeze(1)
            seq_traj_features[seq_id].append(subj_features)
            seq_traj_czyxs[seq_id].append(subj_czyxs.long())
            seq_traj_batch_inds[seq_id].append(subj_batch_inds)
            seq_traj_valid_masks[seq_id].append(valid_mask)
            seq_masks[seq_id].append(seq_flag)
            seq_track_ids[seq_id].append(track_id * torch.ones(len(subj_features)))
    traj_batch_inds = []
    traj_czyxs = []
    traj_features = []
    traj_masks = []
    traj_seq_masks = []
    sample_seq_masks = []
    traj_track_ids = []
    for seq_id, subj_features_list in seq_traj_features.items():
        traj_features.append(torch.stack(subj_features_list))
        traj_czyxs.append(torch.stack(seq_traj_czyxs[seq_id]))
        traj_batch_inds.append(torch.stack(seq_traj_batch_inds[seq_id]))
        traj_masks.append(torch.stack(seq_traj_valid_masks[seq_id]))
        traj_seq_masks = traj_seq_masks + seq_masks[seq_id]
        sample_seq_masks.append(torch.Tensor(seq_masks[seq_id]).sum() > 0)
        traj_track_ids.append(torch.stack(seq_track_ids[seq_id]).long())
    traj_seq_masks = torch.Tensor(traj_seq_masks).bool().reshape(-1)
    sample_seq_masks = torch.Tensor(sample_seq_masks).bool().reshape(-1)
    return traj_features, traj_czyxs, traj_batch_inds, traj_masks, traj_seq_masks, sample_seq_masks, traj_track_ids


def prepare_complete_trajectory_features_withmemory(self, seq_tracking_results, mesh_feature_maps, seq_inds, memory5D=None, det_conf_thresh=0.2, inherent_previous=True):
    if memory5D is None:
        memory5D = {seq_id: None for seq_id in seq_tracking_results}
    seq_traj_features = {}
    seq_traj_czyxs = {}
    seq_traj_batch_inds = {}
    seq_traj_valid_masks = {}
    seq_masks = {}
    seq_track_ids = {}
    seq_inherent_flags = {}
    for seq_id, (center_traj3Ds, batch_inds, seq_flags, track_quality) in seq_tracking_results.items():
        seq_traj_features[seq_id] = []
        seq_traj_czyxs[seq_id] = []
        seq_traj_batch_inds[seq_id] = []
        seq_traj_valid_masks[seq_id] = []
        seq_masks[seq_id] = []
        seq_track_ids[seq_id] = []
        seq_inherent_flags[seq_id] = {}
        if memory5D[seq_id] is None:
            memory5D[seq_id] = {track_id: {'feature': None, 'inherent_flag': {}} for track_id in center_traj3Ds}
        for track_id in center_traj3Ds:
            subj_center3D_traj = center_traj3Ds[track_id]
            subj_batch_inds = batch_inds[track_id]
            seq_flag = seq_flags[track_id]
            det_confs = track_quality[track_id][:, 0]
            tracked_states = track_quality[track_id][:, 1]
            valid_mask = torch.ones(len(subj_batch_inds)).bool()
            if track_id not in memory5D[seq_id]:
                memory5D[seq_id][track_id] = {'feature': None, 'inherent_flag': {}}
            for sbid in range(len(subj_batch_inds)):
                subj_batch_ind = subj_batch_inds[sbid]
                if subj_batch_ind == -1:
                    valid_mask[sbid] = False
            subj_features = self.image_feature_sampling(mesh_feature_maps, subj_center3D_traj, subj_batch_inds)
            inherent_flags = torch.ones(len(subj_features)).bool()
            if inherent_previous:
                for clip_id in range(len(subj_features)):
                    inherent_flag = True
                    if valid_mask[clip_id]:
                        if det_confs[clip_id] > det_conf_thresh and tracked_states[clip_id] > 0.99:
                            memory5D[seq_id][track_id]['feature'] = subj_features[clip_id]
                            inherent_flag = False
                        elif det_confs[clip_id] <= det_conf_thresh and memory5D[seq_id][track_id]['feature'] is not None:
                            subj_features[clip_id] = memory5D[seq_id][track_id]['feature']
                        elif tracked_states[clip_id] < 0.99 and memory5D[seq_id][track_id]['feature'] is not None:
                            subj_features[clip_id] = memory5D[seq_id][track_id]['feature']
                    memory5D[seq_id][track_id]['inherent_flag'][clip_id] = inherent_flag
                    inherent_flags[clip_id] = inherent_flag
            subj_czyxs = subj_center3D_traj
            seq_traj_features[seq_id].append(subj_features)
            seq_traj_czyxs[seq_id].append(subj_czyxs.long())
            seq_traj_batch_inds[seq_id].append(subj_batch_inds)
            seq_traj_valid_masks[seq_id].append(valid_mask)
            seq_masks[seq_id].append(seq_flag)
            seq_track_ids[seq_id].append(track_id * torch.ones(len(subj_features)))
            seq_inherent_flags[seq_id][track_id] = inherent_flags
    traj_batch_inds = []
    traj_czyxs = []
    traj_features = []
    traj_masks = []
    traj_seq_masks = []
    sample_seq_masks = []
    traj_track_ids = []
    for seq_id, subj_features_list in seq_traj_features.items():
        traj_features.append(torch.stack(subj_features_list))
        traj_czyxs.append(torch.stack(seq_traj_czyxs[seq_id]))
        traj_batch_inds.append(torch.stack(seq_traj_batch_inds[seq_id]))
        traj_masks.append(torch.stack(seq_traj_valid_masks[seq_id]))
        traj_seq_masks = traj_seq_masks + seq_masks[seq_id]
        sample_seq_masks.append(torch.Tensor(seq_masks[seq_id]).sum() > 0)
        traj_track_ids.append(torch.stack(seq_track_ids[seq_id]).long())
    traj_seq_masks = torch.Tensor(traj_seq_masks).bool().reshape(-1)
    sample_seq_masks = torch.Tensor(sample_seq_masks).bool().reshape(-1)
    return traj_features, traj_czyxs, traj_batch_inds, traj_masks, traj_seq_masks, sample_seq_masks, traj_track_ids, seq_inherent_flags, memory5D


def progressive_multiply_global_rotation(grots_offsets, cam_rots, clip_length, init_world_grots=None, accum_way='multiply'):
    grots_offsets = grots_offsets.reshape(-1, clip_length, 6)
    cam_grots = cam_rots.detach().reshape(-1, clip_length, 6)
    clip_num = len(grots_offsets)
    if accum_way == 'multiply':
        grots_offsets[..., [0, 4]] = grots_offsets[..., [0, 4]] + 1
        grots_offsets_mat = torch.stack([rotation_6d_to_matrix(grots_offsets[ind]) for ind in range(clip_num)], 0)
        cam_grots_mat = torch.stack([rotation_6d_to_matrix(cam_grots[ind]) for ind in range(clip_num)], 0)
        world_grots_offset_mat = [grots_offsets_mat[:, 0]]
        if init_world_grots is not None:
            world_grots_offset_mat[0] = torch.matmul(world_grots_offset_mat[0], init_world_grots)
        for ind in range(1, clip_length):
            world_grot_offset_mat = torch.matmul(grots_offsets_mat[:, ind], world_grots_offset_mat[-1])
            world_grots_offset_mat.append(world_grot_offset_mat)
        world_grots_offset_mat = torch.stack(world_grots_offset_mat, 1)
        world_grots_mat = torch.matmul(world_grots_offset_mat, cam_grots_mat)
        world_grots = torch.stack([rotation_matrix_to_angle_axis(world_grots_mat[ind]) for ind in range(clip_num)], 0).reshape(-1, 3)
        if init_world_grots is None:
            return world_grots, world_grots_offset_mat.reshape(-1, 3, 3), None
        else:
            return world_grots, world_grots_offset_mat.reshape(-1, 3, 3), world_grots_offset_mat[:, [-1]]
    elif accum_way == 'add':
        accum_offsets = torch.cumsum(cam_grots, -2)
        world_grots = cam_grots + accum_offsets
        if init_world_grots is not None:
            world_grots = world_grots + init_world_grots
        init_world_grots = accum_offsets[:, [-1]]
        world_grots = torch.stack([rot6D_to_angular(world_grots[ind]) for ind in range(clip_num)], 0).reshape(-1, 3)
        return world_grots, None, init_world_grots


def add_image2fig(image, suf_w=128, suf_h=128):
    eight_bit_img = Image.fromarray(image[:, ::-1]).convert('P', palette='WEB', dither=None)
    idx_to_color = np.array(eight_bit_img.getpalette()).reshape((-1, 3))
    colorscale = [[i / 255.0, 'rgb({}, {}, {})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]
    h, w = image.shape[:2]
    x = np.linspace(1, suf_w, w)
    y = np.linspace(1, suf_h, h)
    z = np.zeros((h, w))
    image_fig = go.Surface(x=x, y=y, z=z, surfacecolor=eight_bit_img, cmin=0, cmax=255, colorscale=colorscale, showscale=False, lighting_diffuse=1, lighting_ambient=1, lighting_fresnel=1, lighting_roughness=1, lighting_specular=0.5)
    return image_fig


def prepare_coord_map(d, h, w):
    d_map = np.zeros((d, h, w))
    for ind in range(d):
        d_map[ind] = d - 1 - ind
    h_map = np.zeros((d, h, w))
    for ind in range(h):
        h_map[:, ind] = ind
    w_map = np.zeros((d, h, w))
    for ind in range(w):
        w_map[:, :, ind] = w - 1 - ind
    return [w_map, h_map, d_map]


whd_map = prepare_coord_map(64, 128, 128)


def plot_3D_volume(volume):
    X, Y, Z = whd_map
    fig = go.Figure()
    volume_fig = go.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=volume.flatten(), opacity=0.3, opacityscale=[[-1, 0], [0, 0], [0.3, 1]], surface_count=10, colorscale='RdBu')
    return volume_fig


def update_centermap_layout(fig):
    fig.update_layout(title='CenterMap 3D', width=800, height=800, scene=dict(xaxis_visible=True, yaxis_visible=True, zaxis_visible=True, xaxis_title='W', yaxis_title='H', zaxis_title='D'))
    return fig


def show_plotly_figure(volume=None, image=None):
    fig = go.Figure()
    if volume is not None:
        volume_fig = plot_3D_volume(volume)
        fig.add_trace(volume_fig)
    if image is not None:
        image_fig = add_image2fig(image)
        fig.add_trace(image_fig)
    fig = update_centermap_layout(fig)
    fig.show()


def build_temporal_model(model_type='conv3D', head=1):
    model = THeads[head](model_type=model_type)
    return model


def check_input_data_quality(meta_data, min_num=2):
    return meta_data['all_person_detected_mask'].sum() > min_num


def fix_backbone(params, exclude_key=['backbone.']):
    for exclude_name in exclude_key:
        for index, (name, param) in enumerate(params.named_parameters()):
            if exclude_name in name:
                param.requires_grad = False
    logging.info('Fix params that include in {}'.format(exclude_key))
    return params


def flatten_clip_data(meta_data):
    seq_num, clip_length = meta_data['image'].shape[:2]
    key_names = list(meta_data.keys())
    for key in key_names:
        if isinstance(meta_data[key], torch.Tensor):
            shape_list = meta_data[key].shape
            if len(shape_list) > 2:
                meta_data[key] = meta_data[key].view(-1, *shape_list[2:])
            else:
                meta_data[key] = meta_data[key].view(-1)
        elif isinstance(meta_data[key], list):
            meta_data[key] = list(np.array(meta_data[key]).transpose(1, 0).reshape(-1))
    meta_data['seq_inds'] = torch.stack([torch.arange(seq_num).unsqueeze(1).repeat(1, clip_length).reshape(-1), torch.arange(clip_length).unsqueeze(0).repeat(seq_num, 1).reshape(-1), torch.arange(seq_num * clip_length), torch.ones(seq_num * clip_length)], 1).long()
    return meta_data


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


def load_model(path, model, prefix='module.', drop_prefix='', optimizer=None, **kwargs):
    logging.info('using fine_tune model: {}'.format(path))
    if os.path.exists(path):
        pretrained_model = torch.load(path)
        current_model = model.state_dict()
        if isinstance(pretrained_model, dict):
            if 'model_state_dict' in pretrained_model:
                pretrained_model = pretrained_model['model_state_dict']
        copy_state_dict(current_model, pretrained_model, prefix=prefix, drop_prefix=drop_prefix, **kwargs)
    else:
        logging.warning('model {} not exist!'.format(path))
    return model


def convert2BDsplit(tensor):
    """
    B batch size, N person number, T temp_clip_length, {} means might have this dimension for some input tensor but not all.
    Convert the input tensor from shape (B,...) to (BxT, ...), expanded [0,1,2,...] as [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, ...] if T=5.
    The shape of image feature maps is (BxT) x C x H x W
    In this way, we can use the balanced dataparallel to split data of shape (BxT) to multiple GPUs.
    """
    shape = list(tensor.shape)
    shape[0] = shape[0] * args().temp_clip_length
    repeat_times = [(1) for _ in range(len(shape) + 1)]
    repeat_times[1] = args().temp_clip_length
    return tensor.unsqueeze(1).repeat(*repeat_times).reshape(*shape)


def full_body_bboxes2person_centers(full_body_bboxes):
    valid_mask = (full_body_bboxes != -2).sum(-1) == 0
    person_centers = (full_body_bboxes[:, :, :2] + full_body_bboxes[:, :, 2:]) / 2
    person_centers[~valid_mask] = -2.0
    return person_centers


def ordered_organize_frame_outputs_to_clip(seq_inds, person_centers=None, track_ids=None, cam_params=None, cam_mask=None, full_body_bboxes=None):
    seq_sampling_index = seq_inds[:, 2].reshape(-1, args().temp_clip_length).numpy()
    if full_body_bboxes is not None:
        person_centers = full_body_bboxes2person_centers(full_body_bboxes)
    person_centers_inputs, track_ids_inputs, cam_params_inputs, cam_mask_inputs = [torch.stack([item[ids] for ids in seq_sampling_index], 0).contiguous() for item in [person_centers, track_ids, cam_params, cam_mask]]
    trajectory_info, track_ids_flatten = convert_centers_to_trajectory(person_centers_inputs, track_ids_inputs, cam_params_inputs, cam_mask_inputs, seq_inds=seq_inds)
    return trajectory_info, track_ids_flatten


def save_single_model(model, path):
    logging.info('saving {}'.format(path))
    torch.save(model.module.state_dict(), path)


def save_model(model, title, parent_folder=None):
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    if parent_folder is not None:
        title = os.path.join(parent_folder, title)
    save_single_model(model, title)


def write2log(log_file, massage):
    with open(log_file, 'a') as f:
        f.write(massage)


class Trainer(Base):

    def __init__(self):
        super(Trainer, self).__init__()
        self.determine_root_dir()
        self._load_image_model_()
        self._build_video_model_()
        self.flow_estimator = FlowExtract(torch.device('cuda:{}'.format(self.gpus[0] if len(self.gpus) == 1 else self.gpus[1])))
        self._build_optimizer_video_()
        self.mutli_task_uncertainty_weighted_loss = Learnable_Loss()
        self.loader = self._create_data_loader_video_(train_flag=True)
        self.video_train_cfg = {'mode': 'matching_gts', 'sequence_input': True, 'is_training': True, 'update_data': True, 'calc_loss': True if self.model_return_loss else False, 'input_type': 'sequence', 'with_nms': False, 'with_2d_matching': True, 'new_training': args().new_training, 'regress_params': True, 'traj_conf_threshold': 0.12}
        self.seq_cacher = {}
        logging.info('Initialization of Trainer finished!')

    def determine_root_dir(self):
        local_root_dir = '/home/yusun'
        remote_root_dir = '/home/sunyu15'
        self.show_tracking_results = False
        if os.path.isdir(local_root_dir):
            self.root_dir = local_root_dir
            self.tracking_results_save_dir = '/home/yusun/DataCenter/demo_results/tracking_results'
            self.dataset_dir = '/home/yusun/DataCenter/datasets'
            self.model_path = os.path.join('/home/yusun/Infinity/project_data/trace_data/trained_models', os.path.basename(self.model_path))
            self.temp_model_path = self.temp_model_path
            self.show_tracking_results = False
        elif os.path.isdir(remote_root_dir):
            self.root_dir = remote_root_dir
            self.tracking_results_save_dir = os.path.join(remote_root_dir, 'tracking_results')
            self.dataset_dir = '/home/sunyu15/datasets'
        else:
            raise NotImplementedError("both path : {} and {} don't exist".format(local_root_dir, remote_root_dir))

    def _load_image_model_(self):
        model = build_model(self.backbone, self.model_version, with_loss=False)
        drop_prefix = ''
        model = load_model(self.model_path, model, prefix='module.', drop_prefix=drop_prefix, fix_loaded=True)
        if not args().train_backbone:
            fix_backbone(model, exclude_key=['backbone.', 'head.'])
        if self.train_backbone:
            self.image_model = nn.DataParallel(model)
        else:
            self.image_model_device_id = self.gpus[0] if len(self.gpus) == 1 else self.gpus[1]
            torch.cuda.set_device(self.image_model_device_id)
            self.image_model_device = torch.device(f'cuda:{self.image_model_device_id}')
            self.local_device = torch.device(f'cuda:{self.image_model_device_id}')
            self.image_model = nn.DataParallel(model, device_ids=[self.image_model_device_id])
        if not args().train_backbone:
            self.image_model = self.image_model.eval()
        else:
            self.image_model = self.image_model.train()

    def _build_video_model_(self):
        logging.info('start building learnable video model.')
        temporal_model = build_temporal_model(model_type=args().tmodel_type, head=args().tmodel_version)
        if len(self.temp_model_path) > 0:
            prefix = 'module.' if 'TROMP_v2' in self.temp_model_path else ''
            temporal_model = load_model(self.temp_model_path, temporal_model, prefix=prefix, drop_prefix='', fix_loaded=False)
            if self.loading_bev_head_parameters:
                copy_state_dict(temporal_model.state_dict(), torch.load(self.model_path), prefix='module.')
        self.train_devices = self.gpus
        self.temp_model_device = torch.device(f'cuda:{self.train_devices[0]}')
        if self.master_batch_size != -1:
            self.temporal_model = DataParallel(temporal_model, device_ids=self.train_devices, chunk_sizes=self.chunk_sizes)
        else:
            self.temporal_model = nn.DataParallel(temporal_model, device_ids=self.train_devices)

    def _build_optimizer_video_(self):
        if not args().train_backbone:
            self.optimizer = torch.optim.Adam(list(self.image_model.parameters()) + list(self.temporal_model.parameters()), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.temporal_model.parameters(), lr=self.lr)
        if self.model_precision == 'fp16':
            self.scaler = GradScaler()
        self.e_sche = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 80], gamma=self.adjust_lr_factor)
        logging.info('finished build model.')

    def _create_single_video_sequence_data_loader(self, **kwargs):
        logging.info('gathering a single video datasets, loading a sequence at each time.')
        dataset = SingleVideoDataset(**kwargs)
        batch_size = self.batch_size if kwargs['train_flag'] else self.val_batch_size
        batch_sampler = SequentialBatchSampler('ordered', False, batch_size, dataset)
        data_loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, pin_memory=True, num_workers=self.nw)
        return data_loader

    def _create_data_loader_video_(self, train_flag=True):
        dataset_names = self.datasets.split(',')
        loading_modes = ['video_relative' for _ in range(len(dataset_names))]
        datasets = MixedDataset(dataset_names, self.sample_prob_dict, loading_modes=loading_modes, train_flag=train_flag)
        batch_size = self.batch_size
        None
        data_loader = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=True, drop_last=True if train_flag else False, pin_memory=True, num_workers=self.nw)
        return data_loader

    def reorganize_meta_data(self, meta_data, sampled_ids):
        new_meta_data = {}
        for key in meta_data:
            try:
                if isinstance(meta_data[key], torch.Tensor):
                    new_meta_data[key] = meta_data[key][sampled_ids]
                elif isinstance(meta_data[key], list):
                    new_meta_data[key] = [meta_data[key][ind] for ind in sampled_ids]
                else:
                    None
            except:
                None
            None
        return new_meta_data

    def reorganize_clip_data(self, meta_data, cfg_dict):
        """Each batch contains multiple video clips, this function reorganize them (0,1,2,3,4,5,6,7,8,9,10,11,12,13, ...)
        to each 7 small clips [(0,1,2,3,4,5,6), (7,8,9,10,11,12,13), ...] """
        trajectory_info, meta_data['subject_ids'] = ordered_organize_frame_outputs_to_clip(meta_data['seq_inds'], person_centers=meta_data['person_centers'], track_ids=meta_data['subject_ids'], cam_params=meta_data['cams'], cam_mask=meta_data['cam_mask'])
        meta_data.update(trajectory_info)
        return meta_data

    def network_forward(self, temporal_model, image_model, meta_data, cfg_dict):
        ds_org, imgpath_org = get_remove_keys(meta_data, keys=['data_set', 'imgpath'])
        with autocast():
            if self.train_backbone:
                image_inputs = {'image': meta_data['image']}
            else:
                image_inputs = {'image': meta_data['image']}
            image_outputs = image_model(image_inputs, **{'mode': 'extract_img_feature_maps'})
            meta_data = self.reorganize_clip_data(meta_data, cfg_dict)
            meta_data['batch_ids'] = meta_data['seq_inds'][:, 2]
            temp_inputs = {'image_feature_maps': image_outputs['image_feature_maps'], 'seq_inds': meta_data['seq_inds']}
            if not args().train_backbone:
                temp_inputs['image_feature_maps'] = temp_inputs['image_feature_maps'].detach()
            if args().use_optical_flow:
                optical_flows = self.flow_estimator(image_inputs['image'], meta_data['seq_inds'])
                temp_inputs['optical_flows'] = optical_flows
            outputs = temporal_model(temp_inputs, meta_data, **cfg_dict)
        meta_data.update({'imgpath': imgpath_org, 'data_set': ds_org})
        outputs['meta_data']['data_set'], outputs['meta_data']['imgpath'] = reorganize_items([ds_org, imgpath_org], outputs['reorganize_idx'].cpu().numpy())
        return outputs, image_outputs

    def train_step(self, meta_data):
        self.optimizer.zero_grad()
        outputs, BEV_outputs = self.network_forward(self.temporal_model, self.image_model, meta_data, self.video_train_cfg)
        if not self.model_return_loss:
            outputs.update(self._calc_loss(outputs))
        loss, outputs = self.mutli_task_uncertainty_weighted_loss(outputs, new_training=self.video_train_cfg['new_training'])
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return outputs, loss

    def remove_params_loss(self, outputs, keeped_loss_names=['CenterMap'], remove_loss_names=['MPJPE', 'PAMPJPE', 'P_KP2D', 'Pose', 'Shape', 'Cam']):
        outputs['loss_dict'] = {loss_name: outputs['loss_dict'][loss_name] for loss_name in keeped_loss_names if loss_name in outputs['loss_dict']}
        return outputs

    def train_log_visualization(self, outputs, loss, run_time, data_time, losses, losses_dict, epoch, iter_index):
        losses.update(loss.item())
        losses_dict.update(outputs['loss_dict'])
        if self.global_count % self.print_freq == 0:
            message = 'Epoch: [{0}][{1}/{2}] Time {data_time.avg:.2f} RUN {run_time.avg:.2f} Lr {lr} Loss {loss.avg:.2f} | Losses {3}'.format(epoch, iter_index + 1, len(self.loader), losses_dict.avg(), data_time=data_time, run_time=run_time, loss=losses, lr=self.optimizer.param_groups[0]['lr'])
            None
            write2log(self.log_file, '%s\n' % message)
            self.summary_writer.add_scalar('loss', losses.avg, self.global_count)
            self.summary_writer.add_scalars('loss_items', losses_dict.avg(), self.global_count)
            losses.reset()
            losses_dict.reset()
            data_time.reset()
            self.summary_writer.flush()
        if self.global_count % (4 * self.print_freq) == 0 or self.global_count == 1:
            save_name = '{}'.format(self.global_count)
            for ds_name in set(outputs['meta_data']['data_set']):
                save_name += '_{}'.format(ds_name)
            self.visualizer.visulize_video_result(outputs, outputs['meta_data'], show_items=['mesh', 'motion_offset', 'centermap'], vis_cfg={'settings': ['save_img'], 'save_dir': self.train_img_dir, 'save_name': save_name, 'error_names': ['E']})

    def train_epoch(self, epoch):
        run_time, data_time, losses = [AverageMeter() for i in range(3)]
        losses_dict = AverageMeter_Dict()
        batch_start_time = time.time()
        for iter_index, meta_data in enumerate(self.loader):
            self.global_count += 1
            if args().new_training and self.global_count == args().new_training_iters:
                self.video_train_cfg['new_training'], self.video_eval_cfg['new_training'] = False, False
            data_time.update(time.time() - batch_start_time)
            run_start_time = time.time()
            meta_data = flatten_clip_data(meta_data)
            if check_input_data_quality(meta_data):
                outputs, loss = self.train_step(meta_data)
                if self.local_rank in [-1, 0]:
                    run_time.update(time.time() - run_start_time)
                    self.train_log_visualization(outputs, loss, run_time, data_time, losses, losses_dict, epoch, iter_index)
            if self.global_count % self.test_interval == 0 or self.global_count == self.fast_eval_iter:
                title = '{}_val_{}'.format(self.tab, self.global_count)
                self.save_all_models(title)
            if self.distributed_training:
                torch.distributed.barrier()
            batch_start_time = time.time()
        title = '{}_epoch_{}'.format(self.tab, epoch)
        self.save_all_models(title)
        self.e_sche.step()
        self.loader.dataset.resampling_video_clips()

    def save_all_models(self, title, ext='.pkl', parent_folder=None):
        parent_folder = self.model_save_dir if parent_folder is None else parent_folder
        save_model(self.temporal_model, title + ext, parent_folder=parent_folder)
        if args().train_backbone:
            save_model(self.image_model, title + '_backbone' + ext, parent_folder=parent_folder)

    def train(self):
        init_seeds(self.local_rank, cuda_deterministic=False)
        logging.info('start training')
        self.temporal_model.train()
        for epoch in range(self.epoch):
            self.train_epoch(epoch)
        self.summary_writer.close()


def get_tracked_ids_byte(tracking_points, tracked_objects):
    tracked_ids_out = np.array([obj[4] for obj in tracked_objects])
    tracked_points = np.array([obj[:4] for obj in tracked_objects])
    tracked_ids, tracked_bbox_ids = [], []
    for tid, tracked_point in enumerate(tracked_points):
        org_p_id = np.argmin(np.array([np.linalg.norm(tracked_point - org_point) for org_point in tracking_points]))
        tracked_bbox_ids.append(org_p_id)
        tracked_ids.append(int(tracked_ids_out[tid]))
    return tracked_ids, tracked_bbox_ids


def process_idx(reorganize_idx, vids=None):
    result_size = reorganize_idx.shape[0]
    reorganize_idx = reorganize_idx.cpu().numpy()
    used_org_inds = np.unique(reorganize_idx)
    per_img_inds = [np.where(reorganize_idx == org_idx)[0] for org_idx in used_org_inds]
    return used_org_inds, per_img_inds


color_list = np.array([[0.7, 0.7, 0.6], [0.7, 0.5, 0.5], [0.5, 0.5, 0.7], [0.5, 0.55, 0.3], [0.3, 0.5, 0.55], [1, 0.855, 0.725], [0.588, 0.804, 0.804], [1, 0.757, 0.757], [0.933, 0.474, 0.258], [0.847, 191 / 255, 0.847], [0.941, 1, 1]])


def convert_front_view_to_bird_view_video(verts_t, bv_trans=None, h=512, w=512, focal_length=50):
    R_bv = torch.zeros(3, 3, device=verts_t.device)
    R_bv[0, 0] = R_bv[2, 1] = 1
    R_bv[1, 2] = -1
    verts_tr = torch.einsum('bij,kj->bik', verts_t, R_bv)
    verts_tfar = verts_tr
    p_min, p_max = verts_tfar.view(-1, 3).min(0)[0], verts_tfar.view(-1, 3).max(0)[0]
    p_center = 0.5 * (p_min + p_max)
    verts_center = (verts_tfar.view(-1, 3) - p_center).view(verts_t.shape[0], -1, 3)
    dis_min, dis_max = (verts_tfar.view(-1, 3) - p_center).min(0)[0], (verts_tfar.view(-1, 3) - p_center).max(0)[0]
    ratio_max = abs(0.9 - 0.5)
    z_x = dis_max[0] * focal_length / (ratio_max * w) + torch.abs(dis_min[2])
    z_y = dis_max[1] * focal_length / (ratio_max * h) + torch.abs(dis_min[2])
    z_x_0 = -dis_min[0] * focal_length / (ratio_max * w) + torch.abs(dis_min[2])
    z_y_0 = -dis_min[1] * focal_length / (ratio_max * h) + torch.abs(dis_min[2])
    z = max(z_x, z_y, z_x_0, z_y_0)
    if bv_trans is None:
        pass
    else:
        p_center, z = bv_trans
        p_center, z = p_center, z
    verts_right = verts_tfar - p_center + torch.tensor([0, 0, z], device=verts_t.device)
    return verts_right


INVALID_TRANS = np.ones(3) * -1


def rendering_mesh_to_image(self, outputs, seq_save_dirs):
    used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'])
    seq_save_dirs = [seq_save_dirs[ind] for ind in used_org_inds]
    mesh_colors = torch.Tensor([color_list[idx % len(color_list)] for idx in range(len(outputs['reorganize_idx']))])
    if args().model_version == 1:
        predicts_j3ds = outputs['j3d'].contiguous().detach().cpu().numpy()
        predicts_pj2ds = (outputs['pj2d'].detach().cpu().numpy() + 1) * 256
        predicts_j3ds = predicts_j3ds[:, :24]
        predicts_pj2ds = predicts_pj2ds[:, :24]
        outputs['cam_trans'] = estimate_translation(predicts_j3ds, predicts_pj2ds, focal_length=args().focal_length, img_size=np.array([512, 512]), pnp_algorithm='cv2')
    img_orgs = [outputs['meta_data']['image_1024'][img_id].cpu().numpy().astype(np.uint8) for img_id in range(len(per_img_inds))]
    img_verts = [outputs['verts'][inds] for inds in per_img_inds]
    img_trans = [outputs['cam_trans'][inds] for inds in per_img_inds]
    img_verts_bv = [convert_front_view_to_bird_view_video((iverts + itrans.unsqueeze(1)).detach(), None) for iverts, itrans in zip(img_verts, img_trans)]
    img_names = [np.array(outputs['meta_data']['imgpath'])[inds[0]] for inds in per_img_inds]
    for batch_idx, img_org in enumerate(img_orgs):
        try:
            img_org = img_org[:, :, ::-1]
            rendered_img = self.pyrender_render(img_org[None], [img_verts[batch_idx]], [img_trans[batch_idx]])[0]
            result_image_bv = self.pyrender_render_bv([np.ones_like(img_org) * 255], [img_verts_bv[batch_idx]], [torch.zeros_like(img_trans[batch_idx])])[0]
            render_fv = rendered_img.transpose((1, 2, 0))
            render_bv = result_image_bv.transpose((1, 2, 0))
            save_path = os.path.join(seq_save_dirs[batch_idx], os.path.basename(img_names[batch_idx]))
            img_results = np.concatenate([img_org, render_fv, np.ones((1024, 1024, 3)) * 255], 1)
            img_results[256:256 + 512, 1024 * 2:1024 * 2 + 512] = cv2.resize(render_bv, (512, 512))
            img_results = img_results[200:-200]
            img_results = img_results[:, :-512]
            cv2.imwrite(save_path, img_results)
        except Exception as error:
            None


def reorganize_results(outputs, img_paths, reorganize_idx):
    results = {}
    cam_results = outputs['cam_trans'].detach().cpu().numpy().astype(np.float16)
    pj2d_org_results = outputs['pj2d_org'].detach().cpu().numpy().astype(np.float16)
    center_confs = outputs['center_confs'].detach().cpu().numpy().astype(np.float16)
    vids_org = np.unique(reorganize_idx)
    for idx, vid in enumerate(vids_org):
        verts_vids = np.where(reorganize_idx == vid)[0]
        img_path = img_paths[verts_vids[0]]
        results[img_path] = [{} for idx in range(len(verts_vids))]
        for subject_idx, batch_idx in enumerate(verts_vids):
            results[img_path][subject_idx]['cam_trans'] = cam_results[batch_idx]
            results[img_path][subject_idx]['pj2d_org'] = pj2d_org_results[batch_idx]
            results[img_path][subject_idx]['center_conf'] = center_confs[batch_idx]
    return results


class CenterMap3D(object):

    def __init__(self, conf_thresh):
        None
        self.size = 128
        self.max_person = 64
        self.sigma = 1
        self.conf_thresh = conf_thresh
        self.gk_group, self.pool_group = self.generate_kernels([5])
        self.prepare_parsing()

    def prepare_parsing(self):
        self.coordmap_3d = get_3Dcoord_maps(size=self.size)
        self.maxpool3d = torch.nn.MaxPool3d(5, 1, (5 - 1) // 2)

    def generate_kernels(self, kernel_size_list):
        gk_group, pool_group = {}, {}
        for kernel_size in set(kernel_size_list):
            x = np.arange(0, kernel_size, 1, float)
            y = x[:, np.newaxis]
            x0, y0 = (kernel_size - 1) // 2, (kernel_size - 1) // 2
            gaussian_distribution = -((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)
            gk_group[kernel_size] = np.exp(gaussian_distribution)
            pool_group[kernel_size] = torch.nn.MaxPool2d(kernel_size, 1, (kernel_size - 1) // 2)
        return gk_group, pool_group

    def parse_3dcentermap(self, center_maps):
        center_map_nms = nms(center_maps, pool_func=self.maxpool3d).squeeze(1)
        b, c, h, w = center_map_nms.shape
        K = self.max_person
        topk_scores, topk_inds = torch.topk(center_map_nms.reshape(b, c, -1), K)
        topk_inds = topk_inds % (h * w)
        topk_ys = (topk_inds.long() // w).float()
        topk_xs = (topk_inds % w).int().float()
        topk_score, index = torch.topk(topk_scores.reshape(b, -1), K)
        topk_inds = gather_feature(topk_inds.view(b, -1, 1), index).reshape(b, K)
        topk_zs = index.long() // K
        topk_ys = gather_feature(topk_ys.reshape(b, -1, 1), index).reshape(b, K)
        topk_xs = gather_feature(topk_xs.reshape(b, -1, 1), index).reshape(b, K)
        mask = topk_score > self.conf_thresh
        batch_ids = torch.where(mask)[0]
        center_zyxs = torch.stack([topk_zs[mask].long(), topk_ys[mask].long(), topk_xs[mask].long()]).permute((1, 0)).long()
        return [batch_ids, center_zyxs, topk_score[mask]]


def get_cam3dmap_anchor(FOV, centermap_size):
    depth_level = np.array([1, 10, 20, 100], dtype=np.float32)
    map_coord_range_each_level = (np.array([2 / 64.0, 25 / 64.0, 3 / 64.0, 2 / 64.0], dtype=np.float32) * centermap_size).astype(np.int32)
    scale_level = 1 / np.tan(np.radians(FOV / 2.0)) / depth_level
    cam3dmap_anchor = []
    scale_cache = 8
    for scale, coord_range in zip(scale_level, map_coord_range_each_level):
        cam3dmap_anchor.append(scale_cache - np.arange(1, coord_range + 1) / coord_range * (scale_cache - scale))
        scale_cache = scale
    cam3dmap_anchor = np.concatenate(cam3dmap_anchor)
    return cam3dmap_anchor


class BEVv1(nn.Module):

    def __init__(self, **kwargs):
        super(BEVv1, self).__init__()
        None
        self.backbone = HigherResolutionNet()
        self._build_head()
        self._build_parser(conf_thresh=kwargs.get('center_thresh', 0.1))

    def _build_parser(self, conf_thresh=0.12):
        self.centermap_parser = CenterMap3D(conf_thresh=conf_thresh)

    def _build_head(self):
        params_num, cam_dim = 3 + 22 * 6 + 11, 3
        self.outmap_size = 128
        self.output_cfg = {'NUM_PARAMS_MAP': params_num - cam_dim, 'NUM_CENTER_MAP': 1, 'NUM_CAM_MAP': cam_dim}
        self.head_cfg = {'NUM_BASIC_BLOCKS': 1, 'NUM_CHANNELS': 128}
        self.bv_center_cfg = {'NUM_DEPTH_LEVEL': self.outmap_size // 2, 'NUM_BLOCK': 2}
        self.backbone_channels = self.backbone.backbone_channels
        self.transformer_cfg = {'INPUT_C': self.head_cfg['NUM_CHANNELS'], 'NUM_CHANNELS': 512}
        self._make_transformer()
        self.cam3dmap_anchor = torch.from_numpy(get_cam3dmap_anchor(60, self.outmap_size)).float()
        self.register_buffer('coordmap_3d', get_3Dcoord_maps_halfz(self.outmap_size, z_base=self.cam3dmap_anchor))
        self._make_final_layers(self.backbone_channels)

    def _make_transformer(self, drop_ratio=0.2):
        self.position_embeddings = nn.Embedding(self.outmap_size, self.transformer_cfg['INPUT_C'], padding_idx=0)
        self.transformer = nn.Sequential(nn.Linear(self.transformer_cfg['INPUT_C'], self.transformer_cfg['NUM_CHANNELS']), nn.ReLU(inplace=True), nn.Dropout(drop_ratio), nn.Linear(self.transformer_cfg['NUM_CHANNELS'], self.transformer_cfg['NUM_CHANNELS']), nn.ReLU(inplace=True), nn.Dropout(drop_ratio), nn.Linear(self.transformer_cfg['NUM_CHANNELS'], self.output_cfg['NUM_PARAMS_MAP']))

    def _make_final_layers(self, input_channels):
        self.det_head = self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP'] + self.output_cfg['NUM_CAM_MAP'])
        self.param_head = self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP'], with_outlayer=False)
        self._make_bv_center_layers(input_channels, self.bv_center_cfg['NUM_DEPTH_LEVEL'] * 2)
        self._make_3D_map_refiner()

    def _make_head_layers(self, input_channels, output_channels, num_channels=None, with_outlayer=True):
        head_layers = []
        if num_channels is None:
            num_channels = self.head_cfg['NUM_CHANNELS']
        for _ in range(self.head_cfg['NUM_BASIC_BLOCKS']):
            head_layers.append(nn.Sequential(BasicBlock(input_channels, num_channels, downsample=nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0))))
            input_channels = num_channels
        if with_outlayer:
            head_layers.append(nn.Conv2d(in_channels=num_channels, out_channels=output_channels, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*head_layers)

    def _make_bv_center_layers(self, input_channels, output_channels):
        num_channels = self.outmap_size // 8
        self.bv_pre_layers = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        input_channels = (num_channels + self.output_cfg['NUM_CENTER_MAP'] + self.output_cfg['NUM_CAM_MAP']) * self.outmap_size
        inter_channels = 512
        self.bv_out_layers = nn.Sequential(BasicBlock_1D(input_channels, inter_channels), BasicBlock_1D(inter_channels, inter_channels), BasicBlock_1D(inter_channels, output_channels))

    def _make_3D_map_refiner(self):
        self.center_map_refiner = nn.Sequential(BasicBlock_3D(self.output_cfg['NUM_CENTER_MAP'], self.output_cfg['NUM_CENTER_MAP']))
        self.cam_map_refiner = nn.Sequential(BasicBlock_3D(self.output_cfg['NUM_CAM_MAP'], self.output_cfg['NUM_CAM_MAP']))

    def fv_conditioned_bv_estimation(self, x, center_maps_fv, cam_maps_offset):
        img_feats = self.bv_pre_layers(x)
        summon_feats = torch.cat([center_maps_fv, cam_maps_offset, img_feats], 1).view(img_feats.size(0), -1, self.outmap_size)
        outputs_bv = self.bv_out_layers(summon_feats)
        center_maps_bv = outputs_bv[:, :self.bv_center_cfg['NUM_DEPTH_LEVEL']]
        cam_maps_offset_bv = outputs_bv[:, self.bv_center_cfg['NUM_DEPTH_LEVEL']:]
        center_map_3d = center_maps_fv.repeat(1, self.bv_center_cfg['NUM_DEPTH_LEVEL'], 1, 1) * center_maps_bv.unsqueeze(2).repeat(1, 1, self.outmap_size, 1)
        return center_map_3d, cam_maps_offset_bv

    def coarse2fine_localization(self, x):
        maps_fv = self.det_head(x)
        center_maps_fv = maps_fv[:, :self.output_cfg['NUM_CENTER_MAP']]
        cam_maps_offset = maps_fv[:, self.output_cfg['NUM_CENTER_MAP']:self.output_cfg['NUM_CENTER_MAP'] + self.output_cfg['NUM_CAM_MAP']]
        center_maps_3d, cam_maps_offset_bv = self.fv_conditioned_bv_estimation(x, center_maps_fv, cam_maps_offset)
        center_maps_3d = self.center_map_refiner(center_maps_3d.unsqueeze(1)).squeeze(1)
        cam_maps_3d = self.coordmap_3d + cam_maps_offset.unsqueeze(-1).transpose(4, 1).contiguous()
        cam_maps_3d[:, :, :, :, 2] = cam_maps_3d[:, :, :, :, 2] + cam_maps_offset_bv.unsqueeze(2).contiguous()
        cam_maps_3d = self.cam_map_refiner(cam_maps_3d.unsqueeze(1).transpose(5, 1).squeeze(-1))
        return center_maps_3d, cam_maps_3d, center_maps_fv

    def differentiable_person_feature_sampling(self, feature, pred_czyxs, pred_batch_ids):
        cz, cy, cx = pred_czyxs[:, 0], pred_czyxs[:, 1], pred_czyxs[:, 2]
        position_encoding = self.position_embeddings(cz)
        feature_sampled = feature[pred_batch_ids, :, cy, cx]
        input_features = feature_sampled + position_encoding
        return input_features

    def mesh_parameter_regression(self, fv_f, cams_preds, pred_batch_ids):
        cam_czyx = denormalize_center(convert_cam_params_to_centermap_coords(cams_preds.clone(), self.cam3dmap_anchor), size=self.outmap_size)
        feature_sampled = self.differentiable_person_feature_sampling(fv_f, cam_czyx, pred_batch_ids)
        params_preds = self.transformer(feature_sampled)
        params_preds = torch.cat([cams_preds, params_preds], 1)
        return params_preds, cam_czyx

    @torch.no_grad()
    def forward(self, x):
        x = self.backbone(x)
        center_maps_3d, cam_maps_3d, center_maps_fv = self.coarse2fine_localization(x)
        center_preds_info_3d = self.centermap_parser.parse_3dcentermap(center_maps_3d)
        if len(center_preds_info_3d[0]) == 0:
            None
            return None
        pred_batch_ids, pred_czyxs, center_confs = center_preds_info_3d
        cams_preds = cam_maps_3d[pred_batch_ids, :, pred_czyxs[:, 0], pred_czyxs[:, 1], pred_czyxs[:, 2]]
        front_view_features = self.param_head(x)
        params_preds, cam_czyx = self.mesh_parameter_regression(front_view_features, cams_preds, pred_batch_ids)
        output = {'params_pred': params_preds.float(), 'cam_czyx': cam_czyx.float(), 'center_map': center_maps_fv.float(), 'center_map_3d': center_maps_3d.float().squeeze(), 'pred_batch_ids': pred_batch_ids, 'pred_czyxs': pred_czyxs, 'center_confs': center_confs}
        return output


class ROMPv1(nn.Module):

    def __init__(self, **kwargs):
        super(ROMPv1, self).__init__()
        None
        self.backbone = HigherResolutionNet()
        self._build_head()

    def _build_head(self):
        self.outmap_size = 64
        params_num, cam_dim = 3 + 22 * 6 + 10, 3
        self.head_cfg = {'NUM_HEADS': 1, 'NUM_CHANNELS': 64, 'NUM_BASIC_BLOCKS': 2}
        self.output_cfg = {'NUM_PARAMS_MAP': params_num - cam_dim, 'NUM_CENTER_MAP': 1, 'NUM_CAM_MAP': cam_dim}
        self.final_layers = self._make_final_layers(self.backbone.backbone_channels)
        self.coordmaps = get_coord_maps(128)

    def _make_final_layers(self, input_channels):
        final_layers = [None]
        input_channels += 2
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP']))
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CAM_MAP']))
        return nn.ModuleList(final_layers)

    def _make_head_layers(self, input_channels, output_channels):
        head_layers = []
        num_channels = self.head_cfg['NUM_CHANNELS']
        head_layers.append(nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
        for i in range(self.head_cfg['NUM_HEADS']):
            layers = []
            for _ in range(self.head_cfg['NUM_BASIC_BLOCKS']):
                layers.append(nn.Sequential(BasicBlock(num_channels, num_channels)))
            head_layers.append(nn.Sequential(*layers))
        head_layers.append(nn.Conv2d(in_channels=num_channels, out_channels=output_channels, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*head_layers)

    @torch.no_grad()
    def forward(self, image):
        x = self.backbone(image)
        x = torch.cat((x, self.coordmaps.repeat(x.shape[0], 1, 1, 1)), 1)
        params_maps = self.final_layers[1](x)
        center_maps = self.final_layers[2](x)
        cam_maps = self.final_layers[3](x)
        params_maps = torch.cat([cam_maps, params_maps], 1)
        return center_maps, params_maps


class SMPL_parser(nn.Module):

    def __init__(self, model_path):
        super(SMPL_parser, self).__init__()
        self.smpl_model = SMPL(model_path)

    def forward(self, outputs, root_align=False):
        verts, joints, face = self.smpl_model(outputs['smpl_betas'], outputs['smpl_thetas'], root_align=root_align)
        outputs.update({'verts': verts, 'joints': joints, 'smpl_face': face})
        return outputs


keypoints_select = np.array([4, 5, 7, 8, 16, 17, 18, 19, 20, 21, 24, 35, 36, 12])


def collect_kp_results(outputs, img_paths):
    seq_kp3d_results = {}
    for ind, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path)
        if img_name not in seq_kp3d_results:
            seq_kp3d_results[img_name] = []
        subject_results = [outputs['pj2d_org'][ind].cpu().numpy(), outputs['j3d'][ind].cpu().numpy(), outputs['pj2d_org_h36m17'][ind].cpu().numpy(), outputs['joints_h36m17'][ind].cpu().numpy(), outputs['smpl_thetas'][ind].cpu().numpy(), outputs['smpl_betas'][ind].cpu().numpy(), outputs['cam_trans'][ind].cpu().numpy()]
        seq_kp3d_results[img_name].append(subject_results)
    return seq_kp3d_results


def pj2ds_to_bbox(pj2ds):
    tracked_bbox = np.array([pj2ds[:, 0].min(), pj2ds[:, 1].min(), pj2ds[:, 0].max(), pj2ds[:, 1].max()])
    tracked_bbox[2:] = tracked_bbox[2:] - tracked_bbox[:2]
    return tracked_bbox


def vis_track_bbox(image_path, tracked_ids, tracked_bbox):
    org_img = cv2.imread(image_path)
    for tid, bbox in zip(tracked_ids, tracked_bbox):
        org_img = cv2.rectangle(org_img, tuple(bbox[:2]), tuple(bbox[2:] + bbox[:2]), (255, 0, 0), 3)
        org_img = cv2.putText(org_img, '{}'.format(tid), tuple(bbox[:2]), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 2)
    h, w = org_img.shape[:2]
    cv2.imshow('bbox', cv2.resize(org_img, (w // 2, h // 2)))
    cv2.waitKey(10)


def collect_sequence_tracking_results(outputs, img_paths, reorganize_idx, visualize_results=False):
    track_ids = outputs['track_ids'].numpy()
    pj2d_org_results = outputs['pj2d_org'].detach().cpu().numpy().astype(np.float16)
    tracking_results = {}
    for frame_id, img_path in enumerate(img_paths):
        pred_ids = np.where(reorganize_idx == frame_id)[0]
        img_name = os.path.basename(img_path)
        tracking_results[img_name] = {'track_ids': [], 'track_bbox': [], 'pj2ds': []}
        for batch_id in pred_ids:
            track_id = track_ids[batch_id]
            pj2d_org = pj2d_org_results[batch_id]
            bbox = pj2ds_to_bbox(pj2d_org)
            tracking_results[img_name]['track_ids'].append(track_id)
            tracking_results[img_name]['track_bbox'].append(bbox)
            tracking_results[img_name]['pj2ds'].append(pj2d_org)
        if visualize_results:
            vis_track_bbox(img_path, tracking_results[img_name]['track_ids'], tracking_results[img_name]['track_bbox'])
    return tracking_results


def extract_seq_data(meta_data, seq_num=1):
    collect_items = ['image', 'data_set', 'imgpath', 'offsets']
    seq_data = {key: [] for key in collect_items}
    for key in seq_data:
        if isinstance(meta_data[key], torch.Tensor):
            seq_data[key].append(meta_data[key])
        elif isinstance(seq_data[key], list):
            seq_data[key] += [i[0] for i in meta_data[key]]
    seq_name = meta_data['seq_name'][0]
    for key in collect_items:
        if isinstance(seq_data[key][0], torch.Tensor):
            seq_data[key] = torch.cat(seq_data[key], 1).squeeze(0)
    return seq_data, seq_name


def get_seq_cfgs(args):
    default_cfgs = {'tracker_det_thresh': args.tracker_det_thresh, 'tracker_match_thresh': args.tracker_match_thresh, 'first_frame_det_thresh': args.first_frame_det_thresh, 'accept_new_dets': args.accept_new_dets, 'new_subject_det_thresh': args.new_subject_det_thresh, 'time2forget': args.time2forget, 'large_object_thresh': args.large_object_thresh, 'suppress_duplicate_thresh': args.suppress_duplicate_thresh, 'motion_offset3D_norm_limit': args.motion_offset3D_norm_limit, 'feature_update_thresh': args.feature_update_thresh, 'feature_inherent': args.feature_inherent, 'occlusion_cam_inherent_or_interp': args.occlusion_cam_inherent_or_interp, 'subject_num': args.subject_num, 'axis_times': np.array([1.2, 2.5, 25]), 'smooth_pose_shape': args.smooth_pose_shape, 'pose_smooth_coef': args.pose_smooth_coef, 'smooth_pos_cam': False}
    return default_cfgs


def insert_last_human_state(current, last_state, key, init=None):
    if key in last_state:
        return torch.cat([last_state[key], current], 0).contiguous()
    if key not in last_state:
        return torch.cat([current[[0]], current], 0).contiguous()


def load_config_dict(self, config_dict):
    hparams_dict = {}
    for i, j in config_dict.items():
        setattr(self, i, j)
        hparams_dict[i] = j
    return hparams_dict


class preds_save_paths(object):

    def __init__(self, results_save_dir, prefix='test'):
        self.seq_save_dir = os.path.join(results_save_dir, prefix)
        os.makedirs(self.seq_save_dir, exist_ok=True)
        self.tracking_matrix_save_path = os.path.join(self.seq_save_dir, 'TRACE_{}.txt'.format(prefix))
        self.seq_results_save_path = os.path.join(results_save_dir, prefix + '.npz')
        self.seq_tracking_results_save_path = os.path.join(results_save_dir, prefix + '_tracking.npz')


def prepare_data_loader(sequence_dict, val_batch_size):
    datasets = InternetVideo(sequence_dict)
    data_loader = DataLoader(dataset=datasets, shuffle=False, batch_size=val_batch_size, drop_last=False, pin_memory=True)
    return data_loader


delete_output_keys = ['params_pred', 'verts', 'verts_camed_org', 'world_verts', 'world_j3d', 'world_verts_camed_org', 'detection_flag']


def remove_large_keys(outputs, del_keys=delete_output_keys):
    save_outputs = copy.deepcopy(outputs)
    for key in del_keys:
        del save_outputs[key]
    rest_keys = list(save_outputs.keys())
    for key in rest_keys:
        if torch.is_tensor(save_outputs[key]):
            save_outputs[key] = save_outputs[key].detach().cpu().numpy()
    return save_outputs


def save_last_human_state(cacher, last_state, key):
    if key not in cacher:
        cacher = {}
    cacher[key] = last_state
    return cacher


def update_seq_cfgs(seq_name, default_cfgs):
    seq_cfgs = copy.deepcopy(default_cfgs)
    sequence_cfgs = {}
    if seq_name in sequence_cfgs:
        seq_cfgs.update(sequence_cfgs[seq_name])
    return seq_cfgs


def save_video(frame_save_paths, save_path, frame_rate=24):
    if len(frame_save_paths) == 0:
        return
    height, width = cv2.imread(frame_save_paths[0]).shape[:2]
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
    for frame_path in frame_save_paths:
        writer.write(cv2.imread(frame_path))
    writer.release()


def add_light(scene, light):
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)


class Py3DR(object):

    def __init__(self, FOV=60, height=512, width=512, focal_length=None):
        self.renderer = pyrender.OffscreenRenderer(height, width)
        if focal_length is None:
            self.focal_length = 1 / np.tan(np.radians(FOV / 2))
        else:
            self.focal_length = focal_length / max(height, width) * 2
        self.rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        self.colors = [(0.7, 0.7, 0.6, 1.0), (0.7, 0.5, 0.5, 1.0), (0.5, 0.5, 0.7, 1.0), (0.5, 0.55, 0.3, 1.0), (0.3, 0.5, 0.55, 1.0)]

    def __call__(self, vertices, triangles, image, mesh_colors=None, f=None, persp=True, camera_pose=None):
        img_height, img_width = image.shape[:2]
        self.renderer.viewport_height = img_height
        self.renderer.viewport_width = img_width
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        if camera_pose is None:
            camera_pose = np.eye(4)
        if persp:
            if f is None:
                f = self.focal_length * max(img_height, img_width) / 2
            camera = pyrender.camera.IntrinsicsCamera(fx=f, fy=f, cx=img_width / 2.0, cy=img_height / 2.0)
        else:
            xmag = ymag = np.abs(vertices[:, :, :2]).max() * 1.05
            camera = pyrender.camera.OrthographicCamera(xmag, ymag, znear=0.05, zfar=100.0, name=None)
        scene.add(camera, pose=camera_pose)
        if len(triangles.shape) == 2:
            triangles = [triangles for _ in range(len(vertices))]
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        for n in range(vertices.shape[0]):
            mesh = trimesh.Trimesh(vertices[n], triangles[n])
            mesh.apply_transform(self.rot)
            if mesh_colors is None:
                mesh_color = self.colors[n % len(self.colors)]
            else:
                mesh_color = mesh_colors[n % len(mesh_colors)]
            material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.2, alphaMode='OPAQUE', baseColorFactor=mesh_color)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh')
            add_light(scene, light)
        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32)
        valid_mask = (rend_depth > 0)[:, :, None]
        output_image = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image).astype(np.uint8)
        return output_image, rend_depth

    def delete(self):
        self.renderer.delete()


_norm = lambda arr: arr / np.sqrt(np.sum(arr ** 2, axis=1))[:, None]


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


def convert_type(obj):
    if isinstance(obj, tuple) or isinstance(obj, list):
        return np.array(obj, dtype=np.float32)[None, :]
    return obj


def get_normal(vertices, triangles):
    normal = np.zeros_like(vertices, dtype=np.float32)
    Sim3DR_Cython.get_normal(normal, vertices, triangles, vertices.shape[0], triangles.shape[0])
    return normal


def norm_vertices(vertices):
    vertices -= vertices.min(0)[None, :]
    vertices /= vertices.max()
    vertices *= 2
    vertices -= vertices.max(0)[None, :] / 2
    return vertices


def rasterize(vertices, triangles, colors, bg=None, height=None, width=None, channel=None, reverse=False):
    if bg is not None:
        height, width, channel = bg.shape
    else:
        assert height is not None and width is not None and channel is not None
        bg = np.zeros((height, width, channel), dtype=np.uint8)
    buffer = np.zeros((height, width), dtype=np.float32) - 100000000.0
    if colors.dtype != np.float32:
        colors = colors.astype(np.float32)
    Sim3DR_Cython.rasterize(bg, vertices, triangles, colors, buffer, triangles.shape[0], height, width, channel, reverse=reverse)
    return bg


class Sim3DR(object):

    def __init__(self, **kwargs):
        self.intensity_ambient = convert_type(kwargs.get('intensity_ambient', 0.66))
        self.intensity_directional = convert_type(kwargs.get('intensity_directional', 0.36))
        self.intensity_specular = convert_type(kwargs.get('intensity_specular', 0.1))
        self.specular_exp = kwargs.get('specular_exp', 1)
        self.color_directional = convert_type(kwargs.get('color_directional', (1, 1, 1)))
        self.light_pos = convert_type(kwargs.get('light_pos', (0, 0, -5)))
        self.view_pos = convert_type(kwargs.get('view_pos', (0, 0, 5)))

    def update_light_pos(self, light_pos):
        self.light_pos = convert_type(light_pos)

    def render(self, vertices, triangles, bg, color=np.array([[1, 0.6, 0.4]]), texture=None):
        normal = get_normal(vertices, triangles)
        light = np.zeros_like(vertices, dtype=np.float32)
        if self.intensity_ambient > 0:
            light += self.intensity_ambient * np.array(color)
        vertices_n = norm_vertices(vertices.copy())
        if self.intensity_directional > 0:
            direction = _norm(self.light_pos - vertices_n)
            cos = np.sum(normal * direction, axis=1)[:, None]
            light += self.intensity_directional * (self.color_directional * np.clip(cos, 0, 1))
            if self.intensity_specular > 0:
                v2v = _norm(self.view_pos - vertices_n)
                reflection = 2 * cos * normal - direction
                spe = np.sum((v2v * reflection) ** self.specular_exp, axis=1)[:, None]
                spe = np.where(cos != 0, np.clip(spe, 0, 1), np.zeros_like(spe))
                light += self.intensity_specular * self.color_directional * np.clip(spe, 0, 1)
        light = np.clip(light, 0, 1)
        if texture is None:
            render_img = rasterize(vertices, triangles, light, bg=bg)
            return render_img
        else:
            texture *= light
            render_img = rasterize(vertices, triangles, texture, bg=bg)
            return render_img

    def __call__(self, verts_list, triangles, bg, mesh_colors=np.array([[1, 0.6, 0.4]])):
        rendered_results = bg.copy()
        if len(triangles.shape) == 2:
            triangles = [triangles for _ in range(len(verts_list))]
        for ind, verts in enumerate(verts_list):
            verts = _to_ctype(verts)
            rendered_results = self.render(verts, triangles[ind], rendered_results, mesh_colors[[ind % len(mesh_colors)]])
        return rendered_results


def setup_renderer(name='sim3dr', **kwargs):
    if name == 'sim3dr':
        renderer = Sim3DR(**kwargs)
    elif name == 'pyrender':
        renderer = Py3DR(**kwargs)
    elif name == 'open3d':
        renderer = O3DDR(multi_mode=True, **kwargs)
    return renderer


def visulize_result(renderer, outputs, seq_data, rendering_cfgs, save_dir, alpha=1):
    used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'])


def visualize_predictions(outputs, imgpath, FOV, seq_save_dir, smpl_model_path):
    rendering_cfgs = {'mesh_color': 'identity', 'items': 'mesh,tracking', 'renderer': 'sim3dr'}
    renderer = setup_renderer(name=rendering_cfgs['renderer'], FOV=FOV)
    os.makedirs(seq_save_dir, exist_ok=True)
    render_images_path = visulize_result(renderer, outputs, imgpath, rendering_cfgs, seq_save_dir, smpl_model_path)
    save_video(render_images_path, seq_save_dir + '.mp4', frame_rate=25)


class TRACE(nn.Module):

    def __init__(self, args):
        super(TRACE, self).__init__()
        load_config_dict(self, vars(args))
        self.default_seq_cfgs = get_seq_cfgs(args)
        self.device = torch.device(f'cuda:{self.GPU}') if self.GPU > -1 else torch.device('cpu')
        None
        self.__load_models__()
        self.video_eval_cfg = {'mode': 'parsing', 'sequence_input': True, 'is_training': False, 'update_data': True, 'calc_loss': False, 'input_type': 'sequence', 'with_nms': True, 'with_2d_matching': True, 'new_training': False, 'regress_params': True, 'traj_conf_threshold': 0.12, 'temp_clip_length_eval': 8, 'xs': 2, 'ys': 2}
        self.continuous_state_cacher = {'image': {}, 'image_feats': {}, 'temp_state': {}}
        self.track_id_start = 0

    def __load_models__(self):
        image_backbone = HigherResolutionNet()
        image_backbone = load_model(self.image_backbone_model_path, image_backbone, prefix='module.backbone.', drop_prefix='', fix_loaded=True)
        self.image_backbone = nn.DataParallel(image_backbone).eval()
        self.motion_backbone = FlowExtract(self.raft_model_path, self.device)
        self._result_parser = TempResultParser(self.smpl_path, self.center_thresh)
        temporal_head_model = TRACE_head(self._result_parser, temp_clip_length=self.temp_clip_length, smpl_model_path=self.smpl_path)
        temporal_head_model = load_model(self.trace_head_model_path, temporal_head_model, prefix='', drop_prefix='', fix_loaded=False)
        self.temporal_head_model = nn.DataParallel(temporal_head_model)

    def temp_head_forward(self, feat_inputs, meta_data, seq_name, **cfg):
        temp_states = self.continuous_state_cacher['temp_state'][seq_name] if seq_name in self.continuous_state_cacher['temp_state'] else [None] * 5
        outputs, temp_states = self.temporal_head_model({'image_feature_maps': feat_inputs['image_feature_maps'], 'optical_flows': feat_inputs['optical_flows']}, temp_states=temp_states, temp_clip_length=cfg['temp_clip_length_eval'], track_id_start=self.track_id_start, seq_cfgs=cfg['seq_cfgs'], xs=cfg['xs'], ys=cfg['ys'])
        self.continuous_state_cacher['temp_state'][seq_name] = temp_states
        if outputs is not None:
            outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)
        return outputs

    @torch.no_grad()
    def sequence_inference(self, meta_data, seq_name, cfg_dict):
        input_images = meta_data['image']
        sequence_length = len(input_images)
        image_feature_maps = self.image_backbone(input_images)
        image_feature_maps = insert_last_human_state(image_feature_maps, self.continuous_state_cacher['image_feats'], seq_name)
        padded_input_images = insert_last_human_state(input_images, self.continuous_state_cacher['image'], seq_name)
        target_img_inds = torch.arange(1, len(padded_input_images))
        source_img_inds = target_img_inds - 1
        temp_inputs = {'image_feature_maps': image_feature_maps}
        temp_inputs['optical_flows'] = self.motion_backbone(padded_input_images, source_img_inds, target_img_inds)
        temp_meta_data = {'batch_ids': torch.arange(sequence_length), 'offsets': meta_data['offsets']}
        outputs = self.temp_head_forward(temp_inputs, temp_meta_data, seq_name, **cfg_dict)
        self.continuous_state_cacher['image'] = save_last_human_state(self.continuous_state_cacher['image'], input_images[[-1]], seq_name)
        self.continuous_state_cacher['image_feats'] = save_last_human_state(self.continuous_state_cacher['image_feats'], image_feature_maps[[-1]], seq_name)
        if outputs is None:
            return None, meta_data, None, None
        used_imgpath = reorganize_items([meta_data['imgpath']], outputs['reorganize_idx'].cpu().numpy())[0]
        tracking_results = collect_sequence_tracking_results(outputs, used_imgpath, outputs['reorganize_idx'].cpu().numpy(), show=self.show_tracking)
        kp3d_results = collect_kp_results(outputs, used_imgpath)
        return outputs, meta_data, tracking_results, kp3d_results

    def update_sequence_cfs(self, seq_name):
        return update_seq_cfgs(seq_name, self.default_seq_cfgs)

    @torch.no_grad()
    def forward(self, sequence_dict):
        """
            Please input one sequence per time
        """
        data_loader = prepare_data_loader(sequence_dict, self.val_batch_size)
        seq_outputs, tracking_results, kp3d_results, imgpaths = {}, {}, {}, {}
        start_frame_id = 0
        start_time = time.time()
        None
        for meta_data in data_loader:
            seq_data, seq_name = extract_seq_data(meta_data)
            start_frame_id += len(seq_data['image'])
            if seq_name not in imgpaths:
                imgpaths[seq_name] = []
            imgpaths[seq_name] += seq_data['imgpath']
            sfi = start_frame_id - len(seq_data['image'])
            self.video_eval_cfg['seq_cfgs'] = self.update_sequence_cfs(seq_name)
            outputs, meta_data, seq_tracking_results, seq_kp3d_results = self.sequence_inference(seq_data, seq_name, self.video_eval_cfg)
            if outputs is None:
                None
                continue
            outputs['reorganize_idx'] += sfi
            if seq_name not in seq_outputs:
                seq_outputs[seq_name], tracking_results[seq_name], kp3d_results[seq_name] = {}, {}, {}
            seq_outputs[seq_name] = merge_output(outputs, seq_outputs[seq_name])
            tracking_results[seq_name].update(seq_tracking_results)
            kp3d_results[seq_name].update(seq_kp3d_results)
        None
        return seq_outputs, tracking_results, kp3d_results, imgpaths

    def save_results(self, outputs, tracking_results, kp3d_results, imgpaths):
        for seq_name in outputs:
            save_paths = preds_save_paths(self.results_save_dir, prefix=seq_name)
            np.savez(save_paths.seq_results_save_path, outputs=remove_large_keys(outputs[seq_name]), imgpaths=imgpaths[seq_name])
            np.savez(save_paths.seq_tracking_results_save_path, tracking=tracking_results[seq_name], kp3ds=kp3d_results[seq_name])
            if self.save_video:
                visualize_predictions(outputs[seq_name], imgpaths[seq_name], self.FOV, save_paths.seq_save_dir, self.smpl_model_path)


class ConvGRUCell(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=128 + 128, kernel_size=3):
        super(ConvGRUCell, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class DeformConvPack(DeformConv):

    def __init__(self, *args, **kwargs):
        super(DeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(self.in_channels, self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        deform_conv_cuda.modulated_deform_conv_cuda_forward(input, weight, bias, ctx._bufs[0], offset, mask, output, ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        deform_conv_cuda.modulated_deform_conv_cuda_backward(input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None, None

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = ModulatedDeformConvFunction.apply


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset_mask = nn.Conv2d(self.in_channels, self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx, data, rois, offset, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        ctx.spatial_scale = spatial_scale
        ctx.out_size = out_size
        ctx.out_channels = out_channels
        ctx.no_trans = no_trans
        ctx.group_size = group_size
        ctx.part_size = out_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std
        assert 0.0 <= ctx.trans_std <= 1.0
        if not data.is_cuda:
            raise NotImplementedError
        n = rois.shape[0]
        output = data.new_empty(n, out_channels, out_size, out_size)
        output_count = data.new_empty(n, out_channels, out_size, out_size)
        deform_pool_cuda.deform_psroi_pooling_cuda_forward(data, rois, offset, output, output_count, ctx.no_trans, ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, rois, offset)
        ctx.output_count = output_count
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        data, rois, offset = ctx.saved_tensors
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_rois = None
        grad_offset = torch.zeros_like(offset)
        deform_pool_cuda.deform_psroi_pooling_cuda_backward(grad_output, data, rois, offset, output_count, grad_input, grad_offset, ctx.no_trans, ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        return grad_input, grad_rois, grad_offset, None, None, None, None, None, None, None, None


deform_roi_pooling = DeformRoIPoolingFunction.apply


class DeformRoIPooling(nn.Module):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.out_size = out_size
        self.out_channels = out_channels
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = out_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, data, rois, offset):
        if self.no_trans:
            offset = data.new_empty(0)
        return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)


class DeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, num_offset_fcs=3, deform_fc_channels=1024):
        super(DeformRoIPoolingPack, self).__init__(spatial_scale, out_size, out_channels, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.num_offset_fcs = num_offset_fcs
        self.deform_fc_channels = deform_fc_channels
        if not no_trans:
            seq = []
            ic = self.out_size * self.out_size * self.out_channels
            for i in range(self.num_offset_fcs):
                if i < self.num_offset_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size * self.out_size * 2
                seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_offset_fcs - 1:
                    seq.append(nn.ReLU(inplace=True))
            self.offset_fc = nn.Sequential(*seq)
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
        else:
            n = rois.shape[0]
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, True, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size, self.out_size)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)


class ModulatedDeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, num_offset_fcs=3, num_mask_fcs=2, deform_fc_channels=1024):
        super(ModulatedDeformRoIPoolingPack, self).__init__(spatial_scale, out_size, out_channels, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.num_offset_fcs = num_offset_fcs
        self.num_mask_fcs = num_mask_fcs
        self.deform_fc_channels = deform_fc_channels
        if not no_trans:
            offset_fc_seq = []
            ic = self.out_size * self.out_size * self.out_channels
            for i in range(self.num_offset_fcs):
                if i < self.num_offset_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size * self.out_size * 2
                offset_fc_seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_offset_fcs - 1:
                    offset_fc_seq.append(nn.ReLU(inplace=True))
            self.offset_fc = nn.Sequential(*offset_fc_seq)
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()
            mask_fc_seq = []
            ic = self.out_size * self.out_size * self.out_channels
            for i in range(self.num_mask_fcs):
                if i < self.num_mask_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size * self.out_size
                mask_fc_seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_mask_fcs - 1:
                    mask_fc_seq.append(nn.ReLU(inplace=True))
                else:
                    mask_fc_seq.append(nn.Sigmoid())
            self.mask_fc = nn.Sequential(*mask_fc_seq)
            self.mask_fc[-2].weight.data.zero_()
            self.mask_fc[-2].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
        else:
            n = rois.shape[0]
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, True, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size, self.out_size)
            mask = self.mask_fc(x.view(n, -1))
            mask = mask.view(n, 1, self.out_size, self.out_size)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std) * mask


class VonMisesUniformMix(VonMises):

    def __init__(self, loc, concentration, uniform_mix=0.25, **kwargs):
        super(VonMisesUniformMix, self).__init__(loc, concentration, **kwargs)
        self.uniform_mix = uniform_mix

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        assert len(sample_shape) == 1
        x = np.empty(tuple(self._extended_shape(sample_shape)), dtype=np.float32)
        uniform_samples = round(sample_shape[0] * self.uniform_mix)
        von_mises_samples = sample_shape[0] - uniform_samples
        x[:uniform_samples] = np.random.uniform(-math.pi, math.pi, size=tuple(self._extended_shape((uniform_samples,))))
        x[uniform_samples:] = np.random.vonmises(self.loc.cpu().numpy(), self.concentration.cpu().numpy(), size=tuple(self._extended_shape((von_mises_samples,))))
        return torch.from_numpy(x)

    def log_prob(self, value):
        von_mises_log_prob = super(VonMisesUniformMix, self).log_prob(value) + np.log(1 - self.uniform_mix)
        log_prob = torch.logaddexp(von_mises_log_prob, torch.full_like(von_mises_log_prob, math.log(self.uniform_mix / (2 * math.pi))))
        return log_prob


class EProPnP4DoF(EProPnPBase):
    """
    End-to-End Probabilistic Perspective-n-Points for 4DoF pose estimation.
    The pose is parameterized as [x, y, z, yaw], where yaw is the
    rotation around the Y-axis in radians.
    Adopted proposal distributions:
        position: multivariate t-distribution, degrees of freedom = 3
        orientation: 0.75 von Mises distribution + 0.25 uniform distribution
    """

    def allocate_buffer(self, num_obj, dtype=torch.float32, device=None):
        trans_mode = torch.empty((self.num_iter, num_obj, 3), dtype=dtype, device=device)
        trans_cov_tril = torch.empty((self.num_iter, num_obj, 3, 3), dtype=dtype, device=device)
        rot_mode = torch.empty((self.num_iter, num_obj, 1), dtype=dtype, device=device)
        rot_kappa = torch.empty((self.num_iter, num_obj, 1), dtype=dtype, device=device)
        return trans_mode, trans_cov_tril, rot_mode, rot_kappa

    def initial_fit(self, pose_opt, pose_cov, camera, trans_mode, trans_cov_tril, rot_mode, rot_kappa):
        trans_mode[0], rot_mode[0] = pose_opt.split([3, 1], dim=-1)
        trans_cov_tril[0] = cholesky_wrapper(pose_cov[:, :3, :3], [1.0, 1.0, 4.0])
        rot_kappa[0] = 0.33 / pose_cov[:, 3, 3, None].clamp(min=self.eps)

    @staticmethod
    def gen_new_distr(iter_id, trans_mode, trans_cov_tril, rot_mode, rot_kappa):
        new_trans_distr = MultivariateStudentT(3, trans_mode[iter_id], trans_cov_tril[iter_id])
        new_rot_distr = VonMisesUniformMix(rot_mode[iter_id], rot_kappa[iter_id])
        return new_trans_distr, new_rot_distr

    @staticmethod
    def gen_old_distr(iter_id, trans_mode, trans_cov_tril, rot_mode, rot_kappa):
        mix_trans_distr = MultivariateStudentT(3, trans_mode[:iter_id, None], trans_cov_tril[:iter_id, None])
        mix_rot_distr = VonMisesUniformMix(rot_mode[:iter_id, None], rot_kappa[:iter_id, None])
        return mix_trans_distr, mix_rot_distr

    def estimate_params(self, iter_id, pose_samples, pose_sample_logweights, trans_mode, trans_cov_tril, rot_mode, rot_kappa):
        sample_weights_norm = torch.softmax(pose_sample_logweights, dim=0)
        trans_mode[iter_id + 1] = (sample_weights_norm[..., None] * pose_samples[..., :3]).sum(dim=0)
        trans_dev = pose_samples[..., :3] - trans_mode[iter_id + 1]
        trans_cov = (sample_weights_norm[..., None, None] * trans_dev.unsqueeze(-1) * trans_dev.unsqueeze(-2)).sum(dim=0)
        trans_cov_tril[iter_id + 1] = cholesky_wrapper(trans_cov, [1.0, 1.0, 4.0])
        mean_vector = pose_samples.new_empty((pose_samples.size(1), 2))
        torch.sum(sample_weights_norm[..., None] * pose_samples[..., 3:].sin(), dim=0, out=mean_vector[:, :1])
        torch.sum(sample_weights_norm[..., None] * pose_samples[..., 3:].cos(), dim=0, out=mean_vector[:, 1:])
        rot_mode[iter_id + 1] = torch.atan2(mean_vector[:, :1], mean_vector[:, 1:])
        r_sq = torch.square(mean_vector).sum(dim=-1, keepdim=True)
        rot_kappa[iter_id + 1] = 0.33 * r_sq.sqrt().clamp(min=self.eps) * (2 - r_sq) / (1 - r_sq).clamp(min=self.eps)


class Centermap3dLoss(nn.Module):

    def __init__(self):
        pass


class Aggregate(nn.Module):

    def __init__(self, args, dim, heads=4, dim_head=128):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape
        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        if self.project is not None:
            out = self.project(out)
        out = fmap + self.gamma * out
        return out


class GMAUpdateBlock(nn.Module):

    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.use_setrans = args.use_setrans
        if self.use_setrans:
            self.intra_trans_config = args.intra_trans_config
            self.aggregator = ExpandedFeatTrans(self.intra_trans_config, 'Motion Aggregator')
        else:
            self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=self.args.num_heads)

    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        if self.use_setrans:
            B, C, H, W = motion_features.shape
            motion_features_3d = motion_features.view(B, C, H * W).permute(0, 2, 1)
            motion_features_global_3d = self.aggregator(motion_features_3d, attention)
            motion_features_global = motion_features_global_3d.view(B, H, W, C).permute(0, 3, 1, 2)
        else:
            motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)
        net = self.gru(net, inp_cat)
        delta_flow = self.flow_head(net)
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class PreNormResidual(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x, start=0):
        x = x + self.pe[:, start:start + x.size(1)]
        return x


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        """
        Args:
            - x: [batch_size,seq_len,dim]
            - mask: [batch_size,seq_len] - dytpe= torch.bool - default True everywhere, if False it means that we don't pay attention to this timestep
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, n, 1)
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class FeedForwardResidual(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0, out_dim=24 * 6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + out_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, x, init, n_iter=1):
        pred_pose = init
        for _ in range(n_iter):
            xf = torch.cat([x, init], -1)
            pred_pose = pred_pose + self.net(xf)
        return pred_pose


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class TransformerRegressor(nn.Module):

    def __init__(self, dim, depth=2, heads=8, dim_head=32, mlp_dim=32, dropout=0.1, out=[22 * 6, 3], share_regressor=False, with_norm=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            if with_norm:
                list_modules = [PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]
            else:
                list_modules = [Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout), FeedForward(dim, mlp_dim, dropout=dropout)]
            if i == 0 or not share_regressor:
                for out_i in out:
                    if with_norm:
                        list_modules.append(PreNorm(dim, FeedForwardResidual(dim, mlp_dim, dropout=dropout, out_dim=out_i)))
                    else:
                        list_modules.append(FeedForwardResidual(dim, mlp_dim, dropout=dropout, out_dim=out_i))
            else:
                for j in range(2, len(self.layers[0])):
                    list_modules.append(self.layers[0][j])
            self.layers.append(nn.ModuleList(list_modules))

    def forward(self, x, init, mask=None):
        batch_size, seq_len, *_ = x.size()
        y = init
        for layers_i in self.layers:
            attn, ff = layers_i[0], layers_i[1]
            x = attn(x, mask=mask) + x
            x = ff(x) + x
            for j, reg in enumerate(layers_i[2:]):
                y[j] = reg(x, init=y[j], n_iter=1)
        return y


class TemporalPoseRegressor(nn.Module):

    def __init__(self, in_dim=128, n_jts_out=22, init_pose=None, jt_dim=6, dim=512, depth=2, heads=8, dim_head=64, mlp_dim=512, dropout=0.1, share_regressor=1, with_norm=True, *args, **kwargs):
        super(TemporalPoseRegressor, self).__init__()
        self.pos = PositionalEncoding(dim, 1024)
        self.emb = nn.Linear(in_dim, dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim))
        self.decoder = TransformerRegressor(dim, depth, heads, dim_head, mlp_dim, dropout, [n_jts_out * jt_dim], share_regressor == 1, with_norm=with_norm)
        if init_pose is None:
            init_pose = torch.zeros(n_jts_out * jt_dim).float()
        self.register_buffer('init_pose', init_pose.reshape(1, 1, -1))

    def forward(self, x, mask=None):
        """
        Args:
            - x: torch.Tensor - torch.float32 - [batch_size, seq_len, 128]
            - mask: torch.Tensor - torch.bool - [batch_size, seq_len]
        Return:
            - y: torch.Tensor - [batch_size, seq_len, 24*6] - torch.float32
        """
        batch_size, seq_len, feature_ch = x.size()
        if mask is None:
            mask = torch.ones(batch_size, seq_len).type_as(x).bool()
        x = self.emb(x)
        x = x * mask.float().unsqueeze(-1) + self.mask_token * (1.0 - mask.float().unsqueeze(-1))
        x = self.pos(x)
        init = [self.init_pose.repeat(batch_size, seq_len, 1)]
        y = self.decoder(x, init, mask)[0]
        return y


class TemporalSMPLShapeRegressor(nn.Module):

    def __init__(self, in_dim=128, out_dim=21, init_shape=None, dim=256, depth=1, heads=8, dim_head=64, mlp_dim=512, dropout=0.1, share_regressor=1, *args, **kwargs):
        super(TemporalSMPLShapeRegressor, self).__init__()
        self.pos = PositionalEncoding(dim, 1024)
        self.emb = nn.Linear(in_dim, dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim))
        self.decoder = TransformerRegressor(dim, depth, heads, dim_head, mlp_dim, dropout, [out_dim], share_regressor == 1)
        if init_shape is None:
            init_shape = torch.zeros(out_dim).float()
        self.register_buffer('init_shape', init_shape.reshape(1, 1, -1))

    def forward(self, x, mask=None):
        """
        Args:
            - x: torch.Tensor - torch.float32 - [batch_size, seq_len, 128]
            - mask: torch.Tensor - torch.bool - [batch_size, seq_len]
        Return:
            - y: torch.Tensor - [batch_size, seq_len, 24*6] - torch.float32
        """
        batch_size, seq_len, feature_ch = x.size()
        if mask is None:
            mask = torch.ones(batch_size, seq_len).type_as(x).bool()
        x = self.emb(x)
        x = x * mask.float().unsqueeze(-1) + self.mask_token * (1.0 - mask.float().unsqueeze(-1))
        x = self.pos(x)
        init = [self.init_shape.repeat(batch_size, seq_len, 1)]
        y = self.decoder(x, init, mask)[0]
        return y


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class DataParallelModel(DataParallel):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.DataParallelCriterion`.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """

    def gather(self, outputs, output_device):
        return outputs

    def replicate(self, module, device_ids):
        modules = super(DataParallelModel, self).replicate(module, device_ids)
        return modules


class Reduce(Function):

    @staticmethod
    def forward(ctx, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs)

    @staticmethod
    def backward(ctx, gradOutput):
        return Broadcast.apply(ctx.target_gpus, gradOutput)


def _criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.device(device):
                output = module(*(input + target), **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e
    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input, target, kwargs, device)) for i, (module, input, target, kwargs, device) in enumerate(zip(modules, inputs, targets, kwargs_tup, devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class DataParallelCriterion(DataParallel):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage for
    Semantic Segmentation.

    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """

    def forward(self, inputs, *targets, **kwargs):
        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)
        return Reduce.apply(*outputs) / len(outputs)


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 3D facial landmarks
    """

    def __init__(self, config):
        super(FLAME, self).__init__()
        None
        with open(config.flame_model_path, 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))
        self.NECK_IDX = 1
        self.batch_size = config.batch_size
        self.dtype = torch.float32
        self.use_face_contour = config.use_face_contour
        self.faces = self.flame_model.f
        self.register_buffer('faces_tensor', to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long))
        default_shape = torch.zeros([self.batch_size, 300 - config.shape_params], dtype=self.dtype, requires_grad=False)
        self.register_parameter('shape_betas', nn.Parameter(default_shape, requires_grad=False))
        default_exp = torch.zeros([self.batch_size, 100 - config.expression_params], dtype=self.dtype, requires_grad=False)
        self.register_parameter('expression_betas', nn.Parameter(default_exp, requires_grad=False))
        default_eyball_pose = torch.zeros([self.batch_size, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose, requires_grad=False))
        default_neck_pose = torch.zeros([self.batch_size, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose, requires_grad=False))
        self.use_3D_translation = config.use_3D_translation
        default_transl = torch.zeros([self.batch_size, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('transl', nn.Parameter(default_transl, requires_grad=False))
        self.register_buffer('v_template', to_tensor(to_np(self.flame_model.v_template), dtype=self.dtype))
        shapedirs = self.flame_model.shapedirs
        self.register_buffer('shapedirs', to_tensor(to_np(shapedirs), dtype=self.dtype))
        j_regressor = to_tensor(to_np(self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))
        with open(config.static_landmark_embedding_path, 'rb') as f:
            static_embeddings = Struct(**pickle.load(f, encoding='latin1'))
        lmk_faces_idx = static_embeddings.lmk_face_idx.astype(np.int64)
        self.register_buffer('lmk_faces_idx', torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = static_embeddings.lmk_b_coords
        self.register_buffer('lmk_bary_coords', torch.tensor(lmk_bary_coords, dtype=self.dtype))
        if self.use_face_contour:
            conture_embeddings = np.load(config.dynamic_landmark_embedding_path, allow_pickle=True, encoding='latin1')
            conture_embeddings = conture_embeddings[()]
            dynamic_lmk_faces_idx = np.array(conture_embeddings['lmk_face_idx']).astype(np.int64)
            dynamic_lmk_faces_idx = torch.tensor(dynamic_lmk_faces_idx, dtype=torch.long)
            self.register_buffer('dynamic_lmk_faces_idx', dynamic_lmk_faces_idx)
            dynamic_lmk_bary_coords = conture_embeddings['lmk_b_coords']
            dynamic_lmk_bary_coords = torch.tensor(dynamic_lmk_bary_coords, dtype=self.dtype)
            self.register_buffer('dynamic_lmk_bary_coords', dynamic_lmk_bary_coords)
            neck_kin_chain = []
            curr_idx = torch.tensor(self.NECK_IDX, dtype=torch.long)
            while curr_idx != -1:
                neck_kin_chain.append(curr_idx)
                curr_idx = self.parents[curr_idx]
            self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

    def _find_dynamic_lmk_idx_and_bcoords(self, vertices, pose, dynamic_lmk_faces_idx, dynamic_lmk_b_coords, neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
            Source: Modified for batches from https://github.com/vchoutas/smplx
        """
        batch_size = vertices.shape[0]
        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)
        rel_rot_mat = torch.eye(3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)
        y_rot_angle = torch.round(torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39))
        neg_mask = y_rot_angle.lt(0)
        mask = y_rot_angle.lt(-39)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle
        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def forward(self, shape_params=None, expression_params=None, pose_params=None, neck_pose=None, eye_pose=None, transl=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters
            return:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        betas = torch.cat([shape_params, self.shape_betas, expression_params, self.expression_betas], dim=1)
        neck_pose = neck_pose if neck_pose is not None else self.neck_pose
        eye_pose = eye_pose if eye_pose is not None else self.eye_pose
        transl = transl if transl is not None else self.transl
        full_pose = torch.cat([pose_params[:, :3], neck_pose, pose_params[:, 3:], eye_pose], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(self.batch_size, 1, 1)
        vertices, _ = lbs(betas, full_pose, template_vertices, self.shapedirs, self.posedirs, self.J_regressor, self.parents, self.lbs_weights, dtype=self.dtype)
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).repeat(self.batch_size, 1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        if self.use_face_contour:
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(vertices, full_pose, self.dynamic_lmk_faces_idx, self.dynamic_lmk_bary_coords, self.neck_kin_chain, dtype=self.dtype)
            lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)
        landmarks = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)
        if self.use_3D_translation:
            landmarks += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
        return vertices, landmarks


def _axis_angle_rotation(axis: 'str', angle: 'torch.Tensor') ->torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == 'X':
        R_flat = one, zero, zero, zero, cos, -sin, zero, sin, cos
    elif axis == 'Y':
        R_flat = cos, zero, sin, zero, one, zero, -sin, zero, cos
    elif axis == 'Z':
        R_flat = cos, -sin, zero, sin, cos, zero, zero, zero, one
    else:
        raise ValueError('letter must be either X, Y or Z.')
    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention=('X', 'Y', 'Z')):
    matrices = [_axis_angle_rotation(c, e) for c, e in zip(convention, torch.unbind(euler_angles, -1))]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


class CamPose_IR(nn.Module):

    def __init__(self, world_kp3d, pj2d, cam_K, init_pitch_tx, device=torch.device('cuda:0')):
        super().__init__()
        self.device = device
        self.register_buffer('world_kp3d', world_kp3d.float())
        self.register_buffer('pj2d', pj2d.float())
        self.register_buffer('cam_K', torch.from_numpy(cam_K).float())
        self.camera_pitch_tx = nn.Parameter(init_pitch_tx)

    def forward(self):
        camera_euler_angles = torch.cat([self.camera_pitch_tx[:2], torch.zeros(1)], 0)
        cam_rot_mat = euler_angles_to_matrix(camera_euler_angles)
        points = torch.einsum('ij,kj->ki', cam_rot_mat, self.world_kp3d)
        points[:, 0] = points[:, 0] + self.camera_pitch_tx[2]
        projected_points = points / points[:, -1].unsqueeze(-1)
        projected_points = torch.matmul(self.cam_K[:3, :3], projected_points.contiguous().T).T
        projected_points = projected_points[..., :2]
        loss = torch.norm(projected_points - self.pj2d, p=2, dim=-1).mean()
        return loss, projected_points


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AddCoords,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock_1D,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (BasicBlock_3D,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (BasicBlock_IBN_a,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (BasicMotionEncoder,
     lambda: ([], {'corr_levels': 4, 'corr_radius': 4}),
     lambda: ([torch.rand([4, 2, 64, 64]), torch.rand([4, 324, 64, 64])], {})),
    (CoordConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 2, 4, 4])], {})),
    (DataParallelCriterion,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DataParallelModel,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FlowHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {})),
    (IBN_a,
     lambda: ([], {'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (JointsMSELoss,
     lambda: ([], {'use_target_weight': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (L2Prior,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PreNormResidual,
     lambda: ([], {'dim': 4, 'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Residual,
     lambda: ([], {'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SmallEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SmallMotionEncoder,
     lambda: ([], {'corr_levels': 4, 'corr_radius': 4}),
     lambda: ([torch.rand([4, 2, 64, 64]), torch.rand([4, 324, 64, 64])], {})),
    (_DataParallel,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
]

