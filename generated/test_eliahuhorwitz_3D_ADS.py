
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


from torchvision import transforms


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import numpy as np


import torch


from sklearn import random_projection


from sklearn.metrics import roc_auc_score


from typing import Tuple


import torch.nn as nn


import torch.nn.functional as F


import random


class KNNGaussianBlur(torch.nn.Module):

    def __init__(self, radius: 'int'=4):
        super().__init__()
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(self.unload(img[0] / map_max).filter(self.blur_kernel)) * map_max
        return final_map


class Model(torch.nn.Module):

    def __init__(self, device, backbone_name='wide_resnet50_2', out_indices=(2, 3), checkpoint_path='', pool_last=False):
        super().__init__()
        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})
        self.backbone = timm.create_model(model_name=backbone_name, pretrained=True, checkpoint_path=checkpoint_path, **kwargs)
        self.device = device
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1)) if pool_last else None

    def forward(self, x):
        x = x
        features = self.backbone(x)
        if self.avg_pool:
            fmap = features[-1]
            fmap = self.avg_pool(fmap)
            fmap = torch.flatten(fmap, 1)
            features.append(fmap)
        return features

    def freeze_parameters(self, layers, freeze_bn=False):
        """ Freeze resent parameters. The layers which are not indicated in the layers list are freeze. """
        layers = [str(layer) for layer in layers]
        if '1' not in layers:
            if hasattr(self.backbone, 'conv1'):
                for p in self.backbone.conv1.parameters():
                    p.requires_grad = False
            if hasattr(self.backbone, 'bn1'):
                for p in self.backbone.bn1.parameters():
                    p.requires_grad = False
            if hasattr(self.backbone, 'layer1'):
                for p in self.backbone.layer1.parameters():
                    p.requires_grad = False
        if '2' not in layers:
            if hasattr(self.backbone, 'layer2'):
                for p in self.backbone.layer2.parameters():
                    p.requires_grad = False
        if '3' not in layers:
            if hasattr(self.backbone, 'layer3'):
                for p in self.backbone.layer3.parameters():
                    p.requires_grad = False
        if '4' not in layers:
            if hasattr(self.backbone, 'layer4'):
                for p in self.backbone.layer4.parameters():
                    p.requires_grad = False
        if '-1' not in layers:
            if hasattr(self.backbone, 'fc'):
                for p in self.backbone.fc.parameters():
                    p.requires_grad = False
        if freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()


class GroundTruthComponent:
    """
    Stores sorted anomaly scores of a single ground truth component.
    Used to efficiently compute the region overlap for many increasing thresholds.
    """

    def __init__(self, anomaly_scores):
        """
        Initialize the module.

        Args:
            anomaly_scores: List of all anomaly scores within the ground truth
                            component as numpy array.
        """
        self.anomaly_scores = anomaly_scores.copy()
        self.anomaly_scores.sort()
        self.index = 0
        self.last_threshold = None

    def compute_overlap(self, threshold):
        """
        Compute the region overlap for a specific threshold.
        Thresholds must be passed in increasing order.

        Args:
            threshold: Threshold to compute the region overlap.

        Returns:
            Region overlap for the specified threshold.
        """
        if self.last_threshold is not None:
            assert self.last_threshold <= threshold
        while self.index < len(self.anomaly_scores) and self.anomaly_scores[self.index] <= threshold:
            self.index += 1
        return 1.0 - self.index / len(self.anomaly_scores)


def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
    """
    Extract anomaly scores for each ground truth connected component as well as anomaly scores for each potential false
    positive pixel from anomaly maps.

    Args:
        anomaly_maps:      List of anomaly maps (2D numpy arrays) that contain a real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that contain binary-valued ground truth labels
                           for each pixel. 0 indicates that a pixel is anomaly-free. 1 indicates that a pixel contains
                           an anomaly.

    Returns:
        ground_truth_components: A list of all ground truth connected components that appear in the dataset. For each
                                 component, a sorted list of its anomaly scores is stored.

        anomaly_scores_ok_pixels: A sorted list of anomaly scores of all anomaly-free pixels of the dataset. This list
                                  can be used to quickly select thresholds that fix a certain false positive rate.
    """
    assert len(anomaly_maps) == len(ground_truth_maps)
    ground_truth_components = []
    anomaly_scores_ok_pixels = np.zeros(len(ground_truth_maps) * ground_truth_maps[0].size)
    structure = np.ones((3, 3), dtype=int)
    ok_index = 0
    for gt_map, prediction in zip(ground_truth_maps, anomaly_maps):
        labeled, n_components = label(gt_map, structure)
        num_ok_pixels = len(prediction[labeled == 0])
        anomaly_scores_ok_pixels[ok_index:ok_index + num_ok_pixels] = prediction[labeled == 0].copy()
        ok_index += num_ok_pixels
        for k in range(n_components):
            component_scores = prediction[labeled == k + 1]
            ground_truth_components.append(GroundTruthComponent(component_scores))
    anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
    anomaly_scores_ok_pixels.sort()
    return ground_truth_components, anomaly_scores_ok_pixels


def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds):
    """
    Compute the PRO curve at equidistant interpolation points for a set of anomaly maps with corresponding ground
    truth maps. The number of interpolation points can be set manually.

    Args:
        anomaly_maps:      List of anomaly maps (2D numpy arrays) that contain a real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that contain binary-valued ground truth labels
                           for each pixel. 0 indicates that a pixel is anomaly-free. 1 indicates that a pixel contains
                           an anomaly.

        num_thresholds:    Number of thresholds to compute the PRO curve.
    Returns:
        fprs: List of false positive rates.
        pros: List of correspoding PRO values.
    """
    ground_truth_components, anomaly_scores_ok_pixels = collect_anomaly_scores(anomaly_maps, ground_truth_maps)
    threshold_positions = np.linspace(0, len(anomaly_scores_ok_pixels) - 1, num=num_thresholds, dtype=int)
    fprs = [1.0]
    pros = [1.0]
    for pos in threshold_positions:
        threshold = anomaly_scores_ok_pixels[pos]
        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)
        pro = 0.0
        for component in ground_truth_components:
            pro += component.compute_overlap(threshold)
        pro /= len(ground_truth_components)
        fprs.append(fpr)
        pros.append(pro)
    fprs = fprs[::-1]
    pros = pros[::-1]
    return fprs, pros


def trapezoid(x, y, x_max=None):
    """
    This function calculates the definit integral of a curve given by x- and corresponding y-values.
    In contrast to, e.g., 'numpy.trapz()', this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a warning.

    Args:
        x:     Samples from the domain of the function to integrate need to be sorted in ascending order. May contain
               the same value multiple times. In that case, the order of the corresponding y values will affect the
               integration with the trapezoidal rule.
        y:     Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be determined by interpolating between its
               neighbors. Must not lie outside of the range of x.

    Returns:
        Area under the curve.
    """
    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        None
    x = x[finite_mask]
    y = y[finite_mask]
    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            ins = bisect(x, x_max)
            assert 0 < ins < len(x)
            y_interp = y[ins - 1] + (y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])
        mask = x <= x_max
        x = x[mask]
        y = y[mask]
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def calculate_au_pro(gts, predictions, integration_limit=0.3, num_thresholds=100):
    """
    Compute the area under the PRO curve for a set of ground truth images and corresponding anomaly images.
    Args:
        gts:         List of tensors that contain the ground truth images for a single dataset object.
        predictions: List of tensors containing anomaly images for each ground truth image.
        integration_limit:    Integration limit to use when computing the area under the PRO curve.
        num_thresholds:       Number of thresholds to use to sample the area under the PRO curve.

    Returns:
        au_pro:    Area under the PRO curve computed up to the given integration limit.
        pro_curve: PRO curve values for localization (fpr,pro).
    """
    pro_curve = compute_pro(anomaly_maps=predictions, ground_truth_maps=gts, num_thresholds=num_thresholds)
    au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max=integration_limit)
    au_pro /= integration_limit
    return au_pro, pro_curve


def set_seeds(seed: 'int'=0) ->None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class Features(torch.nn.Module):

    def __init__(self, image_size=224, f_coreset=0.1, coreset_eps=0.9):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.deep_feature_extractor = Model(device=self.device)
        self.deep_feature_extractor
        self.deep_feature_extractor.freeze_parameters(layers=[], freeze_bn=True)
        self.image_size = image_size
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = KNNGaussianBlur(4)
        self.n_reweight = 3
        set_seeds(0)
        self.patch_lib = []
        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))
        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0

    def __call__(self, x):
        with torch.no_grad():
            feature_maps = self.deep_feature_extractor(x)
        feature_maps = [fmap for fmap in feature_maps]
        return feature_maps

    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label):
        raise NotImplementedError

    def compute_s_s_map(self, patch, feature_map_dims, mask, label):
        dist = torch.cdist(patch, self.patch_lib)
        min_val, min_idx = torch.min(dist, dim=1)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)
        m_test = patch[s_idx].unsqueeze(0)
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)
        w_dist = torch.cdist(m_star, self.patch_lib)
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)
        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - torch.exp(s_star / D) / torch.sum(torch.exp(m_star_knn / D))
        s = w * s_star
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map)
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def calculate_metrics(self):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)
        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)

    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)
        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib, n=int(self.f_coreset * self.patch_lib.shape[0]), eps=self.coreset_eps)
            self.patch_lib = self.patch_lib[self.coreset_idx]

    def get_coreset_idx_randomp(self, z_lib, n=1000, eps=0.9, float16=True, force_cpu=False):
        """Returns n coreset idx for given z_lib.
        Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
        CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)
        Args:
            z_lib:      (n, d) tensor of patches.
            n:          Number of patches to select.
            eps:        Agression of the sparse random projection.
            float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
            force_cpu:  Force cpu, useful in case of GPU OOM.
        Returns:
            coreset indices
        """
        None
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps)
            z_lib = torch.tensor(transformer.fit_transform(z_lib))
            None
        except ValueError:
            None
        select_idx = 0
        last_item = z_lib[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        if float16:
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        if torch.cuda.is_available() and not force_cpu:
            last_item = last_item
            z_lib = z_lib
            min_distances = min_distances
        for _ in tqdm(range(n - 1)):
            distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
            min_distances = torch.minimum(distances, min_distances)
            select_idx = torch.argmax(min_distances)
            last_item = z_lib[select_idx:select_idx + 1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx)
        return torch.stack(coreset_idx)


def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])


def get_fpfh_features(organized_pc, voxel_size=0.05):
    organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
    radius_normal = voxel_size * 2
    o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    fpfh = pcd_fpfh.data.T
    full_fpfh = np.zeros((unorganized_pc.shape[0], fpfh.shape[1]), dtype=fpfh.dtype)
    full_fpfh[nonzero_indices, :] = fpfh
    full_fpfh_reshaped = full_fpfh.reshape((organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1]))
    full_fpfh_tensor = torch.tensor(full_fpfh_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
    return full_fpfh_tensor


class FPFHFeatures(Features):

    def add_sample_to_mem_bank(self, sample):
        fpfh_feature_maps = get_fpfh_features(sample[1])
        fpfh_feature_maps_resized = self.resize(self.average(fpfh_feature_maps))
        fpfh_patch = fpfh_feature_maps_resized.reshape(fpfh_feature_maps_resized.shape[1], -1).T
        self.patch_lib.append(fpfh_patch)

    def predict(self, sample, mask, label):
        depth_feature_maps = get_fpfh_features(sample[1])
        depth_feature_maps_resized = self.resize(self.average(depth_feature_maps))
        patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T
        self.compute_s_s_map(patch, depth_feature_maps_resized.shape[-2:], mask, label)


class HoGFeatures(Features):

    def add_sample_to_mem_bank(self, sample):
        sample = sample[2]
        hog_feature = hog(sample[0, 0, :, :], orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=False, feature_vector=False)
        hog_feature = hog_feature.reshape(hog_feature.shape[0], hog_feature.shape[1], hog_feature.shape[2] * hog_feature.shape[3] * hog_feature.shape[4])
        hog_feature = torch.tensor(hog_feature.squeeze()).permute(2, 0, 1).unsqueeze(dim=0)
        hog_depth_patch = hog_feature.reshape(hog_feature.shape[1], -1).T
        self.patch_lib.append(hog_depth_patch)

    def predict(self, sample, mask, label):
        sample = sample[2]
        hog_features = hog(sample[0, 0, :, :], orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=False, feature_vector=False)
        hog_features = hog_features.reshape(hog_features.shape[0], hog_features.shape[1], hog_features.shape[2] * hog_features.shape[3] * hog_features.shape[4])
        depth_feature_maps_resized = torch.tensor(hog_features.squeeze()).permute(2, 0, 1).unsqueeze(dim=0)
        patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T
        self.compute_s_s_map(patch, depth_feature_maps_resized.shape[-2:], mask, label)


class RGBFPFHFeatures(Features):

    def add_sample_to_mem_bank(self, sample):
        rgb_feature_maps = self(sample[0])
        if self.resize is None:
            largest_fmap_size = rgb_feature_maps[0].shape[-2:]
            self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
        rgb_resized_maps = [self.resize(self.average(fmap)) for fmap in rgb_feature_maps]
        rgb_patch = torch.cat(rgb_resized_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        fpfh_feature_maps = get_fpfh_features(sample[1])
        fpfh_feature_maps_resized = self.resize(self.average(fpfh_feature_maps))
        fpfh_patch = fpfh_feature_maps_resized.reshape(fpfh_feature_maps_resized.shape[1], -1).T
        concat_patch = torch.cat([rgb_patch, fpfh_patch], dim=1)
        self.patch_lib.append(concat_patch)

    def predict(self, sample, mask, label):
        rgb_sample = sample[0]
        pc_sample = sample[1]
        rgb_feature_maps = self(rgb_sample)
        rgb_resized_maps = [self.resize(self.average(fmap)) for fmap in rgb_feature_maps]
        rgb_patch = torch.cat(rgb_resized_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        depth_feature_maps = get_fpfh_features(pc_sample)
        depth_feature_maps_resized = self.resize(self.average(depth_feature_maps))
        depth_patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T
        concat_patch = torch.cat([rgb_patch, depth_patch], dim=1)
        concat_feature_maps = torch.cat([rgb_resized_maps[0], depth_feature_maps_resized], dim=1)
        self.compute_s_s_map(concat_patch, concat_feature_maps.shape[-2:], mask, label)


def _get_reshape_kernel(kd: 'int', ky: 'int', kx: 'int') ->torch.Tensor:
    """Utility function, which returns neigh2channels conv kernel."""
    numel: 'int' = kd * ky * kx
    weight = torch.eye(numel)
    return weight.view(numel, kd, ky, kx)


def get_sift_pooling_kernel(ksize: 'int'=25) ->torch.Tensor:
    """Return a weighted pooling kernel for SIFT descriptor.

    Args:
        ksize: kernel_size.

    Returns:
        the pooling kernel with shape :math:`(ksize, ksize)`.
    """
    ks_2: 'float' = float(ksize) / 2.0
    xc2: 'torch.Tensor' = ks_2 - (torch.arange(ksize).float() + 0.5 - ks_2).abs()
    kernel: 'torch.Tensor' = torch.ger(xc2, xc2) / ks_2 ** 2
    return kernel


class DenseSIFTDescriptor(nn.Module):
    """Module, which computes SIFT descriptor densely over the image.

    Args:
        num_ang_bins: Number of angular bins. (8 is default)
        num_spatial_bins: Number of spatial bins per descriptor (4 is default).
    You might want to set odd number and relevant padding to keep feature map size
        spatial_bin_size: Size of a spatial bin in pixels (4 is default)
        clipval: clipping value to reduce single-bin dominance
        rootsift: (bool) if True, RootSIFT (Arandjelović et. al, 2012) is computed
        stride: default 1
        padding: default 0

    Returns:
        torch.Tensor: DenseSIFT descriptor of the image

    Shape:
        - Input: (B, 1, H, W)
        - Output: (B, num_ang_bins * num_spatial_bins ** 2, (H+padding)/stride, (W+padding)/stride)

    Examples::
        >>> input =  torch.rand(2, 1, 200, 300)
        >>> SIFT = DenseSIFTDescriptor()
        >>> descs = SIFT(input) # 2x128x194x294
    """

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(' + 'num_ang_bins=' + str(self.num_ang_bins) + ', ' + 'num_spatial_bins=' + str(self.num_spatial_bins) + ', ' + 'spatial_bin_size=' + str(self.spatial_bin_size) + ', ' + 'rootsift=' + str(self.rootsift) + ', ' + 'stride=' + str(self.stride) + ', ' + 'clipval=' + str(self.clipval) + ')'

    def __init__(self, num_ang_bins: 'int'=8, num_spatial_bins: 'int'=4, spatial_bin_size: 'int'=4, rootsift: 'bool'=True, clipval: 'float'=0.2, stride: 'int'=1, padding: 'int'=1) ->None:
        super().__init__()
        self.eps = 1e-10
        self.num_ang_bins = num_ang_bins
        self.num_spatial_bins = num_spatial_bins
        self.spatial_bin_size = spatial_bin_size
        self.clipval = clipval
        self.rootsift = rootsift
        self.stride = stride
        self.pad = padding
        nw = get_sift_pooling_kernel(ksize=self.spatial_bin_size).float()
        self.bin_pooling_kernel = nn.Conv2d(1, 1, kernel_size=(nw.size(0), nw.size(1)), stride=(1, 1), bias=False, padding=(nw.size(0) // 2, nw.size(1) // 2))
        self.bin_pooling_kernel.weight.data.copy_(nw.reshape(1, 1, nw.size(0), nw.size(1)))
        self.PoolingConv = nn.Conv2d(num_ang_bins, num_ang_bins * num_spatial_bins ** 2, kernel_size=(num_spatial_bins, num_spatial_bins), stride=(self.stride, self.stride), bias=False, padding=(self.pad, self.pad))
        self.PoolingConv.weight.data.copy_(_get_reshape_kernel(num_ang_bins, num_spatial_bins, num_spatial_bins).float())
        return

    def get_pooling_kernel(self) ->torch.Tensor:
        return self.bin_pooling_kernel.weight.detach()

    def forward(self, input):
        if not isinstance(input, torch.Tensor):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError('Invalid input shape, we expect Bx1xHxW. Got: {}'.format(input.shape))
        B, CH, W, H = input.size()
        self.bin_pooling_kernel = self.bin_pooling_kernel.to(input.dtype)
        self.PoolingConv = self.PoolingConv.to(input.dtype)
        grads: 'torch.Tensor' = spatial_gradient(input, 'diff')
        gx: 'torch.Tensor' = grads[:, :, 0]
        gy: 'torch.Tensor' = grads[:, :, 1]
        mag: 'torch.Tensor' = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori: 'torch.Tensor' = torch.atan2(gy, gx + self.eps) + 2.0 * pi
        o_big: 'torch.Tensor' = float(self.num_ang_bins) * ori / (2.0 * pi)
        bo0_big_: 'torch.Tensor' = torch.floor(o_big)
        wo1_big_: 'torch.Tensor' = o_big - bo0_big_
        bo0_big: 'torch.Tensor' = bo0_big_ % self.num_ang_bins
        bo1_big: 'torch.Tensor' = (bo0_big + 1) % self.num_ang_bins
        wo0_big: 'torch.Tensor' = (1.0 - wo1_big_) * mag
        wo1_big: 'torch.Tensor' = wo1_big_ * mag
        ang_bins = []
        for i in range(0, self.num_ang_bins):
            out = self.bin_pooling_kernel((bo0_big == i) * wo0_big + (bo1_big == i) * wo1_big)
            ang_bins.append(out)
        ang_bins = torch.cat(ang_bins, dim=1)
        out_no_norm = self.PoolingConv(ang_bins)
        out = F.normalize(out_no_norm, dim=1, p=2).clamp_(0, float(self.clipval))
        out = F.normalize(out, dim=1, p=2)
        if self.rootsift:
            out = torch.sqrt(F.normalize(out, p=1) + self.eps)
        return out


class SIFTFeatures(Features):

    def __init__(self):
        super().__init__()
        self.dsift = DenseSIFTDescriptor()

    def add_sample_to_mem_bank(self, sample):
        sample = sample[2]
        dsift_feat = self.dsift(sample[:, 0, :, :].unsqueeze(dim=1)).detach()
        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))
        sift_depth_resized_maps = self.resize(self.average(dsift_feat))
        sift_patch = sift_depth_resized_maps.reshape(sift_depth_resized_maps.shape[1], -1).T
        self.patch_lib.append(sift_patch)

    def predict(self, sample, mask, label):
        sample = sample[2]
        feature_maps = self.dsift(sample[:, 0, :, :].unsqueeze(dim=1)).detach()
        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))
        depth_feature_maps_resized = self.resize(self.average(feature_maps))
        patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T
        self.compute_s_s_map(patch, depth_feature_maps_resized.shape[-2:], mask, label)

