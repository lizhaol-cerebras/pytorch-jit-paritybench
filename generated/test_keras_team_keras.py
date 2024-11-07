import sys
_module = sys.modules[__name__]
del sys
api_gen = _module
benchmarks = _module
layer_benchmark = _module
activation_benchmark = _module
attention_benchmark = _module
base_benchmark = _module
conv_benchmark = _module
core_benchmark = _module
merge_benchmark = _module
normalization_benchmark = _module
pooling_benchmark = _module
regularization_benchmark = _module
reshaping_benchmark = _module
rnn_benchmark = _module
model_benchmark = _module
benchmark_utils = _module
bert_benchmark = _module
image_classification_benchmark = _module
torch_ctl_benchmark = _module
benchmark_utils = _module
conv_model_benchmark = _module
dense_model_benchmark = _module
conftest = _module
demo_custom_jax_workflow = _module
demo_custom_layer_backend_agnostic = _module
demo_custom_tf_workflow = _module
demo_custom_torch_workflow = _module
demo_functional = _module
demo_jax_distributed = _module
demo_mnist_convnet = _module
demo_subclass = _module
demo_torch_multi_gpu = _module
custom_train_step_in_jax = _module
custom_train_step_in_tensorflow = _module
custom_train_step_in_torch = _module
distributed_training_with_jax = _module
distributed_training_with_tensorflow = _module
distributed_training_with_torch = _module
functional_api = _module
making_new_layers_and_models_via_subclassing = _module
sequential_model = _module
training_with_built_in_methods = _module
transfer_learning = _module
understanding_masking_and_padding = _module
writing_a_custom_training_loop_in_jax = _module
writing_a_custom_training_loop_in_tensorflow = _module
writing_a_custom_training_loop_in_torch = _module
writing_your_own_callbacks = _module
basic_full_flow = _module
boston_housing_test = _module
california_housing_test = _module
cifar100_test = _module
cifar10_test = _module
fashion_mnist_test = _module
imdb_test = _module
mnist_test = _module
reuters_test = _module
import_test = _module
jax_custom_fit_test = _module
model_visualization_test = _module
numerical_test = _module
tf_custom_fit_test = _module
tf_distribute_training_test = _module
torch_custom_fit_test = _module
torch_workflow_test = _module
keras = _module
api = _module
_tf_keras = _module
activations = _module
applications = _module
convnext = _module
densenet = _module
efficientnet = _module
efficientnet_v2 = _module
imagenet_utils = _module
inception_resnet_v2 = _module
inception_v3 = _module
mobilenet = _module
mobilenet_v2 = _module
mobilenet_v3 = _module
nasnet = _module
resnet = _module
resnet50 = _module
resnet_v2 = _module
vgg16 = _module
vgg19 = _module
xception = _module
backend = _module
callbacks = _module
config = _module
constraints = _module
datasets = _module
boston_housing = _module
california_housing = _module
cifar10 = _module
cifar100 = _module
fashion_mnist = _module
imdb = _module
mnist = _module
reuters = _module
distribution = _module
dtype_policies = _module
export = _module
initializers = _module
layers = _module
legacy = _module
saving = _module
losses = _module
metrics = _module
mixed_precision = _module
models = _module
ops = _module
image = _module
linalg = _module
nn = _module
numpy = _module
optimizers = _module
schedules = _module
preprocessing = _module
sequence = _module
text = _module
quantizers = _module
random = _module
regularizers = _module
tree = _module
utils = _module
bounding_boxes = _module
visualization = _module
src = _module
activations_test = _module
api_export = _module
applications_test = _module
densenet = _module
imagenet_utils = _module
imagenet_utils_test = _module
nasnet = _module
backend = _module
common = _module
backend_utils = _module
backend_utils_test = _module
compute_output_spec_test = _module
dtypes = _module
dtypes_test = _module
global_state = _module
global_state_test = _module
keras_tensor = _module
keras_tensor_test = _module
masking = _module
masking_test = _module
name_scope = _module
name_scope_test = _module
stateless_scope = _module
stateless_scope_test = _module
symbolic_scope = _module
symbolic_scope_test = _module
tensor_attributes = _module
thread_safe_test = _module
variables = _module
variables_test = _module
config = _module
jax = _module
core = _module
distribution_lib = _module
distribution_lib_test = _module
layer = _module
math = _module
optimizer = _module
rnn = _module
sparse = _module
tensorboard = _module
trainer = _module
tensorflow = _module
core = _module
distribute_test = _module
optimizer_distribute_test = _module
saved_model_test = _module
trackable = _module
trainer = _module
device_scope_test = _module
core = _module
image = _module
layer = _module
linalg = _module
math = _module
nn = _module
numpy = _module
optimizers = _module
torch_adadelta = _module
torch_adagrad = _module
torch_adam = _module
torch_adamax = _module
torch_adamw = _module
torch_lion = _module
torch_nadam = _module
torch_optimizer = _module
torch_parallel_optimizer = _module
torch_rmsprop = _module
torch_sgd = _module
random = _module
rnn = _module
trainer = _module
backup_and_restore = _module
backup_and_restore_test = _module
callback = _module
callback_list = _module
callback_test = _module
csv_logger = _module
csv_logger_test = _module
early_stopping = _module
early_stopping_test = _module
history = _module
lambda_callback = _module
lambda_callback_test = _module
learning_rate_scheduler = _module
learning_rate_scheduler_test = _module
model_checkpoint = _module
model_checkpoint_test = _module
progbar_logger = _module
reduce_lr_on_plateau = _module
reduce_lr_on_plateau_test = _module
remote_monitor = _module
remote_monitor_test = _module
swap_ema_weights = _module
swap_ema_weights_test = _module
tensorboard = _module
tensorboard_test = _module
terminate_on_nan = _module
terminate_on_nan_test = _module
constraints_test = _module
cifar = _module
dtype_policy = _module
dtype_policy_map = _module
dtype_policy_map_test = _module
dtype_policy_test = _module
export_lib = _module
export_lib_test = _module
constant_initializers = _module
constant_initializers_test = _module
initializer = _module
random_initializers = _module
random_initializers_test = _module
activation = _module
activation_test = _module
elu = _module
elu_test = _module
leaky_relu = _module
leaky_relu_test = _module
prelu = _module
prelu_test = _module
relu = _module
relu_test = _module
softmax = _module
softmax_test = _module
attention = _module
additive_attention = _module
additive_attention_test = _module
attention_test = _module
grouped_query_attention = _module
grouped_query_attention_test = _module
multi_head_attention = _module
multi_head_attention_test = _module
convolutional = _module
base_conv = _module
base_conv_transpose = _module
base_depthwise_conv = _module
base_separable_conv = _module
conv1d = _module
conv1d_transpose = _module
conv2d = _module
conv2d_transpose = _module
conv3d = _module
conv3d_transpose = _module
conv_test = _module
conv_transpose_test = _module
depthwise_conv1d = _module
depthwise_conv2d = _module
depthwise_conv_test = _module
separable_conv1d = _module
separable_conv2d = _module
separable_conv_test = _module
dense = _module
dense_test = _module
einsum_dense = _module
einsum_dense_test = _module
embedding = _module
embedding_test = _module
identity = _module
identity_test = _module
input_layer = _module
input_layer_test = _module
lambda_layer = _module
lambda_layer_test = _module
wrapper = _module
wrapper_test = _module
input_spec = _module
layer = _module
layer_test = _module
merging = _module
add = _module
average = _module
base_merge = _module
concatenate = _module
dot = _module
maximum = _module
merging_test = _module
minimum = _module
multiply = _module
subtract = _module
normalization = _module
batch_normalization = _module
batch_normalization_test = _module
group_normalization = _module
group_normalization_test = _module
layer_normalization = _module
layer_normalization_test = _module
spectral_normalization = _module
spectral_normalization_test = _module
unit_normalization = _module
unit_normalization_test = _module
pooling = _module
average_pooling1d = _module
average_pooling2d = _module
average_pooling3d = _module
average_pooling_test = _module
base_global_pooling = _module
base_pooling = _module
global_average_pooling1d = _module
global_average_pooling2d = _module
global_average_pooling3d = _module
global_average_pooling_test = _module
global_max_pooling1d = _module
global_max_pooling2d = _module
global_max_pooling3d = _module
global_max_pooling_test = _module
max_pooling1d = _module
max_pooling2d = _module
max_pooling3d = _module
max_pooling_test = _module
category_encoding = _module
category_encoding_test = _module
discretization = _module
discretization_test = _module
feature_space = _module
feature_space_test = _module
hashed_crossing = _module
hashed_crossing_test = _module
hashing = _module
hashing_test = _module
image_preprocessing = _module
auto_contrast = _module
auto_contrast_test = _module
base_image_preprocessing_layer = _module
bounding_box = _module
converters = _module
formats = _module
validation = _module
center_crop = _module
center_crop_test = _module
max_num_bounding_box = _module
max_num_bounding_box_test = _module
random_brightness = _module
random_brightness_test = _module
random_contrast = _module
random_contrast_test = _module
random_crop = _module
random_crop_test = _module
random_flip = _module
random_flip_test = _module
random_rotation = _module
random_rotation_test = _module
random_translation = _module
random_translation_test = _module
random_zoom = _module
random_zoom_test = _module
resizing = _module
resizing_test = _module
solarization = _module
solarization_test = _module
index_lookup = _module
index_lookup_test = _module
integer_lookup = _module
integer_lookup_test = _module
mel_spectrogram = _module
mel_spectrogram_test = _module
normalization_test = _module
pipeline = _module
pipeline_test = _module
rescaling = _module
rescaling_test = _module
stft_spectrogram = _module
stft_spectrogram_test = _module
string_lookup = _module
string_lookup_test = _module
text_vectorization = _module
text_vectorization_test = _module
tf_data_layer = _module
regularization = _module
activity_regularization = _module
activity_regularization_test = _module
alpha_dropout = _module
alpha_dropout_test = _module
dropout = _module
dropout_test = _module
gaussian_dropout = _module
gaussian_dropout_test = _module
gaussian_noise = _module
gaussian_noise_test = _module
spatial_dropout = _module
spatial_dropout_test = _module
reshaping = _module
cropping1d = _module
cropping1d_test = _module
cropping2d = _module
cropping2d_test = _module
cropping3d = _module
cropping3d_test = _module
flatten = _module
flatten_test = _module
permute = _module
permute_test = _module
repeat_vector = _module
repeat_vector_test = _module
reshape = _module
reshape_test = _module
up_sampling1d = _module
up_sampling1d_test = _module
up_sampling2d = _module
up_sampling2d_test = _module
up_sampling3d = _module
up_sampling3d_test = _module
zero_padding1d = _module
zero_padding1d_test = _module
zero_padding2d = _module
zero_padding2d_test = _module
zero_padding3d = _module
zero_padding3d_test = _module
bidirectional = _module
bidirectional_test = _module
conv_lstm = _module
conv_lstm1d = _module
conv_lstm1d_test = _module
conv_lstm2d = _module
conv_lstm2d_test = _module
conv_lstm3d = _module
conv_lstm3d_test = _module
conv_lstm_test = _module
dropout_rnn_cell = _module
dropout_rnn_cell_test = _module
gru = _module
gru_test = _module
lstm = _module
lstm_test = _module
rnn_test = _module
simple_rnn = _module
simple_rnn_test = _module
stacked_rnn_cells = _module
stacked_rnn_cells_test = _module
time_distributed = _module
time_distributed_test = _module
json_utils = _module
json_utils_test = _module
legacy_h5_format = _module
legacy_h5_format_test = _module
saving_options = _module
saving_utils = _module
serialization = _module
loss = _module
loss_test = _module
losses_test = _module
accuracy_metrics = _module
accuracy_metrics_test = _module
confusion_metrics = _module
confusion_metrics_test = _module
correlation_metrics = _module
correlation_metrics_test = _module
f_score_metrics = _module
f_score_metrics_test = _module
hinge_metrics = _module
hinge_metrics_test = _module
iou_metrics = _module
iou_metrics_test = _module
metric = _module
metric_test = _module
metrics_utils = _module
probabilistic_metrics = _module
probabilistic_metrics_test = _module
reduction_metrics = _module
reduction_metrics_test = _module
regression_metrics = _module
regression_metrics_test = _module
cloning = _module
cloning_test = _module
functional = _module
functional_test = _module
model = _module
model_test = _module
sequential = _module
sequential_test = _module
variable_mapping = _module
variable_mapping_test = _module
core_test = _module
function = _module
function_test = _module
image = _module
image_test = _module
linalg_test = _module
math_test = _module
nn_test = _module
node = _module
node_test = _module
numpy = _module
numpy_test = _module
operation = _module
operation_test = _module
operation_utils = _module
operation_utils_test = _module
symbolic_arguments = _module
symbolic_arguments_test = _module
adadelta = _module
adadelta_test = _module
adafactor = _module
adafactor_test = _module
adagrad = _module
adagrad_test = _module
adam = _module
adam_test = _module
adamax = _module
adamax_test = _module
adamw = _module
adamw_test = _module
base_optimizer = _module
ftrl = _module
ftrl_test = _module
lamb = _module
lamb_test = _module
lion = _module
lion_test = _module
loss_scale_optimizer = _module
loss_scale_optimizer_test = _module
nadam = _module
nadam_test = _module
optimizer = _module
optimizer_sparse_test = _module
optimizer_test = _module
rmsprop = _module
rmsprop_test = _module
learning_rate_schedule = _module
learning_rate_schedule_test = _module
sgd = _module
sgd_test = _module
quantizers_test = _module
random_test = _module
seed_generator = _module
seed_generator_test = _module
regularizers_test = _module
file_editor = _module
file_editor_test = _module
keras_saveable = _module
object_registration = _module
object_registration_test = _module
saving_api = _module
saving_api_test = _module
saving_lib = _module
saving_lib_test = _module
serialization_lib = _module
serialization_lib_test = _module
testing = _module
test_case = _module
test_utils = _module
test_utils_test = _module
trainers = _module
compile_utils = _module
compile_utils_test = _module
data_adapters = _module
array_data_adapter = _module
array_data_adapter_test = _module
array_slicing = _module
data_adapter = _module
data_adapter_utils = _module
generator_data_adapter = _module
generator_data_adapter_test = _module
py_dataset_adapter = _module
py_dataset_adapter_test = _module
tf_dataset_adapter = _module
tf_dataset_adapter_test = _module
torch_data_loader_adapter = _module
torch_data_loader_adapter_test = _module
epoch_iterator = _module
epoch_iterator_test = _module
trainer = _module
trainer_test = _module
dmtree_impl = _module
optree_impl = _module
tree_api = _module
tree_test = _module
argument_validation = _module
audio_dataset_utils = _module
audio_dataset_utils_test = _module
backend_utils = _module
backend_utils_test = _module
code_stats = _module
code_stats_test = _module
dataset_utils = _module
dataset_utils_test = _module
dtype_utils = _module
dtype_utils_test = _module
file_utils = _module
file_utils_test = _module
image_dataset_utils = _module
image_dataset_utils_test = _module
image_utils = _module
io_utils = _module
io_utils_test = _module
jax_layer = _module
jax_layer_test = _module
jax_utils = _module
model_visualization = _module
module_utils = _module
naming = _module
naming_test = _module
numerical_utils = _module
numerical_utils_test = _module
progbar = _module
python_utils = _module
python_utils_test = _module
rng_utils = _module
rng_utils_test = _module
sequence_utils = _module
sequence_utils_test = _module
summary_utils = _module
summary_utils_test = _module
text_dataset_utils = _module
text_dataset_utils_test = _module
tf_utils = _module
timeseries_dataset_utils = _module
timeseries_dataset_utils_test = _module
torch_utils = _module
torch_utils_test = _module
traceback_utils = _module
tracking = _module
tracking_test = _module
version = _module
draw_bounding_boxes = _module
draw_segmentation_masks = _module
plot_bounding_box_gallery = _module
plot_image_gallery = _module
plot_segmentation_mask_gallery = _module
pip_build = _module

from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import time


import numpy as np


import torch


import torch.nn as nn


import torch.optim as optim


import torch.multiprocessing as mp


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data.distributed import DistributedSampler


import tensorflow as tf


import re


import warnings


import functools


import itertools


from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice


from tensorflow.python.eager import context as tf_context


from typing import Iterator


from typing import Tuple


import math


import torch.nn.functional as tnn


import torch._dynamo as dynamo


import logging


import collections


import random


import tensorflow.summary as summary


from tensorflow.compat.v1 import SummaryMetadata


from tensorflow.core.util import event_pb2


from tensorflow.python.lib.io import tf_record


import scipy.signal


from numpy.lib.stride_tricks import as_strided


import string


import inspect


from functools import wraps


from tensorflow import data as tf_data


import typing


import scipy.ndimage


from itertools import combinations


import types


import pandas


import scipy


import copy


from torch.utils.data import Dataset as TorchDataset


num_classes = 10


class TorchModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(8192, 64)
        self.activation1 = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(64, 8)
        self.activation2 = torch.nn.ReLU()
        self.dense3 = torch.nn.Linear(8, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dense2(x)
        x = self.activation2(x)
        x = self.dense3(x)
        x = self.softmax(x)
        return x


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = keras.Sequential([layers.Input(shape=(28, 28, 1)), layers.Conv2D(32, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Flatten(), layers.Dropout(0.5), layers.Dense(num_classes)])

    def forward(self, x):
        return self.model(x)


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = layers.Dense(1)

    def forward(self, x):
        x = self.fc1(x)
        return x


class CallContext:

    def __init__(self, entry_layer):
        self.entry_layer = entry_layer
        self.training = None


def to_snake_case(name):
    name = re.sub('\\W+', '', name)
    name = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', name)
    name = re.sub('([a-z])([A-Z])', '\\1_\\2', name).lower()
    return name


def uniquify(name):
    object_name_uids = global_state.get_global_attribute('object_name_uids', default=collections.defaultdict(int), set_to_default=True)
    if name in object_name_uids:
        unique_name = f'{name}_{object_name_uids[name]}'
    else:
        unique_name = name
    object_name_uids[name] += 1
    return unique_name


def auto_name(prefix):
    prefix = to_snake_case(prefix)
    return uniquify(prefix)


_BACKEND = 'tensorflow'


class KerasHistory(collections.namedtuple('KerasHistory', ['operation', 'node_index', 'tensor_index'])):
    """Tracks the Operation call that created a Tensor.

    During construction of Keras Functions, this metadata is added to
    each Tensor produced as the output of an Operation.
    This allows Keras to track how each Tensor was produced, and
    this information is later retraced by the `Function` class to
    reconstruct the Operations graph.

    Attributes:
      operation: The Operation instance that produced the Tensor.
      node_index: The specific call to the Operation that produced this Tensor.
        Operations can be called multiple times in order to share weights. A new
        node is created every time an Operation is called. The corresponding
        node that represents the call event that produced the Tensor can be
        found at `op._inbound_nodes[node_index]`.
      tensor_index: The output index for this Tensor.
        Always zero if the Operation that produced this Tensor
        only has one output. Nested structures of
        Tensors are deterministically assigned an index via `nest.flatten`.
    """
    __slots__ = ()


class SymbolicArguments:

    def __init__(self, *args, **kwargs):
        self.args = tree.map_structure(lambda x: x, args)
        self.kwargs = tree.map_structure(lambda x: x, kwargs)
        self._flat_arguments = tree.flatten((self.args, self.kwargs))
        if not self.kwargs and len(self.args) == 1 and isinstance(self.args[0], KerasTensor):
            self._single_positional_tensor = self.args[0]
        else:
            self._single_positional_tensor = None
        self.keras_tensors = []
        for arg in self._flat_arguments:
            if isinstance(arg, KerasTensor):
                self.keras_tensors.append(arg)

    def convert(self, conversion_fn):
        args = tree.map_structure(conversion_fn, self.args)
        kwargs = tree.map_structure(conversion_fn, self.kwargs)
        return args, kwargs

    def fill_in(self, tensor_dict):
        """Maps KerasTensors to computed values using `tensor_dict`.

        `tensor_dict` maps `KerasTensor` instances to their current values.
        """
        if self._single_positional_tensor is not None:
            return (tensor_dict[id(self._single_positional_tensor)],), {}

        def switch_fn(x):
            if isinstance(x, KerasTensor):
                return tensor_dict.get(id(x), None)
            return x
        return self.convert(switch_fn)

