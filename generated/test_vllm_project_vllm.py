import sys
_module = sys.modules[__name__]
del sys
test_lm_eval_correctness = _module
backend_request_func = _module
benchmark_latency = _module
benchmark_prefix_caching = _module
benchmark_prioritization = _module
benchmark_serving = _module
benchmark_throughput = _module
w8a8_benchmarks = _module
weight_shapes = _module
benchmark_aqlm = _module
benchmark_layernorm = _module
benchmark_machete = _module
benchmark_marlin = _module
benchmark_moe = _module
benchmark_paged_attention = _module
benchmark_quant = _module
benchmark_rope = _module
benchmark_shapes = _module
graph_machete_bench = _module
benchmark_hashing = _module
hipify = _module
collect_env = _module
vllm_cutlass_library_extension = _module
generate = _module
conf = _module
generate_examples = _module
api_client = _module
aqlm_example = _module
cpu_offload = _module
florence2_inference = _module
extract_scales = _module
quantize = _module
gguf_inference = _module
gradio_openai_chatbot_webserver = _module
gradio_webserver = _module
llm_engine_example = _module
lora_with_quantization_inference = _module
multilora_inference = _module
offline_chat_with_tools = _module
offline_inference = _module
offline_inference_arctic = _module
offline_inference_audio_language = _module
offline_inference_chat = _module
offline_inference_distributed = _module
offline_inference_embedding = _module
offline_inference_encoder_decoder = _module
offline_inference_mlpspeculator = _module
offline_inference_neuron = _module
offline_inference_neuron_int8_quantization = _module
offline_inference_pixtral = _module
offline_inference_tpu = _module
offline_inference_vision_language = _module
offline_inference_vision_language_embedding = _module
offline_inference_vision_language_multi_image = _module
offline_inference_with_prefix = _module
offline_inference_with_profiler = _module
offline_profile = _module
openai_chat_completion_client = _module
openai_chat_completion_client_for_multimodal = _module
openai_chat_completion_client_with_tools = _module
openai_chat_embedding_client_for_multimodal = _module
openai_completion_client = _module
openai_embedding_client = _module
dummy_client = _module
save_sharded_state = _module
tensorize_vllm_model = _module
find_cuda_init = _module
python_only_dev = _module
setup = _module
tests = _module
async_engine = _module
api_server_async_engine = _module
test_api_server = _module
test_async_llm_engine = _module
test_request_tracker = _module
basic_correctness = _module
test_basic_correctness = _module
test_chunked_prefill = _module
test_cpu_offload = _module
test_preemption = _module
compile = _module
piecewise = _module
test_simple = _module
test_toy_llama = _module
test_full_graph = _module
test_wrapper = _module
utils = _module
conftest = _module
core = _module
block = _module
e2e = _module
test_correctness = _module
test_correctness_sliding_window = _module
test_block_manager = _module
test_block_table = _module
test_common = _module
test_cpu_gpu_block_allocator = _module
test_naive_block = _module
test_prefix_caching_block = _module
test_chunked_prefill_scheduler = _module
test_num_computed_tokens_update = _module
test_scheduler = _module
test_scheduler_encoder_decoder = _module
test_serialization = _module
distributed = _module
test_ca_buffer_sharing = _module
test_comm_ops = _module
test_custom_all_reduce = _module
test_distributed_oot = _module
test_multi_node_assignment = _module
test_pipeline_parallel = _module
test_pipeline_partition = _module
test_pp_cudagraph = _module
test_pynccl = _module
test_same_node = _module
test_shm_broadcast = _module
test_utils = _module
encoder_decoder = _module
test_e2e_correctness = _module
engine = _module
output_processor = _module
test_multi_step = _module
test_stop_checker = _module
test_arg_utils = _module
test_computed_prefix_blocks = _module
test_custom_executor = _module
test_detokenization = _module
test_multiproc_workers = _module
test_short_mm_context = _module
test_skip_tokenizer_init = _module
test_stop_reason = _module
test_stop_strings = _module
entrypoints = _module
llm = _module
test_chat = _module
test_encode = _module
test_generate = _module
test_generate_multiple_loras = _module
test_guided_generate = _module
test_init = _module
test_lazy_outlines = _module
test_prompt_validation = _module
offline_mode = _module
test_offline_mode = _module
openai = _module
test_accuracy = _module
test_audio = _module
test_basic = _module
test_chat = _module
test_chat_template = _module
test_chunked_prompt = _module
test_cli_args = _module
test_completion = _module
test_embedding = _module
test_encoder_decoder = _module
test_lora_lineage = _module
test_metrics = _module
test_models = _module
test_oot_registration = _module
test_return_tokens_as_ids = _module
test_run_batch = _module
test_serving_chat = _module
test_serving_engine = _module
test_shutdown = _module
test_tokenization = _module
test_vision = _module
test_vision_embedding = _module
test_chat_utils = _module
kernels = _module
allclose_default = _module
quant_utils = _module
test_activation = _module
test_aqlm = _module
test_attention = _module
test_attention_selector = _module
test_awq = _module
test_awq_marlin = _module
test_awq_triton = _module
test_blocksparse_attention = _module
test_cache = _module
test_causal_conv1d = _module
test_cutlass = _module
test_encoder_decoder_attn = _module
test_flash_attn = _module
test_flashinfer = _module
test_fp8_quant = _module
test_ggml = _module
test_gguf = _module
test_gptq = _module
test_int8_quant = _module
test_layernorm = _module
test_machete_gemm = _module
test_mamba_ssm = _module
test_marlin_gemm = _module
test_moe = _module
test_permute_cols = _module
test_pos_encoding = _module
test_prefix_prefill = _module
test_rotary_embedding = _module
test_utils = _module
utils = _module
lora = _module
conftest = _module
data = _module
long_context_test_data = _module
test_baichuan = _module
test_chatglm3 = _module
test_gemma = _module
test_layers = _module
test_llama = _module
test_long_context = _module
test_lora_checkpoints = _module
test_lora_huggingface = _module
test_lora_manager = _module
test_minicpmv = _module
test_minicpmv_tp = _module
test_mixtral = _module
test_phi = _module
test_punica_sizes = _module
test_punica_variation = _module
test_quant_model = _module
test_tokenizer_group = _module
test_utils = _module
test_worker = _module
utils = _module
metrics = _module
model_executor = _module
test_enabled_custom_ops = _module
test_guided_processors = _module
weight_utils = _module
models = _module
decoder_only = _module
audio_language = _module
test_ultravox = _module
language = _module
test_aqlm = _module
test_fp8 = _module
test_gptq_marlin = _module
test_gptq_marlin_24 = _module
test_granite = _module
test_granitemoe = _module
test_jamba = _module
test_mamba = _module
test_mistral = _module
test_modelopt = _module
test_phimoe = _module
vision_language = _module
mm_processor_kwargs = _module
test_llava_next = _module
test_phi3v = _module
test_qwen = _module
test_qwen2_vl = _module
test_h2ovl = _module
test_intern_vit = _module
test_internvl = _module
test_pixtral = _module
vlm_utils = _module
builders = _module
case_filtering = _module
core = _module
custom_inputs = _module
model_utils = _module
runners = _module
types = _module
embedding = _module
test_cls_models = _module
utils = _module
test_llava_next = _module
test_phi3v = _module
test_bart = _module
test_broadcast = _module
test_florence2 = _module
test_mllama = _module
test_registry = _module
utils = _module
mq_llm_engine = _module
test_abort = _module
test_error_handling = _module
test_load = _module
multi_step = _module
test_correctness_async_llm = _module
test_correctness_llm = _module
multimodal = _module
test_base = _module
test_mapper = _module
test_processor_kwargs = _module
vllm_add_dummy_model = _module
my_gemma_embedding = _module
my_llava = _module
my_opt = _module
prefix_caching = _module
test_disable_sliding_window = _module
test_prefix_caching = _module
test_bloom = _module
test_multi_adapter_inference = _module
test_pa_lora = _module
quantization = _module
test_bitsandbytes = _module
test_compressed_tensors = _module
test_configs = _module
test_experts_int8 = _module
test_fp8 = _module
test_ipex_quant = _module
test_lm_head = _module
samplers = _module
test_beam_search = _module
test_ignore_eos = _module
test_logits_processor = _module
test_logprobs = _module
test_no_bad_words = _module
test_ranks = _module
test_rejection_sampler = _module
test_sampler = _module
test_seeded_generate = _module
test_typical_acceptance_sampler = _module
spec_decode = _module
test_compatibility = _module
test_eagle_correctness = _module
test_integration = _module
test_integration_dist_tp2 = _module
test_integration_dist_tp4 = _module
test_medusa_correctness = _module
test_mlp_correctness = _module
test_multistep_correctness = _module
test_ngram_correctness = _module
test_seed = _module
test_batch_expansion = _module
test_dynamic_spec_decode = _module
test_metrics = _module
test_multi_step_worker = _module
test_ngram_worker = _module
test_scorer = _module
test_spec_decode_worker = _module
test_utils = _module
utils = _module
tensorizer_loader = _module
conftest = _module
test_tensorizer = _module
test_cache_block_hashing = _module
test_config = _module
test_embedded_commit = _module
test_inputs = _module
test_logger = _module
test_logits_processor = _module
test_regression = _module
test_sampling_params = _module
test_scalartype = _module
test_sequence = _module
test_sharded_state_loader = _module
tokenization = _module
test_cached_tokenizer = _module
test_detokenize = _module
test_get_eos = _module
test_tokenizer = _module
tool_use = _module
test_chat_completion_request_validations = _module
test_chat_completions = _module
test_jamba_tool_parser = _module
test_parallel_tool_calls = _module
test_tool_calls = _module
tpu = _module
test_compilation = _module
test_custom_dispatcher = _module
tracing = _module
test_tracing = _module
utils = _module
test_weight_loading = _module
worker = _module
test_encoder_decoder_model_runner = _module
test_model_input = _module
test_model_runner = _module
test_profile = _module
test_swap = _module
print_layerwise_table = _module
visualize_layerwise_profile = _module
report_build_time_ninja = _module
use_existing_torch = _module
vllm = _module
_custom_ops = _module
_ipex_ops = _module
adapter_commons = _module
layers = _module
models = _module
request = _module
worker_manager = _module
assets = _module
audio = _module
base = _module
image = _module
video = _module
attention = _module
backends = _module
abstract = _module
blocksparse_attn = _module
flash_attn = _module
flashinfer = _module
hpu_attn = _module
ipex_attn = _module
openvino = _module
pallas = _module
placeholder_attn = _module
rocm_flash_attn = _module
torch_sdpa = _module
utils = _module
xformers = _module
layer = _module
ops = _module
blocksparse_attention = _module
blocksparse_attention_kernel = _module
interface = _module
utils = _module
hpu_paged_attn = _module
ipex_attn = _module
paged_attn = _module
prefix_prefill = _module
triton_flash_attention = _module
selector = _module
beam_search = _module
compilation = _module
backends = _module
compile_context = _module
config = _module
counter = _module
decorators = _module
levels = _module
wrapper = _module
config = _module
connections = _module
block_table = _module
common = _module
cpu_gpu_block_allocator = _module
interfaces = _module
naive_block = _module
prefix_caching_block = _module
block_manager = _module
evictor = _module
placeholder_block_space_manager = _module
scheduler = _module
communication_op = _module
device_communicators = _module
cuda_wrapper = _module
custom_all_reduce = _module
custom_all_reduce_utils = _module
hpu_communicator = _module
pynccl = _module
pynccl_wrapper = _module
shm_broadcast = _module
tpu_communicator = _module
parallel_state = _module
utils = _module
arg_utils = _module
async_llm_engine = _module
async_timeout = _module
llm_engine = _module
metrics_types = _module
multiprocessing = _module
client = _module
single_step = _module
stop_checker = _module
util = _module
protocol = _module
api_server = _module
chat_utils = _module
launcher = _module
logger = _module
cli_args = _module
logits_processors = _module
protocol = _module
run_batch = _module
serving_chat = _module
serving_completion = _module
serving_embedding = _module
serving_engine = _module
serving_tokenization = _module
tool_parsers = _module
abstract_tool_parser = _module
granite_20b_fc_tool_parser = _module
hermes_tool_parser = _module
internlm2_tool_parser = _module
jamba_tool_parser = _module
llama_tool_parser = _module
mistral_tool_parser = _module
envs = _module
executor = _module
cpu_executor = _module
distributed_gpu_executor = _module
executor_base = _module
gpu_executor = _module
hpu_executor = _module
msgspec_utils = _module
multiproc_gpu_executor = _module
multiproc_worker_utils = _module
multiproc_xpu_executor = _module
neuron_executor = _module
openvino_executor = _module
ray_gpu_executor = _module
ray_hpu_executor = _module
ray_tpu_executor = _module
ray_utils = _module
ray_xpu_executor = _module
tpu_executor = _module
xpu_executor = _module
forward_context = _module
inputs = _module
parse = _module
preprocess = _module
registry = _module
logging = _module
formatter = _module
logits_process = _module
fully_sharded_layers = _module
layers = _module
lora = _module
models = _module
bgmv_expand = _module
bgmv_expand_slice = _module
bgmv_shrink = _module
sgmv_expand = _module
sgmv_expand_slice = _module
sgmv_shrink = _module
punica = _module
utils = _module
worker_manager = _module
custom_op = _module
guided_decoding = _module
guided_fields = _module
lm_format_enforcer_decoding = _module
outlines_decoding = _module
outlines_logits_processors = _module
activation = _module
fused_moe = _module
fused_marlin_moe = _module
fused_moe = _module
layer = _module
moe_pallas = _module
layernorm = _module
linear = _module
logits_processor = _module
mamba = _module
mamba_mixer = _module
causal_conv1d = _module
mamba_ssm = _module
pooler = _module
aqlm = _module
awq = _module
awq_marlin = _module
awq_triton = _module
base_config = _module
bitsandbytes = _module
compressed_tensors = _module
compressed_tensors = _module
compressed_tensors_moe = _module
schemes = _module
compressed_tensors_scheme = _module
compressed_tensors_w4a16_24 = _module
compressed_tensors_w8a16_fp8 = _module
compressed_tensors_w8a8_fp8 = _module
compressed_tensors_w8a8_int8 = _module
compressed_tensors_wNa16 = _module
utils = _module
deepspeedfp = _module
experts_int8 = _module
fbgemm_fp8 = _module
fp8 = _module
gguf = _module
gptq = _module
gptq_marlin = _module
gptq_marlin_24 = _module
ipex_quant = _module
MPLinearKernel = _module
exllama = _module
machete = _module
marlin = _module
kv_cache = _module
marlin = _module
modelopt = _module
neuron_quant = _module
qqq = _module
schema = _module
tpu_int8 = _module
layer_utils = _module
machete_utils = _module
marlin_utils = _module
marlin_utils_fp8 = _module
marlin_utils_test = _module
marlin_utils_test_24 = _module
marlin_utils_test_qqq = _module
quant_utils = _module
w8a8_utils = _module
rejection_sampler = _module
resampler = _module
rotary_embedding = _module
sampler = _module
spec_decode_base_sampler = _module
typical_acceptance_sampler = _module
vocab_parallel_embedding = _module
model_loader = _module
loader = _module
neuron = _module
openvino = _module
tensorizer = _module
utils = _module
weight_utils = _module
arctic = _module
baichuan = _module
bart = _module
bert = _module
blip = _module
blip2 = _module
bloom = _module
chameleon = _module
chatglm = _module
clip = _module
commandr = _module
dbrx = _module
decilm = _module
deepseek = _module
deepseek_v2 = _module
eagle = _module
exaone = _module
falcon = _module
florence2 = _module
fuyu = _module
gemma = _module
gemma2 = _module
glm4_vision_encoder = _module
gpt2 = _module
gpt_bigcode = _module
gpt_j = _module
gpt_neox = _module
granite = _module
granitemoe = _module
h2ovl = _module
idefics2_vision_model = _module
idefics3 = _module
interfaces = _module
interfaces_base = _module
intern_vit = _module
internlm2 = _module
internlm2_ve = _module
internvl = _module
jais = _module
jamba = _module
llama = _module
llava = _module
llava_next = _module
llava_next_video = _module
llava_onevision = _module
mamba = _module
mamba_cache = _module
medusa = _module
minicpm = _module
minicpm3 = _module
minicpmv = _module
mixtral = _module
mixtral_quant = _module
mllama = _module
mlp_speculator = _module
module_mapping = _module
molmo = _module
mpt = _module
nemotron = _module
nvlm_d = _module
olmo = _module
olmoe = _module
opt = _module
orion = _module
paligemma = _module
persimmon = _module
phi = _module
phi3 = _module
phi3_small = _module
phi3v = _module
phimoe = _module
pixtral = _module
qwen = _module
qwen2 = _module
qwen2_audio = _module
qwen2_cls = _module
qwen2_moe = _module
qwen2_rm = _module
qwen2_vl = _module
registry = _module
siglip = _module
solar = _module
stablelm = _module
starcoder2 = _module
ultravox = _module
utils = _module
xverse = _module
parameter = _module
pooling_metadata = _module
sampling_metadata = _module
utils = _module
base = _module
image = _module
video = _module
outputs = _module
platforms = _module
cpu = _module
cuda = _module
hpu = _module
interface = _module
openvino = _module
rocm = _module
tpu = _module
xpu = _module
plugins = _module
pooling_params = _module
profiler = _module
layerwise_profile = _module
utils = _module
prompt_adapter = _module
layers = _module
models = _module
utils = _module
worker_manager = _module
sampling_params = _module
scalar_type = _module
scripts = _module
sequence = _module
batch_expansion = _module
draft_model_runner = _module
interfaces = _module
medusa_worker = _module
metrics = _module
mlp_speculator_worker = _module
mqa_scorer = _module
multi_step_worker = _module
ngram_worker = _module
proposer_worker_base = _module
smaller_tp_proposer_worker = _module
spec_decode_worker = _module
target_model_runner = _module
top1_proposer = _module
util = _module
transformers_utils = _module
configs = _module
mpt = _module
detokenizer = _module
detokenizer_utils = _module
processor = _module
tokenizer = _module
tokenizer_group = _module
base_tokenizer_group = _module
ray_tokenizer_group = _module
tokenizers = _module
mistral = _module
triton_utils = _module
custom_cache_manager = _module
importing = _module
libentry = _module
usage = _module
usage_lib = _module
utils = _module
flash_attn = _module
kv_cache_manager = _module
outputs = _module
sample = _module
metadata = _module
sampler = _module
gpu_model_runner = _module
gpu_worker = _module
version = _module
cache_engine = _module
cpu_enc_dec_model_runner = _module
cpu_model_runner = _module
cpu_worker = _module
embedding_model_runner = _module
enc_dec_model_runner = _module
hpu_model_runner = _module
hpu_worker = _module
model_runner = _module
model_runner_base = _module
multi_step_model_runner = _module
multi_step_tpu_worker = _module
multi_step_worker = _module
neuron_model_runner = _module
neuron_worker = _module
openvino_model_runner = _module
openvino_worker = _module
tpu_model_runner = _module
tpu_worker = _module
worker = _module
worker_base = _module
xpu_model_runner = _module
xpu_worker = _module

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


from typing import List


from typing import Optional


import numpy as np


import torch


import random


import copy


import itertools


from typing import Callable


from typing import Iterable


from typing import Tuple


import torch.utils.benchmark as TBenchmark


from torch.utils.benchmark import Measurement as TMeasurement


import torch.nn.functional as F


import math


from itertools import product


import pandas as pd


import torch.utils.benchmark as benchmark


from typing import Any


from typing import Dict


from typing import TypedDict


from itertools import accumulate


import re


from collections import defaultdict


import matplotlib.pyplot as plt


from torch.utils.hipify.hipify_python import hipify


from collections import namedtuple


from collections.abc import Iterable


from typing import Union


import logging


from torch.utils.data import DataLoader


import inspect


from torch.utils.cpp_extension import CUDA_HOME


import uuid


from copy import copy


from torch import nn


from torch.library import Library


from collections import UserList


from enum import Enum


from typing import Type


from typing import TypeVar


import torch.nn as nn


from collections import deque


from typing import Set


from torch import Use


import torch.distributed as dist


import torch.distributed


from typing import NamedTuple


from numbers import Number


from typing import Sequence


from torch._prims_common import TensorLikeType


from collections import OrderedDict


from copy import deepcopy


import types


import warnings


import torch.cuda


from typing import Mapping


from types import SimpleNamespace


from itertools import count


from typing import Sequence as GenericSequence


import functools


from typing import TYPE_CHECKING


import torch.library


from abc import ABC


from abc import abstractmethod


from typing import Hashable


from typing import Literal


from enum import auto


from typing import Generic


from torch.nn.functional import scaled_dot_product_attention


from functools import lru_cache


import enum


from typing import Generator


import torch.fx as fx


from types import CodeType


from typing import ClassVar


from typing import Final


from torch.distributed import ProcessGroup


import torch.multiprocessing as mp


from torch.distributed import ReduceOp


from torch.distributed import Backend


from torch.distributed.distributed_c10d import Backend


from torch.distributed.distributed_c10d import PrefixStore


from torch.distributed.distributed_c10d import _get_default_timeout


from torch.distributed.distributed_c10d import is_nccl_available


from torch.distributed.rendezvous import rendezvous


from typing import cast


from typing import get_args


from functools import partial


from typing import AsyncGenerator


from typing import Coroutine


from typing import overload


from collections import Counter as collectionsCounter


from typing import Deque


from typing import FrozenSet


from typing import Awaitable


from collections import UserDict


from typing import Protocol


import torch.types


from typing import DefaultDict


from torch.nn.parameter import Parameter


from torch.nn.parameter import UninitializedParameter


from enum import IntEnum


from torch.nn import Parameter


from torch.nn import Module


import numpy


from functools import cached_property


import torch.jit


from torch.nn.init import trunc_normal_


from math import inf


from typing import Iterator


import collections


from typing import BinaryIO


from torch.nn import LayerNorm


import torch.utils.checkpoint


from typing import runtime_checkable


import torchvision.transforms as T


from torch.nn import functional as F


from itertools import tee


from torchvision import transforms


from torchvision.transforms import InterpolationMode


from torch.func import functional_call


from typing import final


from functools import wraps


from torch._C._autograd import DeviceType


from torch._C._autograd import _KinetoEvent


from torch._C._autograd import _ProfilerResult


from torch._C._profiler import _EventType


from torch._C._profiler import _ExperimentalConfig


from torch._C._profiler import _ProfilerEvent


from torch.autograd.profiler import FunctionEvent


from torch.profiler import ProfilerActivity


from torch.profiler import profile


from torch._C._profiler import _TensorMetadata


from functools import reduce


from itertools import chain


from uuid import uuid4


from collections.abc import Mapping


from typing import OrderedDict


import numpy.typing as npt


from torch import is_tensor


class CompilationLevel:
    NO_COMPILATION = 0
    DYNAMO_AS_IS = 1
    DYNAMO_ONCE = 2
    PIECEWISE = 3


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_torch_compile_backend() ->Optional[Union[Callable, str]]:
    return _torch_compile_backend


def combine_fx_passes(passes: 'List[Callable]') ->Callable:

    def combined_fx(graph) ->None:
        for fx in passes:
            fx(graph)
    return combined_fx


def fix_functionalization(graph: 'fx.Graph'):
    """
    Rewrite the graph module to replace the pattern involving
    torch._higher_order_ops.auto_functionalize.auto_functionalized
    with a direct call to the inplace custom op.

    # TODO: check if PyTorch nightly has fixed this issue
    """
    nodes_to_remove = []
    for node in graph.nodes:
        if node.op == 'call_function' and node.target == torch._higher_order_ops.auto_functionalize.auto_functionalized:
            if node.args[0] == torch.ops._C.rotary_embedding.default:
                kwargs = node.kwargs
                query = kwargs['query']
                mm_node = query.args[0].args[0]
                with graph.inserting_before(node):
                    graph.call_function(torch.ops._C.rotary_embedding.default, kwargs=kwargs)
                for user in list(node.users):
                    if user.op == 'call_function' and user.target == operator.getitem:
                        for getitem_user in list(user.users):
                            if getitem_user.op == 'call_function' and getitem_user.target == torch.ops.aten.slice_scatter.default:
                                getitem_user.replace_all_uses_with(mm_node)
                                nodes_to_remove.append(getitem_user)
                        nodes_to_remove.append(user)
                nodes_to_remove.append(node)
            elif node.args[0] == torch.ops._C.fused_add_rms_norm.default:
                kwargs = node.kwargs
                input = kwargs['input']
                residual = kwargs['residual']
                with graph.inserting_before(node):
                    graph.call_function(torch.ops._C.fused_add_rms_norm.default, kwargs=kwargs)
                for user in list(node.users):
                    if user.op == 'call_function' and user.target == operator.getitem:
                        if user.args[1] == 1:
                            replace_node = input
                        elif user.args[1] == 2:
                            replace_node = residual
                        user.replace_all_uses_with(replace_node)
                        nodes_to_remove.append(user)
                nodes_to_remove.append(node)
            elif node.args[0] == torch.ops._C.rms_norm.default:
                kwargs = node.kwargs
                input = kwargs['input']
                out = kwargs['out']
                weight = kwargs['weight']
                epsilon = kwargs['epsilon']
                with graph.inserting_before(node):
                    graph.call_function(torch.ops._C.rms_norm.default, args=(out, input, weight, epsilon))
                replace_node = out
                for user in list(node.users):
                    if user.op == 'call_function' and user.target == operator.getitem:
                        user.replace_all_uses_with(replace_node)
                        nodes_to_remove.append(user)
                nodes_to_remove.append(node)
            elif node.args[0] == torch.ops._C.silu_and_mul.default:
                kwargs = node.kwargs
                input = kwargs['input']
                out = kwargs['out']
                with graph.inserting_before(node):
                    graph.call_function(torch.ops._C.silu_and_mul.default, args=(out, input))
                replace_node = out
                for user in list(node.users):
                    if user.op == 'call_function' and user.target == operator.getitem:
                        user.replace_all_uses_with(replace_node)
                        nodes_to_remove.append(user)
                nodes_to_remove.append(node)
    for node in nodes_to_remove:
        graph.erase_node(node)

