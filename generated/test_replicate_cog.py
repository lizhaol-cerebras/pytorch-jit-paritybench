import sys
_module = sys.modules[__name__]
del sys
predict = _module
cog = _module
base_input = _module
base_predictor = _module
code_xforms = _module
command = _module
ast_openapi_schema = _module
openapi_schema = _module
config = _module
errors = _module
files = _module
json = _module
logging = _module
mode = _module
predictor = _module
schema = _module
server = _module
connection = _module
eventtypes = _module
exceptions = _module
helpers = _module
http = _module
probes = _module
response_throttler = _module
runner = _module
telemetry = _module
useragent = _module
webhook = _module
worker = _module
suppress_output = _module
types = _module
test_files = _module
conftest = _module
argv_override = _module
async_setup_uses_same_loop_as_predict = _module
catch_in_predict = _module
complex_output = _module
count_up = _module
exc_in_predict = _module
exc_in_setup = _module
exc_in_setup_and_predict = _module
exc_on_import = _module
exit_in_predict = _module
exit_in_setup = _module
exit_on_import = _module
function = _module
hello_world = _module
hello_world_async = _module
input_choices = _module
input_choices_integer = _module
input_choices_iterable = _module
input_file = _module
input_ge_le = _module
input_integer = _module
input_integer_default = _module
input_literal = _module
input_literal_integer = _module
input_multiple = _module
input_none = _module
input_path = _module
input_path_2 = _module
input_secret = _module
input_string = _module
input_union_integer_or_list_of_integers = _module
input_union_string_or_list_of_strings = _module
input_unsupported_type = _module
input_untyped = _module
killed_in_predict = _module
logging_async = _module
missing_predict = _module
missing_predictor = _module
openapi_complex_input = _module
openapi_custom_output_type = _module
openapi_input_int_choices = _module
openapi_output_list = _module
openapi_output_type = _module
openapi_output_yield = _module
output_complex = _module
output_file = _module
output_file_named = _module
output_iterator_complex = _module
output_numpy = _module
output_path_image = _module
output_path_text = _module
output_wrong_type = _module
setup = _module
setup_uses_async = _module
setup_weights = _module
simple = _module
sleep = _module
sleep_async = _module
slow_predict = _module
slow_setup = _module
steps = _module
stream_redirector_race_condition = _module
train = _module
yield_concatenate_iterator = _module
yield_files = _module
yield_strings = _module
yield_strings_file_input = _module
test_helpers = _module
test_http = _module
test_http_input = _module
test_http_output = _module
test_predictor = _module
test_probes = _module
test_response_throttler = _module
test_runner = _module
test_webhook = _module
test_worker = _module
test_code_xforms = _module
test_config = _module
test_json = _module
test_types = _module
test_integration = _module
pong = _module
forker = _module
mylib = _module
test_build = _module
test_predict = _module
test_run = _module
test_train = _module
util = _module

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


import uuid

