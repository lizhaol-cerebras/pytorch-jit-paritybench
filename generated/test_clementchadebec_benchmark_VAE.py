
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


from typing import List


import numpy as np


import torch


import torch.nn as nn


from torchvision.datasets import MNIST


from torchvision.datasets import CelebA


from torchvision.datasets import CIFAR10


from torchvision.transforms import PILToTensor


import logging


import time


from torch.utils.data import Dataset


from torchvision import transforms


import math


import torch.nn.functional as F


from time import time


from collections import OrderedDict


from typing import Any


from typing import Tuple


from torch.utils.data._utils.collate import default_collate


from typing import Optional


from typing import Union


import inspect


import warnings


from copy import deepcopy


from typing import Dict


import torch.distributed as dist


from math import sqrt


from torch.autograd import grad


from numbers import Number


import torch.distributions as dist


from torch.autograd import Function


from torch.distributions import Normal


from torch.distributions.utils import _standard_normal


from torch.distributions.utils import broadcast_all


from torch.nn import functional as F


from collections import deque


import scipy.special


from sklearn import mixture


from torch.utils.data import DataLoader


from torch.distributions import MultivariateNormal


import itertools


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data.distributed import DistributedSampler


import random


import typing


from torch.optim import Adam


from collections import Counter


class RiemannianLayer(nn.Module):

    def __init__(self, in_features, out_features, manifold, over_param, weight_norm):
        super(RiemannianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self._weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.over_param = over_param
        self.weight_norm = weight_norm
        self._bias = nn.Parameter(torch.Tensor(out_features, 1))
        self.reset_parameters()

    @property
    def weight(self):
        return self.manifold.transp0(self.bias, self._weight)

    @property
    def bias(self):
        if self.over_param:
            return self._bias
        else:
            return self.manifold.expmap0(self._weight * self._bias)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self._weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._weight)
        bound = 4 / math.sqrt(fan_in)
        nn.init.uniform_(self._bias, -bound, bound)
        if self.over_param:
            with torch.no_grad():
                self._bias.set_(self.manifold.expmap0(self._bias))


class GeodesicLayer(RiemannianLayer):

    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(GeodesicLayer, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, input):
        input = input.unsqueeze(0)
        input = input.unsqueeze(-2).expand(*input.shape[:-(len(input.shape) - 2)], self.out_features, self.in_features)
        res = self.manifold.normdist2plane(input, self.bias, self.weight, signed=True, norm=self.weight_norm)
        return res


class NonLinear(nn.Module):

    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()
        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        return h


class GatedDense(nn.Module):

    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))
        return h * g


class BadInheritanceError(Exception):
    pass


class BaseDecoder(nn.Module):
    """This is a base class for Decoders neural networks."""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, z: 'torch.Tensor'):
        """This function must be implemented in a child class.
        It takes the input data and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own decoder network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseDecoder
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_decoder(BaseDecoder):
            ...
            ...    def __init__(self):
            ...        BaseDecoder.__init__(self)
            ...        # your code
            ...
            ...    def forward(self, z: torch.Tensor):
            ...        # your code
            ...        output = ModelOutput(
            ...             reconstruction=reconstruction
            ...         )
            ...        return output

        Parameters:
            z (torch.Tensor): The latent data that must be decoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the decoder

        .. note::

            By convention, the reconstruction tensors should be in [0, 1] and of shape
            BATCH x channels x ...

        """
        raise NotImplementedError()


class BaseEncoder(nn.Module):
    """This is a base class for Encoders neural networks."""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        """This function must be implemented in a child class.
        It takes the input data and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own encoder network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseEncoder
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_Encoder(BaseEncoder):
            ...
            ...     def __init__(self):
            ...         BaseEncoder.__init__(self)
            ...         # your code
            ...
            ...     def forward(self, x: torch.Tensor):
            ...         # your code
            ...         output = ModelOutput(
            ...             embedding=embedding,
            ...             log_covariance=log_var # for VAE based models
            ...         )
            ...         return output

        Parameters:
            x (torch.Tensor): The input data that must be encoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the encoder
        """
        raise NotImplementedError()


class DatasetOutput(OrderedDict):
    """Base DatasetOutput class fixing the output type from the dataset. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for k, v in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) ->Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class ModelOutput(OrderedDict):
    """Base ModelOutput class fixing the output type from the models. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for k, v in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) ->Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class Decoder_AE_MLP(BaseDecoder):

    def __init__(self, args: 'dict'):
        BaseDecoder.__init__(self)
        self.input_dim = args.input_dim
        layers = nn.ModuleList()
        layers.append(nn.Sequential(nn.Linear(args.latent_dim, 512), nn.ReLU()))
        layers.append(nn.Sequential(nn.Linear(512, int(np.prod(args.input_dim))), nn.Sigmoid()))
        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: 'torch.Tensor', output_layer_levels: 'List[int]'=None):
        output = ModelOutput()
        max_depth = self.depth
        if output_layer_levels is not None:
            assert all(self.depth >= levels > 0 or levels == -1 for levels in output_layer_levels), f'Cannot output layer deeper than depth ({self.depth}). Got ({output_layer_levels}).'
            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)
        out = z
        for i in range(max_depth):
            out = self.layers[i](out)
            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f'reconstruction_layer_{i + 1}'] = out
            if i + 1 == self.depth:
                output['reconstruction'] = out.reshape((z.shape[0],) + self.input_dim)
        return output


class LoadError(Exception):
    pass


def hf_hub_is_available():
    return importlib.util.find_spec('huggingface_hub') is not None


logger = logging.getLogger(__name__)


model_card_template = """---
language: en
tags:
- pythae
license: apache-2.0
---

### Downloading this model from the Hub
This model was trained with pythae. It can be downloaded or reloaded using the method `load_from_hf_hub`
```python
>>> from pythae.models import AutoModel
>>> model = AutoModel.load_from_hf_hub(hf_hub_path="your_hf_username/repo_name")
```
"""


class BaseAE(nn.Module):
    """Base class for Autoencoder based models.

    Args:
        model_config (BaseAEConfig): An instance of BaseAEConfig in which any model's parameters is
            made available.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'BaseAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        nn.Module.__init__(self)
        self.model_name = 'BaseAE'
        self.input_dim = model_config.input_dim
        self.latent_dim = model_config.latent_dim
        self.model_config = model_config
        if decoder is None:
            if model_config.input_dim is None:
                raise AttributeError("No input dimension provided !'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' where the shape of the data is (C, H, W ..)]. Unable to build decoderautomatically")
            decoder = Decoder_AE_MLP(model_config)
            self.model_config.uses_default_decoder = True
        else:
            self.model_config.uses_default_decoder = False
        self.set_decoder(decoder)
        self.device = None

    def forward(self, inputs: 'BaseDataset', **kwargs) ->ModelOutput:
        """Main forward pass outputing the VAE outputs
        This function should output a :class:`~pythae.models.base.base_utils.ModelOutput` instance
        gathering all the model outputs

        Args:
            inputs (BaseDataset): The training data with labels, masks etc...

        Returns:
            ModelOutput: A ModelOutput instance providing the outputs of the model.

        .. note::
            The loss must be computed in this forward pass and accessed through
            ``loss = model_output.loss``"""
        raise NotImplementedError()

    def reconstruct(self, inputs: 'torch.Tensor'):
        """This function returns the reconstructions of given input data.

        Args:
            inputs (torch.Tensor): The inputs data to be reconstructed of shape [B x input_dim]
            ending_inputs (torch.Tensor): The starting inputs in the interpolation of shape

        Returns:
            torch.Tensor: A tensor of shape [B x input_dim] containing the reconstructed samples.
        """
        return self(DatasetOutput(data=inputs)).recon_x

    def embed(self, inputs: 'torch.Tensor') ->torch.Tensor:
        """Return the embeddings of the input data.

        Args:
            inputs (torch.Tensor): The input data to be embedded, of shape [B x input_dim].

        Returns:
            torch.Tensor: A tensor of shape [B x latent_dim] containing the embeddings.
        """
        return self(DatasetOutput(data=inputs)).z

    def predict(self, inputs: 'torch.Tensor') ->ModelOutput:
        """The input data is encoded and decoded without computing loss

        Args:
            inputs (torch.Tensor): The input data to be reconstructed, as well as to generate the embedding.

        Returns:
            ModelOutput: An instance of ModelOutput containing reconstruction and embedding
        """
        z = self.encoder(inputs).embedding
        recon_x = self.decoder(z)['reconstruction']
        output = ModelOutput(recon_x=recon_x, embedding=z)
        return output

    def interpolate(self, starting_inputs: 'torch.Tensor', ending_inputs: 'torch.Tensor', granularity: 'int'=10):
        """This function performs a linear interpolation in the latent space of the autoencoder
        from starting inputs to ending inputs. It returns the interpolation trajectories.

        Args:
            starting_inputs (torch.Tensor): The starting inputs in the interpolation of shape
                [B x input_dim]
            ending_inputs (torch.Tensor): The starting inputs in the interpolation of shape
                [B x input_dim]
            granularity (int): The granularity of the interpolation.

        Returns:
            torch.Tensor: A tensor of shape [B x granularity x input_dim] containing the
            interpolation trajectories.
        """
        assert starting_inputs.shape[0] == ending_inputs.shape[0], f'The number of starting_inputs should equal the number of ending_inputs. Got {starting_inputs.shape[0]} sampler for starting_inputs and {ending_inputs.shape[0]} for endinging_inputs.'
        starting_z = self(DatasetOutput(data=starting_inputs)).z
        ending_z = self(DatasetOutput(data=ending_inputs)).z
        t = torch.linspace(0, 1, granularity)
        intep_line = (torch.kron(starting_z.reshape(starting_z.shape[0], -1), (1 - t).unsqueeze(-1)) + torch.kron(ending_z.reshape(ending_z.shape[0], -1), t.unsqueeze(-1))).reshape((starting_z.shape[0] * t.shape[0],) + starting_z.shape[1:])
        decoded_line = self.decoder(intep_line).reconstruction.reshape((starting_inputs.shape[0], t.shape[0]) + starting_inputs.shape[1:])
        return decoded_line

    def update(self):
        """Method that allows model update during the training (at the end of a training epoch)

        If needed, this method must be implemented in a child class.

        By default, it does nothing.
        """

    def save(self, dir_path: 'str'):
        """Method to save the model at a specific location. It saves, the model weights as a
        ``models.pt`` file along with the model config as a ``model_config.json`` file. If the
        model to save used custom encoder (resp. decoder) provided by the user, these are also
        saved as ``decoder.pkl`` (resp. ``decoder.pkl``).

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """
        env_spec = EnvironmentConfig(python_version=f'{sys.version_info[0]}.{sys.version_info[1]}')
        model_dict = {'model_state_dict': deepcopy(self.state_dict())}
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except FileNotFoundError as e:
                raise e
        env_spec.save_json(dir_path, 'environment')
        self.model_config.save_json(dir_path, 'model_config')
        if not self.model_config.uses_default_encoder:
            with open(os.path.join(dir_path, 'encoder.pkl'), 'wb') as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.encoder))
                cloudpickle.dump(self.encoder, fp)
        if not self.model_config.uses_default_decoder:
            with open(os.path.join(dir_path, 'decoder.pkl'), 'wb') as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.decoder))
                cloudpickle.dump(self.decoder, fp)
        torch.save(model_dict, os.path.join(dir_path, 'model.pt'))

    def push_to_hf_hub(self, hf_hub_path: 'str'):
        """Method allowing to save your model directly on the Hugging Face hub.
        You will need to have the `huggingface_hub` package installed and a valid Hugging Face
        account. You can install the package using

        .. code-block:: bash

            python -m pip install huggingface_hub

        end then login using

        .. code-block:: bash

            huggingface-cli login

        Args:
            hf_hub_path (str): path to your repo on the Hugging Face hub.
        """
        if not hf_hub_is_available():
            raise ModuleNotFoundError('`huggingface_hub` package must be installed to push your model to the HF hub. Run `python -m pip install huggingface_hub` and log in to your account with `huggingface-cli login`.')
        logger.info(f'Uploading {self.model_name} model to {hf_hub_path} repo in HF hub...')
        tempdir = tempfile.mkdtemp()
        self.save(tempdir)
        model_files = os.listdir(tempdir)
        api = HfApi()
        hf_operations = []
        for file in model_files:
            hf_operations.append(CommitOperationAdd(path_in_repo=file, path_or_fileobj=f'{str(os.path.join(tempdir, file))}'))
        with open(os.path.join(tempdir, 'model_card.md'), 'w') as f:
            f.write(model_card_template)
        hf_operations.append(CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=os.path.join(tempdir, 'model_card.md')))
        try:
            api.create_commit(commit_message=f'Uploading {self.model_name} in {hf_hub_path}', repo_id=hf_hub_path, operations=hf_operations)
            logger.info(f'Successfully uploaded {self.model_name} to {hf_hub_path} repo in HF hub!')
        except:
            repo_name = os.path.basename(os.path.normpath(hf_hub_path))
            logger.info(f'Creating {repo_name} in the HF hub since it does not exist...')
            create_repo(repo_id=repo_name)
            logger.info(f'Successfully created {repo_name} in the HF hub!')
            api.create_commit(commit_message=f'Uploading {self.model_name} in {hf_hub_path}', repo_id=hf_hub_path, operations=hf_operations)
        shutil.rmtree(tempdir)

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)
        if 'model_config.json' not in file_list:
            raise FileNotFoundError(f"Missing model config file ('model_config.json') in{dir_path}... Cannot perform model building.")
        path_to_model_config = os.path.join(dir_path, 'model_config.json')
        model_config = AutoConfig.from_json_file(path_to_model_config)
        return model_config

    @classmethod
    def _load_model_weights_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)
        if 'model.pt' not in file_list:
            raise FileNotFoundError(f"Missing model weights file ('model.pt') file in{dir_path}... Cannot perform model building.")
        path_to_model_weights = os.path.join(dir_path, 'model.pt')
        try:
            model_weights = torch.load(path_to_model_weights, map_location='cpu')
        except RuntimeError:
            RuntimeError("Enable to load model weights. Ensure they are saves in a '.pt' format.")
        if 'model_state_dict' not in model_weights.keys():
            raise KeyError(f"Model state dict is not available in 'model.pt' file. Got keys:{model_weights.keys()}")
        model_weights = model_weights['model_state_dict']
        return model_weights

    @classmethod
    def _load_custom_encoder_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)
        if 'encoder.pkl' not in file_list:
            raise FileNotFoundError(f"Missing encoder pkl file ('encoder.pkl') in{dir_path}... This file is needed to rebuild custom encoders. Cannot perform model building.")
        else:
            with open(os.path.join(dir_path, 'encoder.pkl'), 'rb') as fp:
                encoder = CPU_Unpickler(fp).load()
        return encoder

    @classmethod
    def _load_custom_decoder_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)
        if 'decoder.pkl' not in file_list:
            raise FileNotFoundError(f"Missing decoder pkl file ('decoder.pkl') in{dir_path}... This file is needed to rebuild custom decoders. Cannot perform model building.")
        else:
            with open(os.path.join(dir_path, 'decoder.pkl'), 'rb') as fp:
                decoder = CPU_Unpickler(fp).load()
        return decoder

    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """
        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)
        if not model_config.uses_default_encoder:
            encoder = cls._load_custom_encoder_from_folder(dir_path)
        else:
            encoder = None
        if not model_config.uses_default_decoder:
            decoder = cls._load_custom_decoder_from_folder(dir_path)
        else:
            decoder = None
        model = cls(model_config, encoder=encoder, decoder=decoder)
        model.load_state_dict(model_weights)
        return model

    @classmethod
    def load_from_hf_hub(cls, hf_hub_path: 'str', allow_pickle=False):
        """Class method to be used to load a pretrained model from the Hugging Face hub

        Args:
            hf_hub_path (str): The path where the model should have been be saved on the
                hugginface hub.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """
        if not hf_hub_is_available():
            raise ModuleNotFoundError('`huggingface_hub` package must be installed to load models from the HF hub. Run `python -m pip install huggingface_hub` and log in to your account with `huggingface-cli login`.')
        logger.info(f'Downloading {cls.__name__} files for rebuilding...')
        _ = hf_hub_download(repo_id=hf_hub_path, filename='environment.json')
        config_path = hf_hub_download(repo_id=hf_hub_path, filename='model_config.json')
        dir_path = os.path.dirname(config_path)
        _ = hf_hub_download(repo_id=hf_hub_path, filename='model.pt')
        model_config = cls._load_model_config_from_folder(dir_path)
        if cls.__name__ + 'Config' != model_config.name and cls.__name__ + '_Config' != model_config.name:
            warnings.warn(f'You are trying to load a `{cls.__name__}` while a `{model_config.name}` is given.')
        model_weights = cls._load_model_weights_from_folder(dir_path)
        if (not model_config.uses_default_encoder or not model_config.uses_default_decoder) and not allow_pickle:
            warnings.warn('You are about to download pickled files from the HF hub that may have been created by a third party and so could potentially harm your computer. If you are sure that you want to download them set `allow_pickle=true`.')
        else:
            if not model_config.uses_default_encoder:
                _ = hf_hub_download(repo_id=hf_hub_path, filename='encoder.pkl')
                encoder = cls._load_custom_encoder_from_folder(dir_path)
            else:
                encoder = None
            if not model_config.uses_default_decoder:
                _ = hf_hub_download(repo_id=hf_hub_path, filename='decoder.pkl')
                decoder = cls._load_custom_decoder_from_folder(dir_path)
            else:
                decoder = None
            logger.info(f'Successfully downloaded {cls.__name__} model!')
            model = cls(model_config, encoder=encoder, decoder=decoder)
            model.load_state_dict(model_weights)
            return model

    def set_encoder(self, encoder: 'BaseEncoder') ->None:
        """Set the encoder of the model"""
        if not issubclass(type(encoder), BaseEncoder):
            raise BadInheritanceError('Encoder must inherit from BaseEncoder class from pythae.models.base_architectures.BaseEncoder. Refer to documentation.')
        self.encoder = encoder

    def set_decoder(self, decoder: 'BaseDecoder') ->None:
        """Set the decoder of the model"""
        if not issubclass(type(decoder), BaseDecoder):
            raise BadInheritanceError('Decoder must inherit from BaseDecoder class from pythae.models.base_architectures.BaseDecoder. Refer to documentation.')
        self.decoder = decoder

    @classmethod
    def _check_python_version_from_folder(cls, dir_path: 'str'):
        if 'environment.json' in os.listdir(dir_path):
            env_spec = EnvironmentConfig.from_json_file(os.path.join(dir_path, 'environment.json'))
            python_version = env_spec.python_version
            python_version_minor = python_version.split('.')[1]
            if python_version_minor == '7' and sys.version_info[1] > 7:
                raise LoadError('Trying to reload a model saved with python3.7 with python3.8+. Please create a virtual env with python 3.7 to reload this model.')
            elif int(python_version_minor) >= 8 and sys.version_info[1] == 7:
                raise LoadError('Trying to reload a model saved with python3.8+ with python3.7. Please create a virtual env with python 3.8+ to reload this model.')


class Encoder_AE_MLP(BaseEncoder):

    def __init__(self, args: 'dict'):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        layers = nn.ModuleList()
        layers.append(nn.Sequential(nn.Linear(np.prod(args.input_dim), 512), nn.ReLU()))
        self.layers = layers
        self.depth = len(layers)
        self.embedding = nn.Linear(512, self.latent_dim)

    def forward(self, x, output_layer_levels: 'List[int]'=None):
        output = ModelOutput()
        max_depth = self.depth
        if output_layer_levels is not None:
            assert all(self.depth >= levels > 0 or levels == -1 for levels in output_layer_levels), f'Cannot output layer deeper than depth ({self.depth}). Got ({output_layer_levels}).'
            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)
        out = x.reshape(-1, np.prod(self.input_dim))
        for i in range(max_depth):
            out = self.layers[i](out)
            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f'embedding_layer_{i + 1}'] = out
            if i + 1 == self.depth:
                output['embedding'] = self.embedding(out)
        return output


class AE(BaseAE):
    """Vanilla Autoencoder model.

    Args:
        model_config (AEConfig): The Autoencoder configuration setting the main parameters of the
            model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'AEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        BaseAE.__init__(self, model_config=model_config, decoder=decoder)
        self.model_name = 'AE'
        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError("No input dimension provided !'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' where the shape of the data is (C, H, W ..). Unable to build encoder automatically")
            encoder = Encoder_AE_MLP(model_config)
            self.model_config.uses_default_encoder = True
        else:
            self.model_config.uses_default_encoder = False
        self.set_encoder(encoder)

    def forward(self, inputs: 'BaseDataset', **kwargs) ->ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        x = inputs['data']
        z = self.encoder(x).embedding
        recon_x = self.decoder(z)['reconstruction']
        loss = self.loss_function(recon_x, x)
        output = ModelOutput(loss=loss, recon_x=recon_x, z=z)
        return output

    def loss_function(self, recon_x, x):
        MSE = F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        return MSE.mean(dim=0)


class BaseDiscriminator(nn.Module):
    """This is a base class for Discriminator neural networks."""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        """This function must be implemented in a child class.
        It takes the input data and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own disctriminator network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseDiscriminator
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_Discriminator(BaseDiscriminator):
            ...
            ...     def __init__(self):
            ...         BaseDiscriminator.__init__(self)
            ...         # your code
            ...
            ...     def forward(self, x: torch.Tensor):
            ...         # your code
            ...         output = ModelOutput(
            ...             adversarial_cost=adversarial_cost
            ...         )
            ...         return output

        Parameters:
            x (torch.Tensor): The input data that must be encoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the encoder
        """
        raise NotImplementedError()


class Discriminator_MLP(BaseDiscriminator):

    def __init__(self, args: 'dict'):
        BaseDiscriminator.__init__(self)
        self.discriminator_input_dim = args.discriminator_input_dim
        layers = nn.ModuleList()
        layers.append(nn.Sequential(nn.Linear(np.prod(args.discriminator_input_dim), 256), nn.ReLU()))
        layers.append(nn.Sequential(nn.Linear(256, 1), nn.Sigmoid()))
        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: 'torch.Tensor', output_layer_levels: 'List[int]'=None):
        """Forward method

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`
        """
        output = ModelOutput()
        max_depth = self.depth
        if output_layer_levels is not None:
            assert all(self.depth >= levels > 0 or levels == -1 for levels in output_layer_levels), f'Cannot output layer deeper than depth ({self.depth}). Got ({output_layer_levels}).'
            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)
        out = z.reshape(z.shape[0], -1)
        for i in range(max_depth):
            out = self.layers[i](out)
            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f'embedding_layer_{i + 1}'] = out
            if i + 1 == self.depth:
                output['embedding'] = out
        return output


class Encoder_VAE_MLP(BaseEncoder):

    def __init__(self, args: 'dict'):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        layers = nn.ModuleList()
        layers.append(nn.Sequential(nn.Linear(np.prod(args.input_dim), 512), nn.ReLU()))
        self.layers = layers
        self.depth = len(layers)
        self.embedding = nn.Linear(512, self.latent_dim)
        self.log_var = nn.Linear(512, self.latent_dim)

    def forward(self, x, output_layer_levels: 'List[int]'=None):
        output = ModelOutput()
        max_depth = self.depth
        if output_layer_levels is not None:
            assert all(self.depth >= levels > 0 or levels == -1 for levels in output_layer_levels), f'Cannot output layer deeper than depth ({self.depth}). Got ({output_layer_levels}).'
            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)
        out = x.reshape(-1, np.prod(self.input_dim))
        for i in range(max_depth):
            out = self.layers[i](out)
            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f'embedding_layer_{i + 1}'] = out
            if i + 1 == self.depth:
                output['embedding'] = self.embedding(out)
                output['log_covariance'] = self.log_var(out)
        return output


class VAE(BaseAE):
    """Vanilla Variational Autoencoder model.

    Args:
        model_config (VAEConfig): The Variational Autoencoder configuration setting the main
        parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'VAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        BaseAE.__init__(self, model_config=model_config, decoder=decoder)
        self.model_name = 'VAE'
        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError("No input dimension provided !'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' where the shape of the data is (C, H, W ..). Unable to build encoder automatically")
            encoder = Encoder_VAE_MLP(model_config)
            self.model_config.uses_default_encoder = True
        else:
            self.model_config.uses_default_encoder = False
        self.set_encoder(encoder)

    def forward(self, inputs: 'BaseDataset', **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        x = inputs['data']
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)['reconstruction']
        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)
        output = ModelOutput(recon_loss=recon_loss, reg_loss=kld, loss=loss, recon_x=recon_x, z=z)
        return output

    def loss_function(self, recon_x, x, mu, log_var, z):
        if self.model_config.reconstruction_loss == 'mse':
            recon_loss = 0.5 * F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        elif self.model_config.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """
        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size
        log_p = []
        for i in range(len(data)):
            x = data[i].unsqueeze(0)
            log_p_x = []
            for j in range(n_full_batch):
                x_rep = torch.cat(batch_size * [x])
                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)
                log_q_z_given_x = -0.5 * (log_var + (z - mu) ** 2 / torch.exp(log_var)).sum(dim=-1)
                log_p_z = -0.5 * (z ** 2).sum(dim=-1)
                recon_x = self.decoder(z)['reconstruction']
                if self.model_config.reconstruction_loss == 'mse':
                    log_p_x_given_z = -0.5 * F.mse_loss(recon_x.reshape(x_rep.shape[0], -1), x_rep.reshape(x_rep.shape[0], -1), reduction='none').sum(dim=-1) - torch.tensor([np.prod(self.input_dim) / 2 * np.log(np.pi * 2)])
                elif self.model_config.reconstruction_loss == 'bce':
                    log_p_x_given_z = -F.binary_cross_entropy(recon_x.reshape(x_rep.shape[0], -1), x_rep.reshape(x_rep.shape[0], -1), reduction='none').sum(dim=-1)
                log_p_x.append(log_p_x_given_z + log_p_z - log_q_z_given_x)
            log_p_x = torch.cat(log_p_x)
            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
        return np.mean(log_p)


class Adversarial_AE(VAE):
    """Adversarial Autoencoder model.

    Args:
        model_config (Adversarial_AE_Config): The Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        discriminator (BaseDiscriminator): An instance of BaseDiscriminator (inheriting from
            `torch.nn.Module` which plays the role of discriminator. This argument allows you to
            use your own neural networks architectures if desired. If None is provided, a simple
            Multi Layer Preception (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used.
            Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'Adversarial_AE_Config', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None, discriminator: 'Optional[BaseDiscriminator]'=None):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        if discriminator is None:
            if model_config.latent_dim is None:
                raise AttributeError("No latent dimension provided !'latent_dim' parameter of Adversarial_AE_Config instance must be set to a value. Unable to build discriminator automatically.")
            self.model_config.discriminator_input_dim = self.model_config.latent_dim
            discriminator = Discriminator_MLP(model_config)
            self.model_config.uses_default_discriminator = True
        else:
            self.model_config.uses_default_discriminator = False
        self.set_discriminator(discriminator)
        self.model_name = 'Adversarial_AE'
        assert 0 <= self.model_config.adversarial_loss_scale <= 1, 'adversarial_loss_scale must be in [0, 1]'
        self.adversarial_loss_scale = self.model_config.adversarial_loss_scale
        self.reconstruction_loss_scale = self.model_config.reconstruction_loss_scale
        self.deterministic_posterior = 1 if self.model_config.deterministic_posterior else 0

    def set_discriminator(self, discriminator: 'BaseDiscriminator') ->None:
        """This method is called to set the discriminator network

        Args:
            discriminator (BaseDiscriminator): The discriminator module that needs to be set to the
                model.

        """
        if not issubclass(type(discriminator), BaseDiscriminator):
            raise BadInheritanceError('Discriminator must inherit from BaseDiscriminator class from pythae.models.base_architectures.BaseDiscriminator. Refer to documentation.')
        self.discriminator = discriminator

    def forward(self, inputs: 'BaseDataset', **kwargs) ->ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        x = inputs['data']
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = (1 - self.deterministic_posterior) * torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)['reconstruction']
        z_prior = torch.randn_like(z, device=x.device).requires_grad_(True)
        recon_loss, autoencoder_loss, discriminator_loss = self.loss_function(recon_x, x, z, z_prior)
        loss = autoencoder_loss + discriminator_loss
        output = ModelOutput(loss=loss, recon_loss=recon_loss, autoencoder_loss=autoencoder_loss, discriminator_loss=discriminator_loss, recon_x=recon_x, z=z)
        return output

    def loss_function(self, recon_x, x, z, z_prior):
        N = z.shape[0]
        if self.model_config.reconstruction_loss == 'mse':
            recon_loss = 0.5 * F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        elif self.model_config.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        gen_adversarial_score = self.discriminator(z).embedding.flatten()
        prior_adversarial_score = self.discriminator(z_prior).embedding.flatten()
        true_labels = torch.ones(N, requires_grad=False)
        fake_labels = torch.zeros(N, requires_grad=False)
        autoencoder_loss = self.adversarial_loss_scale * F.binary_cross_entropy(gen_adversarial_score, true_labels) + recon_loss * self.reconstruction_loss_scale
        gen_adversarial_score_ = self.discriminator(z.detach()).embedding.flatten()
        discriminator_loss = F.binary_cross_entropy(prior_adversarial_score, true_labels) + F.binary_cross_entropy(gen_adversarial_score_, fake_labels)
        return recon_loss.mean(dim=0), autoencoder_loss.mean(dim=0), discriminator_loss.mean(dim=0)

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def save(self, dir_path: 'str'):
        """Method to save the model at a specific location

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """
        super().save(dir_path)
        model_path = dir_path
        model_dict = {'model_state_dict': deepcopy(self.state_dict())}
        if not self.model_config.uses_default_discriminator:
            with open(os.path.join(model_path, 'discriminator.pkl'), 'wb') as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.discriminator))
                cloudpickle.dump(self.discriminator, fp)
        torch.save(model_dict, os.path.join(model_path, 'model.pt'))

    @classmethod
    def _load_custom_discriminator_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)
        if 'discriminator.pkl' not in file_list:
            raise FileNotFoundError(f"Missing discriminator pkl file ('discriminator.pkl') in{dir_path}... This file is needed to rebuild custom discriminators. Cannot perform model building.")
        else:
            with open(os.path.join(dir_path, 'discriminator.pkl'), 'rb') as fp:
                discriminator = CPU_Unpickler(fp).load()
        return discriminator

    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided

        """
        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)
        if not model_config.uses_default_encoder:
            encoder = cls._load_custom_encoder_from_folder(dir_path)
        else:
            encoder = None
        if not model_config.uses_default_decoder:
            decoder = cls._load_custom_decoder_from_folder(dir_path)
        else:
            decoder = None
        if not model_config.uses_default_discriminator:
            discriminator = cls._load_custom_discriminator_from_folder(dir_path)
        else:
            discriminator = None
        model = cls(model_config, encoder=encoder, decoder=decoder, discriminator=discriminator)
        model.load_state_dict(model_weights)
        return model

    @classmethod
    def load_from_hf_hub(cls, hf_hub_path: 'str', allow_pickle: 'bool'=False):
        """Class method to be used to load a pretrained model from the Hugging Face hub

        Args:
            hf_hub_path (str): The path where the model should have been be saved on the
                hugginface hub.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl`` and ``discriminator``) if a custom encoder (resp. decoder and/or
                discriminator) was provided
        """
        if not hf_hub_is_available():
            raise ModuleNotFoundError('`huggingface_hub` package must be installed to load models from the HF hub. Run `python -m pip install huggingface_hub` and log in to your account with `huggingface-cli login`.')
        logger.info(f'Downloading {cls.__name__} files for rebuilding...')
        config_path = hf_hub_download(repo_id=hf_hub_path, filename='model_config.json')
        dir_path = os.path.dirname(config_path)
        _ = hf_hub_download(repo_id=hf_hub_path, filename='model.pt')
        model_config = cls._load_model_config_from_folder(dir_path)
        if cls.__name__ + 'Config' != model_config.name and cls.__name__ + '_Config' != model_config.name:
            warnings.warn(f'You are trying to load a `{cls.__name__}` while a `{model_config.name}` is given.')
        model_weights = cls._load_model_weights_from_folder(dir_path)
        if (not model_config.uses_default_encoder or not model_config.uses_default_decoder or not model_config.uses_default_discriminator) and not allow_pickle:
            warnings.warn('You are about to download pickled files from the HF hub that may have been created by a third party and so could potentially harm your computer. If you are sure that you want to download them set `allow_pickle=true`.')
        else:
            if not model_config.uses_default_encoder:
                _ = hf_hub_download(repo_id=hf_hub_path, filename='encoder.pkl')
                encoder = cls._load_custom_encoder_from_folder(dir_path)
            else:
                encoder = None
            if not model_config.uses_default_decoder:
                _ = hf_hub_download(repo_id=hf_hub_path, filename='decoder.pkl')
                decoder = cls._load_custom_decoder_from_folder(dir_path)
            else:
                decoder = None
            if not model_config.uses_default_discriminator:
                _ = hf_hub_download(repo_id=hf_hub_path, filename='discriminator.pkl')
                discriminator = cls._load_custom_discriminator_from_folder(dir_path)
            else:
                discriminator = None
            logger.info(f'Successfully downloaded {cls.__name__} model!')
            model = cls(model_config, encoder=encoder, decoder=decoder, discriminator=discriminator)
            model.load_state_dict(model_weights)
            return model


class BetaTCVAE(VAE):
    """
    :math:`\\beta`-TCVAE model.

    Args:
        model_config (BetaTCVAEConfig): The Variational Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'BetaTCVAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self.model_name = 'BetaTCVAE'
        self.alpha = model_config.alpha
        self.beta = model_config.beta
        self.gamma = model_config.gamma
        self.use_mss = model_config.use_mss

    def forward(self, inputs: 'BaseDataset', **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        x = inputs['data']
        dataset_size = kwargs.pop('dataset_size', x.shape[0])
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)['reconstruction']
        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z, dataset_size)
        output = ModelOutput(recon_loss=recon_loss, reg_loss=kld, loss=loss, recon_x=recon_x, z=z)
        return output

    def loss_function(self, recon_x, x, mu, log_var, z, dataset_size):
        if self.model_config.reconstruction_loss == 'mse':
            recon_loss = 0.5 * F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        elif self.model_config.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        log_q_z_given_x = self._compute_log_gauss_density(z, mu, log_var).sum(dim=-1)
        log_prior = self._compute_log_gauss_density(z, torch.zeros_like(z), torch.zeros_like(z)).sum(dim=-1)
        log_q_batch_perm = self._compute_log_gauss_density(z.reshape(z.shape[0], 1, -1), mu.reshape(1, z.shape[0], -1), log_var.reshape(1, z.shape[0], -1))
        if self.use_mss:
            logiw_mat = self._log_importance_weight_matrix(z.shape[0], dataset_size)
            log_q_z = torch.logsumexp(logiw_mat + log_q_batch_perm.sum(dim=-1), dim=-1)
            log_prod_q_z = torch.logsumexp(logiw_mat.reshape(z.shape[0], z.shape[0], -1) + log_q_batch_perm, dim=1).sum(dim=-1)
        else:
            log_q_z = torch.logsumexp(log_q_batch_perm.sum(dim=-1), dim=-1) - torch.log(torch.tensor([z.shape[0] * dataset_size]))
            log_prod_q_z = (torch.logsumexp(log_q_batch_perm, dim=1) - torch.log(torch.tensor([z.shape[0] * dataset_size]))).sum(dim=-1)
        mutual_info_loss = log_q_z_given_x - log_q_z
        TC_loss = log_q_z - log_prod_q_z
        dimension_wise_KL = log_prod_q_z - log_prior
        return (recon_loss + self.alpha * mutual_info_loss + self.beta * TC_loss + self.gamma * dimension_wise_KL).mean(dim=0), recon_loss.mean(dim=0), (self.alpha * mutual_info_loss + self.beta * TC_loss + self.gamma * dimension_wise_KL).mean(dim=0)

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def _compute_log_gauss_density(self, z, mu, log_var):
        """element-wise computation"""
        return -0.5 * (torch.log(torch.tensor([2 * np.pi])) + log_var + (z - mu) ** 2 * torch.exp(-log_var))

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        """Compute importance weigth matrix for MSS
        Code from (https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py)
        """
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M + 1] = 1 / N
        W.view(-1)[1::M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()


class BetaVAE(VAE):
    """
    :math:`\\beta`-VAE model.

    Args:
        model_config (BetaVAEConfig): The Variational Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'BetaVAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self.model_name = 'BetaVAE'
        self.beta = model_config.beta

    def forward(self, inputs: 'BaseDataset', **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        x = inputs['data']
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)['reconstruction']
        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)
        output = ModelOutput(recon_loss=recon_loss, reg_loss=kld, loss=loss, recon_x=recon_x, z=z)
        return output

    def loss_function(self, recon_x, x, mu, log_var, z):
        if self.model_config.reconstruction_loss == 'mse':
            recon_loss = 0.5 * F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        elif self.model_config.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        return (recon_loss + self.beta * KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps


class CIWAE(VAE):
    """
    Combination Importance Weighted Autoencoder model.

    Args:
        model_config (CIWAEConfig): The CIWAE configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'CIWAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self.model_name = 'CIWAE'
        self.n_samples = model_config.number_samples
        self.beta = model_config.beta

    def forward(self, inputs: 'BaseDataset', **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        x = inputs['data']
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        mu = mu.unsqueeze(1).repeat(1, self.n_samples, 1)
        log_var = log_var.unsqueeze(1).repeat(1, self.n_samples, 1)
        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z.reshape(-1, self.latent_dim))['reconstruction'].reshape(x.shape[0], self.n_samples, -1)
        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)
        output = ModelOutput(recon_loss=recon_loss, reg_loss=kld, loss=loss, recon_x=recon_x.reshape(x.shape[0], self.n_samples, -1)[:, 0, :].reshape_as(x), z=z.reshape(x.shape[0], self.n_samples, -1)[:, 0, :].reshape(-1, self.latent_dim))
        return output

    def loss_function(self, recon_x, x, mu, log_var, z):
        if self.model_config.reconstruction_loss == 'mse':
            recon_loss = 0.5 * F.mse_loss(recon_x, x.reshape(recon_x.shape[0], -1).unsqueeze(1).repeat(1, self.n_samples, 1), reduction='none').sum(dim=-1)
        elif self.model_config.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x, x.reshape(recon_x.shape[0], -1).unsqueeze(1).repeat(1, self.n_samples, 1), reduction='none').sum(dim=-1)
        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / log_var.exp())).sum(dim=-1)
        log_p_z = -0.5 * (z ** 2).sum(dim=-1)
        KLD = -(log_p_z - log_q_z)
        log_w = -(recon_loss + KLD)
        log_w_minus_max = log_w - log_w.max(1, keepdim=True)[0]
        w = log_w_minus_max.exp()
        w_tilde = (w / w.sum(axis=1, keepdim=True)).detach()
        iwae_elbo = -(w_tilde * log_w).sum(1)
        vae_elbo = -log_w.mean(dim=-1)
        return (self.beta * vae_elbo + (1 - self.beta) * iwae_elbo).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps


class DisentangledBetaVAE(VAE):
    """
    Disentangled :math:`\\beta`-VAE model.

    Args:
        model_config (DisentangledBetaVAEConfig): The Variational Autoencoder configuration setting
            the main parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'DisentangledBetaVAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        assert model_config.warmup_epoch >= 0, f'Provide a value of warmup epoch >= 0, got {model_config.warmup_epoch}'
        self.model_name = 'DisentangledBetaVAE'
        self.beta = model_config.beta
        self.C = model_config.C
        self.warmup_epoch = model_config.warmup_epoch

    def forward(self, inputs: 'BaseDataset', **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        x = inputs['data']
        epoch = kwargs.pop('epoch', self.warmup_epoch)
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)['reconstruction']
        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z, epoch)
        output = ModelOutput(recon_loss=recon_loss, reg_loss=kld, loss=loss, recon_x=recon_x, z=z)
        return output

    def loss_function(self, recon_x, x, mu, log_var, z, epoch):
        if self.model_config.reconstruction_loss == 'mse':
            recon_loss = 0.5 * F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        elif self.model_config.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        C_factor = min(epoch / (self.warmup_epoch + 1), 1)
        KLD_diff = torch.abs(KLD - self.C * C_factor)
        return (recon_loss + self.beta * KLD_diff).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps


class FactorVAEDiscriminator(nn.Module):

    def __init__(self, latent_dim=16, hidden_units=1000) ->None:
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(latent_dim, hidden_units), nn.LeakyReLU(0.2), nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(0.2), nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(0.2), nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(0.2), nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(0.2), nn.Linear(hidden_units, 2))

    def forward(self, z: 'torch.Tensor'):
        return self.layers(z)


class FactorVAE(VAE):
    """
    FactorVAE model.

    Args:
        model_config (FactorVAEConfig): The Variational Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.


    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'FactorVAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self.discriminator = FactorVAEDiscriminator(latent_dim=model_config.latent_dim)
        self.model_name = 'FactorVAE'
        self.gamma = model_config.gamma

    def set_discriminator(self, discriminator: 'BaseDiscriminator') ->None:
        """This method is called to set the discriminator network

        Args:
            discriminator (BaseDiscriminator): The discriminator module that needs to be set to the model.

        """
        if not issubclass(type(discriminator), BaseDiscriminator):
            raise BadInheritanceError('Discriminator must inherit from BaseDiscriminator class from pythae.models.base_architectures.BaseDiscriminator. Refer to documentation.')
        self.discriminator = discriminator

    def forward(self, inputs: 'BaseDataset', **kwargs) ->ModelOutput:
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        x_in = inputs['data']
        if x_in.shape[0] <= 1:
            raise ArithmeticError('At least 2 samples in a batch are required for the `FactorVAE` model')
        idx = torch.randperm(x_in.shape[0])
        idx_1 = idx[int(x_in.shape[0] / 2):]
        idx_2 = idx[:int(x_in.shape[0] / 2)]
        x = inputs['data'][idx_1]
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)['reconstruction']
        x_bis = inputs['data'][idx_2]
        encoder_output = self.encoder(x_bis)
        mu_bis, log_var_bis = encoder_output.embedding, encoder_output.log_covariance
        std_bis = torch.exp(0.5 * log_var_bis)
        z_bis, _ = self._sample_gauss(mu_bis, std_bis)
        z_bis_permuted = self._permute_dims(z_bis).detach()
        recon_loss, autoencoder_loss, discriminator_loss = self.loss_function(recon_x, x, mu, log_var, z, z_bis_permuted)
        loss = autoencoder_loss + discriminator_loss
        output = ModelOutput(loss=loss, recon_loss=recon_loss, autoencoder_loss=autoencoder_loss, discriminator_loss=discriminator_loss, recon_x=recon_x, recon_x_indices=idx_1, z=z, z_bis_permuted=z_bis_permuted)
        return output

    def loss_function(self, recon_x, x, mu, log_var, z, z_bis_permuted):
        N = z.shape[0]
        if self.model_config.reconstruction_loss == 'mse':
            recon_loss = 0.5 * F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        elif self.model_config.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        latent_adversarial_score = self.discriminator(z)
        TC = (latent_adversarial_score[:, 0] - latent_adversarial_score[:, 1]).mean()
        autoencoder_loss = recon_loss + KLD + self.gamma * TC
        permuted_latent_adversarial_score = self.discriminator(z_bis_permuted)
        true_labels = torch.ones(z_bis_permuted.shape[0], requires_grad=False).type(torch.LongTensor)
        fake_labels = torch.zeros(z.shape[0], requires_grad=False).type(torch.LongTensor)
        TC_permuted = F.cross_entropy(latent_adversarial_score, fake_labels) + F.cross_entropy(permuted_latent_adversarial_score, true_labels)
        discriminator_loss = 0.5 * TC_permuted
        return recon_loss.mean(dim=0), autoencoder_loss.mean(dim=0), discriminator_loss.mean(dim=0)

    def reconstruct(self, inputs: 'torch.Tensor'):
        """This function returns the reconstructions of given input data.

        Args:
            inputs (torch.Tensor): The inputs data to be reconstructed of shape [B x input_dim]
            ending_inputs (torch.Tensor): The starting inputs in the interpolation of shape

        Returns:
            torch.Tensor: A tensor of shape [B x input_dim] containing the reconstructed samples.
        """
        encoder_output = self.encoder(inputs)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)['reconstruction']
        return recon_x

    def interpolate(self, starting_inputs: 'torch.Tensor', ending_inputs: 'torch.Tensor', granularity: 'int'=10):
        """This function performs a linear interpolation in the latent space of the autoencoder
        from starting inputs to ending inputs. It returns the interpolation trajectories.

        Args:
            starting_inputs (torch.Tensor): The starting inputs in the interpolation of shape
                [B x input_dim]
            ending_inputs (torch.Tensor): The starting inputs in the interpolation of shape
                [B x input_dim]
            granularity (int): The granularity of the interpolation.

        Returns:
            torch.Tensor: A tensor of shape [B x granularity x input_dim] containing the
            interpolation trajectories.
        """
        assert starting_inputs.shape[0] == ending_inputs.shape[0], f'The number of starting_inputs should equal the number of ending_inputs. Got {starting_inputs.shape[0]} sampler for starting_inputs and {ending_inputs.shape[0]} for endinging_inputs.'
        encoder_output = self.encoder(starting_inputs)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        starting_z, _ = self._sample_gauss(mu, std)
        encoder_output = self.encoder(ending_inputs)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        ending_z, _ = self._sample_gauss(mu, std)
        t = torch.linspace(0, 1, granularity)
        intep_line = (torch.kron(starting_z.reshape(starting_z.shape[0], -1), (1 - t).unsqueeze(-1)) + torch.kron(ending_z.reshape(ending_z.shape[0], -1), t.unsqueeze(-1))).reshape((starting_z.shape[0] * t.shape[0],) + starting_z.shape[1:])
        decoded_line = self.decoder(intep_line).reconstruction.reshape((starting_inputs.shape[0], t.shape[0]) + starting_inputs.shape[1:])
        return decoded_line

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def _permute_dims(self, z):
        permuted = torch.zeros_like(z)
        for i in range(z.shape[-1]):
            perms = torch.randperm(z.shape[0])
            permuted[:, i] = z[perms, i]
        return permuted


class HierarchicalResidualQuantizer(nn.Module):

    def __init__(self, model_config: 'HRQVAEConfig'):
        nn.Module.__init__(self)
        self.model_config = model_config
        self.embedding_dim = model_config.embedding_dim
        self.num_embeddings = model_config.num_embeddings
        self.num_levels = model_config.num_levels
        self.embeddings = nn.ModuleList([nn.Embedding(self.num_embeddings, self.embedding_dim) for hix in range(self.num_levels)])
        init_scale = model_config.init_scale
        init_decay_weight = model_config.init_decay_weight
        for hix, embedding in enumerate(self.embeddings):
            scale = init_scale * init_decay_weight ** hix / sqrt(self.embedding_dim)
            embedding.weight.data.uniform_(-1.0 * scale, scale)
            embedding.weight.data = embedding.weight.data / torch.linalg.vector_norm(embedding.weight, dim=1, keepdim=True) * init_scale * init_decay_weight ** hix

    def forward(self, z: 'torch.Tensor', epoch: 'int', uses_ddp: 'bool'=False):
        if uses_ddp:
            raise Exception("HRQVAE doesn't currently support DDP :(")
        input_shape = z.shape
        z = z.reshape(-1, self.embedding_dim)
        loss = torch.zeros(z.shape[0])
        resid_error = z
        quantized = []
        codes = []
        all_probs = []
        for head_ix, embedding in enumerate(self.embeddings):
            if head_ix > 0:
                resid_error = z - torch.sum(torch.cat(quantized, dim=1), dim=1)
            distances = -1.0 * (torch.sum(resid_error ** 2, dim=-1, keepdim=True) + torch.sum(embedding.weight ** 2, dim=-1) - 2 * torch.matmul(resid_error, embedding.weight.T))
            gumbel_sched_weight = torch.exp(-torch.tensor(float(epoch)) / float(self.model_config.temp_schedule_gamma * 1.5 ** head_ix))
            gumbel_temp = max(gumbel_sched_weight, 0.5)
            if self.training:
                sample_onehot = F.gumbel_softmax(distances, tau=gumbel_temp, hard=True, dim=-1)
            else:
                indices = torch.argmax(distances, dim=-1)
                sample_onehot = F.one_hot(indices, num_classes=self.num_embeddings).float()
            probs = F.softmax(distances / gumbel_temp, dim=-1)
            prior = torch.ones_like(distances).detach() / torch.ones_like(distances).sum(-1, keepdim=True).detach()
            kl_loss = torch.nn.KLDivLoss(reduction='none')
            kl = kl_loss(nn.functional.log_softmax(distances, dim=-1), prior).sum(dim=-1)
            loss += kl * self.model_config.kl_weight
            this_quantized = sample_onehot @ embedding.weight
            this_quantized = this_quantized.reshape_as(z)
            quantized.append(this_quantized.unsqueeze(-2))
            codes.append(torch.argmax(sample_onehot, dim=-1).unsqueeze(-1))
            all_probs.append(probs.unsqueeze(-2))
        quantized = torch.cat(quantized, dim=-2)
        quantized_indices = torch.cat(codes, dim=-1)
        all_probs = torch.cat(all_probs, dim=-2)
        if self.model_config.norm_loss_weight is not None:
            upper_norms = torch.linalg.vector_norm(quantized[:, :-1, :], dim=-1)
            lower_norms = torch.linalg.vector_norm(quantized[:, 1:, :], dim=-1)
            norm_loss = (torch.max(lower_norms / upper_norms * self.model_config.norm_loss_scale, torch.ones_like(lower_norms)) - 1.0) ** 2
            loss += norm_loss.mean(dim=1) * self.model_config.norm_loss_weight
        if self.training:
            drop_dist = torch.distributions.Bernoulli(1 - self.model_config.depth_drop_rate)
            mask = drop_dist.sample(sample_shape=(*quantized.shape[:-1], 1))
            mask = torch.cumprod(mask, dim=1)
            quantized = quantized * mask
        quantized = quantized.sum(dim=-2).reshape(*input_shape)
        quantized = quantized.permute(0, 3, 1, 2)
        loss = loss.reshape(input_shape[0], -1).mean(dim=1)
        quantized_indices = quantized_indices.reshape(*input_shape[:-1], self.num_levels)
        output = ModelOutput(z_orig=z, quantized_vector=quantized, quantized_indices=quantized_indices, loss=loss, probs=all_probs)
        return output


class HRQVAE(AE):
    """
    Hierarchical Residual Quantization-VAE model. Introduced in https://aclanthology.org/2022.acl-long.178/ (Hosking et al., ACL 2022)

    Args:
        model_config (HRQVAEConfig): The Variational Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    """

    def __init__(self, model_config: 'HRQVAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        AE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self._set_quantizer(model_config)
        self.model_name = 'HRQVAE'

    def _set_quantizer(self, model_config):
        if model_config.input_dim is None:
            raise AttributeError("No input dimension provided !'input_dim' parameter of HRQVAEConfig instance must be set to 'data_shape' where the shape of the data is (C, H, W ..). Unable to set quantizer.")
        x = torch.randn((2,) + self.model_config.input_dim)
        z = self.encoder(x).embedding
        if len(z.shape) == 2:
            z = z.reshape(z.shape[0], 1, 1, -1)
        z = z.permute(0, 2, 3, 1)
        self.model_config.embedding_dim = z.shape[-1]
        self.quantizer = HierarchicalResidualQuantizer(model_config=model_config)

    def forward(self, inputs: 'Dict[str, Any]', **kwargs) ->ModelOutput:
        """
        The VAE model

        Args:
            inputs (dict): A dict of samples

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        x = inputs['data']
        uses_ddp = kwargs.pop('uses_ddp', False)
        epoch = kwargs.pop('epoch', 0)
        encoder_output = self.encoder(x)
        embeddings = encoder_output.embedding
        reshape_for_decoding = False
        if len(embeddings.shape) == 2:
            embeddings = embeddings.reshape(embeddings.shape[0], 1, 1, -1)
            reshape_for_decoding = True
        embeddings = embeddings.permute(0, 2, 3, 1)
        quantizer_output = self.quantizer(embeddings, epoch=epoch, uses_ddp=uses_ddp)
        quantized_embed = quantizer_output.quantized_vector
        if reshape_for_decoding:
            quantized_embed = quantized_embed.reshape(embeddings.shape[0], -1)
        recon_x = self.decoder(quantized_embed).reconstruction
        loss, recon_loss, hrq_loss = self.loss_function(recon_x, x, quantizer_output)
        output = ModelOutput(loss=loss, recon_loss=recon_loss, hrq_loss=hrq_loss, recon_x=recon_x, z=quantized_embed, z_orig=quantizer_output.z_orig, quantized_indices=quantizer_output.quantized_indices, probs=quantizer_output.probs)
        return output

    def loss_function(self, recon_x, x, quantizer_output):
        recon_loss = F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        hrq_loss = quantizer_output.loss
        return (recon_loss + hrq_loss).mean(dim=0), recon_loss.mean(dim=0), hrq_loss.mean(dim=0)


class HVAE(VAE):
    """
    Hamiltonian VAE.

    Args:
        model_config (HVAEConfig): A model configuration setting the main parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'HVAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self.model_name = 'HVAE'
        self.n_lf = model_config.n_lf
        self.eps_lf = nn.Parameter(torch.tensor([model_config.eps_lf]), requires_grad=True if model_config.learn_eps_lf else False)
        self.beta_zero_sqrt = nn.Parameter(torch.tensor([model_config.beta_zero]) ** 0.5, requires_grad=True if model_config.learn_beta_zero else False)
        if model_config.reconstruction_loss == 'bce':
            warnings.warn('Carefull, this model expects the encoder to give the *logits* of the Bernouilli distribution. Make sure the encoder actually outputs the logits.')

    def forward(self, inputs: 'BaseDataset', **kwargs) ->ModelOutput:
        """
        The input data is first encoded. The reparametrization is used to produce a sample
        :math:`z_0` from the approximate posterior :math:`q_{\\phi}(z|x)`. Then
        Hamiltonian equations are solved using the leapfrog integrator.

        Args:
            inputs (BaseDataset): The training data with labels

        Returns:
            output (ModelOutput): An instance of ModelOutput containing all the relevant parameters
        """
        x = inputs['data']
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = F.softplus(log_var)
        z0, eps0 = self._sample_gauss(mu, std)
        gamma = torch.randn_like(z0, device=x.device)
        rho = gamma / self.beta_zero_sqrt
        z = z0
        beta_sqrt_old = self.beta_zero_sqrt
        for k in range(self.n_lf):
            rho_ = rho - self.eps_lf / 2 * self._dU_dz(z, x)
            z_ = z + self.eps_lf * rho
            rho__ = rho_ - self.eps_lf / 2 * self._dU_dz(z_, x)
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = beta_sqrt_old / beta_sqrt * rho__
            beta_sqrt_old = beta_sqrt
            z = z_
            rho = rho__
        recon_x = self.decoder(z)['reconstruction'].reshape_as(x)
        loss = self.loss_function(x, z, rho, z0, mu, log_var)
        output = ModelOutput(loss=loss, recon_x=recon_x, z=z, z0=z0, rho=rho, eps0=eps0, gamma=gamma, mu=mu, log_var=log_var)
        return output

    def _dU_dz(self, z, x):
        net_out = self.decoder(z)['reconstruction'].reshape(x.shape[0], -1)
        U = -self._log_p_x_given_z(net_out, x).sum()
        g = grad(U, z)[0]
        return g + z

    def loss_function(self, x, zK, rhoK, z0, mu, log_var):
        recon_x = self.decoder(zK)['reconstruction']
        logpx_given_z = self._log_p_x_given_z(recon_x, x)
        log_zk = -0.5 * torch.pow(zK, 2).sum(dim=-1)
        logrhoK = -0.5 * torch.pow(rhoK, 2).sum(dim=-1)
        logp = logpx_given_z + logrhoK + log_zk
        logq = -0.5 * log_var.sum(dim=-1)
        return -(logp - logq).mean(dim=0)

    def _tempering(self, k, K):
        """Perform tempering step"""
        beta_k = (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2 + 1 / self.beta_zero_sqrt
        return 1 / beta_k

    def _log_p_x_given_z(self, recon_x, x):
        if self.model_config.reconstruction_loss == 'mse':
            logp_x_given_z = -0.5 * F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        elif self.model_config.reconstruction_loss == 'bce':
            logp_x_given_z = torch.distributions.Bernoulli(logits=recon_x.reshape(x.shape[0], -1)).log_prob(x.reshape(x.shape[0], -1)).sum(dim=-1)
        return logp_x_given_z

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """
        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size
        log_p = []
        for i in range(len(data)):
            x = data[i].unsqueeze(0)
            log_p_x = []
            for j in range(n_full_batch):
                x_rep = torch.cat(batch_size * [x]).reshape(-1, 1, 28, 28)
                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                std = torch.exp(0.5 * log_var)
                z0, _ = self._sample_gauss(mu, std)
                gamma = torch.randn_like(z0, device=x.device)
                rho = gamma / self.beta_zero_sqrt
                z = z0
                beta_sqrt_old = self.beta_zero_sqrt
                for k in range(self.n_lf):
                    rho_ = rho - self.eps_lf / 2 * self._dU_dz(z, x_rep)
                    z = z + self.eps_lf * rho_
                    rho__ = rho_ - self.eps_lf / 2 * self._dU_dz(z, x_rep)
                    beta_sqrt = self._tempering(k + 1, self.n_lf)
                    rho = beta_sqrt_old / beta_sqrt * rho__
                    beta_sqrt_old = beta_sqrt
                log_q_z0_given_x = -0.5 * (log_var + (z0 - mu) ** 2 / torch.exp(log_var)).sum(dim=-1)
                log_p_z = -0.5 * (z ** 2).sum(dim=-1)
                log_p_rho = -0.5 * (rho ** 2).sum(dim=-1)
                log_p_rho0 = -0.5 * (rho ** 2).sum(dim=-1) * self.beta_zero_sqrt
                recon_x = self.decoder(z)['reconstruction']
                log_p_x_given_z = self._log_p_x_given_z(recon_x, x_rep)
                log_p_x.append(log_p_x_given_z + log_p_z + log_p_rho - log_p_rho0 - log_q_z0_given_x)
            log_p_x = torch.cat(log_p_x)
            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
            if i % 50 == 0:
                None
        return np.mean(log_p)


class BaseNF(nn.Module):
    """Base Class from Normalizing flows

    Args:
        model_config (BaseNFConfig): The configuration setting the main parameters of the
            model.
    """

    def __init__(self, model_config: 'BaseNFConfig'):
        nn.Module.__init__(self)
        if model_config.input_dim is None:
            raise AttributeError("No input dimension provided !'input_dim' parameter of MADEConfig instance must be set to 'data_shape' where the shape of the data is (C, H, W ..)]. Unable to build networkautomatically")
        self.model_config = model_config
        self.input_dim = np.prod(model_config.input_dim)

    def forward(self, x: 'torch.Tensor', **kwargs) ->ModelOutput:
        """Main forward pass mapping the data towards the prior
        This function should output a :class:`~pythae.models.base.base_utils.ModelOutput` instance
        gathering all the model outputs

        Args:
            x (torch.Tensor): The training data.

        Returns:
            ModelOutput: A ModelOutput instance providing the outputs of the model.
        """
        raise NotImplementedError()

    def inverse(self, y: 'torch.Tensor', **kwargs) ->ModelOutput:
        """Main inverse pass mapping the prior toward the data
        This function should output a :class:`~pythae.models.base.base_utils.ModelOutput` instance
        gathering all the model outputs

        Args:
            inputs (torch.Tensor): Data from the prior.

        Returns:
            ModelOutput: A ModelOutput instance providing the outputs of the model.
        """
        raise NotImplementedError()

    def update(self):
        """Method that allows model update during the training (at the end of a training epoch)

        If needed, this method must be implemented in a child class.

        By default, it does nothing.
        """

    def save(self, dir_path):
        """Method to save the model at a specific location. It saves, the model weights as a
        ``models.pt`` file along with the model config as a ``model_config.json`` file.

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """
        env_spec = EnvironmentConfig(python_version=f'{sys.version_info[0]}.{sys.version_info[1]}')
        model_dict = {'model_state_dict': deepcopy(self.state_dict())}
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except (FileNotFoundError, TypeError) as e:
                raise e
        env_spec.save_json(dir_path, 'environment')
        self.model_config.save_json(dir_path, 'model_config')
        torch.save(model_dict, os.path.join(dir_path, 'model.pt'))

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)
        if 'model_config.json' not in file_list:
            raise FileNotFoundError(f"Missing model config file ('model_config.json') in{dir_path}... Cannot perform model building.")
        path_to_model_config = os.path.join(dir_path, 'model_config.json')
        model_config = AutoConfig.from_json_file(path_to_model_config)
        return model_config

    @classmethod
    def _load_model_weights_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)
        if 'model.pt' not in file_list:
            raise FileNotFoundError(f"Missing model weights file ('model.pt') file in{dir_path}... Cannot perform model building.")
        path_to_model_weights = os.path.join(dir_path, 'model.pt')
        try:
            model_weights = torch.load(path_to_model_weights, map_location='cpu')
        except RuntimeError:
            RuntimeError("Enable to load model weights. Ensure they are saves in a '.pt' format.")
        if 'model_state_dict' not in model_weights.keys():
            raise KeyError(f"Model state dict is not available in 'model.pt' file. Got keys:{model_weights.keys()}")
        model_weights = model_weights['model_state_dict']
        return model_weights

    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided
        """
        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)
        model = cls(model_config)
        model.load_state_dict(model_weights)
        return model


class BatchNorm(nn.Module):
    """A BatchNorm layer used in several flows"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        nn.Module.__init__(self)
        self.eps = eps
        self.momentum = momentum
        self.log_gamma = nn.Parameter(torch.zeros(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            self.batch_mean = x.mean(0).data
            self.batch_var = x.var(0).data
            self.running_mean.mul_(1 - self.momentum).add_(self.batch_mean * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(self.batch_var * self.momentum)
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        y = (x - mean) / (var + self.eps).sqrt() * self.log_gamma.exp() + self.beta
        log_abs_det_jac = self.log_gamma - 0.5 * (var + self.eps).log()
        output = ModelOutput(out=y, log_abs_det_jac=log_abs_det_jac.expand_as(x).sum(dim=-1))
        return output

    def inverse(self, y):
        if self.training:
            if not hasattr(self, 'batch_mean') or not hasattr(self, 'batch_var'):
                mean = torch.zeros(1)
                var = torch.ones(1)
            else:
                mean = self.batch_mean
                var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x = (y - self.beta) * (-self.log_gamma).exp() * (var + self.eps).sqrt() + mean
        log_abs_det_jac = -self.log_gamma + 0.5 * (var + self.eps).log()
        output = ModelOutput(out=x, log_abs_det_jac=log_abs_det_jac.expand_as(x).sum(dim=-1))
        return output


class MaskedLinear(nn.Linear):
    """Masked Linear Layer inheriting from `~torch.nn.Linear` class and applying a mask to consider
    only a selection of weight.
    """

    def __init__(self, in_features, out_features, mask):
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(BaseNF):
    """Masked Autoencoder model

    Args:
        model_config (MADEConfig): The MADE model configuration setting the main parameters of the
            model
    """

    def __init__(self, model_config: 'MADEConfig'):
        BaseNF.__init__(self, model_config=model_config)
        self.net = []
        self.m = {}
        self.model_config = model_config
        self.input_dim = np.prod(model_config.input_dim)
        self.output_dim = np.prod(model_config.output_dim)
        self.hidden_sizes = model_config.hidden_sizes
        self.model_name = 'MADE'
        if model_config.input_dim is None:
            raise AttributeError("No input dimension provided !'input_dim' parameter of MADEConfig instance must be set to 'data_shape' where the shape of the data is (C, H, W ..)]. Unable to build networkautomatically")
        if model_config.output_dim is None:
            raise AttributeError("No input dimension provided !'output_dim' parameter of MADEConfig instance must be set to 'data_shape' where the shape of the data is (C, H, W ..)]. Unable to build networkautomatically")
        hidden_sizes = [self.input_dim] + model_config.hidden_sizes + [self.output_dim]
        masks = self._make_mask(ordering=self.model_config.degrees_ordering)
        for inp, out, mask in zip(hidden_sizes[:-1], hidden_sizes[1:-1], masks[:-1]):
            self.net.extend([MaskedLinear(inp, out, mask), nn.ReLU()])
        self.net.extend([MaskedLinear(self.hidden_sizes[-1], 2 * self.output_dim, masks[-1].repeat(2, 1))])
        self.net = nn.Sequential(*self.net)

    def _make_mask(self, ordering='sequential'):
        if ordering == 'sequential':
            self.m[-1] = torch.arange(self.input_dim)
            for i in range(len(self.hidden_sizes)):
                self.m[i] = torch.arange(self.hidden_sizes[i]) % (self.input_dim - 1)
        else:
            self.m[-1] = torch.randperm(self.input_dim)
            for i in range(len(self.hidden_sizes)):
                self.m[i] = torch.randint(self.m[-1].min(), self.input_dim - 1, (self.hidden_sizes[i],))
        masks = []
        for i in range(len(self.hidden_sizes)):
            masks += [(self.m[i].unsqueeze(-1) >= self.m[i - 1].unsqueeze(0)).float()]
        masks.append((self.m[len(self.hidden_sizes) - 1].unsqueeze(0) < self.m[-1].unsqueeze(-1)).float())
        return masks

    def forward(self, x: 'torch.tensor', **kwargs) ->ModelOutput:
        """The input data is transformed toward the prior

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        net_output = self.net(x.reshape(x.shape[0], -1))
        mu = net_output[:, :self.input_dim]
        log_var = net_output[:, self.input_dim:]
        return ModelOutput(mu=mu, log_var=log_var)


class IAF(BaseNF):
    """Inverse Autoregressive Flow.

    Args:
        model_config (IAFConfig): The IAF model configuration setting the main parameters of the
            model.
    """

    def __init__(self, model_config: 'IAFConfig'):
        BaseNF.__init__(self, model_config=model_config)
        self.net = []
        self.m = {}
        self.model_config = model_config
        self.input_dim = np.prod(model_config.input_dim)
        self.hidden_size = model_config.hidden_size
        self.model_name = 'IAF'
        made_config = MADEConfig(input_dim=(self.input_dim,), output_dim=(self.input_dim,), hidden_sizes=[self.hidden_size] * self.model_config.n_hidden_in_made, degrees_ordering='sequential')
        for i in range(model_config.n_made_blocks):
            self.net.extend([MADE(made_config)])
            if self.model_config.include_batch_norm:
                self.net.extend([BatchNorm(self.input_dim)])
        self.net = nn.ModuleList(self.net)

    def forward(self, x: 'torch.Tensor', **kwargs) ->ModelOutput:
        """The input data is transformed toward the prior (f^{-1})

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        x = x.reshape(x.shape[0], -1)
        sum_log_abs_det_jac = torch.zeros(x.shape[0])
        for layer in self.net:
            if layer.__class__.__name__ == 'MADE':
                y = torch.zeros_like(x)
                for i in range(self.input_dim):
                    layer_out = layer(y.clone())
                    m, s = layer_out.mu, layer_out.log_var
                    y[:, i] = (x[:, i] - m[:, i]) * (-s[:, i]).exp()
                    sum_log_abs_det_jac += -s[:, i]
                x = y
            else:
                layer_out = layer(x)
                x = layer_out.out
                sum_log_abs_det_jac += layer_out.log_abs_det_jac
            x = x.flip(dims=(1,))
        return ModelOutput(out=x, log_abs_det_jac=sum_log_abs_det_jac)

    def inverse(self, y: 'torch.Tensor', **kwargs) ->ModelOutput:
        """The prior is transformed toward the input data (f)

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        y = y.reshape(y.shape[0], -1)
        sum_log_abs_det_jac = torch.zeros(y.shape[0])
        for layer in self.net[::-1]:
            y = y.flip(dims=(1,))
            if layer.__class__.__name__ == 'MADE':
                layer_out = layer(y)
                m, s = layer_out.mu, layer_out.log_var
                y = y * s.exp() + m
                sum_log_abs_det_jac += s.sum(dim=-1)
            else:
                layer_out = layer.inverse(y)
                y = layer_out.out
                sum_log_abs_det_jac += layer_out.log_abs_det_jac
        return ModelOutput(out=y, log_abs_det_jac=sum_log_abs_det_jac)


class INFOVAE_MMD(VAE):
    """Info Variational Autoencoder model.

    Args:
        model_config (INFOVAE_MMD_Config): The Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'INFOVAE_MMD_Config', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self.model_name = 'INFOVAE_MMD'
        self.alpha = self.model_config.alpha
        self.lbd = self.model_config.lbd
        self.kernel_choice = model_config.kernel_choice
        self.scales = model_config.scales if model_config.scales is not None else [1.0]

    def forward(self, inputs: 'BaseDataset', **kwargs) ->ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        x = inputs['data']
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)['reconstruction']
        z_prior = torch.randn_like(z, device=x.device)
        loss, recon_loss, kld_loss, mmd_loss = self.loss_function(recon_x, x, z, z_prior, mu, log_var)
        output = ModelOutput(loss=loss, recon_loss=recon_loss, reg_loss=kld_loss, mmd_loss=mmd_loss, recon_x=recon_x, z=z)
        return output

    def loss_function(self, recon_x, x, z, z_prior, mu, log_var):
        N = z.shape[0]
        if self.model_config.reconstruction_loss == 'mse':
            recon_loss = 0.5 * F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        elif self.model_config.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        if self.kernel_choice == 'rbf':
            k_z = self.rbf_kernel(z, z)
            k_z_prior = self.rbf_kernel(z_prior, z_prior)
            k_cross = self.rbf_kernel(z, z_prior)
        else:
            k_z = self.imq_kernel(z, z)
            k_z_prior = self.imq_kernel(z_prior, z_prior)
            k_cross = self.imq_kernel(z, z_prior)
        mmd_z = (k_z - k_z.diag().diag()).sum() / ((N - 1) * N)
        mmd_z_prior = (k_z_prior - k_z_prior.diag().diag()).sum() / ((N - 1) * N)
        mmd_cross = k_cross.sum() / N ** 2
        mmd_loss = mmd_z + mmd_z_prior - 2 * mmd_cross
        loss = recon_loss + (1 - self.alpha) * KLD + (self.alpha + self.lbd - 1) * mmd_loss
        return loss.mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0), mmd_loss

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def imq_kernel(self, z1, z2):
        """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""
        Cbase = 2.0 * self.model_config.latent_dim * self.model_config.kernel_bandwidth ** 2
        k = 0
        for scale in self.scales:
            C = scale * Cbase
            k += C / (C + torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2)
        return k

    def rbf_kernel(self, z1, z2):
        """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""
        C = 2.0 * self.model_config.latent_dim * self.model_config.kernel_bandwidth ** 2
        k = torch.exp(-torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2 / C)
        return k


class IWAE(VAE):
    """
    Importance Weighted Autoencoder model.

    Args:
        model_config (IWAEConfig): The IWAE configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'IWAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self.model_name = 'IWAE'
        self.n_samples = model_config.number_samples

    def forward(self, inputs: 'BaseDataset', **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        x = inputs['data']
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        mu = mu.unsqueeze(1).repeat(1, self.n_samples, 1)
        log_var = log_var.unsqueeze(1).repeat(1, self.n_samples, 1)
        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z.reshape(-1, self.latent_dim))['reconstruction'].reshape(x.shape[0], self.n_samples, -1)
        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)
        output = ModelOutput(recon_loss=recon_loss, reg_loss=kld, loss=loss, recon_x=recon_x.reshape(x.shape[0], self.n_samples, -1)[:, 0, :].reshape_as(x), z=z.reshape(x.shape[0], self.n_samples, -1)[:, 0, :].reshape(-1, self.latent_dim))
        return output

    def loss_function(self, recon_x, x, mu, log_var, z):
        if self.model_config.reconstruction_loss == 'mse':
            recon_loss = 0.5 * F.mse_loss(recon_x, x.reshape(recon_x.shape[0], -1).unsqueeze(1).repeat(1, self.n_samples, 1), reduction='none').sum(dim=-1)
        elif self.model_config.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x, x.reshape(recon_x.shape[0], -1).unsqueeze(1).repeat(1, self.n_samples, 1), reduction='none').sum(dim=-1)
        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / log_var.exp())).sum(dim=-1)
        log_p_z = -0.5 * (z ** 2).sum(dim=-1)
        KLD = -(log_p_z - log_q_z)
        log_w = -(recon_loss + KLD)
        log_w_minus_max = log_w - log_w.max(1, keepdim=True)[0]
        w = log_w_minus_max.exp()
        w_tilde = (w / w.sum(axis=1, keepdim=True)).detach()
        return -(w_tilde * log_w).sum(1).mean(), recon_loss.mean(), KLD.mean()

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps


class MAF(BaseNF):
    """Masked Autoregressive Flow.

    Args:
        model_config (MAFConfig): The MAF model configuration setting the main parameters of the
            model.
    """

    def __init__(self, model_config: 'MAFConfig'):
        BaseNF.__init__(self, model_config=model_config)
        self.net = []
        self.m = {}
        self.model_config = model_config
        self.hidden_size = model_config.hidden_size
        self.model_name = 'MAF'
        made_config = MADEConfig(input_dim=(self.input_dim,), output_dim=(self.input_dim,), hidden_sizes=[self.hidden_size] * self.model_config.n_hidden_in_made, degrees_ordering='sequential')
        for i in range(model_config.n_made_blocks):
            self.net.extend([MADE(made_config)])
            if self.model_config.include_batch_norm:
                self.net.extend([BatchNorm(self.input_dim)])
        self.net = nn.ModuleList(self.net)

    def forward(self, x: 'torch.Tensor', **kwargs) ->ModelOutput:
        """The input data is transformed toward the prior

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        x = x.reshape(x.shape[0], -1)
        sum_log_abs_det_jac = torch.zeros(x.shape[0])
        for layer in self.net:
            layer_out = layer(x)
            if layer.__class__.__name__ == 'MADE':
                mu, log_var = layer_out.mu, layer_out.log_var
                x = (x - mu) * (-log_var).exp()
                sum_log_abs_det_jac += -log_var.sum(dim=-1)
            else:
                x = layer_out.out
                sum_log_abs_det_jac += layer_out.log_abs_det_jac
            x = x.flip(dims=(1,))
        return ModelOutput(out=x, log_abs_det_jac=sum_log_abs_det_jac)

    def inverse(self, y: 'torch.Tensor', **kwargs) ->ModelOutput:
        """The prior is transformed toward the input data

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        y = y.reshape(y.shape[0], -1)
        sum_log_abs_det_jac = torch.zeros(y.shape[0])
        for layer in self.net[::-1]:
            y = y.flip(dims=(1,))
            if layer.__class__.__name__ == 'MADE':
                x = torch.zeros_like(y)
                for i in range(self.input_dim):
                    layer_out = layer(x.clone())
                    mu, log_var = layer_out.mu, layer_out.log_var
                    x[:, i] = y[:, i] * log_var[:, i].exp() + mu[:, i]
                    sum_log_abs_det_jac += log_var[:, i]
                y = x
            else:
                layer_out = layer.inverse(y)
                y = layer_out.out
                sum_log_abs_det_jac += layer_out.log_abs_det_jac
        return ModelOutput(out=y, log_abs_det_jac=sum_log_abs_det_jac)


class MIWAE(VAE):
    """
    Multiply Importance Weighted Autoencoder model.

    Args:
        model_config (MIWAEConfig): The MIWAE configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'MIWAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self.model_name = 'MIWAE'
        self.gradient_n_estimates = model_config.number_gradient_estimates
        self.n_samples = model_config.number_samples

    def forward(self, inputs: 'BaseDataset', **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        x = inputs['data']
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        mu = mu.unsqueeze(1).unsqueeze(1).repeat(1, self.gradient_n_estimates, self.n_samples, 1)
        log_var = log_var.unsqueeze(1).unsqueeze(1).repeat(1, self.gradient_n_estimates, self.n_samples, 1)
        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z.reshape(-1, self.latent_dim))['reconstruction'].reshape(x.shape[0], self.gradient_n_estimates, self.n_samples, -1)
        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)
        output = ModelOutput(recon_loss=recon_loss, reg_loss=kld, loss=loss, recon_x=recon_x.reshape(x.shape[0], self.gradient_n_estimates, self.n_samples, -1)[:, 0, 0, :].reshape_as(x), z=z.reshape(x.shape[0], self.gradient_n_estimates, self.n_samples, -1)[:, 0, 0, :].reshape(-1, self.latent_dim))
        return output

    def loss_function(self, recon_x, x, mu, log_var, z):
        if self.model_config.reconstruction_loss == 'mse':
            recon_loss = 0.5 * F.mse_loss(recon_x, x.reshape(recon_x.shape[0], -1).unsqueeze(1).unsqueeze(1).repeat(1, self.gradient_n_estimates, self.n_samples, 1), reduction='none').sum(dim=-1)
        elif self.model_config.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x, x.reshape(recon_x.shape[0], -1).unsqueeze(1).unsqueeze(1).repeat(1, self.gradient_n_estimates, self.n_samples, 1), reduction='none').sum(dim=-1)
        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / log_var.exp())).sum(dim=-1)
        log_p_z = -0.5 * (z ** 2).sum(dim=-1)
        KLD = -(log_p_z - log_q_z)
        log_w = -(recon_loss + KLD)
        log_w_minus_max = log_w - log_w.max(1, keepdim=True)[0]
        w = log_w_minus_max.exp()
        w_tilde = (w / w.sum(axis=1, keepdim=True)).detach()
        return -(w_tilde * log_w).sum(1).mean(dim=-1).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps


class MSSSIM(torch.nn.Module):

    def __init__(self, window_size=11):
        super(MSSSIM, self).__init__()
        self.window_size = window_size

    def _gaussian(self, sigma):
        gauss = torch.Tensor([np.exp(-(x - self.window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(self.window_size)])
        return gauss / gauss.sum()

    def _create_window(self, n_dim, channel=1):
        _1D_window = self._gaussian(1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float()
        _3D_window = torch.stack([(_2D_window * x) for x in _1D_window], dim=2).float().unsqueeze(0).unsqueeze(0)
        _2D_window = _2D_window.unsqueeze(0).unsqueeze(0)
        if n_dim == 3:
            return _3D_window.expand(channel, 1, self.window_size, self.window_size, self.window_size).contiguous()
        else:
            return _2D_window.expand(channel, 1, self.window_size, self.window_size).contiguous()

    def ssim(self, img1: 'torch.Tensor', img2: 'torch.Tensor'):
        n_dim = len(img1.shape) - 2
        padd = int(self.window_size / 2)
        if n_dim == 2:
            _, channel, height, width = img1.size()
            convFunction = F.conv2d
        elif n_dim == 3:
            _, channel, height, width, depth = img1.size()
            convFunction = F.conv3d
        window = self._create_window(n_dim=n_dim, channel=channel)
        mu1 = convFunction(img1, window, padding=padd, groups=channel)
        mu2 = convFunction(img2, window, padding=padd, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = convFunction(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = convFunction(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = convFunction(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
        L = 1
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)
        ssim_map = (2 * mu1_mu2 + C1) * v1 / ((mu1_sq + mu2_sq + C1) * v2)
        ret = ssim_map.mean()
        return ret, cs

    def forward(self, img1, img2):
        n_dim = len(img1.shape) - 2
        if min(img1.shape[-n_dim:]) < 4:
            weights = torch.FloatTensor([1.0])
        elif min(img1.shape[-n_dim:]) < 8:
            weights = torch.FloatTensor([0.3222, 0.6778])
        elif min(img1.shape[-n_dim:]) < 16:
            weights = torch.FloatTensor([0.4558, 0.1633, 0.3809])
        elif min(img1.shape[-n_dim:]) < 32:
            weights = torch.FloatTensor([0.3117, 0.3384, 0.2675, 0.0824])
        else:
            weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        levels = weights.size()[0]
        mssim = []
        mcs = []
        pool_size = [2] * n_dim
        if n_dim == 2:
            pool_function = F.avg_pool2d
        elif n_dim == 3:
            pool_function = F.avg_pool3d
        for _ in range(levels):
            sim, cs = self.ssim(img1, img2)
            mssim.append(sim)
            mcs.append(cs)
            img1 = pool_function(img1, pool_size)
            img2 = pool_function(img2, pool_size)
        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2
        pow1 = mcs ** weights
        pow2 = mssim ** weights
        output = torch.prod(pow1[:-1] * pow2[-1])
        return 1 - output


class MSSSIM_VAE(VAE):
    """
    VAE using perseptual similarity metrics model.

    Args:
        model_config (MSSSIM_VAEConfig): The Variational Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'MSSSIM_VAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self.model_name = 'MSSSIM_VAE'
        self.beta = model_config.beta
        self.msssim = MSSSIM(window_size=model_config.window_size)

    def forward(self, inputs: 'BaseDataset', **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        x = inputs['data']
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)['reconstruction']
        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)
        output = ModelOutput(recon_loss=recon_loss, reg_loss=kld, loss=loss, recon_x=recon_x, z=z)
        return output

    def loss_function(self, recon_x, x, mu, log_var, z):
        recon_loss = self.msssim(recon_x, x)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        return (recon_loss + self.beta * KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps


class PIWAE(VAE):
    """
    Partially Importance Weighted Autoencoder model.

    Args:
        model_config (PIWAEConfig): The PIWAE configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'PIWAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self.model_name = 'PIWAE'
        self.gradient_n_estimates = model_config.number_gradient_estimates
        self.n_samples = model_config.number_samples

    def forward(self, inputs: 'BaseDataset', **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        x = inputs['data']
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        mu = mu.unsqueeze(1).unsqueeze(1).repeat(1, self.gradient_n_estimates, self.n_samples, 1)
        log_var = log_var.unsqueeze(1).unsqueeze(1).repeat(1, self.gradient_n_estimates, self.n_samples, 1)
        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z.reshape(-1, self.latent_dim))['reconstruction'].reshape(x.shape[0], self.gradient_n_estimates, self.n_samples, -1)
        miwae_loss, iwae_loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)
        loss = miwae_loss + iwae_loss
        output = ModelOutput(recon_loss=recon_loss, reg_loss=kld, loss=loss, encoder_loss=miwae_loss, decoder_loss=iwae_loss, update_encoder=True, update_decoder=True, recon_x=recon_x.reshape(x.shape[0], self.gradient_n_estimates, self.n_samples, -1)[:, 0, 0, :].reshape_as(x), z=z.reshape(x.shape[0], self.gradient_n_estimates, self.n_samples, -1)[:, 0, 0, :].reshape(-1, self.latent_dim))
        return output

    def loss_function(self, recon_x, x, mu, log_var, z):
        if self.model_config.reconstruction_loss == 'mse':
            recon_loss = 0.5 * F.mse_loss(recon_x, x.reshape(recon_x.shape[0], -1).unsqueeze(1).unsqueeze(1).repeat(1, self.gradient_n_estimates, self.n_samples, 1), reduction='none').sum(dim=-1)
        elif self.model_config.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x, x.reshape(recon_x.shape[0], -1).unsqueeze(1).unsqueeze(1).repeat(1, self.gradient_n_estimates, self.n_samples, 1), reduction='none').sum(dim=-1)
        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / log_var.exp())).sum(dim=-1)
        log_p_z = -0.5 * (z ** 2).sum(dim=-1)
        KLD = -(log_p_z - log_q_z)
        log_w = -(recon_loss + KLD)
        log_w_minus_max = log_w - log_w.max(1, keepdim=True)[0]
        w = log_w_minus_max.exp()
        w_tilde = (w / w.sum(axis=1, keepdim=True)).detach()
        miwae_loss = -(w_tilde * log_w).sum(1).mean(dim=-1)
        log_w_iwae = log_w.reshape(x.shape[0], self.gradient_n_estimates * self.n_samples)
        log_w_iwae_minus_max = log_w_iwae - log_w_iwae.max(1, keepdim=True)[0]
        w_iwae = log_w_iwae_minus_max.exp()
        w_iwae_tilde = (w_iwae / w_iwae.sum(axis=1, keepdim=True)).detach()
        iwae_loss = -(w_iwae_tilde * log_w_iwae).sum(1)
        return miwae_loss.mean(dim=0), iwae_loss.mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps


class MaskedConv2d(nn.Conv2d):

    def __init__(self, mask_type: 'str', in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride: 'int'=1, padding: 'Union[str, int]'=0, dilation: 'int'=1, groups: 'int'=1, bias: 'bool'=True, padding_mode: 'str'='zeros', device=None, dtype=None) ->None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, kH, kW = self.weight.shape
        if mask_type == 'A':
            self.mask[:, :, kH // 2, kW // 2:] = 0
            self.mask[:, :, kH // 2 + 1:] = 0
        else:
            self.mask[:, :, kH // 2, kW // 2 + 1:] = 0
            self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return F.conv2d(input, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation)


class PixelCNN(BaseNF):
    """Pixel CNN model.

    Args:
        model_config (PixelCNNConfig): The PixelCNN model configuration setting the main parameters
            of the model.
    """

    def __init__(self, model_config: 'PixelCNNConfig'):
        BaseNF.__init__(self, model_config=model_config)
        self.model_config = model_config
        self.model_name = 'PixelCNN'
        self.net = []
        pad_shape = model_config.kernel_size // 2
        for i in range(model_config.n_layers):
            if i == 0:
                self.net.extend([nn.Sequential(MaskedConv2d('A', model_config.input_dim[0], 64, model_config.kernel_size, 1, pad_shape), nn.BatchNorm2d(64), nn.ReLU())])
            else:
                self.net.extend([nn.Sequential(MaskedConv2d('B', 64, 64, model_config.kernel_size, 1, pad_shape), nn.BatchNorm2d(64), nn.ReLU())])
        self.net.extend([nn.Conv2d(64, model_config.n_embeddings * model_config.input_dim[0], 1)])
        self.net = nn.Sequential(*self.net)

    def forward(self, inputs: 'BaseDataset', **kwargs) ->ModelOutput:
        """The input data is transformed an output image.

        Args:
            inputs (torch.Tensor): An input tensor image. Be carefull it must be in range
                [0-max_channels_values] (i.e. [0-256] for RGB images) and shaped [B x C x H x W].

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        x = inputs['data']
        out = self.net(x).reshape(x.shape[0], self.model_config.n_embeddings, self.model_config.input_dim[0], x.shape[2], x.shape[3])
        loss = F.cross_entropy(out, x.long())
        return ModelOutput(out=out, loss=loss)


ACTIVATION = {'elu': F.elu, 'tanh': torch.tanh, 'linear': lambda x: x}


ACTIVATION_DERIVATIVES = {'elu': lambda x: torch.ones_like(x) * (x >= 0) + torch.exp(x) * (x < 0), 'tanh': lambda x: 1 - torch.tanh(x) ** 2, 'linear': lambda x: 1}


class PlanarFlow(BaseNF):
    """Planar Flow model.

    Args:
        model_config (PlanarFlowConfig): The PlanarFlow model configuration setting the main parameters of
            the model.
    """

    def __init__(self, model_config: 'PlanarFlowConfig'):
        BaseNF.__init__(self, model_config)
        self.w = nn.Parameter(torch.randn(1, self.input_dim))
        self.u = nn.Parameter(torch.randn(1, self.input_dim))
        self.b = nn.Parameter(torch.randn(1))
        self.activation = ACTIVATION[model_config.activation]
        self.activation_derivative = ACTIVATION_DERIVATIVES[model_config.activation]
        self.model_name = 'PlanarFlow'
        nn.init.normal_(self.w)
        nn.init.normal_(self.u)
        nn.init.normal_(self.b)

    def forward(self, x: 'torch.Tensor', **kwargs) ->ModelOutput:
        """The input data is transformed toward the prior

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        x = x.reshape(x.shape[0], -1)
        lin = x @ self.w.T + self.b
        f = x + self.u * self.activation(lin)
        phi = self.activation_derivative(lin) @ self.w
        log_det = torch.log(torch.abs(1 + phi @ self.u.T) + 0.0001).squeeze()
        output = ModelOutput(out=f, log_abs_det_jac=log_det)
        return output


class Encoder_SVAE_MLP(BaseEncoder):

    def __init__(self, args: 'dict'):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        layers = nn.ModuleList()
        layers.append(nn.Sequential(nn.Linear(np.prod(args.input_dim), 512), nn.ReLU()))
        self.layers = layers
        self.depth = len(layers)
        self.embedding = nn.Linear(512, self.latent_dim)
        self.log_concentration = nn.Linear(512, 1)

    def forward(self, x, output_layer_levels: 'List[int]'=None):
        output = ModelOutput()
        max_depth = self.depth
        if output_layer_levels is not None:
            assert all(self.depth >= levels > 0 or levels == -1 for levels in output_layer_levels), f'Cannot output layer deeper than depth ({self.depth}). Got ({output_layer_levels}).'
            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)
        out = x.reshape(-1, np.prod(self.input_dim))
        for i in range(max_depth):
            out = self.layers[i](out)
            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f'embedding_layer_{i + 1}'] = out
            if i + 1 == self.depth:
                output['embedding'] = self.embedding(out)
                output['log_concentration'] = self.log_concentration(out)
        return output


MIN_NORM = 1e-15


def _gyration(u, v, w, c, dim: 'int'=-1):
    u2 = u.pow(2).sum(dim=dim, keepdim=True)
    v2 = v.pow(2).sum(dim=dim, keepdim=True)
    uv = (u * v).sum(dim=dim, keepdim=True)
    uw = (u * w).sum(dim=dim, keepdim=True)
    vw = (v * w).sum(dim=dim, keepdim=True)
    c2 = c ** 2
    a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
    b = -c2 * vw * u2 - c * uw
    d = 1 + 2 * c * uv + c2 * u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(MIN_NORM)


def _lambda_x(x, c, keepdim: 'bool'=False, dim: 'int'=-1):
    return 2 / (1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim)).clamp_min(MIN_NORM)


def _mobius_add(x, y, c, dim=-1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def artanh(x: 'torch.Tensor'):
    x = x.clamp(-1 + 1e-05, 1 - 1e-05)
    return torch.log_(1 + x).sub_(torch.log_(1 - x)).mul_(0.5)


def tanh(x):
    return x.clamp(-15, 15).tanh()


def _mobius_scalar_mul(r, x, c, dim: 'int'=-1):
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    sqrt_c = c ** 0.5
    res_c = tanh(r * artanh(sqrt_c * x_norm)) * x / (x_norm * sqrt_c)
    return res_c


BALL_EPS = {torch.float32: 0.004, torch.float64: 1e-05}


def _project(x, c, dim: 'int'=-1, eps: 'float'=None):
    norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    if eps is None:
        eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / c ** 0.5
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def arsinh(x: 'torch.Tensor'):
    return (x + torch.sqrt(1 + x.pow(2))).clamp_min(MIN_NORM).log()


class PoincareBall:

    def __init__(self, dim, c=1.0):
        self.c = c
        self.dim = dim

    @property
    def coord_dim(self):
        return int(self.dim)

    @property
    def zero(self):
        return torch.zeros(1, self.dim)

    def dist(self, x: 'torch.Tensor', y: 'torch.Tensor', *, keepdim=False, dim=-1) ->torch.Tensor:
        sqrt_c = self.c ** 0.5
        dist_c = artanh(sqrt_c * _mobius_add(-x, y, self.c, dim=dim).norm(dim=dim, p=2, keepdim=keepdim))
        return dist_c * 2 / sqrt_c

    def lambda_x(self, x: 'torch.Tensor', *, dim=-1, keepdim=False) ->torch.Tensor:
        return _lambda_x(x, c=self.c, dim=dim, keepdim=keepdim)

    def mobius_add(self, x: 'torch.Tensor', y: 'torch.Tensor', *, dim=-1, project=True) ->torch.Tensor:
        res = _mobius_add(x, y, c=self.c, dim=dim)
        if project:
            return _project(res, c=self.c, dim=dim)
        else:
            return res

    def logmap0(self, x: 'torch.Tensor', y: 'torch.Tensor', *, dim=-1) ->torch.Tensor:
        sqrt_c = self.c ** 0.5
        y_norm = y.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
        return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)

    def logmap(self, x: 'torch.Tensor', y: 'torch.Tensor', *, dim=-1) ->torch.Tensor:
        sub = _mobius_add(-x, y, self.c, dim=dim)
        sub_norm = sub.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
        lam = _lambda_x(x, self.c, keepdim=True, dim=dim)
        sqrt_c = self.c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def transp0(self, y: 'torch.Tensor', v: 'torch.Tensor', *, dim=-1) ->torch.Tensor:
        return v * (1 - self.c * y.pow(2).sum(dim=dim, keepdim=True)).clamp_min(MIN_NORM)

    def transp(self, x: 'torch.Tensor', y: 'torch.Tensor', v: 'torch.Tensor', *, dim=-1):
        return _gyration(y, -x, v, self.c, dim=dim) * _lambda_x(x, self.c, keepdim=True, dim=dim) / _lambda_x(y, self.c, keepdim=True, dim=dim)

    def logdetexp(self, x, y, is_vector=False, keepdim=False):
        d = self.norm(x, y, keepdim=keepdim) if is_vector else self.dist(x, y, keepdim=keepdim)
        return (self.dim - 1) * (torch.sinh(math.sqrt(self.c) * d) / math.sqrt(self.c) / d).log()

    def expmap0(self, u, dim: 'int'=-1):
        sqrt_c = self.c ** 0.5
        u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def expmap(self, x, u, dim: 'int'=-1):
        sqrt_c = self.c ** 0.5
        u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
        second_term = tanh(sqrt_c / 2 * _lambda_x(x, self.c, keepdim=True, dim=dim) * u_norm) * u / (sqrt_c * u_norm)
        gamma_1 = _mobius_add(x, second_term, self.c, dim=dim)
        return gamma_1

    def expmap_polar(self, x, u, r, dim: 'int'=-1):
        sqrt_c = self.c ** 0.5
        u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
        second_term = tanh(torch.tensor([sqrt_c]) / 2 * r) * u / (sqrt_c * u_norm)
        gamma_1 = self.mobius_add(x, second_term, dim=dim)
        return gamma_1

    def geodesic(self, t, x, y, dim: 'int'=-1):
        v = _mobius_add(-x, y, self.c, dim=dim)
        tv = _mobius_scalar_mul(t, v, self.c, dim=dim)
        gamma_t = _mobius_add(x, tv, self.c, dim=dim)
        return gamma_t

    def normdist2plane(self, x, a, p, keepdim: 'bool'=False, signed: 'bool'=False, dim: 'int'=-1, norm: 'bool'=False):
        c = self.c
        sqrt_c = c ** 0.5
        diff = self.mobius_add(-p, x, dim=dim)
        diff_norm2 = diff.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(MIN_NORM)
        sc_diff_a = (diff * a).sum(dim=dim, keepdim=keepdim)
        if not signed:
            sc_diff_a = sc_diff_a.abs()
        a_norm = a.norm(dim=dim, keepdim=keepdim, p=2).clamp_min(MIN_NORM)
        num = 2 * sqrt_c * sc_diff_a
        denom = (1 - c * diff_norm2) * a_norm
        res = arsinh(num / denom.clamp_min(MIN_NORM)) / sqrt_c
        if norm:
            res = res * a_norm
        return res

    def _check_point_on_manifold(self, x, *, atol=1e-05, rtol=1e-05):
        px = _project(x, c=self.c)
        ok = torch.allclose(x, px, atol=atol, rtol=rtol)
        if not ok:
            reason = "'x' norm lies out of the bounds [-1/sqrt(c)+eps, 1/sqrt(c)-eps]"
        else:
            reason = None
        return ok, reason

    def _check_vector_on_tangent(self, x: 'torch.Tensor', u: 'torch.Tensor', *, atol=1e-05, rtol=1e-05, dim=-1) ->Tuple[bool, Optional[str]]:
        return True, None


def diff(x):
    return x[:, 1:] - x[:, :-1]


infty = torch.tensor(float('Inf'))


class ARS:
    """
    This class implements the Adaptive Rejection Sampling technique of Gilks and Wild '92.
    Where possible, naming convention has been borrowed from this paper.
    The PDF must be log-concave.
    Currently does not exploit lower hull described in paper- which is fine for drawing
    only small amount of samples at a time.
    """

    def __init__(self, logpdf, grad_logpdf, device, xi, lb=-infty, ub=infty, use_lower=False, ns=50, **fargs):
        """
        initialize the upper (and if needed lower) hulls with the specified params

        Parameters
        ==========
        f: function that computes log(f(u,...)), for given u, where f(u) is proportional to the
           density we want to sample from
        fprima:  d/du log(f(u,...))
        xi: ordered vector of starting points in wich log(f(u,...) is defined
            to initialize the hulls
        use_lower: True means the lower sqeezing will be used; which is more efficient
                   for drawing large numbers of samples


        lb: lower bound of the domain
        ub: upper bound of the domain
        ns: maximum number of points defining the hulls
        fargs: arguments for f and fprima
        """
        self.device = device
        self.lb = lb
        self.ub = ub
        self.logpdf = logpdf
        self.grad_logpdf = grad_logpdf
        self.fargs = fargs
        self.ns = ns
        self.xi = xi
        self.B, self.K = self.xi.size()
        self.h = torch.zeros(self.B, ns)
        self.hprime = torch.zeros(self.B, ns)
        self.x = torch.zeros(self.B, ns)
        self.h[:, :self.K] = self.logpdf(self.xi, **self.fargs)
        self.hprime[:, :self.K] = self.grad_logpdf(self.xi, **self.fargs)
        self.x[:, :self.K] = self.xi
        self.offset = self.h.max(-1)[0].view(-1, 1)
        self.h = self.h - self.offset
        if not (self.hprime[:, 0] > 0).all():
            raise IOError('initial anchor points must span mode of PDF (left)')
        if not (self.hprime[:, self.K - 1] < 0).all():
            raise IOError('initial anchor points must span mode of PDF (right)')
        self.insert()

    def sample(self, shape=torch.Size()):
        """
        Draw N samples and update upper and lower hulls accordingly
        """
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
        samples = torch.ones(self.B, *shape)
        bool_mask = torch.ones(self.B, *shape) == 1
        count = 0
        while bool_mask.sum() != 0:
            count += 1
            xt, i = self.sampleUpper(shape)
            ht = self.logpdf(xt, **self.fargs)
            ht = ht - self.offset
            ut = self.h.gather(1, i) + (xt - self.x.gather(1, i)) * self.hprime.gather(1, i)
            u = torch.rand(shape)
            accept = u < torch.exp(ht - ut)
            reject = ~accept
            samples[bool_mask * accept] = xt[bool_mask * accept]
            bool_mask[bool_mask * accept] = reject[bool_mask * accept]
        return samples.t().unsqueeze(-1)

    def insert(self, nbnew=0, xnew=None, hnew=None, hprimenew=None):
        """
        Update hulls with new point(s) if none given, just recalculate hull from existing x,h,hprime
        #"""
        self.z = torch.zeros(self.B, self.K + 1)
        self.z[:, 0] = self.lb
        self.z[:, self.K] = self.ub
        self.z[:, 1:self.K] = (diff(self.h[:, :self.K]) - diff(self.x[:, :self.K] * self.hprime[:, :self.K])) / -diff(self.hprime[:, :self.K])
        idx = [0] + list(range(self.K))
        self.u = self.h[:, idx] + self.hprime[:, idx] * (self.z - self.x[:, idx])
        self.s = diff(torch.exp(self.u)) / self.hprime[:, :self.K]
        self.s[self.hprime[:, :self.K] == 0.0] = 0.0
        self.cs = torch.cat((torch.zeros(self.B, 1), torch.cumsum(self.s, dim=-1)), dim=-1)
        self.cu = self.cs[:, -1]

    def sampleUpper(self, shape=torch.Size()):
        """
        Return a single value randomly sampled from the upper hull and index of segment
        """
        u = torch.rand(self.B, *shape)
        i = (self.cs / self.cu.unsqueeze(-1)).unsqueeze(-1) <= u.unsqueeze(1).expand(*self.cs.shape, *shape)
        idx = i.sum(1) - 1
        xt = self.x.gather(1, idx) + (-self.h.gather(1, idx) + torch.log(self.hprime.gather(1, idx) * (self.cu.unsqueeze(-1) * u - self.cs.gather(1, idx)) + torch.exp(self.u.gather(1, idx)))) / self.hprime.gather(1, idx)
        return xt, idx


def log_sum_exp_signs(value, signs, dim=0, keepdim=False):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(signs * torch.exp(value0), dim=dim, keepdim=keepdim))


def rexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on right."""
    return A.view(A.shape + (1,) * len(dimensions)).expand(A.shape + tuple(dimensions))


class _log_normalizer_closed_grad(Function):

    @staticmethod
    def forward(ctx, scale, c, dim):
        scale = scale.double()
        c = np.double(c)
        ctx.scale = scale.clone().detach()
        ctx.c = torch.tensor([c])
        ctx.dim = dim
        device = scale.device
        output = 0.5 * (math.log(math.pi) - math.log(2)) + scale.log() - (int(dim) - 1) * (math.log(c) / 2 + math.log(2))
        dim = torch.tensor(int(dim)).double()
        k_float = rexpand(torch.arange(int(dim)), *scale.size()).double()
        s = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 + torch.log1p(torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2)))
        signs = torch.tensor([1.0, -1.0]).double().repeat((int(dim) + 1) // 2 * 2)[:int(dim)]
        signs = rexpand(signs, *scale.size())
        ctx.log_sum_term = log_sum_exp_signs(s, signs, dim=0)
        output = output + ctx.log_sum_term
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        device = grad_input.device
        scale = ctx.scale
        c = ctx.c
        dim = torch.tensor(int(ctx.dim)).double()
        k_float = rexpand(torch.arange(int(dim)), *scale.size()).double()
        signs = torch.tensor([1.0, -1.0]).double().repeat((int(dim) + 1) // 2 * 2)[:int(dim)]
        signs = rexpand(signs, *scale.size())
        log_arg = (dim - 1 - 2 * k_float).pow(2) * c * scale * (1 + torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2))) + torch.exp(-(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2) * 2 / math.sqrt(math.pi) * (dim - 1 - 2 * k_float) * math.sqrt(c) / math.sqrt(2)
        log_arg_signs = torch.sign(log_arg)
        s = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 + torch.log(log_arg_signs * log_arg)
        log_grad_sum_sigma = log_sum_exp_signs(s, log_arg_signs * signs, dim=0)
        grad_scale = torch.exp(log_grad_sum_sigma - ctx.log_sum_term)
        grad_scale = 1 / ctx.scale + grad_scale
        grad_scale = (grad_input * grad_scale.float()).view(-1, *grad_input.shape).sum(0)
        return grad_scale, None, None


def cdf_r(value, scale, c, dim):
    value = value.double()
    scale = scale.double()
    c = np.double(c)
    if dim == 2:
        return 1 / torch.erf(math.sqrt(c) * scale / math.sqrt(2)) * 0.5 * (2 * torch.erf(math.sqrt(c) * scale / math.sqrt(2)) + torch.erf((value - math.sqrt(c) * scale.pow(2)) / math.sqrt(2) / scale) - torch.erf((math.sqrt(c) * scale.pow(2) + value) / math.sqrt(2) / scale))
    else:
        device = value.device
        k_float = rexpand(torch.arange(dim), *value.size()).double()
        dim = torch.tensor(dim).double()
        s1 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 + torch.log(torch.erf((value - (dim - 1 - 2 * k_float) * math.sqrt(c) * scale.pow(2)) / scale / math.sqrt(2)) + torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2)))
        s2 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 + torch.log1p(torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2)))
        signs = torch.tensor([1.0, -1.0]).double().repeat((int(dim) + 1) // 2 * 2)[:int(dim)]
        signs = rexpand(signs, *value.size())
        S1 = log_sum_exp_signs(s1, signs, dim=0)
        S2 = log_sum_exp_signs(s2, signs, dim=0)
        output = torch.exp(S1 - S2)
        zero_value_idx = value == 0.0
        output[zero_value_idx] = 0.0
        return output.float()


def logsinh(x):
    return x + torch.log(1 - torch.exp(-2 * x)) - math.log(2)


def grad_cdf_value_scale(value, scale, c, dim):
    device = value.device
    dim = torch.tensor(int(dim)).double()
    signs = torch.tensor([1.0, -1.0]).double().repeat((int(dim) + 1) // 2 * 2)[:int(dim)]
    signs = rexpand(signs, *value.size())
    k_float = rexpand(torch.arange(dim), *value.size()).double()
    log_arg1 = (dim - 1 - 2 * k_float).pow(2) * c * scale * (torch.erf((value - (dim - 1 - 2 * k_float) * math.sqrt(c) * scale.pow(2)) / scale / math.sqrt(2)) + torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2)))
    log_arg2 = math.sqrt(2 / math.pi) * ((dim - 1 - 2 * k_float) * math.sqrt(c) * torch.exp(-(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2) - (value / scale.pow(2) + (dim - 1 - 2 * k_float) * math.sqrt(c)) * torch.exp(-(value - (dim - 1 - 2 * k_float) * math.sqrt(c) * scale.pow(2)).pow(2) / (2 * scale.pow(2))))
    log_arg = log_arg1 + log_arg2
    sign_log_arg = torch.sign(log_arg)
    s = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 + torch.log(sign_log_arg * log_arg)
    log_grad_sum_sigma = log_sum_exp_signs(s, signs * sign_log_arg, dim=0)
    grad_sum_sigma = torch.sum(signs * sign_log_arg * torch.exp(s), dim=0)
    s1 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 + torch.log(torch.erf((value - (dim - 1 - 2 * k_float) * math.sqrt(c) * scale.pow(2)) / scale / math.sqrt(2)) + torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2)))
    S1 = log_sum_exp_signs(s1, signs, dim=0)
    grad_log_cdf_scale = grad_sum_sigma / S1.exp()
    log_unormalised_prob = -value.pow(2) / (2 * scale.pow(2)) + (dim - 1) * logsinh(math.sqrt(c) * value) - (dim - 1) / 2 * math.log(c)
    with torch.autograd.enable_grad():
        scale = scale.float()
        logZ = _log_normalizer_closed_grad.apply(scale, c, dim)
        grad_logZ_scale = grad(logZ, scale, grad_outputs=torch.ones_like(scale))
    grad_log_cdf_scale = -grad_logZ_scale[0] + 1 / scale + grad_log_cdf_scale.float()
    cdf = cdf_r(value.double(), scale.double(), np.double(c), int(dim)).float().squeeze(0)
    grad_scale = cdf * grad_log_cdf_scale
    grad_value = (log_unormalised_prob.float() - logZ).exp()
    return grad_value, grad_scale


class impl_rsample(Function):

    @staticmethod
    def forward(ctx, value, scale, c, dim):
        ctx.scale = scale.clone().detach().double().requires_grad_(True)
        ctx.value = value.clone().detach().double().requires_grad_(True)
        ctx.c = torch.tensor([c]).double().requires_grad_(True)
        ctx.dim = dim
        return value

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_cdf_value, grad_cdf_scale = grad_cdf_value_scale(ctx.value, ctx.scale, ctx.c, ctx.dim)
        assert not torch.isnan(grad_cdf_value).any()
        assert not torch.isnan(grad_cdf_scale).any()
        grad_value_scale = -grad_cdf_value.pow(-1) * grad_cdf_scale.expand(grad_input.shape)
        grad_scale = (grad_input * grad_value_scale).view(-1, *grad_cdf_scale.shape).sum(0)
        return None, grad_scale, None, None


class HyperbolicRadius(dist.Distribution):
    support = dist.constraints.positive
    has_rsample = True

    def __init__(self, dim, c, scale, ars=True, validate_args=None):
        self.dim = dim
        self.c = c
        self.scale = scale
        self.device = scale.device
        self.ars = ars
        if isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.scale.size()
        self.log_normalizer = self._log_normalizer()
        if torch.isnan(self.log_normalizer).any() or torch.isinf(self.log_normalizer).any():
            None
            raise
        super(HyperbolicRadius, self).__init__(batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        value = self.sample(sample_shape)
        return impl_rsample.apply(value, self.scale, self.c, self.dim)

    def sample(self, sample_shape=torch.Size()):
        if sample_shape == torch.Size():
            sample_shape = torch.Size([1])
        with torch.no_grad():
            mean = self.mean
            stddev = self.stddev
            if torch.isnan(stddev).any():
                stddev[torch.isnan(stddev)] = self.scale[torch.isnan(stddev)]
            if torch.isnan(mean).any():
                mean[torch.isnan(mean)] = ((self.dim - 1) * self.scale.pow(2) * math.sqrt(self.c))[torch.isnan(mean)]
            steps = torch.linspace(0.1, 3, 10)
            steps = torch.cat((-steps.flip(0), steps))
            xi = [(mean + s * torch.min(stddev, 0.95 * mean / 3)) for s in steps]
            xi = torch.cat(xi, dim=1)
            ars = ARS(self.log_prob, self.grad_log_prob, self.device, xi=xi, ns=20, lb=0)
            value = ars.sample(sample_shape)
        return value

    def log_prob(self, value):
        res = -value.pow(2) / (2 * self.scale.pow(2)) + (self.dim - 1) * logsinh(math.sqrt(self.c) * value) - (self.dim - 1) / 2 * math.log(self.c) - self.log_normalizer
        assert not torch.isnan(res).any()
        return res

    def grad_log_prob(self, value):
        res = -value / self.scale.pow(2) + (self.dim - 1) * math.sqrt(self.c) * torch.cosh(math.sqrt(self.c) * value) / torch.sinh(math.sqrt(self.c) * value)
        return res

    def cdf(self, value):
        return cdf_r(value, self.scale, self.c, self.dim)

    @property
    def mean(self):
        c = np.double(self.c)
        scale = self.scale.double()
        dim = torch.tensor(int(self.dim)).double()
        signs = torch.tensor([1.0, -1.0]).double().repeat((self.dim + 1) // 2 * 2)[:self.dim].unsqueeze(-1).unsqueeze(-1).expand(self.dim, *self.scale.size())
        k_float = rexpand(torch.arange(self.dim), *self.scale.size()).double()
        s2 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 + torch.log1p(torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2)))
        S2 = log_sum_exp_signs(s2, signs, dim=0)
        log_arg = (dim - 1 - 2 * k_float) * math.sqrt(c) * scale.pow(2) * (1 + torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2))) + torch.exp(-(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2) * scale * math.sqrt(2 / math.pi)
        log_arg_signs = torch.sign(log_arg)
        s1 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 + torch.log(log_arg_signs * log_arg)
        S1 = log_sum_exp_signs(s1, signs * log_arg_signs, dim=0)
        output = torch.exp(S1 - S2)
        return output.float()

    @property
    def variance(self):
        c = np.double(self.c)
        scale = self.scale.double()
        dim = torch.tensor(int(self.dim)).double()
        signs = torch.tensor([1.0, -1.0]).double().repeat((int(dim) + 1) // 2 * 2)[:int(dim)].unsqueeze(-1).unsqueeze(-1).expand(int(dim), *self.scale.size())
        k_float = rexpand(torch.arange(self.dim), *self.scale.size()).double()
        s2 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 + torch.log1p(torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2)))
        S2 = log_sum_exp_signs(s2, signs, dim=0)
        log_arg = (1 + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2)) * (1 + torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2))) + (dim - 1 - 2 * k_float) * math.sqrt(c) * torch.exp(-(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2) * scale * math.sqrt(2 / math.pi)
        log_arg_signs = torch.sign(log_arg)
        s1 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 + 2 * scale.log() + torch.log(log_arg_signs * log_arg)
        S1 = log_sum_exp_signs(s1, signs * log_arg_signs, dim=0)
        output = torch.exp(S1 - S2)
        output = output.float() - self.mean.pow(2)
        return output

    @property
    def stddev(self):
        return self.variance.sqrt()

    def _log_normalizer(self):
        return _log_normalizer_closed_grad.apply(self.scale, self.c, self.dim)


class HypersphericalUniform(dist.Distribution):
    """Taken from
    https://github.com/nicola-decao/s-vae-pytorch/blob/master/hyperspherical_vae/distributions/von_mises_fisher.py
    """
    support = dist.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim

    def __init__(self, dim, device='cpu', validate_args=None):
        super(HypersphericalUniform, self).__init__(torch.Size([dim]), validate_args=validate_args)
        self._dim = dim
        self._device = device

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = torch.Size([*sample_shape, self._dim + 1])
        output = _standard_normal(shape, dtype=torch.float, device=self._device)
        return output / output.norm(dim=-1, keepdim=True)

    def entropy(self):
        return self.__log_surface_area()

    def log_prob(self, x):
        return -torch.ones(x.shape[:-1]) * self._log_normalizer()

    def _log_normalizer(self):
        return self._log_surface_area()

    def _log_surface_area(self):
        return math.log(2) + (self._dim + 1) / 2 * math.log(math.pi) - torch.lgamma(torch.Tensor([(self._dim + 1) / 2]))


class RiemannianNormal(dist.Distribution):
    support = dist.constraints.interval(-1, 1)
    has_rsample = True

    @property
    def mean(self):
        return self.loc

    def __init__(self, loc, scale, manifold, validate_args=None):
        assert not (torch.isnan(loc).any() or torch.isnan(scale).any())
        self.manifold = manifold
        self.loc = loc
        self.manifold._check_point_on_manifold(self.loc)
        self.scale = scale.clamp(min=0.1, max=7.0)
        self.radius = HyperbolicRadius(manifold.dim, manifold.c, self.scale)
        self.direction = HypersphericalUniform(manifold.dim - 1, device=loc.device)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(RiemannianNormal, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        alpha = self.direction.sample(torch.Size([*shape[:-1]]))
        radius = self.radius.rsample(sample_shape)
        res = self.manifold.expmap_polar(self.loc, alpha, radius)
        return res

    def log_prob(self, value):
        loc = self.loc.expand(value.shape)
        radius_sq = self.manifold.dist(loc, value, keepdim=True).pow(2)
        res = -radius_sq / 2 / self.scale.pow(2) - self.direction._log_normalizer() - self.radius.log_normalizer
        return res


class WrappedNormal(dist.Distribution):
    """Wrapped Normal distribution"""
    arg_constraints = {'loc': dist.constraints.real, 'scale': dist.constraints.positive}
    support = dist.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def scale(self):
        return F.softplus(self._scale) if self.softplus else self._scale

    def __init__(self, loc, scale, manifold, validate_args=None, softplus=False):
        self.dtype = loc.dtype
        self.softplus = softplus
        self.loc, self._scale = broadcast_all(loc, scale)
        self.manifold = manifold
        self.manifold._check_point_on_manifold(self.loc)
        self.device = loc.device
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape, event_shape = torch.Size(), torch.Size()
        else:
            batch_shape = self.loc.shape[:-1]
            event_shape = torch.Size([self.manifold.dim])
        super(WrappedNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        v = self.scale * _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        self.manifold._check_vector_on_tangent(torch.zeros(1, self.manifold.dim), v)
        v = v / self.manifold.lambda_x(torch.zeros(1, self.manifold.dim), keepdim=True)
        u = self.manifold.transp(torch.zeros(1, self.manifold.dim), self.loc, v)
        z = self.manifold.expmap(self.loc, u)
        return z

    def log_prob(self, x):
        shape = x.shape
        loc = self.loc.unsqueeze(0).expand(x.shape[0], *self.batch_shape, self.manifold.coord_dim)
        if len(shape) < len(loc.shape):
            x = x.unsqueeze(1)
        v = self.manifold.logmap(loc, x)
        v = self.manifold.transp(loc, torch.zeros(1, self.manifold.dim), v)
        u = v * self.manifold.lambda_x(torch.zeros(1, self.manifold.dim), keepdim=True)
        norm_pdf = Normal(torch.zeros_like(self.scale), self.scale).log_prob(u).sum(-1, keepdim=True)
        logdetexp = self.manifold.logdetexp(loc, x, keepdim=True)
        result = norm_pdf - logdetexp
        return result


class PoincareVAE(VAE):
    """Poincaré Variational Autoencoder model.

    Args:
        model_config (PoincareVAEConfig): The Poincaré Variational Autoencoder configuration
            setting the main parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'PoincareVAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self.model_name = 'PoincareVAE'
        self.latent_manifold = PoincareBall(dim=model_config.latent_dim, c=model_config.curvature)
        if model_config.prior_distribution == 'riemannian_normal':
            self.prior = RiemannianNormal
        else:
            self.prior = WrappedNormal
        if model_config.posterior_distribution == 'riemannian_normal':
            warnings.warn('Carefull, this model expects the encoder to give a one dimensional `log_concentration` tensor for the Riemannian normal distribution. Make sure the encoder actually outputs this.')
            self.posterior = RiemannianNormal
        else:
            self.posterior = WrappedNormal
        if encoder is None:
            if model_config.posterior_distribution == 'riemannian_normal':
                encoder = Encoder_SVAE_MLP(model_config)
            else:
                encoder = Encoder_VAE_MLP(model_config)
            self.model_config.uses_default_encoder = True
        else:
            self.model_config.uses_default_encoder = False
        self.set_encoder(encoder)
        self._pz_mu = nn.Parameter(torch.zeros(1, model_config.latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=False)

    def forward(self, inputs: 'BaseDataset', **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        x = inputs['data']
        encoder_output = self.encoder(x)
        if self.model_config.posterior_distribution == 'riemannian_normal':
            mu, log_var = encoder_output.embedding, encoder_output.log_concentration
        else:
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        qz_x = self.posterior(loc=mu, scale=std, manifold=self.latent_manifold)
        z = qz_x.rsample(torch.Size([1]))
        recon_x = self.decoder(z.squeeze(0))['reconstruction']
        loss, recon_loss, kld = self.loss_function(recon_x, x, z, qz_x)
        output = ModelOutput(recon_loss=recon_loss, reg_loss=kld, loss=loss, recon_x=recon_x, z=z.squeeze(0))
        return output

    def loss_function(self, recon_x, x, z, qz_x):
        if self.model_config.reconstruction_loss == 'mse':
            recon_loss = 0.5 * F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        elif self.model_config.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        pz = self.prior(loc=self._pz_mu, scale=self._pz_logvar.exp(), manifold=self.latent_manifold)
        KLD = (qz_x.log_prob(z) - pz.log_prob(z)).sum(-1).squeeze(0)
        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def interpolate(self, starting_inputs: 'torch.Tensor', ending_inputs: 'torch.Tensor', granularity: 'int'=10):
        """This function performs a geodesic interpolation in the poincaré disk of the autoencoder
        from starting inputs to ending inputs. It returns the interpolation trajectories.

        Args:
            starting_inputs (torch.Tensor): The starting inputs in the interpolation of shape
                [B x input_dim]
            ending_inputs (torch.Tensor): The starting inputs in the interpolation of shape
                [B x input_dim]
            granularity (int): The granularity of the interpolation.

        Returns:
            torch.Tensor: A tensor of shape [B x granularity x input_dim] containing the
            interpolation trajectories.
        """
        assert starting_inputs.shape[0] == ending_inputs.shape[0], f'The number of starting_inputs should equal the number of ending_inputs. Got {starting_inputs.shape[0]} sampler for starting_inputs and {ending_inputs.shape[0]} for endinging_inputs.'
        starting_z = self.encoder(starting_inputs).embedding
        ending_z = self.encoder(ending_inputs).embedding
        t = torch.linspace(0, 1, granularity)
        inter_geo = torch.zeros(starting_inputs.shape[0], granularity, starting_z.shape[-1])
        for i, t_i in enumerate(t):
            z_i = self.latent_manifold.geodesic(t_i, starting_z, ending_z)
            inter_geo[:, i, :] = z_i
        decoded_geo = self.decoder(inter_geo.reshape((starting_z.shape[0] * t.shape[0],) + starting_z.shape[1:])).reconstruction.reshape((starting_inputs.shape[0], t.shape[0]) + starting_inputs.shape[1:])
        return decoded_geo

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """
        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size
        log_p = []
        for i in range(len(data)):
            x = data[i].unsqueeze(0)
            log_p_x = []
            for j in range(n_full_batch):
                x_rep = torch.cat(batch_size * [x])
                encoder_output = self.encoder(x_rep)
                if self.model_config.posterior_distribution == 'riemannian_normal':
                    mu, log_var = encoder_output.embedding, encoder_output.log_concentration
                else:
                    mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                std = torch.exp(0.5 * log_var)
                qz_x = self.posterior(loc=mu, scale=std, manifold=self.latent_manifold)
                z = qz_x.rsample(torch.Size([1]))
                pz = self.prior(loc=self._pz_mu, scale=self._pz_logvar.exp(), manifold=self.latent_manifold)
                log_q_z_given_x = qz_x.log_prob(z).sum(-1).squeeze(0)
                log_p_z = pz.log_prob(z).sum(-1).squeeze(0)
                recon_x = self.decoder(z.squeeze(0))['reconstruction']
                if self.model_config.reconstruction_loss == 'mse':
                    log_p_x_given_z = -0.5 * F.mse_loss(recon_x.reshape(x_rep.shape[0], -1), x_rep.reshape(x_rep.shape[0], -1), reduction='none').sum(dim=-1) - torch.tensor([np.prod(self.input_dim) / 2 * np.log(np.pi * 2)])
                elif self.model_config.reconstruction_loss == 'bce':
                    log_p_x_given_z = -F.binary_cross_entropy(recon_x.reshape(x_rep.shape[0], -1), x_rep.reshape(x_rep.shape[0], -1), reduction='none').sum(dim=-1)
                log_p_x.append(log_p_x_given_z + log_p_z - log_q_z_given_x)
            log_p_x = torch.cat(log_p_x)
            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
        return np.mean(log_p)


class RAE_GP(AE):
    """Regularized Autoencoder with gradient penalty model.

    Args:
        model_config (RAE_GP_Config): The Autoencoder configuration setting the main parameters of the
            model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'RAE_GP_Config', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        AE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self.model_name = 'RAE_GP'

    def forward(self, inputs: 'BaseDataset', **kwargs) ->ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        x = inputs['data'].requires_grad_(True)
        z = self.encoder(x).embedding
        recon_x = self.decoder(z)['reconstruction']
        loss, recon_loss, gen_reg_loss, embedding_loss = self.loss_function(recon_x, x, z)
        output = ModelOutput(loss=loss, recon_loss=recon_loss, gen_reg_loss=gen_reg_loss, embedding_loss=embedding_loss, recon_x=recon_x, z=z)
        return output

    def loss_function(self, recon_x, x, z):
        recon_loss = F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        gen_reg_loss = self._compute_gp(recon_x, x)
        embedding_loss = 0.5 * torch.linalg.norm(z, dim=-1) ** 2
        return (recon_loss + self.model_config.embedding_weight * embedding_loss + self.model_config.reg_weight * gen_reg_loss).mean(dim=0), recon_loss.mean(dim=0), gen_reg_loss.mean(dim=0), embedding_loss.mean(dim=0)

    def _compute_gp(self, recon_x, x):
        grads = torch.autograd.grad(outputs=recon_x, inputs=x, grad_outputs=torch.ones_like(recon_x), create_graph=True, retain_graph=True)[0].reshape(recon_x.shape[0], -1)
        return grads.norm(dim=-1) ** 2


class RAE_L2(AE):
    """Regularized Autoencoder with L2 decoder params regularization model.

    Args:
        model_config (RAE_L2_Config): The Autoencoder configuration setting the main parameters of the
            model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'RAE_L2_Config', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None):
        AE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self.model_name = 'RAE_L2'

    def forward(self, inputs: 'BaseDataset', **kwargs) ->ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        x = inputs['data']
        z = self.encoder(x).embedding
        recon_x = self.decoder(z)['reconstruction']
        loss, recon_loss, embedding_loss = self.loss_function(recon_x, x, z)
        output = ModelOutput(loss=loss, encoder_loss=loss, decoder_loss=loss, update_encoder=True, update_decoder=True, recon_loss=recon_loss, embedding_loss=embedding_loss, recon_x=recon_x, z=z)
        return output

    def loss_function(self, recon_x, x, z):
        recon_loss = F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        embedding_loss = 0.5 * torch.linalg.norm(z, dim=-1) ** 2
        return (recon_loss + self.model_config.embedding_weight * embedding_loss).mean(dim=0), recon_loss.mean(dim=0), embedding_loss.mean(dim=0)


class BaseMetric(nn.Module):
    """This is a base class for Metrics neural networks
    (only applicable for Riemannian based VAE)
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        """This function must be implemented in a child class.
        It takes the input data and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own metric network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseMetric
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_Metric(BaseMetric):
            ...
            ...    def __init__(self):
            ...        BaseMetric.__init__(self)
            ...        # your code
            ...
            ...    def forward(self, x: torch.Tensor):
            ...        # your code
            ...        output = ModelOutput(
            ...             L=L # L matrices in the metric of  Riemannian based VAE (see docs)
            ...         )
            ...        return output

        Parameters:
            x (torch.Tensor): The input data that must be encoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the metric
        """
        raise NotImplementedError()


class Metric_MLP(BaseMetric):

    def __init__(self, args: 'dict'):
        BaseMetric.__init__(self)
        if args.input_dim is None:
            raise AttributeError("No input dimension provided !'input_dim' parameter of ModelConfig instance must be set to 'data_shape' where the shape of the data is [mini_batch x data_shape]. Unable to build metric automatically")
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.layers = nn.Sequential(nn.Linear(np.prod(args.input_dim), 400), nn.ReLU())
        self.diag = nn.Linear(400, self.latent_dim)
        k = int(self.latent_dim * (self.latent_dim - 1) / 2)
        self.lower = nn.Linear(400, k)

    def forward(self, x):
        h1 = self.layers(x.reshape(-1, np.prod(self.input_dim)))
        h21, h22 = self.diag(h1), self.lower(h1)
        L = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim))
        indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=-1)
        L[:, indices[0], indices[1]] = h22
        L = L + torch.diag_embed(h21.exp())
        output = ModelOutput(L=L)
        return output


def create_inverse_metric(model):

    def G_inv(z):
        return (model.M_tens.unsqueeze(0) * torch.exp(-torch.norm(model.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2 / model.temperature ** 2).unsqueeze(-1).unsqueeze(-1)).sum(dim=1) + model.lbd * torch.eye(model.latent_dim)
    return G_inv


def create_metric(model):

    def G(z):
        return torch.inverse((model.M_tens.unsqueeze(0) * torch.exp(-torch.norm(model.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2 / model.temperature ** 2).unsqueeze(-1).unsqueeze(-1)).sum(dim=1) + model.lbd * torch.eye(model.latent_dim))
    return G


class RHVAE(VAE):
    """
    Riemannian Hamiltonian VAE model.


    Args:
        model_config (RHVAEConfig): A model configuration setting the main parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.


    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(self, model_config: 'RHVAEConfig', encoder: 'Optional[BaseEncoder]'=None, decoder: 'Optional[BaseDecoder]'=None, metric: 'Optional[BaseMetric]'=None):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        self.model_name = 'RHVAE'
        if metric is None:
            metric = Metric_MLP(model_config)
            self.model_config.uses_default_metric = True
        else:
            self.model_config.uses_default_metric = False
        self.set_metric(metric)
        self.temperature = nn.Parameter(torch.Tensor([model_config.temperature]), requires_grad=False)
        self.lbd = nn.Parameter(torch.Tensor([model_config.regularization]), requires_grad=False)
        self.beta_zero_sqrt = nn.Parameter(torch.Tensor([model_config.beta_zero]), requires_grad=False)
        self.n_lf = model_config.n_lf
        self.eps_lf = model_config.eps_lf
        self.M = deque(maxlen=100)
        self.centroids = deque(maxlen=100)
        self.M_tens = torch.randn(1, self.model_config.latent_dim, self.model_config.latent_dim)
        self.centroids_tens = torch.randn(1, self.model_config.latent_dim)

        def G(z):
            return torch.inverse((torch.eye(self.latent_dim, device=z.device).unsqueeze(0) * torch.exp(-torch.norm(z.unsqueeze(1), dim=-1) ** 2).unsqueeze(-1).unsqueeze(-1)).sum(dim=1) + self.lbd * torch.eye(self.latent_dim))

        def G_inv(z):
            return (torch.eye(self.latent_dim, device=z.device).unsqueeze(0) * torch.exp(-torch.norm(z.unsqueeze(1), dim=-1) ** 2).unsqueeze(-1).unsqueeze(-1)).sum(dim=1) + self.lbd * torch.eye(self.latent_dim)
        self.G = G
        self.G_inv = G_inv

    def update(self):
        """
        As soon as the model has seen all the data points (i.e. at the end of 1 loop)
        we update the final metric function using \\mu(x_i) as centroids
        """
        self._update_metric()

    def set_metric(self, metric: 'BaseMetric') ->None:
        """This method is called to set the metric network outputing the
        :math:`L_{\\psi_i}` of the metric matrices

        Args:
            metric (BaseMetric): The metric module that need to be set to the model.

        """
        if not issubclass(type(metric), BaseMetric):
            raise BadInheritanceError('Metric must inherit from BaseMetric class from pythae.models.base_architectures.BaseMetric. Refer to documentation.')
        self.metric = metric

    def forward(self, inputs: 'BaseDataset', **kwargs) ->ModelOutput:
        """
        The input data is first encoded. The reparametrization is used to produce a sample
        :math:`z_0` from the approximate posterior :math:`q_{\\phi}(z|x)`. Then Riemannian
        Hamiltonian equations are solved using the generalized leapfrog integrator. In the meantime,
        the input data :math:`x` is fed to the metric network outputing the matrices
        :math:`L_{\\psi}`. The metric is computed and used with the integrator.

        Args:
            inputs (BaseDataset): The training data with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        x = inputs['data']
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z0, eps0 = self._sample_gauss(mu, std)
        z = z0
        if self.training:
            L = self.metric(x)['L']
            M = L @ torch.transpose(L, 1, 2)
            self.M.append(M.detach().clone())
            self.centroids.append(mu.detach().clone())
            G_inv = (M.unsqueeze(0) * torch.exp(-torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2 / self.temperature ** 2).unsqueeze(-1).unsqueeze(-1)).sum(dim=1) + self.lbd * torch.eye(self.latent_dim)
        else:
            G = self.G(z)
            G_inv = self.G_inv(z)
            L = torch.linalg.cholesky(G)
        G_log_det = -torch.logdet(G_inv)
        gamma = torch.randn_like(z0, device=x.device)
        rho = gamma / self.beta_zero_sqrt
        beta_sqrt_old = self.beta_zero_sqrt
        rho = (L @ rho.unsqueeze(-1)).squeeze(-1)
        recon_x = self.decoder(z)['reconstruction']
        for k in range(self.n_lf):
            rho_ = self._leap_step_1(recon_x, x, z, rho, G_inv, G_log_det)
            z = self._leap_step_2(recon_x, x, z, rho_, G_inv, G_log_det)
            recon_x = self.decoder(z)['reconstruction']
            if self.training:
                G_inv = (M.unsqueeze(0) * torch.exp(-torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2 / self.temperature ** 2).unsqueeze(-1).unsqueeze(-1)).sum(dim=1) + self.lbd * torch.eye(self.latent_dim)
            else:
                G = self.G(z)
                G_inv = self.G_inv(z)
            G_log_det = -torch.logdet(G_inv)
            rho__ = self._leap_step_3(recon_x, x, z, rho_, G_inv, G_log_det)
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = beta_sqrt_old / beta_sqrt * rho__
            beta_sqrt_old = beta_sqrt
        loss = self.loss_function(recon_x, x, z0, z, rho, eps0, gamma, mu, log_var, G_inv, G_log_det)
        output = ModelOutput(loss=loss, recon_x=recon_x, z=z, z0=z0, rho=rho, eps0=eps0, gamma=gamma, mu=mu, log_var=log_var, G_inv=G_inv, G_log_det=G_log_det)
        return output

    def predict(self, inputs: 'torch.Tensor') ->ModelOutput:
        """The input data is encoded and decoded without computing loss

        Args:
            inputs (torch.Tensor): The input data to be reconstructed, as well as to generate the embedding.

        Returns:
            ModelOutput: An instance of ModelOutput containing reconstruction, raw embedding (output of encoder), and the final embedding (output of metric)
        """
        encoder_output = self.encoder(inputs)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z0, _ = self._sample_gauss(mu, std)
        z = z0
        G = self.G(z)
        G_inv = self.G_inv(z)
        L = torch.linalg.cholesky(G)
        G_log_det = -torch.logdet(G_inv)
        gamma = torch.randn_like(z0, device=inputs.device)
        rho = gamma / self.beta_zero_sqrt
        beta_sqrt_old = self.beta_zero_sqrt
        rho = (L @ rho.unsqueeze(-1)).squeeze(-1)
        recon_x = self.decoder(z)['reconstruction']
        for k in range(self.n_lf):
            rho_ = self._leap_step_1(recon_x, inputs, z, rho, G_inv, G_log_det)
            z = self._leap_step_2(recon_x, inputs, z, rho_, G_inv, G_log_det)
            recon_x = self.decoder(z)['reconstruction']
            G = self.G(z)
            G_inv = self.G_inv(z)
            G_log_det = -torch.logdet(G_inv)
            rho__ = self._leap_step_3(recon_x, inputs, z, rho_, G_inv, G_log_det)
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = beta_sqrt_old / beta_sqrt * rho__
            beta_sqrt_old = beta_sqrt
        output = ModelOutput(recon_x=recon_x, raw_embedding=encoder_output.embedding, embedding=z if self.n_lf > 0 else encoder_output.embedding)
        return output

    def _leap_step_1(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves first equation of generalized leapfrog integrator
        using fixed point iterations
        """

        def f_(rho_):
            H = self._hamiltonian(recon_x, x, z, rho_, G_inv, G_log_det)
            gz = grad(H, z, retain_graph=True)[0]
            return rho - 0.5 * self.eps_lf * gz
        rho_ = rho.clone()
        for _ in range(steps):
            rho_ = f_(rho_)
        return rho_

    def _leap_step_2(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves second equation of generalized leapfrog integrator
        using fixed point iterations
        """
        H0 = self._hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        grho_0 = grad(H0, rho)[0]

        def f_(z_):
            H = self._hamiltonian(recon_x, x, z_, rho, G_inv, G_log_det)
            grho = grad(H, rho, retain_graph=True)[0]
            return z + 0.5 * self.eps_lf * (grho_0 + grho)
        z_ = z.clone()
        for _ in range(steps):
            z_ = f_(z_)
        return z_

    def _leap_step_3(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves third equation of generalized leapfrog integrator
        """
        H = self._hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        gz = grad(H, z, create_graph=True)[0]
        return rho - 0.5 * self.eps_lf * gz

    def _hamiltonian(self, recon_x, x, z, rho, G_inv=None, G_log_det=None):
        """
        Computes the Hamiltonian function.
        used for RHVAE
        """
        norm = (torch.transpose(rho.unsqueeze(-1), 1, 2) @ G_inv @ rho.unsqueeze(-1)).sum()
        return -self._log_p_xz(recon_x, x, z).sum() + 0.5 * norm + 0.5 * G_log_det.sum()

    def _update_metric(self):
        self.M_tens = torch.cat(list(self.M))
        self.centroids_tens = torch.cat(list(self.centroids))

        def G(z):
            return torch.inverse((self.M_tens.unsqueeze(0) * torch.exp(-torch.norm(self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2 / self.temperature ** 2).unsqueeze(-1).unsqueeze(-1)).sum(dim=1) + self.lbd * torch.eye(self.latent_dim))

        def G_inv(z):
            return (self.M_tens.unsqueeze(0) * torch.exp(-torch.norm(self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2 / self.temperature ** 2).unsqueeze(-1).unsqueeze(-1)).sum(dim=1) + self.lbd * torch.eye(self.latent_dim)
        self.G = G
        self.G_inv = G_inv
        self.M = deque(maxlen=100)
        self.centroids = deque(maxlen=100)

    def loss_function(self, recon_x, x, z0, zK, rhoK, eps0, gamma, mu, log_var, G_inv, G_log_det):
        logpxz = self._log_p_xz(recon_x, x, zK)
        logrhoK = -0.5 * (torch.transpose(rhoK.unsqueeze(-1), 1, 2) @ G_inv @ rhoK.unsqueeze(-1)).squeeze().squeeze() - 0.5 * G_log_det
        logp = logpxz + logrhoK
        normal = torch.distributions.MultivariateNormal(loc=torch.zeros(self.latent_dim), covariance_matrix=torch.eye(self.latent_dim))
        logq = normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)
        return -(logp - logq).mean(dim=0)

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def _tempering(self, k, K):
        """Perform tempering step"""
        beta_k = (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2 + 1 / self.beta_zero_sqrt
        return 1 / beta_k

    def _log_p_x_given_z(self, recon_x, x):
        if self.model_config.reconstruction_loss == 'mse':
            recon_loss = -0.5 * F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
            -torch.log(torch.tensor([2 * np.pi])) * np.prod(self.input_dim) / 2
        elif self.model_config.reconstruction_loss == 'bce':
            recon_loss = -F.binary_cross_entropy(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').sum(dim=-1)
        return recon_loss

    def _log_z(self, z):
        """
        Return Normal density function as prior on z
        """
        normal = torch.distributions.MultivariateNormal(loc=torch.zeros(self.latent_dim), covariance_matrix=torch.eye(self.latent_dim))
        return normal.log_prob(z)

    def _log_p_xz(self, recon_x, x, z):
        """
        Estimate log(p(x, z)) using Bayes rule
        """
        logpxz = self._log_p_x_given_z(recon_x, x)
        logpz = self._log_z(z)
        return logpxz + logpz

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """
        normal = torch.distributions.MultivariateNormal(loc=torch.zeros(self.model_config.latent_dim), covariance_matrix=torch.eye(self.model_config.latent_dim))
        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size
        log_p = []
        for i in range(len(data)):
            x = data[i].unsqueeze(0)
            log_p_x = []
            for j in range(n_full_batch):
                x_rep = torch.cat(batch_size * [x])
                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                std = torch.exp(0.5 * log_var)
                z0, eps = self._sample_gauss(mu, std)
                gamma = torch.randn_like(z0, device=x.device)
                rho = gamma / self.beta_zero_sqrt
                z = z0
                beta_sqrt_old = self.beta_zero_sqrt
                G = self.G(z0)
                G_inv = self.G_inv(z0)
                G_log_det = -torch.logdet(G_inv)
                L = torch.linalg.cholesky(G)
                gamma = torch.randn_like(z0, device=z.device)
                rho = gamma / self.beta_zero_sqrt
                beta_sqrt_old = self.beta_zero_sqrt
                rho = (L @ rho.unsqueeze(-1)).squeeze(-1)
                recon_x = self.decoder(z)['reconstruction']
                for k in range(self.n_lf):
                    rho_ = self._leap_step_1(recon_x, x_rep, z, rho, G_inv, G_log_det)
                    z = self._leap_step_2(recon_x, x_rep, z, rho_, G_inv, G_log_det)
                    recon_x = self.decoder(z)['reconstruction']
                    G_inv = self.G_inv(z)
                    G_log_det = -torch.logdet(G_inv)
                    rho__ = self._leap_step_3(recon_x, x_rep, z, rho_, G_inv, G_log_det)
                    beta_sqrt = self._tempering(k + 1, self.n_lf)
                    rho = beta_sqrt_old / beta_sqrt * rho__
                    beta_sqrt_old = beta_sqrt
                log_q_z0_given_x = -0.5 * (log_var + (z0 - mu) ** 2 / torch.exp(log_var)).sum(dim=-1)
                log_p_z = -0.5 * (z ** 2).sum(dim=-1)
                log_p_rho0 = normal.log_prob(gamma) - torch.logdet(L / self.beta_zero_sqrt)
                log_p_rho = -0.5 * (torch.transpose(rho.unsqueeze(-1), 1, 2) @ G_inv @ rho.unsqueeze(-1)).squeeze().squeeze() - 0.5 * G_log_det - torch.log(torch.tensor([2 * np.pi])) * self.latent_dim / 2
                if self.model_config.reconstruction_loss == 'mse':
                    log_p_x_given_z = -0.5 * F.mse_loss(recon_x.reshape(x_rep.shape[0], -1), x_rep.reshape(x_rep.shape[0], -1), reduction='none').sum(dim=-1) - torch.tensor([np.prod(self.input_dim) / 2 * np.log(np.pi * 2)])
                elif self.model_config.reconstruction_loss == 'bce':
                    log_p_x_given_z = -F.binary_cross_entropy(recon_x.reshape(x_rep.shape[0], -1), x_rep.reshape(x_rep.shape[0], -1), reduction='none').sum(dim=-1)
                log_p_x.append(log_p_x_given_z + log_p_z + log_p_rho - log_p_rho0 - log_q_z0_given_x)
            log_p_x = torch.cat(log_p_x)
            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
        return np.mean(log_p)

    def save(self, dir_path: 'str'):
        """Method to save the model at a specific location

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """
        super().save(dir_path)
        model_path = dir_path
        model_dict = {'M': deepcopy(self.M_tens.clone().detach()), 'centroids': deepcopy(self.centroids_tens.clone().detach()), 'model_state_dict': deepcopy(self.state_dict())}
        if not self.model_config.uses_default_metric:
            with open(os.path.join(model_path, 'metric.pkl'), 'wb') as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.metric))
                cloudpickle.dump(self.metric, fp)
        torch.save(model_dict, os.path.join(model_path, 'model.pt'))

    @classmethod
    def _load_custom_metric_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)
        if 'metric.pkl' not in file_list:
            raise FileNotFoundError(f"Missing metric pkl file ('metric.pkl') in{dir_path}... This file is needed to rebuild custom metrics. Cannot perform model building.")
        else:
            with open(os.path.join(dir_path, 'metric.pkl'), 'rb') as fp:
                metric = CPU_Unpickler(fp).load()
        return metric

    @classmethod
    def _load_metric_matrices_and_centroids(cls, dir_path):
        """this function can be called safely since it is called after
        _load_model_weights_from_folder which handles FileNotFoundError and
        loading issues"""
        path_to_model_weights = os.path.join(dir_path, 'model.pt')
        model_weights = torch.load(path_to_model_weights, map_location='cpu')
        if 'M' not in model_weights.keys():
            raise KeyError(f"Metric M matrices are not available in 'model.pt' file. Got keys:{model_weights.keys()}. These are needed to build the metric.")
        metric_M = model_weights['M']
        if 'centroids' not in model_weights.keys():
            raise KeyError(f"Metric centroids are not available in 'model.pt' file. Got keys:{model_weights.keys()}. These are needed to build the metric.")
        metric_centroids = model_weights['centroids']
        return metric_M, metric_centroids

    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl`` or/and ``metric.pkl``) if a custom encoder (resp. decoder or/and
                metric) was provided
        """
        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)
        if not model_config.uses_default_encoder:
            encoder = cls._load_custom_encoder_from_folder(dir_path)
        else:
            encoder = None
        if not model_config.uses_default_decoder:
            decoder = cls._load_custom_decoder_from_folder(dir_path)
        else:
            decoder = None
        if not model_config.uses_default_metric:
            metric = cls._load_custom_metric_from_folder(dir_path)
        else:
            metric = None
        model = cls(model_config, encoder=encoder, decoder=decoder, metric=metric)
        metric_M, metric_centroids = cls._load_metric_matrices_and_centroids(dir_path)
        model.M_tens = metric_M
        model.centroids_tens = metric_centroids
        model.G = create_metric(model)
        model.G_inv = create_inverse_metric(model)
        model.load_state_dict(model_weights)
        return model

    @classmethod
    def load_from_hf_hub(cls, hf_hub_path: 'str', allow_pickle: 'bool'=False):
        """Class method to be used to load a pretrained model from the Hugging Face hub

        Args:
            hf_hub_path (str): The path where the model should have been be saved on the
                hugginface hub.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl`` and ``metric.pkl``) if a custom encoder (resp. decoder and/or
                metric) was provided
        """
        if not hf_hub_is_available():
            raise ModuleNotFoundError('`huggingface_hub` package must be installed to load models from the HF hub. Run `python -m pip install huggingface_hub` and log in to your account with `huggingface-cli login`.')
        logger.info(f'Downloading {cls.__name__} files for rebuilding...')
        config_path = hf_hub_download(repo_id=hf_hub_path, filename='model_config.json')
        dir_path = os.path.dirname(config_path)
        _ = hf_hub_download(repo_id=hf_hub_path, filename='model.pt')
        model_config = cls._load_model_config_from_folder(dir_path)
        if cls.__name__ + 'Config' != model_config.name and cls.__name__ + '_Config' != model_config.name:
            warnings.warn(f'You are trying to load a `{cls.__name__}` while a `{model_config.name}` is given.')
        model_weights = cls._load_model_weights_from_folder(dir_path)
        if (not model_config.uses_default_encoder or not model_config.uses_default_decoder or not model_config.uses_default_metric) and not allow_pickle:
            warnings.warn('You are about to download pickled files from the HF hub that may have been created by a third party and so could potentially harm your computer. If you are sure that you want to download them set `allow_pickle=true`.')
        else:
            if not model_config.uses_default_encoder:
                _ = hf_hub_download(repo_id=hf_hub_path, filename='encoder.pkl')
                encoder = cls._load_custom_encoder_from_folder(dir_path)
            else:
                encoder = None
            if not model_config.uses_default_decoder:
                _ = hf_hub_download(repo_id=hf_hub_path, filename='decoder.pkl')
                decoder = cls._load_custom_decoder_from_folder(dir_path)
            else:
                decoder = None
            if not model_config.uses_default_metric:
                _ = hf_hub_download(repo_id=hf_hub_path, filename='metric.pkl')
                metric = cls._load_custom_metric_from_folder(dir_path)
            else:
                metric = None
            logger.info(f'Successfully downloaded {cls.__name__} model!')
            model = cls(model_config, encoder=encoder, decoder=decoder, metric=metric)
            metric_M, metric_centroids = cls._load_metric_matrices_and_centroids(dir_path)
            model.M_tens = metric_M
            model.centroids_tens = metric_centroids
            model.G = create_metric(model)
            model.G_inv = create_inverse_metric(model)
            model.load_state_dict(model_weights)
            return model


class RadialFlow(BaseNF):
    """Radial Flow model.

    Args:
        model_config (RadialFlowConfig): The RadialFlow model configuration setting the main parameters of
            the model.
    """

    def __init__(self, model_config: 'RadialFlowConfig'):
        BaseNF.__init__(self, model_config)
        self.x0 = nn.Parameter(torch.randn(1, self.input_dim))
        self.log_alpha = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))
        self.model_name = 'RadialFlow'
        nn.init.normal_(self.x0)
        nn.init.normal_(self.log_alpha)
        nn.init.normal_(self.beta)

    def forward(self, x: 'torch.Tensor', **kwargs) ->ModelOutput:
        """The input data is transformed toward the prior

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        x = x.reshape(x.shape[0], -1)
        x_sub = x - self.x0
        alpha = torch.exp(self.log_alpha)
        beta = -alpha + torch.log(1 + self.beta.exp())
        r = torch.norm(x_sub, dim=-1, keepdim=True)
        h = 1 / (alpha + r)
        f = x + beta * h * x_sub
        log_det = (self.input_dim - 1) * torch.log(1 + beta * h) + torch.log(1 + beta * h - beta * r / (alpha + r) ** 2)
        output = ModelOutput(out=f, log_abs_det_jac=log_det.squeeze())
        return output

