
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


import numpy


import collections


import time


import itertools


from torch.distributions import Exponential as tExponential


import math


from numpy.testing import assert_array_almost_equal


from numpy.testing import assert_array_equal


def _cast_as_tensor(value, dtype=None):
    """Set the parameter."""
    if value is None:
        return None
    if isinstance(value, (torch.nn.Parameter, torch.Tensor, torch.masked.MaskedTensor)):
        if dtype is None:
            return value
        elif value.dtype == dtype:
            return value
        else:
            return value.type(dtype)
    if isinstance(value, list) and all(isinstance(v, numpy.ndarray) for v in value):
        value = numpy.array(value)
    if isinstance(value, (float, int, list, tuple, numpy.ndarray)):
        if dtype is None:
            return torch.tensor(value)
        else:
            return torch.tensor(value, dtype=dtype)


def _check_parameter(parameter, name, min_value=None, max_value=None, value_sum=None, value_sum_dim=None, value_set=None, dtypes=None, ndim=None, shape=None, check_parameter=True, epsilon=1e-06):
    """Ensures that the parameter falls within a valid range.

	This check accepts several optional conditions that can be used to ensure
	that the value has the desired properties. If these conditions are set to
	`None`, they are not checked. 

	Note: `parameter` can be a single value or it can be a whole tensor/array
	of values. These checks are run either against the entire tensor, e.g.
	ndims, or against each of the values in the parameter.

	
	Parameters
	----------
	parameter: anything
		The parameter meant to be checked

	name: str
		The name of the parameter for error logging purposes.

	min_value: float or None, optional
		The minimum numeric value that any values in the parameter can take.
		Default is None.

	max_value: float or None, optional
		The maximum numeric value that any values in the parameter can take.
		Default is None.

	value_sum: float or None, optional
		The approximate sum, within eps, of the parameter. Default is None.

	value_sum_dim: int or float, optional
		What dimension to sum over. If None, sum over entire tensor. Default
		is None.

	value_set: tuple or list or set or None, optional
		The set of values that each element in the parameter can take. Default
		is None.

	dtypes: tuple or list or set or None, optional
		The set of dtypes that the parameter can take. Default is None.

	ndim: int or list or tuple or None, optional
		The number of dimensions of the tensor. Should not be used when the
		parameter is a single value. Default is None.

	shape: tuple or None, optional
		The shape of the parameter. -1 can be used to accept any value for that
		dimension.

	epsilon: float, optional
		When using `value_sum`, this is the maximum difference acceptable.
		Default is 1e-6.
	"""
    vector = numpy.ndarray, torch.Tensor, torch.nn.Parameter
    if parameter is None:
        return None
    if check_parameter == False:
        return parameter
    if dtypes is not None:
        if isinstance(parameter, vector):
            if parameter.dtype not in dtypes:
                raise ValueError('Parameter {} dtype must be one of {}'.format(name, dtypes))
        elif type(parameter) not in dtypes:
            raise ValueError('Parameter {} dtype must be one of {}'.format(name, dtypes))
    if min_value is not None:
        if isinstance(parameter, vector):
            if (parameter < min_value).sum() > 0:
                raise ValueError('Parameter {} must have a minimum value above {}'.format(name, min_value))
        elif parameter < min_value:
            raise ValueError('Parameter {} must have a minimum value above {}'.format(name, min_value))
    if max_value is not None:
        if isinstance(parameter, vector):
            if (parameter > max_value).sum() > 0:
                raise ValueError('Parameter {} must have a maximum value below {}'.format(name, max_value))
        elif parameter > max_value:
            raise ValueError('Parameter {} must have a maximum value below {}'.format(name, max_value))
    if value_sum is not None:
        if isinstance(parameter, vector):
            if value_sum_dim is None:
                delta = torch.sum(parameter) - value_sum
            else:
                delta = torch.sum(parameter, dim=value_sum_dim) - value_sum
            if torch.any(torch.abs(delta) > epsilon):
                raise ValueError('Parameter {} must sum to {}'.format(name, value_sum))
        elif abs(parameter - value_sum) > epsilon:
            raise ValueError('Parameter {} must sum to {}'.format(name, value_sum))
    if value_set is not None:
        if isinstance(parameter, vector):
            if (~numpy.isin(parameter, value_set)).sum() > 0:
                raise ValueError('Parameter {} must contain values in set {}'.format(name, value_set))
        elif parameter not in value_set:
            raise ValueError('Parameter {} must contain values in set {}'.format(name, value_set))
    if ndim is not None:
        if isinstance(parameter, vector):
            if isinstance(ndim, int):
                if len(parameter.shape) != ndim:
                    raise ValueError('Parameter {} must have {} dims'.format(name, ndim))
            elif len(parameter.shape) not in ndim:
                raise ValueError('Parameter {} must have {} dims'.format(name, ndim))
        elif ndim != 0:
            raise ValueError('Parameter {} must have {} dims'.format(name, ndim))
    if shape is not None:
        if isinstance(parameter, vector):
            if len(parameter.shape) != len(shape):
                raise ValueError('Parameter {} must have shape {}'.format(name, shape))
            for i in range(len(shape)):
                if shape[i] != -1 and shape[i] != parameter.shape[i]:
                    raise ValueError('Parameter {} must have shape {}'.format(name, shape))
    return parameter


def _cast_as_parameter(value, dtype=None, requires_grad=False):
    """Set the parameter."""
    if value is None:
        return None
    value = _cast_as_tensor(value, dtype=dtype)
    return torch.nn.Parameter(value, requires_grad=requires_grad)


def _update_parameter(value, new_value, inertia=0.0, frozen=None):
    """Update a parameter unles.
	"""
    if hasattr(value, 'frozen') and getattr(value, 'frozen') == True:
        return
    if inertia == 0.0:
        value[...] = _cast_as_parameter(new_value)
    elif inertia < 1.0:
        value_ = inertia * value + (1 - inertia) * new_value
        inf_idx = torch.isinf(value)
        inf_idx_new = torch.isinf(new_value)
        value_[inf_idx] = value[inf_idx].type(value_.dtype)
        value_[inf_idx_new] = new_value[inf_idx_new].type(value_.dtype)
        value[:] = _cast_as_parameter(value_)


class BayesMixin(torch.nn.Module):

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        self.register_buffer('_w_sum', torch.zeros(self.k, device=self.device))
        self.register_buffer('_log_priors', torch.log(self.priors))

    def _emission_matrix(self, X, priors=None):
        """Return the emission/responsibility matrix.

		This method returns the log probability of each example under each
		distribution contained in the model with the log prior probability
		of each component added.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.

	
		Returns
		-------
		e: torch.Tensor, shape=(-1, self.k)
			A set of log probabilities for each example under each distribution.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        priors = _check_parameter(_cast_as_tensor(priors), 'priors', ndim=2, shape=(X.shape[0], self.k), min_value=0.0, max_value=1.0, value_sum=1.0, value_sum_dim=-1, check_parameter=self.check_data)
        d = X.shape[0]
        e = torch.empty(d, self.k, device=self.device, dtype=self.dtype)
        for i, d in enumerate(self.distributions):
            e[:, i] = d.log_probability(X)
        if priors is not None:
            e += torch.log(priors)
        return e + self._log_priors

    def probability(self, X, priors=None):
        """Calculate the probability of each example.

		This method calculates the probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format.

		Note: This differs from some other probability calculation
		functions, like those in torch.distributions, because it is not
		returning the probability of each feature independently, but rather
		the total probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		prob: torch.Tensor, shape=(-1,)
			The probability of each example.
		"""
        return torch.exp(self.log_probability(X, priors=priors))

    def log_probability(self, X, priors=None):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For a Bernoulli distribution, each entry in the data must
		be either 0 or 1.

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        e = self._emission_matrix(X, priors=priors)
        return torch.logsumexp(e, dim=1)

    def predict(self, X, priors=None):
        """Calculate the label assignment for each example.

		This method calculates the label for each example as the most likely
		component after factoring in the prior probability.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		y: torch.Tensor, shape=(-1,)
			The predicted label for each example.
		"""
        e = self._emission_matrix(X, priors=priors)
        return torch.argmax(e, dim=1)

    def predict_proba(self, X, priors=None):
        """Calculate the posterior probabilities for each example.

		This method calculates the posterior probabilities for each example
		under each component of the model after factoring in the prior 
		probability and normalizing across all the components.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		y: torch.Tensor, shape=(-1, self.k)
			The posterior probabilities for each example under each component.
		"""
        e = self._emission_matrix(X, priors=priors)
        return torch.exp(e - torch.logsumexp(e, dim=1, keepdims=True))

    def predict_log_proba(self, X, priors=None):
        """Calculate the log posterior probabilities for each example.

		This method calculates the log posterior probabilities for each example
		under each component of the model after factoring in the prior 
		probability and normalizing across all the components.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		y: torch.Tensor, shape=(-1, self.k)
			The log posterior probabilities for each example under each 
			component.
		"""
        e = self._emission_matrix(X, priors=priors)
        return e - torch.logsumexp(e, dim=1, keepdims=True)

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        for d in self.distributions:
            d.from_summaries()
        if self.frozen == True:
            return
        priors = self._w_sum / torch.sum(self._w_sum)
        _update_parameter(self.priors, priors, self.inertia)
        self._reset_cache()


class BufferList(torch.nn.Module):
    """A buffer list."""

    def __init__(self, buffers):
        super(BufferList, self).__init__()
        self.buffers = []
        for i, b in enumerate(buffers):
            name = '_buffer_{}'.format(i)
            self.register_buffer(name, b)
            self.buffers.append(getattr(self, name))

    def __repr__(self):
        return str(self.buffers)

    def __getitem__(self, i):
        return self.buffers[i]

    @property
    def dtype(self):
        return self.buffers[0].dtype


def _reshape_weights(X, sample_weight, device='cpu'):
    """Handle a sample weight tensor by creating and reshaping it.

	This function will take any weight input, including None, 1D weights, and
	2D weights, and shape it into a 2D matrix with the same shape as the data
	X also passed in.

	Both elements must be PyTorch tensors.


	Parameters
	----------
	X: torch.tensor, ndims=2
		The data being weighted. The contents of this tensor are not used, only
		the shape is.

	sample_weight: torch.tensor or None
		The weight for each element in the data or None.


	Returns
	-------
	sample_weight: torch.tensor, shape=X.shape
		A tensor with the same dimensions as X with elements repeated as
		necessary.
	"""
    if sample_weight is None:
        if torch.is_floating_point(X):
            sample_weight = torch.ones(1, device=device, dtype=X.dtype).expand(*X.shape)
        else:
            sample_weight = torch.ones(1, device=device, dtype=torch.float32).expand(*X.shape)
    elif not torch.is_floating_point(sample_weight):
        sample_weight = sample_weight.type(torch.float32)
    if len(sample_weight.shape) == 1:
        sample_weight = sample_weight.reshape(-1, 1).expand(-1, X.shape[1])
        _check_parameter(sample_weight, 'sample_weight', min_value=0)
    elif sample_weight.shape[1] == 1:
        sample_weight = sample_weight.expand(-1, X.shape[1])
        _check_parameter(sample_weight, 'sample_weight', min_value=0)
    if isinstance(X, torch.masked.MaskedTensor):
        if not isinstance(sample_weight, torch.masked.MaskedTensor):
            sample_weight = torch.masked.MaskedTensor(sample_weight, mask=X._masked_mask)
    _check_parameter(sample_weight, 'sample_weight', shape=X.shape, ndim=X.ndim)
    return sample_weight


class Distribution(torch.nn.Module):
    """A base distribution object.

	This distribution is inherited by all the other distributions.
	"""

    def __init__(self, inertia, frozen, check_data):
        super(Distribution, self).__init__()
        self._device = _cast_as_parameter([0.0])
        _check_parameter(inertia, 'inertia', min_value=0, max_value=1, ndim=0)
        _check_parameter(frozen, 'frozen', value_set=[True, False], ndim=0)
        _check_parameter(check_data, 'check_data', value_set=[True, False], ndim=0)
        self.register_buffer('inertia', _cast_as_tensor(inertia))
        self.register_buffer('frozen', _cast_as_tensor(frozen))
        self.register_buffer('check_data', _cast_as_tensor(check_data))
        self._initialized = False

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except:
            return 'cpu'

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def freeze(self):
        self.register_buffer('frozen', _cast_as_tensor(True))
        return self

    def unfreeze(self):
        self.register_buffer('frozen', _cast_as_tensor(False))
        return self

    def forward(self, X):
        self.summarize(X)
        return self.log_probability(X)

    def backward(self, X):
        self.from_summaries()
        return X

    def _initialize(self, d):
        self.d = d
        self._reset_cache()

    def _reset_cache(self):
        raise NotImplementedError

    def probability(self, X):
        return torch.exp(self.log_probability(X))

    def log_probability(self, X):
        raise NotImplementedError

    def fit(self, X, sample_weight=None):
        self.summarize(X, sample_weight=sample_weight)
        self.from_summaries()
        return self

    def summarize(self, X, sample_weight=None):
        if not self._initialized:
            self._initialize(len(X[0]))
        X = _cast_as_tensor(X)
        _check_parameter(X, 'X', ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight), device=self.device)
        return X, sample_weight

    def from_summaries(self):
        raise NotImplementedError


class BayesClassifier(BayesMixin, Distribution):
    """A Bayes classifier object.

	A simple way to produce a classifier using probabilistic models is to plug
	them into Bayes' rule. Basically, inference is the same as the 'E' step in
	EM for mixture models. However, fitting can be significantly faster because
	instead of having to iteratively infer labels and learn parameters, you can
	just learn the parameters given the known labels. Because the learning step
	for most models are simple MLE estimates, this can be done extremely
	quickly.

	Although the most common distribution to use is a Gaussian with a diagonal
	covariance matrix, termed the Gaussian naive Bayes model, any probability
	distribution can be used. Here, you can just drop any distributions or
	probabilistic model in as long as it has the `log_probability`, `summarize`,
	and `from_samples` methods implemented.

	Further, the probabilistic models do not even need to be simple
	distributions. The distributions can be mixture models or hidden Markov
	models or Bayesian networks.


	Parameters
	----------
	distributions: tuple or list
		A set of distribution objects. These objects do not need to be
		initialized, i.e., can be "Normal()". 

	priors: tuple, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The prior probabilities over the given distributions. Default is None.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling. Default is True.
	"""

    def __init__(self, distributions, priors=None, inertia=0.0, frozen=False, check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'BayesClassifier'
        _check_parameter(distributions, 'distributions', dtypes=(list, tuple, numpy.array, torch.nn.ModuleList))
        self.distributions = torch.nn.ModuleList(distributions)
        self.priors = _check_parameter(_cast_as_parameter(priors), 'priors', min_value=0, max_value=1, ndim=1, value_sum=1.0, shape=(len(distributions),))
        self.k = len(distributions)
        if all(d._initialized for d in distributions):
            self._initialized = True
            self.d = distributions[0].d
            if self.priors is None:
                self.priors = _cast_as_parameter(torch.ones(self.k) / self.k)
        else:
            self._initialized = False
            self.d = None
        self._reset_cache()

    def _initialize(self, d):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.
		"""
        self.priors = _cast_as_parameter(torch.ones(self.k, dtype=self.dtype, device=self.device) / self.k)
        self._initialized = True
        super()._initialize(d)

    def fit(self, X, y, sample_weight=None):
        """Fit the model to optionally weighted examples.

		This method implements the core of the learning process. For a
		general Bayes model, this involves fitting each component of the model
		using the labels that are provided. 

		This method is largely a wrapper around the `summarize` and
		`from_summaries` methods. It's primary contribution is serving as a
		loop around these functions and to monitor convergence.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

		y: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1,)
			A set of labels, one per example.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		self
		"""
        self.summarize(X, y, sample_weight=sample_weight)
        self.from_summaries()
        return self

    def summarize(self, X, y, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.

		For a Bayes' classifier, this step involves partitioning the data
		according to the labels and then training each component using MLE
		estimates.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		y: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1,)
			A set of labels, one per example.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""
        X, sample_weight = super().summarize(X, sample_weight=sample_weight)
        y = _check_parameter(_cast_as_tensor(y), 'y', min_value=0, max_value=self.k - 1, ndim=1, shape=(len(X),), check_parameter=self.check_data)
        sample_weight = _check_parameter(sample_weight, 'sample_weight', min_value=0, shape=(-1, self.d), check_parameter=self.check_data)
        for j, d in enumerate(self.distributions):
            idx = y == j
            d.summarize(X[idx], sample_weight[idx])
            if self.frozen == False:
                self._w_sum[j] = self._w_sum[j] + sample_weight[idx].mean(dim=-1).sum()


class ConditionalDistribution(Distribution):

    def __init__(self, inertia, frozen, check_data):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)

    def marginal(self, dim):
        raise NotImplementedError


eps = torch.finfo(torch.float32).eps


class Bernoulli(Distribution):
    """A Bernoulli distribution object.

	A Bernoulli distribution models the probability of a binary variable
	occurring. rates of discrete events, and has a probability parameter
	describing this value. This distribution assumes that each feature is 
	independent of the others.

	There are two ways to initialize this object. The first is to pass in
	the tensor of probability parameters, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the probability
	parameter will be learned from data.


	Parameters
	----------
	probs: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
		The probability parameters for each feature. Default is None.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, probs=None, inertia=0.0, frozen=False, check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'Bernoulli'
        self.probs = _check_parameter(_cast_as_parameter(probs), 'probs', min_value=eps, max_value=1 - eps, ndim=1)
        self._initialized = self.probs is not None
        self.d = self.probs.shape[-1] if self._initialized else None
        self._reset_cache()

    def _initialize(self, d):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.
		"""
        self.probs = _cast_as_parameter(torch.zeros(d, dtype=self.dtype, device=self.device))
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        self.register_buffer('_w_sum', torch.zeros(self.d, device=self.device))
        self.register_buffer('_xw_sum', torch.zeros(self.d, device=self.device))
        self.register_buffer('_log_probs', torch.log(self.probs))
        self.register_buffer('_log_inv_probs', torch.log(-(self.probs - 1)))

    def sample(self, n):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""
        return torch.distributions.Bernoulli(self.probs).sample([n])

    def log_probability(self, X):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For a Bernoulli distribution, each entry in the data must
		be either 0 or 1.

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        X = _check_parameter(_cast_as_tensor(X, dtype=self.probs.dtype), 'X', value_set=(0, 1), ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        return X.matmul(self._log_probs) + (1 - X).matmul(self._log_inv_probs)

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""
        if self.frozen == True:
            return
        X, sample_weight = super().summarize(X, sample_weight=sample_weight)
        _check_parameter(X, 'X', value_set=(0, 1), check_parameter=self.check_data)
        self._w_sum += torch.sum(sample_weight, dim=0)
        self._xw_sum += torch.sum(X * sample_weight, dim=0)

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        if self.frozen == True:
            return
        probs = self._xw_sum / self._w_sum
        _update_parameter(self.probs, probs, self.inertia)
        self._reset_cache()


def _inplace_add(X, Y):
    """Do an in-place addition on X accounting for if Y is a masked tensor."""
    if isinstance(Y, torch.masked.MaskedTensor):
        X += Y._masked_data
    else:
        X += Y


class Categorical(Distribution):
    """A categorical distribution object.

	A categorical distribution models the probability of a set of distinct
	values happening. It is an extension of the Bernoulli distribution to
	multiple values. Sometimes it is referred to as a discrete distribution,
	but this distribution does not enforce that the numeric values used for the
	keys have any relationship based on their identity. Permuting the keys will
	have no effect on the calculation. This distribution assumes that the
	features are independent from each other.

	The keys must be contiguous non-negative integers that begin at zero. 
	Because the probabilities are represented as a single tensor, each feature
	must have values for all keys up to the maximum key of any one distribution.
	Specifically, if one feature has 10 keys and a second feature has only 4,
	the tensor must go out to 10 for each feature but encode probabilities of
	zero for the second feature. 


	Parameters
	----------
	probs: list, numpy.ndarray, torch.tensor or None, shape=(k, d), optional
		Probabilities for each key for each feature, where k is the largest
		number of keys across all features. Default is None

	n_categories: list, numpy.ndarray, torch.tensor or None, optional
		The number of categories for each feature in the data. Only needs to
		be provided when the parameters will be learned directly from data and
		you want to make sure that right number of keys are included in each
		dimension. Default is None.

	pseudocount: float, optional
		A value to add to the observed counts of each feature when training.
		Setting this to a positive value ensures that no probabilities are
		truly zero. Default is 0.

	inertia: float, (0, 1), optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, probs=None, n_categories=None, pseudocount=0.0, inertia=0.0, frozen=False, check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'Categorical'
        self.probs = _check_parameter(_cast_as_parameter(probs), 'probs', min_value=0, max_value=1, ndim=2)
        self.pseudocount = pseudocount
        self._initialized = probs is not None
        self.d = self.probs.shape[-2] if self._initialized else None
        if n_categories is not None:
            self.n_keys = n_categories
        else:
            self.n_keys = self.probs.shape[-1] if self._initialized else None
        self._reset_cache()

    def _initialize(self, d, n_keys):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.

		n_keys: int
			The number of keys the distribution is being initialized with.
		"""
        self.probs = _cast_as_parameter(torch.zeros(d, n_keys, dtype=self.dtype, device=self.device))
        self.n_keys = n_keys
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        self.register_buffer('_w_sum', torch.zeros(self.d, device=self.device))
        self.register_buffer('_xw_sum', torch.zeros(self.d, self.n_keys, device=self.device))
        self.register_buffer('_log_probs', torch.log(self.probs))

    def sample(self, n):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""
        return torch.distributions.Categorical(self.probs).sample([n])

    def log_probability(self, X):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For a categorical distribution, each entry in the data must
		be an integer in the range [0, n_keys).

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', min_value=0.0, max_value=self.n_keys - 1, ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        logps = torch.zeros(X.shape[0], dtype=self.probs.dtype, device=self.device)
        for i in range(self.d):
            if isinstance(X, torch.masked.MaskedTensor):
                logp_ = self._log_probs[i][X[:, i]._masked_data]
                logp_[~X[:, i]._masked_mask] = 0
                logps += logp_
            else:
                logps += self._log_probs[i][X[:, i]]
        return logps

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""
        if self.frozen == True:
            return
        X = _cast_as_tensor(X)
        if not self._initialized:
            if self.n_keys is not None:
                n_keys = self.n_keys
            elif isinstance(X, torch.masked.MaskedTensor):
                n_keys = int(torch.max(X._masked_data)) + 1
            else:
                n_keys = int(torch.max(X)) + 1
            self._initialize(X.shape[1], n_keys)
        X = _check_parameter(X, 'X', min_value=0, max_value=self.n_keys - 1, ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight))
        _inplace_add(self._w_sum, torch.sum(sample_weight, dim=0))
        for i in range(self.n_keys):
            _inplace_add(self._xw_sum[:, i], torch.sum((X == i) * sample_weight, dim=0))

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        if self.frozen == True:
            return
        probs = (self._xw_sum + self.pseudocount) / (self._w_sum + self.pseudocount * self.n_keys).unsqueeze(1)
        _update_parameter(self.probs, probs, self.inertia)
        self._reset_cache()


class ConditionalCategorical(ConditionalDistribution):
    """A conditional categorical distribution.

	This is a categorical distribution that is conditioned on previous
	emissions, meaning that the probability of each character depends on the
	observed character earlier in the sequence. Each feature is conditioned
	independently of the others like a `Categorical` distribution. 

	This conditioning makes the shape of the distribution a bit more
	complicated than the `JointCategorical` distribution. Specifically, a 
	`JointCategorical` distribution is multivariate by definition but a
	`ConditionalCategorical` does not have to be. Although both may appear 
	similar in that they both take in a vector of characters and return 
	probabilities, the vector fed into the JointCategorical are all observed 
	together without some notion of time, whereas the ConditionalCategorical 
	explicitly requires a notion of timing, where the probability of later 
	characters depend on the composition of characters seen before.


	Parameters
	----------
	probs: list of numpy.ndarray, torch.tensor or None, shape=(k, k), optional
		A list of conditional probabilities with one tensor for each feature
		in the data being modeled. Each tensor should have `k+1` dimensions 
		where `k` is the number of timesteps to condition on. Each dimension
		should span the number of keys in that dimension. For example, if
		specifying a univariate conditional categorical distribution where
		k=2, a valid tensor shape would be [(2, 3, 4)]. Default is None.

	n_categories: list, numpy.ndarray, torch.tensor or None, optional
		The number of categories for each feature in the data. Only needs to
		be provided when the parameters will be learned directly from data and
		you want to make sure that right number of keys are included in each
		dimension. Unlike the `Categorical` distribution, this needs to be
		a list of shapes with one shape for each feature and the shape matches
		that specified in `probs`. Default is None.

	pseudocount: float, optional
		A value to add to the observed counts of each feature when training.
		Setting this to a positive value ensures that no probabilities are
		truly zero. Default is 0.

	inertia: float, (0, 1), optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, probs=None, n_categories=None, pseudocount=0, inertia=0.0, frozen=False, check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'ConditionalCategorical'
        if probs is not None:
            self.n_categories = []
            self.probs = torch.nn.ParameterList([])
            for prob in probs:
                prob = _check_parameter(_cast_as_parameter(prob), 'probs', min_value=0, max_value=1)
                self.probs.append(prob)
                self.n_categories.append(tuple(prob.shape))
        else:
            self.probs = None
            self.n_categories = n_categories
        self.pseudocount = _check_parameter(pseudocount, 'pseudocount')
        self._initialized = probs is not None
        self.d = len(self.probs) if self._initialized else None
        self.n_parents = len(self.probs[0].shape) if self._initialized else None
        self._reset_cache()

    def _initialize(self, d, n_categories):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.

		n_categories: list of tuples
			The shape of each conditional distribution, one per feature.
		"""
        self.n_categories = []
        for n_cat in n_categories:
            if isinstance(n_cat, (list, tuple)):
                self.n_categories.append(tuple(n_cat))
            elif isinstance(n_cat, (numpy.ndarray, torch.Tensor)):
                self.n_categories.append(tuple(n_cat.tolist()))
        self.n_parents = len(self.n_categories[0])
        self.probs = torch.nn.ParameterList([_cast_as_parameter(torch.zeros(*cats, dtype=self.dtype, device=self.device, requires_grad=False)) for cats in self.n_categories])
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        _w_sum = []
        _xw_sum = []
        for n_categories in self.n_categories:
            _w_sum.append(torch.zeros(*n_categories[:-1], dtype=self.probs[0].dtype, device=self.device))
            _xw_sum.append(torch.zeros(*n_categories, dtype=self.probs[0].dtype, device=self.device))
        self._w_sum = BufferList(_w_sum)
        self._xw_sum = BufferList(_xw_sum)
        self._log_probs = BufferList([torch.log(prob) for prob in self.probs])

    def sample(self, n, X):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution. For a mixture model, this involves first
		sampling the component using the prior probabilities, and then sampling
		from the chosen distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		
		X: list, numpy.ndarray, torch.tensor, shape=(n, d, *self.probs.shape-1) 
			The values to be conditioned on when generating the samples.

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=3, shape=(-1, self.n_parents - 1, self.d))
        y = []
        for i in range(n):
            y.append([])
            for j in range(self.d):
                idx = tuple(X[i, :, j])
                if len(idx) == 1:
                    idx = idx[0].item()
                probs = self.probs[j][idx]
                y_ = torch.multinomial(probs, 1).item()
                y[-1].append(y_)
        return torch.tensor(y)

    def log_probability(self, X):
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=3, shape=(-1, self.n_parents, self.d), check_parameter=self.check_data)
        logps = torch.zeros(len(X), dtype=self.probs[0].dtype, device=X.device, requires_grad=False)
        for i in range(len(X)):
            for j in range(self.d):
                logps[i] += self._log_probs[j][tuple(X[i, :, j])]
        return logps

    def summarize(self, X, sample_weight=None):
        if self.frozen == True:
            return
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=3, dtypes=(torch.int32, torch.int64), check_parameter=self.check_data)
        if not self._initialized:
            self._initialize(len(X[0][0]), torch.max(X, dim=0)[0].T + 1)
        X = _check_parameter(X, 'X', shape=(-1, self.n_parents, self.d), check_parameter=self.check_data)
        sample_weight = _check_parameter(_cast_as_tensor(sample_weight, dtype=torch.float32), 'sample_weight', min_value=0, ndim=(1, 2))
        if sample_weight is None:
            sample_weight = torch.ones(X[:, 0].shape[0], X[:, 0].shape[-1], dtype=self.probs[0].dtype)
        elif len(sample_weight.shape) == 1:
            sample_weight = sample_weight.reshape(-1, 1).expand(-1, X.shape[2])
        elif sample_weight.shape[1] == 1 and self.d > 1:
            sample_weight = sample_weight.expand(-1, X.shape[2])
        _check_parameter(sample_weight, 'sample_weight', min_value=0, ndim=2, shape=(X.shape[0], X.shape[2]))
        for j in range(self.d):
            strides = torch.tensor(self._xw_sum[j].stride(), device=X.device)
            X_ = torch.sum(X[:, :, j] * strides, dim=-1)
            self._xw_sum[j].view(-1).scatter_add_(0, X_, sample_weight[:, j])
            self._w_sum[j][:] = self._xw_sum[j].sum(dim=-1)

    def from_summaries(self):
        if self.frozen == True:
            return
        for i in range(self.d):
            probs = self._xw_sum[i] / self._w_sum[i].unsqueeze(-1)
            probs = torch.nan_to_num(probs, 1.0 / probs.shape[-1])
            _update_parameter(self.probs[i], probs, self.inertia)
        self._reset_cache()


class DiracDelta(Distribution):
    """A dirac delta distribution object.

	A dirac delta distribution is a probability distribution that has its entire
	density at zero. This distribution assumes that each feature is independent
	of the others. This means that, in practice, it will assign a zero
	probability if any value in an example is non-zero.

	There are two ways to initialize this object. The first is to pass in
	the tensor of alpha values representing the probability to return given a
	zero value, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the probability
	parameter will be learned from data.


	Parameters
	----------
	alphas: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
		The probability parameters for each feature. Default is None.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, alphas=None, inertia=0.0, frozen=False, check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'DiracDelta'
        self.alphas = _check_parameter(_cast_as_parameter(alphas), 'alphas', min_value=0.0, ndim=1)
        self._initialized = alphas is not None
        self.d = len(self.alphas) if self._initialized else None
        self._reset_cache()

    def _initialize(self, d):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.
		"""
        self.alphas = _cast_as_parameter(torch.ones(d, device=self.device))
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        self.register_buffer('_log_alphas', torch.log(self.alphas))

    def log_probability(self, X):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. 

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        return torch.sum(torch.where(X == 0.0, self._log_alphas, float('-inf')), dim=-1)

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.

		For a dirac delta distribution, there are no statistics to extract.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""
        if self.frozen == True:
            return
        X, sample_weight = super().summarize(X, sample_weight=sample_weight)

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		For a dirac delta distribution, there are no updates.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        return


class Exponential(Distribution):
    """An exponential distribution object.

	An exponential distribution models scales of discrete events, and has a
	rate parameter describing the average time between event occurrences.
	This distribution assumes that each feature is independent of the others.
	Although the object is meant to operate on discrete counts, it can be used
	on any non-negative continuous data.

	There are two ways to initialize this object. The first is to pass in
	the tensor of rate parameters, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the rate
	parameter will be learned from data.


	Parameters
	----------
	scales: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
		The rate parameters for each feature. Default is None.

	inertia: float, (0, 1), optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are 
		frozen. If you want to freeze individual pameters, or individual values 
		in those parameters, you must modify the `frozen` attribute of the 
		tensor or parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, scales=None, inertia=0.0, frozen=False, check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'Exponential'
        self.scales = _check_parameter(_cast_as_parameter(scales), 'scales', min_value=0, ndim=1)
        self._initialized = scales is not None
        self.d = self.scales.shape[-1] if self._initialized else None
        self._reset_cache()

    def _initialize(self, d):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.
		"""
        self.scales = _cast_as_parameter(torch.zeros(d, dtype=self.dtype, device=self.device))
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        self.register_buffer('_w_sum', torch.zeros(self.d, device=self.device))
        self.register_buffer('_xw_sum', torch.zeros(self.d, device=self.device))
        self.register_buffer('_log_scales', torch.log(self.scales))

    def sample(self, n):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""
        return tExponential(1.0 / self.scales).sample([n])

    def log_probability(self, X):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For an exponential distribution, the data must be non-negative.

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', min_value=0.0, ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        return torch.sum(-self._log_scales - 1.0 / self.scales * X, dim=1)

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""
        if self.frozen == True:
            return
        X, sample_weight = super().summarize(X, sample_weight=sample_weight)
        _check_parameter(X, 'X', min_value=0, check_parameter=self.check_data)
        self._w_sum[:] = self._w_sum + torch.sum(sample_weight, dim=0)
        self._xw_sum[:] = self._xw_sum + torch.sum(X * sample_weight, dim=0)

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        if self.frozen == True:
            return
        scales = self._xw_sum / self._w_sum
        _update_parameter(self.scales, scales, self.inertia)
        self._reset_cache()


def _check_shapes(parameters, names):
    """Check the shapes of a set of parameters.

	This function takes in a set of parameters, as well as their names, and
	checks that the shape is correct. It will raise an error if the lengths
	of the parameters without the value of None are not equal.


	Parameters
	----------
	parameters: list or tuple
		A set of parameters, which can be None, to check the shape of.

	names: list or tuple
		A set of parameter names to refer to if something is wrong.
	"""
    n = len(parameters)
    for i in range(n):
        for j in range(n):
            if parameters[i] is None:
                continue
            if parameters[j] is None:
                continue
            n1, n2 = names[i], names[j]
            if len(parameters[i]) != len(parameters[j]):
                raise ValueError('Parameters {} and {} must be the same shape.'.format(names[i], names[j]))


class Gamma(Distribution):
    """A gamma distribution object.

	A gamma distribution is the sum of exponential distributions, and has shape
	and rate parameters. This distribution assumes that each feature is
	independent of the others. 

	There are two ways to initialize this objecct. The first is to pass in
	the tensor of rate and shae parameters, at which point they can immediately 
	be used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the rate
	and shape parameters will be learned from data.


	Parameters
	----------
	shapes: torch.tensor or None, shape=(d,), optional
		The shape parameter for each feature. Default is None

	rates: torch.tensor or None, shape=(d,), optional
		The rate parameters for each feature. Default is None.

	inertia: float, (0, 1), optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	tol: float, [0, inf), optional
		The threshold at which to stop fitting the parameters of the
		distribution. Default is 1e-4.

	max_iter: int, [0, inf), optional
		The maximum number of iterations to run EM when fitting the parameters
		of the distribution. Default is 20.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, shapes=None, rates=None, inertia=0.0, tol=0.0001, max_iter=20, frozen=False, check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'Gamma'
        self.shapes = _check_parameter(_cast_as_parameter(shapes), 'shapes', min_value=0, ndim=1)
        self.rates = _check_parameter(_cast_as_parameter(rates), 'rates', min_value=0, ndim=1)
        _check_shapes([self.shapes, self.rates], ['shapes', 'rates'])
        self.tol = _check_parameter(tol, 'tol', min_value=0, ndim=0)
        self.max_iter = _check_parameter(max_iter, 'max_iter', min_value=1, ndim=0)
        self._initialized = shapes is not None and rates is not None
        self.d = self.shapes.shape[-1] if self._initialized else None
        self._reset_cache()

    def _initialize(self, d):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.
		"""
        self.shapes = _cast_as_parameter(torch.zeros(d, dtype=self.dtype, device=self.device))
        self.rates = _cast_as_parameter(torch.zeros(d, dtype=self.dtype, device=self.device))
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        self.register_buffer('_w_sum', torch.zeros(self.d, device=self.device))
        self.register_buffer('_xw_sum', torch.zeros(self.d, device=self.device))
        self.register_buffer('_logx_w_sum', torch.zeros(self.d, device=self.device))
        self.register_buffer('_log_rates', torch.log(self.rates))
        self.register_buffer('_lgamma_shapes', torch.lgamma(self.shapes))
        self.register_buffer('_thetas', self._log_rates * self.shapes - self._lgamma_shapes)

    def sample(self, n):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""
        return torch.distributions.Gamma(self.shapes, self.rates).sample([n])

    def log_probability(self, X):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For a gamma distribution, the data must be non-negative.

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', min_value=0.0, ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        return torch.sum(self._thetas + torch.log(X) * (self.shapes - 1) - self.rates * X, dim=-1)

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""
        if self.frozen == True:
            return
        X, sample_weight = super().summarize(X, sample_weight=sample_weight)
        _check_parameter(X, 'X', min_value=0, check_parameter=self.check_data)
        self._w_sum[:] = self._w_sum + torch.sum(sample_weight, dim=0)
        self._xw_sum[:] = self._xw_sum + torch.sum(X * sample_weight, dim=0)
        self._logx_w_sum[:] = self._logx_w_sum + torch.sum(torch.log(X) * sample_weight, dim=0)

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        if self.frozen == True:
            return
        thetas = torch.log(self._xw_sum / self._w_sum) - self._logx_w_sum / self._w_sum
        numerator = 3 - thetas + torch.sqrt((thetas - 3) ** 2 + 24 * thetas)
        denominator = 12 * thetas
        new_shapes = numerator / denominator
        shapes = new_shapes + self.tol
        for iteration in range(self.max_iter):
            mask = torch.abs(shapes - new_shapes) < self.tol
            if torch.all(mask):
                break
            shapes = new_shapes
            new_shapes = shapes - (torch.log(shapes) - torch.polygamma(0, shapes) - thetas) / (1.0 / shapes - torch.polygamma(1, shapes))
        shapes = new_shapes
        rates = 1.0 / (1.0 / (shapes * self._w_sum) * self._xw_sum)
        _update_parameter(self.shapes, shapes, self.inertia)
        _update_parameter(self.rates, rates, self.inertia)
        self._reset_cache()


class IndependentComponents(Distribution):
    """An independent components distribution object.

	A distribution made up of independent, univariate, distributions that each
	model a single feature in the data. This means that instead of using a
	single type of distribution to model all of the features in your data, you
	use one distribution per feature. Note that this will likely be slower
	than using a single distribution because the amount of batching possible
	will go down significantly.

	There are two ways to initialize this object. The first is to pass in a
	set of distributions that are all initialized with parameters, at which
	point this distribution can be immediately used for inference. The second
	is to pass in a set of distributions that are not initialized with
	parameters, and then call either `fit` or `summary` + `from_summaries` to
	learn the parameters of all the distributions.


	Parameters
	----------
	distributions: list, tuple, numpy.ndarray, torch.Tensor, shape=(d,)
		An ordered iterable containing all of the distributions, one per
		feature, that will be used.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, distributions, check_data=False):
        super().__init__(inertia=0.0, frozen=False, check_data=check_data)
        self.name = 'IndependentComponents'
        if len(distributions) <= 1:
            raise ValueError('Must pass in at least 2 distributions.')
        for distribution in distributions:
            if not isinstance(distribution, Distribution):
                raise ValueError('All passed in distributions must ' + 'inherit from the Distribution object.')
        self.distributions = distributions
        self._initialized = all(d._initialized for d in distributions)
        self.d = len(distributions)
        self._reset_cache()

    def _initialize(self, d):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.
		"""
        for distribution in self.distributions:
            distribution._initialize(d)
        self._initialized = True

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        for distribution in self.distributions:
            distribution._reset_cache()

    def sample(self, n):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""
        return torch.hstack([d.sample(n) for d in self.distributions])

    def log_probability(self, X):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format.

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, shape=(-1, self.d))
        logp = torch.zeros(X.shape[0])
        for i, d in enumerate(self.distributions):
            if isinstance(X, torch.masked.MaskedTensor):
                logp.add_(d.log_probability(X[:, i:i + 1])._masked_data)
            else:
                logp.add_(d.log_probability(X[:, i:i + 1]))
        return logp

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""
        if self.frozen == True:
            return
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, shape=(-1, self.d))
        sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight, dtype=torch.float32), device=self.device)
        for i, d in enumerate(self.distributions):
            d.summarize(X[:, i:i + 1], sample_weight=sample_weight[:, i:i + 1])

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        if self.frozen == True:
            return
        for distribution in self.distributions:
            distribution.from_summaries()


class JointCategorical(Distribution):
    """A joint categorical distribution.

	A joint categorical distribution models the probability of a vector of
	categorical values occurring without assuming that the dimensions are
	independent from each other. Essentially, it is a Categorical distribution
	without the assumption that the dimensions are independent of each other. 

	There are two ways to initialize this object. The first is to pass in
	the tensor of probability parameters, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the 
	probability parameters will be learned from data.


	Parameters
	----------
	probs: list, numpy.ndarray, torch.tensor, or None, shape=*n_categories
		A tensor where each dimension corresponds to one column in the data
		set being modeled and the size of each dimension is the number of
		categories in that column, e.g., if the data being modeled is binary 
		and has shape (5, 4), this will be a tensor with shape (2, 2, 2, 2).
		Default is None.

	n_categories: list, numpy.ndarray, torch.tensor, or None, shape=(d,)
		A vector with the maximum number of categories that each column
		can have. If not given, this will be inferred from the data. Default
		is None.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	pseudocount: float, optional
		A number of observations to add to each entry in the probability
		distribution during training. A higher value will smooth the 
		distributions more. Default is 0.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, probs=None, n_categories=None, pseudocount=0, inertia=0.0, frozen=False, check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'JointCategorical'
        self.probs = _check_parameter(_cast_as_parameter(probs), 'probs', min_value=0, max_value=1, value_sum=1)
        self.n_categories = _check_parameter(n_categories, 'n_categories', min_value=2)
        self.pseudocount = _check_parameter(pseudocount, 'pseudocount')
        self._initialized = probs is not None
        self.d = len(self.probs.shape) if self._initialized else None
        if self._initialized:
            if n_categories is None:
                self.n_categories = tuple(self.probs.shape)
            elif isinstance(n_categories, int):
                self.n_categories = (n_categories for i in range(n_categories))
            else:
                self.n_categories = tuple(n_categories)
        else:
            self.n_categories = None
        self._reset_cache()

    def _initialize(self, d, n_categories):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.

		n_categories: list, numpy.ndarray, torch.tensor, or None, shape=(d,)
			A vector with the maximum number of categories that each column
			can have. If not given, this will be inferred from the data. 
			Default is None.
		"""
        self.probs = _cast_as_parameter(torch.zeros(*n_categories, dtype=self.dtype, device=self.device))
        self.n_categories = n_categories
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        self._w_sum = torch.zeros(self.d, dtype=self.probs.dtype)
        self._xw_sum = torch.zeros(*self.n_categories, dtype=self.probs.dtype)
        self._log_probs = torch.log(self.probs)

    def sample(self, n):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution. For a mixture model, this involves first
		sampling the component using the prior probabilities, and then sampling
		from the chosen distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""
        idxs = torch.multinomial(self.probs.flatten(), num_samples=n, replacement=True)
        X = numpy.unravel_index(idxs.numpy(), self.n_categories)
        X = numpy.stack(X).T
        return torch.from_numpy(X)

    def log_probability(self, X):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For a joint categorical distribution, each value must be an
		integer category that is smaller than the maximum number of categories
		for each feature.

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', value_set=tuple(range(max(self.n_categories) + 1)), ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        logps = torch.zeros(len(X), dtype=self.probs.dtype)
        for i in range(len(X)):
            logps[i] = self._log_probs[tuple(X[i])]
        return logps

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""
        if self.frozen == True:
            return
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, dtypes=(torch.int32, torch.int64), check_parameter=self.check_data)
        if not self._initialized:
            self._initialize(len(X[0]), torch.max(X, dim=0)[0] + 1)
        X = _check_parameter(X, 'X', shape=(-1, self.d), value_set=tuple(range(max(self.n_categories) + 1)), check_parameter=self.check_data)
        sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight, dtype=torch.float32))[:, 0]
        self._w_sum += torch.sum(sample_weight, dim=0)
        for i in range(len(X)):
            self._xw_sum[tuple(X[i])] += sample_weight[i]

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        if self.frozen == True:
            return
        probs = self._xw_sum / self._w_sum[0]
        _update_parameter(self.probs, probs, self.inertia)
        self._reset_cache()


LOG_2_PI = 1.83787706641


SQRT_2_PI = 2.50662827463


class Normal(Distribution):
    """A normal distribution object.

	A normal distribution models the probability of a variable occurring under
	a bell-shaped curve. It is described by a vector of mean values and a
	covariance value that can be zero, one, or two dimensional. This
	distribution can assume that features are independent of the others if
	the covariance type is 'diag' or 'sphere', but if the type is 'full' then
	the features are not independent.

	There are two ways to initialize this object. The first is to pass in
	the tensor of probability parameters, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the probability
	parameter will be learned from data.


	Parameters
	----------
	means: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
		The mean values of the distributions. Default is None.

	covs: list, numpy.ndarray, torch.Tensor, or None, optional
		The variances and covariances of the distribution. If covariance_type
		is 'full', the shape should be (self.d, self.d); if 'diag', the shape
		should be (self.d,); if 'sphere', it should be (1,). Note that this is
		the variances or covariances in all settings, and not the standard
		deviation, as may be more common for diagonal covariance matrices.
		Default is None.

	covariance_type: str, optional
		The type of covariance matrix. Must be one of 'full', 'diag', or
		'sphere'. Default is 'full'. 

	min_cov: float or None, optional
		The minimum variance or covariance.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.
	"""

    def __init__(self, means=None, covs=None, covariance_type='full', min_cov=None, inertia=0.0, frozen=False, check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'Normal'
        self.means = _check_parameter(_cast_as_parameter(means), 'means', ndim=1)
        self.covs = _check_parameter(_cast_as_parameter(covs), 'covs', ndim=(1, 2))
        _check_shapes([self.means, self.covs], ['means', 'covs'])
        self.min_cov = _check_parameter(min_cov, 'min_cov', min_value=0, ndim=0)
        self.covariance_type = covariance_type
        self._initialized = means is not None and covs is not None
        self.d = self.means.shape[-1] if self._initialized else None
        self._reset_cache()

    def _initialize(self, d):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.
		"""
        self.means = _cast_as_parameter(torch.zeros(d, dtype=self.dtype, device=self.device))
        if self.covariance_type == 'full':
            self.covs = _cast_as_parameter(torch.zeros(d, d, dtype=self.dtype, device=self.device))
        elif self.covariance_type == 'diag':
            self.covs = _cast_as_parameter(torch.zeros(d, dtype=self.dtype, device=self.device))
        elif self.covariance_type == 'sphere':
            self.covs = _cast_as_parameter(torch.tensor(0, dtype=self.dtype, device=self.device))
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        self.register_buffer('_w_sum', torch.zeros(self.d, dtype=self.dtype, device=self.device))
        self.register_buffer('_xw_sum', torch.zeros(self.d, dtype=self.dtype, device=self.device))
        if self.covariance_type == 'full':
            self.register_buffer('_xxw_sum', torch.zeros(self.d, self.d, dtype=self.dtype, device=self.device))
            if self.covs.sum() > 0.0:
                chol = torch.linalg.cholesky(self.covs)
                _inv_cov = torch.linalg.solve_triangular(chol, torch.eye(len(self.covs), dtype=self.dtype, device=self.device), upper=False).T
                _inv_cov_dot_mu = torch.matmul(self.means, _inv_cov)
                _log_det = -0.5 * torch.linalg.slogdet(self.covs)[1]
                _theta = _log_det - 0.5 * (self.d * LOG_2_PI)
                self.register_buffer('_inv_cov', _inv_cov)
                self.register_buffer('_inv_cov_dot_mu', _inv_cov_dot_mu)
                self.register_buffer('_log_det', _log_det)
                self.register_buffer('_theta', _theta)
        elif self.covariance_type in ('diag', 'sphere'):
            self.register_buffer('_xxw_sum', torch.zeros(self.d, dtype=self.dtype, device=self.device))
            if self.covs.sum() > 0.0:
                _log_sigma_sqrt_2pi = -torch.log(torch.sqrt(self.covs) * SQRT_2_PI)
                _inv_two_sigma = 1.0 / (2 * self.covs)
                self.register_buffer('_log_sigma_sqrt_2pi', _log_sigma_sqrt_2pi)
                self.register_buffer('_inv_two_sigma', _inv_two_sigma)
            if torch.any(self.covs < 0):
                raise ValueError('Variances must be positive.')

    def sample(self, n):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""
        if self.covariance_type == 'diag':
            return torch.distributions.Normal(self.means, self.covs).sample([n])
        elif self.covariance_type == 'full':
            return torch.distributions.MultivariateNormal(self.means, self.covs).sample([n])

    def log_probability(self, X):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format.

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        X = _check_parameter(_cast_as_tensor(X, dtype=self.means.dtype), 'X', ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        if self.covariance_type == 'full':
            logp = torch.matmul(X, self._inv_cov) - self._inv_cov_dot_mu
            logp = self.d * LOG_2_PI + torch.sum(logp ** 2, dim=-1)
            logp = self._log_det - 0.5 * logp
            return logp
        elif self.covariance_type in ('diag', 'sphere'):
            return torch.sum(self._log_sigma_sqrt_2pi - (X - self.means) ** 2 * self._inv_two_sigma, dim=-1)

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""
        if self.frozen == True:
            return
        X, sample_weight = super().summarize(X, sample_weight=sample_weight)
        X = _cast_as_tensor(X, dtype=self.means.dtype)
        sample_weight = _cast_as_tensor(sample_weight, dtype=self.means.dtype)
        if self.covariance_type == 'full':
            self._w_sum += torch.sum(sample_weight, dim=0)
            self._xw_sum += torch.sum(X * sample_weight, axis=0)
            self._xxw_sum += torch.matmul((X * sample_weight).T, X)
        elif self.covariance_type in ('diag', 'sphere'):
            self._w_sum[:] = self._w_sum + torch.sum(sample_weight, dim=0)
            self._xw_sum[:] = self._xw_sum + torch.sum(X * sample_weight, dim=0)
            self._xxw_sum[:] = self._xxw_sum + torch.sum(X ** 2 * sample_weight, dim=0)

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        if self.frozen == True:
            return
        means = self._xw_sum / self._w_sum
        if self.covariance_type == 'full':
            v = self._xw_sum.unsqueeze(0) * self._xw_sum.unsqueeze(1)
            covs = self._xxw_sum / self._w_sum - v / self._w_sum ** 2.0
        elif self.covariance_type in ['diag', 'sphere']:
            covs = self._xxw_sum / self._w_sum - self._xw_sum ** 2.0 / self._w_sum ** 2.0
            if self.covariance_type == 'sphere':
                covs = covs.mean(dim=-1)
        _update_parameter(self.means, means, self.inertia)
        _update_parameter(self.covs, covs, self.inertia)
        self._reset_cache()


class Poisson(Distribution):
    """An poisson distribution object.

	A poisson distribution models the number of occurrences of events that
	happen in a fixed time span, assuming that the occurrence of each event
	is independent. This distribution also assumes that each feature is
	independent of the others.

	There are two ways to initialize this objecct. The first is to pass in
	the tensor of lambda parameters, at which point they can immediately be
	used. The second is to not pass in the lambda parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the lambda
	parameter will be learned from data.


	Parameters
	----------
	lambdas: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
		The lambda parameters for each feature. Default is None.

	inertia: float, (0, 1), optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, lambdas=None, inertia=0.0, frozen=False, check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'Poisson'
        self.lambdas = _check_parameter(_cast_as_parameter(lambdas), 'lambdas', min_value=0, ndim=1)
        self._initialized = lambdas is not None
        self.d = self.lambdas.shape[-1] if self._initialized else None
        self._reset_cache()

    def _initialize(self, d):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.
		"""
        self.lambdas = _cast_as_parameter(torch.zeros(d, dtype=self.dtype, device=self.device))
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        self.register_buffer('_w_sum', torch.zeros(self.d, device=self.device))
        self.register_buffer('_xw_sum', torch.zeros(self.d, device=self.device))
        self.register_buffer('_log_lambdas', torch.log(self.lambdas))

    def sample(self, n):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""
        return torch.distributions.Poisson(self.lambdas).sample([n])

    def log_probability(self, X):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For a Poisson distribution, each entry in the data must
		be non-negative.

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', min_value=0.0, ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        return torch.sum(X * self._log_lambdas - self.lambdas - torch.lgamma(X + 1), dim=-1)

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""
        if self.frozen == True:
            return
        X, sample_weight = super().summarize(X, sample_weight=sample_weight)
        _check_parameter(X, 'X', min_value=0, check_parameter=self.check_data)
        self._w_sum[:] = self._w_sum + torch.sum(sample_weight, dim=0)
        self._xw_sum[:] = self._xw_sum + torch.sum(X * sample_weight, dim=0)

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        if self.frozen == True:
            return
        lambdas = self._xw_sum / self._w_sum
        _update_parameter(self.lambdas, lambdas, self.inertia)
        self._reset_cache()


class StudentT(Normal):
    """A Student T distribution.

	A Student T distribution models the probability of a variable occurring under
	a bell-shaped curve with heavy tails. Basically, this is a version of the
	normal distribution that is less resistant to outliers.  It is described by 
	a vector of mean values and a vector of variance values. This
	distribution can assume that features are independent of the others if
	the covariance type is 'diag' or 'sphere', but if the type is 'full' then
	the features are not independent.

	There are two ways to initialize this object. The first is to pass in
	the tensor of probability parameters, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the probability
	parameter will be learned from data.


	Parameters
	----------
	means: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
		The mean values of the distributions. Default is None.

	covs: list, numpy.ndarray, torch.Tensor, or None, optional
		The variances and covariances of the distribution. If covariance_type
		is 'full', the shape should be (self.d, self.d); if 'diag', the shape
		should be (self.d,); if 'sphere', it should be (1,). Note that this is
		the variances or covariances in all settings, and not the standard
		deviation, as may be more common for diagonal covariance matrices.
		Default is None.

	covariance_type: str, optional
		The type of covariance matrix. Must be one of 'full', 'diag', or
		'sphere'. Default is 'full'. 

	min_cov: float or None, optional
		The minimum variance or covariance.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, dofs, means=None, covs=None, covariance_type='diag', min_cov=None, inertia=0.0, frozen=False, check_data=True):
        dofs = _check_parameter(_cast_as_tensor(dofs), 'dofs', min_value=1, ndim=0, dtypes=(torch.int32, torch.int64))
        self.dofs = dofs
        super().__init__(means=means, covs=covs, min_cov=min_cov, covariance_type=covariance_type, inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'StudentT'
        del self.dofs
        self.register_buffer('dofs', _cast_as_tensor(dofs))
        self.register_buffer('_lgamma_dofsp1', torch.lgamma((dofs + 1) / 2.0))
        self.register_buffer('_lgamma_dofs', torch.lgamma(dofs / 2.0))

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        super()._reset_cache()
        if self._initialized == False:
            return
        self.register_buffer('_log_sqrt_dofs_pi_cov', torch.log(torch.sqrt(self.dofs * math.pi * self.covs)))

    def sample(self, n):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""
        return torch.distributions.StudentT(self.means, self.covs).sample([n])

    def log_probability(self, X):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format.

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        t = (X - self.means) ** 2 / self.covs
        return torch.sum(self._lgamma_dofsp1 - self._lgamma_dofs - self._log_sqrt_dofs_pi_cov - (self.dofs + 1) / 2.0 * torch.log(1 + t / self.dofs), dim=-1)


inf = float('inf')


class Uniform(Distribution):
    """A uniform distribution.

	A uniform distribution models the probability of a variable occurring given
	a range that has the same probability within it and no probability outside
	it. It is described by a vector of minimum and maximum values for this
	range.  This distribution assumes that the features are independent of
	each other.

	There are two ways to initialize this object. The first is to pass in
	the tensor of probability parameters, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the probability
	parameter will be learned from data.


	Parameters
	----------
	mins: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
		The minimum values of the range.

	maxs: list, numpy.ndarray, torch.Tensor, or None, optional
		The maximum values of the range.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, mins=None, maxs=None, inertia=0.0, frozen=False, check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'Uniform'
        self.mins = _check_parameter(_cast_as_parameter(mins), 'mins', ndim=1)
        self.maxs = _check_parameter(_cast_as_parameter(maxs), 'maxs', ndim=1)
        _check_shapes([self.mins, self.maxs], ['mins', 'maxs'])
        self._initialized = mins is not None and maxs is not None
        self.d = self.mins.shape[-1] if self._initialized else None
        self._reset_cache()

    def _initialize(self, d):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.
		"""
        self.mins = _cast_as_parameter(torch.zeros(d, dtype=self.dtype, device=self.device))
        self.maxs = _cast_as_parameter(torch.zeros(d, dtype=self.dtype, device=self.device))
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        self.register_buffer('_x_mins', torch.full((self.d,), inf, device=self.device))
        self.register_buffer('_x_maxs', torch.full((self.d,), -inf, device=self.device))
        self.register_buffer('_logps', -torch.log(self.maxs - self.mins))

    def sample(self, n):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""
        return torch.distributions.Uniform(self.mins, self.maxs).sample([n])

    def log_probability(self, X):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For a Bernoulli distribution, each entry in the data must
		be either 0 or 1.

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        return torch.where((X >= self.mins) & (X <= self.maxs), self._logps, float('-inf')).sum(dim=1)

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""
        if self.frozen == True:
            return
        X, sample_weight = super().summarize(X, sample_weight=sample_weight)
        self._x_mins = torch.minimum(self._x_mins, X.min(dim=0).values)
        self._x_maxs = torch.maximum(self._x_maxs, X.max(dim=0).values)

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        if self.frozen == True:
            return
        _update_parameter(self.mins, self._x_mins, self.inertia)
        _update_parameter(self.maxs, self._x_maxs, self.inertia)
        self._reset_cache()


class ZeroInflated(Distribution):
    """A wrapper for a zero-inflated distribution.

	Some discrete distributions, e.g. Poisson or negative binomial, are used
	to model data that has many more zeroes in it than one would expect from
	the true signal itself. Potentially, this is because data collection devices
	fail or other gaps exist in the data. A zero-inflated distribution is
	essentially a mixture of these zero values and the real underlying
	distribution.

	Accordingly, this class serves as a wrapper that can be dropped in for
	other probability distributions and makes them "zero-inflated". It is
	similar to a mixture model between the distribution passed in and a dirac
	delta distribution, except that the mixture happens independently for each
	distribution as well as for each example.


	Parameters
	----------
	distribution: pomegranate.distributions.Distribution
		A pomegranate distribution object. It should probably be a discrete
		distribution, but does not technically have to be.

	priors: tuple, numpy.ndarray, torch.Tensor, or None. shape=(2,), optional
		The prior probabilities over the given distribution and the dirac
		delta component. Default is None.

	max_iter: int, optional
		The number of iterations to do in the EM step of fitting the
		distribution. Default is 10.

	tol: float, optional
		The threshold at which to stop during fitting when the improvement
		goes under. Default is 0.1.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	verbose: bool, optional
		Whether to print the improvement and timings during training.
	"""

    def __init__(self, distribution, priors=None, max_iter=10, tol=0.1, inertia=0.0, frozen=False, check_data=False, verbose=False):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'ZeroInflated'
        self.distribution = distribution
        self.priors = _check_parameter(_cast_as_parameter(priors), 'priors', min_value=0, max_value=1, ndim=1, value_sum=1.0)
        self.verbose = verbose
        self._initialized = distribution._initialized is True
        self.d = distribution.d if self._initialized else None
        self.max_iter = max_iter
        self.tol = tol
        if self.priors is None and self.d is not None:
            self.priors = _cast_as_parameter(torch.ones(self.d, device=self.device) / 2)
        self._reset_cache()

    def _initialize(self, X):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(1, self.d)
			The data to use to initialize the model.
		"""
        self.distribution._initialize(X.shape[1])
        self.distribution.fit(X)
        self.priors = _cast_as_parameter(torch.ones(X.shape[1], device=self.device) / 2)
        self._initialized = True
        super()._initialize(X.shape[1])

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        self.register_buffer('_w_sum', torch.zeros(self.d, 2, device=self.device))
        self.register_buffer('_log_priors', torch.log(self.priors))

    def _emission_matrix(self, X):
        """Return the emission/responsibility matrix.

		This method returns the log probability of each example under each
		distribution contained in the model with the log prior probability
		of each component added.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

	
		Returns
		-------
		e: torch.Tensor, shape=(-1, self.k)
			A set of log probabilities for each example under each distribution.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, shape=(-1, self.d))
        e = torch.empty(X.shape[0], self.d, 2, device=self.device)
        e[:, :, 0] = self._log_priors.unsqueeze(0)
        e[:, :, 0] += self.distribution.log_probability(X).unsqueeze(1)
        e[:, :, 1] = torch.log(1 - self.priors).unsqueeze(0)
        e[:, :, 1] += torch.where(X == 0, 0, float('-inf'))
        return e

    def fit(self, X, sample_weight=None):
        """Fit the model to optionally weighted examples.

		This method implements the core of the learning process. For a
		zero-inflated distribution, this involves performing EM until the
		distribution being fit converges.

		This method is largely a wrapper around the `summarize` and
		`from_summaries` methods. It's primary contribution is serving as a
		loop around these functions and to monitor convergence.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		self
		"""
        logp = None
        for i in range(self.max_iter):
            start_time = time.time()
            last_logp = logp
            logp = self.summarize(X, sample_weight=sample_weight)
            if i > 0:
                improvement = logp - last_logp
                duration = time.time() - start_time
                if self.verbose:
                    None
                if improvement < self.tol:
                    break
            self.from_summaries()
        self._reset_cache()
        return self

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""
        X = _cast_as_tensor(X)
        if not self._initialized:
            self._initialize(X)
        _check_parameter(X, 'X', ndim=2, shape=(-1, self.d))
        sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight, dtype=torch.float32), device=self.device)
        e = self._emission_matrix(X)
        logp = torch.logsumexp(e, dim=2, keepdims=True)
        y = torch.exp(e - logp)
        self.distribution.summarize(X, y[:, :, 0] * sample_weight)
        if not self.frozen:
            self._w_sum += torch.sum(y * sample_weight.unsqueeze(-1), dim=(0, 1))
        return torch.sum(logp)

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        self.distribution.from_summaries()
        if self.frozen == True:
            return
        priors = self._w_sum[:, 0] / torch.sum(self._w_sum, dim=-1)
        _update_parameter(self.priors, priors, self.inertia)
        self._reset_cache()


class FactorGraph(Distribution):
    """A factor graph object.

	A factor graph represents a probability distribution as a bipartite graph
	where marginal distributions of each dimension in the distribution are on
	one side of the graph and factors are on the other side. The distributions
	on the factor side encode probability estimates from the model, whereas the
	distributions on the marginal side encode probability estimates from the
	data. 

	Inference is done on the factor graph using the loopy belief propagation
	algorithm. This is an iterative algorithm where "messages" are passed
	along each edge between the marginals and the factors until the estimates
	for the marginals converges. In brief: each message represents what the
	generating node thinks its marginal distribution is with respect to the
	child node. Calculating each message involves marginalizing the parent
	node with respect to *every other* node. When the parent node is already
	a univariate distribution -- either because it is a marginal node or
	a univariate factor node -- no marginalization is needed and it sends
	itself as the message. Basically, a joint probability table will receive
	messages from all the marginal nodes that comprise its dimensions and,
	to each of those marginal nodes, it will send a message back saying what
	it (the joint probability table) thinks its marginal distribution is with
	respect to the messages from the OTHER marginals. More concretely, if a
	joint probability table has two dimensions with marginal node parents
	A and B, it will send a message to A that is itself after marginalizing out
	B, and will send a message to B that is itself after marginalizing out A. 

	..note:: It is worth noting that this algorithm is exact when the structure
	is a tree. If there exist any loops in the factors, i.e., you can draw a
	circle beginning with a factor and then hopping between marginals and
	factors and make it back to the factor without crossing any edges twice,
	the probabilities returned are approximate.


	Parameters
	----------
	factors: tuple or list or None
		A set of distribution objects. These do not need to be initialized,
		i.e. can be "Categorical()". Currently, they must be either Categorical
		or JointCategorical distributions. Default is None.

	marginals: tuple or list or None
		A set of distribution objects. These must be initialized and be
		Categorical distributions.

	edges: list or tuple or None
		A set of edges. Critically, the items in this list must be the
		distribution objects themselves, and the order that edges must match
		the order distributions in a multivariate distribution. Specifically,
		if you have a multivariate distribution, the first edge that includes
		it must correspond to the first dimension, the second edge must
		correspond to the second dimension, etc, and the total number of
		edges cannot exceed the number of dimensions. Default is None.

	max_iter: int, optional
		The number of iterations to do in the inference step as distributions
		are converging. Default is 10.

	tol: float, optional
		The threshold at which to stop during fitting when the improvement
		goes under. Default is 1e-6.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.

	verbose: bool, optional
		Whether to print the improvement and timings during training.
	"""

    def __init__(self, factors=None, marginals=None, edges=None, max_iter=20, tol=1e-06, inertia=0.0, frozen=False, check_data=True, verbose=False):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'FactorGraph'
        self.factors = torch.nn.ModuleList([])
        self.marginals = torch.nn.ModuleList([])
        self._factor_idxs = {}
        self._marginal_idxs = {}
        self._factor_edges = []
        self._marginal_edges = []
        self.max_iter = _check_parameter(max_iter, 'max_iter', min_value=1, dtypes=[int, torch.int16, torch.int32, torch.int64])
        self.tol = _check_parameter(tol, 'tol', min_value=0)
        self.verbose = verbose
        self.d = 0
        self._initialized = factors is not None and factors[0]._initialized
        if factors is not None:
            _check_parameter(factors, 'factors', dtypes=(list, tuple))
            for factor in factors:
                self.add_factor(factor)
        if marginals is not None:
            _check_parameter(marginals, 'marginals', dtypes=(list, tuple))
            for marginal in marginals:
                self.add_marginal(marginal)
        if edges is not None:
            _check_parameter(edges, 'edges', dtypes=(list, tuple))
            for marginal, factor in edges:
                self.add_edge(marginal, factor)
        self._initialized = not factors

    def _initialize(self, d):
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        return

    def add_factor(self, distribution):
        """Adds a distribution to the set of factors.

		
		Parameters
		----------
		distribution: pomegranate.distributions.Distribution
			A distribution object to include as a node.
		"""
        if not isinstance(distribution, (Categorical, JointCategorical)):
            raise ValueError('Must be a Categorical or a JointCategorical distribution.')
        self.factors.append(distribution)
        self._factor_edges.append([])
        self._factor_idxs[distribution] = len(self.factors) - 1
        self._initialized = distribution._initialized

    def add_marginal(self, distribution):
        """Adds a distribution to the set of marginals.

		This adds a distribution to the marginal side of the bipartate graph.
		This distribution must be univariate. 

		Parameters
		----------
		distribution: pomegranate.distributions.Distribution
			A distribution object to include as a node.
		"""
        if not isinstance(distribution, Categorical):
            raise ValueError('Must be a Categorical distribution.')
        self.marginals.append(distribution)
        self._marginal_edges.append([])
        self._marginal_idxs[distribution] = len(self.marginals) - 1
        self.d += 1

    def add_edge(self, marginal, factor):
        """Adds an undirected edge to the set of edges.

		Because a factor graph is a bipartite graph, one of the edges must be
		a marginal distribution and the other edge must be a factor 
		distribution.

		Parameters
		----------
		marginal: pomegranate.distributions.Distribution
			The marginal distribution to include in the edge.

		factor: pomegranate.distributions.Distribution
			The factor distribution to include in the edge.
		"""
        if marginal not in self._marginal_idxs:
            raise ValueError('Marginal distribution does not exist in graph.')
        if factor not in self._factor_idxs:
            raise ValueError('Factor distribution does not exist in graph.')
        m_idx = self._marginal_idxs[marginal]
        f_idx = self._factor_idxs[factor]
        self._factor_edges[f_idx].append(m_idx)
        self._marginal_edges[m_idx].append(f_idx)

    def log_probability(self, X):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.

		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, check_parameter=self.check_data)
        logps = torch.zeros(X.shape[0], device=X.device, dtype=torch.float32)
        for idxs, factor in zip(self._factor_edges, self.factors):
            logps += factor.log_probability(X[:, idxs])
        for i, marginal in enumerate(self.marginals):
            logps += marginal.log_probability(X[:, i:i + 1])
        return logps

    def predict(self, X):
        """Infers the maximum likelihood value for each missing value.

		This method infers a probability distribution for each of the missing
		values in the data. First, the sum-product algorithm is run to infer
		a probability distribution for each variable. Then, the maximum
		likelihood value is returned from that distribution.

		The input to this method must be a torch.masked.MaskedTensor where the
		mask specifies which variables are observed (mask = True) and which ones
		are not observed (mask = False) for each of the values. When setting
		mask = False, it does not matter what the corresponding value in the
		tensor is. Different sets of variables can be observed or missing in
		different examples. 

		Unlike the `predict_proba` and `predict_log_proba` methods, this
		method preserves the dimensions of the original data because it does
		not matter how many categories a variable can take when you're only
		returning the maximally likely one.


		Parameters
		----------
		X: torch.masked.MaskedTensor
			A masked tensor where the observed values are available and the
			unobserved values are missing, i.e., the mask is True for
			observed values and the mask is False for missing values. It does
			not matter what the underlying value in the tensor is for the 
			missing values.
		"""
        y = [t.argmax(dim=1) for t in self.predict_proba(X)]
        return torch.vstack(y).T.contiguous()

    def predict_proba(self, X):
        """Predict the probability of each variable given some evidence.

		Given some evidence about the value that each variable takes, infer
		the value that each other variable takes. If no evidence is given,
		this returns the marginal value of each variable given the dependence
		structure in the network.

		Currently, only hard evidence is supported, where the evidence takes
		the form of a discrete value. The evidence is represented as a
		masked tensor where the masked out values are considered missing.


		Parameters
		----------
		X: torch.masked.MaskedTensor
			A masked tensor where the observed values are available and the
			unobserved values are missing, i.e., the mask is True for
			observed values and the mask is False for missing values. It does
			not matter what the underlying value in the tensor is for the 
			missing values.
		"""
        nm = len(self.marginals)
        nf = len(self.factors)
        if X.shape[1] != nm:
            raise ValueError('X.shape[1] must match the number of marginals.')
        factors = []
        marginals = []
        prior_marginals = []
        current_marginals = []
        for i, m in enumerate(self.marginals):
            p = torch.clone(m.probs[0])
            p = p.repeat((X.shape[0],) + tuple(1 for _ in p.shape))
            for j in range(X.shape[0]):
                if X._masked_mask[j, i] == True:
                    value = X._masked_data[j, i]
                    p[j] = 0
                    p[j, value] = 1.0
            marginals.append(p)
            prior_marginals.append(torch.clone(p))
            current_marginals.append(torch.clone(p))
        for i, f in enumerate(self.factors):
            if not isinstance(f, Categorical):
                p = torch.clone(f.probs)
            else:
                p = torch.clone(f.probs[0])
            p = p.repeat((X.shape[0],) + tuple(1 for _ in p.shape))
            factors.append(p)
        in_messages, out_messages = [], []
        for i, m in enumerate(marginals):
            k = len(self._marginal_edges[i])
            in_messages.append([])
            for j in range(k):
                in_messages[-1].append(m)
        for i in range(len(factors)):
            k = len(self._factor_edges[i])
            out_messages.append([])
            for j in range(k):
                marginal_idx = self._factor_edges[i][j]
                d_j = marginals[marginal_idx]
                out_messages[-1].append(d_j)
        iteration = 0
        while iteration < self.max_iter:
            for i, f in enumerate(factors):
                ni_edges = len(self._factor_edges[i])
                for k in range(ni_edges):
                    message = torch.clone(f)
                    shape = torch.ones(len(message.shape), dtype=torch.int32)
                    shape[0] = X.shape[0]
                    for l in range(ni_edges):
                        if k == l:
                            continue
                        shape[l + 1] = message.shape[l + 1]
                        message *= out_messages[i][l].reshape(*shape)
                        message = torch.sum(message, dim=l + 1, keepdims=True)
                        shape[l + 1] = 1
                    else:
                        message = message.squeeze()
                        if len(message.shape) == 1:
                            message = message.unsqueeze(0)
                    j = self._factor_edges[i][k]
                    for ik, parent in enumerate(self._marginal_edges[j]):
                        if parent == i:
                            dims = tuple(range(1, len(message.shape)))
                            in_messages[j][ik] = message / message.sum(dim=dims, keepdims=True)
                            break
            loss = 0
            for i, m in enumerate(marginals):
                current_marginals[i] = torch.clone(m)
                for k in range(len(self._marginal_edges[i])):
                    current_marginals[i] *= in_messages[i][k]
                dims = tuple(range(1, len(current_marginals[i].shape)))
                current_marginals[i] /= current_marginals[i].sum(dim=dims, keepdims=True)
                loss += torch.nn.KLDivLoss(reduction='batchmean')(torch.log(current_marginals[i] + 1e-08), prior_marginals[i])
            if self.verbose:
                None
            if loss < self.tol:
                break
            for i, m in enumerate(marginals):
                ni_edges = len(self._marginal_edges[i])
                for k in range(ni_edges):
                    message = torch.clone(m)
                    for l in range(ni_edges):
                        if k == l:
                            continue
                        message *= in_messages[i][l]
                    j = self._marginal_edges[i][k]
                    for ik, parent in enumerate(self._factor_edges[j]):
                        if parent == i:
                            dims = tuple(range(1, len(message.shape)))
                            out_messages[j][ik] = message / message.sum(dim=dims, keepdims=True)
                            break
            prior_marginals = [torch.clone(d) for d in current_marginals]
            iteration += 1
        return current_marginals

    def predict_log_proba(self, X):
        """Infers the probability of each category given the model and data.

		This method is a wrapper around the `predict_proba` method and simply
		takes the log of each returned tensor.

		This method infers a log probability distribution for each of the 
		missing  values in the data. It uses the factor graph representation of 
		the Bayesian network to run the sum-product/loopy belief propagation
		algorithm.

		The input to this method must be a torch.masked.MaskedTensor where the
		mask specifies which variables are observed (mask = True) and which ones
		are not observed (mask = False) for each of the values. When setting
		mask = False, it does not matter what the corresponding value in the
		tensor is. Different sets of variables can be observed or missing in
		different examples. 

		An important note is that, because each variable can have a different
		number of categories in the categorical setting, the return is a list
		of tensors where each element in that list is the marginal probability
		distribution for that variable. More concretely: the first element will
		be the distribution of values for the first variable across all
		examples. When the first variable has been provided as evidence, the
		distribution will be clamped to the value provided as evidence.

		..warning:: This inference is exact given a Bayesian network that has
		a tree-like structure, but is only approximate for other cases. When
		the network is acyclic, this procedure will converge, but if the graph
		contains cycles then there is no guarantee on convergence.


		Parameters
		----------
		X: torch.masked.MaskedTensor, shape=(-1, d)
			The data to predict values for. The mask should correspond to
			whether the variable is observed in the example. 
		

		Returns
		-------
		y: list of tensors, shape=(d,)
			A list of tensors where each tensor contains the distribution of
			values for that dimension.
		"""
        return [torch.log(t) for t in self.predict_proba(X)]

    def fit(self, X, sample_weight=None):
        """Fit the factors of the model to optionally weighted examples.

		This method will fit the provided factor distributions to the given
		data and their optional weights. It will not update the marginal
		distributions, as that information is already encoded in the joint
		probabilities.

		..note:: A structure must already be provided. Currently, structure
		learning of factor graphs is not supported.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		self
		"""
        self.summarize(X, sample_weight=sample_weight)
        self.from_summaries()
        return self

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache for each distribution
		in the network. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        if self.frozen:
            return
        X, sample_weight = super().summarize(X, sample_weight=sample_weight)
        X = _check_parameter(X, 'X', ndim=2, check_parameter=self.check_data)
        for i, factor in enumerate(self.factors):
            factor.summarize(X[:, self._factor_edges[i]], sample_weight=sample_weight[:, i])

    def from_summaries(self):
        if self.frozen:
            return
        for distribution in self.factors:
            distribution.from_summaries()


def _initialize_centroids(X, k, algorithm='first-k', random_state=None):
    if isinstance(k, torch.Tensor):
        k = k.item()
    if not isinstance(random_state, numpy.random.mtrand.RandomState):
        random_state = numpy.random.RandomState(random_state)
    if algorithm == 'first-k':
        return _cast_as_tensor(torch.clone(X[:k]), dtype=torch.float32)
    elif algorithm == 'random':
        idxs = random_state.choice(len(X), size=k, replace=False)
        return _cast_as_tensor(torch.clone(X[idxs]), dtype=torch.float32)
    elif algorithm == 'submodular-facility-location':
        selector = FacilityLocationSelection(k, random_state=random_state)
        return _cast_as_tensor(selector.fit_transform(X), dtype=torch.float32)
    elif algorithm == 'submodular-feature-based':
        selector = FeatureBasedSelection(k, random_state=random_state)
        return selector.fit_transform(X)


class KMeans(torch.nn.Module):
    """A KMeans clustering object.

	Although K-means clustering is not a probabilistic model by itself,
	necessarily, it can be a useful initialization for many probabilistic
	methods, and can also be thought of as a specific formulation of a mixture
	model. Specifically, if you have a Gaussian mixture model with diagonal
	covariances set to 1/inf, in theory you will get the exact same results
	as k-means clustering.

	The implementation is provided here not necessarily to compete with other
	implementations, but simply to use as a consistent initialization.


	Parameters
	----------
	k: int or None, optional
		The number of clusters to initialize. Default is None

	centroids: list, numpy.ndarray, torch.Tensor, or None, optional
		A set of centroids to use to initialize the algorithm. Default is None.

	init: str, optional
		The initialization to use if `centroids` are not provided. Default is
		'first-k'. Must be one of:

			'first-k': Use the first k examples from the data set
			'random': Use a random set of k examples from the data set
			'submodular-facility-location': Use a facility location submodular
				objective to initialize the k-means algorithm
			'submodular-feature-based': Use a feature-based submodular objective
				to initialize the k-means algorithm.

	max_iter: int, optional
		The number of iterations to do in the EM step of fitting the
		distribution. Default is 10.

	tol: float, optional
		The threshold at which to stop during fitting when the improvement
		goes under. Default is 0.1.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	verbose: bool, optional
		Whether to print the improvement and timings during training.
	"""

    def __init__(self, k=None, centroids=None, init='first-k', max_iter=10, tol=0.1, inertia=0.0, frozen=False, random_state=None, verbose=False):
        super().__init__()
        self.name = 'KMeans'
        self._device = _cast_as_parameter([0.0])
        self.centroids = _check_parameter(_cast_as_parameter(centroids, dtype=torch.float32), 'centroids', ndim=2)
        self.k = _check_parameter(_cast_as_parameter(k), 'k', ndim=0, min_value=2, dtypes=(int, torch.int32, torch.int64))
        self.init = _check_parameter(init, 'init', value_set=('random', 'first-k', 'submodular-facility-location', 'submodular-feature-based'), ndim=0, dtypes=(str,))
        self.max_iter = _check_parameter(_cast_as_tensor(max_iter), 'max_iter', ndim=0, min_value=1, dtypes=(int, torch.int32, torch.int64))
        self.tol = _check_parameter(_cast_as_tensor(tol), 'tol', ndim=0, min_value=0)
        self.inertia = _check_parameter(_cast_as_tensor(inertia), 'inertia', ndim=0, min_value=0.0, max_value=1.0)
        self.frozen = _check_parameter(_cast_as_tensor(frozen), 'frozen', ndim=0, value_set=(True, False))
        self.random_state = random_state
        self.verbose = _check_parameter(verbose, 'verbose', value_set=(True, False))
        if self.k is None and self.centroids is None:
            raise ValueError('Must specify one of `k` or `centroids`.')
        self.k = len(centroids) if centroids is not None else self.k
        self.d = len(centroids[0]) if centroids is not None else None
        self._initialized = centroids is not None
        self._reset_cache()

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except:
            return 'cpu'

    def _initialize(self, X):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			The data to use to initialize the model.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2)
        centroids = _initialize_centroids(X, self.k, algorithm=self.init, random_state=self.random_state)
        if isinstance(centroids, torch.masked.MaskedTensor):
            centroids = centroids._masked_data * centroids._masked_mask
        if centroids.device != self.device:
            centroids = centroids
        self.centroids = _cast_as_parameter(centroids)
        self.d = X.shape[1]
        self._initialized = True
        self._reset_cache()

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        self.register_buffer('_w_sum', torch.zeros(self.k, self.d, device=self.device))
        self.register_buffer('_xw_sum', torch.zeros(self.k, self.d, device=self.device))
        self.register_buffer('_centroid_sum', torch.sum(self.centroids ** 2, dim=1).unsqueeze(0))

    def _distances(self, X):
        """Calculate the distances between each example and each centroid.

		This method calculates the distance between each example and each
		centroid in the model. These distances make up the backbone of the
		k-means learning and prediction steps.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.


		Returns
		-------
		distances: torch.Tensor, shape=(-1, self.k)
			The Euclidean distance between each example and each cluster.
		"""
        X = _check_parameter(_cast_as_tensor(X, dtype=torch.float32), 'X', ndim=2, shape=(-1, self.d))
        XX = torch.sum(X ** 2, dim=1).unsqueeze(1)
        if isinstance(X, torch.masked.MaskedTensor):
            n = X._masked_mask.sum(dim=1).unsqueeze(1)
            Xc = torch.matmul(X._masked_data * X._masked_mask, self.centroids.T)
        else:
            n = X.shape[1]
            Xc = torch.matmul(X, self.centroids.T)
        distances = torch.empty(X.shape[0], self.k, dtype=X.dtype, device=self.device)
        distances[:] = torch.clamp(XX - 2 * Xc + self._centroid_sum, min=0)
        return torch.sqrt(distances / n)

    def predict(self, X):
        """Calculate the cluster assignment for each example.

		This method calculates cluster assignment for each example as the
		nearest centroid according to the Euclidean distance.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.


		Returns
		-------
		y: torch.Tensor, shape=(-1,)
			The predicted label for each example.
		"""
        return self._distances(X).argmin(dim=1)

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.

		For k-means clustering, this step is essentially performing the 'E' part
		of the EM algorithm on a batch of data, where examples are hard-assigned
		to distributions in the model and summaries are derived from that.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""
        if self.frozen:
            return 0
        if not self._initialized:
            self._initialize(X)
        X = _check_parameter(_cast_as_tensor(X, dtype=torch.float32), 'X', ndim=2, shape=(-1, self.d))
        sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight, dtype=torch.float32), device=self.device)
        distances = self._distances(X)
        y_hat = distances.argmin(dim=1)
        if isinstance(X, torch.masked.MaskedTensor):
            for i in range(self.k):
                idx = y_hat == i
                self._w_sum[i][:] = self._w_sum[i] + sample_weight[idx].sum(dim=0)
                self._xw_sum[i][:] = self._xw_sum[i] + (X[idx] * sample_weight[idx]).sum(dim=0)
        else:
            y_hat = y_hat.unsqueeze(1).expand(-1, self.d)
            self._w_sum.scatter_add_(0, y_hat, sample_weight)
            self._xw_sum.scatter_add_(0, y_hat, X * sample_weight)
        return distances.min(dim=1).values.sum()

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        if self.frozen:
            return
        centroids = self._xw_sum / self._w_sum
        _update_parameter(self.centroids, centroids, self.inertia)
        self._reset_cache()

    def fit(self, X, sample_weight=None):
        """Fit the model to optionally weighted examples.

		This method implements the core of the learning process. For a
		mixture model, this involves performing EM until the distributions that
		are being fit converge according to the threshold set by `tol`, or
		until the maximum number of iterations has been hit.

		This method is largely a wrapper around the `summarize` and
		`from_summaries` methods. It's primary contribution is serving as a
		loop around these functions and to monitor convergence.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		self
		"""
        d_current = None
        for i in range(self.max_iter):
            start_time = time.time()
            d_previous = d_current
            d_current = self.summarize(X, sample_weight=sample_weight)
            if i > 0:
                improvement = d_previous - d_current
                duration = time.time() - start_time
                if self.verbose:
                    None
                if improvement < self.tol:
                    break
            self.from_summaries()
        self._reset_cache()
        return self

    def fit_predict(self, X, sample_weight=None):
        """Fit the model and then return the predictions.

		This function wraps a call to the `fit` function and then to the
		`predict` function. Essentially, k-means will be fit to the data and
		the resulting clustering assignments will be returned.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		y: torch.Tensor, shape=(-1,)
			Cluster assignments for each example.
		"""
        self.fit(X, sample_weight=sample_weight)
        return self.predict(X)


class GeneralMixtureModel(BayesMixin, Distribution):
    """A general mixture model.

	Frequently, data is generated from multiple components. A mixture model
	is a probabilistic model that explicitly models data as having come from
	a set of probability distributions rather than a single one. Usually, the
	abbreviation "GMM" refers to a Gaussian mixture model, but any probability
	distribution or heterogeneous set of distributions can be included in the
	mixture, making it a "general" mixture model.

	However, a mixture model itself has all the same theoretical properties as a
	probability distribution because it is one. Hence, it can be used in any
	situation that a simpler distribution could, such as an emission
	distribution for a HMM or a component of a Bayes classifier.

	Conversely, many models that are usually thought of as composed of
	probability distributions but distinct from them, e.g. hidden Markov models,
	Markov chains, and Bayesian networks, can in theory be passed into this
	object and incorporated into the mixture.

	If the distributions included in the mixture are not initialized, the
	fitting step will first initialize them by running k-means for a small
	number of iterations and fitting the distributions to the clusters that
	are discovered.


	Parameters
	----------
	distributions: tuple or list
		A set of distribution objects. These objects do not need to be
		initialized, i.e., can be "Normal()". 

	priors: tuple, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The prior probabilities over the given distributions. Default is None.

	max_iter: int, optional
		The number of iterations to do in the EM step of fitting the
		distribution. Default is 10.

	tol: float, optional
		The threshold at which to stop during fitting when the improvement
		goes under. Default is 0.1.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling. Default is True.

	verbose: bool, optional
		Whether to print the improvement and timings during training.
	"""

    def __init__(self, distributions, priors=None, init='random', max_iter=1000, tol=0.1, inertia=0.0, frozen=False, random_state=None, check_data=True, verbose=False):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'GeneralMixtureModel'
        _check_parameter(distributions, 'distributions', dtypes=(list, tuple, numpy.array, torch.nn.ModuleList))
        self.distributions = torch.nn.ModuleList(distributions)
        self.priors = _check_parameter(_cast_as_parameter(priors), 'priors', min_value=0, max_value=1, ndim=1, value_sum=1.0, shape=(len(distributions),))
        self.verbose = verbose
        self.k = len(distributions)
        if all(d._initialized for d in distributions):
            self._initialized = True
            self.d = distributions[0].d
            if self.priors is None:
                self.priors = _cast_as_parameter(torch.ones(self.k) / self.k)
        else:
            self._initialized = False
            self.d = None
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self._reset_cache()

    def _initialize(self, X, sample_weight=None):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			The data to use to initialize the model.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2)
        if sample_weight is None:
            sample_weight = torch.ones(1, dtype=self.dtype, device=self.device).expand(X.shape[0], 1)
        else:
            sample_weight = _check_parameter(_cast_as_tensor(sample_weight), 'sample_weight', min_value=0.0, check_parameter=self.check_data)
        model = KMeans(self.k, init=self.init, max_iter=3, random_state=self.random_state)
        if self.device != model.device:
            model
        y_hat = model.fit_predict(X, sample_weight=sample_weight)
        self.priors = _cast_as_parameter(torch.empty(self.k, dtype=self.dtype, device=self.device))
        sample_weight_sum = sample_weight.sum()
        for i in range(self.k):
            idx = y_hat == i
            sample_weight_idx = sample_weight[idx]
            self.distributions[i].fit(X[idx], sample_weight=sample_weight_idx)
            self.priors[i] = sample_weight_idx.sum() / sample_weight_sum
        self._initialized = True
        self._reset_cache()
        super()._initialize(X.shape[1])

    def sample(self, n):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution. For a mixture model, this involves first
		sampling the component using the prior probabilities, and then sampling
		from the chosen distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""
        X = []
        for distribution in self.distributions:
            X_ = distribution.sample(n)
            X.append(X_)
        X = torch.stack(X)
        idxs = torch.multinomial(self.priors, num_samples=n, replacement=True)
        return X[idxs, torch.arange(n)]

    def fit(self, X, sample_weight=None, priors=None):
        """Fit the model to optionally weighted examples.

		This method implements the core of the learning process. For a
		mixture model, this involves performing EM until the distributions that
		are being fit converge according to the threshold set by `tol`, or
		until the maximum number of iterations has been hit.

		This method is largely a wrapper around the `summarize` and
		`from_summaries` methods. It's primary contribution is serving as a
		loop around these functions and to monitor convergence.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			This can be used when only some labels are known by using a
			uniform distribution when the labels are not known. Note that 
			this can be used to assign hard labels, but does not have the 
			same semantics for soft labels, in that it only influences the 
			initial estimate of an observation being generated by a component, 
			not gives a target. Default is None.


		Returns
		-------
		self
		"""
        logp = None
        for i in range(self.max_iter):
            start_time = time.time()
            last_logp = logp
            logp = self.summarize(X, sample_weight=sample_weight, priors=priors)
            if i > 0:
                improvement = logp - last_logp
                duration = time.time() - start_time
                if self.verbose:
                    None
                if improvement < self.tol:
                    break
            self.from_summaries()
        self._reset_cache()
        return self

    def summarize(self, X, sample_weight=None, priors=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example. Labels can be provided for examples but, if provided,
		must be incomplete such that semi-supervised learning can be performed.

		For a mixture model, this step is essentially performing the 'E' part
		of the EM algorithm on a batch of data, where examples are soft-assigned
		to distributions in the model and summaries are derived from that.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		logp: float
			The log probability of X given the model.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2)
        if not self._initialized:
            self._initialize(X, sample_weight=sample_weight)
        sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight, dtype=torch.float32), device=self.device)
        e = self._emission_matrix(X, priors=priors)
        logp = torch.logsumexp(e, dim=1, keepdims=True)
        y = torch.exp(e - logp)
        z = torch.clone(self._w_sum)
        for i, d in enumerate(self.distributions):
            d.summarize(X, y[:, i:i + 1] * sample_weight)
            if self.frozen == False:
                self._w_sum[i] = self._w_sum[i] + (y[:, i:i + 1] * sample_weight).mean(dim=-1).sum()
        return torch.sum(logp)


class Silent(Distribution):

    def __init__(self):
        super().__init__(inertia=0.0, frozen=False, check_data=True)


def _check_inputs(model, X, emissions, priors):
    if X is None and emissions is None:
        raise ValueError('Must pass in one of `X` or `emissions`.')
    emissions = _check_parameter(_cast_as_tensor(emissions), 'emissions', ndim=3)
    if emissions is None:
        emissions = model._emission_matrix(X, priors=priors)
    return emissions


def partition_sequences(X, sample_weight=None, priors=None, n_dists=None):
    """Partition a set of sequences into blobs of equal length.

	This function will take in a list of sequences, where each sequence is
	a different length, and group together sequences of the same length so that
	batched operations can be more efficiently done on them. 

	Alternatively, it can take in sequences in the correct format and simply 
	return them. The correct form is to be either a single tensor that has
	three dimensions or a list of three dimensional tensors, where each
	tensor contains all the sequences of the same length.

	More concretely, the input to this function can be:

		- A single 3D tensor, or blob of data that can be cast into a 3D
		tensor, which gets returned

		- A list of 3D tensors, or elements that can be cast into 3D tensors, 
		which also gets returned

		- A list of 2D tensors, which get partitioned into a list of 3D tensors


	Parameters
	----------
	X: list or tensor
		The input sequences to be partitioned if in an incorrect format.

	sample_weight: list or tensor or None, optional
		The input sequence weights for the sequences or None. If None, return
		None. Default is None.

	priors: list or tensor or None
		The input sequence priors for the sequences or None. If None, return
		None. Default is None.

	n_dists: int or None
		The expected last dimension of the priors tensor. Must be provided if
		`priors` is provided. Default is None.


	Returns
	-------
	X_: list or tensor
		The partitioned and grouped sequences.
	"""
    if priors is not None and n_dists is None:
        raise RuntimeError('If priors are provided, n_dists must be provided as well.')
    try:
        X = [_check_parameter(_cast_as_tensor(X), 'X', ndim=3)]
    except:
        pass
    else:
        if sample_weight is not None:
            sample_weight = [_check_parameter(_cast_as_tensor(sample_weight), 'sample_weight', min_value=0.0)]
        if priors is not None:
            priors = [_check_parameter(_cast_as_tensor(priors), 'priors', ndim=3, shape=(*X[0].shape[:-1], n_dists))]
        return X, sample_weight, priors
    X = [_cast_as_tensor(x) for x in X]
    try:
        X = [_check_parameter(x, 'X', ndim=3) for x in X]
    except:
        pass
    else:
        if sample_weight is not None:
            sample_weight = [_check_parameter(_cast_as_tensor(w_), 'sample_weight', min_value=0.0) for w_ in sample_weight]
        if priors is not None:
            priors = [_check_parameter(_cast_as_tensor(p), 'priors', ndim=3, shape=(*X[i].shape[:-1], n_dists)) for i, p in enumerate(priors)]
        if all([(x.ndim == 3) for x in X]):
            return X, sample_weight, priors
    X_dict = collections.defaultdict(list)
    sample_weight_dict = collections.defaultdict(list)
    priors_dict = collections.defaultdict(list)
    for i, x in enumerate(X):
        x = _check_parameter(x, 'X', ndim=2)
        n = len(x)
        X_dict[n].append(x)
        if sample_weight is not None:
            w = _check_parameter(_cast_as_tensor(sample_weight[i]), 'sample_weight', min_value=0.0)
            sample_weight_dict[n].append(w)
        if priors is not None:
            p = _check_parameter(_cast_as_tensor(priors[i]), 'priors', ndim=2, shape=(*x.shape[:-1], n_dists))
            priors_dict[n].append(p)
    keys = sorted(X_dict.keys())
    X_ = [torch.stack(X_dict[key]) for key in keys]
    if sample_weight is None:
        sample_weight_ = None
    else:
        sample_weight_ = [torch.stack(sample_weight_dict[key]) for key in keys]
    if priors is None:
        priors_ = None
    else:
        priors_ = [torch.stack(priors_dict[key]) for key in keys]
    return X_, sample_weight_, priors_


class _BaseHMM(Distribution):
    """A base hidden Markov model.

	A hidden Markov model is an extension of a mixture model to sequences by
	including a transition matrix between the elements of the mixture. Each of
	the algorithms for a hidden Markov model are essentially just a revision
	of those algorithms to incorporate this transition matrix.

	There are two main ways one can implement a hidden Markov model: either the
	transition matrix can be implemented in a dense, or a sparse, manner. If the
	transition matrix is dense, implementing it in a dense manner allows for
	the primary computation to use matrix multiplications which can be very
	fast. However, if the matrix is sparse, these matrix multiplications will
	be fairly slow and end up significantly slower than the sparse version of
	a matrix multiplication.

	This object is a wrapper for both implementations, which can be specified
	using the `kind` parameter. Choosing the right implementation will not
	effect the accuracy of the results but will change the speed at which they
	are calculated. 	

	Separately, there are two ways to instantiate the hidden Markov model. The
	first is by passing in a set of distributions, a dense transition matrix, 
	and optionally start/end probabilities. The second is to initialize the
	object without these and then to add edges using the `add_edge` method
	and to add nodes using the `add_nodes` method. Importantly, the way that
	you choose to initialize the hidden Markov model is independent of the
	implementation that you end up choosing. If you pass in a dense transition
	matrix, this will be converted to a sparse matrix with all the zeros
	dropped if you choose `kind='sparse'`.


	Parameters
	----------
	distributions: tuple or list
		A set of distribution objects. These objects do not need to be
		initialized, i.e., can be "Normal()". 

	edges: numpy.ndarray, torch.Tensor, or None. shape=(k,k), optional
		A dense transition matrix of probabilities for how each node or
		distribution passed in connects to each other one. This can contain
		many zeroes, and when paired with `kind='sparse'`, will drop those
		elements from the matrix. Default is None.

	starts: list, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The probability of starting at each node. If not provided, assumes
		these probabilities are uniform. Default is None.

	ends: list, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The probability of ending at each node. If not provided, assumes
		these probabilities are uniform. Default is None.

	kind: str, 'sparse' or 'dense', optional
		The underlying implementation of the transition matrix to use.
		Default is 'sparse'. 

	init: str, optional
		The initialization to use for the k-means initialization approach.
		Default is 'first-k'. Must be one of:

			'first-k': Use the first k examples from the data set
			'random': Use a random set of k examples from the data set
			'submodular-facility-location': Use a facility location submodular
				objective to initialize the k-means algorithm
			'submodular-feature-based': Use a feature-based submodular objective
				to initialize the k-means algorithm.

	max_iter: int, optional
		The number of iterations to do in the EM step, which for HMMs is
		sometimes called Baum-Welch. Default is 10.

	tol: float, optional
		The threshold at which to stop during fitting when the improvement
		goes under. Default is 0.1.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	random_state: int or None, optional
		The random state to make randomness deterministic. If None, not
		deterministic. Default is None.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.

	verbose: bool, optional
		Whether to print the improvement and timings during training.
	"""

    def __init__(self, distributions=None, starts=None, ends=None, init='random', max_iter=1000, tol=0.1, sample_length=None, return_sample_paths=False, inertia=0.0, frozen=False, check_data=True, random_state=None, verbose=False):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.distributions = distributions
        n = len(distributions) if distributions is not None else None
        self.start = Silent()
        self.end = Silent()
        self.edges = None
        self.starts = None
        self.ends = None
        if starts is not None:
            starts = _check_parameter(_cast_as_tensor(starts), 'starts', ndim=1, shape=(n,), min_value=0.0, max_value=1.0, value_sum=1.0)
            self.starts = _cast_as_parameter(torch.log(starts))
        if ends is not None:
            ends = _check_parameter(_cast_as_tensor(ends), 'ends', ndim=1, shape=(n,), min_value=0.0, max_value=1.0)
            self.ends = _cast_as_parameter(torch.log(ends))
        if not isinstance(random_state, numpy.random.RandomState):
            self.random_state = numpy.random.RandomState(random_state)
        else:
            self.random_state = random_state
        self.init = init
        self.max_iter = _check_parameter(max_iter, 'max_iter', min_value=1, ndim=0, dtypes=(int, torch.int32, torch.int64))
        self.tol = _check_parameter(tol, 'tol', min_value=0.0, ndim=0)
        self.sample_length = sample_length
        self.return_sample_paths = return_sample_paths
        self.verbose = verbose
        self.d = self.distributions[0].d if distributions is not None else None

    def _initialize(self, X=None, sample_weight=None):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d), optional
			The data to use to initialize the model. Default is None.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, len) or a vector of shape (-1,). If None, defaults to ones.
			Default is None.
		"""
        n = self.n_distributions
        if self.starts is None:
            self.starts = _cast_as_parameter(torch.log(torch.ones(n, dtype=self.dtype, device=self.device) / n))
        if self.ends is None:
            self.ends = _cast_as_parameter(torch.log(torch.ones(n, dtype=self.dtype, device=self.device) / n))
        _init = all(d._initialized for d in self.distributions)
        if X is not None and not _init:
            if isinstance(X, list):
                d = _cast_as_tensor(X[0]).shape[-1]
                X = torch.cat([_cast_as_tensor(x).reshape(-1, d) for x in X], dim=0).unsqueeze(0)
            X = _check_parameter(_cast_as_tensor(X), 'X', ndim=3, check_parameter=self.check_data)
            X = X.reshape(-1, X.shape[-1])
            if sample_weight is None:
                sample_weight = torch.ones(1, dtype=self.dtype, device=self.device).expand(X.shape[0], 1)
            else:
                if isinstance(sample_weight, list):
                    sample_weight = torch.cat([_cast_as_tensor(w).reshape(-1, 1) for w in sample_weight], dim=0)
                sample_weight = _check_parameter(_cast_as_tensor(sample_weight).reshape(-1, 1), 'sample_weight', min_value=0.0, ndim=1, shape=(len(X),), check_parameter=self.check_data).reshape(-1, 1)
            model = KMeans(self.n_distributions, init=self.init, max_iter=1, random_state=self.random_state)
            y_hat = model.fit_predict(X, sample_weight=sample_weight)
            for i in range(self.n_distributions):
                self.distributions[i].fit(X[y_hat == i].cpu(), sample_weight=sample_weight[y_hat == i].cpu())
                self.distributions[i]
            self.d = X.shape[-1]
            super()._initialize(X.shape[-1])
        self._initialized = True
        self._reset_cache()

    def _emission_matrix(self, X, priors=None):
        """Return the emission/responsibility matrix.

		This method returns the log probability of each example under each
		distribution contained in the model with the log prior probability
		of each component added.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to evaluate. 

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.

	
		Returns
		-------
		e: torch.Tensor, shape=(-1, len, self.k)
			A set of log probabilities for each example under each distribution.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=3, shape=(-1, -1, self.d), check_parameter=self.check_data)
        n, k, _ = X.shape
        X = X.reshape(n * k, self.d)
        priors = _check_parameter(_cast_as_tensor(priors), 'priors', ndim=3, shape=(n, k, self.k), min_value=0.0, max_value=1.0, value_sum=1.0, value_sum_dim=-1, check_parameter=self.check_data)
        if not self._initialized:
            self._initialize()
        e = torch.empty((k, self.k, n), dtype=self.dtype, requires_grad=False, device=self.device)
        for i, node in enumerate(self.distributions):
            logp = node.log_probability(X)
            if isinstance(logp, torch.masked.MaskedTensor):
                logp = logp._masked_data
            e[:, i] = logp.reshape(n, k).T
        e = e.permute(2, 0, 1)
        if priors is not None:
            e += torch.log(priors)
        return e

    @property
    def k(self):
        return len(self.distributions) if self.distributions is not None else 0

    @property
    def n_distributions(self):
        return len(self.distributions) if self.distributions is not None else 0

    def freeze(self):
        """Freeze this model and all child distributions."""
        self.register_buffer('frozen', _cast_as_tensor(True))
        for d in self.distributions:
            d.freeze()
        return self

    def unfreeze(self):
        """Unfreeze this model and all child distributions."""
        self.register_buffer('frozen', _cast_as_tensor(False))
        for d in self.distributions:
            d.unfreeze()
        return self

    def add_distribution(self, distribution):
        """Add a distribution to the model.


		Parameters
		----------
		distribution: torchegranate.distributions.Distribution
			A distribution object.
		"""
        if self.distributions is None:
            self.distributions = []
        if not isinstance(distribution, Distribution):
            raise ValueError('distribution must be a distribution object.')
        self.distributions.append(distribution)
        self.d = distribution.d

    def add_distributions(self, distributions):
        """Add a set of distributions to the model.

		This method will iterative call the `add_distribution`.


		Parameters
		----------
		distributions: list, tuple, iterable
			A set of distributions to add to the model.
		"""
        for distribution in distributions:
            self.add_distribution(distribution)

    def add_edge(self, start, end, probability):
        """Add an edge to the model.

		This method takes in two distribution objects and the probability
		connecting the two and adds an edge to the model.


		Parameters
		----------
		start: torchegranate.distributions.Distribution
			The parent node for the edge

		end: torchegranate.distributions.Distribution
			The child node for the edge

		probability: float, (0, 1]
			The probability of connecting the two.
		"""
        if not isinstance(start, Distribution):
            raise ValueError('start must be a distribution.')
        if not isinstance(end, Distribution):
            raise ValueError('end must be a distribution.')
        if not isinstance(probability, float):
            raise ValueError('probability must be a float.')
        if self.edges is None:
            self.edges = []
        self.edges.append((start, end, probability))

    def probability(self, X, priors=None):
        """Calculate the probability of each example.

		This method calculates the probability of each example given the
		parameters of the distribution. The examples must be given in a 3D
		format.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		prob: torch.Tensor, shape=(-1,)
			The probability of each example.
		"""
        return torch.exp(self.log_probability(X, priors=priors))

    def log_probability(self, X, priors=None):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 3D
		format.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        f = self.forward(X, priors=priors)
        return torch.logsumexp(f[:, -1] + self.ends, dim=1)

    def predict_log_proba(self, X, priors=None):
        """Calculate the posterior probabilities for each example.

		This method calculates the log posterior probabilities for each example
		and then normalizes across each component of the model. These
		probabilities are calculated using the forward-backward algorithm.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.n_distributions)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		r: torch.Tensor, shape=(-1, len, self.n_distributions)
			The log posterior probabilities for each example under each 
			component as calculated by the forward-backward algorithm.
		"""
        _, r, _, _, _ = self.forward_backward(X, priors=priors)
        return r

    def predict_proba(self, X, priors=None):
        """Calculate the posterior probabilities for each example.

		This method calculates the posterior probabilities for each example
		and then normalizes across each component of the model. These
		probabilities are calculated using the forward-backward algorithm.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.n_distributions)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		y: torch.Tensor, shape=(-1, len, self.n_distributions)
			The posterior probabilities for each example under each component
			as calculated by the forward-backward algorithm.
		"""
        return torch.exp(self.predict_log_proba(X, priors=priors))

    def predict(self, X, priors=None):
        """Predicts the component for each observation.

		This method calculates the predicted component for each observation
		given the posterior probabilities as calculated by the forward-backward
		algorithm. Essentially, it is just the argmax over components.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		y: torch.Tensor, shape=(-1, len, self.k)
			The posterior probabilities for each example under each component
			as calculated by the forward-backward algorithm.
		"""
        return torch.argmax(self.predict_log_proba(X, priors=priors), dim=-1)

    def fit(self, X, sample_weight=None, priors=None):
        """Fit the model to sequences with optional weights and priors.

		This method implements the core of the learning process. For hidden
		Markov models, this is a form of EM called "Baum-Welch" or "structured
		EM". This iterative algorithm will proceed until converging, either
		according to the threshold set by `tol` or until the maximum number
		of iterations set by `max_iter` has been hit.

		This method is largely a wrapper around the `summarize` and
		`from_summaries` methods. It's primary contribution is serving as a
		loop around these functions and to monitor convergence.

		Unlike other HMM methods, this method can handle variable length
		sequences by accepting a list of tensors where each tensor has a
		different sequence length. Then, summarization is done on each tensor
		sequentially. This will provide an exact update as if the entire data
		set was seen at the same time but will allow batched operations to be
		performed on each variable length tensor.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to evaluate. Because sequences can be variable
			length, there are three ways to format the sequences.
			
				1. Pass in a tensor of shape (n, length, dim), which can only 
				be done when each sequence is the same length. 

				2. Pass in a list of 3D tensors where each tensor has the shape 
				(n, length, dim). In this case, each tensor is a collection of 
				sequences of the same length and so sequences of different 
				lengths can be trained on. 

				3. Pass in a list of 2D tensors where each tensor has the shape
				(length, dim). In this case, sequences of the same length will
				be grouped together into the same tensor and fitting will
				proceed as if you had passed in data like way 2.

		sample_weight: list, numpy.ndarray, torch.Tensor or None, optional
			A set of weights for the examples. These must follow the same format
			as X.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observations
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Must be formatted in the same
			shape as X. Default is None.


		Returns
		-------
		self
		"""
        X, sample_weight, priors = partition_sequences(X, sample_weight=sample_weight, priors=priors, n_dists=self.k)
        if not self._initialized:
            if sample_weight is None:
                self._initialize(X)
            else:
                self._initialize(X, sample_weight=sample_weight)
        logp, last_logp = None, None
        for i in range(self.max_iter):
            start_time = time.time()
            logp = 0
            for j, X_ in enumerate(X):
                w_ = None if sample_weight is None else sample_weight[j]
                p_ = None if priors is None else priors[j]
                logp += self.summarize(X_, sample_weight=w_, priors=p_).sum()
            if i > 0:
                improvement = logp - last_logp
                duration = time.time() - start_time
                if self.verbose:
                    None
                if improvement < self.tol:
                    self._reset_cache()
                    return self
            last_logp = logp
            self.from_summaries()
        if self.verbose:
            logp = 0
            for j, X_ in enumerate(X):
                w_ = None if sample_weight is None else sample_weight[j]
                p_ = None if priors is None else priors[j]
                logp += self.summarize(X_, sample_weight=w_, priors=p_).sum()
            improvement = logp - last_logp
            duration = time.time() - start_time
            None
        self._reset_cache()
        return self

    def summarize(self, X, sample_weight=None, emissions=None, priors=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, length, self.d) or a vector of shape (-1,). Default is ones.

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.n_distributions)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=3, shape=(-1, -1, self.d), check_parameter=self.check_data)
        emissions = _check_inputs(self, X, emissions, priors)
        if sample_weight is None:
            sample_weight = torch.ones(1, device=self.device).expand(emissions.shape[0], 1)
        else:
            sample_weight = _check_parameter(_cast_as_tensor(sample_weight), 'sample_weight', min_value=0.0, ndim=1, shape=(emissions.shape[0],), check_parameter=self.check_data).reshape(-1, 1)
        if not self._initialized:
            self._initialize(X, sample_weight=sample_weight)
        return X, emissions, sample_weight

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        self.from_summaries()
        self._reset_cache()


class DenseHMM(_BaseHMM):
    """A hidden Markov model with a dense transition matrix.

	A hidden Markov model is an extension of a mixture model to sequences by
	including a transition matrix between the elements of the mixture. Each of
	the algorithms for a hidden Markov model are essentially just a revision
	of those algorithms to incorporate this transition matrix.

	This object is a wrapper for a hidden Markov model with a dense transition
	matrix.

	This object is a wrapper for both implementations, which can be specified
	using the `kind` parameter. Choosing the right implementation will not
	effect the accuracy of the results but will change the speed at which they
	are calculated. 	

	Separately, there are two ways to instantiate the hidden Markov model. The
	first is by passing in a set of distributions, a dense transition matrix, 
	and optionally start/end probabilities. The second is to initialize the
	object without these and then to add edges using the `add_edge` method
	and to add distributions using the `add_distributions` method. Importantly, the way that
	you choose to initialize the hidden Markov model is independent of the
	implementation that you end up choosing. If you pass in a dense transition
	matrix, this will be converted to a sparse matrix with all the zeros
	dropped if you choose `kind='sparse'`.


	Parameters
	----------
	distributions: tuple or list
		A set of distribution objects. These objects do not need to be
		initialized, i.e., can be "Normal()". 

	edges: numpy.ndarray, torch.Tensor, or None. shape=(k,k), optional
		A dense transition matrix of probabilities for how each node or
		distribution passed in connects to each other one. This can contain
		many zeroes, and when paired with `kind='sparse'`, will drop those
		elements from the matrix.

	starts: list, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The probability of starting at each node. If not provided, assumes
		these probabilities are uniform.

	ends: list, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The probability of ending at each node. If not provided, assumes
		these probabilities are uniform.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling. Default is True.
	"""

    def __init__(self, distributions=None, edges=None, starts=None, ends=None, init='random', max_iter=1000, tol=0.1, sample_length=None, return_sample_paths=False, inertia=0.0, frozen=False, check_data=True, random_state=None, verbose=False):
        super().__init__(distributions=distributions, starts=starts, ends=ends, init=init, max_iter=max_iter, tol=tol, sample_length=sample_length, return_sample_paths=return_sample_paths, inertia=inertia, frozen=frozen, check_data=check_data, random_state=random_state, verbose=verbose)
        self.name = 'DenseHMM'
        n = len(distributions) if distributions is not None else 0
        if edges is not None:
            self.edges = _cast_as_parameter(torch.log(_check_parameter(_cast_as_tensor(edges), 'edges', ndim=2, shape=(n, n), min_value=0.0, max_value=1.0)))
        self._initialized = self.distributions is not None and self.starts is not None and self.ends is not None and self.edges is not None and all(d._initialized for d in self.distributions)
        if self._initialized:
            self.distributions = torch.nn.ModuleList(self.distributions)
        self._reset_cache()

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        for node in self.distributions:
            node._reset_cache()
        self.register_buffer('_xw_sum', torch.zeros(self.n_distributions, self.n_distributions, dtype=self.dtype, requires_grad=False, device=self.device))
        self.register_buffer('_xw_starts_sum', torch.zeros(self.n_distributions, dtype=self.dtype, requires_grad=False, device=self.device))
        self.register_buffer('_xw_ends_sum', torch.zeros(self.n_distributions, dtype=self.dtype, requires_grad=False, device=self.device))

    def _initialize(self, X=None, sample_weight=None):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d), optional
			The data to use to initialize the model. Default is None.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, len) or a vector of shape (-1,). If None, defaults to ones.
			Default is None.
		"""
        super()._initialize(X, sample_weight=sample_weight)
        n = self.n_distributions
        if self.edges == None:
            self.edges = _cast_as_parameter(torch.log(torch.ones(n, n, dtype=self.dtype, device=self.device) / n))
        self.distributions = torch.nn.ModuleList(self.distributions)

    def add_edge(self, start, end, prob):
        """Add an edge to the model.

		This method will fill in an entry in the dense transition matrix
		at row indexed by the start distribution and the column indexed
		by the end distribution. The value that will be included is the
		log of the probability value provided. Note that this will override
		values that already exist, and that this will initialize a new
		dense transition matrix if none has been passed in so far.


		Parameters
		----------
		start: torch.distributions.distribution
			The distribution that the edge starts at.

		end: torch.distributions.distribution
			The distribution that the edge ends at.

		prob: float, (0.0, 1.0]
			The probability of that edge.
		"""
        if self.distributions is None:
            raise ValueError('Must add distributions before edges.')
        n = self.n_distributions
        if start == self.start:
            if self.starts is None:
                self.starts = torch.empty(n, dtype=self.dtype, device=self.device) - inf
            idx = self.distributions.index(end)
            self.starts[idx] = math.log(prob)
        elif end == self.end:
            if self.ends is None:
                self.ends = torch.empty(n, dtype=self.dtype, device=self.device) - inf
            idx = self.distributions.index(start)
            self.ends[idx] = math.log(prob)
        else:
            if self.edges is None:
                self.edges = torch.empty((n, n), dtype=self.dtype, device=self.device) - inf
            idx1 = self.distributions.index(start)
            idx2 = self.distributions.index(end)
            self.edges[idx1, idx2] = math.log(prob)

    def sample(self, n):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution. Because a HMM describes variable length
		sequences, a list will be returned where each element is one of
		the generated sequences.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: list of torch.tensor, shape=(n,)
			A list of randomly generated samples, where each sample of
			size (length, self.d).
		"""
        if self.sample_length is None and self.ends is None:
            raise ValueError('Must specify a length or have explicit ' + 'end probabilities.')
        if self.ends is None:
            ends = torch.zeros(self.n_distributions, dtype=self.edges.dtype, device=self.edges.device) + float('-inf')
        else:
            ends = self.ends
        distributions, emissions = [], []
        edge_probs = torch.hstack([self.edges, ends.unsqueeze(1)])
        edge_probs = torch.exp(edge_probs).numpy()
        starts = torch.exp(self.starts).numpy()
        for _ in range(n):
            node_i = self.random_state.choice(self.n_distributions, p=starts)
            emission_i = self.distributions[node_i].sample(n=1)
            distributions_, emissions_ = [node_i], [emission_i]
            for i in range(1, self.sample_length or int(100000000.0)):
                node_i = self.random_state.choice(self.n_distributions + 1, p=edge_probs[node_i])
                if node_i == self.n_distributions:
                    break
                emission_i = self.distributions[node_i].sample(n=1)
                distributions_.append(node_i)
                emissions_.append(emission_i)
            distributions.append(distributions_)
            emissions.append(torch.vstack(emissions_))
        if self.return_sample_paths == True:
            return emissions, distributions
        return emissions

    def viterbi(self, X=None, emissions=None, priors=None):
        """Run the Viterbi algorithm on some data.

		Runs the Viterbi algortihm on a batch of sequences. The Viterbi 
		algorithm is a dynamic programming algorithm that begins at the start
		state and calculates the single best path through the model involving
		alignments of symbol i to node j. This is in contrast to the forward
		function, which involves calculating the sum of all paths, not just
		the single best path. Because we have to keep track of the best path,
		the Viterbi algorithm is slightly more conceptually challenging and
		involves keeping track of a traceback matrix.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			A set of examples to evaluate. Does not need to be passed in if
			emissions are. 

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_dists)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		path: torch.Tensor, shape=(-1, -1)
			The state assignment for each observation in each sequence.
		"""
        emissions = _check_inputs(self, X, emissions, priors)
        n, l = emissions.shape[:2]
        v = torch.clone(emissions.permute(1, 0, 2)).contiguous()
        v[0] += self.starts
        traceback = torch.zeros_like(v, dtype=torch.int32)
        traceback[0] = torch.arange(v.shape[-1])
        for i in range(1, l):
            z = v[i - 1].unsqueeze(-1) + self.edges.unsqueeze(0) + v[i][:, None]
            v[i], traceback[i] = torch.max(z, dim=-2)
        ends = self.ends + v[-1]
        best_end_logps, best_end_idxs = torch.max(ends, dim=-1)
        paths = [best_end_idxs]
        for i in range(1, l):
            paths.append(traceback[l - i, torch.arange(n), paths[-1]])
        paths = torch.flip(torch.stack(paths).T, dims=(-1,))
        return paths

    def forward(self, X=None, emissions=None, priors=None):
        """Run the forward algorithm on some data.

		Runs the forward algorithm on a batch of sequences. This is not to be
		confused with a "forward pass" when talking about neural networks. The
		forward algorithm is a dynamic programming algorithm that begins at the
		start state and returns the probability, over all paths through the
		model, that result in the alignment of symbol i to node j.

		
		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			A set of examples to evaluate. Does not need to be passed in if
			emissions are. 

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_dists)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		f: torch.Tensor, shape=(-1, -1, self.n_distributions)
			The log probabilities calculated by the forward algorithm.
		"""
        emissions = _check_inputs(self, X, emissions, priors)
        l = emissions.shape[1]
        t_max = self.edges.max()
        t = torch.exp(self.edges - t_max)
        f = torch.clone(emissions.permute(1, 0, 2)).contiguous()
        f[0] += self.starts
        f[1:] += t_max
        for i in range(1, l):
            p_max = torch.max(f[i - 1], dim=1, keepdims=True).values
            p = torch.exp(f[i - 1] - p_max)
            f[i] += torch.log(torch.matmul(p, t)) + p_max
        f = f.permute(1, 0, 2)
        return f

    def backward(self, X=None, emissions=None, priors=None):
        """Run the backward algorithm on some data.

		Runs the backward algorithm on a batch of sequences. This is not to be
		confused with a "backward pass" when talking about neural networks. The
		backward algorithm is a dynamic programming algorithm that begins at end
		of the sequence and returns the probability, over all paths through the
		model, that result in the alignment of symbol i to node j, working
		backwards.

		
		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			A set of examples to evaluate. Does not need to be passed in if
			emissions are. 

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		b: torch.Tensor, shape=(-1, length, self.n_distributions)
			The log probabilities calculated by the backward algorithm.
		"""
        emissions = _check_inputs(self, X, emissions, priors)
        n, l, _ = emissions.shape
        b = torch.zeros(l, n, self.n_distributions, dtype=self.dtype, device=self.device) + float('-inf')
        b[-1] = self.ends
        t_max = self.edges.max()
        t = torch.exp(self.edges.T - t_max)
        for i in range(l - 2, -1, -1):
            p = b[i + 1] + emissions[:, i + 1]
            p_max = torch.max(p, dim=1, keepdims=True).values
            p = torch.exp(p - p_max)
            b[i] = torch.log(torch.matmul(p, t)) + t_max + p_max
        b = b.permute(1, 0, 2)
        return b

    def forward_backward(self, X=None, emissions=None, priors=None):
        """Run the forward-backward algorithm on some data.

		Runs the forward-backward algorithm on a batch of sequences. This
		algorithm combines the best of the forward and the backward algorithm.
		It combines the probability of starting at the beginning of the sequence
		and working your way to each observation with the probability of
		starting at the end of the sequence and working your way backward to it.

		A number of statistics can be calculated using this information. These
		statistics are powerful inference tools but are also used during the
		Baum-Welch training process. 

		
		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			A set of examples to evaluate. Does not need to be passed in if
			emissions are. 

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		transitions: torch.Tensor, shape=(-1, n, n)
			The expected number of transitions across each edge that occur
			for each example. The returned transitions follow the structure
			of the transition matrix and so will be dense or sparse as
			appropriate.

		responsibility: torch.Tensor, shape=(-1, -1, n)
			The posterior probabilities of each observation belonging to each
			state given that one starts at the beginning of the sequence,
			aligns observations across all paths to get to the current
			observation, and then proceeds to align all remaining observations
			until the end of the sequence.

		starts: torch.Tensor, shape=(-1, n)
			The probabilities of starting at each node given the 
			forward-backward algorithm.

		ends: torch.Tensor, shape=(-1, n)
			The probabilities of ending at each node given the forward-backward
			algorithm.

		logp: torch.Tensor, shape=(-1,)
			The log probabilities of each sequence given the model.
		"""
        emissions = _check_inputs(self, X, emissions, priors)
        n, l, _ = emissions.shape
        f = self.forward(emissions=emissions)
        b = self.backward(emissions=emissions)
        logp = torch.logsumexp(f[:, -1] + self.ends, dim=1)
        f_ = f[:, :-1].unsqueeze(-1)
        b_ = (b[:, 1:] + emissions[:, 1:]).unsqueeze(-2)
        t = f_ + b_ + self.edges.unsqueeze(0).unsqueeze(0)
        t = t.reshape(n, l - 1, -1)
        t = torch.exp(torch.logsumexp(t, dim=1).T - logp).T
        t = t.reshape(n, int(t.shape[1] ** 0.5), -1)
        starts = self.starts + emissions[:, 0] + b[:, 0]
        starts = torch.exp(starts.T - torch.logsumexp(starts, dim=-1)).T
        ends = self.ends + f[:, -1]
        ends = torch.exp(ends.T - torch.logsumexp(ends, dim=-1)).T
        r = f + b
        r = r - torch.logsumexp(r, dim=2).reshape(n, -1, 1)
        return t, r, starts, ends, logp

    def summarize(self, X, sample_weight=None, emissions=None, priors=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: torch.Tensor, shape=(-1, -1, self.d)
			A set of examples to summarize.

		y: torch.Tensor, shape=(-1, -1), optional 
			A set of labels with the same number of examples and length as the
			observations that indicate which node in the model that each
			observation should be assigned to. Passing this in means that the
			model uses labeled training instead of Baum-Welch. Default is None.

		sample_weight: torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.

		emissions: torch.Tensor, shape=(-1, -1, self.n_distributions)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.	

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.
		"""
        X, emissions, sample_weight = super().summarize(X, sample_weight=sample_weight, emissions=emissions, priors=priors)
        t, r, starts, ends, logps = self.forward_backward(emissions=emissions)
        X = X.reshape(-1, X.shape[-1])
        r = torch.exp(r) * sample_weight.unsqueeze(-1)
        for i, node in enumerate(self.distributions):
            w = r[:, :, i].reshape(-1, 1)
            node.summarize(X, sample_weight=w)
        if self.frozen == False:
            self._xw_starts_sum += torch.sum(starts * sample_weight, dim=0)
            self._xw_ends_sum += torch.sum(ends * sample_weight, dim=0)
            self._xw_sum += torch.sum(t * sample_weight.unsqueeze(-1), dim=0)
        return logps

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        for node in self.distributions:
            node.from_summaries()
        if self.frozen:
            return
        node_out_count = torch.sum(self._xw_sum, dim=1, keepdims=True)
        node_out_count += self._xw_ends_sum.unsqueeze(1)
        ends = torch.log(self._xw_ends_sum / node_out_count[:, 0])
        starts = torch.log(self._xw_starts_sum / self._xw_starts_sum.sum())
        edges = torch.log(self._xw_sum / node_out_count)
        _update_parameter(self.ends, ends, inertia=self.inertia)
        _update_parameter(self.starts, starts, inertia=self.inertia)
        _update_parameter(self.edges, edges, inertia=self.inertia)
        self._reset_cache()


def unpack_edges(self, edges, starts, ends):
    """Unpack the edges for a sparse hidden Markov model.

	This function takes in a SparseHMM object and sets of edges and adds the
	edges to the model. It is designed to allow the model to be initialized
	either through passing in the edges initially or created through the
	`add_edge` API. Doing this is slightly more complicated than in the
	DenseHMM case because we use an underlying sparse representation to store
	the edges and so cannot as easily simply modify a random element in each
	call to `add_edge`. 


	Parameters
	----------
	self: SparseHMM
		A torchegranate sparse HMM.

	edges: list
		A list of 3-ples that consist of the parent distribution, the child
		distribution, and the probability on that edge.

	starts: list, tuple, numpy.ndarray, torch.tensor
		A vector of probabilities indicating the probability of starting in
		each state. Must sum to 1.0.

	ends: list, tuple, numpy.ndarray, torch.tensor
		A vector of probabilities indicating the probability of ending at
		each state. Does not have to sum to 1.0.
	"""
    self.n_edges = len(edges)
    n = len(self.distributions)
    self.starts = None
    if starts is not None:
        starts = _check_parameter(_cast_as_tensor(starts), 'starts', ndim=1, shape=(n,), min_value=0.0, max_value=1.0, value_sum=1.0)
        self.starts = _cast_as_parameter(torch.log(starts))
    if ends is None:
        self.ends = torch.empty(n, dtype=self.dtype, device=self.device) - inf
    else:
        ends = _check_parameter(_cast_as_tensor(ends), 'ends', ndim=1, shape=(n,), min_value=0.0, max_value=1.0)
        self.ends = _cast_as_parameter(torch.log(ends))
    _edge_idx_starts = torch.empty(self.n_edges, dtype=torch.int64, device=self.device)
    _edge_idx_ends = torch.empty(self.n_edges, dtype=torch.int64, device=self.device)
    _edge_log_probs = torch.empty(self.n_edges, dtype=self.dtype, device=self.device)
    idx = 0
    for edge in edges:
        if not hasattr(edge, '__len__') or len(edge) != 3:
            raise ValueError('Each edge must have three elements.')
        ni, nj, probability = edge
        if not isinstance(ni, Distribution):
            raise ValueError('First element must be a distribution.')
        if not isinstance(nj, Distribution):
            raise ValueError('Second element must be a distribution.')
        if not isinstance(probability, float):
            raise ValueError('Third element must be a float.')
        if probability < 0 or probability > 1:
            raise ValueError('Third element must be between 0 and 1.')
        if ni is self.start:
            if self.starts is None:
                self.starts = torch.zeros(n, dtype=self.dtype, device=self.device) - inf
            j = self.distributions.index(nj)
            self.starts[j] = math.log(probability)
        elif nj is self.end:
            i = self.distributions.index(ni)
            self.ends[i] = math.log(probability)
        else:
            i = self.distributions.index(ni)
            j = self.distributions.index(nj)
            _edge_idx_starts[idx] = i
            _edge_idx_ends[idx] = j
            _edge_log_probs[idx] = math.log(probability)
            idx += 1
    self._edge_idx_starts = _cast_as_parameter(_edge_idx_starts[:idx])
    self._edge_idx_ends = _cast_as_parameter(_edge_idx_ends[:idx])
    self._edge_log_probs = _cast_as_parameter(_edge_log_probs[:idx])
    self.n_edges = idx
    self.edges = self._edge_log_probs
    if idx == 0:
        raise ValueError('Must pass in edges to a sparse model, cannot ' + 'be uniformly initialized or it would be a dense model.')
    self._edge_keymap = {}
    for i in range(idx):
        start = self._edge_idx_starts[i].item()
        end = self._edge_idx_ends[i].item()
        self._edge_keymap[start, end] = i
    if self.starts is None:
        self.starts = torch.log(torch.ones(n, dtype=self.dtype, device=self.device) / n)
    self.starts = _cast_as_parameter(self.starts)
    self.ends = _cast_as_parameter(self.ends)


class SparseHMM(_BaseHMM):
    """A hidden Markov model with a sparse transition matrix.

	A hidden Markov model is an extension of a mixture model to sequences by
	including a transition matrix between the elements of the mixture. Each of
	the algorithms for a hidden Markov model are essentially just a revision
	of those algorithms to incorporate this transition matrix.

	This object is a wrapper for a hidden Markov model with a sparse transition
	matrix.

	Separately, there are two ways to instantiate the hidden Markov model. The
	first is by passing in a set of distributions, a dense transition matrix, 
	and optionally start/end probabilities. The second is to initialize the
	object without these and then to add edges using the `add_edge` method
	and to add distributions using the `add_distributions` method. Importantly, 
	the way that you choose to initialize the hidden Markov model is independent of the
	implementation that you end up choosing. If you pass in a dense transition
	matrix, this will be converted to a sparse matrix with all the zeros
	dropped if you choose `kind='sparse'`.


	Parameters
	----------
	distributions: tuple or list
		A set of distribution objects. These objects do not need to be
		initialized, i.e., can be "Normal()". 

	edges: numpy.ndarray, torch.Tensor, or None. shape=(k,k)
		A dense transition matrix of probabilities for how each node or
		distribution passed in connects to each other one. This can contain
		many zeroes, and when paired with `kind='sparse'`, will drop those
		elements from the matrix.

	starts: list, numpy.ndarray, torch.Tensor, or None. shape=(k,)
		The probability of starting at each node. If not provided, assumes
		these probabilities are uniform.

	ends: list, numpy.ndarray, torch.Tensor, or None. shape=(k,)
		The probability of ending at each node. If not provided, assumes
		these probabilities are uniform.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, distributions=None, edges=None, starts=None, ends=None, init='random', max_iter=1000, tol=0.1, sample_length=None, return_sample_paths=False, inertia=0.0, frozen=False, check_data=True, random_state=None, verbose=False):
        super().__init__(distributions=distributions, starts=starts, ends=ends, init=init, max_iter=max_iter, tol=tol, sample_length=sample_length, return_sample_paths=return_sample_paths, inertia=inertia, frozen=frozen, check_data=check_data, random_state=random_state, verbose=verbose)
        self.name = 'SparseHMM'
        if edges is not None:
            unpack_edges(self, edges, starts, ends)
            self.n_edges = len(edges)
        self._initialized = False
        if self.distributions is not None:
            if self.ends is not None:
                if self.starts is not None:
                    if all(d._initialized for d in self.distributions):
                        self._initialized = True
                        self.distributions = torch.nn.ModuleList(self.distributions)
        self._reset_cache()

    def _initialize(self, X=None, sample_weight=None):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d), optional
			The data to use to initialize the model. Default is None.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, len) or a vector of shape (-1,). If None, defaults to ones.
			Default is None.
		"""
        n = self.n_distributions
        if not hasattr(self, '_edge_log_probs'):
            unpack_edges(self, self.edges, self.starts, self.ends)
            self.n_edges = len(self.edges)
        self.distributions = torch.nn.ModuleList(self.distributions)
        super()._initialize(X, sample_weight=sample_weight)

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized == False:
            return
        for node in self.distributions:
            node._reset_cache()
        self.register_buffer('_xw_sum', torch.zeros(self.n_edges, dtype=self.dtype, device=self.device))
        self.register_buffer('_xw_starts_sum', torch.zeros(self.n_distributions, dtype=self.dtype, device=self.device))
        self.register_buffer('_xw_ends_sum', torch.zeros(self.n_distributions, dtype=self.dtype, device=self.device))

    def sample(self, n):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution. Because a HMM describes variable length
		sequences, a list will be returned where each element is one of
		the generated sequences.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: list of torch.tensor, shape=(n,)
			A list of randomly generated samples, where each sample of
			size (length, self.d).
		"""
        if self.sample_length is None and self.ends is None:
            raise ValueError('Must specify a length or have explicit ' + 'end probabilities.')
        if self.ends is None:
            ends = torch.zeros(self.n_distributions, dtype=self._edge_log_probs.dtype, device=self._edge_log_probs.device) + float('-inf')
        else:
            ends = self.ends
        distributions, emissions = [], []
        edge_ends, edge_probs = [], []
        for idx in range(self.n_distributions):
            idxs = self._edge_idx_starts == idx
            _ends = numpy.concatenate([self._edge_idx_ends[idxs].numpy(), [self.n_distributions]])
            _probs = numpy.concatenate([torch.exp(self._edge_log_probs[idxs]).numpy(), [numpy.exp(ends[idx])]])
            edge_ends.append(_ends)
            edge_probs.append(_probs)
        starts = torch.exp(self.starts).numpy()
        for _ in range(n):
            node_i = self.random_state.choice(self.n_distributions, p=starts)
            emission_i = self.distributions[node_i].sample(n=1)
            distributions_, emissions_ = [node_i], [emission_i]
            for i in range(1, self.sample_length or int(100000000.0)):
                node_i = self.random_state.choice(edge_ends[node_i], p=edge_probs[node_i])
                if node_i == self.n_distributions:
                    break
                emission_i = self.distributions[node_i].sample(n=1)
                distributions_.append(node_i)
                emissions_.append(emission_i)
            distributions.append(distributions_)
            emissions.append(torch.vstack(emissions_))
        if self.return_sample_paths == True:
            return emissions, distributions
        return emissions

    def viterbi(self, X=None, emissions=None, priors=None):
        """Run the Viterbi algorithm on some data.

		Runs the Viterbi algortihm on a batch of sequences. The Viterbi 
		algorithm is a dynamic programming algorithm that begins at the start
		state and calculates the single best path through the model involving
		alignments of symbol i to node j. This is in contrast to the forward
		function, which involves calculating the sum of all paths, not just
		the single best path. Because we have to keep track of the best path,
		the Viterbi algorithm is slightly more conceptually challenging and
		involves keeping track of a traceback matrix.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			A set of examples to evaluate. Does not need to be passed in if
			emissions are. 

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_dists)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		path: torch.Tensor, shape=(-1, -1)
			The state assignment for each observation in each sequence.
		"""
        emissions = _check_inputs(self, X, emissions, priors)
        n, l = emissions.shape[:2]
        v = torch.full((l, n, self.n_distributions), -inf, dtype=emissions.dtype, device=self.device)
        v[0] = self.starts + emissions[:, 0]
        traceback = torch.zeros_like(v, dtype=torch.int32)
        traceback[0] = torch.arange(v.shape[-1])
        idxs = [torch.where(self._edge_idx_ends == i)[0] for i in range(self.n_distributions)]
        for i in range(1, l):
            p = v[i - 1, :, self._edge_idx_starts]
            p += self._edge_log_probs.expand(n, -1)
            p += emissions[:, i, self._edge_idx_ends]
            for j, idx in enumerate(idxs):
                v[i, :, j], _idx = torch.max(p[:, idx], dim=-1)
                traceback[i, :, j] = self._edge_idx_starts[idx][_idx]
        ends = self.ends + v[-1]
        best_end_logps, best_end_idxs = torch.max(ends, dim=-1)
        paths = [best_end_idxs]
        for i in range(1, l):
            paths.append(traceback[l - i, torch.arange(n), paths[-1]])
        paths = torch.flip(torch.stack(paths).T, dims=(-1,))
        return paths

    def forward(self, X=None, emissions=None, priors=None):
        """Run the forward algorithm on some data.

		Runs the forward algorithm on a batch of sequences. This is not to be
		confused with a "forward pass" when talking about neural networks. The
		forward algorithm is a dynamic programming algorithm that begins at the
		start state and returns the probability, over all paths through the
		model, that result in the alignment of symbol i to node j.

		
		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			A set of examples to evaluate. Does not need to be passed in if
			emissions are. 

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		f: torch.Tensor, shape=(-1, -1, self.n_distributions)
			The log probabilities calculated by the forward algorithm.
		"""
        emissions = _check_inputs(self, X, emissions, priors)
        n, l, _ = emissions.shape
        f = torch.full((l, n, self.n_distributions), -inf, dtype=torch.float32, device=self.device)
        f[0] = self.starts + emissions[:, 0]
        for i in range(1, l):
            p = f[i - 1, :, self._edge_idx_starts]
            p += self._edge_log_probs.expand(n, -1)
            alpha = torch.max(p, dim=1, keepdims=True).values
            p = torch.exp(p - alpha)
            z = torch.zeros_like(f[i])
            z.scatter_add_(1, self._edge_idx_ends.expand(n, -1), p)
            f[i] = alpha + torch.log(z) + emissions[:, i]
        f = f.permute(1, 0, 2)
        return f

    def backward(self, X=None, emissions=None, priors=None):
        """Run the backward algorithm on some data.

		Runs the backward algorithm on a batch of sequences. This is not to be
		confused with a "backward pass" when talking about neural networks. The
		backward algorithm is a dynamic programming algorithm that begins at end
		of the sequence and returns the probability, over all paths through the
		model, that result in the alignment of symbol i to node j, working
		backwards.

		
		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			A set of examples to evaluate. Does not need to be passed in if
			emissions are. 

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		b: torch.Tensor, shape=(-1, length, self.n_distributions)
			The log probabilities calculated by the backward algorithm.
		"""
        emissions = _check_inputs(self, X, emissions, priors)
        n, l, _ = emissions.shape
        b = torch.full((l, n, self.n_distributions), -inf, dtype=torch.float32, device=self.device)
        b[-1] = self.ends
        for i in range(l - 2, -1, -1):
            p = b[i + 1, :, self._edge_idx_ends]
            p += emissions[:, i + 1, self._edge_idx_ends]
            p += self._edge_log_probs.expand(n, -1)
            alpha = torch.max(p, dim=1, keepdims=True).values
            p = torch.exp(p - alpha)
            z = torch.zeros_like(b[i])
            z.scatter_add_(1, self._edge_idx_starts.expand(n, -1), p)
            b[i] = alpha + torch.log(z)
        b = b.permute(1, 0, 2)
        return b

    def forward_backward(self, X=None, emissions=None, priors=None):
        """Run the forward-backward algorithm on some data.

		Runs the forward-backward algorithm on a batch of sequences. This
		algorithm combines the best of the forward and the backward algorithm.
		It combines the probability of starting at the beginning of the sequence
		and working your way to each observation with the probability of
		starting at the end of the sequence and working your way backward to it.

		A number of statistics can be calculated using this information. These
		statistics are powerful inference tools but are also used during the
		Baum-Welch training process. 

		
		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			A set of examples to evaluate. Does not need to be passed in if
			emissions are. 

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		transitions: torch.Tensor, shape=(-1, n, n)
			The expected number of transitions across each edge that occur
			for each example. The returned transitions follow the structure
			of the transition matrix and so will be dense or sparse as
			appropriate.

		responsibility: torch.Tensor, shape=(-1, -1, n)
			The posterior probabilities of each observation belonging to each
			state given that one starts at the beginning of the sequence,
			aligns observations across all paths to get to the current
			observation, and then proceeds to align all remaining observations
			until the end of the sequence.

		starts: torch.Tensor, shape=(-1, n)
			The probabilities of starting at each node given the 
			forward-backward algorithm.

		ends: torch.Tensor, shape=(-1, n)
			The probabilities of ending at each node given the forward-backward
			algorithm.

		logp: torch.Tensor, shape=(-1,)
			The log probabilities of each sequence given the model.
		"""
        emissions = _check_inputs(self, X, emissions, priors)
        n, l, _ = emissions.shape
        f = self.forward(emissions=emissions)
        b = self.backward(emissions=emissions)
        logp = torch.logsumexp(f[:, -1] + self.ends, dim=1)
        t = f[:, :-1, self._edge_idx_starts] + b[:, 1:, self._edge_idx_ends]
        t += emissions[:, 1:, self._edge_idx_ends]
        t += self._edge_log_probs.expand(n, l - 1, -1)
        t = torch.exp(torch.logsumexp(t, dim=1).T - logp).T
        starts = self.starts + emissions[:, 0] + b[:, 0]
        starts = torch.exp(starts.T - torch.logsumexp(starts, dim=-1)).T
        ends = self.ends + f[:, -1]
        ends = torch.exp(ends.T - torch.logsumexp(ends, dim=-1)).T
        r = f + b
        r = r - torch.logsumexp(r, dim=2).reshape(n, -1, 1)
        return t, r, starts, ends, logp

    def summarize(self, X, sample_weight=None, emissions=None, priors=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.

		emissions: torch.Tensor, shape=(-1, -1, self.n_distributions)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.	

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.
		"""
        X, emissions, sample_weight = super().summarize(X, sample_weight=sample_weight, emissions=emissions, priors=priors)
        t, r, starts, ends, logps = self.forward_backward(emissions=emissions)
        X = X.reshape(-1, X.shape[-1])
        r = torch.exp(r) * sample_weight.unsqueeze(1)
        for i, node in enumerate(self.distributions):
            w = r[:, :, i].reshape(-1, 1)
            node.summarize(X, sample_weight=w)
        if self.frozen == False:
            self._xw_starts_sum += torch.sum(starts * sample_weight, dim=0)
            self._xw_ends_sum += torch.sum(ends * sample_weight, dim=0)
            self._xw_sum += torch.sum(t * sample_weight, dim=0)
        return logps

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        for node in self.distributions:
            node.from_summaries()
        if self.frozen:
            return
        node_out_count = torch.clone(self._xw_ends_sum)
        for start, count in zip(self._edge_idx_starts, self._xw_sum):
            node_out_count[start] += count
        ends = torch.log(self._xw_ends_sum / node_out_count)
        starts = torch.log(self._xw_starts_sum / self._xw_starts_sum.sum())
        _edge_log_probs = torch.empty_like(self._edge_log_probs)
        for i in range(self.n_edges):
            t = self._xw_sum[i]
            t_sum = node_out_count[self._edge_idx_starts[i]]
            _edge_log_probs[i] = torch.log(t / t_sum)
        _update_parameter(self.ends, ends, inertia=self.inertia)
        _update_parameter(self.starts, starts, inertia=self.inertia)
        _update_parameter(self._edge_log_probs, _edge_log_probs, inertia=self.inertia)
        self._reset_cache()


class MarkovChain(Distribution):
    """A Markov chain.

	A Markov chain is the simplest sequential model which factorizes the
	joint probability distribution P(X_{0} ... X_{t}) along a chain into the
	product of a marginal distribution P(X_{0}) P(X_{1} | X_{0}) ... with
	k conditional probability distributions for a k-th order Markov chain.

	Despite sometimes being thought of as an independent model, Markov chains
	are probability distributions over sequences just like hidden Markov
	models. Because a Markov chain has the same theoretical properties as a
	probability distribution, it can be used in any situation that a simpler 
	distribution could, such as an emission distribution for a HMM or a 
	component of a Bayes classifier.


	Parameters
	----------
	distributions: tuple or list or None
		A set of distribution objects. These objects do not need to be
		initialized, i.e., can be "Categorical()". 

	k: int or None
		The number of conditional distributions to include in the chain, also
		the number of steps back to model in the sequence. This must be passed
		in if the distributions are not passed in.

	n_categories: list, tuple, or None
		A list or tuple containing the number of categories that each feature
		has. 

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, distributions=None, k=None, n_categories=None, inertia=0.0, frozen=False, check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'MarkovChain'
        self.distributions = _check_parameter(distributions, 'distributions', dtypes=(list, tuple))
        self.k = _check_parameter(_cast_as_tensor(k, dtype=torch.int32), 'k', ndim=0)
        self.n_categories = _check_parameter(n_categories, 'n_categories', dtypes=(list, tuple))
        if distributions is None and k is None:
            raise ValueError("Must provide one of 'distributions', or 'k'.")
        if distributions is not None:
            self.k = len(distributions) - 1
        self.d = None
        self._initialized = distributions is not None and distributions[0]._initialized
        self._reset_cache()

    def _initialize(self, d, n_categories):
        """Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.

		n_categories: int
			The maximum number of categories to model. This single number is
			used as the maximum across all features and all timesteps.
		"""
        if self.distributions is None:
            self.distributions = [Categorical()]
            self.distributions[0]._initialize(d, max(n_categories))
            for i in range(self.k):
                distribution = ConditionalCategorical()
                distribution._initialize(d, [([n_categories[j]] * (i + 2)) for j in range(d)])
                self.distributions.append(distribution)
        self.n_categories = n_categories
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        """Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""
        if self._initialized:
            for distribution in self.distributions:
                distribution._reset_cache()

    def sample(self, n):
        """Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution. For a mixture model, this involves first
		sampling the component using the prior probabilities, and then sampling
		from the chosen distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""
        X = [self.distributions[0].sample(n)]
        for distribution in self.distributions[1:]:
            X_ = torch.stack(X).permute(1, 0, 2)
            samples = distribution.sample(n, X_[:, -self.k - 1:])
            X.append(samples)
        return torch.stack(X).permute(1, 0, 2)

    def log_probability(self, X):
        """Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 3D
		format.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			A set of examples to evaluate.

		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=3, check_parameter=self.check_data)
        self.d = X.shape[1]
        logps = self.distributions[0].log_probability(X[:, 0])
        for i, distribution in enumerate(self.distributions[1:-1]):
            logps += distribution.log_probability(X[:, :i + 2])
        for i in range(X.shape[1] - self.k):
            j = i + self.k + 1
            logps += self.distributions[-1].log_probability(X[:, i:j])
        return logps

    def fit(self, X, sample_weight=None):
        """Fit the model to optionally weighted examples.

		This method will fit the provided distributions given the data and
		their weights. If only `k` has been provided, the relevant set of
		distributions will be initialized.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			A set of examples to evaluate. 

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		self
		"""
        self.summarize(X, sample_weight=sample_weight)
        self.from_summaries()
        return self

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache for each distribution
		in the network. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""
        if self.frozen:
            return
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=3, check_parameter=self.check_data)
        sample_weight = _check_parameter(_cast_as_tensor(sample_weight), 'sample_weight', min_value=0, ndim=(1, 2), check_parameter=self.check_data)
        if not self._initialized:
            if self.n_categories is not None:
                n_keys = self.n_categories
            elif isinstance(X, torch.masked.MaskedTensor):
                n_keys = (torch.max(torch.max(X._masked_data, dim=0)[0], dim=0)[0] + 1).type(torch.int32)
            else:
                n_keys = (torch.max(torch.max(X, dim=0)[0], dim=0)[0] + 1).type(torch.int32)
            self._initialize(len(X[0][0]), n_keys)
        if sample_weight is None:
            sample_weight = torch.ones_like(X[:, 0])
        elif len(sample_weight.shape) == 1:
            sample_weight = sample_weight.reshape(-1, 1).expand(-1, X.shape[2])
        elif sample_weight.shape[1] == 1:
            sample_weight = sample_weight.expand(-1, X.shape[2])
        _check_parameter(_cast_as_tensor(sample_weight), 'sample_weight', min_value=0, ndim=2, shape=(X.shape[0], X.shape[2]), check_parameter=self.check_data)
        self.distributions[0].summarize(X[:, 0], sample_weight=sample_weight)
        for i, distribution in enumerate(self.distributions[1:-1]):
            distribution.summarize(X[:, :i + 2], sample_weight=sample_weight)
        distribution = self.distributions[-1]
        for i in range(X.shape[1] - self.k):
            j = i + self.k + 1
            distribution.summarize(X[:, i:j], sample_weight=sample_weight)

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
        if self.frozen:
            return
        for distribution in self.distributions:
            distribution.from_summaries()


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (DiracDelta,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {})),
    (Exponential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {})),
    (Gamma,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {})),
    (Poisson,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {})),
    (StudentT,
     lambda: ([], {'dofs': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Uniform,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {})),
]

