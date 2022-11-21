"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features), dtype=dtype)
        self.require_bias = bias
        if bias:
          self.bias = Parameter(ops.transpose(init.kaiming_uniform(out_features, 1)), dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.require_bias:
          return ops.matmul(X, self.weight) + ops.broadcast_to(self.bias, (X.shape[0], self.out_features)) 
        else:
          return ops.matmul(X, self.weight)
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        dimension = 1
        for i in X.shape[1:]:
          dimension = dimension * i
        return ops.reshape(X, (batch_size, dimension))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
          out = module(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        m = logits.shape[0]  # num of samples
        k = logits.shape[1]  # dimensionality of output
        return ops.summation(ops.logsumexp(logits, axes = (1, )) - ops.summation(logits * init.one_hot(k, y), axes = (1, ))) / m
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(np.ones(dim), dtype=dtype)
        self.bias = Parameter(np.zeros(dim), dtype=dtype)
        self.running_mean = Tensor(np.zeros(dim), dtype=dtype)
        self.running_var = Tensor(np.ones(dim), dtype=dtype)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = ops.summation(x, axes = (0, )) / x.shape[0]
        centerlized_x = x - ops.broadcast_to(mean, x.shape)
        var = ops.summation(ops.power_scalar(centerlized_x, 2), axes = (0, )) / x.shape[0]
        if self.training:
          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
          self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
          return ops.divide(ops.broadcast_to(self.weight, x.shape) * (centerlized_x), \
          ops.broadcast_to(ops.power_scalar(var + self.eps, 0.5), x.shape)) + ops.broadcast_to(self.bias, x.shape)
        else:
          return ops.divide(ops.broadcast_to(self.weight, x.shape) * (x - ops.broadcast_to(self.running_mean, x.shape)), \
          ops.broadcast_to(ops.power_scalar(self.running_var + self.eps, 0.5), x.shape)) + ops.broadcast_to(self.bias, x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(np.ones(dim), dtype=dtype)
        self.bias = Parameter(np.zeros(dim), dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # lesson : Explicitly write all the broadcast steps. If we omit the broadcast in element-wise division, 
        # the backward method will crash. Same situation will happen if we omit the broadcast in computing 
        # centerlized_x.
        mean = ops.reshape(ops.summation(x, axes = (1, )) / x.shape[1], (x.shape[0], 1))
        centerlized_x = x - ops.broadcast_to(mean, x.shape)
        var = ops.reshape(ops.summation(ops.power_scalar(centerlized_x, 2), axes = (1, )) / x.shape[1], (x.shape[0], 1))
        return ops.divide(ops.broadcast_to(self.weight, x.shape) * (centerlized_x), \
        ops.broadcast_to(ops.power_scalar(var + self.eps, 0.5), x.shape)) + ops.broadcast_to(self.bias, x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          return init.randb(*x.shape, p = 1 - self.p) * x / (1 - self.p)
        else:
          return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION



