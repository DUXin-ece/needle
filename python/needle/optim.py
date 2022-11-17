"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for key, param in enumerate(self.params):
          if key in self.u:
            self.u[key].data = self.momentum * self.u[key].data + (1 - self.momentum) \
            * (param.grad.data + self.weight_decay * param.data) 
          else:
            self.u[key] = (1 - self.momentum) * (param.grad.data + self.weight_decay * param.data)
          param.data = param.data - self.lr * self.u[key].data  
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for key, param in enumerate(self.params):
          penalized_grad = (param.grad + self.weight_decay * param).data 
          if key not in self.m:
            self.m[key] = ndl.init.zeros(*param.grad.shape)
            self.v[key] = ndl.init.zeros(*param.grad.shape)
          self.m[key].data = self.beta1 * self.m[key].data + (1 - self.beta1) * penalized_grad
          self.v[key].data = self.beta2 * self.v[key].data + (1 - self.beta2) * penalized_grad ** 2
          corrected_m = self.m[key].data / (1 - self.beta1 ** self.t)
          corrected_v = self.v[key].data / (1 - self.beta2 ** self.t)
          param.data = param.data - self.lr * corrected_m / (corrected_v ** 0.5 + self.eps)
        ### END YOUR SOLUTION
