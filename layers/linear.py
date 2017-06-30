import numpy as np
import copy
from math import sqrt

class Linear:
    def __init__(self, i, c):
        self.pd_weight = 0
        self.hd_weight = 0
        # Random kernel initialization
#         self.weights = np.random.normal(0, 1e-1, (i, c))
#         self.bias = np.random.normal(0, 1e-4, (c))
        self.weights = np.random.rand(i, c) * sqrt(2.0/(i*c))
        self.bias = np.ones((c)) * 0.01

    def _forward(self, x):
        # Cache the input for backward use
        self.input = copy.deepcopy(x)
        return np.dot(x, self.weights) + self.bias

    def _backward(self, err, res):
        delta = np.multiply(err, res)
        self.d_weights = self.input.T.dot(delta)
        self.d_bias = delta.sum(axis=0)
        return np.dot(delta, self.weights.T), None

    def _update(self, step, mom, decay):
        self.pd_weight = self.hd_weight
        self.hd_weight = (self.hd_weight * mom) - (step * self.d_weights) - (step * decay * self.weights)
        self.weights += -mom * self.pd_weight + (1 + mom) * self.hd_weight
        # var = (self.pd_weight * mom) - (step * self.d_weights) - (step * decay * self.weights)
        # self.pd_weight = var
        # self.weights += var
        self.bias -= step * self.d_bias
