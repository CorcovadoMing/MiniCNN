import numpy as np
import copy

class Linear:
    def __init__(self, i, c):
        self.pd_weight = 0
        # Random kernel initialization
        self.weights = np.random.normal(0, 1e-1, (i, c))
        self.bias = np.random.normal(0, 1e-4, (c))

    def _forward(self, x):
        # Cache the input for backward use
        self.input = copy.deepcopy(x)
        return np.dot(x, self.weights) + self.bias

    def _backward(self, err, res):
        delta = np.multiply(err, res)
        self.d_weights = self.input.dot(delta) / err.shape[0]
        self.d_bias = delta.sum(axis=0) / err.shape[0]
        return np.dot(delta, self.weights.T), None

    def _update(self, step, mom, decay):
        var = (self.pd_weight * mom) - (step * self.d_weights) - (step * decay * self.weights)
        self.pd_weight = var
        self.weights += var
        self.bias -= step * self.d_bias
