import numpy as np
from gemm import im2col
from math import sqrt

class Conv2d:
    def __init__(self, kw, kh, input_channel, output_channel, pw=0, ph=0, sw=1, sh=1):
        # Control flag
        self.is_first_layer = False
        self.no_bias = False
        # Momentum
        self.pd_weight = 0
        self.hd_weight = 0
        self.weights = np.random.rand(output_channel, input_channel * kw * kh) * sqrt(2.0/(input_channel * output_channel * kw * kh))
        self.bias = np.ones((1, output_channel, 1, 1)) * 0.01
        self.k = kw
        self.ic = input_channel
        self.oc = output_channel

    def set_first(self):
        self.is_first_layer = True
        return self.is_first_layer

    def set_no_bias(self):
        self.no_bias = True
        return self.no_bias

    def _forward(self, x):
        self.input = x
        self.input_shape = x.shape # Cache the input for backward use
        out_map_size = np.array(x.shape[2:]) - np.array([self.k, self.k]) + 1
        out_map_size = [self.weights.shape[0]] + [x.shape[0]] + list(out_map_size)
        x_ = im2col(x, self.k, self.k, 0, 1)
        self.input_col = x_ # Cache the input for backward use
        output = self.weights.dot(x_).reshape(out_map_size).transpose(1,0,2,3)
        if self.no_bias:
            return output
        else:
            return output + self.bias

    def _backward(self, err, res):
        delta = np.multiply(err, res)
        delta_vector = delta.reshape(delta.shape[1], -1)
        self.d_weights = delta_vector.dot(self.input_col.T)

        if not self.no_bias:
            self.d_bias = np.sum(delta, axis=(0, 2, 3))[None, :, None, None]

        if not self.is_first_layer:
            padding = self.k - 1
            delta_col = im2col(delta, self.k, self.k, padding, 1)
            w_ = self.weights.reshape(self.oc, self.ic, self.k, self.k)
            w_ = w_.swapaxes(0, 1)
            w_ = w_.reshape(w_.shape[0], -1)
            output = w_.dot(delta_col)
            output = output.reshape(self.oc, self.ic, -1).swapaxes(0, 1)
            return output.reshape(self.input_shape), None
        else:
            return None, None

    def _update(self, step, mom, decay):
        self.pd_weight = self.hd_weight
        self.hd_weight = (self.hd_weight * mom) - (step * self.d_weights) - (step * decay * self.weights)
        self.weights += -mom * self.pd_weight + (1 + mom) * self.hd_weight
        # var = (self.pd_weight * mom) - (step * self.d_weights) - (step * decay * self.weights)
        # self.pd_weight = var
        # self.weights += var
        if not self.no_bias:
            self.bias -= step * self.d_bias
