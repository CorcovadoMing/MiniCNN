import numpy as np
import copy
from scipy import signal
from convop import conv2d_op, deconv2d_op
from gemm import im2col

class Conv2d:
    def __init__(self, kw, kh, input_channel, output_channel, pw=0, ph=0, sw=1, sh=1):
        # Not implement so far
        self.padding_width = pw
        self.padding_height = ph
        self.stride_width = sw
        self.stride_height = sh

        # Momentum
        self.pd_weight = 0
        # Random kernel initialization
        self.weights = np.random.normal(0, 1e-1, (input_channel, output_channel, kw, kh))
        self.bias = np.random.normal(0, 1e-4, (1, output_channel, 1, 1))

    def _forward(self, x):
        self.input_shape = x.shape # Cache the input for backward use
        out_map_size = np.array(x.shape[2:]) - np.array(self.weights.shape[2:]) + 1
        out_map_size = list(self.weights.shape[1:2]) + list(x.shape[:1]) + list(out_map_size)
        x_ = im2col(x, self.weights.shape[2], self.weights.shape[3], 0, 1)
        self.input_col = x_ # Cache the input for backward use
        w_ = self.weights.transpose(1,2,3,0).reshape(self.weights.shape[1], -1)
        output = w_.dot(x_).reshape(out_map_size).transpose(1,0,2,3)
        return output + self.bias

    def _backward(self, err, res):
        delta = np.multiply(err, res)
        delta_vector = delta.transpose(1,0,2,3).reshape(err.shape[1], -1)
        s = self.weights.transpose(1,2,3,0).shape 
        self.d_weights = (delta_vector.dot(self.input_col.T) / err.shape[0]).reshape(s).transpose(3,0,1,2)

        self.d_bias = (np.sum(delta, axis=(0, 2, 3)) / err.shape[0])[None, :, None, None]
        
        delta_col = im2col(delta, self.weights.shape[2], self.weights.shape[3], 2, 1)
        w_ = self.weights[:,:,::-1,::-1].transpose(0,2,3,1).reshape(self.weights.shape[0], -1)
        return w_.dot(delta_col).reshape(self.input_shape), None

    def _update(self, step, mom, decay):
        var = (self.pd_weight * mom) - (step * self.d_weights) - (step * decay * self.weights)
        self.pd_weight = var
        self.weights += var
        self.bias -= step * self.d_bias
