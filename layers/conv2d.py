import numpy as np
import copy
from scipy import signal
from convop import conv2d_op, deconv2d_op

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
        self.weights = np.random.normal(0, 0.1, (input_channel, output_channel, kw, kh))
        self.bias = np.random.normal(0, 0.01, (1, output_channel, 1, 1))

    def _forward(self, x):
        # Cache the input for backward use
        self.input = copy.deepcopy(x)
        out_map_size = np.array(x.shape[2:]) - np.array(self.weights.shape[2:]) + 1
        out_map_size = list(x.shape[:1]) + list(self.weights.shape[1:2]) + list(out_map_size)
        output = np.empty(out_map_size)
        conv2d_op(x, self._rot180_b(self.weights), output)
        self._rot180_b(self.weights) # Rotate back
        return output + self.bias
                
    def _rot180(self, kernel):
        return np.flipud(np.fliplr(kernel))
    
    def _rot180_b(self, data):
        for i in xrange(data.shape[0]):
            for j in xrange(data.shape[1]):
                data[i, j, :, :] = self._rot180(data[i, j, :, :])
        return data

    def _backward(self, err, res):
        self.d_weights = np.zeros_like(self.weights)
        output = np.zeros_like(self.input)
        deconv2d_op(self.input, err, self._rot180_b(self.weights), output, self.d_weights)
        self._rot180_b(self.weights) # Rotate back
        self.d_weights = self.d_weights[:, :, ::-1, ::-1]
        self.d_bias = (np.sum(err, axis=(0, 2, 3)) / err.shape[0])[None, :, None, None]
        return output, None
    
    def _update(self, step, mom, decay):
        var = (self.pd_weight * mom) - (step * self.d_weights)
        self.pd_weight = var
        self.weights += var
        self.weights -= (decay * self.weights)
        #self.weights -= step * self.d_weights
        self.bias -= step * self.d_bias