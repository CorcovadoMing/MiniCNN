import numpy as np
import copy
from scipy import signal

class Conv2d:
    def __init__(self, kw, kh, input_channel, output_channel, pw=0, ph=0, sw=1, sh=1):
        self.padding_width = pw
        self.padding_height = ph
        self.stride_width = sw
        self.stride_height = sh
        self.pd_weight = 0
        # Random kernel initialization
        self.weights = np.random.normal(0, 0.1, (input_channel, output_channel, kw, kh))
        self.bias = np.random.rand()

    def _forward(self, x):
        # Cache the input for backward use
        self.input = copy.deepcopy(x)
        output = []
        for i in x:
            imm_result = []
            for out_ch in xrange(self.weights.shape[1]):
                out_map = np.zeros(np.array(x.shape[2:]) - np.array(self.weights.shape[2:]) + 1)
                for in_ch in xrange(self.weights.shape[0]):
                    out_map += signal.convolve2d(i[in_ch], self.weights[in_ch][out_ch], 'valid')
                imm_result.append(out_map)
            output.append(imm_result)
        return np.array(output)
                
    def _rot180(self, kernel):
        return np.flipud(np.fliplr(kernel))

    def _backward(self, err, res):
        self.d_weights = np.zeros((self.weights.shape))
        output = np.zeros_like(self.input)
        
        for i in xrange(err.shape[0]):
            for in_ch in xrange(self.weights.shape[0]):
                for out_ch in xrange(self.weights.shape[1]):
                    self.d_weights[in_ch][out_ch] += signal.convolve2d(self._rot180(self.input[i][in_ch]),
                                                                    err[i][out_ch],mode='valid')
                    output[i] += signal.convolve2d(err[i][out_ch],
                                                self._rot180(self.weights[in_ch][out_ch]))
        self.d_weights /= err.shape[0]
        return output, None
    
    def _update(self, step, mom):
        var = (self.pd_weight * mom) - (step * self.d_weights) + (1e-4 * self.weights)
        self.pd_weight = var
        self.weights += var
        #self.weights -= step * self.d_weights