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
        self.bias = np.random.rand()

    def _forward(self, x):
        # Cache the input for backward use
        self.input = copy.deepcopy(x)

        # Although it 10x times slower, but it has better acc result, keep the original code here
        '''
        out_map_size = np.array(x.shape[2:]) - np.array(self.weights.shape[2:]) + 1
        out_map_size = list(x.shape[:1]) + list(self.weights.shape[1:2]) + list(out_map_size)
        output = np.zeros(out_map_size)
        for i in xrange(output.shape[0]):
            for out_ch in xrange(self.weights.shape[1]):
                for in_ch in xrange(self.weights.shape[0]):
                    output[i][out_ch] += signal.convolve2d(x[i][in_ch], self.weights[in_ch][out_ch], 'valid')
        return output
        '''
        
        out_map_size = np.array(x.shape[2:]) - np.array(self.weights.shape[2:]) + 1
        out_map_size = list(x.shape[:1]) + list(self.weights.shape[1:2]) + list(out_map_size)
        output = np.empty(out_map_size)
        conv2d_op(x, self.weights, output)
        return output
        
                
    def _rot180(self, kernel):
        return np.flipud(np.fliplr(kernel))

    def _backward(self, err, res):
        # Although it 10x times slower, but it has better acc result, keep the original code here
        '''
        self.d_weights = np.zeros_like(self.weights)
        output = np.zeros_like(self.input)
        for i in xrange(err.shape[0]):
            for in_ch in xrange(self.weights.shape[0]):
                for out_ch in xrange(self.weights.shape[1]):
                    self.d_weights[in_ch][out_ch] += signal.convolve2d(self._rot180(self.input[i][in_ch]), err[i][out_ch], mode='valid')
                    output[i] += signal.convolve2d(err[i][out_ch], self._rot180(self.weights[in_ch][out_ch]))
        self.d_weights /= err.shape[0]
        return output, None
        '''
        
        self.d_weights = np.zeros_like(self.weights)
        output = np.zeros_like(self.input)
        deconv2d_op(self._rot180(self.input), err, self._rot180(self.weights), output, self.d_weights)
        return output, None
        
    
    def _update(self, step, mom, decay):
        var = (self.pd_weight * mom) - (step * self.d_weights)
        self.pd_weight = var
        self.weights += var
        self.weights -= (decay * self.weights)
        #self.weights -= step * self.d_weights