import numpy as np
import copy
from scipy import signal

class Conv2d:
    def __init__(self, kw, kh, input_channel, output_channel, pw=0, ph=0, sw=1, sh=1):
        self.padding_width = pw
        self.padding_height = ph
        self.stride_width = sw
        self.stride_height = sh
        
        # Random kernel initialization
        self.weights = np.random.rand(input_channel, output_channel, kw, kh)
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
        print 'err', err.shape
        print 'w', self.weights.shape
        print 'x', self.input.shape
        self.d_weights = np.zeros((self.weights.shape))
        '''
        for i in xrange(err.shape[0]):
            for out_ch in xrange(self.weights.shape[1]):
                for in_ch in xrange(self.weights.shape[0]):
                    self.d_weights[in_ch][out_ch] += self._rot180(signal.convolve2d(self.input[i],
                                                                 self.rot180(gdY),
                                                                 mode='valid'))
                                                                 '''
        return err, None
    
    def _update(self, step):
        pass