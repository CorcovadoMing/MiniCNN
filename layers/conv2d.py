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
            for in_ch in xrange(self.weights.shape[0]):
                for out_ch in xrange(self.weights.shape[1]):
                    #imm_result.append(self._op_conv2d(i[in_ch], self.weights[in_ch][out_ch]))
                    imm_result.append(signal.convolve2d(i[in_ch], self.weights[in_ch][out_ch], 'valid'))
            output.append(imm_result)
        return np.array(output)
                
    
    def _backward(self):
        pass