import numpy as np

class Conv2d:
    def __init__(self, kw, kh, input_channel, output_channel, pw=0, ph=0, sw=1, sh=1):
        self.padding_width = pw
        self.padding_height = ph
        self.stride_width = sw
        self.stride_height = sh
        
        # Random kernel initialization
        self.weights = np.random.rand(input_channel, output_channel, kw, kh)
        self.bias = np.random.rand()
    
    def _op_conv2d(self, in_map, w):
        result_shape = np.array(in_map.shape) - (np.array(w.shape) - 1)
        result = np.zeros(result_shape)
        for i in xrange(result_shape[0]):
            for j in xrange(result_shape[1]):
                for kw in xrange(w.shape[0]):
                    for kh in xrange(w.shape[1]):
                        result[i][j] += in_map[i+kw][j+kh] * w[kw][kh]
        return result

    def _forward(self, x):
        # Cache the input for backward use
        self.input = x
        output = []
        for in_ch in xrange(self.weights.shape[0]):
            for out_ch in xrange(self.weights.shape[1]):
                output.append(self._op_conv2d(x[in_ch], self.weights[in_ch][out_ch]))
        return np.array(output)
                
    
    def _backward(self):
        pass