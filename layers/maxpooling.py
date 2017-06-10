import copy
import numpy as np
from poolop import pool_bc01, bprop_pool_bc01

class Maxpooling:
    def __init__(self, kw, kh):
        self.kernel_width = kw
        self.kernel_height = kh

    def _forward(self, x):
        B, N, h, w = x.shape
        output = np.empty((B, N, h/self.kernel_width, w/self.kernel_height))
        switch = np.empty(output.shape + (2,), dtype=np.int)
        pool_bc01(x, output, switch, self.kernel_width, self.kernel_height, 2, 2)
        self.masks = switch
        self.input_shape = x.shape
        return output
    
    def _backward(self, err, res):
        output = np.empty(self.input_shape)
        bprop_pool_bc01(err, self.masks, output)
        return output, None
    
    def _update(self, *args):
        pass
            

