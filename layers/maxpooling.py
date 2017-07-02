import copy
import numpy as np
from poolop import maxpooling, unpooling

class Maxpooling:
    def __init__(self, kw, kh):
        self.kernel_width = kw
        self.kernel_height = kh

    def _forward(self, x):
        B, N, h, w = x.shape
        output = np.empty((B, N, h/self.kernel_width, w/self.kernel_height))
        switch = np.empty(output.shape + (2,), dtype=np.int)
        maxpooling(x, output, switch, self.kernel_width, self.kernel_height, self.kernel_width, self.kernel_height)
        self.masks = switch
        self.input_shape = x.shape
        return output

    def _backward(self, err, res):
        output = np.empty(self.input_shape)
        unpooling(err, self.masks, output)
        return output, None

    def _update(self, *args):
        pass


