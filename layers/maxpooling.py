import copy
import numpy as np

class Maxpooling:
    def __init__(self, kw, kh):
        self.kernel_width = kw
        self.kernel_height = kh
        self.mask = []

    def _forward(self, x):
        # Cache the input for backward use
        self.input = copy.deepcopy(x)
        
        B, N, h, w = x.shape
        
        tmp = x.reshape(B, N, h/self.kernel_width, self.kernel_width, w/self.kernel_height, self.kernel_height).swapaxes(3, 4).reshape(B, N, h/self.kernel_width, w/self.kernel_height, self.kernel_width * self.kernel_height)
        
        self.masks = np.zeros(x.shape)
        indexes = tmp.argmax(axis=-1)
        for b in xrange(indexes.shape[0]):
            for i in xrange(indexes.shape[1]):
                for x in xrange(indexes.shape[2]):
                    for y in xrange(indexes.shape[3]):
                        index = indexes[b][i][x][y]
                        self.masks[b][i][x * self.kernel_width + index / self.kernel_width ][y * self.kernel_height + index % self.kernel_height] = 1

        return tmp.max(axis=-1)
    
    def _backward(self, err, res):
        return err, self.masks
    
    def _update(self, step):
        pass
            

