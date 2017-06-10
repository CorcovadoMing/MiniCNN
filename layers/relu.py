import numpy as np

class Relu:
    def __init__(self):
        pass
    
    def _forward(self, x):
        self.mask = (x > 0).astype(int)
        return x * self.mask + 0.
    
    def _backward(self, err, res):
        return err, self.mask
    
    def _update(self, *args):
        pass