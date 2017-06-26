import numpy as np

class LeakyRelu:
    def __init__(self):
        pass
    
    def _forward(self, x):
        pos = (x > 0).astype(float)
        neg = (x < 0).astype(float) * 0.01
        self.mask = (pos + neg)
        return x * self.mask
    
    def _backward(self, err, res):
        return err, self.mask
    
    def _update(self, *args):
        pass