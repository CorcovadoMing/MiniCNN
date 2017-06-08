import numpy as np

class Relu:
    def __init__(self):
        pass
    
    def _forward(self, x):
        # Need to unwrap this after operations
        x = x * [x > 0][0]
        # Fix the -0.0 issue
        x += 0.
        mask = np.ones_like(x)
        self.mask = mask * [x > 0][0]
        return x
    
    def _backward(self, err, res):
        return err, self.mask
    
    def _update(self, *args):
        pass