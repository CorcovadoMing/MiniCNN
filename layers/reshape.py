import numpy as np

class Reshape:
    def __init__(self, shape):
        self.output_shape = shape
    
    def _forward(self, x):
        self.input_shape = x.shape
        return np.reshape(x, self.output_shape) 