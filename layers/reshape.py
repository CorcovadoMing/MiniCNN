import numpy as np

class Reshape:
    def __init__(self, shape):
        self.shape = list(shape)
    
    def _forward(self, x):
        self.input_shape = x.shape
        self.output_shape = [x.shape[0]] + self.shape
        return np.reshape(x, self.output_shape) 