import numpy as np

class Reshape:
    def __init__(self, shape):
        if type(shape) == int:
            self.shape = [shape]
        else:
            self.shape = list(shape)
    
    def _forward(self, x):
        self.input_shape = x.shape
        self.output_shape = [x.shape[0]] + self.shape
        return np.reshape(x, self.output_shape)
    
    def _backward(self, err, res):
        return err.reshape(self.input_shape), None
    
    def _update(self, step):
        pass