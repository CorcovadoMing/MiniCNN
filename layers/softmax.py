import numpy as np

class Softmax:
    def __init__(self, i, c):
        self.input_size = i
        self.classes = c
        
        # Random kernel initialization
        self.weights = np.random.rand(i, c)
        self.bias = np.random.rand()
    
    def _forward(self, x):
        # Cache the input for backward use
        self.input = x
        output = np.dot(np.transpose(self.weights), x) + self.bias
        calibrate = np.max(output)
        output = output - calibrate
        return np.exp(output) / np.sum(np.exp(output))
