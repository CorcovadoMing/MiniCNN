import numpy as np
import copy

class Softmax:
    def __init__(self, i, c):
        self.input_size = i
        self.classes = c
        
        # Random kernel initialization
        self.weights = np.random.rand(i, c)
        self.bias = np.random.rand()
    
    def _forward(self, x):
        # Cache the input for backward use
        self.input = copy.deepcopy(x)
        output = []
        for i in x:
            imm_result = np.dot(self.weights.T, i) + self.bias
            calibrate = np.max(imm_result)
            imm_result -= calibrate
            output.append(np.exp(imm_result) / np.sum(np.exp(imm_result)))
        return np.array(output)[:, :, -1]

    def _backward(self, e):
        self.d_weights = self.input.T.dot(e)
        self.d_bias = e.sum()
        e = e.dot(self.weights.T)
        print e.shape
        print self.input.shape

