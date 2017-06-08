import numpy as np
import copy

class Linear:
    def __init__(self, i, c):
        self.input_size = i
        self.classes = c
        self.v = 0
        
        # Random kernel initialization
        self.weights = np.random.normal(0, 1, (i, c))
        self.bias = np.random.rand()
    
    def _forward(self, x):
        # Cache the input for backward use
        self.input = copy.deepcopy(x)
        output = []
        for i in x:
            imm_result = np.dot(self.weights.T, i) + self.bias
            output.append(imm_result)
        return np.array(output)

    def _backward(self, err, res):
        self.d_weights = np.dot(np.multiply(err, res).T, self.input).T
        self.d_bias = np.multiply(err, res).sum()
        return np.dot(err, self.weights.T), None
    
    def _update(self, step, mom):
        #self.d_weights += (0.01 * self.weights)
        self.v =  mom * self.v - self.d_weights * step
        self.weights += self.v
        self.bias -= step * self.d_bias
