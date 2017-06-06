import numpy as np
import copy

class Linear:
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
            imm_result = np.dot(np.transpose(self.weights), i) + self.bias
            output.append(imm_result)
        return np.array(output)
