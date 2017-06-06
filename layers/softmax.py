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
        output = []
        for i in x:
            imm_result = np.dot(np.transpose(self.weights), i) + self.bias
            calibrate = np.max(imm_result)
            imm_result -= calibrate
            output.append(np.exp(imm_result) / np.sum(np.exp(imm_result)))
        return output
