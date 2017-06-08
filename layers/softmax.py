import numpy as np
import copy

class Softmax:
    def __init__(self, i, c):
        self.input_size = i
        self.classes = c
        self.pd_weight = 0
        # Random kernel initialization
        self.weights = np.random.normal(0, 0.1, (i, c))
        self.bias = np.random.rand()

    def softmax_derivative(self, probs):
        return np.diag(probs) - np.dot(np.expand_dims(probs, 1), np.expand_dims(probs, 0))
    
    def _forward(self, x):
        # Cache the input for backward use
        self.input = copy.deepcopy(x)
        output = []
        for i in x:
            imm_result = np.dot(self.weights.T, i) + self.bias
            calibrate = np.max(imm_result)
            imm_result -= calibrate
            output.append(np.exp(imm_result) / np.sum(np.exp(imm_result)))
        return np.array(output)

    def _backward(self, err, res):
        self.d_weights = np.dot(err.T, self.input).T / err.shape[0]
        self.d_bias = err.sum() / err.shape[0]
        return np.dot(err, self.weights.T), None
    
    def _update(self, step, mom, decay):
        var = (self.pd_weight * mom) - (step * self.d_weights)
        self.pd_weight = var
        self.weights += var
        self.weights -= (decay * self.weights)
        #self.weights -= step * self.d_weights
        self.bias -= step * self.d_bias

