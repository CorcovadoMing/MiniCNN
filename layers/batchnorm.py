import numpy as np

class BatchNorm:
    def __init__(self):
        self.gamma = 1
        self.beta = 0

    def _forward(self, x):
        x_mean = x.mean(axis=0) # size: single image
        x_var = x.var(axis=0) # size: single image
        eps = 1e-8
        self.inv_var = (x_var + eps)**0.5 # size: single image
        self.x_hat = (x - x_mean) / self.inv_var # size: batched image
        return self.gamma * self.x_hat + self.beta
        
    def _backward(self, err, res):
        N = err.shape[0]
        self.d_gamma = np.multiply(err, self.x_hat).sum()
        self.d_beta = err.sum()
        dx_hat = err * self.gamma 
        output = (1. / N) * self.inv_var * (N * dx_hat - dx_hat.sum(axis=0) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=0))        
        return output, None

    def _update(self, step, mom, decay):
        self.gamma -= step * self.d_gamma
        self.beta -= step * self.d_beta
