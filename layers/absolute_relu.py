import numpy as np

class AbsRelu:
    def __init__(self):
        pass

    def _forward(self, x):
        self.mask = (x >= 0).astype(float) + (x < 0).astype(float) * -1
        return x * self.mask

    def _backward(self, err, res):
        return err, self.mask

    def _update(self, *args):
        pass
