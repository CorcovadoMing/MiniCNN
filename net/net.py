import numpy as np
import time

class Net:
    def __init__(self):
        self.layers = []
    
    def push(self, layer):
        self.layers.append(layer)

    def pop(self):
        self.layers.pop()

    def input(self, x, y, type):
        if type == 'train':
            self.train_x = np.array(x, dtype=np.float)
            self.train_y = np.array(y, dtype=np.float)
        elif type == 'test':
            self.test_x = np.array(x, dtype=np.float)
            self.test_y = np.array(y, dtype=np.float)
        else:
            raise TypeError

    def forward(self):
        imm_result = self.train_x
        for i in self.layers:
            now = time.time()
            imm_result = i._forward(imm_result)
            print str(i), time.time() - now
            now = time.time()
        self.output = imm_result
    
    def backward(self):
        E = self.output - self.train_y
        for i in self.layers[-1:-2:-1]:
            now = time.time()
            E = i._backward(E)
            print E.shape
            print str(i), time.time() - now
            now = time.time()
