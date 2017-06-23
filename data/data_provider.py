import numpy as np
import math

class DataProvider:
    def __init__(self):
        self.current = 0
        self.bs = 0
    
    def train_input(self, x, y):
        # TODO: Need to check the data dimension first
        self.train_x = np.array(x, dtype=np.float)
        self.train_y = np.array(y, dtype=np.int)
    
    def test_input(self, x, y):
        # TODO: Need to check the data dimension first
        self.test_x = np.array(x, dtype=np.float)
        self.test_y = np.array(y, dtype=np.int)
    
    def test(self):
        return self.test_x, self.test_y

    def batch_size(self, bs):
        self.bs = bs
    
    def get_batch_size(self):
        return self.bs
    
    def get_count(self):
        return self.current
    
    def batch_run(self):
        return int(math.ceil((int(len(self.train_x)) / float(self.bs))))

    def next_batch(self):
        if self.current == len(self.train_x):
            self.current = 0
        start, end = self.current, None
        if self.current + self.bs > len(self.train_x):
            end = len(self.train_x)
            self.current = end
        else:
            end = self.current + self.bs
            self.current += self.bs
        return self.train_x[start:end], self.train_y[start:end]