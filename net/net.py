import numpy as np

class Net:
    def __init__(self):
        self.layers = []
    
    def push(self, layer):
        self.layers.append(layer)

    def pop(self):
        self.layers.pop()

    def input(self, x, y, type):
        if type == 'train':
            self.train_x = np.array(x)
            self.train_y = np.array(y)
        elif type == 'test':
            self.test_x = np.array(x)
            self.test_y = np.array(y)
        else:
            raise TypeError

    def debug(self):
        try:
            print 'Layers:', self.layers
            print 'Train_x', self.train_x
            print 'Train_y', self.train_y
            print 'Test_x', self.test_x
            print 'Test_y', self.test_y
        except:
            pass
