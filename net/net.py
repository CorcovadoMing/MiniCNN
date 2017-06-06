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
            self.train_y = np.array(y, dtype=np.int)
        elif type == 'test':
            self.test_x = np.array(x, dtype=np.float)
            self.test_y = np.array(y, dtype=np.int)
        else:
            raise TypeError

    def forward(self):
        imm_result = self.train_x
        for i in self.layers:
            now = time.time()
            imm_result = i._forward(imm_result)
            #print str(i), time.time() - now
            now = time.time()
        self.output = imm_result

        # Evaluation
        count = 0.
        for i in xrange(len(self.output)):
            if self.output[i].argmax() == self.train_y[i]:
                count += 1.
        print 'Acc: ' + str(count / len(self.train_y))
    
    def backward(self):
        self.output[range(self.output.shape[0]), self.train_y] -= 1
        E = self.output

        for i in self.layers[::-1]:
            now = time.time()
            E = i._backward(E)
            #print str(i), time.time() - now
            now = time.time()
        
        for i in self.layers[::-1]:
            now = time.time()
            i._update(0.5)
            #print str(i), time.time() - now
            now = time.time()
