import numpy as np
import time

class Net:
    def __init__(self):
        self.record = []
        self.layers = []
        self.profile_forward = []
        self.profile_backward = []
    
    def push(self, layer):
        self.layers.append(layer)

    def pop(self):
        self.layers.pop()

    def input(self, (x, y)):
        self.x = np.array(x, dtype=np.float)
        self.y = np.array(y, dtype=np.int)

    def forward(self):
        pf = []
        imm_result = self.x
        for i in self.layers:
            now = time.time()
            imm_result = i._forward(imm_result)
            pf.append(time.time() - now)
            now = time.time()
        self.output = imm_result
        self.profile_forward.append(pf)

        # Catrgorical Cross-Entropy
        loss = 0.
        for i in xrange(len(self.output)):
            if self.output[i][self.y[i]] == 0:
                loss -= 0
            else:
                loss -= np.log(self.output[i][self.y[i]])
        loss /= len(self.output)

        # Evaluation
        count = 0.
        for i in xrange(len(self.output)):
            if self.output[i].argmax() == self.y[i]:
                count += 1.
        #print 'Acc: ' + str(count / len(self.y)), 'Loss: ' + str(loss)
        self.record.append(count / len(self.y))
    
    def backward(self, lr=0.01):
        pb = []
        self.output[xrange(self.output.shape[0]), self.y] -= 1
        self.output /= self.output.shape[0]
        err = self.output
        res = None

        for i in self.layers[::-1]:
            now = time.time()
            err, res = i._backward(err, res)
            pb.append(time.time() - now)
            now = time.time()
        self.profile_backward.append(pb)
        
        for i in self.layers[::-1]:
            now = time.time()
            i._update(lr, 0.90, 1e-6)
            now = time.time()
    
    def get_record(self):
        return self.record
    
    def get_profile(self):
        return np.array(self.profile_forward).sum(axis=0), \
                np.array(self.profile_backward).sum(axis=0)[::-1]
    
    def clear_record(self):
        self.record = []
        self.profile_forward = []
        self.profile_backward = []
