from layers import Conv2d, Linear, Maxpooling, Relu, Softmax, Reshape
from net import Net
import numpy as np
import time
from keras.datasets import mnist

if __name__ == '__main__':


    (x, y), (_, _) = mnist.load_data()
    x = np.expand_dims(x, 1)
    x = x.astype(np.float)
    x /= 255.0
    #x = np.random.random((n, 1, 28, 28))
    #y = np.random.randint(2, size=(n))

    net = Net()
    net.push(Conv2d(7, 7, 1, 3)) # 1x28x28 -> 3x22x22
    net.push(Relu())
    net.push(Maxpooling(2, 2)) # 3x22x22 -> 3x11x11
    net.push(Reshape((363)))
    net.push(Linear(363, 64))
    net.push(Relu())
    net.push(Softmax(64, 10))
    #net.input(x, y, 'train')

    for epoch in xrange(50):
        print 'Epoch: ', epoch
        n = 128
        now = time.time()
        for i in xrange(0, 5000, n):
            if x[i:i+n].shape[0] != 0:
                net.input(x[i:i+n], y[i:i+n], 'train')
                net.forward()
                net.backward()
        t = time.time() - now
        print 'Acc: ', np.array(net.get_record()).mean(), 'Time: ', t 
    
    #print net.get_record()