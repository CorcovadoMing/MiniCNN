from layers import Conv2d, Linear, Maxpooling, Relu, Softmax, Reshape
from net import Net
import numpy as np
import time
from keras.datasets import mnist

if __name__ == '__main__':


    (x, y), (xt, yt) = mnist.load_data()

    def preprocessing(x):
        x = np.expand_dims(x, 1)
        x = x.astype(np.float)
        x /= 255.0
        x -= x.mean()
        x /= x.std()
        return x
    
    x = preprocessing(x)
    xt = preprocessing(xt)
    #x = np.random.random((n, 1, 28, 28))
    #y = np.random.randint(2, size=(n))

    net = Net()
    net.push(Conv2d(5, 5, 1, 6)) # 1x28x28 -> 6x24x24
    net.push(Relu())
    net.push(Maxpooling(2, 2)) # 6x24x24 -> 6x12x12
    net.push(Conv2d(5, 5, 6, 16)) # 6x12x12 -> 16x8x8
    net.push(Relu())
    net.push(Maxpooling(2, 2)) # 16x8x8 -> 16x4x4
    net.push(Reshape((256)))
    net.push(Linear(256, 84))
    net.push(Relu())
    net.push(Softmax(84, 10))
    #net.input(x, y, 'train')

    for epoch in xrange(50):
        print 'Epoch: ', epoch
        n = 32
        now = time.time()
        for i in xrange(0, 500, n):
            if x[i:i+n].shape[0] != 0:
                net.input(x[i:i+n], y[i:i+n], 'train')
                net.forward()
                net.backward()
        t = time.time() - now
        print 'Acc: ', np.array(net.get_record()).mean(), 'Time: ', t
        net.clear_record()
        net.input(xt[:100], yt[:100], 'train')
        net.forward()
        print 'Val: ', net.get_record()[0]
        net.clear_record()
    #print net.get_record()