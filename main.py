from layers import Conv2d, Linear, Maxpooling, Relu, Softmax, Reshape
from net import Net
import numpy as np

from keras.datasets import mnist

if __name__ == '__main__':


    (x, y), (_, _) = mnist.load_data()
    n = 50
    x = x[:n]
    y = y[:n]
    x = np.expand_dims(x, 1)
    x = x.astype(np.float)
    x /= 255.0
    #x = np.random.random((n, 1, 28, 28))
    #y = np.random.randint(2, size=(n))

    net = Net()
    net.push(Conv2d(7, 7, 1, 3)) # 1x28x28 -> 1x22x22
    net.push(Relu())
    net.push(Maxpooling(2, 2)) # 1x22x22 -> 1x11x11
    net.push(Reshape((363)))
    net.push(Linear(363, 64))
    net.push(Relu())
    net.push(Softmax(64, 10))
    net.input(x, y, 'train')
    for i in xrange(50):
        net.forward()
        net.backward()
    
    print net.get_record()