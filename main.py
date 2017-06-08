from layers import Conv2d, Linear, Maxpooling, Relu, Softmax, Reshape
from net import Net
import numpy as np

if __name__ == '__main__':
    n = 1
    x = np.random.random((n, 3, 28, 28))
    y = np.random.randint(2, size=(n))

    net = Net()
    net.push(Conv2d(3, 3, 3, 6)) # 1x28x28 -> 6x26x26
    net.push(Relu())
    net.push(Maxpooling(2, 2)) # 6x26x26 -> 6x13x13
    net.push(Reshape((1014)))
    net.push(Linear(1014, 64))
    net.push(Relu())
    net.push(Softmax(64, 2))
    net.input(x, y, 'train')
    for i in xrange(1):
        net.forward()
        net.backward()