from layers import Conv2d, Linear, Maxpooling, Relu, Softmax, Reshape
from net import Net
import numpy as np

if __name__ == '__main__':
    n = 1
    x = np.random.random((n, 1, 28, 28))
    y = np.random.randint(2, size=(n))

    net = Net()
    net.push(Conv2d(3, 3, 1, 3)) # 1x28x28 -> 3x26x26
    net.push(Relu())
    net.push(Maxpooling(2, 2)) # 3x26x26 -> 3x13x13
    net.push(Reshape((507)))
    net.push(Linear(507, 64))
    net.push(Relu())
    net.push(Softmax(64, 2))
    net.input(x, y, 'train')
    for i in xrange(1):
        net.forward()
        net.backward()