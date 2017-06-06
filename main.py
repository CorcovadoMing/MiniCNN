from layers import Conv2d, Linear, Maxpooling, Relu, Softmax, Reshape
from net import Net
import numpy as np

if __name__ == '__main__':
    n = 100
    x = np.random.normal(0, 5, (n, 100))
    y = [int(i.sum() > 0 ) for i in x]

    net = Net()
    #net.push(Conv2d(3, 3, 3, 3)) # 1x28x28 -> 3x26x26
    #net.push(Relu())
    #net.push(Maxpooling(2, 2)) # 3x26x26 -> 3x13x13
    #net.push(Reshape((507, 1)))
    #net.push(Linear(507, 128))
    #net.push(Relu())
    #net.push(Linear(128, 32))
    #net.push(Relu())
    net.push(Softmax(100, 2))
    net.input(x, y, 'train')
    for i in xrange(10):
        net.forward()
        net.backward()