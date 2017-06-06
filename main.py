from layers import Conv2d, Linear, Maxpooling, Relu, Softmax, Reshape
from net import Net
import numpy as np

if __name__ == '__main__':
    x = np.random.rand(1, 5, 5)
    y = [0]
    net = Net()
    net.push(Conv2d(3, 3, 1, 3)) # 1x5x5 -> 3x3x3
    net.push(Reshape((27, 1)))
    net.push(Softmax(2))
    net.input(x, y, 'train')
    net.forward()
    net.debug()