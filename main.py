from layers import Conv2d, Linear, Maxpooling, Relu, Softmax, Reshape
from net import Net
import numpy as np

if __name__ == '__main__':
    x = np.random.rand(500, 1, 28, 28)
    y = [0]*50
    net = Net()
    net.push(Conv2d(3, 3, 1, 3)) # 1x28x28 -> 3x26x26
    net.push(Relu())
    net.push(Reshape((2028, 1)))
    net.push(Softmax(2028, 10))
    net.input(x, y, 'train')
    net.forward()
    #net.debug() 