from layers import Conv2d, Linear, Maxpooling, Relu, Softmax
from net import Net
import numpy as np

if __name__ == '__main__':
    x = np.random.rand(3, 3)
    y = [0]
    net = Net()
    net.push(Conv2d(3, 3))
    net.input(x, y, 'train')
    net.debug()