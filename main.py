from layers import Conv2d, Linear, Maxpooling, Relu, Softmax
from net import Net

if __name__ == '__main__':
    net = Net()
    net.push(Conv2d(3, 3))
    net.debug()