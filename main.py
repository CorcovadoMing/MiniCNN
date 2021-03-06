from layers import Conv2d, Linear, Maxpooling, Relu, Softmax, Reshape
from data import DataProvider
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

    # Model
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

    # Data
    data = DataProvider()
    n = 10000
    data.train_input(x[:n], y[:n])
    data.test_input(xt, yt)
    data.batch_size(16)

    lr = 0.0009
    gamma = 0.9
    for epoch in xrange(50):
        print 'Epoch: ', epoch

        # Training (Mini-batch)
        now = time.time()
        for _ in xrange(data.batch_run()):
            net.input(data.next_batch())
            net.forward()
            net.backward(lr)
        t = time.time() - now
        acc, loss = net.get_record()
        print 'Acc:    ', np.array(acc).mean()
        print 'Loss:   ', np.array(loss).mean()
        print 'Time:   ', t
        f, b = net.get_profile()
        net.clear_record()

        # Testing
        net.input(data.test())
        net.forward()
        print 'Val:    ', net.get_record()[0][0]
        print 'Loss:   ', net.get_record()[1][0]
        net.clear_record()
        print

        # Profile
        print 'Forward Time:  ', sum(f)
        print f
        print 'Backward Time: ', sum(b)
        print b
        print

        # Learning rate decay
        lr *= gamma
