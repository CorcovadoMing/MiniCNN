from layers import Conv2d, Linear, Maxpooling, Relu, Softmax, Reshape
from data import DataProvider
from net import Net
import numpy as np
import time
from keras.datasets import mnist

from progressive.bar import Bar

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
    net.push(Conv2d(3, 3, 1, 4)) # 1x28x28 -> 4x26x26
    net.push(Relu())
    net.push(Conv2d(3, 3, 4, 4)) # 4x26x26 -> 4x24x24
    net.push(Relu())
    net.push(Maxpooling(2, 2)) # 4x24x24 -> 4x12x12
    net.push(Conv2d(3, 3, 4, 8)) # 4x12x12 -> 8x10x10
    net.push(Relu())
    net.push(Maxpooling(2, 2)) # 8x10x10 -> 8x5x5
    net.push(Reshape((200)))
    net.push(Linear(200, 64))
    net.push(Relu())
    net.push(Softmax(64, 10))

    # Data
    data = DataProvider()
    n = 60000
    data.train_input(x[:n], y[:n])
    data.test_input(xt, yt)
    data.batch_size(64)

    lr = 0.01
    gamma = 0.99
    mom = 0.99
    l2_decay = 1e-4
    total_epoch = 50
    for epoch in xrange(total_epoch):
        print 'Epoch: {}/{}'.format(epoch, total_epoch)

        # Training (Mini-batch)
        now = time.time()
        bar = Bar(max_value=n)
        bar.cursor.clear_lines(2)  # Make some room
        bar.cursor.save()  # Mark starting line
        for _ in xrange(data.batch_run()):
            net.input(data.next_batch())
            net.forward()
            net.backward(lr, mom, l2_decay)
            bar.cursor.restore()  # Return cursor to start
            bar.draw(value=data.get_count())
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
