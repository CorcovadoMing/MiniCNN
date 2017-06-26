from layers import Conv2d, Linear, Maxpooling, Relu, LeakyRelu, Softmax, Reshape
from data import DataProvider
from net import Net
import numpy as np
import time
from keras.datasets import cifar10

from progressive.bar import Bar

if __name__ == '__main__':
    (x, y), (xt, yt) = cifar10.load_data()

    x = x.reshape(50000, 32, 32, 3).transpose(0,3,1,2)
    xt = xt.reshape(10000, 32, 32, 3).transpose(0,3,1,2)
    def preprocessing(x):
        x = np.expand_dims(x, 1)
        x = x.astype(np.float)
        x /= 255.0
        x -= x.mean()
        x /= x.std()
        return x

    x = preprocessing(x)
    xt = preprocessing(xt)

    x = x.reshape(50000, 32, 32, 3).transpose(0,3,1,2)
    xt = xt.reshape(10000, 32, 32, 3).transpose(0,3,1,2)
    y = y.ravel()
    yt = yt.ravel()

    # Model
    net = Net()
    net.push(Conv2d(3, 3, 3, 8)) # 3x32x32 -> 8x30x30
    net.push(LeakyRelu())
    net.push(Conv2d(3, 3, 8, 8)) # 8x30x30 -> 8x28x28
    net.push(LeakyRelu())
    net.push(Maxpooling(2, 2)) # 8x28x28 -> 8x14x14
    net.push(Conv2d(3, 3, 8, 16)) # 8x14x14 -> 16x12x12
    net.push(LeakyRelu())
    net.push(Conv2d(3, 3, 16, 16)) # 16x12x12 -> 16x10x10
    net.push(LeakyRelu())
    net.push(Conv2d(3, 3, 16, 16)) # 16x10x10 -> 16x8x8
    net.push(LeakyRelu())
    net.push(Maxpooling(2, 2)) # 16x8x8 -> 16x4x4
    net.push(Reshape((256)))
    net.push(Linear(256, 84))
    net.push(LeakyRelu())
    net.push(Softmax(84, 10))

    # Data
    data = DataProvider()
    n = 50000
    nt = 10000
    data.train_input(x[:n], y[:n])
    data.test_input(xt[:nt], yt[:nt])
    data.batch_size(32)

    lr = 0.001
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
        #print 'Forward Time:  ', sum(f)
        #print f
        #print 'Backward Time: ', sum(b)
        #print b
        #print

        # Learning rate decay
        lr *= gamma
