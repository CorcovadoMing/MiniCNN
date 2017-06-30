from layers import Conv2d, Linear, Maxpooling, Relu, LeakyRelu, Softmax, Reshape, BatchNorm
from data import DataProvider
from net import Net
import numpy as np
import time
from keras.datasets import cifar10

from progressive.bar import Bar

if __name__ == '__main__':
    (x, y), (xt, yt) = cifar10.load_data()

    def preprocessing(x):
        x = np.expand_dims(x, 1)
        x = x.astype(np.float)
        x /= (x.max()/4)
        x -= x.mean()
        return x

    x = preprocessing(x)
    xt = preprocessing(xt)
    print x.mean()
    print x.std()

    x = x.reshape(50000, 32, 32, 3).transpose(0,3,1,2)
    xt = xt.reshape(10000, 32, 32, 3).transpose(0,3,1,2)
    y = y.ravel()
    yt = yt.ravel()

    # Model
    net = Net()
    net.push(Conv2d(5, 5, 3, 6)) # 3x32 -> 6x28
    net.push(Relu())
    net.push(BatchNorm())
    net.push(Maxpooling(2, 2)) # 6x28 -> 6x14
    net.push(Conv2d(5, 5, 6, 16)) # 6x14x14 -> 16x10x10
    net.push(Relu())
    net.push(BatchNorm())
    net.push(Maxpooling(2, 2)) # 16x10x10 -> 16x5x5
    net.push(Reshape((400)))
    net.push(Linear(400, 200))
    net.push(Relu())
    net.push(BatchNorm())
    net.push(Softmax(200, 10))

    # Data
    data = DataProvider()
    n = 50000
    nt = 10000
    data.train_input(x[:n], y[:n])
    data.test_input(xt[:nt], yt[:nt])
    data.batch_size(32)
    data.batch_size_test(1000)

    lr = 0.001
    gamma = 1
    mom = 0.95
    l2_decay = 1e-3
    total_epoch = 1000

    loss_cache = 100
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
        loss_avg = np.array(loss).mean()
        loss_diff = loss_avg - loss_cache
        loss_cache = loss_avg
        print 'Acc:    ', np.array(acc).mean()
        print 'Loss:   ', loss_avg
        print 'Time:   ', t
        f, b = net.get_profile()
        net.clear_record()

        bar_t = Bar(max_value=nt)
        bar_t.cursor.clear_lines(2)  # Make some room
        bar_t.cursor.save()  # Mark starting line
        for _ in xrange(data.batch_run_test()):
            net.input(data.next_batch_test())
            net.forward()
            bar_t.cursor.restore()  # Return cursor to start
            bar_t.draw(value=data.get_count_test())
        acc, loss = net.get_record()
        print 'Val:    ', np.array(acc).mean()
        print 'Loss:   ', np.array(loss).mean()
        net.clear_record()
        lr *= gamma
        print lr, loss_diff
        print

        # Profile
        #print 'Forward Time:  ', sum(f)
        #print f
        #print 'Backward Time: ', sum(b)
        #print b
        #print
