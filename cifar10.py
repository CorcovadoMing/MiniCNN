from layers import Conv2d, Linear, Maxpooling, Relu, LeakyRelu, AbsRelu, Softmax, Reshape, BatchNorm
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


    n = 50000
    nt = 10000
    x = x[:n]
    y = y[:n]
    xt = xt[:nt]
    yt = yt[:nt]

    # Model
    net = Net()
    net.push(Conv2d(5, 5, 3, 20)) # 3x32 -> 10x28
    net.push(Relu())
    net.push(BatchNorm())
    net.push(Maxpooling(4, 4)) # 10x28 -> 10x7
    net.push(Reshape((980)))
    net.push(Linear(980, 200))
    net.push(Relu())
    net.push(BatchNorm())
    net.push(Softmax(200, 10))

    # Data
    data = DataProvider()
    data.train_input(x, y)
    data.test_input(xt, yt)
    data.batch_size(32)
    data.batch_size_test(1000)

    lr = 1e-3
    gamma = 1
    beta_1 = 0.9
    beta_2 = 0.999
    total_epoch = 100

    loss_cache = 10
    for epoch in xrange(1, total_epoch+1):
        print 'Epoch: {}/{}'.format(epoch, total_epoch)

        # Training (Mini-batch)
        now = time.time()
        data.shuffle()
        bar = Bar(max_value=n)
        bar.cursor.clear_lines(2)  # Make some room
        bar.cursor.save()  # Mark starting line
        for _ in xrange(data.batch_run()):
            net.input(data.next_batch())
            net.forward()
            net.backward(lr, beta_1, beta_2, epoch)
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
        print 'Forward Time:  ', sum(f)
        print f
        print 'Backward Time: ', sum(b)
        print b
        print
