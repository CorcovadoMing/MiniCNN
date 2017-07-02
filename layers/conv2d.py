import numpy as np
from gemm import im2col
from math import sqrt

class Conv2d:
    def __init__(self, kw, kh, input_channel, output_channel, pw=0, ph=0, sw=1, sh=1):
        # Control flag
        self.is_first_layer = False
        self.no_bias = False
        # Momentum
        self.m = 0
        self.v = 0
        self.c = 1

        self.weights = np.random.rand(input_channel, output_channel, kw, kh) * sqrt(2.0/(input_channel * output_channel * kw * kh))
        self.weights = self.weights.transpose(1,2,3,0).reshape(self.weights.shape[1], -1)

        self.bias = np.ones((1, output_channel, 1, 1)) * 0.01
        self.k = kw
        self.ic = input_channel
        self.oc = output_channel

    def set_first(self):
        self.is_first_layer = True
        return self.is_first_layer

    def set_no_bias(self):
        self.no_bias = True
        return self.no_bias

    def _forward(self, x):
        self.input_shape = x.shape # Cache the input for backward use
        out_map_size = np.array(x.shape[2:]) - np.array([self.k, self.k]) + 1
        out_map_size = [self.weights.shape[0]] + [x.shape[0]] + list(out_map_size)

        x_ = im2col(x, self.k, self.k, 0, 1)
        self.input_col = x_ # Cache the input for backward use
        output = self.weights.dot(x_).reshape(out_map_size).transpose(1,0,2,3)

        if self.no_bias:
            return output
        else:
            return output + self.bias

    def _backward(self, err, res):
        delta = np.multiply(err, res)

        delta_vector = delta.transpose(1,0,2,3).reshape(delta.shape[1], -1)
        self.d_weights = delta_vector.dot(self.input_col.T).reshape(self.oc, self.k, self.k, self.ic).transpose(3,0,1,2)
        self.d_weights = self.d_weights.transpose(1,2,3,0).reshape(self.d_weights.shape[1], -1)

        if not self.no_bias:
            self.d_bias = np.sum(delta, axis=(0, 2, 3))[None, :, None, None]

        if not self.is_first_layer:
            delta_col = im2col(delta, self.k, self.k, self.k-1, 1)
            w_ = self.weights.reshape(self.oc, self.k, self.k, self.ic).transpose(3,0,1,2)
            w_ = w_[:, :, ::-1, ::-1]
            w_ = w_.swapaxes(0,1).transpose(1,2,3,0).reshape(w_.shape[0], -1)
            output = w_.dot(delta_col)
            output = output.reshape(self.ic, err.shape[0], -1).swapaxes(0, 1)
            return output.reshape(self.input_shape), None
        else:
            return None, None

    def _update(self, step, beta_1, beta_2, epoch):
        #self.pd_weight = self.hd_weight
        #self.hd_weight = (self.hd_weight * mom) - (step * self.d_weights) - (step * decay * self.weights)
        #self.weights += -mom * self.pd_weight + (1 + mom) * self.hd_weight
        # var = (self.pd_weight * mom) - (step * self.d_weights) - (step * decay * self.weights)
        # self.pd_weight = var
        # self.weights += var

        self.m = beta_1 * self.m + (1 - beta_1) * self.d_weights
        self.v = beta_2 * self.v + (1 - beta_2) * (self.d_weights ** 2)
        m_hat = self.m / (1 - (beta_1 ** epoch))
        v_hat = self.v / (1 - (beta_2 ** epoch))
        self.weights = self.weights - ((step * m_hat) / (np.sqrt(v_hat) + 1e-8))


        if not self.no_bias:
            self.bias -= step * self.d_bias

def test_convolution(inp, w, out_shape):
    out = np.zeros(out_shape)

    for n in xrange(inp.shape[0]): # batch
        for out_ch in xrange(w.shape[1]):
            for in_ch in xrange(inp.shape[1]):
                for y in xrange(out.shape[2]):
                    for x in xrange(out.shape[3]):
                        for j in xrange(w.shape[2]):
                            for i in xrange(w.shape[3]):
                                out[n, out_ch, y, x] += inp[n, in_ch, y+j, x+i] * w[in_ch, out_ch, j, i]
    return out

if __name__ == '__main__':
    print 'Forward Test'
    print '========================'
    x = np.random.rand(5, 4, 8, 8)
    w = np.random.rand(4, 9, 3, 3)
    y = test_convolution(x, w, (5,9,6,6))

    w_ = w.transpose(1,2,3,0).reshape(w.shape[1], -1)
    x_ = im2col(x, 3, 3, 0, 1)
    y_ = w_.dot(x_).reshape((9,5,6,6)).transpose(1,0,2,3)

    print y.shape, y.sum()
    print y_.shape, y_.sum()
    print np.allclose(y_, y)



    print
    print
    print 'Backward Test (dw)'
    print '========================'
    delta = np.random.rand(5,9,6,6)
    dw = test_convolution(x.transpose(1,0,2,3), delta, (4,9,3,3))


    delta_vector = delta.transpose(1,0,2,3).reshape(delta.shape[1], -1)
    dw_ = delta_vector.dot(x_.T).reshape(9,3,3,4).transpose(3,0,1,2)

    print dw.shape, dw.sum()
    print dw_.shape, dw_.sum()
    print np.allclose(dw_, dw)



    print
    print
    print 'Backward Test (dx)'
    print '========================'
    delta = np.random.rand(5,9,6,6)
    delta_ = np.empty((5,9,10,10))
    for i in xrange(delta.shape[0]):
        for j in xrange(delta.shape[1]):
            delta_[i][j] = np.pad(delta[i][j], 2, 'constant')
    dx = test_convolution(delta_, w.transpose(1,0,2,3), (5,4,8,8))

    delta_col = im2col(delta, 3, 3, 2, 1)
    w_ = w_.reshape(9,3,3,4).transpose(3,0,1,2)
    print 'is w_ == w?'
    print np.allclose(w_, w)

    w_ = w_.swapaxes(0,1).transpose(1,2,3,0).reshape(w_.shape[0], -1)
    dx_ = w_.dot(delta_col)
    dx_ = dx_.reshape(4,5,8,8).transpose(1,0,2,3)

    print dx.shape, dx.sum()
    print dx_.shape, dx_.sum()
    print np.allclose(dx_, dx)
