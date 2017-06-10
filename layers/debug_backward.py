from convop import deconv2d_op, conv2d_op
import numpy as np
from scipy import signal

def _rot180(kernel):
    return np.flipud(np.fliplr(kernel))

def _rot180_b(data):
    for i in xrange(data.shape[0]):
        for j in xrange(data.shape[1]):
            data[i, j, :, :] = _rot180(data[i, j, :, :])
    return data

err = np.random.rand(2,5,3,3)
x = np.random.rand(2,2,5,5)
dx = np.zeros_like(x)

w = np.random.rand(2,5,3,3)
dw = np.zeros_like(w)

deconv2d_op(x, err, _rot180_b(w), dx, dw)

_rot180_b(w)
print dx.dtype
print
print dw[:, :, ::-1, ::-1].dtype
print

dx2 = np.zeros_like(x)
dw2 = np.zeros_like(w)

for i in xrange(err.shape[0]):
    for in_ch in xrange(w.shape[0]):
        for out_ch in xrange(w.shape[1]):
            dw2[in_ch][out_ch] += signal.convolve2d(_rot180(x[i][in_ch]), err[i][out_ch], mode='valid')
            dx2[i][in_ch] += signal.convolve2d(err[i][out_ch], _rot180(w[in_ch][out_ch]))
dw2 /= err.shape[0]

print dx2.dtype
print
print dw2.dtype

#print dx2==dx
#print dw2==dw[:, :, ::-1, ::-1]