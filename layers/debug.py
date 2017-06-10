from convop import deconv2d_op
import numpy as np
from scipy import signal

err = np.array([ [[[1,1,1]]*3]  ], dtype=np.float)
x = np.array([ [[[1,1,1,1,1]]*5]  ], dtype=np.float)
dx = np.zeros_like(x)

w = np.array([ [[[1,1,1]]*3]  ], dtype=np.float)
dw = np.zeros_like(w)

deconv2d_op(x, err, w, dx, dw)

print dx
print
print dw

print
dx = np.zeros_like(x)
dw = np.zeros_like(w)

for i in xrange(err.shape[0]):
    for in_ch in xrange(w.shape[0]):
        for out_ch in xrange(w.shape[1]):
            dw[in_ch][out_ch] += signal.convolve2d(x[i][in_ch], err[i][out_ch], mode='valid')
            dx[i] += signal.convolve2d(err[i][out_ch], w[in_ch][out_ch])
dw /= err.shape[0]

print dx
print
print dw


