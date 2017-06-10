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


x = np.random.rand(5,3,5,5)
w = np.random.rand(3,5,3,3)

out_map_size = np.array(x.shape[2:]) - np.array(w.shape[2:]) + 1
out_map_size = list(x.shape[:1]) + list(w.shape[1:2]) + list(out_map_size)
output = np.zeros(out_map_size)
for i in xrange(output.shape[0]):
    for out_ch in xrange(w.shape[1]):
        for in_ch in xrange(w.shape[0]):
            output[i][out_ch] += signal.convolve2d(x[i][in_ch], w[in_ch][out_ch], 'valid')



print output
print
out_map_size = np.array(x.shape[2:]) - np.array(w.shape[2:]) + 1
out_map_size = list(x.shape[:1]) + list(w.shape[1:2]) + list(out_map_size)
output = np.zeros(out_map_size)
conv2d_op(x, _rot180_b(w), output)
print output

