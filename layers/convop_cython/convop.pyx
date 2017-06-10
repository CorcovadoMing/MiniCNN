from __future__ import division
import numpy as np
import cython
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t
ctypedef Py_ssize_t uint
cdef inline int int_max(int a, int b) nogil: return a if a >= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
def conv2d_op(np.ndarray[DTYPE_t, ndim=4] data,
            np.ndarray[DTYPE_t, ndim=4] w,
            np.ndarray[DTYPE_t, ndim=4] out):

    cdef uint batch_size = data.shape[0]
    cdef uint out_h = out.shape[2]
    cdef uint out_w = out.shape[3]
    cdef uint input_channel = w.shape[0]
    cdef uint output_channel = w.shape[1]
    cdef uint w_h = w.shape[2]
    cdef uint w_w = w.shape[3]

    cdef uint batch_size_, input_ch, output_ch, y, x, y_offset, x_offset
    cdef uint input_y, input_x, kernel_y, kernel_x
    cdef DTYPE_t value

    for batch_size_ in range(batch_size):
        for output_ch in range(output_channel):
            for y in range(out_h):
                for x in range(out_w):
                    value = 0.0
                    for y_offset in range(w_h):
                        for x_offset in range(w_w):
                            input_y = <uint>(y + y_offset)
                            input_x = <uint>(x + x_offset)
                            kernel_y = <uint>(y_offset)
                            kernel_x = <uint>(x_offset)
                            for input_ch in range(input_channel):
                                value += data[batch_size_, input_ch, input_y, input_x] * \
                                        w[input_ch, output_ch, kernel_y, kernel_x]
                    out[batch_size_, output_ch, y, x] = value

@cython.boundscheck(False)
@cython.wraparound(False)
def deconv2d_op(np.ndarray[DTYPE_t, ndim=4] data,
                    np.ndarray[DTYPE_t, ndim=4] err,
                    np.ndarray[DTYPE_t, ndim=4] w,
                    np.ndarray[DTYPE_t, ndim=4] dx,
                    np.ndarray[DTYPE_t, ndim=4] dw):

    cdef uint batch_size = err.shape[0]
    cdef uint out_h = err.shape[2]
    cdef uint out_w = err.shape[3]
    cdef uint input_channel = w.shape[0]
    cdef uint output_channel = w.shape[1]
    cdef uint w_h = w.shape[2]
    cdef uint w_w = w.shape[3]
    cdef int fil_mid_h = w_h // 2
    cdef int fil_mid_w = w_w // 2

    cdef uint batch_size_, output_ch, input_ch
    cdef uint input_y, input_x, kernel_y, kernel_x
    cdef DTYPE_t err_value
    cdef int y, x, y_offset_min, y_offset_max, y_offset, x_offset_min, x_offset_max, x_offset

    for batch_size_ in range(batch_size):
        for output_ch in range(output_channel):
            for y in range(out_h):
                y_offset_min = int_max(-y, -fil_mid_h)
                y_offset_max = int_min(out_h-y, fil_mid_h+1)
                for x in range(out_w):
                    err_value = err[batch_size_, output_ch, y, x]
                    x_offset_min = int_max(-x, -fil_mid_w)
                    x_offset_max = int_min(out_w-x, fil_mid_w+1)
                    for y_offset in range(y_offset_min, y_offset_max):
                        for x_offset in range(x_offset_min, x_offset_max):
                            input_y = <uint>(y + y_offset)
                            input_x = <uint>(x + x_offset)
                            kernel_y = <uint>(fil_mid_w + y_offset)
                            kernel_x = <uint>(fil_mid_h + x_offset)
                            for input_ch in range(input_channel):
                                dx[batch_size_, input_ch, input_y, input_x] += \
                                w[input_ch, output_ch, kernel_y, kernel_x] * err_value

                                dw[input_ch, output_ch, kernel_y, kernel_x] += \
                                data[batch_size_, input_ch, input_y, input_x] * err_value

    dw[...] /= batch_size
