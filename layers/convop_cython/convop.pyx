from __future__ import division
import numpy as np
import cython
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t
ctypedef Py_ssize_t uint


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
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
@cython.nonecheck(False)
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

    cdef uint batch_size_, output_ch, input_ch
    cdef uint input_y, input_x, kernel_y, kernel_x
    cdef DTYPE_t err_value
    cdef int y, x, y_offset, x_offset

    for batch_size_ in range(batch_size):
        for output_ch in range(output_channel):
            for y in range(out_h):
                for x in range(out_w):
                    err_value = err[batch_size_, output_ch, y, x]
                    for y_offset in range(w_h):
                        for x_offset in range(w_w):
                            input_y = <uint>(y + y_offset)
                            input_x = <uint>(x + x_offset)
                            kernel_y = <uint>(y_offset)
                            kernel_x = <uint>(x_offset)
                            for input_ch in range(input_channel):
                                dx[batch_size_, input_ch, input_y, input_x] += \
                                w[input_ch, output_ch, kernel_y, kernel_x] * err_value

                                dw[input_ch, output_ch, kernel_y, kernel_x] += \
                                data[batch_size_, input_ch, input_y, input_x] * err_value

    dw[...] /= batch_size
