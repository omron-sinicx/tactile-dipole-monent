#!/usr/bin/env python

import numpy as np

from functools import partial

from std_msgs.msg import MultiArrayDimension, Float32MultiArray

def _numpy2multiarray(multiarray_type, np_array):
    """Convert numpy.ndarray to multiarray"""
    multiarray = multiarray_type()
    multiarray.layout.dim = [MultiArrayDimension("dim%d" % i, np_array.shape[i], np_array.shape[i] * np_array.dtype.itemsize) for i in range(np_array.ndim)]
    multiarray.data = np_array.reshape(1, -1)[0].tolist()
    return multiarray

def _multiarray2numpy(pytype, dtype, multiarray):
    """Convert multiarray to numpy.ndarray"""
    dims = list(map(lambda x: x.size, multiarray.layout.dim))
    # print(dims)
    return np.array(multiarray.data, dtype=pytype).reshape(dims).astype(dtype)

numpy2f32multi = partial(_numpy2multiarray, Float32MultiArray)
f32multi2numpy = partial(_multiarray2numpy, float, np.float32)