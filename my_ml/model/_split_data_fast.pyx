# cython: language_level=3
from typing import Tuple

import numpy as np
cimport numpy as np


DTYPE = np.float
ctypedef np.float_t DTYPE_t

def _split_data(
    np.ndarray[DTYPE_t, ndim=2] data,
    DTYPE_t val,
    int feature_j
) -> Tuple[np.ndarray]:
    """
    A faster implementation of np.where. Based on:
    https://stackoverflow.com/questions/34885612/\
            fastest-way-to-find-indices-of-condition-in-numpy-array

    Parameters
    ----------
    data : np.ndarray
        Array to split up.
    val : int
        Threshold to split on.
    feature_j : int
        Index of the jth feature.

    Returns
    -------
    low_y, high_y, low_X, high_X : Tuple[np.ndarray]
        The data split by the threshold into data with low values and
        separate data for high values.
    """
    assert data.dtype == DTYPE

    cdef int xmax = data.shape[0]
    cdef int ymax = data.shape[1] - 1
    cdef unsigned int x
    cdef int low_count = 0
    cdef int high_count = 0
    cdef np.ndarray[DTYPE_t, ndim=1] low_y = np.zeros(xmax, dtype=float)
    cdef np.ndarray[DTYPE_t, ndim=1] high_y = np.zeros(xmax, dtype=float)
    cdef np.ndarray[DTYPE_t, ndim=2] low_X = np.zeros((xmax, ymax), dtype=float)
    cdef np.ndarray[DTYPE_t, ndim=2] high_X = np.zeros((xmax, ymax), dtype=float)
    for x in xrange(xmax):
        if data[x, feature_j] <= val:
            low_y[low_count] = data[x, -1]
            for y in xrange(ymax):
                low_X[low_count, y] = data[x, y]
            low_count += 1
        else:
            high_y[high_count] = data[x, -1]
            for y in xrange(ymax):
                high_X[high_count, y] = data[x, y]
            high_count += 1

    return (
        low_y[0:low_count],
        high_y[0:high_count],
        low_X[0:low_count, :], 
        high_X[0:high_count, :]
    )
