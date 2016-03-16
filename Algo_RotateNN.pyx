import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from numpy cimport ndarray
cimport cython
from cython import parallel
from collections import namedtuple
import sys

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef class AlgoRotateNN:

    cdef _cy_fragment_vals_to_words(self, np.ndarray[np.uint8_t, ndim=1] ptxt):
        cdef np.uint8_t[:,:] xxx
        pass