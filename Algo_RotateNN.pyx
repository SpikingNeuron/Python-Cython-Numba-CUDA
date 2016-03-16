
cimport cython
cimport numpy as np
import numpy as np
from cython cimport boundscheck, wraparound, initializedcheck, cdivision
from libc.math cimport log

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
class AlgoRotateNN:

    def cy_rotate_degree_0(self, image, theta):
        pass
