import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from numpy cimport ndarray
cimport cython
from cython import parallel
from collections import namedtuple
import sys
from libc.math cimport sin, cos


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef class AlgoRotateNN:

    cdef _minmax(self, np.float64_t [:] coor, np.float64_t [:] minc, np.float64_t [:] maxc):
        if coor[0] < minc[0]:
            minc[0] = coor[0]
        if coor[0] > maxc[0]:
            maxc[0] = coor[0]
        if coor[1] < minc[1]:
            minc[1] = coor[1]
        if coor[1] > maxc[1]:
            maxc[1] = coor[1]

    def cy_rotate_grey_uint8(self, image, theta_):

        # storage variables
        cdef np.float64_t mincx=0, mincy=0, maxcx=0, maxcy=0, coorx=0, coory=0

        #
        cdef np.uint8_t[:,:] src = image
        cdef np.float64_t theta = np.pi / 180 * theta_
        cdef np.float64_t cos_t = cos(theta)
        cdef np.float64_t sin_t = sin(theta)
        cdef Py_ssize_t ix = image.shape[0]
        cdef Py_ssize_t iy = image.shape[1]
        cdef np.float64_t sin_t_ix = sin_t * ix
        cdef np.float64_t cos_t_ix = cos_t * ix
        cdef np.float64_t sin_t_iy = sin_t * iy
        cdef np.float64_t cos_t_iy = cos_t * iy

        # find dimensions of new image
        mincx = min(mincx, sin_t_ix)
        maxcx = max(maxcx, sin_t_ix)
        mincy = min(mincy, cos_t_ix)
        maxcy = max(maxcy, cos_t_ix)
        mincx = min(mincx, cos_t_iy)
        maxcx = max(maxcx, cos_t_iy)
        mincy = min(mincy, -sin_t_iy)
        maxcy = max(maxcy, -sin_t_iy)
        mincx = min(mincx, sin_t_ix + cos_t_iy)
        maxcx = max(maxcx, sin_t_ix + cos_t_iy)
        mincy = min(mincy, cos_t_ix - sin_t_iy)
        maxcy = max(maxcy, cos_t_ix - sin_t_iy)
        cdef Py_ssize_t oy = <Py_ssize_t>(maxcx - mincx + 0.5)
        cdef Py_ssize_t ox = <Py_ssize_t>(maxcy - mincy + 0.5)

        # create array for return
        cdef np.uint8_t[:,:] dst = np.zeros((ox, oy), dtype=np.uint8)

        # populate the destination image
        cdef Py_ssize_t cx = ox/2
        cdef Py_ssize_t cy = oy/2
        cdef Py_ssize_t index_x, index_y, index_x_new, index_y_new

        # rotation logic
        cos_t = cos(-theta)
        sin_t = sin(-theta)
        for index_x in range(ox):
            for index_y in range(oy):
                index_y_new = <Py_ssize_t>((<np.float64_t>(index_x - cx)*sin_t + <np.float64_t>(index_y - cy)*cos_t) + ix/2)
                index_x_new = <Py_ssize_t>((<np.float64_t>(index_x - cx)*cos_t - <np.float64_t>(index_y - cy)*sin_t) + iy/2)
                if 0 <= index_x_new <= ix and 0 <= index_y_new <= iy:
                    dst[index_x, index_y] = src[index_x_new, index_y_new]





        return dst