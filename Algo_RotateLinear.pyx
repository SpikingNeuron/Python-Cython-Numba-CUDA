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
#@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef class AlgoRotateLin:

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
        cdef Py_ssize_t index_x, index_y, index_x_new, index_y_new, index_x_new_p1, index_y_new_p1

        # rotation logic
        cos_t = cos(-theta)
        sin_t = sin(-theta)
        for index_x in range(ox):
            for index_y in range(oy):
                index_y_new = <Py_ssize_t>((<np.float64_t>(index_x - cx)*sin_t + <np.float64_t>(index_y - cy)*cos_t) + ix/2)
                index_x_new = <Py_ssize_t>((<np.float64_t>(index_x - cx)*cos_t - <np.float64_t>(index_y - cy)*sin_t) + iy/2)
                index_y_new_p1 = index_y_new + 1
                index_x_new_p1 = index_x_new + 1
                if 0 <= index_x_new_p1 < ix and 0 <= index_y_new_p1 < iy:
                    dst[index_x, index_y] = <np.uint8_t>(
                        <np.float64_t>(
                            src[index_x_new, index_y_new] +
                            src[index_x_new, index_y_new_p1] +
                            src[index_x_new_p1, index_y_new] +
                            src[index_x_new_p1, index_y_new_p1]
                        )/4.0)
        return dst


    def cy_rotate_grey_float(self, image, theta_):

        # storage variables
        cdef np.float64_t mincx=0, mincy=0, maxcx=0, maxcy=0, coorx=0, coory=0

        #
        cdef np.float_t[:,:] src = image
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
        cdef np.float_t[:,:] dst = np.zeros((ox, oy), dtype=np.float)

        # populate the destination image
        cdef Py_ssize_t cx = ox/2
        cdef Py_ssize_t cy = oy/2
        cdef Py_ssize_t index_x, index_y, index_x_new, index_y_new, index_x_new_p1, index_y_new_p1

        # rotation logic
        cos_t = cos(-theta)
        sin_t = sin(-theta)
        for index_x in range(ox):
            for index_y in range(oy):
                index_y_new = <Py_ssize_t>((<np.float64_t>(index_x - cx)*sin_t + <np.float64_t>(index_y - cy)*cos_t) + ix/2)
                index_x_new = <Py_ssize_t>((<np.float64_t>(index_x - cx)*cos_t - <np.float64_t>(index_y - cy)*sin_t) + iy/2)
                index_y_new_p1 = index_y_new + 1
                index_x_new_p1 = index_x_new + 1
                if 0 <= index_x_new_p1 < ix and 0 <= index_y_new_p1 < iy:
                    dst[index_x, index_y] = <np.float_t>(
                        <np.float64_t>(
                            src[index_x_new, index_y_new] +
                            src[index_x_new, index_y_new_p1] +
                            src[index_x_new_p1, index_y_new] +
                            src[index_x_new_p1, index_y_new_p1]
                        )/4.0)
        return dst



    def cy_rotate_rgb_uint8(self, image, theta_):

        # storage variables
        cdef np.float64_t mincx=0, mincy=0, maxcx=0, maxcy=0, coorx=0, coory=0

        #
        cdef np.uint8_t[:,:,:] src = image
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
        cdef Py_ssize_t oz = <Py_ssize_t>image.shape[2]

        # create array for return
        cdef np.uint8_t[:,:,:] dst = np.zeros((ox, oy, oz), dtype=np.uint8)

        # populate the destination image
        cdef Py_ssize_t cx = ox/2
        cdef Py_ssize_t cy = oy/2
        cdef Py_ssize_t index_x, index_y, index_x_new, index_y_new, index_x_new_p1, index_y_new_p1

        # rotation logic
        cos_t = cos(-theta)
        sin_t = sin(-theta)
        for index_x in range(ox):
            for index_y in range(oy):
                index_y_new = <Py_ssize_t>((<np.float64_t>(index_x - cx)*sin_t + <np.float64_t>(index_y - cy)*cos_t) + ix/2)
                index_x_new = <Py_ssize_t>((<np.float64_t>(index_x - cx)*cos_t - <np.float64_t>(index_y - cy)*sin_t) + iy/2)
                index_y_new_p1 = index_y_new + 1
                index_x_new_p1 = index_x_new + 1
                if 0 <= index_x_new_p1 < ix and 0 <= index_y_new_p1 < iy:
                    #for index_z in range(oz):
                    for index_z in range(3):
                        dst[index_x, index_y, index_z] = <np.uint8_t>(
                            <np.float64_t>(
                                src[index_x_new, index_y_new, index_z] +
                                src[index_x_new, index_y_new_p1, index_z] +
                                src[index_x_new_p1, index_y_new, index_z] +
                                src[index_x_new_p1, index_y_new_p1, index_z]
                            )/4.0)

        return dst



    def cy_rotate_rgb_float(self, image, theta_):

        # storage variables
        cdef np.float64_t mincx=0, mincy=0, maxcx=0, maxcy=0, coorx=0, coory=0

        #
        cdef np.float_t[:,:,:] src = image
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
        cdef Py_ssize_t oz = <Py_ssize_t>image.shape[2]

        # create array for return
        cdef np.float_t[:,:,:] dst = np.zeros((ox, oy, oz), dtype=np.float)

        # populate the destination image
        cdef Py_ssize_t cx = ox/2
        cdef Py_ssize_t cy = oy/2
        cdef Py_ssize_t index_x, index_y, index_x_new, index_y_new, index_x_new_p1, index_y_new_p1

        # rotation logic
        cos_t = cos(-theta)
        sin_t = sin(-theta)
        for index_x in range(ox):
            for index_y in range(oy):
                index_y_new = <Py_ssize_t>((<np.float64_t>(index_x - cx)*sin_t + <np.float64_t>(index_y - cy)*cos_t) + ix/2)
                index_x_new = <Py_ssize_t>((<np.float64_t>(index_x - cx)*cos_t - <np.float64_t>(index_y - cy)*sin_t) + iy/2)
                index_y_new_p1 = index_y_new + 1
                index_x_new_p1 = index_x_new + 1
                if 0 <= index_x_new_p1 < ix and 0 <= index_y_new_p1 < iy:
                    #for index_z in range(oz):
                    for index_z in range(3):
                        dst[index_x, index_y, index_z] = <np.float_t>(
                            <np.float64_t>(
                                src[index_x_new, index_y_new, index_z] +
                                src[index_x_new, index_y_new_p1, index_z] +
                                src[index_x_new_p1, index_y_new, index_z] +
                                src[index_x_new_p1, index_y_new_p1, index_z]
                            )/4.0)

        return dst

