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
cdef class AlgoSubSampling:

    def cy_subsample_grey_uint8(self, image):
        #
        cdef np.uint8_t[:,:] src = image
        cdef Py_ssize_t ix = image.shape[0]
        cdef Py_ssize_t iy = image.shape[1]

        # find dimensions of new image
        cdef Py_ssize_t ox = ix/2
        cdef Py_ssize_t oy = iy/2

        # create array for return
        cdef np.uint8_t[:,:] dst = np.zeros((ox, oy), dtype=np.uint8)

        # populate the destination image
        cdef Py_ssize_t index_x, index_y, index_x_new, index_y_new, index_x_new_p1, index_y_new_p1

        # sub sampling logic
        for index_x in range(ox):
            for index_y in range(oy):
                index_x_new = index_x * 2
                index_y_new = index_y * 2
                index_x_new_p1 = index_x_new + 1
                index_y_new_p1 = index_y_new + 1
                dst[index_x, index_y] = <np.uint8_t>(
                    <np.float64_t>(
                        src[index_x_new, index_y_new] +
                        src[index_x_new_p1, index_y_new] +
                        src[index_x_new, index_y_new_p1] +
                        src[index_x_new_p1, index_y_new_p1]
                    )/4.0)

        return dst


    def cy_subsample_rgb_uint8(self, image):
        #
        cdef np.uint8_t[:,:,:] src = image
        cdef Py_ssize_t ix = image.shape[0]
        cdef Py_ssize_t iy = image.shape[1]
        cdef Py_ssize_t iz = image.shape[2]

        # find dimensions of new image
        cdef Py_ssize_t ox = ix/2
        cdef Py_ssize_t oy = iy/2

        # create array for return
        cdef np.uint8_t[:,:,:] dst = np.zeros((ox, oy, iz), dtype=np.uint8)

        # populate the destination image
        cdef Py_ssize_t index_x, index_y, index_z, index_x_new, index_y_new, index_x_new_p1, index_y_new_p1

        # sub sampling logic
        for index_x in range(ox):
            for index_y in range(oy):
                index_x_new = index_x * 2
                index_y_new = index_y * 2
                index_x_new_p1 = index_x_new + 1
                index_y_new_p1 = index_y_new + 1
                #for index_z in range(iz):
                for index_z in range(3):
                    dst[index_x, index_y, index_z] = <np.uint8_t>(
                        <np.float64_t>(
                            src[index_x_new, index_y_new, index_z] +
                            src[index_x_new_p1, index_y_new, index_z] +
                            src[index_x_new, index_y_new_p1, index_z] +
                            src[index_x_new_p1, index_y_new_p1, index_z]
                        )/4.0)

        return dst



    def cy_subsample_grey_float(self, image):
        #
        cdef np.float_t[:,:] src = image
        cdef Py_ssize_t ix = image.shape[0]
        cdef Py_ssize_t iy = image.shape[1]

        # find dimensions of new image
        cdef Py_ssize_t ox = ix/2
        cdef Py_ssize_t oy = iy/2

        # create array for return
        cdef np.float_t[:,:] dst = np.zeros((ox, oy), dtype=np.float)

        # populate the destination image
        cdef Py_ssize_t index_x, index_y, index_x_new, index_y_new, index_x_new_p1, index_y_new_p1

        # sub sampling logic
        for index_x in range(ox):
            for index_y in range(oy):
                index_x_new = index_x * 2
                index_y_new = index_y * 2
                index_x_new_p1 = index_x_new + 1
                index_y_new_p1 = index_y_new + 1
                dst[index_x, index_y] = <np.float_t>(
                    <np.float64_t>(
                        src[index_x_new, index_y_new] +
                        src[index_x_new_p1, index_y_new] +
                        src[index_x_new, index_y_new_p1] +
                        src[index_x_new_p1, index_y_new_p1]
                    )/4.0)

        return dst


    def cy_subsample_rgb_float(self, image):
        #
        cdef np.float_t[:,:,:] src = image
        cdef Py_ssize_t ix = image.shape[0]
        cdef Py_ssize_t iy = image.shape[1]
        cdef Py_ssize_t iz = image.shape[2]

        # find dimensions of new image
        cdef Py_ssize_t ox = ix/2
        cdef Py_ssize_t oy = iy/2

        # create array for return
        cdef np.float_t[:,:,:] dst = np.zeros((ox, oy, iz), dtype=np.float)

        # populate the destination image
        cdef Py_ssize_t index_x, index_y, index_z, index_x_new, index_y_new, index_x_new_p1, index_y_new_p1

        # sub sampling logic
        for index_x in range(ox):
            for index_y in range(oy):
                index_x_new = index_x * 2
                index_y_new = index_y * 2
                index_x_new_p1 = index_x_new + 1
                index_y_new_p1 = index_y_new + 1
                #for index_z in range(iz):
                for index_z in range(3):
                    dst[index_x, index_y, index_z] = <np.float_t>(
                        <np.float64_t>(
                            src[index_x_new, index_y_new, index_z] +
                            src[index_x_new_p1, index_y_new, index_z] +
                            src[index_x_new, index_y_new_p1, index_z] +
                            src[index_x_new_p1, index_y_new_p1, index_z]
                        )/4.0)

        return dst


