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
cdef class AlgoBlending:

    def cy_blending_grey_uint8(self, image, image2):
        #
        cdef np.uint8_t[:,:] src = image
        cdef np.uint8_t[:,:] star = image2
        cdef Py_ssize_t ix = image.shape[0]
        cdef Py_ssize_t iy = image.shape[1]

        # create array for return
        cdef np.uint8_t[:,:] dst = np.zeros((ix, iy), dtype=np.uint8)

        # populate the destination image
        cdef Py_ssize_t index_x, index_y

        # sub sampling logic
        for index_x in range(ix):
            for index_y in range(iy):
                dst[index_x, index_y] = <np.uint8_t>(
                    <np.float64_t>(
                        src[index_x, index_y] +
                        star[index_x, index_y]
                    )/2.0)

        return dst

    def cy_blending_rgb_uint8(self, image, image2):
        #
        cdef np.uint8_t[:,:,:] src = image
        cdef np.uint8_t[:,:,:] star = image2
        cdef Py_ssize_t ix = image.shape[0]
        cdef Py_ssize_t iy = image.shape[1]
        cdef Py_ssize_t iz = image.shape[2]

        # create array for return
        cdef np.uint8_t[:,:,:] dst = np.zeros((ix, iy, iz), dtype=np.uint8)

        # populate the destination image
        cdef Py_ssize_t index_x, index_y, index_z

        # sub sampling logic
        for index_x in range(ix):
            for index_y in range(iy):
                #for index_z in range(iz):
                for index_z in range(3):
                    dst[index_x, index_y, index_z] = <np.uint8_t>(
                        <np.float64_t>(
                            src[index_x, index_y, index_z] +
                            star[index_x, index_y, index_z]
                        )/2.0)

        return dst

    def cy_blending_grey_float(self, image, image2):
        #
        cdef np.float_t[:,:] src = image
        cdef np.float_t[:,:] star = image2
        cdef Py_ssize_t ix = image.shape[0]
        cdef Py_ssize_t iy = image.shape[1]

        # create array for return
        cdef np.float_t[:,:] dst = np.zeros((ix, iy), dtype=np.float)

        # populate the destination image
        cdef Py_ssize_t index_x, index_y

        # sub sampling logic
        for index_x in range(ix):
            for index_y in range(iy):
                dst[index_x, index_y] = <np.float_t>(
                    <np.float64_t>(
                        src[index_x, index_y] +
                        star[index_x, index_y]
                    )/2.0)

        return dst

    def cy_blending_rgb_float(self, image, image2):
        #
        cdef np.float_t[:,:,:] src = image
        cdef np.float_t[:,:,:] star = image2
        cdef Py_ssize_t ix = image.shape[0]
        cdef Py_ssize_t iy = image.shape[1]
        cdef Py_ssize_t iz = image.shape[2]

        # create array for return
        cdef np.float_t[:,:,:] dst = np.zeros((ix, iy, iz), dtype=np.float)

        # populate the destination image
        cdef Py_ssize_t index_x, index_y, index_z

        # sub sampling logic
        for index_x in range(ix):
            for index_y in range(iy):
                #for index_z in range(iz):
                for index_z in range(3):
                    dst[index_x, index_y, index_z] = <np.float_t>(
                        <np.float64_t>(
                            src[index_x, index_y, index_z] +
                            star[index_x, index_y, index_z]
                        )/2.0)

        return dst

