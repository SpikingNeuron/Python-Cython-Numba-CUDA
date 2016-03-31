"""

"""

import numpy as np
import unittest
import sys
import time
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import collections
import matplotlib
import pandas as pd
matplotlib.style.use('ggplot')

# for compiling the cython code
import pyximport
pyximport.install(inplace=False,
                  setup_args={"include_dirs": np.get_include()},
                  reload_support=True)

# Generate c and html file that shows respective c code
from Cython.Build import cythonize

# TODO: check with multi thread
cythonize('Algo_RotateNN.pyx', annotate=True)
cythonize('Algo_RotateLinear.pyx', annotate=True)
cythonize('Algo_SubSampling.pyx', annotate=True)

from Algo_RotateNN import AlgoRotateNN
from Algo_RotateLinear import AlgoRotateLin
from Algo_SubSampling import AlgoSubSampling

# To print on console or to file
custom_stream = sys.stdout
# custom_stream = open("report.txt", "w", encoding="utf-8")

# config vars
_PLOT = True
_ITER_NUM = 100
_result_timings = {}

def plot_images(im0, im1, im2, im3, title):
    """
    plot the image
    :param im0:
    :type im0:
    :param im1:
    :type im1:
    :param im2:
    :type im2:
    :param im3:
    :type im3:
    :param title:
    :type title:
    :return:
    :rtype:
    """

    fig = plt.figure()
    fig.suptitle(title)

    a = fig.add_subplot(2, 2, 1)
    plt.imshow(im0)
    a.set_title('Original')

    a = fig.add_subplot(2, 2, 2)
    plt.imshow(im1)
    a.set_title('Numpy')

    a = fig.add_subplot(2, 2, 3)
    plt.imshow(im2)
    a.set_title('Cython')

    a = fig.add_subplot(2, 2, 4)
    plt.imshow(im3)
    a.set_title('CUDA')

    plt.savefig('Report\\' + title + '.png')
    # plt.show()


def get_image(query):
    """
    get the image
    :param query:
    :type query:
    :return:
    :rtype:
    """
    args = query.split(sep='_')
    lena = None

    if args[2] == 'grey':
        lena = ndimage.imread('lena.jpg', mode='L')
    elif args[2] == 'rgb':
        lena = ndimage.imread('lena.jpg', mode='RGB')
    else:
        raise ValueError('Invalid color type. Allowed rgb or grey')

    if args[3] == 'small':
        lena = misc.imresize(lena, (2048, 2048), interp='bilinear')
    elif args[3] == 'large':
        lena = misc.imresize(lena, (4096, 4096), interp='bilinear')
    else:
        raise ValueError('Invalid size. Allowed small or large')

    if args[4] == 'uint8':
        lena = lena.astype(np.uint8)
    elif args[4] == 'float':
        lena = lena.astype(np.float)
    else:
        raise ValueError('Invalid size. Allowed uint8 or float')

    return lena


def print_utility(time_taken_np, time_taken_cy, time_taken_cuda, message, query):
    """
    print utility
    :param query:
    :param time_taken_np:
    :type time_taken_np:
    :param time_taken_cy:
    :type time_taken_cy:
    :param time_taken_cuda:
    :type time_taken_cuda:
    :param message:
    :type message:
    :return:
    :rtype:
    """
    global custom_stream
    # custom_stream.write('\n' + message)
    custom_stream.write('\nTime taken Numpy \t: ' + str(time_taken_np))
    custom_stream.write('\nTime taken Cython\t: ' + str(time_taken_cy))
    # custom_stream.write('\n\t\tSpeed up\t: ' + str(time_taken_np/time_taken_cy))
    custom_stream.write('\nTime taken CUDA  \t: ' + str(time_taken_cuda))
    # custom_stream.write('\n\t\tSpeed up\t: ' + str(time_taken_cuda))

    custom_stream.write('\n...........')

    # store results
    args = query.split(sep='_')
    _result_timings[args[2] + ' ' + args[3] + ' ' + args[4]] = (time_taken_np, time_taken_cy)

    pass


class TestRotateNNGray(unittest.TestCase):
    """
    Unit test utility for rotating with gray image
    """

    def test_image_grey_small_uint8(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateNN().cy_rotate_grey_uint8(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=0)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateNN' + self._testMethodName)

        # test case check
        pass

    def test_image_grey_large_uint8(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateNN().cy_rotate_grey_uint8(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=0)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateNN' + self._testMethodName)

        # test case check
        pass

    def test_image_grey_small_float(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateNN().cy_rotate_grey_float(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=0)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateNN' + self._testMethodName)

        # test case check
        pass

    def test_image_grey_large_float(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateNN().cy_rotate_grey_float(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=0)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateNN' + self._testMethodName)

        # test case check
        pass


class TestRotateNNRGB(unittest.TestCase):
    """
    Unit test utility for rotating with RGB image
    """

    def test_image_rgb_small_uint8(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateNN().cy_rotate_rgb_uint8(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=0)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateNN' + self._testMethodName)

        # test case check
        pass

    def test_image_rgb_large_uint8(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateNN().cy_rotate_rgb_uint8(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=0)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateNN' + self._testMethodName)

        # test case check
        pass

    def test_image_rgb_small_float(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateNN().cy_rotate_rgb_float(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=0)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateNN' + self._testMethodName)

        # test case check
        pass

    def test_image_rgb_large_float(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateNN().cy_rotate_rgb_float(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=0)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateNN' + self._testMethodName)

        # test case check
        pass


class TestRotateLinGray(unittest.TestCase):
    """
    Unit test utility for rotating with gray image
    """

    def test_image_grey_small_uint8(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=1)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateLin().cy_rotate_grey_uint8(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateLin' + self._testMethodName)

        # test case check
        pass

    def test_image_grey_large_uint8(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=1)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateLin().cy_rotate_grey_uint8(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateLin' + self._testMethodName)

        # test case check
        pass

    def test_image_grey_small_float(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=1)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateLin().cy_rotate_grey_float(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateLin' + self._testMethodName)

        # test case check
        pass

    def test_image_grey_large_float(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=1)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateLin().cy_rotate_grey_float(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateLin' + self._testMethodName)

        # test case check
        pass


class TestRotateLinRGB(unittest.TestCase):
    """
    Unit test utility for rotating with RGB image
    """

    def test_image_rgb_small_uint8(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=1)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateLin().cy_rotate_rgb_uint8(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateLin' + self._testMethodName)

        # test case check
        pass

    def test_image_rgb_large_uint8(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=1)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateLin().cy_rotate_rgb_uint8(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateLin' + self._testMethodName)

        # test case check
        pass

    def test_image_rgb_small_float(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=1)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateLin().cy_rotate_rgb_float(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateLin' + self._testMethodName)

        # test case check
        pass

    def test_image_rgb_large_float(self):
        theta = 33.33
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = ndimage.rotate(test_image, theta, order=1)
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoRotateLin().cy_rotate_rgb_float(test_image, theta)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'RotateLin' + self._testMethodName)

        # test case check
        pass


class TestSubSamplingGray(unittest.TestCase):
    """
    Unit test utility for subsampling gray image
    """

    def test_image_grey_small_uint8(self):
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = (
                            test_image[0::2, 0::2] +
                            test_image[1::2, 0::2] +
                            test_image[0::2, 1::2] +
                            test_image[1::2, 1::2]
                        ) / 4.0
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoSubSampling().cy_subsample_grey_uint8(test_image)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'SubSampling' + self._testMethodName)

        # test case check
        pass

    def test_image_grey_large_uint8(self):
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = (
                            test_image[0::2, 0::2] +
                            test_image[1::2, 0::2] +
                            test_image[0::2, 1::2] +
                            test_image[1::2, 1::2]
                        ) / 4.0
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoSubSampling().cy_subsample_grey_uint8(test_image)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'SubSampling' + self._testMethodName)

        # test case check
        pass

    def test_image_grey_small_float(self):
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = (
                            test_image[0::2, 0::2] +
                            test_image[1::2, 0::2] +
                            test_image[0::2, 1::2] +
                            test_image[1::2, 1::2]
                        ) / 4.0
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoSubSampling().cy_subsample_grey_float(test_image)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'SubSampling' + self._testMethodName)

        # test case check
        pass

    def test_image_grey_large_float(self):
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = (
                            test_image[0::2, 0::2] +
                            test_image[1::2, 0::2] +
                            test_image[0::2, 1::2] +
                            test_image[1::2, 1::2]
                        ) / 4.0
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoSubSampling().cy_subsample_grey_float(test_image)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'SubSampling' + self._testMethodName)

        # test case check
        pass

class TestSubSamplingRGB(unittest.TestCase):
    """
    Unit test utility for subsampling gray image
    """

    def test_image_rgb_small_uint8(self):
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = (
                            test_image[0::2, 0::2] +
                            test_image[1::2, 0::2] +
                            test_image[0::2, 1::2] +
                            test_image[1::2, 1::2]
                        ) / 4.0
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoSubSampling().cy_subsample_rgb_uint8(test_image)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'SubSampling' + self._testMethodName)

        # test case check
        pass

    def test_image_rgb_large_uint8(self):
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = (
                            test_image[0::2, 0::2] +
                            test_image[1::2, 0::2] +
                            test_image[0::2, 1::2] +
                            test_image[1::2, 1::2]
                        ) / 4.0
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoSubSampling().cy_subsample_rgb_uint8(test_image)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'SubSampling' + self._testMethodName)

        # test case check
        pass

    def test_image_rgb_small_float(self):
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = (
                            test_image[0::2, 0::2] +
                            test_image[1::2, 0::2] +
                            test_image[0::2, 1::2] +
                            test_image[1::2, 1::2]
                        ) / 4.0
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoSubSampling().cy_subsample_rgb_float(test_image)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'SubSampling' + self._testMethodName)

        # test case check
        pass

    def test_image_rgb_large_float(self):
        test_image = get_image(self._testMethodName)

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest1 = (
                            test_image[0::2, 0::2] +
                            test_image[1::2, 0::2] +
                            test_image[0::2, 1::2] +
                            test_image[1::2, 1::2]
                        ) / 4.0
        end = time.time()
        t1 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            img_dest2 = AlgoSubSampling().cy_subsample_rgb_float(test_image)
        end = time.time()
        t2 = end - start

        start = time.time()
        for i in range(_ITER_NUM):
            # img_dest3 = ndimage.rotate(test_image, theta, order=1)
            pass
        end = time.time()
        t3 = end - start
        t3 = 'Not available ....'

        print_utility(t1, t2, t3, '', self._testMethodName)

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, 'SubSampling' + self._testMethodName)

        # test case check
        pass

def rotateGrayNN():
    # run suite
    global _result_timings
    custom_stream.write('\n----------------------------------------------------------------------\n')
    custom_stream.write('\n       *** Rotation of gray images (with NN-interpolation) ***        \n')
    custom_stream.write('\n----------------------------------------------------------------------\n\n')
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRotateNNGray)
    unittest.TextTestRunner(verbosity=3, stream=custom_stream).run(suite)
    od = collections.OrderedDict(sorted(_result_timings.items()))
    plt.clf()
    df2 = pd.DataFrame.from_dict(od, orient='columns')
    df2.plot.bar()
    plt.ylabel('Time taken')
    plt.xlabel('Numpy vs Cython')
    plt.title('Rotation of gray images (with NN-interpolation)')
    plt.savefig('Report\\RotateGrayNN.png')
    plt.close()
    _result_timings = {}


def rotateRGBNN():
    # run suite
    global _result_timings
    custom_stream.write('\n----------------------------------------------------------------------\n')
    custom_stream.write('\n       *** Rotation of RGB images (with NN-interpolation) ***         \n')
    custom_stream.write('\n----------------------------------------------------------------------\n\n')
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRotateNNRGB)
    unittest.TextTestRunner(verbosity=3, stream=custom_stream).run(suite)
    od = collections.OrderedDict(sorted(_result_timings.items()))
    plt.clf()
    df2 = pd.DataFrame.from_dict(od, orient='columns')
    df2.plot.bar()
    plt.ylabel('Time taken')
    plt.xlabel('Numpy vs Cython')
    plt.title('Rotation of RGB images (with NN-interpolation)')
    plt.savefig('Report\\RotateRGBNN.png')
    plt.close()
    _result_timings = {}


def rotateGrayLin():
    # run suite
    global _result_timings
    custom_stream.write('\n----------------------------------------------------------------------\n')
    custom_stream.write('\n     *** Rotation of gray images (with Linear-interpolation) ***      \n')
    custom_stream.write('\n----------------------------------------------------------------------\n\n')
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRotateLinGray)
    unittest.TextTestRunner(verbosity=3, stream=custom_stream).run(suite)
    od = collections.OrderedDict(sorted(_result_timings.items()))
    plt.clf()
    df2 = pd.DataFrame.from_dict(od, orient='columns')
    df2.plot.bar()
    plt.ylabel('Time taken')
    plt.xlabel('Numpy vs Cython')
    plt.title('Rotation of gray images (with Linear-interpolation)')
    plt.savefig('Report\\RotateGrayLin.png')
    plt.close()
    _result_timings = {}


def rotateRGBLin():
    # run suite
    global _result_timings
    custom_stream.write('\n----------------------------------------------------------------------\n')
    custom_stream.write('\n     *** Rotation of RGB images (with Linear-interpolation) ***       \n')
    custom_stream.write('\n----------------------------------------------------------------------\n\n')
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRotateLinRGB)
    unittest.TextTestRunner(verbosity=3, stream=custom_stream).run(suite)
    od = collections.OrderedDict(sorted(_result_timings.items()))
    plt.clf()
    df2 = pd.DataFrame.from_dict(od, orient='columns')
    df2.plot.bar()
    plt.ylabel('Time taken')
    plt.xlabel('Numpy vs Cython')
    plt.title('Rotation of RGB images (with Linear-interpolation)')
    plt.savefig('Report\\RotateRGBLin.png')
    plt.close()
    _result_timings = {}


def subsamplingGray():
    # run suite
    global _result_timings
    custom_stream.write('\n----------------------------------------------------------------------\n')
    custom_stream.write('\n                 *** Sub sampling of gray images  ***                 \n')
    custom_stream.write('\n----------------------------------------------------------------------\n\n')
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSubSamplingGray)
    unittest.TextTestRunner(verbosity=3, stream=custom_stream).run(suite)
    od = collections.OrderedDict(sorted(_result_timings.items()))
    plt.clf()
    df2 = pd.DataFrame.from_dict(od, orient='columns')
    df2.plot.bar()
    plt.ylabel('Time taken')
    plt.xlabel('Numpy vs Cython')
    plt.title('Sub sampling of gray images')
    plt.savefig('Report\\SubsamplingGray.png')
    plt.close()
    _result_timings = {}


def subsamplingRGB():
    # run suite
    global _result_timings
    custom_stream.write('\n----------------------------------------------------------------------\n')
    custom_stream.write('\n                 *** Sub sampling of RGB images  ***                 \n')
    custom_stream.write('\n----------------------------------------------------------------------\n\n')
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSubSamplingRGB)
    unittest.TextTestRunner(verbosity=3, stream=custom_stream).run(suite)
    od = collections.OrderedDict(sorted(_result_timings.items()))
    plt.clf()
    df2 = pd.DataFrame.from_dict(od, orient='columns')
    df2.plot.bar()
    plt.ylabel('Time taken')
    plt.xlabel('Numpy vs Cython')
    plt.title('Sub sampling of RGB images')
    plt.savefig('Report\\SubsamplingRGB.png')
    plt.close()
    _result_timings = {}


if __name__ == '__main__':
    rotateGrayNN()
    rotateRGBNN()
    rotateGrayLin()
    rotateRGBLin()
    subsamplingGray()
    subsamplingRGB()




