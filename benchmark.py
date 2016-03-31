"""

"""

import numpy as np
import unittest
import sys
import time
from scipy import ndimage, misc
import matplotlib.pyplot as plt

# for compiling the cython code
import pyximport
pyximport.install(inplace=False,
                  setup_args={"include_dirs": np.get_include()},
                  reload_support=True)

# Generate c and html file that shows respective c code
from Cython.Build import cythonize

# TODO: check with multi thread
cythonize('Algo_RotateNN.pyx', annotate=True, nthreads=8)
cythonize('Algo_RotateLinear.pyx', annotate=True, nthreads=8)
cythonize('Algo_SubSampling.pyx', annotate=True, nthreads=8)

from Algo_RotateNN import AlgoRotateNN

# To print on console or to file
custom_stream = sys.stdout
# custom_stream = open("report.txt", "w", encoding="utf-8")

# config vars
_PLOT = False
_ITER_NUM = 1


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

    if not _PLOT:
        return

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

    plt.show()


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
        #lena = misc.imresize(lena, (100, 50), interp='bilinear')
        lena = misc.imresize(lena, (500, 500), interp='bilinear')
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


def print_utility(time_taken_np, time_taken_cy, time_taken_cuda, message):
    """
    print utility
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
    pass


class TestRotateNNGray(unittest.TestCase):
    """
    Unit test utility for rotating
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

        print_utility(t1, t2, t3, '')

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, self._testMethodName)

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

        print_utility(t1, t2, t3, '')

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, self._testMethodName)

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

        print_utility(t1, t2, t3, '')

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, self._testMethodName)

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

        print_utility(t1, t2, t3, '')

        if _PLOT:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, self._testMethodName)

        # test case check
        pass


class TestRotateNNRGB(unittest.TestCase):
    """
    Unit test utility for rotating
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

        print_utility(t1, t2, t3, '')

        if True:
            plot_images(test_image, img_dest1, img_dest2, img_dest2, self._testMethodName)

        # test case check
        pass


if __name__ == '__main__':

    # run suite
    custom_stream.write('\n----------------------------------------------------------------------\n')
    custom_stream.write('\n       *** Rotation of gray images (with NN-interpolation) ***        \n')
    custom_stream.write('\n----------------------------------------------------------------------\n\n')
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRotateNNGray)
    unittest.TextTestRunner(verbosity=3, stream=custom_stream).run(suite)
    custom_stream.write('\n----------------------------------------------------------------------\n')
    custom_stream.write('\n       *** Rotation of RGB images (with NN-interpolation) ***         \n')
    custom_stream.write('\n----------------------------------------------------------------------\n\n')
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRotateNNRGB)
    unittest.TextTestRunner(verbosity=3, stream=custom_stream).run(suite)

