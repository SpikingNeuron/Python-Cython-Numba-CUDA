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
pyximport.install(inplace=True,
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

# variables that store size of images
lena_rgb_uint8 = ndimage.imread('lena.jpg')
lena_rgb_float32 = lena_rgb_uint8.astype(np.float32)
lena_grey_uint8 = ndimage.imread('lena.jpg', mode='L')
lena_grey_float32 = lena_grey_uint8.astype(np.float32)


small_shape_grey = (2048, 2048)
large_shape_grey = (4096, 4096)
small_shape_rgb = (2048, 2048, 3)
large_shape_rgb = (4096, 4096, 3)

misc.imresize(lena_rgb_float32, small_shape_grey, interp='bilinear')


#plot image
def plot_images(im0, im1, im2, im3, title):
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

# get image utility
def get_image(query):
    args = query.split(sep='_')
    lena = None

    if args[2] == 'grey':
        lena = ndimage.imread('lena.jpg', mode='L')
    elif args[2] == 'rgb':
        lena = ndimage.imread('lena.jpg')
    else:
        raise ValueError('Invalid color type. Allowed rgb or grey')

    if args[3] == 'small':
        lena = misc.imresize(lena, (2048, 2048), interp='bilinear')
    elif args[3] == 'large':
        lena = misc.imresize(lena, (4096, 4096), interp='bilinear')
    else:
        raise ValueError('Invalid size. Allowed small or large')

    if args[4] == 'uint8':
        lena = lena
    elif args[4] == 'float32':
        lena = lena.astype(np.float32)
    else:
        raise ValueError('Invalid size. Allowed uint8 or float32')

    return lena


# print utility
def print_utility(time_taken_np, time_taken_cy, time_taken_cuda, message):
    global custom_stream
    # custom_stream.write('\n' + message)
    custom_stream.write('\nTime taken Numpy : ' + str(time_taken_np))
    custom_stream.write('\nTime taken Cython: ' + str(time_taken_cy))
    custom_stream.write('\nTime taken CUDA  : ' + str(time_taken_cuda))
    custom_stream.write('\n...........')
    pass


class TestRotateNN(unittest.TestCase):

    def test_image_grey_small_uint8(self):
        theta = 33.7
        test_image = get_image(self._testMethodName)

        start = time.time()
        img_dest1 = ndimage.rotate(test_image, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        #img_dest2 = AlgoRotateNN().cy_rotate_degree_0(test_image, theta)
        img_dest2 = ndimage.rotate(test_image, theta, order=0)
        end = time.time()
        t2 = end - start

        start = time.time()
        img_dest3 = ndimage.rotate(test_image, theta, order=0)
        end = time.time()
        t3 = end - start

        print_utility(t1, t2, t3, '')

        plot_images(test_image, img_dest1, img_dest2, img_dest3, self._testMethodName)

        # test case check
        pass


if __name__ == '__main__':

    # run suite
    custom_stream.write('\n----------------------------------------------------------------------\n')
    custom_stream.write('\n          *** Rotation of images (with NN-interpolation) *** \n')
    custom_stream.write('\n----------------------------------------------------------------------\n\n')
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRotateNN)
    unittest.TextTestRunner(verbosity=3, stream=custom_stream).run(suite)
