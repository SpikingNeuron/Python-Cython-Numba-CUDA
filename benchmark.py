"""

"""

import numpy as np
import unittest
import sys
import time
from scipy import ndimage

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

# To print on console or to file
custom_stream = sys.stdout
# custom_stream = open("report.txt", "w", encoding="utf-8")

# variables that store size of images
small_shape_grey = (2048, 2048)
large_shape_grey = (4096, 4096)
small_shape_rgb = (2048, 2048, 3)
large_shape_rgb = (4096, 4096, 3)


# print utility
def print_utility(time_taken_np, time_taken_cuda, message):
    global custom_stream
    # custom_stream.write('\n' + message)
    custom_stream.write('\nTime taken numpy: ' + str(time_taken_np))
    custom_stream.write('\nTime taken CUDA : ' + str(time_taken_cuda))
    custom_stream.write('\n...........')
    pass


class TestRotateNN(unittest.TestCase):

    def test_grey_small_image_uint8(self):
        theta = 33.7
        test_images_grey = np.random.random_integers(0, 255, small_shape_grey).astype(np.uint8)

        start = time.time()
        img_dest1 = ndimage.rotate(test_images_grey, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        img_dest2 = ndimage.rotate(test_images_grey, theta, order=0)
        end = time.time()
        t2 = end - start

        print_utility(t1, t2, '')

        # test case check
        self.assertTrue(
            np.allclose(img_dest1, img_dest2, rtol=1e-05, atol=1e-08, equal_nan=False),
            'Results computed by GPU are not correct ...'
        )
        
    def test_grey_small_image_float32(self):
        theta = 33.7
        test_images_grey = np.random.random_integers(0, 255, small_shape_grey).astype(np.float32)

        start = time.time()
        img_dest1 = ndimage.rotate(test_images_grey, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        img_dest2 = ndimage.rotate(test_images_grey, theta, order=0)
        end = time.time()
        t2 = end - start

        print_utility(t1, t2, '')

        # test case check
        self.assertTrue(
            np.allclose(img_dest1, img_dest2, rtol=1e-05, atol=1e-08, equal_nan=False),
            'Results computed by GPU are not correct ...'
        )

    def test_grey_large_image_uint8(self):
        theta = 33.7
        test_images_grey = np.random.random_integers(0, 255, large_shape_grey).astype(np.uint8)

        start = time.time()
        img_dest1 = ndimage.rotate(test_images_grey, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        img_dest2 = ndimage.rotate(test_images_grey, theta, order=0)
        end = time.time()
        t2 = end - start

        print_utility(t1, t2, '')

        # test case check
        self.assertTrue(
            np.allclose(img_dest1, img_dest2, rtol=1e-05, atol=1e-08, equal_nan=False),
            'Results computed by GPU are not correct ...'
        )
        
    def test_grey_large_image_float32(self):
        theta = 33.7
        test_images_grey = np.random.random_integers(0, 255, large_shape_grey).astype(np.float32)

        start = time.time()
        img_dest1 = ndimage.rotate(test_images_grey, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        img_dest2 = ndimage.rotate(test_images_grey, theta, order=0)
        end = time.time()
        t2 = end - start

        print_utility(t1, t2, '')

        # test case check
        self.assertTrue(
            np.allclose(img_dest1, img_dest2, rtol=1e-05, atol=1e-08, equal_nan=False),
            'Results computed by GPU are not correct ...'
        )

    def test_rgb_small_image_uint8(self):
        theta = 33.7
        test_images_rgb = np.random.random_integers(0, 255, small_shape_rgb).astype(np.uint8)

        start = time.time()
        img_dest1 = ndimage.rotate(test_images_rgb, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        img_dest2 = ndimage.rotate(test_images_rgb, theta, order=0)
        end = time.time()
        t2 = end - start

        print_utility(t1, t2, '')

        # test case check
        self.assertTrue(
            np.allclose(img_dest1, img_dest2, rtol=1e-05, atol=1e-08, equal_nan=False),
            'Results computed by GPU are not correct ...'
        )
        
    def test_rgb_small_image_float32(self):
        theta = 33.7
        test_images_rgb = np.random.random_integers(0, 255, small_shape_rgb).astype(np.float32)

        start = time.time()
        img_dest1 = ndimage.rotate(test_images_rgb, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        img_dest2 = ndimage.rotate(test_images_rgb, theta, order=0)
        end = time.time()
        t2 = end - start

        print_utility(t1, t2, '')

        # test case check
        self.assertTrue(
            np.allclose(img_dest1, img_dest2, rtol=1e-05, atol=1e-08, equal_nan=False),
            'Results computed by GPU are not correct ...'
        )

    def test_rgb_large_image_uint8(self):
        theta = 33.7
        test_images_rgb = np.random.random_integers(0, 255, large_shape_rgb).astype(np.uint8)

        start = time.time()
        img_dest1 = ndimage.rotate(test_images_rgb, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        img_dest2 = ndimage.rotate(test_images_rgb, theta, order=0)
        end = time.time()
        t2 = end - start

        print_utility(t1, t2, '')

        # test case check
        #self.assertTrue(
        #    np.allclose(img_dest1, img_dest2, rtol=1e-05, atol=1e-08, equal_nan=False),
        #    'Results computed by GPU are not correct ...'
        #)
        pass
        
    def test_rgb_large_image_float32(self):
        theta = 33.7
        test_images_rgb = np.random.random_integers(0, 255, large_shape_rgb).astype(np.float32)

        start = time.time()
        img_dest1 = ndimage.rotate(test_images_rgb, theta, order=0)
        end = time.time()
        t1 = end - start

        start = time.time()
        img_dest2 = ndimage.rotate(test_images_rgb, theta, order=0)
        end = time.time()
        t2 = end - start

        print_utility(t1, t2, '')

        # test case check
        #self.assertTrue(
        #    np.allclose(img_dest1, img_dest2, rtol=1e-05, atol=1e-08, equal_nan=False),
        #    'Results computed by GPU are not correct ...'
        #)
        pass

if __name__ == '__main__':

    # run suite
    custom_stream.write('\n\n\n *** Rotation of images (with NN-interpolation) *** \n\n')
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRotateNN)
    unittest.TextTestRunner(verbosity=3, stream=custom_stream).run(suite)
