"""

"""

import numpy as np
import unittest
import sys
import time

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

# stream to hold log
global custom_stream

# variable to store ground truth
global ground_truth


# print utility
def print_utility(time_taken):
    global custom_stream
    custom_stream.write('\nTime taken: ')
    custom_stream.write(str(time_taken))
    custom_stream.write('\n...........')
    pass


class TestRotateNN(unittest.TestCase):

    def test_rotate_linear_interpolation(self):
        start = time.time()
        end = time.time()
        print_utility(end - start)
        pass

    def test_rotate_linear_interpolation2(self):
        start = time.time()
        end = time.time()
        print_utility(end - start)
        pass

if __name__ == '__main__':
    global custom_stream

    # To print on console or to file
    custom_stream = sys.stdout
    # custom_stream = open("report.txt", "w", encoding="utf-8")

    # run suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRotateNN)
    unittest.TextTestRunner(verbosity=3, stream=custom_stream).run(suite)
