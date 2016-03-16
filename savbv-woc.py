import numpy as np
from scipy import ndimage

import timeit as ti


small_shape = (2048, 2048)
large_shape = (4096, 4096)

test_configs_grey = ("Testcase 1: grey, small shape, pixeltype: uint8",
                     "Testcase 2: grey, large shape, pixeltype: uint8", 
                     "Testcase 3: grey, small shape, pixeltype: float",
                     "Testcase 4: grey, large shape, pixeltype: float")
test_configs_rgb = ("Testcase 5: RGB,  small shape, pixeltype: (uint8,uint8,uint8)",
                    "Testcase 6: RGB,  large shape, pixeltype: (uint8,uint8,uint8)", 
                    "Testcase 7: RGB,  small shape, pixeltype: (float,float,float)",
                    "Testcase 8: RGB,  large shape, pixeltype: (float,float,float)")

             
setup_string = """
import numpy as np
from scipy import ndimage

small_shape_grey = (2048, 2048)
large_shape_grey = (4096, 4096)

test_images_grey = ((



, np.random.random_integers(0,255, small_shape_grey).astype(np.uint8)),
                    (np.random.random_integers(0,255, large_shape_grey).astype(np.uint8), np.random.random_integers(0,255, large_shape_grey).astype(np.uint8)),
                    (np.random.random_integers(0,255, small_shape_grey).astype(np.float), np.random.random_integers(0,255, small_shape_grey).astype(np.float)),
                    (np.random.random_integers(0,255, large_shape_grey).astype(np.float), np.random.random_integers(0,255, large_shape_grey).astype(np.float)))
                    
thresh_grey = 128

small_shape_rgb = (2048, 2048, 3)
large_shape_rgb = (4096, 4096, 3)

test_images_rgb = ((np.random.random_integers(0,255, small_shape_rgb).astype(np.uint8), np.random.random_integers(0,255, small_shape_rgb).astype(np.uint8)),
                   (np.random.random_integers(0,255, large_shape_rgb).astype(np.uint8), np.random.random_integers(0,255, large_shape_rgb).astype(np.uint8)),
                   (np.random.random_integers(0,255, small_shape_rgb).astype(np.float), np.random.random_integers(0,255, small_shape_rgb).astype(np.float)),
                   (np.random.random_integers(0,255, large_shape_rgb).astype(np.float), np.random.random_integers(0,255, large_shape_rgb).astype(np.float)))
 
thresh_rgb = np.array([128, 128, 128])

def gaussianKernel(size, truncate=3.0):
    temp = np.zeros((size,size))
    temp[size/2,size/2] = 1
    return ndimage.gaussian_filter(temp,(size-1)/(2.0*truncate),truncate=truncate)

mask5x5   = gaussianKernel(5)
mask11x11 = gaussianKernel(11)
                   
theta = 33.7
"""

number_of_executions = 100

tests_grey = (("01",  
          "Summation of arrays", 
          "img_dest = np.clip((test_images_grey[{i}][0] + test_images_grey[{i}][1])/2, 0, 255)"), 
         ("02", 
          "Thresholding of images", 
          "img_dest = test_images_grey[{i}][0] > thresh_grey"), 
         ("03", 
          "Histogram of images", 
          "img_hist = np.histogram(test_images_grey[{i}][0], bins=256, range=(0,255))"), 
         ("04a", 
          "2d-convolution of images with gaussian mask (size: 5x5)", 
          "img_dest = ndimage.convolve(test_images_grey[{i}][0],mask5x5)"), 
         ("04b", 
          "separable convolution of images with gaussian masks (sizes: (1x5) & (5x1))", 
          "img_dest = ndimage.gaussian_filter(test_images_grey[{i}][0],(5-1)/(2.0*3.0),truncate=3.0)"), 
         ("05a", 
          "2d-convolution of images with gaussian mask (size: 11x11)", 
          "img_dest = ndimage.convolve(test_images_grey[{i}][0],mask11x11)"), 
         ("05b", 
          "separable convolution of images with gaussian masks (sizes: (1x11) & (11x1))", 
          "img_dest = ndimage.gaussian_filter(test_images_grey[{i}][0],(11-1)/(2.0*3.0),truncate=3.0)"), 
         ("06", 
          "Anisotropic median filter (size: 5x5)", 
          "img_dest = ndimage.median_filter(test_images_grey[{i}][0],5)"), 
         ("07", 
          "Anisotropic median filter (size: 11x11)", 
          "img_dest = ndimage.median_filter(test_images_grey[{i}][0],11)"), 
         ("08", 
          "Subsampling of images", 
          "img_dest = (test_images_grey[{i}][0][0::2,0::2]+test_images_grey[{i}][0][1::2,::2]+test_images_grey[{i}][0][0::2,1::2]+test_images_grey[{i}][0][1::2,1::2])/4.0"), 
         ("09", 
          "Rotation of images (with NN-interpolation)", 
          "img_dest = ndimage.rotate(test_images_grey[{i}][0],theta, order=0)"), 
         ("10", 
          "Rotation of images (with linear interpolation)", 
          "img_dest = ndimage.rotate(test_images_grey[{i}][0],theta, order=1)"))    

tests_rgb = (("01",  
          "Summation of arrays", 
          "img_dest = np.clip((test_images_rgb[{i}][0] + test_images_rgb[{i}][1])/2, 0, 255)"), 
         ("02", 
          "Thresholding of images", 
          "img_dest = np.all(test_images_rgb[{i}][0] > thresh_rgb,-1) "), 
         ("03", 
          "Histogram of images", 
          "img_hist =  np.zeros((256,3))\n"
          "for c in range(3):\n"
          "    img_hist[...,c],foo = np.histogram((test_images_rgb[{i}][0])[...,c], bins=256, range=(0,255))"), 
         ("04a", 
          "2d-convolution of images with gaussian mask (size: 5x5)",           
          "img_dest =  np.zeros_like(test_images_rgb[{i}][0])\n"
          "for c in range(3):\n"
          "    img_dest[...,c] = ndimage.convolve(test_images_rgb[{i}][0][...,c], mask5x5)"), 
         ("04b", 
          "separable convolution of images with gaussian masks (sizes: (1x5) & (5x1))", 
          "img_dest =  np.zeros_like(test_images_rgb[{i}][0])\n"
          "for c in range(3):\n"
          "    img_dest[...,c] = ndimage.gaussian_filter(test_images_rgb[{i}][0][...,c],(5-1)/(2.0*3.0),truncate=3.0)"), 
         ("05a", 
          "2d-convolution of images with gaussian mask (size: 11x11)", 
          "img_dest =  np.zeros_like(test_images_rgb[{i}][0])\n"
          "for c in range(3):\n"
          "    img_dest[...,c] = ndimage.convolve(test_images_rgb[{i}][0][...,c],mask11x11)"), 
         ("05b", 
          "separable convolution of images with gaussian masks (sizes: (1x11) & (11x1))", 
          "img_dest =  np.zeros_like(test_images_rgb[{i}][0])\n"
          "for c in range(3):\n"
          "    img_dest[...,c] = ndimage.gaussian_filter(test_images_rgb[{i}][0][...,c],(11-1)/(2.0*3.0),truncate=3.0)"), 
         ("06", 
          "Anisotropic median filter (size: 5x5)", 
          "img_dest =  np.zeros_like(test_images_rgb[{i}][0])\n"
          "for c in range(3):\n"
          "    img_dest[...,c] =  ndimage.median_filter(test_images_rgb[{i}][0][...,c],5)"), 
         ("07", 
          "Anisotropic median filter (size: 11x11)", 
          "img_dest =  np.zeros_like(test_images_rgb[{i}][0])\n"
          "for c in range(3):\n"
          "    img_dest[...,c] = ndimage.median_filter(test_images_rgb[{i}][0][...,c],11)"), 
         ("08", 
          "Subsampling of images", 
          "img_dest = (test_images_rgb[{i}][0][0::2,0::2]+test_images_rgb[{i}][0][1::2,::2]+test_images_rgb[{i}][0][0::2,1::2]+test_images_rgb[{i}][0][1::2,1::2])/4.0"), 
         ("09", 
          "Rotation of images (with NN-interpolation)", 
          "img_dest = ndimage.rotate(test_images_rgb[{i}][0],theta, order=0)"), 
         ("10", 
          "Rotation of images (with linear interpolation)", 
          "img_dest = ndimage.rotate(test_images_rgb[{i}][0],theta, order=1)"))                 
                 

for test in tests_grey:
    print "Running grey test ", test[0], "-", test[1], ":"
    for i in range(4):
        print "   ", test_configs_grey[i], "time :", \
              ti.timeit(test[2].format(i=i), setup=setup_string, number=number_of_executions), \
              "seconds"
              
for test in tests_rgb:
    print "Running RGB test ", test[0], "-", test[1], ":"
    for i in range(4):
        print "   ", test_configs_rgb[i], "time :", \
              ti.timeit(test[2].format(i=i), setup=setup_string, number=number_of_executions), \
              "seconds"