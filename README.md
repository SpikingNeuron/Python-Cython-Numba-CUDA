

# Python Cython CUDA

## About Project

### Algorithms implemented

This project has implementation for below algorithms:
+ Blending of two images
+ Rotation without interpolation
+ Rotation with linear interpolation
+ Subsampling by factor of 2

Every algorithm is tested with unittest for three different categories:
+ Two image sizes (large and small)
+ Two data types (uint8 and float)
+ Two color types (gray and RGB)

So basically there are eight test cases per algorithm.

### Three different implementations with numpy, cython and pycuda

We have three implementation of these algorithms for benchmarking:
+ Python Numpy library
+ Cython
+ CUDA with Cython (Not available. Work needs to be done to write compiler wrapper for nvcc, to be called from python.
The `setup.py` file will be updated soon...)

:construction: :warning: CUDA part still needs nvcc compiler patch (Only Numpy and Cython will work) :warning: :construction:

### Additional contents

Once you run `benchmark.py` the cython code will be compiled and for every `*
.pyx` a `*.html` file will be generated. This `*.html` file will have the info 
about  how  well the pycthon code is converted to C.

## Environment
+ Windows 10 
+ CUDA 8.0
+ PyCuda 2016.1.2


## Dependencies
```sh
conda install numpy
conda install cython
conda install pandas
pip install pycuda
```

## Running the code
+ To run the code just type. Compilation of c code for cython will be done by the pyximport utility when you run the
`benchmark.py` file
```py
git clone https://github.com/praveenneuron/Python-Cython-CUDA.git
python benchmark.py
```
+ The unittest framework will run and dump the results in report folder
  + Images with test_image in their name are the output of the algorithm
  + Other images have barplots that will compare numpy and cython implementation

## Benchmark results
Below are the results (Smaller the better as time taken is the y axis)

The code was run for 100 iterations you can adjust it with `_ITER_NUM` variable.

+ **Blending of two images**

|                                                                                   |                                                                                 |
|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
|![Not available check Report folder](Report/BlendingGray.png?raw=true "Gray image")|![Not available check Report folder](Report/BlendingRGB.png?raw=true "RGB image")|


+ **Rotate with no interpolation**

|                                                                                   |                                                                                 |
|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
|![Not available check Report folder](Report/RotateGrayNN.png?raw=true "Gray image")|![Not available check Report folder](Report/RotateRGBNN.png?raw=true "RGB image")|


+ **Rotate with linear interpolation**

|                                                                                    |                                                                                  |
|------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
|![Not available check Report folder](Report/RotateGrayLin.png?raw=true "Gray image")|![Not available check Report folder](Report/RotateRGBLin.png?raw=true "RGB image")|


+ **Subsampling by factor of 2**

|                                                                                      |                                                                                    |
|--------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
|![Not available check Report folder](Report/SubsamplingGray.png?raw=true "Gray image")|![Not available check Report folder](Report/SubsamplingRGB.png?raw=true "RGB image")|

## Output of algorithms

+ **Blending of two images**

![Not available check Report folder](Report/Blendingtest_image_grey_large_uint8.png?raw=true "Gray image")

+ **Rotate with no interpolation**

![Not available check Report folder](Report/RotateNNtest_image_rgb_small_uint8.png?raw=true "Gray image")

+ **Rotate with linear interpolation**

![Not available check Report folder](Report/RotateLintest_image_grey_small_uint8.png?raw=true "Gray image")

+ **Subsampling by factor of 2**

![Not available check Report folder](Report/SubSamplingtest_image_grey_small_float.png?raw=true "Gray image")

## Unittest log

```txt
I:\Anaconda3\python.exe D:/Github/Python-Cython-CUDA/benchmark.py

----------------------------------------------------------------------

                   *** Blending of gray images  ***                   

----------------------------------------------------------------------

test_image_grey_large_float (__main__.TestBlendingGray) ... 
Time taken Numpy 	: 0.27251648902893066
Time taken Cython	: 0.08499431610107422
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_large_uint8 (__main__.TestBlendingGray) ... 
Time taken Numpy 	: 0.13849925994873047
Time taken Cython	: 0.02300095558166504
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_float (__main__.TestBlendingGray) ... 
Time taken Numpy 	: 0.043999671936035156
Time taken Cython	: 0.014496088027954102
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_uint8 (__main__.TestBlendingGray) ... 
Time taken Numpy 	: 0.03500032424926758
Time taken Cython	: 0.005500078201293945
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 15.218s

OK

----------------------------------------------------------------------

                   *** Blending of RGB images  ***                    

----------------------------------------------------------------------

test_image_rgb_large_float (__main__.TestBlendingRGB) ... 
Time taken Numpy 	: 0.5374839305877686
Time taken Cython	: 0.2605001926422119
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_large_uint8 (__main__.TestBlendingRGB) ... 
Time taken Numpy 	: 0.4310019016265869
Time taken Cython	: 0.06849908828735352
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_float (__main__.TestBlendingRGB) ... 
Time taken Numpy 	: 0.13401365280151367
Time taken Cython	: 0.04648733139038086
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_uint8 (__main__.TestBlendingRGB) ... 
Time taken Numpy 	: 0.10849499702453613
Time taken Cython	: 0.017495393753051758
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 14.844s

OK

----------------------------------------------------------------------

       *** Rotation of gray images (with NN-interpolation) ***        

----------------------------------------------------------------------

test_image_grey_large_float (__main__.TestRotateNNGray) ... 
Time taken Numpy 	: 1.5130414962768555
Time taken Cython	: 0.33399128913879395
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_large_uint8 (__main__.TestRotateNNGray) ... 
Time taken Numpy 	: 1.301123857498169
Time taken Cython	: 0.16699862480163574
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_float (__main__.TestRotateNNGray) ... 
Time taken Numpy 	: 0.3525111675262451
Time taken Cython	: 0.07950067520141602
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_uint8 (__main__.TestRotateNNGray) ... 
Time taken Numpy 	: 0.29999709129333496
Time taken Cython	: 0.03302001953125
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 24.842s

OK

----------------------------------------------------------------------

       *** Rotation of RGB images (with NN-interpolation) ***         

----------------------------------------------------------------------

test_image_rgb_large_float (__main__.TestRotateNNRGB) ... 
Time taken Numpy 	: 5.865823984146118
Time taken Cython	: 0.5754995346069336
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_large_uint8 (__main__.TestRotateNNRGB) ... 
Time taken Numpy 	: 4.095231533050537
Time taken Cython	: 0.25600290298461914
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_float (__main__.TestRotateNNRGB) ... 
Time taken Numpy 	: 1.4641034603118896
Time taken Cython	: 0.17099690437316895
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_uint8 (__main__.TestRotateNNRGB) ... 
Time taken Numpy 	: 0.9620048999786377
Time taken Cython	: 0.0559847354888916
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 30.920s

OK

----------------------------------------------------------------------

     *** Rotation of gray images (with Linear-interpolation) ***      

----------------------------------------------------------------------

test_image_grey_large_float (__main__.TestRotateLinGray) ... 
Time taken Numpy 	: 1.8127481937408447
Time taken Cython	: 0.5325160026550293
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_large_uint8 (__main__.TestRotateLinGray) ... 
Time taken Numpy 	: 1.5821540355682373
Time taken Cython	: 0.35249972343444824
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_float (__main__.TestRotateLinGray) ... 
Time taken Numpy 	: 0.4364912509918213
Time taken Cython	: 0.10949945449829102
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_uint8 (__main__.TestRotateLinGray) ... 
Time taken Numpy 	: 0.3880960941314697
Time taken Cython	: 0.06500482559204102
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 26.101s

OK

----------------------------------------------------------------------

     *** Rotation of RGB images (with Linear-interpolation) ***       

----------------------------------------------------------------------

test_image_rgb_large_float (__main__.TestRotateLinRGB) ... 
Time taken Numpy 	: 6.48515772819519
Time taken Cython	: 1.5131480693817139
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_large_uint8 (__main__.TestRotateLinRGB) ... 
Time taken Numpy 	: 4.950857877731323
Time taken Cython	: 0.8226222991943359
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_float (__main__.TestRotateLinRGB) ... 
Time taken Numpy 	: 1.583146095275879
Time taken Cython	: 0.33049964904785156
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_uint8 (__main__.TestRotateLinRGB) ... 
Time taken Numpy 	: 1.3056743144989014
Time taken Cython	: 0.18750357627868652
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 34.996s

OK

----------------------------------------------------------------------

                 *** Sub sampling of gray images  ***                 

----------------------------------------------------------------------

test_image_grey_large_float (__main__.TestSubSamplingGray) ... 
Time taken Numpy 	: 0.0879828929901123
Time taken Cython	: 0.021017074584960938
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_large_uint8 (__main__.TestSubSamplingGray) ... 
Time taken Numpy 	: 0.0345001220703125
Time taken Cython	: 0.007493734359741211
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_float (__main__.TestSubSamplingGray) ... 
Time taken Numpy 	: 0.020996809005737305
Time taken Cython	: 0.004999637603759766
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_uint8 (__main__.TestSubSamplingGray) ... 
Time taken Numpy 	: 0.008991241455078125
Time taken Cython	: 0.001497030258178711
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 7.690s

OK

----------------------------------------------------------------------

                 *** Sub sampling of RGB images  ***                 

----------------------------------------------------------------------

test_image_rgb_large_float (__main__.TestSubSamplingRGB) ... 
Time taken Numpy 	: 0.2674999237060547
Time taken Cython	: 0.06250166893005371
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_large_uint8 (__main__.TestSubSamplingRGB) ... 
Time taken Numpy 	: 0.1490623950958252
Time taken Cython	: 0.029485464096069336
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_float (__main__.TestSubSamplingRGB) ... 
Time taken Numpy 	: 0.06699943542480469
Time taken Cython	: 0.01549673080444336
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_uint8 (__main__.TestSubSamplingRGB) ... 
Time taken Numpy 	: 0.03750109672546387
Time taken Cython	: 0.006997346878051758
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 7.633s

OK

Process finished with exit code 0
```
