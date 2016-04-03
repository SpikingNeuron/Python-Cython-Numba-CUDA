

# Python Cython CUDA

## About Project

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

We have three implementation of these algorithms for benchmarking:
+ Python Numpy library
+ Cython
+ CUDA with Cython (Not available. Work needs to be done to write compiler wrapper for nvcc, to be called from python.
The `setup.py` file will be updated soon...)

:construction: :warning: CUDA part still needs nvcc compiler patch (Only Numpy and Cython will work) :warning: :construction:

## Environment
+ Windows 10 (or Ubuntu 14.04)
+ Python 3.4.3
+ Cython 0.23.4
+ CUDA 7.5
+ Visual Studio 2015


## Dependencies
```sh
conda install python=3.5
conda install numpy
conda install cython
conda install pandas
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

----------------------------------------------------------------------

                   *** Blending of gray images  ***

----------------------------------------------------------------------

test_image_grey_large_float (__main__.TestBlendingGray) ...
Time taken Numpy 	: 22.71826696395874
Time taken Cython	: 8.578590869903564
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_large_uint8 (__main__.TestBlendingGray) ...
Time taken Numpy 	: 17.102105617523193
Time taken Cython	: 2.519599199295044
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_float (__main__.TestBlendingGray) ...
Time taken Numpy 	: 5.634594202041626
Time taken Cython	: 2.110654592514038
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_uint8 (__main__.TestBlendingGray) ...
Time taken Numpy 	: 4.269731521606445
Time taken Cython	: 0.6175847053527832
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 82.689s

OK

----------------------------------------------------------------------

                   *** Blending of RGB images  ***

----------------------------------------------------------------------

test_image_rgb_large_float (__main__.TestBlendingRGB) ...
Time taken Numpy 	: 72.8106141090393
Time taken Cython	: 27.211707830429077
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_large_uint8 (__main__.TestBlendingRGB) ...
Time taken Numpy 	: 51.59686350822449
Time taken Cython	: 7.60575270652771
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_float (__main__.TestBlendingRGB) ...
Time taken Numpy 	: 16.564209699630737
Time taken Cython	: 5.886641979217529
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_uint8 (__main__.TestBlendingRGB) ...
Time taken Numpy 	: 12.357579469680786
Time taken Cython	: 1.907670497894287
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 211.822s

OK

----------------------------------------------------------------------

       *** Rotation of gray images (with NN-interpolation) ***

----------------------------------------------------------------------

test_image_grey_large_float (__main__.TestRotateNNGray) ...
Time taken Numpy 	: 151.31482219696045
Time taken Cython	: 36.94421911239624
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_large_uint8 (__main__.TestRotateNNGray) ...
Time taken Numpy 	: 123.39914846420288
Time taken Cython	: 16.95520257949829
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_float (__main__.TestRotateNNGray) ...
Time taken Numpy 	: 35.40998888015747
Time taken Cython	: 8.452341556549072
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_uint8 (__main__.TestRotateNNGray) ...
Time taken Numpy 	: 30.000380992889404
Time taken Cython	: 3.7491652965545654
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 432.468s

OK

----------------------------------------------------------------------

       *** Rotation of RGB images (with NN-interpolation) ***

----------------------------------------------------------------------

test_image_rgb_large_float (__main__.TestRotateNNRGB) ...
Time taken Numpy 	: 591.0141146183014
Time taken Cython	: 62.84261226654053
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_large_uint8 (__main__.TestRotateNNRGB) ...
Time taken Numpy 	: 388.97081446647644
Time taken Cython	: 25.296037435531616
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_float (__main__.TestRotateNNRGB) ...
Time taken Numpy 	: 136.52729868888855
Time taken Cython	: 14.830699443817139
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_uint8 (__main__.TestRotateNNRGB) ...
Time taken Numpy 	: 95.43055057525635
Time taken Cython	: 5.61405348777771
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 1339.251s

OK

----------------------------------------------------------------------

     *** Rotation of gray images (with Linear-interpolation) ***

----------------------------------------------------------------------

test_image_grey_large_float (__main__.TestRotateLinGray) ...
Time taken Numpy 	: 178.00181245803833
Time taken Cython	: 46.812225580215454
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_large_uint8 (__main__.TestRotateLinGray) ...
Time taken Numpy 	: 153.71812272071838
Time taken Cython	: 32.40706133842468
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_float (__main__.TestRotateLinGray) ...
Time taken Numpy 	: 43.1264967918396
Time taken Cython	: 10.751116752624512
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_uint8 (__main__.TestRotateLinGray) ...
Time taken Numpy 	: 38.20052409172058
Time taken Cython	: 6.512584924697876
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 535.673s

OK

----------------------------------------------------------------------

     *** Rotation of RGB images (with Linear-interpolation) ***

----------------------------------------------------------------------

test_image_rgb_large_float (__main__.TestRotateLinRGB) ...
Time taken Numpy 	: 648.240086555481
Time taken Cython	: 143.8962905406952
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_large_uint8 (__main__.TestRotateLinRGB) ...
Time taken Numpy 	: 478.977468252182
Time taken Cython	: 78.14135432243347
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_float (__main__.TestRotateLinRGB) ...
Time taken Numpy 	: 158.55249094963074
Time taken Cython	: 34.45909404754639
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_uint8 (__main__.TestRotateLinRGB) ...
Time taken Numpy 	: 116.78005647659302
Time taken Cython	: 18.58473801612854
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 1698.162s

OK

----------------------------------------------------------------------

                 *** Sub sampling of gray images  ***

----------------------------------------------------------------------

test_image_grey_large_float (__main__.TestSubSamplingGray) ...
Time taken Numpy 	: 10.621656894683838
Time taken Cython	: 3.1351451873779297
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_large_uint8 (__main__.TestSubSamplingGray) ...
Time taken Numpy 	: 3.7409982681274414
Time taken Cython	: 0.7685854434967041
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_float (__main__.TestSubSamplingGray) ...
Time taken Numpy 	: 2.3244316577911377
Time taken Cython	: 0.8110001087188721
Time taken CUDA  	: Not available ....
...........ok
test_image_grey_small_uint8 (__main__.TestSubSamplingGray) ...
Time taken Numpy 	: 0.9299986362457275
Time taken Cython	: 0.19199752807617188
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 31.435s

OK

----------------------------------------------------------------------

                 *** Sub sampling of RGB images  ***

----------------------------------------------------------------------

test_image_rgb_large_float (__main__.TestSubSamplingRGB) ...
Time taken Numpy 	: 30.34756565093994
  max_open_warning, RuntimeWarning)
Time taken Cython	: 8.097901582717896
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_large_uint8 (__main__.TestSubSamplingRGB) ...
Time taken Numpy 	: 14.92826771736145
Time taken Cython	: 3.053738832473755
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_float (__main__.TestSubSamplingRGB) ...
Time taken Numpy 	: 8.16323471069336
Time taken Cython	: 2.373201608657837
Time taken CUDA  	: Not available ....
...........ok
test_image_rgb_small_uint8 (__main__.TestSubSamplingRGB) ...
Time taken Numpy 	: 3.756680965423584
Time taken Cython	: 0.7535483837127686
Time taken CUDA  	: Not available ....
...........ok

----------------------------------------------------------------------
Ran 4 tests in 79.440s

OK

Process finished with exit code 0
```
