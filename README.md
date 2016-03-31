

# Python Cython CUDA

## About Project

This project has implementation for three algorithms:
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
+ CUDA with Cython (Not available work needs to be done to build compiler wrapper for nvcc to be called from python.
The `setup.py` file will be updated soon...)

## Environment
+ Windows 10
+ Python 3.4.3
+ Cython 0.23.4
+ CUDA 7.5

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
python benchmark.py
```
+ The unittest framework will run and dump the results in report folder
  + Images with test_image in their name are the output of the algorithm
  + Other images have barplots that will compare numpy and cython implementation

## Benchmark results
Below are the results (Smaller the better as time taken is the y axis)

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


