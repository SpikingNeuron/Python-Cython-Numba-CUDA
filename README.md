

# Python Cython CUDA

## About Project

Python is very slow. To improve the speed we can use Cython. Also most libraries that come with Python are implemented using Cython for performance. 

When it comes to GPU programming CUDA has good support with C++. 

But with Python we have only two options as below.

+ PyCUDA
    + Open source
    + Needs to write kernels in string which is not elegant
+ Continuum accelerate library (previously known as numbapro)
    + Paid version
    + Full integration with Python
    
Third option we have is use C++ CUDA with Cython.

Advantages

+ Development in pure C++ for CUDA code so all IDE functionality is available.
+ Compile Cython code to use the C++ CUDA code 
+ Python can easily understand Cython, and Cython can easily call C++ code
+ Minimum overhead

Disadvantage

+ Need to know both C++ but this disadvantage is still there with PyCUDA but not with Continuum accelerate library 

## Environment
+ Windows 10
+ Python 3.4.3
+ Cython 0.23.4
+ CUDA 7.5
