# Lossless-Data-Compression

## Task Description
This task was part of a project at the end of High Performance Computing Laboratory. The goal of this course was to learn programming on GPUs with OpenCL. This is used to speed up execution time of any code by parallelizing bottlenecks and running them on GPU.

For any program, if there are many segments of the code that can be run in parallel, considerable speed up can be achieved by running the code on a GPU as opposed to CPU.

### Algorithm to optimize
Encoding voxel of data using Exponential Golomb encoding for lossless compression using Lorenzo Predictor.

[1] P. Lindstrom et al. , "Fast and Efficient Compression of Floating-Point Data", *IEEE Transactions on Visualization and Computer Graphics 12(5):1245-50 · September 2006*

[2] L.Ibarria, et.al., "Out-of-core compression and decompression of large n-dimensional scalar fields.", *Eurographics 2003*

## Code description
 
 ```
src
│   CPU_implementation.cpp
│   GPU_implementation.cl    
```
**CPU_implementation.cpp**

This program has the below functions:

* Run an implementation of the code on CPU
* Another implementation to run code on CPU+GPU combination with the parallelized code being run on GPU
* Compute speed up achieved

**GPU_implementation.cl**

This program contains the menthods of the algorithm that can be parallelized and are run on the GPU.

## Authors

The project was developed by the below three team members:

* Rafael Villanueva Ferrari
* [Pushpam Choudhury](https://github.com/PushpamChoudhury)
* [Vijay Ravichandran](https://github.com/vijay-ravichandran)
