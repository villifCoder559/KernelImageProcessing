# Kernel Image Processing

This project aims to compare the performance of parallel CPU, sequential CPU and 3 GPU versions(gloabl,constant and shared memory) of kernel image processing. 
The goal is to demonstrate the improvements that can be achieved by utilizing a GPU for image processing tasks, despite the
overhead of data transfer from host/device to device/host.
<br/>
You can read a detailed pdf **project report** inside the folder "project_report".

## Prerequisites

- A computer with an NVIDIA GPU
- CUDA and cuDNN libraries installed
- C++ compiler that supports OpenMP

## Results

You can compare the performance of the parallel CPU, sequential CPU and GPU versions of the kernel image processing by comparing the execution time, memory usage and data 
transfer time using different kernel sizes (3x3, 5x5, 7x7).
The GPU versions should show improvements in terms of execution time and memory usage when compared to the CPU versions, despite the data transfer time from host to device.

## Conclusion

This project demonstrates the benefits of utilizing a GPU for image processing tasks despite the overhead of data transfer from host to device.
The GPU versions is able to perform the task much faster and more efficiently than the CPU versions.
In conclusion, using GPU's for image processing tasks should be considered when performance is a critical factor, despite the data transfer time.
