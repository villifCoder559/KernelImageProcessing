#ifndef CONV_GPU
#define CONV_GPU
#include "image.h"
#include "kernel.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define MAX_SIZE_KERNEL 7
#define BLOCK_WIDTH 16

class ConvolutionGPU {
public:
  ConvolutionGPU(){};
  static Image *apply_convolution_constant_memory(Image *image, Kernel *kernel);
  static Image *apply_convolution_base(Image *image, Kernel *kernel);
  static Image *apply_convolution_shared_memory(Image *image, Kernel *kernel);
  ~ConvolutionGPU(){};
};
#endif