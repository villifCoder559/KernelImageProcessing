#ifndef CONV_GPU
#define CONV_GPU
#include "image.h"
#include "kernel.h"
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "padding_image.h"
#define MAX_SIZE_KERNEL 7
#define BLOCK_WIDTH 16

class ConvolutionGPU {
public:
  ConvolutionGPU(){};
  static Image *apply_convolution_constant_memory(Image *image, Kernel *kernel, type_padding padding = zero);
  static Image *apply_convolution_base(Image *image, Kernel *kernel, type_padding padding = zero);
  static Image *apply_convolution_shared_memory(Image *image, Kernel *kernel, type_padding padding = zero);
  ~ConvolutionGPU(){};
};
#endif