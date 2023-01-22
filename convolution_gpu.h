#ifndef CONV_GPU
#define CONV_GPU
#include "image.h"
#include "kernel.h"
#include "padding_image.h"
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#define MAX_SIZE_KERNEL 7
#define BLOCK_WIDTH 16
enum type_memory { global, constant, shared };
class ConvolutionGPU {
public:
  ConvolutionGPU(){};
  static Image *apply_convolution_constant_memory(Image *image, Kernel *kernel, type_padding padding = zero);
  static Image *apply_convolution_global_memory(Image *image, Kernel *kernel, type_padding padding = zero);
  static Image *apply_convolution_shared_memory(Image *image, Kernel *kernel, type_padding padding = zero, int version = 1);
  // static Image *apply_convolution(Image *image, Kernel *kernel, type_padding padding = zero, type_memory type_mem = constant, int version = 1);
  ~ConvolutionGPU(){};
};
#endif