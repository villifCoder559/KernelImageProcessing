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
#define BLOCK_HEIGHT 16

enum type_memory { global_memory, constant_memory, shared_memory};
class ConvolutionGPU {
public:
  ConvolutionGPU(){};
  static double total_time;
  static double kernel_time;
  static Image *apply_convolution_constant_memory(Image *image, Kernel *kernel, type_padding padding = zero);
  static Image *apply_convolution_global_memory(Image *image, Kernel *kernel, type_padding padding = zero);
  // static Image *apply_convolution_shared_memory(Image *image, Kernel *kernel, type_padding padding = zero);
  static Image *apply_convolution_shared_constant_memory(Image *image, Kernel *kernel, type_padding padding = zero);
  ~ConvolutionGPU(){};
};
#endif