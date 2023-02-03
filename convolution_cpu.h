#ifndef CONV_CPU
#define CONV_CPU
#include "image.h"
#include "kernel.h"
#include "padding_image.h"
#include <iostream>
#include <omp.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

class ConvolutionCPU {
public:
  ConvolutionCPU(){};
  static double total_time;
  static Image *apply_convolution_parallel(Image *image, Kernel *kernel,type_padding padding=zero);
  static Image *apply_convolution_sequential(Image *image,Kernel *kernel,type_padding padding=zero);
  ~ConvolutionCPU(){};
};
#endif