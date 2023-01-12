#include "image.h"
#include "kernel.h"
#include <iostream>
#include <omp.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

class ConvolutionCPU {
public:
  ConvolutionCPU(){};
  static Image *apply_convolution_parallel(Image *image, Kernel *kernel);
  static Image *apply_convolution_sequential(Image *image,Kernel *kernel);
  ~ConvolutionCPU(){};
};