#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
// enum type_kernel { edge_detection, sharpen, constant_blur, gaussian_blur_3x3, gaussian_blur_5x5, gaussian_blur_7x7 };

Kernel::Kernel(type_kernel type) {
  switch (type) {
  case edge_detection:
    size = 3;
    kernel = {new float[size * size]{-1, -1, -1, -1, 8, -1, -1, -1, -1}};
    break;
  case sharpen:
    size = 3;
    kernel = {new float[size * size]{0, -1, 0, -1, 5, -1, 0, -1, 0}};
    break;
  case constant_blur:
    size = 3;
    kernel = {new float[size * size]{1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f}};
    break;
  case gaussian_blur_3x3:
    size = 3;
    kernel = {new float[size * size]{1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f, 2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}};
    break;
  case gaussian_blur_5x5:
    size = 5;
    kernel = {new float[size * size]{
        1.0f / 256.0f,  4.0f / 256.0f, 6.0f / 256.0f,  4.0f / 256.0f,  1.0f / 256.0f,  4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f,
        4.0f / 256.0f,  6.0f / 256.0f, 24.0f / 256.0f, 36.0f / 256.0f, 24.0f / 256.0f, 6.0f / 256.0f, 4.0f / 256.0f,  16.0f / 256.0f, 24.0f / 256.0f,
        16.0f / 256.0f, 4.0f / 256.0f, 1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f
    }};
    break;
  case gaussian_blur_7x7:
    size = 7;
    kernel = {new float[size * size]{0.0f,
                                     0.0f,
                                     1.0f / 1003.0f,
                                     2.0f / 1033.0f,
                                     1.0f / 1033.0f,
                                     0.0f,
                                     0.0f,
                                     0.0f,
                                     3.0f / 1003.0f,
                                     13.0f / 1003.0f,
                                     22.0f / 1003.0f,
                                     13.0f / 1003.0f,
                                     3.0f / 1003.0f,
                                     0.0f,
                                     1.0f / 1003.0f,
                                     13.0f / 1003.0f,
                                     59.0f / 1003.0f,
                                     97.0f / 1003.0f,
                                     59.0f / 1003.0f,
                                     13.0f / 1003.0f,
                                     1.0f / 1003.0f,
                                     2.0f / 1003.0f,
                                     22.0f / 1003.0f,
                                     97.0f / 1003.0f,
                                     159.0f / 1003.0f,
                                     97.0f / 1003.0f,
                                     22.0f / 1003.0f,
                                     2.0f / 1003.0f,
                                     1.0f / 1003.0f,
                                     13.0f / 1003.0f,
                                     59.0f / 1003.0f,
                                     97.0f / 1003.0f,
                                     59.0f / 1003.0f,
                                     13.0f / 1003.0f,
                                     1.0f / 1003.0f,
                                     0.0f,
                                     3.0f / 1003.0f,
                                     13.0f / 1003.0f,
                                     22.0f / 1003.0f,
                                     13.0f / 1003.0f,
                                     3.0f / 1003.0f,
                                     0.0f,
                                     0.0f,
                                     0.0f,
                                     1.0f / 1003.0f,
                                     2.0f / 1003.0f,
                                     1.0f / 1003.0f,
                                     0.0f,
                                     0.0f}};
    break;
  }
}

int Kernel::get_size() { return size; }
float *Kernel::get_kernel() { return kernel; }
Kernel::~Kernel() { delete[] kernel; }