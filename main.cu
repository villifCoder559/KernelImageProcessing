#include "convolution_gpu.h"
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/*
  1) Create kernel that uses shared memory OK
  2) Manage data with size greater than constant memory OK
  2.1) Manage the case where the image is too big (DIM=500) OK
  3) Adapt algorithm to non-squared images OK
  4) Create more filters OK
  4.1) Manage the case where the product exceeds 255 OK
  5) Create classes OK
  6) Fix apply_convolution passing all class Image OK
  7) Add const to some params OK
  8) Test all
  END) Measure divergence and general performance (profiling) OK
*/
void compare_result(Image *cmp1, Image *cmp2);
void print_image(Image *img, std::string title = "");

int main() {
  int width = 200;
  int height = 200;
  Image *img = new Image(width, height);
  Kernel *mask = new Kernel(constant_blur);
  Image *result_img_base = ConvolutionGPU::apply_convolution_base(img, mask);
  Image *result_img_constant = ConvolutionGPU::apply_convolution_constant_memory(img, mask);
  Image *result_img_shared = ConvolutionGPU::apply_convolution_shared_memory(img, mask);
  compare_result(result_img_base, result_img_shared);
  compare_result(result_img_shared,result_img_constant);
}

void compare_result(Image *cmp1, Image *cmp2) {
  if (cmp1->get_height() == cmp2->get_height() && cmp1->get_width() == cmp2->get_width()) {
    int width = cmp1->get_width();
    for (int i = 0; i < cmp1->get_width(); i++) {
      for (int j = 0; j < cmp1->get_height(); j++)
        if ((cmp1->get_image())[i * width + j] != (cmp2->get_image())[i * width + j])
          printf("(%d,%d) \t", i, j);
    }
  } else
    printf("Sizes not equal");
}
void print_image(Image *img, std::string title) {
  int width = img->get_width();
  std::cout << title << std::endl;
  for (int i = 0; i < img->get_width(); i++) {
    for (int j = 0; j < img->get_height(); j++)
      printf("%d \t", (img->get_image())[i * width + j]);
    printf("\n");
  }
}
