#include "convolution_cpu.h"
#include "convolution_gpu.h"
#include "padding_image.h"
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
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
  8) Test all OK
  9) Add pixel replication and pixel mirroring in conv_base and constant memory OK
  10) Add replication and mirroring in conv_shared_memory OK
  10.1) Clean the code (apply base and constant convolution) OK
  11) test new features OK
  END) Measure divergence and general performance (profiling) OK
*/
void compare_result(Image *cmp1, Image *cmp2);
void print_image(Image *img, std::string title = "");
void test_all_convolutions(int tot_tests, int tot_kernels, int *width, int *height, type_kernel *kernels, type_padding padding);
void test_multiple_time_convolution(int n, int width, int height, type_kernel t_kernel, type_padding t_padding, type_memory memory);
void print_partial_image(Image *img, int r_center, int c_center, int K_SIZE);

double ConvolutionGPU::total_time = 0;
double ConvolutionGPU::kernel_time = 0;
double ConvolutionCPU::total_time = 0;
// int width[] = {1250, 1970, 2500, 8546};
// int height[] = {850, 1298, 1530, 4450};
int main() {
  int tot_tests = 4;
  int tot_kernels = 3;
  int width[] = {1250, 1970, 2500, 8546};
  int height[] = {850, 1298, 1530, 4450};
  type_kernel kernels[] = {gaussian_blur_3x3, gaussian_blur_5x5, gaussian_blur_7x7};
  type_padding paddings[] = {zero, pixel_mirroring, pixel_replication};
  // Image *img = new Image(1920, 1080, 100);
  // Kernel *mask = new Kernel(kernels[2]);
  // Image *result_img_shared = ConvolutionGPU::apply_convolution_global_memory(img, mask, zero);
  ConvolutionGPU::total_time = 0;
  ConvolutionGPU::kernel_time = 0;
  int size = 3;
  test_multiple_time_convolution(100, width[size], height[size], kernels[1], paddings[2], shared_constant);
  // // cudaDeviceProp *prop=new cudaDeviceProp();
  // cudaGetDeviceProperties_v2(prop,1);
  // test_all_convolutions(tot_tests, tot_kernels, width, height, kernels, paddings[1]);
  // Image *result_img_base = ConvolutionGPU::apply_convolution_global_memory(img, mask, zero);
  // compare_result(result_img_base, result_img_shared);
  return 0;
}

void test_multiple_time_convolution(int n, int width, int height, type_kernel t_kernel, type_padding t_padding, type_memory memory) {
  Kernel *mask = new Kernel(t_kernel);
  double total_time = 0;
  printf("\n(%d,%d) using %dx%d gaussian filter with %d images", width, height, mask->get_size(), mask->get_size(), n);
  for (int i = 0; i < n; i++) {
    Image *img = new Image(width, height, 0);
    Image *result_img;
    switch (memory) {
    case global:
      if (i == 0)
        std::cout << "global " << std::endl;
      result_img = ConvolutionGPU::apply_convolution_global_memory(img, mask, t_padding);
      break;
    case constant:
      if (i == 0)
        std::cout << "constant " << std::endl;
      result_img = ConvolutionGPU::apply_convolution_constant_memory(img, mask, t_padding);
      break;
    case shared_constant:
      if (i == 0)
        std::cout << "shared_constant " << std::endl;
      result_img = ConvolutionGPU::apply_convolution_shared_constant_memory(img, mask, t_padding);
      break;
    }
    delete img;
    delete result_img;
  }
  std::cout << "No_transfer_time: " << ConvolutionGPU::kernel_time << std::endl;
  std::cout << "With_transfer_time: " << ConvolutionGPU::total_time << std::endl;
  ConvolutionGPU::total_time = 0;
  ConvolutionGPU::kernel_time = 0;
  delete mask;
}
void test_all_convolutions(int tot_tests, int tot_kernels, int *width, int *height, type_kernel *kernels, type_padding padding) {
  for (int i = 0; i < tot_kernels; i++) {
    Kernel *mask = new Kernel(kernels[i]);
    for (int j = 0; j < tot_tests; j++) {
      Image *img = new Image(width[j], height[j], 100);
      Image *img_padded = PaddingImage::apply_padding_to_image(img, 3, zero);
      // print_image(img);
      // print_partial_image(img_padded, 2, 12, mask->get_size()); // padded(1,1)->out(0,0)
      printf("\n(%d,%d) using %dx%d gaussian filter \n", height[j], width[j], mask->get_size(), mask->get_size());
      Image *result_img_base = ConvolutionGPU::apply_convolution_global_memory(img, mask, padding);
      Image *result_img_shared_const = ConvolutionGPU::apply_convolution_constant_memory(img, mask, padding);
      Image *result_img_constant = ConvolutionGPU::apply_convolution_constant_memory(img, mask, padding);
      Image *result_cpu_seq = ConvolutionCPU::apply_convolution_sequential(img, mask, padding);
      Image *result_cpu_par = ConvolutionCPU::apply_convolution_parallel(img, mask, padding);
      // print_image(result_img_constant);
      // print_image(result_img_shared);
      compare_result(result_img_base, result_img_constant);
      compare_result(result_img_base, result_img_shared_const);
      compare_result(result_img_base, result_cpu_seq);
      compare_result(result_cpu_seq, result_cpu_par);
      delete img;
      delete img_padded;
      delete result_img_base;
      delete result_img_constant;
      delete result_img_shared_const;
      delete result_cpu_seq;
      delete result_cpu_par;
    }
    delete mask;
    printf("\n");
  }
}
void compare_result(Image *cmp1, Image *cmp2) {
  int count = 0;
  if (cmp1->get_height() == cmp2->get_height() && cmp1->get_width() == cmp2->get_width()) {
    int width = cmp1->get_width();
    for (int i = 0; i < cmp1->get_height(); i++) {
      for (int j = 0; j < cmp1->get_width(); j++)
        if ((cmp1->get_image())[i * width + j] != (cmp2->get_image())[i * width + j]) {
          printf("(%d,%d)->%d!=%d ", i, j, cmp1->get_image()[i * width + j], cmp2->get_image()[i * width + j]);
          // printf("%d!=%d \t", cmp1->get_image()[i * width + j], cmp2->get_image()[i * width + j]);
          count++;
        }
    }
  } else
    printf("Sizes not equal");
  printf("\n");
  std::cout << count << std::endl;
}
void print_image(Image *img, std::string title) {
  int width = img->get_width();
  std::cout << title << std::endl;
  for (int i = 0; i < img->get_height(); i++) {
    for (int j = 0; j < img->get_width(); j++)
      printf("%d \t", (img->get_image())[i * width + j]);
    printf("\n");
  }
}
void print_partial_image(Image *img, int r_center, int c_center, int K_SIZE) {
  for (int i = r_center - K_SIZE / 2; i <= r_center + K_SIZE / 2; i++) {
    for (int j = c_center - K_SIZE / 2; j <= c_center + K_SIZE / 2; j++) {
      if (i >= 0 && j >= 0 && i < img->get_height() && j < img->get_width())
        printf("(%d,%d)-> %d \t", i, j, (img->get_image())[i * img->get_width() + j]);
    }
    printf("\n");
  }
}
