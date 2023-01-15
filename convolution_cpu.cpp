#include "convolution_cpu.h"

Image *ConvolutionCPU::apply_convolution_parallel(Image *image, Kernel *kernel, type_padding padding) {
  Image *img_padded = PaddingImage::apply_padding_to_image(image, kernel->get_size(), padding);
  const unsigned char *img = img_padded->get_image();
  const int WIDTH = image->get_width();
  const int HEIGHT = image->get_height();
  const int K_SIZE = kernel->get_size();
  const float *mask = kernel->get_kernel();
  const int shift = K_SIZE / 2;
  const float norm = kernel->get_normalization_factor();
  unsigned char *result_img = (unsigned char *)malloc(sizeof(unsigned char) * WIDTH * HEIGHT);
  float sum = 0.0f;
  auto start = omp_get_wtime();
#pragma omp parallel for firstprivate(HEIGHT, WIDTH, K_SIZE, shift, sum, norm)
  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < WIDTH; j++) {
      for (int k = 0; k < K_SIZE; k++) {
        for (int w = 0; w < K_SIZE; w++) {
          int index_r = i + k;
          int index_c = j + w;
          // if (index_r >= 0 && index_r < HEIGHT && index_c >= 0 && index_c < WIDTH) {
            sum += img[index_r * (WIDTH + (K_SIZE - 1)) + index_c] * mask[k * K_SIZE + w];
            // if (i == 1 && j == 7902)
            //   printf("[%d_%d]->%d*%f \n", index_r, index_c, img[index_r * WIDTH + index_c], mask[k * K_SIZE + w]);
          // }
        }
      }
      sum = sum * norm;
      result_img[i * WIDTH + j] = (unsigned char)sum > 255 ? 255 : sum < 0 ? 0 : sum;
      sum = 0.0f;
    }
  }
  auto end = omp_get_wtime();
  std::cout << "TIME_CPU_parallel: " << end - start << "\n";
  Image *out_image = new Image(result_img, WIDTH, HEIGHT);
  return out_image;
}
Image *ConvolutionCPU::apply_convolution_sequential(Image *image, Kernel *kernel, type_padding padding) {
  Image *img_padded = PaddingImage::apply_padding_to_image(image, kernel->get_size(), padding);
  const unsigned char *img = img_padded->get_image();
  const int WIDTH = image->get_width();
  const int HEIGHT = image->get_height();
  const int K_SIZE = kernel->get_size();
  const float *mask = kernel->get_kernel();
  const int shift = K_SIZE / 2;
  const float norm = kernel->get_normalization_factor();
  unsigned char *result_img = (unsigned char *)malloc(sizeof(unsigned char) * WIDTH * HEIGHT);
  float sum = 0.0f;
  auto start = omp_get_wtime();
  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < WIDTH; j++) {
      for (int k = 0; k < K_SIZE; k++) {
        int index_r = i + k;
        for (int w = 0; w < K_SIZE; w++) {
          int index_c = j + w;
          // if (index_r >= 0 && index_r < HEIGHT && index_c >= 0 && index_c < WIDTH) {
            sum += img[index_r * (WIDTH+K_SIZE-1) + index_c] * mask[k * K_SIZE + w];
          // }
        }
      }
      sum = sum * norm;
      result_img[i * WIDTH + j] = (unsigned char)(sum > 255 ? 255 : sum < 0 ? 0 : sum);
      sum = 0.0f;
    }
  }
  auto end = omp_get_wtime();
  std::cout << "TIME_CPU_sequential: " << end - start << "\n";
  Image *out_image = new Image(result_img, WIDTH, HEIGHT);
  return out_image;
}