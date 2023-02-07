#include "convolution_gpu.h"
__constant__ float kernel_const[MAX_SIZE_KERNEL * MAX_SIZE_KERNEL]; // 48KB max size [202,202]

__global__ void convolution_global_memory(const float *d_kernel, unsigned char *out_img, unsigned char *d_img, const int WIDTH, const int HEIGHT, const int K_SIZE, const float norm) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  float sum = 0.0f;
  if (col < WIDTH && row < HEIGHT) {
    for (int i = 0; i < K_SIZE; i++) {
      for (int j = 0; j < K_SIZE; j++) {
        int index_c = col + j;
        int index_r = row + i;
        sum += d_img[index_r * (WIDTH + K_SIZE - 1) + index_c] * d_kernel[i * K_SIZE + j];
      }
    }
    sum = fmaxf(0.0f, fminf(sum * norm, 255.0f));
    out_img[row * WIDTH + col] = (unsigned char)sum;
  }
}

__global__ void convolution_constant_memory(unsigned char *out_img, unsigned char *d_img, const int WIDTH, const int HEIGHT, const int K_SIZE, const float norm) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  float sum = 0.0f;
  if (col < WIDTH && row < HEIGHT) {
    for (int i = 0; i < K_SIZE; i++) {
      for (int j = 0; j < K_SIZE; j++) {
        int index_c = col + j;
        int index_r = row + i;
        sum += d_img[index_r * (WIDTH + K_SIZE - 1) + index_c] * kernel_const[i * K_SIZE + j];
        // if (row == 2 && col == 9)
        //   printf("(%d,%d):[%d_%d]->%d*%f \n", blockIdx.x, blockIdx.y, index_r, index_c, d_img[index_r * (WIDTH + K_SIZE - 1) + index_c], kernel_const[i * K_SIZE + j]);
      }
    }
    sum = fmaxf(0.0f, fminf(sum * norm, 255.0f));
    // if (row == 2 && col == 9)
    //   printf("SUM: %d \n", (unsigned char)sum);
    out_img[row * WIDTH + col] = (unsigned char)sum;
  }
}
__global__ void convolution_shared_memory_constant(unsigned char *img, unsigned char *out_img, const int WIDTH, const int HEIGHT, const int K_SIZE, const float norm) {
  extern __shared__ unsigned char d_img[];
  const int size_d_img = (BLOCK_WIDTH + K_SIZE - 1);
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int c = threadIdx.x;
  int r = threadIdx.y;
  int shift = K_SIZE / 2;
  bool row_above = (row - shift >= 0 && r < shift);
  bool row_below = (r >= BLOCK_HEIGHT - shift && (row + shift) < HEIGHT);
  bool col_left = (col - shift >= 0 && c < shift);
  bool col_right = ((c >= BLOCK_WIDTH - shift) && (col + shift < WIDTH));
  bool bottom_right = (col_right && row_below);
  bool top_left = (col_left && (shift - r > 0) && row_above);
  bool bottom_left = (col_left && row_below && (shift - c > 0));
  bool top_right = (col_right && row_above && (c >= BLOCK_WIDTH - shift) && (shift - r > 0));
  if (col >= WIDTH && row >= HEIGHT)
    return;
  // if (col < WIDTH && row < HEIGHT) {
  d_img[(r + shift) * size_d_img + c + shift] = img[col + WIDTH * row];
  // check if exists row above
  if (row_above) {
    d_img[r * size_d_img + c + shift] = img[col + (row - shift) * WIDTH];
  }
  // row below
  if (row_below) {
    d_img[(r + K_SIZE - 1) * size_d_img + c + shift] = img[col + (row + shift) * WIDTH];
  }
  // column left
  if (col_left) {
    d_img[(r + shift) * size_d_img + c] = img[col - shift + (row)*WIDTH];
  }
  // column right
  if (col_right) {
    d_img[(r + shift) * size_d_img + c + K_SIZE - 1] = img[col + shift + row * WIDTH];
  }
  // corner bottom right
  if (bottom_right) {
    d_img[(r + K_SIZE - 1) * size_d_img + c + K_SIZE - 1] = img[col + shift + (row + shift) * WIDTH];
  }
  // corner top left
  if (top_left) {
    d_img[r * size_d_img + c] = img[col - shift + (row - shift) * WIDTH];
  }
  // corner bottom left
  if (bottom_left) {
    d_img[(r + K_SIZE - 1) * size_d_img + c] = img[col - shift + (row + shift) * WIDTH];
  }
  // corner top right
  if (top_right) {
    d_img[r * size_d_img + c + K_SIZE - 1] = img[col + shift + (row - shift) * WIDTH];
  }
  // }
  __syncthreads();
  float sum = 0.0f;
  if (col < WIDTH - shift && row < HEIGHT - shift && col_left && row_above) {
    for (int i = 0; i < K_SIZE; i++) {
      for (int j = 0; j < K_SIZE; j++) {
        int index_c = c + j;
        int index_r = r + i;
        sum += d_img[index_r * size_d_img + index_c] * kernel_const[i * K_SIZE + j];
      }
    }
    sum = fmaxf(0.0f, fminf(sum * norm, 255.0f));
    out_img[(row - shift) * (WIDTH - (K_SIZE - 1)) + col - shift] = (unsigned char)sum;
  }
}

// __global__ void convolution_shared_memory(unsigned char *img, unsigned char *out_img, float *kernel, const int WIDTH, const int HEIGHT, const int K_SIZE, const float norm) {
//   extern __shared__ int dynamic_array[];
//   unsigned char *d_img = (unsigned char *)&dynamic_array[K_SIZE * K_SIZE];
//   float *mask = (float *)&dynamic_array;
//   const int size_d_img = (BLOCK_WIDTH + K_SIZE - 1);
//   int col = threadIdx.x + blockIdx.x * blockDim.x;
//   int row = threadIdx.y + blockIdx.y * blockDim.y;
//   int c = threadIdx.x;
//   int r = threadIdx.y;
//   int shift = K_SIZE / 2;
//   bool row_above = (row - shift >= 0 && r < shift);
//   bool row_below = (r >= BLOCK_HEIGHT - shift && (row + shift) < HEIGHT);
//   bool col_left = (col - shift >= 0 && c < shift);
//   bool col_right = ((c >= BLOCK_WIDTH - shift) && (col + shift < WIDTH));
//   bool bottom_right = (col_right && row_below);
//   bool top_left = (col_left && (shift - r > 0) && row_above);
//   bool bottom_left = (col_left && row_below && (shift - c > 0));
//   bool top_right = (col_right && row_above && (c >= BLOCK_WIDTH - shift) && (shift - r > 0));
//   if (col >= WIDTH && row >= HEIGHT)
//     return;
//   // if (col < WIDTH && row < HEIGHT) {
//   d_img[(r + shift) * size_d_img + c + shift] = img[col + WIDTH * row];
//   // check if exists row above
//   if (row_above) {
//     d_img[r * size_d_img + c + shift] = img[col + (row - shift) * WIDTH];
//   }
//   // row below
//   if (row_below) {
//     d_img[(r + K_SIZE - 1) * size_d_img + c + shift] = img[col + (row + shift) * WIDTH];
//   }
//   // column left
//   if (col_left) {
//     d_img[(r + shift) * size_d_img + c] = img[col - shift + (row)*WIDTH];
//   }
//   // column right
//   if (col_right) {
//     d_img[(r + shift) * size_d_img + c + K_SIZE - 1] = img[col + shift + row * WIDTH];
//   }
//   // corner bottom right
//   if (bottom_right) {
//     d_img[(r + K_SIZE - 1) * size_d_img + c + K_SIZE - 1] = img[col + shift + (row + shift) * WIDTH];
//   }
//   // corner top left
//   if (top_left) {
//     d_img[r * size_d_img + c] = img[col - shift + (row - shift) * WIDTH];
//   }
//   // corner bottom left
//   if (bottom_left) {
//     d_img[(r + K_SIZE - 1) * size_d_img + c] = img[col - shift + (row + shift) * WIDTH];
//   }
//   // corner top right
//   if (top_right) {
//     d_img[r * size_d_img + c + K_SIZE - 1] = img[col + shift + (row - shift) * WIDTH];
//   }
//   if (r * BLOCK_WIDTH + c < K_SIZE * K_SIZE)
//     mask[r * BLOCK_WIDTH + c] = kernel[r * BLOCK_WIDTH + c];
//   // }
//   __syncthreads();
//   float sum = 0.0f;
//   if (col < WIDTH - shift && row < HEIGHT - shift && col_left && row_above) {
//     for (int i = 0; i < K_SIZE; i++) {
//       for (int j = 0; j < K_SIZE; j++) {
//         int index_c = c + j;
//         int index_r = r + i;
//         sum += d_img[index_r * size_d_img + index_c] * mask[i * K_SIZE + j];
//         if (row == 10 && col == 10)
//           printf("(%d,%d):[%d_%d]->%d*%f \n", blockIdx.x, blockIdx.y, index_r, index_c, d_img[index_r * (WIDTH + K_SIZE - 1) + index_c], kernel_const[i * K_SIZE + j]);
//       }
//     }
//     sum = fmaxf(0.0f, fminf(sum * norm, 255.0f));
//     out_img[(row - shift) * (WIDTH - (K_SIZE - 1)) + col - shift] = (unsigned char)sum;
//   }
// }

Image *ConvolutionGPU::apply_convolution_constant_memory(Image *image, Kernel *kernel, type_padding padding) {
  unsigned char *d_out_img;
  unsigned char *d_image;
  const int WIDTH = image->get_width();
  const int HEIGHT = image->get_height();
  const int K_SIZE = kernel->get_size();
  Image *img_padded = PaddingImage::apply_padding_to_image(image, K_SIZE, padding);
  const unsigned char *img = image->get_image();
  const float norm = kernel->get_normalization_factor();
  double start_time_data_transfer;
  double end_time_data_transfer;
  double start_convolution;
  double end_convolution;
  int grid_size_x = (WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  int grid_size_y = (HEIGHT + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
  dim3 block_dim(BLOCK_WIDTH, BLOCK_HEIGHT);
  dim3 grid_dim(grid_size_x, grid_size_y);
  // printf("grid_x:%d, grid_y:%d \n", grid_size_x, grid_size_y);
  unsigned char *h_out_img = (unsigned char *)malloc(sizeof(unsigned char) * WIDTH * HEIGHT);
  start_time_data_transfer = omp_get_wtime();
  cudaMalloc(&d_out_img, WIDTH * HEIGHT * sizeof(unsigned char));
  cudaMalloc(&d_image, img_padded->get_width() * img_padded->get_height() * sizeof(unsigned char));
  cudaMemcpy(d_image, img_padded->get_image(), img_padded->get_width() * img_padded->get_height() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(kernel_const, kernel->get_kernel(), K_SIZE * K_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
  start_convolution = omp_get_wtime();
  convolution_constant_memory<<<grid_dim, block_dim>>>(d_out_img, d_image, WIDTH, HEIGHT, K_SIZE, norm);
  end_convolution = omp_get_wtime();
  cudaMemcpy(h_out_img, d_out_img, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  end_time_data_transfer = omp_get_wtime();
  Image *out_img = new Image(h_out_img, WIDTH, HEIGHT);
  total_time += end_time_data_transfer - start_time_data_transfer;
  kernel_time += end_convolution - start_convolution;
  // std::cout << "TOTAL_constant -> " << end_time_data_transfer - start_time_data_transfer - (end_convolution - start_convolution)
  //           << " + " << end_convolution - start_convolution << " = "
  //           << end_time_data_transfer - start_time_data_transfer << std::endl;
  delete img_padded;
  cudaFree(d_image);
  cudaFree(d_out_img);
  return out_img;
}
Image *ConvolutionGPU::apply_convolution_global_memory(Image *image, Kernel *kernel, type_padding padding) {
  unsigned char *d_out_img;
  float *d_kernel;
  unsigned char *d_img;
  const int K_SIZE = kernel->get_size();
  Image *img_padded = PaddingImage::apply_padding_to_image(image, K_SIZE, padding);
  const int WIDTH = image->get_width();
  const int HEIGHT = image->get_height();
  const float norm = kernel->get_normalization_factor();
  double start_time_data_transfer;
  double end_time_data_transfer;
  double start_convolution;
  double end_convolution;
  int grid_size_x = (WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  int grid_size_y = (HEIGHT + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
  // std::cout << grid_size_x << ";" << grid_size_y<<std::endl;
  dim3 block_dim(BLOCK_WIDTH, BLOCK_HEIGHT);
  dim3 grid_dim(grid_size_x, grid_size_y);
  unsigned char *h_out_img = (unsigned char *)malloc(WIDTH * HEIGHT * sizeof(unsigned char));
  start_time_data_transfer = omp_get_wtime();
  cudaMalloc(&d_kernel, K_SIZE * K_SIZE * sizeof(float));
  cudaMalloc(&d_img, img_padded->get_width() * img_padded->get_height() * sizeof(unsigned char));
  cudaMalloc(&d_out_img, WIDTH * HEIGHT * sizeof(unsigned char));
  cudaMemcpy(d_img, img_padded->get_image(), img_padded->get_width() * img_padded->get_height() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel->get_kernel(), K_SIZE * K_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  start_convolution = omp_get_wtime();
  convolution_global_memory<<<grid_dim, block_dim>>>(d_kernel, d_out_img, d_img, WIDTH, HEIGHT, K_SIZE, norm);
  end_convolution = omp_get_wtime();
  cudaMemcpy(h_out_img, d_out_img, image->get_width() * image->get_height() * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  end_time_data_transfer = omp_get_wtime();
  total_time += end_time_data_transfer - start_time_data_transfer;
  kernel_time += end_convolution - start_convolution;
  Image *out_img = new Image(h_out_img, image->get_width(), image->get_height());
  // std::cout << "TOTAL_global -> " << end_time_data_transfer - start_time_data_transfer - (end_convolution - start_convolution)
  //           << " + " << end_convolution - start_convolution << " = "
  //           << end_time_data_transfer - start_time_data_transfer << std::endl;
  delete img_padded;
  cudaFree(d_img);
  cudaFree(d_kernel);
  cudaFree(d_out_img);
  return out_img;
}

// Image *ConvolutionGPU::apply_convolution_shared_memory(Image *image, Kernel *kernel, type_padding padding) {
//   unsigned char *d_out_img;
//   unsigned char *d_img;
//   float *d_kernel;
//   const int K_SIZE = kernel->get_size();
//   Image *img_padded = PaddingImage::apply_padding_to_image(image, K_SIZE, padding);
//   const int WIDTH = image->get_width();
//   const int HEIGHT = image->get_height();
//   const int WIDTH_PADDED = WIDTH + K_SIZE - 1;
//   const int HEIGHT_PADDED = HEIGHT + K_SIZE - 1;
//   const float norm = kernel->get_normalization_factor();
//   const unsigned char *img = img_padded->get_image();
//   double start_time_data_transfer;
//   double end_time_data_transfer;
//   double start_convolution;
//   double end_convolution;
//   int grid_size_x = (WIDTH_PADDED + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
//   int grid_size_y = (HEIGHT_PADDED + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
//   dim3 block_dim(BLOCK_WIDTH, BLOCK_HEIGHT);
//   dim3 grid_dim(grid_size_x, grid_size_y);
//   unsigned char *h_out_img = (unsigned char *)malloc(HEIGHT * WIDTH * sizeof(unsigned char));
//   start_time_data_transfer = omp_get_wtime();
//   cudaMalloc(&d_out_img, HEIGHT * WIDTH * sizeof(unsigned char));
//   cudaMalloc(&d_img, WIDTH_PADDED * HEIGHT_PADDED * sizeof(unsigned char));
//   cudaMalloc(&d_kernel, K_SIZE * K_SIZE * sizeof(float));
//   cudaMemcpy(d_img, img, WIDTH_PADDED * HEIGHT_PADDED * sizeof(unsigned char), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_kernel, kernel->get_kernel(), K_SIZE * K_SIZE * sizeof(float), cudaMemcpyHostToDevice);
//   // cudaMemcpyToSymbol(kernel_const, kernel->get_kernel(), K_SIZE * K_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
//   // printf("SHARED_grid_x:%d, grid_y:%d \n", grid_size_x, grid_size_y);
//   start_convolution = omp_get_wtime();
//   convolution_shared_memory<<<grid_dim, block_dim, sizeof(unsigned char) * (BLOCK_WIDTH + K_SIZE - 1) * (BLOCK_HEIGHT + K_SIZE - 1) + K_SIZE * K_SIZE * sizeof(float)>>>(
//       d_img, d_out_img, d_kernel, WIDTH_PADDED, HEIGHT_PADDED, K_SIZE, norm);
//   // applyKernel<<<grid_dim, block_dim, sizeof(unsigned char) * (BLOCK_WIDTH + K_SIZE - 1) * (BLOCK_WIDTH + K_SIZE - 1)>>>(d_img, d_out_img, WIDTH, HEIGHT, K_SIZE, norm);
//   end_convolution = omp_get_wtime();
//   cudaMemcpy(h_out_img, d_out_img, HEIGHT * WIDTH * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//   end_time_data_transfer = omp_get_wtime();
//   total_time += end_time_data_transfer - start_time_data_transfer;
//   kernel_time += end_convolution - start_convolution;
//   Image *out_img = new Image(h_out_img, WIDTH, HEIGHT);
//   // std::cout << "TOTAL_shared"
//   //           << " -> " << end_time_data_transfer - start_time_data_transfer - (end_convolution - start_convolution)
//   //           << " + " << end_convolution - start_convolution << " = "
//   //           << end_time_data_transfer - start_time_data_transfer << std::endl;
//   delete img_padded;
//   cudaFree(d_img);
//   cudaFree(d_out_img);
//   cudaFree(d_kernel);
//   return out_img;
// }
Image *ConvolutionGPU::apply_convolution_shared_constant_memory(Image *image, Kernel *kernel, type_padding padding) {
  unsigned char *d_out_img;
  unsigned char *d_img;
  // float *d_kernel;
  const int K_SIZE = kernel->get_size();
  Image *img_padded = PaddingImage::apply_padding_to_image(image, K_SIZE, padding);
  const int WIDTH = image->get_width();
  const int HEIGHT = image->get_height();
  const int WIDTH_PADDED = WIDTH + K_SIZE - 1;
  const int HEIGHT_PADDED = HEIGHT + K_SIZE - 1;
  const float norm = kernel->get_normalization_factor();
  const unsigned char *img = img_padded->get_image();
  double start_time_data_transfer;
  double end_time_data_transfer;
  double start_convolution;
  double end_convolution;
  int grid_size_x = (WIDTH_PADDED + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  int grid_size_y = (HEIGHT_PADDED + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
  dim3 block_dim(BLOCK_WIDTH, BLOCK_HEIGHT);
  dim3 grid_dim(grid_size_x, grid_size_y);
  unsigned char *h_out_img = (unsigned char *)malloc(HEIGHT * WIDTH * sizeof(unsigned char));
  start_time_data_transfer = omp_get_wtime();
  cudaMalloc(&d_out_img, HEIGHT * WIDTH * sizeof(unsigned char));
  cudaMalloc(&d_img, WIDTH_PADDED * HEIGHT_PADDED * sizeof(unsigned char));
  // cudaMalloc(&d_kernel, K_SIZE * K_SIZE * sizeof(float));
  cudaMemcpy(d_img, img, WIDTH_PADDED * HEIGHT_PADDED * sizeof(unsigned char), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_kernel, kernel->get_kernel(), K_SIZE * K_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(kernel_const, kernel->get_kernel(), K_SIZE * K_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
  // printf("SHARED_grid_x:%d, grid_y:%d \n", grid_size_x, grid_size_y);
  start_convolution = omp_get_wtime();
  convolution_shared_memory_constant<<<grid_dim, block_dim, sizeof(unsigned char) * (BLOCK_WIDTH + K_SIZE - 1) * (BLOCK_HEIGHT + K_SIZE - 1)>>>(d_img, d_out_img, WIDTH_PADDED, HEIGHT_PADDED, K_SIZE,
                                                                                                                                                norm);
  // applyKernel<<<grid_dim, block_dim, sizeof(unsigned char) * (BLOCK_WIDTH + K_SIZE - 1) * (BLOCK_WIDTH + K_SIZE - 1)>>>(d_img, d_out_img, WIDTH, HEIGHT, K_SIZE, norm);
  end_convolution = omp_get_wtime();
  cudaMemcpy(h_out_img, d_out_img, HEIGHT * WIDTH * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  end_time_data_transfer = omp_get_wtime();
  total_time += end_time_data_transfer - start_time_data_transfer;
  kernel_time += end_convolution - start_convolution;
  Image *out_img = new Image(h_out_img, WIDTH, HEIGHT);
  // std::cout << "TOTAL_shared"
  //           << " -> " << end_time_data_transfer - start_time_data_transfer - (end_convolution - start_convolution)
  //           << " + " << end_convolution - start_convolution << " = "
  //           << end_time_data_transfer - start_time_data_transfer << std::endl;
  delete img_padded;
  cudaFree(d_img);
  cudaFree(d_out_img);
  return out_img;
}