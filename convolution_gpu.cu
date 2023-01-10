#include "convolution_gpu.h"
__constant__ float kernel_const[MAX_SIZE_KERNEL * MAX_SIZE_KERNEL]; // 48KB max size [202,202]

__global__ void convolution_base(const float *d_kernel, unsigned char *out_img, unsigned char *d_img, const int WIDTH, const int HEIGHT, const int K_SIZE) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  float sum = 0.0f;
  if (col < WIDTH && row < HEIGHT) {
    for (int i = 0; i < K_SIZE; i++) {
      for (int j = 0; j < K_SIZE; j++) {
        int index_r = row - K_SIZE / 2 + i;
        int index_c = col - K_SIZE / 2 + j;
        if (index_r >= 0 && index_r < HEIGHT && index_c >= 0 && index_c < WIDTH) {
          sum += d_img[index_r * WIDTH + index_c] * d_kernel[i * K_SIZE + j];
          // if (row == 190 && col == 0)
          //   printf("B(%d,%d):[%d_%d]->%d*%f \n", blockIdx.x, blockIdx.y, index_r, index_c, d_img[index_r * WIDTH + index_c], d_kernel[i * K_SIZE + j]);
        }
      }
    }
    // if (row == 190 && col == 0)
    //   printf("SUM: %f \n", sum);
    sum = fmaxf(0.0f, fminf(sum, 255.0f));
    out_img[row * WIDTH + col] = (unsigned char)sum;
  }
}
__global__ void convolution_constant_memory(unsigned char *out_img, unsigned char *d_img, const int WIDTH, const int HEIGHT, const int K_SIZE) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  // int tid_block = threadIdx.x + threadIdx.y * BLOCK_WIDTH;
  float sum = 0.0f;
  if (col < WIDTH && row < HEIGHT) {
    for (int i = 0; i < K_SIZE; i++) {
      for (int j = 0; j < K_SIZE; j++) {
        int index_r = row - K_SIZE / 2 + i;
        int index_c = col - K_SIZE / 2 + j;
        if (index_r >= 0 && index_r < HEIGHT && index_c >= 0 && index_c < WIDTH) {
          sum += d_img[index_r * WIDTH + index_c] * kernel_const[i * K_SIZE + j];
        }
      }
    }
    sum = fmaxf(0.0f, fminf(sum, 255.0f));
    out_img[row * WIDTH + col] = (unsigned char)sum;
  }
}
__global__ void convolution_shared_memory(const float *d_kernel, unsigned char *img, unsigned char *out_img, const int WIDTH, const int HEIGHT, const int K_SIZE) {
  extern __shared__ float dynamic_array[];
  const int size_d_img = (BLOCK_WIDTH + K_SIZE - 1);
  float *mask = &dynamic_array[size_d_img * size_d_img];
  unsigned char *d_img = (unsigned char *)&dynamic_array;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int c = threadIdx.x;
  int r = threadIdx.y;
  int tid_block = threadIdx.x + threadIdx.y * BLOCK_WIDTH;
  int shift = K_SIZE / 2;
  // initialize d_img
  d_img[(r + shift) * size_d_img + c + shift] = 0;
  // init row above
  if ((r < shift)) {
    d_img[r * size_d_img + c + shift] = 0;
  }
  // init row below
  if (r > BLOCK_WIDTH - shift - 1) {
    d_img[(r + K_SIZE - 1) * size_d_img + c + shift] = 0;
  }
  // init colomun left
  if ((c < shift)) {
    d_img[(r + shift) * size_d_img + c] = 0;
  }
  // init column right
  if ((c > BLOCK_WIDTH - shift - 1)) {
    d_img[(r + shift) * size_d_img + c + K_SIZE - 1] = 0;
  }
  // init corner top right
  if ((c == BLOCK_WIDTH - 1) && (r == BLOCK_WIDTH - 1)) {
    d_img[(r + K_SIZE - 1) * size_d_img + c + K_SIZE - 1] = 0;
  }
  // init corner top left
  if ((c == 0) && (r == 0)) {
    d_img[r * size_d_img + c] = 0;
  }
  // init corner bottom left
  if ((c == 0) && (r == BLOCK_WIDTH - 1)) {
    d_img[(r + K_SIZE - 1) * size_d_img + c] = 0;
  }
  // init corner bottom right
  if ((c == BLOCK_WIDTH - 1) && (r == 0)) {
    d_img[r * size_d_img + c + K_SIZE - 1] = 0;
  }
  __syncthreads();
  if (col < WIDTH && row < HEIGHT) {
    d_img[(r + shift) * size_d_img + c + shift] = img[col + WIDTH * row];
    // check if exists row above
    if ((row - shift >= 0 && r < shift)) {
      // printf("A(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row - shift, col - shift);
      d_img[r * size_d_img + c + shift] = img[col + (row - shift) * WIDTH];
    }
    // row below
    if (r > BLOCK_WIDTH - shift - 1 && (row + shift) < HEIGHT) {
      d_img[(r + K_SIZE - 1) * size_d_img + c + shift] = img[(col) + (row + shift) * WIDTH];
      // printf("B(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row + shift, col);
    }
    // colomun left
    if ((c < shift) && (col - shift >= 0)) {
      d_img[(r + shift) * size_d_img + c] = img[col - shift + (row)*WIDTH];
      // printf("C(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row, col - shift);
    }
    // column right
    if ((c > BLOCK_WIDTH - shift - 1) && (col + shift < WIDTH)) {
      d_img[(r + shift) * size_d_img + c + K_SIZE - 1] = img[col + shift + (row)*WIDTH];
      // printf("D(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row, col + shift);
    }
    // corner top right
    if ((c == BLOCK_WIDTH - 1) && (r == BLOCK_WIDTH - 1) && row + shift < HEIGHT && col + shift < WIDTH) {
      d_img[(r + K_SIZE - 1) * size_d_img + c + K_SIZE - 1] = img[col + shift + (row + shift) * WIDTH];
      // printf("E(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row + shift, col + shift);
    }
    // corner top left
    if ((c == 0) && (r == 0) && row - shift >= 0 && col - shift >= 0) {
      d_img[r * size_d_img + c] = img[col - shift + (row - shift) * WIDTH];
      // printf("F(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row - shift, col - shift);
    }
    // corner bottom left
    if ((c == 0) && (r == BLOCK_WIDTH - 1) && row + shift < WIDTH && col - shift >= 0) {
      d_img[(r + K_SIZE - 1) * size_d_img + c] = img[col - shift + (row + shift) * WIDTH];
      // printf("G(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row + shift, col - shift);
    }
    // corner bottom right
    if ((c == BLOCK_WIDTH - 1) && (r == 0) && row - shift >= 0 && col + shift < WIDTH) {
      d_img[r * size_d_img + c + K_SIZE - 1] = img[col + shift + (row - shift) * WIDTH];
      // printf("H(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row - shift, col + shift);
    }
  }
  if (tid_block < K_SIZE * K_SIZE) {
    mask[tid_block] = d_kernel[tid_block];
  }
  float sum = 0.0f;
  __syncthreads();
  // printf("%d \t", blockIdx.x, blockIdx.y, r, c, d_img[r * WIDTH + c]);
  if (col < WIDTH && row < HEIGHT) {
    for (int i = 0; i < K_SIZE; i++) {
      for (int j = 0; j < K_SIZE; j++) {
        int index_r = r + i;
        int index_c = c + j;
        // if (index_r < HEIGHT && index_c < WIDTH) {
        sum += d_img[index_r * size_d_img + index_c] * mask[i * K_SIZE + j];
        // if (row == 0 && col == 0)
        //   printf("(%d,%d):[%d_%d]->%d*%f \n", blockIdx.x, blockIdx.y, index_r, index_c, d_img[index_r * size_d_img + index_c], mask[i * K_SIZE + j]);
        // }
      }
    }
    // if (row == 190 && col == 0)
    //   printf("SUM: %f \n", sum);
    sum = fmaxf(0.0f, fminf(sum, 255.0f));
    out_img[row * WIDTH + col] = (unsigned char)sum;
  }
}

Image *ConvolutionGPU::apply_convolution_constant_memory(Image *image, Kernel *kernel) {
  unsigned char *d_out_img;
  unsigned char *d_image;
  const int WIDTH = image->get_width();
  const int HEIGHT = image->get_height();
  const int K_SIZE = kernel->get_size();
  const unsigned char *img = image->get_image();
  cudaMalloc(&d_out_img, WIDTH * HEIGHT * sizeof(unsigned char));
  cudaMalloc(&d_image, WIDTH * HEIGHT * sizeof(unsigned char));
  cudaMemcpy(d_image, img, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(kernel_const, kernel->get_kernel(), K_SIZE * K_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
  int grid_size_x = (WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  int grid_size_y = (HEIGHT + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH);
  dim3 grid_dim(grid_size_x, grid_size_y);

  double start = omp_get_wtime();
  convolution_constant_memory<<<grid_dim, block_dim>>>(d_out_img, d_image, WIDTH, HEIGHT, K_SIZE);
  double end = omp_get_wtime();
  printf("TIME_GPU_constant: %f \n", end - start);
  // unsigned char h_out_img[DIM*DIM];
  unsigned char *h_out_img = (unsigned char *)malloc(sizeof(unsigned char) * WIDTH * HEIGHT);
  cudaMemcpy(h_out_img, d_out_img, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  Image *out_img = new Image(h_out_img, WIDTH, HEIGHT);
  cudaFree(d_image);
  cudaFree(d_out_img);
  return out_img;
}
Image *ConvolutionGPU::apply_convolution_base(Image *image, Kernel *kernel) {
  unsigned char *d_out_img;
  float *d_kernel;
  unsigned char *d_img;
  const int WIDTH = image->get_width();
  const int HEIGHT = image->get_height();
  const int K_SIZE = kernel->get_size();
  const unsigned char *img = image->get_image();
  cudaMalloc(&d_kernel, K_SIZE * K_SIZE * sizeof(float));
  cudaMalloc(&d_img, WIDTH * HEIGHT * sizeof(unsigned char));
  cudaMalloc(&d_out_img, WIDTH * HEIGHT * sizeof(unsigned char));
  cudaMemcpy(d_img, img, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel->get_kernel(), K_SIZE * K_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  int grid_size_x = (WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  int grid_size_y = (HEIGHT + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH);
  dim3 grid_dim(grid_size_x, grid_size_y);

  double start = omp_get_wtime();
  convolution_base<<<grid_dim, block_dim>>>(d_kernel, d_out_img, d_img, WIDTH, HEIGHT, K_SIZE);
  double end = omp_get_wtime();
  printf("TIME_GPU_base: %f \n", end - start);
  // unsigned char h_out_img[DIM*DIM];
  unsigned char *h_out_img = (unsigned char *)malloc(sizeof(unsigned char) * WIDTH * HEIGHT);
  cudaMemcpy(h_out_img, d_out_img, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  Image *out_img = new Image(h_out_img, WIDTH, HEIGHT);
  cudaFree(d_img);
  cudaFree(d_kernel);
  cudaFree(d_out_img);
  return out_img;
}
Image *ConvolutionGPU::apply_convolution_shared_memory(Image *image, Kernel *kernel) {
  unsigned char *d_out_img;
  float *d_kernel;
  unsigned char *d_img;
  const int WIDTH = image->get_width();
  const int HEIGHT = image->get_height();
  const int K_SIZE = kernel->get_size();
  const unsigned char *img = image->get_image();
  cudaMalloc(&d_kernel, K_SIZE * K_SIZE * sizeof(float));
  cudaMalloc(&d_out_img, WIDTH * HEIGHT * sizeof(unsigned char));
  cudaMalloc(&d_img, WIDTH * HEIGHT * sizeof(unsigned int));
  cudaMemcpy(d_img, img, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel->get_kernel(), K_SIZE * K_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  int grid_size_x = (WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  int grid_size_y = (HEIGHT + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH);
  dim3 grid_dim(grid_size_x, grid_size_y);

  double start = omp_get_wtime();
  convolution_shared_memory<<<grid_dim, block_dim, sizeof(float) * K_SIZE * K_SIZE + sizeof(unsigned char) * (BLOCK_WIDTH + K_SIZE - 1) * (BLOCK_WIDTH + K_SIZE - 1)>>>(d_kernel, d_img, d_out_img,
                                                                                                                                                                        WIDTH, HEIGHT, K_SIZE);
  double end = omp_get_wtime();
  printf("TIME_GPU_shared: %f \n", end - start);
  // unsigned char h_out_img[DIM*DIM];
  unsigned char *h_out_img = (unsigned char *)malloc(sizeof(unsigned char) * WIDTH * HEIGHT);
  cudaMemcpy(h_out_img, d_out_img, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  Image *out_img = new Image(h_out_img, WIDTH, HEIGHT);
  cudaFree(d_img);
  cudaFree(d_kernel);
  cudaFree(d_out_img);
  return out_img;
}
