#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DIM 300
#define K_size 3
#define BLOCK_WIDTH 16

/*
  1) Create kernel that uses shared memory OK
  2) Manage data with size greater than constant memory OK
  3) Try to increse the size of constant memory if possible (control if there is a difference between shared/constant memory)
  3.1) Adapt algorithm to non-squared images
  4) Create more filters
  5) Manage the case where the image is too big to fit the constant/shared memory
  6) Measure divergence and general performance
*/

__constant__ unsigned char d_image[DIM][2];

__global__ void convolution_base(float *d_kernel, unsigned char *out_img, unsigned char *d_img) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  float sum = 0.0f;
  if (col < DIM && row < DIM) {
    for (int i = 0; i < K_size; i++) {
      for (int j = 0; j < K_size; j++) {
        int index_r = row - K_size / 2 + i;
        int index_c = col - K_size / 2 + j;
        if (index_r >= 0 && index_r < DIM && index_c >= 0 && index_c < DIM) {
          sum += d_img[index_r * DIM + index_c] * d_kernel[i * K_size + j];
          // if (row == 16 && col == 0)
          //   printf("B(%d,%d):[%d_%d]->%d*%f \n", blockIdx.x, blockIdx.y, index_r, index_c, d_img[index_r * DIM + index_c], d_kernel[i * K_size + j]);
        }
      }
    }
    // if (row == 16 && col == 0)
    //   printf("SUM: %f \n", sum);
    out_img[row * DIM + col] = (unsigned char)sum;
  }
}

__global__ void convolution_constant_memory(float *d_kernel, unsigned char *out_img) {
  __shared__ float mask[K_size * K_size];
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int tid_block = threadIdx.x + threadIdx.y * BLOCK_WIDTH;
  if (tid_block < K_size * K_size) {
    mask[tid_block] = d_kernel[tid_block];
  }
  float sum = 0.0f;
  __syncthreads();
  if (col < DIM && row < DIM) {
    for (int i = 0; i < K_size; i++) {
      for (int j = 0; j < K_size; j++) {
        int index_r = row - K_size / 2 + i;
        int index_c = col - K_size / 2 + j;
        if (index_r >= 0 && index_r < DIM && index_c >= 0 && index_c < DIM) {
          sum += d_image[index_r][index_c] * mask[i * K_size + j];
        }
      }
    }
    out_img[row * DIM + col] = (unsigned char)sum;
  }
}

__global__ void convolution_shared_memory(float *d_kernel, unsigned char *img, unsigned char *out_img) {
  __shared__ float mask[K_size * K_size];
  __shared__ unsigned char d_img[BLOCK_WIDTH + K_size - 1][BLOCK_WIDTH + K_size - 1];
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int c = threadIdx.x;
  int r = threadIdx.y;
  int tid_block = threadIdx.x + threadIdx.y * BLOCK_WIDTH;
  if (col < DIM && row < DIM) {
    d_img[r + K_size / 2][c + K_size / 2] = img[col + DIM * row];
    // check if exists row above
    if ((row - K_size / 2 >= 0 && r < K_size / 2)) {
      // printf("A(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row - K_size / 2, col - K_size / 2);
      d_img[r][c + K_size / 2] = img[col + (row - K_size / 2) * DIM];
    }
    // row below
    if (r > BLOCK_WIDTH - K_size / 2 - 1 && (row + K_size / 2) < DIM) {
      // printf("B(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row + K_size / 2, col);
      d_img[r + K_size - 1][c + K_size / 2] = img[(col) + (row + K_size / 2) * DIM];
    }
    // colomun left
    if ((c < K_size / 2) && (col - K_size / 2 >= 0)) {
      // printf("C(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row, col - K_size / 2);
      d_img[r + K_size / 2][c] = img[col - K_size / 2 + (row)*DIM];
    }
    // column right
    if ((c > BLOCK_WIDTH - K_size / 2 - 1) && (col + K_size / 2 < DIM)) {
      // printf("D(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row, col + K_size / 2);
      d_img[r + K_size / 2][c + K_size - 1] = img[col + K_size / 2 + (row)*DIM];
    }
    // corner top right
    if ((c == BLOCK_WIDTH - 1) && (r == BLOCK_WIDTH - 1) && row + K_size / 2 < DIM && col + K_size / 2 < DIM) {
      d_img[r + K_size - 1][c + K_size - 1] = img[col + K_size / 2 + (row + K_size / 2) * DIM];
      // printf("E(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row + K_size / 2, col + K_size / 2);
    }
    // corner top left
    if ((c == 0) && (r == 0) && row - K_size / 2 >= 0 && col - K_size / 2 >= 0) {
      d_img[r][c] = img[col + -K_size / 2 + (row - K_size / 2) * DIM];
      // printf("F(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row - K_size / 2, col - K_size / 2);
    }
    // corner bottom left
    if ((c == 0) && (r == BLOCK_WIDTH - 1) && row + K_size / 2 < DIM && col - K_size / 2 >= 0) {
      d_img[r + K_size - 1][c] = img[col - K_size / 2 + (row + K_size / 2) * DIM];
      // printf("G(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row + K_size / 2, col - K_size / 2);
    }
    // corner bottom right
    if ((c == BLOCK_WIDTH - 1) && (r == 0) && row - K_size / 2 >= 0 && col + K_size / 2 < DIM) {
      d_img[r][c + K_size - 1] = img[col + K_size / 2 + (row - K_size / 2) * DIM];
      // printf("H(%d,%d):%d_%d->%d_%d ", blockIdx.x, blockIdx.y, r, c, row - K_size / 2, col + K_size / 2);
    }
  }
  if (tid_block < K_size * K_size) {
    mask[tid_block] = d_kernel[tid_block];
  }
  __syncthreads();
  float sum = 0.0f;
  if (col < DIM && row < DIM) {
    for (int i = 0; i < K_size; i++) {
      for (int j = 0; j < K_size; j++) {
        int index_r = r + i;
        int index_c = c + j;
        if (index_r < DIM && index_c < DIM) {
          sum += d_img[index_r][index_c] * mask[i * K_size + j];
          // if (row == 16 && col == 0)
          //   printf("(%d,%d):[%d_%d]->%d*%f \n", blockIdx.x, blockIdx.y, index_r, index_c, d_img[index_r][index_c], mask[i * K_size + j]);
        }
      }
    }
    // if (row == 16 && col == 0)
    //   printf("SUM: %f \n", sum);
    out_img[row * DIM + col] = (unsigned char)sum;
  }
}
unsigned char *apply_convolution_constant_memory(unsigned char (&image)[DIM * DIM], float (&kernel)[K_size * K_size]) {
  unsigned char *d_out_img;
  float *d_kernel;
  cudaMalloc(&d_kernel, K_size * K_size * sizeof(float));
  cudaMalloc(&d_out_img, DIM * DIM * sizeof(unsigned char));
  cudaMemcpyToSymbol(d_image, &image, DIM * DIM * sizeof(unsigned char), 0, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, K_size * K_size * sizeof(float), cudaMemcpyHostToDevice);
  int grid_size = (DIM + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH);
  dim3 grid_dim(grid_size, grid_size);

  double start = omp_get_wtime();
  convolution_constant_memory<<<grid_dim, block_dim>>>(d_kernel, d_out_img);
  double end = omp_get_wtime();
  printf("TIME_GPU_efficient: %f \n", end - start);
  // unsigned char h_out_img[DIM*DIM];
  unsigned char *h_out_img = (unsigned char *)malloc(sizeof(unsigned char) * DIM * DIM);
  cudaMemcpy(h_out_img, d_out_img, DIM * DIM * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaFree(d_image);
  cudaFree(d_kernel);
  cudaFree(d_out_img);
  return h_out_img;
}
unsigned char *apply_convolution_base(unsigned char (&image)[DIM * DIM], float (&kernel)[K_size * K_size]) {
  unsigned char *d_out_img;
  float *d_kernel;
  unsigned char *d_img;
  cudaMalloc(&d_kernel, K_size * K_size * sizeof(float));
  cudaMalloc(&d_img, DIM * DIM * sizeof(unsigned char));
  cudaMalloc(&d_out_img, DIM * DIM * sizeof(unsigned char));
  cudaMemcpy(d_img, image, DIM * DIM * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, K_size * K_size * sizeof(float), cudaMemcpyHostToDevice);
  int grid_size = (DIM + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH);
  dim3 grid_dim(grid_size, grid_size);

  double start = omp_get_wtime();
  convolution_base<<<grid_dim, block_dim>>>(d_kernel, d_out_img, d_img);
  double end = omp_get_wtime();
  printf("TIME_GPU_base: %f \n", end - start);
  // unsigned char h_out_img[DIM*DIM];
  unsigned char *h_out_img = (unsigned char *)malloc(sizeof(unsigned char) * DIM * DIM);
  cudaMemcpy(h_out_img, d_out_img, DIM * DIM * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaFree(d_img);
  cudaFree(d_kernel);
  cudaFree(d_out_img);
  return h_out_img;
}
unsigned char *apply_convolution_shared_memory(unsigned char (&image)[DIM * DIM], float (&kernel)[K_size * K_size]) {
  unsigned char *d_out_img;
  float *d_kernel;
  unsigned char *d_img;
  cudaMalloc(&d_kernel, K_size * K_size * sizeof(float));
  cudaMalloc(&d_out_img, DIM * DIM * sizeof(unsigned char));
  cudaMalloc(&d_img, DIM * DIM * sizeof(unsigned int));
  cudaMemcpy(d_img, image, DIM * DIM * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, K_size * K_size * sizeof(float), cudaMemcpyHostToDevice);
  int grid_size = (DIM + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH);
  dim3 grid_dim(grid_size, grid_size);

  double start = omp_get_wtime();
  convolution_shared_memory<<<grid_dim, block_dim>>>(d_kernel, d_img, d_out_img);
  double end = omp_get_wtime();
  printf("TIME_GPU_shared: %f \n", end - start);
  // unsigned char h_out_img[DIM*DIM];
  unsigned char *h_out_img = (unsigned char *)malloc(sizeof(unsigned char) * DIM * DIM);
  cudaMemcpy(h_out_img, d_out_img, DIM * DIM * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaFree(d_image);
  cudaFree(d_kernel);
  cudaFree(d_out_img);
  return h_out_img;
}

int main() {
  unsigned char image[DIM * DIM];
  float kernel[K_size * K_size];
  unsigned int seed = time(NULL);
  printf("%d \n", seed);
  srand(seed);
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      int value = rand() % 255;
      image[i * DIM + j] = value;
      // printf("[%d,%d]=%d ", i, j, value);
    }
    // printf("\n");
  }
  for (int i = 0; i < K_size; i++) // remember to flip matrix to compute convolution
    for (int j = 0; j < K_size; j++)
      kernel[i * K_size + j] = 0.33f;
  unsigned char *result_base = apply_convolution_base(image, kernel);
  unsigned char *result_efficient = apply_convolution_shared_memory(image, kernel);
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++)
      if (result_base[i * DIM + j] != result_efficient[i * DIM + j])
        printf("%d!=%d[%d,%d] ", result_base[i * DIM + j], result_efficient[i * DIM + j], i, j);
  return 0;
}