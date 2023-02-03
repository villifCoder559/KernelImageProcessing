#include "padding_image.h"
Image *PaddingImage::apply_padding_to_image(Image *img, const int kernel_size, type_padding padding) {
  const int shift = kernel_size / 2;
  const int WIDTH = img->get_width() + kernel_size - 1;
  const int HEIGHT = img->get_height() + kernel_size - 1;
  Image *padded_img = new Image(WIDTH, HEIGHT);
  unsigned char *padded_img_matrix = padded_img->get_image();
  unsigned char *old_img_matrix = img->get_image();
  const int OLD_WIDTH = img->get_width();
  const int OLD_HEIGHT = img->get_height();
  switch (padding) {
  case zero:
    for (int i = 0; i < HEIGHT; i++) {
      for (int j = 0; j < WIDTH; j++) {
        padded_img_matrix[i * WIDTH + j] = 0;
      }
    }
    for (int i = 0; i < OLD_HEIGHT; i++)
      for (int j = 0; j < OLD_WIDTH; j++)
        padded_img_matrix[(i + shift) * WIDTH + j + shift] = old_img_matrix[i * OLD_WIDTH + j];
    break;
  // case pixel_replication:
  //   // #pragma omp parallel firstprivate(shift, OLD_HEIGHT, OLD_WIDTH, WIDTH, HEIGHT)
  //   //   {
  //   // copy all values
  //   // #pragma omp for
  //   for (int i = 0; i < OLD_HEIGHT; i++)
  //     for (int j = 0; j < OLD_WIDTH; j++)
  //       padded_img_matrix[(i + shift) * WIDTH + j + shift] = old_img_matrix[i * OLD_WIDTH + j];
  //   // copy top and bottom row
  //   // #pragma omp for collapse(2)
  //   for (int i = 0; i < shift; i++)
  //     for (int j = 0; j < OLD_WIDTH; j++) {
  //       padded_img_matrix[i * WIDTH + j + shift] = old_img_matrix[j];                                               //[0 * OLD_WIDTH + j]
  //       padded_img_matrix[(HEIGHT - 1 - i) * WIDTH + j + shift] = old_img_matrix[(OLD_HEIGHT - 1) * OLD_WIDTH + j]; //[(OLD_HEIGHT - 1 - 0) * OLD_WIDTH + j]
  //     }
  //   // copy right and left column
  //   // #pragma omp for collapse(2)
  //   for (int i = 0; i < OLD_HEIGHT; i++)
  //     for (int j = 0; j < shift; j++) {
  //       padded_img_matrix[(i + shift) * WIDTH + j] = old_img_matrix[i * OLD_WIDTH];                             // [i * OLD_WIDTH + 0]
  //       padded_img_matrix[(i + shift) * WIDTH + WIDTH - 1 - j] = old_img_matrix[i * OLD_WIDTH + OLD_WIDTH - 1]; //[i * OLD_WIDTH + OLD_WIDTH - 1 - 0]
  //     }
  //   // copy corners top-left,top-right,bottom-left,bottom-right
  //   for (int i = 0; i < shift; i++)
  //     for (int j = 0; j < shift; j++) {
  //       padded_img_matrix[i * WIDTH + j] = old_img_matrix[0];                                                                           // [0 * OLD_WIDTH + 0]
  //       padded_img_matrix[i * WIDTH + WIDTH - 1 - j] = old_img_matrix[OLD_WIDTH - 1];                                                   // [0 * OLD_WIDTH + OLD_WIDTH - j - 0]
  //       padded_img_matrix[(HEIGHT - 1 - i) * WIDTH + j] = old_img_matrix[(OLD_HEIGHT - 1) * OLD_WIDTH];                                 // [(OLD_HEIGHT - 0 - 1) * OLD_WIDTH + 0]
  //       padded_img_matrix[(HEIGHT - 1 - i) * WIDTH + (WIDTH - 1 - j)] = old_img_matrix[(OLD_HEIGHT - 1) * OLD_WIDTH + (OLD_WIDTH - 1)]; // [(OLD_HEIGHT - 0 - 1) * OLD_WIDTH + (OLD_WIDTH - 0 - 1)]
  //                                                                                                                                       // }
  //     }
  //   break;
  // case pixel_mirroring:
  //   // #pragma omp parallel firstprivate(shift, OLD_HEIGHT, OLD_WIDTH, WIDTH, HEIGHT)
  //   //   {
  //   // copy all values
  //   // #pragma omp for
  //   for (int i = 0; i < OLD_HEIGHT; i++)
  //     for (int j = 0; j < OLD_WIDTH; j++)
  //       padded_img_matrix[(i + shift) * WIDTH + j + shift] = old_img_matrix[i * OLD_WIDTH + j];
  //   // copy top and bottom row
  //   // #pragma omp for collapse(2)
  //   for (int i = 0; i < shift; i++)
  //     for (int j = 0; j < OLD_WIDTH; j++) {
  //       padded_img_matrix[i * WIDTH + j + shift] = old_img_matrix[(shift - i) * OLD_WIDTH + j];                                   //[0 * OLD_WIDTH + j]
  //       padded_img_matrix[(HEIGHT - 1 - i) * WIDTH + j + shift] = old_img_matrix[(OLD_HEIGHT - 1 - (shift - i)) * OLD_WIDTH + j]; //[(OLD_HEIGHT - 1 - 0) * OLD_WIDTH + j]
  //     }
  //   // copy right and left column
  //   // #pragma omp for collapse(2)
  //   for (int i = 0; i < OLD_HEIGHT; i++)
  //     for (int j = 0; j < shift; j++) {
  //       padded_img_matrix[(i + shift) * WIDTH + j] = old_img_matrix[i * OLD_WIDTH + (shift - i)];                             // [i * OLD_WIDTH + 0]
  //       padded_img_matrix[(i + shift) * WIDTH + WIDTH - 1 - j] = old_img_matrix[i * OLD_WIDTH + OLD_WIDTH - 1 - (shift - i)]; //[i * OLD_WIDTH + OLD_WIDTH - 1 - 0]
  //     }
  //   // copy corners top-left,top-right,bottom-left,bottom-right
  //   for (int i = 0; i < shift; i++)
  //     for (int j = 0; j < shift; j++) {
  //       padded_img_matrix[i * WIDTH + j] = old_img_matrix[(shift - i) * OLD_WIDTH + (shift - j)];                                   // [0 * OLD_WIDTH + 0]
  //       padded_img_matrix[i * WIDTH + WIDTH - 1 - j] = old_img_matrix[(shift - i) * OLD_WIDTH + OLD_WIDTH - j - (shift - j)];       // [0 * OLD_WIDTH + OLD_WIDTH - j - 0]
  //       padded_img_matrix[(HEIGHT - 1 - i) * WIDTH + j] = old_img_matrix[(OLD_HEIGHT - (shift - i) - 1) * OLD_WIDTH + (shift - j)]; // [(OLD_HEIGHT - 0 - 1) * OLD_WIDTH + 0]
  //       padded_img_matrix[(HEIGHT - 1 - i) * WIDTH + (WIDTH - 1 - j)] =
  //           old_img_matrix[(OLD_HEIGHT - (shift - i) - 1) * OLD_WIDTH + (OLD_WIDTH - (shift - j) - 1)]; // [(OLD_HEIGHT - 0 - 1) * OLD_WIDTH + (OLD_WIDTH - 0 - 1)]
  //                                                                                                       // }
  //     }
  //   break;
  case pixel_mirroring:
  case pixel_replication:
    // #pragma omp parallel firstprivate(shift, OLD_HEIGHT, OLD_WIDTH, WIDTH, HEIGHT)
    //   {
    // copy all values
    // #pragma omp for
    for (int i = 0; i < OLD_HEIGHT; i++)
      for (int j = 0; j < OLD_WIDTH; j++)
        padded_img_matrix[(i + shift) * WIDTH + j + shift] = old_img_matrix[i * OLD_WIDTH + j];
    // copy top and bottom row
    // #pragma omp for collapse(2)
    for (int i = 0; i < shift; i++)
      for (int j = 0; j < OLD_WIDTH; j++) {
        int shift_padding_row = padding == pixel_mirroring ? shift - i : 0;
        padded_img_matrix[i * WIDTH + j + shift] = old_img_matrix[(shift_padding_row)*OLD_WIDTH + j];                                     //[0 * OLD_WIDTH + j]
        padded_img_matrix[(HEIGHT - 1 - i) * WIDTH + j + shift] = old_img_matrix[(OLD_HEIGHT - 1 - (shift_padding_row)) * OLD_WIDTH + j]; //[(OLD_HEIGHT - 1 - 0) * OLD_WIDTH + j]
      }
    // copy right and left column
    // #pragma omp for collapse(2)
    for (int i = 0; i < OLD_HEIGHT; i++)
      for (int j = 0; j < shift; j++) {
        int shift_padding_col = padding == pixel_mirroring ? shift - j : 0;
        padded_img_matrix[(i + shift) * WIDTH + j] = old_img_matrix[i * OLD_WIDTH + (shift_padding_col)];                             // [i * OLD_WIDTH + 0]
        padded_img_matrix[(i + shift) * WIDTH + WIDTH - 1 - j] = old_img_matrix[i * OLD_WIDTH + OLD_WIDTH - 1 - (shift_padding_col)]; //[i * OLD_WIDTH + OLD_WIDTH - 1 - 0]
      }
    // copy corners top-left,top-right,bottom-left,bottom-right
    if (padding == pixel_mirroring)
      for (int i = 0; i < shift; i++)
        for (int j = 0; j < shift; j++) {
          int shift_padding_row = shift - i;
          int shift_padding_col = shift - j;
          padded_img_matrix[i * WIDTH + j] = old_img_matrix[(shift_padding_row)*OLD_WIDTH + (shift_padding_col)];                                   // [0 * OLD_WIDTH + 0]
          padded_img_matrix[i * WIDTH + WIDTH - 1 - j] = old_img_matrix[(shift_padding_row)*OLD_WIDTH + OLD_WIDTH - (shift_padding_col)-1];         // [0 * OLD_WIDTH + OLD_WIDTH - j - 0]
          padded_img_matrix[(HEIGHT - 1 - i) * WIDTH + j] = old_img_matrix[(OLD_HEIGHT - (shift_padding_row)-1) * OLD_WIDTH + (shift_padding_col)]; // [(OLD_HEIGHT - 0 - 1) * OLD_WIDTH + 0]
          padded_img_matrix[(HEIGHT - 1 - i) * WIDTH + (WIDTH - 1 - j)] =
              old_img_matrix[(OLD_HEIGHT - (shift_padding_row)-1) * OLD_WIDTH + (OLD_WIDTH - (shift_padding_col)-1)]; // [(OLD_HEIGHT - 0 - 1) * OLD_WIDTH + (OLD_WIDTH - 0 - 1)]
        }
    else
      for (int i = 0; i < shift; i++)
        for (int j = 0; j < shift; j++) {
          padded_img_matrix[i * WIDTH + j] = old_img_matrix[0];                                                                           // [0 * OLD_WIDTH + 0]
          padded_img_matrix[i * WIDTH + WIDTH - 1 - j] = old_img_matrix[OLD_WIDTH - 1];                                                   // [0 * OLD_WIDTH + OLD_WIDTH - j - 0]
          padded_img_matrix[(HEIGHT - 1 - i) * WIDTH + j] = old_img_matrix[(OLD_HEIGHT - 1) * OLD_WIDTH];                                 // [(OLD_HEIGHT - 0 - 1) * OLD_WIDTH + 0]
          padded_img_matrix[(HEIGHT - 1 - i) * WIDTH + (WIDTH - 1 - j)] = old_img_matrix[(OLD_HEIGHT - 1) * OLD_WIDTH + (OLD_WIDTH - 1)]; // [(OLD_HEIGHT - 0 - 1) * OLD_WIDTH + (OLD_WIDTH - 0 - 1)]
        }
    break;
  }
  return padded_img;
}