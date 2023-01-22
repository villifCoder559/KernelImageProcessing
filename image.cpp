#include "image.h"

Image::Image(int width, int height) {
  this->width = width;
  this->height = height;
  image = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
}
Image::Image(int width, int height, unsigned int seed) {
  this->width = width;
  this->height = height;
  image = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
  seed = seed ? seed : (unsigned int)time(NULL);
  printf("seed: %d \n", seed);
  srand(seed);
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      image[i * width + j] = rand() % 256;
}
Image::Image(unsigned char *matrix, int width, int height) {
  image = matrix;
  this->width = width;
  this->height = height;
}
unsigned char &Image::operator[](int index) {
  if (index >= width * height)
    exit(0);
  return image[index];
}
int Image::get_height() { return height; }
int Image::get_width() { return width; }
unsigned char *Image::get_image() { return image; }
Image::~Image() { delete[] image; }