#ifndef IMAGE
#define IMAGE
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

class Image {
private:
  int width;
  int height;
  unsigned char *image;

public:
  Image(int width, int height);
  Image(int width, int height, unsigned int seed);
  Image(unsigned char *matrix, int width, int height);
  unsigned char &operator[](int index);
  int get_width();
  int get_height();
  unsigned char *get_image();
  ~Image();
};
#endif