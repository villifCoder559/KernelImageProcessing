#ifndef KERNEL
#define KERNEL

enum type_kernel { edge_detection, sharpen, constant_blur, gaussian_blur_3x3, gaussian_blur_5x5, gaussian_blur_7x7 };

class Kernel {
private:
  int size;
  float *kernel;
  float normalization_factor;

public:
  Kernel(type_kernel type);
  /*
    The kernel is squared
  */
  int get_size();
  float get_normalization_factor();
  float *get_kernel();
  ~Kernel();
};
#endif