#ifndef PAD_IMG
#define PAD_IMG
#include "image.h"
#include "omp.h"
enum type_padding { zero, pixel_replication, pixel_mirroring };

class PaddingImage {
public:
  PaddingImage(){};
  static Image *apply_padding_to_image(Image *img, const int kernel_size, type_padding padding);
  ~PaddingImage(){};
};
#endif