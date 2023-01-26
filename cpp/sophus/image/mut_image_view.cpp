// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/mut_image_view.h"

namespace sophus {

namespace details {

void pitchedCopy(
    uint8_t* dst,
    size_t dst_pitch_bytes,
    uint8_t const* src,
    size_t src_pitch_bytes,
    sophus::ImageSize size,
    uint8_t size_of_pixel) {
  size_t width_bytes = size.width * size_of_pixel;
#if 0
  // once we have CUDA support
  cudaMemcpy2D(
      dst,
      dst_pitch_bytes,
      src,
      src_pitch_bytes,
      width_bytes,
      size.height,
      cudaMemcpyDefault);
#else
  if (dst_pitch_bytes == width_bytes && src_pitch_bytes == width_bytes) {
    std::memcpy(dst, src, size.height * width_bytes);
  } else {
    for (int row = 0; row < size.height; ++row) {
      std::memcpy(dst, src, width_bytes);
      dst += dst_pitch_bytes;
      src += src_pitch_bytes;
    }
  }
#endif
}
}  // namespace details

}  // namespace sophus
