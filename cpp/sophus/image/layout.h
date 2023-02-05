// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/common.h"
#include "sophus/image/image_size.h"

#include <Eigen/Dense>

#include <iostream>

namespace sophus {

/// Layout of the image: width, height and pitch in bytes.
///
class ImageLayout {
 public:
  ImageLayout() = default;

  ImageLayout(ImageSize image_size, size_t pitch_bytes)
      : image_size_(image_size), pitch_bytes_(pitch_bytes) {}

  ImageLayout(int width, int height, size_t pitch_bytes)
      : image_size_(width, height), pitch_bytes_(pitch_bytes) {}

  template <class TPixelFormat>
  [[nodiscard]] static ImageLayout makeFromSizeAndPitch(
      ImageSize image_size, size_t pitch_bytes) {
    SOPHUS_ASSERT_GE(pitch_bytes, image_size.width * sizeof(TPixelFormat));
    ImageLayout layout;
    layout.image_size_ = image_size;
    layout.pitch_bytes_ = pitch_bytes;

    return layout;
  }

  [[nodiscard]] static ImageLayout makeFromSizeAndPitchUnchecked(
      ImageSize image_size, size_t pitch_bytes) {
    ImageLayout layout;
    layout.image_size_ = image_size;
    layout.pitch_bytes_ = pitch_bytes;
    return layout;
  }

  template <class TpixelFormatT>
  [[nodiscard]] static ImageLayout makeFromSize(sophus::ImageSize image_size) {
    return makeFromSizeAndPitch<TpixelFormatT>(
        image_size, image_size.width * sizeof(TpixelFormatT));
  }

  [[nodiscard]] sophus::ImageSize const& imageSize() const {
    return image_size_;
  }
  [[nodiscard]] int width() const { return image_size_.width; }
  [[nodiscard]] int height() const { return image_size_.height; }
  [[nodiscard]] size_t pitchBytes() const { return pitch_bytes_; }

  [[nodiscard]] int area() const { return this->width() * this->height(); }
  [[nodiscard]] int sizeBytes() const { return pitch_bytes_ * height(); }

  [[nodiscard]] bool isEmpty() const { return sizeBytes() == 0; }

 private:
  sophus::ImageSize image_size_ = {0, 0};
  size_t pitch_bytes_ = 0;
};

/// Equality operator.
bool operator==(ImageLayout const& lhs, ImageLayout const& rhs);

bool operator!=(ImageLayout const& lhs, ImageLayout const& rhs);

/// Ostream operator.
std::ostream& operator<<(std::ostream& os, ImageLayout const& layout);

}  // namespace sophus
