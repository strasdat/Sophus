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

/// Shape of image: width, height and pitch in bytes.
class ImageShape {
 public:
  ImageShape() = default;

  ImageShape(ImageSize image_size, size_t pitch_bytes)
      : image_size_(image_size), pitch_bytes_(pitch_bytes) {}

  ImageShape(int width, int height, size_t pitch_bytes)
      : image_size_(width, height), pitch_bytes_(pitch_bytes) {}

  template <class TPixelType>
  [[nodiscard]] static ImageShape makeFromSizeAndPitch(
      ImageSize image_size, size_t pitch_bytes) {
    SOPHUS_ASSERT_GE(pitch_bytes, image_size.width * sizeof(TPixelType));
    ImageShape shape;
    shape.image_size_ = image_size;
    shape.pitch_bytes_ = pitch_bytes;

    return shape;
  }

  [[nodiscard]] static ImageShape makeFromSizeAndPitchUnchecked(
      ImageSize image_size, size_t pitch_bytes) {
    ImageShape shape;
    shape.image_size_ = image_size;
    shape.pitch_bytes_ = pitch_bytes;
    return shape;
  }

  template <class TPixelTypeT>
  [[nodiscard]] static ImageShape makeFromSize(sophus::ImageSize image_size) {
    return makeFromSizeAndPitch<TPixelTypeT>(
        image_size, image_size.width * sizeof(TPixelTypeT));
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
bool operator==(ImageShape const& lhs, ImageShape const& rhs);

bool operator!=(ImageShape const& lhs, ImageShape const& rhs);

/// Ostream operator.
std::ostream& operator<<(std::ostream& os, ImageShape const& shape);

}  // namespace sophus
