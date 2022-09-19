// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include <Eigen/Dense>
#include <farm_ng/core/logging/logger.h>

#include <iostream>

namespace sophus {

/// Image size, hence its width and height.
struct ImageSize {
  ImageSize() = default;
  ImageSize(int width, int height) : width(width), height(height) {}

  [[nodiscard]] [[nodiscard]] int area() const { return width * height; }

  /// Horizontal width of images, i.e. number of columns.
  int width = 0;

  /// Vertical height of images, i.e. number of rows.
  int height = 0;

  /// Returns true if obs is within image.
  ///
  /// Positive border makes the image frame smaller.
  [[nodiscard]] [[nodiscard]] bool contains(
      const Eigen::Vector2i& obs, int border = 0) const;
  [[nodiscard]] [[nodiscard]] bool contains(
      const Eigen::Vector2f& obs, float border = 0.f) const;
  [[nodiscard]] [[nodiscard]] bool contains(
      const Eigen::Vector2d& obs, double border = 0.0) const;
};

/// Equality operator.
bool operator==(const ImageSize& lhs, const ImageSize& rhs);

/// If the original width [height] is odd, the new width [height] will be:
/// (width+1)/2 [height+1)/2].
ImageSize half(ImageSize size);

/// Ordering operator, for keys in sets and maps
bool operator<(const ImageSize& lhs, const ImageSize& rhs);

/// Ostream operator.
std::ostream& operator<<(std::ostream& os, const ImageSize& image_size);

/// Shape of image: width, height and pitch in bytes.
class ImageShape {
 public:
  ImageShape() = default;

  template <class PixelType>
  [[nodiscard]] static ImageShape makeFromSizeAndPitch(
      ImageSize image_size, size_t pitch_bytes) {
    FARM_CHECK_GE(pitch_bytes, image_size.width * sizeof(PixelType));
    ImageShape shape;
    shape.image_size_ = image_size;
    shape.pitch_bytes_ = pitch_bytes;

    return shape;
  }

  template <class PixelTypeT>
  [[nodiscard]] static ImageShape makeFromSize(sophus::ImageSize image_size) {
    return makeFromSizeAndPitch<PixelTypeT>(
        image_size, image_size.width * sizeof(PixelTypeT));
  }

  [[nodiscard]] const sophus::ImageSize& imageSize() const {
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
bool operator==(const ImageShape& lhs, const ImageShape& rhs);

/// Ostream operator.
std::ostream& operator<<(std::ostream& os, const ImageShape& shape);

}  // namespace sophus
