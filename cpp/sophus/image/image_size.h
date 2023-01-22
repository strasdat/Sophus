// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/calculus/region.h"
#include "sophus/common/common.h"

#include <Eigen/Dense>

#include <iostream>

namespace sophus {

/// Image size, hence its width and height.
struct ImageSize {
  ImageSize() = default;
  ImageSize(int width, int height) : width(width), height(height) {}
  static ImageSize from(Eigen::Array2<int> const& arr) {
    return {arr[0], arr[1]};
  }

  [[nodiscard]] int area() const { return width * height; }

  /// Returns true if obs is within image.
  ///
  /// Positive border makes the image frame smaller.
  [[nodiscard]] bool contains(Eigen::Vector2i const& obs, int border = 0) const;
  [[nodiscard]] bool contains(
      Eigen::Vector2f const& obs, float border = 0.f) const;
  [[nodiscard]] bool contains(
      Eigen::Vector2d const& obs, double border = 0.0) const;

  [[nodiscard]] bool isEmpty() const { return width == 0 && height == 0; }

  [[nodiscard]] Eigen::Array2<int> array() const {
    return Eigen::Array2<int>(width, height);
  }

  /// Horizontal width of images, i.e. number of columns.
  int width = 0;

  /// Vertical height of images, i.e. number of rows.
  int height = 0;
};

// TODO: make member function?
template <class TPixel>
inline Region2I imageCoordsInterval(ImageSize image_size, int border = 0) {
  // e.g. 10x10 image has valid values [0, ..., 9] in both dimensions
  // a border of 2 would make valid range [2, ..., 7]
  return Region2I::fromMinMax(
      Eigen::Vector2i(border, border),
      Eigen::Vector2i(
          image_size.width - border - 1, image_size.height - border - 1));
}

/// Equality operator.
bool operator==(ImageSize const& lhs, ImageSize const& rhs);

bool operator!=(ImageSize const& lhs, ImageSize const& rhs);

/// If the original width [height] is odd, the new width [height] will be:
/// (width+1)/2 [height+1)/2].
ImageSize half(ImageSize size);

/// Ordering operator, for keys in sets and maps
bool operator<(ImageSize const& lhs, ImageSize const& rhs);

/// Ostream operator.
std::ostream& operator<<(std::ostream& os, ImageSize const& image_size);

}  // namespace sophus
