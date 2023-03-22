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

  static auto from(Eigen::Array2<int> const& arr) -> ImageSize {
    return {arr[0], arr[1]};
  }

  [[nodiscard]] auto area() const -> size_t { return width * height; }

  /// Returns true if obs is within image.
  ///
  /// Positive border makes the image frame smaller.
  [[nodiscard]] auto contains(Eigen::Vector2i const& obs, int border = 0) const
      -> bool;
  [[nodiscard]] auto contains(
      Eigen::Vector2f const& obs, float border = 0.f) const -> bool;
  [[nodiscard]] auto contains(
      Eigen::Vector2d const& obs, double border = 0.0) const -> bool;

  [[nodiscard]] auto isEmpty() const -> bool {
    return width == 0 && height == 0;
  }

  [[nodiscard]] auto array() const -> Eigen::Array2<int> {
    return Eigen::Array2<int>(width, height);
  }

  [[nodiscard]] auto iwidth() const -> int { return this->width; }
  [[nodiscard]] auto iheight() const -> int { return this->height; }

  /// Horizontal width of images, i.e. number of columns.
  int width = 0;

  /// Vertical height of images, i.e. number of rows.
  int height = 0;
};

// TODO: make member function?
template <class TPixel>
inline auto imageCoordsInterval(ImageSize image_size, int border = 0)
    -> Region2I {
  // e.g. 10x10 image has valid values [0, ..., 9] in both dimensions
  // a border of 2 would make valid range [2, ..., 7]
  return Region2I::fromMinMax(
      Eigen::Vector2i(border, border),
      Eigen::Vector2i(
          image_size.width - border - 1, image_size.height - border - 1));
}

/// Equality operator.
auto operator==(ImageSize const& lhs, ImageSize const& rhs) -> bool;

auto operator!=(ImageSize const& lhs, ImageSize const& rhs) -> bool;

/// If the original width [height] is odd, the new width [height] will be:
/// (width+1)/2 [height+1)/2].
auto half(ImageSize size) -> ImageSize;

/// Ordering operator, for keys in sets and maps
auto operator<(ImageSize const& lhs, ImageSize const& rhs) -> bool;

/// Ostream operator.
auto operator<<(std::ostream& os, ImageSize const& image_size) -> std::ostream&;

}  // namespace sophus
