// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/common.h"
#include "sophus/concepts/image.h"
#include "sophus/image/image_size.h"

#include <Eigen/Dense>

#include <iostream>

namespace sophus {

/// Layout of the image: width, height and pitch in bytes.
///
class ImageLayout {
 public:
  ImageLayout() = default;

  explicit ImageLayout(ImageSize image_size, size_t pitch_bytes)
      : image_size_(image_size), pitch_bytes_(pitch_bytes) {}

  explicit ImageLayout(int width, int height, size_t pitch_bytes)
      : image_size_(ImageSize(width, height)), pitch_bytes_(pitch_bytes) {}

  template <class TPixelFormat>
  [[nodiscard]] static auto makeFromSizeAndPitch(
      ImageSize image_size, size_t pitch_bytes) -> ImageLayout {
    SOPHUS_ASSERT_GE(pitch_bytes, image_size.width * sizeof(TPixelFormat));
    ImageLayout layout;
    layout.image_size_ = image_size;
    layout.pitch_bytes_ = pitch_bytes;

    return layout;
  }

  template <class TPixelFormat>
  [[nodiscard]] static auto makeFromSize(sophus::ImageSize image_size)
      -> ImageLayout {
    return makeFromSizeAndPitch<TPixelFormat>(
        image_size, image_size.width * sizeof(TPixelFormat));
  }

  [[nodiscard]] auto imageSize() const -> sophus::ImageSize const& {
    return image_size_;
  }
  [[nodiscard]] auto width() const -> int { return image_size_.width; }
  [[nodiscard]] auto height() const -> int { return image_size_.height; }

  [[nodiscard]] auto pitchBytes() const -> size_t { return pitch_bytes_; }

  [[nodiscard]] auto area() const -> size_t {
    return this->width() * this->height();
  }
  [[nodiscard]] auto sizeBytes() const -> size_t {
    return pitch_bytes_ * height();
  }

  [[nodiscard]] auto isEmpty() const -> bool { return sizeBytes() == 0; }

 private:
  sophus::ImageSize image_size_ = {0, 0};
  size_t pitch_bytes_ = 0;
};

static_assert(concepts::ImageLayoutTrait<ImageLayout>);

/// Equality operator.
auto operator==(ImageLayout const& lhs, ImageLayout const& rhs) -> bool;

auto operator!=(ImageLayout const& lhs, ImageLayout const& rhs) -> bool;

/// Ostream operator.
auto operator<<(std::ostream& os, ImageLayout const& layout) -> std::ostream&;

}  // namespace sophus
