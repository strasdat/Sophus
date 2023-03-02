// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image/image.h"
#include "sophus/image/image_types.h"
#include "sophus/image/pixel_format.h"

#include <variant>

namespace sophus {

struct AnyImagePredicate {
  template <class TPixel>
  static auto constexpr isTypeValid() -> bool {
    return true;
  }
};

template <class TPredicate = AnyImagePredicate>
class DynImageView {
 public:
  /// Create type-erased image view from ImageView.
  ///
  /// By design not "explicit".
  template <class TPixel>
  DynImageView(ImageView<TPixel> const& image)
      : DynImageView(
            image.layout(), PixelFormat::fromTemplate<TPixel>(), image.ptr()) {
    static_assert(TPredicate::template isTypeValid<TPixel>());
  }

  DynImageView(
      ImageLayout const& layout, PixelFormat const& pixel_type, void const* ptr)
      : layout_(layout),
        pixel_format_(pixel_type),
        ptr_(reinterpret_cast<uint8_t const*>(ptr)) {}

  /// Return true is this contains data of type TPixel.
  template <class TPixel>
  [[nodiscard]] auto has() const noexcept -> bool {
    PixelFormat expected_type = PixelFormat::fromTemplate<TPixel>();
    return expected_type == pixel_format_;
  }

  /// Returns v-th row pointer.
  ///
  /// Precondition: v must be in [0, height).
  [[nodiscard]] auto rawRowPtr(int v) const -> uint8_t const* {
    return ((uint8_t*)(rawPtr()) + v * layout_.pitchBytes());
  }

  [[nodiscard]] auto rawPtr() const -> uint8_t const* { return ptr_; }

  /// Number of bytes per channel of a single pixel.
  ///
  /// E.g. a pixel of Eigen::Matrix<uint8_t, 3, 1> has 1 byte per channel.
  [[nodiscard]] auto numBytesPerPixelChannel() const -> int {
    return pixel_format_.num_bytes_per_pixel_channel;
  }
  [[nodiscard]] auto numberType() const -> NumberType {
    return pixel_format_.number_type;
  }

  [[nodiscard]] auto layout() const -> ImageLayout const& { return layout_; }

  [[nodiscard]] auto imageSize() const -> ImageSize const& {
    return layout_.imageSize();
  }

  [[nodiscard]] auto width() const -> int { return layout().width(); }
  [[nodiscard]] auto height() const -> int { return layout().height(); }
  [[nodiscard]] auto pitchBytes() const -> size_t {
    return layout().pitchBytes();
  }
  [[nodiscard]] auto sizeBytes() const -> size_t {
    return height() * pitchBytes();
  }
  [[nodiscard]] auto isEmpty() const -> bool {
    return this->rawPtr() == nullptr;
  }

  [[nodiscard]] auto pixelFormat() const -> PixelFormat {
    return pixel_format_;
  }
  [[nodiscard]] auto numChannels() const -> int {
    return this->pixel_format_.num_channels;
  }

  /// Returns subview with shared ownership semantics of whole image.
  [[nodiscard]] auto subview(Eigen::Vector2i uv, sophus::ImageSize size) const
      -> DynImageView {
    SOPHUS_ASSERT(imageSize().contains(uv));
    SOPHUS_ASSERT_LE(uv.x() + size.width, this->layout_.width());
    SOPHUS_ASSERT_LE(uv.y() + size.height, this->layout_.height());

    auto const layout =
        ImageLayout::makeFromSizeAndPitchUnchecked(size, pitchBytes());
    const size_t row_offset =
        uv.x() * numBytesPerPixelChannel() * numChannels();
    uint8_t const* ptr = this->rawPtr() + uv.y() * pitchBytes() + row_offset;
    return DynImageView{layout, this->pixel_format_, ptr};
  }

  /// Returns typed image.
  ///
  /// Precondition: this->has<TPixel>()
  template <class TPixel>
  [[nodiscard]] auto imageView() const noexcept -> ImageView<TPixel> {
    if (!this->has<TPixel>()) {
      PixelFormat expected_type = PixelFormat::fromTemplate<TPixel>();

      SOPHUS_PANIC(
          "expected type: {}\n"
          "actual type: {}",
          expected_type,
          this->pixel_format_);
    }

    return ImageView<TPixel>(
        this->layout_, reinterpret_cast<TPixel const*>(ptr_));
  }

  void setViewToEmpty() {
    this->layout_ = {};
    this->ptr_ = nullptr;
  }

 protected:
  DynImageView() = default;

  ImageLayout layout_ = {};
  PixelFormat pixel_format_;
  uint8_t const* ptr_ = nullptr;
};
}  // namespace sophus
