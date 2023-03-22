// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image/dyn_image_view.h"

#include <variant>

namespace sophus {

template <class TPredicate = AnyImagePredicate>
class MutDynImageView : public DynImageView<TPredicate> {
 public:
  /// Create type-erased image view from ImageView.
  ///
  /// By design not "explicit".
  template <class TPixel>
  MutDynImageView(MutImageView<TPixel> const& image)
      : MutDynImageView(
            image.layout(), PixelFormat::fromTemplate<TPixel>(), image.ptr()) {
    static_assert(TPredicate::template isTypeValid<TPixel>());
  }

  /// Return true is this contains data of type TPixel.
  template <class TPixel>
  [[nodiscard]] auto has() const noexcept -> bool {
    PixelFormat expected_type = PixelFormat::fromTemplate<TPixel>();
    return expected_type == this->pixel_format_;
  }

  /// Returns v-th row pointer.
  ///
  /// Precondition: v must be in [0, height).
  [[nodiscard]] auto rawMutRowPtr(int v) const -> uint8_t* {
    return this->rawMutPtr() + v * this->layout_.pitchBytes();
  }

  [[nodiscard]] auto rawMutPtr() const -> uint8_t* {
    return const_cast<uint8_t*>(this->rawPtr());
  }

  /// Returns subview with shared ownership semantics of whole image.
  [[nodiscard]] auto mutSubview(
      Eigen::Vector2i uv, sophus::ImageSize size) const -> MutDynImageView {
    SOPHUS_ASSERT(this->imageSize().contains(uv));
    SOPHUS_ASSERT_LE(uv.x() + size.width, this->layout_.width());
    SOPHUS_ASSERT_LE(uv.y() + size.height, this->layout_.height());

    auto const layout = ImageLayout(size, this->pitchBytes());
    const size_t row_offset =
        uv.x() * this->numBytesPerPixelChannel() * this->numChannels();
    uint8_t* ptr = this->rawMutPtr() + uv.y() * this->pitchBytes() + row_offset;
    return MutDynImageView{layout, this->pixel_format_, ptr};
  }

  /// Returns typed image view.
  ///
  /// Precondition: this->has<TPixel>()
  template <class TPixel>
  [[nodiscard]] auto mutImageView() const noexcept -> MutImageView<TPixel> {
    if (!this->has<TPixel>()) {
      PixelFormat expected_type = PixelFormat::fromTemplate<TPixel>();

      SOPHUS_PANIC(
          "expected type: {}\n"
          "actual type: {}",
          expected_type,
          this->pixel_format_);
    }

    return MutImageView<TPixel>(
        this->layout_, reinterpret_cast<TPixel const*>(this->ptr_));
  }

  /// Copies data from view into this.
  ///
  /// Preconditions:
  ///  * this->isEmpty() == view.isEmpty()
  ///  * this->size() == view.size()
  ///
  /// No-op if view is empty.
  void copyDataFrom(DynImageView<TPredicate> view) const {
    SOPHUS_ASSERT_EQ(this->isEmpty(), view.isEmpty());

    if (this->isEmpty()) {
      return;
    }
    SOPHUS_ASSERT_EQ(this->imageSize(), view.imageSize());
    SOPHUS_ASSERT_EQ(this->pixel_format_, view.pixelFormat());
    details::pitchedCopy(
        (uint8_t*)this->rawMutPtr(),
        this->layout().pitchBytes(),
        (uint8_t const*)view.rawPtr(),
        view.layout().pitchBytes(),
        this->imageSize(),
        this->imageSize().width * this->pixel_format_.numBytesPerPixel());
  }

 protected:
  MutDynImageView(
      ImageLayout const& layout,
      PixelFormat const& pixel_format,
      void const* ptr)
      : DynImageView<TPredicate>(layout, pixel_format, ptr) {
    SOPHUS_ASSERT(TPredicate::isFormatValid(pixel_format));
  }

  MutDynImageView() = default;
};
}  // namespace sophus
