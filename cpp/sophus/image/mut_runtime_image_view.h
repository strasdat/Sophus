// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image/runtime_image_view.h"

#include <variant>

namespace sophus {

template <class TPredicate = AnyImagePredicate>
class MutRuntimeImageView : public RuntimeImageView<TPredicate> {
 public:
  /// Create type-erased image view from ImageView.
  ///
  /// By design not "explicit".
  template <class TPixel>
  MutRuntimeImageView(MutImageView<TPixel> const& image)
      : MutRuntimeImageView(
            image.shape(),
            RuntimePixelType::fromTemplate<TPixel>(),
            image.ptr()) {
    static_assert(TPredicate::template isTypeValid<TPixel>());
  }

  MutRuntimeImageView(
      ImageShape const& image_shape,
      RuntimePixelType const& pixel_type,
      void const* ptr)
      : RuntimeImageView<TPredicate>(image_shape, pixel_type, ptr) {}

  /// Return true is this contains data of type TPixel.
  template <class TPixel>
  [[nodiscard]] bool has() const noexcept {
    RuntimePixelType expected_type = RuntimePixelType::fromTemplate<TPixel>();
    return expected_type == this->pixel_type_;
  }

  /// Returns v-th row pointer.
  ///
  /// Precondition: v must be in [0, height).
  [[nodiscard]] uint8_t* rawMutRowPtr(int v) {
    return this->rawMutPtr() + v * this->shape_.pitchBytes();
  }

  [[nodiscard]] uint8_t* rawMutPtr() {
    return const_cast<uint8_t*>(this->rawPtr());
  }

  /// Returns subview with shared ownership semantics of whole image.
  [[nodiscard]] MutRuntimeImageView mutSubview(
      Eigen::Vector2i uv, sophus::ImageSize size) const {
    SOPHUS_ASSERT(this->imageSize().contains(uv));
    SOPHUS_ASSERT_LE(uv.x() + size.width, this->shape_.width());
    SOPHUS_ASSERT_LE(uv.y() + size.height, this->shape_.height());

    auto const shape =
        ImageShape::makeFromSizeAndPitchUnchecked(size, this->pitchBytes());
    const size_t row_offset =
        uv.x() * this->numBytesPerPixelChannel() * this->numChannels();
    uint8_t* ptr = this->rawMutPtr() + uv.y() * this->pitchBytes() + row_offset;
    return MutRuntimeImageView{shape, this->pixel_type_, ptr};
  }

  /// Returns typed image view.
  ///
  /// Precondition: this->has<TPixel>()
  template <class TPixel>
  [[nodiscard]] MutImageView<TPixel> mutImageView() const noexcept {
    if (!this->has<TPixel>()) {
      RuntimePixelType expected_type = RuntimePixelType::fromTemplate<TPixel>();

      SOPHUS_PANIC(
          "expected type: {}\n"
          "actual type: {}",
          expected_type,
          this->pixel_type_);
    }

    return MutImageView<TPixel>(
        this->shape_, reinterpret_cast<TPixel const*>(this->ptr_));
  }

  /// Copies data from view into this.
  ///
  /// Preconditions:
  ///  * this->isEmpty() == view.isEmpty()
  ///  * this->size() == view.size()
  ///
  /// No-op if view is empty.
  void copyDataFrom(RuntimeImageView<TPredicate> view) const {
    SOPHUS_ASSERT_EQ(this->isEmpty(), view.isEmpty());

    if (this->isEmpty()) {
      return;
    }
    SOPHUS_ASSERT_EQ(this->imageSize(), view.imageSize());
    SOPHUS_ASSERT_EQ(this->pixel_type_, view.pixelType());
    details::pitchedCopy(
        (uint8_t*)this->mutRawPtr(),
        this->shape().pitchBytes(),
        (uint8_t const*)view.rawPtr(),
        view.shape().pitchBytes(),
        this->imageSize(),
        this->size.width * this->pixel_type_.bytesPerPixel());
  }

 protected:
  MutRuntimeImageView() = default;
};
}  // namespace sophus
