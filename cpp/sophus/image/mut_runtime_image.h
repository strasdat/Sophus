// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image/mut_runtime_image_view.h"

#include <variant>

namespace sophus {

template <class TPredicate, class TAllocator>
class RuntimeImage;

template <
    class TPredicate = AnyImagePredicate,
    class TAllocator = Eigen::aligned_allocator<uint8_t>>
class MutRuntimeImage : public MutRuntimeImageView<TPredicate> {
 public:
  /// Empty image.
  MutRuntimeImage() = default;

  /// Not copy constructable
  MutRuntimeImage(MutRuntimeImage const& other) = delete;
  /// Not copy assignable
  MutRuntimeImage& operator=(MutRuntimeImage const&) = delete;

  /// Nothrow move constructable
  MutRuntimeImage(MutRuntimeImage&& other) noexcept = default;
  /// Nothrow move assignable
  MutRuntimeImage& operator=(MutRuntimeImage&&) noexcept = default;

  /// Create type-erased image from MutImage.
  ///
  /// By design not "explicit".
  template <class TPixel>
  MutRuntimeImage(MutImage<TPixel, TAllocator>&& image)
      : MutRuntimeImage(
            image.shape(),
            RuntimePixelType::fromTemplate<TPixel>(),
            std::move(image.unique_)) {
    image.reset();
    static_assert(TPredicate::template isTypeValid<TPixel>());
  }

  /// Create type-image image from provided size and pixel type.
  /// Pixel data is left uninitialized
  MutRuntimeImage(ImageSize const& size, RuntimePixelType const& pixel_type)
      : MutRuntimeImage(
            ImageShape::makeFromSizeAndPitch<uint8_t>(
                size, size.width * pixel_type.bytesPerPixel()),
            pixel_type) {}

  /// Create type-image image from provided size and pixel type.
  /// Pixel data is left uninitialized
  MutRuntimeImage(ImageShape const& shape, RuntimePixelType const& pixel_type)
      : MutRuntimeImage(shape, pixel_type, nullptr) {
    if (this->shape_.sizeBytes() != 0u) {
      this->unique_ = UniqueDataArea<TAllocator>(
          TAllocator().allocate(this->shape_.sizeBytes()),
          MaybeLeakingUniqueDataAreaDeleter<TAllocator>(
              UniqueDataAreaDeleter<TAllocator>(this->shape_.sizeBytes())));
    }
    this->ptr_ = this->unique_.get();
  }

  template <class TT>
  static MutRuntimeImage makeCopyFrom(ImageView<TT> image_view) {
    return MutImage<TT>::makeCopyFrom(image_view);
  }

  static MutRuntimeImage makeCopyFrom(RuntimeImageView<TPredicate> image_view) {
    MutRuntimeImage image(image_view.imageSize(), image_view.pixelType());
    image.copyDataFrom(image_view);
    return image;
  }

  /// Return true is this contains data of type TPixel.
  template <class TPixel>
  [[nodiscard]] bool has() const noexcept {
    RuntimePixelType expected_type = RuntimePixelType::fromTemplate<TPixel>();
    return expected_type == this->pixel_type_;
  }

  /// Returns typed MutImage.
  ///
  /// Precondition: this->has<TPixel>()
  template <class TPixel>
  [[nodiscard]] MutImage<TPixel, TAllocator> moveOutAs() noexcept {
    SOPHUS_ASSERT(this->has<TPixel>());
    MutImage<TPixel, TAllocator> mut_image;
    mut_image.shape_ = this->shape_;
    mut_image.unique_ = std::move(unique_);
    mut_image.ptr_ = (TPixel*)mut_image.unique_.get();
    this->unique_.reset();
    this->setViewToEmpty();
    return mut_image;
  }

 protected:
  template <class TPredicate2, class TAllocator2T>
  friend class RuntimeImage;

  // Private constructor mainly available for constructing sub-views
  MutRuntimeImage(
      ImageShape shape,
      RuntimePixelType pixel_type,
      UniqueDataArea<TAllocator> unique)
      : MutRuntimeImageView<TPredicate>(shape, pixel_type, unique.get()),
        unique_(std::move(unique)) {}

  UniqueDataArea<TAllocator> unique_;
};

}  // namespace sophus
