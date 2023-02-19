// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image/mut_dyn_image_view.h"

#include <variant>

namespace sophus {

template <class TPredicate, class TAllocator>
class DynImage;

template <
    class TPredicate = AnyImagePredicate,
    class TAllocator = Eigen::aligned_allocator<uint8_t>>
class MutDynImage : public MutDynImageView<TPredicate> {
 public:
  /// Empty image.
  MutDynImage() = default;

  /// Not copy constructable
  MutDynImage(MutDynImage const& other) = delete;
  /// Not copy assignable
  MutDynImage& operator=(MutDynImage const&) = delete;

  /// Nothrow move constructable
  MutDynImage(MutDynImage&& other) noexcept = default;
  /// Nothrow move assignable
  MutDynImage& operator=(MutDynImage&&) noexcept = default;

  /// Create type-erased image from MutImage.
  ///
  /// By design not "explicit".
  template <class TPixel>
  MutDynImage(MutImage<TPixel, TAllocator>&& image)
      : MutDynImage(
            image.layout(),
            PixelFormat::fromTemplate<TPixel>(),
            std::move(image.unique_)) {
    image.reset();
    static_assert(TPredicate::template isTypeValid<TPixel>());
  }

  /// Create type-image image from provided size and pixel type.
  /// Pixel data is left uninitialized
  MutDynImage(ImageSize const& size, PixelFormat const& pixel_type)
      : MutDynImage(
            ImageLayout::makeFromSizeAndPitch<uint8_t>(
                size, size.width * pixel_type.bytesPerPixel()),
            pixel_type) {}

  /// Create type-image image from provided size and pixel type.
  /// Pixel data is left uninitialized
  MutDynImage(ImageLayout const& layout, PixelFormat const& pixel_type)
      : MutDynImage(layout, pixel_type, nullptr) {
    if (this->layout_.sizeBytes() != 0u) {
      this->unique_ = UniqueDataArea<TAllocator>(
          TAllocator().allocate(this->layout_.sizeBytes()),
          MaybeLeakingUniqueDataAreaDeleter<TAllocator>(
              UniqueDataAreaDeleter<TAllocator>(this->layout_.sizeBytes())));
    }
    this->ptr_ = this->unique_.get();
  }

  template <class TT>
  static MutDynImage makeCopyFrom(ImageView<TT> image_view) {
    return MutImage<TT>::makeCopyFrom(image_view);
  }

  static MutDynImage makeCopyFrom(DynImageView<TPredicate> image_view) {
    MutDynImage image(image_view.imageSize(), image_view.pixelFormat());
    image.copyDataFrom(image_view);
    return image;
  }

  /// Return true is this contains data of type TPixel.
  template <class TPixel>
  [[nodiscard]] bool has() const noexcept {
    PixelFormat expected_type = PixelFormat::fromTemplate<TPixel>();
    return expected_type == this->pixel_format_;
  }

  /// Returns typed MutImage.
  ///
  /// Precondition: this->has<TPixel>()
  template <class TPixel>
  [[nodiscard]] MutImage<TPixel, TAllocator> moveOutAs() noexcept {
    SOPHUS_ASSERT(this->has<TPixel>());
    MutImage<TPixel, TAllocator> mut_image;
    mut_image.layout_ = this->layout_;
    mut_image.unique_ = std::move(unique_);
    mut_image.ptr_ = (TPixel*)mut_image.unique_.get();
    this->unique_.reset();
    this->setViewToEmpty();
    return mut_image;
  }

 protected:
  template <class TPredicate2, class TAllocator2T>
  friend class DynImage;

  // Private constructor mainly available for constructing sub-views
  MutDynImage(
      ImageLayout layout,
      PixelFormat pixel_type,
      UniqueDataArea<TAllocator> unique)
      : MutDynImageView<TPredicate>(layout, pixel_type, unique.get()),
        unique_(std::move(unique)) {}

  UniqueDataArea<TAllocator> unique_;
};

}  // namespace sophus