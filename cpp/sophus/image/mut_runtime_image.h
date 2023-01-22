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

/// Type-erased image with shared ownership, and read-only access to pixels.
/// Type is nullable.
template <
    class TPredicate = AnyImagePredicate,
    template <class> class TAllocator = Eigen::aligned_allocator>
class MutRuntimeImage : public MutRuntimeImageView<TPredicate> {
 public:
  /// Empty image.
  MutRuntimeImage() = default;

  /// Create type-erased image from MutImage.
  ///
  /// By design not "explicit".
  template <class TPixel>
  MutRuntimeImage(MutImage<TPixel, TAllocator>&& image)
      : MutRuntimeImage(
            image.shape(),
            RuntimePixelType::fromTemplate<TPixel>(),
            std::move(image.shared_)) {
    static_assert(TPredicate::template isTypeValid<TPixel>());
  }

  /// Create type-image image from provided shape and pixel type.
  /// Pixel data is left uninitialized
  MutRuntimeImage(ImageShape const& shape, RuntimePixelType const& pixel_type)
      : MutRuntimeImage(
            shape,
            pixel_type,
            std::shared_ptr<uint8_t>(TAllocator<uint8_t>().allocate(
                shape.height() * shape.pitchBytes()))) {
    // TODO: Missing check on ImagePredicate against pixel_type
    //       has to be a runtime check, since we don't know at compile time.

    SOPHUS_ASSERT_LE(
        shape.width() * pixel_type.num_channels *
            pixel_type.num_bytes_per_pixel_channel,
        (int)shape.pitchBytes());
  }

  /// Create type-image image from provided size and pixel type.
  /// Pixel data is left uninitialized
  MutRuntimeImage(ImageSize const& size, RuntimePixelType const& pixel_type)
      : MutRuntimeImage(
            ImageShape::makeFromSizeAndPitch<uint8_t>(
                size,
                size.width * pixel_type.num_channels *
                    pixel_type.num_bytes_per_pixel_channel),
            pixel_type) {}

  template <class TT>
  static MutRuntimeImage makeCopyFrom(ImageView<TT> image_view) {
    return MutImage<TT>::makeCopyFrom(image_view);
  }

  /// Not copy constructable
  MutRuntimeImage(MutRuntimeImage const& other) = delete;

  /// Not copy assignable
  MutRuntimeImage& operator=(MutRuntimeImage const&) = delete;

  /// Return true is this contains data of type TPixel.
  template <class TPixel>
  [[nodiscard]] bool has() const noexcept {
    RuntimePixelType expected_type = RuntimePixelType::fromTemplate<TPixel>();
    return expected_type == this->pixel_type_;
  }

  /// Returns typed image.
  ///
  /// Precondition: this->has<TPixel>()
  template <class TPixel>
  [[nodiscard]] MutImage<TPixel, TAllocator> mutImage() const noexcept {
    FARM_UNIMPLEMENTED();
  }

 private:
  // Private constructor mainly available for constructing sub-views
  MutRuntimeImage(
      ImageShape shape,
      RuntimePixelType pixel_type,
      std::shared_ptr<uint8_t> shared)
      : RuntimeImageView<TPredicate>(shape, pixel_type, shared.get()),
        shared_(shared) {}

  std::shared_ptr<uint8_t> shared_;
};

}  // namespace sophus
