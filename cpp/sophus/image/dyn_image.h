// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image/dyn_image_view.h"
#include "sophus/image/mut_dyn_image.h"

#include <variant>

namespace sophus {

/// Type-erased image with shared ownership, and read-only access to pixels.
/// Type is nullable.
///
template <
    class TPredicate = AnyImagePredicate,
    class TAllocator = Eigen::aligned_allocator<uint8_t>>
class DynImage : public DynImageView<TPredicate> {
 public:
  /// Empty image.
  DynImage() = default;

  /// Create type-erased image from Image.
  ///
  /// Ownership is shared between DynImage and Image, and hence the
  /// reference count will be increased by one (unless input is empty).
  /// By design not "explicit".
  template <class TPixel>
  DynImage(Image<TPixel, TAllocator> const& image)
      : DynImage(
            image.layout(),
            PixelFormat::fromTemplate<TPixel>(),
            image.shared_) {
    static_assert(TPredicate::template isTypeValid<TPixel>());
  }

  /// Create type-erased image from MutImage.
  /// By design not "explicit".
  template <class TPixel>
  DynImage(MutImage<TPixel>&& image)
      : DynImage(Image<TPixel>(std::move(image))) {
    static_assert(TPredicate::template isTypeValid<TPixel>());
  }

  /// Create type-erased image from MutImage.
  /// By design not "explicit".
  DynImage(MutDynImage<TPredicate, TAllocator>&& image)
      : DynImage(
            image.layout(), image.pixel_format_, std::move(image.unique_)) {
    image.unique_.reset();
    image.setViewToEmpty();
  }

  /// Tries to create image from provided size and format.
  /// Returns error if format does not satisfy TPredicate.
  static Expected<DynImage<TPredicate, TAllocator>> tryFromFormat(
      ImageSize const& size, PixelFormat const& pixel_format) {
    using Mut = MutDynImage<TPredicate, TAllocator>;
    FARM_TRY(auto, mut, Mut::tryFromFormat(size, pixel_format));

    return DynImage(std::move(mut));
  }

  // Creates image from provided size and format.
  //
  // Panics if format does not satisfy TPredicate.
  static DynImage<TPredicate, TAllocator> fromFormat(
      ImageSize const& size, PixelFormat const& pixel_format) {
    auto maybe = tryFromFormat(size, pixel_format);
    return SOPHUS_UNWRAP(maybe);
  }

  // Creates image from provided size and format.
  //
  // Panics if format does not satisfy TPredicate.
  static DynImage<TPredicate, TAllocator> fromFormat(
      ImageLayout const& layout, PixelFormat const& pixel_format) {
    auto maybe = tryFromFormat(layout, pixel_format);
    return SOPHUS_UNWRAP(maybe);
  }

  /// Tries to create image from provided size and format.
  /// Returns error if format does not satisfy TPredicate.
  static Expected<DynImage<TPredicate, TAllocator>> tryFromFormat(
      ImageLayout const& layout, PixelFormat const& pixel_format) {
    if (!TPredicate::isFormatValid(pixel_format)) {
      return SOPHUS_UNEXPECTED("pixel format does not satisfy predicate");
    }
    return DynImage(MutDynImage<TPredicate, TAllocator>(layout, pixel_format));
  }

  template <class TOtherPredicate>
  static auto tryFrom(DynImage<TOtherPredicate, TAllocator> other_image)
      -> Expected<DynImage<TPredicate, TAllocator>> {
    if (!TPredicate::isFormatValid(other_image.pixelFormat())) {
      return SOPHUS_UNEXPECTED("pixel format does not satisfy predicate");
    }
    return DynImage<TPredicate, TAllocator>(
        other_image.layout(), other_image.pixelFormat(), other_image.shared_);
  }

  /// Return true is this contains data of type TPixel.
  template <class TPixel>
  [[nodiscard]] auto has() const noexcept -> bool {
    PixelFormat expected_type = PixelFormat::fromTemplate<TPixel>();
    return expected_type == this->pixel_format_;
  }

  /// Returns typed image.
  ///
  /// Precondition: this->has<TPixel>()
  template <class TPixel>
  [[nodiscard]] auto image() const noexcept -> Image<TPixel, TAllocator> {
    if (!this->has<TPixel>()) {
      PixelFormat expected_type = PixelFormat::fromTemplate<TPixel>();

      SOPHUS_PANIC(
          "expected type: {}\n"
          "actual type: {}",
          expected_type,
          this->pixel_format_);
    }

    return Image<TPixel, TAllocator>(
        ImageView<TPixel>(
            this->layout_, reinterpret_cast<TPixel*>(shared_.get())),
        shared_);
  }

  template <class TPixel>
  auto reinterpretAs(ImageSize reinterpreted_size) const noexcept
      -> Image<TPixel, TAllocator> {
    SOPHUS_ASSERT_LE(
        reinterpreted_size.width * sizeof(TPixel), this->layout().pitch_bytes_);
    SOPHUS_ASSERT_LE(reinterpreted_size.height, this->height());

    SOPHUS_UNIMPLEMENTED();
  }

  [[nodiscard]] auto useCount() const -> size_t { return shared_.use_count(); }

 protected:
  template <class TPredicate2, class TAllocator2T>
  friend class DynImage;

  // Private constructor mainly available for constructing sub-views
  DynImage(
      ImageLayout layout,
      PixelFormat pixel_format,
      std::shared_ptr<uint8_t> shared)
      : DynImageView<TPredicate>(layout, pixel_format, shared.get()),
        shared_(shared) {}

  std::shared_ptr<uint8_t> shared_;
};

}  // namespace sophus
