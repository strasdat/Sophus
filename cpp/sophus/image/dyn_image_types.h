// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image/dyn_image.h"
#include "sophus/image/mut_dyn_image.h"

namespace sophus {

/// Image representing any number of channels (>=1) and any floating and
/// unsigned integral channel type.
template <class TAllocator = Eigen::aligned_allocator<uint8_t>>
using AnyImage = DynImage<AnyImagePredicate, TAllocator>;
using AnyImageView = DynImageView<AnyImagePredicate>;
template <class TAllocator = Eigen::aligned_allocator<uint8_t>>
using MutAnyImage = MutDynImage<AnyImagePredicate, TAllocator>;
using MutAnyImageView = MutDynImageView<AnyImagePredicate>;

template <class TPixelVariant>
struct VariantImagePredicate {
  using PixelVariant = TPixelVariant;

  template <class TPixel>
  static auto constexpr isTypeValid() -> bool {
    return has_type_v<TPixel, TPixelVariant>;
  }
};

using IntensityImagePredicate = VariantImagePredicate<std::variant<
    uint8_t,
    uint16_t,
    float,
    Pixel3U8,
    Pixel3U16,
    Pixel3F32,
    Pixel4U8,
    Pixel4U16,
    Pixel4F32>>;

/// Image to represent intensity image / texture as grayscale (=1 channel),
/// RGB (=3 channel ) and RGBA (=4 channel), either uint8_t [0-255],
/// uint16 [0-65535] or float [0.0-1.0] channel type.
template <class TAllocator = Eigen::aligned_allocator<uint8_t>>
using IntensityImage = DynImage<IntensityImagePredicate, TAllocator>;
using IntensityImageView = DynImageView<IntensityImagePredicate>;
template <class TAllocator = Eigen::aligned_allocator<uint8_t>>
using MutIntensityImage = MutDynImage<IntensityImagePredicate, TAllocator>;
using MutIntensityImageView = MutDynImageView<IntensityImagePredicate>;

namespace detail {
// Call UserFunc with TDynImage cast to the appropriate concrete type
// from the options in pixelFormats...
template <class TUserFunc, typename TDynImage, typename... TTpixelFormats>
struct VisitImpl;

// base case
template <class TUserFunc, typename TDynImage, typename TPixelFormat>
struct VisitImpl<TUserFunc, TDynImage, std::variant<TPixelFormat>> {
  static void visit(TUserFunc&& func, TDynImage const& image) {
    if (image.pixelFormat().template is<TPixelFormat>()) {
      func(image.template imageView<TPixelFormat>());
    }
  }
};

// inductive case
template <
    typename TUserFunc,
    typename TDynImage,
    typename TPixelFormat,
    typename... TRest>
struct VisitImpl<TUserFunc, TDynImage, std::variant<TPixelFormat, TRest...>> {
  static void visit(TUserFunc&& func, TDynImage const& image) {
    if (image.pixelFormat().template is<TPixelFormat>()) {
      func(image.template imageView<TPixelFormat>());
    } else {
      VisitImpl<TUserFunc, TDynImage, std::variant<TRest...>>::visit(
          std::forward<TUserFunc>(func), image);
    }
  }
};
}  // namespace detail

// DynImage visitor
template <
    typename TUserFunc,
    class TPredicate = IntensityImagePredicate,
    class TAllocator = Eigen::aligned_allocator<uint8_t>>
void visitImage(
    TUserFunc&& func, DynImage<TPredicate, TAllocator> const& image) {
  using TDynImage = DynImage<TPredicate, TAllocator>;
  detail::VisitImpl<TUserFunc, TDynImage, typename TPredicate::PixelVariant>::
      visit(std::forward<TUserFunc>(func), image);
}

// DynImageView visitor - shares same implementation than the one for
// DynImage
template <class TUserFunc, class TPredicate = IntensityImagePredicate>
void visitImage(TUserFunc&& func, DynImageView<TPredicate> const& image) {
  using DynImageView = DynImageView<TPredicate>;
  detail::
      VisitImpl<TUserFunc, DynImageView, typename TPredicate::PixelVariant>::
          visit(std::forward<TUserFunc>(func), image);
}

}  // namespace sophus
