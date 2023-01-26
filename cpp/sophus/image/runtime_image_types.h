// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image/mut_runtime_image.h"
#include "sophus/image/runtime_image.h"

namespace sophus {

/// Image representing any number of channels (>=1) and any floating and
/// unsigned integral channel type.
template <class TAllocator = Eigen::aligned_allocator<uint8_t>>
using AnyImage = RuntimeImage<AnyImagePredicate, TAllocator>;
using AnyImageView = RuntimeImageView<AnyImagePredicate>;
template <class TAllocator = Eigen::aligned_allocator<uint8_t>>
using MutAnyImage = MutRuntimeImage<AnyImagePredicate, TAllocator>;
using MutAnyImageView = MutRuntimeImageView<AnyImagePredicate>;

template <class TPixelVariant>
struct VariantImagePredicate {
  using PixelVariant = TPixelVariant;

  template <class TPixel>
  static bool constexpr isTypeValid() {
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
using IntensityImage = RuntimeImage<IntensityImagePredicate, TAllocator>;
using IntensityImageView = RuntimeImageView<IntensityImagePredicate>;
template <class TAllocator = Eigen::aligned_allocator<uint8_t>>
using MutIntensityImage = MutRuntimeImage<IntensityImagePredicate, TAllocator>;
using MutIntensityImageView = MutRuntimeImageView<IntensityImagePredicate>;

namespace detail {
// Call UserFunc with TRuntimeImage cast to the appropriate concrete type
// from the options in PixelTypes...
template <class TUserFunc, typename TRuntimeImage, typename... TPixelTypes>
struct VisitImpl;

// base case
template <class TUserFunc, typename TRuntimeImage, typename TPixelType>
struct VisitImpl<TUserFunc, TRuntimeImage, std::variant<TPixelType>> {
  static void visit(TUserFunc&& func, TRuntimeImage const& image) {
    if (image.pixelType().template is<TPixelType>()) {
      func(image.template imageView<TPixelType>());
    }
  }
};

// inductive case
template <
    typename TUserFunc,
    typename TRuntimeImage,
    typename TPixelType,
    typename... TRest>
struct VisitImpl<TUserFunc, TRuntimeImage, std::variant<TPixelType, TRest...>> {
  static void visit(TUserFunc&& func, TRuntimeImage const& image) {
    if (image.pixelType().template is<TPixelType>()) {
      func(image.template imageView<TPixelType>());
    } else {
      VisitImpl<TUserFunc, TRuntimeImage, std::variant<TRest...>>::visit(
          std::forward<TUserFunc>(func), image);
    }
  }
};
}  // namespace detail

// RuntimeImage visitor
template <
    typename TUserFunc,
    class TPredicate = IntensityImagePredicate,
    class TAllocator = Eigen::aligned_allocator<uint8_t>>
void visitImage(
    TUserFunc&& func, RuntimeImage<TPredicate, TAllocator> const& image) {
  using TRuntimeImage = RuntimeImage<TPredicate, TAllocator>;
  detail::
      VisitImpl<TUserFunc, TRuntimeImage, typename TPredicate::PixelVariant>::
          visit(std::forward<TUserFunc>(func), image);
}

// RuntimeImageView visitor - shares same implementation than the one for
// RuntimeImage
template <class TUserFunc, class TPredicate = IntensityImagePredicate>
void visitImage(TUserFunc&& func, RuntimeImageView<TPredicate> const& image) {
  using RuntimeImageView = RuntimeImageView<TPredicate>;
  detail::VisitImpl<
      TUserFunc,
      RuntimeImageView,
      typename TPredicate::PixelVariant>::
      visit(std::forward<TUserFunc>(func), image);
}

}  // namespace sophus
