// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image/image.h"

#include <farm_ng/core/enum/enum.h>

namespace sophus {

// Pixel type defs:
template <class TChannel>
using Pixel3 = Eigen::Matrix<TChannel, 3, 1>;
using Pixel3U8 = Pixel3<uint8_t>;
using Pixel3U16 = Pixel3<uint16_t>;
using Pixel3F32 = Pixel3<float>;

template <class TChannel>
using Pixel4 = Eigen::Matrix<TChannel, 4, 1>;
using Pixel4U8 = Pixel4<uint8_t>;
using Pixel4U16 = Pixel4<uint16_t>;
using Pixel4F32 = Pixel4<float>;

// Image view type defs:
using ImageViewBool = ImageView<bool>;
using MutImageViewBool = MutImageView<bool>;

using ImageViewU8 = ImageView<uint8_t>;
using ImageViewU16 = ImageView<uint16_t>;
using ImageViewF32 = ImageView<float>;

using MutImageViewU8 = MutImageView<uint8_t>;
using MutImageViewU16 = MutImageView<uint16_t>;
using MutImageViewF32 = MutImageView<float>;

template <class TChannel>
using ImageView3 = ImageView<Pixel3<TChannel>>;
using ImageView3U8 = ImageView3<uint8_t>;
using ImageView3U16 = ImageView3<uint16_t>;
using ImageView3F32 = ImageView3<float>;

template <class TChannel>
using MutImageView3 = MutImageView<Pixel3<TChannel>>;
using MutImageView3U8 = MutImageView3<uint8_t>;
using MutImageView3U16 = MutImageView3<uint16_t>;
using MutImageView3F32 = MutImageView3<float>;

template <class TChannel>
using ImageView4 = ImageView<Pixel4<TChannel>>;
using ImageView4U8 = ImageView4<uint8_t>;
using ImageView4U16 = ImageView4<uint16_t>;
using ImageView4F32 = ImageView4<float>;

template <class TChannel>
using MutImageView4 = MutImageView<Pixel4<TChannel>>;
using MutImageView4U8 = MutImageView4<uint8_t>;
using MutImageView4U16 = MutImageView4<uint16_t>;
using MutImageView4F32 = MutImageView4<float>;

// Image type defs:
using ImageBool = Image<bool>;
using MutImageBool = MutImage<bool>;

using ImageU8 = Image<uint8_t>;
using ImageU16 = Image<uint16_t>;
using ImageF32 = Image<float>;

using MutImageU8 = MutImage<uint8_t>;
using MutImageU16 = MutImage<uint16_t>;
using MutImageF32 = MutImage<float>;

template <class TChannel>
using Image3 = Image<Pixel3<TChannel>>;
using Image3U8 = Image3<uint8_t>;
using Image3U16 = Image3<uint16_t>;
using Image3F32 = Image3<float>;

template <class TChannel>
using MutImage3 = MutImage<Pixel3<TChannel>>;
using MutImage3U8 = MutImage3<uint8_t>;
using MutImage3U16 = MutImage3<uint16_t>;
using MutImage3F32 = MutImage3<float>;

template <class TChannel>
using Image4 = Image<Pixel4<TChannel>>;
using Image4U8 = Image4<uint8_t>;
using Image4U16 = Image4<uint16_t>;
using Image4F32 = Image4<float>;

template <class TChannel>
using MutImage4 = MutImage<Pixel4<TChannel>>;
using MutImage4U8 = MutImage4<uint8_t>;
using MutImage4U16 = MutImage4<uint16_t>;
using MutImage4F32 = MutImage4<float>;

/// Number type.
FARM_ENUM(NumberType, (fixed_point, floating_point));

template <class TT>
struct ImageTraits {
  static int const kNumChannels = 1;
  using TPixel = TT;
  using ChannelT = TPixel;
  // static_assert(
  //     std::is_floating_point_v<ChannelT> || std::is_unsigned_v<ChannelT>);
};

template <class TT, int kNumChannelsT>
struct ImageTraits<Eigen::Matrix<TT, kNumChannelsT, 1>> {
  static int const kNumChannels = kNumChannelsT;
  using TPixel = Eigen::Matrix<TT, kNumChannels, 1>;
  using ChannelT = TT;
  static_assert(
      std::is_floating_point_v<ChannelT> || std::is_unsigned_v<ChannelT>);
};

/// Returns boolean image with the result per pixel:
///
/// mask(..) = lhs(..) == rhs (..)
template <class TPixel>
MutImageBool isEqualMask(ImageView<TPixel> lhs, ImageView<TPixel> rhs) {
  return MutImageBool::makeFromTransform(
      lhs, rhs, [](TPixel lhs, TPixel rhs) { return lhs == rhs; });
}

/// Returns boolean image with the result per pixel:
///
/// mask(..) = lhs(..) < rhs (..)
template <class TPixel>
MutImageBool isLessMask(ImageView<TPixel> lhs, ImageView<TPixel> rhs) {
  return MutImageBool::makeFromTransform(
      lhs, rhs, [](TPixel lhs, TPixel rhs) { return lhs < rhs; });
}

/// Returns boolean image with the result per pixel:
///
/// mask(..) = lhs(..) > rhs (..)
template <class TPixel>
MutImageBool isGreaterMask(ImageView<TPixel> lhs, ImageView<TPixel> rhs) {
  return MutImageBool::makeFromTransform(
      lhs, rhs, [](TPixel lhs, TPixel rhs) { return lhs > rhs; });
}

/// Returns boolean image with the result per pixel:
///
/// mask(..) = ||lhs(..), rhs (..)|| <= thr
template <class TPixel>
MutImageBool isNearMask(
    ImageView<TPixel> lhs,
    ImageView<TPixel> rhs,
    typename ImageTraits<TPixel>::ChannelT thr) {
  return MutImageBool::makeFromTransform(
      lhs, rhs, [thr](TPixel lhs, TPixel rhs) {
        return MaxMetric(lhs, rhs) <= thr;
      });
}

/// Returns number of pixels equal `truth_value` in mask.
int count(ImageViewBool mask, bool truth_value);

/// Returns number of true pixels in mask.
int countTrue(ImageViewBool mask);

/// Returns number of false pixels in mask.
int countFalse(ImageViewBool mask);

/// Returns true if all pixels are true.
bool isAllTrue(ImageViewBool mask);

/// Returns true if at least one pixel is true.
bool isAnyTrue(ImageViewBool mask);

/// Returns boolean image with the result per pixel:
///
/// out_mask(..) = !mask(..)
[[nodiscard]] MutImageBool neg(ImageViewBool mask);

/// Returns first pixel of mask which equals `truth_value`, nullopt otherwise.
std::optional<Eigen::Vector2i> firstPixel(ImageViewBool mask, bool truth_value);

/// Returns first true pixel, nullopt otherwise.
std::optional<Eigen::Vector2i> firstTruePixel(ImageViewBool mask);

/// Returns first false pixel, nullopt otherwise.
std::optional<Eigen::Vector2i> firstFalsePixel(ImageViewBool mask);

/// If it is false that `left_image` == `right_image`, print formatted error
/// message and then panic.
#define FARM_CHECK_IMAGE_EQ(left_image, right_image, ...)                     \
  FARM_CHECK_EQ(                                                              \
      (left_image).imageSize(),                                               \
      (right_image).imageSize(),                                              \
      "Inside: FARM_CHECK_IMAGE_EQ.");                                        \
  do {                                                                        \
    if (!(left_image).hasSameData(right_image)) {                             \
      ::sophus::MutImageBool mask = isEqualMask((left_image), (right_image)); \
      FARM_IMPL_LOG_HEADER("FARM_CHECK_IMAGE_EQ failed");                     \
      FARM_IMPL_LOG_PRINTLN(                                                  \
          "Number of pixel failing: {} / {}",                                 \
          countFalse(mask),                                                   \
          mask.imageSize().area());                                           \
      auto maybe_uv = firstFalsePixel(mask);                                  \
      ::Eigen::Vector2i uv = FARM_UNWRAP(maybe_uv);                           \
      int u = uv[0];                                                          \
      int v = uv[1];                                                          \
      FARM_IMPL_LOG_PRINTLN(                                                  \
          "First failed pixel: ({},{}).\nLeft:\n{}\nRigth:\n{}",              \
          u,                                                                  \
          v,                                                                  \
          left_image.checked(u, v),                                           \
          right_image.checked(u, v));                                         \
      FARM_IMPL_LOG_PRINTLN(__VA_ARGS__);                                     \
      FARM_IMPL_ABORT();                                                      \
    }                                                                         \
  } while (false)

}  // namespace sophus
