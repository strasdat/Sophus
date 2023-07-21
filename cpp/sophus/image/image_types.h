// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/enum.h"
#include "sophus/image/image.h"

namespace sophus {

// Pixel type defs:
template <class TChannel>
using Pixel2 = Eigen::Matrix<TChannel, 2, 1>;
using Pixel2U8 = Pixel2<uint8_t>;
using Pixel2U16 = Pixel2<uint16_t>;
using Pixel2F32 = Pixel2<float>;

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

static_assert(concepts::ImageView<ImageViewF32>);

using MutImageViewU8 = MutImageView<uint8_t>;
using MutImageViewU16 = MutImageView<uint16_t>;
using MutImageViewF32 = MutImageView<float>;

static_assert(concepts::ImageView<MutImageViewF32>);

template <class TChannel>
using ImageView3 = ImageView<Pixel3<TChannel>>;
using ImageView3U8 = ImageView3<uint8_t>;
using ImageView3U16 = ImageView3<uint16_t>;
using ImageView3F32 = ImageView3<float>;

static_assert(concepts::ImageView<ImageView3F32>);

template <class TChannel>
using MutImageView3 = MutImageView<Pixel3<TChannel>>;
using MutImageView3U8 = MutImageView3<uint8_t>;
using MutImageView3U16 = MutImageView3<uint16_t>;
using MutImageView3F32 = MutImageView3<float>;

static_assert(concepts::ImageView<MutImageView3F32>);

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

template <class TT>
struct ImageTraits {
  static int const kNumChannels = 1;
  using TPixel = TT;
  using ChannelT = TPixel;
  static_assert(
      std::is_floating_point_v<ChannelT> || std::is_unsigned_v<ChannelT>);
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
auto isEqualMask(ImageView<TPixel> lhs, ImageView<TPixel> rhs) -> MutImageBool {
  return MutImageBool::makeFromTransform(
      lhs, rhs, [](TPixel lhs, TPixel rhs) { return lhs == rhs; });
}

/// Returns boolean image with the result per pixel:
///
/// mask(..) = lhs(..) < rhs (..)
template <class TPixel>
auto isLessMask(ImageView<TPixel> lhs, ImageView<TPixel> rhs) -> MutImageBool {
  return MutImageBool::makeFromTransform(
      lhs, rhs, [](TPixel lhs, TPixel rhs) { return lhs < rhs; });
}

/// Returns boolean image with the result per pixel:
///
/// mask(..) = lhs(..) > rhs (..)
template <class TPixel>
auto isGreaterMask(ImageView<TPixel> lhs, ImageView<TPixel> rhs)
    -> MutImageBool {
  return MutImageBool::makeFromTransform(
      lhs, rhs, [](TPixel lhs, TPixel rhs) { return lhs > rhs; });
}

/// Returns boolean image with the result per pixel:
///
/// mask(..) = ||lhs(..), rhs (..)|| <= thr
template <class TPixel>
auto isNearMask(
    ImageView<TPixel> lhs,
    ImageView<TPixel> rhs,
    typename ImageTraits<TPixel>::ChannelT thr) -> MutImageBool {
  return MutImageBool::makeFromTransform(
      lhs, rhs, [thr](TPixel lhs, TPixel rhs) {
        return MaxMetric(lhs, rhs) <= thr;
      });
}

/// Returns number of pixels equal `truth_value` in mask.
auto count(ImageViewBool mask, bool truth_value) -> int;

/// Returns number of true pixels in mask.
auto countTrue(ImageViewBool mask) -> int;

/// Returns number of false pixels in mask.
auto countFalse(ImageViewBool mask) -> int;

/// Returns true if all pixels are true.
auto isAllTrue(ImageViewBool mask) -> bool;

/// Returns true if at least one pixel is true.
auto isAnyTrue(ImageViewBool mask) -> bool;

/// Returns boolean image with the result per pixel:
///
/// out_mask(..) = !mask(..)
[[nodiscard]] auto neg(ImageViewBool mask) -> MutImageBool;

/// Returns first pixel of mask which equals `truth_value`, nullopt otherwise.
auto firstPixel(ImageViewBool mask, bool truth_value)
    -> std::optional<Eigen::Vector2i>;

/// Returns first true pixel, nullopt otherwise.
auto firstTruePixel(ImageViewBool mask) -> std::optional<Eigen::Vector2i>;

/// Returns first false pixel, nullopt otherwise.
auto firstFalsePixel(ImageViewBool mask) -> std::optional<Eigen::Vector2i>;

/// If it is false that `left_image` == `right_image`, print formatted error
/// message and then panic.
#define SOPHUS_ASSERT_IMAGE_EQ(left_image, right_image, ...)                  \
  SOPHUS_ASSERT_EQ(                                                           \
      (left_image).imageSize(),                                               \
      (right_image).imageSize(),                                              \
      "Inside: SOPHUS_ASSERT_IMAGE_EQ.");                                     \
  do {                                                                        \
    if (!(left_image).hasSameData(right_image)) {                             \
      ::sophus::MutImageBool mask = isEqualMask((left_image), (right_image)); \
      FARM_IMPL_LOG_HEADER("SOPHUS_ASSERT_IMAGE_EQ failed");                  \
      FARM_IMPL_LOG_PRINTLN(                                                  \
          "Number of pixel failing: {} / {}",                                 \
          countFalse(mask),                                                   \
          mask.imageSize().area());                                           \
      auto maybe_uv = firstFalsePixel(mask);                                  \
      ::Eigen::Vector2i uv = SOPHUS_UNWRAP(maybe_uv);                         \
      int u = uv[0];                                                          \
      int v = uv[1];                                                          \
      FARM_IMPL_LOG_PRINTLN(                                                  \
          "First failed pixel: ({},{}).\nLeft:\n{}\nRigth:\n{}",              \
          u,                                                                  \
          v,                                                                  \
          left_image(u, v),                                                   \
          right_image(u, v));                                                 \
      FARM_IMPL_LOG_PRINTLN(__VA_ARGS__);                                     \
      FARM_IMPL_ABORT();                                                      \
    }                                                                         \
  } while (false)

}  // namespace sophus
