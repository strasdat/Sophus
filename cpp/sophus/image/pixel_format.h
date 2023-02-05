// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image/image_types.h"

namespace sophus {

struct PixelFormat {
  template <class TPixel>
  static PixelFormat fromTemplate() {
    return PixelFormat{
        .number_type =
            std::is_floating_point_v<typename ImageTraits<TPixel>::ChannelT>
                ? NumberType::floating_point
                : NumberType::fixed_point,
        .num_channels = ImageTraits<TPixel>::kNumChannels,
        .num_bytes_per_pixel_channel =
            sizeof(typename ImageTraits<TPixel>::ChannelT)};
  }

  [[nodiscard]] inline int bytesPerPixel() const {
    return num_channels * num_bytes_per_pixel_channel;
  }

  template <class TPixel>
  [[nodiscard]] bool is() {
    return fromTemplate<TPixel>() == *this;
  }

  NumberType number_type;
  int num_channels;
  int num_bytes_per_pixel_channel;
};

bool operator==(PixelFormat const& lhs, PixelFormat const& rhs);

/// Example:
/// PixelFormat::fromTemplate<float>() outputs: "1F32";
/// PixelFormat::fromTemplate<Eigen::Matrix<uint8_t,4,1>>() outputs:
/// "4U8";
std::ostream& operator<<(std::ostream& os, PixelFormat const& type);
}  // namespace sophus
