// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once
#include "sophus/common/enum.h"
#include "sophus/concepts/params.h"

namespace sophus {

struct ImageSize;
struct PixelFormat;

SOPHUS_ENUM(NumberType, (fixed_point, floating_point));

namespace concepts {

template <class TT>
concept ImageSize = requires(TT self) {
  // group operations
  { self.width() } -> ConvertibleTo<int>;

  { self.height() } -> ConvertibleTo<int>;

  { self.area() } -> ConvertibleTo<int>;
};

// Ideally, the LieSubgroupFunctions is not necessary and all these
// properties can be deduced.
template <class TT>
concept ImageLayout = ImageSize<TT> && requires(TT self) {
  { self.sizeBytes() } -> ConvertibleTo<int>;

  { self.pitchBytes() } -> ConvertibleTo<int>;

  { self.isEmpty() } -> ConvertibleTo<bool>;

  { self.imageSize() } -> ConvertibleTo<sophus::ImageSize>;
};

template <class TT>
concept DynImageView = ImageLayout<TT> && requires(TT self) {
  { self.pixelFormat() } -> ConvertibleTo<PixelFormat>;
};

}  // namespace concepts
}  // namespace sophus
