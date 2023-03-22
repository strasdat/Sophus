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
class ImageLayout;
struct PixelFormat;

SOPHUS_ENUM(NumberType, (fixed_point, floating_point));

namespace concepts {

template <class TT>
concept ImageSizeTrait = requires(TT self) {
  // group operations
  { self.width() } -> SameAs<int>;

  { self.height() } -> SameAs<int>;

  { self.area() } -> SameAs<size_t>;
};

// Ideally, the LieSubgroupFunctions is not necessary and all these
// properties can be deduced.
template <class TT>
concept ImageLayoutTrait = ImageSizeTrait<TT> && requires(TT self) {
  { self.sizeBytes() } -> SameAs<size_t>;

  { self.pitchBytes() } -> SameAs<size_t>;

  { self.isEmpty() } -> SameAs<bool>;

  { self.imageSize() } -> ConvertibleTo<sophus::ImageSize>;
};

template <class TT>
concept ImageView = ImageLayoutTrait<TT> && requires(TT self) {
  { self.layout() } -> ConvertibleTo<sophus::ImageLayout>;
};

template <class TT>
concept DynImageView = ImageLayoutTrait<TT> && requires(TT self) {
  { self.layout() } -> ConvertibleTo<sophus::ImageLayout>;
  { self.pixelFormat() } -> ConvertibleTo<PixelFormat>;
};

}  // namespace concepts
}  // namespace sophus
