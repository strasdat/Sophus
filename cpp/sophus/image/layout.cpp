// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/layout.h"

namespace sophus {

auto operator==(ImageLayout const& lhs, ImageLayout const& rhs) -> bool {
  return lhs.imageSize() == rhs.imageSize() &&
         lhs.pitchBytes() == rhs.pitchBytes();
}

auto operator!=(ImageLayout const& lhs, ImageLayout const& rhs) -> bool {
  return lhs.imageSize() != rhs.imageSize() ||
         lhs.pitchBytes() != rhs.pitchBytes();
}

auto operator<<(std::ostream& os, ImageLayout const& layout) -> std::ostream& {
  os << "[" << layout.imageSize() << ", pitch: " << layout.pitchBytes() << "]";
  return os;
}

}  // namespace sophus
