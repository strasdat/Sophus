// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/layout.h"

namespace sophus {

bool operator==(ImageLayout const& lhs, ImageLayout const& rhs) {
  return lhs.imageSize() == rhs.imageSize() &&
         lhs.pitchBytes() == rhs.pitchBytes();
}

bool operator!=(ImageLayout const& lhs, ImageLayout const& rhs) {
  return lhs.imageSize() != rhs.imageSize() ||
         lhs.pitchBytes() != rhs.pitchBytes();
}

std::ostream& operator<<(std::ostream& os, ImageLayout const& layout) {
  os << "[" << layout.imageSize() << ", pitch: " << layout.pitchBytes() << "]";
  return os;
}

}  // namespace sophus
