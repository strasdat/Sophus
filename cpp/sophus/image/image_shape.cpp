// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/image_shape.h"

namespace sophus {

bool operator==(ImageShape const& lhs, ImageShape const& rhs) {
  return lhs.imageSize() == rhs.imageSize() &&
         lhs.pitchBytes() == rhs.pitchBytes();
}

bool operator!=(ImageShape const& lhs, ImageShape const& rhs) {
  return lhs.imageSize() != rhs.imageSize() ||
         lhs.pitchBytes() != rhs.pitchBytes();
}

std::ostream& operator<<(std::ostream& os, ImageShape const& shape) {
  os << "[" << shape.imageSize() << ", pitch: " << shape.pitchBytes() << "]";
  return os;
}

}  // namespace sophus
