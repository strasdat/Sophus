// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/runtime_image.h"

namespace sophus {

bool operator==(const RuntimePixelType& lhs, const RuntimePixelType& rhs) {
  return lhs.number_type == rhs.number_type &&
         lhs.num_channels == rhs.num_channels &&
         lhs.num_bytes_per_pixel_channel == rhs.num_bytes_per_pixel_channel;
}

std::ostream& operator<<(std::ostream& os, const RuntimePixelType& type) {
  std::string type_char = "?";
  switch (type.number_type) {
    case NumberType::fixed_point: {
      type_char = "U";
      break;
    }
    case NumberType::floating_point: {
      type_char = "F";
      break;
    }
  }
  os << FARM_FORMAT(
      "{}{}{}",
      type.num_channels,
      type_char,
      8 * type.num_bytes_per_pixel_channel);
  return os;
}

}  // namespace sophus
