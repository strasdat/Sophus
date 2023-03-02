// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/image_types.h"

namespace sophus {

auto count(ImageViewBool mask, bool truth_value) -> int {
  return mask.reduce(
      [truth_value](bool pixel, int& true_count) {
        true_count += int(pixel == truth_value);
      },
      0);
}

auto countTrue(ImageViewBool mask) -> int { return count(mask, true); }

auto countFalse(ImageViewBool mask) -> int { return count(mask, false); }

auto isAllTrue(ImageViewBool mask) -> bool {
  return mask.shortCircuitReduce(
      [](bool pixel, bool& is_all_true) {
        if (!pixel) {
          is_all_true = false;
          return true;
        }
        return false;
      },
      true);
}

auto isAnyTrue(ImageViewBool mask) -> bool {
  return mask.shortCircuitReduce(
      [](bool pixel, bool& is_any_true) {
        if (pixel) {
          is_any_true = true;
          return true;
        }
        return false;
      },
      false);
}

auto neg(ImageViewBool mask) -> MutImageBool {
  return MutImageBool::makeFromTransform(mask, [](bool val) { return !val; });
}

auto firstPixel(ImageViewBool mask, bool truth_value)
    -> std::optional<Eigen::Vector2i> {
  for (int v = 0; v < mask.height(); ++v) {
    bool const* p = mask.rowPtr(v);
    for (int u = 0; u < mask.width(); ++u) {
      if (p[u] == truth_value) {
        return Eigen::Vector2i(u, v);
      }
    }
  }
  return std::nullopt;
}

auto firstTruePixel(ImageViewBool mask) -> std::optional<Eigen::Vector2i> {
  return firstPixel(mask, true);
}

auto firstFalsePixel(ImageViewBool mask) -> std::optional<Eigen::Vector2i> {
  return firstPixel(mask, false);
}

}  // namespace sophus
