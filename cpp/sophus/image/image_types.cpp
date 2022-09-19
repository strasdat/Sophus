// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/image_types.h"

namespace sophus {

int count(ImageViewBool mask, bool truth_value) {
  return mask.reduce(
      [truth_value](bool pixel, int& true_count) {
        true_count += int(pixel == truth_value);
      },
      0);
}

int countTrue(ImageViewBool mask) { return count(mask, true); }

int countFalse(ImageViewBool mask) { return count(mask, false); }

bool isAllTrue(ImageViewBool mask) {
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

bool isAnyTrue(ImageViewBool mask) {
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

MutImageBool neg(ImageViewBool mask) {
  return MutImageBool::makeFromTransform(mask, [](bool val) { return !val; });
}

std::optional<Eigen::Vector2i> firstPixel(
    ImageViewBool mask, bool truth_value) {
  for (int v = 0; v < mask.height(); ++v) {
    const bool* p = mask.rowPtr(v);
    for (int u = 0; u < mask.width(); ++u) {
      if (p[u] == truth_value) {
        return Eigen::Vector2i(u, v);
      }
    }
  }
  return std::nullopt;
}

std::optional<Eigen::Vector2i> firstTruePixel(ImageViewBool mask) {
  return firstPixel(mask, true);
}

std::optional<Eigen::Vector2i> firstFalsePixel(ImageViewBool mask) {
  return firstPixel(mask, false);
}

}  // namespace sophus
