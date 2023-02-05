// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/dyn_image_view.h"

#include <gtest/gtest.h>

using namespace sophus;

#define SOPHUS_TEST_IMG_EQ(lhs, rhs) /* NOLINT*/        \
  do {                                                  \
    SOPHUS_ASSERT_EQ(lhs.imageSize(), rhs.imageSize()); \
    for (int v = 0; v < lhs.imageSize().height; ++v) {  \
      for (int u = 0; u < lhs.imageSize().width; ++u) { \
        SOPHUS_ASSERT_EQ(lhs(u, v), rhs(u, v));         \
      }                                                 \
    }                                                   \
  } while (false)
