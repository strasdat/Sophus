// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/common/common.h"

#include <gtest/gtest.h>

using namespace sophus;

TEST(check_near, unit) {
  SOPHUS_ASSERT_NEAR(
      Eigen::Vector3d(1.0, 2.0, 0.0),
      Eigen::Vector3d(1.0, 2.0, 0.0),
      0.0000001);
  ASSERT_DEATH(
      {
        SOPHUS_ASSERT_NEAR(
            Eigen::Vector3d(1.0, 2.0, 0.001),
            Eigen::Vector3d(1.0, 2.0, 0.0),
            0.00001);
      },
      "ASSERT_NEAR");
  SOPHUS_ASSERT_NEAR(
      Eigen::Vector3d(1.0, 2.0, 999.001),
      Eigen::Vector3d(1.0, 2.0, 999.0),
      0.0001);
}
