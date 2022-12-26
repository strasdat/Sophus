// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/geometry/inverse_depth.h"

#include "sophus/lie/se3.h"

#include <gtest/gtest.h>

using namespace sophus;

TEST(inverse_depth, integrations) {
  for (Eigen::Vector3d const &point : {Eigen::Vector3d(0.1, 0.3, 2.0)}) {
    auto inv_depth_point = InverseDepthPoint3F64::fromEuclideanPoint3(point);
    Eigen::Vector3d point2 = inv_depth_point.toEuclideanPoint3();

    SOPHUS_ASSERT_NEAR(point, point2, kEpsilonF64);
  }
}
