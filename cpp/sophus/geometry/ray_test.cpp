// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/geometry/ray.h"

#include <gtest/gtest.h>

using namespace sophus;

TEST(ray, simple_ray3) {
  auto const dir1 = UnitVector3<double>::fromUnitVector({0.0, 0.0, 1.0});
  auto const dir2 = UnitVector3<double>::fromUnitVector({1.0, 0.0, 0.0});
  Ray3<double> const line_a({0.0, 0.0, 0.0}, dir1);
  Ray3<double> const line_b({1.0, 0.0, 0.0}, dir1);
  Ray3<double> const line_c({1.0, 0.0, 0.0}, dir2);

  auto const intersect_a_b = closestApproachParameters(line_a, line_b);
  EXPECT_EQ(intersect_a_b, std::nullopt);

  auto const intersect_a_c = closestApproachParameters(line_a, line_c);
  EXPECT_TRUE((bool)intersect_a_c);
  EXPECT_EQ(intersect_a_c->lambda0, 0.0);
  EXPECT_EQ(intersect_a_c->lambda1, -1.0);

  auto maybe_mid_a_c = closestApproach(line_a, line_c);
  EXPECT_TRUE((bool)maybe_mid_a_c);
  SOPHUS_ASSERT_NEAR(
      SOPHUS_UNWRAP(maybe_mid_a_c),
      Eigen::Vector3d(0.0, 0.0, 0.0),
      kEpsilonF64);
}
