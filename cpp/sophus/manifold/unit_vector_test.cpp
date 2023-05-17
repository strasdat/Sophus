// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/manifold/unit_vector.h"

#include "sophus/concepts/manifold_prop_tests.h"

#include <gtest/gtest.h>
namespace sophus::test {

TEST(unitvec, unit2) {
  EXPECT_NO_FATAL_FAILURE(UnitVector2<double>::fromUnitVector({1.0, 0.0}));
  EXPECT_NO_FATAL_FAILURE(UnitVector2<double>::fromUnitVector({0.0, 1.0}));
  EXPECT_FALSE(UnitVector2<double>::tryFromUnitVector({0.0, 0.5}));
}

TEST(unitvec, unit3) {
  EXPECT_NO_FATAL_FAILURE(UnitVector3<double>::fromUnitVector({1.0, 0.0, 0.0}));
  EXPECT_NO_FATAL_FAILURE(UnitVector3<double>::fromUnitVector({0.0, 1.0, 0.0}));
  EXPECT_NO_FATAL_FAILURE(UnitVector3<double>::fromUnitVector({0.0, 0.0, 1.0}));
  EXPECT_FALSE(UnitVector3<double>::tryFromUnitVector({0.0, 0.5, 0.0}));
}

TEST(unitvec, copy3) {
  auto a = UnitVector3<double>::fromUnitVector({1.0, 0.0, 0.0});
  auto b = UnitVector3<double>::fromUnitVector({0.0, 1.0, 0.0});
  UnitVector3<double> c = a;
  SOPHUS_ASSERT_NEAR(a.vector(), c.vector(), kEpsilonF64);

  c = UnitVector3<double>(b);
  SOPHUS_ASSERT_NEAR(b.vector(), c.vector(), kEpsilonF64);

  auto d = (a = b);
  SOPHUS_ASSERT_NEAR(d.vector(), a.vector(), kEpsilonF64);
  SOPHUS_ASSERT_NEAR(d.vector(), b.vector(), kEpsilonF64);
}

TEST(unitvec, manifold_prop_tests) {
  // ManifoldPropTestSuite<UnitVector2<double>>::runAllTests("UnitVector");
  // ManifoldPropTestSuite<UnitVector2<float>>::runAllTests("UnitVector");
  ManifoldPropTestSuite<UnitVector3<double>>::runAllTests("UnitVector");
  ManifoldPropTestSuite<UnitVector3<float>>::runAllTests("UnitVector");
}
}  // namespace sophus::test

// TEST(unitvec, testRotThroughPoints) {
//   std::default_random_engine generator(0);

//   for (size_t trial = 0; trial < 500; trial++) {
//     std::normal_distribution<double> normal(0, 10);

//     Eigen::Vector3d point_from(
//         normal(generator), normal(generator), normal(generator));
//     Eigen::Vector3d point_to(
//         normal(generator), normal(generator), normal(generator));

//     if (point_from.norm() > kEpsilonF64 && point_to.norm() > kEpsilonF64) {
//       std::cerr << point_from.transpose() << std::endl;
//       std::cerr << point_to.transpose() << std::endl;

//       Rotation3F64 to_rot_from = rotThroughPoints(
//           UnitVector3F64::fromVectorAndNormalize(point_from),
//           UnitVector3F64::fromVectorAndNormalize(point_to));
//       Rotation3F64 to_rot_from2 = rotThroughPoints(point_from, point_to);

//       // Check that the resulting rotation can take ``from`` into a vector
//       // collinear with ``to``
//       SOPHUS_ASSERT_NEAR(
//           point_to.cross(to_rot_from * point_from).norm(), 0.0, kEpsilonF64);
//       SOPHUS_ASSERT_NEAR(
//           point_to.cross(to_rot_from2 * point_from).norm(), 0.0,
//           kEpsilonF64);

//       // And the reverse as a sanity check
//       SOPHUS_ASSERT_NEAR(
//           point_from.cross(to_rot_from.inverse() * point_to).norm(),
//           0.0,
//           kEpsilonF64);
//       SOPHUS_ASSERT_NEAR(
//           point_from.cross(to_rot_from2.inverse() * point_to).norm(),
//           0.0,
//           kEpsilonF64);
//     }
//   }
// }
