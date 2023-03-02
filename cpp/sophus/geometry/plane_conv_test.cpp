// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/geometry/plane_conv.h"

#include <gtest/gtest.h>

using namespace sophus;

namespace {

template <class TScalar>
auto test2dGeometry() -> bool {
  bool passed = true;
  TScalar const eps = 10.0 * kEpsilon<TScalar>;

  for (int i = 0; i < 20; ++i) {
    // Roundtrip test:
    Eigen::Vector2<TScalar> normal_in_foo =
        Eigen::Vector2<TScalar>::Random().normalized();
    sophus::Rotation2<TScalar> foo_rotation_plane =
        rotation2FromNormal(normal_in_foo);
    Eigen::Vector2<TScalar> result_normal_foo =
        normalFromRotation2(foo_rotation_plane);
    SOPHUS_ASSERT_NEAR(normal_in_foo, result_normal_foo, eps, "");
  }

  for (int i = 0; i < 20; ++i) {
    // Roundtrip test:
    Eigen::Hyperplane<TScalar, 2> line_in_foo =
        makeHyperplaneUnique(Eigen::Hyperplane<TScalar, 2>(
            Eigen::Vector2<TScalar>::Random().normalized(),
            Eigen::Vector2<TScalar>::Random()));
    sophus::Isometry2<TScalar> foo_from_plane = isometryFromLine(line_in_foo);
    Eigen::Hyperplane<TScalar, 2> result_plane_foo =
        lineFromIsometry(foo_from_plane);
    SOPHUS_ASSERT_NEAR(
        line_in_foo.normal().eval(), result_plane_foo.normal().eval(), eps, "");
    SOPHUS_ASSERT_NEAR(
        line_in_foo.offset(), result_plane_foo.offset(), eps, "");
  }

  for (auto params : Isometry2<TScalar>::paramsExamples()) {
    Isometry2<TScalar> const& foo_from_line =
        Isometry2<TScalar>::fromParams(params);
    Eigen::Hyperplane<TScalar, 2> line_in_foo = lineFromIsometry(foo_from_line);
    Isometry2<TScalar> t2_foo_line = isometryFromLine(line_in_foo);
    Eigen::Hyperplane<TScalar, 2> line2_foo = lineFromIsometry(t2_foo_line);
    SOPHUS_ASSERT_NEAR(
        line_in_foo.normal().eval(), line2_foo.normal().eval(), eps, "");
    SOPHUS_ASSERT_NEAR(line_in_foo.offset(), line2_foo.offset(), eps, "");
  }

  return passed;
}

template <class TScalar>
auto test3dGeometry() -> bool {
  bool passed = true;
  TScalar const eps = 10.0 * kEpsilon<TScalar>;

  Eigen::Vector3<TScalar> normal_in_foo =
      Eigen::Vector3<TScalar>(1, 2, 0).normalized();
  Eigen::Matrix3<TScalar> foo_rotation_plane =
      rotation3FromNormal(normal_in_foo);
  SOPHUS_ASSERT_NEAR(normal_in_foo, foo_rotation_plane.col(2).eval(), eps, "");
  // Just testing that the function normalizes the input normal and hint
  // direction correctly:
  Eigen::Matrix3<TScalar> r2_foo_plane =
      rotation3FromNormal((TScalar(0.9) * normal_in_foo).eval());
  SOPHUS_ASSERT_NEAR(normal_in_foo, r2_foo_plane.col(2).eval(), eps, "");
  Eigen::Matrix3<TScalar> r3_foo_plane = rotation3FromNormal(
      normal_in_foo,
      Eigen::Vector3<TScalar>(TScalar(0.9), TScalar(0), TScalar(0)),
      Eigen::Vector3<TScalar>(TScalar(0), TScalar(1.1), TScalar(0)));
  SOPHUS_ASSERT_NEAR(normal_in_foo, r3_foo_plane.col(2).eval(), eps, "");

  normal_in_foo = Eigen::Vector3<TScalar>(1, 0, 0);
  foo_rotation_plane = rotation3FromNormal(normal_in_foo);
  SOPHUS_ASSERT_NEAR(normal_in_foo, foo_rotation_plane.col(2).eval(), eps, "");
  SOPHUS_ASSERT_NEAR(
      Eigen::Vector3<TScalar>(0, 1, 0),
      foo_rotation_plane.col(1).eval(),
      eps,
      "");

  normal_in_foo = Eigen::Vector3<TScalar>(0, 1, 0);
  foo_rotation_plane = rotation3FromNormal(normal_in_foo);
  SOPHUS_ASSERT_NEAR(normal_in_foo, foo_rotation_plane.col(2).eval(), eps, "");
  SOPHUS_ASSERT_NEAR(
      Eigen::Vector3<TScalar>(1, 0, 0),
      foo_rotation_plane.col(0).eval(),
      eps,
      "");

  for (int i = 0; i < 20; ++i) {
    // Roundtrip test:
    Eigen::Vector3<TScalar> normal_in_foo =
        Eigen::Vector3<TScalar>::Random().normalized();
    sophus::Rotation3<TScalar> foo_rotation_plane =
        rotation3FromPlane(normal_in_foo);
    Eigen::Vector3<TScalar> result_normal_foo =
        normalFromRotation3(foo_rotation_plane);
    SOPHUS_ASSERT_NEAR(normal_in_foo, result_normal_foo, eps, "");
  }

  for (int i = 0; i < 20; ++i) {
    // Roundtrip test:
    Eigen::Hyperplane<TScalar, 3> plane_in_foo =
        makeHyperplaneUnique(Eigen::Hyperplane<TScalar, 3>(
            Eigen::Vector3<TScalar>::Random().normalized(),
            Eigen::Vector3<TScalar>::Random()));
    sophus::Isometry3<TScalar> foo_from_plane = isometryFromPlane(plane_in_foo);
    Eigen::Hyperplane<TScalar, 3> result_plane_foo =
        planeFromIsometry(foo_from_plane);
    SOPHUS_ASSERT_NEAR(
        plane_in_foo.normal().eval(),
        result_plane_foo.normal().eval(),
        eps,
        "");
    SOPHUS_ASSERT_NEAR(
        plane_in_foo.offset(), result_plane_foo.offset(), eps, "");
  }

  for (auto params : Isometry3<TScalar>::paramsExamples()) {
    Isometry3<TScalar> const& foo_from_plane =
        Isometry3<TScalar>::fromParams(params);
    Eigen::Hyperplane<TScalar, 3> plane_in_foo =
        planeFromIsometry(foo_from_plane);
    Isometry3<TScalar> t2_foo_plane = isometryFromPlane(plane_in_foo);
    Eigen::Hyperplane<TScalar, 3> plane2_foo = planeFromIsometry(t2_foo_plane);
    SOPHUS_ASSERT_NEAR(
        plane_in_foo.normal().eval(), plane2_foo.normal().eval(), eps, "");
    SOPHUS_ASSERT_NEAR(plane_in_foo.offset(), plane2_foo.offset(), eps, "");
  }

  return passed;
}
}  // namespace

TEST(plane_conv, integrations) {
  test2dGeometry<double>();
  test3dGeometry<double>();
  test2dGeometry<float>();
  test3dGeometry<float>();
}

// int main() { sophus::runAll(); }
