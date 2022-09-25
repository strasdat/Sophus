// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/geometry/plane_conv.h"

#include "sophus/common/test_macros.h"
#include "sophus/lie/details/test_impl.h"

#include <iostream>

namespace sophus {

namespace {

template <class TScalar>
bool test2dGeometry() {
  bool passed = true;
  TScalar const eps = kEpsilon<TScalar>;

  for (int i = 0; i < 20; ++i) {
    // Roundtrip test:
    Eigen::Vector2<TScalar> normal_in_foo =
        Eigen::Vector2<TScalar>::Random().normalized();
    sophus::So2<TScalar> foo_rotation_plane = so2FromNormal(normal_in_foo);
    Eigen::Vector2<TScalar> result_normal_foo =
        normalFromSo2(foo_rotation_plane);
    SOPHUS_TEST_APPROX(passed, normal_in_foo, result_normal_foo, eps, "");
  }

  for (int i = 0; i < 20; ++i) {
    // Roundtrip test:
    Eigen::Hyperplane<TScalar, 2> line_in_foo =
        makeHyperplaneUnique(Eigen::Hyperplane<TScalar, 2>(
            Eigen::Vector2<TScalar>::Random().normalized(),
            Eigen::Vector2<TScalar>::Random()));
    sophus::Se2<TScalar> foo_pose_plane = se2FromLine(line_in_foo);
    Eigen::Hyperplane<TScalar, 2> result_plane_foo =
        lineFromSe2(foo_pose_plane);
    SOPHUS_TEST_APPROX(
        passed,
        line_in_foo.normal().eval(),
        result_plane_foo.normal().eval(),
        eps,
        "");
    SOPHUS_TEST_APPROX(
        passed, line_in_foo.offset(), result_plane_foo.offset(), eps, "");
  }

  std::vector<Se2<TScalar>, Eigen::aligned_allocator<Se2<TScalar>>>
      ts_foo_line = getTestSE2s<TScalar>();

  for (Se2<TScalar> const& foo_pose_line : ts_foo_line) {
    Eigen::Hyperplane<TScalar, 2> line_in_foo = lineFromSe2(foo_pose_line);
    Se2<TScalar> t2_foo_line = se2FromLine(line_in_foo);
    Eigen::Hyperplane<TScalar, 2> line2_foo = lineFromSe2(t2_foo_line);
    SOPHUS_TEST_APPROX(
        passed,
        line_in_foo.normal().eval(),
        line2_foo.normal().eval(),
        eps,
        "");
    SOPHUS_TEST_APPROX(
        passed, line_in_foo.offset(), line2_foo.offset(), eps, "");
  }

  return passed;
}

template <class TScalar>
bool test3dGeometry() {
  bool passed = true;
  TScalar const eps = kEpsilon<TScalar>;

  Eigen::Vector3<TScalar> normal_in_foo =
      Eigen::Vector3<TScalar>(1, 2, 0).normalized();
  Eigen::Matrix3<TScalar> foo_rotation_plane =
      rotationFromNormal(normal_in_foo);
  SOPHUS_TEST_APPROX(
      passed, normal_in_foo, foo_rotation_plane.col(2).eval(), eps, "");
  // Just testing that the function normalizes the input normal and hint
  // direction correctly:
  Eigen::Matrix3<TScalar> r2_foo_plane =
      rotationFromNormal((TScalar(0.9) * normal_in_foo).eval());
  SOPHUS_TEST_APPROX(
      passed, normal_in_foo, r2_foo_plane.col(2).eval(), eps, "");
  Eigen::Matrix3<TScalar> r3_foo_plane = rotationFromNormal(
      normal_in_foo,
      Eigen::Vector3<TScalar>(TScalar(0.9), TScalar(0), TScalar(0)),
      Eigen::Vector3<TScalar>(TScalar(0), TScalar(1.1), TScalar(0)));
  SOPHUS_TEST_APPROX(
      passed, normal_in_foo, r3_foo_plane.col(2).eval(), eps, "");

  normal_in_foo = Eigen::Vector3<TScalar>(1, 0, 0);
  foo_rotation_plane = rotationFromNormal(normal_in_foo);
  SOPHUS_TEST_APPROX(
      passed, normal_in_foo, foo_rotation_plane.col(2).eval(), eps, "");
  SOPHUS_TEST_APPROX(
      passed,
      Eigen::Vector3<TScalar>(0, 1, 0),
      foo_rotation_plane.col(1).eval(),
      eps,
      "");

  normal_in_foo = Eigen::Vector3<TScalar>(0, 1, 0);
  foo_rotation_plane = rotationFromNormal(normal_in_foo);
  SOPHUS_TEST_APPROX(
      passed, normal_in_foo, foo_rotation_plane.col(2).eval(), eps, "");
  SOPHUS_TEST_APPROX(
      passed,
      Eigen::Vector3<TScalar>(1, 0, 0),
      foo_rotation_plane.col(0).eval(),
      eps,
      "");

  for (int i = 0; i < 20; ++i) {
    // Roundtrip test:
    Eigen::Vector3<TScalar> normal_in_foo =
        Eigen::Vector3<TScalar>::Random().normalized();
    sophus::So3<TScalar> foo_rotation_plane = so3FromPlane(normal_in_foo);
    Eigen::Vector3<TScalar> result_normal_foo =
        normalFromSo3(foo_rotation_plane);
    SOPHUS_TEST_APPROX(passed, normal_in_foo, result_normal_foo, eps, "");
  }

  for (int i = 0; i < 20; ++i) {
    // Roundtrip test:
    Eigen::Hyperplane<TScalar, 3> plane_in_foo =
        makeHyperplaneUnique(Eigen::Hyperplane<TScalar, 3>(
            Eigen::Vector3<TScalar>::Random().normalized(),
            Eigen::Vector3<TScalar>::Random()));
    sophus::Se3<TScalar> foo_pose_plane = se3FromPlane(plane_in_foo);
    Eigen::Hyperplane<TScalar, 3> result_plane_foo =
        planeFromSe3(foo_pose_plane);
    SOPHUS_TEST_APPROX(
        passed,
        plane_in_foo.normal().eval(),
        result_plane_foo.normal().eval(),
        eps,
        "");
    SOPHUS_TEST_APPROX(
        passed, plane_in_foo.offset(), result_plane_foo.offset(), eps, "");
  }

  std::vector<Se3<TScalar>, Eigen::aligned_allocator<Se3<TScalar>>>
      ts_foo_plane = getTestSE3s<TScalar>();

  for (Se3<TScalar> const& foo_pose_plane : ts_foo_plane) {
    Eigen::Hyperplane<TScalar, 3> plane_in_foo = planeFromSe3(foo_pose_plane);
    Se3<TScalar> t2_foo_plane = se3FromPlane(plane_in_foo);
    Eigen::Hyperplane<TScalar, 3> plane2_foo = planeFromSe3(t2_foo_plane);
    SOPHUS_TEST_APPROX(
        passed,
        plane_in_foo.normal().eval(),
        plane2_foo.normal().eval(),
        eps,
        "");
    SOPHUS_TEST_APPROX(
        passed, plane_in_foo.offset(), plane2_foo.offset(), eps, "");
  }

  return passed;
}

void runAll() {
  std::cerr << "Geometry (Lines/Planes) tests:" << std::endl;
  std::cerr << "Double tests: " << std::endl;
  bool passed = test2dGeometry<double>();
  passed = test3dGeometry<double>();
  processTestResult(passed);
  std::cerr << "Float tests: " << std::endl;
  passed = test2dGeometry<float>();
  passed = test3dGeometry<float>();
  processTestResult(passed);
}

}  // namespace
}  // namespace sophus

int main() { sophus::runAll(); }
