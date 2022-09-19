// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/geometry/plane_conv.h"

#include "sophus/core/test_macros.h"
#include "sophus/lie/details/test_impl.h"

#include <iostream>

namespace sophus {

namespace {

template <class ScalarT>
bool test2dGeometry() {
  bool passed = true;
  ScalarT const eps = kEpsilon<ScalarT>;

  for (int i = 0; i < 20; ++i) {
    // Roundtrip test:
    Eigen::Vector2<ScalarT> normal_in_foo =
        Eigen::Vector2<ScalarT>::Random().normalized();
    sophus::So2<ScalarT> foo_rotation_plane = so2FromNormal(normal_in_foo);
    Eigen::Vector2<ScalarT> result_normal_foo =
        normalFromSo2(foo_rotation_plane);
    SOPHUS_TEST_APPROX(passed, normal_in_foo, result_normal_foo, eps, "");
  }

  for (int i = 0; i < 20; ++i) {
    // Roundtrip test:
    Eigen::Hyperplane<ScalarT, 2> line_in_foo =
        makeHyperplaneUnique(Eigen::Hyperplane<ScalarT, 2>(
            Eigen::Vector2<ScalarT>::Random().normalized(),
            Eigen::Vector2<ScalarT>::Random()));
    sophus::Se2<ScalarT> foo_pose_plane = se2FromLine(line_in_foo);
    Eigen::Hyperplane<ScalarT, 2> result_plane_foo =
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

  std::vector<Se2<ScalarT>, Eigen::aligned_allocator<Se2<ScalarT>>>
      ts_foo_line = getTestSE2s<ScalarT>();

  for (Se2<ScalarT> const& foo_pose_line : ts_foo_line) {
    Eigen::Hyperplane<ScalarT, 2> line_in_foo = lineFromSe2(foo_pose_line);
    Se2<ScalarT> t2_foo_line = se2FromLine(line_in_foo);
    Eigen::Hyperplane<ScalarT, 2> line2_foo = lineFromSe2(t2_foo_line);
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

template <class ScalarT>
bool test3dGeometry() {
  bool passed = true;
  ScalarT const eps = kEpsilon<ScalarT>;

  Eigen::Vector3<ScalarT> normal_in_foo =
      Eigen::Vector3<ScalarT>(1, 2, 0).normalized();
  Eigen::Matrix3<ScalarT> foo_rotation_plane =
      rotationFromNormal(normal_in_foo);
  SOPHUS_TEST_APPROX(
      passed, normal_in_foo, foo_rotation_plane.col(2).eval(), eps, "");
  // Just testing that the function normalizes the input normal and hint
  // direction correctly:
  Eigen::Matrix3<ScalarT> r2_foo_plane =
      rotationFromNormal((ScalarT(0.9) * normal_in_foo).eval());
  SOPHUS_TEST_APPROX(
      passed, normal_in_foo, r2_foo_plane.col(2).eval(), eps, "");
  Eigen::Matrix3<ScalarT> r3_foo_plane = rotationFromNormal(
      normal_in_foo,
      Eigen::Vector3<ScalarT>(ScalarT(0.9), ScalarT(0), ScalarT(0)),
      Eigen::Vector3<ScalarT>(ScalarT(0), ScalarT(1.1), ScalarT(0)));
  SOPHUS_TEST_APPROX(
      passed, normal_in_foo, r3_foo_plane.col(2).eval(), eps, "");

  normal_in_foo = Eigen::Vector3<ScalarT>(1, 0, 0);
  foo_rotation_plane = rotationFromNormal(normal_in_foo);
  SOPHUS_TEST_APPROX(
      passed, normal_in_foo, foo_rotation_plane.col(2).eval(), eps, "");
  SOPHUS_TEST_APPROX(
      passed,
      Eigen::Vector3<ScalarT>(0, 1, 0),
      foo_rotation_plane.col(1).eval(),
      eps,
      "");

  normal_in_foo = Eigen::Vector3<ScalarT>(0, 1, 0);
  foo_rotation_plane = rotationFromNormal(normal_in_foo);
  SOPHUS_TEST_APPROX(
      passed, normal_in_foo, foo_rotation_plane.col(2).eval(), eps, "");
  SOPHUS_TEST_APPROX(
      passed,
      Eigen::Vector3<ScalarT>(1, 0, 0),
      foo_rotation_plane.col(0).eval(),
      eps,
      "");

  for (int i = 0; i < 20; ++i) {
    // Roundtrip test:
    Eigen::Vector3<ScalarT> normal_in_foo =
        Eigen::Vector3<ScalarT>::Random().normalized();
    sophus::So3<ScalarT> foo_rotation_plane = so3FromPlane(normal_in_foo);
    Eigen::Vector3<ScalarT> result_normal_foo =
        normalFromSo3(foo_rotation_plane);
    SOPHUS_TEST_APPROX(passed, normal_in_foo, result_normal_foo, eps, "");
  }

  for (int i = 0; i < 20; ++i) {
    // Roundtrip test:
    Eigen::Hyperplane<ScalarT, 3> plane_in_foo =
        makeHyperplaneUnique(Eigen::Hyperplane<ScalarT, 3>(
            Eigen::Vector3<ScalarT>::Random().normalized(),
            Eigen::Vector3<ScalarT>::Random()));
    sophus::Se3<ScalarT> foo_pose_plane = se3FromPlane(plane_in_foo);
    Eigen::Hyperplane<ScalarT, 3> result_plane_foo =
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

  std::vector<Se3<ScalarT>, Eigen::aligned_allocator<Se3<ScalarT>>>
      ts_foo_plane = getTestSE3s<ScalarT>();

  for (Se3<ScalarT> const& foo_pose_plane : ts_foo_plane) {
    Eigen::Hyperplane<ScalarT, 3> plane_in_foo = planeFromSe3(foo_pose_plane);
    Se3<ScalarT> t2_foo_plane = se3FromPlane(plane_in_foo);
    Eigen::Hyperplane<ScalarT, 3> plane2_foo = planeFromSe3(t2_foo_plane);
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
