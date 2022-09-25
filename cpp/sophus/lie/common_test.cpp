// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/calculus/num_diff.h"
#include "sophus/common/test_macros.h"
#include "sophus/lie/interp/interpolate.h"
#include "sophus/lie/interp/spline.h"
#include "sophus/lie/se3.h"

#include <cmath>
#include <iostream>

namespace sophus {

namespace {

bool testSmokeDetails() {
  bool passed = true;
  std::cout << details::pretty(4.2) << std::endl;
  std::cout << details::pretty(Eigen::Vector2f(1, 2)) << std::endl;
  bool dummy = true;
  details::testFailed(
      dummy, "dummyFunc", "dummyFile", 99, "This is just a pratice alarm!");
  SOPHUS_TEST_EQUAL(passed, dummy, false, "");

  double val = transpose(42.0);
  SOPHUS_TEST_EQUAL(passed, val, 42.0, "");
  Eigen::Matrix<float, 1, 2> row = transpose(Eigen::Vector2f(1, 7));
  Eigen::Matrix<float, 1, 2> expected_row(1, 7);
  SOPHUS_TEST_EQUAL(passed, row, expected_row, "");

  std::optional<int> opt(std::nullopt);
  SOPHUS_TEST(passed, !opt, "");

  return passed;
}

bool testSpline() {
  double const small_eps_sqrt = kEpsilonSqrt<double>;

  bool passed = true;
  for (double delta_t : {0.1, 0.5, 1.0}) {
    for (double u : {0.0000, 0.1, 0.5, 0.51, 0.9, 0.999}) {
      std::cout << details::pretty(SplineBasisFunction<double>::b(u))
                << std::endl;

      Eigen::Vector3d dt_b = SplineBasisFunction<double>::dtB(u, delta_t);
      std::cout << details::pretty(dt_b) << std::endl;

      Eigen::Vector3d dt_b2 = curveNumDiff(
          [delta_t](double u_bar) -> Eigen::Vector3d {
            return SplineBasisFunction<double>::b(u_bar) / delta_t;
          },
          u);
      SOPHUS_TEST_APPROX(
          passed,
          dt_b,
          dt_b2,
          small_eps_sqrt,
          "Dt_, u={}, delta_t={}",
          u,
          delta_t);

      Eigen::Vector3d dt2_b = SplineBasisFunction<double>::dt2B(u, delta_t);

      Eigen::Vector3d dt2_b2 = curveNumDiff(
          [delta_t](double u_bar) -> Eigen::Vector3d {
            return SplineBasisFunction<double>::dtB(u_bar, delta_t) / delta_t;
          },
          u);
      SOPHUS_TEST_APPROX(
          passed,
          dt2_b,
          dt2_b2,
          small_eps_sqrt,
          "Dt2_B, u={}, delta_t={}",
          u,
          delta_t);
    }
  }

  SE3d t_world_foo = SE3d::rotX(0.4) * SE3d::rotY(0.2);
  SE3d t_world_bar =
      SE3d::rotZ(-0.4) * SE3d::rotY(0.4) * SE3d::trans(0.0, 1.0, -2.0);

  std::vector<SE3d> control_poses;
  control_poses.push_back(interpolate(t_world_foo, t_world_bar, 0.0));

  for (double p = 0.2; p < 1.0; p += 0.2) {
    SE3d t_world_inter = interpolate(t_world_foo, t_world_bar, p);
    control_poses.push_back(t_world_inter);
  }

  BasisSplineImpl<SE3d> spline(control_poses, 1.0);

  SE3d mat_t = spline.parentPoseSpline(0.0, 1.0);
  SE3d mat_t2 = spline.parentPoseSpline(1.0, 0.0);

  Eigen::Matrix3d r = mat_t.so3().matrix();
  Eigen::Matrix3d r2 = mat_t2.so3().matrix();
  Eigen::Vector3d t = mat_t.translation();
  Eigen::Vector3d t2 = mat_t2.translation();

  SOPHUS_TEST_APPROX(passed, r, r2, small_eps_sqrt, "lambdsa");

  SOPHUS_TEST_APPROX(passed, t, t2, small_eps_sqrt, "lambdsa");

  Eigen::Matrix4d dt_parent_transform_spline =
      spline.dtParentPoseSpline(0.0, 0.5);
  Eigen::Matrix4d dt_parent_transform_spline2 = curveNumDiff(
      [&](double u_bar) -> Eigen::Matrix4d {
        return spline.parentPoseSpline(0.0, u_bar).matrix();
      },
      0.5);
  SOPHUS_TEST_APPROX(
      passed,
      dt_parent_transform_spline,
      dt_parent_transform_spline2,
      small_eps_sqrt,
      "Dt_parent_T_spline");

  Eigen::Matrix4d dt2_parent_transform_spline =
      spline.dt2ParentPoseSpline(0.0, 0.5);
  Eigen::Matrix4d dt2_parent_transform_spline2 = curveNumDiff(
      [&](double u_bar) -> Eigen::Matrix4d {
        return spline.dtParentPoseSpline(0.0, u_bar).matrix();
      },
      0.5);
  SOPHUS_TEST_APPROX(
      passed,
      dt2_parent_transform_spline,
      dt2_parent_transform_spline2,
      small_eps_sqrt,
      "Dt2_parent_T_spline");

  return passed;
}

void runAll() {
  std::cerr << "Common tests:" << std::endl;
  bool passed = testSmokeDetails();
  passed &= testSpline();
  processTestResult(passed);
}

}  // namespace
}  // namespace sophus

int main() { sophus::runAll(); }
