#include <iostream>

#include <sophus/interpolate.hpp>
#include <sophus/num_diff.hpp>
#include <sophus/se3.hpp>
#include <sophus/spline.hpp>
#include <sophus/test_macros.hpp>

namespace Sophus {

namespace {

bool testSmokeDetails() {
  bool passed = true;
  std::cout << details::pretty(4.2) << std::endl;
  std::cout << details::pretty(Vector2f(1, 2)) << std::endl;

  SOPHUS_TEST(passed, false, "Just a practice alarm");
  if (passed) {
    exit(-1);
  }

  passed = true;

  double val = transpose(42.0);
  SOPHUS_TEST_EQUAL(passed, val, 42.0, "");
  Matrix<float, 1, 2> row = transpose(Vector2f(1, 7));
  Matrix<float, 1, 2> expected_row(1, 7);
  SOPHUS_TEST_EQUAL(passed, row, expected_row, "");

  return passed;
}

bool testSpline() {
  double const kSmallEpsSqrt = Constants<double>::epsilonSqrt();

  bool passed = true;
  for (double delta_t : {0.1, 0.5, 1.0}) {
    for (double u : {0.0000, 0.1, 0.5, 0.51, 0.9, 0.999}) {
      std::cout << details::pretty(SplineBasisFunction<double>::B(u))
                << std::endl;

      Eigen::Vector3d Dt_B = SplineBasisFunction<double>::Dt_B(u, delta_t);
      std::cout << details::pretty(Dt_B) << std::endl;

      Eigen::Vector3d Dt_B2 = curveNumDiff(
          [delta_t](double u_bar) -> Eigen::Vector3d {
            return SplineBasisFunction<double>::B(u_bar) / delta_t;
          },
          u);
      SOPHUS_TEST_APPROX(passed, Dt_B, Dt_B2, kSmallEpsSqrt,
                         "Dt_, u={}, delta_t={}", u, delta_t);

      Eigen::Vector3d Dt2_B = SplineBasisFunction<double>::Dt2_B(u, delta_t);

      Eigen::Vector3d Dt2_B2 = curveNumDiff(
          [delta_t](double u_bar) -> Eigen::Vector3d {
            return SplineBasisFunction<double>::Dt_B(u_bar, delta_t) / delta_t;
          },
          u);
      SOPHUS_TEST_APPROX(passed, Dt2_B, Dt2_B2, kSmallEpsSqrt,
                         "Dt2_B, u={}, delta_t={}", u, delta_t);
    }
  }

  SE3d T_world_foo = SE3d::rotX(0.4) * SE3d::rotY(0.2);
  SE3d T_world_bar =
      SE3d::rotZ(-0.4) * SE3d::rotY(0.4) * SE3d::trans(0.0, 1.0, -2.0);

  std::vector<SE3d> control_poses;
  control_poses.push_back(interpolate(T_world_foo, T_world_bar, 0.0));

  for (double p = 0.2; p < 1.0; p += 0.2) {
    SE3d T_world_inter = interpolate(T_world_foo, T_world_bar, p);
    control_poses.push_back(T_world_inter);
  }

  BasisSplineImpl<SE3d> spline(control_poses, 1.0);

  SE3d T = spline.parent_T_spline(0, 1.0);
  SE3d T2 = spline.parent_T_spline(1, 0.0);

  Eigen::Matrix3d R = T.so3().matrix();
  Eigen::Matrix3d R2 = T2.so3().matrix();
  Eigen::Vector3d t = T.translation();
  Eigen::Vector3d t2 = T2.translation();

  SOPHUS_TEST_APPROX(passed, R, R2, kSmallEpsSqrt, "lambdsa");

  SOPHUS_TEST_APPROX(passed, t, t2, kSmallEpsSqrt, "lambdsa");

  Eigen::Matrix4d Dt_parent_T_spline = spline.Dt_parent_T_spline(0, 0.5);
  Eigen::Matrix4d Dt_parent_T_spline2 = curveNumDiff(
      [&](double u_bar) -> Eigen::Matrix4d {
        return spline.parent_T_spline(0, u_bar).matrix();
      },
      0.5);
  SOPHUS_TEST_APPROX(passed, Dt_parent_T_spline, Dt_parent_T_spline2,
                     kSmallEpsSqrt, "Dt_parent_T_spline");

  Eigen::Matrix4d Dt2_parent_T_spline = spline.Dt2_parent_T_spline(0, 0.5);
  Eigen::Matrix4d Dt2_parent_T_spline2 = curveNumDiff(
      [&](double u_bar) -> Eigen::Matrix4d {
        return spline.Dt_parent_T_spline(0, u_bar).matrix();
      },
      0.5);
  SOPHUS_TEST_APPROX(passed, Dt2_parent_T_spline, Dt2_parent_T_spline2,
                     kSmallEpsSqrt, "Dt2_parent_T_spline");

  return passed;
}

void runAll() {
  std::cerr << "Common tests:" << std::endl;
  bool passed = testSmokeDetails();
  passed &= testSpline();
  processTestResult(passed);
}

}  // namespace
}  // namespace Sophus

int main() { Sophus::runAll(); }
