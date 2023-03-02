// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/geometry/point_transform.h"

#include "sophus/calculus/num_diff.h"
#include "sophus/sensor/camera_projection/projection_z1.h"

#include <gtest/gtest.h>

using namespace sophus;

TEST(inverse_depth, integrations) {
  for (InverseDepthPoint3F64 const& inv_dept_point :
       {InverseDepthPoint3F64::fromAbAndPsi(Eigen::Vector3d(0.1, 0.3, 2.0)),
        InverseDepthPoint3F64::fromAbAndPsi(Eigen::Vector3d(0.7, 0.2, 1.0)),
        InverseDepthPoint3F64::fromAbAndPsi(Eigen::Vector3d(-0.2, 1.0, 0.1))}) {
    Eigen::Matrix<double, 2, 3> dx = dxProjX(inv_dept_point);

    Eigen::Matrix<double, 2, 3> const num_dx = vectorFieldNumDiff<double, 2, 3>(
        [](Eigen::Vector3d const& ab_psi) {
          return proj(InverseDepthPoint3F64::fromAbAndPsi(ab_psi));
        },
        Eigen::Vector3d(
            inv_dept_point.projInZ1Plane().x(),
            inv_dept_point.projInZ1Plane().y(),
            inv_dept_point.psi()));
    SOPHUS_ASSERT_NEAR(dx, num_dx, kEpsilonF64);

    {
      Eigen::Vector3d point = inv_dept_point.toEuclideanPoint3();

      Eigen::Matrix<double, 2, 3> dx = ProjectionZ1::dxProjX(point);
      Eigen::Matrix<double, 2, 3> const num_dx =
          vectorFieldNumDiff<double, 2, 3>(
              [](Eigen::Vector3d const& x) { return proj(x); }, point);
      SOPHUS_ASSERT_NEAR(dx, num_dx, kEpsilonSqrtF64);
    }
  }

  for (InverseDepthPoint3F64 const& point :
       {InverseDepthPoint3F64::fromAbAndPsi(Eigen::Vector3d(0.1, 0.3, 2.0))}) {
    Eigen::Matrix<double, 2, 6> dx = dxProjExpXPointAt0(point);

    Eigen::Vector<double, 6> zero;
    zero.setZero();
    Eigen::Matrix<double, 2, 6> const num_dx = vectorFieldNumDiff<double, 2, 6>(
        [point](Eigen::Vector<double, 6> const& vec_a) {
          return proj(
              Isometry3<double>::exp(vec_a) * point.toEuclideanPoint3());
        },
        zero);
    SOPHUS_ASSERT_NEAR(dx, num_dx, kEpsilonSqrtF64);

    Eigen::Matrix<double, 2, 6> const num_dx2 =
        vectorFieldNumDiff<double, 2, 6>(
            [point](Eigen::Vector<double, 6> const& vec_a) {
              return projTransform(Isometry3<double>::exp(vec_a), point);
            },
            zero);
    SOPHUS_ASSERT_NEAR(num_dx2, num_dx, kEpsilonF64);
  }
}

TEST(point_transform, integrations) {
  std::vector<Eigen::Vector<double, 6>> tangent_vec;

  Eigen::Vector<double, 6> tmp;
  tmp << 0, 0, 0, 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 1, 0, 0, 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 0, 1, 0, 1, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 0, -5, 10, 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << -1, 1, 0, 0, 0, 1;
  tangent_vec.push_back(tmp);
  tmp << 20, -1, 0, -1, 1, 0;
  tangent_vec.push_back(tmp);
  tmp << 30, 5, -1, 20, -1, 0;
  tangent_vec.push_back(tmp);

  std::vector<Eigen::Vector3d> point_vec;
  point_vec.push_back(Eigen::Vector3d(1, 2, 4));
  point_vec.push_back(Eigen::Vector3d(1, -3, 0.5));
  point_vec.push_back(Eigen::Vector3d(-5, -6, 7));

  double const eps = sophus::kEpsilon<double>;

  for (Eigen::Vector<double, 6> const& t : tangent_vec) {
    sophus::SE3d foo_from_bar_isometry = sophus::SE3d::exp(t);
    PointTransformer<double> foo_from_bar(foo_from_bar_isometry);

    // For points not at infinity
    for (auto const& point_in_bar : point_vec) {
      InverseDepthPoint3F64 inverse_depth_in_bar =
          InverseDepthPoint3F64::fromEuclideanPoint3(point_in_bar);

      Eigen::Vector3d point_in_foo = foo_from_bar_isometry * point_in_bar;
      Eigen::Vector3d point_in_foo2 = foo_from_bar.transform(point_in_bar);

      SOPHUS_ASSERT_NEAR(point_in_foo, point_in_foo2, eps);

      Eigen::Vector2d xy1_by_z_in_foo =
          proj(foo_from_bar.transform(point_in_bar));
      Eigen::Vector2d xy1_by_z_in_foo2 =
          foo_from_bar.projTransform(point_in_bar);
      Eigen::Vector2d xy1_by_z_in_foo3 =
          projTransform(foo_from_bar_isometry, inverse_depth_in_bar);
      Eigen::Vector2d xy1_by_z_in_foo4 =
          foo_from_bar.projTransform(inverse_depth_in_bar);

      SOPHUS_ASSERT_NEAR(xy1_by_z_in_foo, xy1_by_z_in_foo2, eps);
      SOPHUS_ASSERT_NEAR(xy1_by_z_in_foo, xy1_by_z_in_foo3, eps);
      SOPHUS_ASSERT_NEAR(xy1_by_z_in_foo, xy1_by_z_in_foo4, eps);

      Eigen::Matrix<double, 2, 6> dx =
          foo_from_bar.dxProjExpXTransformPointAt0(inverse_depth_in_bar);

      Eigen::Vector<double, 6> zero;
      zero.setZero();
      Eigen::Matrix<double, 2, 6> const num_dx =
          vectorFieldNumDiff<double, 2, 6>(
              [&](Eigen::Vector<double, 6> const& vec_a) {
                return proj(
                    sophus::Isometry3F64::exp(vec_a) *
                    foo_from_bar.transform(point_in_bar));
              },
              zero);
      Eigen::Matrix<double, 2, 6> const num_dx2 =
          vectorFieldNumDiff<double, 2, 6>(
              [&](Eigen::Vector<double, 6> const& vec_a) {
                sophus::Isometry3F64 exp_a = sophus::Isometry3F64::exp(vec_a);
                return proj(
                    exp_a.so3().matrix() *
                        foo_from_bar.scaledTransform(inverse_depth_in_bar) +
                    inverse_depth_in_bar.psi() * exp_a.translation());
              },
              zero);
      SOPHUS_ASSERT_NEAR(dx, num_dx, 10 * kEpsilonSqrtF64);
      SOPHUS_ASSERT_NEAR(num_dx, num_dx2, kEpsilonSqrtF64);
      {
        Eigen::Matrix<double, 2, 3> dx =
            foo_from_bar.dxProjTransformX(inverse_depth_in_bar);

        Eigen::Matrix<double, 2, 3> const num_dx =
            vectorFieldNumDiff<double, 2, 3>(
                [&](Eigen::Vector<double, 3> const& ab_psi) {
                  return proj(foo_from_bar.scaledTransform(
                      InverseDepthPoint3F64::fromAbAndPsi(ab_psi)));
                },
                inverse_depth_in_bar.params());
        SOPHUS_ASSERT_NEAR(dx, num_dx, 0.002);
      }
    }

    // For points at infinity
    for (auto const& point_in_bar : point_vec) {
      InverseDepthPoint3F64 inverse_depth_in_bar =
          InverseDepthPoint3F64::fromEuclideanPoint3(point_in_bar);
      inverse_depth_in_bar.psi() = 0.0;  // set z to infinity

      Eigen::Vector2d xy1_by_inf_in_foo =
          projTransform(foo_from_bar_isometry, inverse_depth_in_bar);
      Eigen::Vector2d xy1_by_inf_in_foo1 =
          foo_from_bar.projTransform(inverse_depth_in_bar);
      Eigen::Vector2d xy1_by_inf_in_foo2 =
          proj(foo_from_bar_isometry.so3() * point_in_bar.normalized());

      SOPHUS_ASSERT_NEAR(xy1_by_inf_in_foo, xy1_by_inf_in_foo1, eps);
      SOPHUS_ASSERT_NEAR(xy1_by_inf_in_foo, xy1_by_inf_in_foo2, eps);

      Eigen::Matrix<double, 2, 6> dx =
          foo_from_bar.dxProjExpXTransformPointAt0(inverse_depth_in_bar);
      Eigen::Vector<double, 6> zero;
      zero.setZero();
      Eigen::Matrix<double, 2, 6> const num_dx2 =
          vectorFieldNumDiff<double, 2, 6>(
              [&](Eigen::Vector<double, 6> const& vec_a) {
                sophus::Isometry3F64 exp_a = sophus::Isometry3F64::exp(vec_a);
                return proj(
                    exp_a.so3().matrix() *
                        foo_from_bar.scaledTransform(inverse_depth_in_bar) +
                    inverse_depth_in_bar.psi() * exp_a.translation());
              },
              zero);
      SOPHUS_ASSERT_NEAR(dx, num_dx2, kEpsilonSqrtF64);
    }
  }
}
