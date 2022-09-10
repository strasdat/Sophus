// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/ceres/jet_helpers.h"
#include "sophus/core/common.h"
#include "sophus/geometry/projection.h"
#include "sophus/sensor/camera_transforms/affine.h"

#include <Eigen/Dense>

namespace sophus {

// https://github.com/facebookincubator/isometric_pattern_matcher/blob/main/IsometricPatternMatcher/CameraModels.h
//
// parameters = fx, fy, cx, cy, kb0, kb1, kb2, kb3
class KannalaBrandtK3Transform {
 public:
  static constexpr int kNumDistortionParams = 4;
  static constexpr int kNumParams = kNumDistortionParams + 4;
  static const constexpr std::string_view kProjectionModel =
      "KannalaBrandtK3: fx, fy, cx, cy, kb0, kb1, kb2, kb3";

  template <class ScalarT>
  using ProjInCameraZ1Plane = Eigen::Matrix<ScalarT, 2, 1>;
  template <class ScalarT>
  using PixelImage = Eigen::Matrix<ScalarT, 2, 1>;
  template <class ScalarT>
  using Params = Eigen::Matrix<ScalarT, kNumParams, 1>;
  template <class ScalarT>
  using DistorationParams = Eigen::Matrix<ScalarT, kNumDistortionParams, 1>;

  template <class ParamsTypeT, class PointTypeT>
  static PixelImage<typename PointTypeT::Scalar> warp(
      const Eigen::MatrixBase<ParamsTypeT>& params,
      const Eigen::MatrixBase<PointTypeT>& proj_point_in_camera_z1_plane) {
    static_assert(
        ParamsTypeT::ColsAtCompileTime == 1, "params must be a column-vector");
    static_assert(
        ParamsTypeT::RowsAtCompileTime == kNumParams,
        "params must have exactly kNumParams rows");
    static_assert(
        PointTypeT::ColsAtCompileTime == 1,
        "point_camera must be a column-vector");
    static_assert(
        PointTypeT::RowsAtCompileTime == 2,
        "point_camera must have exactly 2 columns");

    const auto k0 = params[4];
    const auto k1 = params[5];
    const auto k2 = params[6];
    const auto k3 = params[7];

    const auto radius_squared =
        proj_point_in_camera_z1_plane[0] * proj_point_in_camera_z1_plane[0] +
        proj_point_in_camera_z1_plane[1] * proj_point_in_camera_z1_plane[1];
    using std::atan2;
    using std::sqrt;

    if (radius_squared > sophus::kEpsilonF64) {
      const auto radius = sqrt(radius_squared);
      const auto radius_inverse = 1.0 / radius;
      const auto theta = atan2(radius, typename PointTypeT::Scalar(1.0));
      const auto theta2 = theta * theta;
      const auto theta4 = theta2 * theta2;
      const auto theta6 = theta4 * theta2;
      const auto theta8 = theta4 * theta4;
      const auto r_distorted =
          theta * (1.0 + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8);
      const auto scaling = r_distorted * radius_inverse;

      return scaling * proj_point_in_camera_z1_plane.cwiseProduct(
                           params.template head<2>()) +
             params.template segment<2>(2);
    }  // linearize r around radius=0

    return AffineTransform::warp(

        params.template head<4>(), proj_point_in_camera_z1_plane);
  }

  template <class ScalarT>
  static ProjInCameraZ1Plane<ScalarT> unwarp(
      const Params<ScalarT>& params, const PixelImage<ScalarT>& pixel_image) {
    using std::abs;
    using std::sqrt;
    using std::tan;

    // Undistortion
    const ScalarT fu = params[0];
    const ScalarT fv = params[1];
    const ScalarT u0 = params[2];
    const ScalarT v0 = params[3];

    const ScalarT k0 = params[4];
    const ScalarT k1 = params[5];
    const ScalarT k2 = params[6];
    const ScalarT k3 = params[7];

    const ScalarT un = (pixel_image(0) - u0) / fu;
    const ScalarT vn = (pixel_image(1) - v0) / fv;
    const ScalarT rth2 = un * un + vn * vn;

    if (rth2 < sophus::kEpsilon<ScalarT> * sophus::kEpsilon<ScalarT>) {
      return Eigen::Matrix<ScalarT, 2, 1>(un, vn);
    }

    const ScalarT rth = sqrt(rth2);

    // Use Newtons method to solve for theta, 50 iterations max
    ScalarT th = sqrt(rth);
    for (int i = 0; i < 500; ++i) {
      const ScalarT th2 = th * th;
      const ScalarT th4 = th2 * th2;
      const ScalarT th6 = th4 * th2;
      const ScalarT th8 = th4 * th4;

      const ScalarT thd =
          th * (ScalarT(1.0) + k0 * th2 + k1 * th4 + k2 * th6 + k3 * th8);

      const ScalarT d_thd_wtr_th =
          ScalarT(1.0) + ScalarT(3.0) * k0 * th2 + ScalarT(5.0) * k1 * th4 +
          ScalarT(7.0) * k2 * th6 + ScalarT(9.0) * k3 * th8;

      const ScalarT step = (thd - rth) / d_thd_wtr_th;
      th -= step;
      // has converged?
      if (abs(jet_helpers::GetValue<ScalarT>::impl(step)) <
          sophus::kEpsilon<ScalarT>) {
        break;
      }
    }

    ScalarT radius_undistorted = tan(th);

    if (radius_undistorted < ScalarT(0.0)) {
      return Eigen::Matrix<ScalarT, 2, 1>(
          -radius_undistorted * un / rth, -radius_undistorted * vn / rth);
    }
    return Eigen::Matrix<ScalarT, 2, 1>(
        radius_undistorted * un / rth, radius_undistorted * vn / rth);
  }

  template <class ParamsTypeT, class PointTypeT>
  static Eigen::Matrix<typename PointTypeT::Scalar, 2, 2> dxWarp(
      const Eigen::MatrixBase<ParamsTypeT>& params,
      const Eigen::MatrixBase<PointTypeT>& proj_point_in_camera_z1_plane) {
    static_assert(
        ParamsTypeT::ColsAtCompileTime == 1, "params must be a column-vector");
    static_assert(
        ParamsTypeT::RowsAtCompileTime == kNumParams,
        "params must have exactly kNumParams rows");
    static_assert(
        PointTypeT::ColsAtCompileTime == 1,
        "point_camera must be a column-vector");
    static_assert(
        PointTypeT::RowsAtCompileTime == 2,
        "point_camera must have exactly 2 columns");
    using Scalar = typename PointTypeT::Scalar;

    Scalar a = proj_point_in_camera_z1_plane[0];
    Scalar b = proj_point_in_camera_z1_plane[1];
    Scalar const fx = params[0];
    Scalar const fy = params[1];
    Eigen::Matrix<Scalar, kNumDistortionParams, 1> k =
        params.template tail<kNumDistortionParams>();

    const auto radius_squared = a * a + b * b;
    using std::atan2;
    using std::sqrt;

    Eigen::Matrix<typename PointTypeT::Scalar, 2, 2> dx;

    if (radius_squared < sophus::kEpsilonSqrtF64) {
      // clang-format off
      dx <<  //
          fx,  0,
           0, fy;
      // clang-format on
      return dx;
    }

    using std::atan2;
    using std::pow;
    using std::sqrt;

    Scalar const c0 = pow(a, 2);
    Scalar const c1 = pow(b, 2);
    Scalar const c2 = c0 + c1;
    Scalar const c3 = pow(c2, 5.0 / 2.0);
    Scalar const c4 = c2 + 1;
    Scalar const c5 = atan(sqrt(c2));
    Scalar const c6 = pow(c5, 2);
    Scalar const c7 = c6 * k[0];
    Scalar const c8 = pow(c5, 4);
    Scalar const c9 = c8 * k[1];
    Scalar const c10 = pow(c5, 6);
    Scalar const c11 = c10 * k[2];
    Scalar const c12 = pow(c5, 8) * k[3];
    Scalar const c13 = 1.0 * c4 * c5 * (c11 + c12 + c7 + c9 + 1.0);
    Scalar const c14 = c13 * c3;
    Scalar const c15 = pow(c2, 3.0 / 2.0);
    Scalar const c16 = c13 * c15;
    Scalar const c17 =
        1.0 * c11 + 1.0 * c12 +
        2.0 * c6 * (4 * c10 * k[3] + 2 * c6 * k[1] + 3 * c8 * k[2] + k[0]) +
        1.0 * c7 + 1.0 * c9 + 1.0;
    Scalar const c18 = c17 * pow(c2, 2);
    Scalar const c19 = 1.0 / c4;
    Scalar const c20 = c19 / pow(c2, 3);
    Scalar const c21 = a * b * c19 * (-c13 * c2 + c15 * c17) / c3;

    dx(0, 0) = c20 * fx * (-c0 * c16 + c0 * c18 + c14);
    dx(0, 1) = c21 * fx;

    dx(1, 0) = c21 * fy;
    dx(1, 1) = c20 * fy * (-c1 * c16 + c1 * c18 + c14);

    return dx;
  }
};

}  // namespace sophus
