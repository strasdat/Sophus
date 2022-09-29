// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/ceres/jet_helpers.h"
#include "sophus/common/common.h"
#include "sophus/geometry/projection.h"
#include "sophus/sensor/camera_distortion/affine.h"

#include <Eigen/Dense>

namespace sophus {

// https://github.com/facebookincubator/isometric_pattern_matcher/blob/main/IsometricPatternMatcher/CameraModels.h
//
// parameters = fx, fy, cx, cy, kb0, kb1, kb2, kb3
class KannalaBrandtK3Transform {
 public:
  static int constexpr kNumDistortionParams = 4;
  static int constexpr kNumParams = kNumDistortionParams + 4;
  static constexpr const std::string_view kProjectionModel =
      "KannalaBrandtK3: fx, fy, cx, cy, kb0, kb1, kb2, kb3";

  template <class TScalar>
  using ProjInCameraZ1Plane = Eigen::Matrix<TScalar, 2, 1>;
  template <class TScalar>
  using PixelImage = Eigen::Matrix<TScalar, 2, 1>;
  template <class TScalar>
  using Params = Eigen::Matrix<TScalar, kNumParams, 1>;
  template <class TScalar>
  using DistorationParams = Eigen::Matrix<TScalar, kNumDistortionParams, 1>;

  template <class TParamsTypeT, class TPointTypeT>
  static PixelImage<typename TPointTypeT::Scalar> distort(
      Eigen::MatrixBase<TParamsTypeT> const& params,
      Eigen::MatrixBase<TPointTypeT> const& proj_point_in_camera_z1_plane) {
    static_assert(
        TParamsTypeT::ColsAtCompileTime == 1, "params must be a column-vector");
    static_assert(
        TParamsTypeT::RowsAtCompileTime == kNumParams,
        "params must have exactly kNumParams rows");
    static_assert(
        TPointTypeT::ColsAtCompileTime == 1,
        "point_camera must be a column-vector");
    static_assert(
        TPointTypeT::RowsAtCompileTime == 2,
        "point_camera must have exactly 2 columns");

    auto const k0 = params[4];
    auto const k1 = params[5];
    auto const k2 = params[6];
    auto const k3 = params[7];

    auto const radius_squared =
        proj_point_in_camera_z1_plane[0] * proj_point_in_camera_z1_plane[0] +
        proj_point_in_camera_z1_plane[1] * proj_point_in_camera_z1_plane[1];
    using std::atan2;
    using std::sqrt;

    if (radius_squared > sophus::kEpsilonF64) {
      auto const radius = sqrt(radius_squared);
      auto const radius_inverse = 1.0 / radius;
      auto const theta = atan2(radius, typename TPointTypeT::Scalar(1.0));
      auto const theta2 = theta * theta;
      auto const theta4 = theta2 * theta2;
      auto const theta6 = theta4 * theta2;
      auto const theta8 = theta4 * theta4;
      auto const r_distorted =
          theta * (1.0 + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8);
      auto const scaling = r_distorted * radius_inverse;

      return scaling * proj_point_in_camera_z1_plane.cwiseProduct(
                           params.template head<2>()) +
             params.template segment<2>(2);
    }  // linearize r around radius=0

    return AffineTransform::distort(

        params.template head<4>(), proj_point_in_camera_z1_plane);
  }

  template <class TScalar>
  static ProjInCameraZ1Plane<TScalar> undistort(
      Params<TScalar> const& params, PixelImage<TScalar> const& pixel_image) {
    using std::abs;
    using std::sqrt;
    using std::tan;

    // Undistortion
    const TScalar fu = params[0];
    const TScalar fv = params[1];
    const TScalar u0 = params[2];
    const TScalar v0 = params[3];

    const TScalar k0 = params[4];
    const TScalar k1 = params[5];
    const TScalar k2 = params[6];
    const TScalar k3 = params[7];

    const TScalar un = (pixel_image(0) - u0) / fu;
    const TScalar vn = (pixel_image(1) - v0) / fv;
    const TScalar rth2 = un * un + vn * vn;

    if (rth2 < sophus::kEpsilon<TScalar> * sophus::kEpsilon<TScalar>) {
      return Eigen::Matrix<TScalar, 2, 1>(un, vn);
    }

    const TScalar rth = sqrt(rth2);

    // Use Newtons method to solve for theta, 50 iterations max
    TScalar th = sqrt(rth);
    for (int i = 0; i < 500; ++i) {
      const TScalar th2 = th * th;
      const TScalar th4 = th2 * th2;
      const TScalar th6 = th4 * th2;
      const TScalar th8 = th4 * th4;

      const TScalar thd =
          th * (TScalar(1.0) + k0 * th2 + k1 * th4 + k2 * th6 + k3 * th8);

      const TScalar d_thd_wtr_th =
          TScalar(1.0) + TScalar(3.0) * k0 * th2 + TScalar(5.0) * k1 * th4 +
          TScalar(7.0) * k2 * th6 + TScalar(9.0) * k3 * th8;

      const TScalar step = (thd - rth) / d_thd_wtr_th;
      th -= step;
      // has converged?
      if (abs(jet_helpers::GetValue<TScalar>::impl(step)) <
          sophus::kEpsilon<TScalar>) {
        break;
      }
    }

    TScalar radius_undistorted = tan(th);

    if (radius_undistorted < TScalar(0.0)) {
      return Eigen::Matrix<TScalar, 2, 1>(
          -radius_undistorted * un / rth, -radius_undistorted * vn / rth);
    }
    return Eigen::Matrix<TScalar, 2, 1>(
        radius_undistorted * un / rth, radius_undistorted * vn / rth);
  }

  template <class TParamsTypeT, class TPointTypeT>
  static Eigen::Matrix<typename TPointTypeT::Scalar, 2, 2> dxDistort(
      Eigen::MatrixBase<TParamsTypeT> const& params,
      Eigen::MatrixBase<TPointTypeT> const& proj_point_in_camera_z1_plane) {
    static_assert(
        TParamsTypeT::ColsAtCompileTime == 1, "params must be a column-vector");
    static_assert(
        TParamsTypeT::RowsAtCompileTime == kNumParams,
        "params must have exactly kNumParams rows");
    static_assert(
        TPointTypeT::ColsAtCompileTime == 1,
        "point_camera must be a column-vector");
    static_assert(
        TPointTypeT::RowsAtCompileTime == 2,
        "point_camera must have exactly 2 columns");
    using Scalar = typename TPointTypeT::Scalar;

    Scalar a = proj_point_in_camera_z1_plane[0];
    Scalar b = proj_point_in_camera_z1_plane[1];
    Scalar const fx = params[0];
    Scalar const fy = params[1];
    Eigen::Matrix<Scalar, kNumDistortionParams, 1> k =
        params.template tail<kNumDistortionParams>();

    auto const radius_squared = a * a + b * b;
    using std::atan2;
    using std::sqrt;

    Eigen::Matrix<typename TPointTypeT::Scalar, 2, 2> dx;

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
