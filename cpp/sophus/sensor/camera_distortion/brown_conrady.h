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
#include "sophus/sensor/camera_distortion/affine.h"

#include <Eigen/Dense>

namespace sophus {
class BrownConradyZ1Projection {
 public:
  static int constexpr kNumDistortionParams = 5;
  static int constexpr kNumParams = kNumDistortionParams + 4;
  static constexpr const std::string_view kProjectionModel =
      "BrownConrady: fx, fy, cx, cy, k1, k2, p1, p2, k3";

  template <class TScalar>
  using ProjInCameraZ1Plane = Eigen::Matrix<TScalar, 2, 1>;
  template <class TScalar>
  using PixelImage = Eigen::Matrix<TScalar, 2, 1>;
  template <class TScalar>
  using Params = Eigen::Matrix<TScalar, kNumParams, 1>;
  template <class TScalar>
  using DistorationParams = Eigen::Matrix<TScalar, kNumDistortionParams, 1>;

  template <class TParamScalarT, class TPointScalarT>
  static PixelImage<typename Eigen::ScalarBinaryOpTraits<
      TParamScalarT,
      TPointScalarT>::ReturnType>
  projImpl(
      DistorationParams<TParamScalarT> const& distortion,
      PixelImage<TPointScalarT> const& point_normalized) {
    using ReturnScalar = typename Eigen::
        ScalarBinaryOpTraits<TParamScalarT, TPointScalarT>::ReturnType;

    auto x = point_normalized[0];
    auto y = point_normalized[1];

    // From:
    // https://github.com/opencv/opencv/blob/63bb2abadab875fc648a572faccafee134f06fc8/modules/calib3d/src/calibration.cpp#L791

    auto r2 = x * x + y * y;
    auto r4 = r2 * r2;
    auto r6 = r4 * r2;
    auto a1 = 2.0 * x * y;
    auto a2 = r2 + 2.0 * x * x;
    auto a3 = r2 + 2.0 * y * y;

    auto cdist =
        1.0 + distortion[0] * r2 + distortion[1] * r4 + distortion[4] * r6;

    double icdist2 = 1.0;
    return PixelImage<ReturnScalar>(
        x * cdist * icdist2 + distortion[2] * a1 + distortion[3] * a2,
        y * cdist * icdist2 + distortion[2] * a3 + distortion[3] * a1);
  }

  template <class TScalar>
  static PixelImage<TScalar> unprojImpl(
      DistorationParams<TScalar> const& distortion,
      PixelImage<TScalar> const& uv_normalized) {
    // We had no luck with OpenCV's undistort. It seems not to be accurate if
    // "icdist" is close to 0.
    // https://github.com/opencv/opencv/blob/63bb2abadab875fc648a572faccafee134f06fc8/modules/calib3d/src/undistort.dispatch.cpp#L365
    //
    // Hence, we derive the inverse approximation scheme from scratch.
    //
    //
    // Objective: find that xy such that proj_impl(xy) = uv
    //
    // Using multivariate Newton scheme, by defining f and find the root of it:
    //
    //  f: R^2 -> R^2
    //  f(xy) :=  proj_impl(xy) - uv
    //
    //  xy_i+1 = xy_i + J^{-1} * f(xy)   with J being the Jacobian of f.
    //
    // TODO(hauke): There is most likely a 1-dimensional embedding and one only
    // need to solve a less computational heavy newton iteration...

    // initial guess
    PixelImage<TScalar> xy = uv_normalized;

    TScalar p0 = distortion[0];
    TScalar p1 = distortion[1];
    TScalar p2 = distortion[2];
    TScalar p3 = distortion[3];
    TScalar p4 = distortion[4];

    for (int i = 0; i < 50; ++i) {
      TScalar x = xy[0];
      TScalar y = xy[1];
      TScalar x2 = x * x;
      TScalar y2 = y * y;
      TScalar r2 = x2 + y2;
      TScalar r4 = r2 * r2;
      TScalar r6 = r2 * r4;

      PixelImage<TScalar> f_xy =
          projImpl(distortion, Eigen::Matrix<TScalar, 2, 1>(x, y)) -
          uv_normalized;

      // calculating Jacobian of proj_impl wrt. point_normalized
      TScalar du_dx = p0 * r2 + p1 * r4 + TScalar(2) * p2 * y +
                      TScalar(6) * p3 * x + p4 * r6 +
                      x * (TScalar(2) * p0 * x + TScalar(4) * p1 * x * r2 +
                           TScalar(6) * p4 * x * r4) +
                      TScalar(1);
      TScalar du_dy = TScalar(2) * p2 * x + TScalar(2) * p3 * y +
                      x * (TScalar(2) * p0 * y + TScalar(4) * p1 * y * r2 +
                           TScalar(6) * p4 * y * r4);
      TScalar dv_dx = TScalar(2) * p2 * x + TScalar(2) * p3 * y +
                      y * (TScalar(2) * p0 * x + TScalar(4) * p1 * x * r2 +
                           TScalar(6) * p4 * x * r4);
      TScalar dv_dy = p0 * r2 + p1 * r4 + TScalar(6) * p2 * y +
                      TScalar(2) * p3 * x + p4 * r6 +
                      y * (TScalar(2) * p0 * y + TScalar(4) * p1 * y * r2 +
                           TScalar(6) * p4 * y * r4) +
                      TScalar(1);

      //     | du_dx  du_dy |      | a  b |
      // J = |              |  =:  |      |
      //     | dv_dx  dv_dy |      | c  d |

      TScalar a = du_dx;
      TScalar b = du_dy;
      TScalar c = dv_dx;
      TScalar d = dv_dy;

      // | a  b | -1       1   |  d  -b |
      // |      |     =  ----- |        |
      // | c  d |        ad-bc | -c   a |

      Eigen::Matrix<TScalar, 2, 2> m;
      // clang-format off
      m <<  d, -b,
           -c,  a;
      // clang-format on

      Eigen::Matrix<TScalar, 2, 2> j_inv = TScalar(1) / (a * d - b * c) * m;
      PixelImage<TScalar> step = j_inv * f_xy;

      if (abs(jet_helpers::GetValue<TScalar>::impl(step.squaredNorm())) <
          sophus::kEpsilon<TScalar> * sophus::kEpsilon<TScalar>) {
        break;
      }
      xy -= step;
    }

    return xy;
  }

  template <class TParamsTypeT, class TPointTypeT>
  static PixelImage<typename TPointTypeT::Scalar> distort(
      Eigen::MatrixBase<TParamsTypeT> const& params,
      Eigen::MatrixBase<TPointTypeT> const& proj_point_in_camera_z1_plane) {
    using ParamScalar = typename TParamsTypeT::Scalar;

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

    Eigen::Matrix<ParamScalar, kNumDistortionParams, 1> distortion =
        params.template tail<kNumDistortionParams>();

    PixelImage<typename TPointTypeT::Scalar>
        distorted_point_in_camera_z1_plane =
            projImpl(distortion, proj_point_in_camera_z1_plane.eval());

    return AffineZ1Projection::distort(
        params.template head<4>(), distorted_point_in_camera_z1_plane);
  }

  template <class TScalar>
  static ProjInCameraZ1Plane<TScalar> undistort(
      Params<TScalar> const& params, PixelImage<TScalar> const& pixel_image) {
    PixelImage<TScalar> proj_point_in_camera_z1_plane = unprojImpl(
        params.template tail<kNumDistortionParams>().eval(),
        AffineZ1Projection::undistort(
            params.template head<4>().eval(), pixel_image));

    return ProjInCameraZ1Plane<TScalar>(
        proj_point_in_camera_z1_plane[0], proj_point_in_camera_z1_plane[1]);
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

    Eigen::Matrix<Scalar, kNumDistortionParams, 1> d =
        params.template tail<kNumDistortionParams>();

    Scalar a = proj_point_in_camera_z1_plane[0];
    Scalar b = proj_point_in_camera_z1_plane[1];

    using std::pow;

    Scalar const c0 = 2 * d[0];
    Scalar const c1 = pow(a, 2) + pow(b, 2);
    Scalar const c2 = 4 * c1 * d[1];
    Scalar const c3 = pow(c1, 2);
    Scalar const c4 = 6 * c3 * d[4];
    Scalar const c5 = a * c0 + a * c2 + a * c4;
    Scalar const c6 = 1.0 * a;
    Scalar const c7 = 2.0 * d[2];
    Scalar const c8 = a * d[3];
    Scalar const c9 =
        1.0 * pow(c1, 3) * d[4] + 1.0 * c1 * d[0] + 1.0 * c3 * d[1] + 1.0;
    Scalar const c10 = b * d[3];
    Scalar const c11 = b * c0 + b * c2 + b * c4;
    Scalar const c12 = 1.0 * b;

    Eigen::Matrix<typename TPointTypeT::Scalar, 2, 2> dx;
    Scalar const fx = params[0];
    Scalar const fy = params[1];

    dx(0, 0) = fx * (b * c7 + c5 * c6 + 6.0 * c8 + c9);
    dx(0, 1) = fy * (a * c7 + 2 * c10 + c11 * c6);

    dx(1, 0) = fx * (2 * a * d[2] + 2.0 * c10 + c12 * c5);
    dx(1, 1) = fy * (6.0 * b * d[2] + c11 * c12 + 2.0 * c8 + c9);

    return dx;
  }
};

}  // namespace sophus
