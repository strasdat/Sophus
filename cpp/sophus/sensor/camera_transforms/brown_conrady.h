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
#include "sophus/sensor/camera_transforms/affine.h"

#include <Eigen/Dense>

namespace sophus {
class BrownConradyTransform {
 public:
  static constexpr int kNumDistortionParams = 5;
  static constexpr int kNumParams = kNumDistortionParams + 4;
  static const constexpr std::string_view kProjectionModel =
      "BrownConrady: fx, fy, cx, cy, k1, k2, p1, p2, k3";

  template <class ScalarT>
  using ProjInCameraZ1Plane = Eigen::Matrix<ScalarT, 2, 1>;
  template <class ScalarT>
  using PixelImage = Eigen::Matrix<ScalarT, 2, 1>;
  template <class ScalarT>
  using Params = Eigen::Matrix<ScalarT, kNumParams, 1>;
  template <class ScalarT>
  using DistorationParams = Eigen::Matrix<ScalarT, kNumDistortionParams, 1>;

  template <class ParamScalarT, class PointScalarT>
  static PixelImage<typename Eigen::ScalarBinaryOpTraits<
      ParamScalarT,
      PointScalarT>::ReturnType>
  projImpl(
      const DistorationParams<ParamScalarT>& distortion,
      const PixelImage<PointScalarT>& point_normalized) {
    using ReturnScalar = typename Eigen::
        ScalarBinaryOpTraits<ParamScalarT, PointScalarT>::ReturnType;

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

  template <class ScalarT>
  static PixelImage<ScalarT> unprojImpl(
      const DistorationParams<ScalarT>& distortion,
      const PixelImage<ScalarT>& uv_normalized) {
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
    PixelImage<ScalarT> xy = uv_normalized;

    ScalarT p0 = distortion[0];
    ScalarT p1 = distortion[1];
    ScalarT p2 = distortion[2];
    ScalarT p3 = distortion[3];
    ScalarT p4 = distortion[4];

    for (int i = 0; i < 50; ++i) {
      ScalarT x = xy[0];
      ScalarT y = xy[1];
      ScalarT x2 = x * x;
      ScalarT y2 = y * y;
      ScalarT r2 = x2 + y2;
      ScalarT r4 = r2 * r2;
      ScalarT r6 = r2 * r4;

      PixelImage<ScalarT> f_xy =
          projImpl(distortion, Eigen::Matrix<ScalarT, 2, 1>(x, y)) -
          uv_normalized;

      // calculating Jacobian of proj_impl wrt. point_normalized
      ScalarT du_dx = p0 * r2 + p1 * r4 + ScalarT(2) * p2 * y +
                      ScalarT(6) * p3 * x + p4 * r6 +
                      x * (ScalarT(2) * p0 * x + ScalarT(4) * p1 * x * r2 +
                           ScalarT(6) * p4 * x * r4) +
                      ScalarT(1);
      ScalarT du_dy = ScalarT(2) * p2 * x + ScalarT(2) * p3 * y +
                      x * (ScalarT(2) * p0 * y + ScalarT(4) * p1 * y * r2 +
                           ScalarT(6) * p4 * y * r4);
      ScalarT dv_dx = ScalarT(2) * p2 * x + ScalarT(2) * p3 * y +
                      y * (ScalarT(2) * p0 * x + ScalarT(4) * p1 * x * r2 +
                           ScalarT(6) * p4 * x * r4);
      ScalarT dv_dy = p0 * r2 + p1 * r4 + ScalarT(6) * p2 * y +
                      ScalarT(2) * p3 * x + p4 * r6 +
                      y * (ScalarT(2) * p0 * y + ScalarT(4) * p1 * y * r2 +
                           ScalarT(6) * p4 * y * r4) +
                      ScalarT(1);

      //     | du_dx  du_dy |      | a  b |
      // J = |              |  =:  |      |
      //     | dv_dx  dv_dy |      | c  d |

      ScalarT a = du_dx;
      ScalarT b = du_dy;
      ScalarT c = dv_dx;
      ScalarT d = dv_dy;

      // | a  b | -1       1   |  d  -b |
      // |      |     =  ----- |        |
      // | c  d |        ad-bc | -c   a |

      Eigen::Matrix<ScalarT, 2, 2> m;
      // clang-format off
      m <<  d, -b,
           -c,  a;
      // clang-format on

      Eigen::Matrix<ScalarT, 2, 2> j_inv = ScalarT(1) / (a * d - b * c) * m;
      PixelImage<ScalarT> step = j_inv * f_xy;

      if (abs(jet_helpers::GetValue<ScalarT>::impl(step.squaredNorm())) <
          sophus::kEpsilon<ScalarT> * sophus::kEpsilon<ScalarT>) {
        break;
      }
      xy -= step;
    }

    return xy;
  }

  template <class ParamsTypeT, class PointTypeT>
  static PixelImage<typename PointTypeT::Scalar> warp(
      const Eigen::MatrixBase<ParamsTypeT>& params,
      const Eigen::MatrixBase<PointTypeT>& proj_point_in_camera_z1_plane) {
    using ParamScalar = typename ParamsTypeT::Scalar;

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

    Eigen::Matrix<ParamScalar, kNumDistortionParams, 1> distortion =
        params.template tail<kNumDistortionParams>();

    PixelImage<typename PointTypeT::Scalar> distorted_point_in_camera_z1_plane =
        projImpl(distortion, proj_point_in_camera_z1_plane.eval());

    return AffineTransform::warp(
        params.template head<4>(), distorted_point_in_camera_z1_plane);
  }

  template <class ScalarT>
  static ProjInCameraZ1Plane<ScalarT> unwarp(
      const Params<ScalarT>& params, const PixelImage<ScalarT>& pixel_image) {
    PixelImage<ScalarT> proj_point_in_camera_z1_plane = unprojImpl(
        params.template tail<kNumDistortionParams>().eval(),
        AffineTransform::unwarp(params.template head<4>().eval(), pixel_image));

    return ProjInCameraZ1Plane<ScalarT>(
        proj_point_in_camera_z1_plane[0], proj_point_in_camera_z1_plane[1]);
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

    Eigen::Matrix<typename PointTypeT::Scalar, 2, 2> dx;
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
