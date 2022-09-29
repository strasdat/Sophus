// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include <Eigen/Dense>

namespace sophus {

class AffineTransform {
 public:
  static int constexpr kNumDistortionParams = 0;
  static int constexpr kNumParams = 4;
  static constexpr const std::string_view kProjectionModel =
      "Pinhole:fx,fy,cx,cy";

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

    return PixelImage<typename TPointTypeT::Scalar>(
        proj_point_in_camera_z1_plane[0] * params[0] + params[2],
        proj_point_in_camera_z1_plane[1] * params[1] + params[3]);
  }

  template <class TScalar>
  static ProjInCameraZ1Plane<TScalar> undistort(
      Params<TScalar> const& params, PixelImage<TScalar> const& pixel_image) {
    TScalar proj_x_in_camera_z1_plane =
        (pixel_image.x() - TScalar(params[2])) / TScalar(params[0]);
    TScalar proj_y_in_camera_z1_plane =
        (pixel_image.y() - TScalar(params[3])) / TScalar(params[1]);
    return ProjInCameraZ1Plane<TScalar>(
        proj_x_in_camera_z1_plane, proj_y_in_camera_z1_plane);
  }

  template <class TParamsTypeT, class TPointTypeT>
  static Eigen::Matrix<typename TPointTypeT::Scalar, 2, 2> dxDistort(
      Eigen::MatrixBase<TParamsTypeT> const& params,
      Eigen::MatrixBase<TPointTypeT> const& /*proj_point_in_camera_z1_plane*/) {
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

    Eigen::Matrix<typename TPointTypeT::Scalar, 2, 2> dx;

    // clang-format off
    dx <<  //
      params[0],         0,
              0, params[1];
    // clang-format on
    return dx;
  }
};

}  // namespace sophus
