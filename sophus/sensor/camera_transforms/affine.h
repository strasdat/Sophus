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
  static constexpr int kNumDistortionParams = 0;
  static constexpr int kNumParams = 4;
  static const constexpr std::string_view kProjectionModel =
      "Pinhole:fx,fy,cx,cy";

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

    return PixelImage<typename PointTypeT::Scalar>(
        proj_point_in_camera_z1_plane[0] * params[0] + params[2],
        proj_point_in_camera_z1_plane[1] * params[1] + params[3]);
  }

  template <class ScalarT>
  static ProjInCameraZ1Plane<ScalarT> unwarp(
      const Params<ScalarT>& params, const PixelImage<ScalarT>& pixel_image) {
    ScalarT proj_x_in_camera_z1_plane =
        (pixel_image.x() - ScalarT(params[2])) / ScalarT(params[0]);
    ScalarT proj_y_in_camera_z1_plane =
        (pixel_image.y() - ScalarT(params[3])) / ScalarT(params[1]);
    return ProjInCameraZ1Plane<ScalarT>(
        proj_x_in_camera_z1_plane, proj_y_in_camera_z1_plane);
  }

  template <class ParamsTypeT, class PointTypeT>
  static Eigen::Matrix<typename PointTypeT::Scalar, 2, 2> dxWarp(
      const Eigen::MatrixBase<ParamsTypeT>& params,
      const Eigen::MatrixBase<PointTypeT>& /*proj_point_in_camera_z1_plane*/) {
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

    Eigen::Matrix<typename PointTypeT::Scalar, 2, 2> dx;

    // clang-format off
    dx <<  //
      params[0],         0,
              0, params[1];
    // clang-format on
    return dx;
  }
};

}  // namespace sophus
