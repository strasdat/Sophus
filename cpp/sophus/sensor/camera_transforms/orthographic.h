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

class OrthographicProjection {
 public:
  static constexpr int kNumDistortionParams = 0;
  static constexpr int kNumParams = 4;
  static const constexpr std::string_view kProjectionModel =
      "Orthographic:scale_x,scale_y,offset_x,offset_y";

  template <class ScalarT>
  using PointCamera = Eigen::Matrix<ScalarT, 3, 1>;
  template <class ScalarT>
  using PixelImage = Eigen::Matrix<ScalarT, 2, 1>;
  template <class ScalarT>
  using Params = Eigen::Matrix<ScalarT, kNumParams, 1>;
  template <class ScalarT>
  using DistorationParams = Eigen::Matrix<ScalarT, kNumDistortionParams, 1>;

  template <class ParamsTypeT, class PointTypeT>
  static PixelImage<typename PointTypeT::Scalar> camProj(
      const Eigen::MatrixBase<ParamsTypeT>& params,
      const Eigen::MatrixBase<PointTypeT>& point_camera) {
    using ParamScalar = typename ParamsTypeT::Scalar;
    using PointScalar = typename PointTypeT::Scalar;
    using ReturnScalar = typename Eigen::
        ScalarBinaryOpTraits<ParamScalar, PointScalar>::ReturnType;

    static_assert(
        ParamsTypeT::ColsAtCompileTime == 1, "params must be a column-vector");
    static_assert(
        ParamsTypeT::RowsAtCompileTime == kNumParams,
        "params must have exactly kNumParams rows");
    static_assert(
        PointTypeT::ColsAtCompileTime == 1,
        "point_camera must be a column-vector");
    static_assert(
        PointTypeT::RowsAtCompileTime == 3,
        "point_camera must have exactly 3 columns");
    FARM_CHECK_NE(
        point_camera.z(),
        PointScalar(0),
        "z(={}) must not be zero.",
        point_camera.z());
    auto scale_x = params[0];
    auto scale_y = params[1];
    auto offset_x = params[2];
    auto offset_y = params[3];
    return PixelImage<ReturnScalar>(
        (point_camera.x() - offset_x) / scale_x,
        (point_camera.y() - offset_y) / scale_y);
  }

  template <class ScalarT>
  static PointCamera<ScalarT> camUnproj(
      const Params<ScalarT>& params,
      const PixelImage<ScalarT>& pixel_image,
      double depth_z) {
    ScalarT scale_x = params[0];
    ScalarT scale_y = params[1];
    ScalarT offset_x = params[2];
    ScalarT offset_y = params[3];
    return PointCamera<ScalarT>(
        scale_x * pixel_image.x() + offset_x,
        scale_y * pixel_image.y() + offset_y,
        depth_z);
  }
};

}  // namespace sophus
