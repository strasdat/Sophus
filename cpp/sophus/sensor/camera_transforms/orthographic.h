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
  static int constexpr kNumDistortionParams = 0;
  static int constexpr kNumParams = 4;
  static constexpr const std::string_view kProjectionModel =
      "Orthographic:scale_x,scale_y,offset_x,offset_y";

  template <class TScalar>
  using PointCamera = Eigen::Matrix<TScalar, 3, 1>;
  template <class TScalar>
  using PixelImage = Eigen::Matrix<TScalar, 2, 1>;
  template <class TScalar>
  using Params = Eigen::Matrix<TScalar, kNumParams, 1>;
  template <class TScalar>
  using DistorationParams = Eigen::Matrix<TScalar, kNumDistortionParams, 1>;

  template <class TParamsTypeT, class TPointTypeT>
  static PixelImage<typename TPointTypeT::Scalar> camProj(
      Eigen::MatrixBase<TParamsTypeT> const& params,
      Eigen::MatrixBase<TPointTypeT> const& point_camera) {
    using ParamScalar = typename TParamsTypeT::Scalar;
    using PointScalar = typename TPointTypeT::Scalar;
    using ReturnScalar = typename Eigen::
        ScalarBinaryOpTraits<ParamScalar, PointScalar>::ReturnType;

    static_assert(
        TParamsTypeT::ColsAtCompileTime == 1, "params must be a column-vector");
    static_assert(
        TParamsTypeT::RowsAtCompileTime == kNumParams,
        "params must have exactly kNumParams rows");
    static_assert(
        TPointTypeT::ColsAtCompileTime == 1,
        "point_camera must be a column-vector");
    static_assert(
        TPointTypeT::RowsAtCompileTime == 3,
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

  template <class TScalar>
  static PointCamera<TScalar> camUnproj(
      Params<TScalar> const& params,
      PixelImage<TScalar> const& pixel_image,
      double depth_z) {
    TScalar scale_x = params[0];
    TScalar scale_y = params[1];
    TScalar offset_x = params[2];
    TScalar offset_y = params[3];
    return PointCamera<TScalar>(
        scale_x * pixel_image.x() + offset_x,
        scale_y * pixel_image.y() + offset_y,
        depth_z);
  }
};

}  // namespace sophus
