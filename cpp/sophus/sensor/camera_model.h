// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/common.h"
#include "sophus/geometry/point_transform.h"
#include "sophus/geometry/projection.h"
#include "sophus/image/image.h"
#include "sophus/image/image_size.h"
#include "sophus/lie/se3.h"
#include "sophus/sensor/camera_distortion/affine.h"
#include "sophus/sensor/camera_distortion/brown_conrady.h"
#include "sophus/sensor/camera_distortion/kannala_brandt.h"
#include "sophus/sensor/orthographic.h"

#include <Eigen/Dense>
#include <farm_ng/core/enum/enum.h>
#include <farm_ng/core/logging/logger.h>

#include <numeric>
#include <variant>

namespace sophus {

/// Subsamples pixel down, factor of 0.5.
///
/// See for details:
/// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.97r8rr8owwpc
template <class TScalar>
Eigen::Matrix<TScalar, 2, 1> subsampleDown(
    Eigen::Matrix<TScalar, 2, 1> const& in) {
  return TScalar(0.5) * in;
}

/// Subsamples pixel up, factor of 2.0.
///
/// See for details:
/// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.97r8rr8owwpc
template <class TScalar>
Eigen::Matrix<TScalar, 2, 1> subsampleUp(
    Eigen::Matrix<TScalar, 2, 1> const& in) {
  return TScalar(2.0) * in;
}

/// Bins pixel down, factor of 0.5.
///
/// See for details:
/// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.elfm6123mecj
template <class TScalar>
Eigen::Matrix<TScalar, 2, 1> binDown(Eigen::Matrix<TScalar, 2, 1> const& in) {
  Eigen::Matrix<TScalar, 2, 1> out;
  // Map left image border from -0.5 to 0.0, then scale down, then
  // map the left image border from 0.0 back to -0.5
  out[0] = TScalar(0.5) * (in[0] + 0.5) - 0.5;  // cx
  out[1] = TScalar(0.5) * (in[1] + 0.5) - 0.5;  // cy
  return out;
}

/// Bins pixel up, factor of 2.0.
///
/// See for details:
/// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.elfm6123mecj
template <class TScalar>
Eigen::Matrix<TScalar, 2, 1> binUp(Eigen::Matrix<TScalar, 2, 1> const& in) {
  Eigen::Matrix<TScalar, 2, 1> out;
  // Map left image border from -0.5 to 0.0, then scale up, then
  // map the left image border from 0.0 back to -0.5
  out[0] = TScalar(2.0) * (in[0] + 0.5) - 0.5;  // cx
  out[1] = TScalar(2.0) * (in[1] + 0.5) - 0.5;  // cy
  return out;
}

/// Camera model class template for pinhole-like camera projections.
///
/// In particular, for those models a point in the world is first projected onto
/// the z=1 plane before being projected to a pixel in the image. The first step
/// is identical for all instances of this class template, while the projection
/// is specific to the concrete specialization and includes the concrete lens
/// distortions.
///
/// This the z=1 parametrization, cameras with up to 180 degree FOV can be
/// represented, so there will be likely numerical issues when approaching that
/// limit.
///
/// And alternative representation is the map the world point on a unit sphere
/// (and not the z=q plan) before projecting in the image. This way large FOV
/// such as fish eye lenses can be represented without the above mentioned
/// limitation.
///
/// Broadly speaking, cameras with pinhole-like distortion and FOV are well
/// represented by this z=1 representation, while fish-eye lenses better use an
/// intermediate sphere projection.
template <class TScalar, class TProjection>
class Z1ProjCameraModelT {
 public:
  using Proj = TProjection;
  static int constexpr kNumDistortionParams = Proj::kNumDistortionParams;
  static int constexpr kNumParams = Proj::kNumParams;
  static constexpr const std::string_view kProjectionModel =
      Proj::kProjectionModel;

  using PointCamera = Eigen::Matrix<TScalar, 3, 1>;
  using PixelImage = Eigen::Matrix<TScalar, 2, 1>;
  using ProjInCameraZ1Plane = Eigen::Matrix<TScalar, 2, 1>;
  using Params = Eigen::Matrix<TScalar, kNumParams, 1>;
  using DistorationParams = Eigen::Matrix<TScalar, kNumDistortionParams, 1>;

  /// Constructs camera model from image size and set up parameters.
  Z1ProjCameraModelT(ImageSize const& image_size, Params const& params)
      : image_size_(image_size), params_(params) {}

  Z1ProjCameraModelT() : image_size_({0, 0}) {
    params_.setZero();
    params_.template head<2>().setOnes();
  }

  /// Returns camera model from raw data pointer. To be used within ceres
  /// optimization only.
  static Z1ProjCameraModelT fromData(TScalar const* const ptr) {
    Z1ProjCameraModelT out;
    Eigen::Map<Eigen::Matrix<TScalar, kNumParams, 1> const> map(
        ptr, kNumParams, 1);
    out.params_ = map;
    return out;
  }

  /// Focal length.
  [[nodiscard]] PixelImage focalLength() const {
    return params_.template head<2>();
  }

  /// Focal length.
  void setFocalLength(PixelImage const& focal_length) {
    params_.template head<2>() = focal_length;
  }

  [[nodiscard]] PixelImage principalPoint() const {
    return params_.template segment<2>(2);
  }

  /// Focal length.
  void setPrincipalPoint(PixelImage const& principal_point) {
    params_.template segment<2>(2) = principal_point;
  }

  /// Returns distortion parameters by value.
  [[nodiscard]] DistorationParams distortionParams() const {
    return params_.template tail<kNumDistortionParams>();
  }

  /// Parameters mutator
  Eigen::Matrix<TScalar, kNumParams, 1>& params() { return params_; }

  /// Parameters accessor
  [[nodiscard]] Eigen::Matrix<TScalar, kNumParams, 1> const& params() const {
    return params_;
  }

  /// Subsamples pixel down, factor of 0.5.
  ///
  /// See for details:
  /// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.97r8rr8owwpc
  ///
  /// If the original width [height] is odd, the new width [height] will be:
  /// (width+1)/2 [height+1)/2].
  [[nodiscard]] Z1ProjCameraModelT subsampleDown() const {
    Params params = this->params_;
    params[0] = TScalar(0.5) * params[0];  // fx
    params[1] = TScalar(0.5) * params[1];  // fy
    params.template segment<2>(2) = ::sophus::subsampleDown(
        params.template segment<2>(2).eval());  // cx, cy
    return Z1ProjCameraModelT(half(image_size_), params);
  }

  /// Subsamples pixel up, factor of 2.0.
  ///
  /// See for details:
  /// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.97r8rr8owwpc
  [[nodiscard]] Z1ProjCameraModelT subsampleUp() const {
    Params params = this->params_;
    params[0] = TScalar(2.0) * params[0];  // fx
    params[1] = TScalar(2.0) * params[1];  // fy
    params.template segment<2>(2) =
        ::sophus::subsampleUp(params.template segment<2>(2).eval());  // cx, cy
    return Z1ProjCameraModelT(
        {image_size_.width * 2, image_size_.height * 2}, params);
  }

  /// Bins pixel down, factor of 0.5.
  ///
  /// See for details:
  /// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.elfm6123mecj
  ///
  /// If the original width [height] is odd, the new width [height] will be:
  /// (width+1)/2 [height+1)/2].
  [[nodiscard]] Z1ProjCameraModelT binDown() const {
    Params params = this->params_;
    params[0] = TScalar(0.5) * params[0];  // fx
    params[1] = TScalar(0.5) * params[1];  // fy
    params.template segment<2>(2) =
        ::sophus::binDown(params.template segment<2>(2).eval());  // cx, cy
    return Z1ProjCameraModelT(half(image_size_), params);
  }

  /// Bins pixel up, factor of 2.0.
  ///
  /// See for details:
  /// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.elfm6123mecj
  [[nodiscard]] Z1ProjCameraModelT binUp() const {
    Params params = this->params_;
    params[0] = TScalar(2.0) * params[0];  // fx
    params[1] = TScalar(2.0) * params[1];  // fy
    params.template segment<2>(2) =
        ::sophus::binUp(params.template segment<2>(2).eval());  // cx, cy
    return Z1ProjCameraModelT(
        {image_size_.width * 2, image_size_.height * 2}, params);
  }

  [[nodiscard]] Z1ProjCameraModelT scale(ImageSize const& image_size) const {
    Params params = this->params_;
    params[0] = TScalar(image_size.width) / TScalar(image_size_.width) *
                params[0];  // fx
    params[1] = TScalar(image_size.height) / TScalar(image_size_.height) *
                params[1];  // fy
    params[2] = TScalar(image_size.width) / TScalar(image_size_.width) *
                params[2];  // cx
    params[3] = TScalar(image_size.height) / TScalar(image_size_.height) *
                params[3];  // cy
    return Z1ProjCameraModelT({image_size.width, image_size.height}, params);
  }

  /// Region of interest given `top_left` and ``roi_size`.
  [[nodiscard]] Z1ProjCameraModelT roi(
      Eigen::Vector2i const& top_left, ImageSize roi_size) const {
    Params params = this->params_;
    params[2] = params[2] - top_left.x();  // cx
    params[3] = params[3] - top_left.y();  // cy
    return Z1ProjCameraModelT(roi_size, params);
  }

  /// Maps a 2-point in the z=1 plane of the camera to a pixel in the image.
  [[nodiscard]] PixelImage distort(
      ProjInCameraZ1Plane const& point2_in_camera_z1_plane) const {
    return Proj::template distort(params_, point2_in_camera_z1_plane);
  }

  [[nodiscard]] Eigen::Matrix<TScalar, 2, 2> dxDistort(
      PixelImage const& pixel_in_image) const {
    return Proj::template dxDistort(params_, pixel_in_image);
  }

  /// Maps a pixel in the image to a 2-point in the z=1 plane of the camera.
  [[nodiscard]] ProjInCameraZ1Plane undistort(
      PixelImage const& pixel_in_image) const {
    return Proj::template undistort(params_, pixel_in_image);
  }

  [[nodiscard]] MutImage<Eigen::Vector2f> undistortTable() const {
    MutImage<Eigen::Vector2f> table(image_size_);
    for (int v = 0; v < table.height(); ++v) {
      Eigen::Vector2f* row_ptr = table.rowPtrMut(v);
      for (int u = 0; u < table.width(); ++u) {
        row_ptr[u] = this->undistort(PixelImage(u, v)).template cast<float>();
      }
    }
    return table;
  }

  /// Projects 3-point in camera frame to a pixel in the image.
  [[nodiscard]] PixelImage camProj(PointCamera const& point_in_camera) const {
    return Proj::template distort(params_, ::sophus::proj(point_in_camera));
  }

  [[nodiscard]] Eigen::Matrix<TScalar, 2, 3> dxCamProjX(
      PointCamera const& point_in_camera) const {
    ProjInCameraZ1Plane point_in_z1plane = ::sophus::proj(point_in_camera);
    return dxDistort(point_in_z1plane) * dxProjX(point_in_camera);
  }

  /// Unprojects pixel in the image to point in camera frame.
  ///
  /// The point is projected onto the xy-plane at z = `depth_z`.
  [[nodiscard]] PointCamera camUnproj(
      PixelImage const& pixel_in_image, double depth_z) const {
    return ::sophus::unproj(
        Proj::template undistort(params_, pixel_in_image), depth_z);
  }

  /// Raw data access. To be used in ceres optimization only.
  TScalar* data() { return params_.data(); }

  /// Accessor of image size.
  [[nodiscard]] ImageSize const& imageSize() const { return image_size_; }

  /// Returns true if obs is within image.
  ///
  /// Note: Postiive border makes the image frame smaller.
  [[nodiscard]] bool contains(
      Eigen::Vector2i const& obs, int border = 0) const {
    return this->image_size_.contains(obs, border);
  }

  /// Returns true if obs is within image.
  ///
  /// Note: Postiive border makes the image frame smaller.
  [[nodiscard]] bool contains(
      PixelImage const& obs, TScalar border = TScalar(0)) const {
    return this->image_size_.contains(obs, border);
  }

 private:
  ImageSize image_size_;
  Eigen::Matrix<TScalar, kNumParams, 1> params_;
};

/// Camera model projection type.
FARM_ENUM(Z1ProjDistortionType, (pinhole, brown_conrady, kannala_brandt_k3));

/// Pinhole camera model.
using PinholeModel = Z1ProjCameraModelT<double, AffineZ1Projection>;

/// Brown Conrady camera model.
using BrownConradyModel = Z1ProjCameraModelT<double, BrownConradyZ1Projection>;

/// KannalaBrandt camera model with k0, k1, k2, k3.
using KannalaBrandtModel =
    Z1ProjCameraModelT<double, KannalaBrandtZ1Projection>;

/// Variant of z1 camera models.
using Z1ProjDistortationVariant =
    std::variant<PinholeModel, BrownConradyModel, KannalaBrandtModel>;

static_assert(
    std::variant_size_v<Z1ProjDistortationVariant> ==
        getCount(Z1ProjDistortionType()),
    "When the variant Z1ProjDistortationVariant is updated, one needs to "
    "update the enum Z1ProjDistortionType as well, and vice versa.");

/// Concrete camera model class.
class Z1ProjCameraModel {
 public:
  Z1ProjCameraModel()
      : model_(PinholeModel({0, 0}, Eigen::Vector4d(1.0, 1.0, 0.0, 0.0))) {}

  /// Constructs camera model from concrete projection model.
  template <class Z1ProjCameraModelT>
  Z1ProjCameraModel(Z1ProjCameraModelT model) : model_(model) {}

  /// Constructs camera model from `frame_name`, `image_size`, `projection_type`
  /// flag and `params` vector.
  ///
  /// Precondition: ``params.size()`` must match the number of parameters of the
  ///               specified `projection_type` (TransformModel::kNumParams).
  Z1ProjCameraModel(
      ImageSize image_size,
      Z1ProjDistortionType projection_type,
      Eigen::VectorXd const& params);

  /// Creates default pinhole model from `image_size`.
  static Z1ProjCameraModel createDefaultPinholeModel(ImageSize image_size);

  /// Returns name of the camera distortion model.
  [[nodiscard]] std::string_view distortionModelName() const;

  /// Distortion variant mutator.
  Z1ProjDistortationVariant& modelVariant() { return model_; }

  /// Distortion variant accessor.
  [[nodiscard]] Z1ProjDistortationVariant const& modelVariant() const {
    return model_;
  }

  /// Camera distortion flag
  [[nodiscard]] Z1ProjDistortionType distortionType() const;

  [[nodiscard]] Eigen::Vector2d focalLength() const;

  /// Focal length.
  void setFocalLength(Eigen::Vector2d const& focal_length);

  [[nodiscard]] Eigen::Vector2d principalPoint() const;

  /// Focal length.
  void setPrincipalPoint(Eigen::Vector2d const& principal_point);

  /// Returns `params` vector by value.
  [[nodiscard]] Eigen::VectorXd params() const;

  /// Sets `params` vector.
  ///
  /// Precontion: ``params.size()`` must match the number of parameters of the
  ///             specivied `projection_type` (Z1Projection::kNumParams).
  void setParams(Eigen::VectorXd const& params);

  /// Returns distortion parameters vector by value.
  [[nodiscard]] Eigen::VectorXd distortionParams() const;

  /// Given a point in 3D space in the camera frame, compute the corresponding
  /// pixel coordinates in the image.
  [[nodiscard]] Eigen::Vector2d camProj(
      Eigen::Vector3d const& point_camera) const;

  /// Maps a 2-point in the z=1 plane of the camera to a (distorted) pixel in
  /// the image.
  [[nodiscard]] Eigen::Vector2d distort(
      Eigen::Vector2d const& point2_in_camera_z1_plane) const;

  [[nodiscard]] Eigen::Matrix2d dxDistort(
      Eigen::Vector2d const& point2_in_camera_z1_plane) const;

  /// Maps a pixel in the image to a 2-point in the z=1 plane of the camera.
  [[nodiscard]] Eigen::Vector2d undistort(
      Eigen::Vector2d const& pixel_image) const;

  [[nodiscard]] MutImage<Eigen::Vector2f> undistortTable() const;

  /// Derivative of camProj(x) with respect to x=0.
  [[nodiscard]] Eigen::Matrix<double, 2, 3> dxCamProjX(
      Eigen::Vector3d const& point_in_camera) const;

  /// Derivative of camProj(exp(x) * point) with respect to x=0.
  [[nodiscard]] Eigen::Matrix<double, 2, 6> dxCamProjExpXPointAt0(
      Eigen::Vector3d const& point_in_camera) const;

  /// Given pixel coordinates in the distorted image, and a corresponding
  /// depth, reproject to a 3d point in the camera's reference frame.
  [[nodiscard]] Eigen::Vector3d camUnproj(
      Eigen::Vector2d const& pixel_image, double depth_z) const;

  /// Subsamples pixel down, factor of 0.5.
  ///
  /// See for details:
  /// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.97r8rr8owwpc
  ///
  /// If the original width [height] is odd, the new width [height] will be:
  /// (width+1)/2 [height+1)/2].
  [[nodiscard]] Z1ProjCameraModel subsampleDown() const;

  /// Subsamples pixel up, factor of 2.0.
  [[nodiscard]] Z1ProjCameraModel subsampleUp() const;

  /// Bins pixel down, factor of 0.5.
  ///
  /// See for details:
  /// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.elfm6123mecj
  ///
  /// If the original width [height] is odd, the new width [height] will be:
  /// (width+1)/2 [height+1)/2].
  [[nodiscard]] Z1ProjCameraModel binDown() const;

  /// Bins pixel up, factor of 2.0.
  ///
  /// See for details:
  /// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.elfm6123mecj
  [[nodiscard]] Z1ProjCameraModel binUp() const;

  /// Image size accessor
  [[nodiscard]] ImageSize const& imageSize() const;

  /// Region of interest given `top_left` and ``roi_size`.
  [[nodiscard]] Z1ProjCameraModel roi(
      Eigen::Vector2i const& top_left, ImageSize roi_size) const;

  /// Returns true if obs is within image.
  ///
  /// Note: Postiive border makes the image frame smaller.
  [[nodiscard]] bool contains(Eigen::Vector2i const& obs, int border = 0) const;

  /// Returns true if obs is within image.
  ///
  /// Postiive border makes the image frame smaller.
  [[nodiscard]] bool contains(
      Eigen::Vector2d const& obs, double border = 0) const;

  [[nodiscard]] Z1ProjCameraModel scale(ImageSize image_size) const;

 private:
  Z1ProjDistortationVariant model_;
};

/// Creates default pinhole model from `image_size`.
PinholeModel createDefaultPinholeModel(ImageSize image_size);

/// Variant of camera models.
using CameraModelVariant = std::variant<Z1ProjCameraModel, OrthographicModel>;

/// Concrete camera model class.
class CameraModel {
 public:
  CameraModel()
      : model_(PinholeModel({0, 0}, Eigen::Vector4d(1.0, 1.0, 0.0, 0.0))) {}

  /// Constructs camera model from  concrete projection model.
  template <class ProjCameraModelT>
  CameraModel(ProjCameraModelT model) : model_(model) {}

  // /// Constructs camera model from `frame_name`, `image_size`,
  // `projection_type`
  // /// flag and `params` vector.
  // ///
  // /// Precondition: ``params.size()`` must match the number of parameters of
  // the
  // ///               specified `projection_type` (TransformModel::kNumParams).
  // Z1ProjCameraModel(
  //     ImageSize image_size,
  //     Z1ProjDistortionType projection_type,
  //     Eigen::VectorXd const& params);

  /// Creates default pinhole model from `image_size`.
  static CameraModel createDefaultPinholeModel(ImageSize image_size);

  /// Returns name of the camera projection model.
  [[nodiscard]] std::string_view projectionModelName() const;

  /// Camera model variant mutator.
  CameraModelVariant& modelVariant() { return model_; }

  /// Camera model variant accessor.
  [[nodiscard]] CameraModelVariant const& modelVariant() const {
    return model_;
  }

  bool isZ1Proj() const {
    return std::holds_alternative<Z1ProjCameraModel>(model_);
  }

  bool isOrtho() const {
    return std::holds_alternative<OrthographicModel>(model_);
  }

  /// Returns `params` vector by value.
  [[nodiscard]] Eigen::VectorXd params() const;

  /// Sets `params` vector.
  ///
  /// Precontion: ``params.size()`` must match the number of parameters of the
  ///             specivied `projection_type` (Projection::kNumParams).
  void setParams(Eigen::VectorXd const& params);

  /// Given a point in 3D space in the camera frame, compute the corresponding
  /// pixel coordinates in the image.
  [[nodiscard]] Eigen::Vector2d camProj(
      Eigen::Vector3d const& point_camera) const;

  /// Given pixel coordinates in the distorted image, and a corresponding
  /// depth, reproject to a 3d point in the camera's reference frame.
  [[nodiscard]] Eigen::Vector3d camUnproj(
      Eigen::Vector2d const& pixel_image, double depth_z) const;

  /// Subsamples pixel down, factor of 0.5.
  ///
  /// See for details:
  /// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.97r8rr8owwpc
  ///
  /// If the original width [height] is odd, the new width [height] will be:
  /// (width+1)/2 [height+1)/2].
  [[nodiscard]] CameraModel subsampleDown() const;

  /// Subsamples pixel up, factor of 2.0.
  [[nodiscard]] CameraModel subsampleUp() const;

  /// Bins pixel down, factor of 0.5.
  ///
  /// See for details:
  /// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.elfm6123mecj
  ///
  /// If the original width [height] is odd, the new width [height] will be:
  /// (width+1)/2 [height+1)/2].
  [[nodiscard]] CameraModel binDown() const;

  /// Bins pixel up, factor of 2.0.
  ///
  /// See for details:
  /// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.elfm6123mecj
  [[nodiscard]] CameraModel binUp() const;

  /// Image size accessor
  [[nodiscard]] ImageSize imageSize() const;

  /// Returns true if obs is within image.
  ///
  /// Note: Postiive border makes the image frame smaller.
  [[nodiscard]] bool contains(Eigen::Vector2i const& obs, int border = 0) const;

  /// Returns true if obs is within image.
  ///
  /// Postiive border makes the image frame smaller.
  [[nodiscard]] bool contains(
      Eigen::Vector2d const& obs, double border = 0) const;

 private:
  CameraModelVariant model_;
};

}  // namespace sophus
