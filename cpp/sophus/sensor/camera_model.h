// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/core/common.h"
#include "sophus/geometry/point_transform.h"
#include "sophus/geometry/projection.h"
#include "sophus/image/image_size.h"
#include "sophus/lie/se3.h"
#include "sophus/sensor/camera_transforms/affine.h"
#include "sophus/sensor/camera_transforms/brown_conrady.h"
#include "sophus/sensor/camera_transforms/kannala_brandt.h"
#include "sophus/sensor/camera_transforms/orthographic.h"

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
template <class ScalarT>
Eigen::Matrix<ScalarT, 2, 1> subsampleDown(
    const Eigen::Matrix<ScalarT, 2, 1>& in) {
  return ScalarT(0.5) * in;
}

/// Subsamples pixel up, factor of 2.0.
///
/// See for details:
/// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.97r8rr8owwpc
template <class ScalarT>
Eigen::Matrix<ScalarT, 2, 1> subsampleUp(
    const Eigen::Matrix<ScalarT, 2, 1>& in) {
  return ScalarT(2.0) * in;
}

/// Bins pixel down, factor of 0.5.
///
/// See for details:
/// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.elfm6123mecj
template <class ScalarT>
Eigen::Matrix<ScalarT, 2, 1> binDown(const Eigen::Matrix<ScalarT, 2, 1>& in) {
  Eigen::Matrix<ScalarT, 2, 1> out;
  // Map left image border from -0.5 to 0.0, then scale down, then
  // map the left image border from 0.0 back to -0.5
  out[0] = ScalarT(0.5) * (in[0] + 0.5) - 0.5;  // cx
  out[1] = ScalarT(0.5) * (in[1] + 0.5) - 0.5;  // cy
  return out;
}

/// Bins pixel up, factor of 2.0.
///
/// See for details:
/// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.elfm6123mecj
template <class ScalarT>
Eigen::Matrix<ScalarT, 2, 1> binUp(const Eigen::Matrix<ScalarT, 2, 1>& in) {
  Eigen::Matrix<ScalarT, 2, 1> out;
  // Map left image border from -0.5 to 0.0, then scale up, then
  // map the left image border from 0.0 back to -0.5
  out[0] = ScalarT(2.0) * (in[0] + 0.5) - 0.5;  // cx
  out[1] = ScalarT(2.0) * (in[1] + 0.5) - 0.5;  // cy
  return out;
}

/// Camera model class template for pinhole-like camera projections.
template <class ScalarT, class ProjectionT>
class CameraModelT {
 public:
  using Proj = ProjectionT;
  static constexpr int kNumDistortionParams = Proj::kNumDistortionParams;
  static constexpr int kNumParams = Proj::kNumParams;
  static const constexpr std::string_view kProjectionModel =
      Proj::kProjectionModel;

  using PointCamera = Eigen::Matrix<ScalarT, 3, 1>;
  using PixelImage = Eigen::Matrix<ScalarT, 2, 1>;
  using ProjInCameraZ1Plane = Eigen::Matrix<ScalarT, 2, 1>;
  using Params = Eigen::Matrix<ScalarT, kNumParams, 1>;
  using DistorationParams = Eigen::Matrix<ScalarT, kNumDistortionParams, 1>;

  /// Constructs camera model from image size and set up parameters.
  CameraModelT(const ImageSize& image_size, const Params& params)
      : image_size_(image_size), params_(params) {}

  /// Returns camera model from raw data pointer. To be used within ceres
  /// optimization only.
  static CameraModelT fromData(ScalarT const* const ptr) {
    CameraModelT out;
    Eigen::Map<Eigen::Matrix<ScalarT, kNumParams, 1> const> map(
        ptr, kNumParams, 1);
    out.params_ = map;
    return out;
  }

  /// Focal length in x.
  [[nodiscard]] ScalarT fx() const { return params_[0]; }

  /// Focal length in y.
  [[nodiscard]] ScalarT fy() const { return params_[1]; }

  /// Camera/projection center x.
  [[nodiscard]] ScalarT cx() const { return params_[2]; }

  /// Camera/projection center y.
  [[nodiscard]] ScalarT cy() const { return params_[3]; }

  /// Returns distortion parameters by value.
  [[nodiscard]] DistorationParams distortionParams() const {
    return params_.template tail<kNumDistortionParams>();
  }

  /// Parameters mutator
  Eigen::Matrix<ScalarT, kNumParams, 1>& mutParams() { return params_; }

  /// Parameters accessor
  [[nodiscard]] const Eigen::Matrix<ScalarT, kNumParams, 1>& params() const {
    return params_;
  }

  /// Subsamples pixel down, factor of 0.5.
  ///
  /// See for details:
  /// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.97r8rr8owwpc
  ///
  /// If the original width [height] is odd, the new width [height] will be:
  /// (width+1)/2 [height+1)/2].
  [[nodiscard]] CameraModelT subsampleDown() const {
    Params params = this->params_;
    params[0] = ScalarT(0.5) * params[0];  // fx
    params[1] = ScalarT(0.5) * params[1];  // fy
    params.template segment<2>(2) = ::sophus::subsampleDown(
        params.template segment<2>(2).eval());  // cx, cy
    return CameraModelT(half(image_size_), params);
  }

  /// Subsamples pixel up, factor of 2.0.
  ///
  /// See for details:
  /// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.97r8rr8owwpc
  [[nodiscard]] CameraModelT subsampleUp() const {
    Params params = this->params_;
    params[0] = ScalarT(2.0) * params[0];  // fx
    params[1] = ScalarT(2.0) * params[1];  // fy
    params.template segment<2>(2) =
        ::sophus::subsampleUp(params.template segment<2>(2).eval());  // cx, cy
    return CameraModelT(
        {image_size_.width * 2, image_size_.height * 2}, params);
  }

  /// Bins pixel down, factor of 0.5.
  ///
  /// See for details:
  /// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.elfm6123mecj
  ///
  /// If the original width [height] is odd, the new width [height] will be:
  /// (width+1)/2 [height+1)/2].
  [[nodiscard]] CameraModelT binDown() const {
    Params params = this->params_;
    params[0] = ScalarT(0.5) * params[0];  // fx
    params[1] = ScalarT(0.5) * params[1];  // fy
    params.template segment<2>(2) =
        ::sophus::binDown(params.template segment<2>(2).eval());  // cx, cy
    return CameraModelT(half(image_size_), params);
  }

  /// Bins pixel up, factor of 2.0.
  ///
  /// See for details:
  /// https://docs.google.com/document/d/1xmhCMWklP2UoQMGaMqFnsoPWoeMvBfXN7S8-ie9k0UA/edit#heading=h.elfm6123mecj
  [[nodiscard]] CameraModelT binUp() const {
    Params params = this->params_;
    params[0] = ScalarT(2.0) * params[0];  // fx
    params[1] = ScalarT(2.0) * params[1];  // fy
    params.template segment<2>(2) =
        ::sophus::binUp(params.template segment<2>(2).eval());  // cx, cy
    return CameraModelT(
        {image_size_.width * 2, image_size_.height * 2}, params);
  }

  [[nodiscard]] CameraModelT scale(const ImageSize& image_size) const {
    Params params = this->params_;
    params[0] = ScalarT(image_size.width) / ScalarT(image_size_.width) *
                params[0];  // fx
    params[1] = ScalarT(image_size.height) / ScalarT(image_size_.height) *
                params[1];  // fy
    params[2] = ScalarT(image_size.width) / ScalarT(image_size_.width) *
                params[2];  // cx
    params[3] = ScalarT(image_size.height) / ScalarT(image_size_.height) *
                params[3];  // cy
    return CameraModelT({image_size.width, image_size.height}, params);
  }

  /// Region of interest given `top_left` and ``roi_size`.
  [[nodiscard]] CameraModelT roi(
      const Eigen::Vector2i& top_left, ImageSize roi_size) const {
    Params params = this->params_;
    params[2] = params[2] - top_left.x();  // cx
    params[3] = params[3] - top_left.y();  // cy
    return CameraModelT(roi_size, params);
  }

  /// Maps a 2-point in the z=1 plane of the camera to a pixel in the image.
  [[nodiscard]] PixelImage warp(
      const ProjInCameraZ1Plane& point2_in_camera_z1_plane) const {
    return Proj::template warp(params_, point2_in_camera_z1_plane);
  }

  /// Projects 3-point in camera frame to a pixel in the image.
  [[nodiscard]] PixelImage camProj(const PointCamera& point_in_camera) const {
    return Proj::template warp(params_, ::sophus::proj(point_in_camera));
  }

  [[nodiscard]] Eigen::Matrix<ScalarT, 2, 3> dxCamProjX(
      const PointCamera& point_in_camera) const {
    ProjInCameraZ1Plane point_in_z1plane = ::sophus::proj(point_in_camera);
    return dxWarp(point_in_z1plane) * dxProjX(point_in_camera);
  }

  /// Maps a pixel in the image to a 2-point in the z=1 plane of the camera.
  [[nodiscard]] ProjInCameraZ1Plane unwarp(
      const PixelImage& pixel_in_image) const {
    return Proj::template unwarp(params_, pixel_in_image);
  }

  [[nodiscard]] Eigen::Matrix<ScalarT, 2, 2> dxWarp(
      const PixelImage& pixel_in_image) const {
    return Proj::template dxWarp(params_, pixel_in_image);
  }

  /// Unprojects pixel in the image to point in camera frame.
  ///
  /// The point is projected onto the xy-plane at z = `depth_z`.
  [[nodiscard]] PointCamera camUnproj(
      const PixelImage& pixel_in_image, double depth_z) const {
    return ::sophus::unproj(
        Proj::template unwarp(params_, pixel_in_image), depth_z);
  }

  /// Raw data access. To be used in ceres optimization only.
  ScalarT* mutData() { return params_.data(); }

  /// Accessor of image size.
  [[nodiscard]] const ImageSize& imageSize() const { return image_size_; }

  /// Returns true if obs is within image.
  ///
  /// Note: Postiive border makes the image frame smaller.
  [[nodiscard]] bool contains(
      const Eigen::Vector2i& obs, int border = 0) const {
    return this->image_size_.contains(obs, border);
  }

  /// Returns true if obs is within image.
  ///
  /// Note: Postiive border makes the image frame smaller.
  [[nodiscard]] bool contains(
      const PixelImage& obs, ScalarT border = ScalarT(0)) const {
    return this->image_size_.contains(obs, border);
  }

 private:
  CameraModelT() = default;
  ImageSize image_size_;
  Eigen::Matrix<ScalarT, kNumParams, 1> params_;
};

/// Orthographic camera model class template.
template <class ScalarT>
class OrthographicModelT {
 public:
  using Proj = OrthographicProjection;
  static constexpr int kNumDistortionParams = Proj::kNumDistortionParams;
  static constexpr int kNumParams = Proj::kNumParams;
  static const constexpr std::string_view kProjectionModel =
      Proj::kProjectionModel;

  using PointCamera = Eigen::Matrix<ScalarT, 3, 1>;
  using PixelImage = Eigen::Matrix<ScalarT, 2, 1>;
  using Params = Eigen::Matrix<ScalarT, kNumParams, 1>;
  using DistorationParams = Eigen::Matrix<ScalarT, kNumDistortionParams, 1>;

  OrthographicModelT(ImageSize image_size, const Params& params)
      : image_size_(image_size), params_(params) {}

  OrthographicModelT(
      ImageSize image_size,
      const Eigen::Matrix<ScalarT, 2, 1>& scale,
      const Eigen::Matrix<ScalarT, 2, 1>& offset)
      : image_size_(image_size),
        params_(scale[0], scale[1], offset[0], offset[1]) {}

  static OrthographicModelT fromData(ScalarT const* const ptr) {
    OrthographicModelT out;
    Eigen::Map<Eigen::Matrix<ScalarT, kNumDistortionParams, 1> const> map(
        ptr, kNumDistortionParams, 1);
    out.params_ = map;
    return out;
  }

  [[nodiscard]] ScalarT scaleX() const { return params_[0]; }

  [[nodiscard]] ScalarT scaleY() const { return params_[1]; }

  [[nodiscard]] ScalarT offsetX() const { return params_[2]; }

  [[nodiscard]] ScalarT offsetY() const { return params_[3]; }

  // maps left most horizontal pixel coordinate (-0.5) to x-coordinate in 3d.
  [[nodiscard]] ScalarT left() const {
    return scaleX() * ScalarT(-0.5) + offsetX();
  }

  // maps right most horizontal pixel coordinate (width-0.5) to xn-coordinate in
  // 3d.
  [[nodiscard]] ScalarT right() const {
    return scaleX() * (image_size_.width - ScalarT(0.5)) + offsetX();
  }

  // maps top most vertical pixel coordinate (-0.5) to y-coordinate in 3d.
  [[nodiscard]] ScalarT top() const {
    return scaleY() * ScalarT(-0.5) * offsetY();
  }

  // maps bottom most vertical pixel coordinate (height-0.5) to y-coordinate in
  // 3d.
  [[nodiscard]] ScalarT bottom() const {
    return scaleY() * (image_size_.height - ScalarT(0.5)) + offsetX();
  }

  [[nodiscard]] OrthographicModelT subsampleDown() const {
    FARM_FATAL("not implemented");
  }

  [[nodiscard]] OrthographicModelT subsampleUp() const {
    FARM_FATAL("not implemented");
  }

  [[nodiscard]] OrthographicModelT binDown() const {
    FARM_FATAL("not implemented");
  }

  [[nodiscard]] OrthographicModelT binUp() const {
    FARM_FATAL("not implemented");
  }

  [[nodiscard]] DistorationParams distortionParams() const {
    return params_.template tail<kNumDistortionParams>();
  }

  Eigen::Matrix<ScalarT, kNumParams, 1>& mutParams() { return params_; }

  [[nodiscard]] const Eigen::Matrix<ScalarT, kNumParams, 1>& params() const {
    return params_;
  }

  [[nodiscard]] PixelImage camProj(const PointCamera& point_camera) const {
    return Proj::template camProj(params_, point_camera);
  }

  [[nodiscard]] PointCamera camUnproj(
      const PixelImage& pixel_image, double depth_z) const {
    return Proj::template camUnproj(params_, pixel_image, depth_z);
  }

  ScalarT* mutData() { return params_.data(); }

  [[nodiscard]] const ImageSize& imageSize() const { return image_size_; }

  /// Creates default pinhole model from `image_size`.
  [[nodiscard]] bool contains(
      const Eigen::Vector2i& obs, int border = 0) const {
    return this->image_size_.contains(obs, border);
  }

  /// Creates default pinhole model from `image_size`.
  [[nodiscard]] bool contains(
      const PixelImage& obs, ScalarT border = ScalarT(0)) const {
    return this->image_size_.contains(obs, border);
  }

 private:
  OrthographicModelT() = default;
  ImageSize image_size_;
  Eigen::Matrix<ScalarT, kNumParams, 1> params_;
};

/// Returns orthographic camera model given bounding box and image size.
template <class ScalarT>
OrthographicModelT<ScalarT> orthoCamFromBoundingBox(
    const Eigen::AlignedBox<ScalarT, 2>& bounding_box, ImageSize image_size) {
  ScalarT min_x = bounding_box.min().x();
  ScalarT min_y = bounding_box.min().y();
  ScalarT max_x = bounding_box.max().x();
  ScalarT max_y = bounding_box.max().y();

  // (-0.5, -0.5)   -> (min.x, min.y)
  // (-0.5, h-0.5)  -> (min.x, max.y)
  // (w-0.5, -0.5)  -> (max.x, min.y)
  // (w-0.5, h-0.5) -> (max.x, max.y)
  //
  // Thus we have the following relationship:
  //
  // u*sx + ox = x
  // v*sy + oy = y
  //
  // -0.5*sx  + ox = min.x   => ox = min.x + 0.5*sx
  // (w-0.5)*sx + ox = max.x
  //
  // (w-0.5)*sx + min.x + 0.5*sx = max.x  =>  sx = (max.x-min.x) / w

  Eigen::Matrix<ScalarT, 2, 1> scale(
      (max_x - min_x) / (image_size.width),
      (max_y - min_y) / (image_size.height));
  Eigen::Matrix<ScalarT, 2, 1> offset(
      min_x + 0.5 * scale.x(), min_y + 0.5 * scale.y());
  return OrthographicModelT<ScalarT>(image_size, scale, offset);
}

/// Returns 2d bounding box corresponding the the given orthographic camera
/// model.
template <class ScalarT>
Eigen::AlignedBox<ScalarT, 2> boundingBoxFromOrthoCam(
    const OrthographicModelT<ScalarT>& ortho_cam) {
  Eigen::AlignedBox<ScalarT, 2> bounding_box;
  Eigen::Matrix<ScalarT, 2, 1>& min = bounding_box.min();
  Eigen::Matrix<ScalarT, 2, 1>& max = bounding_box.max();

  min.x() = ortho_cam.offsetX() - 0.5 * ortho_cam.scaleX();
  min.y() = ortho_cam.offsetY() - 0.5 * ortho_cam.scaleY();
  max.x() = ortho_cam.imageSize().width * ortho_cam.scaleX() + min.x();
  max.y() = ortho_cam.imageSize().height * ortho_cam.scaleY() + min.y();

  return bounding_box;
}

/// Camera model projection type.
FARM_ENUM(CameraTransformType, (pinhole, brown_conrady, kannala_brandt_k3));

/// Pinhole camera model.
using PinholeModel = CameraModelT<double, AffineTransform>;

/// Brown Conrady camera model.
using BrownConradyModel = CameraModelT<double, BrownConradyTransform>;

/// KannalaBrandt camera model with k0, k1, k2, k3.
using KannalaBrandtK3Model = CameraModelT<double, KannalaBrandtK3Transform>;
using OrthographicModel = OrthographicModelT<double>;

/// Variant of camera models.
using CameraTransformVariant =
    std::variant<PinholeModel, BrownConradyModel, KannalaBrandtK3Model>;

static_assert(
    std::variant_size_v<CameraTransformVariant> ==
        getCount(CameraTransformType()),
    "When the variant CameraTransformVariant is updated, one needs to "
    "update the enum CameraTransformType as well, and vice versa.");

/// Concrete camera model class.
class CameraModel {
 public:
  /// Constructs camera model from `frame_name` and concrete projection model.
  template <class TransformModelT>
  CameraModel(std::string frame_name, TransformModelT model)
      : frame_name_(std::move(frame_name)), model_(model) {}

  /// Constructs camera model from `frame_name`, `image_size`, `projection_type`
  /// flag and `params` vector.
  ///
  /// Precondition: ``params.size()`` must match the number of parameters of the
  ///               specified `projection_type` (TransformModel::kNumParams).
  CameraModel(
      std::string frame_name,
      ImageSize image_size,
      CameraTransformType projection_type,
      const Eigen::VectorXd& params);

  /// Creates default pinhole model from `image_size`.
  static CameraModel createDefaultPinholeModel(
      std::string frame_name, ImageSize image_size);

  /// Returns string representation for the concrete camera transform flag.
  [[nodiscard]] std::string_view cameraTransformName() const;

  /// Frame name mutator.
  std::string& mutFrameName() { return frame_name_; }

  /// Frame name accessor.
  [[nodiscard]] const std::string& frameName() const { return frame_name_; }

  /// Distortion variant mutator.
  CameraTransformVariant& mutModelVariant() { return model_; }

  /// Distortion variant accessor.
  [[nodiscard]] const CameraTransformVariant& modelVariant() const {
    return model_;
  }

  /// Camera transform flag
  [[nodiscard]] CameraTransformType transformType() const;

  /// Returns `params` vector by value.
  [[nodiscard]] Eigen::VectorXd params() const;

  /// Sets `params` vector.
  ///
  /// Precontion: ``params.size()`` must match the number of parameters of the
  ///             specivied `projection_type` (TransformModel::kNumParams).
  void setParams(const Eigen::VectorXd& params);

  /// Returns distortion parameters vector by value.
  [[nodiscard]] Eigen::VectorXd distortionParams() const;

  /// Given a point in 3D space in the camera frame, compute the corresponding
  /// pixel coordinates in the image.
  [[nodiscard]] Eigen::Vector2d camProj(
      const Eigen::Vector3d& point_camera) const;

  /// Maps a 2-point in the z=1 plane of the camera to a pixel in the image.
  [[nodiscard]] Eigen::Vector2d warp(
      const Eigen::Vector2d& point2_in_camera_z1_plane) const;

  [[nodiscard]] Eigen::Matrix2d dxWarp(
      const Eigen::Vector2d& point2_in_camera_z1_plane) const;

  /// Derivative of camProj(x) with respect to x=0.
  [[nodiscard]] Eigen::Matrix<double, 2, 3> dxCamProjX(
      const Eigen::Vector3d& point_in_camera) const;

  /// Derivative of camProj(exp(x) * point) with respect to x=0.
  [[nodiscard]] Eigen::Matrix<double, 2, 6> dxCamProjExpXPointAt0(
      const Eigen::Vector3d& point_in_camera) const;

  /// Given pixel coordinates in the distorted image, and a corresponding
  /// depth, reproject to a 3d point in the camera's reference frame.
  [[nodiscard]] Eigen::Vector3d camUnproj(
      const Eigen::Vector2d& pixel_image, double depth_z) const;

  /// Maps a pixel in the image to a 2-point in the z=1 plane of the camera.
  [[nodiscard]] Eigen::Vector2d unwarp(
      const Eigen::Vector2d& pixel_image) const;

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
  [[nodiscard]] const ImageSize& imageSize() const;

  /// Region of interest given `top_left` and ``roi_size`.
  [[nodiscard]] CameraModel roi(
      const Eigen::Vector2i& top_left, ImageSize roi_size) const;

  /// Returns true if obs is within image.
  ///
  /// Note: Postiive border makes the image frame smaller.
  [[nodiscard]] bool contains(const Eigen::Vector2i& obs, int border = 0) const;

  /// Returns true if obs is within image.
  ///
  /// Postiive border makes the image frame smaller.
  [[nodiscard]] bool contains(
      const Eigen::Vector2d& obs, double border = 0) const;

  [[nodiscard]] CameraModel scale(ImageSize image_size) const;

 private:
  std::string frame_name_;

  CameraTransformVariant model_;
};

/// Creates default pinhole model from `image_size`.
PinholeModel createDefaultPinholeModel(ImageSize image_size);

}  // namespace sophus
