// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/sensor/camera_model.h"

namespace sophus {

namespace {
auto getModelFromType(
    CameraDistortionType projection_type,
    ImageSize image_size,
    Eigen::VectorXd const& params) -> CameraDistortionVariant {
  switch (projection_type) {
    case CameraDistortionType::pinhole: {
      return PinholeModel(image_size, params);
      break;
    }
    case CameraDistortionType::brown_conrady: {
      return BrownConradyModel(image_size, params);
      break;
    }
    case CameraDistortionType::kannala_brandt_k3: {
      return KannalaBrandtK3Model(image_size, params);
      break;
    }
    case CameraDistortionType::orthographic: {
      return OrthographicModel(image_size, params);
      break;
    }
  }
  SOPHUS_PANIC("logic error");
}
}  // namespace

CameraModel::CameraModel(
    ImageSize image_size,
    CameraDistortionType projection_type,
    Eigen::VectorXd const& params)
    : model_(getModelFromType(projection_type, image_size, params)) {}

auto CameraModel::distortionModelName() const -> std::string_view {
  return std::visit(
      [](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        return T::kProjectionModel;
      },
      model_);
}

auto CameraModel::params() const -> Eigen::VectorXd {
  return std::visit(
      [](auto&& arg) -> Eigen::VectorXd { return arg.params(); }, model_);
}

auto CameraModel::focalLength() const -> Eigen::Vector2d {
  return std::visit(
      [](auto&& arg) -> Eigen::Vector2d { return arg.focalLength(); }, model_);
}

void CameraModel::setFocalLength(Eigen::Vector2d const& focal_length) {
  std::visit(
      [&focal_length](auto&& arg) { return arg.setFocalLength(focal_length); },
      model_);
}

auto CameraModel::principalPoint() const -> Eigen::Vector2d {
  return std::visit(
      [](auto&& arg) -> Eigen::Vector2d { return arg.principalPoint(); },
      model_);
}

void CameraModel::setPrincipalPoint(Eigen::Vector2d const& principal_point) {
  std::visit(
      [&principal_point](auto&& arg) {
        return arg.setPrincipalPoint(principal_point);
      },
      model_);
}

auto CameraModel::distortionParams() const -> Eigen::VectorXd {
  return std::visit(
      [](auto&& arg) -> Eigen::VectorXd { return arg.distortionParams(); },
      model_);
}

void CameraModel::setParams(Eigen::VectorXd const& params) {
  std::visit([&params](auto&& arg) { arg.params() = params; }, model_);
}

auto CameraModel::camProj(Eigen::Vector3d const& point_camera) const
    -> Eigen::Vector2d {
  return std::visit(
      [&](auto&& arg) -> Eigen::Vector2d { return arg.camProj(point_camera); },
      model_);
}

auto CameraModel::camUnproj(Eigen::Vector2d const& pixel_image, double depth_z)
    const -> Eigen::Vector3d {
  return std::visit(
      [&](auto&& arg) -> Eigen::Vector3d {
        return arg.camUnproj(pixel_image, depth_z);
      },
      model_);
}

auto CameraModel::distort(Eigen::Vector2d const& point2_in_camera_lifted) const
    -> Eigen::Vector2d {
  return std::visit(
      [&](auto&& arg) -> Eigen::Vector2d {
        return arg.distort(point2_in_camera_lifted);
      },
      model_);
}

auto CameraModel::dxDistort(
    Eigen::Vector2d const& point2_in_camera_lifted) const -> Eigen::Matrix2d {
  return std::visit(
      [&](auto&& arg) -> Eigen::Matrix2d {
        return arg.dxDistort(point2_in_camera_lifted);
      },
      model_);
}

auto CameraModel::undistort(Eigen::Vector2d const& pixel_image) const
    -> Eigen::Vector2d {
  return std::visit(
      [&](auto&& arg) -> Eigen::Vector2d { return arg.undistort(pixel_image); },
      model_);
}

[[nodiscard]] auto CameraModel::undistortTable() const
    -> MutImage<Eigen::Vector2f> {
  return std::visit(
      [](auto&& arg) -> MutImage<Eigen::Vector2f> {
        return arg.undistortTable();
      },
      model_);
}

auto CameraModel::dxCamProjX(Eigen::Vector3d const& point_in_camera) const
    -> Eigen::Matrix<double, 2, 3> {
  return std::visit(
      [&](auto&& arg) -> Eigen::Matrix<double, 2, 3> {
        return arg.dxCamProjX(point_in_camera);
      },
      model_);
}

auto CameraModel::dxCamProjExpXPointAt0(Eigen::Vector3d const& point_in_camera)
    const -> Eigen::Matrix<double, 2, 6> {
  return std::visit(
      [&](auto&& arg) -> Eigen::Matrix<double, 2, 6> {
        auto dx2 = Isometry3F64::dxExpXTimesPointAt0(point_in_camera);

        return arg.dxCamProjX(point_in_camera) * dx2;
      },
      model_);
}

auto CameraModel::subsampleDown() const -> CameraModel {
  return CameraModel(std::visit(
      [](auto&& arg) -> CameraDistortionVariant { return arg.subsampleDown(); },
      this->model_));
}

auto CameraModel::subsampleUp() const -> CameraModel {
  return CameraModel(std::visit(
      [](auto&& arg) -> CameraDistortionVariant { return arg.subsampleUp(); },
      this->model_));
}

auto CameraModel::binDown() const -> CameraModel {
  return CameraModel(std::visit(
      [](auto&& arg) -> CameraDistortionVariant { return arg.binDown(); },
      this->model_));
}

auto CameraModel::binUp() const -> CameraModel {
  return CameraModel(std::visit(
      [](auto&& arg) -> CameraDistortionVariant { return arg.binUp(); },
      this->model_));
}

auto CameraModel::roi(Eigen::Vector2i const& top_left, ImageSize roi_size) const
    -> CameraModel {
  SOPHUS_ASSERT_LE(top_left.x() + roi_size.width, imageSize().width);
  SOPHUS_ASSERT_LE(top_left.y() + roi_size.height, imageSize().height);
  return CameraModel(std::visit(
      [&](auto&& arg) -> CameraDistortionVariant {
        return arg.roi(top_left, roi_size);
      },
      this->model_));
}

auto CameraModel::contains(Eigen::Vector2i const& obs, int border) const
    -> bool {
  return std::visit(
      [&](auto&& arg) -> bool { return arg.contains(obs, border); },
      this->model_);
}

auto CameraModel::imageSize() const -> ImageSize const& {
  return std::visit(
      [](auto&& arg) -> ImageSize const& { return arg.imageSize(); },
      this->model_);
}

auto CameraModel::contains(Eigen::Vector2d const& obs, double border) const
    -> bool {
  return std::visit(
      [&](auto&& arg) -> bool { return arg.contains(obs, border); },
      this->model_);
}

auto CameraModel::distortionType() const -> CameraDistortionType {
  return std::visit(
      [](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, PinholeModel>) {
          return CameraDistortionType::pinhole;
        } else if constexpr (std::is_same_v<T, BrownConradyModel>) {
          return CameraDistortionType::brown_conrady;
        } else if constexpr (std::is_same_v<T, KannalaBrandtK3Model>) {
          return CameraDistortionType::kannala_brandt_k3;
        } else if constexpr (std::is_same_v<T, OrthographicModel>) {
          return CameraDistortionType::orthographic;
        } else {
          static_assert(AlwaysFalse<T>, "non-exhaustive visitor!");
        }
      },
      this->modelVariant());
}

auto CameraModel::createDefaultPinholeModel(ImageSize image_size)
    -> CameraModel {
  return CameraModel(::sophus::createDefaultPinholeModel(image_size));
}

auto CameraModel::scale(ImageSize image_size) const -> CameraModel {
  return CameraModel(std::visit(
      [&image_size](auto&& arg) -> CameraDistortionVariant {
        return arg.scale(image_size);
      },
      this->model_));
}

auto createDefaultPinholeModel(ImageSize image_size) -> PinholeModel {
  double const fx = image_size.width * 0.5;
  double const fy = fx;
  double const cx = (image_size.width - 1.0) * 0.5;
  double const cy = (image_size.height - 1.0) * 0.5;

  return PinholeModel(image_size, {fx, fy, cx, cy});
}

auto createDefaultOrthoModel(ImageSize image_size) -> OrthographicModel {
  double const sx = 1.0;
  double const sy = 1.0;
  double const cx = 0.0;
  double const cy = 0.0;

  return OrthographicModel(image_size, {sx, sy, cx, cy});
}

}  // namespace sophus
