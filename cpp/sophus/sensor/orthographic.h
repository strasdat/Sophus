// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image/image_size.h"
#include "sophus/sensor/camera_distortion/orthographic.h"

namespace sophus {

/// Orthographic camera model class template.
template <class TScalar>
class OrthographicModelT {
 public:
  using Proj = OrthographicProjection;
  static int constexpr kNumDistortionParams = Proj::kNumDistortionParams;
  static int constexpr kNumParams = Proj::kNumParams;
  static constexpr const std::string_view kProjectionModel =
      Proj::kProjectionModel;

  using PointCamera = Eigen::Matrix<TScalar, 3, 1>;
  using PixelImage = Eigen::Matrix<TScalar, 2, 1>;
  using Params = Eigen::Matrix<TScalar, kNumParams, 1>;
  using DistorationParams = Eigen::Matrix<TScalar, kNumDistortionParams, 1>;

  OrthographicModelT(ImageSize image_size, Params const& params)
      : image_size_(image_size), params_(params) {}

  OrthographicModelT(
      ImageSize image_size,
      Eigen::Matrix<TScalar, 2, 1> const& scale,
      Eigen::Matrix<TScalar, 2, 1> const& offset)
      : image_size_(image_size),
        params_(scale[0], scale[1], offset[0], offset[1]) {}

  static OrthographicModelT fromData(TScalar const* const ptr) {
    OrthographicModelT out;
    Eigen::Map<Eigen::Matrix<TScalar, kNumDistortionParams, 1> const> map(
        ptr, kNumDistortionParams, 1);
    out.params_ = map;
    return out;
  }

  [[nodiscard]] TScalar scaleX() const { return params_[0]; }

  [[nodiscard]] TScalar scaleY() const { return params_[1]; }

  [[nodiscard]] TScalar offsetX() const { return params_[2]; }

  [[nodiscard]] TScalar offsetY() const { return params_[3]; }

  // maps left most horizontal pixel coordinate (-0.5) to x-coordinate in 3d.
  [[nodiscard]] TScalar left() const {
    return scaleX() * TScalar(-0.5) + offsetX();
  }

  // maps right most horizontal pixel coordinate (width-0.5) to xn-coordinate in
  // 3d.
  [[nodiscard]] TScalar right() const {
    return scaleX() * (image_size_.width - TScalar(0.5)) + offsetX();
  }

  // maps top most vertical pixel coordinate (-0.5) to y-coordinate in 3d.
  [[nodiscard]] TScalar top() const {
    return scaleY() * TScalar(-0.5) * offsetY();
  }

  // maps bottom most vertical pixel coordinate (height-0.5) to y-coordinate in
  // 3d.
  [[nodiscard]] TScalar bottom() const {
    return scaleY() * (image_size_.height - TScalar(0.5)) + offsetX();
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

  Eigen::Matrix<TScalar, kNumParams, 1>& params() { return params_; }

  [[nodiscard]] Eigen::Matrix<TScalar, kNumParams, 1> const& params() const {
    return params_;
  }

  [[nodiscard]] PixelImage camProj(PointCamera const& point_camera) const {
    return Proj::template camProj(params_, point_camera);
  }

  [[nodiscard]] PointCamera camUnproj(
      PixelImage const& pixel_image, double depth_z) const {
    return Proj::template camUnproj(params_, pixel_image, depth_z);
  }

  TScalar* data() { return params_.data(); }

  [[nodiscard]] ImageSize const& imageSize() const { return image_size_; }

  /// Creates default pinhole model from `image_size`.
  [[nodiscard]] bool contains(
      Eigen::Vector2i const& obs, int border = 0) const {
    return this->image_size_.contains(obs, border);
  }

  /// Creates default pinhole model from `image_size`.
  [[nodiscard]] bool contains(
      PixelImage const& obs, TScalar border = TScalar(0)) const {
    return this->image_size_.contains(obs, border);
  }

 private:
  OrthographicModelT() = default;
  ImageSize image_size_;
  Eigen::Matrix<TScalar, kNumParams, 1> params_;
};

using OrthographicModel = OrthographicModelT<double>;

/// Returns orthographic camera model given bounding box and image size.
template <class TScalar>
OrthographicModelT<TScalar> orthoCamFromBoundingBox(
    Eigen::AlignedBox<TScalar, 2> const& bounding_box, ImageSize image_size) {
  TScalar min_x = bounding_box.min().x();
  TScalar min_y = bounding_box.min().y();
  TScalar max_x = bounding_box.max().x();
  TScalar max_y = bounding_box.max().y();

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

  Eigen::Matrix<TScalar, 2, 1> scale(
      (max_x - min_x) / (image_size.width),
      (max_y - min_y) / (image_size.height));
  Eigen::Matrix<TScalar, 2, 1> offset(
      min_x + 0.5 * scale.x(), min_y + 0.5 * scale.y());
  return OrthographicModelT<TScalar>(image_size, scale, offset);
}

/// Returns 2d bounding box corresponding the the given orthographic camera
/// model.
template <class TScalar>
Eigen::AlignedBox<TScalar, 2> boundingBoxFromOrthoCam(
    OrthographicModelT<TScalar> const& ortho_cam) {
  Eigen::AlignedBox<TScalar, 2> bounding_box;
  Eigen::Matrix<TScalar, 2, 1>& min = bounding_box.min();
  Eigen::Matrix<TScalar, 2, 1>& max = bounding_box.max();

  min.x() = ortho_cam.offsetX() - 0.5 * ortho_cam.scaleX();
  min.y() = ortho_cam.offsetY() - 0.5 * ortho_cam.scaleY();
  max.x() = ortho_cam.imageSize().width * ortho_cam.scaleX() + min.x();
  max.y() = ortho_cam.imageSize().height * ortho_cam.scaleY() + min.y();

  return bounding_box;
}

}  // namespace sophus
