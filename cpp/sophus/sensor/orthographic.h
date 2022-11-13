// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image/image_size.h"
#include "sophus/sensor/camera_model.h"

namespace sophus {

template <class TScalar>
using OrthographicModelT =
    CameraModelT<TScalar, AffineTransform, ProjectionOrtho>;

/// Returns orthographic camera model given bounding box and image size.
template <class TScalar>
OrthographicModelT<TScalar> orthoCamFromBoundingBox(
    Eigen::AlignedBox<TScalar, 2> const& bounding_box, ImageSize image_size) {
  // (-0.5, -0.5)   -> (min.x, min.y)
  // (-0.5, h-0.5)  -> (min.x, max.y)
  // (w-0.5, -0.5)  -> (max.x, min.y)
  // (w-0.5, h-0.5) -> (max.x, max.y)
  //
  // Thus we have the following relationship:
  // u = x*sx + ox
  // v = y*sy + oy
  //
  // -0.5 = min.x * sx  + ox   => ox = -(0.5 + min.x * sx)
  // (w-0.5) = max.x * sx + ox
  //
  // w = sx * (max.x  - min.x)
  // sx = w / (max.x  - min.x)

  Eigen::Array<TScalar, 2, 1> const range =
      bounding_box.max() - bounding_box.min();
  Eigen::Array<TScalar, 2, 1> const scale =
      Eigen::Array<TScalar, 2, 1>(image_size.width, image_size.height) / range;
  Eigen::Array<TScalar, 2, 1> const offset =
      -(0.5 + bounding_box.min().array() * scale);

  Eigen::Matrix<TScalar, 4, 1> const params(
      scale.x(), scale.y(), offset.x(), offset.y());

  return OrthographicModelT<TScalar>(image_size, params);
}

/// Returns 2d bounding box corresponding the the given orthographic camera
/// model.
template <class TScalar>
Eigen::AlignedBox<TScalar, 2> boundingBoxFromOrthoCam(
    OrthographicModelT<TScalar> const& ortho_cam) {
  Eigen::AlignedBox<TScalar, 2> bounding_box;

  Eigen::Array<TScalar, 2, 1> const imsize(
      ortho_cam.imageSize().width, ortho_cam.imageSize().height);

  bounding_box.min() = (-ortho_cam.principalPoint().array() - 0.5) /
                       ortho_cam.focalLength().array();
  bounding_box.max() =
      bounding_box.min().array() + imsize / ortho_cam.focalLength().array();

  return bounding_box;
}

}  // namespace sophus
