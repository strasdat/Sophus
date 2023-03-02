// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/calculus/region.h"
#include "sophus/image/image_size.h"
#include "sophus/sensor/camera_model.h"

namespace sophus {

template <class TScalar>
using OrthographicModelT =
    CameraModelT<TScalar, AffineTransform, ProjectionOrtho>;

/// Returns orthographic camera model given bounding box and image size.
template <class TScalar>
auto orthoCamFromBoundingBox(
    Region2<TScalar> const& bounding_box, ImageSize image_size)
    -> OrthographicModelT<TScalar> {
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

  Eigen::Array<TScalar, 2, 1> const range = bounding_box.range();
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
auto boundingBoxFromOrthoCam(OrthographicModelT<TScalar> const& ortho_cam)
    -> Region2<TScalar> {
  Eigen::Vector<TScalar, 2> min = (-ortho_cam.principalPoint().array() - 0.5) /
                                  ortho_cam.focalLength().array();
  return Region2<TScalar>::fromMinMax(
      min,
      min.array() + ortho_cam.imageSize().array().template cast<TScalar>() /
                        ortho_cam.focalLength().array());
}

}  // namespace sophus
