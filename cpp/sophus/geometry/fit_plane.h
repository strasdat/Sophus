// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace sophus {

inline auto fitPlaneToPoints(Eigen::Matrix3Xd const& points)
    -> Eigen::Hyperplane<double, 3> {
  Eigen::Vector3d mean = points.rowwise().mean();
  Eigen::Matrix3Xd points_centered = points.colwise() - mean;
  Eigen::JacobiSVD<Eigen::Matrix3Xd> svd(
      points_centered, Eigen::ComputeFullU | Eigen::ComputeThinV);
  Eigen::Vector3d normal = svd.matrixU().col(2);
  return Eigen::Hyperplane<double, 3>(normal, mean);
}

// convenience overloads
auto fitPlaneToPoints(std::vector<Eigen::Vector3d> const& points)
    -> Eigen::Hyperplane<double, 3> {
  Eigen::Map<const Eigen::Matrix3Xd> map(
      points.data()->data(), 3, points.size());
  return fitPlaneToPoints(map.eval());
}

auto fitPlaneToPoints(std::vector<Eigen::Vector3f> const& points)
    -> Eigen::Hyperplane<double, 3> {
  Eigen::Map<const Eigen::Matrix3Xf> map(
      points.data()->data(), 3, points.size());
  return fitPlaneToPoints(map.cast<double>().eval());
}

}  // namespace sophus
