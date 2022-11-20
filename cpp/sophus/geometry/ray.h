// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/common.h"
#include "sophus/geometry/unit_vector.h"
#include "sophus/lie/se2.h"
#include "sophus/lie/se3.h"
#include "sophus/lie/sim2.h"
#include "sophus/lie/sim3.h"

#include <Eigen/Dense>
#include <farm_ng/core/logging/expected.h>
#include <farm_ng/core/logging/logger.h>

#include <optional>

namespace sophus {

// Forward declarations
template <class TScalar, int kN>
class Ray;

// Convenience typedefs
template <class TScalar>
using Ray2 = Ray<TScalar, 2>;
template <class TScalar>
using Ray3 = Ray<TScalar, 3>;

using Ray2F64 = Ray2<double>;
using Ray3F64 = Ray3<double>;

template <class TScalar, int kN>
class Ray {
 public:
  Ray(Eigen::Matrix<TScalar, kN, 1> const& origin,
      UnitVector<TScalar, kN> const& direction)
      : origin_(origin), direction_(direction) {}

  Ray(Ray const&) = default;
  Ray& operator=(Ray const&) = default;

  Eigen::Matrix<TScalar, kN, 1> const& origin() const { return origin_; }
  Eigen::Matrix<TScalar, kN, 1>& origin() { return origin_; }

  UnitVector<TScalar, kN> const& direction() const { return direction_; }
  UnitVector<TScalar, kN>& direction() { return direction_; }

  Eigen::Matrix<TScalar, kN, 1> pointAt(TScalar lambda) const {
    return this->origin_ + lambda * this->direction_.vector();
  }

  struct IntersectionResult {
    double lambda;
    Eigen::Matrix<TScalar, kN, 1> point;
  };

  std::optional<IntersectionResult> intersect(
      Eigen::Hyperplane<TScalar, kN> const& plane) const {
    using std::abs;
    TScalar dot_prod = plane.normal().dot(this->direction_.vector());
    if (abs(dot_prod) < sophus::kEpsilon<TScalar>) {
      return std::nullopt;
    }
    IntersectionResult result;
    result.lambda =
        -(plane.offset() + plane.normal().dot(this->origin_)) / dot_prod;
    result.point = this->pointAt(result.lambda);
    return result;
  }

  Eigen::Matrix<TScalar, kN, 1> projection(
      Eigen::Matrix<TScalar, kN, 1> const& point) const {
    return origin_ +
           direction_.getVector().dot(point - origin_) * direction_.vector();
  }

 private:
  Eigen::Matrix<TScalar, kN, 1> origin_;
  UnitVector<TScalar, kN> direction_;
};

template <class TScalar>
inline Ray<TScalar, 2> operator*(
    Se2<TScalar> const& bar_pose_foo, Ray<TScalar, 2> const& ray_foo) {
  return Ray<TScalar, 2>(
      bar_pose_foo * ray_foo.origin(),
      bar_pose_foo.so2() * ray_foo.direction());
}

template <class TScalar>
inline Ray<TScalar, 3> operator*(
    Se3<TScalar> const& bar_pose_foo, Ray<TScalar, 3> const& ray_foo) {
  return Ray<TScalar, 3>(
      bar_pose_foo * ray_foo.origin(),
      bar_pose_foo.so3() * ray_foo.direction());
}

// Arbitrary 6-DoF transformation of a unit vector promotes it to a ray
// having a potentially non-zero origin.
template <class TScalar>
inline Ray<TScalar, 3> operator*(
    Se3<TScalar> const& bar_pose_foo, UnitVector<TScalar, 3> const& v_foo) {
  return Ray<TScalar, 3>(
      bar_pose_foo.translation(), bar_pose_foo.so3() * v_foo);
}

template <typename T>
Ray2<T> operator*(Sim2<T> const& b_from_a, Ray2<T> const& ray_a) {
  return Ray2<T>(
      b_from_a * ray_a.origin(), b_from_a.rxso2() * ray_a.direction());
}

template <typename T>
Ray3<T> operator*(Sim3<T> const& b_from_a, Ray3<T> const& ray_a) {
  return Ray3<T>(
      b_from_a * ray_a.origin(), b_from_a.rxso3() * ray_a.direction());
}

// For two lines:
//   line_a: x = A + lambda * B
//   line_b: y = C + mu * D
// returns distances [lambda, mu] along the respective rays, corresponding to
// the closest approach of x and y according to an l2 distance measure. lambda
// and mu may be positive or negative.
//
// TODO: what if they are parallel? Presumably E * line_a.direction() will be 0.
//       should we return an optional?
template <class T>
std::array<T, 2> closestApproachDistances(
    Ray3<T> const& line_a, Ray3<T> const& line_b) {
  // E = B.D.D^T \in R^3
  const Eigen::Matrix<T, 1, 3> E = line_a.direction().transpose() *
                                   line_b.direction() *
                                   line_b.direction().transpose();

  // lambda = E*(A-C) / (E*B)
  const T lambda =
      E * (line_a.origin() - line_b.origin()) / (E * line_a.direction());

  // mu = (A-C-lambda*B)*D
  const T mu =
      (line_a.origin() - line_b.origin() - lambda * line_a.direction()) *
      line_b.direction();
  return {lambda, mu};
}

// For two lines ``line_a`` and ``line_b`` returns the mid-point of the line
// segment connecting one point from each of the lines which are closest to
// one another according to the l2 distance measure.
template <class T>
Eigen::Vector3<T> closestApproach(
    Ray3<T> const& line_a, Ray3<T> const& line_b) {
  auto [lambda, mu] = closestApproachDistances(line_a, line_b);
  return (line_a.pointAt(lambda) + line_b.pointAt(mu)) / static_cast<T>(2.0);
}

}  // namespace sophus
