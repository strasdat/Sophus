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

  [[nodiscard]] Eigen::Matrix<TScalar, kN, 1> const& origin() const {
    return origin_;
  }
  [[nodiscard]] Eigen::Matrix<TScalar, kN, 1>& origin() { return origin_; }

  [[nodiscard]] UnitVector<TScalar, kN> const& direction() const {
    return direction_;
  }
  [[nodiscard]] UnitVector<TScalar, kN>& direction() { return direction_; }

  [[nodiscard]] Eigen::Matrix<TScalar, kN, 1> pointAt(TScalar lambda) const {
    return this->origin_ + lambda * this->direction_.vector();
  }

  struct IntersectionResult {
    double lambda;
    Eigen::Matrix<TScalar, kN, 1> point;
  };

  [[nodiscard]] std::optional<IntersectionResult> intersect(
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

  [[nodiscard]] Eigen::Matrix<TScalar, kN, 1> projection(
      Eigen::Matrix<TScalar, kN, 1> const& point) const {
    return origin_ +
           direction_.getVector().dot(point - origin_) * direction_.vector();
  }

 private:
  Eigen::Matrix<TScalar, kN, 1> origin_;
  UnitVector<TScalar, kN> direction_;
};

template <class TT>
inline Ray<TT, 2> operator*(
    Se2<TT> const& bar_pose_foo, Ray<TT, 2> const& ray_foo) {
  return Ray<TT, 2>(
      bar_pose_foo * ray_foo.origin(),
      bar_pose_foo.so2() * ray_foo.direction());
}

template <class TT>
inline Ray<TT, 3> operator*(
    Se3<TT> const& bar_pose_foo, Ray<TT, 3> const& ray_foo) {
  return Ray<TT, 3>(
      bar_pose_foo * ray_foo.origin(),
      bar_pose_foo.so3() * ray_foo.direction());
}

// Arbitrary 6-DoF transformation of a unit vector promotes it to a ray
// having a potentially non-zero origin.
template <class TT>
inline Ray<TT, 3> operator*(
    Se3<TT> const& bar_pose_foo, UnitVector<TT, 3> const& v_foo) {
  return Ray<TT, 3>(bar_pose_foo.translation(), bar_pose_foo.so3() * v_foo);
}

template <class TT>
Ray2<TT> operator*(Sim2<TT> const& b_from_a, Ray2<TT> const& ray_a) {
  return Ray2<TT>(
      b_from_a * ray_a.origin(), b_from_a.rxso2() * ray_a.direction());
}

template <class TT>
Ray3<TT> operator*(Sim3<TT> const& b_from_a, Ray3<TT> const& ray_a) {
  return Ray3<TT>(
      b_from_a * ray_a.origin(), b_from_a.rxso3() * ray_a.direction());
}

template <class TT>
struct ClosestApproachResult {
  TT lambda0;
  TT lambda1;
  TT min_distance;
};

/// For two parametric lines in lambda0 and lambda1 respectively,
/// ```
///   line_0: x(lambda0) = o0 + lambda0 * d0
///   line_1: y(lambda1) = o1 + lambda1 * d1
/// ```
/// returns distances [lambda0, lambda1] along the respective rays,
/// corresponding to the closest approach of x and y according to an l2 distance
/// measure. lambda0 and lambda1 may be positive or negative. If line_0 and
/// line_1 are parallel, returns nullopt as no unique solution exists
template <class TT>
std::optional<ClosestApproachResult<TT>> closestApproachParameters(
    Ray3<TT> const& line_0, Ray3<TT> const& line_1) {
  using std::abs;
  // Closest approach when line segment connecting closest points on each line
  // is perpendicular to both d0 and d1, thus:
  // ```
  //    x(lambda0)-y(lambda1) = thi*(d0 X d1), for free scalar thi.
  // => o0-o1 + lambda0*d0 - lambda1*d1 - thi*(d0Xd1) = 0
  // => (d0|-d1|-d0Xd1).(lambda0,lambda1,thi)^T = -(o0-o1)
  //           A       .        x               =   b
  // ```

  Eigen::Vector<TT, 3> const d0_cross_d1 =
      line_0.direction().vector().cross(line_1.direction().vector());

  TT const d0_cross_s1_length = d0_cross_d1.norm();

  if (d0_cross_s1_length < kEpsilon<TT>) {
    // Rays are parrallel so no unique solution exists.
    return std::nullopt;
  }

  Eigen::Matrix<TT, 3, 3> mat_a;
  mat_a << line_0.direction().vector(), -line_1.direction().vector(),
      -d0_cross_d1;

  Eigen::Vector<TT, 3> const b = line_1.origin() - line_0.origin();

  Eigen::Vector<TT, 3> const x = mat_a.lu().solve(b);
  TT const lambda0 = x[0];
  TT const lambda1 = x[1];
  TT const min_distance = d0_cross_s1_length * x[2];

  return ClosestApproachResult<TT>{lambda0, lambda1, min_distance};
}

/// For two lines ``line_0`` and ``line_1`` returns the mid-point of the line
/// segment connecting one point from each of the lines which are closest to
/// one another according to the l2 distance measure.
///
/// If line_0 and line_1 are parallel, returns nullopt as no unique solution
/// exists
template <class TT>
std::optional<Eigen::Vector3<TT>> closestApproach(
    Ray3<TT> const& line_0, Ray3<TT> const& line_1) {
  auto maybe_result = closestApproachParameters(line_0, line_1);
  if (!maybe_result) {
    return std::nullopt;
  }
  return (line_0.pointAt(maybe_result->lambda0) +
          line_1.pointAt(maybe_result->lambda1)) /
         static_cast<TT>(2.0);
}

}  // namespace sophus
