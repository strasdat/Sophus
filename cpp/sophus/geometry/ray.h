// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/common.h"

#include <Eigen/Dense>
#include <farm_ng/core/logging/expected.h>
#include <farm_ng/core/logging/logger.h>

#include <optional>

namespace sophus {

// Forward declarations
template <class TScalar, int kN>
class UnitVector;
template <class TScalar, int kN>
class Ray;

// Convenience typedefs
template <class TScalar>
using UnitVector3 = UnitVector<TScalar, 3>;
template <class TScalar>
using UnitVector2 = UnitVector<TScalar, 2>;

using UnitVector2F64 = UnitVector2<double>;
using UnitVector3F64 = UnitVector3<double>;

template <class TScalar>
using Ray2 = Ray<TScalar, 2>;
template <class TScalar>
using Ray3 = Ray<TScalar, 3>;

using Ray2F64 = Ray2<double>;
using Ray3F64 = Ray3<double>;

template <class TScalar, int kN>
class UnitVector {
 public:
  // Precondition: v must be of unit length.
  static UnitVector fromUnitVector(Eigen::Matrix<TScalar, kN, 1> const& v) {
    farm_ng::Expected<UnitVector> e_vec = tryFromUnitVector(v);
    if (!e_vec.has_value()) {
      FARM_FATAL("{}", e_vec.error());
    }
    return e_vec.value();
  }

  static farm_ng::Expected<UnitVector> tryFromUnitVector(
      Eigen::Matrix<TScalar, kN, 1> const& v) {
    using std::abs;
    if (abs(v.squaredNorm() - TScalar(1.0)) > sophus::kEpsilon<TScalar>) {
      return FARM_ERROR(
          "v must be unit length is {}.\n{}", v.squaredNorm(), v.transpose());
    }
    UnitVector unit_vector;
    unit_vector.vector_ = v;
    return unit_vector;
  }

  static UnitVector fromVectorAndNormalize(
      Eigen::Matrix<TScalar, kN, 1> const& v) {
    return fromUnitVector(v.normalized());
  }

  [[nodiscard]] Eigen::Matrix<TScalar, kN, 1> const& vector() const {
    return vector_;
  }

  // Class invariant established from sibling we're copying from
  UnitVector(UnitVector const&) = default;
  UnitVector& operator=(UnitVector const&) = default;

 private:
  UnitVector() {}

  // Class invariant: v_ is of unit length.
  Eigen::Matrix<TScalar, kN, 1> vector_;
};

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

}  // namespace sophus
