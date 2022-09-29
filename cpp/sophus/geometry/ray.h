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

template <class TScalar>
class UnitVector3 {
 public:
  // Precondition: v must be of unit length.
  static UnitVector3 fromUnitVector(Eigen::Matrix<TScalar, 3, 1> const& v) {
    farm_ng::Expected<UnitVector3> e_vec = tryFromUnitVector(v);
    if (!e_vec.has_value()) {
      FARM_FATAL("{}", e_vec.error());
    }
    return e_vec.value();
  }

  static farm_ng::Expected<UnitVector3> tryFromUnitVector(
      Eigen::Matrix<TScalar, 3, 1> const& v) {
    using std::abs;
    if ((v.squaredNorm() - TScalar(1.0)) < sophus::kEpsilon<TScalar>) {
      return FARM_ERROR(
          "v must be unit length is {}.\n{}", v.squaredNorm(), v.transpose());
    }
    UnitVector3 unit_vector;
    unit_vector.vector_ = v;
    return unit_vector;
  }

  static UnitVector3 fromVectorAndNormalize(
      Eigen::Matrix<TScalar, 3, 1> const& v) {
    return fromUnitVector(v.normalized());
  }

  [[nodiscard]] Eigen::Matrix<TScalar, 3, 1> const& vector() const {
    return vector_;
  }

 private:
  UnitVector3() {}

  // Class invariant: v_ is of unit length.
  Eigen::Matrix<TScalar, 3, 1> vector_;
};

template <class TScalar>
class Ray3 {
 public:
  Ray3(
      Eigen::Matrix<TScalar, 3, 1> const& origin,
      UnitVector3<TScalar> const& direction)
      : origin_(origin), direction_(direction) {}

  Eigen::Matrix<TScalar, 3, 1> const& origin() const { return origin_; }
  Eigen::Matrix<TScalar, 3, 1>& origin() { return origin_; }

  UnitVector3<TScalar> const& direction() const { return direction_; }
  UnitVector3<TScalar>& direction() { return direction_; }

  Eigen::Matrix<TScalar, 3, 1> pointAt(TScalar lambda) const {
    return this->origin_ + lambda * this->direction_.getVector();
  }

  struct IntersectionResult {
    double lambda;
    Eigen::Matrix<TScalar, 3, 1> point;
  };

  std::optional<IntersectionResult> intersect(
      Eigen::Hyperplane<TScalar, 3> const& plane) const {
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

  Eigen::Matrix<TScalar, 3, 1> projection(
      Eigen::Matrix<TScalar, 3, 1> const& point) const {
    return origin_ +
           direction_.getVector().dot(point - origin_) * direction_.vector();
  }

 private:
  Eigen::Matrix<TScalar, 3, 1> origin_;
  UnitVector3<TScalar> direction_;
};

using UnitVector3F64 = UnitVector3<double>;
using Ray3F64 = Ray3<double>;

}  // namespace sophus
