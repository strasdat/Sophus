// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/core/common.h"

#include <Eigen/Dense>
#include <farm_ng/core/logging/logger.h>

#include <optional>

namespace sophus {

template <class T>
class UnitVector3 {
 public:
  // Precondition: v must be of unit length.
  static UnitVector3 fromUnitVector(const Eigen::Matrix<T, 3, 1>& v) {
    using std::abs;
    FARM_CHECK_LE((v.squaredNorm() - T(1.0)), sophus::kEpsilon<T>);
    UnitVector3 unit_vector;
    unit_vector.vector_ = v;
    return unit_vector;
  }

  static UnitVector3 fromVectorAndNormalize(const Eigen::Matrix<T, 3, 1>& v) {
    return fromUnitVector(v.normalized());
  }

  const Eigen::Matrix<T, 3, 1>& getVector() const { return vector_; }

 private:
  UnitVector3() {}

  // Class invariant: v_ is of unit length.
  Eigen::Matrix<T, 3, 1> vector_;
};

template <class TT>
class Ray3 {
 public:
  Ray3(const Eigen::Matrix<TT, 3, 1>& origin, const UnitVector3<TT>& direction)
      : origin_(origin), direction_(direction) {}

  const Eigen::Matrix<TT, 3, 1>& origin() const { return origin_; }
  Eigen::Matrix<TT, 3, 1>& origin() { return origin_; }

  const UnitVector3<TT>& direction() const { return direction_; }
  UnitVector3<TT>& direction() { return direction_; }

  Eigen::Matrix<TT, 3, 1> pointAt(TT lambda) const {
    return this->origin_ + lambda * this->direction_.getVector();
  }

  struct IntersectionResult {
    double lambda;
    Eigen::Matrix<TT, 3, 1> point;
  };

  std::optional<IntersectionResult> intersect(
      const Eigen::Hyperplane<TT, 3>& plane) const {
    using std::abs;
    TT dot_prod = plane.normal().dot(this->direction_.getVector());
    if (abs(dot_prod) < sophus::kEpsilon<TT>) {
      return std::nullopt;
    }
    IntersectionResult result;
    result.lambda =
        -(plane.offset() + plane.normal().dot(this->origin_)) / dot_prod;
    result.point = this->pointAt(result.lambda);
    return result;
  }

  Eigen::Matrix<TT, 3, 1> projection(
      const Eigen::Matrix<TT, 3, 1>& point) const {
    return origin_ +
           direction_.getVector().dot(point - origin_) * direction_.getVector();
  }

 private:
  Eigen::Matrix<TT, 3, 1> origin_;
  UnitVector3<TT> direction_;
};

using UnitVector3F64 = UnitVector3<double>;
using Ray3F64 = Ray3<double>;

}  // namespace sophus
