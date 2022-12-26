// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/common.h"
#include "sophus/lie/rxso2.h"
#include "sophus/lie/rxso3.h"
#include "sophus/lie/so2.h"
#include "sophus/lie/so3.h"

#include <Eigen/Dense>

namespace sophus {

// Forward declarations
template <class TScalar, int kN>
class UnitVector;

// Convenience typedefs
template <class TScalar>
using UnitVector3 = UnitVector<TScalar, 3>;
template <class TScalar>
using UnitVector2 = UnitVector<TScalar, 2>;

using UnitVector2F64 = UnitVector2<double>;
using UnitVector3F64 = UnitVector3<double>;

template <class TScalar, int kN>
class UnitVector {
 public:
  // Precondition: v must be of unit length.
  static UnitVector fromUnitVector(Eigen::Matrix<TScalar, kN, 1> const& v) {
    Expected<UnitVector> e_vec = tryFromUnitVector(v);
    if (!e_vec.has_value()) {
      SOPHUS_PANIC("{}", e_vec.error());
    }
    return e_vec.value();
  }

  static Expected<UnitVector> tryFromUnitVector(
      Eigen::Matrix<TScalar, kN, 1> const& v) {
    using std::abs;
    SOPHUS_ASSERT_OR_ERROR(
        abs(v.squaredNorm() - TScalar(1.0)) < sophus::kEpsilon<TScalar>,
        "v must be unit length is {}.\n{}",
        v.squaredNorm(),
        v.transpose());

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

template <class TScalar>
inline UnitVector<TScalar, 2> operator*(
    So2<TScalar> const& bar_rotation_foo, UnitVector<TScalar, 2> const& v_foo) {
  return UnitVector<TScalar, 2>::fromUnitVector(
      bar_rotation_foo * v_foo.vector());
}

template <class TScalar>
inline UnitVector<TScalar, 3> operator*(
    So3<TScalar> const& bar_rotation_foo, UnitVector<TScalar, 3> const& v_foo) {
  return UnitVector<TScalar, 3>::fromUnitVector(
      bar_rotation_foo * v_foo.vector());
}

template <class TScalar>
inline UnitVector<TScalar, 2> operator*(
    RxSo2<TScalar> const& bar_rotscale_foo,
    UnitVector<TScalar, 2> const& v_foo) {
  return UnitVector<TScalar, 2>::fromVectorAndNormalize(
      bar_rotscale_foo * v_foo.vector());
}

template <class TScalar>
inline UnitVector<TScalar, 3> operator*(
    RxSo3<TScalar> const& bar_rotscale_foo,
    UnitVector<TScalar, 3> const& v_foo) {
  return UnitVector<TScalar, 3>::fromVectorAndNormalize(
      bar_rotscale_foo * v_foo.vector());
}

/// Construct rotation which would take unit direction vector ``from`` into
/// ``to`` such that ``to = rotThroughPoints(from,to) * from``. I.e. that the
/// rotated point ``from`` is colinear with ``to`` (equal up to scale)
///
/// The axis of rotation is perpendicular to both ``from`` and ``to``.
///
template <class TScalar>
SOPHUS_FUNC So3<TScalar> rotThroughPoints(
    UnitVector3<TScalar> const& from, UnitVector3<TScalar> const& to) {
  using std::abs;
  using std::atan2;
  Eigen::Vector<TScalar, 3> from_cross_to = from.vector().cross(to.vector());
  TScalar n = from_cross_to.norm();
  if (abs(n) < sophus::kEpsilon<TScalar>) {
    return So3<TScalar>();
  }
  // https://stackoverflow.com/a/32724066
  TScalar angle = atan2(n, from.vector().dot(to.vector()));

  return So3<TScalar>::exp(angle * from_cross_to / n);
}

/// Construct rotation which would take direction vector ``from`` into ``to``
/// such that ``to \propto rotThroughPoints(from,to) * from``. I.e. that the
/// rotated point ``from`` is colinear with ``to`` (equal up to scale)
///
/// The axis of rotation is perpendicular to both ``from`` and ``to``.
///
/// Precondition: Neither ``from`` nor ``to`` must be zero. This is unchecked.
template <class TScalar>
SOPHUS_FUNC So3<TScalar> rotThroughPoints(
    Eigen::Vector<TScalar, 3> const& from,
    Eigen::Vector<TScalar, 3> const& to) {
  using std::abs;
  using std::atan2;
  Eigen::Vector<TScalar, 3> from_cross_to = from.cross(to);
  TScalar n = from_cross_to.norm();
  if (abs(n) < sophus::kEpsilon<TScalar>) {
    return So3<TScalar>();
  }
  // https://stackoverflow.com/a/32724066
  TScalar angle = atan2(n, from.dot(to));

  return So3<TScalar>::exp(angle * from_cross_to / n);
}

}  // namespace sophus
