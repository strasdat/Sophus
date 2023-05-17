// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/concepts/group_accessors.h"
#include "sophus/lie/impl/rotation3.h"
#include "sophus/lie/lie_group.h"
#include "sophus/linalg/orthogonal.h"

namespace sophus {

// definition: origin and distance preserving mapping in R^2
template <class TScalar>
class Rotation3 : public lie::Group<Rotation3, TScalar, lie::Rotation3Impl> {
 public:
  using Scalar = TScalar;
  using Base = lie::Group<Rotation3, TScalar, lie::Rotation3Impl>;

  using Tangent = typename Base::Tangent;
  using Params = typename Base::Params;
  using Point = typename Base::Point;

  Rotation3() = default;

  explicit Rotation3(UninitTag /*unused*/) {}

  template <class TOtherScalar>
  auto cast() const -> Rotation3<TOtherScalar> {
    return Rotation3<TOtherScalar>::fromParams(
        this->params_.template cast<TOtherScalar>());
  }

  /// Construct x-axis rotation.
  ///
  static auto fromRx(TScalar const& x) -> Rotation3 {
    return Rotation3::exp(Eigen::Vector3<TScalar>(x, TScalar(0), TScalar(0)));
  }

  /// Construct y-axis rotation.
  ///
  static auto fromRy(TScalar const& y) -> Rotation3 {
    return Rotation3::exp(Eigen::Vector3<TScalar>(TScalar(0), y, TScalar(0)));
  }

  /// Construct z-axis rotation.
  ///
  static auto fromRz(TScalar const& z) -> Rotation3 {
    return Rotation3::exp(Eigen::Vector3<TScalar>(TScalar(0), TScalar(0), z));
  }

  static auto fromUnitQuaternion(Quaternion<TScalar> const& q) -> Rotation3 {
    return Rotation3::fromParams(q.params());
  }

  static auto fitFromQuaternion(Quaternion<TScalar> const& q) -> Rotation3 {
    return Rotation3::fromParams(q.params().normalized());
  }

  static auto fromRotationMatrix(Eigen::Matrix3<TScalar> const& mat_r)
      -> Rotation3 {
    SOPHUS_ASSERT(
        isOrthogonal(mat_r),
        "R is not orthogonal:\n {}",
        mat_r * mat_r.transpose());
    SOPHUS_ASSERT(
        mat_r.determinant() > TScalar(0),
        "det(R) is not positive: {}",
        mat_r.determinant());
    return Rotation3::fromParams(Eigen::Quaternion<TScalar>(mat_r).coeffs());
  }

  static auto fitFromMatrix(Eigen::Matrix3<TScalar> const& mat_r) -> Rotation3 {
    return Rotation3::fromRotationMatrix(makeRotationMatrix(mat_r));
  }

  [[nodiscard]] auto rotationMatrix() const -> Eigen::Matrix3<TScalar> {
    return this->matrix();
  }

  [[nodiscard]] auto unitQuaternion() const -> Quaternion<TScalar> {
    return Quaternion<TScalar>::fromParams(this->params_);
  }

  auto setUnitQuaternion(Quaternion<TScalar> const& z) const -> void {
    this->setParams(z);
  }
};

/// Construct rotation which would take unit direction vector ``from`` into
/// ``to`` such that ``to = rotThroughPoints(from,to) * from``. I.e. that the
/// rotated point ``from`` is colinear with ``to`` (equal up to scale)
///
/// The axis of rotation is perpendicular to both ``from`` and ``to``.
///
template <class TScalar>
auto rotThroughPoints(
    UnitVector3<TScalar> const& from, UnitVector3<TScalar> const& to)
    -> Rotation3<TScalar> {
  using std::abs;
  using std::atan2;
  Eigen::Vector<TScalar, 3> from_cross_to = from.vector().cross(to.vector());
  TScalar n = from_cross_to.norm();
  if (abs(n) < sophus::kEpsilon<TScalar>) {
    return Rotation3<TScalar>();
  }
  // https://stackoverflow.com/a/32724066
  TScalar angle = atan2(n, from.vector().dot(to.vector()));

  return Rotation3<TScalar>::exp(angle * from_cross_to / n);
}

/// Construct rotation which would take direction vector ``from`` into ``to``
/// such that ``to \propto rotThroughPoints(from,to) * from``. I.e. that the
/// rotated point ``from`` is colinear with ``to`` (equal up to scale)
///
/// The axis of rotation is perpendicular to both ``from`` and ``to``.
///
/// Precondition: Neither ``from`` nor ``to`` must be zero. This is
// unchecked.
template <class TScalar>
auto rotThroughPoints(
    Eigen::Vector<TScalar, 3> const& from, Eigen::Vector<TScalar, 3> const& to)
    -> Rotation3<TScalar> {
  using std::abs;
  using std::atan2;
  Eigen::Vector<TScalar, 3> from_cross_to = from.cross(to);
  TScalar n = from_cross_to.norm();
  if (abs(n) < sophus::kEpsilon<TScalar>) {
    return Rotation3<TScalar>();
  }
  // https://stackoverflow.com/a/32724066
  TScalar angle = atan2(n, from.dot(to));

  return Rotation3<TScalar>::exp(angle * from_cross_to / n);
}

using Rotation3F32 = Rotation3<float>;
using Rotation3F64 = Rotation3<double>;
static_assert(concepts::Rotation3<Rotation3F64>);

namespace details {
template <class TT>
class Cast<sophus::Rotation3<TT>> {
 public:
  template <class TTo>
  static auto impl(sophus::Rotation3<TT> const& v) {
    return v.template cast<typename TTo::Scalar>();
  }
  template <class TTo>
  static auto implScalar(sophus::Rotation3<TT> const& v) {
    return v.template cast<TTo>();
  }
};
}  // namespace details

template <class TScalar>
using SO3 = Rotation3<TScalar>;  // NOLINT
using SO3f = Rotation3<float>;   // NOLINT
using SO3d = Rotation3<double>;  // NOLINT

}  // namespace sophus
