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
#include "sophus/lie/impl/translation_factor_group_product.h"
#include "sophus/lie/lie_group.h"
#include "sophus/lie/rotation3.h"

namespace sophus {

// definition: distance preserving mapping in R^3
//             <==> shape and size preserving mapping in R^3
template <class TScalar>
class Isometry3
    : public lie::Group<
          Isometry3,
          TScalar,
          lie::WithDimAndSubgroup<3, lie::Rotation3Impl>::SemiDirectProduct> {
 public:
  using Scalar = TScalar;

  using Base = lie::Group<
      Isometry3,
      TScalar,
      lie::WithDimAndSubgroup<3, lie::Rotation3Impl>::SemiDirectProduct>;
  using Rotation = Rotation3<Scalar>;

  using Tangent = typename Base::Tangent;
  using Params = typename Base::Params;
  using Point = typename Base::Point;

  Isometry3() = default;

  explicit Isometry3(UninitTag /*unused*/) : Base(UninitTag{}) {}

  Isometry3(
      Eigen::Vector<Scalar, 3> const& translation,
      Rotation3<Scalar> const& group)
      : Isometry3(UninitTag{}) {
    this->params_.template head<4>() = group.params();
    this->params_.template tail<3>() = translation;
  }

  [[deprecated(
      "Use Isometry3(t, R) instead. Rotation and translation do not "
      "commute and constructor arguments are sorted in sophus2 based on order "
      "of operation from right to left:"
      "Isometry3(t, R) * point == t + R * point.")]]  //
  explicit Isometry3(
      Rotation3<Scalar> const& group,
      Eigen::Vector<Scalar, 3> const& translation)
      : Isometry3(translation, group) {}

  Isometry3(Rotation3<Scalar> const& group) {
    this->params_.template head<4>() = group.params();
  }

  Isometry3(Eigen::Vector<Scalar, 3> const& translation) {
    this->params_.template tail<3>() = translation;
  }

  static auto fromRotationMatrix(Eigen::Matrix3<Scalar> const& mat_r)
      -> Isometry3 {
    return Isometry3(Rotation::fromRotationMatrix(mat_r));
  }

  static auto fromUnitQuaternion(Quaternion<Scalar> const& q) -> Isometry3 {
    return Isometry3(Rotation::fromUnitQuaternion(q));
  }

  static auto fitFromQuaternion(Quaternion<Scalar> const& q) -> Isometry3 {
    return Isometry3(Rotation::fitFromQuaternion(q));
  }

  /// Construct a translation only Isometry3 instance.
  ///
  template <class TT0, class TT1, class TT2>
  static auto fromT(TT0 const& x, TT1 const& y, TT2 const& z) -> Isometry3 {
    return Isometry3(Eigen::Vector3<Scalar>(x, y, z));
  }

  /// Construct x-axis translation.
  ///
  static auto fromTx(Scalar const& x) -> Isometry3 {
    return Isometry3::fromT(x, Scalar(0), Scalar(0));
  }

  /// Construct y-axis translation.
  ///
  static auto fromTy(Scalar const& y) -> Isometry3 {
    return Isometry3::fromT(Scalar(0), y, Scalar(0));
  }

  /// Construct z-axis translation.
  ///
  static auto fromTz(Scalar const& z) -> Isometry3 {
    return Isometry3::fromT(Scalar(0), Scalar(0), z);
  }

  /// Construct x-axis rotation.
  ///
  static auto fromRx(Scalar const& x) -> Isometry3 {
    return Isometry3(Rotation3<Scalar>::fromRx(x));
  }

  /// Construct y-axis rotation.
  ///
  static auto fromRy(Scalar const& y) -> Isometry3 {
    return Isometry3(Rotation3<Scalar>::fromRy(y));
  }

  /// Construct z-axis rotation.
  ///
  static auto fromRz(Scalar const& z) -> Isometry3 {
    return Isometry3(Rotation3<Scalar>::fromRz(z));
  }

  template <class TOtherScalar>
  auto cast() const -> Isometry3<TOtherScalar> {
    return Isometry3<TOtherScalar>::fromParams(
        this->params_.template cast<TOtherScalar>());
  }

  auto translation() -> Eigen::VectorBlock<Params, 3> {
    return this->params_.template tail<3>();
  }

  [[nodiscard]] auto translation() const
      -> Eigen::VectorBlock<Params const, 3> {
    return this->params_.template tail<3>();
  }

  [[nodiscard]] auto rotation() const -> Rotation3<Scalar> const {
    return Rotation3<Scalar>::fromParams(
        this->params_.template head<Rotation3<Scalar>::kNumParams>().eval());
  }

  auto setRotation(Rotation3<Scalar> const& rotation) {
    this->params_.template head<Rotation3<Scalar>::kNumParams>() =
        rotation.params();
  }

  [[nodiscard]] auto rotationMatrix() const -> Eigen::Matrix3<Scalar> {
    return this->rotation().matrix();
  }

  [[nodiscard]] auto unitQuaternion() const -> Quaternion<Scalar> {
    return Quaternion<Scalar>::fromParams(this->params_.template head<4>());
  }

  auto setUnitQuaternion(Quaternion<Scalar> const& z) const -> void {
    this->setRotation(Rotation::fromUnitQuaternion(z));
  }

  [[nodiscard]] auto so3() const -> Rotation3<Scalar> const {
    return rotation();
  }
};

using Isometry3F32 = Isometry3<float>;
using Isometry3F64 = Isometry3<double>;
static_assert(concepts::Isometry3<Isometry3F64>);

namespace details {
template <class TT>
class Cast<sophus::Isometry3<TT>> {
 public:
  template <class TTo>
  static auto impl(sophus::Isometry3<TT> const& v) {
    return v.template cast<typename TTo::Scalar>();
  }
  template <class TTo>
  static auto implScalar(sophus::Isometry3<TT> const& v) {
    return v.template cast<TTo>();
  }
};
}  // namespace details

template <class TScalar>
using SE3 = Isometry3<TScalar>;  // NOLINT
using SE3f = Isometry3<float>;   // NOLINT
using SE3d = Isometry3<double>;  // NOLINT

}  // namespace sophus
