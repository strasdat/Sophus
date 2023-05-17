// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/concepts/group_accessors.h"
#include "sophus/lie/impl/rotation2.h"
#include "sophus/lie/impl/translation_factor_group_product.h"
#include "sophus/lie/lie_group.h"
#include "sophus/lie/rotation2.h"

namespace sophus {

// definition: distance preserving mapping in R^2
//             <==> shape and size preserving mapping in R^2
template <class TScalar>
class Isometry2
    : public lie::Group<
          Isometry2,
          TScalar,
          lie::WithDimAndSubgroup<2, lie::Rotation2Impl>::SemiDirectProduct> {
 public:
  using Scalar = TScalar;

  using Base = lie::Group<
      Isometry2,
      TScalar,
      lie::WithDimAndSubgroup<2, lie::Rotation2Impl>::SemiDirectProduct>;
  using Rotation = Rotation2<Scalar>;

  using Tangent = typename Base::Tangent;
  using Params = typename Base::Params;
  using Point = typename Base::Point;

  Isometry2() = default;

  explicit Isometry2(UninitTag /*unused*/) : Base(UninitTag{}) {}

  Isometry2(
      Eigen::Vector<Scalar, 2> const& translation,
      Rotation2<Scalar> const& rotation)
      : Isometry2(UninitTag{}) {
    this->params_.template head<2>() = rotation.params();
    this->params_.template tail<2>() = translation;
  }

  Isometry2(Rotation2<Scalar> const& rotation) {
    this->params_.template head<2>() = rotation.params();
  }

  Isometry2(Eigen::Vector<Scalar, 2> const& translation) {
    this->params_.template tail<2>() = translation;
  }

  static auto fromRotationMatrix(Eigen::Matrix2<Scalar> const& mat_r)
      -> Isometry2 {
    return Isometry2(Rotation::fromRotationMatrix(mat_r));
  }

  static auto fitFromComplex(Complex<Scalar> const& z) -> Isometry2 {
    return Isometry2(Rotation::fitFromComplex(z));
  }

  static auto fromUnitComplex(Complex<Scalar> const& z) -> Isometry2 {
    return Isometry2(Rotation::fromUnitComplex(z));
  }

  static auto fromAngle(Scalar const& theta) -> Isometry2 {
    return Isometry2(Rotation(theta));
  }

  /// Construct a translation only Isometry3 instance.
  ///
  template <class TT0, class TT1>
  static auto fromT(TT0 const& x, TT1 const& y) -> Isometry2 {
    return Isometry2(Eigen::Vector2<Scalar>(x, y));
  }

  /// Construct x-axis translation.
  ///
  static auto fromTx(Scalar const& x) -> Isometry2 {
    return Isometry2::fromT(x, Scalar(0));
  }

  /// Construct y-axis translation.
  ///
  static auto fromTy(Scalar const& y) -> Isometry2 {
    return Isometry2::fromT(Scalar(0), y);
  }

  template <class TOtherScalar>
  auto cast() const -> Isometry2<TOtherScalar> {
    return Isometry2<TOtherScalar>::fromParams(
        this->params_.template cast<TOtherScalar>());
  }

  auto translation() -> Eigen::VectorBlock<Params, 2> {
    return this->params_.template tail<2>();
  }

  [[nodiscard]] auto translation() const
      -> Eigen::VectorBlock<Params const, 2> {
    return this->params_.template tail<2>();
  }

  [[nodiscard]] auto rotation() const -> Rotation2<Scalar> const {
    return Rotation2<Scalar>::fromParams(
        this->params_.template head<Rotation2<Scalar>::kNumParams>());
  }

  auto setRotation(Rotation2<Scalar> const& rotation) {
    this->params_.template head<Rotation2<Scalar>::kNumParams>() =
        rotation.params();
  }

  [[nodiscard]] auto rotationMatrix() const -> Eigen::Matrix2<Scalar> {
    return this->rotation().matrix();
  }

  [[nodiscard]] auto unitComplex() const -> Complex<Scalar> {
    return this->rotation().unitComplex();
  }

  auto setUnitComplex(Complex<Scalar> const& z) const -> void {
    this->setRotation(Rotation::fromUnitComplex(z));
  }

  [[nodiscard]] auto angle() const -> Scalar {
    return this->rotation().angle();
  }
};

using Isometry2F32 = Isometry2<float>;
using Isometry2F64 = Isometry2<double>;

static_assert(concepts::Isometry2<Isometry2F64>);

}  // namespace sophus
