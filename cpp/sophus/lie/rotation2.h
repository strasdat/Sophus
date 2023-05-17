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
#include "sophus/lie/lie_group.h"
#include "sophus/linalg/orthogonal.h"

namespace sophus {

// definition: origin and distance preserving mapping in R^2
template <class TScalar>
class Rotation2 : public lie::Group<Rotation2, TScalar, lie::Rotation2Impl> {
 public:
  using Scalar = TScalar;

  using Base = lie::Group<Rotation2, TScalar, lie::Rotation2Impl>;

  using Tangent = typename Base::Tangent;
  using Params = typename Base::Params;
  using Point = typename Base::Point;

  Rotation2() = default;

  explicit Rotation2(UninitTag /*unused*/) {}

  explicit Rotation2(TScalar angle) : Rotation2(Rotation2::fromAngle(angle)) {}

  template <class TOtherScalar>
  explicit Rotation2(TOtherScalar angle)
      : Rotation2(Rotation2::fromAngle(angle)) {}

  template <class TOtherScalar>
  auto cast() const -> Rotation2<TOtherScalar> {
    return Rotation2<TOtherScalar>::fromParams(
        this->params_.template cast<TOtherScalar>());
  }

  static auto fromRotationMatrix(Eigen::Matrix2<TScalar> const& mat_r)
      -> Rotation2 {
    SOPHUS_ASSERT(
        isOrthogonal(mat_r),
        "R is not orthogonal:\n {}",
        mat_r * mat_r.transpose());
    SOPHUS_ASSERT(
        mat_r.determinant() > TScalar(0),
        "det(R) is not positive: {}",
        mat_r.determinant());

    return Rotation2::fromParams(mat_r.col(0));
  }

  static auto fromAngle(TScalar const& theta) -> Rotation2 {
    return Rotation2::exp(Eigen::Vector<TScalar, 1>{theta});
  }

  static auto fitFromComplex(Complex<TScalar> const& z) -> Rotation2 {
    return Rotation2::fromParams(z.params().normalized());
  }

  static auto fromUnitComplex(Complex<TScalar> const& z) -> Rotation2 {
    return Rotation2::fromParams(z.params());
  }

  static auto fitFromMatrix(Eigen::Matrix2<TScalar> const& mat_r) -> Rotation2 {
    return Rotation2::fromRotationMatrix(makeRotationMatrix(mat_r));
  }

  [[nodiscard]] auto rotationMatrix() const -> Eigen::Matrix2<TScalar> {
    return this->matrix();
  }

  [[nodiscard]] auto angle() const -> TScalar { return this->log()[0]; }

  [[nodiscard]] auto unitComplex() const -> Complex<TScalar> {
    return Complex<TScalar>::fromParams(this->params_);
  }

  auto setUnitComplex(Complex<TScalar> const& z) const -> void {
    this->setParams(z);
  }
};

using Rotation2F32 = Rotation2<float>;
using Rotation2F64 = Rotation2<double>;
static_assert(concepts::Rotation2<Rotation2F64>);

}  // namespace sophus
