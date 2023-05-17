// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/concepts/group_accessors.h"
#include "sophus/lie/impl/spiral_similarity2.h"
#include "sophus/lie/impl/translation_factor_group_product.h"
#include "sophus/lie/isometry2.h"
#include "sophus/lie/lie_group.h"
#include "sophus/lie/spiral_similarity2.h"

namespace sophus {

// origin, coordinate axis directions, and shape preserving mapping
template <class TScalar>
class Similarity2 : public lie::Group<
                        Similarity2,
                        TScalar,
                        lie::WithDimAndSubgroup<2, lie::SpiralSimilarity2Impl>::
                            SemiDirectProduct> {
 public:
  using Scalar = TScalar;
  using Base = lie::Group<
      Similarity2,
      TScalar,
      lie::WithDimAndSubgroup<2, lie::SpiralSimilarity2Impl>::
          SemiDirectProduct>;
  using Rotation = Rotation2<Scalar>;
  using SpiralSimilarity = SpiralSimilarity2<Scalar>;
  using Isometry = Isometry2<Scalar>;

  using Tangent = typename Base::Tangent;
  using Params = typename Base::Params;
  using Point = typename Base::Point;

  Similarity2() = default;

  explicit Similarity2(UninitTag /*unused*/) {}

  Similarity2(
      Eigen::Vector<Scalar, 2> const& translation,
      SpiralSimilarity const& scaled_rotation)
      : Similarity2(UninitTag{}) {
    this->params_.template head<2>() = scaled_rotation.params();
    this->params_.template tail<2>() = translation;
  }

  Similarity2(Rotation const& rotation)
      : Similarity2(Eigen::Vector<Scalar, 2>::Zero(), rotation) {}

  Similarity2(SpiralSimilarity const& scaled_rotation)
      : Similarity2(Eigen::Vector<Scalar, 2>::Zero(), scaled_rotation) {}

  Similarity2(Isometry const& isometry)
      : Similarity2(isometry.translation(), isometry.rotation()) {}

  Similarity2(
      Eigen::Vector<Scalar, 2> const& translation, Rotation const& rotation)
      : Similarity2(translation, SpiralSimilarity(rotation)) {}

  Similarity2(
      Eigen::Vector<Scalar, 2> const& translation,
      Rotation const& rotation,
      Scalar scale)
      : Similarity2(translation, SpiralSimilarity(rotation, scale)) {}

  Similarity2(Rotation const& rotation, Scalar scale)
      : Similarity2(Eigen::Vector<Scalar, 2>::Zero(), rotation, scale) {}

  Similarity2(Eigen::Vector<Scalar, 2> const& translation)
      : Similarity2(translation, SpiralSimilarity{}) {}

  static auto fromScale(Scalar const& scale) -> Similarity2 {
    return Similarity2(Rotation{}, scale);
  }

  static auto fromAngle(Scalar const& theta) -> Similarity2 {
    return Similarity2(Rotation(theta));
  }

  /// Construct a translation only Isometry3 instance.
  ///
  template <class TT0, class TT1>
  static auto fromT(TT0 const& x, TT1 const& y) -> Similarity2 {
    return Similarity2(Eigen::Vector2<Scalar>(x, y));
  }

  /// Construct x-axis translation.
  ///
  static auto fromTx(Scalar const& x) -> Similarity2 {
    return Similarity2::fromT(x, Scalar(0));
  }

  /// Construct y-axis translation.
  ///
  static auto fromTy(Scalar const& y) -> Similarity2 {
    return Similarity2::fromT(Scalar(0), y);
  }

  static auto fromRotationMatrix(Eigen::Matrix2<Scalar> const& mat_r)
      -> Similarity2 {
    return Similarity2(Rotation::fromRotationMatrix(mat_r));
  }

  static auto fromComplex(Complex<Scalar> const& z) -> Similarity2 {
    return Similarity2(SpiralSimilarity::fromComplex(z));
  }

  template <class TOtherScalar>
  auto cast() const -> Similarity2<TOtherScalar> {
    return Similarity2<TOtherScalar>::fromParams(
        this->params_.template cast<TOtherScalar>());
  }

  auto translation() -> Eigen::VectorBlock<Params, 2> {
    return this->params_.template tail<2>();
  }

  [[nodiscard]] auto translation() const
      -> Eigen::VectorBlock<Params const, 2> {
    return this->params_.template tail<2>();
  }

  [[nodiscard]] auto rotation() const {
    return this->spiralSimilarity().rotation();
  }

  void setRotation(Rotation const& rotation) { this->rotation() = rotation; }

  [[nodiscard]] auto rotationMatrix() const -> Eigen::Matrix2<Scalar> {
    return this->rotation().matrix();
  }

  [[nodiscard]] auto spiralSimilarity() const {
    return SpiralSimilarity2<Scalar>::fromParams(
        this->params_.template head<SpiralSimilarity2<Scalar>::kNumParams>());
  }

  auto setSpiralSimilarity(SpiralSimilarity2<Scalar> const& rotation) {
    this->params_.template head<SpiralSimilarity2<Scalar>::kNumParams>() =
        rotation.params();
  }

  [[nodiscard]] auto scale() const -> Scalar {
    return this->spiralSimilarity().scale();
  }

  void setScale(Scalar scale) {
    SpiralSimilarity2<Scalar> s = spiralSimilarity();
    s.setScale(scale);
    this->setSpiralSimilarity(s);
  }

  [[nodiscard]] auto angle() const -> Scalar {
    return this->rotation().angle();
  }

  auto setComplex(Complex<Scalar> const& z) const -> void {
    this->setSpiralSimilarity(SpiralSimilarity::fromComplex(z));
  }

  [[nodiscard]] auto complex() const -> Complex<Scalar> {
    return this->spiralSimilarity().complex();
  }
};

using Similarity2F32 = Similarity2<float>;
using Similarity2F64 = Similarity2<double>;

static_assert(concepts::Similarity2<Similarity2F64>);

}  // namespace sophus
