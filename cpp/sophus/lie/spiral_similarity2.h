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
#include "sophus/lie/lie_group.h"
#include "sophus/lie/rotation2.h"

namespace sophus {

// origin and shape preserving mapping
template <class TScalar>
class SpiralSimilarity2
    : public lie::
          Group<SpiralSimilarity2, TScalar, lie::SpiralSimilarity2Impl> {
 public:
  using Scalar = TScalar;
  using Base =
      lie::Group<SpiralSimilarity2, TScalar, lie::SpiralSimilarity2Impl>;
  using Rotation = Rotation2<Scalar>;

  using Tangent = typename Base::Tangent;
  using Params = typename Base::Params;
  using Point = typename Base::Point;

  SpiralSimilarity2() = default;

  explicit SpiralSimilarity2(UninitTag /*unused*/) {}

  SpiralSimilarity2(Rotation2<Scalar> const& rotation, Scalar scale = 1.0)
      : Base(Base::fromParams(rotation.params())) {
    this->setScale(scale);
  }

  static auto fromComplex(Complex<Scalar> const& z) -> SpiralSimilarity2 {
    return SpiralSimilarity2::fromParams(z.params());
  }

  static auto fromRotationMatrix(Eigen::Matrix2<Scalar> const& mat_r)
      -> SpiralSimilarity2 {
    return SpiralSimilarity2(Rotation::fromRotationMatrix(mat_r));
  }

  static auto fromAngle(Scalar theta) -> SpiralSimilarity2 {
    return SpiralSimilarity2(Rotation::fromAngle(theta));
  }

  static auto fromScale(Scalar scale) -> SpiralSimilarity2 {
    SpiralSimilarity2 spiral_sim;
    spiral_sim.setScale(scale);
    return spiral_sim;
  }

  template <class TOtherScalar>
  auto cast() const -> SpiralSimilarity2<TOtherScalar> {
    return SpiralSimilarity2<TOtherScalar>::fromParams(
        this->params_.template cast<TOtherScalar>());
  }

  [[nodiscard]] auto rotation() const {
    return Rotation2<Scalar>::fromParams(
        this->params_.template head<Rotation2<Scalar>::kNumParams>()
            .normalized());
  }

  [[nodiscard]] auto rotationMatrix() const -> Eigen::Matrix2<Scalar> {
    return this->rotation().matrix();
  }

  void setRotation(Rotation rot) { SOPHUS_UNIMPLEMENTED(); }

  [[nodiscard]] auto scale() const -> Scalar { return this->params_.norm(); }

  void setScale(Scalar scale) {
    using std::sqrt;
    this->params_.normalize();
    this->params_ *= scale;
  }

  [[nodiscard]] auto angle() const -> Scalar {
    return this->rotation().angle();
  }

  [[nodiscard]] auto complex() const -> Complex<Scalar> {
    return Complex<Scalar>::fromParams(this->params_);
  }

  auto setComplex(Complex<Scalar> const& z) const -> void {
    this->setParams(z);
  }
};

using SpiralSimilarity2F32 = SpiralSimilarity2<float>;
using SpiralSimilarity2F64 = SpiralSimilarity2<double>;

static_assert(concepts::SpiralSimilarity2<SpiralSimilarity2F32>);

}  // namespace sophus
