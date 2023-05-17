// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/concepts/group_accessors.h"
#include "sophus/lie/impl/spiral_similarity3.h"
#include "sophus/lie/lie_group.h"
#include "sophus/lie/rotation3.h"

namespace sophus {

// origin and shape preserving mapping
template <class TScalar>
class SpiralSimilarity3
    : public lie::
          Group<SpiralSimilarity3, TScalar, lie::SpiralSimilarity3Impl> {
 public:
  using Scalar = TScalar;
  using Rotation = Rotation3<Scalar>;
  using Base =
      lie::Group<SpiralSimilarity3, TScalar, lie::SpiralSimilarity3Impl>;

  using Tangent = typename Base::Tangent;
  using Params = typename Base::Params;
  using Point = typename Base::Point;

  SpiralSimilarity3() = default;

  explicit SpiralSimilarity3(UninitTag /*unused*/) {}

  SpiralSimilarity3(Rotation3<Scalar> const& rotation, Scalar scale = 1.0)
      : Base(Base::fromParams(rotation.params())) {
    this->setScale(scale);
  }

  static auto fromRotationMatrix(Eigen::Matrix3<Scalar> const& mat_r)
      -> SpiralSimilarity3 {
    return SpiralSimilarity3(Rotation::fromRotationMatrix(mat_r));
  }

  static auto fromRx(Scalar x) -> SpiralSimilarity3 {
    return SpiralSimilarity3(Rotation::fromRx(x));
  }

  static auto fromRy(Scalar y) -> SpiralSimilarity3 {
    return SpiralSimilarity3(Rotation::fromRy(y));
  }

  static auto fromRz(Scalar z) -> SpiralSimilarity3 {
    return SpiralSimilarity3(Rotation::fromRy(z));
  }

  static auto fromScale(Scalar scale) -> SpiralSimilarity3 {
    SpiralSimilarity3 spiral_sim;
    spiral_sim.setScale(scale);
    return spiral_sim;
  }

  static auto fromQuaternion(Quaternion<Scalar> const& q) -> SpiralSimilarity3 {
    return SpiralSimilarity3::fromParams(q.params());
  }

  static auto fitToRotation(Eigen::Matrix3<Scalar> const& mat_r) {
    return SpiralSimilarity3(Rotation::fitToRotation(mat_r));
  }
  template <class TOtherScalar>
  auto cast() const -> SpiralSimilarity3<TOtherScalar> {
    return SpiralSimilarity3<TOtherScalar>::fromParams(
        this->params_.template cast<TOtherScalar>());
  }

  [[nodiscard]] auto rotation() const {
    return Rotation3<Scalar>::fromParams(
        this->params_.template head<Rotation3<Scalar>::kNumParams>()
            .normalized());
  }

  void setRotation(Rotation rot) { SOPHUS_UNIMPLEMENTED(); }

  [[nodiscard]] auto rotationMatrix() const -> Eigen::Matrix3<Scalar> {
    return this->rotation().matrix();
  }

  [[nodiscard]] auto scale() const -> Scalar {
    return this->params_.squaredNorm();
  }

  void setScale(Scalar scale) {
    using std::sqrt;
    this->params_.normalize();
    this->params_ *= sqrt(scale);
  }

  [[nodiscard]] auto quaternion() const -> Quaternion<Scalar> {
    return Quaternion<Scalar>::fromParams(this->params_);
  }

  auto setQuaternion(Quaternion<Scalar> const& z) const -> void {
    this->setParams(z);
  }
};

using SpiralSimilarity3F32 = SpiralSimilarity3<float>;
using SpiralSimilarity3F64 = SpiralSimilarity3<double>;

static_assert(concepts::SpiralSimilarity3<SpiralSimilarity3F32>);

}  // namespace sophus
