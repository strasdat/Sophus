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
#include "sophus/lie/impl/translation_factor_group_product.h"
#include "sophus/lie/isometry3.h"
#include "sophus/lie/lie_group.h"
#include "sophus/lie/spiral_similarity3.h"

namespace sophus {

// origin, coordinate axis directions, and shape preserving mapping
template <class TScalar>
class Similarity3 : public lie::Group<
                        Similarity3,
                        TScalar,
                        lie::WithDimAndSubgroup<3, lie::SpiralSimilarity3Impl>::
                            SemiDirectProduct> {
 public:
  using Scalar = TScalar;
  using Base = lie::Group<
      Similarity3,
      TScalar,
      lie::WithDimAndSubgroup<3, lie::SpiralSimilarity3Impl>::
          SemiDirectProduct>;
  using Rotation = Rotation3<Scalar>;
  using SpiralSimilarity = SpiralSimilarity3<Scalar>;
  using Isometry = Isometry3<Scalar>;

  using Tangent = typename Base::Tangent;
  using Params = typename Base::Params;
  using Point = typename Base::Point;

  Similarity3() = default;

  explicit Similarity3(UninitTag /*unused*/) {}

  Similarity3(
      Eigen::Vector<Scalar, 3> const& translation,
      SpiralSimilarity const& scaled_rotation)
      : Similarity3(UninitTag{}) {
    this->params_.template head<4>() = scaled_rotation.params();
    this->params_.template tail<3>() = translation;
  }

  Similarity3(Rotation const& rotation)
      : Similarity3(Eigen::Vector<Scalar, 3>::Zero(), rotation) {}

  Similarity3(SpiralSimilarity const& scaled_rotation)
      : Similarity3(Eigen::Vector<Scalar, 3>::Zero(), scaled_rotation) {}

  Similarity3(Isometry const& isometry)
      : Similarity3(isometry.translation(), isometry.rotation()) {}

  Similarity3(
      Eigen::Vector<Scalar, 3> const& translation, Rotation const& rotation)
      : Similarity3(translation, SpiralSimilarity(rotation)) {}

  Similarity3(
      Eigen::Vector<Scalar, 3> const& translation,
      Rotation const& rotation,
      Scalar scale)
      : Similarity3(translation, SpiralSimilarity(rotation, scale)) {}

  Similarity3(Rotation const& rotation, Scalar scale)
      : Similarity3(Eigen::Vector<Scalar, 3>::Zero(), rotation, scale) {}

  Similarity3(Eigen::Vector<Scalar, 3> const& translation)
      : Similarity3(translation, SpiralSimilarity{}) {}

  static auto fromRotationMatrix(Eigen::Matrix3<Scalar> const& mat_r)
      -> Similarity3 {
    return Similarity3(Rotation::fromRotationMatrix(mat_r));
  }

  static auto fromQuaternion(Quaternion<Scalar> const& q) -> Similarity3 {
    return Similarity3(SpiralSimilarity::fromQuaternion(q));
  }

  static auto fromScale(Scalar const& scale) -> Similarity3 {
    return Similarity3(Rotation{}, scale);
  }

  /// Construct a translation only Isometry3 instance.
  ///
  template <class TT0, class TT1, class TT2>
  static auto fromT(TT0 const& x, TT1 const& y, TT2 const& z) -> Similarity3 {
    return Similarity3(Eigen::Vector3<Scalar>(x, y, z));
  }

  /// Construct x-axis translation.
  ///
  static auto fromTx(Scalar const& x) -> Similarity3 {
    return Similarity3::fromT(x, Scalar(0), Scalar(0));
  }

  /// Construct y-axis translation.
  ///
  static auto fromTy(Scalar const& y) -> Similarity3 {
    return Similarity3::fromT(Scalar(0), y, Scalar(0));
  }

  /// Construct z-axis translation.
  ///
  static auto fromTz(Scalar const& z) -> Similarity3 {
    return Similarity3::fromT(Scalar(0), Scalar(0), z);
  }

  /// Construct x-axis rotation.
  ///
  static auto fromRx(Scalar const& x) -> Similarity3 {
    return Similarity3(Rotation3<Scalar>::fromRx(x));
  }

  /// Construct y-axis rotation.
  ///
  static auto fromRy(Scalar const& y) -> Similarity3 {
    return Similarity3(Rotation3<Scalar>::fromRy(y));
  }

  /// Construct z-axis rotation.
  ///
  static auto fromRz(Scalar const& z) -> Similarity3 {
    return Similarity3(Rotation3<Scalar>::fromRz(z));
  }

  template <class TOtherScalar>
  auto cast() const -> Similarity3<TOtherScalar> {
    return Similarity3<TOtherScalar>::fromParams(
        this->params_.template cast<TOtherScalar>());
  }

  auto translation() -> Eigen::VectorBlock<Params, 3> {
    return this->params_.template tail<3>();
  }

  [[nodiscard]] auto translation() const
      -> Eigen::VectorBlock<Params const, 3> {
    return this->params_.template tail<3>();
  }

  [[nodiscard]] auto rotation() const {
    return this->spiralSimilarity().rotation();
  }

  void setRotation(Rotation const& rotation) { this->rotation() = rotation; }

  [[nodiscard]] auto rotationMatrix() const -> Eigen::Matrix3<Scalar> {
    return this->rotation().matrix();
  }

  [[nodiscard]] auto spiralSimilarity() const {
    return SpiralSimilarity3<Scalar>::fromParams(
        this->params_.template head<SpiralSimilarity3<Scalar>::kNumParams>());
  }

  auto setSpiralSimilarity(SpiralSimilarity3<Scalar> const& rotation) {
    this->params_.template head<SpiralSimilarity3<Scalar>::kNumParams>() =
        rotation.params();
  }
  [[nodiscard]] auto scale() const -> Scalar {
    return this->spiralSimilarity().scale();
  }

  void setScale(Scalar scale) {
    SpiralSimilarity3<Scalar> s = spiralSimilarity();
    s.setScale(scale);
    this->setSpiralSimilarity(s);
  }

  auto setQuaternion(Quaternion<Scalar> const& z) const -> void {
    this->setSpiralSimilarity(SpiralSimilarity::fromQuaternion(z));
  }

  [[nodiscard]] auto quaternion() const -> Quaternion<Scalar> {
    return this->spiralSimilarity().quaternion();
  }
};

using Similarity3F32 = Similarity3<float>;
using Similarity3F64 = Similarity3<double>;

static_assert(concepts::Similarity3<Similarity3F64>);

}  // namespace sophus
