// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once
#include "sophus/concepts/lie_group.h"

namespace sophus {

template <class TT>
class Complex;

template <class TT>
class Quaternion;

namespace concepts {

namespace accessors {

template <class TT>
concept Translation = LieGroup<TT> &&
    ConstructibleFrom<TT, typename TT::Point> && requires(TT g) {
  { g.translation() } -> ConvertibleTo<typename TT::Point>;
};

template <class TT>
concept Rotation = LieGroup<TT> && requires(
    TT g,
    Eigen::Matrix<typename TT::Scalar, TT::kPointDim, TT::kPointDim> matrix) {
  { TT::fromRotationMatrix(matrix) } -> ConvertibleTo<TT>;
  {
    g.rotationMatrix()
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kPointDim, TT::kPointDim>>;
};

template <class TT>
concept TxTy = requires(typename TT::Scalar s) {
  { TT::fromT(s, s) } -> ConvertibleTo<TT>;
  { TT::fromTx(s) } -> ConvertibleTo<TT>;
  { TT::fromTy(s) } -> ConvertibleTo<TT>;
};

template <class TT>
concept TxTyTz = requires(typename TT::Scalar s) {
  { TT::fromT(s, s, s) } -> ConvertibleTo<TT>;
  { TT::fromTx(s) } -> ConvertibleTo<TT>;
  { TT::fromTy(s) } -> ConvertibleTo<TT>;
  { TT::fromTz(s) } -> ConvertibleTo<TT>;
};

template <class TT>
concept Isometry = LieGroup<TT> && Translation<TT> &&
    ConstructibleFrom<TT, typename TT::Rotation> &&
    ConstructibleFrom<TT, typename TT::Point, typename TT::Rotation> &&
    requires(TT g, typename TT::Rotation rotation) {
  { g.rotation() } -> ConvertibleTo<typename TT::Rotation>;
  {g.setRotation(rotation)};
};

template <class TT>
concept Similarity = LieGroup<TT> && Translation<TT> &&
    ConstructibleFrom<TT, typename TT::Isometry> && ConstructibleFrom<
        TT,
        typename TT::Point,
        typename TT::Rotation,
        typename TT::Scalar> &&
    ConstructibleFrom<TT, typename TT::SpiralSimilarity> &&
    ConstructibleFrom<TT, typename TT::Point, typename TT::SpiralSimilarity>;

template <class TT>
concept SpiralSimilarity = Rotation<TT> &&
    ConstructibleFrom<TT, typename TT::Rotation> &&
    ConstructibleFrom<TT, typename TT::Rotation, typename TT::Scalar> &&
    requires(TT g, typename TT::Scalar scale, typename TT::Rotation rotation) {
  { TT::fromScale(scale) } -> ConvertibleTo<TT>;
  { g.rotation() } -> ConvertibleTo<typename TT::Rotation>;
  {g.setRotation(rotation)};
  { g.scale() } -> ConvertibleTo<typename TT::Scalar>;
  {g.setScale(scale)};
};

template <class TT>
concept UnitComplex =
    requires(TT g, typename TT::Scalar s, Complex<typename TT::Scalar> z) {
  { TT::fitFromComplex(z) } -> ConvertibleTo<TT>;
  { TT::fromUnitComplex(z) } -> ConvertibleTo<TT>;
  { g.unitComplex() } -> ConvertibleTo<Complex<typename TT::Scalar>>;
  {g.setUnitComplex(z)};
};

template <class TT>
concept UnitQuaternion =
    requires(TT g, typename TT::Scalar s, Quaternion<typename TT::Scalar> q) {
  { TT::fitFromQuaternion(q) } -> ConvertibleTo<TT>;
  { TT::fromUnitQuaternion(q) } -> ConvertibleTo<TT>;
  { g.unitQuaternion() } -> ConvertibleTo<Quaternion<typename TT::Scalar>>;
  {g.setUnitQuaternion(q)};
};

template <class TT>
concept Rotation2 = Rotation<TT> &&
    requires(TT g, typename TT::Scalar s, Complex<typename TT::Scalar> z) {
  { TT::fromAngle(s) } -> ConvertibleTo<TT>;
  { g.angle() } -> ConvertibleTo<typename TT::Scalar>;
};

template <class TT>
concept Rotation3 = Rotation<TT> &&
    requires(TT g, typename TT::Scalar s, Quaternion<typename TT::Scalar> q) {
  { TT::fromRx(s) } -> ConvertibleTo<TT>;
  { TT::fromRy(s) } -> ConvertibleTo<TT>;
  { TT::fromRz(s) } -> ConvertibleTo<TT>;
};

template <class TT>
concept Isometry2 = Isometry<TT> && Rotation2<TT> && TxTy<TT>;

template <class TT>
concept Isometry3 = Isometry<TT> && Rotation3<TT> && TxTyTz<TT>;

template <class TT>
concept SpiralSimilarity2 = Rotation2<TT> && SpiralSimilarity<TT> &&
    requires(TT g, Complex<typename TT::Scalar> z) {
  { TT::fromComplex(z) } -> ConvertibleTo<TT>;
  { g.complex() } -> ConvertibleTo<Complex<typename TT::Scalar>>;
  {g.setComplex(z)};
};

template <class TT>
concept SpiralSimilarity3 = Rotation3<TT> && SpiralSimilarity<TT> &&
    requires(TT g, Quaternion<typename TT::Scalar> q) {
  { TT::fromQuaternion(q) } -> ConvertibleTo<TT>;
  { g.quaternion() } -> ConvertibleTo<Quaternion<typename TT::Scalar>>;
  {g.setQuaternion(q)};
};

template <class TT>
concept Similarity2 = Similarity<TT> && SpiralSimilarity2<TT> && TxTy<TT>;

template <class TT>
concept Similarity3 = Similarity<TT> && SpiralSimilarity3<TT> && TxTyTz<TT>;

}  // namespace accessors

namespace base {

template <class TT>
concept Rotation = LieGroup<TT> && requires(
    TT g,
    Eigen::Matrix<typename TT::Scalar, TT::kPointDim, TT::kPointDim> matrix) {
  { TT::fitFromMatrix(matrix) } -> ConvertibleTo<TT>;
};

}  // namespace base

template <class TT>
concept Rotation2 =
    base::Rotation<TT> && ConstructibleFrom<TT, typename TT::Scalar> &&
    accessors::Rotation2<TT> && accessors::UnitComplex<TT>;

template <class TT>
concept Rotation3 = base::Rotation<TT> && accessors::Rotation3<TT> &&
    accessors::UnitQuaternion<TT>;

template <class TT>
concept SpiralSimilarity2 = accessors::SpiralSimilarity2<TT>;

template <class TT>
concept SpiralSimilarity3 = accessors::SpiralSimilarity3<TT>;

template <class TT>
concept Isometry2 = accessors::Isometry2<TT> && accessors::Rotation2<TT> &&
    accessors::UnitComplex<TT>;

template <class TT>
concept Isometry3 = accessors::Isometry3<TT> && accessors::Rotation3<TT> &&
    accessors::UnitQuaternion<TT>;

template <class TT>
concept Similarity2 =
    accessors::Similarity<TT> && accessors::SpiralSimilarity2<TT>;

template <class TT>
concept Similarity3 =
    accessors::Similarity<TT> && accessors::SpiralSimilarity3<TT>;

template <class TT>
concept Translation = accessors::Translation<TT>;

}  // namespace concepts
}  // namespace sophus
