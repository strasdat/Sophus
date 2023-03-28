// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once
#include "sophus/concepts/params.h"

namespace sophus {
template <class TScalar, int kN>
class UnitVector;

namespace concepts {

template <class TT>
concept LieGroupImpl =
    ParamsImpl<TT> && TangentImpl<TT> && std::is_same_v<
        typename TT::Point,
        Eigen::Vector<typename TT::Scalar, TT::kPointDim>> &&
    (TT::kPointDim == TT::kAmbientDim  // inhomogeneous point representation
     || TT::kPointDim + 1 ==
            TT::kAmbientDim)  // or homogeneous point representation
    && requires(
           typename TT::Tangent tangent,
           typename TT::Point point,
           UnitVector<typename TT::Scalar, TT::kPointDim> direction,
           typename TT::Params params,
           Eigen::Matrix<typename TT::Scalar, TT::kAmbientDim, TT::kAmbientDim>
               matrix,
           Eigen::Matrix<typename TT::Scalar, TT::kDof, TT::kDof> adjoint) {
  // constructors and factories
  { TT::identityParams() } -> ConvertibleTo<typename TT::Params>;

  // Manifold / Lie Group concepts

  { TT::exp(tangent) } -> ConvertibleTo<typename TT::Params>;

  { TT::log(params) } -> ConvertibleTo<typename TT::Tangent>;

  {
    TT::hat(tangent)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kAmbientDim, TT::kAmbientDim>>;

  { TT::vee(matrix) } -> ConvertibleTo<typename TT::Tangent>;

  // group operations
  { TT::multiplication(params, params) } -> ConvertibleTo<typename TT::Params>;

  { TT::inverse(params) } -> ConvertibleTo<typename TT::Params>;

  // Group actions
  { TT::action(params, point) } -> ConvertibleTo<typename TT::Point>;

  {
    TT::toAmbient(point)
    } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kAmbientDim>>;

  // {
  //   T::action(params, direction)
  //   } -> ConvertibleTo<UnitVector<typename T::Scalar, T::kPointDim>>;

  {
    TT::adj(params)
    } -> ConvertibleTo<Eigen::Matrix<typename TT::Scalar, TT::kDof, TT::kDof>>;

  // Matrices

  {
    TT::compactMatrix(params)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kPointDim, TT::kAmbientDim>>;

  {
    TT::matrix(params)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kAmbientDim, TT::kAmbientDim>>;

  // Derivatives

  {
    TT::ad(tangent)
    } -> ConvertibleTo<Eigen::Matrix<typename TT::Scalar, TT::kDof, TT::kDof>>;

  // {
  //   TT::dxExpX(tangent)
  //   } -> ConvertibleTo<
  //       Eigen::Matrix<typename TT::Scalar, TT::kNumParams, TT::kDof>>;

  {
    TT::dxExpXAt0()
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kNumParams, TT::kDof>>;

  {
    TT::dxExpXTimesPointAt0(point)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kPointDim, TT::kDof>>;

  {
    TT::dxThisMulExpXAt0(params)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kNumParams, TT::kDof>>;

  {
    TT::dxLogThisInvTimesXAtThis(params)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kDof, TT::kNumParams>>;
};

// Ideally, the LieSubgroupFunctions is not necessary and all these
// properties can be deduced.
template <class TT>
concept LieSubgroupImpl = LieGroupImpl<TT> && requires(
    typename TT::Tangent tangent,
    typename TT::Params params,
    typename TT::Point point) {
  {
    TT::matV(params, tangent)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kPointDim, TT::kPointDim>>;

  {
    TT::matVInverse(params, tangent)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kPointDim, TT::kPointDim>>;

  {
    TT::adjOfTranslation(params, point)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kPointDim, TT::kDof>>;

  {
    TT::adOfTranslation(point)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kPointDim, TT::kPointDim>>;
};

template <class TT>
concept LieGroup = LieGroupImpl<typename TT::Impl> && ParamsConcept<TT> &&
    std::is_same_v<
        typename TT::Point,
        Eigen::Vector<typename TT::Scalar, TT::kPointDim>> &&
    requires(
        TT g,
        typename TT::Tangent tangent,
        typename TT::Point point,
        UnitVector<typename TT::Scalar, TT::kPointDim> direction,
        typename TT::Params params,
        Eigen::Matrix<typename TT::Scalar, TT::kAmbientDim, TT::kAmbientDim>
            matrix,
        Eigen::Matrix<typename TT::Scalar, TT::kDof, TT::kDof> adjoint) {
  // Manifold / Lie Group concepts

  { TT::exp(tangent) } -> ConvertibleTo<TT>;

  { g.log() } -> ConvertibleTo<typename TT::Tangent>;

  // group operations
  { g.operator*(g) } -> ConvertibleTo<TT>;

  { g.inverse() } -> ConvertibleTo<TT>;

  // Group actions
  { g.operator*(point) } -> ConvertibleTo<typename TT::Point>;

  {
    g.operator*(direction)
    } -> ConvertibleTo<UnitVector<typename TT::Scalar, TT::kPointDim>>;

  {
    g.adj()
    } -> ConvertibleTo<Eigen::Matrix<typename TT::Scalar, TT::kDof, TT::kDof>>;

  // Matrices

  {
    g.compactMatrix()
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kPointDim, TT::kAmbientDim>>;

  {
    g.matrix()
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kAmbientDim, TT::kAmbientDim>>;

  // Derivatives
  {
    TT::ad(tangent)
    } -> ConvertibleTo<Eigen::Matrix<typename TT::Scalar, TT::kDof, TT::kDof>>;

  // {
  //   T::dxExpX(tangent)
  //   }
  //   -> ConvertibleTo<Eigen::Matrix<typename T::Scalar, T::kNumParams,
  //   T::kDof>>;

  {
    TT::dxExpXAt0()
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kNumParams, TT::kDof>>;

  {
    TT::dxExpXTimesPointAt0(point)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kPointDim, TT::kDof>>;

  {
    g.dxThisMulExpXAt0()
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kNumParams, TT::kDof>>;

  {
    g.dxLogThisInvTimesXAtThis()
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kDof, TT::kNumParams>>;

  { TT::elementExamples() } -> ConvertibleTo<std::vector<TT>>;
};

}  // namespace concepts
}  // namespace sophus
