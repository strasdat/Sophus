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
    ParamsImpl<TT> && TangentImpl<TT> &&
    (TT::kPointDim == 2 || TT::kPointDim == 3) &&  // 2d or 3d points
    (TT::kPointDim == TT::kAmbientDim  // inhomogeneous point representation
     || TT::kPointDim + 1 ==
            TT::kAmbientDim)  // or homogeneous point representation
    && requires(
           Eigen::Vector<typename TT::Scalar, TT::kDof> tangent,
           Eigen::Vector<typename TT::Scalar, TT::kPointDim> point,
           UnitVector<typename TT::Scalar, TT::kPointDim> direction,
           Eigen::Vector<typename TT::Scalar, TT::kNumParams> params,
           Eigen::Matrix<typename TT::Scalar, TT::kAmbientDim, TT::kAmbientDim>
               matrix,
           Eigen::Matrix<typename TT::Scalar, TT::kDof, TT::kDof> adjoint) {
  // constructors and factories
  {
    TT::identityParams()
    } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kNumParams>>;

  // Manifold / Lie Group concepts

  {
    TT::exp(tangent)
    } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kNumParams>>;

  {
    TT::log(params)
    } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kDof>>;

  {
    TT::hat(tangent)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kAmbientDim, TT::kAmbientDim>>;

  {
    TT::vee(matrix)
    } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kDof>>;

  // group operations
  {
    TT::multiplication(params, params)
    } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kNumParams>>;

  {
    TT::inverse(params)
    } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kNumParams>>;

  // Group actions
  {
    TT::action(params, point)
    } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kPointDim>>;

  {
    TT::toAmbient(point)
    } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kAmbientDim>>;

  // {
  //   T::action(params, direction)
  //   } -> ConvertibleTo<UnitVector<typename T::Scalar, T::kPointDim>>;

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
    TT::adj(params)
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
    Eigen::Vector<typename TT::Scalar, TT::kDof> tangent,
    Eigen::Vector<typename TT::Scalar, TT::kNumParams> params,
    Eigen::Vector<typename TT::Scalar, TT::kPointDim> point) {
  {
    TT::matV(params, tangent)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kPointDim, TT::kPointDim>>;

  {
    TT::matVInverse(params, tangent)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kPointDim, TT::kPointDim>>;

  {
    TT::topRightAdj(params, point)
    } -> ConvertibleTo<
        Eigen::Matrix<typename TT::Scalar, TT::kPointDim, TT::kDof>>;
};

template <class TT>
concept LieGroup = LieGroupImpl<typename TT::Impl> && ParamsConcept<TT> &&
    requires(
        TT g,
        Eigen::Vector<typename TT::Scalar, TT::kDof> tangent,
        Eigen::Vector<typename TT::Scalar, TT::kPointDim> point,
        UnitVector<typename TT::Scalar, TT::kPointDim> direction,
        Eigen::Vector<typename TT::Scalar, TT::kNumParams> params,
        Eigen::Matrix<typename TT::Scalar, TT::kAmbientDim, TT::kAmbientDim>
            matrix,
        Eigen::Matrix<typename TT::Scalar, TT::kDof, TT::kDof> adjoint) {
  // Manifold / Lie Group concepts

  { TT::exp(tangent) } -> ConvertibleTo<TT>;

  { g.log() } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kDof>>;

  // group operations
  { g.operator*(g) } -> ConvertibleTo<TT>;

  { g.inverse() } -> ConvertibleTo<TT>;

  // Group actions
  {
    g.operator*(point)
    } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kPointDim>>;

  {
    g.operator*(direction)
    } -> ConvertibleTo<UnitVector<typename TT::Scalar, TT::kPointDim>>;

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
    g.adj()
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
