// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once
#include "sophus/concepts/params.h"
#include "sophus/concepts/utils.h"

namespace sophus {
namespace concepts {

template <class TScalar, int kN>
class UnitVector;

template <class TT>
concept DivisionRingImpl =
    ::sophus::concepts::ParamsImpl<TT>  // or homogeneous point representation
    && requires(
        typename TT::Params params,
        Eigen::Vector<CompatScalarEx<typename TT::Scalar>, TT::kNumParams>
            compatible_params) {
  // constructors and factories
  { TT::one() } -> ::sophus::concepts::ConvertibleTo<typename TT::Params>;

  { TT::zero() } -> ::sophus::concepts::ConvertibleTo<typename TT::Params>;

  // operations

  {
    TT::addition(params, params)
    } -> ::sophus::concepts::ConvertibleTo<typename TT::Params>;

  {
    TT::multiplication(params, params)
    } -> ::sophus::concepts::ConvertibleTo<typename TT::Params>;

#if __cplusplus >= 202002L
  {
    TT::template addition<CompatScalarEx<typename TT::Scalar>>(
        params, compatible_params)
    } -> ::sophus::concepts::ConvertibleTo<typename TT::template ParamsReturn<
        CompatScalarEx<typename TT::Scalar>>>;

  {
    TT::template multiplication<CompatScalarEx<typename TT::Scalar>>(
        params, compatible_params)
    } -> ::sophus::concepts::ConvertibleTo<typename TT::template ParamsReturn<
        CompatScalarEx<typename TT::Scalar>>>;
#endif

  {
    TT::conjugate(params)
    } -> ::sophus::concepts::ConvertibleTo<typename TT::Params>;

  {
    TT::inverse(params)
    } -> ::sophus::concepts::ConvertibleTo<typename TT::Params>;

  // reduction
  {
    TT::norm(params)
    } -> ::sophus::concepts::ConvertibleTo<typename TT::Scalar>;

  {
    TT::squaredNorm(params)
    } -> ::sophus::concepts::ConvertibleTo<typename TT::Scalar>;
};

template <class TT>
concept DivisionRingConcept = DivisionRingImpl<typename TT::Impl> && requires(
    TT r,
    typename TT::Scalar real,
    typename TT::Imag imag,
    typename TT::Params params) {
  // operations
  { r.operator+(r) } -> ::sophus::concepts::ConvertibleTo<TT>;

  { r.operator*(r) } -> ::sophus::concepts::ConvertibleTo<TT>;

  { r.conjugate() } -> ::sophus::concepts::ConvertibleTo<TT>;

  { r.inverse() } -> ::sophus::concepts::ConvertibleTo<TT>;

  // reduction
  { r.norm() } -> ::sophus::concepts::ConvertibleTo<typename TT::Scalar>;

  { r.squaredNorm() } -> ::sophus::concepts::ConvertibleTo<typename TT::Scalar>;

  { r.real() } -> ::sophus::concepts::ConvertibleTo<typename TT::Scalar>;

  { r.imag() } -> ::sophus::concepts::ConvertibleTo<typename TT::Imag>;
};
}  // namespace concepts
}  // namespace sophus
