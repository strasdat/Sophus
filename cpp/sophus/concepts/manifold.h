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
namespace concepts {

template <class TT>
concept ManifoldImpl = ParamsImpl<TT> && TangentImpl<TT> && requires(
    Eigen::Vector<typename TT::Scalar, TT::kDof> tangent,
    Eigen::Vector<typename TT::Scalar, TT::kNumParams> params) {
  {
    TT::oplus(params, tangent)
    } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kNumParams>>;

  {
    TT::ominus(params, params)
    } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kDof>>;
};

template <class TT>
concept Manifold = ParamsConcept<TT> && ParamsImpl<TT> && TangentImpl<TT> &&
    requires(
        TT m,
        Eigen::Vector<typename TT::Scalar, TT::kDof> tangent,
        Eigen::Vector<typename TT::Scalar, TT::kNumParams> params) {
  // Manifold concepts
  { m.oplus(tangent) } -> ConvertibleTo<TT>;

  {
    m.ominus(m)
    } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kDof>>;
};

}  // namespace concepts
}  // namespace sophus
