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
concept ManifoldImpl = ParamsImpl<TT> && TangentImpl<TT> &&
    requires(typename TT::Tangent tangent, typename TT::Params params) {
  { TT::oplus(params, tangent) } -> ConvertibleTo<typename TT::Params>;

  { TT::ominus(params, params) } -> ConvertibleTo<typename TT::Tangent>;
};

template <class TT>
concept Manifold = ParamsConcept<TT> && ParamsImpl<TT> && TangentImpl<TT> &&
    requires(TT m, typename TT::Tangent tangent, typename TT::Params params) {
  // Manifold concepts
  { m.oplus(tangent) } -> ConvertibleTo<TT>;

  { m.ominus(m) } -> ConvertibleTo<typename TT::Tangent>;
};

}  // namespace concepts
}  // namespace sophus
