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
concept ManifoldImpl = Tangent<TT> &&
    requires(typename TT::Tangent tangent, typename TT::Params params) {
  { TT::oplus(params, tangent) } -> ConvertibleTo<typename TT::Params>;

  { TT::ominus(params, params) } -> ConvertibleTo<typename TT::Tangent>;
};

template <class TT>
concept BaseManifold = Tangent<TT> &&
    requires(TT m, typename TT::Tangent tangent) {
  // Manifold concepts
  { m.oplus(tangent) } -> ConvertibleTo<TT>;

  { m.ominus(m) } -> ConvertibleTo<typename TT::Tangent>;
};

template <class TT>
concept Manifold = Params<TT> && Tangent<TT> && BaseManifold<TT> &&
    requires(std::vector<TT> points) {
  { TT::average(points) } -> ConvertibleTo<std::optional<TT>>;
};

}  // namespace concepts
}  // namespace sophus
