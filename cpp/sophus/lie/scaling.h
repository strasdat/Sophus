// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/concepts/group_accessors.h"
#include "sophus/lie/impl/scaling.h"
#include "sophus/lie/lie_group.h"

namespace sophus {

// origin, coordinate axis directions, and shape preserving mapping

template <class TScalar, int kDim>
class Scaling : public lie::Group<
                    lie::ScalingWithDim<kDim>::template Group,
                    TScalar,
                    lie::ScalingWithDim<kDim>::template Impl> {
 public:
  using Scalar = TScalar;
  using Base = lie::Group<
      lie::ScalingWithDim<kDim>::template Group,
      TScalar,
      lie::ScalingWithDim<kDim>::template Impl>;

  using Tangent = typename Base::Tangent;
  using Params = typename Base::Params;
  using Point = typename Base::Point;

  Scaling() = default;

  explicit Scaling(UninitTag /*unused*/) {}

  template <class TOtherScalar>
  auto cast() const -> Scaling<TOtherScalar, kDim> {
    return Scaling<TOtherScalar, kDim>::fromParams(
        this->params_.template cast<TOtherScalar>());
  }

  [[nodiscard]] auto scaleFactors() const -> Params const& {
    return this->params_;
  }

  auto setScaleFactors(Params const& scale_factors) -> void {
    this->params_ = scale_factors;
  }
};

template <class TScalar>
using Scaling2 = Scaling<TScalar, 2>;
template <class TScalar>
using Scaling3 = Scaling<TScalar, 3>;

}  // namespace sophus
