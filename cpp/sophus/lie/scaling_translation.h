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
#include "sophus/lie/impl/semi_direct_product.h"
#include "sophus/lie/lie_group.h"

namespace sophus {

// origin, coordinate axis directions, and shape preserving mapping
template <class TScalar, int kDim>
class ScalingTranslation : public lie::Group<
                               ScalingTranslation<TScalar, kDim>,
                               lie::SemiDirectProductWithTranslation<
                                   TScalar,
                                   kDim,
                                   lie::ScalingImpl>> {
 public:
  ScalingTranslation() = default;

  explicit ScalingTranslation(UninitTag /*unused*/) {}

  template <class TOtherScalar>
  auto cast() const -> ScalingTranslation<TOtherScalar, kDim> {
    return ScalingTranslation<TOtherScalar, kDim>::fromParams(
        this->params_.template cast<TOtherScalar>());
  }
};

template <class TScalar>
using ScalingTranslation2 = ScalingTranslation<TScalar, 2>;
template <class TScalar>
using ScalingTranslation3 = ScalingTranslation<TScalar, 3>;

}  // namespace sophus
