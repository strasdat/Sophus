// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/concepts/group_accessors.h"
#include "sophus/lie/impl/roto_scaling3.h"
#include "sophus/lie/lie_group.h"

namespace sophus {

// origin, coordinate axis directions, and shape preserving mapping
template <class TScalar>
class RotoScaling3
    : public lie::Group<RotoScaling3<TScalar>, lie::RotoScaling3Impl<TScalar>> {
 public:
  RotoScaling3() = default;

  explicit RotoScaling3(UninitTag /*unused*/) {}

  template <class TOtherScalar>
  auto cast() const -> RotoScaling3<TOtherScalar> {
    return RotoScaling3<TOtherScalar>::fromParams(
        this->params_.template cast<TOtherScalar>());
  }
};

}  // namespace sophus
