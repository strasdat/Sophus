// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/lie/impl/identity.h"
#include "sophus/lie/lie_group.h"

namespace sophus {

// definition: identity mapping in R^n
//             <==> origin, direction and distance preserving mapping R^n
//             <==> origin, direction, shape and size preserving mapping R^n

template <class TScalar, int kDim>
class Identity : public lie::Group<
                     lie::IdentityWithDim<kDim>::template Group,
                     TScalar,
                     lie::IdentityWithDim<kDim>::template Impl> {
 public:
  Identity() = default;

  explicit Identity(UninitTag /*unused*/) {}

  template <class TOtherScalar>
  auto cast() const -> Identity<TOtherScalar, kDim> {
    return Identity<TOtherScalar, kDim>::fromParams(
        this->params_.template cast<TOtherScalar>());
  }
};

template <class TScalar>
using Identity2 = Identity<TScalar, 2>;
template <class TScalar>
using Identity3 = Identity<TScalar, 3>;

}  // namespace sophus
