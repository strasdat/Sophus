// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/concepts/group_accessors.h"
#include "sophus/lie/impl/identity.h"
#include "sophus/lie/impl/semi_direct_product.h"
#include "sophus/lie/lie_group.h"

namespace sophus {

// scale and direction preserving mapping
template <class TScalar, int kDim>
class Translation : public lie::Group<
                        Translation<TScalar, kDim>,
                        lie::SemiDirectProductWithTranslation<
                            TScalar,
                            kDim,
                            lie::IdentityImpl>> {
 public:
  Translation() = default;

  explicit Translation(UninitTag /*unused*/) {}

  template <class TOtherScalar>
  auto cast() const -> Translation<TOtherScalar, kDim> {
    return Translation<TOtherScalar, kDim>::fromParams(
        this->params_.template cast<TOtherScalar>());
  }
};

template <class TScalar>
using Translation2 = Translation<TScalar, 2>;
template <class TScalar>
using Translation3 = Translation<TScalar, 3>;

using Translation2F32 = Translation2<float>;
using Translation2F64 = Translation2<double>;

using Translation3F32 = Translation3<float>;
using Translation3F64 = Translation3<double>;

}  // namespace sophus
