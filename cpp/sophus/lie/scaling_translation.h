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
#include "sophus/lie/impl/translation_factor_group_product.h"
#include "sophus/lie/lie_group.h"

namespace sophus {

template <class TScalar, int kDim>
class ScalingTranslation;

namespace lie {
template <int kDim>
struct ScalingTranslationWithDim {
  template <class TScalar>
  using Group = ScalingTranslation<TScalar, kDim>;
};

}  // namespace lie

// origin, coordinate axis directions, and shape preserving mapping
template <class TScalar, int kDim>
class ScalingTranslation
    : public lie::Group<
          lie::ScalingTranslationWithDim<kDim>::template Group,
          TScalar,
          lie::WithDimAndSubgroup<
              kDim,
              lie::ScalingWithDim<kDim>::template Impl>::
              template SemiDirectProduct> {
 public:
  using Scalar = TScalar;
  using Base = lie::Group<
      lie::ScalingTranslationWithDim<kDim>::template Group,
      TScalar,
      lie::WithDimAndSubgroup<kDim, lie::ScalingWithDim<kDim>::template Impl>::
          template SemiDirectProduct>;

  using Tangent = typename Base::Tangent;
  using Params = typename Base::Params;
  using Point = typename Base::Point;

  ScalingTranslation() = default;

  explicit ScalingTranslation(UninitTag /*unused*/) {}

  template <class TOtherScalar>
  auto cast() const -> ScalingTranslation<TOtherScalar, kDim> {
    return ScalingTranslation<TOtherScalar, kDim>::fromParams(
        this->params_.template cast<TOtherScalar>());
  }

  auto translation() -> Eigen::VectorBlock<Params, kDim> {
    return this->params_.template tail<kDim>();
  }

  [[nodiscard]] auto translation() const
      -> Eigen::VectorBlock<Params const, kDim> {
    return this->params_.template tail<kDim>();
  }

  [[nodiscard]] auto scaleFactors() const -> Eigen::Vector<Scalar, kDim> {
    return this->params_.template head<kDim>();
  }

  auto setScaleFactors(Eigen::Vector<Scalar, kDim> const& scale_factors)
      -> void {
    this->params_.template head<kDim>() = scale_factors;
  }
};

template <class TScalar>
using ScalingTranslation2 = ScalingTranslation<TScalar, 2>;
template <class TScalar>
using ScalingTranslation3 = ScalingTranslation<TScalar, 3>;

using ScalingTranslation2F64 = ScalingTranslation2<double>;

using ScalingTranslation3F64 = ScalingTranslation3<double>;

}  // namespace sophus
